"""Vocabulary-restricted LM head for the DSpark draft pass.

The draft only proposes tokens; the verify step scores them with the full head and keeps a token only if the target
agrees. So the draft head may look at a subset of the vocabulary: the emitted text is unchanged, only the acceptance
rate can move (a target token outside the subset can never be drafted). The restricted head is the exact rows of the
full head, so for the tokens it covers its logits equal the full-head logits.

EXL3 granularity: the output side of a Linear is transformed by a 128-wide Hadamard, so the unit of selection is a
block of 128 consecutive token ids (8 trellis tiles). The blocks are gathered into a new, contiguous trellis; no
re-quantisation, no extra weight bytes beyond the selected rows.

    RestrictedHead.prefix(head, vocab, 32768)          # the first 32768 ids (BPE ids are roughly frequency ordered)
    RestrictedHead.from_counts(head, counts, 32768)    # the 256 most frequent 128-blocks of a token histogram
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

BLOCK = 128


def blocks_from_counts(counts: torch.Tensor, n_ids: int) -> torch.Tensor:
    """Sorted block indices of the n_ids // 128 blocks with the largest summed token count."""
    nb = counts.numel() // BLOCK
    per = counts[:nb * BLOCK].double().view(nb, BLOCK).sum(1)
    k = max(1, min(nb, n_ids // BLOCK))
    return per.topk(k).indices.sort().values


def _inner_of(head):
    """The framework Linear inside a PackLin (None for DenseLin / other)."""
    lin = getattr(head, "lin", None)
    return getattr(lin, "inner", None)


class RestrictedHead:
    def __init__(self, head, vocab: int, blocks: torch.Tensor):
        self.vocab = vocab
        blocks = blocks.to("cpu", torch.int64).sort().values
        assert int(blocks.max()) < (vocab + BLOCK - 1) // BLOCK
        self.blocks = blocks
        ids = (blocks[:, None] * BLOCK + torch.arange(BLOCK)[None]).flatten()
        self.in_features = head.in_features
        inner = _inner_of(head)
        dev = inner.trellis.device if inner is not None else head.w.device
        self.ids = ids[ids < vocab].to(dev)
        self.n_pad = int((ids >= vocab).sum())               # only the tail block of an unaligned vocab
        self._mode = None
        if inner is not None and getattr(inner, "quant_type", None) == "exl3":
            from ..modules.quant.exl3 import LinearEXL3
            assert self.n_pad == 0 and vocab % BLOCK == 0, "EXL3 head vocab must be a multiple of 128"
            tiles = (blocks[:, None] * (BLOCK // 16) + torch.arange(BLOCK // 16)[None]).flatten().to(dev)
            rows = ids.to(dev)
            self.inner = LinearEXL3(
                None, inner.in_features, int(ids.numel()), suh=inner.suh, svh=inner.svh[rows].contiguous(),
                trellis=inner.trellis[:, tiles].contiguous(), mcg=inner.mcg_tensor, mul1=inner.mul1_tensor,
                bias=None if inner.bias is None else inner.bias[rows].contiguous(), key="draft_head")
            self._mode = "exl3"
        elif inner is not None:                              # fp16 LinearFP16 inside a PackLin
            self.w = inner.weight[:, self.ids].contiguous()  # stored [in, out]
            self._mode = "fp16"
        else:                                                # DenseLin
            self.w = head.w[self.ids].contiguous()
            self._mode = "dense"

    @classmethod
    def prefix(cls, head, vocab: int, n_ids: int):
        return cls(head, vocab, torch.arange(min(vocab // BLOCK, max(1, n_ids // BLOCK))))

    @classmethod
    def from_counts(cls, head, counts: torch.Tensor, n_ids: int):
        return cls(head, counts.numel(), blocks_from_counts(counts, n_ids))

    def nbytes(self) -> int:
        if self._mode == "exl3":
            return self.inner.trellis.numel() * 2
        return self.w.numel() * self.w.element_size()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """x [..., D] -> float32 logits [..., len(ids)] for the selected token ids."""
        lead = x.shape[:-1]
        if self._mode == "exl3":
            y = self.inner.forward(x.half().reshape(-1, x.shape[-1]).contiguous(), {}, torch.half)
        elif self._mode == "fp16":
            y = torch.matmul(x.half().reshape(-1, x.shape[-1]), self.w)
        else:
            y = F.linear(x.to(self.w.dtype).reshape(-1, x.shape[-1]), self.w)
        return y.float().reshape(*lead, -1)
