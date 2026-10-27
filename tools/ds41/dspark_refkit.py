"""CPU harness around the official DeepSeek-V4.1-Flash-1017 `inference/` code (vendored in testdata/reference/inference).

* `load_official()` imports the unmodified official `model.py` and `generate.py` with a pure-torch stand-in for the CUDA
  `kernel.py` (sparse attention, hyper-connection Sinkhorn, identity quantisation) and a stub `encoding` module.
* `build_tiny()` builds a random-weight `Transformer` with the real structure (sliding window ring, compressors at ratios
  2 and 1, indexers, MoE, hyper-connections, three DSpark stages, Markov head) and small dimensions.
* `OfficialTarget` / `OfficialDrafter` put the official classes behind the `dspark.Target` / `dspark.Drafter` protocols.
  The official forward takes a full prefill at position 0 or ONE token at a later position, so `verify` runs the tokens
  one by one (logits are bit identical to `generate.py`) and `truncate` restores a snapshot of every cache buffer and
  replays the kept tokens: a ring slot overwritten by a rejected position cannot be recovered any other way.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
REF = HERE / "testdata" / "reference" / "inference"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import dspark  # noqa: E402

# ---- kernel stand-in ----------------------------------------------------------------------------------------------


def _act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    assert inplace, "only the fused quant+dequant form is used with bf16 weights"
    return x


def _fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=None):
    return x


def _sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """Same maths as kernel.sparse_attn: per query, softmax over the gathered kv rows (index -1 = absent) plus the
    per-head sink; a query with no valid row gets 0."""
    b = q.size(0)
    idx = topk_idxs.long()
    valid = idx >= 0
    rows = kv[torch.arange(b)[:, None, None], idx.clamp(min=0)].float()               # [b, m, k, d]
    s = torch.einsum("bmhd,bmkd->bmhk", q.float(), rows) * softmax_scale
    s = s.masked_fill(~valid[:, :, None, :], -torch.inf)
    mx = s.amax(-1, keepdim=True).clamp(min=-1e30)
    e = torch.exp(s - mx)
    denom = e.sum(-1) + torch.exp(attn_sink.float()[None, None, :] - mx.squeeze(-1))
    return (torch.einsum("bmhk,bmkd->bmhd", e, rows) / denom.unsqueeze(-1)).to(q.dtype)


def _hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    hc = hc_mult
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = (mixes[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]).unflatten(-1, (hc, hc))
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def _stub(name: str, **attrs: Any) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    return mod


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_official: tuple[Any, Any] | None = None


def load_official() -> tuple[Any, Any]:
    """(model module, generate module) of the vendored official code, imported once."""
    global _official
    if _official is not None:
        return _official
    kernel = _stub("kernel", act_quant=_act_quant, fp4_act_quant=_fp4_act_quant, sparse_attn=_sparse_attn,
                   hc_split_sinkhorn=_hc_split_sinkhorn, fp8_gemm=None, fp4_gemm=None)
    noop = lambda *a, **k: None  # noqa: E731
    encoding = _stub("encoding", encode_case=noop, encode_messages=noop, load_cases=noop,
                     parse_message_from_completion_text=noop, parse_tagged_text=noop, to_json=noop)
    stubs = {"kernel": kernel, "encoding": encoding, "transformers": _stub("transformers", AutoTokenizer=None),
             "safetensors": _stub("safetensors"), "safetensors.torch": _stub("safetensors.torch", load_model=None)}
    borrowed = (*stubs, "model", "engram", "vision", "image_processor")
    saved = {n: sys.modules.get(n) for n in borrowed}
    sys.path.insert(0, str(REF))
    try:
        sys.modules.update(stubs)                    # generate.py imports these; generate() itself needs none of them
        model = _load("dsv41_model", REF / "model.py")
        sys.modules["model"] = model
        generate = _load("dsv41_generate", REF / "generate.py")
    finally:
        sys.path.remove(str(REF))
        for n, m in saved.items():
            if m is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = m
    _official = (model, generate)
    return _official


# ---- tiny model ---------------------------------------------------------------------------------------------------

TINY = dict(
    max_batch_size=1, max_seq_len=128, temperature=0, dtype="bf16", expert_dtype=None, vocab_size=32, dim=64,
    moe_inter_dim=32, n_layers=6, n_mtp_layers=3, n_heads=4, n_routed_experts=8, n_shared_experts=1,
    n_activated_experts=2, swiglu_limit=10.0, q_lora_rank=32, head_dim=32, rope_head_dim=16, o_groups=2,
    o_lora_rank=16, window_size=8,
    # layers 0,1 window only; 2,3 compress 2-to-1 (2 is the source); 4,5 plain projection (4 is the source);
    # the last three entries are the DSpark stages: window only
    compress_ratios=(0, 0, 2, 2, 1, 1, 0, 0, 0), kv_source_layers=(2, 4), index_source_layers=(2, 4),
    # 16 index heads: with few heads many groups score exactly 0 after the relu, and top-k then breaks the ties
    # differently for different tensor shapes, so prefill and decode would pick different groups
    index_n_heads=16, index_head_dim=16, index_topk=4, candidate_source_layer=-1, hc_mult=2, hc_sinkhorn_iters=4,
    original_seq_len=0, compress_rope_theta=10000.0, route_scale=1.5,
    # engram on layers 1 and 3: 3-gram hash state kept in a per-position token cache (see FakeTokenizer)
    engram_layer_ids=(1, 3), engram_num_embeddings=(1200, 1200), engram_max_ngram_size=3, engram_vocab_size=50,
    engram_n_heads=2, engram_head_dim=32, engram_pad_id=2, engram_compressed_vocab_size=26,
    dspark_block_size=5, dspark_noise_token_id=31, dspark_target_layer_ids=(3, 4, 5), dspark_markov_rank=8,
    dspark_n_routed_experts=8, dspark_n_activated_experts=3,
)


class FakeTokenizer:
    """The three things engram.build_compressed_token_map reads from a tokenizer: token i is the letter a..z for i < 26 and
    A..F after, so ids 26..31 collapse onto 0..5 (26 compressed ids)."""

    def __init__(self, n=32):
        self.n = n
        self.backend_tokenizer = self

    def __len__(self):
        return self.n

    def _text(self, i):
        return chr(97 + i) if i < 26 else chr(65 + i - 26)

    def decode(self, ids, skip_special_tokens=False):
        return self._text(ids[0])

    def id_to_token(self, i):
        return self._text(i)


def build_tiny(seed: int = 0, **overrides: Any):
    """Random-weight official Transformer, float32, temperature 0 on the target and on every DSpark stage (so the
    model's own `sample` never touches the RNG)."""
    mod, _ = load_official()
    args = mod.ModelArgs(**{**TINY, **overrides})
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        model = mod.Transformer(args, FakeTokenizer(args.vocab_size))
    finally:
        torch.set_default_dtype(prev)
    model.float().eval()
    for layer in model.layers:                # the engram table is fp8 in the checkpoint: float32 here, bf16 out of the lookup
        if layer.engram is not None:
            layer.engram.embed.register_forward_hook(lambda mod, inp, out: out.float())
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("norm.weight"):
                p.copy_(1 + 0.1 * torch.randn(p.shape, generator=g))
            elif name.endswith("attn_sink"):
                p.copy_(torch.randn(p.shape, generator=g))
            elif name.endswith((".bias", "_base")):
                p.copy_(0.3 * torch.randn(p.shape, generator=g))
            elif name.endswith(("_scale", "embed.scale")):
                p.copy_(0.5 + 0.2 * torch.rand(p.shape, generator=g))
            elif p.ndim >= 2:
                gain = {"embed.weight": 1.0, "head.weight": 3.0}.get(name, 1.0)
                p.copy_(gain * torch.randn(p.shape, generator=g) / p.shape[-1] ** 0.5)
            else:
                p.copy_(0.2 * torch.randn(p.shape, generator=g))
    model.temperature = 0
    for stage in model.mtp:
        stage.temperature = 0
    model.fresh_state = capture(model)        # empty caches, to start a request from
    return model


def tiny_ids(model) -> tuple[int, int]:
    """(vocab size, noise token id)"""
    return model.embed.vocab_size, model.mtp[0].noise_token_id


# ---- cache state --------------------------------------------------------------------------------------------------


def capture(model) -> dict[str, torch.Tensor]:
    return {n: b.clone() for n, b in model.named_buffers() if not n.endswith("freqs_cis")}


def restore(model, snap: dict[str, torch.Tensor]) -> None:
    bufs = dict(model.named_buffers())
    for n, v in snap.items():
        bufs[n].copy_(v)


def live_view(model, n: int, *, draft_n: int | None = None) -> dict[str, torch.Tensor]:
    """The cache entries that still matter once `n` positions are committed: the last `window_size` ring slots, the
    compressed rows of completed groups, the tail of the open group, the indexer keys. Stale slots are left out, they
    legitimately differ between two histories. The DSpark stages' windows are viewed at `draft_n` positions."""
    out: dict[str, torch.Tensor] = {}
    if model.engram_hash is not None:
        out["engram_hash.cache"] = model.engram_hash.cache[:, :n].clone()
    for name, mod in model.named_modules():
        attn = mod if hasattr(mod, "window_kv_cache") else None
        if attn is not None:
            count = draft_n if (name.startswith("mtp.") and draft_n is not None) else n
            win = attn.window_size
            slots = [p % win for p in range(max(0, count - win), count)]
            out[f"{name}.window"] = attn.window_kv_cache[:, slots].clone()
            if getattr(attn, "is_kv_source", False):
                ratio = attn.compress_ratio
                out[f"{name}.compress"] = attn.compress_kv_cache[:, : n // ratio].clone()
                comp = attn.compressor
                if ratio > 1 and n % ratio:
                    out[f"{name}.kv_state"] = comp.kv_state[:, : n % ratio].clone()
                    out[f"{name}.score_state"] = comp.score_state[:, : n % ratio].clone()
                if attn.indexer is not None and attn.indexer.owns_k:
                    out[f"{name}.index_k"] = attn.indexer.k_cache[:, : n // ratio].clone()
    return out


# ---- protocol adapters --------------------------------------------------------------------------------------------


class OfficialTarget:
    """`dspark.Target` over the official Transformer (single-token decode steps, snapshot rollback)."""

    def __init__(self, model):
        self.model = model
        self._verified: tuple[int, list[int], dict] | None = None
        self.length = 0

    def reset(self) -> None:
        restore(self.model, self.model.fresh_state)
        self._verified, self.length = None, 0

    def _step(self, token: int, pos: int):
        _, logits, hidden = self.model.forward(torch.tensor([[token]]), pos)
        return logits[0], hidden[0, 0]

    def prefill(self, tokens):
        self.reset()
        _, logits, hidden = self.model.forward(torch.tensor([list(tokens)]), 0)
        self.length = len(tokens)
        return dspark.TargetOut(logits=logits, hidden=hidden[0])

    def verify(self, tokens, start_pos):
        assert start_pos == self.length, (start_pos, self.length)
        self._verified = (start_pos, list(tokens), capture(self.model))
        rows = [self._step(t, start_pos + i) for i, t in enumerate(tokens)]
        self.length = start_pos + len(tokens)
        return dspark.TargetOut(logits=torch.stack([r[0] for r in rows]), hidden=torch.stack([r[1] for r in rows]))

    def truncate(self, length):
        start, tokens, snap = self._verified
        assert start <= length <= start + len(tokens), (start, length, len(tokens))
        restore(self.model, snap)
        for i, t in enumerate(tokens[: length - start]):
            self._step(t, start + i)
        self.length = length


class OfficialDrafter:
    """`dspark.Drafter` over `Transformer.forward_spec`. Greedy drafts (stage temperature 0), or sampled at
    `temperature` > 0 with the laws returned as `q` (the stage draws from softmax(logits / T) of the returned logits)."""

    def __init__(self, model, temperature: float = 0.0):
        self.model = model
        self.block_size = model.mtp[0].block_size
        self.noise = model.mtp[0].noise_token_id
        self.temperature = temperature
        self.fed = 0                          # positions whose hidden state the window caches hold
        for stage in model.mtp:
            stage.temperature = temperature

    def draft(self, token, hidden, start_pos):
        n = hidden.shape[0]
        first = start_pos - n + 1
        if first == 0:                        # prompt: seed the windows with every row but the last (start_pos 0 path)
            assert n >= 2, "the official stack cannot draft from a one-token context"
            self.model.forward_spec(torch.tensor([token]), hidden[None, : n - 1], 0)
            first, hidden, n = n - 1, hidden[n - 1:], 1
        for j in range(n - 1):                # accepted positions: one window slot per call, outputs unused
            self.model.forward_spec(torch.tensor([self.noise]), hidden[j][None, None], first + j)
        ids, logits, conf = self.model.forward_spec(torch.tensor([token]), hidden[n - 1][None, None], start_pos)
        self.fed = start_pos + 1
        toks = [int(t) for t in ids[0, 1:]]
        scores = [float(c) for c in conf[0]]        # confidence head: one raw score per slot, known before the slot's token
        if self.temperature > 0:
            q = [torch.softmax(logits[0, i] / max(self.temperature, 1e-5), dim=-1) for i in range(self.block_size)]
            return dspark.DraftOut(toks, q, scores)
        return dspark.DraftOut(toks, conf=scores)


class OracleDrafter:
    """Proposes the true continuation `truth` (the plain loop's tokens, EOS included when it ended there) and replaces
    each token with a random one with probability `p_err`, so every acceptance length shows up. Not a model."""

    def __init__(self, truth, n_prompt, vocab, rng, p_err=0.2, block_size=5, conf_sigma=None):
        self.truth, self.n_prompt, self.vocab, self.rng = list(truth), n_prompt, vocab, rng
        self.p_err, self.block_size = p_err, block_size
        self.conf_sigma = conf_sigma          # not None: also return a synthetic confidence, +-2 plus N(0, sigma) noise

    def draft(self, token, hidden, start_pos):
        c = start_pos + 1 - self.n_prompt                  # index of the anchor in the completion
        assert self.truth[c] == token, (c, self.truth[c], token)
        toks, conf = [], []
        for i in range(self.block_size):
            t = self.truth[c + 1 + i] if c + 1 + i < len(self.truth) else int(self.rng.integers(self.vocab))
            if self.rng.random() < self.p_err:
                t = int(self.rng.integers(self.vocab))
            toks.append(t)
            if self.conf_sigma is not None:
                right = c + 1 + i < len(self.truth) and t == self.truth[c + 1 + i]
                conf.append((2.0 if right else -2.0) + float(self.rng.normal()) * self.conf_sigma)
        return dspark.DraftOut(toks, conf=conf or None)


def reference_generate(model, prompt, max_new, eos_id, temperature=0.0, seed=None):
    """The official generate.py loop (batch of one) from empty caches. Returns the completion, EOS cut off."""
    _, gen = load_official()
    restore(model, model.fresh_state)
    model.temperature = temperature
    try:
        if seed is not None:
            torch.manual_seed(seed)
        return gen.generate(model, [list(prompt)], max_new, eos_id)[0]
    finally:
        model.temperature = 0


def official_config() -> dict:
    return json.loads((REF / "config.json").read_text())
