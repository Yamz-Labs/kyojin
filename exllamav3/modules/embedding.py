from __future__ import annotations
import os
from typing_extensions import override
import os
import torch
from torch import nn
from ..model.config import Config
from ..util.tensor import to2
from ..util.device_copy import to_device
from . import Module
from ..tokenizer.mm_embedding import FIRST_MM_EMBEDDING_INDEX, mm_rows
from ..model.model_tp_alloc import TPAllocation

class Embedding(Module):

    def __init__(
        self,
        config: Config | None,
        key: str,
        vocab_size: int,
        hidden_size: int,
        out_dtype: torch.dtype | None = torch.float,
        qmap: str | None = None,
        normalize: bool = False,
        multiplier: float = 1.0
    ):
        super().__init__(config, key, None)
        assert qmap is None, "No quant scheme for Embedding"

        self.key = key
        self.embedding = None
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.out_dtype = out_dtype
        self._pinned_staging = {}
        self._numel = vocab_size * hidden_size
        self.normalize = normalize
        self.multiplier = multiplier

        self.caps.update({
            "prefer_cpu": True,
        })

    @override
    def optimizer_targets(self):
        return []

    @override
    def load(self, device: torch.device, **kwargs):
        self.device = device
        weight = None
        if self.device.type == "cpu" and os.environ.get("EXL3_EMB_MMAP", "1") != "0" and hasattr(self.config.stc, "get_tensor_mmap"):
            # The table is only gathered from: map it from the file (page cache) rather than holding a 2+ GiB anonymous copy
            weight = self.config.stc.get_tensor_mmap(self.key + ".weight")
        if weight is None:
            weight = self.config.stc.get_tensor(self.key + ".weight", self.device, float2half = True, allow_bf16 = True)
        self._numel = weight.numel()
        self.embedding = nn.Embedding(
            self.vocab_size,
            self.hidden_size,
            device = "meta"
        )
        self.embedding.weight = nn.Parameter(weight)

    @override
    def unload(self):
        self.device = None
        self.embedding = None

    @override
    def get_tensors(self):
        return {
            f"{self.key}.weight": self.embedding.weight.data.contiguous()
        }

    @override
    def weights_numel(self):
        return self._numel
        
    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:

        # Ensure input IDs in params
        if "input_ids" not in params:
            params["input_ids"] = x

        indexed_emb = params.get("indexed_embeddings")
        input_ids = x
        explicit_dtype = out_dtype is not None
        out_dtype = out_dtype or self.out_dtype or x.dtype

        # Indexed embedding masks
        if indexed_emb:
            standard_mask = input_ids < FIRST_MM_EMBEDDING_INDEX
            indexed_masks = [
                (input_ids >= e.first_index) & (input_ids < (e.first_index + e.mm_length))
                for e in indexed_emb
            ]
            indexed_act = [im.any() for im in indexed_masks]
            use_indexed_emb = any(indexed_act)

        # Mixed embeddings when needed
        if indexed_emb and use_indexed_emb:
            bsz, seq_len = input_ids.shape
            combined_emb = torch.empty((bsz, seq_len, self.hidden_size), device = self.device, dtype = out_dtype)

            # Prepare deepstack embedding tensors
            if any(ie.deepstack_embeddings is not None for ie in indexed_emb) and indexed_act:
                assert all(ie.deepstack_embeddings is not None for ie in indexed_emb)
                num_layers = len(indexed_emb[0].deepstack_embeddings)
                assert all(num_layers == len(ie.deepstack_embeddings) is not None for ie in indexed_emb)
                deepstack_emb = [torch.zeros_like(combined_emb) for _ in range(num_layers)]
            else:
                deepstack_emb = None

            # Insert standard embeddings
            if standard_mask.any():
                for i in range(bsz):
                    standard_ids_row = input_ids[i][standard_mask[i]]
                    standard_emb_row = self.embedding(standard_ids_row)
                    combined_emb[i][standard_mask[i]] = standard_emb_row.to(out_dtype)

            # Only normalize standard embeddings
            if self.normalize:
                combined_emb *= combined_emb.shape[-1] ** 0.5

            # Also only scale standard embeddings
            if self.multiplier != 1.0:
                combined_emb *= self.multiplier

            # Insert indexed embeddings
            for im, ie, act in zip(indexed_masks, indexed_emb, indexed_act):
                if not act:
                    continue
                for i in range(bsz):
                    indexed_ids_row = input_ids[i][im[i]] - ie.first_index
                    # mm_rows() is the same gather when the rows are a plain tensor, and dequantizes 8 bit
                    # rows on the device when a server stores them that way (see QuantizedMMRows)
                    combined_emb[i][im[i]] = mm_rows(ie, indexed_ids_row, out_dtype, combined_emb.device)

                    # Prepare deepstack embeddings
                    if ie.deepstack_embeddings is not None:
                        for layer, de in enumerate(ie.deepstack_embeddings):
                            deepstack_emb[layer][i][im[i]] = de[indexed_ids_row].to(out_dtype)

            # Save deepstack embeddings to params
            if deepstack_emb is not None:
                params["deepstack_emb"] = deepstack_emb

            return combined_emb

        # No indexed embeddings, or none in current batch
        else:
            dev = params.get("emb_next_device")
            if dev is not None and not explicit_dtype and self._narrow_upload_ok(x, params):
                return self._upload_narrow(self._gather_narrow(x), x, out_dtype, dev)
            x = self.embedding.forward(x)
            if self.multiplier != 1.0:
                x *= self.multiplier
            x = to2(x, out_dtype, self.out_dtype)
            if self.normalize:
                x *= x.shape[-1] ** 0.5
            # When the embedding resides on the CPU, its output is uploaded to the first
            # device layer; staging it through a reused pinned buffer makes that upload
            # asynchronous. Only callers that guarantee a sync point between forward passes
            # (the generator's decode loop) may set the pinned_staging flag.
            if params.get("pinned_staging") and x.device.type == "cpu":
                key = (x.shape, x.dtype)
                buf = self._pinned_staging.get(key)
                if buf is None:
                    if len(self._pinned_staging) > 8:
                        self._pinned_staging.clear()
                    buf = torch.empty_like(x, pin_memory = True)
                    self._pinned_staging[key] = buf
                buf.copy_(x)
                x = buf
            return x

    # EXL3_PF_EMB_UPLOAD=1: a long chunk (>= 1024 rows) of a CPU-resident embedding is gathered straight into a pinned buffer in the
    # table's own 16-bit dtype and uploaded asynchronously; the widening to out_dtype (exact, fp16/bf16 -> fp32) happens on the device
    # in Module.prepare_for_device (finish_narrow). The default path gathers, widens on the host and copies 4 bytes per element from
    # pageable memory (94 MB per 4583-row chunk: ~25 ms with the GPU idle, 3.8 GB/s). Same values bit for bit.
    def _narrow_upload_ok(self, ids: torch.Tensor, params: dict) -> bool:
        if os.environ.get("EXL3_PF_EMB_UPLOAD", "0") != "1" or ids.device.type != "cpu" or self.normalize or params.get("pinned_staging"):
            return False
        w = self.embedding.weight
        return w.device.type == "cpu" and w.dtype in (torch.half, torch.bfloat16) and ids.numel() >= 1024 and torch.cuda.is_available()

    def forward_device(self, ids: torch.Tensor, params: dict, out_dtype: torch.dtype, device) -> torch.Tensor:
        """forward(ids, params, out_dtype) followed by a move to `device`, with the long-chunk upload of _gather_narrow."""
        if ids.device.type == "cpu" and not params.get("indexed_embeddings") and self._narrow_upload_ok(ids, params):
            # the MTP block embeds the very rows the target forward just uploaded: reuse them (same bf16 -> fp16 rounding, on the device)
            hit = getattr(self, "_dev_rows", None)
            self._dev_rows = None
            if hit is not None and hit[0].shape == ids.shape and hit[1].device == torch.device(device) and torch.equal(hit[0], ids):
                return hit[1] if hit[1].dtype == out_dtype else hit[1].to(out_dtype)
            return self._upload_narrow(self._gather_narrow(ids), ids, out_dtype, device)
        return to_device(self.forward(ids, params, out_dtype = out_dtype), device)

    def _gather_narrow(self, ids: torch.Tensor) -> torch.Tensor:
        w = self.embedding.weight.detach()
        n = ids.numel()
        buf = getattr(self, "_narrow_buf", None)
        if buf is not None and getattr(self, "_narrow_ev", None) is not None:
            self._narrow_ev.synchronize()           # the previous upload out of this buffer has completed
        if buf is None or buf.shape[0] < n or buf.dtype != w.dtype:
            buf = torch.empty((n, self.hidden_size), dtype = w.dtype, pin_memory = True)
            self._narrow_buf = buf
            self._narrow_ev = None
        y = buf[:n]
        torch.index_select(w, 0, ids.reshape(-1), out = y)
        if self.multiplier != 1.0:
            y *= self.multiplier
        return y.view(*ids.shape, self.hidden_size)

    def _upload_narrow(self, y: torch.Tensor, ids: torch.Tensor, out_dtype, device) -> torch.Tensor:
        """Upload the pinned 16-bit rows without blocking the host and widen on the device (exact for bf16/fp16 -> fp32, the same
        round-to-nearest the host conversion does for bf16 -> fp16). The rows stay cached for the MTP block (forward_device)."""
        d = y.to(device, non_blocking = True)
        if self._narrow_ev is None:
            self._narrow_ev = torch.cuda.Event()
        self._narrow_ev.record()
        self._dev_rows = (ids.clone(), d)        # consumed by forward_device (the MTP block), else replaced by the next chunk
        return d if d.dtype == out_dtype else d.to(out_dtype)

    def make_tp_allocation(self, options: dict) -> list[TPAllocation]:
        return []

    def tp_export(self, plan, producer):
        assert self.device is not None, "Cannot export module for TP before loading."
        return {
            "cls": Embedding,
            "kwargs": {
                "key": self.key,
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size,
                "out_dtype": self.out_dtype,
                "normalize": self.normalize,
                "multiplier": self.multiplier,
            },
            "embedding.weight": producer.send(self.embedding.weight),
            "device": self.device
        }

    @staticmethod
    def tp_import(local_context, exported, plan):
        consumer = local_context["consumer"]
        module = Embedding(
            config = None,
            **exported["kwargs"],
        )
        module.device = exported["device"]
        module.embedding = nn.Embedding(
            module.vocab_size,
            module.hidden_size,
            device = "meta"
        )
        emb = consumer.recv(exported["embedding.weight"], cuda = False)
        module.embedding.weight = nn.Parameter(emb)
        return module
