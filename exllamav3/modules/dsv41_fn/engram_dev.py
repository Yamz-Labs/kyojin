# Device-side Engram (DeepSeek-V4.1 layers 1 and 14): hash on the GPU, rows fetched by the host row store (O_DIRECT), MXFP4 decode and
# the gate on the GPU. Replaces dsv41/engram.py::NgramHasher + engram_inject for the serving path; same maths, kernels in dsv41_hip.hip.
#
# Data flow per chunk of R tokens (decode: R = 1, or a draft block):
#   ids -> ds41_engram_comp (compressed ids into the per-sequence cache) -> ds41_engram_hash -> rows int64 [R, nl, 24]      (GPU)
#   rows of the layer -> host (R * 24 * 8 bytes) -> unique -> store.fetch (O_DIRECT, 136 B/row) -> pinned -> GPU           (I/O)
#   ds41_mxfp4_decode (with the dedup inverse map) -> emb [R, 6144] fp16 -> wkv (EXL3 Linear, callable) -> kv [R, 25600]    (GPU)
#   ds41_engram_gate: gate + h += gate * value in place on the fp32 streams                                                (GPU)
# The hash only depends on token ids, so the host can fetch the rows of every Engram layer as soon as the tokens are known
# (`rows_for(...)` returns them without touching h): layer 14's fetch then overlaps the 13 layers before it.
import numpy as np
import torch
from . import dsv41_hip as K


class EngramDevice:
    def __init__(self, layout, token_map, pad_token_id: int, multipliers, max_seq: int, store, device="cuda"):
        self.layout, self.store, self.device = layout, store, device
        nl, ng, nh = len(layout.layer_ids), layout.max_ngram_size, layout.n_heads
        self.nl, self.ng, self.nh, self.nc = nl, ng, nh, (ng - 1) * nh
        self.tmap = torch.as_tensor(np.asarray(token_map, dtype=np.int32)).to(device)
        self.pad_token_id, self.pad = pad_token_id, int(self.tmap[pad_token_id])
        self.mult = torch.as_tensor(np.asarray(multipliers, dtype=np.int64)).to(device).contiguous()            # [nl, ng]
        self.primes = torch.tensor(layout.primes, dtype=torch.int64).flatten(1).to(device).contiguous()         # [nl, nc]
        self.offsets = torch.tensor(np.stack([layout.offsets(i) for i in range(nl)]), dtype=torch.int64).to(device).contiguous()
        self.cache = torch.full((max_seq,), -1, dtype=torch.int64, device=device)

    def reset(self):
        self.cache.fill_(-1)
        if getattr(self, "prefetcher", None) is not None:
            self.prefetcher.reset()

    def attach_prefetcher(self, cache_bytes: int = 1 << 30, max_rows: int = 2048, slots: int = 3, token_map=None, pad_token_id=None,
                          multipliers=None, max_seq: int = 4096):
        """Async host-side fetch (engram_prefetch.py). Submit at step start with `prefetch(ids_host, start)`, pass the ticket to
        inject(ticket=...) and call prefetcher.release(ticket) once the step's kernels are enqueued."""
        from .engram_prefetch import EngramPrefetcher
        self.prefetcher = EngramPrefetcher(self.layout, token_map if token_map is not None else self.tmap.cpu().numpy(),
                                           self.pad_token_id if pad_token_id is None else pad_token_id,
                                           multipliers if multipliers is not None else self.mult.cpu().numpy(), self.store,
                                           max_seq=max_seq, cache_bytes=cache_bytes, device=self.device, max_rows=max_rows, slots=slots)
        return self.prefetcher

    def prefetch(self, ids_host, start: int):
        return self.prefetcher.submit(ids_host, start)

    def speculate(self, cands, start: int):
        """See EngramPrefetcher.speculate: call right after the logits of the previous step, before sampling is final."""
        return self.prefetcher.speculate(cands, start)

    def hash(self, ids: torch.Tensor, start: int, mask: torch.Tensor = None) -> torch.Tensor:
        """ids int64 [R] (device) at positions start .. start + R - 1 -> table rows int64 [R, nl, nc]."""
        R = ids.numel()
        K.engram_comp(ids, self.tmap, self.cache, mask, start)
        rows = torch.empty(R, self.nl, self.nc, dtype=torch.int64, device=self.device)
        K.engram_hash(self.cache, self.mult, self.primes, self.offsets, rows, start, R, self.pad)
        return rows

    def fetch(self, layer_pos: int, rows_l: torch.Tensor):
        """rows_l int64 [R, nc] (device) -> (raw uint8 [U, 136 or 264] on device, inverse int32 [R * nc] on device)."""
        flat = rows_l.reshape(-1).cpu()
        uniq, inv = torch.unique(flat, return_inverse=True)
        raw = self.store.fetch(layer_pos, uniq)
        return raw.to(self.device, non_blocking=True), inv.int().to(self.device, non_blocking=True)

    def inject(self, h: torch.Tensor, rows_l, layer_pos: int, wkv, qk: torch.Tensor, eps: float, fetched=None, ticket=None):
        """h fp32 [R, 4, 5120] in place. rows_l: this layer's rows [R, nc]; wkv: callable [R, 6144] fp16 -> [R, 25600] fp16;
        qk = q_weight * k_weight fp32 [4, 5120]; fetched = (raw, inv) from an earlier fetch() (prefetch) or None;
        ticket = an EngramPrefetcher ticket (engram_prefetch.py) submitted at step start: rows_l is then unused (may be None)."""
        R = h.size(0)
        if ticket is not None:
            fetched = self.prefetcher.take(ticket, layer_pos)
        raw, inv = fetched if fetched is not None else self.fetch(layer_pos, rows_l)
        emb = torch.empty(R * self.nc, 256, dtype=torch.half, device=self.device)
        (K.mxfp4_decode if raw.size(1) == 136 else K.fp8_decode)(raw, emb, inv)     # 136 B MXFP4 rows, 264 B official FP8 rows
        kv = wkv(emb.view(R, self.nc * 256))
        K.engram_gate(h, kv.contiguous(), qk, eps)
        return h
