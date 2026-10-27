"""
Routed-expert backends of the DS4.1 serving model.

The model computes the router itself (sqrtsoftplus, bias only steers the selection, weights from the unbiased scores,
x1.5) and hands `(x, weights, picks)` to a backend that returns the weighted sum of the picked experts in fp32:

  LoopMoE      per expert, only the experts the router picked (any device, any storage: fp16 weights, EXL3 Linear, K-agnostic).
               The shape-agnostic fallback and the CPU path of the tests; at decode it is 6 x 3 small GEMVs per layer.
  GroupedMoE   the mixed-K grouped HIP kernels (`moe_mixedk.MixedKStore`): per-expert bitrates inside a layer (hot-3 / cold-2,
               `quantization_config.expert_bits`), one launch set per K group. Decode (rows <= 16): `exl3_moe_gfx12_k3`
               with the DS4.1 swiglu clamp (`act_limit` = swiglu_limit), prefill (rows >= 32): `exl3_moe_prefill_wmma`.
               Rows in between, or a layer the kernels cannot take, go through the LoopMoE fallback.

The shared expert is not part of a backend (a plain gated MLP on Linear objects, added by the model).
"""
from __future__ import annotations
import os
import torch
from . import ops


class LoopMoE:
    name = "loop"

    def __init__(self, provider, cfg):
        self.provider, self.cfg = provider, cfg

    def run(self, l: int, x: torch.Tensor, wts: torch.Tensor, idx: torch.Tensor, n_exp: int) -> torch.Tensor:
        """x [n, D] (activation dtype), wts [n, k] fp32, idx [n, k] int64 -> fp32 [n, D]."""
        y = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        counts = torch.bincount(idx.flatten(), minlength=n_exp).tolist()
        used = [e for e in range(n_exp) if counts[e]]
        self.provider.prefetch(l, used)
        limit = self.cfg.swiglu_limit
        for e in used:
            h = self.provider.lookup(l, e)
            ti, top = torch.where(idx == e)
            y[ti] += h.ffn(x[ti], wts[ti, top, None], limit)
            self.provider.release(h)
        return y


def grouped_available(device) -> bool:
    dev = torch.device(device)
    if dev.type != "cuda" or os.environ.get("DSV41_GROUPED_MOE", "1") == "0":
        return False
    try:
        from ..ext import exllamav3_ext as ext
        return (hasattr(ext, "exl3_moe_gfx12_k3") and hasattr(ext, "exl3_moe_prefill_wmma") and
                bool(ext.exl3_gemv_supported(dev.index if dev.index is not None else torch.cuda.current_device())))
    except Exception:
        return False


_NATIVE_TOPK = {3: os.environ.get("EXL3_MOE_TOPK3", "0") == "1"}   # top-k values the loaded extension decodes natively (3 needs the rebuilt extension)


class GroupedMoE:
    """One MoE layer on the mixed-K grouped kernels. `gates/ups/downs` are the loaded per-expert Linear modules of the layer
    (EXL3, one K per expert, gate K == up K); the store builds one pointer-table triple per distinct K."""

    name = "grouped"

    def __init__(self, gates, ups, downs, dcfg, device, expert_bits=None, fallback: LoopMoE = None, layer: int = -1):
        from ..modules.moe_mixedk import MixedKStore, check_expert_bits
        from ..modules.block_sparse_mlp_routing import _HIP_GROUPED_MAX_ROWS
        from ..util.tensor import g_tensor_cache
        self.device = torch.device(device)
        self.dcfg, self.layer, self.fallback = dcfg, layer, fallback
        self.store = MixedKStore(self.device, gates, ups, downs)
        if expert_bits is not None:
            check_expert_bits(expert_bits, self.store)
        self.H = dcfg.hidden_size
        self.I = (dcfg.moe_intermediate_size + 127) // 128 * 128
        self.max_rows = _HIP_GROUPED_MAX_ROWS
        n = len(gates)
        ma = self.max_rows * n
        # projection-major scratch shared by every layer of the device (views are taken per call)
        c = lambda shape, dt, tag: g_tensor_cache.get(self.device, shape, dt, tag)
        self.buffers = _Buf(
            gu_had=c((2 * ma, self.H), torch.half, "moe_gfx12_gu_had"),
            gu_out=c((2 * ma, self.I), torch.half, "moe_gfx12_gu_out"),
            down_had=c((ma, self.I), torch.half, "moe_gfx12_down_had"),
            down_out=c((ma, self.H), torch.float, "moe_gfx12_down_out"),
            output=c((self.max_rows, self.H), torch.float, "moe_gfx12_output"))
        self.outputs = [self.buffers.output] + [
            c((self.max_rows, self.H), torch.float, f"moe_gfx12_output_k{i}") for i in range(1, self.store.num_groups)]
        self.decode_ok = self.store.decode_ok
        self.wmma_ok = self.store.wmma_ok
        self.prefill_min = int(os.environ.get("EXL3_MOE_WMMA_MIN_ROWS", "32"))
        self.align_chunk = int(os.environ.get("EXL3_MOE_ALIGN_CHUNK", "4096"))
        self.calls = {"decode": 0, "prefill": 0, "fallback": 0}
        self.uni = None
        # K=4 experts run the WMMA body in the per-group path (k-split follows EXL3_MOE_CFG); the unified kernel is the VALU
        # matvec (k-split 16, = CFG 0), so K=4 joins it only when the two are bit-identical
        ks_ok = (1.0, 2.0, 3.0, 4.0) if os.environ.get("EXL3_MOE_CFG", "0") in ("", "0") else (1.0, 2.0, 3.0)
        if self.decode_ok and self.store.num_groups <= 3 and all(g.k_gu in ks_ok for g in self.store.groups):
            from ..modules.dsv41_fn import dsv41_moe as UM
            if UM.available(self.device):
                self.uni = UM.UnifiedMoE(self.store, self.buffers, self.outputs[:self.store.num_groups], self.H, self.I,
                                         float(dcfg.swiglu_limit), self.device)
        self._uni_ver = None
        self.tier = None        # expert_store.TierSet when the SSD expert tier holds this layer's experts

    def prepare(self, idx):
        """Tiered layer: make the experts of `idx` ([rows, k] picks) resident before any kernel of this layer runs."""
        if self.tier is not None:
            self.tier.store.prepare(self.tier, idx, {}, None)
            if self.uni is not None:
                # the unified kernel reads a concatenated copy of the pointer tables: refresh it when the tier moved slots
                ver = tuple(g.version for g in self.tier.groups)
                if ver != self._uni_ver:
                    self.uni.refresh()
                    self._uni_ver = ver

    def run(self, l, x, wts, idx, n_exp):
        from ..ext import exllamav3_ext as ext
        from ..modules.block_sparse_mlp import _hip_wmma_workspace, _scatter_expert_count
        rows = x.size(0)
        y = x if x.dtype == torch.half else x.half()
        y = y.contiguous()
        sel = idx.contiguous()
        limit = float(self.dcfg.swiglu_limit)
        if rows <= self.max_rows and self.decode_ok:
            self.calls["decode"] += 1
            if sel.size(1) == 3 and not _NATIVE_TOPK[3]:
                # the grouped decode kernel is instantiated for top-k 6 / 8 / 10: repeat the picks with weight 0 (exact: adds 0)
                r = 2
                sel = torch.cat([sel] * r, dim=1)[:, :6].contiguous()
                wts = torch.cat([wts, torch.zeros_like(wts).repeat(1, r - 1)], dim=1)[:, :6]
            out = self.store.decode(ext, y, sel, wts.half().contiguous(), self.buffers, self.outputs, limit, True)
            return out[:rows].float().clone()
        if rows < self.prefill_min and self.decode_ok:
            # rows between the decode kernel's limit and the WMMA minimum: run the decode kernel on slices of max_rows.
            # The LoopMoE fallback gave wrong logits here under the SSD tier (17-31 row prompts / last chunks, 20/10).
            # Per-row math, so this equals decoding the rows one slice at a time.
            self.calls["decode"] += 1
            if sel.size(1) == 3 and not _NATIVE_TOPK[3]:
                sel = torch.cat([sel] * 2, dim=1)[:, :6].contiguous()
                wts = torch.cat([wts, torch.zeros_like(wts)], dim=1)[:, :6]
            wh = wts.half().contiguous()
            parts = []
            for a in range(0, rows, self.max_rows):
                b = min(rows, a + self.max_rows)
                out = self.store.decode(ext, y[a:b].contiguous(), sel[a:b].contiguous(), wh[a:b].contiguous(),
                                        self.buffers, self.outputs, limit, True)
                parts.append(out[:b - a].float().clone())
            return torch.cat(parts, dim=0)
        if rows >= self.prefill_min and self.wmma_ok:
            self.calls["prefill"] += 1
            return self.store.prefill(
                ext, y, sel, wts.float().contiguous(),
                lambda a, e: _hip_wmma_workspace(self.device, a, self.H, self.I, e), _scatter_expert_count,
                limit, self.H, self.I, sh_add=None, chunk=self.align_chunk)
        self.calls["fallback"] += 1
        return self.fallback.run(l, x, wts, idx, n_exp)

    def fused_ok(self, x) -> bool:
        return x.size(0) == 1 and self.decode_ok and x.dtype == torch.half and 1 <= self.store.num_groups <= 3

    def decode_parts(self, x, sel, wh, gsel):
        """Decode (1 row) without the torch glue of `run`: picks `sel`, fp16 weights `wh` and the per-group slots `gsel` [G, 1, k]
        come from the fused router. Returns the K-group outputs (fp32 [1, H] views of the persistent buffers); their ordered
        sum is what `run` returns (MixedKStore.decode adds them in place in group order)."""
        from ..ext import exllamav3_ext as ext
        self.calls["decode"] += 1
        y = x.contiguous()
        if self.uni is not None and sel.shape[1] == 6:
            return [o[:1] for o in self.uni.run(y, sel, wh)]
        limit = float(self.dcfg.swiglu_limit)
        outs = []
        for g in self.store.groups:
            out = self.outputs[g.index][:1]
            ext.exl3_moe_gfx12_k3(y, out, sel if g.identity else gsel[g.index], wh, *g.tables(),
                                  self.buffers.gu_had[:2 * sel.shape[1]], self.buffers.gu_out[:2 * sel.shape[1]],
                                  self.buffers.down_had[:sel.shape[1]], self.buffers.down_out[:sel.shape[1]],
                                  float(g.k_gu), True, limit)
            outs.append(out)
        return outs

    def unload(self):
        self.store.unload()


class _Buf:
    def __init__(self, **kw):
        self.__dict__.update(kw)
