"""Fused MoE half-layer decode (EXL3_MOE_FUSED=1, ROCm gfx11.5, default off).

One persistent launch per MoE half-layer replaces `mlp_hc.mix`, `mlp.forward` and `mlp_hc.apply_` for R = 1..4 rows (decode and short verify
rounds): HC dots + finalize, router, top-k, expert gate/up/down VALU matvecs on weights repacked in read order, Hadamard post, weighted sum,
shared expert, gated-residual apply in place on the stack. Kernel: quant/exl3_moe_fused.cuh. Byte-identical to the flag-off engine at R > 1
and row-identical at R = 1 (one reduction order for every R, same as EXL3_MOE_VALU).

Weights (core5): the kernel reads the engine's own trellis tensors through pointer tables (layout "native", zero extra memory). The
EXL3_MOE_FUSED_LAYOUT=repack option builds the older 32-column-group copy for A/B timing (extra memory, reported by `extra_bytes`).
Only the router / shared-gate weights are copied when they are not already contiguous half.
"""
from __future__ import annotations
import os
import time
import torch
from ..ext import exllamav3_ext as ext
from .hyperconnections import HyperConnection

MAX_ROWS = 4
# EXL3_VERIFY_ROW_SPLIT=1: 5..8 rows of one sequence run as fused sub-calls of at most 4 rows (per-row arithmetic unchanged)
SPLIT_MAX_ROWS = 8 if os.environ.get("EXL3_VERIFY_ROW_SPLIT", "0") != "0" else MAX_ROWS


# EXL3_MOE_FUSED9=<variant>: 0 off, 393216 (0x60000) = forced-inline phases + two blocks per WGP (core9), 917504 = same kernel, ONE block per WGP, 1441792 (0x160000, default) = same phases and block partition as 0x60000, but every grid barrier is a kernel boundary (six stage launches):
# no block waits for another, so a desktop client on the GPU cannot stall it or change the output; 0.58 ms/token faster than 917504. 393216 / 917504 keep the barrier kernels.
# The barrier kernels need every block resident, and two 512-thread blocks fill a WGP's VGPR file exactly, so a desktop client on the same GPU can leave half the grid unscheduled. Runtime-mutable for one-load A/B.
MF9 = {"variant": int(os.environ.get("EXL3_MOE_FUSED9", "1441792"), 0), "timing": 0,   # timing 1 = phase stamps into seldbg (tests)
       # ballot top-k in the expert selection (variant bit 0x1000), bit-identical to the plain loop; EXL3_MOE_TK2=0 turns it off
       "tk2": os.environ.get("EXL3_MOE_TK2", "1") not in ("", "0")}


def enabled_by_env() -> bool:
    return bool(torch.version.hip) and os.environ.get("EXL3_MOE_FUSED", "0") not in ("", "0")


def _rep(tr: torch.Tensor) -> torch.Tensor:
    """(kt, nt, nw) int16 trellis -> (nt/2, kt, 2*nw) int16: one 32-column group per leading index, k-slice major."""
    kt, nt, nw = tr.shape
    assert nt % 2 == 0
    return tr.reshape(kt, nt // 2, 2 * nw).permute(1, 0, 2).contiguous()


class FusedMoEHalf:
    """Static plan + workspace of one qwen4_exp-style MoE half-layer (HC sites, BlockSparseMLP + shared expert)."""

    @staticmethod
    def why_not(block) -> str | None:
        """None if the block is eligible, else the reason (kept for the verdict / debugging)."""
        from .block_sparse_mlp_routing import routing_std
        if not hasattr(ext, "exl3_moe_fused_half"):
            return "extension has no exl3_moe_fused_half"
        gr, mlp = block.mlp_hc, block.mlp
        if gr is None or mlp is None or block.mlp_norm is not None:
            return "needs mlp_hc, mlp and no mlp_norm"
        if not getattr(gr, "use_q8", False) or gr.hc_mult != 4:
            return "GatedResidual int8 mixer (use_q8) with 4 streams required"
        if getattr(mlp, "routing_fn", None) is not routing_std or mlp.routing_gate is None or mlp.shared_gate is None or mlp.shared_experts is None:
            return "needs routing_std, shared_gate and shared_experts"
        if getattr(mlp.routing_gate.inner, "bias", None) is not None or mlp.shared_experts_post_norm is not None:
            return "router bias / shared post norm"
        if len(mlp.gates) != mlp.num_experts or len(mlp.ups) != mlp.num_experts or len(mlp.downs) != mlp.num_experts:
            return "experts are not one linear each"
        return None

    def __init__(self, block):
        t0 = time.time()
        gr, mlp = block.mlp_hc, block.mlp
        why = self.why_not(block)
        assert why is None, why
        dev = gr.fn_q8.device
        self.dev = dev
        self.gr, self.mlp = gr, mlp
        self.D = mlp.hidden_size
        self.H = gr.hc_mult
        self.LR = gr.fn_q8.shape[0] - self.H
        self.NEXP = mlp.num_experts
        self.TOPK = mlp.num_experts_per_tok
        self.INTER = mlp.intermediate_size
        g0, u0, d0 = mlp.gates[0].inner, mlp.ups[0].inner, mlp.downs[0].inner
        sh = mlp.shared_experts
        sg, su, sd = sh.gates[0].inner, sh.ups[0].inner, sh.downs[0].inner
        self.RB = int(round(g0.K))
        self.SB = int(round(sg.K))
        assert g0.K == self.RB and sg.K == self.SB, "integer K only (K2.5 not supported)"
        for l in (g0, u0, d0, sg, su, sd):
            assert l.mul1 and not l.mcg, "mul1 codebook required"
        self.supported = bool(ext.exl3_moe_fused_supported(self.D, self.H, self.LR, self.NEXP, self.TOPK, self.INTER, self.RB, self.SB))
        if not self.supported:
            self.extra_bytes = 0
            return
        self.shape = (self.D, self.H, self.LR, self.NEXP, self.TOPK, self.INTER, self.RB, self.SB)
        E = self.NEXP
        self.native = os.environ.get("EXL3_MOE_FUSED_LAYOUT", "native") != "repack"
        lin = [(mlp.gates[e].inner, mlp.ups[e].inner, mlp.downs[e].inner) for e in range(E)] + [(sg, su, sd)]
        wt, svt, keep = [], [], []
        self.extra_bytes = 0
        if self.native:
            # zero copy: the tables point at the engine's own trellis and scale-vector tensors (they stay alive through the modules)
            for tri in lin:
                for l in tri:
                    assert l.trellis.dtype == torch.int16 and l.trellis.is_contiguous() and l.trellis.device == dev
                    wt.append(l.trellis.data_ptr())
                for l in tri:
                    for v in (l.suh, l.svh):
                        assert v.dtype == torch.half and v.is_contiguous() and v.device == dev
                svt += [tri[0].suh.data_ptr(), tri[0].svh.data_ptr(), tri[1].suh.data_ptr(), tri[1].svh.data_ptr(), tri[2].suh.data_ptr(), tri[2].svh.data_ptr()]
        else:
            # A/B layout: 32-column groups, k-slice major, one copy (the originals stay: prefill and the flag-off path read them)
            mats = [[_rep(l.trellis).reshape(-1) for l in tri] for tri in lin]
            tot = sum(m.numel() for tri in mats for m in tri)
            self.rep = torch.empty(tot, dtype=torch.int16, device=dev)
            o = 0
            for tri in mats:
                for m in tri:
                    self.rep[o: o + m.numel()].copy_(m)
                    wt.append(self.rep.data_ptr() + 2 * o)
                    o += m.numel()
            self.svs = torch.stack([torch.cat([t[0].suh, t[0].svh, t[1].suh, t[1].svh, t[2].suh, t[2].svh]).half() for t in lin]).contiguous()
            sl = self.svs.shape[1]
            for i in range(E + 1):
                base = self.svs.data_ptr() + 2 * i * sl
                offs = [0, self.D, self.D + self.INTER, 2 * self.D + self.INTER, 2 * self.D + 2 * self.INTER, 2 * self.D + 3 * self.INTER]
                svt += [base + 2 * q for q in offs]
            self.extra_bytes += self.rep.numel() * 2 + self.svs.numel() * 2
        self.wt = torch.tensor(wt, dtype=torch.int64, device=dev)
        self.svt = torch.tensor(svt, dtype=torch.int64, device=dev)
        assert self.wt.numel() == 3 * (E + 1) and self.svt.numel() == 6 * (E + 1)
        self.extra_bytes += self.wt.numel() * 8 + self.svt.numel() * 8
        rw = mlp.routing_gate.inner.weight
        rr = rw.t() if tuple(rw.shape) == (self.D, self.NEXP) else rw
        self.router = rr if (rr.dtype == torch.half and rr.is_contiguous()) else rr.half().contiguous()   # kernel wants [NEXP][D]; alias when already so
        sw = mlp.shared_gate.inner.weight
        self.sgate = sw.reshape(-1) if (sw.dtype == torch.half and sw.is_contiguous()) else sw.half().reshape(-1).contiguous()
        if self.router.data_ptr() != rr.data_ptr():
            self.extra_bytes += self.router.numel() * 2
        if self.sgate.data_ptr() != sw.data_ptr():
            self.extra_bytes += self.sgate.numel() * 2
        assert self.router.shape == (self.NEXP, self.D) and self.sgate.numel() == self.D
        off = ext.exl3_moe_fused_ws_offsets(*self.shape)
        self.off = off
        self.ws = torch.zeros(off[-1], dtype=torch.uint8, device=dev)
        self.grid = int(ext.exl3_moe_fused_grid(*self.shape))
        self.extra_bytes += self.ws.numel()
        torch.cuda.synchronize()
        self.build_seconds = time.time() - t0

    # ---- views of the workspace (tests / debugging)
    def ws_view(self, name: str, R: int):
        idx = {"dots": 1, "post": 2, "mixed": 3, "scores": 4, "sgl": 5, "gu": 6, "dn": 7, "ydbg": 8, "seldbg": 9}[name]
        dt = {"dots": torch.float, "post": torch.float, "mixed": torch.half, "scores": torch.half, "sgl": torch.float,
              "gu": torch.half, "dn": torch.float, "ydbg": torch.float, "seldbg": torch.int32}[name]
        nbytes = self.off[idx + 1] - self.off[idx]
        return self.ws[self.off[idx]: self.off[idx] + nbytes].view(dt)

    def ctl(self):
        return self.ws[: 12].view(torch.int32)      # [bar, done, err]

    def applicable(self, x: torch.Tensor, params: dict) -> bool:
        if not self.supported or "capture" in params or "quant_preserve" in params:
            return False
        if x.dtype != torch.float or not x.is_contiguous() or x.dim() != 4 or x.shape[2] != self.H or x.shape[3] != self.D:
            return False
        return 1 <= x.shape[0] * x.shape[1] <= SPLIT_MAX_ROWS

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """x (b, s, H, D) fp32 residual stack, updated in place and returned."""
        if x.shape[0] * x.shape[1] > MAX_ROWS:
            assert x.shape[0] == 1
            for lo in range(0, x.shape[1], MAX_ROWS):
                self(x[:, lo:lo + MAX_ROWS])
            return x
        gr = self.gr
        fq, fsc, uq, usc = gr.q8_set()
        if MF9["variant"] and self.native and hasattr(torch.ops, "mf9") and hasattr(torch.ops.mf9, "half"):
            # core9 kernel (forced-inline phases, two blocks per WGP), byte-identical to the kernel below; same workspace layout
            torch.ops.mf9.half(x, fq, fsc, uq, usc, gr.w_h, self.router, self.sgate,
                               self.wt, self.svt, self.ws, gr.rms_eps, 1, MF9["timing"], MF9["variant"] | (0x1000 if MF9["tk2"] else 0), *self.shape)
            return x
        ext.exl3_moe_fused_half(x, fq, fsc, uq, usc, gr.w_h, self.router, self.sgate,
                                self.wt, self.svt, self.ws, gr.rms_eps, int(self.native), *self.shape)
        return x


def block_status(block) -> str | None:
    """None if the fused half-layer will run for this block, else one short reason (no weights are touched)."""
    why = FusedMoEHalf.why_not(block)
    if why is not None:
        return why
    try:
        gr, mlp = block.mlp_hc, block.mlp
        g0, sg = mlp.gates[0].inner, mlp.shared_experts.gates[0].inner
        rb, sb = int(round(g0.K)), int(round(sg.K))
        if g0.K != rb or sg.K != sb:
            return f"fractional K (routed {g0.K}, shared {sg.K})"
        if not (g0.mul1 and not g0.mcg and sg.mul1 and not sg.mcg):
            return "codebook is not mul1"
        H = gr.hc_mult
        if not ext.exl3_moe_fused_supported(mlp.hidden_size, H, gr.fn_q8.shape[0] - H, mlp.num_experts, mlp.num_experts_per_tok,
                                            mlp.intermediate_size, rb, sb):
            return f"routed K{rb} / shared K{sb} at this shape is not instantiated"
    except Exception as exc:                                              # noqa: BLE001
        return f"status check failed: {exc!r}"
    return None


def report(model, label: str = "") -> str | None:
    """One log line per model load: which MoE layers run the fused half-layer and which fall back to the slow path, and why.
    Silent when EXL3_MOE_FUSED is off or the model has no fusable MoE layers. Returns the line (also printed to stderr)."""
    if not enabled_by_env():
        return None
    ok, bad = [], {}
    for m in getattr(model, "modules", []):
        if not hasattr(m, "_moe_fused_on") or getattr(m, "mlp", None) is None or getattr(m, "mlp_hc", None) is None:
            continue
        if not hasattr(m.mlp, "gates"):
            continue
        idx = getattr(m, "layer_idx", None)
        idx = idx if idx is not None else len(ok) + sum(len(v) for v in bad.values())
        why = block_status(m)
        if why is None:
            ok.append(idx)
        else:
            bad.setdefault(why, []).append(idx)
    if not ok and not bad:
        return None
    def rng(v):
        v = sorted(v); out, a = [], v[0]
        for i, x in enumerate(v):
            if i + 1 == len(v) or v[i + 1] != x + 1:
                out.append(f"{a}" if a == x else f"{a}-{x}"); a = v[i + 1] if i + 1 < len(v) else None
        return ",".join(out)
    nb = sum(len(v) for v in bad.values())
    if nb == 0:
        line = f"[moe_fused]{label} all {len(ok)} MoE layers run the fused half-layer"
    else:
        det = "; ".join(f"layers {rng(v)}: {k}" for k, v in sorted(bad.items(), key=lambda kv: kv[1][0]))
        line = f"[moe_fused]{label} WARNING {nb} of {nb + len(ok)} MoE layers are NOT fused (slow path): {det}"
    import sys
    print(line, file=sys.stderr, flush=True)
    return line


def maybe_apply(block, x: torch.Tensor, params: dict, hc_pending):
    """Hook for TransformerBlock._forward_mlp. Returns the new residual, or None to fall through to the unfused path."""
    if not block._moe_fused_on:
        return None
    fh = block._moe_fused
    if fh is None:
        if block._moe_fused_tried:
            return None
        block._moe_fused_tried = True
        if FusedMoEHalf.why_not(block) is not None:
            return None
        fh = FusedMoEHalf(block)
        block._moe_fused = fh
    if not fh.applicable(x, params):
        return None
    if hc_pending is not None:
        HyperConnection.flush_pending(x, hc_pending)
    return fh(x)


_CHK = {"n": 0, "on": True}      # "on" False only in the determinism harness arms that deliberately run the two-blocks-per-WGP grid


def check_barrier(model, every: int = 8, force: bool = False) -> None:
    """Raise if any fused half-layer grid barrier timed out (its workgroups would have read partial results). Reads the ctl record of every layer
    with one gather every `every` calls (one decode step per call), or at once with force=True."""
    if not _CHK["on"] and not force:
        return
    _CHK["n"] += 1
    if not force and _CHK["n"] % every:
        return
    fhs = [(m.key, m._moe_fused) for m in getattr(model, "modules", []) if getattr(m, "_moe_fused", None) is not None]
    if not fhs:
        return
    rec = torch.cat([f.ws[8:44].view(torch.int32) for _, f in fhs]).cpu().view(len(fhs), 9)   # ctl words 2..10
    bad = (rec[:, 0] != 0) | (rec[:, 1] != 0)
    if bad.any():
        i = int(bad.nonzero()[0])
        e, cnt, ph, blk, seen, ticks, miss, _, _ = [int(v) for v in rec[i]]
        with torch.inference_mode():
            for _, f in fhs:
                f.ws[8:64].zero_()
        raise RuntimeError(f"fused MoE grid barrier timed out in {fhs[i][0]} (phase {e}, {cnt} timeouts, block {blk} saw {seen} arrivals after {ticks / 1e5:.0f} ms, "
                           f"first missing block {miss}): the output of this step is invalid; the GPU is shared and not all workgroups were resident")
