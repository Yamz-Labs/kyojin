"""
Op dispatch of the DS4.1 serving model: the hyper-connection chain (delayed pre-mix) and the router, each with a
torch implementation (CPU / any device, the reference semantics of `ops.py`) and a HIP implementation on the ds41kern
kernels (`exllamav3/modules/dsv41_fn/dsv41_hip.py`) for the real dims (hidden 5120, hc 4) on the GPU.

Hyper-connection chain, per layer (the dataflow of the official block):

    x       = norm(collapse(h, pre_prev), attn_norm)        # pre_prev: the pre-mix of the PREVIOUS ffn site
    pre,post,comb = mix(h, hc_attn)                          # computed on h before the collapse
    y       = attention(x);   h = post * y + comb @ h        # pending until the next collapse
    x       = norm(collapse(h, pre_attn), ffn_norm)          # applies the pending attention update first
    pre,post,comb = mix(h, hc_ffn)
    y       = moe(x);        h = post * y + comb @ h

`HCStream.cn(pre, w)` = [apply the pending update] + collapse + rms norm (one kernel on HIP: `hc_cn`), `mix` needs
the applied stream, so a caller always calls `cn` before `mix` (the order the kernels were tested in).
The stream is fp32 in both implementations; the sublayer outputs may be fp16 or fp32.
"""
from __future__ import annotations
import os
import torch
from . import ops, dec2


def hip_available(device, dim: int, hc: int) -> bool:
    """HIP kernels are usable: a CUDA/HIP device (or the CPU emulator of the tests), real dims."""
    if os.environ.get("DSV41_KERNELS", "1") == "0" or dim != 5120 or hc != 4:
        return False
    try:
        from ..modules.dsv41_fn import dsv41_hip as K
    except Exception:
        return False
    if K._state.get("emu"):
        return True
    return torch.device(device).type == "cuda" and torch.cuda.is_available()


class TorchHC:
    """Reference dataflow in torch (any device / dtype of the sublayer output). Shapes: h [B, L, 4, D] fp32."""

    name = "torch"

    def __init__(self, cfg, act_dtype):
        self.cfg, self.adt = cfg, act_dtype

    def new(self, h):
        return TorchStream(self, h)


class TorchStream:
    def __init__(self, be: TorchHC, h: torch.Tensor):
        self.be, self.h, self.pend = be, h, None

    def apply(self):
        if self.pend is not None:
            y, post, comb = self.pend
            self.h = ops.hc_post(y.float(), self.h, post, comb)          # fp32 stream (type_as(y.float()))
            self.pend = None
        return self.h

    def push(self, y, post, comb):
        self.pend = (y, post, comb)

    def mix(self, fn, scale, base):
        cfg = self.be.cfg
        h = self.apply()
        return ops.hc_mixes(h, fn, scale, base, cfg.hc_mult, cfg.hc_sinkhorn_iters, cfg.hc_eps, cfg.rms_norm_eps)

    def cn(self, pre, w):
        """[apply pending] + rmsnorm(collapse(h, pre), w) -> [B, L, D] in the activation dtype."""
        h = self.apply()
        x = ops.rmsnorm(ops.hc_pre(h, pre), w, self.be.cfg.rms_norm_eps)
        return x.to(self.be.adt)

    def inject(self, fn):
        """Replace the (applied) stream: engram injection."""
        self.h = fn(self.apply())
        return self.h


class HipHC:
    """The same chain on the ds41kern kernels: hc_mix (3 launches' worth fused) and hc_cn (apply + collapse + norm).
    Rows are the flattened (batch, position) tokens; h lives as [R, 4, 5120] fp32, outputs fp16."""

    name = "hip"

    def __init__(self, cfg, act_dtype):
        from ..modules.dsv41_fn import dsv41_hip as K
        self.K, self.cfg = K, cfg
        assert act_dtype == torch.float16, "the HIP chain emits fp16 activations"
        self._part = {}

    def new(self, h):
        return HipStream(self, h)

    def part(self, R, dev):
        n = R * self.K.HC_PART_CH * 25
        t = self._part.get(dev)
        if t is None or t.numel() < n:
            t = self._part[dev] = torch.empty(n, dtype=torch.float, device=dev)
        return t


class HipStream:
    def __init__(self, be: HipHC, h: torch.Tensor):
        B, L = h.shape[:2]
        self.be, self.B, self.L = be, B, L
        self.hf = h.reshape(B * L, 4, -1).contiguous()
        self.pend = None
        self.R = B * L
        self.mix_side = None            # Side whose mix_done event the main stream must still wait for (dec2 stream overlap)

    @property
    def h(self):
        return self.apply().view(self.B, self.L, 4, -1)

    def _join_mix(self):
        sd = self.mix_side
        if sd is not None:
            torch.cuda.current_stream().wait_event(sd.mix_done)
            self.mix_side = None

    def apply(self):
        self._join_mix()
        if self.pend is not None:                       # only reached when no cn follows (engram, taps, final): norm unused
            y, post, comb = self.pend
            w = torch.ones(self.hf.size(-1), dtype=torch.float, device=self.hf.device)
            pre = torch.zeros(self.R, 4, dtype=torch.float, device=self.hf.device)
            pre[:, 0] = 1
            out = torch.empty(self.R, self.hf.size(-1), dtype=torch.half, device=self.hf.device)
            self.be.K.hc_cn(self.hf, y.reshape(self.R, -1).contiguous(), post.reshape(self.R, 4).contiguous(),
                            comb.reshape(self.R, 16).contiguous(), pre, w, out, self.be.cfg.rms_norm_eps, apply=True)
            self.pend = None
        return self.hf

    def push(self, y, post, comb):
        self.pend = (y, post, comb)

    def mix(self, fn, scale, base):
        K, R, dev = self.be.K, self.R, self.hf.device
        hf = self.apply()
        pre = torch.empty(R, 4, dtype=torch.float, device=dev)
        post = torch.empty(R, 4, dtype=torch.float, device=dev)
        comb = torch.empty(R, 16, dtype=torch.float, device=dev)
        cfg = self.be.cfg
        sd = dec2.side(dev) if R == 1 else None
        if sd is not None:
            # the mix only reads the applied stream: run it beside the sublayer (attention / MoE) that follows
            main = torch.cuda.current_stream()
            sd.fork.record(main)
            with torch.cuda.stream(sd.stream):
                sd.stream.wait_event(sd.fork)
                K.hc_mix(hf, fn, scale, base, pre, post, comb, self.be.part(R, dev), cfg.hc_sinkhorn_iters,
                         cfg.rms_norm_eps, cfg.hc_eps)
                sd.mix_done.record(sd.stream)
            self.mix_side = sd
        else:
            K.hc_mix(hf, fn, scale, base, pre, post, comb, self.be.part(R, dev), cfg.hc_sinkhorn_iters,
                     cfg.rms_norm_eps, cfg.hc_eps)
        return pre.view(self.B, self.L, 4), post.view(self.B, self.L, 4), comb.view(self.B, self.L, 4, 4)

    def cn(self, pre, w):
        K, R, dev = self.be.K, self.R, self.hf.device
        out = torch.empty(R, self.hf.size(-1), dtype=torch.half, device=dev)
        pre = pre.reshape(R, 4).contiguous()
        self._join_mix()
        if self.pend is not None:
            y, post, comb = self.pend
            K.hc_cn(self.hf, y.reshape(R, -1).contiguous(), post.reshape(R, 4).contiguous(),
                    comb.reshape(R, 16).contiguous(), pre, w, out, self.be.cfg.rms_norm_eps, apply=True)
            self.pend = None
        else:
            z = torch.zeros(R, 4, dtype=torch.float, device=dev)
            K.hc_cn(self.hf, None, z, z.new_zeros(R, 16), pre, w, out, self.be.cfg.rms_norm_eps, apply=False)
        return out.view(self.B, self.L, -1)

    def inject(self, fn):
        h = self.apply().view(self.B, self.L, 4, -1)
        self.hf = fn(h).reshape(self.R, 4, -1).contiguous()
        return self.hf.view(self.B, self.L, 4, -1)


def make_hc(cfg, act_dtype, device):
    if hip_available(device, cfg.hidden_size, cfg.hc_mult) and act_dtype == torch.float16:
        return HipHC(cfg, act_dtype)
    return TorchHC(cfg, act_dtype)


# ----------------------------------------------------------------------------------------------- router
class Router:
    """sqrtsoftplus noaux_tc gate: scores = sqrt(softplus(x @ W^T)), top-k on scores + bias, weights from the unbiased
    scores normalised and scaled. torch (`ops.route`) or the HIP kernels (`router`: fp16 x, 384 experts, 5120)."""

    def __init__(self, cfg, device, use_hip: bool = None):
        self.cfg = cfg
        use = hip_available(device, cfg.hidden_size, cfg.hc_mult) if use_hip is None else use_hip
        self.hip = use and cfg.scoring_func == "sqrtsoftplus"
        self._ws = {}

    def __call__(self, x, gate_w, gate_b, topk):
        cfg = self.cfg
        # a 2-D gate_b is the per-row bias of an image-span pass (VL routing): the HIP kernel takes one bias vector
        if (self.hip and x.dtype == torch.half and gate_b.dim() == 1 and gate_w.size(0) % 8 == 0 and gate_w.size(0) <= 512
                and topk <= 16):
            from ..modules.dsv41_fn import dsv41_hip as K
            R, E = x.size(0), gate_w.size(0)
            dev = x.device
            sel = torch.empty(R, topk, dtype=torch.int64, device=dev)
            wts = torch.empty(R, topk, dtype=torch.half, device=dev)
            wf = torch.empty(R, topk, dtype=torch.float, device=dev)
            scores = torch.empty(R, E, dtype=torch.float, device=dev)
            gw = gate_w if gate_w.dtype in (torch.float, torch.half, torch.bfloat16) else gate_w.float()
            K.router(x.contiguous(), gw.contiguous(), gate_b.float().contiguous(), sel, wts, scores, topk,
                     cfg.routed_scaling_factor, cfg.gate_temp, bool(cfg.norm_topk_prob), wf)
            return wf, sel
        return ops.route(x, gate_w, gate_b, topk, cfg.scoring_func, cfg.norm_topk_prob, cfg.routed_scaling_factor,
                         cfg.gate_temp)

    def gate16(self, gate_w):
        """gate_w as a 16-bit tensor when that is lossless (fp32 weights that are bf16 values, the checkpoint dtype), else None.
        Cached per weight tensor; halves the bytes the decode router reads."""
        if gate_w.dtype in (torch.half, torch.bfloat16):
            return gate_w
        k = id(gate_w)
        hit = self._ws.get(("g16", k))
        if hit is None or hit[0] is not gate_w:
            g16 = None
            if gate_w.dtype == torch.float32 and gate_w.is_cuda and gate_w.is_contiguous():
                if bool(((gate_w.view(torch.int32) & 0xFFFF) == 0).all()):
                    g16 = gate_w.bfloat16().contiguous()
            hit = self._ws[("g16", k)] = (gate_w, g16)
        return hit[1]

    def fused_ok(self, x, gate_w, gate_b, topk) -> bool:
        """Decode router + K-group slot gather in 2 launches (dsv41_fuse): one row, fp16 x, fp16 / bf16 gate."""
        if not (self.hip and x.dtype == torch.half and x.size(0) == 1 and x.is_cuda and self.gate16(gate_w) is not None
                and gate_w.size(0) % 8 == 0 and gate_w.size(0) <= 512 and topk <= 16 and gate_b.dtype == torch.float):
            return False
        from ..modules.dsv41_fn import dsv41_fuse as FZ
        return FZ.available(x.device)

    def fused(self, x, gate_w, gate_b, topk, lut):
        """(weights fp32 [1,k], weights fp16 [1,k], picks int64 [1,k], group slots int64 [G,1,k]); same values as __call__
        + MixedKStore.group_selected, bit for bit. Buffers are per-router scratch (consumed in stream order)."""
        from ..modules.dsv41_fn import dsv41_fuse as FZ
        cfg, dev = self.cfg, x.device
        E, G = gate_w.size(0), lut.size(0)
        key = (dev, E, topk, G)
        b = self._ws.get(key)
        if b is None:
            b = self._ws[key] = (torch.empty(1, topk, dtype=torch.int64, device=dev), torch.empty(1, topk, dtype=torch.half, device=dev),
                                 torch.empty(1, topk, dtype=torch.float, device=dev), torch.empty(1, E, dtype=torch.float, device=dev),
                                 torch.empty(G, 1, topk, dtype=torch.int64, device=dev))
        sel, wh, wf, scores, gsel = b
        FZ.router_scores_d(x.contiguous(), self.gate16(gate_w), scores, cfg.gate_temp)
        FZ.router_topk2(scores, gate_b, sel, wh, wf, topk, bool(cfg.norm_topk_prob), cfg.routed_scaling_factor, lut, gsel)
        return wf, wh, sel, gsel


from .linear import DenseLin


class HipDenseLin(DenseLin):
    """DenseLin whose decode calls (<= 8 rows) run on the ds41kern dense GEMV (fp32 accumulation, reads the weight once).
    fp32 weights that are bf16 values (the checkpoint dtype) are stored as bf16 (lossless, half the bytes); fp16 weights stay.
    Other shapes / dtypes use the torch GEMM."""

    def __call__(self, x):
        k = self.hk
        if x.is_contiguous() and 0 < x.numel() // x.size(-1) <= 8 and ((x.dtype == torch.float32 and k in "fb")
                                                                       or (x.dtype == torch.float16 and k == "h")):
            from ..modules.dsv41_fn import dsv41_hip as K
            x2 = x.view(-1, x.size(-1))
            y = torch.empty(x2.size(0), self.hw.size(0), dtype=x.dtype, device=x.device)
            K.dense_gemv(self.hw, k, x2, y)
            return y.view(*x.shape[:-1], y.size(-1))
        return DenseLin.__call__(self, x)


class HipDense:
    @staticmethod
    def attach(lin):
        w = lin.w
        if w.dim() != 2 or w.size(1) % 256 or not w.is_cuda:
            return
        if w.dtype == torch.float32:
            bits = w.contiguous().view(torch.int32)
            if bool(((bits & 0xFFFF) == 0).all()):
                lin.hw, lin.hk = w.bfloat16().contiguous(), "b"
            else:
                lin.hw, lin.hk = w.contiguous(), "f"
        elif w.dtype == torch.float16:
            lin.hw, lin.hk = w.contiguous(), "h"
        else:
            return
        lin.__class__ = HipDenseLin
