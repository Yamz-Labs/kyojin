from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn.functional as F
from .module import Module
from .rmsnorm import RMSNorm
from ..model.config import Config
from ..ext import exllamav3_ext as ext
from ..util.tensor import g_tensor_cache
import os

# gfx1151: store the GatedResidual mixer weights (the only fp16 bulk in an EXL3 model, 22% of
# decode weight bytes) as per-row int8 and dequantize in the fused kernel. ROCm only; opt out
# with EXL3_HIP_GR_MIX_Q8=0.
_GR_MIX_Q8 = bool(torch.version.hip) and os.environ.get("EXL3_HIP_GR_MIX_Q8", "1") != "0"

# hc mix fused with the consumer RMSNorm (hc_mix_norm: one finalize+norm launch instead of
# finalize + rms_norm); bit-identical to mix() -> half -> norm.forward(out_dtype = half)
_FUSE_HC_NORM = os.environ.get("EXL3_FUSE_HC_NORM", "1") != "0" and os.environ.get("EXL3_DEC_NORM", "1") != "0"

_hc_mix_support_cache: dict[tuple[object, int], bool] = {}


def _hc_mix_supported(device: torch.device) -> bool:
    """Return whether the fused kernels' fixed 32-lane shuffle layout is valid."""
    support_fn = getattr(ext, "hc_mix_supported", None)
    if support_fn is None or device.type != "cuda":
        return False
    index = device.index if device.index is not None else torch.cuda.current_device()
    key = (support_fn, index)
    if key not in _hc_mix_support_cache:
        _hc_mix_support_cache[key] = bool(support_fn(index))
    return _hc_mix_support_cache[key]


# mHC (manifold-constrained hyper-connections, DeepSeek-V4): the residual is carried as
# hc_mult parallel fp32 streams shaped (bsz, seq, hc_mult, hidden). ExpandStreams broadcasts
# the embedding into the streams, each sublayer site mixes them through a HyperConnection
# (sigmoid pre/post weights + Sinkhorn-normalized combine matrix), and HyperHead collapses
# them before the final norm. TransformerBlock consumes HyperConnection via optional
# attn_hc/mlp_hc parameters.


class ExpandStreams(Module):
    """Broadcast the embedding into hc_mult parallel residual streams, fp32."""

    def __init__(self, config: Config, key: str, hc_mult: int):
        super().__init__(config = config, key = key, qmap = None)
        self.hc_mult = hc_mult

    @override
    def optimizer_targets(self):
        return []

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None):
        return x.float().unsqueeze(2).expand(-1, -1, self.hc_mult, -1).contiguous()

    def tp_export(self, plan, producer):
        # Stateless stream broadcast; the residual (and its stream stack) is replicated
        return {
            "cls": ExpandStreams,
            "kwargs": {
                "key": self.key,
                "hc_mult": self.hc_mult,
            },
            "device": self.device,
        }

    @staticmethod
    def tp_import(local_context, exported, plan):
        module = ExpandStreams(config = None, **exported["kwargs"])
        module.device = local_context["device"]
        return module


class HyperConnection(Module):
    """mHC mixer for one sublayer site. Owns raw fp32 tensors {key}_fn ((2 + H) * H rows,
    H * hidden cols), {key}_base, {key}_scale. Not a standalone graph module: TransformerBlock
    calls mix() around its attn/mlp sites."""

    def __init__(
        self,
        config: Config | None,
        key: str,                    # e.g. "layers.{idx}.hc_attn"; tensors at "{key}_fn" etc.
        hc_mult: int,
        hidden_size: int,
        sinkhorn_iters: int,
        hc_eps: float,
        rms_norm_eps: float,
    ):
        super().__init__(config = config, key = key, qmap = None)
        self.hc_mult = hc_mult
        self.hidden_size = hidden_size
        self.sinkhorn_iters = sinkhorn_iters
        self.hc_eps = hc_eps
        self.rms_eps = rms_norm_eps
        self.norm = RMSNorm(config, f"{key}.norm", rms_norm_eps, unweighted = True,
                            out_dtype = torch.float)
        self.register_submodule(self.norm)
        self.fn = None
        self.fn_h = None
        self.base = None
        self.scale = None

    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        stc = self.config.stc
        self.fn = stc.get_tensor(f"{self.key}_fn", device, no_defer = True).float().contiguous()
        self.base = stc.get_tensor(f"{self.key}_base", device, no_defer = True).float().contiguous()
        self.scale = stc.get_tensor(f"{self.key}_scale", device, no_defer = True).float().contiguous()

    @override
    def unload(self):
        super().unload()
        self.fn = self.fn_h = self.base = self.scale = None

    @override
    def get_tensors(self):
        return {
            f"{self.key}_fn": self.fn.contiguous(),
            f"{self.key}_base": self.base.contiguous(),
            f"{self.key}_scale": self.scale.contiguous(),
        }

    @override
    def weights_numel(self):
        h = self.hc_mult
        return (2 * h + h * h) * (h * self.hidden_size + 1) + 3

    @override
    def optimizer_targets(self):
        return []

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None):
        raise RuntimeError("HyperConnection is not a standalone module; use mix()")

    def mix(self, streams: torch.Tensor, params: dict):
        """streams (b, s, H, D) fp32 -> (post (b,s,H), comb (b,s,H,H), collapsed (b,s,D)).
        Fused ext path (2 kernel launches, see benchmarks/hc_mix/) returns collapsed as HALF
        (both block consumers cast it immediately); the torch fallback keeps fp32."""
        hc = self.hc_mult
        b, s, H, D = streams.shape
        if hasattr(ext, "hc_mix_num_chunks") and hasattr(ext, "hc_mix") \
                and _hc_mix_supported(streams.device) \
                and hc == 4 and streams.dtype == torch.float and D % 4 == 0 \
                and streams.is_contiguous():
            R = b * s
            st = streams.view(R, H, D)
            chunks = ext.hc_mix_num_chunks(R, H * D)
            M1 = 2 * H + H * H + 1
            dev = streams.device
            # Decode-class row counts take the static workspaces (allocation latency matters and
            # the graphed callers rely on them); prefill chunks allocate per call so the static
            # cache holds only small buffers
            def ws(numel, dtype, tag):
                if R <= 32:
                    return g_tensor_cache.get_bucketed(dev, numel, dtype, tag)
                return torch.empty((numel,), dtype = dtype, device = dev)
            partials = ws(R * chunks * M1, torch.float, "hc_mix_partials").view(R, chunks, M1)
            post = ws(R * H, torch.float, "hc_post").view(R, H)
            comb = ws(R * H * H, torch.float, "hc_comb").view(R, H, H)
            collapsed = ws(R * D, torch.half, "hc_coll").view(R, D)
            # Small R (decode): fn in fp16 -- the (M, H * D) matrix is the partials kernel's
            # dominant traffic and the kernel dots it in fp32 either way
            if R <= 32 or os.environ.get("EXL3_HC_FN_HALF_PF", "0") != "0":
                if self.fn_h is None:
                    self.fn_h = self.fn.half()
                fn = self.fn_h
            else:
                fn = self.fn
            ext.hc_mix(st, fn, self.base, self.scale, self.rms_eps, self.hc_eps,
                       self.sinkhorn_iters, partials, post, comb, collapsed)
            return post.view(b, s, H), comb.view(b, s, H, H), collapsed.view(b, s, D)
        flat = self.norm.forward(streams.flatten(2), params)
        mix = F.linear(flat, self.fn)
        pre_w, post_w, comb_w = mix.split([hc, hc, hc * hc], dim = -1)
        pre_b, post_b, comb_b = self.base.split([hc, hc, hc * hc])
        pre_s, post_s, comb_s = self.scale.unbind(0)

        pre = torch.sigmoid(pre_w * pre_s + pre_b) + self.hc_eps
        post = 2.0 * torch.sigmoid(post_w * post_s + post_b)
        comb = comb_w.view(*comb_w.shape[:-1], hc, hc) * comb_s + comb_b.view(hc, hc)
        comb = torch.softmax(comb, dim = -1) + self.hc_eps
        comb = comb / (comb.sum(dim = -2, keepdim = True) + self.hc_eps)
        for _ in range(self.sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim = -1, keepdim = True) + self.hc_eps)
            comb = comb / (comb.sum(dim = -2, keepdim = True) + self.hc_eps)
        collapsed = (pre.unsqueeze(-1) * streams).sum(dim = 2)
        return post, comb, collapsed

    def mix_norm(self, streams: torch.Tensor, params: dict, norm, pending = None):
        """mix_norm with an optional deferred residual apply of the previous site (pf_can_defer).
        With pending, streams are updated in place either way: fused when the site pair
        qualifies, else flushed with the plain apply followed by the normal path."""
        if pending is not None:
            fused = self._pf_apply_mix_norm(streams, params, norm, pending)
            if fused is not None:
                return fused
            HyperConnection.flush_pending(streams, pending)
        return self._mix_norm(streams, params, norm)

    @staticmethod
    def pf_can_defer(x: torch.Tensor, y: torch.Tensor, post: torch.Tensor, comb: torch.Tensor,
                     params: dict) -> bool:
        """EXL3_PF_HC_FUSE (default off, read per call): prefill rows may defer this site's
        residual apply into the next site's mix_norm (ext.hc_pf_apply_mix_norm: the stream
        stack is written once and re-read once instead of twice). True only when the
        in-place ext.hc_apply could also run, so a consumer that cannot fuse flushes with
        flush_pending(). Requires the fused kernel's single-chunk regime (R > 256)."""
        if os.environ.get("EXL3_PF_HC_FUSE", "0") == "0" or not hasattr(ext, "hc_pf_apply_mix_norm"):
            return False
        if "quant_preserve" in params or "capture" in params:
            return False
        b, s, H, D = x.shape
        return (b * s > 256 and b * s <= 4096 and H == 4 and D % 4 == 0
                and x.dtype == torch.float and x.is_contiguous()
                and y.dtype in (torch.float, torch.half) and y.is_contiguous() and y.numel() == b * s * D
                and post.dtype == torch.float and post.is_contiguous()
                and comb.dtype == torch.float and comb.is_contiguous() and _hc_mix_supported(x.device))

    @staticmethod
    def flush_pending(x: torch.Tensor, pending) -> torch.Tensor:
        """Run a deferred residual apply (see pf_can_defer) in place on x."""
        px, y, post, comb = pending
        if px is not x:
            raise RuntimeError("EXL3_PF_HC_FUSE: deferred hc apply reached a different stream tensor")
        b, s, H, D = x.shape
        R = b * s
        ext.hc_apply(x.view(R, H, D), y.view(R, D), post.view(R, H), comb.view(R, H, H))
        return x

    def _pf_apply_mix_norm(self, streams: torch.Tensor, params: dict, norm, pending):
        """Fused site-A apply + this site's mix_norm_1pass (bit-identical to the pair)."""
        px, ya, post_a, comb_a = pending
        if px is not streams or not hasattr(ext, "hc_pf_apply_mix_norm"):
            return None
        if not _FUSE_HC_NORM or not isinstance(norm, RMSNorm) or norm.span_heads or norm.groups != 1 \
                or norm.key in params.get("export_state_norm_keys", ()):
            return None
        b, s, H, D = streams.shape
        w = None if norm.unweighted else norm.weight
        if w is not None and not (w.dtype in (torch.half, torch.bfloat16) and w.numel() == D and w.is_contiguous()):
            return None
        if not (self.hc_mult == 4 and H == 4 and streams.dtype == torch.float and D % 4 == 0
                and D <= 8192 and streams.is_contiguous() and _hc_mix_supported(streams.device)):
            return None
        R = b * s
        if R <= 256 or R > 4096:
            return None
        if ext.hc_mix_num_chunks(R, H * D) != 1:
            return None
        if ya.numel() != R * D:
            return None
        M1 = 2 * H + H * H + 1
        dev = streams.device
        # Same fn choice as _mix_norm: prefill rows read the fp16 copy when enabled.
        if R <= 32 or os.environ.get("EXL3_HC_FN_HALF_PF", "0") != "0":
            if self.fn_h is None:
                self.fn_h = self.fn.half()
            fn = self.fn_h
        else:
            fn = self.fn
        partials = torch.empty((R, 1, M1), dtype = torch.float, device = dev)
        post = torch.empty((R, H), dtype = torch.float, device = dev)
        comb = torch.empty((R, H * H), dtype = torch.float, device = dev).view(R, H, H)
        y = torch.empty((b, s, D), dtype = torch.half, device = dev)
        ext.hc_pf_apply_mix_norm(streams.view(R, H, D), ya.contiguous().view(R, D),
                                 post_a.contiguous().view(R, H), comb_a.contiguous().view(R, H, H),
                                 fn, self.base, self.scale, self.rms_eps, self.hc_eps,
                                 self.sinkhorn_iters, partials, post, comb, y.view(R, D), w,
                                 norm.rms_norm_eps, norm.constant_bias, norm.constant_scale)
        return post.view(b, s, H), comb.view(b, s, H, H), y

    def _mix_norm(self, streams: torch.Tensor, params: dict, norm):
        """mix() followed by norm.forward(collapsed.half(), out_dtype = half), fused. Returns
        (post, comb, normed half (b, s, D)), or None when the fused kernel does not apply."""
        if not _FUSE_HC_NORM or not hasattr(ext, "hc_mix_norm") or not isinstance(norm, RMSNorm):
            return None
        if norm.span_heads or norm.groups != 1 or norm.key in params.get("export_state_norm_keys", ()) \
                or "quant_preserve" in params or "capture" in params:
            return None
        b, s, H, D = streams.shape
        w = None if norm.unweighted else norm.weight
        if w is not None and not (w.dtype in (torch.half, torch.bfloat16) and w.numel() == D and w.is_contiguous()):
            return None
        if not (self.hc_mult == 4 and H == 4 and streams.dtype == torch.float and D % 4 == 0 and D <= 8192
                and streams.is_contiguous() and _hc_mix_supported(streams.device)):
            return None
        R = b * s
        chunks = ext.hc_mix_num_chunks(R, H * D)
        M1 = 2 * H + H * H + 1
        dev = streams.device
        def ws(numel, dtype, tag):
            if R <= 32:
                return g_tensor_cache.get_bucketed(dev, numel, dtype, tag)
            return torch.empty((numel,), dtype = dtype, device = dev)
        partials = ws(R * chunks * M1, torch.float, "hc_mix_partials").view(R, chunks, M1)
        post = ws(R * H, torch.float, "hc_post").view(R, H)
        comb = ws(R * H * H, torch.float, "hc_comb").view(R, H, H)
        # EXL3_HC_FN_HALF_PF=1 (default off, read per call for in-process A/B): prefill rows also
        # read the fp16 fn copy decode uses. fp32 fn (2.4 MB) spills L2 on gfx1151; fp16 fits.
        if R <= 32 or os.environ.get("EXL3_HC_FN_HALF_PF", "0") != "0":
            if self.fn_h is None:
                self.fn_h = self.fn.half()
            fn = self.fn_h
        else:
            fn = self.fn
        y = torch.empty((b, s, D), dtype = torch.half, device = dev)
        ext.hc_mix_norm(streams.view(R, H, D), fn, self.base, self.scale, self.rms_eps, self.hc_eps,
                        self.sinkhorn_iters, partials, post, comb, y.view(R, D), w,
                        norm.rms_norm_eps, norm.constant_bias, norm.constant_scale)
        return post.view(b, s, H), comb.view(b, s, H, H), y

    def apply_(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        params: dict
    ):
        """Residual update for one sublayer site: x <- post ⊗ y + combᵀ x. Fused ext path
        updates x IN PLACE (each output column depends only on the same column of the H
        stream rows); the torch fallback allocates. Conversion must NOT run the in-place
        path: the capture and advance passes forward the SAME stored input states twice."""
        b, s, H, D = x.shape
        converting = "quant_preserve" in params or "capture" in params
        if hasattr(ext, "hc_apply") and not converting and _hc_mix_supported(x.device) \
                and H == 4 and x.dtype == torch.float and x.is_contiguous() and D % 4 == 0 \
                and y.dtype in (torch.float, torch.half) and y.is_contiguous() \
                and post.dtype == torch.float and post.is_contiguous() and comb.is_contiguous():
            R = b * s
            ext.hc_apply(x.view(R, H, D), y.view(R, D), post.view(R, H), comb.view(R, H, H))
            return x
        return post.unsqueeze(-1) * y.float().unsqueeze(-2) + torch.matmul(comb.transpose(-1, -2), x)

    def tp_export(self, plan, producer):
        # Streams are replicated across TP workers (like the residual), so plain replication
        return {
            "cls": HyperConnection,
            "kwargs": {
                "key": self.key,
                "hc_mult": self.hc_mult,
                "hidden_size": self.hidden_size,
                "sinkhorn_iters": self.sinkhorn_iters,
                "hc_eps": self.hc_eps,
                "rms_norm_eps": self.rms_eps,
            },
            "fn": producer.send(self.fn),
            "base": producer.send(self.base),
            "scale": producer.send(self.scale),
            "device": self.device,
        }

    @staticmethod
    def tp_import(local_context, exported, plan):
        consumer = local_context["consumer"]
        module = HyperConnection(config = None, **exported["kwargs"])
        module.fn = consumer.recv(exported["fn"], cuda = True)
        module.base = consumer.recv(exported["base"], cuda = True)
        module.scale = consumer.recv(exported["scale"], cuda = True)
        module.device = local_context["device"]
        return module


class GatedResidual(Module):
    """
    Qwen4Exp-style gated residual: the low-rank, elementwise cousin of mHC. The residual is the
    same (bsz, seq, hc_mult, hidden) fp32 stream stack, but mixing is per-channel instead of a
    stream-mixing matrix: per-stream grouped RMSNorm (zero-init weight, applied as 1 + w), a
    low-rank sigmoid gate over the normed stack picks what each stream contributes to the
    elementwise MEAN that feeds the sublayer, and the sublayer output is injected back into the
    raw streams with a per-stream scalar 2*sigmoid gate. No Sinkhorn, no combine matrix.

    Site form (use_combine = True): TransformerBlock calls mix() / apply_() like HyperConnection,
    with comb = None. Final-mixer form (use_combine = False, HF hyper_connection_mixer): a
    standalone module whose forward() collapses the stack.

    Two compute paths sharing the hc_mix.cu machinery: small R (decode) runs the fused
    ext.gr_mix pair (per-stream partial dots on the raw streams + a finalize that derives the
    low-rank gate inline), large R (prefill) runs half GEMMs + a few elementwise ops where
    launch count amortizes and tensor cores carry the FLOPs. apply_() is ext.hc_apply without
    a comb (x[h] += post[h] * y), shared with mHC. _mix_ref() keeps the fp32 torch reference
    the parity tests compare against.

    Tensors: {key}.hc_norm.weight, {key}.input_mix_weight_down.weight,
    {key}.input_mix_weight_up.weight and, for the site form, {key}.block_inject_weight.weight.
    """

    FUSED_MAX_R = 32

    def __init__(
        self,
        config: Config | None,
        key: str,
        hc_mult: int,
        hidden_size: int,
        rms_norm_eps: float,
        use_combine: bool = True,
        out_dtype: torch.dtype | None = None,
    ):
        super().__init__(config = config, key = key, qmap = None)
        self.hc_mult = hc_mult
        self.hidden_size = hidden_size
        self.rms_eps = rms_norm_eps
        self.use_combine = use_combine
        self.out_dtype = out_dtype
        self.norm_w_raw = None
        self.norm_w = None          # (hc_mult, hidden) fp32, includes the + 1.0 (reference path)
        self.w_h = None             # (hc_mult * hidden) half, includes the + 1.0
        self.down_h = None          # (rank, hc_mult * hidden) half
        self.up_h = None            # (hc_mult * hidden, rank) half, checkpoint orientation
        self.upx_h = None           # (hc_mult, hidden / 4, rank, 4) half (fused-kernel layout)
        self.inject_h = None        # (hc_mult, hc_mult * hidden) half (site form)
        self.proj_h = None          # cat(down, inject) half, unfolded (GEMM path)
        self.fn_h = None            # cat(down, inject) * w half, folded (fused path)
        self.use_q8 = False         # int8 fused-path weights (ROCm), see _prepare
        self.fn_q8 = self.fn_scale = self.upx_q8 = self.up_scale = None
        self.rank = 0

    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        stc = self.config.stc
        self.norm_w_raw = stc.get_tensor(f"{self.key}.hc_norm.weight", device, no_defer = True)
        down = stc.get_tensor(f"{self.key}.input_mix_weight_down.weight", device, no_defer = True)
        up = stc.get_tensor(f"{self.key}.input_mix_weight_up.weight", device, no_defer = True)
        inject = stc.get_tensor(f"{self.key}.block_inject_weight.weight", device,
                                no_defer = True) if self.use_combine else None
        self._prepare(down, up, inject)

    def _prepare(self, down, up, inject):
        # Derived buffers are deduplicated (down/inject live as views of proj_h; up is kept in
        # its checkpoint orientation and the GEMM path transposes by view), and the fp32 folding
        # intermediates go through a REUSED scratch: load interleaves these preparations with
        # the persistent weight allocations, and per-site transient churn splinters the
        # allocator's segments (measured ~15 GB reserved-not-allocated on the full model)
        from ..util.tensor import g_tensor_cache
        dev = down.device
        H, Dh = self.hc_mult, self.hidden_size
        self.norm_w = (self.norm_w_raw.float() + 1.0).view(H, Dh).contiguous()
        self.w_h = self.norm_w.flatten().half().contiguous()
        self.rank = down.shape[0]
        if inject is None:
            self.proj_h = down.half().contiguous()
            self.inject_h = None
        else:
            self.proj_h = torch.cat((down.half(), inject.half())).contiguous()
            self.inject_h = self.proj_h[self.rank :]
        self.down_h = self.proj_h[: self.rank]
        M = self.proj_h.shape[0]
        tmp = g_tensor_cache.get_bucketed(dev, M * H * Dh, torch.float, "gr_prep_tmp") \
            .view(M, H * Dh)
        tmp.copy_(self.proj_h)
        tmp *= self.w_h.float()
        self.up_h = up.half().contiguous()          # (H * D, rank), checkpoint orientation
        self.use_q8 = _GR_MIX_Q8 and hasattr(ext, "gr_mix_q8") and dev.type == "cuda"
        if self.use_q8:
            # Symmetric per-row int8: fn row j (the dot product's output) and up output column
            # h*D+d both get one scale, applied once per result inside the kernel
            sc = tmp.abs().amax(dim = 1).clamp_min_(1e-12) / 127.0
            self.fn_q8 = torch.round(tmp / sc[:, None]).clamp_(-127, 127).to(torch.int8).contiguous()
            self.fn_scale = sc.contiguous()
            up32 = self.up_h.float()
            usc = up32.abs().amax(dim = 1).clamp_min_(1e-12) / 127.0
            upq = torch.round(up32 / usc[:, None]).clamp_(-127, 127).to(torch.int8)
            self.upx_q8 = upq.view(H, Dh // 4, 4, self.rank).permute(0, 1, 3, 2).contiguous()
            self.up_scale = usc.contiguous()
            self.fn_h = None
            self.upx_h = None
            del up32, upq
        else:
            self.fn_h = tmp.half().contiguous()
            # up repacked (H, D/4, rank, 4) so the fused kernel's rank loop reads lane-contiguous
            self.upx_h = self.up_h.view(H, Dh // 4, 4, self.rank) \
                .permute(0, 1, 3, 2).contiguous()

    @override
    def unload(self):
        super().unload()
        self.norm_w_raw = self.norm_w = self.w_h = None
        self.down_h = self.up_h = self.upx_h = self.inject_h = self.proj_h = self.fn_h = None
        self.fn_q8 = self.fn_scale = self.upx_q8 = self.up_scale = None

    @override
    def get_tensors(self):
        t = {
            f"{self.key}.hc_norm.weight": self.norm_w_raw.contiguous(),
            f"{self.key}.input_mix_weight_down.weight": self.down_h.contiguous(),
            f"{self.key}.input_mix_weight_up.weight": self.up_h.contiguous(),
        }
        if self.use_combine:
            t[f"{self.key}.block_inject_weight.weight"] = self.inject_h.contiguous()
        return t

    @override
    def weights_numel(self):
        n = self.hc_mult * self.hidden_size
        return n + 2 * self.rank * n + (self.hc_mult * n if self.use_combine else 0)

    @override
    def optimizer_targets(self):
        return []

    def _mix_ref(self, streams: torch.Tensor):
        """fp32 torch reference of the mix (the parity tests' ground truth): returns
        (post (b, s, H) or None, mixed (b, s, D)), both fp32."""
        x = streams.float()
        normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + self.rms_eps) * self.norm_w
        flat = normed.flatten(-2)
        t = F.silu(F.linear(flat, self.down_h.float()) / self.hc_mult)
        w = torch.sigmoid(F.linear(t, self.up_h.float()))
        mixed = (w.unflatten(-1, (self.hc_mult, self.hidden_size)) * normed).mean(dim = -2)
        post = 2.0 * torch.sigmoid(F.linear(flat, self.inject_h.float()) / self.hc_mult) \
            if self.use_combine else None
        return post, mixed

    def _mix(self, streams: torch.Tensor, cached: bool = True):
        """streams (b, s, H, D) fp32 -> (post (R, H) fp32 or None, mixed (R, D) half).
        cached: small-R outputs may come from the per-device static workspaces (see below);
        callers that hold the result across another mix on the device pass False."""
        H, Dh = self.hc_mult, self.hidden_size
        R = streams.shape[0] * streams.shape[1]
        s3 = streams.reshape(R, H, Dh)
        if s3.dtype != torch.float:
            s3 = s3.float()          # MTP sample_from_state passes the half draft stack
        if not s3.is_contiguous():
            s3 = s3.contiguous()
        dev = s3.device
        has_gr_mix = hasattr(ext, "gr_mix")
        native_mix = has_gr_mix and _hc_mix_supported(dev)
        if not has_gr_mix or (R <= self.FUSED_MAX_R and not native_mix):
            post, mixed = self._mix_ref(s3.view(1, R, H, Dh))
            return (post.view(R, H) if post is not None else None), mixed.view(R, Dh).half()

        if R <= self.FUSED_MAX_R:
            # Decode/MTP-class row counts (the fused path's whole domain) take bucketed
            # workspaces from the per-device static cache, shared by every GatedResidual site
            # on the device: a site's outputs are consumed (block input, apply_) before the
            # next site mixes on the same stream, so one set per device suffices and no
            # per-site statics are needed. Sized by numel, so a rebuilt fn_h with another rank
            # simply lands in a different bucket; nearby R share a backing via slices.
            def ws(numel, dtype, tag):
                if cached:
                    return g_tensor_cache.get_bucketed(dev, numel, dtype, tag)
                return torch.empty((numel,), dtype = dtype, device = dev)
            M = self.proj_h.shape[0] + 1
            dots = ws(R * M * H, torch.float, "gr_mix_dots").view(R, M, H)
            post = ws(R * H, torch.float, "gr_mix_post").view(R, H) if self.use_combine else None
            mixed = ws(R * Dh, torch.half, "gr_mix_mixed").view(R, Dh)
            if self.use_q8:
                ext.gr_mix_q8(s3, self.fn_q8, self.fn_scale, self.upx_q8, self.up_scale, self.w_h,
                              self.rms_eps, dots, post, mixed)
            else:
                ext.gr_mix(s3, self.fn_h, self.upx_h, self.w_h, self.rms_eps, dots, post, mixed)
        else:
            post = torch.empty((R, H), dtype = torch.float, device = dev) \
                if self.use_combine else None
            normed = torch.empty((R * H, Dh), dtype = torch.half, device = dev)
            ext.rms_norm(s3.view(R * H, Dh), self.w_h, normed,
                         self.rms_eps, 0.0, 1.0, False, False, H)
            dm = torch.matmul(normed.view(R, H * Dh), self.proj_h.t())     # (R, rank [+ H])
            t = F.silu(dm[:, : self.rank] / H)
            if self.use_combine:
                post.copy_(2.0 * torch.sigmoid(dm[:, self.rank :].float() / H))
            g = torch.matmul(t, self.up_h.t())                             # (R, H * Dh)
            mixed = (torch.sigmoid(g.float()).view(R, H, Dh)
                     * normed.float().view(R, H, Dh)).mean(dim = -2).half()
        return post, mixed

    def mix(self, streams: torch.Tensor, params: dict):
        """(b, s, H, D) fp32 -> (inject gates (b, s, H) fp32, None, collapsed (b, s, D) half)."""
        b, s = streams.shape[:2]
        post, mixed = self._mix(streams)
        return post.view(b, s, self.hc_mult), None, mixed.view(b, s, self.hidden_size)

    def apply_(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor | None,
        params: dict
    ):
        """Residual update for one sublayer site, in place: x <- x + post (x) y (comb unused).
        Conversion must NOT run the in-place path: the capture and advance passes forward the
        SAME stored input states twice (mHC apply_ has the same guard)."""
        b, s, H, D = x.shape
        if not hasattr(ext, "hc_apply") or "quant_preserve" in params or "capture" in params \
                or not _hc_mix_supported(x.device) or H != 4 or D % 4 != 0 \
                or x.dtype != torch.float or not x.is_contiguous() \
                or y.dtype not in (torch.half, torch.float) or not y.is_contiguous() \
                or post.dtype != torch.float or not post.is_contiguous():
            return x + post.unsqueeze(-1) * y.float().unsqueeze(-2)
        y2 = y.reshape(b * s, self.hidden_size)
        ext.hc_apply(
            x.view(b * s, self.hc_mult, self.hidden_size),
            y2.contiguous(),
            post.reshape(b * s, self.hc_mult).contiguous(),
            None,
        )
        return x

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None):
        """Final-mixer form only: collapse the stream stack."""
        assert not self.use_combine, "site-form GatedResidual is consumed via mix()/apply_()"
        # MTP trunk tap: models without a final norm export the PRE-collapse stream stack here
        # (flattened), the analog of the RMSNorm export hook
        if self.key in params.get("export_state_norm_keys", ()):
            states = params.get("export_states")
            if states is None:
                states = params["export_states"] = []
            states.append(x.flatten(-2).half())
        b, s = x.shape[:2]
        # Conversion passes hold this output while other modules run; give them fresh tensors
        _, mixed = self._mix(x, cached = "capture" not in params and "quant_preserve" not in params)
        mixed = mixed.view(b, s, self.hidden_size)
        dt = out_dtype or self.out_dtype
        return mixed if dt is None else mixed.to(dt)

    def tp_export(self, plan, producer):
        # Streams are replicated across TP workers (like the residual), so plain replication
        return {
            "cls": GatedResidual,
            "kwargs": {
                "key": self.key,
                "hc_mult": self.hc_mult,
                "hidden_size": self.hidden_size,
                "rms_norm_eps": self.rms_eps,
                "use_combine": self.use_combine,
                "out_dtype": self.out_dtype,
            },
            "norm_w_raw": producer.send(self.norm_w_raw),
            "down": producer.send(self.down_h),
            "up": producer.send(self.up_h),
            "inject": producer.send(self.inject_h) if self.use_combine else None,
            "device": self.device,
        }

    @staticmethod
    def tp_import(local_context, exported, plan):
        consumer = local_context["consumer"]
        module = GatedResidual(config = None, **exported["kwargs"])
        module.norm_w_raw = consumer.recv(exported["norm_w_raw"], cuda = True)
        down = consumer.recv(exported["down"], cuda = True)
        up = consumer.recv(exported["up"], cuda = True)
        inject = consumer.recv(exported["inject"], cuda = True) if module.use_combine else None
        module._prepare(down, up, inject)
        module.device = local_context["device"]
        return module


class HyperHead(Module):
    """Final mHC stream collapse before the model norm. Top-level raw tensors {key}_fn etc.
    mean = True (GLM5.3): parameterless unweighted mean over the streams, no tensors."""

    def __init__(self, config: Config, key: str, hc_mult: int, rms_norm_eps: float, hc_eps: float,
                 mean: bool = False):
        super().__init__(config = config, key = key, qmap = None)
        self.hc_mult = hc_mult
        self.rms_eps = rms_norm_eps
        self.hc_eps = hc_eps
        self.mean = mean
        self.norm = RMSNorm(config, f"{key}.norm", rms_norm_eps, unweighted = True,
                            out_dtype = torch.float)
        self.register_submodule(self.norm)
        self.fn = None
        self.fn_h = None
        self.base = None
        self.scale = None

    def _tensor_names(self):
        return [f"{self.key}_fn", f"{self.key}_base", f"{self.key}_scale"]

    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        if self.mean:
            return
        stc = self.config.stc
        self.fn = stc.get_tensor(f"{self.key}_fn", device, no_defer = True).float().contiguous()
        self.base = stc.get_tensor(f"{self.key}_base", device, no_defer = True).float().contiguous()
        self.scale = stc.get_tensor(f"{self.key}_scale", device, no_defer = True).float().contiguous()

    @override
    def unload(self):
        super().unload()
        self.fn = self.fn_h = self.base = self.scale = None

    @override
    def get_tensors(self):
        if self.mean:
            return {}
        return {
            f"{self.key}_fn": self.fn,
            f"{self.key}_base": self.base,
            f"{self.key}_scale": self.scale,
        }

    # The compile step enumerates a module's output tensors by "{key}." prefix; this module's
    # tensor names are underscore-joined at the top level (hc_head_fn etc.), so the prefix trie
    # never matches them and they would be silently dropped from the compiled shards
    @override
    def get_compile_sizes(self, stc):
        if self.mean:
            return []
        return [stc.get_tensor_size(k) for k in self._tensor_names()]

    @override
    def get_compile_tensors(self, stc):
        if self.mean:
            return {}
        return {k: stc.get_tensor(k, allow_bf16 = True) for k in self._tensor_names()}

    @override
    def optimizer_targets(self):
        return []

    def tp_export(self, plan, producer):
        # Stream collapse runs on the replicated stream stack: plain replication
        return {
            "cls": HyperHead,
            "kwargs": {
                "key": self.key,
                "hc_mult": self.hc_mult,
                "rms_norm_eps": self.rms_eps,
                "hc_eps": self.hc_eps,
            },
            "fn": producer.send(self.fn),
            "base": producer.send(self.base),
            "scale": producer.send(self.scale),
            "device": self.device,
        }

    @staticmethod
    def tp_import(local_context, exported, plan):
        consumer = local_context["consumer"]
        module = HyperHead(config = None, **exported["kwargs"])
        module.fn = consumer.recv(exported["fn"], cuda = True)
        module.base = consumer.recv(exported["base"], cuda = True)
        module.scale = consumer.recv(exported["scale"], cuda = True)
        module.device = local_context["device"]
        return module

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None):
        pending = params.pop("hc_pending", None)
        if pending is not None:
            HyperConnection.flush_pending(x, pending)
        if self.mean:
            # MTP tap (EXL3_MTP_PRENORM_H, default OFF): the reference feeds the MTP head the
            # mean-collapsed streams BEFORE model.norm, not the post-norm state. Keyed on the
            # head's own norm key so it rides the same export_state_norm_keys plumbing.
            if self.norm.key in params.get("export_state_norm_keys", ()):
                states = params.get("export_states")
                if states is None:
                    states = params["export_states"] = []
                states.append(x.mean(dim = 2).half())
            return x.mean(dim = 2)
        b, s, H, D = x.shape
        if hasattr(ext, "hc_mix_num_chunks") and hasattr(ext, "hc_head") \
                and _hc_mix_supported(x.device) \
                and H == 4 and x.dtype == torch.float and D % 4 == 0 and x.is_contiguous():
            R = b * s
            chunks = ext.hc_mix_num_chunks(R, H * D)
            partials = g_tensor_cache.get_bucketed(
                x.device, R * chunks * (H + 1), torch.float, "hc_head_partials").view(R, chunks, H + 1)
            collapsed = g_tensor_cache.get_bucketed(
                x.device, R * D, torch.float, "hc_head_coll").view(R, D)
            if R <= 32:
                if self.fn_h is None:
                    self.fn_h = self.fn.half()
                fn = self.fn_h
            else:
                fn = self.fn
            ext.hc_head(x.view(R, H, D), fn, self.base, self.scale,
                        self.rms_eps, self.hc_eps, partials, collapsed)
            return collapsed.view(b, s, D)
        flat = self.norm.forward(x.flatten(2), params)
        mixes = F.linear(flat, self.fn)
        pre = torch.sigmoid(mixes * self.scale + self.base) + self.hc_eps
        return (pre.unsqueeze(-1) * x).sum(dim = 2)
