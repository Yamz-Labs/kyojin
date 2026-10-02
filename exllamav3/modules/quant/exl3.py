from __future__ import annotations
import torch
from ...model.config import Config
from .exl3_lib.quantize import preapply_had_l, preapply_had_r, had_k, had_n
from ...ext import exllamav3_ext as ext
from ...util.tensor import g_tensor_cache
import os
from ...util import profile_opt

MAX_BSZN_GEMV_R = 8  # must match MOE_R_MAX in exllamav3_ext/quant/exl3_dec.cu
# K >= this runs R batch-1 launches instead of one gemv_r launch (A/B knob; 99 = never loop)
DEC_GEMV_R_LOOP_MIN_K = int(os.environ.get("EXL3_DEC_GEMV_R_LOOP_MIN_K", "99"))
# verifyfuse1: EXL3_VERIFY_FUSE=1 routes every R-row (1 < R <= 8) dense EXL3 linear of the MTP/DFlash
# verify forward to exl3_dec_gemv_r (both Hadamards inside, 1 launch) instead of had_in + exl3_gemv +
# had_out (3 launches), and fuses the shared-expert gate+up pair into one gemv_r_multi launch
# (mlp.py). Rows stay bit-exact vs R batch-1 dec_gemv calls (the R=1 plain-decode math). R=1 untouched.
# Mutable at run time for one-load A/B: set VERIFY_FUSE["on"] (the harness also calls block_graph.purge()).
VERIFY_FUSE = {"on": os.environ.get("EXL3_VERIFY_FUSE", "0") != "0"}
AUTO_RECONSTRUCT_THRESHOLD = 144
MAX_RECONSTRUCT_SLICE_N = 32768
RECONSTRUCT_SLICE_GRANULARITY_N = 128

def _hip_gemv_max_rows() -> int:
    try:
        return max(1, min(16, int(os.environ.get("EXL3_GEMV_HIP_MAX_M", "16"))))
    except ValueError:
        return 16


# Runtime rollback cap for the ROCm 16-row GEMV path; larger decode batches reconstruct.
EXL3_GEMV_HIP_MAX_M = _hip_gemv_max_rows()
_EXL3_GEMV_HIP_MMODE1_MAX_M = 8
_EXL3_GEMV_HIP_MMODE2_VARIANTS = frozenset({
    (3, False, True), (4, True, False), (4, False, True),
    (5, False, True), (6, True, False), (6, False, True),
})

no_fused_reconstruct = os.environ.get("EXL3_NO_FUSED_RECONSTRUCT", "0") != "0"

# gfx1151: hipblaslt has no good fp16-in/fp32-out kernel. For a [2048,2560]@[2560,10240] it
# picks Cijk_..._HSS_MT64x32x8 and runs at 6.2 TFLOP/s, where the identical GEMM with an fp16
# output runs at 34 (the card's practical peak). The gated-delta-net projections all declare
# out_dtype=torch.float, so at prefill chunk sizes that one kernel choice was 28% of device
# time. Compute in fp16 and widen afterwards: 4.7x faster including the cast, and the inputs
# are fp16 anyway so the only loss is rounding the product to fp16 before the fp32 consumer.
# Decode (few rows) keeps the direct fp32 path, where the tile choice does not matter.
_f32_via_f16 = os.environ.get("EXL3_HIP_F32OUT_VIA_F16", "1") != "0" and bool(torch.version.hip)
_f32_via_f16_min_rows = int(os.environ.get("EXL3_HIP_F32OUT_VIA_F16_MIN_ROWS", "32"))
_hip_gemv_support_cache: dict[int, bool] = {}

# Batch-1 decode kernels (exl3_dec.cu, gfx11.5): one launch per GEMV with both Hadamard stages
# fused. EXL3_DEC=0 disables them (falls back to exl3_gemv / reconstruct).
EXL3_DEC = os.environ.get("EXL3_DEC", "1") != "0"
_DEC_KB2 = (4, 5, 6, 8, 10, 12)
DEC_SCRATCH_FLOATS = 2 << 20
_dec_workspaces: dict = {}


def dec_workspace(device: torch.device):
    """fp32 scratch (cross-block partial sums) + int32 arrival counters, one pair per device. The
    kernels leave the counters zeroed; all launches share them because they are stream-ordered."""
    key = str(device)
    ws = _dec_workspaces.get(key)
    if ws is None:
        ws = (torch.zeros(DEC_SCRATCH_FLOATS, dtype = torch.float, device = device),
              torch.zeros(4096, dtype = torch.int, device = device))
        _dec_workspaces[key] = ws
    return ws


def dec_supported(inner) -> bool:
    """True when a LinearEXL3 can run the exl3_dec kernels (mul1 codebook, supported K, shapes)."""
    return (
        EXL3_DEC and bool(torch.version.hip) and hasattr(ext, "exl3_dec_gemv") and
        inner.mul1 and not inner.mcg and int(round(inner.K * 2)) in _DEC_KB2 and
        inner.K * 2 == int(round(inner.K * 2)) and
        inner.in_features % 512 == 0 and inner.out_features % 128 == 0 and
        tuple(inner.trellis.shape[:2]) == (inner.in_features // 16, inner.out_features // 16) and
        inner.suh is not None and inner.svh is not None
    )


def dec_moe_supported(inner) -> bool:
    """True when a LinearEXL3 can feed the fused routed-MoE decode (exl3_dec_moe / _union).

    Same as dec_supported() except that both codebooks qualify: the MoE kernels decode the mcg
    codebook as well as mul1 in current packs (GLM-5.3-Flash's published packs are all mcg), while
    the dense exl3_dec_gemv route stays mul1-only.
    """
    return (
        EXL3_DEC and bool(torch.version.hip) and hasattr(ext, "exl3_dec_moe") and
        (inner.mul1 or inner.mcg) and int(round(inner.K * 2)) in _DEC_KB2 and
        inner.K * 2 == int(round(inner.K * 2)) and
        inner.in_features % 512 == 0 and inner.out_features % 128 == 0 and
        tuple(inner.trellis.shape[:2]) == (inner.in_features // 16, inner.out_features // 16) and
        inner.suh is not None and inner.svh is not None
    )


def _hip_gemv_supported(device: torch.device) -> bool:
    """Cache the extension's runtime architecture check for the HIP-only decode route."""
    if not torch.version.hip or not hasattr(ext, "exl3_gemv_supported"):
        return False
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _hip_gemv_support_cache:
        _hip_gemv_support_cache[index] = ext.exl3_gemv_supported(index)
    return _hip_gemv_support_cache[index]


class LinearEXL3:

    quant_type: str = "exl3"

    def __init__(
        self,
        config: Config | None,
        in_features: int,
        out_features: int,
        scale: torch.Tensor | None = None,
        su: torch.Tensor | None = None,
        sv: torch.Tensor | None = None,
        suh: torch.Tensor | None = None,
        svh: torch.Tensor | None = None,
        trellis: torch.Tensor | None = None,
        mcg: torch.Tensor | None = None,
        mul1: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
        transformers_fix: bool = False,
        key: str | None = None
    ):
        assert scale is None, "scale is no longer used"
        assert su is not None or suh is not None, "either su (packed) or suh (unpacked) is required"
        assert sv is not None or svh is not None, "either sv (packed) or svh (unpacked) is required"
        assert trellis is not None, "trellis is required"
        if su is not None: assert su.dtype == torch.int16, "su is wrong datatype"
        if sv is not None: assert sv.dtype == torch.int16, "sv is wrong datatype"
        if suh is not None: assert suh.dtype == torch.half, "suh is wrong datatype"
        if svh is not None: assert svh.dtype == torch.half, "svh is wrong datatype"
        assert trellis.dtype == torch.int16, "trellis is wrong datatype"
        assert len(trellis.shape) == 3, "trellis must have dim = 3"

        if bias is not None and bias.dtype == torch.float: bias = bias.to(torch.half)

        # Not a Module subclass, so the config-or-NullConfig default doesn't apply here; TP imports pass
        # config=None and forward() reads config.infer_params
        if config is None:
            from ...model.config import NullConfig
            config = NullConfig()
        self.config = config
        self.transformers_fix = transformers_fix
        self.key = key

        # self.scale = scale.item()
        self.su = None
        self.sv = None
        self.suh = suh if suh is not None else self.unpack_bf(su)
        self.svh = svh if svh is not None else self.unpack_bf(sv)
        self.trellis = trellis
        self.K = trellis.shape[-1] / 16
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.swap_device = None
        self.out_dtype = out_dtype
        self.default_out_dtype = out_dtype or torch.half

        self.mcg_tensor = mcg
        self.mul1_tensor = mul1
        self.mcg = self.mcg_tensor is not None
        self.mul1 = self.mul1_tensor is not None

        self._fused_reconstruct = None
        self.dec_ok = dec_supported(self)
        self.dec_ok_moe = dec_moe_supported(self)
        self.bsz1_xh_args = (self.trellis.device, (1, self.in_features), self.out_dtype)
        self.bc = ext.BC_LinearEXL3(
            self.trellis,
            self.suh,
            self.svh,
            self.K,
            self.bias,
            self.mcg,
            self.mul1,
            g_tensor_cache.get(*self.bsz1_xh_args)
        )


    def unload(self):
        # g_tensor_cache.drop(*self.bsz1_xh_args)
        pass


    def get_tensors(self, key: str):
        return {
            f"{key}.{subkey}": tensor.contiguous()
            for subkey, tensor in [
                ("su", self.su),
                ("sv", self.sv),
                ("suh", self.suh),
                ("svh", self.svh),
                ("trellis", self.trellis),
                ("bias", self.bias),
                ("mcg", self.mcg_tensor),
                ("mul1", self.mul1_tensor),
            ] if tensor is not None
        }


    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

        if "ovr" in params:
            ovr = params["ovr"]
            if self.key in ovr and ovr[self.key].inner is not self:
                return ovr[self.key].forward(x, params, out_dtype)

        # The EXL3 kernels read x as contiguous rows; a strided view (e.g. a head-group slice
        # of a wider tensor) would be silently misread as interleaved garbage. Producers are
        # responsible for contiguity (a silent copy here would hide a hot-path inefficiency
        # and break CUDA-graph address stability)
        assert x.is_contiguous(), f"LinearEXL3 {self.key}: non-contiguous input {tuple(x.shape)}"

        reconstruct = params.get("reconstruct")
        if not reconstruct:
            rows = x.numel() // x.shape[-1]
            if rows <= AUTO_RECONSTRUCT_THRESHOLD or self.config.infer_params.no_reconstruct:
                if rows == 1 and self.dec_ok:
                    return self.dec_gemv(x, out_dtype)
                # R-row DFlash verify: same kernel family as dec_gemv, one weight
                # tile decode shared across rows -- see exl3_dec_gemv_r's docstring for why this
                # is bit-exact vs R independent dec_gemv calls. Opt-in (EXL3_DEC_MOE_UNION, the
                # same flag dec_norm_route_r/the union MoE branch use, so one flag controls the
                # whole verify path) and capped at MAX_BSZN_GEMV_R: prefill also reaches this
                # branch with much larger row counts, where R sequential-launch overhead would
                # matter and row-chunking hasn't been budgeted for non-power-of-2/huge R.
                if (
                    1 < rows <= MAX_BSZN_GEMV_R and self.dec_ok and
                    hasattr(ext, "exl3_dec_gemv_r") and
                    (VERIFY_FUSE["on"] or
                     os.environ.get("EXL3_VERIFY_GEMV_R", os.environ.get("EXL3_DEC_MOE_UNION", "0")) != "0") and
                    params.get("dflash_verify") and
                    x.is_contiguous()
                ):
                    return self.dec_gemv_r(x, rows, out_dtype)
                if self.bc is not None:
                    dtype = out_dtype or self.default_out_dtype
                    return self.bc.run_alloc(x, self.out_features, dtype == torch.float)
                # ROCm HIP decode GEMV (tensor-core path). NVIDIA builds are untouched:
                # torch.version.hip is None there. Only the shapes the kernel hard-requires
                # are routed here; everything else falls through to reconstruct + hgemm.
                if (torch.version.hip and os.environ.get("EXL3_GEMV", "1") != "0"
                        and hasattr(ext, "exl3_gemv") and _hip_gemv_supported(x.device)
                        and rows <= EXL3_GEMV_HIP_MAX_M
                        and (rows <= _EXL3_GEMV_HIP_MMODE1_MAX_M or
                             (self.K, self.mcg, self.mul1) in _EXL3_GEMV_HIP_MMODE2_VARIANTS)
                        and self.in_features % 128 == 0
                        and self.out_features % 128 == 0
                        and 2 <= self.K <= 6
                        and (self.K == 4 or self.mcg or self.mul1)):
                    return self.hip_gemv(x, out_dtype)

        return self.reconstruct_hgemm(x, out_dtype)


    def dec_gemv(self, x: torch.Tensor, out_dtype):
        # One-row decode GEMV through exl3_dec (input and output Hadamards fused in the kernel)
        out_shape = x.shape[:-1] + (self.out_features,)
        y = torch.empty(out_shape, dtype = out_dtype or self.default_out_dtype, device = x.device)
        if x.dtype != torch.half:
            x = x.half()
        scratch, counters = dec_workspace(x.device)
        ext.exl3_dec_gemv(x.view(1, self.in_features), self.trellis, self.suh, self.svh,
                          y.view(1, self.out_features), scratch, counters, self.K)
        if self.bias is not None:
            y += self.bias
        return y


    def dec_gemv_r(self, x: torch.Tensor, rows: int, out_dtype):
        """R-row decode GEMV (exl3_dec_gemv_r): one launch, each weight tile decoded
        once and applied to every row -- bit-exact vs R independent dec_gemv calls, see the
        kernel's own docstring. Reuses the global dec_workspace() scratch/counters: this call
        never runs concurrently (same stream) with a bsz==1 dec_gemv call reading the same
        counter range, and DEC_SCRATCH_FLOATS was already sized to cover the worst R-row case
        (the MoE union kernel's, which is >= any single dense projection's R*kbs*N here)."""
        out_shape = x.shape[:-1] + (self.out_features,)
        y = torch.empty(out_shape, dtype = out_dtype or self.default_out_dtype, device = x.device)
        if x.dtype != torch.half:
            x = x.half()
        scratch, counters = dec_workspace(x.device)
        if self.K >= DEC_GEMV_R_LOOP_MIN_K:
            # reference path: R batch-1 launches, bit-identical to gemv_r. gemv_r lost at K >= 5 only
            # while its 2-deep tile ring spilled; with ring 1 at RM > 1 it wins (GLM lm_head K 5,
            # N 154880: R 2 1.80 vs 3.56 ms, R 4 1.81 vs 7.12 ms, tools/glm/gemv_r_bench.py)
            x2, y2 = x.view(rows, self.in_features), y.view(rows, self.out_features)
            for r in range(rows):
                ext.exl3_dec_gemv(x2[r:r + 1], self.trellis, self.suh, self.svh,
                                  y2[r:r + 1], scratch, counters, self.K)
        else:
            ext.exl3_dec_gemv_r(x.view(rows, self.in_features), self.trellis, self.suh, self.svh,
                                y.view(rows, self.out_features), scratch, counters, self.K)
        if self.bias is not None:
            y += self.bias
        return y


    def hip_gemv(self, x: torch.Tensor, out_dtype):
        # ROCm decode GEMV: same output contract as reconstruct_hgemm (shape[:-1] +
        # out_features, contiguous, bias applied). A_had is the fp16 workspace the
        # kernel's input-Hadamard stage writes into before the main loop.
        shape = x.shape
        rows = x.numel() // shape[-1]
        out_shape = shape[:-1] + (self.out_features,)
        x_flat = x.view(rows, self.in_features)
        y = torch.empty(out_shape, dtype=out_dtype or self.default_out_dtype, device=x.device)
        y_flat = y.view(rows, self.out_features)
        A_had = g_tensor_cache.get(
            x.device, (rows, self.in_features), torch.half, "exl3_gemv_a_had")
        ext.exl3_gemv(x_flat, self.trellis, y_flat, self.suh, A_had, self.svh, self.mcg, self.mul1)
        if self.bias is not None:
            y += self.bias
        return y


    def unpack_bf(self, bitfield: torch.Tensor):
        # For some reason this operation causes a GPU assert on Transformers. Running on CPU seems to fix it
        device = bitfield.device
        if self.transformers_fix:
            bitfield = bitfield.cpu()

        # (Only used for full reconstruct and loading old models, not during inference)
        bitfield = bitfield.view(torch.uint16).to(torch.int)
        masks = (1 << torch.arange(16)).to(bitfield.device)
        expanded = (bitfield.unsqueeze(-1) & masks) > 0
        expanded = expanded.flatten()
        # NOT torch.where with CPU scalar tensors: that path misses the device guard when the
        # condition lives on a non-current device (observed on torch 2.11.0+cu130) — the kernel
        # launches on the current device's context, faults there, silently zero-fills the output
        # and leaves every other device in the process unusable. Map bool -> {-1, +1} arithmetically
        expanded = 1.0 - expanded.to(torch.float16) * 2.0
        return expanded.contiguous().to(device)


    def reconstruct_hgemm(self, x: torch.Tensor, out_dtype):

        shape = x.shape
        rows = x.numel() // shape[-1]
        out_shape = shape[:-1] + (self.out_features,)
        x = x.view(rows, self.in_features)
        dtype = out_dtype or self.default_out_dtype
        y = torch.empty(out_shape, dtype = dtype, device = x.device)

        # See _f32_via_f16 above: run the gemm into a half buffer and widen, rather than let
        # hipblaslt pick its 6 TFLOP/s fp32-output kernel.
        via_f16 = (_f32_via_f16 and dtype == torch.float and rows >= _f32_via_f16_min_rows)
        if via_f16:
            y_ = torch.empty((rows, self.out_features), dtype = torch.half, device = x.device)
        else:
            y_ = y.view(rows, self.out_features)

        # Fused path: reconstruct emits ORIGINAL-basis weights (both Hadamards + sign
        # vectors folded into the memory-bound reconstruct kernel), so the gemm runs on the
        # raw input and the standalone input/output had_r_128 launches disappear (~14% of
        # long-chunk prefill GPU time). Requires 128-divisible dims (always true for EXL3
        # tensors: both sides are had-transformed at quant time)
        if self._fused_reconstruct is None:
            self._fused_reconstruct = (
                self.in_features % 128 == 0 and self.out_features % 128 == 0
                and not no_fused_reconstruct
            )

        # The fused kernel costs ~4x plain reconstruct (k*n-proportional) while the saved
        # had launches scale with rows*(k+n); breakeven is rows ~400-900 across shapes
        use_fused = self._fused_reconstruct and rows >= 1024

        if use_fused:
            xh = x
        else:
            xh = torch.empty_like(x)
            ext.had_r_128(x, xh, self.suh, None, 1.0)

        if self.out_features <= MAX_RECONSTRUCT_SLICE_N:
            w = torch.empty((self.in_features, self.out_features), dtype = torch.half, device = self.trellis.device)
            if use_fused:
                ext.reconstruct_had_slice(w, self.trellis, self.suh, self.svh, self.K, self.mcg, self.mul1, 0)
            else:
                ext.reconstruct(w, self.trellis, self.K, self.mcg, self.mul1)
            # EXL3_PF_MSPLIT=<rows> (default 0 = off): run big prefill GEMMs as row blocks. Same kernel
            # per block on gfx1151 for the listed shapes, so bit-exact; the smaller A block stays in the
            # MALL (scratch/pfbig1/mb_dense.py). Only shapes with N >= 16384 or K >= 16384.
            mb = int(os.environ.get("EXL3_PF_MSPLIT", "0"))
            if mb and rows > mb and max(self.in_features, self.out_features) >= 16384:
                for r in range(0, rows, mb):
                    ext.hgemm_recon(xh[r:r + mb], w, y_[r:r + mb])
            else:
                ext.hgemm_recon(xh, w, y_)
        else:
            numel_ = self.in_features * MAX_RECONSTRUCT_SLICE_N
            w_ = torch.empty((numel_,), dtype = torch.half, device = self.trellis.device)
            for n_start in range(0, self.out_features, MAX_RECONSTRUCT_SLICE_N):
                n_end = min(n_start + MAX_RECONSTRUCT_SLICE_N, self.out_features)
                numel = self.in_features * (n_end - n_start)
                w = w_[:numel].view(self.in_features, n_end - n_start)
                if use_fused:
                    ext.reconstruct_had_slice(
                        w, self.trellis, self.suh, self.svh[n_start:], self.K, self.mcg, self.mul1, n_start)
                else:
                    ext.reconstruct_slice(w, self.trellis, self.K, self.mcg, self.mul1, n_start)
                ext.hgemm_recon(xh, w, y_[:, n_start:n_end])

        if not use_fused:
            ext.had_r_128(y_, y_, None, self.svh, 1.0)

        if via_f16:
            y.view(rows, self.out_features).copy_(y_)

        if self.bias is not None:
            y += self.bias
        return y


    def get_inner_weight_tensor(self, n_offset: int = 0, n_features: int | None = None):
        w = torch.empty((self.in_features, self.out_features), dtype = torch.half, device = self.trellis.device)
        ext.reconstruct(w, self.trellis, self.K, self.mcg, self.mul1)
        return w


    def get_weight_tensor(self):
        # suh = self.unpack_bf(self.su).unsqueeze(1)
        suh = self.unpack_bf(self.su).unsqueeze(1) if self.su else self.suh.unsqueeze(1)
        svh = self.unpack_bf(self.sv).unsqueeze(0) if self.sv else self.svh.unsqueeze(0)
        w = self.get_inner_weight_tensor()
        w = preapply_had_l(w, had_k)
        w *= suh
        w = preapply_had_r(w, had_n)
        w *= svh
        # w *= self.scale
        return w


    def get_bias_tensor(self) -> torch.Tensor | None:
        return self.bias


    # Swap tensors to CPU (to free some space while quantizing)
    def swap_cpu(self):
        if self.swap_device is not None:
            return
        self.swap_device = self.trellis.device
        if self.su is not None: self.su = self.su.cpu()
        if self.sv is not None: self.sv = self.sv.cpu()
        if self.suh is not None: self.suh = self.suh.cpu()
        if self.svh is not None: self.svh = self.svh.cpu()
        if self.trellis is not None: self.trellis = self.trellis.cpu()
        if self.bias is not None: self.bias = self.bias.cpu()


    def unswap_cpu(self):
        if self.swap_device is None:
            return
        if self.su is not None: self.su = self.su.to(self.swap_device)
        if self.sv is not None: self.sv = self.sv.to(self.swap_device)
        if self.suh is not None: self.suh = self.suh.to(self.swap_device)
        if self.svh is not None: self.svh = self.svh.to(self.swap_device)
        if self.trellis is not None: self.trellis = self.trellis.to(self.swap_device)
        if self.bias is not None: self.bias = self.bias.to(self.swap_device)
        self.swap_device = None


    def tp_export(self, plan, producer):
        return {
            "cls": LinearEXL3,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "suh": producer.send(self.suh),
            "svh": producer.send(self.svh),
            "trellis": producer.send(self.trellis),
            "bias": producer.send(self.bias),
            "mcg": producer.send(self.mcg_tensor),
            "mul1": producer.send(self.mul1_tensor),
            "out_dtype": self.out_dtype,
        }


    @staticmethod
    def tp_import_split(local_context, exported, plan, split):
        consumer = local_context["consumer"]
        device = local_context["device"]
        id_suh = exported["suh"]
        id_svh = exported["svh"]
        id_trellis = exported["trellis"]
        id_bias = exported["bias"]
        mcg = consumer.recv(exported["mcg"], cuda = True)
        mul1 = consumer.recv(exported["mul1"], cuda = True)

        if split is not None:
            split_out, first, last = split
        else:
            split_out, first, last = True, 0, exported["out_features"]

        if split_out:
            suh = consumer.recv(id_suh, cuda = True)
            svh = consumer.recv(id_svh, cuda = True, slice_dim = 0, first = first, last = last)
            trellis = consumer.recv(id_trellis, cuda = True, slice_dim = 1, first = first // 16, last = last // 16)
            bias = consumer.recv(id_bias, cuda = True, slice_dim = 0, first = first, last = last)
            in_features = exported["in_features"]
            out_features = last - first
        else:
            suh = consumer.recv(id_suh, cuda = True, slice_dim = 0, first = first, last = last)
            svh = consumer.recv(id_svh, cuda = True)
            trellis = consumer.recv(id_trellis, cuda = True, slice_dim = 0, first = first // 16, last = last // 16)
            bias = consumer.recv(id_bias, cuda = True) if first == 0 else None
            in_features = last - first
            out_features = exported["out_features"]

        module = LinearEXL3(
            config = None,
            in_features = in_features,
            out_features = out_features,
            scale = None,
            su = None,
            sv = None,
            suh = suh,
            svh = svh,
            trellis = trellis,
            mcg = mcg,
            mul1 = mul1,
            bias = bias,
            out_dtype = exported["out_dtype"],
        )
        return module


    @staticmethod
    def tp_import_split_3(local_context, exported, plan, split_0, split_1, split_2, dbg = False):
        return LinearEXL3.tp_import_split_n(local_context, exported, plan, [split_0, split_1, split_2], dbg)


    @staticmethod
    def tp_import_split_n(local_context, exported, plan, splits, dbg = False):
        consumer = local_context["consumer"]
        device = local_context["device"]
        id_suh = exported["suh"]
        id_svh = exported["svh"]
        id_trellis = exported["trellis"]
        id_bias = exported["bias"]
        mcg = consumer.recv(exported["mcg"], cuda = True)
        mul1 = consumer.recv(exported["mul1"], cuda = True)

        svh_ = []
        trellis_ = []
        bias_ = []
        in_features = 0
        out_features = 0

        for split in splits:
            assert split is not None
            split_out, first, last = split
            assert split_out

            suh = consumer.recv(id_suh, cuda = True)
            svh = consumer.recv(id_svh, cuda = True, slice_dim = 0, first = first, last = last)
            trellis = consumer.recv(id_trellis, cuda = True, slice_dim = 1, first = first // 16, last = last // 16)
            bias = consumer.recv(id_bias, cuda = True, slice_dim = 0, first = first, last = last)
            in_features = exported["in_features"]
            out_features += last - first
            svh_.append(svh)
            trellis_.append(trellis)
            bias_.append(bias)

        svh = torch.cat(svh_, dim = 0)
        trellis = torch.cat(trellis_, dim = 1)
        bias = torch.cat(bias_, dim = 0) if bias_[0] is not None else None

        module = LinearEXL3(
            config = None,
            in_features = in_features,
            out_features = out_features,
            scale = None,
            su = None,
            sv = None,
            suh = suh,
            svh = svh,
            trellis = trellis,
            mcg = mcg,
            mul1 = mul1,
            bias = bias,
            out_dtype = exported["out_dtype"],
        )
        return module
