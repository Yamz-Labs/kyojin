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

# row-padded activations for the prefill GEMMs. An fp16 row of K = 6144 is 12288 B; at that pitch the
# rocBLAS GEMM reads A at about half rate (out_proj 50 % of peak in place). One extra 128 B per row runs the same
# product, bit-exact, at 87-90 %. Keep PAD_ELEMS and the K rule in step with exllamav3_ext/pad_pitch.h.
_PAD_HIP = torch.version.hip is not None
PAD_ELEMS = 64
PAD_MIN_ROWS = 1024          # the fused reconstruct path (no input Hadamard launch) starts at 1024 rows

def _pad_view_ok(x: torch.Tensor) -> bool:
    """x is one (rows, k) matrix with columns contiguous and a single row pitch >= k (leading dims collapsible)."""
    if x.dim() < 2 or x.stride(-1) != 1:
        return False
    k = x.shape[-1]
    if x.stride(-2) < k:
        return False
    for i in range(x.dim() - 3, -1, -1):
        if x.shape[i + 1] != 1 and x.stride(i) != x.stride(i + 1) * x.shape[i + 1]:
            return False
    return True

# Consumer-side padding: a producer we do not own hands a contiguous x whose pitch is a hazard; copy it once into a padded
# buffer (one read + one write of x, bit-exact) before the GEMM. EXL3_PAD_COPY=0 turns it off (read per call).
def pad_copy_in(x: torch.Tensor, k: int) -> torch.Tensor:
    """x is (rows, k) fp16 contiguous with a hazard pitch: return a (rows, k) view at row pitch k + PAD_ELEMS, else x."""
    if (not _PAD_HIP or x.dtype not in (torch.half, torch.bfloat16) or x.dim() != 2 or (k * 2) % 4096 != 0 or x.shape[0] < PAD_MIN_ROWS
            or not x.is_contiguous() or x.data_ptr() % 16 != 0 or os.environ.get("EXL3_PAD_COPY", "1") == "0" or os.environ.get("EXL3_OPROJ_PAD", "1") == "0"):
        return x
    pitch = k + PAD_ELEMS
    buf = torch.empty((x.shape[0], pitch), dtype = x.dtype, device = x.device)
    y = buf[:, :k]
    ext.pad_copy(x, y, pitch)
    return y

def _pad_eligible(lin, k: int, rows: int, params: dict) -> bool:
    """One predicate for producer and consumer: this linear can take a row-padded input of k columns."""
    if not _PAD_HIP or rows < PAD_MIN_ROWS or k % 128 != 0:
        return False
    inner = getattr(lin, "inner", lin)
    if type(inner).__name__ != "LinearEXL3" or inner.in_features != k or inner.out_features % 128 != 0:
        return False
    if no_fused_reconstruct or inner.config.infer_params.no_reconstruct or inner.bias is not None:
        return False
    if any(p in params for p in ("capture", "quant_preserve", "ovr", "reconstruct")):
        return False
    if getattr(lin, "lora_a_tensors", None):
        return False
    if getattr(lin, "pre_scale", 1.0) != 1.0 or getattr(lin, "post_scale", 1.0) != 1.0 or getattr(lin, "softcap", 0.0) != 0.0:
        return False
    return True

def row_pad_pitch(lin, k: int, rows: int, params: dict) -> int:
    """Row pitch (elements) a producer should give the activation of `lin` (0 = leave it contiguous).
    EXL3_OPROJ_PAD=0 turns it off (read per call)."""
    if os.environ.get("EXL3_OPROJ_PAD", "1") == "0" or (k * 2) % 1024 != 0:
        return 0
    return k + PAD_ELEMS if _pad_eligible(lin, k, rows, params) else 0

# hadP: prefill reconstruct cache. A long prefill runs as several row-chunks, and every
# LinearEXL3 reconstructs its (rows-independent!) fp16 weight once per chunk, so the
# same weight is rebuilt once per chunk: 2x at 4K, N-chunk-times beyond. The reconstruct
# is a pure function of the trellis, so keeping the fp16 result and reusing it on the next
# chunk is bit-exact. LRU with a byte budget (EXL3_HAD_PF_FAST_MB, default 2048): the
# unique fp16 weight set of GLM-5.3-Flash is 14.4 GB and cannot all be held.
# EXL3_HAD_PF_FAST=0 (default) leaves the code path untouched.
# EXL3_HAD_PF_FAST_PRIORITY=1 (default) admits a weight only if its measured cost per
# written MB (_had_cost_per_mb, filled in by the unit benchmark) is at or above the best
# cost-per-byte the budget can still afford, so a small budget buys the shapes the kernel
# is worst at rather than an arbitrary LRU slice.
# NOTE: the knobs are read per call, not at import: the A/B harness (glm_base --ab-sets)
# flips os.environ between variants inside one process, so an import-time read would
# silently measure the base arm four times.
def had_pf_fast():
    return os.environ.get("EXL3_HAD_PF_FAST", "0") != "0"


def _had_budget_bytes():
    return int(os.environ.get("EXL3_HAD_PF_FAST_MB", "2048")) << 20


def _had_priority():
    return os.environ.get("EXL3_HAD_PF_FAST_PRIORITY", "1") != "0"


_had_cache: dict = {}
_had_cache_bytes = 0
_had_cache_clock = 0
# hit/miss counters: the A/B must be able to prove the cache actually engaged. A knob read
# at import time while the harness flips os.environ at runtime produced four identical
# "cache" arms before this was added; the count is the cheap guard against that class of
# silent no-op.
_had_stats = {"hit": 0, "miss": 0, "recon": 0}
# measured reconstruct cost in us per fp16 MB written, per (k, n); from mb_had.py
_had_cost_per_mb: dict = {
    (4096, 24576): 6.96, (8192, 4096): 7.38, (4096, 12288): 6.95, (12288, 4096): 7.30,
    (4096, 2048): 6.20, (2048, 4096): 6.32, (4096, 1536): 6.51, (1536, 16384): 7.58,
    (4096, 512): 9.77, (16384, 4096): 7.18, (1536, 4096): 6.51, (4096, 4096): 7.74,
}


def _had_cache_get(inner, n_offset, n):
    """Return the cached fp16 weight for this (trellis, n_offset, n) if it is resident."""
    global _had_cache_clock
    _had_cache_clock += 1
    key = (inner.trellis.data_ptr(), float(inner.K), bool(inner.mcg), bool(inner.mul1),
           int(n_offset), int(n))
    e = _had_cache.get(key)
    if e is None:
        _had_stats["miss"] += 1
        return None
    e[1] = _had_cache_clock
    _had_stats["hit"] += 1
    return e[0]


def _had_cache_admit(inner, n_offset, n):
    """Decide whether to keep this weight, given the budget. Returns the LRU clock value to
    stamp on the new entry, or None.

    A plain LRU is wrong here: one 4K prefill walks all 255 weights in the same order and
    does it twice, so under a 2 GB budget an LRU filled by the first (largest) weights
    evicts everything before it is ever reused. Instead admit by *value per byte* -- once the
    budget is actually short, a shape joins only if its measured reconstruct cost per fp16
    MB is at least as good as the weakest shape already resident, so a small budget buys
    exactly the shapes the kernel is worst at. The ordering deliberately does NOT bind
    while the budget has room (see the comment below): that would cap large budgets too.

    Admission is sticky: a shape already resident stays resident, and the first call of a
    new shape does not pay anything.
    """
    global _had_cache_bytes, _had_cache_clock
    nbytes = inner.in_features * n * 2
    budget = _had_budget_bytes()
    if nbytes > budget:
        return None
    cost = _had_cost_per_mb.get((inner.in_features, n))
    # Value-ordering is a *scarcity* policy, so it must only bind once the budget is
    # actually short. Applied unconditionally it also throttles a large budget: the first
    # weak shape to be admitted becomes the floor and every worse shape is refused for the
    # rest of the session, which capped a 24 GB budget at 10.9 GB resident / 104 hits.
    if _had_priority() and cost is not None and _had_cache \
            and _had_cache_bytes + nbytes > budget * 0.95:
        costs = [_had_cost_per_mb[e[3]] for e in _had_cache.values() if e[3] in _had_cost_per_mb]
        if costs and cost < min(costs):
            return None
    while _had_cache and _had_cache_bytes + nbytes > budget:
        oldest = min(_had_cache.items(), key=lambda kv: kv[1][1])[0]
        _had_cache_bytes -= _had_cache.pop(oldest)[2]
    return _had_cache_clock


def _had_cache_put(inner, n_offset, n, w, clock):
    global _had_cache_bytes
    nbytes = inner.in_features * n * 2
    key = (inner.trellis.data_ptr(), float(inner.K), bool(inner.mcg), bool(inner.mul1),
           int(n_offset), int(n))
    if clock is None:
        return
    # entry: [w, clock, nbytes, (in_features, n)] -- a list, not a tuple: the LRU stamp
    # (element 1) is updated in place on every hit.
    _had_cache[key] = [w, clock, nbytes, (int(inner.in_features), int(n))]
    _had_cache_bytes += nbytes

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


# specrow1: wide-N R-row GEMV (lm_head, N=248320: R*kbs*N partial sums > DEC_SCRATCH_FLOATS even at R=2) used to fall back to
# R batch-1 launches, each re-reading the whole weight (R x 1.36 ms). A dedicated scratch lets one gemv_r launch carry all rows
# (same kernel, same per-row reduction order, so bit-exact). Counters are shared (R * ceil(N/512) <= 3880 <= 4096 for R <= 8).
# Mutable at run time for one-load A/B: set WIDE_R["on"].
WIDE_R = {"on": os.environ.get("EXL3_VERIFY_WIDE_R", "1") != "0"}
WIDE_SCRATCH_FLOATS = 16 << 20
_wide_scratch: dict = {}


def wide_scratch(device: torch.device):
    key = str(device)
    t = _wide_scratch.get(key)
    if t is None:
        t = _wide_scratch[key] = torch.zeros(WIDE_SCRATCH_FLOATS, dtype = torch.float, device = device)
    return t


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
    codebook as well as mul1 since REPORT-21 (GLM-5.3-Flash's published packs are all mcg), while
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
        if not x.is_contiguous():
            # a producer's row-padded output (row_pad_pitch): same product, bit-exact, GEMM reads it at full rate.
            # Only the reconstruct + hgemm path takes a pitch (hgemm checks the view against its storage)
            assert _pad_view_ok(x) and _pad_eligible(self, x.shape[-1], x.numel() // x.shape[-1], params), \
                f"LinearEXL3 {self.key}: non-contiguous input {tuple(x.shape)}"
            return self.reconstruct_hgemm(x, out_dtype)

        reconstruct = params.get("reconstruct")
        if not reconstruct:
            rows = x.numel() // x.shape[-1]
            if rows <= AUTO_RECONSTRUCT_THRESHOLD or self.config.infer_params.no_reconstruct:
                if rows == 1 and self.dec_ok and not params.get("moe_valu"):
                    return self.dec_gemv(x, out_dtype)
                # R-row DFlash verify (REPORT-17): same kernel family as dec_gemv, one weight
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
                    params.get("dflash_verify") and not params.get("moe_valu") and
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
        """R-row decode GEMV (exl3_dec_gemv_r, REPORT-17): one launch, each weight tile decoded
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
        loop = self.K >= DEC_GEMV_R_LOOP_MIN_K
        if not loop:
            try:
                ext.exl3_dec_gemv_r(x.view(rows, self.in_features), self.trellis, self.suh, self.svh,
                                    y.view(rows, self.out_features), scratch, counters, self.K)
            except RuntimeError as e:
                # The R-row partial sums of a very wide N (lm_head) can exceed the shared scratch; the
                # size check runs before any launch, so fall back to the batch-1 loop (same order)
                if "scratch too small" not in str(e): raise
                loop = True
                if WIDE_R["on"] and rows <= 8:
                    try:
                        ext.exl3_dec_gemv_r(x.view(rows, self.in_features), self.trellis, self.suh, self.svh,
                                            y.view(rows, self.out_features), wide_scratch(x.device), counters, self.K)
                        loop = False
                    except RuntimeError as e2:
                        if "scratch too small" not in str(e2): raise
        if loop:
            # reference path: R batch-1 launches, bit-identical to gemv_r. gemv_r lost at K >= 5 only
            # while its 2-deep tile ring spilled; with ring 1 at RM > 1 it wins (GLM lm_head K 5,
            # N 154880: R 2 1.80 vs 3.56 ms, R 4 1.81 vs 7.12 ms, tools/glm/gemv_r_bench.py)
            x2, y2 = x.view(rows, self.in_features), y.view(rows, self.out_features)
            for r in range(rows):
                ext.exl3_dec_gemv(x2[r:r + 1], self.trellis, self.suh, self.svh,
                                  y2[r:r + 1], scratch, counters, self.K)
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


    def _had_weight(self, w, n_offset, n):
        """reconstruct_had_slice into w, or reuse the cached fp16 weight for this
        (trellis, n_offset, n). The reconstruct is a pure function of the trellis and
        does not depend on the row count, so a hit is bit-exact. Returns the tensor to
        hand to hgemm_recon (a resident cache entry, or w after a fresh reconstruct)."""
        if not had_pf_fast():
            ext.reconstruct_had_slice(w, self.trellis, self.suh, self.svh if n_offset == 0
                                      else self.svh[n_offset:], self.K, self.mcg, self.mul1, n_offset)
            return w
        hit = _had_cache_get(self, n_offset, n)
        if hit is not None:
            return hit
        _had_stats["recon"] += 1
        clock = _had_cache_admit(self, n_offset, n)
        if clock is None:
            ext.reconstruct_had_slice(w, self.trellis, self.suh, self.svh if n_offset == 0
                                      else self.svh[n_offset:], self.K, self.mcg, self.mul1, n_offset)
            return w
        # admitted: the caller allocates a fresh w per call, so w itself can be the resident
        # entry (no copy) -- nothing else holds a reference to it
        ext.reconstruct_had_slice(w, self.trellis, self.suh, self.svh if n_offset == 0
                                  else self.svh[n_offset:], self.K, self.mcg, self.mul1, n_offset)
        _had_cache_put(self, n_offset, n, w, clock)
        return w

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
            xh = pad_copy_in(x, self.in_features)
        else:
            xh = torch.empty_like(x)
            ext.had_r_128(x, xh, self.suh, None, 1.0)

        if self.out_features <= MAX_RECONSTRUCT_SLICE_N:
            w = torch.empty((self.in_features, self.out_features), dtype = torch.half, device = self.trellis.device)
            if use_fused:
                w = self._had_weight(w, 0, self.out_features)
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
                    w = self._had_weight(w, n_start, n_end - n_start)
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
