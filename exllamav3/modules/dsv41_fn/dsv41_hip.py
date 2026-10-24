# Loader + launchers for dsv41_hip.hip (DeepSeek-V4.1-Flash small-op kernels, gfx1151). hipcc genco on first use (cached by
# source + header hash), launched through torch's libamdhip64 on the current stream (capturable in a HIP graph).
# Each function names the reference op chain it replaces (exllamav3/dsv41/{ops,quant,engram}.py).
import ctypes, hashlib, os, subprocess
import torch
from exllamav3.util import hip_compiler
from exllamav3.util.hip_lib import load_hip_runtime

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_DIR, "dsv41_hip.hip")
_HDR = os.path.join(_DIR, "dsv41_num.h")
_state = {}
_FUNCS = ("ds41_kv_prep", "ds41_rmsnorm_rows", "ds41_fq_rows", "ds41_idx_q_prep", "ds41_idx_k_store", "ds41_pool_store",
          "ds41_compress", "ds41_compress_state", "ds41_hc_mix_part", "ds41_hc_mix_part_h", "ds41_hc_mix_fin", "ds41_hc_cn",
          "ds41_router_scores", "ds41_router_scores16", "ds41_fq_rows_v2", "ds41_router_topk", "ds41_engram_comp", "ds41_engram_hash", "ds41_mxfp4_decode", "ds41_fp8_decode",
          "ds41_engram_gate", "ds41_attn_dec", "ds41_rmsnorm_wide",
          "ds41_dgemv_ff", "ds41_dgemv_bf", "ds41_dgemv_hf", "ds41_dgemv_fh", "ds41_dgemv_bh", "ds41_dgemv_hh")
HC_PART_CH = 20        # chunks of 1024 columns of the 20480-wide stream row per mix block (rows = 1 -> 20 blocks, one per CU)
FQ_FP8, FQ_FP4_UE8M0, FQ_FP4_E4M3 = 0, 1, 2


def compile_hsaco(arch="gfx1151", src_path=None):
    src_path = src_path or _SRC
    src = open(src_path, "rb").read() + open(_HDR, "rb").read()
    flags = ["--genco", f"--offload-arch={arch}", "-O3", "-ffp-contract=off"]
    gcc = os.environ.get("EXL3_GCC_INSTALL_DIR", "/usr/lib/gcc/x86_64-linux-gnu/13")
    if os.path.isdir(gcc):
        flags.append(f"--gcc-install-dir={gcc}")
    tag = hashlib.sha1(src + " ".join(flags).encode() + b"\0" + hip_compiler.key().encode()).hexdigest()[:16]
    cache = os.path.join(os.path.expanduser("~/.cache/exllamav3"), f"dsv41_{arch}_{tag}.hsaco")
    if not os.path.isfile(cache):
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        tmp = cache + f".{os.getpid()}.tmp"
        r = subprocess.run([hip_compiler.hipcc(), *flags, "-o", tmp, src_path], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stderr[-3000:])
        os.replace(tmp, cache)
    return cache


def _load():
    if "fns" in _state:
        return _state["fns"]
    cache = compile_hsaco(torch.cuda.get_device_properties(0).gcnArchName.split(":")[0])
    lib = load_hip_runtime()
    torch.cuda.init()
    mod = ctypes.c_void_p()
    if lib.hipModuleLoad(ctypes.byref(mod), cache.encode()) != 0:
        raise RuntimeError("dsv41_hip: hipModuleLoad failed")
    fns = {}
    for n in _FUNCS:
        fn = ctypes.c_void_p()
        if lib.hipModuleGetFunction(ctypes.byref(fn), mod, n.encode()) != 0:
            raise RuntimeError("dsv41_hip: no function " + n)
        fns[n] = fn
    lib.hipModuleLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p] * 3
    _state["lib"], _state["fns"], _state["mod"] = lib, fns, mod
    return fns


def _pack(args):
    """ctypes values + the pointer array hipModuleLaunchKernel wants. Tensors are passed by data_ptr (the caller keeps them alive)."""
    cargs = []
    for a in args:
        if a is None:
            cargs.append(ctypes.c_void_p(0))
        elif isinstance(a, torch.Tensor):
            cargs.append(ctypes.c_void_p(a.data_ptr()))
        elif isinstance(a, float):
            cargs.append(ctypes.c_float(a))
        elif isinstance(a, tuple):            # ("l", value): long
            cargs.append(ctypes.c_long(a[1]))
        else:
            cargs.append(ctypes.c_int(a))
    params = (ctypes.c_void_p * len(cargs))(*[ctypes.cast(ctypes.byref(a), ctypes.c_void_p) for a in cargs])
    return cargs, params


def _go(name, g, block, params):
    emu = _state.get("emu")          # tests only: CPU emulation of the kernel model (tests/dsv41/hipemu), CPU tensors
    if emu:
        return emu(name, g, block, params)
    fn = _load()[name]
    r = _state["lib"].hipModuleLaunchKernel(fn, g[0], g[1], g[2], block, 1, 1, 0,
                                            torch.cuda.current_stream().cuda_stream, params, None)
    if r != 0:
        raise RuntimeError(f"dsv41_hip: launch of {name} failed ({r})")


def _launch(name, grid, block, args):
    cargs, params = _pack(args)
    _go(name, tuple(grid) + (1,) * (3 - len(grid)), block, params)


class Launch:
    """A kernel launch with its argument pack built once (static buffers): one ctypes call per replay instead of ~15 for the
    convenience wrappers, and a stable pack for HIP-graph capture. `set(i, v)` updates a scalar argument in place (e.g. start)."""

    def __init__(self, name, grid, block, args):
        self.name, self.block = name, block
        self.grid = tuple(grid) + (1,) * (3 - len(grid))
        self._keep = [a for a in args if isinstance(a, torch.Tensor)]
        self.cargs, self.params = _pack(args)

    def set(self, i, v):
        self.cargs[i].value = v

    def __call__(self):
        _go(self.name, self.grid, self.block, self.params)


def prepare(name, grid, block, args):
    """Launch object for any kernel in the code object (names: dsv41_hip._FUNCS; args as the wrappers pass them to _launch)."""
    return Launch(name, grid, block, args)


def _c(*ts, align=4):
    for t in ts:
        if t is not None:
            assert (t.is_cuda or _state.get("emu")) and t.is_contiguous() and t.data_ptr() % align == 0


def rope_table_f32(fc: torch.Tensor) -> torch.Tensor:
    """ops.rope_tables output (complex64 [max_pos, 32]) -> fp32 [max_pos, 32, 2] (cos, sin) for the kernels."""
    return torch.view_as_real(fc).contiguous()


def _i32(t):
    return t if t.dtype == torch.int32 else t.int()


# ---------------------------------------------------------------------------------------------------- attention
def kv_prep(kv, w, fc, pos, slot, out, ring, eps):
    """rmsnorm(kv, w) -> rope(last 64) at pos -> fp8 fake-quant (ue8m0/32) -> out rows and ring[slot] rows.
    Replaces model.py:157-170 (rmsnorm, rope_apply_, act_quant_fp8_, ring write). kv [R,512] fp16, w [512] fp32,
    fc [max_pos,32,2] fp32, pos/slot [R] int32 (slot -1 = no ring write), out [R,512] fp16 or None, ring [win,512] fp16 or None."""
    R = kv.size(0)
    assert kv.dtype == torch.half and kv.size(1) == 512 and w.dtype == torch.float and w.numel() == 512
    _c(kv, w, fc, pos, slot, out, ring)
    _launch("ds41_kv_prep", (R,), 32, (kv, w, fc, pos, slot, out, ring, float(eps)))


def rmsnorm_rows(x, w, y, eps):
    """y = fp16(w * x * rsqrt(mean(x^2) + eps)), x fp16 [R,N], w fp32 [N] (ops.rmsnorm; y may alias x)."""
    R, N = x.shape
    assert x.dtype == torch.half and w.dtype == torch.float and N % 32 == 0 and y.dtype == torch.half
    _c(x, w, y)
    if N % 8 == 0 and N <= 2048 and not _state.get("emu") and x.data_ptr() % 16 == 0 and y.data_ptr() % 16 == 0:
        _launch("ds41_rmsnorm_wide", (R,), 256, (x, w, y, N, float(eps)))
    else:
        _launch("ds41_rmsnorm_rows", (R,), 32, (x, w, y, N, float(eps)))


def fq_rows(x, mode):
    """In-place fake-quant of fp16 x (any shape, last dim a multiple of the group): mode FQ_FP8 (o before wo_a, group 32),
    FQ_FP4_UE8M0 (group 32), FQ_FP4_E4M3 (group 16). Replaces quant.act_quant_fp8_ / fp4_act_quant_."""
    g = 16 if mode == FQ_FP4_E4M3 else 32
    assert x.dtype == torch.half and x.size(-1) % g == 0
    n = x.numel() // g
    if (os.environ.get("DSV41_FQ_V1", "0") == "0" and x.is_contiguous() and x.data_ptr() % 16 == 0
            and (x.is_cuda or _state.get("emu"))):
        # v2: vectorised, same bits as v1 (see ds41_fq_rows_v2)
        nv = x.numel() // 8
        _launch("ds41_fq_rows_v2", ((nv + 255) // 256,), 256, (x, ("l", nv), mode))
        return
    _c(x)
    _launch("ds41_fq_rows", ((n + 255) // 256,), 256, (x, ("l", n), mode))


def idx_q_prep(q, fc, pos, heads=32):
    """Indexer q [R, heads*128] fp16 in place: rope(dims 64..127 of each head) at pos[r], then fp4 ue8m0/32."""
    R = q.size(0)
    assert q.dtype == torch.half and q.numel() == R * heads * 128
    _c(q, fc, pos)
    _launch("ds41_idx_q_prep", ((R * heads * 4 + 255) // 256,), 256, (q, fc, pos, R, heads))


def idx_k_store(k, w, fc, ck_pos, dst, pool, eps):
    """pool[dst[r]] = fp4_ue8m0(rope(rmsnorm(k[r], w), ck_pos[r])): k [R,128] fp16, w [128] fp32, pool [rows,128] fp16."""
    R = k.size(0)
    assert k.dtype == torch.half and k.size(1) == 128 and w.dtype == torch.float and pool.dtype == torch.half
    _c(k, w, fc, ck_pos, dst, pool)
    _launch("ds41_idx_k_store", (R,), 32, (k, w, fc, ck_pos, dst, pool, float(eps)))


def pool_store(lat, fc, ck_pos, dst, pool):
    """pool[dst[r]] = fp4_e4m3scale16(rope(lat[r], ck_pos[r])): lat [R,512] fp16 (compressor output), pool [rows,512] fp16."""
    R = lat.size(0)
    assert lat.dtype == torch.half and lat.size(1) == 512 and pool.dtype == torch.half
    _c(lat, fc, ck_pos, dst, pool)
    _launch("ds41_pool_store", (R,), 32, (lat, fc, ck_pos, dst, pool))


def compress(kv32, sc32, kst, sst, w, out, start, ratio, eps):
    """Ratio-`ratio` (<= 8) compressor for R rows at positions start..start+R-1: softmax-pool completed groups into out
    [(start+R)//ratio - start//ratio, 512] fp16 and update the incomplete-group state kst/sst [ratio,512] fp32.
    Replaces Graph._compressor (r > 1). Returns the number of groups written."""
    R = kv32.size(0)
    ng = (start + R) // ratio - start // ratio
    assert kv32.dtype == torch.float and sc32.dtype == torch.float and ratio <= 8 and out.size(0) >= ng
    _c(kv32, sc32, kst, sst, w, out)
    if ng:
        _launch("ds41_compress", (ng,), 32, (kv32, sc32, kst, sst, w, out, start, ratio, float(eps)))
    _launch("ds41_compress_state", (ratio,), 256, (kv32, sc32, kst, sst, start, R, ratio))
    return ng


# ---------------------------------------------------------------------------------------------------- hyper-connections
def hc_mix(h, fn, scale, base, pre, post, comb, part, iters, norm_eps, hc_eps):
    """ops.hc_mixes: h [R,4,5120] fp32, fn [24,20480] fp32 or fp16, scale [3], base [24] fp32 -> pre [R,4], post [R,4],
    comb [R,4,4] fp32 (part: fp32 workspace of R*HC_PART_CH*25)."""
    R = h.size(0)
    assert h.dtype == torch.float and h.is_contiguous() and h.numel() == R * 4 * 5120 and fn.shape == (24, 4 * 5120)
    _c(h, fn, scale, base, pre, post, comb, part, align=4)
    assert h.data_ptr() % 16 == 0 and fn.data_ptr() % 16 == 0
    name = "ds41_hc_mix_part" if fn.dtype == torch.float else "ds41_hc_mix_part_h"
    assert fn.dtype in (torch.float, torch.half)
    _launch(name, (HC_PART_CH, R), 256, (h, fn, part, HC_PART_CH))
    _launch("ds41_hc_mix_fin", (R,), 32, (part, scale, base, pre, post, comb, HC_PART_CH, int(iters), float(norm_eps),
                                          float(hc_eps)))


def hc_cn(h, y, post, comb, pre, w, out, eps, apply=True):
    """[hc_post(y, h, post, comb) in place on h] then x = rmsnorm(hc_pre(h, pre), w) -> out [R,5120] fp16.
    pre is the (delayed) pre of an earlier site. y fp16 or fp32 [R,5120] (ignored when apply is False)."""
    R = h.size(0)
    assert h.dtype == torch.float and w.dtype == torch.float and out.dtype == torch.half
    _c(h, y, post, comb, pre, w, out)
    assert h.data_ptr() % 16 == 0 and out.data_ptr() % 8 == 0
    if apply:
        assert y.dtype in (torch.half, torch.float) and y.shape == (R, 5120) and y.data_ptr() % 16 == 0
    _launch("ds41_hc_cn", (R,), 256, (h, y, int(y is not None and y.dtype == torch.float), post, comb, pre, w, out,
                                      int(apply), float(eps)))


# ---------------------------------------------------------------------------------------------------- router
def router(x, W, bias, sel, wts, scores, topk, scale, temp=1.0, norm=True, wts_f32=None):
    """ops.route for the sqrtsoftplus gate: x [R,5120] fp16, W [E,5120] fp32/fp16/bf16, bias [E] fp32 -> sel [R,topk] int64,
    wts [R,topk] fp16 (wts_f32 [R,topk] fp32 optional); scores [R,E] fp32 workspace (the unbiased scores)."""
    R, E = x.size(0), W.size(0)
    wtype = {torch.float: 0, torch.half: 1, torch.bfloat16: 2}[W.dtype]
    assert x.dtype == torch.half and x.size(1) == 5120 and E % 8 == 0 and E <= 512 and topk <= 16
    assert bias.dtype == torch.float and sel.dtype == torch.int64 and wts.dtype == torch.half
    _c(x, W, bias, sel, wts, scores, wts_f32, align=16)
    if R >= 32 and E % 32 == 0:
        _launch("ds41_router_scores16", (E // 32, (R + 7) // 8), 256, (x, W, wtype, scores, R, E, 1.0 / float(temp)))
    else:
        _launch("ds41_router_scores", (E // 8, (R + 3) // 4), 256, (x, W, wtype, scores, R, E, 1.0 / float(temp)))
    _launch("ds41_router_topk", (R,), 32, (scores, bias, sel, wts, wts_f32, E, topk, int(norm), float(scale)))


# ---------------------------------------------------------------------------------------------------- engram
def engram_comp(ids, tmap, cache, mask, start):
    """cache[start + r] = token_map[ids[r]] (-1 where mask is 0). ids int64 [R], tmap int32 [vocab], cache int64 [max_seq]."""
    R = ids.numel()
    assert ids.dtype == torch.int64 and tmap.dtype == torch.int32 and cache.dtype == torch.int64
    _c(ids, tmap, cache, mask, align=8)
    _launch("ds41_engram_comp", ((R + 255) // 256,), 256, (ids, tmap, cache, mask, start, R))


def engram_hash(cache, mult, primes, offsets, rows, start, R, pad):
    """NgramHasher rows: rows int64 [R, nl, (ng-1)*nh] from the compressed-id cache (already holding this chunk).
    mult [nl, ng], primes/offsets [nl, (ng-1)*nh] int64."""
    nl, ng = mult.shape
    nh = primes.shape[1] // (ng - 1)
    assert rows.dtype == torch.int64 and nl * (ng - 1) * nh <= 64 and ng <= 8
    _c(cache, mult, primes, offsets, rows, align=8)
    _launch("ds41_engram_hash", (R,), 64, (cache, mult, primes, offsets, rows, start, nl, ng, nh, ("l", int(pad))))


def mxfp4_decode(raw, out, inv=None):
    """raw uint8 [N,136] -> out fp16 [M,256] (M = N, or len(inv) rows gathered through inv int32)."""
    M = out.size(0)
    assert raw.dtype == torch.uint8 and raw.size(1) == 136 and out.dtype == torch.half and out.size(1) == 256
    _c(raw, inv, out, align=4)
    assert out.data_ptr() % 16 == 0
    _launch("ds41_mxfp4_decode", ((M * 32 + 255) // 256,), 256, (raw, inv, out, ("l", M)))


def fp8_decode(raw, out, inv=None):
    """raw uint8 [N,264] (official e4m3 + e8m0 rows) -> out fp16 [M,256] (M = N, or len(inv) rows gathered through inv int32)."""
    M = out.size(0)
    assert raw.dtype == torch.uint8 and raw.size(1) == 264 and out.dtype == torch.half and out.size(1) == 256
    _c(raw, inv, out, align=4)
    assert out.data_ptr() % 16 == 0
    _launch("ds41_fp8_decode", ((M * 32 + 255) // 256,), 256, (raw, inv, out, ("l", M)))


def engram_gate(h, kv, qk, eps, dim=5120, clamp_value=1e-6):
    """engram_inject's gate + residual update in place on h [R,4,5120] fp32; kv [R, 5*5120] fp16 (wkv output), qk [4,5120] fp32."""
    R = h.size(0)
    assert h.dtype == torch.float and kv.dtype == torch.half and qk.dtype == torch.float and dim == 5120
    _c(h, kv, qk, align=16)
    _launch("ds41_engram_gate", (R,), 128, (h, kv, qk, float(eps), float(dim ** -0.5), float(clamp_value)))


def attn_dec(q, ring, comp, cidx, nc, coff, sink, fc, pos, out, scale):
    """Decode sparse attention, 64 heads, q [64*512] fp16 (un-roped), ring [128,512] fp16 holding the current token, comp
    [rows,512] fp16 + cidx [nc] int32 (value - coff = row, -1 masked; both None when nc == 0), sink [64] fp32, fc [max_pos,32,2],
    pos [1] int32, out [64*512] fp16 (rope-inverted, fp8 fake-quantised). Replaces the rope / window idxs / cat /
    sparse_attention / rope / act_quant chain of DSV41Serving._attention for one token."""
    assert q.dtype == torch.half and ring.dtype == torch.half and sink.dtype == torch.float and out.dtype == torch.half
    assert nc <= 1152 - 128
    _c(q, ring, comp, cidx, sink, fc, pos, out, align=4)
    _launch("ds41_attn_dec", (32,), 256, (q, ring, comp, cidx, nc, coff, sink, fc, pos, out, float(scale)))


def dense_gemv(W, wkind, x, y):
    """y[r] = W @ x[r] for x [R<=8, K] fp32 or fp16 (y same dtype), W [N, K] fp32 (wkind 'f'), bf16 ('b') or fp16 ('h'), K % 256 == 0.
    Replaces F.linear of the compressor / indexer projections at decode (Tensile fp32 GEMM at ~45 GB/s)."""
    R, K = x.shape
    N = W.size(0)
    xk = {torch.float: "f", torch.half: "h"}[x.dtype]
    assert R <= 8 and K % 256 == 0 and W.size(1) == K and y.dtype == x.dtype and y.shape == (R, N)
    _c(W, x, y, align=16)
    _launch("ds41_dgemv_" + wkind + xk, ((N + 7) // 8,), 256, (W, x, y, K, N, R))

