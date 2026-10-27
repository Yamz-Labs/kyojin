"""Loader of ds41_pf.hip: sparse prefill attention of the DS4.1 serving model (one wave per row, head group, d half)."""
import ctypes, hashlib, os, subprocess
import torch
from exllamav3.util import hip_compiler
from exllamav3.util.hip_lib import load_hip_runtime

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ds41_pf.hip")
_state = {}
_FUNCS = ("ds41_sparse_attn_pf", "ds41_gather_sum", "ds41_rope_rows")


def compile_hsaco(arch="gfx1151"):
    src = open(_SRC, "rb").read()
    flags = ["--genco", f"--offload-arch={arch}", "-O3", "-ffp-contract=off", "--no-gpu-bundle-output"]
    gcc = os.environ.get("EXL3_GCC_INSTALL_DIR", "/usr/lib/gcc/x86_64-linux-gnu/13")
    if os.path.isdir(gcc):
        flags.append(f"--gcc-install-dir={gcc}")
    tag = hashlib.sha1(src + " ".join(flags).encode() + b"\0" + hip_compiler.key().encode()).hexdigest()[:16]
    cache = os.path.join(os.path.expanduser("~/.cache/exllamav3"), f"ds41_pf_{arch}_{tag}.hsaco")
    if not os.path.isfile(cache):
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        tmp = cache + f".{os.getpid()}.tmp"
        r = subprocess.run([hip_compiler.hipcc(), *flags, "-o", tmp, _SRC], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stderr[-3000:])
        os.replace(tmp, cache)
    return cache


def _fn(name):
    if name in _state:
        return _state[name]
    cache = compile_hsaco(torch.cuda.get_device_properties(0).gcnArchName.split(":")[0])
    lib = load_hip_runtime()
    torch.cuda.init()
    mod = ctypes.c_void_p()
    if lib.hipModuleLoad(ctypes.byref(mod), cache.encode()) != 0:
        raise RuntimeError("ds41_pf: hipModuleLoad failed")
    lib.hipModuleLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p] * 3
    _state.update(lib=lib, mod=mod)
    for nm in _FUNCS:
        fn = ctypes.c_void_p()
        if lib.hipModuleGetFunction(ctypes.byref(fn), mod, nm.encode()) != 0:
            raise RuntimeError("ds41_pf: no function " + nm)
        _state[nm] = fn
    return _state[name]


def _launch(name, grid, block, args):
    fn = _fn(name)
    cargs = []
    for a in args:
        if isinstance(a, torch.Tensor):
            cargs.append(ctypes.c_void_p(a.data_ptr()))
        elif isinstance(a, float):
            cargs.append(ctypes.c_float(a))
        elif isinstance(a, tuple):
            cargs.append(ctypes.c_long(a[1]))
        else:
            cargs.append(ctypes.c_int(a))
    params = (ctypes.c_void_p * len(cargs))(*[ctypes.cast(ctypes.byref(a), ctypes.c_void_p) for a in cargs])
    r = _state["lib"].hipModuleLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, torch.cuda.current_stream().cuda_stream, params, None)
    if r != 0:
        raise RuntimeError(f"ds41_pf: launch of {name} failed ({r})")


def gather_sum(res, inv, out, T, K, H):
    """out[t] = sum_k res[inv[t*K+k]] (slot order, inv < 0 skipped); res fp32 [N, H], inv int32 [T*K], out fp32 [T, H] contiguous."""
    assert out.is_contiguous() and out.dtype == torch.float32 and res.is_contiguous() and inv.dtype == torch.int32
    if T:
        _launch("ds41_gather_sum", T, 256, (res, inv, out, K, H))
    return out


def rope_rows(x, fc32, pos, heads, inverse=False):
    """In place rope of the last 64 dims of fp16 x (contiguous, [rows, heads, 512] or [rows, 512]); pos int32 [rows]; fc32 [max_pos, 32, 2]."""
    assert x.dtype == torch.half and x.is_contiguous() and x.size(-1) == 512
    total = x.numel() // 512 * 32
    if total:
        _launch("ds41_rope_rows", (total + 255) // 256, 256, (x, fc32, pos, (None, total), heads, 1 if inverse else 0))
    return x


def usable(q, kv, idx, sink) -> bool:
    return (os.environ.get("DSV41_KERNELS", "1") != "0" and q.is_cuda and q.dtype == kv.dtype == torch.half
            and q.dim() == 4 and q.size(0) == 1 and q.size(2) == 64 and q.size(3) == 512 and q.size(1) > 1
            and kv.dim() == 3 and kv.size(2) == 512 and sink.numel() == 64)


def sparse_attention(q, kv, sink, idx, scale):
    """ops.sparse_attention for q [1, R, 64, 512] fp16, kv [1, N, 512] fp16, idx [1, R, K] int32 (-1 masked), sink [64]."""
    R, K, N = q.size(1), idx.size(-1), kv.size(1)
    q = q.contiguous()
    kv = kv.contiguous()
    idx = idx.to(torch.int32).contiguous()
    sink = sink.float().contiguous()
    o = torch.empty_like(q)
    if K == 0 or N == 0:
        return o.zero_()
    fn = _fn("ds41_sparse_attn_pf")
    cargs = [ctypes.c_void_p(a.data_ptr()) for a in (q, kv, idx, sink, o)] + [ctypes.c_int(K), ctypes.c_int(N), ctypes.c_float(scale)]
    params = (ctypes.c_void_p * len(cargs))(*[ctypes.cast(ctypes.byref(a), ctypes.c_void_p) for a in cargs])
    r = _state["lib"].hipModuleLaunchKernel(fn, R, 1, 1, 512, 1, 1, 0, torch.cuda.current_stream().cuda_stream, params, None)
    if r != 0:
        raise RuntimeError(f"ds41_pf: launch failed ({r})")
    return o
