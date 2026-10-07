# Loader for the hand-written HIP fused GDN chunk kernel (gdn_fused_h.hip). Compiles a gfx code object with the
# ROCm SDK hipcc on first use (cached by source hash) and launches it through torch's own libamdhip64 via
# hipModuleLaunchKernel on the current torch stream: no extension rebuild, no second HIP runtime.

import ctypes, hashlib, os, shutil, subprocess
import torch
from exllamav3.util import hip_compiler
from exllamav3.util.hip_lib import load_hip_runtime

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gdn_fused_h.hip")
_state = {}


def _hipcc():
    return hip_compiler.hipcc()


def _compile(defs, arch):
    # hsaco path for these defines (compiled once, cached by source + flags hash); needs no GPU
    src = open(_SRC, "rb").read()
    flags = ["--genco", f"--offload-arch={arch}", "-O3", "-ffp-contract=off"] + [x if x.startswith("-") else f"-D{x}" for x in defs.split()]
    gcc = os.environ.get("EXL3_GCC_INSTALL_DIR", "/usr/lib/gcc/x86_64-linux-gnu/13")
    if os.path.isdir(gcc):
        flags.append(f"--gcc-install-dir={gcc}")
    tag = hashlib.sha1(src + " ".join(flags).encode() + b"\0" + hip_compiler.key().encode()).hexdigest()[:16]
    cache = os.path.join(os.path.expanduser("~/.cache/exllamav3"), f"gdn_fused_h_{arch}_{tag}.hsaco")
    if not os.path.isfile(cache):
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        tmp = cache + f".{os.getpid()}.tmp"
        subprocess.run([_hipcc(), *flags, "-o", tmp, _SRC], check=True, capture_output=True)
        os.replace(tmp, cache)
    return cache


def _load(defs=""):
    # defs: extra -D defines (timing experiments only, e.g. "ABL=4"); one module per defines string
    if defs in _state:
        return _state[defs]
    cache = _compile(defs, torch.cuda.get_device_properties(0).gcnArchName.split(":")[0])
    lib = load_hip_runtime()
    torch.cuda.init()
    mod, fn = ctypes.c_void_p(), ctypes.c_void_p()
    if lib.hipModuleLoad(ctypes.byref(mod), cache.encode()) != 0:
        raise RuntimeError("gdn_fused_h_hip: hipModuleLoad failed")
    if lib.hipModuleGetFunction(ctypes.byref(fn), mod, b"gdn_fused_h_o") != 0:
        raise RuntimeError("gdn_fused_h_hip: hipModuleGetFunction failed")
    lib.hipModuleLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p] * 3
    _state["lib"] = lib
    _state[defs] = fn
    _state["mod", defs] = mod
    return fn


def chunk_gdn_fwd_fused_hip(q, k, v, g, beta, A, scale, initial_state=None, output_final_state=False, defs="", dbg=None, grid=None, ckpt=None):
    """recompute_w_u + chunk_h + chunk_o in one kernel. q, k [B,T,H,128] bf16; v [B,T,HV,128] bf16; g [B,T,HV] fp32
    (chunk-local cumsum, log2 domain); beta [B,T,HV] bf16; A [B,T,HV,64] bf16 (solved kkt). Returns (o, final_state).
    dbg: optional dict of bf16 buffers w, u, vn [T,HV,128] and h [NT,HV,128,128] (build with defs containing DBG=1).
    ckpt: optional dict or list of up to 2 dicts {"s": fp32 [B, HV, K, V] contiguous, "chunk": n}: the kernel (CKPT=1 build, a separate
    code object; the default build is untouched) also writes the state after chunk n - 1 (row 64 n) to ckpt["s"] and sets ckpt["ok"]."""
    B, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[-1]
    assert K == 128 and V == 128 and HV % H == 0
    assert q.dtype == k.dtype == v.dtype == beta.dtype == A.dtype == torch.bfloat16 and g.dtype == torch.float32
    for t in (q, k, g, beta, A):
        assert t.is_contiguous()
    # v: row pitch (elements) may exceed HV * V (slice of the conv output); rows of one batch item are pitch apart, items T * pitch apart
    PV = v.stride(1)
    assert v.stride(3) == 1 and v.stride(2) == V and PV >= HV * V and PV % 2 == 0 and v.storage_offset() % 2 == 0
    assert B == 1 or v.stride(0) == T * PV
    assert q.shape == k.shape and g.shape == (B, T, HV) and beta.shape == (B, T, HV) and A.shape == (B, T, HV, 64)
    if not defs and os.environ.get("EXL3_GDN_OST", "1") == "1":
        defs = "OST=1"   # EXL3_GDN_OST: o tile staged through LDS, 16 B row stores (same bits, see gdn_fused_h.hip)
    cks = [] if ckpt is None else (list(ckpt) if isinstance(ckpt, (list, tuple)) else [ckpt])
    assert len(cks) <= 2
    cks = [c for c in cks if 0 < c["chunk"] <= (T + 63) // 64]
    for c in cks:
        assert c["s"].dtype == torch.float32 and c["s"].is_contiguous() and c["s"].shape == (B, HV, K, V)
    if cks:
        defs = (defs + " CKPT=1").strip()
    fn = _load(defs)
    o = torch.empty(B, T, HV, V, device=v.device, dtype=v.dtype)
    final_state = k.new_empty(B, HV, K, V, dtype=torch.float32) if output_final_state else None
    h0 = initial_state
    if h0 is not None:
        assert h0.dtype == torch.float32 and h0.is_contiguous() and h0.shape == (B, HV, K, V)
    P = ctypes.c_void_p
    d = dbg or {}
    args = [P(q.data_ptr()), P(k.data_ptr()), P(v.data_ptr()), P(g.data_ptr()), P(beta.data_ptr()), P(A.data_ptr()),
            P(o.data_ptr()), P(0 if h0 is None else h0.data_ptr()), P(final_state.data_ptr() if output_final_state else 0),
            ctypes.c_float(scale), ctypes.c_int(T), ctypes.c_int(H), ctypes.c_int(HV), ctypes.c_int(PV)]
    for n in ("w", "u", "vn", "h"):
        args.append(P(d[n].data_ptr() if n in d else 0))
    if cks:
        for i in range(2):
            c = cks[i] if i < len(cks) else None
            args += [P(0 if c is None else c["s"].data_ptr()), ctypes.c_int(-1 if c is None else c["chunk"])]
    params = (ctypes.c_void_p * len(args))(*[ctypes.cast(ctypes.byref(a), ctypes.c_void_p) for a in args])
    stream = torch.cuda.current_stream().cuda_stream
    nwg = B * HV if grid is None else int(grid)   # grid < B * HV launches the first nwg heads only (timing experiments)
    assert 0 < nwg <= B * HV
    r = _state["lib"].hipModuleLaunchKernel(fn, nwg, 1, 1, 256, 1, 1, 0, stream, params, None)
    if r != 0:
        raise RuntimeError(f"gdn_fused_h_hip: launch failed ({r})")
    for c in cks:
        c["ok"] = True
    return o, final_state
