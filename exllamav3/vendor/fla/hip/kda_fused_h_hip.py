# Loader for the hand-written HIP fused KDA state loop (kda_fused_h.hip). Compiles a gfx code object with the
# ROCm SDK hipcc on first use (cached by source hash) and launches it through torch's own libamdhip64 via
# hipModuleLaunchKernel on the current torch stream: no extension rebuild, no second HIP runtime.

import ctypes, hashlib, os, shutil, subprocess
import torch

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kda_fused_h.hip")
_state = {}


def _hipcc():
    for p in (os.environ.get("EXL3_HIPCC"),
              os.path.join(os.environ.get("EXL3_ROCM_SDK", ""), "bin", "hipcc"),
              "~/kyojin/.venv-gfx1151/lib/python3.12/site-packages/_rocm_sdk_devel/bin/hipcc",
              shutil.which("hipcc"), "/opt/rocm/bin/hipcc"):
        if p and os.path.isfile(p):
            return p
    raise RuntimeError("kda_fused_h_hip: hipcc not found (set EXL3_HIPCC)")


def _compile(defs, arch):
    # hsaco path for these defines (compiled once, cached by source + flags hash); needs no GPU
    src = open(_SRC, "rb").read()
    flags = ["--genco", f"--offload-arch={arch}", "-O3", "-ffp-contract=off"] + [x if x.startswith("-") else f"-D{x}" for x in defs.split()]
    gcc = os.environ.get("EXL3_GCC_INSTALL_DIR", "/usr/lib/gcc/x86_64-linux-gnu/13")
    if os.path.isdir(gcc):
        flags.append(f"--gcc-install-dir={gcc}")
    tag = hashlib.sha1(src + " ".join(flags).encode()).hexdigest()[:16]
    cache = os.path.join(os.path.expanduser("~/.cache/exllamav3"), f"kda_fused_h_{arch}_{tag}.hsaco")
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
    lib = ctypes.CDLL(os.path.join(os.path.dirname(torch.__file__), "lib", "libamdhip64.so"))
    torch.cuda.init()
    mod, fn = ctypes.c_void_p(), ctypes.c_void_p()
    if lib.hipModuleLoad(ctypes.byref(mod), cache.encode()) != 0:
        raise RuntimeError("kda_fused_h_hip: hipModuleLoad failed")
    if lib.hipModuleGetFunction(ctypes.byref(fn), mod, b"kda_fused_h_o") != 0:
        raise RuntimeError("kda_fused_h_hip: hipModuleGetFunction failed")
    lib.hipModuleLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p] * 3
    _state["lib"] = lib
    _state[defs] = fn
    _state["mod", defs] = mod
    return fn


def chunk_kda_fwd_h_fused_hip(q, k, v, gk, beta, Akk, Aqk, scale, initial_state=None, output_final_state=False,
                              chunk_size=64, o=None, final_state=None, defs="", ckpt=None):
    """Same contract as chunk_kda_fwd_h_fused(fuse_o=True): returns (None, None, o, final_state).
    ckpt: optional dict {"s": fp32 [B, HV, K, V] contiguous, "chunk": n}; the kernel also writes the state after
    chunk n - 1 (row 64 n) to ckpt["s"] and sets ckpt["ok"] (EXL3_MIDCHUNK_CKPT=2)."""
    B, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[-1]
    assert K == 128 and V == 128 and chunk_size == 64 and HV % H == 0
    assert q.dtype == k.dtype == v.dtype == beta.dtype == Akk.dtype == Aqk.dtype == torch.bfloat16 and gk.dtype == torch.float32
    for t in (q, k, v, gk, beta, Akk, Aqk):
        assert t.is_contiguous()
    pf = os.environ.get("EXL3_KDA_PF_SPLIT", "0") if not defs else ""
    if pf == "2":
        # kdapf2: spill-free build (0 VGPR spills vs 462): sched barriers (SB), opaque lane
        # indices (OPQ), pinned WMMA accumulators (OPV), uniform o-store base (ADR), no LSR,
        # plus PH1=4 PF=1 next-chunk register prefetch. Bitwise-identical to defs="".
        # Microbench T=4096 HV=64 (test box): 8.70 -> 5.40 ms (-38 %). Default off.
        defs = "BRL=1 UNI=1 OPQ=15 OPV=11 SB=15 ADR=1 PH1=4 PF=1 -mllvm -disable-lsr"
    elif pf == "1":
        # kdapf1: BRL (branchless f2bf, single basic block) + UNI (wave-uniform rows).
        # Bitwise-identical to defs="" (sweep T=4096: 10.979 -> 8.615 ms/layer, -21.5 %),
        # default off. Name kept per brief; the 2-WG split (lever a) was ruled NO-GO by
        # construction (w/tril/vn couple all halves; see scratch/kdapf1/NOTES.md).
        defs = "BRL=1 UNI=1"
    fn = _load(defs)
    if o is None:
        o = torch.empty_like(v)
    if output_final_state and final_state is None:
        final_state = k.new_empty(B, HV, K, V, dtype=torch.float32)
    h0 = initial_state
    if h0 is not None:
        assert h0.dtype == torch.float32 and h0.is_contiguous()
    P = ctypes.c_void_p
    args = [P(q.data_ptr()), P(k.data_ptr()), P(v.data_ptr()), P(gk.data_ptr()), P(beta.data_ptr()),
            P(Akk.data_ptr()), P(Aqk.data_ptr()), P(o.data_ptr()), P(0 if h0 is None else h0.data_ptr()),
            P(final_state.data_ptr() if output_final_state else 0), ctypes.c_float(scale), ctypes.c_int(T),
            ctypes.c_int(H), ctypes.c_int(HV)]
    # ckpt: one dict or a list of up to 2 (EXL3_PF_NO_TAIL adds the prompt's last full page)
    cks = [] if ckpt is None else (list(ckpt) if isinstance(ckpt, (list, tuple)) else [ckpt])
    assert len(cks) <= 2
    cks = [c for c in cks if 0 < c["chunk"] <= (T + 63) // 64]
    for c in cks:
        assert c["s"].dtype == torch.float32 and c["s"].is_contiguous() and c["s"].shape == (B, HV, K, V)
    for i in range(2):
        c = cks[i] if i < len(cks) else None
        args += [P(0 if c is None else c["s"].data_ptr()), ctypes.c_int(-1 if c is None else c["chunk"])]
    params = (ctypes.c_void_p * len(args))(*[ctypes.cast(ctypes.byref(a), ctypes.c_void_p) for a in args])
    stream = torch.cuda.current_stream().cuda_stream
    r = _state["lib"].hipModuleLaunchKernel(fn, B * HV, 1, 1, 288 if "PFW=" in defs and "PFW=0" not in defs else 256, 1, 1, 0, stream, params, None)
    if r != 0:
        raise RuntimeError(f"kda_fused_h_hip: launch failed ({r})")
    for c in cks:
        c["ok"] = True
    return None, None, o, (final_state if output_final_state else None)


def read_prof(defs, n):
    # PROF=1 builds only: per-(WG, wave) phase cycle sums, uint32[n * 8 * 9]
    import numpy as np
    lib, mod = _state["lib"], _state["mod", defs]
    ptr, size = ctypes.c_void_p(), ctypes.c_size_t()
    if lib.hipModuleGetGlobal(ctypes.byref(ptr), ctypes.byref(size), mod, b"kda_prof") != 0:
        raise RuntimeError("kda_prof missing")
    torch.cuda.synchronize()
    out = np.zeros(n * 8 * 9, dtype=np.uint32)
    lib.hipMemcpyDtoH.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    if lib.hipMemcpyDtoH(out.ctypes.data, ptr, out.nbytes) != 0:
        raise RuntimeError("hipMemcpyDtoH failed")
    return out.reshape(n, 8, 9)
