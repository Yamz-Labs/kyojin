# EXL3_GDN_HF (default off): Qwen GDN prefill recurrence with the 16-row-chunk WY kernel adapted from hipfire (gdn_hf.hip, Apache-2.0).
# One kernel per layer replaces kkt_solve + gdn_fused_h_o (the gated norm still runs after). F16 WMMA operands with fp32 accumulation
# (S, T, Vnew rounded to f16 per 16-row chunk): NOT bitwise equal to the Triton / gdn_fused_h chain; quality-gated (KLD).
# Compiled with the ROCm SDK hipcc on first use (cached by source hash), launched on torch's stream like gdn_fused_h_hip.
import ctypes, hashlib, os, subprocess
import torch
from exllamav3.util import hip_compiler
from exllamav3.util.hip_lib import load_hip_runtime

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gdn_hf.hip")
_state = {}


def _load():
    if "fn" in _state:
        return _state["fn"]
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    src = open(_SRC, "rb").read()
    flags = ["--genco", f"--offload-arch={arch}", "-O3", "-DKYO_RAW=1"]
    gcc = os.environ.get("EXL3_GCC_INSTALL_DIR", "/usr/lib/gcc/x86_64-linux-gnu/13")
    if os.path.isdir(gcc):
        flags.append(f"--gcc-install-dir={gcc}")
    tag = hashlib.sha1(src + " ".join(flags).encode() + b"\0" + hip_compiler.key().encode()).hexdigest()[:16]
    cache = os.path.join(os.path.expanduser("~/.cache/exllamav3"), f"gdn_hf_{arch}_{tag}.hsaco")
    if not os.path.isfile(cache):
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        tmp = cache + f".{os.getpid()}.tmp"
        subprocess.run([hip_compiler.hipcc(), *flags, "-o", tmp, _SRC], check=True, capture_output=True)
        os.replace(tmp, cache)
    lib = load_hip_runtime()
    torch.cuda.init()
    mod, fn = ctypes.c_void_p(), ctypes.c_void_p()
    if lib.hipModuleLoad(ctypes.byref(mod), cache.encode()) != 0:
        raise RuntimeError("gdn_hf_hip: hipModuleLoad failed")
    if lib.hipModuleGetFunction(ctypes.byref(fn), mod, b"gated_delta_chunk_gate_wmma") != 0:
        raise RuntimeError("gdn_hf_hip: hipModuleGetFunction failed")
    lib.hipModuleLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p] * 3
    _state.update(lib=lib, mod=mod, fn=fn)
    return fn


def gdn_hf_ok(q_raw, v, qn, kn, g, beta, initial_state):
    """q_raw, v: views of the conv output row (q at column 0, v at column 2 H K, same token pitch); qn, kn [1, T, H, 128] bf16
    contiguous (l2-normalized); g [1, T, HV] fp32 raw log decay (natural log, before the chunk cumsum); beta [1, T, HV]."""
    if os.environ.get("EXL3_GDN_HF", "0") != "1" or _state.get("fail"):
        return False
    B, T, H, K = qn.shape
    HV, V = v.shape[2], v.shape[3]
    bf = torch.bfloat16
    ok = (B == 1 and K == 128 and V == 128 and HV % H == 0 and T >= 1
            and qn.dtype == kn.dtype == v.dtype == q_raw.dtype == bf and qn.is_contiguous() and kn.is_contiguous() and kn.shape == qn.shape
            and v.stride(3) == 1 and v.stride(2) == V and q_raw.stride(1) == v.stride(1) and v.stride(1) >= 2 * H * K + HV * V
            and v.data_ptr() == q_raw.data_ptr() + 2 * H * K * 2 and q_raw.stride(3) == 1
            and g.shape == (1, T, HV) and g.dtype == torch.float32 and beta.shape == (1, T, HV)
            and (initial_state is None or (initial_state.dtype == torch.float32 and initial_state.shape == (1, HV, K, V))))
    if not ok and not _state.get("warned"):
        _state["warned"] = True
        print(f" -- EXL3_GDN_HF: shape/layout gate refused (q {tuple(q_raw.shape)} {q_raw.dtype} stride {q_raw.stride()}, v stride {v.stride()}, "
              f"v - q {v.data_ptr() - q_raw.data_ptr()} B, g {tuple(g.shape)} {g.dtype}, beta {tuple(beta.shape)}); Triton/HIP chain used", flush=True)
    return ok


def chunk_gdn_hf(q_raw, v, qn, kn, g, beta, scale, initial_state, output_final_state, ckpt=None):
    """Returns (o [1, T, HV, 128] bf16, final_state or None). ckpt: None, a dict or a list of <= 8 dicts {"s": fp32 [1, HV, K, V]
    contiguous, "chunk": n} (gdn_fused_h_hip contract): the state after row 64 n is written to ck["s"] and ck["ok"] set."""
    try:
        fn = _load()
    except Exception:
        _state["fail"] = True     # no hipcc / no module: the Triton chain for the rest of the process
        raise
    _, T, H, K = qn.shape
    HV = v.shape[2]
    g = g.contiguous()
    beta = beta.float().contiguous()
    state = (torch.zeros(1, HV, K, K, device=v.device, dtype=torch.float32) if initial_state is None
             else initial_state.contiguous().clone())          # the kernel reads the entry state and writes the final one in place
    o = torch.empty(1, T, HV, 128, device=v.device, dtype=torch.bfloat16)
    P = ctypes.c_void_p
    args = [P(q_raw.data_ptr()), P(qn.data_ptr()), P(kn.data_ptr()), P(g.data_ptr()), P(beta.data_ptr()), P(state.data_ptr()),
            P(0), P(0), P(o.data_ptr()), ctypes.c_int(T), ctypes.c_int(v.stride(1)), ctypes.c_int(H), ctypes.c_int(HV),
            ctypes.c_float(scale), ctypes.c_int(1)]
    cks = [] if ckpt is None else (list(ckpt) if isinstance(ckpt, (list, tuple)) else [ckpt])
    cks = [c for c in cks if 0 < 64 * c["chunk"] <= T]
    assert len(cks) <= 8
    for c in cks:
        assert c["s"].dtype == torch.float32 and c["s"].is_contiguous() and c["s"].shape == (1, HV, K, K)
    for i in range(8):
        c = cks[i] if i < len(cks) else None
        args += [P(0 if c is None else c["s"].data_ptr()), ctypes.c_int(-1 if c is None else 64 * c["chunk"])]
    params = (ctypes.c_void_p * len(args))(*[ctypes.cast(ctypes.byref(a), ctypes.c_void_p) for a in args])
    stream = torch.cuda.current_stream().cuda_stream
    r = _state["lib"].hipModuleLaunchKernel(fn, HV, 1, 1, 256, 1, 1, 0, stream, params, None)
    if r != 0:
        _state["fail"] = True
        raise RuntimeError(f"gdn_hf_hip: launch failed ({r})")
    for c in cks:
        c["ok"] = True
    _state["calls"] = _state.get("calls", 0) + 1
    return o, (state if output_final_state else None)
