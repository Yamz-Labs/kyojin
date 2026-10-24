# EXL3_KDA_HF (default off): GLM KDA prefill recurrence with 16-row WY chunks (kda_hf.hip, derived from hipfire's GDN kernel,
# Apache-2.0). fla's token_parallel kernel builds the exact 16x16 diagonal blocks, then one kernel per layer replaces kda_inter_off
# + inter_merge + kda_fused_h_o (EXL3_KDA_HF_DIAG=1: blocks computed in the kernel, token_parallel skipped). F16 WMMA operands with fp32 accumulation: NOT bitwise equal to the 64-row
# chain; quality-gated (KLD). Compiled with the ROCm SDK hipcc on first use (cached by source hash), launched on torch's stream.
import ctypes, hashlib, os, subprocess
import torch
import triton
from exllamav3.util import hip_compiler
from exllamav3.util.hip_lib import load_hip_runtime

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kda_hf.hip")
_state = {}


def _load(defs=""):
    # defs: extra -D defines (experiments only, e.g. "KHF_TP=1"); one module per defines string
    if ("fn", defs) in _state:
        return _state["fn", defs]
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    src = open(_SRC, "rb").read()
    flags = ["--genco", f"--offload-arch={arch}", "-O3", "-ffp-contract=off"] + [f"-D{x}" for x in defs.split()]
    gcc = os.environ.get("EXL3_GCC_INSTALL_DIR", "/usr/lib/gcc/x86_64-linux-gnu/13")
    if os.path.isdir(gcc):
        flags.append(f"--gcc-install-dir={gcc}")
    tag = hashlib.sha1(src + " ".join(flags).encode() + b"\0" + hip_compiler.key().encode()).hexdigest()[:16]
    cache = os.path.join(os.path.expanduser("~/.cache/exllamav3"), f"kda_hf_{arch}_{tag}.hsaco")
    if not os.path.isfile(cache):
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        tmp = cache + f".{os.getpid()}.tmp"
        subprocess.run([hip_compiler.hipcc(), *flags, "-o", tmp, _SRC], check=True, capture_output=True)
        os.replace(tmp, cache)
    lib = load_hip_runtime()
    torch.cuda.init()
    mod, fn = ctypes.c_void_p(), ctypes.c_void_p()
    if lib.hipModuleLoad(ctypes.byref(mod), cache.encode()) != 0:
        raise RuntimeError("kda_hf_hip: hipModuleLoad failed")
    if lib.hipModuleGetFunction(ctypes.byref(fn), mod, b"kda_chunk16_wmma") != 0:
        raise RuntimeError("kda_hf_hip: hipModuleGetFunction failed")
    lib.hipModuleLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p] * 3
    _state["lib"] = lib
    _state["fn", defs] = fn
    _state["mod", defs] = mod
    return fn


def kda_hf_ok(q, k, v, g, beta, initial_state, chunk_size):
    """q, k [1, T, H, 128] bf16 (l2-normalized), v [1, T, HV, 128] bf16, g [1, T, HV, 128] fp32 = 64-row-local log2 cumsum,
    beta [1, T, HV] bf16, all contiguous."""
    if os.environ.get("EXL3_KDA_HF", "0") != "1" or _state.get("fail"):
        return False
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    bf = torch.bfloat16
    return (B == 1 and K == 128 and V == 128 and chunk_size == 64 and HV % H == 0 and k.shape == q.shape
            and q.dtype == k.dtype == v.dtype == beta.dtype == bf and g.dtype == torch.float32 and g.shape == (1, T, HV, K)
            and beta.shape == (1, T, HV) and all(t.is_contiguous() for t in (q, k, v, g, beta))
            and (initial_state is None or (initial_state.dtype == torch.float32 and initial_state.shape == (1, HV, K, V))))


def chunk_kda_hf(q, k, v, g, beta, scale, initial_state, output_final_state, ckpt=None, defs=None):
    """Returns (o [1, T, HV, 128] bf16, final_state or None). ckpt: None, a dict or a list of <= 4 dicts {"s": fp32 [1, HV, K, V]
    contiguous, "chunk": n} (kda_fused_h_hip contract): the state after row 64 n is written to ck["s"] and ck["ok"] set."""
    if defs is None:
        # diagonal 16x16 blocks: from fla's token_parallel kernel (exact fp32 pairwise decay) by default; EXL3_KDA_HF_DIAG=1 (read
        # per call) computes them in the kernel with bf16 WMMA (faster, but heavier per-token error tails on real GLM inputs)
        defs = "" if os.environ.get("EXL3_KDA_HF_DIAG", "0") == "1" else "KHF_TP=1"
    try:
        fn = _load(defs)
    except Exception:
        _state["fail"] = True     # no hipcc / no module: the Triton/HIP chain for the rest of the process
        raise
    from ..kda_chunk_intra_token_parallel import chunk_kda_fwd_kernel_intra_token_parallel as _tp
    _, T, H, K = q.shape
    HV = v.shape[2]
    # diagonal 16x16 blocks, launched as chunk_kda_fwd_intra_pinned does (BH 16, 16 warps): Aqk [T, HV, 64] (columns j % 64),
    # Akkd [T, HV, 16] fp32; only j <= i (Aqk) / j < i (Akkd) entries are written and read
    Aqk = Akkd = None
    if "KHF_TP" in defs:
        Aqk = torch.empty(1, T, HV, 64, device=q.device, dtype=q.dtype)
        Akkd = torch.empty(1, T, HV, 16, device=q.device, dtype=torch.float32)
    if "KHF_TP" in defs:      # timing/accuracy variant: diagonal blocks from token_parallel (default: computed in the kernel)
        _tp.fn.fn[(T, triton.cdiv(HV, 16))](q, k, g, beta, Aqk, Akkd, scale, None, 1, T, H=H, HV=HV, K=K, BT=64,
                                            BC=16, BH=16, IS_VARLEN=False, num_warps=16, num_stages=1)
    state = (torch.zeros(1, HV, K, K, device=q.device, dtype=torch.float32) if initial_state is None
             else initial_state.contiguous().clone())          # the kernel reads the entry state and writes the final one in place
    o = torch.empty(1, T, HV, 128, device=q.device, dtype=torch.bfloat16)
    P = ctypes.c_void_p
    args = [P(q.data_ptr()), P(k.data_ptr()), P(v.data_ptr()), P(g.data_ptr()), P(beta.data_ptr()), P(0 if Akkd is None else Akkd.data_ptr()),
            P(0 if Aqk is None else Aqk.data_ptr()), P(state.data_ptr()), P(o.data_ptr()), ctypes.c_int(T), ctypes.c_int(H), ctypes.c_int(HV),
            ctypes.c_float(scale)]
    cks = [] if ckpt is None else (list(ckpt) if isinstance(ckpt, (list, tuple)) else [ckpt])
    cks = [c for c in cks if 0 < 64 * c["chunk"] <= T]
    assert len(cks) <= 4      # EXL3_NC2_CKPT4: an 8192-row chunk carries 3 interior checkpoints + the last full page
    for c in cks:
        assert c["s"].dtype == torch.float32 and c["s"].is_contiguous() and c["s"].shape == (1, HV, K, K)
    for i in range(4):
        c = cks[i] if i < len(cks) else None
        args += [P(0 if c is None else c["s"].data_ptr()), ctypes.c_int(-1 if c is None else 64 * c["chunk"])]
    params = (ctypes.c_void_p * len(args))(*[ctypes.cast(ctypes.byref(a), ctypes.c_void_p) for a in args])
    stream = torch.cuda.current_stream().cuda_stream
    r = _state["lib"].hipModuleLaunchKernel(fn, HV, 1, 1, 256, 1, 1, 0, stream, params, None)
    if r != 0:
        _state["fail"] = True
        raise RuntimeError(f"kda_hf_hip: launch failed ({r})")
    for c in cks:
        c["ok"] = True
    _state["calls"] = _state.get("calls", 0) + 1
    return o, (state if output_final_state else None)
