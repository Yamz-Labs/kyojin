# Loader + launchers for ple_hip.hip (hand HIP kernels: PLE streams chain after the projections, deinterleave_qg).
# hipcc genco (cached by source hash), launched through torch's libamdhip64 on the current stream.
import ctypes, hashlib, os, shutil, subprocess
import torch
from exllamav3.util.hip_lib import load_hip_runtime

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ple_hip.hip")
_state = {}
HS, DD, ST = 4, 2560, 9
_FUNCS = ("ple_rs", "ple_gate", "ple_norm_conv", "ple_copy_state", "ple_conv_out", "deinterleave_qg")


def _hipcc():
    for p in (os.environ.get("EXL3_HIPCC"), os.path.join(os.environ.get("EXL3_ROCM_SDK", ""), "bin", "hipcc"),
              "$EXL3_ROOT",
              "$EXL3_ROOT",
              shutil.which("hipcc"), "/opt/rocm/bin/hipcc"):
        if p and os.path.isfile(p):
            return p
    raise RuntimeError("ple_hip: hipcc not found (set EXL3_HIPCC)")


def compile_hsaco(arch="gfx1151", src_path=None):
    src_path = src_path or _SRC
    src = open(src_path, "rb").read()
    flags = ["--genco", f"--offload-arch={arch}", "-O3", "-ffp-contract=off"]
    gcc = os.environ.get("EXL3_GCC_INSTALL_DIR", "/usr/lib/gcc/x86_64-linux-gnu/13")
    if os.path.isdir(gcc):
        flags.append(f"--gcc-install-dir={gcc}")
    tag = hashlib.sha1(src + " ".join(flags).encode()).hexdigest()[:16]
    cache = os.path.join(os.path.expanduser("~/.cache/exllamav3"), f"ple_{arch}_{tag}.hsaco")
    if not os.path.isfile(cache):
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        tmp = cache + f".{os.getpid()}.tmp"
        r = subprocess.run([_hipcc(), *flags, "-o", tmp, src_path], capture_output=True, text=True)
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
        raise RuntimeError("ple_hip: hipModuleLoad failed")
    fns = {}
    for n in _FUNCS:
        fn = ctypes.c_void_p()
        if lib.hipModuleGetFunction(ctypes.byref(fn), mod, n.encode()) != 0:
            raise RuntimeError("ple_hip: no function " + n)
        fns[n] = fn
    lib.hipModuleLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p] * 3
    _state["lib"], _state["fns"], _state["mod"] = lib, fns, mod
    return fns


def _launch(name, grid, block, args):
    fn = _load()[name]
    cargs = []
    for a in args:
        if isinstance(a, torch.Tensor):
            cargs.append(ctypes.c_void_p(a.data_ptr()))
        elif isinstance(a, float):
            cargs.append(ctypes.c_float(a))
        elif isinstance(a, tuple):            # ("sz", value): size_t
            cargs.append(ctypes.c_size_t(a[1]))
        else:
            cargs.append(ctypes.c_int(a))
    params = (ctypes.c_void_p * len(cargs))(*[ctypes.cast(ctypes.byref(a), ctypes.c_void_p) for a in cargs])
    g = tuple(grid) + (1,) * (3 - len(grid))
    r = _state["lib"].hipModuleLaunchKernel(fn, g[0], g[1], g[2], block, 1, 1, 0,
                                            torch.cuda.current_stream().cuda_stream, params, None)
    if r != 0:
        raise RuntimeError(f"ple_hip: launch of {name} failed ({r})")


def supported(key, value, streams, conv_state, norm_w, conv_w, constant_scale, hc_mult, hidden):
    """Shapes and dtypes the kernels were proved for (everything else keeps the torch reference)."""
    if hc_mult != HS or hidden != DD or constant_scale != 1.0:
        return False
    ts = (key, value, streams, conv_w) + tuple(norm_w)
    if not all(t.is_cuda and t.is_contiguous() and t.data_ptr() % 16 == 0 for t in ts):
        return False
    if key.dtype != torch.half or value.dtype != torch.half or streams.dtype != torch.float or conv_w.dtype != torch.half:
        return False
    if any(w.dtype != torch.bfloat16 or w.numel() != HS * DD for w in norm_w):
        return False
    if conv_w.numel() != HS * DD * 4:
        return False
    if conv_state is not None and (conv_state.dtype != torch.half or not conv_state.is_contiguous()):
        return False
    return True


def ple_chain(key, value, streams, conv_state, norm_key_w, norm_query_w, norm_conv_w, conv_w, eps, gate_scale,
              delta=None, conv_stream=None, dbg=None):
    """
    key (B*S, H*D) half, value (B*S, D) half, streams (B, S, H, D) fp32, conv_state (B, H*D, 9) half or None,
    norm weights (H*D) bf16 (raw, +1 applied inside), conv_w (H*D, 1, 4) half.
    Returns delta (B, S, H, D) fp32 and conv_stream (B, H*D, 9 + S) half, bitwise equal to the torch chain.
    """
    B, S = streams.shape[:2]
    RT, RH = B * S, B * S * HS
    dev = streams.device
    if delta is None:
        delta = torch.empty_like(streams)
    if conv_stream is None:
        conv_stream = torch.empty((B, HS * DD, ST + S), dtype=torch.half, device=dev)
    rs = torch.empty(2 * RH, dtype=torch.float, device=dev)
    gs = torch.empty(RH, dtype=torch.float, device=dev)
    eps = float(eps); gate_scale = float(gate_scale)
    nrow = 2 * RH
    _launch("ple_rs", ((nrow + 7) // 8,), 256, (key, streams, rs, RH, eps))
    _launch("ple_gate", ((RH + 63) // 64,), 64, (key, streams, rs, norm_key_w, norm_query_w, gs, RH, gate_scale))
    _launch("ple_norm_conv", ((RT + 63) // 64, HS), 256, (value, gs, norm_conv_w, conv_stream, RT, S, eps))
    n = B * HS * DD * ST
    cs = conv_state if conv_state is not None else conv_stream
    _launch("ple_copy_state", ((n + 255) // 256,), 256, (cs, conv_stream, B, S, 1 if conv_state is not None else 0))
    _launch("ple_conv_out", (HS * DD // 64, (S + 63) // 64, B), 256, (conv_stream, conv_w, value, gs, delta, S))
    if dbg is not None:
        dbg.update(rs=rs, gs=gs)
    return delta, conv_stream


def deinterleave_qg(qg, q, g, head_dim):
    assert head_dim % 8 == 0 and qg.is_contiguous() and q.is_contiguous() and g.is_contiguous()
    assert q.numel() == g.numel() and q.numel() * 2 == qg.numel()
    assert all(t.data_ptr() % 16 == 0 for t in (qg, q, g))
    n8 = q.numel() // 8
    _launch("deinterleave_qg", ((n8 + 255) // 256,), 256, (qg, q, g, head_dim // 8, ("sz", n8)))
