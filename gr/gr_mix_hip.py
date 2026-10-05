# Loader + launchers for gr_mix.hip (hand-written WMMA f16 gated-residual mix). hipcc genco, cached by source hash,
# launched through torch's libamdhip64 on the current stream (same method as gdn_fused_h_hip.py).
import ctypes, hashlib, os, shutil, subprocess
import torch
from exllamav3.util.hip_lib import load_hip_runtime

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gr_mix.hip")
_state = {}
H, DD, RANK, NTOT, K1 = 4, 2560, 320, 324, 10240

def _hipcc():
    for p in (os.environ.get("EXL3_HIPCC"), os.path.join(os.environ.get("EXL3_ROCM_SDK", ""), "bin", "hipcc"),
              shutil.which("hipcc"), "/opt/rocm/bin/hipcc"):
        if p and os.path.isfile(p): return p
    raise RuntimeError("gr_mix_hip: hipcc not found (set EXL3_HIPCC)")

def compile_hsaco(defs="", arch="gfx1151", src_path=None):
    src_path = src_path or _SRC
    src = open(src_path, "rb").read()
    flags = ["--genco", f"--offload-arch={arch}", "-O3", "-ffp-contract=off"] + [x if x.startswith("-") else f"-D{x}" for x in defs.split()]
    gcc = os.environ.get("EXL3_GCC_INSTALL_DIR", "/usr/lib/gcc/x86_64-linux-gnu/13")
    if os.path.isdir(gcc): flags.append(f"--gcc-install-dir={gcc}")
    tag = hashlib.sha1(src + " ".join(flags).encode()).hexdigest()[:16]
    cache = os.path.join(os.path.expanduser("~/.cache/exllamav3"), f"gr_mix_{arch}_{tag}.hsaco")
    if not os.path.isfile(cache):
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        tmp = cache + f".{os.getpid()}.tmp"
        r = subprocess.run([_hipcc(), *flags, "-o", tmp, src_path], capture_output=True, text=True)
        if r.returncode: raise RuntimeError(r.stderr[-3000:])
        os.replace(tmp, cache)
    return cache

def _load(defs=""):
    if defs in _state: return _state[defs]
    cache = compile_hsaco(defs, torch.cuda.get_device_properties(0).gcnArchName.split(":")[0])
    lib = load_hip_runtime()
    torch.cuda.init()
    mod = ctypes.c_void_p()
    if lib.hipModuleLoad(ctypes.byref(mod), cache.encode()) != 0: raise RuntimeError("gr_mix_hip: hipModuleLoad failed")
    fns = {}
    for n in ("gr_g1", "gr_g2", "gr_g2s", "gr_elem_test", "gr_gate_test"):
        fn = ctypes.c_void_p()
        if lib.hipModuleGetFunction(ctypes.byref(fn), mod, n.encode()) != 0: raise RuntimeError("gr_mix_hip: no function " + n)
        fns[n] = fn
    lib.hipModuleLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p] * 3
    _state["lib"] = lib; _state[defs] = fns; _state["mod", defs] = mod
    return fns

def _launch(fn, grid, block, args):
    P = ctypes.c_void_p
    cargs = [P(a.data_ptr()) if isinstance(a, torch.Tensor) else ctypes.c_int(a) for a in args]
    params = (ctypes.c_void_p * len(cargs))(*[ctypes.cast(ctypes.byref(a), ctypes.c_void_p) for a in cargs])
    r = _state["lib"].hipModuleLaunchKernel(fn, grid[0], grid[1], 1, block, 1, 1, 0, torch.cuda.current_stream().cuda_stream, params, None)
    if r != 0: raise RuntimeError(f"gr_mix_hip: launch failed ({r})")

def _tile(defs, name):
    d = dict(x.split("=") for x in defs.split() if "=" in x and not x.startswith("-"))
    if name in d: return int(d[name])
    import re
    return int(re.search(r"#ifndef " + name + r"\n#define " + name + r"\s+(\d+)", open(_SRC).read()).group(1))

def gr_g1(normed, proj, t, post, defs=""):
    """normed (R, 10240) fp16, proj (324, 10240) fp16 contiguous; t (R, 320) fp16, post (R, 4) fp32 outputs."""
    R = normed.shape[0]
    assert normed.dtype == proj.dtype == t.dtype == torch.half and post.dtype == torch.float and R >= 1
    assert normed.shape[1] == K1 and proj.shape == (NTOT, K1) and t.shape == (R, RANK) and post.shape == (R, H)
    for x in (proj, t, post): assert x.is_contiguous()
    # normed may be a (R, K1) view of a (R, xp) buffer (xp >= K1, multiple of 8); contiguous = pitch K1
    xp = normed.stride(0) if R > 1 else K1
    assert normed.stride(1) == 1 and xp >= K1 and xp % 8 == 0 and normed.storage_offset() % 8 == 0, "gr_g1: normed layout"
    assert normed.storage_offset() + (R - 1) * xp + K1 <= normed.untyped_storage().nbytes() // 2, "gr_g1: normed outside its storage"
    # 64-row tiles for big chunks (R >= 3072: 1.18 vs 1.35 ms at R = 4096, bitwise equal to the 32-row tile; 32-row stays
    # the best below that). EXL3_GR_G1_BM64=0 goes back; an explicit defs string always wins
    if not defs and R >= 3072 and os.environ.get("EXL3_GR_G1_BM64", "1") != "0": defs = "BM1=64"
    bn1, bm1 = _tile(defs, "BN1"), _tile(defs, "BM1")
    _launch(_load(defs)["gr_g1"], ((NTOT + bn1 - 1) // bn1, (R + bm1 - 1) // bm1), 256, [normed, proj, t, post, R, xp])

def gr_g2(t, up, normed, mixed, defs=""):
    """t (R, 320) fp16, up (10240, 320) fp16 contiguous, normed (R*4, 2560) fp16; mixed (R, 2560) fp16 output."""
    R = t.shape[0]
    assert t.dtype == up.dtype == normed.dtype == mixed.dtype == torch.half and R >= 1
    assert t.shape[1] == RANK and up.shape == (K1, RANK) and normed.shape == (R * H, DD) and mixed.shape == (R, DD)
    for x in (t, up, normed, mixed): assert x.is_contiguous()
    _launch(_load(defs)["gr_g2"], (DD // 32, (R + 63) // 64), 256, [t, up, normed, mixed, R])

def gr_g2s(t, up, normed, mixed, defs=""):
    """LDS-staged G2 (same contract as gr_g2; normed may also be the (R, K1) token-pitched view gr_g1 reads, stride(0) >= K1)."""
    R = t.shape[0]
    assert t.dtype == up.dtype == normed.dtype == mixed.dtype == torch.half and R >= 1
    if normed.shape == (R, K1) and normed.stride(0) != K1:
        ngp = normed.stride(0)
        assert normed.stride(1) == 1 and ngp >= K1 and ngp % 8 == 0 and normed.storage_offset() % 8 == 0, "gr_g2s: normed layout"
        assert normed.storage_offset() + (R - 1) * ngp + K1 <= normed.untyped_storage().nbytes() // 2, "gr_g2s: normed outside its storage"
    else:
        assert normed.shape in ((R * H, DD), (R, K1)) and normed.is_contiguous(), "gr_g2s: normed"
        ngp = H * DD
    assert t.shape[1] == RANK and up.shape == (K1, RANK) and mixed.shape == (R, DD)
    for x in (t, up): assert x.is_contiguous()
    # mixed may be a (R, DD) view of a (R, pitch) buffer (pitch >= DD, multiple of 8); contiguous = pitch DD
    pitch = mixed.stride(0) if R > 1 else DD
    assert mixed.stride(1) == 1 and pitch >= DD and pitch % 8 == 0 and mixed.storage_offset() % 8 == 0, "gr_g2s: mixed layout"
    assert mixed.storage_offset() + (R - 1) * pitch + DD <= mixed.untyped_storage().nbytes() // 2, "gr_g2s: mixed outside its storage"
    _launch(_load(defs)["gr_g2s"], (DD // 32, (R + 127) // 128), 256, [t, up, normed, mixed, R, pitch, ngp])
