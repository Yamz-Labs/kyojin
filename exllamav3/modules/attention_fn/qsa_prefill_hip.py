"""Loader for qsa_prefill.hip (one wave per (row, kv head)). Same call shape as qsa_prefill.qsa_prefill_attend (fp16, shared or per-row block table)."""
import ctypes, hashlib, os, re, shutil, subprocess
import torch
from exllamav3.util import hip_compiler
from exllamav3.util.hip_lib import load_hip_runtime
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qsa_prefill.hip")
_state = {}

def _hipcc():
    return hip_compiler.hipcc()

def proof_tag(src_path, defs):
    return hashlib.sha1((open(src_path).read() + "\0" + " ".join(sorted(defs.split()))).encode()).hexdigest()[:16]

def require_proof(src_path, defs):
    """no GPU build of a kernel variant without a passing CPU proof of exactly this source + defs (check_expanded.py --mark writes the marker)"""
    d = os.environ.get("QSA_PROOF_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "qsa_proof"))
    if not os.path.isfile(os.path.join(d, proof_tag(src_path, defs) + ".ok")):
        raise RuntimeError(f"qsa_prefill_hip: no CPU bounds proof for {src_path} defs={defs!r} (tag {proof_tag(src_path, defs)}); run check_expanded.py --mark first")

def compile_hsaco(defs = "", arch = "gfx1151", src_path = None, extra = ()):
    src_path = src_path or _SRC
    require_proof(src_path, defs)
    src = open(src_path, "rb").read()
    flags = ["--genco", f"--offload-arch={arch}", "-O3", "-ffp-contract=off", "--no-gpu-bundle-output"] + list(extra) + [x if x.startswith("-") else f"-D{x}" for x in defs.split()]
    gcc = os.environ.get("EXL3_GCC_INSTALL_DIR", "/usr/lib/gcc/x86_64-linux-gnu/13")
    if os.path.isdir(gcc): flags.append(f"--gcc-install-dir={gcc}")
    tag = hashlib.sha1(src + " ".join(flags).encode() + b"\0" + hip_compiler.key().encode()).hexdigest()[:16]
    cache = os.path.join(os.path.expanduser("~/.cache/exllamav3"), f"qsa_pf_{arch}_{tag}.hsaco")
    if not os.path.isfile(cache):
        os.makedirs(os.path.dirname(cache), exist_ok = True)
        tmp = cache + f".{os.getpid()}.tmp"
        r = subprocess.run([_hipcc(), *flags, "-o", tmp, src_path], capture_output = True, text = True)
        if r.returncode: raise RuntimeError(r.stderr[-3000:])
        os.replace(tmp, cache)
    return cache

def hsaco_stats(path):
    """vgpr / spill / lds from the code object metadata (no GPU)."""
    t = ""
    for tool in ("llvm-readelf", "llvm-readobj"):
        p = shutil.which(tool) or os.path.join(os.path.dirname(_hipcc()), "..", "lib", "llvm", "bin", tool)
        if os.path.isfile(p):
            t = subprocess.run([p, "--notes", path], capture_output = True, text = True).stdout; break
    g = lambda k: (int(re.search(k + r":\s+(\d+)", t).group(1)) if re.search(k + r":\s+(\d+)", t) else -1)
    return dict(vgpr = g(r"\.vgpr_count"), spill = g(r"\.vgpr_spill_count"), lds = g(r"\.group_segment_fixed_size"))

def _load(defs = ""):
    if defs in _state: return _state[defs]
    cache = compile_hsaco(defs, torch.cuda.get_device_properties(0).gcnArchName.split(":")[0])
    lib = load_hip_runtime()
    torch.cuda.init()
    mod = ctypes.c_void_p()
    if lib.hipModuleLoad(ctypes.byref(mod), cache.encode()) != 0: raise RuntimeError("qsa_prefill_hip: hipModuleLoad failed")
    fn = ctypes.c_void_p()
    if lib.hipModuleGetFunction(ctypes.byref(fn), mod, b"qsa_pf") != 0: raise RuntimeError("qsa_prefill_hip: no function")
    lib.hipModuleLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p] * 3
    _state["lib"] = lib; _state[defs] = fn; _state["mod", defs] = mod
    return fn

def qsa_prefill_hip(q, k, v, indices, sm_scale, block_table, page_size, out = None, defs = ""):
    R, H, hd = q.shape
    assert (H, hd, k.shape[1], page_size, indices.shape[1]) == (24, 256, 2, 256, 2080), "qsa_prefill_hip: Qwen shape only"
    assert q.dtype == k.dtype == v.dtype == torch.half and q.stride(2) == 1 and q.stride(1) == hd
    assert k.stride(1) == hd and k.stride(2) == 1 and k.stride() == v.stride() and indices.is_contiguous() and indices.dtype == torch.int32
    assert indices.shape[0] == R and R >= 1 and R <= 4096 * 4
    assert q.stride(0) % 8 == 0 and q.data_ptr() % 16 == 0 and k.data_ptr() % 16 == 0 and v.data_ptr() % 16 == 0 and k.stride(0) % 8 == 0
    if out is None: out = torch.empty((R, H, hd), dtype = torch.half, device = q.device)
    assert out.stride(2) == 1 and out.stride(1) == hd and out.stride(0) % 8 == 0 and out.data_ptr() % 16 == 0
    assert block_table.is_contiguous() and block_table.dtype == torch.int32
    bt_stride = 0 if block_table.dim() == 1 else block_table.shape[1]
    npages = block_table.shape[-1]
    ntok = k.shape[0]
    assert v.shape[0] == ntok
    fn = _load(defs)
    cargs = [ctypes.c_void_p(a.data_ptr()) for a in (q, k, v, block_table, indices, out)] + \
            [ctypes.c_int(x) for x in (q.stride(0), out.stride(0), bt_stride, k.stride(0), ntok, npages)] + [ctypes.c_float(sm_scale)]
    params = (ctypes.c_void_p * len(cargs))(*[ctypes.cast(ctypes.byref(a), ctypes.c_void_p) for a in cargs])
    r = _state["lib"].hipModuleLaunchKernel(fn, R * 2, 1, 1, 32, 1, 1, 0, torch.cuda.current_stream().cuda_stream, params, None)
    if r != 0: raise RuntimeError(f"qsa_prefill_hip: launch failed ({r})")
    return out
