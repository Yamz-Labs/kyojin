"""HIP / gfx1151 WMMA grouped MoE expert GEMM on the stored bytes of GGUF k-quants (Q4_K, Q3_K). Generic by quant type and shape.

gate_up(x, Wg, Wu, sp, be, P, topk, E, ebytes, lim, K, N, rowb, fmt, maxb) -> H fp16 [maxb*128, N]   (SwiGLU applied)
down(H, Wd, wts, sp, be, P, E, ebytes, K, N, rowb, maxb)                  -> out fp32 [P, N]         (rows scaled by wts[pair])
Block tables as in prefill_kernels.route_blocks with BM = 128. The code object is built by build_moe.sh (separate process, SDK toolchain);
it is loaded here through the HIP runtime torch already has in the process.
"""
import ctypes, os, re
import torch

MT = 128
_lib = None
_mod = {}


def _hip():
    global _lib
    if _lib is None:
        path = None
        for line in open("/proc/self/maps"):
            m = re.search(r"(/\S*libamdhip64\.so[\d.]*)", line)
            if m:
                path = m.group(1)
                break
        _lib = ctypes.CDLL(path)
    return _lib


def _fn(co, name):
    key = (co, name)
    if key not in _mod:
        lib = _hip()
        data = open(co, "rb").read()
        mod = ctypes.c_void_p()
        assert lib.hipModuleLoadData(ctypes.byref(mod), ctypes.c_char_p(data)) == 0, "hipModuleLoadData"
        fn = ctypes.c_void_p()
        assert lib.hipModuleGetFunction(ctypes.byref(fn), mod, name.encode()) == 0, "hipModuleGetFunction " + name
        _mod[key] = (mod, fn)
    return _mod[key][1]


CO = os.environ.get("KQ_MOE_CO", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exllamav3_ext", "hip_kq", "moe_hip.co"))


def _launch(name, grid, args):
    lib = _hip()
    fn = _fn(CO, name)
    cargs = []
    for a in args:
        if isinstance(a, torch.Tensor):
            cargs.append(ctypes.c_void_p(a.data_ptr()))
        elif isinstance(a, float):
            cargs.append(ctypes.c_float(a))
        elif isinstance(a, tuple):            # (value, ctypes type)
            cargs.append(a[1](a[0]))
        else:
            cargs.append(ctypes.c_int(a))
    ptrs = (ctypes.c_void_p * len(cargs))(*[ctypes.cast(ctypes.pointer(c), ctypes.c_void_p) for c in cargs])
    stream = torch.cuda.current_stream().cuda_stream
    rc = lib.hipModuleLaunchKernel(fn, grid[0], grid[1], 1, 256, 1, 1, 0, ctypes.c_void_p(stream), ptrs, None)
    assert rc == 0, f"hipModuleLaunchKernel {name} rc={rc}"


def gate_up(x, Wg, Wu, sp, be, P, topk, E, ebytes, lim, K, N, rowb, fmt, maxb):
    """fmt: 0 = Q4_K, 1 = Q3_K (FMT ids of prefill_kernels)"""
    assert K % 64 == 0 and N % 128 == 0
    H = torch.empty(maxb * MT, N, dtype=torch.float16, device=x.device)
    _launch("gu_q4" if fmt == 0 else "gu_q3", (N // 128, maxb),
            [x, Wg, Wu, H, sp, be, P, topk, E, (ebytes, ctypes.c_long), float(lim), K, N, rowb])
    return H


def down(H, Wd, wts, sp, be, P, E, ebytes, K, N, rowb, maxb, fmt=0):
    assert fmt == 0 and K % 64 == 0 and N % 256 == 0
    out = torch.empty(P, N, dtype=torch.float32, device=H.device)
    _launch("down_q4", (N // 256, maxb), [H, Wd, out, wts, sp, be, P, E, (ebytes, ctypes.c_long), K, N, rowb])
    return out


# ---- the shared generic GGUF kernel (gguf-moe gm_ext / gm_pfmoe.cu: same kernels, per-expert pointer tables, Q3_K gate/up via fmt 3) ----
_gm = None
_ptrs = {}


def shared_module():
    global _gm
    if _gm is None:
        import importlib.util
        so = os.environ.get("HY4_GM_EXT", os.path.expanduser("~/.cache/exl3_gm_ext/exl3_gm_ext.so"))
        spec = importlib.util.spec_from_file_location("exl3_gm_ext", so)
        _gm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_gm)
    return _gm


def _table(buf, ebytes, E):
    key = (buf.data_ptr(), ebytes, E)
    if key not in _ptrs:
        _ptrs[key] = (buf.data_ptr() + torch.arange(E, dtype=torch.int64, device=buf.device) * ebytes)
    return _ptrs[key]


def gate_up_shared(x, Wg, Wu, sp, be, P, topk, E, ebytes, lim, K, N, rowb, fmt, maxb):
    m = shared_module()
    H = torch.empty(maxb * MT, N, dtype=torch.float16, device=x.device)
    m.gm_pf_gate_up(x, _table(Wg, ebytes, E), _table(Wu, ebytes, E), H, sp, be, P, topk, E, float(lim), K, N, rowb, maxb, 0 if fmt == 0 else 3)
    return H


def down_shared(H, Wd, wts, sp, be, P, E, ebytes, K, N, rowb, maxb):
    m = shared_module()
    out = torch.empty(P, N, dtype=torch.float32, device=H.device)
    m.gm_pf_down(H, _table(Wd, ebytes, E), wts, sp, be, out, P, E, K, N, rowb, 0, maxb)
    return out
