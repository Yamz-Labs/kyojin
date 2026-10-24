"""Launchers of dsv41_fuse.hip: DS4.1 decode glue fusion (swiglu, MoE tail, decode router scores, router top-k with the K-group
slots). Same loader model as dsv41_idx.py (hipcc genco on first use, launched on torch's current stream). Every kernel is
bit-identical to the torch chain it replaces (tools/ds41/fuse_check.py)."""
import ctypes, os
import torch
from . import dsv41_hip as H
from exllamav3.util.hip_lib import load_hip_runtime

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dsv41_fuse.hip")
_FUNCS = ("ds41_swiglu_h", "ds41_moe_tail", "ds41_router_scores_d", "ds41_router_topk2")
_state = {}
SWIGLU_MODE = 0


def available(device) -> bool:
    if os.environ.get("DSV41_FUSE", "1") == "0":
        return False
    return torch.device(device).type == "cuda" and torch.cuda.is_available() and torch.version.hip is not None


def _load():
    if "fns" in _state:
        return _state["fns"]
    cache = H.compile_hsaco(torch.cuda.get_device_properties(0).gcnArchName.split(":")[0], src_path=_SRC)
    lib = load_hip_runtime()
    torch.cuda.init()
    mod = ctypes.c_void_p()
    if lib.hipModuleLoad(ctypes.byref(mod), cache.encode()) != 0:
        raise RuntimeError("dsv41_fuse: hipModuleLoad failed")
    fns = {}
    for n in _FUNCS:
        fn = ctypes.c_void_p()
        if lib.hipModuleGetFunction(ctypes.byref(fn), mod, n.encode()) != 0:
            raise RuntimeError("dsv41_fuse: no function " + n)
        fns[n] = fn
    lib.hipModuleLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p] * 3
    _state.update(lib=lib, fns=fns, mod=mod)
    return fns


def _launch(name, grid, block, args):
    fn = _load()[name]
    cargs, params = H._pack(args)
    r = _state["lib"].hipModuleLaunchKernel(fn, grid[0], 1, 1, block, 1, 1, 0,
                                            torch.cuda.current_stream().cuda_stream, params, None)
    if r != 0:
        raise RuntimeError(f"dsv41_fuse: launch of {name} failed ({r})")


def swiglu_h(gate, up, limit, mode=None):
    """half(silu(min(gate, L)) * clamp(up, -L, L)) in fp32 math (ops.swiglu(...).to(half)); gate, up fp16 contiguous."""
    assert gate.dtype == torch.half and up.dtype == torch.half and gate.shape == up.shape and gate.is_contiguous() and up.is_contiguous()
    out = torch.empty_like(gate)
    n = gate.numel()
    _launch("ds41_swiglu_h", ((n + 255) // 256,), 256, (gate, up, out, n, float(limit), SWIGLU_MODE if mode is None else mode))
    return out


def moe_tail(outs, shared, out=None):
    """half(((o0 + o1) + o2) + float(shared)); outs: 1..3 contiguous fp32 tensors (the K-group outputs), shared fp16, same numel."""
    assert 1 <= len(outs) <= 3 and shared.dtype == torch.half and shared.is_contiguous()
    n = shared.numel()
    o = list(outs) + [None] * (3 - len(outs))
    for t in outs:
        assert t.dtype == torch.float and t.is_contiguous() and t.numel() == n
    if out is None:
        out = torch.empty_like(shared)
    _launch("ds41_moe_tail", ((n + 255) // 256,), 256, (o[0], o[1], o[2], shared, out, n))
    return out


def router_scores_d(x, W, scores, temp=1.0):
    """Decode router scores (x [1,5120] fp16, W [E,5120] fp16/bf16, scores [1,E] fp32); bitwise = dsv41_hip router scores."""
    E = W.size(0)
    assert x.dtype == torch.half and x.numel() == 5120 and W.dtype in (torch.half, torch.bfloat16) and E % 8 == 0
    assert x.data_ptr() % 16 == 0 and W.data_ptr() % 16 == 0 and W.is_contiguous()
    _launch("ds41_router_scores_d", (E // 8,), 256, (x, W, 1 if W.dtype == torch.half else 2, scores, E, 1.0 / float(temp)))


def router_topk2(scores, bias, sel, wts, wts_f32, topk, norm, scale, lut, gsel):
    """ds41_router_topk plus gsel [G, R, topk] int64 = lut[g][pick] (lut [G, E] int64)."""
    R, E = scores.shape
    G = lut.size(0)
    assert sel.dtype == torch.int64 and lut.dtype == torch.int64 and gsel.dtype == torch.int64 and lut.is_contiguous()
    _launch("ds41_router_topk2", (R,), 32, (scores, bias, sel, wts, wts_f32, E, topk, int(norm), float(scale), lut, G, gsel, R))
