"""Launchers of dsv41_idx.hip: the DS4.1 indexer scoring kernels (dense pool scan, candidate-slot scan).
Same loader model as dsv41_hip.py (hipcc genco on first use, launched on torch's current stream)."""
import ctypes, os
import torch
from . import dsv41_hip as H
from exllamav3.util.hip_lib import load_hip_runtime

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dsv41_idx.hip")
_FUNCS = ("ds41_idx_score_dense", "ds41_idx_score_cand")
_state = {}


def available(device) -> bool:
    if os.environ.get("DSV41_IDX_KERNELS", "1") == "0":
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
        raise RuntimeError("dsv41_idx: hipModuleLoad failed")
    fns = {}
    for n in _FUNCS:
        fn = ctypes.c_void_p()
        if lib.hipModuleGetFunction(ctypes.byref(fn), mod, n.encode()) != 0:
            raise RuntimeError("dsv41_idx: no function " + n)
        fns[n] = fn
    lib.hipModuleLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p] * 3
    _state.update(lib=lib, fns=fns, mod=mod)
    return fns


def _launch(name, grid, block, args):
    fn = _load()[name]
    cargs, params = H._pack(args)
    r = _state["lib"].hipModuleLaunchKernel(fn, grid[0], grid[1], 1, block, 1, 1, 0,
                                            torch.cuda.current_stream().cuda_stream, params, None)
    if r != 0:
        raise RuntimeError(f"dsv41_idx: launch of {name} failed ({r})")


def score_dense(q, w, kpool, nt, start, ratio, blocks_hint=160, packed=None):
    """q [R, 32, 128] fp16, w [R, 32] fp32, kpool [rows, 128] fp16 (rows >= nt) -> score [R, ntp] fp32, ntp = nt rounded up to 8.
    Keys at or beyond the causal reach of a row, and the pad, are -inf."""
    R = q.size(0)
    assert q.is_contiguous() and w.is_contiguous() and kpool.stride(0) == 128 and q.dtype == torch.half
    ntp = -(-nt // 8) * 8
    out = torch.empty(R, ntp, dtype=torch.float32, device=q.device)
    gx = -(-ntp // 128)
    gy = max(1, min(R, -(-blocks_hint // gx)))
    rpb = -(-R // gy)
    gy = -(-R // rpb)
    _launch("ds41_idx_score_dense", (gx, gy), 128, (q, w, kpool, *(packed or (None, None)), out, R, nt, ntp, start, ratio, rpb))
    return out


def score_cand(q, w, kpool, blk, start, ratio, packed=None):
    """blk [R, nsel] int32 block ids (-1 = none) -> score [R, nsel * 8] fp32 of the keys in those blocks."""
    R, nsel = blk.shape
    assert q.is_contiguous() and w.is_contiguous() and blk.is_contiguous() and blk.dtype == torch.int32
    out = torch.empty(R, nsel * 8, dtype=torch.float32, device=q.device)
    tpw = 2 if R <= 8 else 8
    _launch("ds41_idx_score_cand", (-(-(nsel * 8) // (64 * tpw)), R), 128, (q, w, kpool, *(packed or (None, None)), blk, out, R, nsel, start, ratio, tpw))
    return out


_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def pack_keys(k):
    """fp4-valued keys [n, 128] fp16 -> (q [n, 64] uint8, s [n, 4] uint8 e8m0). Exact for tensors produced by
    quant.fp4_act_quant_(k, 32): every value is code * 2^e with code in the e2m1 set."""
    n = k.size(0)
    x = k.float().view(n, 4, 32)
    amax = x.abs().amax(-1, keepdim=True)
    e = torch.ceil(torch.log2((amax / 6.0).clamp_min(2.0 ** -100))).clamp(-126, 127)
    e = torch.where(amax == 0, torch.zeros_like(e), e)
    y = x / torch.exp2(e)
    tab = torch.tensor(_E2M1, device=k.device)
    code = (y.abs().unsqueeze(-1) - tab).abs().argmin(-1)
    nib = (code | ((y < 0).long() << 3)).to(torch.uint8).view(n, 64, 2)
    return (nib[..., 0] | (nib[..., 1] << 4)).contiguous(), (e.squeeze(-1) + 127).to(torch.uint8).contiguous()
