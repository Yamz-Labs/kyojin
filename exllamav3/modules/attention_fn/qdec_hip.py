"""
HIP (gfx11.5, wave32 WMMA) split kernel for quantized-KV decode attention (qdec.hip), the ROCm counterpart of qdec_cuda.py. Compiled
at first use with hipcc to an hsaco cached on disk, launched through the HIP module API. qdec_split writes per-split partials,
qdec_combine reduces them, rotates the result back with h32 and writes the fp16 output, so one call replaces the Triton split +
combine pair of triton_paged._paged_attn_decode_split_kernel for QCK > 0 and QCV > 0 with full (non-windowed) attention and no softcap.
make_cfg returns None for call shapes outside what the kernel covers; the caller then keeps the Triton kernels.
Switch: EXL3_QDEC_HIP=0/1 (see ENABLED; default on for ROCm).
"""
import ctypes
import hashlib
import os
import subprocess

import torch

from ...util.knobs import knob, knob_int, knob_on, on_change

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qdec.hip")
_CACHE = os.path.expanduser(os.environ.get("EXL3_QDEC_CACHE", "~/.cache/exllamav3"))


_family_cache: dict[int, bool] = {}


def _gfx115_family(device=None) -> bool:
    """qdec.hip uses gfx11.5-only (RDNA3.5) wave32 WMMA; enable it only on that family.
    The check is per device, not per process: on a mixed gfx1201+gfx1151 box the model
    device may be index 1 while device 0 is the other part. The result is cached per
    device index because make_cfg runs per layer per chunk. An extension without the
    family probe keeps the historical behavior."""
    if not torch.version.hip:
        return False
    try:
        from ...ext import exllamav3_ext as ext
    except Exception:                                   # noqa: BLE001
        return False
    if not hasattr(ext, "exl3_gemv_wmma_family"):
        return True
    try:
        if device is None:
            idx = torch.cuda.current_device()
        elif isinstance(device, torch.device):
            idx = device.index if device.index is not None else torch.cuda.current_device()
        else:
            idx = int(device)
        cached = _family_cache.get(idx)
        if cached is None:
            cached = _family_cache[idx] = (ext.exl3_gemv_wmma_family(idx) == 2)
        return cached
    except Exception:                                   # noqa: BLE001
        return False


ENABLED = bool(torch.version.hip) and knob_on("EXL3_QDEC_HIP")


@on_change
def _refresh_knobs():
    global ENABLED
    ENABLED = bool(torch.version.hip) and knob_on("EXL3_QDEC_HIP")

_kernels = {}
_bufs = {}
_lib = None
_cus = {}


def _hip():
    global _lib
    if _lib is None:
        from exllamav3.util.hip_lib import load_hip_runtime
        _lib = load_hip_runtime()
        _lib.hipModuleLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p] * 3
    return _lib


def pick_dw(hd, nt, nw):
    """waves per stream splitting the head dim: the smallest count that keeps the O accumulators (NT * HD/16 * 8 / DW floats) in budget.
    Shapes past that budget (4 and 5 q tiles at hd 256) still take DW=4, with the accumulators spilling, when they fit the second budget:
    the alternative there is the Triton fallback, measured 6.6x / 4.8x slower per call at 131K (same rows, same cache bytes)."""
    if knob("EXL3_QDEC_HIP_DW"):
        dw = int(knob("EXL3_QDEC_HIP_DW"))
        return dw if dw in (1, 2, 4) and nw % dw == 0 and hd // 16 // dw >= 4 else None
    for dw in (1, 2, 4):
        if nw % dw == 0 and hd // 16 // dw >= 4 and nt * (hd // 16) * 8 // dw <= knob_int("EXL3_QDEC_HIP_MAXO", 96):
            return dw
    dw = 4
    if nw % dw == 0 and hd // 16 // dw >= 4 and nt * (hd // 16) * 8 // dw <= knob_int("EXL3_QDEC_HIP_MAXO2", 160):
        return dw
    return None


def geometry(hd, nqh, nkv, qlen, nw, dw, smm = 0, sxmul = 0):
    """(NT, RP, shared bytes): q tile + max(32x32 fp32 q-rotation matrix, S exchange, cross-stream merge buffer), mirrors qdec.hip"""
    rows = qlen * (nqh // nkv)
    nt = (rows + 15) // 16
    rp = nt * 16
    qs = (rp * (hd + 8) * 2 + 15) // 16 * 16
    nsw = nw // dw
    no = nt * (hd // 16 // dw) * 8
    merge = ((nsw - 1) * dw * no * 32 + 2 * (nsw - 1) * nt * 32) * 4
    sx = (sxmul or (1 if smm else 2)) * nw * nt * 8 * 32 * 4 if dw > 1 else 0
    return nt, rp, qs + max(4096, merge, sx)


def make_cfg(q_len, n_q_heads, n_kv_heads, head_dim, k_bits, v_bits, page_size, causal, sm_scale,
             window_left, window_right, softcap, kv_append_len, has_sinks, key_lo = 0, tq = False, bsz = 1, device = None):
    """Kernel config for a call, or None when the Triton kernels must run it. key_lo: KVWIN, positions below it are not attended
    bsz: sequences in the call; it only feeds the EXL3_QDEC_HIP_SMM default (auto: on for one sequence, see below)
    device: the tensor's device; the gfx11.5 gate is per device, not per process"""
    if (not ENABLED or not _gfx115_family(device) or window_left >= 0 or window_right >= 0 or softcap or kv_append_len != q_len
            or not (2 <= k_bits <= 8) or not (2 <= v_bits <= 8)):
        return None
    if head_dim not in (128, 256) or n_q_heads % n_kv_heads != 0 or page_size % 16 != 0:
        return None
    nw = knob_int("EXL3_QDEC_HIP_NW", 0) or 4
    nt = (q_len * (n_q_heads // n_kv_heads) + 15) // 16
    dw = pick_dw(head_dim, nt, nw)
    if dw is None:
        return None
    smmk = knob("EXL3_QDEC_HIP_SMM", "auto")
    smmv = (1 if bsz == 1 else 0) if smmk == "auto" else int(smmk)
    # measured (131K, 36/48 rows, 3 q tiles): single buffer -37 % with one sequence, +1 % with 2 or 4
    # sequences batched (fewer, longer splits hide the extra per-slab barrier); nt2 +1 %, nt1 flat
    smm = int(smmv != 0 and dw > 1 and nt >= 3)
    sxmul = 2 if smmv == 2 else 0                 # SMM=2: same kernel as SMM=1 but the old (double) allocation, a footprint probe
    _, _, shared = geometry(head_dim, n_q_heads, n_kv_heads, q_len, nw, dw, smm, sxmul)
    if shared > 64 * 1024:
        return None
    return dict(hd=head_dim, kb=k_bits, vb=v_bits, nkv=n_kv_heads, nqh=n_q_heads, qlen=q_len, page=page_size, nw=nw, dw=dw,
                causal=bool(causal), scale=float(sm_scale), sinks=bool(has_sinks), key_lo=int(key_lo), tq=bool(tq),
                vpf=bool(knob_int("EXL3_QDEC_HIP_VPF", 1)), smm=smm, sxmul=sxmul)


def _compile(cfg, arch):
    from exllamav3.util import hip_compiler
    defs = dict(HD=cfg["hd"], KB=cfg["kb"], VB=cfg["vb"], NKV=cfg["nkv"], NQH=cfg["nqh"], QLEN=cfg["qlen"], PAGE=cfg["page"],
                NW=cfg["nw"], DW=cfg["dw"], CAUSAL=int(cfg["causal"]), HAS_SINKS=int(cfg["sinks"]), KEY_LO=cfg["key_lo"], TQ=int(cfg.get("tq", False)),
                VPF=int(cfg.get("vpf", False)), SMM1=int(cfg.get("smm", 0)), SCALE=repr(float(cfg["scale"])) + "f")
    extra = [f"-D{k}={v}" for k, v in defs.items()] + knob("EXL3_QDEC_HIP_FLAGS", "").split()
    flags = ["--genco", f"--offload-arch={arch}", "-O3", "-ffp-contract=off", "--no-gpu-bundle-output"] + extra
    flags += hip_compiler.gcc_flags()
    src = open(_SRC, "rb").read()
    tag = hashlib.sha1(src + " ".join(flags).encode() + b"\0" + hip_compiler.key().encode()).hexdigest()[:16]
    path = os.path.join(_CACHE, f"qdec_hip_{arch}_{tag}.hsaco")
    if not os.path.isfile(path):
        os.makedirs(_CACHE, exist_ok=True)
        tmp = path + f".{os.getpid()}.tmp"
        r = subprocess.run([hip_compiler.hipcc(), *flags, "-o", tmp, _SRC], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError("qdec_hip: hipcc failed:\n" + r.stderr[-3000:])
        os.replace(tmp, path)
    return path


class _Kernel:
    def __init__(self, cfg, device):
        lib = _hip()
        torch.cuda.init()
        arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
        self.cfg = cfg
        self.path = _compile(cfg, arch)
        _, self.rp, self.shared = geometry(cfg["hd"], cfg["nqh"], cfg["nkv"], cfg["qlen"], cfg["nw"], cfg["dw"],
                                           cfg.get("smm", 0), cfg.get("sxmul", 0))
        self.mod = ctypes.c_void_p()
        if lib.hipModuleLoad(ctypes.byref(self.mod), self.path.encode()) != 0:
            raise RuntimeError("qdec_hip: hipModuleLoad failed")
        self.split = ctypes.c_void_p()
        self.combine = ctypes.c_void_p()
        if lib.hipModuleGetFunction(ctypes.byref(self.split), self.mod, b"qdec_split") != 0 or \
           lib.hipModuleGetFunction(ctypes.byref(self.combine), self.mod, b"qdec_combine") != 0:
            raise RuntimeError("qdec_hip: kernel symbols missing")

    def _launch(self, fn, grid, block, shared, args):
        cargs = args                                     # ctypes values: c_void_p for pointers, c_int for ints
        params = (ctypes.c_void_p * len(cargs))(*[ctypes.cast(ctypes.byref(a), ctypes.c_void_p) for a in cargs])
        r = _hip().hipModuleLaunchKernel(fn, grid[0], grid[1], 1, block, 1, 1, shared, torch.cuda.current_stream().cuda_stream, params, None)
        if r != 0:
            raise RuntimeError(f"qdec_hip: launch failed ({r})")


def get_kernel(cfg, device):
    key = (device.index, tuple(sorted(cfg.items())), knob("EXL3_QDEC_HIP_FLAGS", ""))
    k = _kernels.get(key)
    if k is None:
        k = _kernels[key] = _Kernel(cfg, device)
    return k


def _buf(device, name, n):
    key = (device.index, name)
    b = _bufs.get(key)
    if b is None or b.numel() < n:
        b = _bufs[key] = torch.empty(n, dtype=torch.float32, device=device)
    return b


def cu_count(device):
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _cus:
        _cus[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return _cus[idx]


def lds_per_mp(device):
    """LDS bytes per multiprocessor as torch counts multiprocessors: a gfx11.5 WGP is 2 CUs of 64 KiB, so 128 KiB per MP.
    Matches the measured residency steps (3 CTAs of 36 KiB at 3/MP = the one-wave 60-CTA config in the qdsmm cells)."""
    idx = device.index if device.index is not None else torch.cuda.current_device()
    p = torch.cuda.get_device_properties(idx)
    return 2 * int(getattr(p, "shared_memory_per_multiprocessor", 65536))


def resident_ctas(kern, device):
    """CTAs of this kernel resident per multiprocessor (dynamic LDS only: the per-CTA static part is ~5 KiB and the
    2 CTAs of a 49 KiB kernel, i.e. the 4/5 q tile tier, are what the split probe measured)."""
    return max(1, lds_per_mp(device) // max(kern.shared, 1))


def num_splits(cfg, device, programs, max_k_len):
    if knob("EXL3_QDEC_HIP_SPL"):
        return int(knob("EXL3_QDEC_HIP_SPL"))
    waves = knob_int("EXL3_QDEC_HIP_WAVES", 0) or cu_count(device) * (16 if cfg.get("tq") and cfg["qlen"] * (cfg["nqh"] // cfg["nkv"]) <= 16 else 12)   # tq, one q tile: 4 waves per SIMD (its V loads stay in flight)
    return max(1, min(-(-waves // (programs * cfg["nw"])), -(-max_k_len // 16 // cfg["nw"]), 128))


def decode(cfg, q, k_cache, v_cache, block_table, cache_seqlens, k_scales, v_scales, h32, sinks, out, max_k_len, ring = None, tq = None):
    """q [bsz, q_len, n_q_heads, hd] fp16 -> out (same shape, fp16); caches are the packed int32 words.
    ring (KVWIN, cfg built with key_lo = S): the split kernel attends [S, q - W] (cache_seqlens shifted by W) and one more
    partial, written by the fp16 ring kernel after the last split, carries the sinks and the window."""
    bsz = q.shape[0]
    kern = get_kernel(cfg, q.device)
    programs = bsz * cfg["nkv"]
    splits = num_splits(cfg, q.device, programs, max_k_len)
    # never launch more CTAs than one full residency wave: the surplus CTAs queue behind the resident ones and each
    # CTA then walks fewer, shorter splits (measured at 4/5 q tiles: 60 CTAs = 1.5 waves -> 40 CTAs -21 % / -13 %;
    # 1-3 q tiles resolve to 25/15/15, i.e. today's count, so those shapes are untouched). tq keeps its own count:
    # there the extra splits keep V loads in flight rather than trading waves (see num_splits), and an explicit
    # EXL3_QDEC_HIP_SPL still wins over this cap.
    if not cfg.get("tq") and not knob("EXL3_QDEC_HIP_SPL"):
        splits = max(1, min(splits, resident_ctas(kern, q.device) * cu_count(q.device) // programs))
    split_len = -(-(-(-max_k_len // splits)) // 16) * 16
    rows = cfg["qlen"] * (cfg["nqh"] // cfg["nkv"])
    ns_total = splits + (1 if ring is not None else 0)
    po = _buf(q.device, "po", programs * ns_total * kern.rp * cfg["hd"])
    pml = _buf(q.device, "pml", programs * ns_total * kern.rp * 2)
    pool_seqlens = cache_seqlens if ring is None else cache_seqlens - ring.cfg.W
    P, I = ctypes.c_void_p, ctypes.c_int
    kern._launch(kern.split, (programs, splits), 32 * cfg["nw"], kern.shared,
                 [P(q.data_ptr()), P(k_cache.data_ptr()), P(v_cache.data_ptr()), P(block_table.data_ptr()), P(pool_seqlens.data_ptr()),
                  P(po.data_ptr()), P(pml.data_ptr()), P(k_scales.data_ptr()), P(v_scales.data_ptr()), P(h32.data_ptr()),
                  I(split_len), I(block_table.shape[1]), I(ns_total),
                  P(tq.cbk16.data_ptr() if tq is not None else None), P(tq.cbv16.data_ptr() if tq is not None else None)])
    if ring is not None:
        import triton
        from .kvwin_triton import ring_split
        ring_split(ring, q, cache_seqlens, h32, po, pml, ns_total, programs, cfg["qlen"], cfg["nqh"], cfg["nkv"], cfg["hd"],
                   cfg["hd"], cfg["scale"], tq is None, 2, 1, 1, max(16, triton.next_power_of_2(kern.rp)), rpp = kern.rp)
    kern._launch(kern.combine, (programs, rows), cfg["hd"], 0,
                 [P(po.data_ptr()), P(pml.data_ptr()), P(out.data_ptr()), P(h32.data_ptr()),
                  P(sinks.data_ptr() if cfg["sinks"] else None), I(ns_total)])
    return out
