"""
CUDA split kernel for quantized-KV decode attention (cu/qdec.cu), compiled at first use with NVRTC to a cubin cached on
disk, launched through the driver API. Replaces triton_paged_cu._paged_attn_decode_split_kernel when both K and V are
packed (QCK > 0, QCV > 0) on sm_80+ with full (non-windowed) attention and no softcap; the Triton combine kernel
reduces its partials, so the downstream contract is unchanged. Returns None / raises QdecUnsupported when a call
shape is outside what the kernel covers, so callers fall back to the Triton kernels.
"""
import hashlib
import math
import os

import torch
import triton

_SRC = os.path.join(os.path.dirname(__file__), "cu", "qdec.cu")
_SRC_ROWS = os.path.join(os.path.dirname(__file__), "cu", "qdec_rows.cu")      # many-row kernel (verify steps), see qdec_rows.cu
_COMMON = os.path.join(os.path.dirname(__file__), "cu", "qdec_common.cuh")
ROWS_MODE = os.environ.get("EXL3_QDEC_ROWS", "1")          # 0: no many-row kernel, 1: only where qdec.cu cannot, 2: always
_CACHE = os.path.expanduser(os.environ.get("EXL3_QDEC_CACHE", "~/.cache/exl3_qdec"))
_kernels = {}


class QdecUnsupported(RuntimeError):
    pass


def geometry(cfg):
    """Shared-memory sizing, mirrors the macros of qdec.cu / qdec_rows.cu"""
    hd, kb, vb = cfg["hd"], cfg["kb"], cfg["vb"]
    if cfg.get("rows_kernel"):
        g = hd // 32
        nw = cfg["nw"]
        gw = g // nw
        rows = cfg["qlen"] * (cfg["nqh"] // cfg["nkv"])
        nt = (rows + 7) // 8
        kww, vww = gw * kb, gw * vb
        ksw = kww + (0 if kww % 8 == 4 else 4)
        vsw = vww + (0 if vww % 8 == 4 else 4)
        sc_b = (16 * g * 2 + 15) // 16 * 16
        stage_b = 16 * ksw * 4 + 16 * vsw * 4 + 2 * sc_b
        sx = nw * nt * 32 * 16 + nt * 32 * 16
        return dict(nt=nt, rows=rows, stage_b=stage_b, shared=sx + max(nw * cfg["st"] * stage_b, 4096))
    g = hd // 32
    group = cfg["nqh"] // cfg["nkv"]
    rows = cfg["qlen"] * group
    nt = (rows + 7) // 8
    kchw, vchw = g * kb, g * vb
    kstr = kchw + (0 if kchw % 8 == 4 else 4)
    vstr = vchw + (0 if vchw % 8 == 4 else 4)
    sc_b = (16 * g * 2 + 15) // 16 * 16
    stage_b = 16 * kstr * 4 + 16 * vstr * 4 + 2 * sc_b
    ring = cfg["nw"] * cfg["st"] * stage_b
    qs = (nt * 8 * (hd + 8) * 2 + 15) // 16 * 16
    merge = (cfg["nw"] * nt * 8 * hd + 2 * cfg["nw"] * nt * 8) * 4
    return dict(nt=nt, rows=rows, stage_b=stage_b, shared=qs + max(ring, merge, 4096))


def supported(cfg):
    hd = cfg["hd"]
    if hd % 64 != 0 or hd > 256 or not (2 <= cfg["kb"] <= 8) or not (2 <= cfg["vb"] <= 8):
        return False
    if cfg["nqh"] % cfg["nkv"] != 0 or cfg["page"] % 16 != 0:
        return False
    geo = geometry(cfg)
    if cfg.get("rows_kernel"):
        # one warp per 32-dim group; O accumulator regs per thread: 2 * nt * 4
        return (hd // 32) % cfg["nw"] == 0 and geo["nt"] <= 8
    if geo["rows"] > 32 or geo["nt"] * hd > 512:       # O accumulator regs: hd/16 * nt * 4 per thread
        return False
    return True


def _compile(cfg):
    from cuda.bindings import nvrtc
    defs = dict(HD=cfg["hd"], KB=cfg["kb"], VB=cfg["vb"], NKV=cfg["nkv"], NQH=cfg["nqh"], QLEN=cfg["qlen"],
                PAGE=cfg["page"], NW=cfg["nw"], ST=cfg["st"], CAUSAL=int(cfg["causal"]), BLOCK_ROWS=cfg["block_rows"],
                HD_PAD=triton.next_power_of_2(cfg["hd"]), SCALE=repr(float(cfg["scale"])) + "f")
    if os.environ.get("EXL3_QDEC_ABL"):
        defs["ABL"] = int(os.environ["EXL3_QDEC_ABL"])
    src = "#define INFINITY __int_as_float(0x7f800000)\n" + open(_COMMON).read() + "\n" + \
        open(_SRC_ROWS if cfg.get("rows_kernel") else _SRC).read()
    major, minor = torch.cuda.get_device_capability()
    opts = [f"--gpu-architecture=sm_{major}{minor}", "-std=c++17", "-default-device", "--fmad=true"]
    opts += [f"-D{k}={v}" for k, v in defs.items()]
    key = hashlib.sha1((src + " ".join(opts)).encode()).hexdigest()[:20]
    path = os.path.join(_CACHE, key + ".cubin")
    if os.path.exists(path):
        return open(path, "rb").read()

    def chk(r):
        if r[0] != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError(f"nvrtc: {r[0]}")
        return r[1:] if len(r) > 1 else None
    prog, = chk(nvrtc.nvrtcCreateProgram(src.encode(), b"qdec.cu", 0, [], []))
    rc = nvrtc.nvrtcCompileProgram(prog, len(opts), [o.encode() for o in opts])
    if rc[0] != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        n, = chk(nvrtc.nvrtcGetProgramLogSize(prog))
        buf = b" " * n
        nvrtc.nvrtcGetProgramLog(prog, buf)
        raise RuntimeError("qdec nvrtc compile failed:\n" + buf.decode(errors="replace"))
    n, = chk(nvrtc.nvrtcGetCUBINSize(prog))
    cubin = b" " * n
    chk(nvrtc.nvrtcGetCUBIN(prog, cubin))
    os.makedirs(_CACHE, exist_ok=True)
    tmp = path + f".{os.getpid()}"
    open(tmp, "wb").write(cubin)
    os.replace(tmp, path)
    return cubin


class _Kernel:
    def __init__(self, cfg):
        from cuda.bindings import driver
        self.cfg = cfg
        self.cubin = _compile(cfg)
        self.shared = geometry(cfg)["shared"]
        torch.cuda.current_stream()   # context
        r, self.mod = driver.cuModuleLoadData(self.cubin)
        assert r == driver.CUresult.CUDA_SUCCESS, r
        r, self.fn = driver.cuModuleGetFunction(self.mod, b"qdec_split_rows" if cfg.get("rows_kernel") else b"qdec_split")
        assert r == driver.CUresult.CUDA_SUCCESS, r
        if self.shared > 48 * 1024:
            r, = driver.cuFuncSetAttribute(self.fn, driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, self.shared)
            assert r == driver.CUresult.CUDA_SUCCESS, r

    def blocks_per_sm(self):
        """resident CTAs per SM for this kernel's block size / shared memory / registers"""
        if not hasattr(self, "_bps"):
            from cuda.bindings import driver
            r, n = driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(self.fn, 32 * self.cfg["nw"], self.shared)
            assert r == driver.CUresult.CUDA_SUCCESS, r
            self._bps = max(1, n)
        return self._bps

    def launch(self, grid, ptrs, ints):
        """ptrs: 11 device pointer ints (q .. h32 incl. out), ints: split_len, num_pages_per_seq, num_splits"""
        import ctypes
        from cuda.bindings import driver
        vals = list(ptrs) + list(ints) + [0, 0, 0]       # + sinks, 2 scratch slots (unused)
        types = [ctypes.c_void_p] * len(ptrs) + [ctypes.c_int] * len(ints) + [ctypes.c_void_p] * 3
        stream = torch.cuda.current_stream().cuda_stream
        r, = driver.cuLaunchKernel(self.fn, grid[0], grid[1], 1, 32 * self.cfg["nw"], 1, 1, self.shared, stream, (tuple(vals), tuple(types)), 0)
        assert r == driver.CUresult.CUDA_SUCCESS, r


def get_kernel(cfg):
    key = tuple(sorted(cfg.items()))
    k = _kernels.get(key)
    if k is None:
        k = _kernels[key] = _Kernel(cfg)
    return k


def rows_nw(hd):
    """warps per CTA of the many-row kernel (each takes hd/32/nw 32-dim groups)"""
    return int(os.environ.get("EXL3_QDEC_RNW", "0")) or (4 if (hd // 32) % 4 == 0 else 2)


def pick_nw_st(cfg):
    """(warps per CTA, ring depth), measured on a 3090 at 128K context (K/V bits 4/4, 6/6, 8/8): 8 x 2 for hd128 with
    narrow codes, 4 x 2 otherwise; shrinks until the shared memory fits the opt-in limit"""
    c = dict(cfg)
    if cfg.get("rows_kernel"):
        c.update(nw=rows_nw(cfg["hd"]))
        for st in ((int(os.environ["EXL3_QDEC_RST"]),) if os.environ.get("EXL3_QDEC_RST") else (3, 2, 1)):
            c.update(st=st)
            if geometry(c)["shared"] <= 99 * 1024:
                return c["nw"], st
        raise QdecUnsupported("no ring configuration fits")
    narrow = cfg["hd"] == 128 and cfg["kb"] + cfg["vb"] <= 8
    for nw, st in (((8, 2), (4, 2), (4, 1), (2, 2), (2, 1)) if narrow else ((4, 2), (4, 1), (2, 2), (2, 1))):
        c.update(nw=nw, st=st)
        if geometry(c)["shared"] <= 99 * 1024:
            return nw, st
    raise QdecUnsupported("no ring configuration fits")


ENABLED = os.environ.get("EXL3_QDEC", "1") != "0" and not torch.version.hip
# graph (BC) path switch (EXL3_QDEC_GRAPH=0 keeps the Triton split kernel in captured graphs)
GRAPH_ENABLED = os.environ.get("EXL3_QDEC_GRAPH", "1") != "0"
_sm = {}


def make_cfg(q_len, n_q_heads, n_kv_heads, head_dim, k_bits, v_bits, page_size, causal, sm_scale, block_rows,
             window_left, window_right, softcap, kv_append_len):
    """Kernel config for a call, or None when the Triton kernels must run it"""
    if (not ENABLED or window_left >= 0 or window_right >= 0 or softcap or kv_append_len != q_len
            or k_bits == 0 or v_bits == 0):
        return None
    cfg = dict(hd=head_dim, kb=k_bits, vb=v_bits, nkv=n_kv_heads, nqh=n_q_heads, qlen=q_len, page=page_size,
               nw=4, st=2, causal=bool(causal), scale=float(sm_scale), block_rows=block_rows)
    old_ok = supported(cfg) and geometry(cfg)["nt"] * 8 <= block_rows
    # the many-row kernel also serves 2-row hd256 steps (qdec.cu needs 74 KB there: one CTA per SM)
    if ROWS_MODE == "2" or (ROWS_MODE == "1" and (not old_ok or (cfg["hd"] >= 256 and cfg["qlen"] == 2))):
        rc = dict(cfg, rows_kernel=True, nw=rows_nw(cfg["hd"]), block_rows=max(16, triton.next_power_of_2(geometry(cfg)["rows"])))
        if supported(rc):
            cfg = rc
            old_ok = True
    if not old_ok:
        return None
    try:
        cfg["nw"], cfg["st"] = pick_nw_st(cfg)
    except QdecUnsupported:
        return None
    return cfg


def split_cap(cfg, device, programs):
    """kv splits per program: one full wave of resident CTAs (the many-row kernel's CTAs are small and several fit an SM;
    qdec.cu keeps one CTA's share per SM)"""
    sm = sm_count(device)
    if cfg.get("rows_kernel") and not os.environ.get("EXL3_QDEC_SPL"):
        sm *= get_kernel(cfg).blocks_per_sm()
    elif os.environ.get("EXL3_QDEC_SPL"):
        return int(os.environ["EXL3_QDEC_SPL"])
    return max(1, sm // programs)


def sm_count(device):
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _sm:
        _sm[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return _sm[idx]


def cubin_for(cfg):
    return _compile(cfg)
