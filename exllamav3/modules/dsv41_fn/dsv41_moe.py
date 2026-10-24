"""Launchers of dsv41_moe.hip: DS4.1 batch-1 routed experts of one layer in 5 launches for all K groups at once (hipcc genco on
first use, launched on torch's current stream, same loader model as dsv41_fuse.py). Bit-identical to the per-K-group
exl3_moe_gfx12_k3 path: the kernels reuse the engine's own matvec / Hadamard bodies and keep the grouped reduction order."""
import ctypes, hashlib, os, subprocess
import torch
from . import dsv41_hip as H
from exllamav3.util import hip_compiler
from exllamav3.util.hip_lib import load_hip_runtime

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_DIR, "dsv41_moe.hip")
_FUNCS = ("ds41m_had_in", "ds41m_valu_gu", "ds41m_valu_down", "ds41m_mid", "ds41m_final")
_EXT = os.path.join(os.path.dirname(os.path.dirname(_DIR)), "exllamav3_ext")
_HEADERS = ("quant/exl3_moe_valu.cuh", "quant/exl3_moe_valu_idx.h", "quant/exl3_gemv_kernel.cuh", "quant/hadamard_inner.cuh",
            "quant/exl3_dq.cuh", "quant/codebook.cuh", "compat.cuh", "compat_rocm.cuh", "util.cuh")
_state = {}


def available(device) -> bool:
    if os.environ.get("DSV41_MOE_UNI", "1") == "0":
        return False
    return torch.device(device).type == "cuda" and torch.cuda.is_available() and torch.version.hip is not None


def _compile(arch):
    src = open(_SRC, "rb").read()
    for h in _HEADERS:
        src += open(os.path.join(_EXT, h), "rb").read()
    flags = ["--genco", f"--offload-arch={arch}", "-O3", f"-I{os.path.join(_DIR, 'stub')}"]
    flags += os.environ.get("DSV41_MOE_DEFS", "").split()   # experiments: -DMV_VU=3 -DEXL3_MOE_VALU_W=8
    gcc = os.environ.get("EXL3_GCC_INSTALL_DIR", "/usr/lib/gcc/x86_64-linux-gnu/13")
    if os.path.isdir(gcc):
        flags.append(f"--gcc-install-dir={gcc}")
    tag = hashlib.sha1(src + " ".join(flags).encode() + b"\0" + hip_compiler.key().encode()).hexdigest()[:16]
    cache = os.path.join(os.path.expanduser("~/.cache/exllamav3"), f"dsv41moe_{arch}_{tag}.hsaco")
    if not os.path.isfile(cache):
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        tmp = cache + f".{os.getpid()}.tmp"
        r = subprocess.run([hip_compiler.hipcc(), *flags, "-o", tmp, _SRC], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stderr[-3000:])
        os.replace(tmp, cache)
    return cache


def _load():
    if "fns" in _state:
        return _state["fns"]
    cache = _compile(torch.cuda.get_device_properties(0).gcnArchName.split(":")[0])
    lib = load_hip_runtime()
    torch.cuda.init()
    mod = ctypes.c_void_p()
    if lib.hipModuleLoad(ctypes.byref(mod), cache.encode()) != 0:
        raise RuntimeError("dsv41_moe: hipModuleLoad failed")
    fns = {}
    for n in _FUNCS:
        fn = ctypes.c_void_p()
        if lib.hipModuleGetFunction(ctypes.byref(fn), mod, n.encode()) != 0:
            raise RuntimeError("dsv41_moe: no function " + n)
        fns[n] = fn
    lib.hipModuleLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p] * 3
    _state.update(lib=lib, fns=fns, mod=mod)
    return fns


def _W():
    return int(os.environ.get("DSV41_MOE_W", "4"))


class _L:
    """One launch with a prebuilt argument pack; `ptr(i, t)` re-points a tensor argument."""

    def __init__(self, name, grid, block, args):
        self.fn = _load()[name]
        self.grid, self.block = tuple(grid) + (1,) * (3 - len(grid)), block
        self._keep = [a for a in args if isinstance(a, torch.Tensor)]
        self.cargs, self.params = H._pack(args)

    def ptr(self, i, t):
        self.cargs[i].value = t.data_ptr()
        return self

    def __call__(self, stream):
        r = _state["lib"].hipModuleLaunchKernel(self.fn, self.grid[0], self.grid[1], self.grid[2], self.block, 1, 1, 0,
                                                stream, self.params, None)
        if r != 0:
            raise RuntimeError(f"dsv41_moe: launch failed ({r})")


class UnifiedMoE:
    """Batch-1 top-6 routed experts of one GroupedMoE layer. All K groups run in the same 5 launches; outputs are the per-group fp32
    [1, H] buffers `outs` (what ds41_moe_tail sums). Supports expert K in {1, 2, 3, 4} and top-6."""

    TOPK = 6

    def __init__(self, store, buffers, outs, H_, I_, limit, device):
        dev = torch.device(device)
        self.store, self.outs, self.limit = store, outs, float(limit)
        E = store.num_experts
        self.G = store.num_groups
        assert self.G <= 3 and all(g.k_gu in (1.0, 2.0, 3.0, 4.0) for g in store.groups)

        self._gidx = [torch.tensor(g.experts, dtype=torch.long, device=dev) for g in store.groups]

        def uni(fn):
            t = torch.zeros(E, dtype=torch.int64, device=dev)
            for g, ix in zip(store.groups, self._gidx):
                t[ix] = fn(g).to(dev)
            return t
        self._fns = dict(gt=lambda g: g.gate.ptrs_trellis, ut=lambda g: g.up.ptrs_trellis, dt=lambda g: g.down.ptrs_trellis,
                         gsuh=lambda g: g.gate.ptrs_suh, usuh=lambda g: g.up.ptrs_suh, dsuh=lambda g: g.down.ptrs_suh,
                         gsvh=lambda g: g.gate.ptrs_svh, usvh=lambda g: g.up.ptrs_svh, dsvh=lambda g: g.down.ptrs_svh)
        self.t = dict(gt=uni(lambda g: g.gate.ptrs_trellis), ut=uni(lambda g: g.up.ptrs_trellis), dt=uni(lambda g: g.down.ptrs_trellis),
                      gsuh=uni(lambda g: g.gate.ptrs_suh), usuh=uni(lambda g: g.up.ptrs_suh), dsuh=uni(lambda g: g.down.ptrs_suh),
                      gsvh=uni(lambda g: g.gate.ptrs_svh), usvh=uni(lambda g: g.up.ptrs_svh), dsvh=uni(lambda g: g.down.ptrs_svh))
        kb = torch.zeros(E, dtype=torch.int32)
        for g in store.groups:
            kb[torch.tensor(g.experts, dtype=torch.long)] = int(g.k_gu)
        self.kbits = kb.to(dev)
        self.kgroup = store.kmap.to(dev)
        self.H, self.I, self.E = H_, I_, E
        t = self.t
        K = self.TOPK
        self._x = self._sel = self._wh = None
        self.l_had = _L("ds41m_had_in", (2 * K, H_ // 128), 32,
                        (buffers.gu_had, buffers.gu_had, buffers.gu_had, t["gsuh"], t["usuh"], H_, E))
        self.l_gu = _L("ds41m_valu_gu", (mv_grid_x(I_ // 16), K, 2), 32 * _W(),
                       (buffers.gu_had, buffers.gu_had, self.kbits, t["gt"], t["ut"], buffers.gu_out, H_, I_, E))
        self.l_mid = _L("ds41m_mid", (K, I_ // 128), 32,
                        (buffers.gu_out, buffers.down_had, buffers.gu_had, t["gsvh"], t["usvh"], t["dsuh"], I_, E, self.limit))
        self.l_down = _L("ds41m_valu_down", (mv_grid_x(H_ // 16), K, 1), 32 * _W(),
                         (buffers.down_had, buffers.gu_had, self.kbits, t["dt"], t["dt"], buffers.down_out, I_, H_, E))
        o = list(outs) + [None] * (3 - len(outs))
        self.l_final = _L("ds41m_final", (1, H_ // 128), 32,
                          (buffers.down_out, buffers.gu_had, buffers.gu_had, t["dsvh"], self.kgroup, o[0], o[1], o[2], H_, E, self.G))
        self.bufs = buffers
        self.sig = None

    def refresh(self):
        """Re-copy the per-group pointer tables into the concatenated [E] tables (device-side, no host sync). Needed when the
        group tables are views of tier buffers whose slot pointers move; the tensors (and so the kernel args) stay the same."""
        for k, fn in self._fns.items():
            t = self.t[k]
            for g, ix in zip(self.store.groups, self._gidx):
                t[ix] = fn(g)

    def run(self, x, sel, wh):
        """x fp16 [1, H] contiguous, sel int64 [1, 6], wh fp16 [1, 6] (router scratch). Returns the group outputs."""
        sig = (x.data_ptr(), sel.data_ptr(), wh.data_ptr())
        if sig != self.sig:
            self.l_had.ptr(0, x).ptr(2, sel)
            self.l_gu.ptr(1, sel)
            self.l_mid.ptr(2, sel)
            self.l_down.ptr(1, sel)
            self.l_final.ptr(1, sel).ptr(2, wh)
            self.sig = sig
        s = torch.cuda.current_stream().cuda_stream
        self.l_had(s); self.l_gu(s); self.l_mid(s); self.l_down(s); self.l_final(s)
        return self.outs


def mv_grid_x(ntiles):
    W = int(os.environ.get("DSV41_MOE_W", "4"))
    return (ntiles + W - 1) // W
