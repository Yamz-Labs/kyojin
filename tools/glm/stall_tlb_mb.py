"""stall1 microbench: does GPU GEMM slow down when an operand sits on scattered 4 KiB pages?

Places the weight operand of the prefill GEMM (ext.hgemm_recon, x[M,K] @ w[K,N]) in:
  torch   : normal torch caching-allocator tensor (GTT BO)
  up_thp  : hipHostRegister'd anonymous memory, MADV_HUGEPAGE (2 MiB physical runs)
  up_4k   : hipHostRegister'd anonymous memory, MADV_NOHUGEPAGE, pages interleaved with a
            twin buffer at first touch so no two neighbours are physically adjacent
Same bytes, same kernel. If up_4k is several x slower than up_thp, 4 KiB-page placement of a
hot transient buffer is a sufficient cause for the served stall.
Usage: python tools/glm/stall_tlb_mb.py [M=4096] [K=6144] [N=12288] [reps=20]
"""
import ctypes, mmap, os, sys, time, json
import torch

M = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
K = int(sys.argv[2]) if len(sys.argv) > 2 else 6144
N = int(sys.argv[3]) if len(sys.argv) > 3 else 12288
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 20
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from exllamav3.ext import exllamav3_ext as ext

hip = ctypes.CDLL("libamdhip64.so")
libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
MADV_HUGEPAGE, MADV_NOHUGEPAGE = 14, 15
PG = 4096

class Dev:
    def __init__(self, ptr, shape):
        self.__cuda_array_interface__ = {"shape": shape, "typestr": "<f2", "data": (ptr, False), "version": 3}

def host_buf(nbytes, huge):
    size = (nbytes + (2 << 20) - 1) // (2 << 20) * (2 << 20)
    bufs = []
    for _ in range(1 if huge else 2):
        m = mmap.mmap(-1, size + (2 << 20), flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        base = ctypes.addressof(ctypes.c_char.from_buffer(m))
        al = (base + (2 << 20) - 1) & ~((2 << 20) - 1)
        libc.madvise(ctypes.c_void_p(al), size, MADV_HUGEPAGE if huge else MADV_NOHUGEPAGE)
        bufs.append((m, al))
    # first touch; 4k case interleaves the two buffers page by page
    arrs = [(ctypes.c_char * size).from_address(al) for _, al in bufs]
    if huge:
        ctypes.memset(bufs[0][1], 0, size)
    else:
        for off in range(0, size, PG):
            arrs[0][off] = b"\0"; arrs[1][off] = b"\0"
    al = bufs[0][1]
    r = hip.hipHostRegister(ctypes.c_void_p(al), ctypes.c_size_t(size), 0)
    assert r == 0, f"hipHostRegister {r}"
    dp = ctypes.c_void_p()
    r = hip.hipHostGetDevicePointer(ctypes.byref(dp), ctypes.c_void_p(al), 0)
    assert r == 0, f"hipHostGetDevicePointer {r}"
    return bufs, dp.value, size

def thp_share(al, size):
    # fraction of the range backed by AnonHugePages (smaps of the containing vma)
    cur = None; tot = huge = 0
    for line in open("/proc/self/smaps"):
        if "-" in line.split()[0] and not line.startswith(("Anon", "Size")):
            lo, hi = (int(v, 16) for v in line.split()[0].split("-")); cur = lo <= al < hi
        elif cur and line.startswith("AnonHugePages:"):
            huge = int(line.split()[1]) * 1024
        elif cur and line.startswith("Rss:"):
            tot = int(line.split()[1]) * 1024
    return round(huge / max(tot, 1), 3)

def bench(w, x, y):
    for _ in range(3): ext.hgemm_recon(x, w, y)
    torch.cuda.synchronize()
    ts = []
    for _ in range(REPS):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record(); ext.hgemm_recon(x, w, y); e1.record(); e1.synchronize()
        ts.append(e0.elapsed_time(e1))
    ts.sort()
    return round(ts[len(ts) // 2], 3)

torch.manual_seed(0)
x = torch.randn(M, K, dtype=torch.half, device="cuda")
y = torch.empty(M, N, dtype=torch.half, device="cuda")
w_t = torch.randn(K, N, dtype=torch.half, device="cuda") * 0.02
ref = torch.empty_like(y); ext.hgemm_recon(x, w_t, ref)
res = {"shape": [M, K, N]}
keep = []
for name, huge in (("up_thp", True), ("up_4k", False)):
    bufs, dp, size = host_buf(K * N * 2, huge)
    keep.append(bufs)
    w = torch.as_tensor(Dev(dp, (K, N)), device="cuda")
    w.copy_(w_t); torch.cuda.synchronize()
    res[name + "_thp_share"] = thp_share(bufs[0][1], size)
    res[name] = None
    y.zero_(); ext.hgemm_recon(x, w, y); torch.cuda.synchronize()
    res[name + "_equal"] = bool(torch.equal(y, ref))
    globals()["w_" + name] = w
for rnd in range(3):  # interleaved rounds
    for name in ("torch", "up_thp", "up_4k"):
        w = w_t if name == "torch" else globals()["w_" + name]
        res.setdefault(name + "_ms", []).append(bench(w, x, y))
for name in ("torch", "up_thp", "up_4k"):
    res[name] = min(res[name + "_ms"])
fl = 2 * M * K * N
res["tflops"] = {n: round(fl / res[n] / 1e9, 1) for n in ("torch", "up_thp", "up_4k")}
print(json.dumps(res))
