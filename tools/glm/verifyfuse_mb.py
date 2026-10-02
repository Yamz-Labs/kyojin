# verifyfuse1 microbench: R-row dense EXL3 GEMV at GLM verify shapes, no model load.
#   A = served R=2 path (bc.run_alloc / exl3_gemv: had_in + exl3_gemv + had_out, 3 launches)
#   B = exl3_dec_gemv_r (input/output Hadamards inside, 1 launch; bit-exact vs R batch-1 dec_gemv)
#   B2 = exl3_dec_gemv_r_multi for shared gate+up (1 launch for 2 matrices)
# Timing: N back-to-back calls in one stream, CUDA events, median of reps -> us/call incl. gaps.
#   usage: verifyfuse_mb.py [R] [out.json]
import json, os, sys, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.quant.exl3 import dec_workspace

R = int(sys.argv[1]) if len(sys.argv) > 1 else 2
OUT = sys.argv[2] if len(sys.argv) > 2 else None
dev = torch.device("cuda:0"); torch.manual_seed(0)
# (name, k, n, K, out_fp32, launches of this linear per verify round)
SHAPES = [
    ("sh_gate", 4096, 2048, 4, False, 42), ("sh_up", 4096, 2048, 4, False, 42),
    ("sh_down", 2048, 4096, 4, True, 42), ("kda_o", 8192, 4096, 4, True, 34),
    ("kda_qkv", 4096, 24576, 4, True, 34), ("dense_gu", 4096, 12288, 3, False, 6),
    ("dense_down", 12288, 4096, 3, True, 3),
]

def mk(k, n, K):
    tr = torch.randint(-32768, 32767, (k // 16, n // 16, 16 * K), dtype=torch.int16, device=dev)
    suh = (torch.randint(0, 2, (k,), device=dev) * 2 - 1).half()
    svh = (torch.randint(0, 2, (n,), device=dev) * 2 - 1).half()
    return tr, suh, svh

def timeit(fn, n=100, reps=7):
    for _ in range(5): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(n): fn()
        b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1000 / n)
    return statistics.median(ts)

def census(fn):
    from torch.profiler import profile, ProfilerActivity
    fn(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        fn(); torch.cuda.synchronize()
    return [e.name[:40] for e in p.events() if e.device_type.name == "CUDA"]

def rel(a, b): return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()

scratch, counters = dec_workspace(dev)
res = {"R": R, "rows": []}
tot_a = tot_b = 0.0
for name, k, n, K, f32, cnt in SHAPES:
    # rotate NC weight copies so the working set (> 96 MB) never sits in the 32 MB Infinity Cache
    NC = max(1, min(24, -(-96 * 2**20 // (k * n * K // 8))))
    cps = [mk(k, n, K) for _ in range(NC)]
    tr, suh, svh = cps[0]
    it = [0]
    def nxt():
        it[0] = (it[0] + 1) % NC
        return cps[it[0]]
    x = (torch.randn(R, k, device=dev) * 0.5).half()
    od = torch.float if f32 else torch.half
    xh = torch.empty((1, k), dtype=torch.half, device=dev)
    bc = ext.BC_LinearEXL3(tr, suh, svh, K, None, False, True, xh) if hasattr(ext, "BC_LinearEXL3") else None
    ya = torch.empty(R, n, dtype=od, device=dev); yb = torch.empty_like(ya); yg = torch.empty_like(ya)
    A_had = torch.empty(R, k, dtype=torch.half, device=dev)
    fa = (lambda: bc.run_alloc(x, n, f32)) if bc is not None else (lambda: (ext.exl3_gemv(x, tr, ya, suh, A_had, svh, False, True), ya)[1])
    fg = lambda: ext.exl3_gemv(x, tr, yg, suh, A_had, svh, False, True)
    fb = lambda: ext.exl3_dec_gemv_r(x, tr, suh, svh, yb, scratch, counters, K)
    def fgr():
        t, su_, sv_ = nxt(); ext.exl3_gemv(x, t, yg, su_, A_had, sv_, False, True)
    def fbr():
        t, su_, sv_ = nxt(); ext.exl3_dec_gemv_r(x, t, su_, sv_, yb, scratch, counters, K)
    ra = fa().clone(); fg(); fb(); torch.cuda.synchronize()
    # batch-1 reference: R x exl3_dec_gemv (R=1 plain decode math)
    y1 = torch.empty_like(ya)
    for r in range(R): ext.exl3_dec_gemv(x[r:r + 1], tr, suh, svh, y1[r:r + 1], scratch, counters, K)
    torch.cuda.synchronize()
    row = dict(name=name, k=k, n=n, K=K, cnt=cnt,
               census_bc=census(fa), census_gemv=census(fg), census_r=census(fb),
               rel_bc_vs_gemv=rel(ra, yg), rel_r_vs_bc=rel(yb, ra), r_eq_batch1=bool(torch.equal(yb, y1)),
               us_bc=timeit(fgr), us_gemv=timeit(fg), us_r=timeit(fbr), us_r_hot=timeit(fb), ncopies=NC)
    row["save_us_round"] = (row["us_bc"] - row["us_r"]) * cnt
    tot_a += row["us_bc"] * cnt; tot_b += row["us_r"] * cnt
    res["rows"].append(row)
    print(f"{name:10s} k{k:6d} n{n:6d} K{K} bc {row['us_bc']:7.1f} gemv {row['us_gemv']:7.1f} r {row['us_r']:7.1f} us"
          f"  save {row['save_us_round']:7.1f} us/rd  rel(r,bc) {row['rel_r_vs_bc']:.2e} rel(bc,gemv) {row['rel_bc_vs_gemv']:.1e}"
          f" hot gemv {row['us_gemv']:.1f} r {row['us_r_hot']:.1f} x{NC} r==b1 {row['r_eq_batch1']}  launches bc {len(row['census_bc'])} r {len(row['census_r'])}", flush=True)
    del tr, cps
# shared gate+up in one multi launch vs two single launches
tg, sg, vg = mk(4096, 2048, 4); tu, su, vu = mk(4096, 2048, 4)
x = (torch.randn(R, 4096, device=dev) * 0.5).half()
g2 = torch.empty(R, 2048, dtype=torch.half, device=dev); u2 = torch.empty_like(g2)
fm = lambda: ext.exl3_dec_gemv_r_multi(x, [tg, tu], [sg, su], [vg, vu], [g2, u2], scratch, counters, [4.0, 4.0], False)
fs = lambda: (ext.exl3_dec_gemv_r(x, tg, sg, vg, g2, scratch, counters, 4.0), ext.exl3_dec_gemv_r(x, tu, su, vu, u2, scratch, counters, 4.0))
fm(); torch.cuda.synchronize(); gm, um = g2.clone(), u2.clone(); fs(); torch.cuda.synchronize()
res["multi_gu"] = dict(us_multi=timeit(fm), us_two=timeit(fs), eq=bool(torch.equal(gm, g2) and torch.equal(um, u2)))
print("shared gate+up multi", res["multi_gu"])
res["tot_bc_ms_round"] = tot_a / 1000; res["tot_r_ms_round"] = tot_b / 1000
print(f"TOTAL per round: bc {tot_a/1000:.3f} ms  gemv_r {tot_b/1000:.3f} ms  save {(tot_a-tot_b)/1000:.3f} ms")
if OUT: json.dump(res, open(OUT, "w"), indent=1)
