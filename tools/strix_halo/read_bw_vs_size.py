#!/usr/bin/env python
"""How fast can this GPU read N bytes in ONE kernel, as a function of N?

Small kernels never reach the 236 GB/s roofline measured on big copies: each launch pays
ramp-up + DRAM latency before the memory system is saturated. This maps the achievable
read bandwidth per kernel size so per-kernel targets are realistic (gr_dots reads 6.3 MiB).
"""
import torch
dev = "cuda"
junk = torch.empty(96 << 20, device=dev, dtype=torch.uint8)

def bench(fn, iters=100):
    for _ in range(5): fn()
    torch.cuda.synchronize()
    st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
    tot = 0.0
    for _ in range(iters):
        junk.zero_()            # evict from L2 / MALL
        st.record(); fn(); en.record(); en.synchronize()
        tot += st.elapsed_time(en)
    return tot / iters * 1000

print(f"{'MiB':>6} {'sum() us':>9} {'GB/s':>7}   {'clone us':>9} {'GB/s(r+w)':>10}")
for mib in (0.25, 0.5, 1, 2, 4, 6.3, 8, 16, 32, 64):
    n = int(mib * 2**20)
    x = torch.empty(n // 2, device=dev, dtype=torch.half)
    us_sum = bench(lambda: x.float().sum() if n < 2**22 else torch.sum(x, dtype=torch.float))
    us_cl = bench(lambda: x.clone())
    print(f"{mib:>6.2f} {us_sum:>9.1f} {n/us_sum/1e3:>7.1f}   {us_cl:>9.1f} {2*n/us_cl/1e3:>10.1f}")

# MALL check: same 6.3 MiB read twice back to back without eviction
x = torch.empty(int(6.3 * 2**20) // 2, device=dev, dtype=torch.half)
def two():
    torch.sum(x, dtype=torch.float); torch.sum(x, dtype=torch.float)
one = bench(lambda: torch.sum(x, dtype=torch.float))
print(f"\n6.3 MiB: cold read {one:.1f} us, cold+warm pair {bench(two):.1f} us -> warm ~{bench(two)-one:.1f} us (MALL/L2 hit if << cold)")
