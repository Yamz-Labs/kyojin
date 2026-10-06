#!/usr/bin/env python3
"""CPU proof (T80): gemv_kernel_r with jobs.swap=1 visits exactly the same (job, strip, kb, row0, nrows) work items as swap=0,
each once, and every index it derives stays in range. Emulates the kernel's index code (bid/chunk -> ji/local/strip/kb/row0/nrows)
copied from exl3_dec.cu; mutants must fail."""
import random, itertools, sys
def kernel_items(jobs_block_base, count, kbs, R, rpb, swap, grid, mut=None):
    gx, gy = grid
    items = []
    for by in range(gy):
        for bx in range(gx):
            bid = by if swap else bx
            chunk = bx if swap else by
            if mut == "nochunk": chunk = 0
            ji = 0
            for i in range(1, 4):
                if i < count and bid >= jobs_block_base[i]: ji = i
            local = bid - jobs_block_base[ji]
            strip = local // kbs; kb = local % kbs
            row0 = chunk * rpb
            if mut == "rowoff": row0 += 1
            nrows = min(rpb, R - row0)
            items.append((ji, strip, kb, row0, nrows))
    return items
def check(mut=None, trials=3000):
    random.seed(7)
    for _ in range(trials):
        count = random.randint(1, 4); kbs = random.choice([1, 2, 3, 4, 6, 8])
        strips = [random.randint(1, 40) for _ in range(count)]
        base = []; acc = 0
        for s in strips: base.append(acc); acc += s * kbs
        blocks = acc; R = random.randint(5, 8); rpb = random.choice([2, 3, 4]); nch = (R + rpb - 1) // rpb
        a = kernel_items(base, count, kbs, R, rpb, 0, (blocks, nch), mut)
        swap = nch > 1
        b = kernel_items(base, count, kbs, R, rpb, 1, (nch, blocks), mut)
        if sorted(a) != sorted(b) or len(set(b)) != len(b): return False
        if blocks > 65535: return False
        for ji, strip, kb, row0, nrows in b:
            if not (0 <= ji < count and 0 <= strip < strips[ji] and 0 <= kb < kbs and 0 <= row0 < R and 1 <= nrows <= rpb and row0 + nrows <= R): return False
        # coverage: every (job,strip,kb,row) exactly once
        cov = {}
        for ji, strip, kb, row0, nrows in b:
            for r in range(row0, row0 + nrows): cov[(ji, strip, kb, r)] = cov.get((ji, strip, kb, r), 0) + 1
        if any(v != 1 for v in cov.values()) or len(cov) != blocks * R: return False
    return True
def test_grid_swap_mapping():
    assert check(None)
    assert not check("nochunk") and not check("rowoff")


if __name__ == "__main__":
    ok = check(None)
    mut_ok = [check(m) for m in ("nochunk", "rowoff")]
    print("original mapping equal + covering:", ok, "| mutants rejected:", [not m for m in mut_ok])
    sys.exit(0 if ok and not any(mut_ok) else 1)
