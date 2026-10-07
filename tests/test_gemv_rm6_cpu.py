#!/usr/bin/env python3
"""CPU proof (row5): EXL3_GEMV_R_RM6 runs 5 or 6 rows as ONE chunk (rpb = R, grid (blocks, 1)) of gemv_kernel_r<KB2, CB, 6, 1>.
Checks the index/size arithmetic copied from exl3_dec.cu / dec_gemv_r_impl over the real shapes: every (job, strip, kb, row) is
covered exactly once, row < R, xs staging rows * ktw * 128 halves <= LDS (MOE_R_MAX * MOE_R_KTW_MAX * 128), acc row < RM = 6,
scratch / counter offsets stay inside R * kbs * N / R * strips. Mutants must fail."""
import random
BUDGET = 8 * 16          # GEMV_R_ROW_BUDGET = MOE_R_MAX * MOE_R_KTW_MAX
LDS_HALVES = 8 * 16 * 128
def run(mut=None, trials=4000):
    random.seed(11)
    for _ in range(trials):
        R = random.choice([5, 6]); ktw = random.choice([1, 2, 4, 8, 16, 21, 32])
        guard = (BUDGET // ktw >= R) if mut != "noguard" else True
        if not guard: continue                   # launch falls back to the 4-row chunks
        rpb = R
        if mut == "rpb7": rpb = 7
        count = random.randint(1, 4); kbs = random.choice([1, 2, 3, 4, 6, 8])
        strips = [random.randint(1, 40) for _ in range(count)]; Ns = [s * 512 - random.choice([0, 128, 256, 384]) for s in strips]
        base = []; acc = 0
        for s in strips: base.append(acc); acc += s * kbs
        blocks = acc; nch = (R + rpb - 1) // rpb
        if nch != 1: return False
        if rpb * ktw * 128 > LDS_HALVES: return False
        cov = {}
        for bx in range(blocks):
            ji = 0
            for i in range(1, 4):
                if i < count and bx >= base[i]: ji = i
            local = bx - base[ji]; strip = local // kbs; kb = local % kbs
            row0 = 0; nrows = min(rpb, R - row0)
            if nrows != R or not (0 <= strip < strips[ji] and 0 <= kb < kbs): return False
            for j in range(nrows):
                if j >= 6 or row0 + j >= R: return False        # acc[6][16], s_xsum[j], out row
                part = (row0 + j) * kbs * Ns[ji] + kb * Ns[ji] + Ns[ji]
                if part > R * kbs * Ns[ji]: return False
                ctr = (row0 + j) * strips[ji] + strip
                if ctr >= R * strips[ji]: return False
                cov[(ji, strip, kb, row0 + j)] = cov.get((ji, strip, kb, row0 + j), 0) + 1
        if any(v != 1 for v in cov.values()) or len(cov) != blocks * R: return False
    return True
def test_rm6():
    assert run(None)
    assert not run("noguard") and not run("rpb7")
if __name__ == "__main__":
    test_rm6(); print("rm6 cpu proof ok, mutants rejected")
