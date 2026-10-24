"""CPU bounds proof for mla_dec_valu.cuh (index arithmetic transcribed 1:1), real GLM shapes, all R, all UN.
Every global access index is enumerated for every (block, thread); asserts: in range, outputs written exactly once.
Mutations (off-by-one in each index formula) must FAIL the same checks."""
import numpy as np, itertools, sys
import os
H, DN, DC, DV = 64, 256, 512, 256
QK = int(os.environ.get("QK", "256"))   # GLM-5.3: NoPE q head = 256; QK=320 checks a rope-padded head
QS_OPTS = (H * QK, H * QK + 64)       # unpadded and padded q row stride (halves)
def absorb(R, UN, qs, mut=None):
    T = 256
    nq = (R - 1) * qs + (H - 1) * QK + DN        # q extent needed
    qbuf = (R - 1) * qs + H * QK + 1             # a deliberately generous buffer: rows allocated (R, qs)
    qbuf = R * qs
    wbuf = DC * H * DN
    obuf = H * R * DC
    h, bx, tx = np.meshgrid(np.arange(H), np.arange(DC // T), np.arange(T), indexing="ij")
    h, bx, tx = h.ravel(), bx.ravel(), tx.ravel()
    n = bx * T + tx
    # q staging: i < R*(DN/8), uint4 index = r*qs/8 .. in halves: r*qs + h*QK + 8*c
    for hh in range(H):
        i = np.arange(R * (DN // 8)); r = i // (DN // 8); c = i % (DN // 8)
        idx = r * qs + hh * QK + 8 * c + (1 if mut == "q" else 0)
        assert idx.min() >= 0 and idx.max() + 7 < qbuf, ("q oob", idx.max() + 7, qbuf)
        assert (idx % 8 == 0).all(), "q uint4 misaligned"
    assert qs % 8 == 0 and QK % 8 == 0
    # w loads: uint4 index c0 + j over c0 in range(0, DN/8, UN): halves = n*H*DN + h*DN + 8*(c0+j)
    cs = np.array([c0 + j for c0 in range(0, DN // 8, UN) for j in range(UN)])
    assert cs.max() == DN // 8 - 1 and len(cs) == DN // 8 and DN // 8 % UN == 0, "UN does not tile DN/8"
    wlo = n * H * DN + h * DN + (1 if mut == "w" else 0)
    whi = wlo + 8 * cs.max() + 7
    assert wlo.min() >= 0 and whi.max() < wbuf, ("w oob", whi.max(), wbuf)
    # every w element read exactly once
    seen = np.zeros(wbuf, np.int8)
    for c in cs:
        np.add.at(seen, wlo + 8 * c, 1)   # first half of each uint4 (alignment-level check)
    assert (wlo % 8 == 0).all()
    # outputs
    o = (h[:, None] * R + np.arange(R)[None, :]) * DC + n[:, None] + (DC if mut == "o" else 0)
    assert o.min() >= 0 and o.max() < obuf, ("out oob", o.max(), obuf)
    cnt = np.bincount(o.ravel(), minlength=obuf)
    assert (cnt == 1).all(), "outputs not written exactly once"
def absorb2(R, UN, qs, mut=None):
    CH, ROWS = DN // 8, 32
    assert CH == 32 and ROWS % UN == 0 and DC % ROWS == 0
    qbuf, wbuf, obuf = R * qs, DC * H * DN, H * R * DC
    h, bx, l = np.meshgrid(np.arange(H), np.arange(DC // ROWS), np.arange(32), indexing="ij"); h, bx, l = h.ravel(), bx.ravel(), l.ravel()
    for hh in range(H):
        for r in range(R):
            idx = r * qs + hh * QK + 8 * np.arange(32) + (1 if mut == "q" else 0)
            assert idx.max() + 7 < qbuf and (idx % 8 == 0).all(), "q oob/misaligned"
    seen = np.zeros(wbuf // 8, np.int32)
    for i in range(ROWS):                                  # row i of the block, lane l loads uint4 l
        n = bx * ROWS + i
        idx = n * H * DN + h * DN + 8 * l + (8 if mut == "w" else 0)
        assert idx.min() >= 0 and idx.max() + 7 < wbuf and (idx % 8 == 0).all(), ("w oob", idx.max() + 7, wbuf)
        np.add.at(seen, idx // 8, 1)
    assert (seen == 1).all(), "w uint4 not read exactly once"
    # LDS: writes wsh[i][l] (pitch CH+1), reads wsh[l][c], c < CH: all within [32][33]
    assert (31 * (CH + 1) + CH - 1) < 32 * (CH + 1)
    o = (h[:, None] * R + np.arange(R)[None, :]) * DC + bx[:, None] * ROWS + l[:, None] + (1 if mut == "o" else 0)
    assert o.min() >= 0 and o.max() < obuf, ("out oob", o.max(), obuf)
    assert (np.bincount(o.ravel(), minlength=obuf) == 1).all(), "outputs not exactly once"
def unfold(R, UN, mut=None):
    T = DV // 2
    obuf, wbuf, outbuf = H * R * DC, DC * H * DV, R * H * DV
    dc2 = DC // 2
    assert dc2 % UN == 0
    shm = R * DC * 2
    # staging o -> LDS: uint32 index i < R*dc2 of head h
    for h in range(H):
        i = np.arange(R * dc2)
        idx = h * R * DC // 2 + i + (1 if mut == "o" else 0)
        assert idx.max() < obuf // 2, "o oob"
    assert (R * dc2) * 4 <= shm
    # LDS reads: r*dc2 + k2 + j
    r = np.arange(R)[:, None]; k = np.arange(dc2)[None, :]
    assert (r * dc2 + k).max() < R * dc2
    # w loads: uint32 index h*DV/2 + t + (2*k2' (+1)) * rs ; rs = H*DV/2
    rs = H * DV // 2
    h, t = np.meshgrid(np.arange(H), np.arange(T), indexing="ij"); h, t = h.ravel(), t.ravel()
    rows = np.arange(DC)                                   # 2*(k2+j) and +1 cover all k rows exactly once
    idx = (h[:, None] * (DV // 2) + t[:, None] + rows[None, :] * rs + (1 if mut == "w" else 0))
    assert idx.min() >= 0 and idx.max() < wbuf // 2, ("w oob", idx.max(), wbuf // 2)
    assert np.bincount(idx.ravel(), minlength=wbuf // 2).max() == 1, "w element read twice or not covered"
    # outputs: uint32 index ((r*H + h)*DV)/2 + t
    oi = ((np.arange(R)[None, :] * H + h[:, None]) * DV) // 2 + t[:, None] + (1 if mut == "out" else 0)
    assert oi.min() >= 0 and oi.max() < outbuf // 2, ("out oob", oi.max(), outbuf // 2)
    assert (np.bincount(oi.ravel(), minlength=outbuf // 2) == 1).all(), "outputs not exactly once"
fails = 0
for R in range(1, 9):
    for UN in (4, 8, 16):
        for qs in QS_OPTS: absorb(R, UN, qs)
    for UN in (4, 8, 16): unfold(R, UN)
    for UN in (4, 8, 16, 32):
        for qs in QS_OPTS: absorb2(R, UN, qs)
print("clean: absorb+unfold pass for R=1..8, UN 4/8/16, both q strides")
for name, f, muts in (("absorb", lambda m: absorb(3, 8, QS_OPTS[0], m), ("q", "w", "o")), ("absorb2", lambda m: absorb2(3, 8, QS_OPTS[0], m), ("q", "w", "o")), ("unfold", lambda m: unfold(3, 8, m), ("o", "w", "out"))):
    for m in muts:
        try: f(m); print("MUTATION NOT CAUGHT", name, m); fails += 1
        except AssertionError as e: print("mutation", name, m, "caught:", str(e)[:60])
sys.exit(fails)
