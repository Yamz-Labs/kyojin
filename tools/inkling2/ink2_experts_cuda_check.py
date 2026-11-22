"""CPU numpy check of the CUDA kernel's lane / fragment maths (no GPU): emulates, lane by lane, the loads, nibble decode, fp16 scale fold, B-pair order and the
mma.sync.m16n8k16 fragment layout of ink2_experts_cuda.cu for one 16 x 8 tile over random NVFP4 bytes (all 128 UE4M3 codes, incl. subnormals), and compares with an fp64 decode.
  python -I ink2_experts_cuda_check.py     -> prints CHECK ... lines, exit 7 on failure"""
import sys
import numpy as np

KV = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], np.float64)


def np_nvfp4(raw):  # same independent decode as experts_micro.np_nvfp4 (ggml: ue4m3(scale) * 0.5 * kvalues[nibble]; 0 and 0x7f -> 0)
    N = raw.shape[0]
    b = raw.reshape(N, -1, 36)
    s = b[..., :4].astype(np.int64)
    ex, ma = (s >> 3) & 15, s & 7
    sv = np.where(ex == 0, ma * 2.0 ** -9, (1 + ma / 8.0) * 2.0 ** (ex - 7)) * 0.5
    sv = np.where((s == 0) | (s == 0x7F), 0.0, sv)
    qs = b[..., 4:].reshape(N, -1, 4, 8)
    vals = np.concatenate([KV[qs & 15], KV[qs >> 4]], -1)
    return (vals * sv[..., None]).reshape(N, -1)


def h16(bits):
    return np.array(bits, np.uint16).view(np.float16)


def dec_pair(t):  # u32 -> (lo half bits, hi half bits)
    r = ((t & 0x00070007) << 9) | ((t & 0x00080008) << 12)
    return np.uint16(r & 0xffff), np.uint16(r >> 16)


def scale2(sb):  # fp16 scale * 2^7
    s = np.uint16((sb & 0x7f) << 7)
    return np.float16(h16([s])[0] * np.float16(32768.0))


def pack(x):
    return np.clip(np.asarray(x, np.float64), -65504, 65504).astype(np.float16)


def run(NB, seed):
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 256, size=(16, NB * 36), dtype=np.uint8)
    codes = rng.integers(1, 0x7f, size=(16, NB, 4))            # valid UE4M3 codes (0x7f = NaN in the format; real tensors have none)
    raw3 = raw.reshape(16, NB, 36)
    raw3[..., :4] = codes
    x = rng.standard_normal((8, NB * 64))
    ref = np_nvfp4(raw) @ x.T                                      # [16, 8] fp64; scale: ggml 0.5 * ue4m3 -> kernel folds 2^-1? check below
    u32 = raw3.reshape(16, NB, 9, 4).view("<u4").reshape(16, NB, 9)    # u32 view of each 36-byte block (little endian)
    C = np.zeros((16, 8))                                              # mma accumulators, filled lane by lane
    maxerr_A = 0.0
    for blk in range(NB):
        for lane in range(32):
            g, t = lane >> 2, lane & 3
            rows = [g, g + 8]
            A = {}
            for r in rows:
                d0 = int(u32[r, blk, 0]); w0 = int(u32[r, blk, 1 + 2 * t]); w1 = int(u32[r, blk, 2 + 2 * t])
                s2 = scale2(d0 >> (8 * t))
                regs = []
                for w, sh in ((w0, 0), (w0, 8), (w0, 4), (w0, 12), (w1, 0), (w1, 8), (w1, 4), (w1, 12)):
                    lo, hi = dec_pair(w >> sh)
                    regs.append((np.float16(h16([lo])[0] * s2), np.float16(h16([hi])[0] * s2)))
                A[r] = regs
                # exactness of the fp16 fold: compare with the fp64 value (E2M1 * scale * 2^-7); ggml's own 0.5 factor is applied below
                exact = []
                for w, sh in ((w0, 0), (w0, 8), (w0, 4), (w0, 12), (w1, 0), (w1, 8), (w1, 4), (w1, 12)):
                    n0, n1 = (w >> sh) & 0xf, (w >> (sh + 16)) & 0xf
                    sv = (codes[r, blk, t] & 0x7f)
                    ex, ma = sv >> 3, sv & 7
                    scl = ma * 2.0 ** -9 if ex == 0 else (1 + ma / 8.0) * 2.0 ** (ex - 7)
                    exact.append((KV[n0] * 0.5 * scl * 2.0 ** -7, KV[n1] * 0.5 * scl * 2.0 ** -7))
                got = [(float(a), float(b)) for a, b in regs]
                maxerr_A = max(maxerr_A, max(abs(a - b) for gp, ep in zip(got, exact) for a, b in zip(gp, ep)))
            # B pairs for column n = g, k group (blk, t)
            xe = pack(x[g, blk * 64 + 16 * t: blk * 64 + 16 * t + 16])
            e = xe
            xb = [(e[0], e[2]), (e[1], e[3]), (e[8], e[10]), (e[9], e[11]), (e[4], e[6]), (e[5], e[7]), (e[12], e[14]), (e[13], e[15])]
            # mma: step s uses P = reg 2s, Q = reg 2s + 1; thread (g, t) holds A[row g] k slots (2t, 2t+1) = P, (2t+8, 2t+9) = Q; B[k slot][n = g] same slots.
            # Accumulate into C by the PTX definition: C[m][n] += sum_k A[m][k] B[k][n]; each (row, slot, col) product is attributed to the lane that owns A and B.
            # Here the A element of row r at slot k is held by lane (r % 8, k//2 % 4); B element at slot k column n by lane (n, k//2 % 4): emulate by assigning
            # k slot of thread t: slots 2t, 2t+1 (P) and 2t+8, 2t+9 (Q), which every column n and row m of the mma sees through its own lane. So the sum over the 4 t of the tile
            # equals the sum over the 16 slots; the kernel's per-lane values (A rows g/g+8, all 8 columns via the B lanes) are accumulated as:
            for s in range(4):
                for r_i, r in enumerate(rows):
                    pa = A[r][2 * s], A[r][2 * s + 1]
                    # this lane's slots contribute to every column n: need B of column n for the SAME t -> built from lane (n, t)
                    for n in range(8):
                        en = pack(x[n, blk * 64 + 16 * t: blk * 64 + 16 * t + 16])
                        xbn = [(en[0], en[2]), (en[1], en[3]), (en[8], en[10]), (en[9], en[11]), (en[4], en[6]), (en[5], en[7]), (en[12], en[14]), (en[13], en[15])]
                        P, Q = xbn[2 * s], xbn[2 * s + 1]
                        C[r, n] += float(pa[0][0]) * float(P[0]) + float(pa[0][1]) * float(P[1]) + float(pa[1][0]) * float(Q[0]) + float(pa[1][1]) * float(Q[1])
    # kernel result = C * 2^7, ggml convention has weights = ue4m3 * 0.5 * kv: the HIP kernel decodes E2M1 as raw kv/2 (E2M1 = 0.5 * kv) -> A = E2M1 * scale_true * 2^-7 with scale_true = ue4m3
    out = C * 128.0
    rel = np.linalg.norm(out - ref) / np.linalg.norm(ref)
    xr = pack(x).astype(np.float64)                                    # reference with the fp16-rounded activations: only fp32 accumulation order differs
    rel16 = np.linalg.norm(out - np_nvfp4(raw) @ xr.T) / np.linalg.norm(ref)
    return rel, rel16, maxerr_A


def run_q8(NP, seed):
    """Q8_0: 34-byte blocks (fp16 d + 32 int8), block pairs of 68 bytes = 17 dwords; lane (g, t) owns bytes 8t..8t+7 of both blocks of every pair"""
    rng = np.random.default_rng(seed)
    rows = rng.integers(-128, 128, size=(16, NP * 2, 32)).astype(np.int8)
    d = (rng.random((16, NP * 2)) * 0.05 + 0.001).astype(np.float16)
    raw = np.zeros((16, NP * 2, 34), np.uint8)
    raw[..., :2] = d.view(np.uint8).reshape(16, NP * 2, 2)
    raw[..., 2:] = rows.view(np.uint8)
    raw = raw.reshape(16, NP * 68)
    ref_w = (rows.astype(np.float64) * d.astype(np.float64)[..., None]).reshape(16, NP * 64)
    x = rng.standard_normal((8, NP * 64))
    u32 = raw.reshape(16, NP, 17, 4).view("<u4").reshape(16, NP, 17).astype(np.uint64)
    M32 = np.uint64(0xffffffff)

    def funnel(lo, hi):                                            # __funnelshift_r(lo, hi, 16)
        return ((hi << np.uint64(32)) | lo) >> np.uint64(16) & M32

    def dec_q8(w):                                                 # -> pairs P (b0, b1), Q (b2, b3) as fp16 values
        u = int(w) ^ 0x80808080
        bytes_ = [(u >> (8 * i)) & 0xff for i in range(4)]
        v = [np.uint16(0x6400 | b).view(np.float16) - np.float16(1152.0) for b in bytes_]   # byte_perm 0x5140 / 0x5342 + sub.f16x2
        return (v[0], v[1]), (v[2], v[3])

    C = np.zeros((16, 8))
    maxerr = 0.0
    for p in range(NP):
        for lane in range(32):
            g, t = lane >> 2, lane & 3
            for blk in range(2):
                words, dsc = {}, {}
                for r in (g, g + 8):
                    dw = u32[r, p]
                    if blk == 0:
                        w = [funnel(dw[2 * t], dw[2 * t + 1]), funnel(dw[2 * t + 1], dw[2 * t + 2])]
                        sc = np.uint16(int(dw[0]) & 0xffff).view(np.float16)
                    else:
                        w = [dw[9 + 2 * t], dw[10 + 2 * t]]
                        sc = np.uint16(int(dw[8]) >> 16).view(np.float16)
                    words[r], dsc[r] = w, float(sc)
                xe = pack(x[:, p * 64 + 32 * blk + 8 * t: p * 64 + 32 * blk + 8 * t + 8])        # [8 cols, 8]
                for s in range(2):
                    for r in (g, g + 8):
                        P, Q = dec_q8(words[r][s])
                        for n in range(8):
                            e = xe[n]
                            C[r, n] += dsc[r] * (float(P[0]) * float(e[4 * s]) + float(P[1]) * float(e[4 * s + 1]) + float(Q[0]) * float(e[4 * s + 2]) + float(Q[1]) * float(e[4 * s + 3]))
                        # exactness of the int8 decode
                        base = rows[r, 2 * p + blk, 8 * t + 4 * s: 8 * t + 4 * s + 4].astype(np.float64)
                        maxerr = max(maxerr, abs(float(P[0]) - base[0]), abs(float(P[1]) - base[1]), abs(float(Q[0]) - base[2]), abs(float(Q[1]) - base[3]))
    ref = ref_w @ pack(x).astype(np.float64).T
    return float(np.linalg.norm(C - ref) / np.linalg.norm(ref)), maxerr


if __name__ == "__main__":
    bad = 0
    for NB, seed in ((2, 1), (4, 2)):
        rel, rel16, ea = run(NB, seed)
        ok = rel < 1e-3 and rel16 < 1e-9 and ea == 0.0
        bad += (not ok)
        print(f"CHECK NB={NB} rel_l2_vs_fp64={rel:.3e} rel_vs_fp16_x_ref={rel16:.3e} fp16_fold_max_abs_err={ea:.3e} ok={ok}", flush=True)
    for NP, seed in ((2, 3), (3, 4)):
        rel, ea = run_q8(NP, seed)
        ok = rel < 1e-6 and ea == 0.0
        bad += (not ok)
        print(f"CHECK Q8_0 pairs={NP} rel_vs_fp16_x_ref={rel:.3e} int8_decode_max_err={ea:.1f} ok={ok}", flush=True)
    sys.exit(7 if bad else 0)
