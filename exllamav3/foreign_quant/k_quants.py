"""K-quant + IQ4_XS dequant references (ggml, numpy-only).

Block layouts after llama.cpp ``ggml/src/ggml-common.h`` (MIT, ggml-org),
decoded by ``dequantize_row_*`` in ``ggml/src/ggml-quants.c``. All formats
use QK_K = 256 elements per super-block:

- Q4_K (144 B): half dm[2] (d, dmin), scales[12] (6-bit sc/min pairs via
  get_scale_min_k4), qs[128] (4-bit nibbles, 8 groups of 32).
- Q5_K (176 B): dm[2], scales[12], qh[32] (extra high bit), qs[128].
- Q6_K (210 B): ql[128] (low 4 bits), qh[64] (high 2 bits), scales[16]
  (int8), d (half). y = d * sc * (q - 32).
- IQ4_XS (136 B): d (half), scales_h (u16), scales_l[4], qs[128] (nibbles
  into kvalues_iq4nl, per-32 scale ls-32).

These mirror the CUDA dots in fq_gemv.cuh bit-for-bit at the integer level
(the gate compares the CUDA q8_1-activation integer dot against
``q8_dot_numpy`` below, which replicates the dp4a arithmetic exactly), and
provide the fp32 dequant path for the q8-noise tolerance check.
"""
from __future__ import annotations

import numpy as np

QK_K = 256

KVALUES_IQ4NL = np.array(
    [-127, -104, -83, -65, -49, -35, -22, -10,
     1, 13, 25, 38, 53, 69, 89, 113], dtype=np.int32)


def _f16(b0: int, b1: int) -> float:
    return float(np.frombuffer(bytes((b0, b1)), dtype=np.float16)[0])


def _get_scale_min_k4(scales: np.ndarray, j: int) -> tuple[int, int]:
    q = [int(v) for v in scales]
    if j < 4:
        return q[j] & 63, q[j + 4] & 63
    return ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)), \
           ((q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4))


def dequant_q4_K_block(block: bytes) -> np.ndarray:
    b = np.frombuffer(block, dtype=np.uint8)
    assert len(b) == 144
    d = _f16(int(b[0]), int(b[1]))
    dmin = _f16(int(b[2]), int(b[3]))
    scales = b[4:16]
    qs = b[16:144]
    y = np.empty(QK_K, dtype=np.float32)
    is_ = 0
    p = 0
    q = [int(v) for v in qs]
    for j in range(0, QK_K, 64):
        sc, m = _get_scale_min_k4(scales, is_ + 0)
        d1, m1 = d * sc, dmin * m
        sc, m = _get_scale_min_k4(scales, is_ + 1)
        d2, m2 = d * sc, dmin * m
        for l in range(32):
            y[p] = d1 * (q[l] & 0xF) - m1
            p += 1
        for l in range(32):
            y[p] = d2 * (q[l] >> 4) - m2
            p += 1
        qs = qs[32:]
        q = [int(v) for v in qs]
        is_ += 2
    return y


def dequant_q5_K_block(block: bytes) -> np.ndarray:
    b = np.frombuffer(block, dtype=np.uint8)
    assert len(b) == 176
    d = _f16(int(b[0]), int(b[1]))
    dmin = _f16(int(b[2]), int(b[3]))
    scales = b[4:16]
    qh = b[16:48]
    qs = b[48:176]
    y = np.empty(QK_K, dtype=np.float32)
    is_ = 0
    p = 0
    u1, u2 = 1, 2
    qhb = [int(v) for v in qh]  # 32 high-bit bytes, shared across groups (u1/u2 shift)
    for j in range(0, QK_K, 64):
        sc, m = _get_scale_min_k4(scales, is_ + 0)
        d1, m1 = d * sc, dmin * m
        sc, m = _get_scale_min_k4(scales, is_ + 1)
        d2, m2 = d * sc, dmin * m
        ql = [int(v) for v in qs[:32]]
        for l in range(32):
            y[p] = d1 * ((ql[l] & 0xF) + (16 if qhb[l] & u1 else 0)) - m1
            p += 1
        for l in range(32):
            y[p] = d2 * ((ql[l] >> 4) + (16 if qhb[l] & u2 else 0)) - m2
            p += 1
        qs = qs[32:]
        u1 <<= 2
        u2 <<= 2
        is_ += 2
    return y


def dequant_q6_K_block(block: bytes) -> np.ndarray:
    b = np.frombuffer(block, dtype=np.uint8)
    assert len(b) == 210
    ql = b[0:128]
    qh = b[128:192]
    sc = np.frombuffer(bytes(b[192:208]), dtype=np.int8).astype(np.int32)
    d = _f16(int(b[208]), int(b[209]))
    y = np.empty(QK_K, dtype=np.float32)
    p = 0
    for n in range(0, QK_K, 128):
        for l in range(32):
            is_ = l // 16
            q1 = ((int(ql[l]) & 0xF) | (((int(qh[l]) >> 0) & 3) << 4)) - 32
            q2 = ((int(ql[l + 32]) & 0xF) | (((int(qh[l]) >> 2) & 3) << 4)) - 32
            q3 = ((int(ql[l]) >> 4) | (((int(qh[l]) >> 4) & 3) << 4)) - 32
            q4 = ((int(ql[l + 32]) >> 4) | (((int(qh[l]) >> 6) & 3) << 4)) - 32
            y[p] = d * int(sc[is_ + 0]) * q1
            y[p + 32] = d * int(sc[is_ + 2]) * q2
            y[p + 64] = d * int(sc[is_ + 4]) * q3
            y[p + 96] = d * int(sc[is_ + 6]) * q4
            p += 1
        p += 96
        ql, qh, sc = ql[64:], qh[32:], sc[8:]
    return y


def dequant_iq4_xs_block(block: bytes) -> np.ndarray:
    b = np.frombuffer(block, dtype=np.uint8)
    assert len(b) == 136
    d = _f16(int(b[0]), int(b[1]))
    scales_h = int(b[2]) | (int(b[3]) << 8)
    scales_l = b[4:8]
    qs = b[8:136]
    y = np.empty(QK_K, dtype=np.float32)
    p = 0
    for ib in range(QK_K // 32):
        ls = ((int(scales_l[ib // 2]) >> (4 * (ib % 2))) & 0xF) | \
             (((scales_h >> (2 * ib)) & 3) << 4)
        dl = d * (ls - 32)
        for j in range(16):
            y[p + j] = dl * int(KVALUES_IQ4NL[int(qs[j]) & 0xF])
            y[p + j + 16] = dl * int(KVALUES_IQ4NL[int(qs[j]) >> 4])
        p += 32
        qs = qs[16:]
    return y


def dequant_row(blocks: list[bytes], fmt: str) -> np.ndarray:
    fn = {"q4_k": dequant_q4_K_block, "q5_k": dequant_q5_K_block,
          "q6_k": dequant_q6_K_block, "iq4_xs": dequant_iq4_xs_block}[fmt]
    return np.concatenate([fn(b) for b in blocks]).astype(np.float32)


def dequant_iq4_xs_bulk(raw: np.ndarray) -> np.ndarray:
    """Vectorized IQ4_XS dequant: raw uint8 [nb, 136] -> float16 [nb, 256].

    Same math as dequant_iq4_xs_block (kvalues table, per-32 ls-32 scale),
    no per-block Python (the embedding table is ~1M blocks).
    """
    raw = np.ascontiguousarray(raw).reshape(-1, 136)
    nb = raw.shape[0]
    d = raw[:, 0:2].view(np.float16).astype(np.float32).reshape(nb, 1, 1)
    sh = (raw[:, 2].astype(np.uint16) | (raw[:, 3].astype(np.uint16) << 8))
    sl = raw[:, 4:8]
    qs = raw[:, 8:136].reshape(nb, 8, 16)
    grp = np.arange(8).reshape(1, 8)
    ls = (((sl[:, (grp // 2).ravel()] >> (4 * (grp % 2)).ravel()) & 0xF).reshape(nb, 8) |
          ((((sh[:, None] >> (2 * grp)) & 3) << 4).reshape(nb, 8)))
    dl = d.reshape(nb, 1) * (ls.astype(np.float32) - 32.0)
    lo = KVALUES_IQ4NL[(qs & 0xF).ravel()].reshape(nb, 8, 16).astype(np.float32)
    hi = KVALUES_IQ4NL[(qs >> 4).ravel()].reshape(nb, 8, 16).astype(np.float32)
    y = np.empty((nb, 8, 32), dtype=np.float32)
    y[:, :, 0:16] = lo
    y[:, :, 16:32] = hi
    y *= dl[:, :, None]
    return y.reshape(nb, 256).astype(np.float16)


def dequant_q4_K_bulk(raw: np.ndarray) -> np.ndarray:
    """Vectorized Q4_K dequant: raw uint8 [nb, 144] -> float16 [nb, 256].

    Same math as dequant_q4_K_block (per-32 d*sc/m*m, 4-bit lo/hi nibbles),
    no per-block Python (the Q4_K embedding table is ~2M blocks).
    """
    raw = np.ascontiguousarray(raw).reshape(-1, 144)
    nb = raw.shape[0]
    d = raw[:, 0:2].view(np.float16).astype(np.float32).reshape(nb)
    dmin = raw[:, 2:4].view(np.float16).astype(np.float32).reshape(nb)
    sc8 = raw[:, 4:16].astype(np.int32)
    sc = np.empty((nb, 8), dtype=np.float32)
    mn = np.empty((nb, 8), dtype=np.float32)
    sc[:, 0:4] = sc8[:, 0:4] & 63
    mn[:, 0:4] = sc8[:, 4:8] & 63
    sc[:, 4:8] = ((sc8[:, 8:12] & 0xF) | ((sc8[:, 0:4] >> 6) << 4))
    mn[:, 4:8] = ((sc8[:, 8:12] >> 4) | ((sc8[:, 4:8] >> 6) << 4))
    dl = d[:, None] * sc
    ml = dmin[:, None] * mn
    qs = raw[:, 16:144].reshape(nb, 4, 32)
    lo = (qs & 0xF).astype(np.float32)
    hi = ((qs >> 4) & 0xF).astype(np.float32)
    y = np.empty((nb, 256), dtype=np.float32)
    for c in range(4):
        y[:, c * 64:c * 64 + 32] = dl[:, 2 * c, None] * lo[:, c, :] - ml[:, 2 * c, None]
        y[:, c * 64 + 32:c * 64 + 64] = dl[:, 2 * c + 1, None] * hi[:, c, :] - ml[:, 2 * c + 1, None]
    return y.astype(np.float16)


def quantize_q8_1_row(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ggml quantize_q8_1: per-32 amax/127, round-half-away, returns (q, d, s)."""
    x = np.asarray(x, dtype=np.float32)
    assert x.size % 32 == 0
    xr = x.reshape(-1, 32)
    amax = np.abs(xr).max(axis=1)
    d = amax / 127.0
    q = np.zeros_like(xr, dtype=np.int32)
    nz = amax != 0
    q[nz] = np.floor(xr[nz] / d[nz, None] + 0.5).astype(np.int32)
    q = np.clip(q, -127, 127)
    s = xr.sum(axis=1)
    return q.astype(np.int8), d.astype(np.float32), s.astype(np.float32)


def _dp4a(a: int, b: int) -> int:
    s = 0
    for m in range(4):
        av = ((a >> (8 * m)) & 0xFF) - 256 * (((a >> (8 * m)) & 0x80) >> 7)
        bv = ((b >> (8 * m)) & 0xFF) - 256 * (((b >> (8 * m)) & 0x80) >> 7)
        s += av * bv
    return s


def _u32(le_bytes) -> int:
    v = 0
    for m, by in enumerate(bytes(le_bytes)):
        v |= int(by) << (8 * m)
    return v


def q8_dot_numpy(packed_row: bytes, x: np.ndarray, fmt: str) -> float:
    """Exact integer replica of the CUDA q8_1-activation dots in fq_gemv.cuh.

    Same op order as ggml vec_dot_*_q8_1 (int32 dp4a accumulation, one float
    scale per chunk, float sum over chunks). Used for the bitwise gate.
    """
    b = np.frombuffer(packed_row, dtype=np.uint8)
    q8, da32, _ = quantize_q8_1_row(x)
    # CUDA stores the q8 scales as fp16 (round-to-nearest on quantize);
    # replicate the stored values for the bitwise gate.
    da = np.float16(da32).astype(np.float32)
    nq8 = q8.reshape(-1, 32)
    acc = np.float32(0.0)
    if fmt == "q4_k":
        assert len(b) % 144 == 0
        nb = len(b) // 144
        for bb in range(nb):
            blk = b[bb * 144:(bb + 1) * 144]
            dm = np.frombuffer(bytes(blk[0:4]), dtype=np.float16).astype(np.float32)
            dm0 = np.float32(dm[0])
            dm1 = np.float32(dm[1])
            qs = blk[16:144]
            for t in range(16):
                bo = 2 * (t // 4)
                qo = (t % 4)
                v0 = _u32(qs[16 * bo + 4 * qo:16 * bo + 4 * qo + 4])
                v1 = _u32(qs[16 * bo + 4 * qo + 16:16 * bo + 4 * qo + 20])
                j = bo // 2
                jm = j & 1
                # u16 pairs, matching the CUDA reinterpret path
                s0 = int(blk[4 + jm * 2]) | (int(blk[4 + jm * 2 + 1]) << 8)
                s2 = int(blk[4 + (jm + 2) * 2]) | (int(blk[4 + (jm + 2) * 2 + 1]) << 8)
                s4 = int(blk[4 + (jm + 4) * 2]) | (int(blk[4 + (jm + 4) * 2 + 1]) << 8)
                hi = 0xFFFFFFFF if j >= 2 else 0
                aux0 = (((s0 & 0x3F3F) & (~hi & 0xFFFF)) |
                        ((((s4 >> 0) & 0x0F0F) | ((s0 & 0xC0C0) >> 2)) & (hi & 0xFFFF))) & 0xFFFF
                aux1 = (((s2 & 0x3F3F) & (~hi & 0xFFFF)) |
                        ((((s4 >> 4) & 0x0F0F) | ((s2 & 0xC0C0) >> 2)) & (hi & 0xFFFF))) & 0xFFFF
                sc = [aux0 & 0xFF, (aux0 >> 8) & 0xFF]
                mm = [aux1 & 0xFF, (aux1 >> 8) & 0xFF]
                sumf_d = np.float32(0.0)
                sumf_m = np.float32(0.0)
                for i in range(2):
                    v0i = (v0 >> (4 * i)) & 0x0F0F0F0F
                    v1i = (v1 >> (4 * i)) & 0x0F0F0F0F
                    qb = bb * 8 + bo + i
                    aqb = nq8[qb].tobytes()
                    u0 = _u32(aqb[4 * qo:4 * qo + 4])
                    u1 = _u32(aqb[16 + 4 * qo:16 + 4 * qo + 4])
                    d8 = np.float32(da[qb])
                    dot1 = _dp4a(v0i, u0) + _dp4a(v1i, u1)
                    dot2 = _dp4a(0x01010101, u0) + _dp4a(0x01010101, u1)
                    sumf_d = np.float32(sumf_d + np.float32(d8 * np.float32(dot1 * sc[i])))
                    sumf_m = np.float32(sumf_m + np.float32(d8 * np.float32(dot2 * mm[i])))
                m0 = np.float32(dm0 * sumf_d)
                m1 = np.float32(dm1 * sumf_m)
                acc = np.float32(acc + np.float32(m0 + np.float32(-m1)))
        return float(acc)
    if fmt == "q5_k":
        assert len(b) % 176 == 0
        nb = len(b) // 176
        for bb in range(nb):
            blk = b[bb * 176:(bb + 1) * 176]
            dm = np.frombuffer(bytes(blk[0:4]), dtype=np.float16).astype(np.float32)
            dm0 = np.float32(dm[0])
            dm1 = np.float32(dm[1])
            qh = blk[16:48]
            qs = blk[48:176]
            for t in range(16):
                bo = 2 * (t // 4)
                qo = (t % 4)
                vl0 = _u32(qs[16 * bo + 4 * qo:16 * bo + 4 * qo + 4])
                vl1 = _u32(qs[16 * bo + 4 * qo + 16:16 * bo + 4 * qo + 20])
                vh0 = _u32(qh[4 * qo:4 * qo + 4]) >> bo
                vh1 = _u32(qh[4 * qo + 16:4 * qo + 20]) >> bo
                j = bo // 2
                jm = j & 1
                s0 = int(blk[4 + jm * 2]) | (int(blk[4 + jm * 2 + 1]) << 8)
                s2 = int(blk[4 + (jm + 2) * 2]) | (int(blk[4 + (jm + 2) * 2 + 1]) << 8)
                s4 = int(blk[4 + (jm + 4) * 2]) | (int(blk[4 + (jm + 4) * 2 + 1]) << 8)
                hi = 0xFFFFFFFF if j >= 2 else 0
                aux0 = (((s0 & 0x3F3F) & (~hi & 0xFFFF)) |
                        ((((s4 >> 0) & 0x0F0F) | ((s0 & 0xC0C0) >> 2)) & (hi & 0xFFFF))) & 0xFFFF
                aux1 = (((s2 & 0x3F3F) & (~hi & 0xFFFF)) |
                        ((((s4 >> 4) & 0x0F0F) | ((s2 & 0xC0C0) >> 2)) & (hi & 0xFFFF))) & 0xFFFF
                sc = [aux0 & 0xFF, (aux0 >> 8) & 0xFF]
                mm = [aux1 & 0xFF, (aux1 >> 8) & 0xFF]
                sumf_d = np.float32(0.0)
                sumf_m = np.float32(0.0)
                for i in range(2):
                    v0i = ((vl0 >> (4 * i)) & 0x0F0F0F0F) | ((((vh0 >> i) << 4) & 0x10101010))
                    v1i = ((vl1 >> (4 * i)) & 0x0F0F0F0F) | ((((vh1 >> i) << 4) & 0x10101010))
                    qb = bb * 8 + bo + i
                    aqb = nq8[qb].tobytes()
                    u0 = _u32(aqb[4 * qo:4 * qo + 4])
                    u1 = _u32(aqb[16 + 4 * qo:16 + 4 * qo + 4])
                    d8 = np.float32(da[qb])
                    dot1 = _dp4a(v0i, u0) + _dp4a(v1i, u1)
                    dot2 = _dp4a(0x01010101, u0) + _dp4a(0x01010101, u1)
                    sumf_d = np.float32(sumf_d + np.float32(d8 * np.float32(dot1 * sc[i])))
                    sumf_m = np.float32(sumf_m + np.float32(d8 * np.float32(dot2 * mm[i])))
                m0 = np.float32(dm0 * sumf_d)
                m1 = np.float32(dm1 * sumf_m)
                acc = np.float32(acc + np.float32(m0 + np.float32(-m1)))
        return float(acc)
    if fmt == "q6_k":
        assert len(b) % 210 == 0
        nb = len(b) // 210
        for bb in range(nb):
            blk = b[bb * 210:(bb + 1) * 210]
            ql = blk[0:128]
            qhb = blk[128:192]
            sc = np.frombuffer(bytes(blk[192:208]), dtype=np.int8).astype(np.int32)
            d = float(np.frombuffer(bytes(blk[208:210]), dtype=np.float16)[0])
            df = np.float32(d)
            for s8 in range(32):
                iqs = s8
                bo = 4 * (iqs // 16) + (iqs % 16) // 8
                sco = 8 * (iqs // 16) + (iqs % 16) // 4
                vsh = 2 * ((iqs % 16) // 8)
                vl = _u32(ql[4 * iqs:4 * iqs + 4])
                vh = _u32(qhb[4 * (8 * (iqs // 16) + iqs % 8):4 * (8 * (iqs // 16) + iqs % 8) + 4]) >> vsh
                sumf = np.float32(0.0)
                for i in range(2):
                    scc = int(sc[sco + 4 * i])
                    vil = (vl >> (4 * i)) & 0x0F0F0F0F
                    vih = ((vh >> (4 * i)) << 4) & 0x30303030
                    vi = 0
                    for mm_ in range(4):
                        vb = ((((vil >> (8 * mm_)) & 0xFF) + (((vih >> (8 * mm_)) & 0xFF))) - 0x20) & 0xFF
                        vi |= vb << (8 * mm_)
                    qb = bb * 8 + bo + 2 * i
                    aqb = nq8[qb].tobytes()
                    u = _u32(aqb[4 * (iqs % 8):4 * (iqs % 8) + 4])
                    d8 = np.float32(da[qb])
                    sumf = np.float32(sumf + np.float32(d8 * np.float32(_dp4a(vi, u) * scc)))
                acc = np.float32(acc + np.float32(df * sumf))
        return float(acc)
    if fmt == "iq4_xs":
        assert len(b) % 136 == 0
        nb = len(b) // 136
        for bb in range(nb):
            blk = b[bb * 136:(bb + 1) * 136]
            dw = float(np.frombuffer(bytes(blk[0:2]), dtype=np.float16)[0])
            dwf = np.float32(dw)
            sh = int(blk[2]) | (int(blk[3]) << 8)
            sl = blk[4:8]
            qs = blk[8:136]
            for sub in range(8):
                iqs = 4 * sub
                ls = ((int(sl[sub // 2]) >> (4 * (sub % 2))) & 0xF) | \
                     (((sh >> (2 * sub)) & 3) << 4)
                qb = bb * 8 + sub
                dah = np.float32(da[qb])
                aq = nq8[qb].tobytes()
                sumi = 0
                for j in range(4):
                    aux = _u32(qs[4 * (iqs + j):4 * (iqs + j) + 4])
                    vx, vy = 0, 0
                    for k in range(4):
                        vx |= (int(KVALUES_IQ4NL[(aux >> (8 * k)) & 0xF]) & 0xFF) << (8 * k)
                        vy |= (int(KVALUES_IQ4NL[(aux >> (8 * k + 4)) & 0xF]) & 0xFF) << (8 * k)
                    u0 = _u32(aq[4 * j:4 * j + 4])
                    u1 = _u32(aq[4 * (j + 4):4 * (j + 4) + 4])
                    sumi += _dp4a(vx & 0xFFFFFFFF, u0) + _dp4a(vy & 0xFFFFFFFF, u1)
                sumi *= ls - 32
                acc = np.float32(acc + np.float32(np.float32(dwf * dah) * np.float32(sumi)))
        return float(acc)
    raise ValueError(fmt)
