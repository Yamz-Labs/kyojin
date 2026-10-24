"""CPU proof for dsa_topk_gfx12_v2 (exllamav3_ext/dsa_topk_gfx12.cu): emulates the kernel's thread/chunk ownership,
histogram, threshold search, scans and writes in python and checks (a) every memory read the kernel would issue is inside
[0, width) of its row, (b) every write lands in [0, out_w), (c) the result equals the fp16-bit-key oracle that pins the v1
contract (all keys above the threshold + the first ties_needed ties by ascending index, ascending output, -1 padding)."""
import numpy as np
import pytest

NEG = 0x03ff


def topk_key(bits):
    bits = np.asarray(bits, dtype=np.uint32)
    return np.where(bits & 0x8000, (~bits) & 0xffff, bits | 0x8000).astype(np.uint32)


def oracle(bits, k, out_w):
    keys = topk_key(bits)
    cand = [(int(kv), i) for i, kv in enumerate(keys) if kv > NEG]
    chosen = sorted(cand, key=lambda t: (-t[0], t[1]))[:k]
    res = sorted(i for _, i in chosen)
    return res + [-1] * (out_w - len(res))


def desc_bucket(hist, target):
    total = 0
    for b in range(255, -1, -1):
        if total + hist[b] >= target:
            return b, total
        total += hist[b]
    return -1, total


def emulate(bits, width, K, out_w, THREADS, CH):
    assert width <= THREADS * CH * 8
    n_chunks = (width + 7) >> 3
    reads = []
    keys_t = []
    c0s = []
    for t in range(THREADS):
        c0 = n_chunks * t // THREADS
        c1 = n_chunks * (t + 1) // THREADS
        assert c1 - c0 <= CH
        kk = np.zeros(CH * 8, dtype=np.uint32)
        for j in range(CH):
            c = c0 + j
            base = c * 8
            if c < c1 and base + 8 <= width:
                reads.append((base, base + 8))
                kk[j * 8:(j + 1) * 8] = topk_key(bits[base:base + 8])
            elif c < c1:
                for e in range(8):
                    if base + e < width:
                        reads.append((base + e, base + e + 1))
                        kk[j * 8 + e] = topk_key(bits[base + e:base + e + 1])[0]
        keys_t.append(kk)
        c0s.append(c0)
    for lo, hi in reads:
        assert 0 <= lo < hi <= width
    allk = np.concatenate(keys_t)
    hist = np.bincount((allk[allk > NEG] >> 8).astype(int), minlength=256)
    hb, above = desc_bucket(hist, K)
    thr, ties_needed = NEG, 0
    if hb >= 0:
        sel = allk[(allk > NEG) & ((allk >> 8) == hb)]
        h2 = np.bincount((sel & 0xff).astype(int), minlength=256)
        lb, below = desc_bucket(h2, K - above)
        thr = (hb << 8) | lb
        ties_needed = K - above - below
    hc = [int(((k > thr) & (k > NEG)).sum()) for k in keys_t]
    tc = [int(((k == thr) & (k > NEG)).sum()) for k in keys_t]
    tie_off = np.concatenate(([0], np.cumsum(tc)[:-1]))
    local = [max(0, min(tc[t], ties_needed - int(tie_off[t]))) for t in range(THREADS)]
    selc = [hc[t] + local[t] for t in range(THREADS)]
    out_off = np.concatenate(([0], np.cumsum(selc)[:-1]))
    out = [None] * out_w
    for t in range(THREADS):
        o = int(out_off[t])
        seen = 0
        for i in range(CH * 8):
            kv = keys_t[t][i]
            idx = c0s[t] * 8 + i
            if kv > thr and kv > NEG:
                assert 0 <= o < out_w
                out[o] = idx; o += 1
            elif kv == thr and kv > NEG:
                if seen < local[t]:
                    assert 0 <= o < out_w
                    out[o] = idx; o += 1
                seen += 1
        if t == THREADS - 1:
            while o < out_w:
                out[o] = -1; o += 1
    assert None not in out
    return out


@pytest.mark.parametrize("THREADS,CH", [(512, 8), (1024, 4)])
@pytest.mark.parametrize("kind", ["randn", "ties", "const", "ninf", "special", "sparse"])
def test_v2_emulation_matches_oracle(THREADS, CH, kind):
    rng = np.random.default_rng(1234)
    for width in (1, 7, 8, 9, 63, 64, 65, 1000, 4095, 4097, 8192, 20001, 32768):
        for K, out_w in ((512, 512), (1, 8), (17, 32), (300, 512)):
            if kind == "randn":
                bits = rng.standard_normal(width).astype(np.float16).view(np.uint16)
            elif kind == "ties":
                bits = ((rng.integers(0, 6, width) - 2) * 0.25).astype(np.float16).view(np.uint16)
            elif kind == "const":
                bits = np.full(width, 0.5, dtype=np.float16).view(np.uint16)
            elif kind == "ninf":
                bits = np.full(width, -np.inf, dtype=np.float16).view(np.uint16)
            elif kind == "special":
                bits = rng.integers(0, 65536, width).astype(np.uint16)
            else:
                x = rng.standard_normal(width).astype(np.float16)
                x[rng.random(width) > 0.02] = -np.inf
                bits = x.view(np.uint16)
            assert emulate(bits, width, K, out_w, THREADS, CH) == oracle(bits, K, out_w), (kind, width, K, out_w)
