"""CPU proof that the parallel selection (dsa_topk_par: one row spread over nb blocks, two-pass
radix select with a per-row merge) returns exactly what the one-block kernel (dsa_topk_gfx12)
returns, ties / -inf / NaN / fewer-valid-than-k included.

Both sides are emulated from their contracts in torch, no GPU: the reference is a stable descending
sort of the fp16 order key (score descending, lowest position first among ties, ascending output,
-1 padded), the parallel form replays the four phases of dsa_topk_par.cu block for block (per-block
256-bin histograms merged into the row's bins, the descending bucket search, per-block counts, the
tie and output offsets, then the write pass). The GPU gate (scratch/qw_par_gate.py) is what proves
the kernel implements this algorithm; this file proves the algorithm is exact.
"""
import torch

K_NEG_INF = 0x03FF


def _blocks(width, nb):
    return [(width * b // nb, width * (b + 1) // nb) for b in range(nb)]


def _keys(scores):
    bits = scores.view(torch.int16).to(torch.int32) & 0xFFFF
    return torch.where(bits >= 0x8000, ~bits & 0xFFFF, bits | 0x8000)


def _desc_bucket(hist, target):
    """dsa_topk_gfx12's find_descending_bucket: highest bucket whose running count from the top
    reaches target, with the count above it; (-1, total) when the row holds fewer than target."""
    total = 0
    for bucket in range(255, -1, -1):
        if total + int(hist[bucket]) >= target:
            return bucket, total
        total += int(hist[bucket])
    return -1, total


def one_block_select(scores, k, out_w):
    """dsa_topk_gfx12's contract."""
    R, T = scores.shape
    keys = _keys(scores)
    kk = min(k, T)
    order = torch.sort(keys, dim = 1, descending = True, stable = True).indices[:, :kk]
    ok = torch.gather(keys, 1, order) > K_NEG_INF
    sel = torch.where(ok, order, T + 1).sort(dim = 1).values
    res = torch.full((R, out_w), -1, dtype = torch.int32)
    res[:, :kk] = torch.where(sel <= T - 1, sel, -1).to(torch.int32)
    return res


def par_select(scores, k, nb, out_w):
    """dsa_topk_par.cu's four phases, vectorised but block for block."""
    R, width = scores.shape
    keys = _keys(scores)
    valid = keys > K_NEG_INF
    blocks = _blocks(width, nb)
    pos = torch.arange(width)
    blk = torch.zeros(width, dtype = torch.long)
    for b, (lo, hi) in enumerate(blocks):
        blk[lo:hi] = b

    out = torch.full((R, out_w), -1, dtype = torch.int32)
    for row in range(R):
        krow = keys[row]
        vrow = valid[row]
        brow = blk

        # phase 1: coarse histogram over every block, merged per row
        coarse = torch.where(vrow, (krow >> 8) + 256 * brow, torch.full_like(brow, -1))
        hist_hi = torch.bincount(coarse[vrow], minlength = nb * 256).reshape(nb, 256).sum(0)
        hi_bucket, count_above = _desc_bucket(hist_hi, k)
        if hi_bucket < 0:
            threshold, ties_needed = K_NEG_INF, 0        # fewer valid keys than k: take them all
        else:
            # phase 2: fine histogram inside the coarse bucket
            inside = vrow & ((krow >> 8) == hi_bucket)
            fine = torch.where(inside, (krow & 0xFF) + 256 * brow, torch.full_like(brow, -1))
            hist_lo = torch.bincount(fine[inside], minlength = nb * 256).reshape(nb, 256).sum(0)
            low, below = _desc_bucket(hist_lo, k - count_above)
            threshold, ties_needed = (hi_bucket << 8) | low, k - count_above - below

        # phase 3: per-block counts, then the tie and output offsets
        is_high = vrow & (krow > threshold)
        is_tie = vrow & (krow == threshold)
        cnt_hi = torch.bincount(brow[is_high], minlength = nb)
        cnt_tie = torch.bincount(brow[is_tie], minlength = nb)
        sel_off = torch.zeros(nb, dtype = torch.long)
        tie_off = torch.zeros(nb, dtype = torch.long)
        tie_run = sel_run = 0
        for b in range(nb):
            sel_off[b] = sel_run
            tie_off[b] = tie_run
            tie_run += int(cnt_tie[b])
            sel_run += int(cnt_hi[b]) + max(0, min(int(cnt_tie[b]), ties_needed - int(tie_off[b])))
        total = sel_run

        # phase 4: admit the earliest ties of each block, then write ascending
        tie_rank = torch.cumsum(is_tie.long(), 0) - is_tie.long()   # ties before this column, row-wide
        base = torch.cat((torch.zeros(1, dtype = torch.long), torch.cumsum(is_tie.long(), 0)))
        first_of_block = torch.zeros(width, dtype = torch.long)
        for b, (lo, hi) in enumerate(blocks):
            first_of_block[lo:hi] = base[lo]
        rank_in_block = tie_rank - first_of_block                   # ties before it inside its block
        block_ties = torch.minimum(cnt_tie, torch.clamp(ties_needed - tie_off, min = 0))
        admit = is_tie & (rank_in_block < block_ties[brow])
        selected = is_high | admit
        sel_rank = torch.cumsum(selected.long(), 0) - selected.long()
        sel_base = torch.zeros(width, dtype = torch.long)
        for b, (lo, hi) in enumerate(blocks):
            sel_base[lo:hi] = int(sel_off[b])
        rank_in_block_sel = sel_rank - sel_base
        dest = sel_off[brow] + rank_in_block_sel
        out[row, dest[selected]] = pos[selected].to(torch.int32)
        assert int(selected.sum()) == total <= k <= out_w, (int(selected.sum()), total, k, out_w)
        out[row, total:] = -1
    return out


def gen_scores(R, T, levels, neg_inf_frac, gen, nan = False):
    vals = torch.tensor([0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 0.25, 2.0, -2.0, 3.0][:levels], dtype = torch.half)
    s = vals[torch.randint(0, levels, (R, T), generator = gen)]
    s[torch.rand(R, T, generator = gen) < neg_inf_frac] = float("-inf")
    if nan:
        s[torch.rand(R, T, generator = gen) < 0.01] = float("nan")
        s[torch.rand(R, T, generator = gen) < 0.01] = -float("nan")
    return s


def aligned(s):
    """The kernels take a 128-aligned row stride; pad the row so the reference emulation sees one."""
    pad = (-s.shape[1]) % 128
    return torch.cat([s, torch.full((s.shape[0], pad), float("-inf"), dtype = torch.half)], 1) \
        if pad else s


CASES = [  # levels, -inf fraction, R, T, k, out_w, nb, nan
    (3, 0.0, 4, 700, 40, 64, 4, False), (10, 0.3, 4, 1000, 64, 64, 8, False),
    (2, 0.0, 3, 513, 32, 32, 3, False), (6, 0.9, 4, 900, 50, 64, 7, False),
    (10, 0.0, 3, 2048, 128, 128, 16, False), (4, 0.99, 3, 700, 64, 64, 5, False),
    (5, 0.2, 3, 4000, 96, 128, 32, True), (2, 0.0, 2, 129, 512, 512, 128, False),
    # decode shapes: K = 512, pools 8K..128K, every block count, plus unaligned widths
    (3, 0.0, 1, 8192, 512, 512, 8, False), (3, 0.0, 1, 32768, 512, 512, 32, False),
    (3, 0.5, 1, 32769, 512, 512, 33, False), (5, 0.0, 1, 65536, 512, 512, 64, True),
    (2, 0.0, 1, 131072, 512, 512, 128, False), (3, 0.0, 1, 131072, 512, 544, 256, False),
    (3, 0.9995, 1, 131072, 512, 512, 128, False), (4, 0.2, 1, 40000, 512, 512, 100, True),
]


def test_par_equals_one_block_kernel():
    gen = torch.Generator().manual_seed(3)
    for levels, frac, R, T, k, out_w, nb, nan in CASES:
        s = aligned(gen_scores(R, T, levels, frac, gen, nan))
        ref = one_block_select(s, k, out_w)
        for blocks in {1, 2, 3, min(nb, 256), 256}:
            got = par_select(s, k, blocks, out_w)
            assert torch.equal(ref, got), (levels, frac, R, T, k, out_w, blocks)


def test_fewer_valid_than_k_is_padded():
    s = aligned(torch.full((2, 600), float("-inf"), dtype = torch.half))
    s[0, 10] = 1.0; s[0, 400] = 1.0; s[1, 599] = -3.0
    ref = one_block_select(s, 8, 32)
    for nb in (1, 3, 7, 16):
        assert torch.equal(ref, par_select(s, 8, nb, 32))
    assert ref[0].tolist() == [10, 400] + [-1] * 30
    assert ref[1].tolist() == [599] + [-1] * 31


def test_tie_storm_takes_lowest_indices_across_blocks():
    s = aligned(torch.full((1, 4096), 1.0, dtype = torch.half))   # every key equal
    ref = one_block_select(s, 512, 512)
    for nb in (1, 2, 5, 16, 64, 256):
        assert ref[0].tolist() == list(range(512))
        assert torch.equal(ref, par_select(s, 512, nb, 512))
    s[0, 4000] = 2.0                       # one winner in the last block, the rest tie below it
    s[0, 5] = 0.5                          # and one loser: the k-1 lowest tied indices win
    ref = one_block_select(s, 512, 512)
    expected = [i for i in range(4096) if i != 5][:511] + [4000]
    assert ref[0].tolist() == expected
    for nb in (1, 3, 8, 32, 128):
        assert torch.equal(ref, par_select(s, 512, nb, 512))


def test_winners_only_in_the_last_block():
    s = aligned(torch.zeros((1, 8192), dtype = torch.half))
    s[0, 7500:] = 1.0
    ref = one_block_select(s, 512, 512)
    for nb in (1, 4, 16, 64):
        assert torch.equal(ref, par_select(s, 512, nb, 512))


def test_special_values_rank_like_the_one_block_kernel():
    raw = torch.full((1, 1024), -1024, dtype = torch.int16)      # -inf
    raw[0, :300] = 0                                            # +0
    raw[0, 300] = -32768                                        # -0 ranks below +0
    raw[0, 301] = 0x7C00                                        # +inf
    raw[0, 302] = 0x7E00                                        # positive NaN ranks above +inf
    raw[0, 303] = -512                                          # negative NaN: excluded
    raw[0, 304] = -32767                                     # -0x8001: a negative denormal, valid
    s = aligned(raw.view(torch.float16))
    ref = one_block_select(s, 512, 512)
    # ascending output: the 300 +0 columns, then -0, +inf, the positive NaN and the denormal;
    # the negative NaN at 303 is excluded like -inf
    assert ref[0, :304].tolist() == list(range(303)) + [304]
    assert (ref[0, 304:] == -1).all()
    for nb in (1, 5, 32, 256):
        assert torch.equal(ref, par_select(s, 512, nb, 512))