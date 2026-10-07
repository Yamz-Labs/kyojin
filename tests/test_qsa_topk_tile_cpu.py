"""CPU proof that the tiled selection (per-tile kernel pass + one merge pass of the same kernel) equals the single pass,
tie order included. The kernel is emulated by its contract: the k largest 16-bit order keys above the -inf key, lowest
position first among equal keys, ascending output, -1 padded to the output width."""
import ast, os, torch

SRC = os.environ.get("QSA_SRC") or os.path.join(os.path.dirname(__file__), "..", "exllamav3", "modules", "qsa_indexer.py")
NAMES = ("tile_candidates", "merge_candidates", "gather_tile", "merge_tiles_topk")


def _load():
    tree = ast.parse(open(SRC).read())
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in NAMES]
    ns = {"torch": torch}
    exec(compile(ast.Module(body = keep, type_ignores = []), SRC, "exec"), ns)
    return ns


def _key(scores):
    bits = scores.view(torch.int16).to(torch.int32) & 0xFFFF
    return torch.where(bits >= 0x8000, ~bits & 0xFFFF, bits | 0x8000)


def kernel(scores, out, k, t_ptr = None, t_seq = 0):
    """dsa_topk_gfx12 contract, vectorised: stable descending sort of the key keeps the lowest position first among ties."""
    R, T = scores.shape
    assert scores.stride(1) == 1 and scores.stride(0) % 128 == 0 and out.shape[1] >= k and out.is_contiguous()
    key = _key(scores)
    order = torch.sort(key, dim = 1, descending = True, stable = True).indices[:, :k]
    ok = torch.gather(key, 1, order) > 0x03FF
    sel = torch.where(ok, order, T + 1).sort(dim = 1).values
    res = torch.full(out.shape, -1, dtype = torch.int32)
    res[:, :k] = torch.where(sel <= T - 1, sel, -1).to(torch.int32)
    out.copy_(res)


def single_pass(scores, kp, k):
    pad = (-scores.shape[1]) % 128
    s = torch.cat([scores, torch.full((scores.shape[0], pad), float("-inf"), dtype = torch.half)], 1) if pad else scores
    out = torch.empty((scores.shape[0], kp), dtype = torch.int32)
    kernel(s, out, min(k, scores.shape[1]))
    return out


def tiled(scores, kp, k, tile, ns = None):
    ns = ns or _load()
    R, T = scores.shape
    n_tiles = -(-T // tile)
    w = -(-(-(-T // n_tiles)) // 128) * 128
    W = -(-(n_tiles * kp) // 128) * 128
    cs = torch.full((R, W), float("-inf"), dtype = torch.half)
    ci = torch.full((R, W), -1, dtype = torch.int32)
    tile_idx = torch.empty((R, kp), dtype = torch.int32)
    for n, t0 in enumerate(range(0, T, w)):
        t1 = min(t0 + w, T)
        sc = torch.full((R, -(-(t1 - t0) // 128) * 128), float("-inf"), dtype = torch.half)
        sc[:, :t1 - t0] = scores[:, t0:t1]
        kernel(sc, tile_idx, min(k, t1 - t0))
        ns["gather_tile"](sc, tile_idx, t0, cs[:, n * kp:(n + 1) * kp], ci[:, n * kp:(n + 1) * kp])
    pool = torch.empty((R, kp), dtype = torch.int32)
    ns["merge_tiles_topk"](kernel, cs, ci, tile_idx, pool, k)
    return pool


def gen_scores(R, T, levels, neg_inf_frac, gen, nan = False):
    vals = torch.tensor([0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 0.25, 2.0, -2.0, 3.0][:levels], dtype = torch.half)
    s = vals[torch.randint(0, levels, (R, T), generator = gen)]
    s[torch.rand(R, T, generator = gen) < neg_inf_frac] = float("-inf")
    if nan:
        s[torch.rand(R, T, generator = gen) < 0.01] = float("nan")
        s[torch.rand(R, T, generator = gen) < 0.01] = -float("nan")
    return s


CASES = [  # levels, -inf fraction, R, T, k, tile, nan
    (3, 0.0, 4, 700, 40, 128, False), (10, 0.3, 4, 1000, 64, 256, False), (2, 0.0, 3, 513, 32, 100, False),
    (6, 0.9, 4, 900, 50, 300, False), (10, 0.0, 3, 2048, 128, 512, False), (4, 0.99, 3, 700, 64, 200, False),
    (5, 0.2, 3, 4000, 96, 1024, True),
    # model shape: K = 512, tile 32768, pools 8K..262K (padding, T not a multiple of the tile, <k valid, heavy ties)
    (3, 0.0, 2, 8192, 512, 32768, False), (3, 0.0, 2, 32768, 512, 32768, False), (3, 0.0, 2, 32769, 512, 32768, False),
    (3, 0.5, 2, 65000, 512, 32768, False), (5, 0.0, 2, 100000, 512, 32768, True), (2, 0.0, 2, 131072, 512, 32768, False),
    (3, 0.0, 1, 196609, 512, 32768, False), (3, 0.9995, 2, 262144, 512, 32768, False), (4, 0.2, 1, 262144, 512, 32768, True),
]


def test_tiled_equals_single_pass():
    ns = _load()
    gen = torch.Generator().manual_seed(3)
    for levels, frac, R, T, k, tile, nan in CASES:
        s = gen_scores(R, T, levels, frac, gen, nan)
        kp = -(-k // 32) * 32
        ref, got = single_pass(s, kp, k), tiled(s, kp, k, tile, ns)
        assert torch.equal(ref, got), (levels, frac, R, T, k, tile)


def test_fewer_valid_than_k_is_padded():
    s = torch.full((2, 600), float("-inf"), dtype = torch.half)
    s[0, 10] = 1.0; s[0, 400] = 1.0; s[1, 599] = -3.0
    got = tiled(s, 32, 8, 128)
    assert got[0].tolist() == [10, 400] + [-1] * 30
    assert got[1].tolist() == [599] + [-1] * 31


def test_ties_across_tile_border_take_lowest_index():
    s = torch.full((1, 512), 1.0, dtype = torch.half)   # every key equal: the k lowest indices win, across tiles
    got = tiled(s, 32, 32, 128)
    assert got[0].tolist() == list(range(32))
    s[0, 130] = 2.0; s[0, 5] = 0.5
    got = tiled(s, 32, 32, 128)
    assert got[0].tolist() == [0, 1, 2, 3, 4] + list(range(6, 32)) + [130]


def test_pad_slots_do_not_duplicate_position_zero():
    s = torch.full((2, 700), float("-inf"), dtype = torch.half)
    s[:, 0] = 3.0; s[:, 650] = 1.0; s[1, 300] = 2.0
    got = tiled(s, 32, 8, 128)
    assert got[0].tolist() == [0, 650] + [-1] * 30
    assert got[1].tolist() == [0, 300, 650] + [-1] * 29
