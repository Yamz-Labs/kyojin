import ast, os, torch

SRC = os.environ.get("QSA_SRC") or os.path.join(os.path.dirname(__file__), "..", "exllamav3", "modules", "qsa_indexer.py")


def _load():
    tree = ast.parse(open(SRC).read())
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("tile_candidates", "merge_candidates")]
    ns = {"torch": torch}
    exec(compile(ast.Module(body = keep, type_ignores = []), SRC, "exec"), ns)
    return ns["tile_candidates"], ns["merge_candidates"]


def _key(bits):
    return torch.where(bits >= 0x8000, ~bits & 0xFFFF, bits | 0x8000)


def _single_pass(scores, k):
    # the one-block-per-row kernel's contract: the k largest 16-bit order keys above the -inf key, earliest index first
    # among equal keys, emitted as ascending indices, -1 padded to the row width k
    R, T = scores.shape
    key = _key(scores.view(torch.int16).to(torch.int32) & 0xFFFF).long()
    out = torch.full((R, k), -1, dtype = torch.int32)
    for r in range(R):
        valid = (key[r] > 0x03FF).nonzero().flatten()
        order = sorted(valid.tolist(), key = lambda i: (-int(key[r, i]), i))[:k]
        sel = sorted(order)
        out[r, :len(sel)] = torch.tensor(sel, dtype = torch.int32)
    return out


def _tiled(scores, k, tile):
    tile_candidates, merge_candidates = _load()
    R, T = scores.shape
    cand = []
    for t0 in range(0, T, tile):
        sc = scores[:, t0: t0 + tile]
        local = _single_pass(sc, min(k, sc.shape[1]))
        kp = torch.full((R, k), -1, dtype = torch.int32)
        kp[:, :local.shape[1]] = local
        cand.append(tile_candidates(sc, kp, t0))
    return merge_candidates(cand, k)


def _scores(R, T, levels, neg_inf_frac, gen):
    vals = torch.tensor([0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 0.25, 2.0, -2.0, 3.0][:levels], dtype = torch.half)
    s = vals[torch.randint(0, levels, (R, T), generator = gen)]
    s[torch.rand(R, T, generator = gen) < neg_inf_frac] = float("-inf")
    return s


def test_tiled_merge_equals_single_pass():
    gen = torch.Generator().manual_seed(3)
    for levels, frac, R, T, k, tile in [(3, 0.0, 6, 700, 40, 128), (10, 0.3, 5, 1000, 64, 256), (2, 0.0, 4, 513, 32, 100),
                                        (6, 0.9, 5, 900, 50, 300), (10, 0.0, 4, 2048, 128, 512), (4, 0.99, 3, 700, 64, 200)]:
        s = _scores(R, T, levels, frac, gen)
        ref = _single_pass(s, k)
        got = _tiled(s, k, tile)
        assert torch.equal(ref, got), (levels, frac, R, T, k, tile)


def test_fewer_valid_than_k_is_padded():
    s = torch.full((2, 600), float("-inf"), dtype = torch.half)
    s[0, 10] = 1.0; s[0, 400] = 1.0; s[1, 599] = -3.0
    got = _tiled(s, 8, 128)
    assert got[0].tolist() == [10, 400] + [-1] * 6
    assert got[1].tolist() == [599] + [-1] * 7
