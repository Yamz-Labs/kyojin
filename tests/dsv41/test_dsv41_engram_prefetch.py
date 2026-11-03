"""Async Engram prefetch: host hash == NgramHasher, prefetched rows == synchronous store.fetch byte for byte (cache hits, evictions,
chunk splits, verify-block rewrites). CPU only."""
import numpy as np
import pytest
import torch

from hardware import require
require("ext")
from exllamav3.dsv41.engram import EngramLayout, NgramHasher, compute_multipliers, EngramRowStore
from exllamav3.modules.dsv41_fn.engram_prefetch import HostEngramHash, EngramPrefetcher

V, NTOK = 5000, 129280


class HashStore(EngramRowStore):
    """Deterministic 136-byte rows from (layer, row); counts reads."""
    fmt = "mxfp4"

    def __init__(self):
        self.reads = 0

    def num_rows(self, lp):
        return 1 << 31

    def fetch(self, lp, rows):
        self.reads += rows.numel()
        r = rows.long().unsqueeze(1)
        return (((r * 2654435761 + torch.arange(136).unsqueeze(0) * 40503 + lp * 977) >> 5) & 255).to(torch.uint8)


@pytest.fixture(scope="module")
def setup():
    layers = [1, 14]
    lay = EngramLayout.build(layers, None, 4, 16_000_000, 8, 256)
    g = torch.Generator().manual_seed(5)
    tmap = torch.randint(0, V, (NTOK,), generator=g).tolist()
    return lay, tmap, compute_multipliers(layers, 4, V)


def test_host_hash_matches(setup):
    lay, tmap, mult = setup
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, NTOK, (1, 300), generator=g)
    ref = NgramHasher(lay, tmap, 2, V, mult, 1, 512, "cpu")
    hh = HostEngramHash(lay, tmap, 2, mult, 16)                                  # also exercises cache growth
    for a, b in ((0, 1), (1, 40), (40, 41), (41, 300)):
        want = ref(ids[:, a:b], a)[0].numpy()
        got = hh(ids[0, a:b].numpy(), a)
        assert np.array_equal(want, got)
    # rewrite of a verify block (rejected draft) gives the same rows as the reference
    alt = torch.randint(0, NTOK, (1, 5), generator=g)
    want = ref(alt, 296)[0].numpy()
    assert np.array_equal(want, hh(alt[0].numpy(), 296))


@pytest.mark.parametrize("cache_rows", [0, 50, 100_000])
def test_prefetch_equals_sync(setup, cache_rows):
    lay, tmap, mult = setup
    store = HashStore()
    pf = EngramPrefetcher(lay, tmap, 2, mult, store, max_seq=64, cache_bytes=cache_rows * 136, max_rows=64, slots=2)
    ref = NgramHasher(lay, tmap, 2, V, mult, 1, 1024, "cpu")
    g = torch.Generator().manual_seed(9)
    ids = torch.randint(0, 4, (1, 400), generator=g)                            # small alphabet: many repeated n-grams
    pos, tickets = 0, []
    plan = [1, 6, 11, 1, 1, 64, 5, 64, 64, 11, 1, 6]
    chunks = []
    for L in plan:
        if pos + L > 400:
            break
        chunks.append((pos, L))
        pos += L
    # two tickets in flight (next chunk submitted before the current one is checked)
    pending = pf.submit(ids[0, chunks[0][0]:chunks[0][0] + chunks[0][1]], chunks[0][0])
    for k, (s, L) in enumerate(chunks):
        nxt = pf.submit(ids[0, chunks[k + 1][0]:chunks[k + 1][0] + chunks[k + 1][1]], chunks[k + 1][0]) if k + 1 < len(chunks) else None
        rows = ref(ids[:, s:s + L], s)[0]                                         # [L, nl, nc]
        for lp in range(2):
            raw, inv = pf.take(pending, lp)
            flat = rows[:, lp].reshape(-1)
            want = store.fetch(lp, flat)
            assert torch.equal(raw[inv.long()], want)
        pf.release(pending)
        pending = nxt
    st = pf.stats()
    if cache_rows == 100_000:
        assert st["hit_rate"] > 0.3 and st["rows_read"] < st["rows_unique"]
    if cache_rows == 0:
        assert st["rows_read"] == st["rows_unique"]
    pf.close()


@pytest.mark.parametrize("hit", [True, False])
def test_speculate_exact(setup, hit):
    """Decode with speculation: a right guess reuses the ticket, a wrong one is dropped; rows stay equal to the reference either way."""
    lay, tmap, mult = setup
    store = HashStore()
    pf = EngramPrefetcher(lay, tmap, 2, mult, store, max_seq=64, cache_bytes=100_000 * 136, max_rows=64, slots=3)
    ref = NgramHasher(lay, tmap, 2, V, mult, 1, 1024, "cpu")
    g = torch.Generator().manual_seed(11)
    ids = torch.randint(0, NTOK, (1, 120), generator=g)
    ref(ids[:, :40], 0)
    for s, L in ((0, 40), ):                                                      # prefill
        j = pf.submit(ids[0, s:s + L], s)
        [pf.take(j, lp) for lp in range(2)]
        pf.release(j)
    for s in range(40, 120):
        tok = int(ids[0, s])
        pf.speculate([tok, 3, 5] if hit else [(tok + 1) % NTOK, tok, 5], s)
        j = pf.submit(ids[0, s:s + 1], s)
        rows = ref(ids[:, s:s + 1], s)[0]
        for lp in range(2):
            raw, inv = pf.take(j, lp)
            assert torch.equal(raw[inv.long()], store.fetch(lp, rows[:, lp].reshape(-1)))
        pf.release(j)
    st = pf.stats()
    assert (st["spec_hit"], st["spec_miss"]) == ((80, 0) if hit else (0, 80))
    pf.close()


@pytest.mark.parametrize("cache_rows,depth", [(100_000, 3), (3_000, 2), (0, 2)])
def test_lookahead_equals_sync(setup, cache_rows, depth):
    """Whole-prompt lookahead (rows read ahead into the cache, bounded by the cache): tickets stay byte-identical to store.fetch,
    also with a cache too small for the lookahead budget, no cache, and a start inside the prompt."""
    lay, tmap, mult = setup
    store = HashStore()
    pf = EngramPrefetcher(lay, tmap, 2, mult, store, max_seq=64, cache_bytes=cache_rows * 136, max_rows=64, slots=2)
    ref = NgramHasher(lay, tmap, 2, V, mult, 1, 1024, "cpu")
    g = torch.Generator().manual_seed(21)
    ids = torch.randint(0, NTOK, (1, 640), generator=g)
    ref(ids[:, :64], 0)
    j = pf.submit(ids[0, :64], 0)                                                    # a first chunk without lookahead
    [pf.take(j, lp) for lp in range(2)]
    pf.release(j)
    base = pf.stats()["rows_read"]
    pf.lookahead(ids[0, 64:].numpy(), 64, 64, depth=depth)
    for s in range(64, 640, 64):
        L = min(64, 640 - s)
        j = pf.submit(ids[0, s:s + L], s)
        rows = ref(ids[:, s:s + L], s)[0]
        for lp in range(2):
            raw, inv = pf.take(j, lp)
            assert torch.equal(raw[inv.long()], store.fetch(lp, rows[:, lp].reshape(-1)))
        pf.release(j)
    st = pf.stats()
    if cache_rows == 100_000:
        assert st["la_rows_read"] > 0 and st["rows_read"] == base                      # every row came from the lookahead
    pf.close()
