"""ParallelRowReader == DirectRowReader byte for byte on a scratch file (O_DIRECT when the filesystem allows it). CPU only."""
import os, pathlib, sys
import numpy as np
import pytest
import torch

from exllamav3.dsv41.engram import DirectRowReader
from exllamav3.modules.dsv41_fn.engram_io import ParallelRowReader, NativeRowReader

ROOT = pathlib.Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def tables():
    d = ROOT / "scratch" / "io"
    d.mkdir(parents=True, exist_ok=True)
    n = 100_003
    rng = np.random.default_rng(3)
    w, s = d / "w.bin", d / "s.bin"
    w.write_bytes(rng.integers(0, 256, n * 128, dtype=np.uint8).tobytes())
    s.write_bytes(rng.integers(0, 256, n * 8, dtype=np.uint8).tobytes())
    return str(w), str(s), n


@pytest.mark.parametrize("threads", [1, 4, 16])
def test_same_bytes(tables, threads):
    w, s, n = tables
    segs = [(w, 0, 128, 128), (s, 0, 8, 8)]
    ref_r = DirectRowReader(segs, allow_buffered=True)
    par_r = ParallelRowReader(segs, threads=threads, allow_buffered=True)
    rng = np.random.default_rng(threads)
    rows = rng.integers(0, n, 5000).tolist() + [0, n - 1, 31, 32, 511, 512]          # duplicates and sector edges
    a, b = ref_r.read_rows(rows), par_r.read_rows(rows)
    assert torch.equal(a, b)
    assert par_r.reads <= 2 * len(set(rows))                                           # shared sectors cost one read
    ref_r.close(); par_r.close()


@pytest.mark.parametrize("threads", [1, 8, 32])
def test_native_same_bytes(tables, threads):
    w, s, n = tables
    segs = [(w, 0, 128, 128), (s, 0, 8, 8)]
    ref_r = DirectRowReader(segs, allow_buffered=True)
    nat = NativeRowReader(segs, threads=threads, allow_buffered=True)
    rng = np.random.default_rng(100 + threads)
    rows = rng.integers(0, n, 5000).tolist() + [0, n - 1, 31, 32, 511, 512, 0]
    assert torch.equal(ref_r.read_rows(rows), nat.read_rows(rows))
    assert nat.read_rows([]).shape == (0, 136)
    ref_r.close(); nat.close()


def test_private_pools_concurrent(tables):
    """Two readers with private pools gathering at the same time from two threads == DirectRowReader byte for byte."""
    import threading
    w, s, n = tables
    segs = [(w, 0, 128, 128), (s, 0, 8, 8)]
    ref_r = DirectRowReader(segs, allow_buffered=True)
    rd = [NativeRowReader(segs, threads=t, allow_buffered=True, private_pool=True) for t in (8, 128)]
    rng = np.random.default_rng(7)
    rows = [rng.integers(0, n, 4000).tolist() + [0, n - 1] for _ in rd]
    got = [None, None]

    def go(i):
        for _ in range(5):
            got[i] = rd[i].read_rows(rows[i])
    th = [threading.Thread(target=go, args=(i,)) for i in range(2)]
    [t.start() for t in th]; [t.join() for t in th]
    for i in range(2):
        assert torch.equal(ref_r.read_rows(rows[i]), got[i])
    ref_r.close(); [r.close() for r in rd]
