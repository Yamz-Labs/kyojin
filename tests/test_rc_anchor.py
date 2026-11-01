"""RecurrentCache eviction spares the shared-prefix anchors (power-of-two positions 2K..16K) while other checkpoints exist. CPU only."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from types import SimpleNamespace
from exllamav3.cache import recurrent
from exllamav3.cache.recurrent import RecurrentCache


def make(max_entries):
    model = SimpleNamespace(loaded_tp=False)
    c = RecurrentCache(model, max_size=max_entries * 100)
    return c


def st(pos):
    return {"position": pos, "checkpoint_size": 100}


def test_anchors_survive_lru():
    c = make(6)
    for i, pos in enumerate([2048, 4096, 8192, 9472, 20480, 40960]):
        c.put(("k", i), None, stashed=st(pos))
    for j in range(10):                          # a long session keeps adding prompt-end checkpoints
        c.put(("n", j), None, stashed=st(50000 + 256 * j))
    kept = sorted(v["position"] for v in c.values())
    assert 2048 in kept and 4096 in kept and 8192 in kept, kept
    assert len(kept) == 6


def test_anchors_do_not_fill_the_cache():
    c = make(4)
    for i, pos in enumerate([2048, 4096, 8192, 16384]):
        c.put(("a", i), None, stashed=st(pos))
    c.put(("n", 0), None, stashed=st(60000))
    c.put(("n", 1), None, stashed=st(60256))
    kept = sorted(v["position"] for v in c.values())
    assert kept[-2:] == [60000, 60256] or 60256 in kept, kept      # anchors beyond half the cache go by plain LRU
    assert len(kept) == 4


def test_off_is_plain_lru(monkeypatch):
    monkeypatch.setattr(recurrent, "ANCHOR_MAX_POS", 0)
    c = make(3)
    for i, pos in enumerate([2048, 4096, 8192]):
        c.put(("a", i), None, stashed=st(pos))
    c.put(("n", 0), None, stashed=st(60000))
    assert 2048 not in [v["position"] for v in c.values()]
