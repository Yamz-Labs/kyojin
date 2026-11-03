"""exllamav3/util/disk_store.py: atomic async writes, LRU byte cap, free-space floor, torn entries. CPU only."""
import os
import time

import torch

from hardware import require
require("ext")
from exllamav3.util import disk_store as ds


def mk(tmp_path, **kw):
    kw.setdefault("floor_gb", 0)
    kw.setdefault("mb_per_min", 1e9)
    return ds.DiskStore(str(tmp_path), "ns", **kw)


def blob(mb):
    return {"x": torch.zeros(int(mb * 262144), dtype=torch.float32)}


def test_roundtrip_and_index(tmp_path):
    s = mk(tmp_path, cap_gb=1)
    s.put("a", {"x": torch.arange(10)}, {"k": 1})
    s.flush()
    assert s.entries() == ["a"] and s.read_idx("a") == {"k": 1}
    assert torch.equal(s.read("a")["x"], torch.arange(10))
    assert not [f for f in os.listdir(s.dir) if f.endswith(".tmp")]
    s2 = mk(tmp_path, cap_gb=1)                                   # a restarted process sees the same entry
    assert s2.entries() == ["a"]


def test_cap_evicts_least_recently_used(tmp_path):
    s = mk(tmp_path, cap_gb=3.5 / 1024)                           # 3.5 MiB: three 1 MiB entries fit, a fourth evicts
    for n in "abc":
        s.put(n, blob(1), {"n": n})
        s.flush()
        time.sleep(0.02)
    s.touch("a")                                                  # a is now the most recent
    time.sleep(0.02)
    s.put("d", blob(1), {"n": "d"})
    s.flush()
    assert "a" in s.entries() and "d" in s.entries() and "b" not in s.entries()
    assert s.total() <= s.cap and s.stats["evicted"] >= 1


def test_floor_skips_write(tmp_path):
    s = mk(tmp_path, cap_gb=1, floor_gb=1e6)                      # an impossible floor: nothing is written
    s.put("a", blob(1), {})
    s.flush()
    assert s.entries() == [] and s.stats["skipped_floor"] == 1
    assert not [f for f in os.listdir(s.dir)]


def test_torn_payload_without_index_is_swept(tmp_path):
    s = mk(tmp_path, cap_gb=1)
    (tmp_path / "ns" / "z.pt").write_bytes(b"junk")
    (tmp_path / "ns" / "y.pt.tmp").write_bytes(b"junk")
    s2 = mk(tmp_path, cap_gb=1)
    assert s2.entries() == [] and os.listdir(s2.dir) == []


def test_existing_entry_is_not_rewritten(tmp_path):
    s = mk(tmp_path, cap_gb=1)
    s.put("a", {"x": torch.zeros(4)}, {"v": 1})
    s.flush()
    s.put("a", {"x": torch.ones(4)}, {"v": 2})
    s.flush()
    assert s.read_idx("a") == {"v": 1} and s.stats["written"] == 1


def test_close_flushes_queue(tmp_path):
    s = mk(tmp_path, cap_gb=1)
    for i in range(5):
        s.put(f"e{i}", blob(0.5), {})
    s.close()
    assert len(s.entries()) == 5


def test_cap_is_shared_by_all_namespaces(tmp_path):
    old = ds.DiskStore(str(tmp_path), "old_build", cap_gb=3.5 / 1024, floor_gb=0, mb_per_min=1e9)
    for n in "ab":
        old.put(n, blob(1), {})
        old.flush()
        time.sleep(0.02)
    new = ds.DiskStore(str(tmp_path), "new_build", cap_gb=3.5 / 1024, floor_gb=0, mb_per_min=1e9)
    for n in "cd":
        new.put(n, blob(1), {})
        new.flush()
        time.sleep(0.02)
    assert old.entries() == ["b"] and new.entries() == ["c", "d"]       # the old build's oldest entry went first
