"""Disk-first recurrent cache: cap 0 keeps one entry, evictions spill to the disk tier, spilled stashes read back. CPU only."""
import os
import types

import torch

from hardware import require
require("ext")
from exllamav3.cache.recurrent import RecurrentCache
from exllamav3.constants import PAGE_SIZE
from exllamav3.generator.prefix_disk import PagedPrefixDisk
from exllamav3.util import disk_store as ds

MIB = 1 << 20


def stash(pos, mib = 1):
    return {"checkpoint_size": mib * MIB, "position": pos, "x": torch.full((8,), float(pos))}


def cache(max_size):
    return RecurrentCache(types.SimpleNamespace(loaded_tp = False), max_size)


def test_cap_zero_keeps_only_the_entry_in_flight():
    rc = cache(0)
    rc.put(b"a", None, stashed = stash(PAGE_SIZE))
    assert list(rc) == [b"a"]
    rc.put(b"b", None, stashed = stash(2 * PAGE_SIZE))
    assert list(rc) == [b"b"] and rc.metrics["stash_evictions"] == 1


def test_one_entry_cap():
    rc = cache(MIB)
    for i, k in enumerate((b"a", b"b", b"c")):
        rc.put(k, None, stashed = stash((i + 1) * PAGE_SIZE))
    assert list(rc) == [b"c"] and rc.metrics["stash_evictions"] == 2


def test_four_gib_cap_evicts_nothing():
    rc = cache(4 * 1024**3)
    for i in range(6):
        rc.put(bytes([i]), None, stashed = stash((i + 1) * PAGE_SIZE, mib = 142))
    assert len(rc) == 6 and rc.metrics["stash_evictions"] == 0


def test_eviction_hook_gets_the_victim_and_failures_are_swallowed():
    rc = cache(0)
    seen = []
    rc.on_evict = lambda k, v: seen.append((k, v["position"]))
    rc.put(b"a", None, stashed = stash(PAGE_SIZE))
    rc.put(b"b", None, stashed = stash(2 * PAGE_SIZE))
    assert seen == [(b"a", PAGE_SIZE)]
    def boom(k, v):
        raise RuntimeError("disk gone")
    rc.on_evict = boom
    rc.put(b"c", None, stashed = stash(3 * PAGE_SIZE))      # must not raise
    assert list(rc) == [b"c"]


def disk_tier(tmp_path, rc):
    p = object.__new__(PagedPrefixDisk)
    p.ns = "ns"
    p.disk = ds.DiskStore(str(tmp_path), "ns", cap_gb = 1, floor_gb = 0, mb_per_min = 1e9)
    p.stashes = {}
    p.slots = types.SimpleNamespace(generator = types.SimpleNamespace(recurrent_cache = rc))
    return p


def test_spill_writes_aligned_anchors_and_restores_from_disk(tmp_path):
    rc = cache(0)
    p = disk_tier(tmp_path, rc)
    rc.on_evict = p.spill
    keys = {40: b"k40", 64: b"k64", 100: b"k100", 128: b"k128"}      # depth in pages
    for d in (40, 64, 100, 128):
        rc.put(keys[d], None, stashed = stash(d * PAGE_SIZE))
    rc.put(b"last", None, stashed = stash(129 * PAGE_SIZE))
    p.disk.flush()
    # depth 64 and 128 are powers of two (kept); 40 and 100 are neither pow2 nor in the pinned first 8K (32 pages): dropped
    assert set(p.stashes) >= {b"k64", b"k128"} and b"k40" not in p.stashes and b"k100" not in p.stashes
    name, depth = p.stashes[b"k64"]
    got = p.disk.read(name)["stash"]
    assert got["position"] == 64 * PAGE_SIZE and torch.equal(got["x"], torch.full((8,), float(64 * PAGE_SIZE)))
    assert p.disk.read_idx(name)["depth"] == 64


def test_spill_skips_unaligned_and_tp_and_keeps_branch_checkpoints(tmp_path):
    rc = cache(0)
    p = disk_tier(tmp_path, rc)
    p.spill(b"u", stash(PAGE_SIZE + 1))
    p.spill(b"t", {**stash(64 * PAGE_SIZE), "tp_handle": 1})
    assert not p.stashes
    rc.branch_keys = {b"br"}
    p.spill(b"br", stash(40 * PAGE_SIZE))                 # depth 40: not a power of two, kept as a branch checkpoint
    assert b"br" in p.stashes
