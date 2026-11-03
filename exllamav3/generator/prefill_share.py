"""Concurrent prefill of one shared prefix. Pure python (no torch), importable without a GPU.

A page is shared by hash as soon as a job allocates it, but it only counts as cached once its K/V is complete. Jobs that start
together on one prompt prefix (an agent and the sub-agents it spawns: same system + tools + parent context) therefore each
prefill the whole prefix themselves: N-fold work, N-fold wall time, and every one of them writes the same pages. The leader is
the active job that is still prefilling the longest shared run of leading pages; a pending job waits for it to pass that run, then
starts on pages (and recurrent checkpoints) that are cached. EXL3_PF_SHARE=0 turns it off; EXL3_PF_SHARE_MIN_PAGES (default 4)
is the shortest shared run worth waiting for.
"""
from __future__ import annotations

import os


def share_enabled() -> bool:
    return os.environ.get("EXL3_PF_SHARE", "1") != "0"


def min_pages() -> int:
    return int(os.environ.get("EXL3_PF_SHARE_MIN_PAGES", "4"))


def common_pages(a: list, b: list) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def find_leader(hashes: list, prefilling: list, page: int, min_run: int | None = None):
    """The entry of `prefilling` (page_hashes, kv_position, key) that has not yet prefilled the leading pages it shares with
    `hashes`, or None. Longest shared run first, so a job waits for the one that unlocks the most."""
    need = min_pages() if min_run is None else min_run
    best, best_k = None, 0
    for ph, kv, key in prefilling:
        k = common_pages(hashes, ph)
        if k >= need and k > best_k and kv < k * page:
            best, best_k = key, k
    return best
