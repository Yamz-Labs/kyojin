#!/usr/bin/env python3
"""
pagecache.py -- evict a model directory's shards from the page cache.

Why this exists
---------------
The GB10 has one 121.6 GiB pool shared by CPU and GPU. exllamav3's `stloader`
reads shards with buffered `pread` on a `FILE*` (exllamav3_ext/stloader.cpp:541,
`fopen(filename, "rb")` + `pread`), so after loading an 87 GiB model the SAME
87 GiB is also sitting in page cache as clean pages. Nominally reclaimable -- but
`torch.cuda.mem_get_info()` reports MemFree, not MemAvailable, and the CUDA
allocator's large pinned/unified allocations do not wait politely for reclaim.
On 2026-09-22 this combination starved the machine and cost a power cycle.

`posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)` drops the *clean* pages of a file
regardless of which process read them, needs no root and no mmap ownership, and
costs a few ms per shard. Call `evict_dir(model_dir)` from the same process right
after `model.load()`.

CLI:
    python util/pagecache.py /data/hf/exl3/mimo-2.25bpw-hq          # evict
    python util/pagecache.py --report /data/hf/exl3/mimo-2.25bpw-hq # measure only

`--report` uses mincore(2) via ctypes to count resident pages per file, which is
what `vmtouch`/`fincore` do (neither is installed here).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import glob
import mmap
import os
import sys

GIB = 1024.0 ** 3
_PAGE = os.sysconf("SC_PAGE_SIZE")

_DEFAULT_PATTERNS = ("*.safetensors", "*.bin", "*.pt", "*.gguf")


def _libc():
    return ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)


def resident_bytes(path: str) -> int:
    """
    Bytes of `path` currently resident in page cache, via mmap(PROT_READ) + mincore(2)
    -- what `vmtouch`/`fincore` report (neither is installed on this box).

    The mapping is made with libc mmap directly rather than Python's `mmap` module:
    a read-only Python mmap refuses `ctypes.c_char.from_buffer`, and mapping it
    writable would dirty nothing but would need write access to the file.
    """
    size = os.path.getsize(path)
    if size == 0:
        return 0
    libc = _libc()
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_long,
    ]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]

    fd = os.open(path, os.O_RDONLY)
    addr = None
    try:
        addr = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
        if addr is None or addr == ctypes.c_void_p(-1).value:
            return -1
        npages = (size + _PAGE - 1) // _PAGE
        vec = ctypes.create_string_buffer(npages)
        if libc.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(size), vec) != 0:
            return -1
        # mincore does not fault pages in, so this measures, it does not populate.
        return sum(1 for b in vec.raw if b & 1) * _PAGE
    finally:
        if addr:
            libc.munmap(ctypes.c_void_p(addr), size)
        os.close(fd)


def files_in(model_dir: str, patterns=_DEFAULT_PATTERNS) -> list[str]:
    if os.path.isfile(model_dir):
        return [model_dir]
    out: list[str] = []
    for pat in patterns:
        out.extend(sorted(glob.glob(os.path.join(model_dir, pat))))
    return out


def evict_file(path: str) -> int:
    """posix_fadvise(DONTNEED) one file. Returns its size in bytes."""
    fd = os.open(path, os.O_RDONLY)
    try:
        # A DONTNEED on a file with dirty pages is a no-op for those pages, so sync
        # first. These are read-only model shards, so this is normally free.
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        except AttributeError:  # pragma: no cover - POSIX_FADV_DONTNEED missing
            _libc().posix_fadvise64(fd, ctypes.c_int64(0), ctypes.c_int64(0), 4)
        return os.fstat(fd).st_size
    finally:
        os.close(fd)


def evict_dir(model_dir: str, *, verbose: bool = True, report: bool = False) -> dict:
    """
    Evict every weight file under `model_dir` from the page cache.

    Returns a dict with the MemAvailable / Cached figures before and after and the
    number of bytes fadvised, so callers can log one line.
    """

    def _mi():
        d = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, _, rest = line.partition(":")
                rest = rest.strip()
                if rest.endswith(" kB"):
                    d[k] = int(rest[:-3]) * 1024
        return d

    paths = files_in(model_dir)
    before = _mi()
    res_before = sum(max(0, resident_bytes(p)) for p in paths) if report else -1

    os.sync()
    total = 0
    for p in paths:
        total += evict_file(p)

    after = _mi()
    res_after = sum(max(0, resident_bytes(p)) for p in paths) if report else -1

    info = {
        "files": len(paths),
        "bytes_fadvised": total,
        "cached_before": before.get("Cached", 0),
        "cached_after": after.get("Cached", 0),
        "avail_before": before.get("MemAvailable", 0),
        "avail_after": after.get("MemAvailable", 0),
        "resident_before": res_before,
        "resident_after": res_after,
    }
    if verbose:
        msg = (
            f" -- pagecache: fadvised DONTNEED on {len(paths)} files "
            f"({total / GIB:.1f} GiB); Cached {before.get('Cached', 0) / GIB:.1f} -> "
            f"{after.get('Cached', 0) / GIB:.1f} GiB, MemAvailable "
            f"{before.get('MemAvailable', 0) / GIB:.1f} -> {after.get('MemAvailable', 0) / GIB:.1f} GiB"
        )
        if report and res_before >= 0:
            msg += f"; shard pages resident {res_before / GIB:.1f} -> {res_after / GIB:.1f} GiB"
        print(msg, flush=True)
    return info


def main() -> int:
    args = sys.argv[1:]
    report = False
    if args and args[0] == "--report":
        report = True
        args = args[1:]
    if not args:
        print(__doc__)
        return 2
    d = args[0]
    if report:
        paths = files_in(d)
        tot = 0
        res = 0
        for p in paths:
            r = resident_bytes(p)
            s = os.path.getsize(p)
            tot += s
            res += max(0, r)
            print(f"  {os.path.basename(p):40s} {s / GIB:7.2f} GiB  resident {r / GIB:7.2f} GiB")
        print(f"  TOTAL {tot / GIB:.2f} GiB, resident {res / GIB:.2f} GiB")
    evict_dir(d, report=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
