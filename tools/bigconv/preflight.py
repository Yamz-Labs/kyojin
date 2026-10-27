"""Pre-flight checks before a conversion starts. Each check returns (name, status, message); status ok | warn | FAIL.

    python -m bigconv preflight --root WORK --spool-gb 25 --capture-min-per-layer 20 [--gpurun path --who bigconv-q0]
"""
from __future__ import annotations

import os
import shutil
import time

from .defaults import DEFAULTS
from .orchestra import idle_exempt


def _read_int(path):
    try:
        return int(open(path).read().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def check_dirty(read=_read_int):
    """vm.dirty_* above the kernel defaults: with a large memmap/spool the page cache fills with dirty pages, then one
    flush stalls every process (a raise to 90/80 hard-hung a host). Never raise them; put scratch on tmpfs instead."""
    r = read("/proc/sys/vm/dirty_ratio")
    b = read("/proc/sys/vm/dirty_background_ratio")
    if r is None or b is None:
        return ("dirty", "warn", "vm.dirty_* unreadable")
    if r > DEFAULTS["dirty_ratio_max"] or b > DEFAULTS["dirty_background_ratio_max"]:
        return ("dirty", "FAIL", f"vm.dirty_ratio={r} background={b} above defaults "
                                 f"({DEFAULTS['dirty_ratio_max']}/{DEFAULTS['dirty_background_ratio_max']}): reset them, do not tune")
    return ("dirty", "ok", f"dirty_ratio={r} background={b}")


def check_disk(path, floor_gb=None, need_gb=0.0):
    floor_gb = DEFAULTS["floor_gb"] if floor_gb is None else floor_gb
    free = shutil.disk_usage(path).free / 1e9
    if free - need_gb < floor_gb:
        return ("disk", "FAIL", f"{free:.0f} GB free - {need_gb:.0f} needed < floor {floor_gb} GB")
    return ("disk", "ok", f"{free:.0f} GB free, floor {floor_gb} GB")


def spool_budget(free_gb, spool_gb, floor_gb=None, lead=None, capture_min_per_layer=None, qlc=True, burst_gb=None):
    """Spool budget for the capture host. Returns dict(max_pending, write_min_per_layer, qlc_warning).
    max_pending = what the disk allows above the floor, capped by the lead. On a QLC SSD, sustained writes beyond the
    SLC cache (burst_gb, ~100-140 GB) fold to ~300 MB/s: if a layer's spool takes longer to write than the capture takes
    to produce it, the disk is the bottleneck, and the pending cap must stay small so the writer is not in debt."""
    floor_gb = DEFAULTS["floor_gb"] if floor_gb is None else floor_gb
    lead = DEFAULTS["lead"] if lead is None else lead
    by_disk = int((free_gb - floor_gb) / spool_gb) if spool_gb > 0 else lead
    mp = max(0, min(lead, by_disk))
    out = dict(max_pending=mp, write_min_per_layer=None, qlc_warning=None)
    if qlc:
        wmin = spool_gb * 1000.0 / DEFAULTS["qlc_sustained_write_mbs"] / 60.0
        out["write_min_per_layer"] = round(wmin, 1)
        if capture_min_per_layer is not None and wmin > capture_min_per_layer:
            out["qlc_warning"] = (f"spool write {wmin:.1f} min/layer at {DEFAULTS['qlc_sustained_write_mbs']} MB/s sustained "
                                  f"> capture {capture_min_per_layer} min/layer: disk-bound, shrink the spool or trim first")
    return out


def check_spool(free_gb, spool_gb, capture_min_per_layer=None, qlc=True, floor_gb=None):
    b = spool_budget(free_gb, spool_gb, floor_gb, capture_min_per_layer=capture_min_per_layer, qlc=qlc)
    if b["max_pending"] < DEFAULTS["min_lead"]:
        return ("spool", "FAIL", f"disk allows {b['max_pending']} pending spools < {DEFAULTS['min_lead']}: the capture would serialise")
    if b["qlc_warning"]:
        return ("spool", "warn", f"max_pending={b['max_pending']}; {b['qlc_warning']}")
    return ("spool", "ok", f"max_pending={b['max_pending']}, QLC write {b['write_min_per_layer']} min/layer")


def check_fstrim(stamp="/var/lib/systemd/timers/stamp-fstrim.timer", max_age_days=7, now=None):
    """Trim before a long write run (a full QLC drive with stale trim folds sooner). Only a hint: run `fstrim -v` yourself."""
    now = time.time() if now is None else now
    try:
        age = (now - os.path.getmtime(stamp)) / 86400
    except OSError:
        return ("fstrim", "warn", "no trim timer stamp: run `sudo fstrim -v <mount>` before the capture")
    if age > max_age_days:
        return ("fstrim", "warn", f"last timer trim {age:.0f} days ago: run `sudo fstrim -v <mount>`")
    return ("fstrim", "ok", f"trim {age:.1f} days ago")


def check_scratch(path, need_gb, mounts="/proc/mounts"):
    """Capture memmap scratch must be on tmpfs (an SSD scratch means TBs of writes into the QLC cliff) and large enough.
    GTT / unified-memory use is NOT in the sheet: on unified-memory GPUs it comes out of the same RAM, keep margin."""
    path = os.path.realpath(path)
    best = None
    try:
        for ln in open(mounts):
            dev, mp, fs = ln.split()[:3]
            if (path == mp or path.startswith(mp.rstrip("/") + "/")) and (best is None or len(mp) > len(best[0])):
                best = (mp, fs)
    except OSError:
        return ("scratch", "warn", "mount table unreadable")
    if best is None or best[1] not in ("tmpfs", "ramfs"):
        return ("scratch", "FAIL", f"{path} is on {best[1] if best else '?'}, not tmpfs: memmap scratch on the SSD writes TBs")
    free = shutil.disk_usage(path).free / 1e9
    if free < need_gb:
        return ("scratch", "FAIL", f"tmpfs has {free:.0f} GB free < memmap need {need_gb:.0f} GB")
    return ("scratch", "ok", f"tmpfs {free:.0f} GB free >= {need_gb:.0f} GB (GTT not modelled: keep RAM margin)")


def check_calib(rows, row_len):
    t = rows * row_len
    want = DEFAULTS["rows"] * DEFAULTS["row_len"]
    if t > want * 2:
        return ("calib", "warn", f"{t} tokens > 2x default {want}: capture time and spool grow; measured gain beyond default is small")
    return ("calib", "ok", f"{rows} x {row_len} = {t} tokens")


def check_gpurun_exempt(gpurun_path, who):
    try:
        text = open(gpurun_path).read()
    except OSError:
        return ("idle-exempt", "warn", f"{gpurun_path} unreadable: cannot verify the idle-kill exemption of {who}")
    if idle_exempt(who, text):
        return ("idle-exempt", "ok", f"{who} is exempt from the idle kill")
    return ("idle-exempt", "FAIL", f"{who} is NOT exempt from the GPU idle kill: quantizer CPU phases get killed; add it to the wrapper")


def run(root, spool_gb, capture_min_per_layer=None, rows=None, row_len=None, gpurun=None, who=None, qlc=True, floor_gb=None,
        scratch=None, memmap_gb=0.0, src=None, tmap=None, config=None):
    os.makedirs(root, exist_ok=True)
    free = shutil.disk_usage(root).free / 1e9
    res = [check_dirty(), check_disk(root, floor_gb, 0.0), check_spool(free, spool_gb, capture_min_per_layer, qlc, floor_gb),
           check_fstrim(), check_calib(rows or DEFAULTS["rows"], row_len or DEFAULTS["row_len"])]
    if scratch:
        res.append(check_scratch(scratch, memmap_gb))
    if src or tmap:
        from .formats import check_input_format, load_source
        res.append(check_input_format(*load_source(src, tmap, config)))
    if gpurun:
        res.append(check_gpurun_exempt(gpurun, who or DEFAULTS["quantizer_who_prefix"] + "0"))
    return res


def cli(a):
    res = run(a.root, a.spool_gb, a.capture_min_per_layer, a.rows, a.row_len, a.gpurun, a.who, not a.no_qlc, a.floor_gb,
              a.scratch, a.memmap_gb, a.src, a.map, a.config)
    for n, s, m in res:
        print(f"{s:5s} {n:12s} {m}")
    raise SystemExit(1 if any(s == "FAIL" for _, s, _ in res) else 0)
