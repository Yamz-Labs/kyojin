#!/usr/bin/env python3
"""Characterise the eGPU link between the Strix Halo iGPU and the RX 9070.

Why: the contiguous layer split, expert-parallel split and tensor-parallel split all
pay a crossing cost that is set by the cable, not by the model.  On USB4 the 9070 is
behind an ASMedia tunnel with no usable peer path, so every crossing is host-staged;
on OCuLink the link is plain PCIe and the trade-off changes.  One tool records the
fingerprint, the latency floor, sustained staged/ring/direct throughput, the patterns
the estimator consumes, and the same evidence as machine-readable JSON so the two
cables can be compared and the model estimates use identical numbers.

    python scripts/link_bench.py measure  [options]     # default subcommand
    python scripts/link_bench.py estimate --link <json>
    python scripts/link_bench.py compare  <a.json> <b.json>

Run from the repo root after `source tools/strix_halo/env.sh` with
`EXL3_VENV=/home/m/engines/exllamav3/.venv`.  No model weights are loaded: `estimate`
reads config.json plus the safetensors headers only.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import math
import os
import platform
import re
import statistics
import struct
import subprocess
import sys
import time

import torch

torch.set_grad_enabled(False)

GIB = 1024 ** 3
MIB = 1024 ** 2
KIB = 1024

LATENCY_SIZES = [
    0, 4, 16, 64, 256, 1 * KIB, 4 * KIB, 8 * KIB, 10 * KIB, 16 * KIB,
    20 * KIB, 40 * KIB, 64 * KIB, 80 * KIB, 128 * KIB, 256 * KIB,
]
THROUGHPUT_SIZES = [1, 4, 16, 64, 128, 256, 512, 1024]
PREFILL_SIZES = [84, 320, 512, 640]
MODEL_HOP_SIZES = [8, 10, 20, 40, 64, 80]
MODEL_ROWS = [1, 8, 64, 512, 4096, 8192]

TRUNK_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
PCI_BDF_RE = re.compile(r"^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$")

BUILTIN_SPECS = {
    "glm": dict(hidden=4096, layers=45, hc=4, experts=288, topk=8, moe_inter=2048,
                dense=3, mtp=1, vision=True, vocab=154880, decode_tps=32.0,
                prefill_tps=600.0, total_bytes=106.1 * GIB, label="GLM-5.3-Flash"),
    "qwen": dict(hidden=2560, layers=48, hc=4, experts=512, topk=10, moe_inter=640,
                 dense=0, mtp=1, vision=False, vocab=248320, decode_tps=50.0,
                 prefill_tps=1500.0, total_bytes=None, label="Qwen3.8-Flash-Next"),
    "ds41": dict(hidden=5120, layers=40, hc=4, experts=384, topk=6, moe_inter=2304,
                 dense=0, mtp=3, vision=False, vocab=129280, decode_tps=7.0,
                 prefill_tps=200.0, total_bytes=172.0 * GIB, label="DeepSeek-V4.1-Flash"),
}

MODEL_GLOBS = {
    "glm": [
        "/data/hf/hub/models--NicSir--GLM-5.3-Flash-EXL3-2.72bpw/snapshots/*",
        "/data/models/NicSir--GLM-5.3-Flash-EXL3-2.72bpw",
        "/data/models/*GLM-5.3-Flash*",
        "~/models/*GLM-5.3*",
        "~/models/glm53*",
    ],
    "qwen": [
        "~/models/Qwen3.8-Flash-Next-EXL3*",
        "~/models/*Qwen3.8-Flash-Next*",
        "/data/models/*Qwen3.8-Flash-Next*",
    ],
    "ds41": [
        "~/models/*DeepSeek-V4.1*",
        "/data/models/*DeepSeek-V4.1*",
        "/data/hf/hub/models--yamz-labs--DeepSeek-V4.1-Flash-EXL3-Yamz/snapshots/*",
    ],
}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def arch_of(props) -> str:
    arch = getattr(props, "gcnArchName", None) or getattr(props, "gcn_arch_name", "") or ""
    return str(arch).lower().split(":")[0]


def devices_by_arch() -> dict[str, torch.device]:
    out: dict[str, torch.device] = {}
    for i in range(torch.cuda.device_count()):
        out[arch_of(torch.cuda.get_device_properties(i))] = torch.device("cuda", i)
    return out


def fmt_bytes(n: int | float) -> str:
    n = float(n)
    if n >= GIB:
        return f"{n / GIB:.2f} GiB"
    if n >= MIB:
        return f"{n / MIB:.1f} MiB"
    if n >= KIB:
        return f"{n / KIB:.1f} KiB"
    return f"{int(n)} B"


def us(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    return f"{seconds * 1e6:.1f}"


def gbs_str(entry: dict | None) -> str:
    return f"{entry['payload_gbs']:.2f}" if entry else "-"


def pctile(sorted_a: list[float], p: float) -> float:
    if not sorted_a:
        return float("nan")
    if len(sorted_a) == 1:
        return sorted_a[0]
    k = (len(sorted_a) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_a[int(k)]
    return sorted_a[f] * (c - k) + sorted_a[c] * (k - f)


def summary(samples: list[float]) -> dict:
    a = sorted(samples)
    return {
        "n": len(a),
        "min": a[0] if a else None,
        "max": a[-1] if a else None,
        "mean": statistics.fmean(a) if a else None,
        "median": statistics.median(a) if a else None,
        "stdev": statistics.pstdev(a) if len(a) > 1 else 0.0,
        "p10": pctile(a, 0.10) if a else None,
        "p90": pctile(a, 0.90) if a else None,
        "p95": pctile(a, 0.95) if a else None,
        "p99": pctile(a, 0.99) if a else None,
    }


def _run(cmd: list[str], timeout: float = 20.0) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (r.stdout or "") + (("\n[stderr] " + r.stderr) if r.stderr else "")
    except Exception as exc:                        # noqa: BLE001 - probe must never crash the run
        return f"[unavailable: {type(exc).__name__}: {exc}]"


def _read(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf8", errors="replace") as f:
            return f.read().strip()
    except Exception:                                # noqa: BLE001
        return None


def _link_attrs(dirpath: str) -> dict:
    out = {}
    for k in ("current_link_speed", "current_link_width", "max_link_speed", "max_link_width"):
        out[k] = _read(os.path.join(dirpath, k))
    return out


def bdf_of(index: int) -> str | None:
    p = torch.cuda.get_device_properties(index)
    dom = int(getattr(p, "pci_domain_id", 0))
    bus = int(getattr(p, "pci_bus_id", 0))
    dev = int(getattr(p, "pci_device_id", 0))
    prefix = f"{dom:04x}:{bus:02x}:{dev:02x}."
    root = "/sys/bus/pci/devices"
    if not os.path.isdir(root):
        return None
    matches = [e for e in os.listdir(root) if e.startswith(prefix)]
    if not matches:
        return None
    # Prefer function 0 (the VGA/compute function), which is the one lspci shows.
    matches.sort(key=lambda e: (not e.endswith(".0"), e))
    return matches[0]


def ancestor_links(bdf: str) -> list[dict]:
    root = os.path.realpath(os.path.join("/sys/bus/pci/devices", bdf))
    out: list[dict] = []
    cur = root
    while cur and cur != "/" and cur.startswith("/sys/"):
        base = os.path.basename(cur)
        if PCI_BDF_RE.match(base):
            item = {"bdf": base, "vendor": _read(os.path.join(cur, "vendor")),
                    "device": _read(os.path.join(cur, "device"))}
            item.update(_link_attrs(cur))
            out.append(item)
        cur = os.path.dirname(cur)
    return out


def thunderbolt_devices() -> list[dict]:
    root = "/sys/bus/thunderbolt/devices"
    out: list[dict] = []
    if not os.path.isdir(root):
        return out
    for entry in sorted(os.listdir(root)):
        d = os.path.join(root, entry)
        if not os.path.isdir(d):
            continue
        item = {"name": entry}
        for k in ("generation", "rx_speed", "tx_speed", "device_name", "vendor_name", "unique_id"):
            v = _read(os.path.join(d, k))
            if v is not None:
                item[k] = v
        if len(item) > 1:
            out.append(item)
    return out


def dmesg_tail(tag: str) -> dict:
    text = _run(["dmesg", "--ctime"], timeout=10.0)
    if (text.startswith("[unavailable") or "Operation not permitted" in text
            or "read kernel buffer failed" in text):
        return {"available": False, "tag": tag,
                "reason": "dmesg is root-restricted on this host"}
    keep = ("amdgpu", "thunderbolt", "pcieport", "AER", "xhci", "usb4")
    lines = [ln for ln in text.splitlines() if any(k.lower() in ln.lower() for k in keep)]
    return {"available": True, "tag": tag, "lines": lines[-80:]}


def git_info() -> dict:
    head = _run(["git", "rev-parse", "HEAD"]).strip()
    status = _run(["git", "status", "--porcelain"])
    return {
        "commit": head.splitlines()[0] if head and not head.startswith("[unavail") else None,
        "dirty": bool(status.strip()) if not status.startswith("[unavail") else None,
        "status_porcelain": status.splitlines()[:40],
    }


def theoretical_gbs(speed: str | None, width: int | None) -> float | None:
    if not speed or not width:
        return None
    m = re.search(r"([0-9.]+)\s*GT/s", speed)
    if not m:
        return None
    gt = float(m.group(1))
    enc = 0.8 if gt <= 5.0 else 0.9846            # Gen1/2 8b/10b, Gen3+ 128b/130b
    return gt * width * enc / 8.0


def link_ceiling(fp: dict, method: str) -> float | None:
    """Best software-visible ceiling for the whole path, in payload GB/s.

    USB4: the Thunderbolt device reports the per-lane rate (20 Gb/s on this box);
    two lanes give the 40 Gb/s tunnel.  OCuLink: the narrowest parent link that is
    at least x4 and Gen3+, which is the connector/root-port limit.
    """
    if method == "usb4":
        speeds = []
        for t in fp.get("thunderbolt", []):
            for k in ("rx_speed", "tx_speed"):
                m = re.search(r"([0-9.]+)\s*Gb/s", str(t.get(k, "")))
                if m:
                    speeds.append(float(m.group(1)))
        if speeds:
            return max(speeds) * 2.0 / 8.0
    front = next((d for d in fp.get("devices", [])
                  if str(d.get("arch", "")).startswith("gfx12")), None)
    if front is None:
        return None
    vals = []
    for a in front.get("ancestors", []):
        w = int(a.get("current_link_width") or 0) or 0
        g = theoretical_gbs(a.get("current_link_speed"), w or None)
        if w >= 4 and g and g >= 1.0:
            vals.append(g)
    if vals:
        return min(vals)
    return front.get("gpu_link_gbs")


def fingerprint(main: torch.device, front: torch.device) -> dict:
    devices = []
    for d in (main, front):
        i = d.index or 0
        p = torch.cuda.get_device_properties(i)
        bdf = bdf_of(i)
        free, total = torch.cuda.mem_get_info(i)
        link = _link_attrs(os.path.join("/sys/bus/pci/devices", bdf)) if bdf else None
        gpu_ceiling = None
        if link:
            gpu_ceiling = theoretical_gbs(link.get("current_link_speed"),
                                          int(link.get("current_link_width") or 0) or None)
        devices.append({
            "torch_index": i,
            "arch": arch_of(p),
            "name": p.name,
            "total_memory": int(p.total_memory),
            "free_memory": int(free),
            "uuid": str(getattr(p, "uuid", "")),
            "pci_bus_id": int(getattr(p, "pci_bus_id", 0)),
            "pci_device_id": int(getattr(p, "pci_device_id", 0)),
            "pci_domain_id": int(getattr(p, "pci_domain_id", 0)),
            "bdf": bdf,
            "link": link,
            "gpu_link_gbs": gpu_ceiling,
            "ancestors": ancestor_links(bdf) if bdf else [],
        })

    front_i = front.index or 0
    prev = torch.cuda.current_device()
    try:
        torch.cuda.set_device(front_i)
        front_props = torch.cuda.get_device_properties(front_i)
        p2p = getattr(front_props, "p2p", None)
    finally:
        torch.cuda.set_device(prev)

    tb = thunderbolt_devices()
    try:
        peer_01 = bool(torch.cuda.can_device_access_peer(main.index or 0, front.index or 0))
        peer_10 = bool(torch.cuda.can_device_access_peer(front.index or 0, main.index or 0))
    except Exception:                                # noqa: BLE001
        peer_01 = peer_10 = None

    method = _detect_method(tb, devices[1])
    lspci_links = {}
    for d in devices:
        if not d.get("bdf"):
            continue
        for a in [{"bdf": d["bdf"]}] + d.get("ancestors", []):
            bdf = a["bdf"]
            if bdf not in lspci_links:
                lspci_links[bdf] = _run(["lspci", "-Dnnvv", "-s", bdf], timeout=10.0)
    return {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "host": {
            "hostname": platform.node(),
            "kernel": platform.release(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "hip": getattr(torch.version, "hip", None),
            "rocm": getattr(torch.version, "rocm", None),
            "git": git_info(),
        },
        "devices": devices,
        "peer_access": {"main_to_front": peer_01, "front_to_main": peer_10,
                        "front_p2p_property": p2p},
        "thunderbolt": tb,
        "method": method,
        "lspci_links": lspci_links,
        "rocm_smi": {
            "topo": _run(["rocm-smi", "--showtopo"]),
            "bus": _run(["rocm-smi", "--showbus"]),
            "hw": _run(["rocm-smi", "--showhw"]),
        },
        "dmesg_before": dmesg_tail("before"),
    }


def _detect_method(tb: list[dict], front_dev: dict) -> str:
    """USB4 only if a Thunderbolt/USB4 bridge is actually upstream of the 9070.

    The host's own Thunderbolt routers stay enumerated when the eGPU is on OCuLink,
    so mere presence is not enough.  On this box the USB4/TB docks expose an ASMedia
    (0x1b21) or Intel JHL (0x8086) bridge in the 9070's parent chain; OCuLink is a
    plain AMD switch/root-port chain.  The TB device names are a second signal.
    """
    parents = front_dev.get("ancestors") or []
    tb_vendors = {"0x1b21", "0x8086"}
    if any(str(a.get("vendor", "")).lower() in tb_vendors for a in parents):
        return "usb4"
    host = platform.node()
    for t in tb:
        name = str(t.get("device_name", ""))
        vendor = str(t.get("vendor_name", ""))
        if name and name != host and ("ASMedia" in vendor or name.startswith("246")
                                      or "TB" in vendor or "Thunderbolt" in vendor):
            return "usb4"
    if parents and (parents[0].get("current_link_speed") or "").startswith(("16", "32")):
        return "oculink"
    return "unknown"


def link_label(fmt: dict, requested: str | None) -> tuple[str, str | None]:
    detected = fmt.get("method", "unknown")
    if not requested or requested == "auto":
        return detected, None
    warn = None
    if requested != detected:
        warn = f"--label {requested!r} contradicts fingerprint method {detected!r}"
    return requested, warn


# ---------------------------------------------------------------------------
# measurement primitives
# ---------------------------------------------------------------------------

def _sync(*devs: torch.device) -> None:
    for d in devs:
        torch.cuda.synchronize(d)


RAW_DIR: str | None = None
CURRENT_GROUP = "start"


def _emit_raw(record: dict) -> None:
    if not RAW_DIR:
        return
    os.makedirs(RAW_DIR, exist_ok=True)
    with open(os.path.join(RAW_DIR, "samples.jsonl"), "a", encoding="utf8") as f:
        f.write(json.dumps(record) + "\n")


def _verify(src: torch.Tensor, dst: torch.Tensor) -> bool:
    n = src.numel()
    if n == 0:
        return True
    if n <= 64 * MIB:
        return bool(torch.equal(src.cpu(), dst.cpu()))
    chunk = MIB
    spots = [0, max(0, n // 2 - chunk // 2), max(0, n - chunk)]
    spots.append(int(n * 0.618) // chunk * chunk)
    for s in spots:
        e = min(n, s + chunk)
        if not torch.equal(src[s:e].cpu(), dst[s:e].cpu()):
            return False
    return True


def _timed(fn, iters: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn()
    _sync_all(fn)
    out = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return out


def _sync_all(fn) -> None:
    # The closures sync their own devices; this is a belt-and-braces barrier for warmup.
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(torch.device("cuda", i))


def _make_src(dev: torch.device, n: int) -> torch.Tensor:
    if n <= 0:
        return torch.empty(0, dtype=torch.uint8, device=dev)
    return torch.randint(0, 256, (n,), dtype=torch.uint8, device=dev)


def _pinned(n: int) -> torch.Tensor:
    if n <= 0:
        return torch.empty(0, dtype=torch.uint8, device="cpu").pin_memory()
    return torch.empty(n, dtype=torch.uint8, device="cpu").pin_memory()


def _iterations(n: int) -> tuple[int, int]:
    if n <= MIB:
        return 2000, 50
    if n <= 64 * MIB:
        return 200, 10
    return 20, 5


def probe_direct(main: torch.device, front: torch.device, n: int = MIB) -> dict:
    """Run one direct D2D copy in a child process.

    A broken USB4/TB tunnel can segfault the copy (this box did, rc=139) instead of
    merely slowing it down, so the parent must never issue the probe itself.
    """
    mi = main.index or 0
    fi = front.index or 0
    code = f"""
import time
import torch
a = torch.empty({n}, dtype=torch.uint8, device={mi})
b = torch.empty({n}, dtype=torch.uint8, device={fi})
s = torch.cuda.Stream(device={fi})
for _ in range(3):
    with torch.cuda.stream(s):
        b.copy_(a, non_blocking=True)
torch.cuda.synchronize({fi})
t0 = time.perf_counter()
with torch.cuda.stream(s):
    b.copy_(a, non_blocking=True)
torch.cuda.synchronize({fi})
print({n} / (time.perf_counter() - t0) / 1e9)
"""
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, timeout=90)
    except Exception as exc:                        # noqa: BLE001
        return {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}
    gbs = None
    lines = r.stdout.strip().splitlines()
    if r.returncode == 0 and lines:
        try:
            gbs = float(lines[-1])
        except ValueError:
            gbs = None
    return {"ok": r.returncode == 0 and gbs is not None,
            "returncode": r.returncode, "payload_gbs": gbs,
            "stderr": r.stderr.strip()[-600:]}


def measure_latency_pair(src: torch.device, dst: torch.device, n: int,
                         skip_direct: bool = False) -> dict:
    iters, warmup = _iterations(n)
    src_t = _make_src(src, n)
    pin = _pinned(n)
    dst_t = torch.empty(n, dtype=torch.uint8, device=dst)
    src_rt = torch.empty(n, dtype=torch.uint8, device=src)
    pin_rt = _pinned(n)
    src_s = torch.cuda.Stream(device=src)
    dst_s = torch.cuda.Stream(device=dst)
    with torch.cuda.device(src):
        ev_s0 = torch.cuda.Event(enable_timing=True)   # recorded on src_s only
        ev_s1 = torch.cuda.Event(enable_timing=True)
    with torch.cuda.device(dst):
        ev_d0 = torch.cuda.Event(enable_timing=True)   # recorded on dst_s only
        ev_d1 = torch.cuda.Event(enable_timing=True)

    def empty():
        with torch.cuda.stream(src_s):
            ev_s0.record()
        dst_s.wait_event(ev_s0)
        torch.cuda.synchronize(dst)

    def staged():
        with torch.cuda.stream(src_s):
            pin.copy_(src_t, non_blocking=True)
            ev_s0.record()
        dst_s.wait_event(ev_s0)
        with torch.cuda.stream(dst_s):
            dst_t.copy_(pin, non_blocking=True)
        torch.cuda.synchronize(dst)

    def direct():
        with torch.cuda.stream(dst_s):
            dst_t.copy_(src_t, non_blocking=True)
        torch.cuda.synchronize(dst)

    def finalize() -> None:
        with torch.cuda.stream(src_s):
            pin.copy_(src_t, non_blocking=True)
            ev_s0.record()
        dst_s.wait_event(ev_s0)
        with torch.cuda.stream(dst_s):
            dst_t.copy_(pin, non_blocking=True)
        torch.cuda.synchronize(dst)

    def round_trip():
        # A -> B staged, then B -> A staged with the return dependent on the forward.
        with torch.cuda.stream(src_s):
            pin.copy_(src_t, non_blocking=True)
            ev_s0.record()
        dst_s.wait_event(ev_s0)
        with torch.cuda.stream(dst_s):
            dst_t.copy_(pin, non_blocking=True)
            ev_d0.record()
        src_s.wait_event(ev_d0)
        with torch.cuda.stream(src_s):
            pin_rt.copy_(dst_t, non_blocking=True)
            ev_s1.record()
        dst_s.wait_event(ev_s1)
        with torch.cuda.stream(dst_s):
            src_rt.copy_(pin_rt, non_blocking=True)
        _sync(src, dst)

    def leg_d2h():
        with torch.cuda.stream(src_s):
            ev_s0.record()
            pin.copy_(src_t, non_blocking=True)
            ev_s1.record()
        src_s.synchronize()
        return ev_s0.elapsed_time(ev_s1) / 1000.0

    def leg_h2d():
        with torch.cuda.stream(dst_s):
            ev_d0.record()
            dst_t.copy_(pin, non_blocking=True)
            ev_d1.record()
        dst_s.synchronize()
        return ev_d0.elapsed_time(ev_d1) / 1000.0

    # Warmups, then samples.  ``empty`` has no copy so reuse the same counters.
    warm_fns = [empty, staged, round_trip, leg_d2h, leg_h2d]
    if not skip_direct:
        warm_fns.append(direct)
    for fn in warm_fns:
        for _ in range(warmup):
            fn()
    _sync(src, dst)

    empty_s = _timed(empty, iters, 0)
    staged_s = _timed(staged, iters, 0)
    direct_s = [] if skip_direct else _timed(direct, iters, 0)
    rt_s = _timed(round_trip, iters, 0)

    d2h_ev: list[float] = []
    h2d_ev: list[float] = []
    d2h_host: list[float] = []
    h2d_host: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        d2h_ev.append(leg_d2h())
        d2h_host.append(time.perf_counter() - t0)
    for _ in range(iters):
        t0 = time.perf_counter()
        h2d_ev.append(leg_h2d())
        h2d_host.append(time.perf_counter() - t0)

    # Correctness of the two real paths.
    finalize()
    staged_ok = _verify(src_t, dst_t)
    if skip_direct:
        direct_ok = None
    else:
        with torch.cuda.stream(dst_s):
            dst_t.copy_(src_t, non_blocking=True)
        torch.cuda.synchronize(dst)
        direct_ok = _verify(src_t, dst_t)

    _emit_raw({"group": "latency", "bytes": n, "src": str(src), "dst": str(dst),
               "empty_s": empty_s, "staged_s": staged_s, "direct_s": direct_s,
               "round_trip_s": rt_s, "d2h_s": d2h_ev, "h2d_s": h2d_ev})

    return {
        "bytes": n,
        "empty": summary(empty_s),
        "staged": summary(staged_s),
        "direct": None if skip_direct else summary(direct_s),
        "round_trip": summary(rt_s),
        "d2h": summary(d2h_ev),
        "h2d": summary(h2d_ev),
        "d2h_host": summary(d2h_host),
        "h2d_host": summary(h2d_host),
        "staged_bit_exact": staged_ok,
        "direct_bit_exact": direct_ok,
    }


class StagedRing:
    """A depth-slot pinned ring that reproduces the bridge's stream ordering."""

    def __init__(self, src: torch.device, dst: torch.device, n: int, depth: int,
                 src_stream: torch.cuda.Stream | None = None,
                 dst_stream: torch.cuda.Stream | None = None):
        self.src, self.dst, self.n = src, dst, n
        self.depth = max(1, int(depth))
        self.src_stream = src_stream or torch.cuda.Stream(device=src)
        self.dst_stream = dst_stream or torch.cuda.Stream(device=dst)
        self.src_t = _make_src(src, n)
        self.pinned = [_pinned(n) for _ in range(self.depth)]
        self.dsts = [torch.empty(n, dtype=torch.uint8, device=dst) for _ in range(self.depth)]
        with torch.cuda.device(src):
            self.filled = [torch.cuda.Event() for _ in range(self.depth)]
        with torch.cuda.device(dst):
            self.read = [torch.cuda.Event() for _ in range(self.depth)]
        self.used = [False] * self.depth
        self.i = 0

    def enqueue(self) -> None:
        s = self.i % self.depth
        if self.used[s]:
            self.src_stream.wait_event(self.read[s])
        with torch.cuda.stream(self.src_stream):
            self.pinned[s].copy_(self.src_t, non_blocking=True)
            self.filled[s].record(self.src_stream)
        self.dst_stream.wait_event(self.filled[s])
        with torch.cuda.stream(self.dst_stream):
            self.dsts[s].copy_(self.pinned[s], non_blocking=True)
            self.read[s].record(self.dst_stream)
        self.used[s] = True
        self.i += 1

    def sync(self) -> None:
        _sync(self.src, self.dst)

    def last_ok(self) -> bool:
        return _verify(self.src_t, self.dsts[(self.i - 1) % self.depth])

    def free(self) -> None:
        self.sync()
        self.pinned.clear()
        self.dsts.clear()


class DirectRing:
    def __init__(self, src: torch.device, dst: torch.device, n: int, depth: int):
        self.src, self.dst, self.n = src, dst, n
        self.depth = max(1, int(depth))
        self.dst_stream = torch.cuda.Stream(device=dst)
        self.src_t = _make_src(src, n)
        self.dsts = [torch.empty(n, dtype=torch.uint8, device=dst) for _ in range(self.depth)]
        self.i = 0

    def enqueue(self) -> None:
        s = self.i % self.depth
        with torch.cuda.stream(self.dst_stream):
            self.dsts[s].copy_(self.src_t, non_blocking=True)
        self.i += 1

    def sync(self) -> None:
        _sync(self.src, self.dst)

    def last_ok(self) -> bool:
        return _verify(self.src_t, self.dsts[(self.i - 1) % self.depth])

    def free(self) -> None:
        self.sync()
        self.dsts.clear()


class LegRing:
    def __init__(self, src: torch.device, dst: torch.device, n: int, depth: int, leg: str):
        self.src, self.dst, self.n, self.leg = src, dst, n, leg
        self.depth = max(1, int(depth))
        self.src_t = _make_src(src, n)
        self.pinned = [_pinned(n) for _ in range(self.depth)]
        self.dsts = [torch.empty(n, dtype=torch.uint8, device=dst) for _ in range(self.depth)]
        self.stream = torch.cuda.Stream(device=src if leg == "d2h" else dst)
        self.i = 0

    def enqueue(self) -> None:
        s = self.i % self.depth
        if self.leg == "d2h":
            with torch.cuda.stream(self.stream):
                self.pinned[s].copy_(self.src_t, non_blocking=True)
        else:
            with torch.cuda.stream(self.stream):
                self.dsts[s].copy_(self.pinned[s], non_blocking=True)
        self.i += 1

    def sync(self) -> None:
        self.stream.synchronize()

    def last_ok(self) -> bool:
        s = (self.i - 1) % self.depth
        if self.leg == "d2h":
            return _verify(self.src_t, self.pinned[s])
        return _verify(self.pinned[s], self.dsts[s])

    def free(self) -> None:
        self.sync()
        self.pinned.clear()
        self.dsts.clear()


def _ring_time(make_ring, iters: int, serial: bool) -> tuple[float, str]:
    ring = None
    try:
        ring = make_ring()
        for _ in range(min(3, iters)):
            ring.enqueue()
        ring.sync()
        t0 = time.perf_counter()
        for _ in range(iters):
            ring.enqueue()
            if serial:
                ring.sync()
        ring.sync()
        dt = time.perf_counter() - t0
        ok = ring.last_ok()
        return dt / iters, ("ok" if ok else "MISMATCH")
    finally:
        if ring is not None:
            ring.free()


def _throughput_iters(n: int) -> int:
    target = 2 * GIB
    return max(4, min(200, int(target / max(n, 1))))


def _cap_depth(n: int, depth: int, dst: torch.device) -> int:
    free = torch.cuda.mem_get_info(dst.index or 0)[0]
    cap = max(1, int(free * 0.6 / max(n, 1)))
    return max(1, min(depth, cap))


def measure_throughput_pair(src: torch.device, dst: torch.device, n: int,
                            max_depth: int = 4, skip_direct: bool = False) -> dict:
    iters = _throughput_iters(n)
    depth2 = _cap_depth(n, 2, dst) if n else 1
    depth4 = _cap_depth(n, max_depth, dst) if n else 1
    out = {"bytes": n}

    t, ok = _ring_time(lambda: StagedRing(src, dst, n, 1), iters, serial=True)
    out["staged_serial"] = {"seconds": t, "payload_gbs": n / t / 1e9,
                            "link_gbs": 2 * n / t / 1e9, "verify": ok}

    t, ok = _ring_time(lambda: StagedRing(src, dst, n, depth2), iters, serial=False)
    out["staged_ring2"] = {"seconds": t, "payload_gbs": n / t / 1e9,
                           "link_gbs": 2 * n / t / 1e9, "depth": depth2, "verify": ok}

    t, ok = _ring_time(lambda: StagedRing(src, dst, n, depth4), iters, serial=False)
    out["staged_ring4"] = {"seconds": t, "payload_gbs": n / t / 1e9,
                           "link_gbs": 2 * n / t / 1e9, "depth": depth4, "verify": ok}

    if skip_direct:
        out["direct"] = None
    else:
        t, ok = _ring_time(lambda: DirectRing(src, dst, n, depth2), iters, serial=False)
        out["direct"] = {"seconds": t, "payload_gbs": n / t / 1e9,
                         "link_gbs": n / t / 1e9, "depth": depth2, "verify": ok}

    t, ok = _ring_time(lambda: LegRing(src, dst, n, depth2, "d2h"), iters, serial=False)
    out["d2h"] = {"seconds": t, "payload_gbs": n / t / 1e9, "verify": ok}

    t, ok = _ring_time(lambda: LegRing(src, dst, n, depth2, "h2d"), iters, serial=False)
    out["h2d"] = {"seconds": t, "payload_gbs": n / t / 1e9, "verify": ok}
    return out


def measure_bidirectional(main: torch.device, front: torch.device, n: int) -> dict:
    iters = max(4, min(100, int(GIB / max(n, 1))))
    fwd = StagedRing(main, front, n, 2)
    bwd = StagedRing(front, main, n, 2)
    try:
        for _ in range(3):
            fwd.enqueue()
            bwd.enqueue()
        fwd.sync()
        bwd.sync()
        t0 = time.perf_counter()
        for _ in range(iters):
            fwd.enqueue()
            bwd.enqueue()
        fwd.sync()
        bwd.sync()
        dt = (time.perf_counter() - t0) / iters
        ok_f = fwd.last_ok()
        ok_b = bwd.last_ok()
    finally:
        fwd.free()
        bwd.free()
    return {"bytes": n, "seconds": dt,
            "payload_gbs_each": n / dt / 1e9,
            "aggregate_payload_gbs": 2 * n / dt / 1e9,
            "link_gbs": 4 * n / dt / 1e9,
            "verify_fwd": ok_f, "verify_bwd": ok_b}


# ---------------------------------------------------------------------------
# pattern tests
# ---------------------------------------------------------------------------

def measure_pattern_size(main: torch.device, front: torch.device, n: int, label: str = "") -> dict:
    iters = max(4, min(60, int(GIB / max(n, 1))))
    t, ok = _ring_time(lambda: StagedRing(main, front, n, 2), iters, serial=False)
    t2, ok2 = _ring_time(lambda: StagedRing(front, main, n, 2), iters, serial=False)
    return {
        "label": label,
        "bytes": n,
        "main_to_front": {"seconds": t, "payload_gbs": n / t / 1e9,
                          "link_gbs": 2 * n / t / 1e9, "verify": ok},
        "front_to_main": {"seconds": t2, "payload_gbs": n / t2 / 1e9,
                          "link_gbs": 2 * n / t2 / 1e9, "verify": ok2},
    }


def measure_batched_small(main: torch.device, front: torch.device,
                          msize: int, k: int, reps: int = 20) -> dict:
    def packed():
        pin = _pinned(msize * k)
        srcp = _make_src(main, msize * k)
        dstp = torch.empty(msize * k, dtype=torch.uint8, device=front)
        src_s = torch.cuda.Stream(device=main)
        dst_s = torch.cuda.Stream(device=front)
        with torch.cuda.device(main):
            ev = torch.cuda.Event()
        times = []
        for r in range(reps + 3):
            t0 = time.perf_counter()
            with torch.cuda.stream(src_s):
                pin.copy_(srcp, non_blocking=True)
                ev.record()
            dst_s.wait_event(ev)
            with torch.cuda.stream(dst_s):
                dstp.copy_(pin, non_blocking=True)
            _sync(main, front)
            if r >= 3:
                times.append(time.perf_counter() - t0)
        ok = _verify(srcp, dstp)
        return summary(times), ok

    def separate():
        pins = [_pinned(msize) for _ in range(k)]
        srcs = [_make_src(main, msize) for _ in range(k)]
        dsts = [torch.empty(msize, dtype=torch.uint8, device=front) for _ in range(k)]
        src_s = torch.cuda.Stream(device=main)
        dst_s = torch.cuda.Stream(device=front)
        with torch.cuda.device(main):
            evs = [torch.cuda.Event() for _ in range(k)]
        times = []
        for r in range(reps + 3):
            t0 = time.perf_counter()
            for i in range(k):
                with torch.cuda.stream(src_s):
                    pins[i].copy_(srcs[i], non_blocking=True)
                    evs[i].record()
                dst_s.wait_event(evs[i])
                with torch.cuda.stream(dst_s):
                    dsts[i].copy_(pins[i], non_blocking=True)
            _sync(main, front)
            if r >= 3:
                times.append(time.perf_counter() - t0)
        ok = all(_verify(srcs[i], dsts[i]) for i in range(k))
        return summary(times), ok

    p, pok = packed()
    s, sok = separate()
    return {
        "msg_bytes": msize, "k": k,
        "packed": p, "packed_verify": pok,
        "separate": s, "separate_verify": sok,
        "packed_per_msg_us": (p["median"] or 0) * 1e6 / k,
        "separate_per_msg_us": (s["median"] or 0) * 1e6 / k,
        "speedup": (s["median"] / p["median"]) if p["median"] and s["median"] else None,
    }


def measure_gather_scatter(main: torch.device, front: torch.device,
                           rows: int, k: int, hidden: int = 4096) -> dict:
    src = torch.randn(rows, hidden, dtype=torch.float16, device=main)
    idx = torch.randperm(rows, device=main)[:k].to(torch.int64)
    # Tiny index, but move it via the host: ``idx.to(front)`` is a direct D2D copy,
    # which segfaults on a broken USB4 tunnel (the probe may have skipped direct).
    idx_f = idx.cpu().to(front)
    gathered = torch.empty(k, hidden, dtype=torch.float16, device=main)
    recv = torch.empty(k, hidden, dtype=torch.float16, device=front)
    dst = torch.empty(rows, hidden, dtype=torch.float16, device=front)
    src_s = torch.cuda.Stream(device=main)
    dst_s = torch.cuda.Stream(device=front)
    pin = torch.empty(k * hidden, dtype=torch.float16, device="cpu").pin_memory()
    with torch.cuda.device(main):
        ev_a = torch.cuda.Event(enable_timing=True)
        ev_b = torch.cuda.Event(enable_timing=True)
    with torch.cuda.device(front):
        ev_c = torch.cuda.Event(enable_timing=True)
        ev_d = torch.cuda.Event(enable_timing=True)

    def gather_time():
        with torch.cuda.device(main):
            st = torch.cuda.Event(enable_timing=True)
            en = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(src_s):
            st.record()
            gathered.copy_(src.index_select(0, idx))
            en.record()
        src_s.synchronize()
        return st.elapsed_time(en) / 1000.0

    def scatter_time():
        with torch.cuda.device(front):
            st = torch.cuda.Event(enable_timing=True)
            en = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(dst_s):
            st.record()
            dst.index_copy_(0, idx_f, recv)
            en.record()
        dst_s.synchronize()
        return st.elapsed_time(en) / 1000.0

    def stage_once() -> tuple[float, float]:
        with torch.cuda.stream(src_s):
            ev_a.record()
            pin.view(k, hidden).copy_(gathered, non_blocking=True)
            ev_b.record()
        src_s.synchronize()
        d2h_s = ev_a.elapsed_time(ev_b) / 1000.0
        dst_s.wait_event(ev_b)
        with torch.cuda.stream(dst_s):
            ev_c.record()
            recv.copy_(pin.view(k, hidden), non_blocking=True)
            ev_d.record()
        torch.cuda.synchronize(front)
        return d2h_s, ev_c.elapsed_time(ev_d) / 1000.0

    gather_time()
    scatter_time()
    stage_once()
    g = gather_time()
    legs = [stage_once() for _ in range(10)]
    d2h = statistics.median(d for d, _ in legs)
    h2d = statistics.median(h for _, h in legs)
    s = scatter_time()
    total = g + d2h + h2d + s
    ok = torch.equal(src.index_select(0, idx).cpu(), dst.index_select(0, idx_f).cpu())
    return {
        "rows": rows, "k": k, "hidden": hidden,
        "gather_us": g * 1e6, "d2h_us": d2h * 1e6, "h2d_us": h2d * 1e6,
        "stage_us": (d2h + h2d) * 1e6, "scatter_us": s * 1e6,
        "total_us": total * 1e6, "bytes": k * hidden * 2, "verify": ok,
    }


def measure_overlap(main: torch.device, front: torch.device, n: int = 84 * MIB,
                    h: int = 2048) -> dict:
    a_m = torch.randn(h, h, dtype=torch.float16, device=main)
    b_m = torch.randn(h, h, dtype=torch.float16, device=main)
    a_f = torch.randn(h, h, dtype=torch.float16, device=front)
    b_f = torch.randn(h, h, dtype=torch.float16, device=front)
    o_m = torch.empty(h, h, dtype=torch.float16, device=main)
    o_f = torch.empty(h, h, dtype=torch.float16, device=front)
    calc_m = torch.cuda.Stream(device=main)
    calc_f = torch.cuda.Stream(device=front)
    xfer = StagedRing(main, front, n, 2)
    reps = {"n": 1}

    def compute():
        with torch.cuda.stream(calc_m):
            for _ in range(reps["n"]):
                torch.mm(a_m, b_m, out=o_m)
        with torch.cuda.stream(calc_f):
            for _ in range(reps["n"]):
                torch.mm(a_f, b_f, out=o_f)
        _sync(main, front)

    def transfer():
        xfer.enqueue()
        xfer.sync()

    def timeit(fn, count=5):
        vals = []
        for _ in range(count):
            t0 = time.perf_counter()
            fn()
            vals.append(time.perf_counter() - t0)
        return statistics.median(vals)

    # Warm, then match the compute duration to one hop so the overlap metric is meaningful.
    compute()
    transfer()
    t_one = timeit(compute, 5)
    t_xfer = timeit(transfer, 5)
    reps["n"] = max(1, min(256, round(t_xfer / max(t_one, 1e-6))))
    t_compute = timeit(compute, 5)

    def both():
        xfer.enqueue()
        with torch.cuda.stream(calc_m):
            for _ in range(reps["n"]):
                torch.mm(a_m, b_m, out=o_m)
        with torch.cuda.stream(calc_f):
            for _ in range(reps["n"]):
                torch.mm(a_f, b_f, out=o_f)
        xfer.sync()
        _sync(main, front)

    t_both = timeit(both)
    denom = min(t_xfer, t_compute)
    overlap_eff = 1.0 - (t_both - max(t_xfer, t_compute)) / denom if denom else None
    return {
        "bytes": n, "matmul": h, "matmul_repeats": reps["n"],
        "transfer_s": t_xfer, "compute_s": t_compute, "total_s": t_both,
        "overlap_efficiency": overlap_eff,
        "serialization_ratio": t_both / (t_xfer + t_compute) if (t_xfer + t_compute) else None,
        "verify": xfer.last_ok(),
    }


def measure_stability(main: torch.device, front: torch.device, seconds: float,
                      n: int = 64 * MIB) -> dict:
    if seconds <= 0:
        return {"requested_s": 0, "skipped": True}
    fwd = StagedRing(main, front, n, 2)
    bwd = StagedRing(front, main, n, 2)
    buckets: dict[int, list[float]] = {}
    t_start = time.perf_counter()
    hops = 0
    try:
        while time.perf_counter() - t_start < seconds:
            ring = fwd if (hops % 2 == 0) else bwd
            t0 = time.perf_counter()
            ring.enqueue()
            ring.sync()
            dt = time.perf_counter() - t0
            bucket = int(time.perf_counter() - t_start)
            buckets.setdefault(bucket, []).append(n / dt / 1e9)
            hops += 1
    finally:
        fwd.free()
        bwd.free()
    med = [statistics.median(v) for _, v in sorted(buckets.items())]
    return {
        "requested_s": seconds, "hop_bytes": n, "hops": hops,
        "bucket_medians_gbs": med,
        "gbs_median": statistics.median(med) if med else None,
        "gbs_min": min(med) if med else None,
        "gbs_max": max(med) if med else None,
        "drift_gbs": (med[-1] - med[0]) if len(med) >= 2 else None,
        "dmesg_after": dmesg_tail("after"),
    }


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------

def write_outputs(result: dict, out_json: str | None, out_md: str | None) -> None:
    if out_json:
        os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
        tmp = out_json + ".tmp"
        with open(tmp, "w", encoding="utf8") as f:
            json.dump(result, f, indent=1)
        os.replace(tmp, out_json)
    if out_md:
        os.makedirs(os.path.dirname(os.path.abspath(out_md)), exist_ok=True)
        with open(out_md, "w", encoding="utf8") as f:
            f.write(render_measure_md(result))


def render_measure_md(result: dict) -> str:
    link = result.get("link", {})
    lines = [f"# eGPU link bench — {result.get('label', '?')}", ""]
    lines.append(f"Run {result.get('started', '')} / method `{link.get('method')}`")
    lines.append("")

    host = link.get("host", {})
    lines += ["## Fingerprint", "", "| Item | Value |", "| --- | --- |"]
    for k in ("hostname", "kernel", "python", "torch", "hip", "rocm"):
        lines.append(f"| {k} | {host.get(k)} |")
    g = host.get("git", {})
    lines.append(f"| git | {g.get('commit')} (dirty {g.get('dirty')}) |")
    for d in link.get("devices", []):
        lines.append(f"| cuda:{d['torch_index']} | {d['arch']} {d['name']} "
                     f"{d['total_memory'] / GIB:.2f} GiB BDF {d.get('bdf')} |")
    pa = link.get("peer_access", {})
    lines.append(f"| peer access | {pa.get('main_to_front')} / {pa.get('front_to_main')} |")
    tb = link.get("thunderbolt", [])
    if tb:
        lines.append(f"| thunderbolt | {len(tb)} devices, "
                     f"{', '.join(t.get('device_name', t['name']) for t in tb[:4])} |")
    lines.append("")

    front = next((d for d in link.get("devices", []) if str(d.get("arch", "")).startswith("gfx12")),
                 None)
    if front and front.get("ancestors"):
        lines += ["### 9070 PCIe parent chain (device upward)", "",
                  "| BDF | current | max |", "| --- | --- | --- |"]
        for a in front["ancestors"]:
            lines.append(f"| {a['bdf']} | {a.get('current_link_speed')} x"
                         f"{a.get('current_link_width')} | {a.get('max_link_speed')} x"
                         f"{a.get('max_link_width')} |")
        lines.append("")

    lat = result.get("latency", {})
    if lat:
        lines += ["## Latency", "",
                  "Medians in microseconds (host wall for the hop; same-device events for legs).",
                  ""]
        for direction, rows in lat.items():
            lines += [f"### {direction}", "",
                      "| bytes | empty | direct | staged | round-trip | d2h | h2d |",
                      "| --- | --- | --- | --- | --- | --- | --- |"]
            for r in rows:
                lines.append(
                    f"| {fmt_bytes(r['bytes'])} | {us(r['empty'].get('median'))} | "
                    f"{us((r.get('direct') or {}).get('median'))} | "
                    f"{us(r['staged'].get('median'))} | "
                    f"{us(r['round_trip'].get('median'))} | {us(r['d2h'].get('median'))} | "
                    f"{us(r['h2d'].get('median'))} |")
            lines.append("")

    thr = result.get("throughput", {})
    if thr:
        lines += ["## Throughput", "",
                  "Payload GB/s; link GB/s counts both host passes.  Direct is diagnostic only."
                  "  `util` is payload over the link ceiling (USB4: two Thunderbolt lanes;"
                  " OCuLink: narrowest x4+ parent).", ""]
        for direction, rows in thr.items():
            if direction == "bidirectional":
                continue
            lines += [f"### {direction}", "",
                      "| size | serial | ring2 | ring4 | direct | link(ring2) | util(ring2) |",
                      "| --- | --- | --- | --- | --- | --- | --- |"]
            for r in rows:
                ceil = r.get("theoretical_gbs")
                eff = (r["staged_ring2"]["payload_gbs"] / ceil * 100.0) if ceil else None
                lines.append(
                    f"| {fmt_bytes(r['bytes'])} | {r['staged_serial']['payload_gbs']:.2f} | "
                    f"{r['staged_ring2']['payload_gbs']:.2f} | "
                    f"{r['staged_ring4']['payload_gbs']:.2f} | "
                    f"{gbs_str(r.get('direct'))} | "
                    f"{r['staged_ring2']['link_gbs']:.2f} | "
                    f"{(f'{eff:.0f}%' if eff is not None else '-')} |")
            lines.append("")
        bid = thr.get("bidirectional")
        if bid:
            lines += ["### Bidirectional", "",
                      "| size | per-hop s | each GB/s | aggregate GB/s | verify |",
                      "| --- | --- | --- | --- | --- |"]
            for r in bid:
                lines.append(f"| {fmt_bytes(r['bytes'])} | {r['seconds'] * 1e3:.2f} ms | "
                             f"{r['payload_gbs_each']:.2f} | "
                             f"{r['aggregate_payload_gbs']:.2f} | "
                             f"{r['verify_fwd']}/{r['verify_bwd']} |")
            lines.append("")

    pat = result.get("patterns", {})
    if pat:
        lines += ["## Patterns", ""]
        if "model_hops" in pat:
            lines += ["### hidden-sized and prefill hops", "",
                      "| case | bytes | main->front GB/s | front->main GB/s | verify |",
                      "| --- | --- | --- | --- | --- |"]
            for r in pat["model_hops"]:
                lines.append(f"| {r.get('label', '')} | {fmt_bytes(r['bytes'])} | "
                             f"{r['main_to_front']['payload_gbs']:.2f} | "
                             f"{r['front_to_main']['payload_gbs']:.2f} | "
                             f"{r['main_to_front']['verify']}/{r['front_to_main']['verify']} |")
            lines.append("")
        if "batched_small" in pat:
            lines += ["### batched small messages", "",
                      "| msg | k | packed us/msg | separate us/msg | speedup |",
                      "| --- | --- | --- | --- | --- |"]
            for r in pat["batched_small"]:
                lines.append(f"| {fmt_bytes(r['msg_bytes'])} | {r['k']} | "
                             f"{r['packed_per_msg_us']:.2f} | {r['separate_per_msg_us']:.2f} | "
                             f"{(f'{r['speedup']:.2f}x' if r['speedup'] else '-')} |")
            lines.append("")
        if "gather_scatter" in pat:
            lines += ["### gather / stage / scatter", "",
                      "| rows | k | gather us | D2H us | H2D us | scatter us | total us | GB/s |",
                      "| --- | --- | --- | --- | --- | --- | --- | --- |"]
            for r in pat["gather_scatter"]:
                gbs = r["bytes"] / (r["total_us"] / 1e6) / 1e9
                lines.append(f"| {r['rows']} | {r['k']} | {r['gather_us']:.1f} | "
                             f"{r.get('d2h_us', 0):.1f} | {r.get('h2d_us', 0):.1f} | "
                             f"{r['scatter_us']:.1f} | "
                             f"{r['total_us']:.1f} | {gbs:.2f} |")
            lines.append("")
        if "overlap" in pat:
            r = pat["overlap"]
            lines += ["### overlap with compute", "",
                      f"transfer {r['transfer_s'] * 1e3:.2f} ms, "
                      f"compute {r['compute_s'] * 1e3:.2f} ms "
                      f"({r.get('matmul_repeats', 1)}x {r['matmul']}^2), "
                      f"total {r['total_s'] * 1e3:.2f} ms, "
                      f"efficiency {r['overlap_efficiency']:.3f}, "
                      f"serialization {r['serialization_ratio']:.3f}", ""]

    st = result.get("stability", {})
    if st and not st.get("skipped"):
        drift = (f"drift {st['drift_gbs']:+.2f} GB/s"
                 if st.get("drift_gbs") is not None else "drift unavailable")
        lines += ["## Stability", "",
                  f"{st['hops']} hops of {fmt_bytes(st['hop_bytes'])}, "
                  f"median {st['gbs_median']:.2f} GB/s, "
                  f"min {st['gbs_min']:.2f}, max {st['gbs_max']:.2f}, "
                  f"{drift}", ""]

    corr = result.get("correctness", {})
    if corr:
        lines += ["## Correctness", "",
                  f"{corr.get('staged_pass')}/{corr.get('staged_total')} staged paths bit-exact."]
        if corr.get("direct_total"):
            lines.append(f"{corr.get('direct_pass')}/{corr.get('direct_total')} direct paths "
                         f"bit-exact.")
        else:
            probe = result.get("direct_probe") or {}
            lines.append(f"Direct D2D skipped: probe rc={probe.get('returncode')} "
                         f"(staged path is unaffected).")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# measure command
# ---------------------------------------------------------------------------

def run_measure(args: argparse.Namespace) -> int:
    global RAW_DIR, CURRENT_GROUP
    RAW_DIR = args.raw_dir
    if RAW_DIR:
        os.makedirs(RAW_DIR, exist_ok=True)
        open(os.path.join(RAW_DIR, "samples.jsonl"), "w").close()
    devs = devices_by_arch()
    main = devs.get(args.main_arch)
    front = devs.get(args.front_arch)
    label = args.label or "auto"
    out_json = args.out
    out_md = args.md

    if main is None or front is None:
        print(f"link bench: missing device (have {sorted(devs)}, "
              f"need {args.main_arch} and {args.front_arch}). Is the 9070 visible?",
              flush=True)
        if main is not None:
            fp = fingerprint(main, main)
            print(f"fingerprint: method={fp['method']} devices="
                  f"{[(d['torch_index'], d['arch'], d['bdf']) for d in fp['devices']]}",
                  flush=True)
        return 2

    result: dict = {
        "schema": "kyojin-link-bench/1",
        "label": label,
        "started": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "quick": bool(args.quick),
        "max_bytes": int(args.max_bytes),
    }

    print("[link] fingerprint...", flush=True)
    fp = fingerprint(main, front)
    result["link"] = fp
    auto_label, warn = link_label(fp, args.label)
    result["label"] = auto_label
    if warn:
        print(f"[link] WARNING: {warn}", flush=True)
        result["label_warning"] = warn
    if not out_json:
        out_json = f"docs-notes/link-bench/{auto_label}.json"
    if not out_md:
        out_md = f"docs-notes/link-bench/{auto_label}.md"
    result["files"] = {"json": out_json, "md": out_md}
    write_outputs(result, out_json, out_md)
    print(f"[link] method={fp['method']} label={auto_label} -> {out_json}", flush=True)

    # Theoretical ceiling: USB4 uses the Thunderbolt lane rate, OCuLink the narrowest
    # x4+ parent link.  The raw sysfs ASMedia link (2.5 GT/s) under-reports USB4.
    front_dev = next((d for d in fp["devices"] if d["arch"] == args.front_arch), None)
    path_vals = []
    if front_dev:
        for a in front_dev.get("ancestors", []):
            w = int(a.get("current_link_width") or 0) or None
            path_vals.append(theoretical_gbs(a.get("current_link_speed"), w))
    path_vals = [v for v in path_vals if v]
    theory = link_ceiling(fp, fp.get("method", "unknown"))
    result["theoretical_gbs"] = theory
    result["sysfs_path_min_gbs"] = min(path_vals) if path_vals else None
    result["ancestor_gbs"] = path_vals
    if theory:
        print(f"[link] link ceiling {theory:.2f} GB/s (sysfs path min "
              f"{result['sysfs_path_min_gbs']})", flush=True)

    CURRENT_GROUP = "direct_probe"
    probe = probe_direct(main, front)
    skip_direct = bool(args.skip_direct) or not probe["ok"]
    result["direct_probe"] = probe
    result["skip_direct"] = skip_direct
    if skip_direct:
        print(f"[link] direct D2D probe failed (rc={probe.get('returncode')}); "
              f"using staged only", flush=True)
    else:
        print(f"[link] direct D2D probe {probe['payload_gbs']:.2f} GB/s", flush=True)

    latency: dict[str, list] = {"main_to_front": [], "front_to_main": []}
    throughput: dict[str, list] = {"main_to_front": [], "front_to_main": [], "bidirectional": []}
    patterns: dict = {}
    correctness = {"staged_pass": 0, "staged_total": 0, "direct_pass": 0, "direct_total": 0}
    result.update(latency=latency, throughput=throughput, patterns=patterns,
                  correctness=correctness)

    lat_sizes = LATENCY_SIZES if not args.quick else LATENCY_SIZES[:10]
    thr_mib = THROUGHPUT_SIZES if not args.quick else [1, 4, 16, 64, 128]
    free_front = int(torch.cuda.mem_get_info(front.index or 0)[0])
    max_bytes = min(int(args.max_bytes), max(1, free_front - 512 * MIB))

    CURRENT_GROUP = "latency"
    print("[link] latency matrix...", flush=True)
    for n in lat_sizes:
        for name, src, dst in (("main_to_front", main, front), ("front_to_main", front, main)):
            r = measure_latency_pair(src, dst, n, skip_direct=skip_direct)
            latency[name].append(r)
            correctness["staged_total"] += 1
            correctness["staged_pass"] += int(r["staged_bit_exact"])
            if not skip_direct:
                correctness["direct_total"] += 1
                correctness["direct_pass"] += int(bool(r["direct_bit_exact"]))
            if not r["staged_bit_exact"]:
                print(f"[link] MISMATCH latency {name} {n} B", flush=True)
        print(f"[link]   {fmt_bytes(n)} {us(latency['main_to_front'][-1]['staged']['median'])} us "
              f"/ {us(latency['front_to_main'][-1]['staged']['median'])} us", flush=True)
        write_outputs(result, out_json, out_md)

    CURRENT_GROUP = "throughput"
    print("[link] throughput matrix...", flush=True)
    for mib in thr_mib:
        n = mib * MIB
        if n > max_bytes:
            print(f"[link]   skip {fmt_bytes(n)} > cap {fmt_bytes(max_bytes)}", flush=True)
            continue
        for name, src, dst in (("main_to_front", main, front), ("front_to_main", front, main)):
            r = measure_throughput_pair(src, dst, n, skip_direct=skip_direct)
            r["theoretical_gbs"] = theory
            throughput[name].append(r)
            print(f"[link]   {name} {fmt_bytes(n)}: "
                  f"ring2 {r['staged_ring2']['payload_gbs']:.2f} GB/s", flush=True)
        throughput["bidirectional"].append(measure_bidirectional(main, front, n))
        write_outputs(result, out_json, out_md)
        for d in (main, front):
            with torch.cuda.device(d):
                torch.cuda.empty_cache()

    CURRENT_GROUP = "patterns"
    print("[link] patterns...", flush=True)
    hops: list[dict] = []
    for kib in MODEL_HOP_SIZES:
        hops.append(measure_pattern_size(main, front, kib * KIB, f"token stream {kib} KiB"))
    for rows in MODEL_ROWS:
        for key, spec in BUILTIN_SPECS.items():
            n = rows * spec["hidden"] * 2
            if n <= max_bytes:
                hops.append(measure_pattern_size(main, front, n, f"{key} rows={rows}"))
    for mib in ([] if args.quick else PREFILL_SIZES):
        n = mib * MIB
        if n <= max_bytes:
            hops.append(measure_pattern_size(main, front, n, f"prefill {mib} MiB"))
    patterns["model_hops"] = hops
    write_outputs(result, out_json, out_md)

    singles = [(8, 4), (8, 16), (16, 8), (32, 4), (64, 4)]
    if args.quick:
        singles = [(8, 4), (16, 8)]
    patterns["batched_small"] = [
        measure_batched_small(main, front, m * KIB, k, reps=10) for m, k in singles]
    write_outputs(result, out_json, out_md)

    gs_pairs = [(1, 1), (8, 8), (64, 64), (512, 128), (4096, 1024), (8192, 2048)]
    if args.quick:
        gs_pairs = [(1, 1), (64, 64), (4096, 1024)]
    patterns["gather_scatter"] = [
        measure_gather_scatter(main, front, rows, k) for rows, k in gs_pairs]
    write_outputs(result, out_json, out_md)

    if not args.quick:
        patterns["overlap"] = measure_overlap(main, front)
        write_outputs(result, out_json, out_md)

    CURRENT_GROUP = "soak"
    result["stability"] = measure_stability(main, front, args.soak)
    result["dmesg_after"] = dmesg_tail("after")
    result["finished"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    write_outputs(result, out_json, out_md)

    print(f"[link] wrote {out_json} and {out_md}", flush=True)
    floor = next((r for r in latency["main_to_front"] if r["bytes"] == 10 * KIB),
                 latency["main_to_front"][-1])
    floor_b = next((r for r in latency["front_to_main"] if r["bytes"] == 10 * KIB),
                   latency["front_to_main"][-1])
    print(f"[link] latency floor 10 KiB: {us(floor['staged']['median'])} us fwd / "
          f"{us(floor_b['staged']['median'])} us bwd", flush=True)
    for r in throughput["main_to_front"]:
        if r["bytes"] == 84 * MIB:
            print(f"[link] 84 MiB staged ring2: {r['staged_ring2']['payload_gbs']:.2f} GB/s",
                  flush=True)
    return 0


# ---------------------------------------------------------------------------
# estimator
# ---------------------------------------------------------------------------

def discover_packs() -> dict[str, str | None]:
    found: dict[str, str | None] = {}
    for key, globs in MODEL_GLOBS.items():
        hit = None
        for pattern in globs:
            pattern = os.path.expanduser(pattern)
            for path in sorted(glob.glob(pattern), reverse=True):
                if os.path.isfile(os.path.join(path, "config.json")):
                    hit = path
                    break
            if hit:
                break
        found[key] = hit
    return found


def load_config(pack: str) -> dict:
    with open(os.path.join(pack, "config.json"), encoding="utf8") as f:
        return json.load(f)


def spec_from_config(key: str, pack: str | None) -> dict:
    base = dict(BUILTIN_SPECS[key])
    base["pack"] = pack
    base["header_bytes"] = False
    base["source"] = "builtin"
    if pack is None:
        return base

    cfg = load_config(pack)
    tc = cfg.get("text_config", cfg) or cfg
    base["arch"] = cfg.get("architectures", [None])[0]
    base["hidden"] = int(tc.get("hidden_size") or base["hidden"])
    base["layers"] = int(tc.get("num_hidden_layers") or base["layers"])
    base["hc"] = int(tc.get("hc_mult") or tc.get("hc_count") or base["hc"])
    experts = tc.get("n_routed_experts") or tc.get("num_experts") or base["experts"]
    base["experts"] = int(experts)
    base["topk"] = int(tc.get("num_experts_per_tok") or base["topk"])
    base["moe_inter"] = int(tc.get("moe_intermediate_size") or base["moe_inter"])
    mtp = (tc.get("num_nextn_predict_layers") or tc.get("mtp_num_hidden_layers")
           or base["mtp"])
    base["mtp"] = int(mtp)
    base["vocab"] = int(tc.get("vocab_size") or base["vocab"])
    base["vision"] = "vision_config" in cfg or "vision_config" in tc
    if "first_k_dense_replace" in tc:
        base["dense"] = int(tc["first_k_dense_replace"])
    mlt = tc.get("mlp_layer_types")
    if isinstance(mlt, list) and mlt:
        base["dense"] = sum(1 for x in mlt if "dense" in str(x).lower())
    qc = cfg.get("quantization_config") or tc.get("quantization_config") or {}
    base["bits"] = qc.get("bits")
    if tc.get("ple_layer_ids"):
        base["external_tables_bytes"] = 24.4 * GIB
        base["external_tables_note"] = "PLE / n-gram table (not in the trunk index)"
    if tc.get("engram_layer_ids"):
        base["external_tables_bytes"] = 190e9
        base["external_tables_note"] = "Engram tables (not in the trunk index)"
    base["source"] = "config+headers"

    index = os.path.join(pack, "model.safetensors.index.json")
    if not os.path.isfile(index):
        return base

    scan = scan_pack_headers(pack, index, base["layers"])
    base.update(scan)
    base["header_bytes"] = True
    base["source"] = "headers"
    return base


def scan_pack_headers(pack: str, index_path: str, num_layers: int) -> dict:
    with open(index_path, encoding="utf8") as f:
        index = json.load(f)
    shards = sorted(set(index["weight_map"].values()))
    layer_bytes = [0] * num_layers
    layer_expert = [0] * num_layers
    endcap = 0
    cpu = 0
    seen = set()
    total = 0
    per_layer_dense_hit = [False] * num_layers

    for shard in shards:
        path = os.path.join(pack, shard)
        if not os.path.isfile(path):
            continue
        try:
            header = read_st_header(path)
        except Exception:                            # noqa: BLE001 - one bad shard must not lose the rest
            continue
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            b, e = meta["data_offsets"]
            length = e - b
            total += length
            m = TRUNK_LAYER_RE.search(name)
            if m and int(m.group(1)) < num_layers:
                idx = int(m.group(1))
                layer_bytes[idx] += length
                if ".mlp.experts." in name or re.search(r"\.experts\.\d+\.", name):
                    layer_expert[idx] += length
                elif ".mlp." in name:
                    per_layer_dense_hit[idx] = True
            elif "embed_tokens" in name:
                cpu += length
            else:
                endcap += length
            seen.add(name)

    return {
        "total_bytes": total,
        "layer_bytes": layer_bytes,
        "layer_expert_bytes": layer_expert,
        "layer_dense_hit": per_layer_dense_hit,
        "endcap_bytes": endcap,
        "cpu_bytes": cpu,
        "tensor_count": len(seen),
        "shards": len(shards),
    }


def read_st_header(path: str) -> dict:
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) < 8:
            raise ValueError(f"{path}: short header")
        n = struct.unpack("<Q", raw)[0]
        if n <= 0 or n > 512 * MIB:
            raise ValueError(f"{path}: implausible header length {n}")
        return json.loads(f.read(n))


def estimate_spec(key: str, spec: dict) -> dict:
    layers = spec["layers"]
    if not spec.get("layer_bytes"):
        tot = spec.get("total_bytes")
        if tot:
            uniform = int(tot / max(layers, 1))
            spec["layer_bytes"] = [uniform] * layers
            experts_total = uniform * layers * (layers - spec["dense"]) / max(layers, 1)
            per = int(experts_total / max(layers - spec["dense"], 1))
            spec["layer_expert_bytes"] = [0 if i < spec["dense"] else per for i in range(layers)]
            spec["endcap_bytes"] = 0
            spec["cpu_bytes"] = 0
            spec["approximated"] = True
        else:
            spec["layer_bytes"] = [0] * layers
            spec["layer_expert_bytes"] = [0] * layers
            spec["endcap_bytes"] = spec.get("total_bytes") or 0
            spec["cpu_bytes"] = 0
            spec["approximated"] = True
            spec["size_unknown"] = True
    if not spec.get("layer_dense_hit"):
        spec["layer_dense_hit"] = [i < spec["dense"] for i in range(layers)]
    return spec


def _nearest(rows: list[dict], n: int, key: str) -> dict | None:
    if not rows:
        return None
    return min(rows, key=lambda r: abs(r["bytes"] - n))


def _interp_bw(rows: list[dict], n: int, key: str) -> tuple[float, str]:
    pts = []
    for r in rows:
        v = r.get(key)
        if v and v.get("payload_gbs"):
            pts.append((r["bytes"], v["payload_gbs"]))
    pts.sort()
    if not pts:
        return 0.0, "unavailable"
    if len(pts) == 1:
        return pts[0][1], f"single point {fmt_bytes(pts[0][0])}"
    if n <= pts[0][0]:
        return pts[0][1], f"clamped low {fmt_bytes(pts[0][0])}"
    if n >= pts[-1][0]:
        return pts[-1][1], f"clamped high {fmt_bytes(pts[-1][0])}"
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= n <= x1:
            t = (math.log(max(n, 1)) - math.log(max(x0, 1))) / (
                math.log(max(x1, 1)) - math.log(max(x0, 1)))
            return math.exp(math.log(max(y0, 1e-12)) * (1 - t) + math.log(max(y1, 1e-12)) * t), \
                f"log-interp {fmt_bytes(x0)}..{fmt_bytes(x1)}"
    return pts[-1][1], "fallback"


def hop_cost(link: dict, direction: str, n: int) -> tuple[float, dict]:
    lat_rows = link.get("latency", {}).get(direction, [])
    thr_rows = link.get("throughput", {}).get(direction, [])
    row = _nearest(lat_rows, n, "staged")
    if row and n <= 256 * KIB:
        latency_s = row["staged"]["median"]
        lat_note = fmt_bytes(row["bytes"])
    elif row:
        latency_s = row["staged"]["median"]
        lat_note = f"{fmt_bytes(row['bytes'])} (floor)"
    else:
        latency_s = 0.0
        lat_note = "none"
    bw, bw_note = _interp_bw(thr_rows, n, "staged_ring2")
    transfer_s = n / (bw * 1e9) if bw else 0.0
    return latency_s + transfer_s, {"bytes": n, "latency_s": latency_s,
                                    "latency_source": lat_note, "bw_gbs": bw,
                                    "bw_source": bw_note, "transfer_s": transfer_s}


def estimate_layer_split(key: str, spec: dict, link: dict, args) -> dict:
    if spec.get("size_unknown"):
        return {"method": "layer_split", "size_unknown": True,
                "activation_bytes_per_token": 0, "activation_note": "unknown",
                "front_total": 0, "front_free": 0, "front_budget": 0,
                "front_budget_shortctx": 0, "endcap_bytes": 0, "cpu_bytes": 0,
                "best_N": None, "best_N_shortctx": None, "sweep": [],
                "decode": None, "prefill": []}
    front = _front_device(link)
    if front is None:
        front_total = 15.92 * GIB
        free = 15.27 * GIB
    else:
        front_total = int(front["total_memory"])
        free = int(front.get("free_memory") or front_total)
    budget = max(0.0, front_total - args.front_reserve_gb * GIB - args.kv_workspace_gb * GIB)

    layers = spec["layers"]
    act = spec["hc"] * spec["hidden"] * 4
    act_note = f"{spec['hc']}x fp32"
    if not act:
        act = spec["hidden"] * 2
        act_note = "hidden fp16 fallback"

    endcap = int(spec.get("endcap_bytes") or 0)
    short_budget = max(0.0, front_total - args.front_reserve_gb * GIB)
    sweep = []
    best = None
    best_short = None
    for N in range(layers + 1):
        fb = sum(spec["layer_bytes"][:N]) + (endcap if args.endcaps == "on" else 0)
        fits = fb <= budget
        fits_short = fb <= short_budget
        sweep.append({"N": N, "front_bytes": fb, "fits": fits,
                      "fits_shortctx": fits_short})
        if fits:
            best = N
        if fits_short:
            best_short = N

    crossings = 2 + max(0, args.draft_steps)
    dec_size = act
    pre_sizes = [int(x) for x in str(args.prefill_tokens).split(",") if x.strip()]
    dec_cost, dec_note = hop_cost(link, "front_to_main", dec_size)
    dec_cost2, dec_note2 = hop_cost(link, "main_to_front", dec_size)
    dec_ms = 2 * (dec_cost + dec_cost2) / 2 * 1e3 * (crossings / 2.0)
    dec_base_ms = 1000.0 / spec["decode_tps"]
    prefill = []
    for T in pre_sizes:
        size = T * act
        c1, n1 = hop_cost(link, "front_to_main", size)
        c2, n2 = hop_cost(link, "main_to_front", size)
        t_s = 2 * (c1 + c2) / 2
        base_s = T / spec["prefill_tps"]
        prefill.append({"tokens": T, "bytes": size, "added_s": t_s,
                        "added_ms": t_s * 1e3, "baseline_s": base_s,
                        "pct": 100.0 * t_s / base_s if base_s else None,
                        "note": n1})

    return {
        "method": "layer_split",
        "activation_bytes_per_token": act,
        "activation_note": act_note,
        "front_total": front_total, "front_free": free, "front_budget": budget,
        "front_budget_shortctx": short_budget,
        "endcap_bytes": endcap, "cpu_bytes": int(spec.get("cpu_bytes") or 0),
        "best_N": best, "best_N_shortctx": best_short,
        "sweep": sweep,
        "decode": {"bytes_per_crossing": dec_size, "crossings": crossings,
                   "added_ms_per_token": dec_ms, "baseline_ms": dec_base_ms,
                   "pct_baseline": 100.0 * dec_ms / dec_base_ms if dec_base_ms else None,
                   "crossing_cost_ms": (dec_cost + dec_cost2) / 2 * 1e3,
                   "note": dec_note},
        "prefill": prefill,
    }


def estimate_expert_split(key: str, spec: dict, link: dict, args) -> dict:
    if spec.get("size_unknown"):
        return {"method": "expert_split", "size_unknown": True,
                "expert_total_bytes": 0, "expert_estimated": False,
                "capacity_fraction_front": None, "f_remote": None,
                "moe_layers": 0, "dense_layers": spec["dense"],
                "dense_byte_fraction": None, "activation_row_bytes": None,
                "decode_uniform": None, "decode_skew": None, "prefill_8k": None,
                "prefill_verdict": "size unknown", "verdict": "size unknown"}
    front = _front_device(link)
    free = int(front.get("free_memory") or 16 * GIB) if front else int(16 * GIB)
    budget = max(0.0, free - args.front_reserve_gb * GIB)

    expert_total = sum(spec.get("layer_expert_bytes") or [])
    if expert_total <= 0 and spec.get("total_bytes"):
        moe_layers = spec["layers"] - spec["dense"]
        bits = float(spec.get("bits") or 2.5)
        expert_total = int(moe_layers * spec["experts"] * 3 * spec["moe_inter"]
                            * spec["hidden"] * bits / 8)
        spec["expert_estimated"] = True
    dense_bytes = sum(spec["layer_bytes"][i] for i in range(min(spec["dense"], spec["layers"])))
    total_bytes = sum(spec["layer_bytes"]) or spec.get("total_bytes") or 1

    f_front = min(1.0, budget / expert_total) if expert_total else 0.0
    f_remote = max(0.0, 1.0 - f_front)
    moe_layers = spec["layers"] - spec["dense"]
    hidden = spec["hidden"]
    act_row = hidden * 2                         # collapsed hidden at the MoE site

    def run(tokens: int, skew: float) -> dict:
        remote = min(tokens * spec["topk"], tokens * spec["topk"] * f_remote * skew)
        per_layer_bytes = 2 * remote * act_row   # dispatch + combine
        cost, info = hop_cost(link, "front_to_main", max(int(per_layer_bytes), 1))
        latency_s = info["latency_s"]
        # hop_cost returns transfer_s = 0 when the throughput table has no usable
        # rows (bw_source "unavailable"); use it instead of dividing by bw_gbs so a
        # partial measure JSON still produces a latency-only estimate.
        transfer_s = info["transfer_s"]
        per_layer_s = 2 * latency_s + transfer_s
        total_s = moe_layers * per_layer_s
        return {"tokens": tokens, "f_remote": f_remote, "skew": skew,
                "remote_assignments_per_layer": remote,
                "bytes_per_layer": per_layer_bytes,
                "latency_s_per_layer": 2 * latency_s,
                "transfer_s_per_layer": transfer_s,
                "per_layer_s": per_layer_s, "total_s": total_s,
                "latency_bound": (2 * latency_s * moe_layers) / total_s
                if total_s else None,
                "note": info["bw_source"]}

    dec = run(1, 1.0)
    dec_skew = run(1, args.expert_skew)
    pre = run(8192, 1.0)
    lat_share = dec["latency_bound"] or 0.0
    if f_front < 0.05:
        verdict = "capacity-limited"
    elif lat_share > 0.7:
        verdict = "latency-bound"
    else:
        verdict = "bandwidth-bound"
    pre_lat = pre["latency_bound"] or 0.0
    prefill_verdict = "latency-bound" if pre_lat > 0.5 else "bandwidth-bound"
    extra = {}
    if key == "ds41":
        link_bw, _ = _interp_bw(link.get("throughput", {}).get("front_to_main", []),
                                256 * MIB, "staged_ring2")
        extra = {"ssd_tier_gbs": 3.0, "link_over_ssd": link_bw / 3.0}
    return {
        "method": "expert_split",
        "expert_total_bytes": expert_total,
        "expert_estimated": bool(spec.get("expert_estimated")),
        "capacity_fraction_front": f_front,
        "f_remote": f_remote,
        "moe_layers": moe_layers,
        "dense_layers": spec["dense"],
        "dense_byte_fraction": dense_bytes / total_bytes if total_bytes else None,
        "activation_row_bytes": act_row,
        "decode_uniform": dec,
        "decode_skew": dec_skew,
        "prefill_8k": pre,
        "prefill_verdict": prefill_verdict,
        "verdict": verdict,
        **extra,
    }


def estimate_tp(key: str, spec: dict, link: dict, args) -> dict:
    hidden = spec["hidden"]
    rows = 1
    dec_bytes = rows * hidden * 2
    pre_bytes = 8192 * hidden * 2
    bid = link.get("throughput", {}).get("bidirectional", [])
    row = _nearest(bid, dec_bytes, "seconds")
    bw = 0.0
    if row:
        bw = max(row.get("payload_gbs_each", 0), 1e-9)
    if not bw:
        bw = 3.5
    dec_s = 2 * dec_bytes * spec["layers"] / (bw * 1e9)
    pre_s = 2 * pre_bytes * spec["layers"] / (bw * 1e9)
    return {
        "method": "tensor_parallel_ref",
        "bytes_per_allreduce_decode": dec_bytes,
        "bytes_per_allreduce_prefill_8k": pre_bytes,
        "ring_volume_factor": 2,
        "bandwidth_gbs": bw,
        "decode_added_ms_per_token": dec_s * 1e3,
        "prefill_added_s_8k": pre_s,
        "note": "ring all-reduce of tokens*hidden*2, 2x volume per layer",
    }


def _front_device(link: dict) -> dict | None:
    for d in link.get("link", link).get("devices", []):
        if str(d.get("arch", "")).startswith("gfx12"):
            return d
    return None


def render_estimate_md(result: dict) -> str:
    lines = ["# Split estimates", "",
             f"Link: `{result.get('link_label')}` ({result.get('link_method')}).  "
             "Baselines are context from doc/benchmarks.md, not predictions.", ""]
    for key, est in result["models"].items():
        lines += [f"## {est['label']} ({key})", ""]
        spec = est["spec"]
        if spec.get("size_unknown"):
            lines += ["Model size unknown: no pack headers were found and the builtin has "
                      "no total_bytes, so layer/expert split estimates are unavailable.", ""]
            continue
        lines += ["| Item | Value |", "| --- | --- |",
                  f"| pack | {spec.get('pack')} |",
                  f"| source | {spec.get('source')} |",
                  f"| bytes | {fmt_bytes(spec.get('total_bytes') or 0)} |",
                  f"| layers | {spec['layers']} ({spec['dense']} dense) |",
                  f"| hidden / hc | {spec['hidden']} / {spec['hc']} |",
                  f"| experts / top-k | {spec['experts']} / {spec['topk']} |",
                  f"| activation/crossing | {fmt_bytes(est['layer']['activation_bytes_per_token'])} "
                  f"({est['layer']['activation_note']}) |", ""]
        if spec.get("external_tables_note"):
            lines.append(f"Note: {spec['external_tables_note']} "
                         f"({fmt_bytes(spec['external_tables_bytes'])}) is excluded from the "
                         f"trunk budget; it stays on CPU/SSD.")
            lines.append("")
        if spec.get("approximated"):
            lines.append("Note: header bytes were unavailable; per-layer bytes are a uniform "
                         "approximation and are marked config-only.")
            lines.append("")
        lyr = est["layer"]
        lines += ["| Method | Front N / capacity | Decode add ms/token | Decode % baseline | "
                  "Prefill add s per 8K | Prefill % baseline | Verdict |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        pre8 = next((p for p in lyr["prefill"] if p["tokens"] == 8192), lyr["prefill"][-1])
        lines.append(
            f"| layer split | N={lyr['best_N']} of {spec['layers']} | "
            f"{lyr['decode']['added_ms_per_token']:.3f} | "
            f"{lyr['decode']['pct_baseline']:.2f}% | "
            f"{pre8['added_s']:.4f} | {pre8['pct']:.3f}% | "
            f"{'fits' if lyr['best_N'] is not None else 'does not fit'} |")
        exp = est["expert"]
        dec = exp["decode_uniform"]
        lines.append(
            f"| expert split | {exp['capacity_fraction_front'] * 100:.1f}% of experts | "
            f"{dec['total_s'] * 1e3:.3f} | "
            f"{100 * dec['total_s'] / (1 / spec['decode_tps']):.2f}% | "
            f"{exp['prefill_8k']['total_s']:.4f} | "
            f"{100 * exp['prefill_8k']['total_s'] / (8192 / spec['prefill_tps']):.3f}% | "
            f"{exp['verdict']} |")
        tp = est["tp"]
        lines.append(
            f"| tensor parallel (ref) | 2 devices | {tp['decode_added_ms_per_token']:.3f} | "
            f"{100 * tp['decode_added_ms_per_token'] / (1000 / spec['decode_tps']):.2f}% | "
            f"{tp['prefill_added_s_8k']:.4f} | "
            f"{100 * tp['prefill_added_s_8k'] / (8192 / spec['prefill_tps']):.3f}% | "
            f"bandwidth-bound reference |")
        lines += ["", f"Front budget {fmt_bytes(lyr['front_budget'])} (long-context, "
                  f"reserve + KV workspace) or {fmt_bytes(lyr['front_budget_shortctx'])} "
                  f"(short-context, reserve only) of {fmt_bytes(lyr['front_total'])}; "
                  f"endcaps {fmt_bytes(lyr['endcap_bytes'])}; CPU-resident "
                  f"{fmt_bytes(lyr['cpu_bytes'])}.  Short-context best N = "
                  f"{lyr['best_N_shortctx']}.", ""]
        if exp["dense_layers"]:
            lines.append(f"Decode is {exp['verdict']}; prefill is "
                         f"{exp['prefill_verdict']}.  {exp['dense_layers']} dense layers "
                         f"({(exp['dense_byte_fraction'] or 0) * 100:.2f}% of bytes) cannot be "
                         f"expert-split; they need layer split or replication.")
            lines.append("")
        else:
            lines.append(f"Decode is {exp['verdict']}; prefill is "
                         f"{exp['prefill_verdict']}.")
            lines.append("")
        if "ssd_tier_gbs" in exp:
            lines.append(f"DS41 tier check: link {exp['link_over_ssd']:.2f}x the SSD tier "
                         f"({exp['ssd_tier_gbs']:.1f} GB/s).")
            lines.append("")
        lines += ["| N | front bytes | long-ctx fits | short-ctx fits |",
                  "| --- | --- | --- | --- |"]
        lo = max(0, (lyr["best_N"] or 0) - 3)
        hi = min(spec["layers"], (lyr["best_N"] or 0) + 3)
        for row in lyr["sweep"][lo:hi + 1]:
            lines.append(f"| {row['N']} | {fmt_bytes(row['front_bytes'])} | "
                         f"{row['fits']} | {row.get('fits_shortctx')} |")
        lines.append("")
        m = est["measured"]
        direct = (f"- direct iGPU->9070 D2D 84 MiB: {m['direct_gbs']:.2f} GB/s "
                  f"(not used by the estimate)"
                  if m.get("direct_gbs") is not None
                  else "- direct iGPU->9070 D2D: skipped (probe failed)")
        lines += ["Measured inputs used:", "",
                  f"- latency floor (staged, ~64 KiB): {m['latency_us']:.1f} us",
                  f"- staged ring2 bandwidth at ~256 MiB: {m['bw_gbs']:.2f} GB/s",
                  f"- bidirectional aggregate: {m['duplex_gbs']:.2f} GB/s",
                  direct, ""]
    lines += ["```json", json.dumps(result["models"], indent=1)[:60000], "```", ""]
    return "\n".join(lines)


def run_estimate(args: argparse.Namespace) -> int:
    with open(args.link, encoding="utf8") as f:
        link = json.load(f)
    overrides = {}
    for item in args.model or []:
        if "=" not in item:
            print(f"estimate: --model needs NAME=PATH, got {item!r}", file=sys.stderr)
            return 2
        name, path = item.split("=", 1)
        overrides[name] = path

    packs = discover_packs()
    packs.update(overrides)
    if not any(packs.values()):
        print("estimate: no model packs found; using built-ins", flush=True)

    result = {
        "schema": "kyojin-link-estimate/1",
        "link_label": link.get("label"),
        "link_method": link.get("link", {}).get("method"),
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "models": {},
    }
    label = link.get("label") or "unknown"
    out_md = args.out or f"docs-notes/link-bench/estimates-{label}.md"
    out_json = args.json or out_md.replace(".md", ".json")

    for key in ("glm", "qwen", "ds41"):
        pack = packs.get(key)
        spec = spec_from_config(key, pack)
        spec = estimate_spec(key, spec)
        est = {
            "label": spec["label"],
            "spec": spec,
            "layer": estimate_layer_split(key, spec, link, args),
            "expert": estimate_expert_split(key, spec, link, args),
            "tp": estimate_tp(key, spec, link, args),
            "measured": measured_inputs(link),
        }
        result["models"][key] = est
        n = "size unknown" if spec.get("size_unknown") else est["layer"]["best_N"]
        print(f"[estimate] {key}: pack={pack or 'builtin'} "
              f"N={n} expert={est['expert']['verdict']}", flush=True)

    with open(out_json, "w", encoding="utf8") as f:
        json.dump(result, f, indent=1)
    with open(out_md, "w", encoding="utf8") as f:
        f.write(render_estimate_md(result))
    print(f"[estimate] wrote {out_md} and {out_json}", flush=True)
    return 0


def measured_inputs(link: dict) -> dict:
    lat = link.get("latency", {})
    rows = lat.get("front_to_main", [])
    row = _nearest(rows, 64 * KIB, "staged")
    latency_us = (row["staged"]["median"] * 1e6) if row else float("nan")
    thr = link.get("throughput", {}).get("front_to_main", [])
    bw, _ = _interp_bw(thr, 256 * MIB, "staged_ring2")
    bid = link.get("throughput", {}).get("bidirectional", [])
    duplex = max((r.get("aggregate_payload_gbs", 0) for r in bid), default=0.0)
    direct = None
    mf = link.get("throughput", {}).get("main_to_front", [])
    drow = _nearest(mf, 84 * MIB, "direct")
    if drow and drow.get("direct"):
        direct = drow["direct"].get("payload_gbs")
    return {"latency_us": latency_us, "bw_gbs": bw, "duplex_gbs": duplex,
            "direct_gbs": direct}


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

def run_compare(args: argparse.Namespace) -> int:
    a = json.load(open(args.a, encoding="utf8"))
    b = json.load(open(args.b, encoding="utf8"))
    out = args.out or "docs-notes/link-bench/compare.md"
    lines = ["# eGPU link comparison", "",
             f"A: `{a.get('label')}` ({a.get('link', {}).get('method')})  ",
             f"B: `{b.get('label')}` ({b.get('link', {}).get('method')})", ""]
    lines += ["## Latency (staged median, us)", "",
              "| direction | bytes | A | B | B/A |", "| --- | --- | --- | --- | --- |"]
    for direction in ("main_to_front", "front_to_main"):
        ra = {r["bytes"]: r for r in a.get("latency", {}).get(direction, [])}
        rb = {r["bytes"]: r for r in b.get("latency", {}).get(direction, [])}
        for n in sorted(set(ra) & set(rb)):
            va = ra[n]["staged"]["median"]
            vb = rb[n]["staged"]["median"]
            lines.append(f"| {direction} | {fmt_bytes(n)} | {us(va)} | {us(vb)} | "
                         f"{vb / va:.3f} |")
    lines += ["", "## Throughput (staged ring2 payload GB/s)", "",
              "| direction | bytes | A | B | B/A | theory A | theory B |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for direction in ("main_to_front", "front_to_main"):
        ra = {r["bytes"]: r for r in a.get("throughput", {}).get(direction, [])}
        rb = {r["bytes"]: r for r in b.get("throughput", {}).get(direction, [])}
        for n in sorted(set(ra) & set(rb)):
            va = ra[n]["staged_ring2"]["payload_gbs"]
            vb = rb[n]["staged_ring2"]["payload_gbs"]
            lines.append(f"| {direction} | {fmt_bytes(n)} | {va:.2f} | {vb:.2f} | "
                         f"{vb / va:.2f}x | {ra[n].get('theoretical_gbs') or 0:.2f} | "
                         f"{rb[n].get('theoretical_gbs') or 0:.2f} |")
    lines += ["", "## Key findings", ""]
    lines.append(f"- Method A/B: {a.get('link', {}).get('method')} / "
                 f"{b.get('link', {}).get('method')}")
    lines.append(f"- Method A/B: {a.get('link', {}).get('method')} / "
                 f"{b.get('link', {}).get('method')}")
    da, db = _direct_gbs(a), _direct_gbs(b)
    fa = f"{da:.2f}" if da is not None else "skipped/segfault"
    fb = f"{db:.2f}" if db is not None else "skipped/segfault"
    lines.append(f"- Direct D2D 84 MiB A/B: {fa} / {fb} GB/s")
    lines.append("- Ring depth 2 versus serial and the bidirectional duplex factor are in the "
                 "per-run JSON.")
    with open(out, "w", encoding="utf8") as f:
        f.write("\n".join(lines) + "\n")
    with open(out.replace(".md", ".json"), "w", encoding="utf8") as f:
        json.dump({"a": a.get("label"), "b": b.get("label"),
                   "latency": {"a": a.get("latency"), "b": b.get("latency")},
                   "throughput": {"a": a.get("throughput"), "b": b.get("throughput")}}, f, indent=1)
    print(f"[compare] wrote {out}", flush=True)
    return 0


def _direct_gbs(run: dict) -> float | None:
    rows = run.get("throughput", {}).get("main_to_front", [])
    row = _nearest(rows, 84 * MIB, "direct")
    if not row or not row.get("direct"):
        return None
    return row["direct"].get("payload_gbs")


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------

def self_test() -> int:
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"[self-test] {'PASS' if cond else 'FAIL'} {name}")

    s = summary([1.0, 2.0, 3.0, 4.0])
    check("summary median", s["median"] == 2.5)
    check("summary p10", abs(s["p10"] - 1.3) < 1e-9)
    check("summary p90", abs(s["p90"] - 3.7) < 1e-9)

    rows = [{"bytes": 1024, "staged_ring2": {"payload_gbs": 1.0}},
            {"bytes": 1024 * 1024, "staged_ring2": {"payload_gbs": 10.0}}]
    bw, note = _interp_bw(rows, 32 * 1024, "staged_ring2")
    check("interp monotonic", 1.0 <= bw <= 10.0)

    name = "model.language_model.layers.17.mlp.experts.3.down_proj.trellis"
    m = TRUNK_LAYER_RE.search(name)
    check("layer regex", m is not None and m.group(1) == "17")
    name2 = "model.language_model.layers.45.eh_proj"
    check("mtp layer regex", TRUNK_LAYER_RE.search(name2).group(1) == "45")

    # Ring dependency: simulate two slots and ensure slot i+2 waits for slot i's read.
    class FakeEvent:
        def __init__(self):
            self.recorded = False

    waits = []

    class FakeStream:
        def wait_event(self, ev):
            waits.append(ev)

    class Sim:
        def __init__(self):
            self.depth = 2
            self.read = [FakeEvent() for _ in range(2)]
            self.used = [False, False]
            self.i = 0
            self.stream = FakeStream()

        def step(self):
            s = self.i % self.depth
            if self.used[s]:
                self.stream.wait_event(self.read[s])
            self.used[s] = True
            self.read[s].recorded = True
            self.i += 1

    sim = Sim()
    for _ in range(4):
        sim.step()
    check("ring waits on slot reuse", len(waits) == 2)

    check("theoretical gen4x4", abs(theoretical_gbs("16.0 GT/s PCIe", 4) - 7.877) < 0.2)
    check("fmt_bytes", fmt_bytes(64 * KIB) == "64.0 KiB")
    print(f"[self-test] {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_measure_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--label", default="auto",
                   help="filename label; auto uses the fingerprint method")
    p.add_argument("--out", default=None, help="JSON path (default docs-notes/link-bench/<label>.json)")
    p.add_argument("--md", default=None, help="markdown path")
    p.add_argument("--main-arch", default="gfx1151")
    p.add_argument("--front-arch", default="gfx1201")
    p.add_argument("--quick", action="store_true", help="about 30 s instead of 2-3 min")
    p.add_argument("--soak", type=float, default=0.0, help="seconds of sustained hops (0 = off)")
    p.add_argument("--max-bytes", type=int, default=GIB)
    p.add_argument("--raw-dir", default=None, help="write per-iteration JSONL here")
    p.add_argument("--skip-direct", action="store_true",
                   help="skip direct D2D (also auto-skipped when the probe crashes)")
    p.add_argument("--self-test", action="store_true", help="statistics/ring checks only")


def _default_front_reserve() -> float:
    """SplitPlan's reserve, honouring the same env override as the engine."""
    env = os.environ.get("KYOJIN_FRONT_RESERVE_GB")
    if env:
        try:
            return float(env)
        except ValueError:
            pass
    try:
        from exllamav3.device_map import DEFAULT_FRONT_RESERVE_GB
        return float(DEFAULT_FRONT_RESERVE_GB)
    except Exception:   # noqa: BLE001 - standalone use outside the repo
        return 1.2


def add_estimate_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--link", required=True, help="measure JSON to read")
    p.add_argument("--model", action="append", metavar="NAME=PATH")
    p.add_argument("--out", default=None, help="markdown path")
    p.add_argument("--json", default=None, help="JSON path")
    p.add_argument("--endcaps", choices=("on", "off"), default="on")
    p.add_argument("--front-reserve-gb", type=float, default=_default_front_reserve())
    p.add_argument("--kv-workspace-gb", type=float, default=1.5)
    p.add_argument("--draft-steps", type=int, default=0)
    p.add_argument("--prefill-tokens", default="4096,8192")
    p.add_argument("--expert-skew", type=float, default=1.5)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd")
    m = sub.add_parser("measure", help="fingerprint + link measurements")
    add_measure_args(m)
    e = sub.add_parser("estimate", help="model-aware split estimates from a measure JSON")
    add_estimate_args(e)
    c = sub.add_parser("compare", help="compare two measure JSONs")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--out", default=None)
    return ap


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("measure", "estimate", "compare"):
        if argv and argv[0] in ("-h", "--help"):
            build_parser().print_help()
            return 0
        argv = ["measure"] + argv
    args = build_parser().parse_args(argv)

    if args.cmd == "measure":
        if args.self_test:
            return self_test()
        try:
            return run_measure(args)
        except Exception as exc:                    # noqa: BLE001 - report, keep partial JSON
            print(f"[link] ERROR in {CURRENT_GROUP}: {type(exc).__name__}: {exc}", flush=True)
            return 1
    if args.cmd == "estimate":
        return run_estimate(args)
    if args.cmd == "compare":
        return run_compare(args)
    build_parser().print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
