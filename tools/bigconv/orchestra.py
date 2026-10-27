"""Capture -> spool -> quantize orchestration policy, model-agnostic and CPU-testable.

On-disk protocol (same as the DS4.1 conversion):
    <root>/work/spool/layers.<L>/CAPTURE_DONE      layer captured (spool complete)
    <root>/claims/L<NN>/{owner,hb,FAILED,COPIED}   atomic `mkdir` claim; hb = heartbeat file; COPIED = a verified remote copy exists
    <root>/claims/L<NN>.done                       shard verified (JSON record), written last, atomically

Every function here fixes one operational failure seen in a live run:
  * pending/lead counting ignores finished spools and a separate sweeper removes them (capture never waits for a spool nobody needs)
  * the lead is never below 2 (lead 1 = capture waits for the previous layer to be quantized)
  * prefetch refills a free slot at any time, not only when a layer starts
  * every step is judged by rc AND outputs AND the output text (a traceback can come back as rc 0)
  * a FAILED claim is resumed from partial/, never restarted; partial/ is never wiped
  * a quantizer on a GPU shared with the capture follows `shared_gpu_gate`
"""
from __future__ import annotations

import collections
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from .defaults import DEFAULTS


# --------------------------------------------------------------------------------------------- layout and spool state
@dataclass
class Layout:
    root: str

    @property
    def claims(self):
        return os.path.join(self.root, "claims")

    @property
    def spool(self):
        return os.path.join(self.root, "work", "spool")

    def nn(self, L):
        return f"L{L:02d}"

    def claim(self, L):
        return os.path.join(self.claims, self.nn(L))

    def spool_dir(self, L):
        return os.path.join(self.spool, f"layers.{L}")


def _spool_layers(lay: Layout):
    if not os.path.isdir(lay.spool):
        return []
    return sorted(int(d[7:]) for d in os.listdir(lay.spool) if d.startswith("layers.") and d[7:].isdigit())


def spool_ready(lay: Layout, L):
    return os.path.exists(os.path.join(lay.spool_dir(L), "CAPTURE_DONE"))


def spool_finished(lay: Layout, L):
    """Done, or a verified copy exists elsewhere: the local spool is not needed any more."""
    c = lay.claim(L)
    return os.path.exists(c + ".done") or os.path.exists(os.path.join(c, "COPIED"))


def pending_spools(lay: Layout):
    """Captured spools somebody still needs. Finished spools do NOT count (the old count included them: deadlock)."""
    return [L for L in _spool_layers(lay) if spool_ready(lay, L) and not spool_finished(lay, L)]


def captured_free(lay: Layout):
    """Captured spools with no claim."""
    return [L for L in _spool_layers(lay) if spool_ready(lay, L) and not os.path.exists(lay.claim(L))]


def sweep_finished(lay: Layout, log: Callable = lambda m: None):
    """Delete the spools of finished layers. Independent of the capture: run it in a Sweeper, never 'between captures'."""
    gone = []
    for L in _spool_layers(lay):
        if spool_finished(lay, L):
            shutil.rmtree(lay.spool_dir(L), ignore_errors=True)
            gone.append(L)
            log(f"swept spool layers.{L}")
    return gone


class Sweeper(threading.Thread):
    """Background sweep every `period` seconds until stop() (or the DONE_ALL file appears)."""

    def __init__(self, lay: Layout, period=30.0, log: Callable = lambda m: None):
        super().__init__(daemon=True)
        self.lay, self.period, self.log = lay, period, log
        self._stop_ev = threading.Event()

    def run(self):
        while not self._stop_ev.is_set():
            if os.path.exists(os.path.join(self.lay.claims, "DONE_ALL")):
                return
            try:
                sweep_finished(self.lay, self.log)
            except OSError as e:
                self.log(f"sweep error: {e}")
            self._stop_ev.wait(self.period)

    def stop(self):
        self._stop_ev.set()


def effective_lead(lead):
    """lead 1 serialises capture behind the quantizers (capture blocked until the previous layer is done)."""
    return max(int(lead), DEFAULTS["min_lead"])


def capture_cap(lead, free_gb, floor_gb, spool_gb):
    """How many pending spools the disk allows: the disk, not only the count, pauses the capture."""
    by_disk = int((free_gb - floor_gb) / spool_gb) if spool_gb > 0 else effective_lead(lead)
    return max(0, min(effective_lead(lead), by_disk))


def capture_may_start(lay: Layout, lead, free_gb, floor_gb, spool_gb):
    return len(pending_spools(lay)) < capture_cap(lead, free_gb, floor_gb, spool_gb)


# --------------------------------------------------------------------------------------------- prefetch
def plan_prefetch(free_layers, resident, slots, free_gb, floor_gb, spool_gb):
    """Layers to start copying now. Call it every tick of the worker loop: a slot that is free is refilled at once.
    free_layers: captured, unclaimed (lowest first); resident: layers whose spool is local or being copied.
    Claim a layer (try_claim) BEFORE copying it, so two workers never copy the same layer."""
    room = slots - len(resident)
    if spool_gb > 0:
        room = min(room, int((free_gb - floor_gb) / spool_gb))
    out = []
    for L in sorted(free_layers):
        if len(out) >= room:
            break
        if L not in resident:
            out.append(L)
    return out


# --------------------------------------------------------------------------------------------- claims
def try_claim(lay: Layout, L, owner):
    os.makedirs(lay.claims, exist_ok=True)
    try:
        os.mkdir(lay.claim(L))
    except FileExistsError:
        return False
    _write(os.path.join(lay.claim(L), "owner"), owner + "\n")
    touch(os.path.join(lay.claim(L), "hb"))
    return True


def _write(path, text):
    with open(path, "w") as f:
        f.write(text)


def touch(path):
    with open(path, "a"):
        os.utime(path, None)


def owner_of(lay: Layout, L):
    try:
        return open(os.path.join(lay.claim(L), "owner")).read().strip()
    except OSError:
        return None


def _claim_layers(lay: Layout):
    if not os.path.isdir(lay.claims):
        return []
    return sorted(int(m.group(1)) for d in os.listdir(lay.claims)
                  if (m := re.fullmatch(r"L(\d+)", d)) and os.path.isdir(os.path.join(lay.claims, d)))


def own_open(lay: Layout, owner):
    """Own claims neither done nor failed (a restarted worker resumes these first)."""
    return [L for L in _claim_layers(lay)
            if owner_of(lay, L) == owner and not os.path.exists(lay.claim(L) + ".done")
            and not os.path.exists(os.path.join(lay.claim(L), "FAILED"))]


def mark_failed(lay: Layout, L, msg):
    _write(os.path.join(lay.claim(L), "FAILED"), str(msg)[:300] + "\n")


def failed_count(lay: Layout, L):
    d = lay.claim(L)
    return len([f for f in os.listdir(d) if f.startswith("FAILED.")]) if os.path.isdir(d) else 0


def recover_failed(lay: Layout, owner, max_retries=None):
    """Own FAILED claims that may be retried: the FAILED marker moves to FAILED.<n> (history), the claim reopens, and the
    caller resumes from its partial/ directory (never wiped). Returns the layers reopened."""
    max_retries = DEFAULTS["max_failed_retries"] if max_retries is None else max_retries
    out = []
    for L in _claim_layers(lay):
        d = lay.claim(L)
        fp = os.path.join(d, "FAILED")
        if owner_of(lay, L) != owner or not os.path.exists(fp) or os.path.exists(d + ".done"):
            continue
        n = failed_count(lay, L)
        if n >= max_retries:
            continue
        os.replace(fp, os.path.join(d, f"FAILED.{n + 1}"))
        touch(os.path.join(d, "hb"))
        out.append(L)
    return out


def stale_claims(lay: Layout, now=None, stale_s=None):
    """Claims without .done/FAILED whose heartbeat is older than stale_s (dead worker). A foreign stale claim is reported,
    never taken: its partial/ lives on the other host."""
    now = time.time() if now is None else now
    stale_s = DEFAULTS["hb_stale_s"] if stale_s is None else stale_s
    out = []
    for L in _claim_layers(lay):
        d = lay.claim(L)
        if os.path.exists(d + ".done") or os.path.exists(os.path.join(d, "FAILED")):
            continue
        try:
            age = now - os.path.getmtime(os.path.join(d, "hb"))
        except OSError:
            age = 1e18
        if age > stale_s:
            out.append(L)
    return out


def pick_work(lay: Layout, owner, nlayers, max_retries=None):
    """Next layer for a worker: own open claim, then own FAILED (resume), then the lowest captured unclaimed layer."""
    o = own_open(lay, owner)
    if o:
        return o[0], "resume"
    r = recover_failed(lay, owner, max_retries)
    if r:
        return r[0], "retry"
    for L in captured_free(lay):
        if L < nlayers and try_claim(lay, L, owner):
            return L, "new"
    return None, None


class Heartbeat:
    """with Heartbeat(path): ...   touches `path` every period seconds."""

    def __init__(self, path, period=None):
        self.path = path
        self.period = DEFAULTS["hb_period_s"] if period is None else period
        self._ev = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._ev.is_set():
            try:
                touch(self.path)
            except OSError:
                pass
            self._ev.wait(self.period)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._ev.set()
        self._t.join(timeout=5)


# --------------------------------------------------------------------------------------------- shared GPU
def capture_blocked(lay: Layout, lead, free_gb, floor_gb, spool_gb):
    """True when the capture is paused for the same reason it uses itself: pending spools at the cap, where the cap is
    min(lead, disk room). Compute 'blocked' with this, never with the lead alone (a disk pause idles the capture too)."""
    return not capture_may_start(lay, lead, free_gb, floor_gb, spool_gb)


def shared_gpu_gate(capture_running, blocked, free_captured, own_open_claims, capture_waits_gpu=False, min_free_layers=None):
    """May a quantizer that SHARES the capture GPU (iGPU) take a GPU hold now?
    * capture not running: yes.
    * capture is waiting for the GPU lock: no (the quantizer must never hold the lock the capture needs).
    * capture running: only while it is blocked (`capture_blocked`) AND the dedicated cards have enough free work
      (>= min_free_layers captured unclaimed layers, so they never starve), or the worker only finishes its own claim."""
    min_free_layers = DEFAULTS["shared_gpu_min_free_layers"] if min_free_layers is None else min_free_layers
    if not capture_running:
        return True
    if capture_waits_gpu:
        return False
    return bool(blocked) and (free_captured >= min_free_layers or bool(own_open_claims))


def gpurun_who(index=0, prefix=None):
    """--who name of a quantizer GPU window. The prefix must be exempt from the wrapper's idle kill (quantizers have long
    CPU-only phases; the wrapper killed a layer at 1155/1165 linears)."""
    return f"{prefix or DEFAULTS['quantizer_who_prefix']}{index}"


def idle_exempt(who, gpurun_text):
    """True when the gpurun script text exempts `who` from the idle kill: a line `case "$who" in <glob>*) idle=0;;`."""
    for m in re.finditer(r'case\s+"\$who"\s+in\s+([^\n]*?)\)\s*idle=0', gpurun_text):
        for g in m.group(1).split("|"):
            g = g.strip()
            if re.fullmatch(g.replace("*", ".*"), who):
                return True
    return False


# --------------------------------------------------------------------------------------------- verified steps
class StepFailed(RuntimeError):
    pass


class Held(Exception):
    """The step stopped at its time budget (rc in hold_codes) and is resumable."""


TRACE_MARK = "Traceback (most recent call last)"


@dataclass
class RunResult:
    rc: int
    tail: list = field(default_factory=list)
    held: bool = False


def checked_run(cmd, verify: Optional[Callable] = None, hold_codes=None, tail_lines=60, env=None, cwd=None, out=None):
    """Run a step and judge it by rc, by its output text and by `verify()` (raises on missing/invalid outputs).
    A wrapper chain (gpu wrapper -> shell -> python) can return rc 0 for a crashed python (probable cause, not reproduced:
    the engine import swallows exit codes), so with rc 0 the outputs must verify; without a verify(), a traceback in the output
    tail is a failure (with one, outputs decide, since some paths log handled tracebacks). rc in hold_codes = resumable stop (held=True, outputs not verified)."""
    hold_codes = DEFAULTS["hold_codes"] if hold_codes is None else hold_codes
    out = out or sys.stdout
    p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, cwd=cwd,
                         errors="replace")
    tail = collections.deque(maxlen=tail_lines)
    for line in p.stdout:
        tail.append(line.rstrip("\n"))
        out.write(line)
        out.flush()
    rc = p.wait()
    if rc in hold_codes:
        return RunResult(rc, list(tail), held=True)
    if rc != 0:
        raise StepFailed(f"rc={rc}: {' | '.join(list(tail)[-3:])[:200]}")
    if verify is not None:
        verify()          # outputs decide: a handled, logged traceback on a good run must not fail it
    elif any(TRACE_MARK in ln for ln in tail):
        raise StepFailed(f"rc=0 but a traceback is in the output tail: {' | '.join(list(tail)[-2:])[:200]}")
    return RunResult(rc, list(tail))


def run_holds(cmd_fn: Callable[[], list], verify: Callable, max_holds=60, should_stop: Callable = lambda: False,
              between: Callable = lambda: None, **kw):
    """Repeat a resumable step until it exits clean. `between()` runs between holds (release the GPU lock, yield to the
    capture, sync code). Returns the number of holds; raises StepFailed on a real failure or too many holds."""
    for n in range(1, max_holds + 1):
        if should_stop():
            raise StepFailed("stop requested; the layer stays resumable")
        r = checked_run(cmd_fn(), verify=None, **kw)
        if not r.held:
            verify()
            return n
        between()
    raise StepFailed(f"not finished after {max_holds} holds")


def verify_shard(out_dir, L, min_bytes=1e8, suffix="layers"):
    """Shard of layer L is there and sane: done.json parses, the safetensors header lists tensors, size above min_bytes.
    Returns the facts for the .done record."""
    sh = os.path.join(out_dir, f"{suffix}.{L}.safetensors")
    dj = os.path.join(out_dir, f"{suffix}.{L}.done.json")
    try:
        json.load(open(dj))
        with open(sh, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            hdr = json.loads(f.read(n))
    except (OSError, ValueError) as e:
        raise StepFailed(f"layer {L}: outputs missing or unreadable ({e})")
    keys = [k for k in hdr if k != "__metadata__"]
    sz = os.path.getsize(sh)
    if not keys or sz < min_bytes:
        raise StepFailed(f"layer {L}: shard looks wrong: {len(keys)} keys, {sz} bytes")
    return dict(path=sh, bytes=sz, keys=len(keys))


def mark_done(lay: Layout, L, record: dict):
    """Write claims/L<NN>.done atomically; call it last, after verify_shard."""
    p = lay.claim(L) + ".done"
    _write(p + ".tmp", json.dumps(record) + "\n")
    os.replace(p + ".tmp", p)


def accept_cross_hw(err, ref_err, ratio=None, abs_tol=1e-5):
    """A shard quantized on another GPU family differs in bytes (trellis rounding, scales ~1e-5) but not in quality:
    accept on the proxy error (mean relative error of the layer) against the reference arm, never on a hash."""
    ratio = DEFAULTS["cross_hw_proxy_ratio"] if ratio is None else ratio
    return err <= ref_err * ratio + abs_tol


def exit_guard(main: Callable, hold_exceptions=(), hold_code=76):
    """Entry-point wrapper for quantizer scripts: exceptions print a traceback and leave with os._exit(1) (rc survives any
    import-time exit hook); a hold exception leaves with os._exit(hold_code)."""
    import traceback
    try:
        main()
    except hold_exceptions as e:  # noqa
        sys.stdout.flush()
        os._exit(hold_code)
    except SystemExit as e:
        sys.stdout.flush()
        sys.stderr.flush()
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        os._exit(code)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
