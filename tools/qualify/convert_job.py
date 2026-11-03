"""Conversion as a qualify step (drill-0 gap 3). Drill 0 did this by hand for 2 h 40: a wrapper script (lock, resume flag, setsid), a second script that
SIGSTOPped/CONTed the converter when MemAvailable fell, milestone lines, a zombie to reap, the chat template copied into the pack by hand and the
converter log (per-layer SQNR) lost. Here: one call, own-PID pause/resume guard, <pack>/convert.log, template carried over, resume on rerun."""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path

import backends
import pack as packmod


def pack_complete(pack: str | os.PathLike) -> bool:
    p = Path(pack).expanduser()
    return (p / "config.json").exists() and (p / "quantization_config.json").exists() and not packmod.verify_shards(p)


def carry_template(src: Path, pack: Path) -> str | None:
    """The pack must hold the working chat template (serve reads --chat-template from it): copy the source's chat_template.jinja, or write the
    one embedded in tokenizer_config.json. Returns where it came from, None when the source has none (qualify.meta then reports it)."""
    dst = pack / "chat_template.jinja"
    if dst.exists():
        return "pack"
    if (src / "chat_template.jinja").exists():
        shutil.copy2(src / "chat_template.jinja", dst)
        return "source file"
    try:
        t = json.loads((src / "tokenizer_config.json").read_text()).get("chat_template")
    except (OSError, ValueError):
        t = None
    if isinstance(t, str) and t.strip():
        dst.write_text(t)
        return "tokenizer_config"
    return None


class MemGuard(threading.Thread):
    """SIGSTOP the converter's process group when MemAvailable < pause_gib, SIGCONT at >= resume_gib (own group only, never a name match)."""

    def __init__(self, pgid: int, pause_gib: float, resume_gib: float, log, poll_s: float = 5.0):
        super().__init__(daemon=True)
        self.pgid, self.pause, self.resume, self.log, self.poll = pgid, pause_gib, resume_gib, log, poll_s
        self.stop_ev, self.stopped, self.paused_s, self.pauses = threading.Event(), False, 0.0, 0

    def run(self):
        t_stop = 0.0
        while not self.stop_ev.wait(self.poll):
            a = backends.mem_available_gib()
            try:
                if not self.stopped and a < self.pause:
                    os.killpg(self.pgid, signal.SIGSTOP); self.stopped, t_stop = True, time.time(); self.pauses += 1
                    self.log(f"convert: PAUSED (MemAvailable {a:.1f} GiB < {self.pause})")
                elif self.stopped and a >= self.resume:
                    os.killpg(self.pgid, signal.SIGCONT); self.stopped = False; self.paused_s += time.time() - t_stop
                    self.log(f"convert: RESUMED (MemAvailable {a:.1f} GiB)")
            except ProcessLookupError:
                return

    def finish(self):
        self.stop_ev.set()
        if self.stopped:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.pgid, signal.SIGCONT)


def convert_argv(src: str, work: str, pack: str, bits: float, head_bits: int, resume: bool, extra: list[str]) -> list[str]:
    a = ["convert.py", "-i", src, "-w", work, "-o", pack, "-b", str(bits), "-hb", str(head_bits)]
    return a + (["-r"] if resume else []) + list(extra)


def disk_need_gb(src: str) -> float:
    """Converter work files + output: drill 0 saw 41 GB of work + 18 GB pack for a 62 GB source, i.e. ~0.95 x the source size."""
    return 0.95 * sum(f.stat().st_size for f in Path(src).glob("*.safetensors")) / 1e9


def run_convert(prof: dict, repo: Path, a: dict, out: Path, log) -> dict:
    src, pack = Path(a["convert_from"]).expanduser(), Path(a["pack"])
    work = Path(a.get("conv_work") or out / "conv_work")
    problems = packmod.verify_shards(src)
    if problems:
        raise RuntimeError(f"convert: source {src} is not a complete checkpoint: {problems[:2]}")
    free, need = packmod.disk_free_gb(pack), disk_need_gb(str(src))
    if free - need < a["disk_floor_gb"]:
        raise RuntimeError(f"convert: needs ~{need:.0f} GB of work + output, {free:.0f} GB free, floor {a['disk_floor_gb']} GB")
    resume = (work / "state.json").exists()
    argv = convert_argv(str(src), str(work), str(pack), a["bits"], a["head_bits"], resume, a.get("conv_extra", "").split())
    env = {**os.environ, **prof["env"]}
    pack.parent.mkdir(parents=True, exist_ok=True)
    logf = pack.parent / f"{pack.name}.convert.log" if not pack.exists() else pack / "convert.log"
    t0 = time.time()
    with open(logf, "ab") as fh:
        p = subprocess.Popen([prof["python"], "-u", *argv], cwd=repo, env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        g = MemGuard(p.pid, a["conv_pause_gib"], a["conv_resume_gib"], log)
        g.start()
        last = 0.0
        try:
            while p.poll() is None:
                if time.time() - last > 1200:       # a milestone line every 20 min (fleet rule)
                    last = time.time(); log(f"convert: running {time.time()-t0:.0f} s, MemAvailable {backends.mem_available_gib():.0f} GiB, paused {g.paused_s:.0f} s")
                time.sleep(5)
        finally:
            g.finish()
            if p.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(p.pid, signal.SIGTERM)
    if p.returncode:
        raise RuntimeError(f"convert.py rc={p.returncode}; see {logf}")
    if logf.parent != pack:                              # the converter created the pack dir: move the log into it
        shutil.move(str(logf), str(pack / "convert.log"))
    tmpl = carry_template(src, pack)
    if not a.get("keep_work"):
        shutil.rmtree(work, ignore_errors=True)
    return dict(wall_s=round(time.time() - t0, 1), paused_s=round(g.paused_s, 1), pauses=g.pauses, resumed=resume, template_from=tmpl, cmd=[prof["python"], *argv],
                bits=a["bits"], head_bits=a["head_bits"])
