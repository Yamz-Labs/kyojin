"""Opt-in host-RSS accounting for serve.py (rssleak1). Off unless EXL3_SERVE_RSS_LOG is set.

The served GLM process parks recurrent checkpoints in system RAM (exllamav3/cache/recurrent.py),
so "process RSS" is the number that decides whether a lane OOMs the 128 GB box, and the device
allocator numbers say nothing about it. This probe writes one JSON line per finished request with

  - VmRSS / VmHWM from /proc/self/status, and the same split the kernel reports (RssAnon, RssFile,
    RssShmem) so a growth in mapped device pages is distinguishable from anonymous host memory
  - torch device allocated/reserved (unchanging in the served lane, kept as a control)
  - the pinned-host allocator (torch.cuda.host_memory_stats) if this build has it
  - the recurrent cache: entry count, accounted bytes, and the bytes actually held by the tensors
  - a GC census of live torch storages, so a leak shows up as an owner instead of a number

The last request of a run passes finish=True: it then re-reads RSS after gc.collect(), after
malloc_trim(0), and after dropping the recurrent cache, which splits "live references" from
"allocator residue".
"""
import ctypes
import inspect
import gc
import json
import os
import time
from pathlib import Path

import torch

_libc = ctypes.CDLL("libc.so.6")


def _status(*keys):
    out = {}
    for line in open("/proc/self/status"):
        k = line.split(":")[0]
        if k in keys:
            out[k] = int(line.split()[1]) // 1024
    return out


def _rss():
    return _status("VmRSS").get("VmRSS", 0)


def _tensor_bytes(t):
    try:
        if t.is_meta or t.device.type != "cpu":
            return 0
        return t.untyped_storage().nbytes()
    except Exception:
        return 0


class RSSProbe:

    def __init__(self, path, engine=None):
        self.path = Path(path)
        self.engine = engine
        self.fh = open(self.path, "a", buffering=1)
        self.n = 0
        # EXL3_SERVE_RSS_FINISH=<n>: run the gc/trim/census diagnostic on request n only. Running it
        # on every request would malloc_trim() and empty the recurrent cache each time, i.e.
        # measure a server nobody serves.
        self.finish_at = int(os.environ.get("EXL3_SERVE_RSS_FINISH", "0") or 0)
        self.finished = False
        self.write({"ev": "start", "t": time.time(), "pid": os.getpid(),
                    "rss": _rss(), "finish_at": self.finish_at,
                    "status": _status("VmRSS", "VmHWM", "RssAnon", "RssFile", "RssShmem")})

    # ---- internals ----------------------------------------------------------------

    def _rc(self):
        gen = getattr(self.engine, "greedy_generator", None)
        rc = getattr(gen, "recurrent_cache", None)
        if rc is None:
            return None
        held, nbytes = 0, 0
        for v in rc.values():
            held += 1
            nbytes += _tensor_bytes(v)
            for k, s in v.items():
                if torch.is_tensor(s):
                    nbytes += _tensor_bytes(s)
                elif isinstance(s, (list, tuple)):
                    nbytes += sum(_tensor_bytes(x) for x in s)
        return {"n": held, "accounted_mb": round(rc.current_size / 2**20, 1),
                "tensor_mb": round(nbytes / 2**20, 1), "max_mb": round(rc.max_size / 2**20, 1),
                "metrics": dict(rc.metrics)}

    def _census(self, top=12):
        """Live CPU tensors by owner: the fastest way to name an unaccounted holder."""
        agg = {}
        for o in gc.get_objects():
            try:
                if not torch.is_tensor(o) or o.device.type != "cpu" or o.is_meta:
                    continue
                b = _tensor_bytes(o)
                if b < (1 << 20):
                    continue
                key = f"{tuple(o.shape)} {o.dtype}"
                e = agg.setdefault(key, [0, 0, ""])
                e[0] += b
                e[1] += 1
                if not e[2]:
                    e[2] = self._referrer(o)
            except Exception:
                continue
        rows = sorted(((v[0], k, v[1], v[2]) for k, v in agg.items()), reverse=True)[:top]
        return [{"mb": round(b / 2**20, 1), "what": k, "n": n, "held_by": p} for b, k, n, p in rows]

    @staticmethod
    def _referrer(t, depth=4):
        """One plausible owner path for a live tensor, so a census row names a suspect.

        Instances hold their tensors in __dict__, so the first referrer of a tensor is usually that
        dict; the name comes from the object owning it. A frame referrer is reported as
        file:line in a function, which is the most useful answer when there is one."""
        seen, cur = {id(t)}, t
        for _ in range(depth):
            refs = [r for r in gc.get_referrers(cur) if id(r) not in seen]
            seen.add(id(cur))
            refs = [r for r in refs if r is not RSSProbe._referrer.__globals__]
            frames = [r for r in refs if inspect.isframe(r)]
            if frames:
                f = frames[0]
                return f"{f.f_code.co_filename.split('/')[-1]}:{f.f_lineno} in {f.f_code.co_name}()"
            objs = [r for r in refs if not isinstance(r, (list, dict, set, tuple))]
            if objs:
                cur = objs[0]
                continue
            dicts = [r for r in refs if isinstance(r, dict)]
            if not dicts:
                return type(cur).__name__
            d = dicts[0]
            keys = [k for k, v in list(d.items())[:2000] if v is cur][:2]
            owners = [r for r in gc.get_referrers(d) if hasattr(r, "__dict__") and vars(r) is d]
            if owners:
                o = owners[0]
                return f"{type(o).__module__}.{type(o).__name__}.{keys}"
            return f"dict keys {keys}"
        return type(cur).__name__

    def _host(self):
        try:
            s = torch.cuda.host_memory_stats()
            return {k: round(s.get(k, 0) / 2**20, 1) if "bytes" in k else s.get(k, 0)
                    for k in ("allocated_bytes.all.current", "segment.all.current")}
        except Exception:
            return None

    def write(self, rec):
        rec.setdefault("t", time.time())
        rec["rss"] = _rss()
        self.fh.write(json.dumps(rec) + "\n")

    # ---- per request --------------------------------------------------------------

    def request(self, prompt_tokens=0, gen_tokens=0, wall=0.0, finish=False):
        self.n += 1
        rec = {"ev": "req", "n": self.n, "ptok": prompt_tokens, "gtok": gen_tokens,
               "wall": round(wall, 2),
               "status": _status("VmRSS", "VmHWM", "RssAnon", "RssFile", "RssShmem"),
               "dev_alloc_mb": round(torch.cuda.memory_allocated() / 2**20, 1),
               "dev_resv_mb": round(torch.cuda.memory_reserved() / 2**20, 1),
               "host_alloc": self._host(), "rc": self._rc()}
        self.write(rec)
        # One shot: the diagnostic trims and empties the recurrent cache, so running it twice
        # measures a server nobody serves.
        if not self.finished and (finish or (self.finish_at and self.n >= self.finish_at)):
            self.finished = True
            self.diag()
        return rec

    def diag(self):
            self.write({"ev": "diag", "step": "raw", "rc": self._rc()})
            gc.collect()
            self.write({"ev": "diag", "step": "after_gc", "rc": self._rc()})
            _libc.malloc_trim(0)
            self.write({"ev": "diag", "step": "after_trim", "rc": self._rc()})
            gen = getattr(self.engine, "greedy_generator", None)
            rc = getattr(gen, "recurrent_cache", None)
            if rc is not None:
                rc.clear()
                rc.update_total_size()
            gc.collect()
            self.write({"ev": "diag", "step": "after_rc_clear", "rc": self._rc()})
            _libc.malloc_trim(0)
            self.write({"ev": "diag", "step": "after_rc_clear_trim", "rc": self._rc()})
            self.write({"ev": "diag", "step": "census", "census": self._census()})
            self.write({"ev": "end", "n": self.n})


def install(engine, path=None):
    path = path or os.environ.get("EXL3_SERVE_RSS_LOG")
    if not path:
        return None
    probe = RSSProbe(path, engine)
    engine.rss_probe = probe
    return probe
