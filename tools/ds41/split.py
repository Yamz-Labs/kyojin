"""Two-machine split of the routed-expert quantization (coordinator / worker roles).

In short: one coordinator process (on the box) owns the output directory and the
per-layer work queue; any number of workers (a local subprocess on the box, an `ssh worker ...` process on
the 3090 machine) pull chunks of experts, fetch the FP4 source themselves, quantize, and send the
quantized tensors back over the same pipe. A worker keeps nothing on disk. Per-expert results land on the
coordinator as `experts/<unit>/e<NNN>.safetensors` + `.json` (marker, written last), so a killed worker or
a restarted coordinator loses at most the chunks in flight. The finished layer is assembled from those
files into the usual `<unit>.safetensors` + `<unit>.done.json`.

Calibration Hessians come through a spool directory (`--hessians spool`): the capture process writes
`<spool>/<unit>/gu.h` (the Hessian shared by every expert's gate/up), `down.<e>.h` per expert and
`lin.<key>.h` for the other linears, each by atomic rename, then `CAPTURE_DONE`. The coordinator releases an
expert as soon as its Hessians exist, so quantization overlaps the rest of the capture.
`--hessians synthetic` fills the spool with deterministic random SPD matrices (tests, smoke), `none` is the
uncalibrated path.

Wire format (pipe or ssh stdio, both directions): 12-byte prefix `<I header_len><Q payload_len>`, JSON header,
raw payload. The worker's stdout is the protocol channel; its prints go to stderr.
"""

import base64
import collections
import shutil
import json
import os
import struct
import sys
import threading
import time
import types
import zlib
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

MODULES = ["common", "dequant", "fetch", "quant", "split"]   # shipped to remote workers
RE_EXPERT_KEY = None


# ----------------------------------------------------------------------------------------------
# Framing
# ----------------------------------------------------------------------------------------------

def send_frame(f, header: dict, payload=b""):
    parts = payload if isinstance(payload, (list, tuple)) else [payload]
    n = sum(len(p) for p in parts)
    hb = json.dumps(header, separators=(",", ":")).encode()
    f.write(struct.pack("<IQ", len(hb), n))
    f.write(hb)
    for p in parts:
        f.write(p)
    f.flush()


def _read_exact(f, n):
    buf = bytearray()
    while len(buf) < n:
        c = f.read(n - len(buf))
        if not c:
            return None if not buf else bytes(buf)  # short read = EOF mid frame
        buf += c
    return bytes(buf)


def recv_frame(f):
    """-> (header, payload) or None on EOF."""
    pre = _read_exact(f, 12)
    if pre is None or len(pre) < 12:
        return None
    hl, pl = struct.unpack("<IQ", pre)
    hb = _read_exact(f, hl)
    payload = _read_exact(f, pl) if pl else b""
    if hb is None or len(hb) < hl or (pl and (payload is None or len(payload) < pl)):
        return None
    return json.loads(hb), payload


# ----------------------------------------------------------------------------------------------
# Hessian blobs
# ----------------------------------------------------------------------------------------------

def pack_hessian(H: torch.Tensor, count: int, num_total: int = 0, inf_nan=(0, 0), group: str | None = None) -> bytes:
    """Symmetric fp32 [n, n] -> upper triangle, row by row (half the bytes), CRC32 trailer.
    The *sum* of x^T x (not divided by count) is stored, as the capture accumulates it."""
    assert H.dtype == torch.float32 and H.ndim == 2 and H.shape[0] == H.shape[1]
    n = H.shape[0]
    hdr = json.dumps({"n": n, "count": int(count), "num_total": int(num_total),
                      "inf_nan": [int(inf_nan[0]), int(inf_nan[1])], "fmt": "tri_f32",
                      **({"group": group} if group else {})}).encode()
    tri = torch.cat([H[i, i:] for i in range(n)]).contiguous().cpu().numpy().tobytes()
    body = struct.pack("<I", len(hdr)) + hdr + tri
    return body + struct.pack("<I", zlib.crc32(body))


def unpack_hessian(blob: bytes) -> dict:
    body, crc = blob[:-4], struct.unpack("<I", blob[-4:])[0]
    if zlib.crc32(body) != crc:
        raise ValueError("Hessian blob CRC mismatch")
    hl = struct.unpack("<I", body[:4])[0]
    meta = json.loads(body[4:4 + hl])
    n = meta["n"]
    v = torch.frombuffer(bytearray(body[4 + hl:]), dtype=torch.float32)
    assert v.numel() == n * (n + 1) // 2, (v.numel(), n)
    H = torch.zeros(n, n)
    off = 0
    for i in range(n):
        H[i, i:] = v[off:off + n - i]
        off += n - i
    H = H.triu() + H.triu(1).T   # canonical symmetric matrix, identical for every consumer
    return {"H": H, "count": meta["count"], "num_total": meta["num_total"], "inf_nan": meta["inf_nan"], "n": n}


NOH = b"NOH"   # spool marker: no Hessian for this linear (expert without routed tokens): quantize uncalibrated


class HessianEntry:
    """One Hessian as seen by a quantizer; `cache` is the quantizer's derived state (H_data)."""

    def __init__(self, name: str, blob: bytes):
        self.name, self.blob, self.cache = name, blob, None

    @classmethod
    def from_blob(cls, blob: bytes, default_name: str):
        """None for the NOH marker. The entry name is the blob's `group` (linears of one group, and all gate/up of a
        layer, carry the same group and therefore draw the same sign vector), else `default_name`."""
        if blob == NOH:
            return None
        hl = struct.unpack("<I", blob[:4])[0]
        name = json.loads(blob[4:4 + hl]).get("group") or default_name
        return cls(name, blob)

    def digest(self) -> int:
        return struct.unpack("<I", self.blob[-4:])[0]

    def drop(self):
        """Free the quantizer's derived state (finalized H, L, raw H): the entry is not needed any more."""
        self.cache = None
        self.__dict__.pop("raw", None)
        self.__dict__.pop("energy_cache", None)

    def load(self) -> dict:
        return unpack_hessian(self.blob)


def synthetic_hessian(n: int, seed: int, rows: int | None = None):
    """Deterministic SPD matrix with a decaying spectrum (sum of x^T x over `rows` samples)."""
    rows = rows or max(n, 256)
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, n, generator=g)
    x *= (1.0 / (1.0 + 6.0 * torch.arange(n, dtype=torch.float32) / n)).sqrt()
    return x.T @ x, rows


class Spool:
    """Hessian hand-off directory between the capture process and the coordinator."""

    def __init__(self, root: str):
        self.root = root

    def dir(self, unit: str) -> str:
        return os.path.join(self.root, unit)

    def path(self, unit: str, name: str) -> str:
        return os.path.join(self.dir(unit), f"{name}.h")

    def put(self, unit: str, name: str, blob: bytes):
        os.makedirs(self.dir(unit), exist_ok=True)
        p = self.path(unit, name)
        with open(p + ".tmp", "wb") as f:
            f.write(blob)
        os.replace(p + ".tmp", p)

    def has(self, unit: str, name: str) -> bool:
        return os.path.exists(self.path(unit, name))

    def read(self, unit: str, name: str) -> bytes:
        with open(self.path(unit, name), "rb") as f:
            return f.read()

    def get(self, unit: str, name: str):
        """HessianEntry, or None when the spool holds the NOH marker."""
        return HessianEntry.from_blob(self.read(unit, name), name)

    def capture_done(self, unit: str) -> bool:
        return os.path.exists(os.path.join(self.dir(unit), "CAPTURE_DONE"))

    def mark_capture_done(self, unit: str):
        os.makedirs(self.dir(unit), exist_ok=True)
        open(os.path.join(self.dir(unit), "CAPTURE_DONE"), "w").close()

    def clear(self, unit: str):
        d = self.dir(unit)
        if os.path.isdir(d):
            for fn in os.listdir(d):
                os.remove(os.path.join(d, fn))
            os.rmdir(d)


def synthetic_fill(spool: Spool, unit: str, experts, dims, slices: int, gap: float, log, lins=()):
    """Write deterministic Hessians for one layer, `slices` groups at a time (simulates the capture).
    `lins`: (key, in_features) of the non-expert linears. The gate/up Hessian is generated once and linked per
    expert (`in.<e>`), the down ones are per expert."""
    in_gu, in_down = dims
    for key, n_in in lins:
        if not spool.has(unit, f"lin.{key}"):
            H, n = synthetic_hessian(n_in, zlib.crc32(f"lin.{key}".encode()))
            spool.put(unit, f"lin.{key}", pack_hessian(H, n))
    gu_name = f"{unit}.ffn.experts.gu"
    H, n = synthetic_hessian(in_gu, zlib.crc32(f"{unit}.gu".encode()))
    gu_blob = pack_hessian(H, n, group=gu_name)
    experts = list(experts)
    per = max(1, -(-len(experts) // max(1, slices)))
    for i in range(0, len(experts), per):
        for e in experts[i:i + per]:
            if not spool.has(unit, f"down.{e}"):
                H, n = synthetic_hessian(in_down, zlib.crc32(f"{unit}.down.{e}".encode()))
                spool.put(unit, f"down.{e}", pack_hessian(H, n))
            if not spool.has(unit, f"in.{e}"):
                spool.put(unit, f"in.{e}", gu_blob)
        if gap and i + per < len(experts):
            time.sleep(gap)
    spool.mark_capture_done(unit)


# ----------------------------------------------------------------------------------------------
# Expert result files on the coordinator
# ----------------------------------------------------------------------------------------------

def expert_paths(out_dir: str, unit: str, e: int):
    d = os.path.join(out_dir, "experts", unit)
    return os.path.join(d, f"e{e:03d}.safetensors"), os.path.join(d, f"e{e:03d}.json")


def expert_done(out_dir: str, unit: str, e: int) -> bool:
    f, m = expert_paths(out_dir, unit, e)
    if not (os.path.exists(f) and os.path.exists(m)):
        return False
    try:
        return os.path.getsize(f) == json.load(open(m))["size"]
    except (ValueError, KeyError, OSError):
        return False


def store_expert(out_dir: str, unit: str, e: int, blob: bytes, meta: dict) -> bool:
    """Atomically store one expert result. False if it was already there (a re-issued job lost the race)."""
    if expert_done(out_dir, unit, e):
        return False
    f, m = expert_paths(out_dir, unit, e)
    os.makedirs(os.path.dirname(f), exist_ok=True)
    with open(f + ".tmp", "wb") as fh:
        fh.write(blob)
    os.replace(f + ".tmp", f)
    C.atomic_write_json(m, {**meta, "size": len(blob), "crc32": zlib.crc32(blob)})   # marker last
    return True


def P_exl3_bytes(in_f, out_f, k):
    import plan as P
    return P.exl3_bytes(in_f, out_f, k)


def cand_paths(out_dir: str, unit: str, e: int, k: int):
    d = os.path.join(out_dir, "cands", unit)
    return os.path.join(d, f"e{e:03d}.k{k}.safetensors"), os.path.join(d, f"e{e:03d}.k{k}.json")


def store_cand(out_dir: str, unit: str, e: int, k: int, blob: bytes, meta: dict) -> bool:
    """Alternative-K candidate of an expert (kept for the allocation optimizer; the primary K lives in the layer
    shard). Same atomic protocol as store_expert; stored BEFORE the primary so a done primary implies its candidates."""
    f, m = cand_paths(out_dir, unit, e, k)
    try:
        if os.path.exists(f) and os.path.getsize(f) == json.load(open(m))["size"]:
            return False
    except (ValueError, KeyError, OSError):
        pass
    os.makedirs(os.path.dirname(f), exist_ok=True)
    with open(f + ".tmp", "wb") as fh:
        fh.write(blob)
    os.replace(f + ".tmp", f)
    C.atomic_write_json(m, {**meta, "size": len(blob), "crc32": zlib.crc32(blob)})
    return True


def routing_stats_path(out_dir: str, unit: str) -> str:
    return os.path.join(out_dir, "stats", f"{unit}.routing.json")


def zipf_stats(unit: str, n_exp: int = 384, n_tokens: int = 262144, topk: int = 6) -> dict:
    """Deterministic stand-in for the capture's routing statistics (synthetic / uncalibrated runs, tests): a Zipf-like
    skew over a per-layer permutation of the experts, sum of counts = n_tokens * topk."""
    import random
    rnd = random.Random(zlib.crc32(unit.encode()))
    order = list(range(n_exp))
    rnd.shuffle(order)
    wts = [0.0] * n_exp
    for rank, e in enumerate(order):
        wts[e] = 1.0 / (rank + 3.0) ** 1.0
    tot = sum(wts)
    counts = [int(round(w / tot * n_tokens * topk)) for w in wts]
    return {"n_tokens": n_tokens, "topk": topk, "counts": counts,
            "w2sum": [c * (1.0 / topk) ** 2 for c in counts], "synthetic": True}


def choose_alts(counts, primary: dict, alt_ks, up_frac: float, down_frac: float) -> dict:
    """Which experts also get quantized at another K (candidates for the allocation optimizer).

    counts: routing count per expert of the layer (calibration capture); primary: {expert: plan K}.
    An expert gets every K of `alt_ks` that differs from its primary: higher K only if it is among the hottest
    `up_frac` of the layer (hot experts are where more bits pay: error weight and RAM-resident reads), lower K only if
    it is among the coldest `down_frac` (cheap to demote). Ties break by expert id, so the choice is deterministic."""
    experts = sorted(primary)
    order = sorted(experts, key=lambda e: (-counts[e], e))
    rank = {e: i for i, e in enumerate(order)}
    n = len(experts)
    n_up, n_down = int(round(up_frac * n)), int(round(down_frac * n))
    out = {}
    for e in experts:
        ks = []
        for k in sorted(alt_ks):
            if k > primary[e] and rank[e] < n_up:
                ks.append(k)
            elif k < primary[e] and rank[e] >= n - n_down:
                ks.append(k)
        if ks:
            out[e] = ks
    return out


def alloc_update(out_dir: str, unit: str, patch: dict):
    """Merge `patch` (top-level keys; dict values are merged one level deep) into alloc/<unit>.json."""
    d = os.path.join(out_dir, "alloc")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{unit}.json")
    cur = json.load(open(path)) if os.path.exists(path) else {"unit": unit}
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(cur.get(k), dict):
            cur[k].update(v)
        else:
            cur[k] = v
    C.atomic_write_json(path, cur)


# ----------------------------------------------------------------------------------------------
# Worker
# ----------------------------------------------------------------------------------------------

def mem_available_gib() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1048576.0
    except OSError:
        pass
    return 1e9


class Worker:
    """Pulls expert jobs, returns serialized quantized experts."""

    def __init__(self, fin, fout, log):
        self.fin, self.fout, self.log = fin, fout, log
        self.send_lock = threading.Lock()
        self.jobs = deque()
        self.cv = threading.Condition()
        self.closed = False

    def send(self, header, payload=b""):
        with self.send_lock:
            send_frame(self.fout, header, payload)

    def reader(self):
        while True:
            fr = recv_frame(self.fin)
            if fr is None:
                break
            h, payload = fr
            op = h["op"]
            if op == "job":
                with self.cv:
                    self.jobs.append((h, payload))
                    self.cv.notify()
            elif op == "layer_done":
                pass
            elif op == "bye":
                break
        with self.cv:
            self.closed = True
            self.cv.notify_all()

    def run(self):
        hello = recv_frame(self.fin)
        if hello is None:
            return
        cfg = hello[0]
        from quant import StubQuantizer
        if cfg["quantizer"] == "stub":
            self.quantizer = StubQuantizer(cfg.get("stub_delay", 0.0))
        else:
            from quant import Exl3Quantizer
            self.quantizer = Exl3Quantizer(cfg["device"], cfg.get("codebook", "mul1"))
        self.min_mem = cfg.get("min_mem_gib", 6.0)
        self.threads = max(1, cfg.get("threads", 1))
        self.qpool = ThreadPoolExecutor(max_workers=self.threads, initializer=self._thread_init)
        import socket
        self.send({"op": "hello", "host": socket.gethostname(), "pid": os.getpid(), "device": cfg["device"],
                   "torch": torch.__version__, "mem_gib": round(mem_available_gib(), 1)})
        threading.Thread(target=self.reader, daemon=True).start()
        for _ in range(cfg.get("depth", 2)):
            self.send({"op": "credit"})
        while True:
            with self.cv:
                while not self.jobs and not self.closed:
                    self.cv.wait()
                if not self.jobs:
                    break
                h, payload = self.jobs.popleft()
            while mem_available_gib() < self.min_mem:
                self.log(f"MemAvailable {mem_available_gib():.1f} GiB < {self.min_mem}: waiting")
                time.sleep(10)
            try:
                hdr, blobs = self.do_job(h, payload)
                self.send(hdr, blobs)
            except Exception as ex:  # report, the coordinator re-queues
                import traceback
                traceback.print_exc()
                self.send({"op": "error", "id": h["id"], "msg": f"{type(ex).__name__}: {ex}"})
            self.send({"op": "credit"})
            _free()

    # -- one job
    def _fetch_expert(self, fetcher, spec):
        out = {}
        for w in ("w1", "w3", "w2"):
            t = spec["tensors"][w]
            out[w] = (fetcher.fetch(t["weight"]), fetcher.fetch(t["scale"]))
        return out

    def _thread_init(self):
        init = getattr(self.quantizer, "thread_init", None)
        if init is not None and self.threads > 1:
            init()

    def _one_expert(self, fetcher, s, hs, gu, down):
        # gu / down: HessianEntry or None (uncalibrated) when hs is set
        """Fetch, dequantize and quantize one expert (runs on a compute thread). The expert is quantized at its
        primary K (`s["bits"]`) and at every alternative K of `s["alts"]` from the same dequantized weights and the
        same Hessians (candidates for the allocation optimizer). Returns [(blob, meta)], primary first."""
        import dequant as D
        from quant import quant_record
        from safetensors.torch import save as st_save
        raw = self._fetch_expert(fetcher, s)
        ks = [s["bits"]] + [k for k in s.get("alts", []) if k != s["bits"]]
        tensors = {k: {} for k in ks}
        stored = {k: {} for k in ks}
        proxy = {k: {} for k in ks}
        energy = {}
        secs = {k: 0.0 for k in ks}
        st = {"bf16_bytes": 0, "fetched_bytes": 0}
        for w in ("w1", "w3", "w2"):
            wt, sc = raw[w]
            st["fetched_bytes"] += wt.numel() * wt.element_size() + sc.numel()
            bf = D.dequant(wt, sc)
            st["bf16_bytes"] += bf.numel() * 2
            key = s["keys"][w]
            hess = (gu if w in ("w1", "w3") else down) if hs else None
            for k in ks:
                tq = time.time()
                q = self.quantizer(key, bf, k, hess)
                for suffix, t in q.items():
                    tensors[k][f"{key}.{suffix}"] = t
                stored[k][key] = quant_record(k, q)
                pe = getattr(self.quantizer, "proxy", {}).pop(key, None)
                if pe is not None:
                    proxy[k][key.rsplit(".", 1)[1]] = round(pe, 6)
                en = getattr(self.quantizer, "energy", {}).pop(key, None)
                if en is not None:
                    energy[w] = en
                secs[k] += time.time() - tq
            del bf, wt, sc
        for h_ in (gu, down):
            if h_ is not None:
                h_.drop()
        out = []
        for k in ks:
            blob = st_save(tensors[k], metadata={"format": "pt"})
            out.append((blob, {"e": s["e"], "bits": k, "stored": stored[k], "len": len(blob),
                               "quant_seconds": round(secs[k], 2), "proxy": proxy[k], "energy": energy,
                               **({"alt": True} if k != s["bits"] else {}), **st}))
        return out

    def do_job(self, h, payload):
        import fetch as F
        t0 = time.time()
        unit = h["unit"]
        pos = 0
        hs = h.get("hess")
        specs = h["experts"]
        gus, downs = {}, {}
        for s in specs:
            if hs:
                e = s["e"]
                n = hs["in_len"][str(e)]
                gus[e] = HessianEntry.from_blob(payload[pos:pos + n], hs["gu_name"])
                pos += n
                n = hs["down_len"][str(e)]
                downs[e] = HessianEntry.from_blob(payload[pos:pos + n], s["down_name"])
                pos += n
        fetcher = F.TensorFetcher({"repo": h["repo"], "revision": h["revision"], "tensors": h["tensors"]},
                                  cache=h.get("cache"))
        futs = [self.qpool.submit(self._one_expert, fetcher, s, hs, gus.get(s["e"]), downs.get(s["e"]))
                for s in specs]
        blobs, metas = [], []
        for f in futs:
            res = f.result()
            for blob, meta in res:
                blobs.append(blob)
                metas.append(meta)
        return ({"op": "result", "id": h["id"], "unit": unit, "experts": metas,
                 "seconds": round(time.time() - t0, 2), "mem_gib": round(mem_available_gib(), 1)}, blobs)


def _free():
    import gc
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def worker_main(log=None):
    """Protocol on stdin/stdout; everything else printed goes to stderr."""
    fin = sys.stdin.buffer
    fout = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    name = os.environ.get("DS41_WORKER_NAME", "worker")
    log = log or (lambda m: print(f"[{time.strftime('%H:%M:%S')}] [{name}] {m}", file=sys.stderr, flush=True))
    Worker(fin, fout, log).run()


# ----------------------------------------------------------------------------------------------
# Remote bootstrap (worker): code travels over the pipe, nothing is installed there
# ----------------------------------------------------------------------------------------------

LOADER = r'''
import struct, sys, types
n = struct.unpack("<Q", sys.stdin.buffer.read(8))[0]
buf = sys.stdin.buffer.read(n)
import io, zipfile
z = zipfile.ZipFile(io.BytesIO(buf))
for name in %(mods)r:
    m = types.ModuleType(name)
    m.__file__ = "<ds41:%%s>" %% name
    sys.modules[name] = m
    exec(compile(z.read(name + ".py"), m.__file__, "exec"), m.__dict__)
sys.modules["split"].worker_main()
'''


def code_bundle() -> bytes:
    import io
    import zipfile
    here = os.path.dirname(os.path.abspath(__file__))
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as z:
        for m in MODULES:
            z.write(os.path.join(here, m + ".py"), m + ".py")
    return bio.getvalue()


def ssh_command(host: str, python: str, pythonpath: str, device_index: int = 0, lock: str | None = None,
                name: str = "worker", env: str = "") -> str:
    """Shell command (run locally) that starts a worker on `host`. The loader script is passed as -c, the
    code zip follows on stdin (see Coordinator.start_worker)."""
    loader = LOADER % {"mods": MODULES}
    b64 = base64.b64encode(loader.encode()).decode()
    remote = (f"cd /dev/shm && {'flock -n ' + lock + ' ' if lock else ''}env PYTHONPATH={pythonpath} "
              f"CUDA_VISIBLE_DEVICES={device_index} PYTHONUNBUFFERED=1 DS41_WORKER_NAME={name} {env} "
              f"{python} -c \"import base64;exec(base64.b64decode('{b64}'))\"")
    import shlex
    return (f"ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=6 "
            f"-c aes128-gcm@openssh.com {host} {shlex.quote(remote)}")


# ----------------------------------------------------------------------------------------------
# Coordinator
# ----------------------------------------------------------------------------------------------

class StaleJob(Exception):
    pass


class Job:
    _next = 0

    def __init__(self, unit, experts):
        Job._next += 1
        self.id = Job._next
        self.unit, self.experts = unit, list(experts)
        self.t0 = None
        self.holders = set()


class LayerQueue:
    """Released-but-not-started experts, in-flight jobs and finished experts of the layer being processed."""

    def __init__(self, unit, todo, n_workers_hint, chunk):
        self.unit = unit
        self.cv = threading.Condition()
        self.pending = deque()
        self.todo = set(todo)          # experts still without a result
        self.inflight = {}
        self.chunk = chunk
        self.nw = max(1, n_workers_hint)
        self.times = []                # seconds per finished job (for the straggler rule)
        self.closed = False

    def release(self, experts):
        with self.cv:
            self.pending.extend(e for e in experts if e in self.todo)
            self.cv.notify_all()

    def take(self, worker, can_take=lambda: True, reissue_after=None, timeout=0.5):
        """Next job for `worker` or None (after `timeout`). Re-issues a straggling job when nothing is pending."""
        with self.cv:
            if self.closed or not can_take():
                self.cv.wait(timeout)
                return None
            if self.pending:
                size = max(1, min(self.chunk, -(-len(self.pending) // (2 * self.nw))))
                exps = [self.pending.popleft() for _ in range(min(size, len(self.pending)))]
                j = Job(self.unit, exps)
                j.t0 = time.time()
                j.holders.add(worker)
                self.inflight[j.id] = j
                return j
            if reissue_after is not None and self.inflight:
                med = sorted(self.times)[len(self.times) // 2] if self.times else None
                limit = reissue_after if med is None else max(reissue_after, 3 * med)
                old = [j for j in self.inflight.values() if worker not in j.holders and time.time() - j.t0 > limit]
                if old:
                    j = min(old, key=lambda j: j.t0)
                    j.holders.add(worker)
                    return j
            self.cv.wait(timeout)
            return None

    def finished(self, job, worker):
        with self.cv:
            j = self.inflight.get(job.id)
            if j is not None:
                self.times.append(time.time() - j.t0)
                del self.inflight[job.id]
            for e in job.experts:
                self.todo.discard(e)
            self.cv.notify_all()

    def requeue(self, job, worker):
        """Worker lost or failed: back to the front of the queue unless another holder still runs it."""
        with self.cv:
            j = self.inflight.get(job.id)
            if j is None:
                return
            j.holders.discard(worker)
            if not j.holders:
                del self.inflight[job.id]
                self.pending.extendleft(reversed([e for e in job.experts if e in self.todo]))
            self.cv.notify_all()

    def wait_all(self, alive, poll=0.5):
        with self.cv:
            while self.todo:
                self.cv.wait(poll)
                if self.todo and not alive():
                    raise RuntimeError(f"{self.unit}: {len(self.todo)} experts left and no worker alive")

    def close(self):
        with self.cv:
            self.closed = True
            self.cv.notify_all()


class WorkerSlot:
    """One worker connection (supervised: restarted on death up to `restarts` times)."""

    def __init__(self, name, cmd, coord, after_capture=False, restarts=3, retry_s=60.0, ship_code=False,
                 preflight=None):
        self.name, self.cmd, self.co = name, cmd, coord
        self.after_capture = after_capture
        self.restarts, self.retry_s = restarts, retry_s
        self.ship_code = ship_code
        self.preflight = preflight
        self.min_pending = 0
        self.stats = collections.Counter()
        self.proc = None
        self.thread = None
        self.stop = False
        self.info = {}

    def start(self):
        self.thread = threading.Thread(target=self.supervise, daemon=True)
        self.thread.start()

    def supervise(self):
        import subprocess
        attempts = 0
        while not self.stop:
            if self.preflight is not None:
                ok, why = self.preflight()
                if not ok:
                    self.co.log(f"[{self.name}] not started: {why}; retry in {self.retry_s:.0f}s")
                    time.sleep(self.retry_s)
                    continue
            self.co.log(f"[{self.name}] start: {self.cmd[:120]}")
            self.proc = subprocess.Popen("exec " + self.cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, bufsize=0, start_new_session=True)
            threading.Thread(target=self._pump_stderr, args=(self.proc,), daemon=True).start()
            try:
                self.session(self.proc)
            except Exception as ex:
                import traceback
                self.co.log(f"[{self.name}] session error: {ex}\n{traceback.format_exc()}")
            self.requeue_all()
            try:
                self.proc.kill()
            except Exception:
                pass
            self.proc.wait()
            if self.stop:
                break
            attempts += 1
            if attempts > self.restarts:
                self.co.log(f"[{self.name}] gave up after {attempts} sessions")
                return
            self.co.log(f"[{self.name}] died (rc={self.proc.returncode}), restart {attempts}/{self.restarts} "
                        f"in {self.retry_s:.0f}s")
            time.sleep(self.retry_s)

    def _pump_stderr(self, p):
        for line in iter(p.stderr.readline, b""):
            self.co.log(f"[{self.name}] {line.decode(errors='replace').rstrip()}")

    def send(self, header, payload=b""):
        with self.send_lock:
            send_frame(self.fout, header, payload)

    def requeue_all(self):
        for job in list(getattr(self, "mine", [])):
            self.co.requeue(job, self)
        self.mine = []

    def session(self, p):
        fin = os.fdopen(p.stdout.fileno(), "rb", closefd=False)
        fout = os.fdopen(p.stdin.fileno(), "wb", closefd=False)
        if self.ship_code:
            blob = code_bundle()
            fout.write(struct.pack("<Q", len(blob)) + blob)
            fout.flush()
        co = self.co
        send_frame(fout, {"op": "hello", "quantizer": co.quantizer, "device": co.device_of(self),
                          "codebook": co.codebook, "stub_delay": co.stub_delay, "depth": co.depth,
                          "min_mem_gib": co.min_mem_gib, "threads": co.threads_of(self)})
        fr = recv_frame(fin)
        if fr is None:
            raise IOError("worker exited before hello")
        self.info = fr[0]
        co.log(f"[{self.name}] hello: {self.info}")
        self.fout = fout
        self.send_lock = threading.Lock()
        self.mine = []
        self.credits = 0
        self.cv = threading.Condition()
        self.dead = False
        feeder = threading.Thread(target=self._feeder, args=(fout,), daemon=True)
        feeder.start()
        try:
            while True:
                fr = recv_frame(fin)
                if fr is None:
                    break
                h, payload = fr
                if h["op"] == "credit":
                    with self.cv:
                        self.credits += 1
                        self.cv.notify_all()
                elif h["op"] == "result":
                    self._result(h, payload)
                elif h["op"] == "error":
                    job = next((j for j in self.mine if j.id == h["id"]), None)
                    self.co.log(f"[{self.name}] job {h['id']} failed: {h['msg']}")
                    self.stats["errors"] += 1
                    if job is not None:
                        self.mine.remove(job)
                        self.co.requeue(job, self)
                        if self.stats["errors"] >= 3:
                            raise RuntimeError("too many job errors")
        finally:
            self.dead = True
            with self.cv:
                self.cv.notify_all()
            try:
                self.send({"op": "bye"})
            except Exception:
                pass

    def _feeder(self, fout):
        co = self.co
        while not self.dead and not self.stop:
            with self.cv:
                while self.credits <= 0 and not self.dead:
                    self.cv.wait(0.5)
                if self.dead:
                    return
            q = co.queue
            if q is None:
                time.sleep(0.2)
                continue
            can = (lambda: ((not self.after_capture) or co.capture_finished(q.unit))
                   and (not self.min_pending or len(q.pending) >= self.min_pending))
            job = q.take(self, can, co.reissue_after)
            if job is None:
                continue
            with self.cv:
                self.credits -= 1
            try:
                header, payload = co.build_job(job, self)
                self.mine.append(job)
                self.send(header, payload)
                self.stats["bytes_up"] += len(payload) if not isinstance(payload, list) else sum(map(len, payload))
            except StaleJob:
                with self.cv:
                    self.credits += 1
                continue
            except Exception as ex:
                co.log(f"[{self.name}] send failed: {ex}")
                q.requeue(job, self)
                self.dead = True
                return

    def _result(self, h, payload):
        co = self.co
        job = next((j for j in self.mine if j.id == h["id"]), None)
        q = co.queue
        if q is None or q.unit != h["unit"]:   # late duplicate of an already assembled layer
            if job in self.mine:
                self.mine.remove(job)
            return
        if job is None:
            job = Job(q.unit, [m["e"] for m in h["experts"]])
        pos = 0
        new = 0
        entries = []
        for m in h["experts"]:
            entries.append((m, payload[pos:pos + m["len"]]))
            pos += m["len"]
        # alternative-K candidates first, the primary last: the primary marker is what "expert done" means
        for m, blob in sorted(entries, key=lambda x: not x[0].get("alt")):
            meta = {k: m[k] for k in ("e", "bits", "stored", "bf16_bytes", "fetched_bytes", "quant_seconds", "proxy",
                                      "energy") if k in m}
            meta.update({"worker": self.name, "device": self.info.get("device"), "host": self.info.get("host")})
            if m.get("alt"):
                if store_cand(co.out_dir, q.unit, m["e"], m["bits"], blob, meta):
                    self.stats["cands"] += 1
                continue
            if store_expert(co.out_dir, q.unit, m["e"], blob, meta):
                new += 1
                self.stats["experts"] += 1
                co.drop_expert_hessians(q.unit, m["e"])
        self.stats["bytes_down"] += len(payload)
        self.stats["seconds"] += h.get("seconds", 0)
        if job in self.mine:
            self.mine.remove(job)
        q.finished(job, self)


# ----------------------------------------------------------------------------------------------
# RAM-fit allocation at queue time (ramfit.py): K per routed expert from the layer's routing statistics
# ----------------------------------------------------------------------------------------------

def ramfit_path(out_dir: str, unit: str) -> str:
    return os.path.join(out_dir, "ramfit", f"{unit}.json")


def expert_index(key: str) -> int:
    return int(key.split(".")[4])


def apply_ramfit(entries: list, ks: list):
    """Impose the per-expert K of one layer (K = 0: pruned) on the plan entries IN PLACE (the entries are the dicts of
    plan["entries"], so the finalize step sees them): bits and out_bytes of the three linears, pruned experts become
    action "pruned" (no tensors, no job), and the layer's router bias entries record the pruned set (the copy step writes
    -1e4 there so the router never selects them). Idempotent. Returns the pruned expert ids."""
    pruned = sorted(e for e, k in enumerate(ks) if k == 0)
    for ent in entries:
        if ent["kind"] != "routed" or ent["action"] not in ("quant", "pruned") or ent.get("tier") == "mtp":
            continue
        k = int(ks[expert_index(ent["key"])])
        if k == 0:
            ent["action"], ent["bits"], ent["out_bytes"], ent["tier"] = "pruned", 0, 0, "pruned"
            ent.pop("alt_bits", None)
        else:
            ent["action"], ent["bits"], ent["tier"] = "quant", k, "ramfit"
            ent["out_bytes"] = P_exl3_bytes(ent["in"], ent["out"], k)
    for ent in entries:
        if ent.get("action") == "copy" and (ent["name"].endswith(".ffn.gate.bias")
                                            or ent["name"].endswith(".ffn.gate.bias_vl")):
            if pruned:
                ent["mask_experts"] = pruned
            else:
                ent.pop("mask_experts", None)
    return pruned


def ramfit_replay(plan: dict, out_dir: str, log=print) -> int:
    """Re-apply the persisted per-layer decisions (a restarted coordinator keeps the K of experts already quantized)."""
    d = os.path.join(out_dir, "ramfit")
    if not os.path.isdir(d):
        return 0
    by_unit = {}
    for e in plan["entries"]:
        by_unit.setdefault(e["unit"], []).append(e)
    n = 0
    for fn in sorted(os.listdir(d)):
        if fn.startswith("layers.") and fn.endswith(".json") and fn[:-5] in by_unit:
            apply_ramfit(by_unit[fn[:-5]], json.load(open(os.path.join(d, fn)))["ks"])
            n += 1
    if n:
        plan["summary"] = P_summarize(plan["entries"])
        log(f"ramfit: replayed the decisions of {n} layers")
    return n


def P_summarize(entries):
    import plan as P
    return P.summarize(entries)


class Coordinator:
    def __init__(self, a, plan, tmap, S, log):
        self.a, self.plan, self.tmap, self.S, self.log = a, plan, tmap, S, log
        self.out_dir = a.out
        self.queue = None
        self.spool = Spool(a.spool or os.path.join(a.out, "spool"))
        self.hess_mode = a.hessians
        self.quantizer = a.quantizer
        self.codebook = a.codebook
        self.stub_delay = a.stub_delay
        self.depth = a.depth
        self.chunk = a.chunk
        self.reissue_after = a.reissue_after
        self.min_mem_gib = a.min_mem_gib
        self.alt_ks = [int(x) for x in (getattr(a, "alt_experts", None) or "").split(",") if x.strip()]
        self.ramfit = json.load(open(a.ramfit)) if getattr(a, "ramfit", None) else None
        if self.ramfit and self.alt_ks:
            log("ramfit: --alt-experts ignored (the K of every expert is chosen at queue time, no candidates)")
            self.alt_ks = []
        self.slots = []
        self.cur = None   # per-layer expert table
        self.qlock = threading.Lock()

    # -- workers
    def threads_of(self, slot):
        return getattr(slot, "threads", None) or self.a.worker_threads

    def device_of(self, slot):
        return getattr(slot, "device", self.a.device)

    def add_slots(self):
        for spec in self.a.worker or []:
            name, _, cmd = spec.partition("=")
            late = name.startswith("late:")
            name = name.removeprefix("late:")
            name, *opts = name.split(",")     # NAME[,threads:N][,gate:N]  (gate: take jobs only while >= N experts wait)
            s = WorkerSlot(name, cmd, self, after_capture=late, restarts=self.a.worker_restarts,
                           retry_s=self.a.retry_s)
            s.device = self.a.device
            for o in opts:
                k, _, v = o.partition(":")
                if k == "threads":
                    s.threads = int(v)
                elif k == "gate":
                    s.min_pending = int(v)
                elif k == "device":
                    s.device = v
            self.slots.append(s)
        for spec in self.a.worker_ssh or []:
            # host[:device_index]  e.g. worker:0
            host, _, idx = spec.partition(":")
            cmd = ssh_command(host, self.a.ssh_python, self.a.ssh_pythonpath, int(idx or 0), self.a.ssh_lock, host,
                              self.a.ssh_env)
            s = WorkerSlot(host, cmd, self, restarts=self.a.ssh_restarts, retry_s=self.a.ssh_retry_s,
                           ship_code=True, preflight=lambda h=host: self.ssh_preflight(h))
            s.device = "cuda:0"
            self.slots.append(s)
        if not self.slots:
            raise SystemExit("coordinator: give at least one --worker or --worker-ssh")
        for s in self.slots:
            s.start()

    def ssh_preflight(self, host):
        """Never touch a GPU someone else is using; never start with little host RAM."""
        import subprocess
        try:
            r = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host,
                 "nvidia-smi --query-compute-apps=pid,name --format=csv,noheader; "
                 "echo MEM $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)"],
                capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
        except Exception as ex:
            return False, f"ssh failed: {ex}"
        lines = [x for x in r.stdout.splitlines() if x.strip()]
        mem = next((int(x.split()[1]) for x in lines if x.startswith("MEM")), 0)
        apps = [x for x in lines if not x.startswith("MEM")]
        if apps:
            return False, f"GPU busy: {apps[0]}"
        if mem < self.a.ssh_min_mem_mib:
            return False, f"MemAvailable {mem} MiB"
        return True, ""

    def drop_expert_hessians(self, unit, e):
        """The result is on disk: the expert's Hessians are no longer needed (bounds the spool)."""
        if self.a.keep_spool:
            return
        for kind in ("in", "down"):
            try:
                os.remove(self.spool.path(unit, f"{kind}.{e}"))
            except OSError:
                pass

    def capture_finished(self, unit):
        return self.hess_mode == "none" or self.spool.capture_done(unit)

    def requeue(self, job, slot):
        if self.queue is not None:
            self.queue.requeue(job, slot)

    def shutdown(self):
        for s in self.slots:
            s.stop = True
            try:
                s.send({"op": "bye"})
                s.proc.stdin.close()
            except Exception:
                pass
        for s in self.slots:
            if s.thread:
                s.thread.join(10)
            try:
                s.proc.kill()
            except Exception:
                pass

    # -- job construction
    def build_job(self, job, slot):
        cur = self.cur
        if cur is None or cur["unit"] != job.unit:
            raise StaleJob(job.unit)   # a re-issued duplicate that outlived its layer
        specs, tensors = [], {}
        hess = None
        payload = []
        for e in job.experts:
            ent = cur["experts"][e]
            keys = {w: ent[w]["key"] for w in ("w1", "w2", "w3")}
            tinfo = {}
            for w in ("w1", "w2", "w3"):
                for nm in (ent[w]["weight"], ent[w]["scale"]):
                    tensors[nm] = self.tmap["tensors"][nm]
                tinfo[w] = {"weight": ent[w]["weight"], "scale": ent[w]["scale"]}
            specs.append({"e": e, "bits": ent["w1"]["bits"], "keys": keys, "tensors": tinfo,
                          "down_name": f"{cur['unit']}.ffn.experts.{e}.w2",
                          "alts": cur.get("alts", {}).get(e, [])})
        hdr = {"op": "job", "id": job.id, "unit": cur["unit"], "experts": specs, "tensors": tensors,
               "repo": self.tmap.get("repo", C.REPO), "revision": self.tmap.get("revision", C.REVISION)}
        if self.a.cache:
            hdr["cache"] = self.a.cache
        if self.hess_mode != "none":
            hs = {"gu_name": f"{cur['unit']}.ffn.experts.gu", "in_len": {}, "down_len": {}}
            for e in job.experts:
                for kind, key in (("in", "in_len"), ("down", "down_len")):
                    b = self.spool.read(cur["unit"], f"{kind}.{e}")
                    hs[key][str(e)] = len(b)
                    payload.append(b)
            hdr["hess"] = hs
        return hdr, payload

    # -- one layer
    def run_layer(self, unit, entries):
        S = self.S
        t0 = time.time()
        if self.ramfit and unit.startswith("layers."):
            self.ramfit_layer(unit, entries)
        routed = [e for e in entries if e["kind"] == "routed" and e["action"] == "quant"]
        table = OrderedDict()
        for e in routed:
            idx = int(e["key"].split(".")[4])
            table.setdefault(idx, {})[e["key"].rsplit(".", 1)[1]] = e
        todo = [e for e in table if not expert_done(self.out_dir, unit, e)]
        self.log(f"{unit}: {len(table)} experts, {len(table) - len(todo)} already done")
        alts = {}
        if self.alt_ks and todo and unit.startswith("layers."):     # not the MTP experts (fixed 4 bit)
            st = self.routing_stats(unit, len(table))
            if st is not None:
                prim = {e: table[e]["w1"]["bits"] for e in table}
                alts = {e: ks for e, ks in choose_alts(st["counts"], prim, self.alt_ks, self.a.alt_up_frac,
                                                       self.a.alt_down_frac).items() if e in set(todo)}
                self.log(f"{unit}: alternative-K candidates for {len(alts)}/{len(todo)} experts "
                         f"({sum(len(v) for v in alts.values())} extra quantizations)")
            else:
                self.log(f"{unit}: WARNING no routing statistics, no alternative candidates for this layer")
        pruned_ids = [expert_index(e["key"]) for e in entries if e.get("action") == "pruned" and e["key"].endswith(".w1")]
        self.cur = {"unit": unit, "experts": table, "alts": alts, "pruned": pruned_ids}
        q = LayerQueue(unit, todo, len(self.slots) or 1, self.chunk)
        released = set()

        def hess_ready():
            """Release the experts whose gate/up and down Hessians are in the spool."""
            for pe in pruned_ids:        # the capture still emits their Hessians: nobody needs them, keep the spool small
                if self.spool.has(unit, f"in.{pe}") or self.spool.has(unit, f"down.{pe}"):
                    self.drop_expert_hessians(unit, pe)
            if self.hess_mode == "none":
                if not released:
                    released.update(todo)
                    q.release(todo)
                return
            new = [e for e in todo if e not in released and self.spool.has(unit, f"in.{e}")
                   and self.spool.has(unit, f"down.{e}")]
            if new:
                released.update(new)
                q.release(new)

        feeder = None
        if self.hess_mode == "synthetic" and todo:
            dims = (next(e["in"] for e in routed if e["key"].endswith(".w1")),
                    next(e["in"] for e in routed if e["key"].endswith(".w2")))
            lins = [(e["key"], e["in"]) for e in entries if e["kind"] != "routed" and e["action"] == "quant"]
            feeder = threading.Thread(target=synthetic_fill, args=(
                self.spool, unit, todo, dims, self.a.synthetic_slices, self.a.synthetic_gap, self.log, lins),
                daemon=True)
            feeder.start()
        stop = threading.Event()

        def release_loop():
            while not stop.is_set():
                hess_ready()
                time.sleep(0.2)

        hess_ready()
        rl = threading.Thread(target=release_loop, daemon=True)
        rl.start()
        self.queue = q
        asm = None
        try:
            alive = lambda: any(sl.thread.is_alive() for sl in self.slots)
            if self.a.experts_only:   # benchmark / smoke: no assembly, the unit stays unfinished
                q.wait_all(alive)
            else:
                # the non-expert linears are quantized and the shard is assembled in the background, so the
                # workers move on to the next layer as soon as the last expert of this one is back
                asm = AsmThread(self, unit, entries, table, q, t0)
                asm.start()
                try:
                    q.wait_all(alive)
                except Exception:
                    if asm.error is not None:
                        raise asm.error
                    raise
        finally:
            stop.set()
            q.close()
            self.queue = None
            for s in self.slots:
                try:
                    s.send({"op": "layer_done", "unit": unit})
                except Exception:
                    pass
        tot = {s.name: dict(s.stats) for s in self.slots}
        self.log(f"{unit}: experts done in {time.time() - t0:.1f}s, workers {tot}")
        return asm

    def ramfit_layer(self, unit, entries):
        """Decide K (1/2/3/..., 0 = pruned) of every routed expert of this layer from its routing statistics, so that the
        layer's experts fill the byte budget of the ramfit plan with the least proxy error (ramfit.decide_layer). Runs once
        per layer: the decision is persisted (ramfit/<unit>.json) and replayed after a restart."""
        import ramfit as RF
        path = ramfit_path(self.out_dir, unit)
        if os.path.exists(path):
            ks = json.load(open(path))["ks"]
            apply_ramfit(entries, ks)
            return
        cfg = self.ramfit["experts"]
        n = len([e for e in entries if e["kind"] == "routed" and e.get("tier") != "mtp" and e["key"].endswith(".w1")])
        st = self.routing_stats(unit, n)
        layer = int(unit.split(".")[1])
        w = (cfg.get("layer_weights") or {}).get(str(layer), 1.0)
        budget = float(cfg["bytes_per_expert"]) * n * float(w)
        smooth = float(getattr(self.a, "ramfit_smooth", 0.02))
        tau = getattr(self.a, "ramfit_prune_tau", None)
        tau = float(tau if tau is not None else cfg.get("prune_tau", 1e-4))
        if st is not None and sum(st["w2sum"]) > 0:
            w2 = np.asarray(st["w2sum"], dtype=np.float64)
            share = w2 + smooth * w2.mean()                           # unseen is not the same as never used
            ntok = float(st.get("n_tokens") or max(1.0, sum(st["counts"]) / float(st.get("topk", 6))))
            frac = np.asarray(st["counts"], dtype=np.float64) / max(1.0, ntok)      # share of tokens routed to e
            prunable = frac < tau if cfg.get("prune") else None
            how = "routing statistics" + (" (synthetic stand-in)" if st.get("synthetic") else "")
        else:
            share = np.ones(n)
            prunable = None
            how = "NO routing statistics: equal shares"
        ks_set = tuple(int(k) for k in cfg["ks"])
        w1 = next(e for e in entries if e["kind"] == "routed" and e.get("tier") != "mtp" and e["key"].endswith(".w1"))
        ks, used, d = RF.decide_layer(share, budget, ks_set, float(cfg.get("prune_cost", 1.3)),
                                      prunable=prunable if cfg.get("prune") else None, dims=(w1["in"], w1["out"]))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        C.atomic_write_json(path, {"ks": ks, "bytes": used, "budget": budget, "D": d, "how": how})
        pruned = apply_ramfit(entries, ks)
        for e in pruned:
            self.drop_expert_hessians(unit, e)
        hist = {k: ks.count(k) for k in sorted(set(ks))}
        self.log(f"{unit}: ramfit K {hist} from {how}, {used / 1e9:.2f} of {budget / 1e9:.2f} GB, proxy D {d * 100:.2f}%")

    def routing_stats(self, unit, n_exp):
        """The capture's routing statistics of the layer. The capture writes them BEFORE it emits the layer's first Hessian,
        so once any Hessian of the layer is in the spool and the file is still missing, the recorder is broken: give up
        at once (and for every later layer) instead of stalling the conversion; the pack is then simply built without
        alternative candidates. Synthetic / uncalibrated runs generate a deterministic Zipf stand-in."""
        path = routing_stats_path(self.out_dir, unit)
        t0 = time.time()
        seen = None
        while not os.path.exists(path):
            if self.hess_mode in ("synthetic", "none"):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                C.atomic_write_json(path, zipf_stats(unit, n_exp))
                break
            if getattr(self, "stats_dead", False):
                return None
            d = self.spool.dir(unit)
            if os.path.isdir(d) and any(not f.endswith(".tmp") for f in os.listdir(d)):
                seen = seen or time.time()
                if time.time() - seen > 20.0:
                    self.stats_dead = True
                    self.log(f"{unit}: Hessians are being captured but there are no routing statistics: "
                             f"alternative candidates disabled for the rest of the run")
                    return None
            if time.time() - t0 > self.a.stats_timeout:
                self.stats_dead = True
                self.log(f"{unit}: no routing statistics after {self.a.stats_timeout:.0f}s: candidates disabled")
                return None
            time.sleep(1.0)
        return json.load(open(path))

    def local_and_merge(self, unit, entries, table, q, t0):
        S = self.S
        local = [e for e in entries if e["kind"] != "routed"]
        order = [e.get("key") or e["name"] for e in entries]
        hq = HessQuantizer(self.local_quantizer, self.spool, unit, self.hess_mode, self.log, self.a.hess_timeout)

        def merge(tensors, stored, stats):
            q.wait_all(lambda: any(sl.thread.is_alive() for sl in self.slots))
            from safetensors.torch import load as st_load
            prox = {}
            for e, ent in table.items():
                f, m = expert_paths(self.out_dir, unit, e)
                meta = json.load(open(m))
                with open(f, "rb") as fh:
                    blob = fh.read()
                if zlib.crc32(blob) != meta["crc32"]:
                    raise IOError(f"{f}: CRC mismatch")
                tensors.update(st_load(blob))
                stored.update(meta["stored"])
                for w, v in (meta.get("proxy") or {}).items():
                    prox.setdefault((meta["bits"], w), []).append((v, e))
                stats["bf16_bytes"] += meta["bf16_bytes"]
                stats["fetched_bytes"] += meta["fetched_bytes"]
                stats["planned_bytes"] += sum(ent[w]["out_bytes"] for w in ent)
                stats["tensors"] += len(ent)
            self.report_proxy(unit, prox)
            try:
                self.write_expert_table(unit, table)
            except Exception as ex:   # allocation statistics must never fail a layer
                import traceback
                self.log(f"{unit}: WARNING allocation table not written: {ex}\n{traceback.format_exc()}")
            ordered = OrderedDict((k, stored[k]) for k in order if k in stored)
            ordered.update((k, v) for k, v in stored.items() if k not in ordered)
            stored.clear()
            stored.update(ordered)

        S.process_unit(unit, local, S_fetcher(self), self.out_dir, False, hq, self.a.prefetch, self.log, merge=merge)
        # expert files are kept until the whole run is finalized (cheap, and the unit file can be rebuilt)

    def write_expert_table(self, unit, table):
        """alloc/<unit>.json, experts part: per expert the routing count and, per candidate K, the proxy error of
        w1/w3/w2, the bytes and where the tensors are (shard = primary, cands = alternative file)."""
        sp = routing_stats_path(self.out_dir, unit)
        st = json.load(open(sp)) if os.path.exists(sp) else None
        tab = {}
        for e, ent in table.items():
            _, mp = expert_paths(self.out_dir, unit, e)
            meta = json.load(open(mp))
            kinfo = {}
            nbytes = lambda k: sum(P_exl3_bytes(ent[w]["in"], ent[w]["out"], k) for w in ("w1", "w2", "w3"))
            kinfo[str(meta["bits"])] = {"proxy": meta.get("proxy") or {}, "bytes": nbytes(meta["bits"]),
                                        "where": "shard"}
            for k in (1, 2, 3, 4, 5, 6):
                if k == meta["bits"]:
                    continue
                _, cm = cand_paths(self.out_dir, unit, e, k)
                if os.path.exists(cm):
                    cmeta = json.load(open(cm))
                    kinfo[str(k)] = {"proxy": cmeta.get("proxy") or {}, "bytes": nbytes(k), "where": "cand"}
            row = {"E": meta.get("energy") or {}, "k": kinfo}
            if st is not None:
                row["count"] = st["counts"][e]
                row["w2sum"] = st["w2sum"][e]
            tab[str(e)] = row
        patch = {"experts": tab}
        if st is not None:
            patch["stats"] = {k: v for k, v in st.items() if k not in ("counts", "w2sum")}
        alloc_update(self.out_dir, unit, patch)

    def report_proxy(self, unit, prox):
        """Per bit width and projection: median proxy error, and a warning for outliers (> 3x median), the
        symptom of a corrupted quantization (seen when compute threads ran off their stream)."""
        for (bits, w), vals in sorted(prox.items()):
            v = sorted(x for x, _ in vals)
            med = v[len(v) // 2]
            bad = [e for x, e in vals if med > 0 and x > 3 * med]
            self.log(f"{unit}: proxy K={bits} {w}: median {med:.5f} max {v[-1]:.5f} n={len(v)}"
                     + (f"  WARNING {len(bad)} outliers > 3x median: experts {sorted(bad)[:12]}" if bad else ""))

    def cleanup_layer(self, unit):
        keep = os.path.join(self.out_dir, "hess_keep", unit)    # attention/shared Hessians: lets a later pass redo them
        d0 = self.spool.dir(unit)
        if os.path.isdir(d0) and self.a.keep_lin_hess:
            os.makedirs(keep, exist_ok=True)
            for fn in os.listdir(d0):
                if fn.startswith("lin.") and not os.path.exists(os.path.join(keep, fn)):
                    try:
                        os.link(os.path.join(d0, fn), os.path.join(keep, fn))
                    except OSError:
                        shutil.copy(os.path.join(d0, fn), os.path.join(keep, fn))
        self.spool.clear(unit)
        if self.a.cache:
            shutil.rmtree(os.path.join(self.a.cache, unit), ignore_errors=True)
        d = os.path.join(self.out_dir, "experts", unit)
        if os.path.isdir(d) and not self.a.keep_experts:
            for fn in os.listdir(d):
                os.remove(os.path.join(d, fn))
            os.rmdir(d)


def S_fetcher(co):
    import fetch as F
    return F.TensorFetcher(co.tmap, cache=co.a.cache)


class AsmThread(threading.Thread):
    """Quantizes the non-expert linears of a layer and assembles its shard (runs next to the expert queue)."""

    def __init__(self, co, unit, entries, table, q, t0):
        super().__init__(daemon=True)
        self.co, self.unit, self.entries, self.table, self.q, self.t0 = co, unit, entries, table, q, t0
        self.error = None

    def run(self):
        try:
            self.co.local_and_merge(self.unit, self.entries, self.table, self.q, self.t0)
            self.co.cleanup_layer(self.unit)
            self.co.log(f"{self.unit}: assembled, {time.time() - self.t0:.0f}s since start")
        except BaseException as ex:  # noqa: BLE001
            import traceback
            self.error = ex
            self.co.log(f"{self.unit}: assembly failed: {ex}\n{traceback.format_exc()}")


class HessQuantizer:
    """Local (coordinator-side) quantizer for the non-expert linears, with Hessians from the spool."""

    def __init__(self, base, spool, unit, mode, log, timeout):
        self.base, self.spool, self.unit, self.mode, self.log, self.timeout = base, spool, unit, mode, log, timeout
        self.cache = {}

    def __call__(self, key, w, bits, hess=None):
        if self.mode == "none":
            return self.base(key, w, bits, None)
        name = f"lin.{key}"
        t0 = time.time()
        while not self.spool.has(self.unit, name):
            if time.time() - t0 > self.timeout:
                raise TimeoutError(f"{self.unit}: no Hessian '{name}' in the spool after {self.timeout:.0f}s")
            time.sleep(0.5)
        if name not in self.cache:
            self.cache[name] = self.spool.get(self.unit, name)
        return self.base(key, w, bits, self.cache[name])


def coordinate(a, plan, tmap, sel, by_unit, S, log):
    """Process the selected units: routed layers through the work queue, the rest locally."""
    if a.quantizer == "stub":
        from quant import StubQuantizer
        local_q = StubQuantizer(a.stub_delay)
    elif a.experts_only:
        local_q = None
    else:
        from quant import Exl3Quantizer
        local_q = Exl3Quantizer(a.local_device or a.device, a.codebook)
    co = Coordinator(a, plan, tmap, S, log)
    if co.ramfit:
        ramfit_replay(plan, a.out, log)
        by_unit.clear()
        for e in plan["entries"]:
            by_unit.setdefault(e["unit"], []).append(e)
    co.local_quantizer = local_q
    co.add_slots()
    asms = []
    try:
        for u in sel:
            if S.unit_done(u, a.out, False):
                log(f"{u}: done, skipping")
                continue
            ents = by_unit[u]
            if any(e["kind"] == "routed" for e in ents):
                t = co.run_layer(u, ents)     # returns once every expert is back; the shard assembles in the background
                if t is not None:
                    asms.append(t)
                while len(asms) > 1:          # keep one assembly overlapping the next layer's experts
                    t0_ = asms.pop(0)
                    t0_.join()
                    if t0_.error is not None:
                        raise t0_.error
            else:
                import fetch as F
                unc = u == "global" or u.startswith("mtp.")   # no calibration capture for these: meta Hessian
                S.process_unit(u, ents, F.TensorFetcher(tmap), a.out, False,
                               HessQuantizer(local_q, co.spool, u, "none" if unc else a.hessians, log,
                                             a.hess_timeout),
                               a.prefetch, log)
        for t in asms:
            t.join()
            if t.error is not None:
                raise t.error
    finally:
        co.shutdown()
