"""Calibration capture for the DS4.1 conversion: writes the Hessians of every linear to the spool.

    python capture.py --map tensor_map.json --config config.json --calib calib.npy --rows 128 \
        --out OUT --cache CACHE --engram-subset DIR --device cuda:0

Runs the reference forward (StreamCalibrator of the ds41 port) layer by layer over the calibration rows and writes
`<spool>/<unit>/{lin.<key>,in.<e>,down.<e>}.h` + `CAPTURE_DONE` for each layer. The rows are advanced through the
REFERENCE (unquantized) layer in the same pass (capture and advance in one forward), so capture of layer i+1 never
waits for the quantization of layer i: the quantizers lag behind and the spool is bounded by `--lead`.

Source tensors come from the Hugging Face repo with Range requests into a per-layer disk cache (`--cache`), which the
split workers share. A prefetch thread keeps the next `--prefetch-layers` layers downloaded.
"""
import argparse
import json
import os
import re
import shutil
import sys
import threading
import time
import zlib
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
import fetch as F  # noqa: E402
import split as SP  # noqa: E402

LOG = lambda m: print(f"[{time.strftime('%H:%M:%S')}] [capture] {m}", flush=True)


# ---------------------------------------------------------------------------------------------- Hessian blobs
def pack_gpu(H: torch.Tensor, count: int, num_total: int, inf_nan, group: str) -> bytes:
    """Same format as split.pack_hessian (upper triangle, row by row) but gathers on the device."""
    import struct
    n = H.shape[0]
    hdr = json.dumps({"n": n, "count": int(count), "num_total": int(num_total), "group": group,
                      "inf_nan": [int(inf_nan[0]), int(inf_nan[1])], "fmt": "tri_f32"}).encode()
    mask = torch.ones(n, n, dtype=torch.bool, device=H.device).triu()
    tri = H[mask].cpu().numpy().tobytes()           # row-major order of the upper triangle
    body = struct.pack("<I", len(hdr)) + hdr + tri
    return body + struct.pack("<I", zlib.crc32(body))


NOH = b"NOH"   # "no Hessian" marker (expert the router never used): quantized uncalibrated


def spool_bytes(root: str) -> int:
    tot = 0
    for d in os.scandir(root):
        if d.is_dir():
            for f in os.scandir(d.path):
                try:
                    tot += f.stat().st_size
                except OSError:
                    pass
    return tot


class SpoolSink:
    """StreamCalibrator sink -> spool files."""

    def __init__(self, spool: SP.Spool, unit: str, prefix: str, log, cap_bytes: int = 0):
        self.spool, self.unit, self.prefix, self.log, self.cap = spool, unit, prefix, log, cap_bytes
        self.n = 0
        self.bytes = 0
        self.none = 0
        self.counts = {}

    def _put(self, name, blob):
        while self.cap and spool_bytes(self.spool.root) > self.cap:   # backpressure: quantizers lag behind
            time.sleep(2)
        self.spool.put(self.unit, name, blob)
        self.bytes += len(blob)

    def emit(self, group, linears, H):
        m = re.match(r"ffn\.experts\.(\d+)\.(in|mid)$", group)
        if m:
            e, kind = int(m.group(1)), m.group(2)
            name = f"in.{e}" if kind == "in" else f"down.{e}"
            if H is None:
                self.none += 1
                self._put(name, NOH)
            else:
                self._put(name, pack_gpu(H["H"], H["count"], H["num_total"], H["inf_nan"].tolist(),
                                         f"{self.unit}.ffn.experts.gu" if kind == "in" else
                                         f"{self.unit}.ffn.experts.{e}.w2"))
                self.counts[name] = int(H["count"])
            self.n += 1
            return
        if H is None:
            self.log(f"{self.unit}: group {group} has no Hessian (stored as NOH)")
        names = [f"lin.{self.prefix}.{ln}" for ln in linears if not ln.startswith("engram.")]
        if not names:
            return
        blob = NOH if H is None else pack_gpu(H["H"], H["count"], H["num_total"], H["inf_nan"].tolist(),
                                              f"{self.prefix}.{group}")
        first = None
        for nm in names:
            if first is None:
                self._put(nm, blob)
                first = self.spool.path(self.unit, nm)
            else:   # linears that share an input share the file (and the group name -> the same sign vector)
                dst = self.spool.path(self.unit, nm)
                if os.path.exists(dst):
                    os.remove(dst)
                os.link(first, dst)
        self.n += 1


# ---------------------------------------------------------------------------------------------- routing statistics
def routing_stats(wts: torch.Tensor, idx: torch.Tensor, n_experts: int, n_tokens: int) -> dict:
    """Per-layer routing statistics of the calibration rows: for every routed expert the number of tokens the router
    sent to it (`counts`) and the sum of the squared routing weights of those tokens (`w2sum`, the gate scaling of
    the expert's output energy). P(token picks e) = counts[e] / n_tokens; sum over e = topk."""
    flat = idx.reshape(-1).long()
    counts = torch.bincount(flat, minlength=n_experts)
    w2 = torch.zeros(n_experts, dtype=torch.float64, device=flat.device)
    w2.index_add_(0, flat, (wts.reshape(-1).double()) ** 2)
    return {"n_tokens": int(n_tokens), "topk": int(idx.shape[-1]), "counts": [int(c) for c in counts.tolist()],
            "w2sum": [float(v) for v in w2.tolist()], "moe_out_energy": None}


class RoutingRecorder:
    """Writes `<out>/stats/<unit>.routing.json` the moment the router has run on a layer (before the layer's first
    expert Hessian is emitted, so the coordinator can pick the alternative-K candidates), and adds the mean output
    energy of the MoE block (routed + shared) when the layer is done. Hooks the reference router
    (`exllamav3.dsv41.calib.moe_route`) and the calibrator's `_moe`; neither changes any numerics."""

    def __init__(self, out_dir: str):
        self.dir = os.path.join(out_dir, "stats")
        os.makedirs(self.dir, exist_ok=True)
        self.unit = None
        self.skip_existing = False

    def path(self, unit):
        return os.path.join(self.dir, f"{unit}.routing.json")

    def begin(self, unit, skip_existing=False):
        self.unit, self.skip_existing, self.cur = unit, skip_existing, None

    def end(self):
        self.unit = None

    def wrap_route(self, orig):
        def moe_route(cfg, x, gate_w, gate_b, topk):
            wts, idx = orig(cfg, x, gate_w, gate_b, topk)
            if self.unit is not None and not (self.skip_existing and os.path.exists(self.path(self.unit))):
                self.cur = routing_stats(wts, idx, gate_w.shape[0], x.shape[0])
                C.atomic_write_json(self.path(self.unit), self.cur)
            return wts, idx
        return moe_route

    def wrap_moe(self, orig):
        def _moe(*a, **k):
            y = orig(*a, **k)
            if self.unit is not None and self.cur is not None:
                # chunked: y is [N, d] fp32 with N = rows x tokens (~262k): no full-size temporaries
                n = torch.linalg.vector_norm(y.float(), dim=-1).double()
                self.cur["moe_out_energy"] = float((n * n).mean())
                C.atomic_write_json(self.path(self.unit), self.cur)
            return y
        return _moe

    def install(self, cal):
        from exllamav3.dsv41 import calib
        calib.moe_route = self.wrap_route(calib.moe_route)
        cal._moe = self.wrap_moe(cal._moe)


# ---------------------------------------------------------------------------------------------- weights
class FetchMap(Mapping):
    """name -> tensor (stored dtype) through the cached fetcher."""

    def __init__(self, fetcher: F.TensorFetcher):
        self.f = fetcher

    def __contains__(self, k):
        return k in self.f.map["tensors"]

    def __getitem__(self, k):
        if k not in self.f.map["tensors"]:
            raise KeyError(k)
        return self.f.fetch(k)

    def __iter__(self):
        return iter(self.f.map["tensors"])

    def __len__(self):
        return len(self.f.map["tensors"])


def make_source(fetcher, n_layers, dev):
    from exllamav3.dsv41.weights import MapWeightSource
    from exllamav3.dsv41 import quant as Q

    class DevSource(MapWeightSource):
        """MapWeightSource that decodes FP4 expert weights on the device (the scales become fp32 first)."""

        def tensor(self, name, dtype=torch.float32, device="cpu"):
            w = self._raw(name)
            sc = self._scale(name)
            if sc is None:
                return w.to(dtype).to(device)
            if w.dtype == torch.int8 and str(dev) != "cpu":
                b = w.to(dev).view(torch.uint8)
                lut = Q.fp4_lut(b.device)
                v = torch.stack([lut[(b & 0x0F).long()], lut[(b >> 4).long()]], dim=-1).flatten(-2)
                s = Q.e8m0_to_float(sc).to(dev).repeat_interleave(32, -1)
                out = v * s
            else:
                out = self._decode(w, sc)
                if name.endswith("wo_a"):
                    out = out.to(torch.bfloat16)
            return out.to(dtype).to(device)

    return DevSource(FetchMap(fetcher), n_layers)


class SubsetEngramStore:
    """Engram rows of the calibration set only (engram_subset.py): rows.<p>.npy + data.<p>.bin."""
    fmt = "mxfp4"

    def __init__(self, directory, n_layers_pos):
        self.rows, self.data = [], []
        for p in range(n_layers_pos):
            r = np.load(os.path.join(directory, f"rows.{p}.npy"))
            d = np.fromfile(os.path.join(directory, f"data.{p}.bin"), dtype=np.uint8).reshape(-1, 136)
            assert d.shape[0] == r.shape[0]
            self.rows.append(torch.from_numpy(r))
            self.data.append(torch.from_numpy(d))

    def num_rows(self, p):
        return 1 << 40

    def fetch(self, p, rows):
        r = rows.cpu()
        idx = torch.searchsorted(self.rows[p], r)
        idx.clamp_(max=self.rows[p].numel() - 1)
        assert bool((self.rows[p][idx] == r).all()), "engram row not in the subset (different calibration rows?)"
        return self.data[p][idx]


# ---------------------------------------------------------------------------------------------- layer prefetch
def layer_tensor_names(tmap, unit):
    pre = unit + "."
    return [k for k in tmap["tensors"] if k.startswith(pre) and ".engram.embed" not in k]


def prefetch_layer(fetcher, tmap, unit, threads, log):
    names = layer_tensor_names(tmap, unit)
    t0 = time.time()
    tot = 0

    def one(k):
        fetcher.fetch_bytes(k)    # fills the cache
        return tmap["tensors"][k]["length"]

    with ThreadPoolExecutor(threads) as ex:
        for n in ex.map(one, names):
            tot += n
    log(f"prefetched {unit}: {tot / 1e9:.2f} GB in {time.time() - t0:.0f}s ({tot / 1e6 / max(1e-3, time.time() - t0):.0f} MB/s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--rows", type=int, default=128)
    ap.add_argument("--out", required=True)
    ap.add_argument("--spool")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--engram-subset")
    ap.add_argument("--tokenizer")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--layers", default="0-39")
    ap.add_argument("--lead", type=int, default=3, help="max layers captured but not yet assembled")
    ap.add_argument("--spool-cap-gb", type=float, default=24.0, help="capture waits while the spool holds more")
    ap.add_argument("--prefetch-layers", type=int, default=2)
    ap.add_argument("--fetch-threads", type=int, default=6)
    ap.add_argument("--min-expert-tokens", type=int, default=2048)
    ap.add_argument("--state-ckpt", help="directory for row-state checkpoints (resume)")
    ap.add_argument("--ckpt-every", type=int, default=4)
    ap.add_argument("--row-len", type=int, default=0, help="truncate rows to this many tokens (0 = all)")
    a = ap.parse_args()
    from exllamav3.dsv41.config import DSV41Config
    from exllamav3.dsv41.stream_calib import StreamCalibrator, SourceLayerWeights, EngramSupport

    cfg = DSV41Config.from_file(a.config)
    tmap = json.load(open(a.map))
    spool = SP.Spool(a.spool or os.path.join(a.out, "spool"))
    os.makedirs(spool.root, exist_ok=True)
    os.makedirs(a.cache, exist_ok=True)
    dev = a.device
    fetcher = F.TensorFetcher(tmap, cache=a.cache)
    src = make_source(fetcher, cfg.num_hidden_layers, dev)
    lo, _, hi = a.layers.partition("-")
    lo, hi = int(lo), int(hi or lo)

    arr = np.load(a.calib)[:a.rows].astype(np.int64)
    if a.row_len:
        arr = arr[:, :a.row_len]
    rows = [torch.from_numpy(r).view(1, -1) for r in arr]
    LOG(f"{len(rows)} rows x {arr.shape[1]} tokens, device {dev}, layers {lo}-{hi}")

    engram = None
    if a.engram_subset:
        from engram_subset import token_map_from_file
        tm, _ = token_map_from_file(a.tokenizer)
        engram = EngramSupport(cfg, tm, SubsetEngramStore(a.engram_subset, len(cfg.engram_layer_ids)))
    t0 = time.time()
    embed = src.tensor("embed", torch.float32, "cpu")
    cal = StreamCalibrator(cfg, dev, embed, engram, state_device="cpu", min_expert_tokens=a.min_expert_tokens)
    del embed
    cal.start(rows)
    rec = RoutingRecorder(a.out)
    try:
        rec.install(cal)
    except Exception as ex:   # never let the allocation statistics stop the conversion
        LOG(f"WARNING routing recorder not installed ({type(ex).__name__}: {ex}); no alternative-K candidates")
        rec.begin = lambda *a_, **k_: None
        rec.end = lambda: None
    LOG(f"calibrator ready in {time.time() - t0:.0f}s")

    done_layers = lambda: [l for l in range(lo, hi + 1) if os.path.exists(os.path.join(a.out, f"layers.{l}.done.json"))]
    outstanding = lambda: [d for d in os.listdir(spool.root) if os.path.isdir(os.path.join(spool.root, d))]

    # state checkpoint / resume
    start_layer = lo
    if a.state_ckpt:
        os.makedirs(a.state_ckpt, exist_ok=True)
        cp = os.path.join(a.state_ckpt, "state.pt")
        if os.path.exists(cp):
            st = torch.load(cp, weights_only=False)
            cal.h, cal.ho, start_layer = st["h"], st["ho"], st["next_layer"]
            LOG(f"resumed row state at layer {start_layer}")
            del st
    pf = {}
    pool = ThreadPoolExecutor(1)

    def ensure_prefetch(l):
        for j in range(l, min(hi, l + a.prefetch_layers) + 1):
            if j not in pf:
                pf[j] = pool.submit(prefetch_layer, fetcher, tmap, f"layers.{j}", a.fetch_threads, LOG)

    cap_times = []
    for l in range(start_layer, hi + 1):
        unit = f"layers.{l}"
        ensure_prefetch(l)
        pf[l].result()
        if spool.capture_done(unit) and os.path.exists(os.path.join(a.out, f"{unit}.done.json")):
            pass
        while len(outstanding()) >= a.lead:
            time.sleep(5)
        W = SourceLayerWeights(src, l, dev)
        t1 = time.time()
        if os.path.exists(os.path.join(a.out, f"{unit}.done.json")):
            # resume replay of an assembled layer: advance the rows only, nothing to emit
            rec.begin(unit, skip_existing=True)
            cal.run_layer(l, W, None, commit=True)
            rec.end()
            if str(dev).startswith("cuda"):
                torch.cuda.synchronize()
            LOG(f"{unit}: already assembled, replayed (no Hessians) in {time.time() - t1:.0f}s")
            shutil.rmtree(os.path.join(a.cache, unit), ignore_errors=True)
            pf.pop(l, None)
            ensure_prefetch(l + 1)
            continue
        sink = SpoolSink(spool, unit, unit, LOG, int(a.spool_cap_gb * 1e9))
        rec.begin(unit)
        cal.run_layer(l, W, sink, commit=True)
        rec.end()
        if str(dev).startswith("cuda"):
            torch.cuda.synchronize()
        spool.mark_capture_done(unit)
        cap_times.append(time.time() - t1)
        LOG(f"{unit}: captured in {cap_times[-1]:.0f}s, {sink.n} Hessians ({sink.bytes / 1e9:.1f} GB spool), "
            f"{sink.none} without tokens; rss {SP.mem_available_gib():.0f} GiB avail")
        # cache of assembled layers goes away
        for j in done_layers():
            shutil.rmtree(os.path.join(a.cache, f"layers.{j}"), ignore_errors=True)
            pf.pop(j, None)
        if a.state_ckpt and (l + 1 - lo) % a.ckpt_every == 0 and l < hi:
            t2 = time.time()
            torch.save({"h": cal.h, "ho": cal.ho, "next_layer": l + 1}, os.path.join(a.state_ckpt, "state.pt.tmp"))
            os.replace(os.path.join(a.state_ckpt, "state.pt.tmp"), os.path.join(a.state_ckpt, "state.pt"))
            LOG(f"row state checkpoint at layer {l + 1}: {time.time() - t2:.0f}s")
        ensure_prefetch(l + 1)
    LOG(f"capture of layers {lo}-{hi} finished; mean {sum(cap_times) / max(1, len(cap_times)):.0f}s/layer")
    open(os.path.join(a.out, "CAPTURE_TRUNK_DONE"), "w").write("1\n")


if __name__ == "__main__":
    main()
