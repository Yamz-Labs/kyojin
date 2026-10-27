"""Day-0 autotune of one pack on this device (qualify stage `autotune`).

Reads every tensor shape of the pack from its safetensors headers (no weights are loaded), runs the engine's GEMM autotuner for each
distinct (in, out, K, codebook) at every row class a request can reach, and writes the winners next to the pack:
    <pack>/tuning/<device signature>.bin  (+ .json manifest with per-shape timings)
Serve start merges that file into the live cache (exllamav3/util/tune_cache.install_pack), so the first request of a new head_dim /
hidden size / expert count runs on tuned kernels with no hand work.

Pure helpers (shapes, row classes, manifest) are CPU-testable; `run()` needs the engine venv and a free GPU.
    python autotune.py PACK [--dtypes half,float] [--rows 1,2,4,...] [--out DIR] [--repeat 20]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import struct
import sys
import time
from pathlib import Path

# Row classes of the GEMM autotune key (exl3_gemm.cu gemm_autotune_hash): <=16 -> next pow2, 17..24 -> 24, then 16-steps capped at 64.
# The BC path (CUDA/ROCm exl3_gemm) serves rows <= AUTO_RECONSTRUCT_THRESHOLD (192 CUDA / 144 ROCm); beyond that reconstruct + dense GEMM.
ROW_CLASSES = (1, 2, 4, 8, 16, 24, 32, 48, 64)


def row_class(m: int) -> int:
    """Python mirror of the row term in gemm_autotune_hash (size_m is clamped to >= 2 by the caller)."""
    m = max(m, 2)
    if m <= 16:
        c = 1
        while c < m:
            c <<= 1
        return c
    return 24 if m <= 24 else min(-(-m // 16) * 16, 64)


def pack_shapes(pack: str | os.PathLike) -> dict:
    """{(in, out, K, mcg, mul1): count} for every dense EXL3 tensor, from the safetensors headers. Expert stacks are reported apart."""
    dense, experts = collections.Counter(), collections.Counter()
    for f in sorted(glob.glob(str(Path(pack) / "*.safetensors"))):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for k, v in hdr.items():
            if not k.endswith(".trellis"):
                continue
            shp = v["shape"]
            base = k[:-len(".trellis")]
            if len(shp) == 3:
                key = (shp[0] * 16, shp[1] * 16, shp[2] / 16, (base + ".mcg") in hdr, (base + ".mul1") in hdr)
                (experts if ".experts." in k else dense)[key] += 1
            else:
                experts[("stack", tuple(shp))] += 1
    return dict(dense=dict(dense), experts=dict(experts))


def tune_plan(shapes: dict, rows=ROW_CLASSES, dtypes=("half", "float")) -> list:
    """Ordered list of (in, out, K, mcg, mul1, rows, dtype) the tuner must run (integral K only: half-integer K takes the Python path)."""
    out = []
    for (i, o, K, mcg, mul1), _n in sorted(shapes["dense"].items(), key=lambda kv: (-kv[0][0] * kv[0][1], kv[0])):
        if float(K) != int(K):
            continue
        for dt in dtypes:
            for r in rows:
                out.append((i, o, int(K), mcg, mul1, r, dt))
    return out


def run(pack: str, out_dir: str | None = None, dtypes=("half", "float"), rows=ROW_CLASSES, repeat: int = 20, log=print) -> dict:
    """Tune every shape. The tuner is pointed at a scratch cache (EXLLAMAV3_TUNE_CACHE) so the file holds exactly this pack's winners."""
    t0 = time.time()
    pack = str(pack)
    out = Path(out_dir) if out_dir else Path(pack) / "tuning"
    out.mkdir(parents=True, exist_ok=True)
    scratch = out / f".scratch-{os.getpid()}.bin"
    scratch.unlink(missing_ok=True)
    os.environ["EXLLAMAV3_TUNE_CACHE"] = str(scratch)       # before the first GEMM: the C++ side reads the path once, lazily
    import torch
    from exllamav3.util import tune_cache
    from exllamav3.modules.quant.exl3 import LinearEXL3
    sig = tune_cache.signature()
    shapes = pack_shapes(pack)
    plan = tune_plan(shapes, rows, dtypes)
    log(f"autotune {pack}: {len(shapes['dense'])} dense shapes, {len(plan)} (shape,rows,dtype) points on {sig['name']} ({sig['cc']}, {sig['sms']} SMs)")
    res = []
    dev = torch.device("cuda", sig["device_index"])
    torch.manual_seed(0)
    cur = None
    for (i, o, K, mcg, mul1, r, dt) in plan:
        if cur is None or cur[0] != (i, o, K, mcg, mul1):
            tr = torch.randint(-32768, 32767, (i // 16, o // 16, 16 * K), dtype=torch.int16, device=dev)
            suh = (torch.randint(0, 2, (i,), device=dev) * 2 - 1).half()
            svh = (torch.randint(0, 2, (o,), device=dev) * 2 - 1).half()
            mc = torch.tensor(0xC0DE, dtype=torch.int32, device=dev) if mcg else None
            m1 = torch.tensor(0xC0DE, dtype=torch.int32, device=dev) if mul1 else None
            cur = ((i, o, K, mcg, mul1), LinearEXL3(None, i, o, suh=suh, svh=svh, trellis=tr, mcg=mc, mul1=m1))
        lin = cur[1]
        x = torch.randn(r, i, dtype=torch.half, device=dev) * 0.1
        f32 = dt == "float"
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        lin.bc.run_alloc(x, o, f32)                         # first call at this key: the autotuner times its candidates here
        torch.cuda.synchronize()
        first_ms = (time.perf_counter() - t1) * 1e3
        t1 = time.perf_counter()
        for _ in range(repeat):
            lin.bc.run_alloc(x, o, f32)
        torch.cuda.synchronize()
        steady_us = (time.perf_counter() - t1) / repeat * 1e6
        res.append(dict(i=i, o=o, K=K, mcg=mcg, mul1=mul1, rows=r, dtype=dt, first_ms=round(first_ms, 2), steady_us=round(steady_us, 1)))
    recs = tune_cache.read_records(scratch)
    dst = out / f"{sig['file']}.bin"
    tune_cache.write_records(dst, recs)
    scratch.unlink(missing_ok=True)
    man = dict(schema=1, signature=sig, records=len(recs), plan_points=len(plan), wall_s=round(time.time() - t0, 1),
               shapes=[dict(i=k[0], o=k[1], K=k[2], mcg=k[3], mul1=k[4], tensors=n) for k, n in shapes["dense"].items()],
               experts_not_tuned=len(shapes["experts"]), points=res, torch=torch.__version__,
               engine_git=_git(Path(__file__).resolve().parents[2]))
    (out / f"{sig['file']}.json").write_text(json.dumps(man, indent=1))
    log(f"autotune: {len(recs)} records -> {dst} in {man['wall_s']} s")
    return man


def _git(repo: Path) -> str | None:
    import subprocess
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:   # noqa: BLE001
        return None


if __name__ == "__main__":
    if "--signature" in sys.argv:                      # device signature of this process (the stage runs it with the backend's python)
        from exllamav3.util import tune_cache
        print(json.dumps(tune_cache.signature()))
        sys.exit(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("pack")
    ap.add_argument("--out")
    ap.add_argument("--dtypes", default="half,float")
    ap.add_argument("--rows", default=",".join(map(str, ROW_CLASSES)))
    ap.add_argument("--repeat", type=int, default=20)
    a = ap.parse_args()
    m = run(a.pack, a.out, tuple(a.dtypes.split(",")), tuple(int(x) for x in a.rows.split(",")), a.repeat)
    print(json.dumps({k: m[k] for k in ("records", "plan_points", "wall_s")}))
