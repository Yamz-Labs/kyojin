#!/usr/bin/env python3
"""Per-shape and streaming timings for the ROCm EXL3 GEMV route.

Micro mode times `ext.exl3_gemv` per distinct (in, out, K, codebook) shape at
m = 1 and reports effective weight bandwidth. Stream mode runs one pass over
every EXL3 linear in the model, which removes the per-shape L2 reuse and is the
number to compare with a decode step. `--gemm` also times `ext.exl3_gemm` when
the tree exports it (upstream routing), for an A/B against the direct GEMV.

`--census` ranks the distinct shapes by cost (ms/token), not by bytes: a narrow
shape that carries little data but runs many times per token can matter more
than a wide one. `calls/token` is the number of LinearEXL3 modules with that
shape, which is exact for a dense trunk; for a MoE pack the experts are routed,
so the module count overstates the call count and the census must be read as an
upper bound. A separately loaded MTP/draft head is not traversed here (the
harness loads only the trunk), so its linears do not appear in the table.

`--sweep` runs the same cost-ranked shapes under each existing RDNA knob and
prints the delta against the default configuration. `env_int()` in
`exl3_gemv_rdna.cu` reads the environment on every launch, so one process can
sweep configs by mutating `os.environ`; the environment is restored in a
`finally`. A forced config that cannot launch (e.g. shared memory over 64 KB)
falls back to the generic path rather than throwing here, so a "win" from such
a config must be confirmed with `--verify` before it is trusted.

`--verify` compares the selected shapes/config against the generic WMMA GEMV
(`EXL3_GEMV=0`) and reports max abs and NaN. It is a correctness screen, not a
bit-exactness claim: the two paths use different reduction orders.

Example:
    HIP_VISIBLE_DEVICES=1 python tools/strix_halo/gemv_perf_ab.py \
        -m <model> --mode both --census --top 40
"""
import argparse
import contextlib
import os
import statistics
import torch
from exllamav3 import Config, Model
from exllamav3.modules.quant.exl3 import LinearEXL3
from exllamav3.util.tensor import g_tensor_cache
import exllamav3.ext as extmod

ext = extmod.exllamav3_ext

# Every measurement knob the RDNA launcher reads. The baseline clears them so
# an inherited shell environment cannot silently change what "default" means.
KNOBS = (
    "EXL3_GEMV",
    "EXL3_ROCM_GEMV_TILES",
    "EXL3_GEMV_LDS",
    "EXL3_GEMV_SPLITK",
    "EXL3_GEMV_SPLITK_WARPS",
    "EXL3_GEMV_WARPS",
    "EXL3_GEMV_FUSE_OUT",
)


def collect(model):
    mods = []
    stack = list(model.modules)
    while stack:
        mod = stack.pop()
        inner = getattr(mod, "inner", None)
        if isinstance(inner, LinearEXL3):
            mods.append(inner)
        stack.extend(getattr(mod, "modules", []) or [])
    return mods


def shape_key(l):
    return (l.in_features, l.out_features, int(l.K), int(l.mcg) + 2 * int(l.mul1))


def shape_bytes(key):
    k, n, K, _ = key
    return k * n * K // 8


def fmt_shape(key):
    return f"{key[0]}x{key[1]} K{key[2]} cb{key[3]}"


@contextlib.contextmanager
def env_config(cfg):
    """Apply cfg (None unsets) and restore the previous environment."""
    old = {}
    try:
        for k, v in cfg.items():
            old[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def time_us(fn, iters, repeats=1):
    """Median and spread (max-min) of `repeats` event-timed runs.

    The spread is None with a single repeat, so callers cannot mistake an
    unmeasured spread for perfect stability.
    """
    vals = []
    for _ in range(repeats):
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(iters):
            fn()
        e1.record()
        torch.cuda.synchronize()
        vals.append(e0.elapsed_time(e1) / iters * 1e3)
    spread = max(vals) - min(vals) if len(vals) > 1 else None
    return statistics.median(vals), spread


def fmt_spread(spread):
    return "n/a" if spread is None else f"{spread:.1f}"


def buffers(l, dev):
    x = torch.randn((1, l.in_features), dtype=torch.half, device=dev) * 0.5
    y = torch.empty((1, l.out_features), dtype=torch.half, device=dev)
    ah = g_tensor_cache.get(dev, (1, l.in_features), torch.half, "exl3_gemv_a_had")
    return x, y, ah


def launch(l, x, y, ah):
    ext.exl3_gemv(x, l.trellis, y, l.suh, ah, l.svh, l.mcg, l.mul1)


def baseline_cfg():
    return {k: None for k in KNOBS}


def sweep_configs():
    cfgs = [("default", baseline_cfg())]
    cfgs.append(("splitk=0", {"EXL3_GEMV_SPLITK": 0}))
    for w in (4, 8, 16):
        cfgs.append((f"splitk_warps={w}", {"EXL3_GEMV_SPLITK_WARPS": w}))
    for w in (4, 8, 16):
        cfgs.append((f"warps={w}", {"EXL3_GEMV_WARPS": w}))
    # EXL3_GEMV_WARPS only reaches warps_for(); with split-K enabled (the
    # default for size_n <= 32768) splitk_warps() overwrites it. These rows
    # isolate the single-warp form so the knob is actually measured.
    for w in (4, 8, 16):
        cfgs.append((f"splitk=0,warps={w}",
                     {"EXL3_GEMV_SPLITK": 0, "EXL3_GEMV_WARPS": w}))
    cfgs.append(("lds=1", {"EXL3_GEMV_LDS": 1}))
    cfgs.append(("tiles=0", {"EXL3_ROCM_GEMV_TILES": 0}))
    cfgs.append(("fuse_out=0", {"EXL3_GEMV_FUSE_OUT": 0}))
    return cfgs


def reference_out(l, x, y, ah):
    y.zero_()
    with env_config({"EXL3_GEMV": 0}):
        launch(l, x, y, ah)
        torch.cuda.synchronize()
    return y.clone()


def check_against_reference(l, x, y, ah, ref):
    y.zero_()
    launch(l, x, y, ah)
    torch.cuda.synchronize()
    nan = bool(torch.isnan(y).any().item())
    if ref is None:
        return nan, None
    err = (y.float() - ref.float()).abs().max().item()
    return nan, err


def census(by_shape, dev, iters, repeats, top):
    rows = []
    for key, mods in by_shape.items():
        l = mods[0]
        x, y, ah = buffers(l, dev)
        us, spread = time_us(lambda: launch(l, x, y, ah), iters, repeats)
        b = shape_bytes(key)
        calls = len(mods)
        rows.append({
            "key": key,
            "MB": b / 1e6,
            "calls": calls,
            "us": us,
            "spread": spread,
            "GBs": b / us / 1e3,
            "ms_token": calls * us / 1000,
        })
    rows.sort(key=lambda r: -r["ms_token"])
    if not rows:
        print("\ncensus: no shapes selected")
        return rows, 0.0
    total_ms = sum(r["ms_token"] for r in rows)
    total_gb = sum(r["MB"] * r["calls"] for r in rows) / 1e3
    print("\ncensus (distinct shapes, m=1, median of "
          f"{repeats} x {iters} iters), ranked by ms/token:")
    print(f"{'in':>6} {'out':>7} {'K':>2} {'cb':>2} {'MB':>9} {'calls':>6} "
          f"{'us':>8} {'+-us':>7} {'GB/s':>7} {'ms/tok':>8} {'share':>7}")
    show = rows if top <= 0 else rows[:top]
    for r in show:
        k, n, K, cb = r["key"]
        share = 100.0 * r["ms_token"] / total_ms if total_ms else 0.0
        print(f"{k:6d} {n:7d} {K:2d} {cb:2d} {r['MB']:9.3f} {r['calls']:6d} "
              f"{r['us']:8.1f} {fmt_spread(r['spread']):>7} {r['GBs']:7.0f} "
              f"{r['ms_token']:8.3f} {share:7.2f}%")
    print(f"census total: {total_ms:.2f} ms/token over {len(rows)} shapes "
          f"({total_gb * 1e3:.3f} MB stream, {total_gb / (total_ms / 1e3):.0f} GB/s implied)")
    print("note: calls/token is a module count; MoE routing makes it an upper bound.")
    return rows, total_ms


def parse_shapes(spec):
    want = set()
    for token in spec.split(";"):
        token = token.strip()
        if not token:
            continue
        parts = [int(x) for x in token.replace(" ", ",").split(",") if x.strip()]
        if len(parts) != 4:
            raise SystemExit(f"bad --shapes entry: {token!r} (want in,out,K,cb)")
        want.add(tuple(parts))
    return want


def filter_shapes(by_shape, spec):
    if not spec:
        return dict(by_shape)
    want = parse_shapes(spec)
    missing = want - set(by_shape)
    if missing:
        print("warning: shapes not found: " +
              ", ".join(fmt_shape(k) for k in sorted(missing)))
    return {k: v for k, v in by_shape.items() if k in want}


def select_shapes(by_shape, args, dev, iters, repeats):
    by_shape = filter_shapes(by_shape, args.shapes)
    keys = list(by_shape)
    bufs = {k: buffers(by_shape[k][0], dev) for k in keys}
    # Rank without the full census print. Measure the default for the selected
    # shapes, then keep the top N by cost.
    base = measure(keys, by_shape, dev, iters, repeats, baseline_cfg(), bufs)
    ranked = sorted((k for k in keys if base[k][0] is not None),
                    key=lambda k: -(len(by_shape[k]) * base[k][0]))
    if args.top > 0:
        ranked = ranked[:args.top]
    return ranked, base, bufs


def measure(keys, by_shape, dev, iters, repeats, cfg, bufs, refs=None):
    out = {}
    with env_config(cfg):
        for k in keys:
            l = by_shape[k][0]
            x, y, ah = bufs[k]
            try:
                us, spread = time_us(lambda: launch(l, x, y, ah), iters, repeats)
                nan = err = None
                if refs is not None:
                    nan, err = check_against_reference(l, x, y, ah, refs[k])
            except RuntimeError as e:
                print(f"  {fmt_shape(k)}: n/a ({e})", flush=True)
                out[k] = (None, None, None, None)
            else:
                out[k] = (us, spread, nan, err)
    return out


def sweep(by_shape, args, dev, iters, repeats):
    keys, base, bufs = select_shapes(by_shape, args, dev, iters, repeats)
    if not keys:
        print("sweep: no shapes selected")
        return
    base_total = sum(len(by_shape[k]) * base[k][0] for k in keys) / 1000
    print(f"\nsweep over {len(keys)} shapes, baseline {base_total:.3f} ms/token, "
          f"median of {repeats} x {iters} iters:")
    refs = {}
    if args.verify:
        for k in keys:
            l = by_shape[k][0]
            x, y, ah = bufs[k]
            refs[k] = reference_out(l, x, y, ah)

    default_verify = {}
    if args.verify:
        # Verify the auto-routed default too, without re-timing it.
        with env_config(baseline_cfg()):
            for k in keys:
                l = by_shape[k][0]
                x, y, ah = bufs[k]
                default_verify[k] = check_against_reference(l, x, y, ah, refs[k])

    for name, cfg in sweep_configs():
        # Every config starts from the cleared baseline so an inherited
        # EXL3_* knob cannot contaminate a single-knob comparison.
        full_cfg = {**baseline_cfg(), **cfg}
        if name == "default":
            res = {k: (base[k][0], base[k][1],
                       *(default_verify.get(k, (None, None))))
                   for k in keys}
        else:
            res = measure(keys, by_shape, dev, iters, repeats, full_cfg, bufs,
                          refs if args.verify else None)
        # Compare only shapes that completed under both arms. Dropping a
        # failed shape from one side alone manufactures a false win.
        ok = [k for k in keys
              if base.get(k) and base[k][0] is not None
              and res.get(k) and res[k][0] is not None]
        if not ok:
            print(f"\nCONFIG {name}  env={ {k: v for k, v in cfg.items() if v is not None} }")
            print("  n/a (no shape completed)")
            continue
        total = sum(len(by_shape[k]) * res[k][0] for k in ok) / 1000
        base_cmp = sum(len(by_shape[k]) * base[k][0] for k in ok) / 1000
        delta = (100.0 * (total - base_cmp) / base_cmp) if base_cmp else 0.0
        spreads_known = all(base[k][1] is not None and res[k][1] is not None
                            for k in ok)
        if spreads_known:
            spread_ms = sum(len(by_shape[k]) * (base[k][1] + res[k][1])
                            for k in ok) / 1000
            spread_pct = (100.0 * spread_ms / base_cmp) if base_cmp else 0.0
        else:
            spread_pct = None
        within = spread_pct is not None and abs(delta) <= spread_pct
        partial = (f"  partial {len(keys) - len(ok)}/{len(keys)} shapes n/a"
                   if len(ok) != len(keys) else "")
        spread_txt = f"spread +-{spread_pct:.2f}%" if spread_pct is not None else "spread n/a"
        print(f"\nCONFIG {name}  env={ {k: v for k, v in cfg.items() if v is not None} }")
        print(f"  total {total:.3f} ms/token  delta {delta:+.2f}%  {spread_txt}"
              f"{'  (within spread)' if within else ''}{partial}")
        for k in ok:
            b = base[k]
            r = res[k]
            d = 100.0 * (r[0] - b[0]) / b[0] if b[0] else 0.0
            if b[1] is not None and r[1] is not None:
                shape_spread = 100.0 * (b[1] + r[1]) / b[0] if b[0] else 0.0
                marker = "  (within spread)" if abs(d) <= shape_spread else ""
            else:
                marker = ""
            line = (f"  {fmt_shape(k):>26} base {b[0]:7.1f}+-{fmt_spread(b[1])} us  "
                    f"cfg {r[0]:7.1f}+-{fmt_spread(r[1])} us  {d:+6.1f}%")
            if args.verify and r[2] is not None:
                line += f"  maxabs={r[3]:.2e} nan={r[2]}"
            print(line + marker)
    print("\nsweep note: a forced config that exceeds 64 KB smem falls back to the "
          "generic WMMA path; confirm with --verify before trusting its delta.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default=os.environ.get("EXL3_TEST_MODEL", ""))
    ap.add_argument("--mode", choices=("micro", "stream", "both"), default="both")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--census-iters", type=int, default=300)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--top", type=int, default=16)
    ap.add_argument("--gemm", action="store_true",
                    help="also time ext.exl3_gemm when present")
    ap.add_argument("--census", action="store_true",
                    help="cost-ranked per-shape census (ms/token, calls/token)")
    ap.add_argument("--sweep", action="store_true",
                    help="sweep the RDNA knobs over the cost-ranked shapes")
    ap.add_argument("--shapes", default="",
                    help='limit --census, --sweep and the micro table to '
                         '"in,out,K,cb;..." shapes (stream stays a full pass)')
    ap.add_argument("--verify", action="store_true",
                    help="compare against EXL3_GEMV=0 generic path")
    args = ap.parse_args()
    if not args.model:
        raise SystemExit("--model or EXL3_TEST_MODEL is required")

    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    model.load(progressbar=False)
    dev = torch.device("cuda:0")
    linears = collect(model)
    if not linears:
        raise SystemExit("no LinearEXL3 modules found")
    total_bytes = sum(shape_bytes(shape_key(l)) for l in linears)
    print(f"model: {args.model}")
    print(f"ext:   {ext.__file__}")
    print(f"linears: {len(linears)}  weight stream: {total_bytes / 1e9:.3f} GB")
    if args.gemm and not hasattr(ext, "exl3_gemm"):
        print("ext.exl3_gemm not exported; skipping the --gemm A/B")

    by_shape = {}
    for l in linears:
        by_shape.setdefault(shape_key(l), []).append(l)

    def buffers_for(l):
        x = torch.randn((1, l.in_features), dtype=torch.half, device=dev) * 0.5
        y = torch.empty((1, l.out_features), dtype=torch.half, device=dev)
        ah = g_tensor_cache.get(dev, (1, l.in_features), torch.half, "exl3_gemv_a_had")
        return x, y, ah

    if args.mode in ("micro", "both"):
        shape_filter = parse_shapes(args.shapes) if args.shapes else None
        seen = {}
        for l in linears:
            key = shape_key(l)
            if shape_filter is not None and key not in shape_filter:
                continue
            seen.setdefault(key, l)
        rows = sorted(seen.items(), key=lambda kv: -shape_bytes(kv[0]))
        has_gemm = args.gemm and hasattr(ext, "exl3_gemm")
        head = f"{'k':>6} {'n':>7} {'K':>2} {'cb':>2} {'MB':>8} {'us':>8} {'GB/s':>7}"
        if has_gemm:
            head += f" {'gemm_us':>8} {'gemm_GB/s':>9}"
        print("\n" + head)
        for (k, n, K, cb), l in rows[:args.top]:
            x, y, ah = buffers_for(l)
            b = shape_bytes((k, n, K, cb))
            f = lambda: launch(l, x, y, ah)
            us, _ = time_us(f, args.iters, args.repeats)
            line = f"{k:6d} {n:7d} {K:2d} {cb:2d} {b/1e6:8.2f} {us:8.1f} {b/us/1e3:7.0f}"
            if has_gemm:
                g = lambda: ext.exl3_gemm(x, l.trellis, y, l.suh, ah, l.svh, -1,
                                          l.mcg, l.mul1, 0)
                usg = time_us(g, args.iters, args.repeats)[0]
                line += f" {usg:8.1f} {b/usg/1e3:9.0f}"
            print(line, flush=True)

    if args.mode in ("stream", "both"):
        work = []
        for l in linears:
            x, y, ah = buffers_for(l)
            work.append((l, x, y, ah))

        def run_gemv():
            for l, x, y, ah in work:
                launch(l, x, y, ah)

        def run_gemm():
            for l, x, y, ah in work:
                ext.exl3_gemm(x, l.trellis, y, l.suh, ah, l.svh, -1, l.mcg, l.mul1, 0)

        def stream(fn, passes=3):
            fn()
            torch.cuda.synchronize()
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(passes):
                fn()
            e1.record()
            torch.cuda.synchronize()
            return e0.elapsed_time(e1) / passes

        print("\nstreaming pass over all linears (median of 3):")
        ms = statistics.median([stream(run_gemv) for _ in range(3)])
        print(f"  exl3_gemv: {ms:.2f} ms  {total_bytes / ms * 1e-6:.0f} GB/s")
        if args.gemm and hasattr(ext, "exl3_gemm"):
            msg = statistics.median([stream(run_gemm) for _ in range(3)])
            print(f"  exl3_gemm: {msg:.2f} ms  {total_bytes / msg * 1e-6:.0f} GB/s")

    if args.census:
        census(filter_shapes(by_shape, args.shapes), dev,
               args.census_iters, args.repeats, args.top)

    if args.sweep:
        sweep(by_shape, args, dev, args.census_iters, args.repeats)


if __name__ == "__main__":
    main()
