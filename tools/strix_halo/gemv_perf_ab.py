#!/usr/bin/env python3
"""Per-shape and streaming timings for the ROCm EXL3 GEMV route.

Micro mode times `ext.exl3_gemv` per distinct (in, out, K, codebook) shape and
reports effective weight bandwidth. The default is m = 1 (decode); `--rows R`
(2..8; 1 is the default) times the dense multi-row GEMV on gfx1150/1151/1152
and gfx1200/1201.
Only shapes inside the MR gate are measured: integer K 2..6, mcg or mul1, and
inputs/outputs divisible by 128; other shapes are reported and skipped rather
than silently timed on the generic fallback. Stream mode runs one pass over
the eligible EXL3 linears in the model, which removes the per-shape L2 reuse
and is the number to compare with a decode step. `--gemm` also times
`ext.exl3_gemm` when the tree exports it (upstream routing), for an A/B
against the direct GEMV.

`--census` ranks the distinct shapes by cost (ms/token), not by bytes: a narrow
shape that carries little data but runs many times per token can matter more
than a wide one. `calls/token` is the number of LinearEXL3 modules with that
shape, which is exact for a dense trunk; for a MoE pack the experts are routed,
so the module count overstates the call count and the census must be read as an
upper bound. A separately loaded MTP/draft head is not traversed here (the
harness loads only the trunk), so its linears do not appear in the table.
With `--rows R` one call serves R tokens, so ms/token is the per-call cost
divided by R.

`--dec` adds a column for `ext.exl3_dec_gemv` on shapes where `dec_supported()`
holds. That predicate includes the gfx11.5 family, mul1, K in the dense dec
set, and the ambient `EXL3_DEC` / `EXL3_DEC_DENSE` gates, so the column is a
kernel timing task, not a sweep of the Python routing decision; see the note
below.

`--sweep` runs the same cost-ranked shapes under each existing RDNA knob and
prints the delta against the default configuration. `env_int()` in
`exl3_gemv_rdna.cu` reads the environment on every launch, so one process can
sweep configs by mutating `os.environ`; the environment is restored in a
`finally`. A forced config that cannot launch (e.g. shared memory over 64 KB)
falls back to the generic path rather than throwing here, so a "win" from such
a config may be the generic path; `--verify` checks correctness, not which
kernel actually ran.

The sweep is paired: for every shape, each config is measured in the same
interleaved bursts, and a duplicate "default#2" arm measures the run-to-run
noise floor of an identical config. A config is reported as a win only when its
median delta is outside 2 x that noise floor and the sign reproduces across
every burst; otherwise it is labelled "within noise" or "sign flips". The
within-run spread is the standard deviation across those paired bursts, not a
max-min range.

`--verify` compares the selected sweep configs against the generic WMMA GEMV
(`EXL3_GEMV=0` plus the other route knobs cleared) and reports max abs and NaN.
For K7/K8 and other shapes the generic GEMV cannot serve, the reference falls
back to `ext.exl3_gemm` where bound, otherwise to reconstruct + hgemm, and the
line is labelled `ref=gemm` or `ref=reconstruct`. It is a
correctness screen, not a bit-exactness claim: the paths use different
reduction orders. `--verify` only applies to `--sweep`; it warns otherwise.

This harness calls `ext.exl3_gemv` / `ext.exl3_dec_gemv` directly, so Python
routing gates such as `EXL3_DEC` and `EXL3_DEC_DENSE` (read at import time in
`exllamav3.modules.quant.exl3`) are not swept. Use `--dec` for the dec kernel
timing and evaluate the routing decision at the model level.

When `--sweep` is used, inherited route knobs are cleared before the first
launch as well as per config, so no first-launch-time decision can see an
inherited value.

Example:
    python tools/strix_halo/gemv_perf_ab.py \
        -m <model> --mode both --census --top 40 --sweep --verify --repeats 6

Select the GPU with HIP_VISIBLE_DEVICES=<index> if several are visible; pick the
index whose gcnArchName matches. Enumeration order is a driver/boot property.
"""
import argparse
import contextlib
import os
import statistics
import torch
from exllamav3 import Config, Model
from exllamav3.modules.quant.exl3 import LinearEXL3, dec_supported, dec_workspace
from exllamav3.util.tensor import g_tensor_cache
import exllamav3.ext as extmod

ext = extmod.exllamav3_ext

# Every route knob the dispatch or the RDNA launcher reads. The baseline clears
# them so an inherited shell environment cannot silently change what "default"
# means. EXL3_INT8_GEMV is included because the fused int8 kernel is tried
# before the RDNA launcher. EXL3_DEC_DENSE is a Python import-time gate and
# cannot be swept through the environment; see the --dec column.
KNOBS = (
    "EXL3_GEMV",
    "EXL3_INT8_GEMV",
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


def mr_eligible(l):
    """Mirror the multi-row gate in exl3_gemv_rdna_launch().

    Rows > 1 take the MR kernel only for integer K 2..6 with a codebook; the
    dispatcher also declines the RDNA path for half-integer bitrates. Shapes
    outside this gate are skipped by --rows rather than silently measured on
    the generic fallback. The device is assumed to be RDNA (the caller warns).
    """
    return (l.K == int(l.K) and 2 <= int(l.K) <= 6
            and bool(l.mcg or l.mul1)
            and l.in_features % 128 == 0 and l.out_features % 128 == 0)


def rdna_arch(dev):
    try:
        name = torch.cuda.get_device_properties(dev).gcnArchName
    except Exception:
        return False
    return name.startswith(("gfx1150", "gfx1151", "gfx1152",
                            "gfx1200", "gfx1201"))


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


def time_burst(fn, iters):
    """One event-timed burst of `iters` launches, in microseconds per launch."""
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
    return e0.elapsed_time(e1) / iters * 1e3


def time_us(fn, iters, repeats=1):
    """Median and standard deviation of `repeats` event-timed bursts.

    The spread is None with a single repeat, so callers cannot mistake an
    unmeasured spread for perfect stability. A standard deviation is used
    instead of max-min because the sweep compares paired deltas against it.
    """
    vals = [time_burst(fn, iters) for _ in range(repeats)]
    spread = statistics.stdev(vals) if len(vals) > 1 else None
    return statistics.median(vals), spread


def fmt_spread(spread):
    return "n/a" if spread is None else f"{spread:.1f}"


def buffers(l, dev, rows=1):
    x = torch.randn((rows, l.in_features), dtype=torch.half, device=dev) * 0.5
    y = torch.empty((rows, l.out_features), dtype=torch.half, device=dev)
    ah = g_tensor_cache.get(dev, (rows, l.in_features), torch.half, "exl3_gemv_a_had")
    return x, y, ah


def launch(l, x, y, ah):
    ext.exl3_gemv(x, l.trellis, y, l.suh, ah, l.svh, l.mcg, l.mul1)


def dec_launch(l, x, y, scratch, counters):
    # Batch-1 dense dec kernel; dec_supported() already checked the codebook
    # and shape gates. The Python dec_gemv() wrapper adds bias, which is not
    # part of this kernel comparison.
    ext.exl3_dec_gemv(x.view(1, l.in_features), l.trellis, l.suh, l.svh,
                      y.view(1, l.out_features), scratch, counters, l.K)


def baseline_cfg():
    return {k: None for k in KNOBS}


def reference_cfg():
    # The int8 gate runs before the RDNA launcher and does not consult
    # EXL3_GEMV, so the generic reference must clear the whole route set.
    return {**baseline_cfg(), "EXL3_GEMV": 0}


def sweep_configs(rows=1):
    cfgs = [("default", baseline_cfg())]
    if rows > 1:
        # The multi-row path uses splitk_warps() and the core mode; it ignores
        # EXL3_GEMV_SPLITK, EXL3_GEMV_WARPS and EXL3_GEMV_FUSE_OUT, so those
        # rows would be no-ops (and no-op rows are exactly the noise controls
        # this harness should not mistake for measurements).
        for w in (4, 8, 16):
            cfgs.append((f"splitk_warps={w}", {"EXL3_GEMV_SPLITK_WARPS": w}))
        cfgs.append(("lds=1", {"EXL3_GEMV_LDS": 1}))
        cfgs.append(("tiles=0", {"EXL3_ROCM_GEMV_TILES": 0}))
        return cfgs
    cfgs.append(("splitk=0", {"EXL3_GEMV_SPLITK": 0}))
    for w in (4, 8, 16):
        cfgs.append((f"splitk_warps={w}", {"EXL3_GEMV_SPLITK_WARPS": w}))
    for w in (4, 8, 16):
        cfgs.append((f"warps={w}", {"EXL3_GEMV_WARPS": w}))
    # EXL3_GEMV_WARPS only reaches warps_for(); with split-K enabled (the
    # default for size_n <= 32768) splitk_warps() overwrites it. These rows
    # isolate the single-warp form so the knob is actually measured. The plain
    # warps rows are no-ops for those shapes and serve as extra noise controls.
    for w in (4, 8, 16):
        cfgs.append((f"splitk=0,warps={w}",
                     {"EXL3_GEMV_SPLITK": 0, "EXL3_GEMV_WARPS": w}))
    cfgs.append(("lds=1", {"EXL3_GEMV_LDS": 1}))
    cfgs.append(("tiles=0", {"EXL3_ROCM_GEMV_TILES": 0}))
    cfgs.append(("fuse_out=0", {"EXL3_GEMV_FUSE_OUT": 0}))
    return cfgs


def reference_out(l, x, y, ah):
    """Return (output, kind). Generic GEMV first; GEMM/reconstruct for K7/K8."""
    y.zero_()
    try:
        with env_config(reference_cfg()):
            launch(l, x, y, ah)
            torch.cuda.synchronize()
        return y.clone(), "generic"
    except RuntimeError as generic_error:
        if "not eligible" not in str(generic_error):
            raise
        if hasattr(ext, "exl3_gemm"):
            y.zero_()
            try:
                # EXL3_GEMV=0 keeps the GEMM reference off the GEMV shortcut.
                with env_config(reference_cfg()):
                    ext.exl3_gemm(x, l.trellis, y, l.suh, ah, l.svh, -1,
                                  l.mcg, l.mul1, 0)
                    torch.cuda.synchronize()
            except RuntimeError:
                raise generic_error
            return y.clone(), "gemm"
        # ROCm does not bind exl3_gemm; dequantize and run an hgemm instead.
        try:
            ref = l.reconstruct_hgemm(x, torch.half)
        except RuntimeError:
            raise generic_error
        if l.bias is not None:
            ref = ref - l.bias
        return ref, "reconstruct"


def check_against_reference(l, x, y, ah, ref):
    y.zero_()
    launch(l, x, y, ah)
    torch.cuda.synchronize()
    nan = bool(torch.isnan(y).any().item())
    if ref is None:
        return nan, None
    err = (y.float() - ref.float()).abs().max().item()
    return nan, err


def census(by_shape, dev, iters, repeats, top, rows=1, dec=False):
    dec = bool(dec and rows == 1)
    if rows > 1:
        skipped = [k for k, mods in by_shape.items() if not mr_eligible(mods[0])]
        if skipped:
            shown = ", ".join(fmt_shape(k) for k in skipped[:6])
            more = "" if len(skipped) <= 6 else f" (+{len(skipped) - 6} more)"
            print(f"census rows>1: skipping {len(skipped)} shapes outside the "
                  f"MR gate: {shown}{more}")
        by_shape = {k: v for k, v in by_shape.items()
                    if mr_eligible(v[0])}
    out_rows = []
    dec_ws = None
    dec_gate = 0
    for key, mods in by_shape.items():
        l = mods[0]
        x, y, ah = buffers(l, dev, rows)
        try:
            us, spread = time_us(lambda: launch(l, x, y, ah), iters, repeats)
        except RuntimeError as e:
            print(f"  {fmt_shape(key)}: n/a ({e})", flush=True)
            continue
        b = shape_bytes(key)
        calls = len(mods)
        dec_us = None
        if dec and dec_supported(l):
            dec_gate += 1
            if dec_ws is None:
                dec_ws = dec_workspace(dev)
            scratch, counters = dec_ws
            try:
                dec_us, _ = time_us(
                    lambda: dec_launch(l, x, y, scratch, counters), iters, repeats)
            except RuntimeError as e:
                print(f"  {fmt_shape(key)}: dec n/a ({e})", flush=True)
        out_rows.append({
            "key": key,
            "MB": b / 1e6,
            "calls": calls,
            "us": us,
            "spread": spread,
            "GBs": b / us / 1e3,
            "ms_token": calls * us / (1000 * rows),
            "ms_call": calls * us / 1000,
            "dec_us": dec_us,
        })
    out_rows.sort(key=lambda r: -r["ms_token"])
    if not out_rows:
        print("\ncensus: no shapes selected")
        return out_rows, 0.0
    total_ms = sum(r["ms_token"] for r in out_rows)
    total_ms_call = sum(r["ms_call"] for r in out_rows)
    total_gb = sum(r["MB"] * r["calls"] for r in out_rows) / 1e3
    print(f"\ncensus (distinct shapes, m={rows}, median of "
          f"{repeats} x {iters} iters), ranked by ms/token:")
    head = (f"{'in':>6} {'out':>7} {'K':>2} {'cb':>2} {'MB':>9} {'calls':>6} "
            f"{'us':>8} {'+-us':>7} {'GB/s':>7} {'ms/tok':>8} {'share':>7}")
    if dec:
        head += f" {'dec_us':>8} {'dec%':>7}"
    print(head)
    show = out_rows if top <= 0 else out_rows[:top]
    for r in show:
        k, n, K, cb = r["key"]
        share = 100.0 * r["ms_token"] / total_ms if total_ms else 0.0
        line = (f"{k:6d} {n:7d} {K:2d} {cb:2d} {r['MB']:9.3f} {r['calls']:6d} "
                f"{r['us']:8.1f} {fmt_spread(r['spread']):>7} {r['GBs']:7.0f} "
                f"{r['ms_token']:8.3f} {share:7.2f}%")
        if dec:
            if r["dec_us"] is None:
                line += f" {'-':>8} {'-':>7}"
            else:
                d = 100.0 * (r["dec_us"] - r["us"]) / r["us"] if r["us"] else 0.0
                line += f" {r['dec_us']:8.1f} {d:+7.1f}"
        print(line)
    print(f"census total: {total_ms:.2f} ms/token over {len(out_rows)} shapes "
          f"({total_ms_call:.2f} ms/forward at m={rows}, "
          f"{total_gb * 1e3:.3f} MB stream, "
          f"{total_gb / (total_ms_call / 1e3):.0f} GB/s implied)")
    if dec:
        timed = [r for r in out_rows if r["dec_us"] is not None]
        if timed:
            dec_ms = sum(r["calls"] * r["dec_us"] / (1000 * rows) for r in timed)
            gemv_ms = sum(r["calls"] * r["us"] / (1000 * rows) for r in timed)
            d = 100.0 * (dec_ms - gemv_ms) / gemv_ms if gemv_ms else 0.0
            print(f"dec: {dec_ms:.2f} ms/token over {len(timed)}/{len(out_rows)} "
                  f"shapes vs gemv {gemv_ms:.2f} ({d:+.2f}%, negative = dec faster)")
        elif dec_gate:
            print(f"dec: {dec_gate} shape(s) passed dec_supported() but none "
                  f"could be timed")
        else:
            print("dec: no shape passed dec_supported() (needs gfx11.5, mul1, "
                  "the dense K set, and EXL3_DEC/EXL3_DEC_DENSE on)")
    print("note: calls/token is a module count; MoE routing makes it an upper bound.")
    return out_rows, total_ms


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
    if args.rows > 1:
        skipped = [k for k in keys if not mr_eligible(by_shape[k][0])]
        if skipped:
            shown = ", ".join(fmt_shape(k) for k in skipped[:6])
            more = "" if len(skipped) <= 6 else f" (+{len(skipped) - 6} more)"
            print(f"rows>1: skipping {len(skipped)} shapes outside the MR gate: "
                  f"{shown}{more}")
        keys = [k for k in keys if mr_eligible(by_shape[k][0])]
    bufs = {k: buffers(by_shape[k][0], dev, args.rows) for k in keys}
    # Rank without the full census print. Measure the default for the selected
    # shapes, then keep the top N by cost. The sweep re-measures the default
    # interleaved with every arm; this pass is only for choosing the shapes.
    base = measure_config(keys, by_shape, iters, repeats, baseline_cfg(), bufs)
    ranked = sorted((k for k in keys if base[k][0] is not None),
                    key=lambda k: -(len(by_shape[k]) * base[k][0]))
    if args.top > 0:
        ranked = ranked[:args.top]
    return ranked, bufs


def measure_config(keys, by_shape, iters, repeats, cfg, bufs):
    out = {}
    with env_config({**baseline_cfg(), **cfg}):
        for k in keys:
            l = by_shape[k][0]
            x, y, ah = bufs[k]
            try:
                out[k] = time_us(lambda: launch(l, x, y, ah), iters, repeats)
            except RuntimeError as e:
                print(f"  {fmt_shape(k)}: n/a ({e})", flush=True)
                out[k] = (None, None)
    return out


def interleaved_samples(keys, by_shape, iters, repeats, configs, bufs):
    """Time every (shape, config) in paired, interleaved bursts.

    For each shape, all configs are measured back to back within a burst, and
    the config order is reversed on alternate bursts. Any clock or thermal drift
    then affects both arms of a comparison nearly equally, instead of landing
    entirely on whichever arm happened to run second.
    """
    samples = {name: {k: [] for k in keys} for name, _ in configs}
    full = [(name, {**baseline_cfg(), **cfg}) for name, cfg in configs]
    failed = set()
    for k in keys:
        l = by_shape[k][0]
        x, y, ah = bufs[k]
        fn = lambda: launch(l, x, y, ah)
        for r in range(repeats):
            order = full if r % 2 == 0 else list(reversed(full))
            for name, cfg in order:
                with env_config(cfg):
                    try:
                        us = time_burst(fn, iters)
                    except RuntimeError as e:
                        if (name, k) not in failed:
                            print(f"  {fmt_shape(k)} [{name}]: n/a ({e})",
                                  flush=True)
                            failed.add((name, k))
                        samples[name][k].append(None)
                    else:
                        samples[name][k].append(us)
    return samples


def verify_configs(keys, by_shape, configs, bufs, refs):
    checks = {name: {} for name, _ in configs}
    for k in keys:
        l = by_shape[k][0]
        x, y, ah = bufs[k]
        for name, cfg in configs:
            with env_config({**baseline_cfg(), **cfg}):
                try:
                    checks[name][k] = check_against_reference(
                        l, x, y, ah, refs.get(k))
                except RuntimeError as e:
                    print(f"  {fmt_shape(k)} [{name}]: verify n/a ({e})",
                          flush=True)
                    checks[name][k] = (None, None)
    return checks


def _stats(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    spread = statistics.stdev(vals) if len(vals) > 1 else None
    return statistics.median(vals), spread


def _paired(base_vals, cfg_vals):
    """Median paired difference and whether every pair moves the same way."""
    diffs = [c - b for b, c in zip(base_vals, cfg_vals)
             if b is not None and c is not None]
    if not diffs:
        return None, False
    same = all(d > 0 for d in diffs) or all(d < 0 for d in diffs)
    return statistics.median(diffs), same


def sweep(by_shape, args, dev, iters, repeats):
    keys, bufs = select_shapes(by_shape, args, dev, iters, repeats)
    if not keys:
        print("sweep: no shapes selected")
        return

    # "default#2" is the same configuration as "default", measured in the same
    # interleaved bursts. Its delta is the empirical noise floor: if an
    # identical config moves by X%, no win smaller than that is credible.
    configs = sweep_configs(args.rows) + [("default#2", baseline_cfg())]
    names = [name for name, _ in configs]

    refs = {}
    ref_kind = {}
    if args.verify:
        for k in keys:
            l = by_shape[k][0]
            x, y, ah = bufs[k]
            try:
                refs[k], ref_kind[k] = reference_out(l, x, y, ah)
            except RuntimeError as e:
                print(f"  {fmt_shape(k)}: reference n/a ({e})", flush=True)

    samples = interleaved_samples(keys, by_shape, iters, repeats, configs, bufs)
    checks = (verify_configs(keys, by_shape, configs, bufs, refs)
              if args.verify else {name: {} for name in names})

    # Compare one common shape set everywhere. Dropping a shape from one arm
    # alone would manufacture a win; dropping it from every arm is fair.
    ok = [k for k in keys
          if all(all(samples[n][k][r] is not None for r in range(repeats))
                 for n in names)]
    if not ok:
        print("sweep: no shape completed under every config")
        return
    partial = len(keys) - len(ok)

    calls = {k: len(by_shape[k]) for k in ok}

    def totals(name):
        return [sum(calls[k] * samples[name][k][r] for k in ok) / 1000
                for r in range(repeats)]

    default_totals = totals("default")
    null_deltas = [100.0 * (n - d) / d if d else 0.0
                   for n, d in zip(totals("default#2"), default_totals)]
    noise = max(abs(d) for d in null_deltas)
    threshold = 2.0 * noise
    base_total = statistics.median(default_totals) / args.rows

    print(f"\nsweep over {len(ok)} shapes, m={args.rows}, "
          f"baseline {base_total:.3f} ms/token, "
          f"{repeats} paired bursts x {iters} iters:")
    if partial:
        print(f"  partial: {partial}/{len(keys)} shapes n/a in at least one arm")
    if repeats >= 2:
        print(f"  noise floor (default#2): +-{noise:.2f}% identical-config "
              f"drift, win threshold {threshold:.2f}%")
    else:
        print("  no noise floor (need --repeats >= 2)")

    for name, cfg in configs:
        if name == "default#2":
            continue
        if name == "default":
            cfg_totals = default_totals
            delta = 0.0
            base_med = statistics.median(default_totals)
            spread = (100.0 * statistics.stdev(default_totals) / base_med
                      if len(default_totals) > 1 and base_med else None)
            total = base_total
            verdict = "baseline"
            label = "BASELINE"
        else:
            cfg_totals = totals(name)
            deltas = [100.0 * (c - d) / d if d else 0.0
                      for c, d in zip(cfg_totals, default_totals)]
            delta = statistics.median(deltas)
            spread = statistics.stdev(deltas) if len(deltas) > 1 else None
            same = all(d > 0 for d in deltas) or all(d < 0 for d in deltas)
            if repeats < 2:
                verdict = "no noise estimate"
            elif not same:
                verdict = "sign flips (noise)"
            elif abs(delta) <= threshold:
                verdict = "within noise"
            elif delta < 0:
                verdict = "faster (outside noise)"
            else:
                verdict = "slower (outside noise)"
            total = statistics.median(cfg_totals) / args.rows
            label = "CONFIG"
        spread_txt = f"sd +-{spread:.2f}%" if spread is not None else "sd n/a"
        env = {k: v for k, v in cfg.items() if v is not None}
        print(f"\n{label} {name}  env={env}")
        print(f"  total {total:.3f} ms/token  delta {delta:+.2f}%  "
              f"{spread_txt}  noise {noise:.2f}%  -> {verdict}")
        for k in ok:
            b_us, b_sd = _stats(samples["default"][k])
            c_us, c_sd = _stats(samples[name][k])
            d_us, d_same = _paired(samples["default"][k],
                                   samples[name][k])
            d_pct = 100.0 * d_us / b_us if (d_us is not None and b_us) else 0.0
            # Same noise definition as the aggregate: 2 x the worst
            # identical-config (default#2) paired move for this shape.
            n_pcts = [100.0 * (c - b) / b
                      for b, c in zip(samples["default"][k],
                                      samples["default#2"][k])
                      if b is not None and c is not None and b]
            shape_noise = 2.0 * max((abs(x) for x in n_pcts), default=0.0)
            if name == "default":
                marker = "  (baseline)"
            elif repeats < 2:
                marker = "  (no noise estimate)"
            elif not d_same:
                marker = "  (sign flips)"
            elif abs(d_pct) <= shape_noise:
                marker = "  (within noise)"
            else:
                marker = ""
            line = (f"  {fmt_shape(k):>26} base {b_us:7.1f}+-{fmt_spread(b_sd)} us  "
                    f"cfg {c_us:7.1f}+-{fmt_spread(c_sd)} us  {d_pct:+6.1f}%")
            nan, err = checks.get(name, {}).get(k, (None, None))
            if args.verify and nan is not None:
                err_txt = "n/a" if err is None else f"{err:.2e}"
                line += f"  maxabs={err_txt} nan={nan} ref={ref_kind.get(k, 'n/a')}"
            print(line + marker)

    print("\nsweep note: a forced config that exceeds 64 KB smem falls back to the "
          "generic WMMA path; --verify proves correctness, not that the fast "
          "kernel actually ran.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default=os.environ.get("EXL3_TEST_MODEL", ""))
    ap.add_argument("--mode", choices=("micro", "stream", "both"), default="both")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--census-iters", type=int, default=300)
    ap.add_argument("--repeats", type=int, default=6,
                    help="paired timing bursts per (shape, config); >= 2 "
                         "estimates the run-to-run noise (use an even count)")
    ap.add_argument("--top", type=int, default=16)
    ap.add_argument("--rows", type=int, default=1,
                    help="dense rows per GEMV call (1..8; >1 uses the RDNA "
                         "multi-row path on MR-eligible shapes)")
    ap.add_argument("--gemm", action="store_true",
                    help="also time ext.exl3_gemm when present")
    ap.add_argument("--census", action="store_true",
                    help="cost-ranked per-shape census (ms/token, calls/token)")
    ap.add_argument("--dec", action="store_true",
                    help="census: also time exl3_dec_gemv where dec_supported "
                         "holds (batch-1, gfx11.5, mul1, dense K set, "
                         "EXL3_DEC/EXL3_DEC_DENSE on)")
    ap.add_argument("--sweep", action="store_true",
                    help="sweep the RDNA knobs over the cost-ranked shapes")
    ap.add_argument("--shapes", default="",
                    help='limit --census, --sweep and the micro table to '
                         '"in,out,K,cb;..." shapes (stream stays a pass over '
                         'all eligible linears)')
    ap.add_argument("--verify", action="store_true",
                    help="with --sweep, compare each config against the "
                         "EXL3_GEMV=0 generic GEMV (GEMM/reconstruct fallback "
                         "for K7/K8)")
    args = ap.parse_args()
    if not args.model:
        raise SystemExit("--model or EXL3_TEST_MODEL is required")
    if args.rows < 1 or args.rows > 8:
        raise SystemExit("--rows must be 1..8 (the RDNA multi-row kernel's range)")
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    if args.sweep and args.repeats % 2:
        print("warning: odd --repeats leaves the interleaved order unbalanced; "
              "use an even count")
    if args.iters < 1 or args.census_iters < 1:
        raise SystemExit("--iters and --census-iters must be >= 1")
    if args.verify and not args.sweep:
        print("warning: --verify only applies to --sweep; ignored")
    if args.dec and not args.census:
        print("warning: --dec only applies to --census; ignored")
    if args.dec and args.rows > 1:
        print("warning: --dec measures the batch-1 dec kernel; ignored with --rows > 1")
    ambient = {k: os.environ[k] for k in KNOBS if k in os.environ}
    if args.sweep:
        # Clear before the first launch, not only per config: some routes may
        # sample the environment at first use, so an inherited value must not
        # be visible to that decision.
        for k in KNOBS:
            os.environ.pop(k, None)
        if ambient:
            print("cleared inherited route knobs: " +
                  ", ".join(f"{k}={v}" for k, v in ambient.items()))
    elif ambient:
        print("note: ambient route knobs are in effect for micro/census/stream: " +
              ", ".join(f"{k}={v}" for k, v in ambient.items()))

    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    model.load(progressbar=False)
    dev = torch.device("cuda:0")
    if args.rows > 1 and not rdna_arch(dev):
        print("warning: --rows > 1 expects an RDNA device; this device will "
              "use the generic GEMV path")
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
        return buffers(l, dev, args.rows)

    if args.mode in ("micro", "both"):
        shape_filter = parse_shapes(args.shapes) if args.shapes else None
        seen = {}
        skipped = 0
        for l in linears:
            key = shape_key(l)
            if shape_filter is not None and key not in shape_filter:
                continue
            if args.rows > 1 and not mr_eligible(l):
                skipped += 1
                continue
            seen.setdefault(key, l)
        if skipped:
            print(f"rows>1: skipping {skipped} micro linears outside the MR gate")
        rows = sorted(seen.items(), key=lambda kv: -shape_bytes(kv[0]))
        has_gemm = args.gemm and hasattr(ext, "exl3_gemm")
        head = f"{'k':>6} {'n':>7} {'K':>2} {'cb':>2} {'MB':>8} {'us':>8} {'GB/s':>7}"
        if has_gemm:
            head += f" {'gemm_us':>8} {'gemm_GB/s':>9}"
        print(f"\nmicro timing (m={args.rows}; us is per call)")
        print(head)
        for (k, n, K, cb), l in rows[:args.top]:
            x, y, ah = buffers_for(l)
            b = shape_bytes((k, n, K, cb))
            f = lambda: launch(l, x, y, ah)
            try:
                us, _ = time_us(f, args.iters, args.repeats)
            except RuntimeError as e:
                print(f"  {k}x{n} K{K} cb{cb}: n/a ({e})", flush=True)
                continue
            line = f"{k:6d} {n:7d} {K:2d} {cb:2d} {b/1e6:8.2f} {us:8.1f} {b/us/1e3:7.0f}"
            if has_gemm:
                g = lambda: ext.exl3_gemm(x, l.trellis, y, l.suh, ah, l.svh, -1,
                                          l.mcg, l.mul1, 0)
                try:
                    usg = time_us(g, args.iters, args.repeats)[0]
                except RuntimeError:
                    line += f" {'n/a':>8} {'n/a':>9}"
                else:
                    line += f" {usg:8.1f} {b/usg/1e3:9.0f}"
            print(line, flush=True)

    if args.mode in ("stream", "both"):
        work = []
        skipped = 0
        for l in linears:
            if args.rows > 1 and not mr_eligible(l):
                skipped += 1
                continue
            x, y, ah = buffers_for(l)
            work.append((l, x, y, ah))
        stream_bytes = sum(shape_bytes(shape_key(l)) for l, _, _, _ in work)
        if skipped:
            print(f"rows>1: skipping {skipped} stream linears outside the MR gate")
        if not work:
            print("stream: no eligible linears")
        else:
            def run_gemv():
                for l, x, y, ah in work:
                    launch(l, x, y, ah)

            def run_gemm():
                for l, x, y, ah in work:
                    ext.exl3_gemm(x, l.trellis, y, l.suh, ah, l.svh, -1,
                                  l.mcg, l.mul1, 0)

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

            print(f"\nstreaming pass over {len(work)} linears "
                  f"({stream_bytes / 1e9:.3f} GB, median of 3, m={args.rows}):")
            try:
                ms = statistics.median([stream(run_gemv) for _ in range(3)])
            except RuntimeError as e:
                print(f"  exl3_gemv: n/a ({e})")
            else:
                per_tok = (f" ({ms / args.rows:.2f} ms/token)"
                           if args.rows > 1 else "")
                print(f"  exl3_gemv: {ms:.2f} ms{per_tok}  "
                      f"{stream_bytes / ms * 1e-6:.0f} GB/s")
            if args.gemm and hasattr(ext, "exl3_gemm"):
                try:
                    msg = statistics.median([stream(run_gemm) for _ in range(3)])
                except RuntimeError as e:
                    print(f"  exl3_gemm: n/a ({e})")
                else:
                    per_tok = (f" ({msg / args.rows:.2f} ms/token)"
                               if args.rows > 1 else "")
                    print(f"  exl3_gemm: {msg:.2f} ms{per_tok}  "
                          f"{stream_bytes / msg * 1e-6:.0f} GB/s")

    if args.census:
        census(filter_shapes(by_shape, args.shapes), dev,
               args.census_iters, args.repeats, args.top,
               rows=args.rows, dec=args.dec)

    if args.sweep:
        sweep(by_shape, args, dev, args.census_iters, args.repeats)


if __name__ == "__main__":
    main()
