#!/usr/bin/env python
"""In-process depth profiler: per-step cost vs context depth in minutes, no real prefill.

  tools/depth/depth_profile.py MODEL_DIR --serve glm|qwen -o out.json [--depths 8192,...] [--cache-bits 16|8]
        [--ndt 2] [--chunk 4096] [--reps 3] [--kernels] [--no-prefill] [--no-decode]

Loads the model once with the SERVED environment (the SERVE_ENV / SPEED_ENV tables of tools/{qwen,glm}/serve.py,
read with ast, applied before exllamav3 is imported), allocates one cache for the deepest depth and fills EVERY
cache tensor with synthetic data once (random fp16 K/V / latent / indexer keys; random packed words and sane
scales for quantized pages), then for each depth D:
  - recurrent state: Cache.get_test_state(D)
  - decode: N timed single-token steps at position D (spec off), and N verify steps of ndt+1 rows
  - prefill: one chunk of --chunk rows at offset D (model.prefill, no lm_head)
  - --kernels: stamps time windows (4 decode/verify steps, one prefill chunk); run under rocprofv3 --kernel-trace and
    join with tools/depth/kernels.py -> us per kernel name per step, GPU busy vs wall (the gap is host / launch cost)
Output JSON is rewritten after every depth, so a crash keeps what finished. Decode is content independent
except top-k selection (random keys give the realistic score spread of a long document).
The speculative drafter is NOT loaded: spec-off and verify are the target model's cost; MTP acceptance is content.
Run inside:  flock ~/ds41_work/gpu.lock ~/ds41_work/cur1510/quiet.sh ...  (needs ~110 GiB free for GLM).
"""
import argparse, ast, collections, hashlib, json, os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PAGE = 256


def served_env(which):
    """EXL3_* defaults of the server, read from its source without importing it."""
    path, name = {"qwen": ("tools/qwen/serve.py", "SERVE_ENV"), "glm": ("tools/glm/serve.py", "SPEED_ENV")}[which]
    for node in ast.parse(open(os.path.join(ROOT, path)).read()).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            v = ast.literal_eval(node.value)
            return dict(v)
    raise SystemExit(f"{name} not found in {path}")


def avail_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1048576
    return -1.0


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--serve", choices=["glm", "qwen", "none"], default="none", help="apply this server's EXL3_* environment")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--depths", default="8192,32768,65536,131072,262144,500000")
    ap.add_argument("--cache-bits", type=int, default=16)
    ap.add_argument("--ndt", type=int, default=2, help="drafted tokens: verify step = ndt + 1 rows")
    ap.add_argument("--chunk", type=int, default=int(os.environ.get("EXL3_PREFILL_CHUNK", "4096")))
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--reps", type=int, default=2, help="prefill chunk repetitions per depth (interleaved over depths)")
    ap.add_argument("--kernels", action="store_true")
    ap.add_argument("--no-prefill", action="store_true")
    ap.add_argument("--no-decode", action="store_true")
    ap.add_argument("--min-avail-gb", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ndt-list", default="", help="draft depths to scan in ONE load, interleaved per --reps visit (e.g. 3,4,5,6 = verify rows 4..7); overrides --ndt")
    ap.add_argument("--arm", action="append", default=[], help="NAME:K=V,K2=V2 (repeatable): A/B arms of exllamav3.util.knobs switched in this ONE load")
    ap.add_argument("--order", default="", help="visit order of the arms, e.g. ABBAAB (default: each arm once, in the order given)")
    args = ap.parse_args()
    if args.serve != "none":
        for k, v in served_env(args.serve).items():
            os.environ.setdefault(k, v)
        os.environ.setdefault("EXL3_MOE_UNION_V2", "1")
    os.environ.setdefault("TQDM_DISABLE", "1")
    sys.path.insert(0, ROOT)
    import torch
    from exllamav3 import Config, Model, Cache
    from exllamav3.cache import CacheLayer_quant
    torch.set_grad_enabled(False)

    depths = [int(x) for x in args.depths.split(",")]
    ndt_scan = [int(x) for x in args.ndt_list.split(",")] if args.ndt_list else []
    R = args.ndt + 1
    Rmax = max([R] + [n + 1 for n in ndt_scan])
    top = max(depths) + args.chunk + args.steps + Rmax + 64
    cap = ((top + PAGE - 1) // PAGE + 1) * PAGE
    RES = {"model": args.model, "args": vars(args), "env": {k: v for k, v in os.environ.items() if k.startswith(("EXL3_", "MPW"))},
           "points": {}}

    def save():
        tmp = args.out + ".tmp"
        json.dump(RES, open(tmp, "w"), indent=1)
        os.replace(tmp, args.out)

    def guard(tag):
        a = avail_gb()
        if a < args.min_avail_gb:
            RES["abort"] = f"MemAvailable {a:.1f} GiB < {args.min_avail_gb} at {tag}"
            save(); print(RES["abort"], flush=True); sys.exit(3)

    t0 = time.time()
    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    qkw = dict(layer_type=CacheLayer_quant, k_bits=args.cache_bits, v_bits=args.cache_bits) if args.cache_bits < 16 else {}
    cache = Cache(model, max_num_tokens=cap, max_history=max(3, Rmax), max_batch_size=1, **qkw)
    model.load(device="cuda:0", progressbar=False)
    RES["load_s"] = round(time.time() - t0, 1)
    RES["avail_after_load_gb"] = round(avail_gb(), 1)
    guard("after_load")

    # ---- synthetic fill of every cache tensor
    g = torch.Generator(device="cuda:0"); g.manual_seed(args.seed)
    nbytes = 0
    for layer in cache.layers.values():
        for name, v in vars(layer).items():
            if name.startswith("_") or not (isinstance(v, torch.Tensor) and v.is_cuda):
                continue
            f = v.view(-1)
            scale_like = name in ("sk", "sv", "pool_s")
            for i in range(0, f.numel(), 1 << 26):
                sl = f[i: i + (1 << 26)]
                if sl.dtype == torch.half:
                    x = torch.randn(sl.shape, generator=g, device=sl.device, dtype=torch.float32)
                    sl.copy_((x.abs_().mul_(0.05) if scale_like else x.mul_(0.5)))
                elif sl.dtype in (torch.int32, torch.int16, torch.uint8, torch.int8):
                    sl.copy_(torch.randint(-2**31, 2**31 - 1, sl.shape, generator=g, device=sl.device, dtype=torch.int64).to(sl.dtype))
            nbytes += v.numel() * v.element_size()
    RES["cache_bytes"] = nbytes
    RES["cache_bytes_per_token"] = nbytes / cap
    print(f"loaded {RES['load_s']}s, cache {cap} tokens = {nbytes/1e9:.2f} GB filled, avail {RES['avail_after_load_gb']} GiB", flush=True)
    save()

    bt_full = torch.arange(cap // PAGE, dtype=torch.int32).unsqueeze(0)
    st = {"s": cache.get_new_state()}

    def fresh(pos):
        st["s"].free()
        st["s"] = cache.get_test_state(pos)

    def fwd(ids, pos, prefill=False, verify=False):
        T = ids.shape[-1]
        pages = (pos + T + PAGE - 1) // PAGE
        params = {"attn_mode": "flash_attn", "block_table": bt_full[:, :pages], "cache": cache,
                  "cache_seqlens": torch.tensor([pos], dtype=torch.int32), "recurrent_states": [st["s"]]}
        if verify:      # the Generator's verify forward: R-row fast paths are gated on these markers
            params.update({"recurrent_history": True, "dflash_verify": True,
                           "dflash_table_pages": [(pages + 15) // 16 * 16]})
        return model.prefill(ids, params) if prefill else model.forward(ids, params)

    V = config.vocab_size if hasattr(config, "vocab_size") else 100000
    gen = torch.Generator().manual_seed(args.seed)
    chunk_ids = torch.randint(1000, min(V, 100000), (1, args.chunk), generator=gen)
    one = torch.randint(1000, min(V, 100000), (1, 1), generator=gen)
    rows = torch.randint(1000, min(V, 100000), (1, R), generator=gen)
    rows_of = {R: rows}

    def rows_for(r):
        if r not in rows_of:                       # one synthetic row set per row count, reused across visits
            rows_of[r] = torch.randint(1000, min(V, 100000), (1, r), generator=gen)
        return rows_of[r]

    import hashlib
    HASH = hashlib.md5()

    def timed_steps(ids, d, n, verify=False):
        ts = []
        for k in range(n):
            torch.cuda.synchronize(); a = time.perf_counter()
            lg = fwd(ids, d + k, verify=verify)
            _ = lg[0, -1].argmax().item()            # the served step ends with the token readback
            if os.environ.get("DP_HASH") == "1":     # bit-identity probe: md5 of the logits of every step
                HASH.update(lg.float().cpu().numpy().tobytes())
            ts.append(time.perf_counter() - a)
        return ts

    def window(fn, steps):
        """Time window stamps (ns, two clocks) around fn(): tools/depth/kernels.py joins them with a rocprofv3
        kernel trace (run the whole profiler under `rocprofv3 --kernel-trace`). torch.profiler sees no GPU
        activity on this stack."""
        torch.cuda.synchronize()
        w0 = {"mono": time.monotonic_ns(), "boot": time.clock_gettime_ns(time.CLOCK_BOOTTIME)}
        fn(); torch.cuda.synchronize()
        w1 = {"mono": time.monotonic_ns(), "boot": time.clock_gettime_ns(time.CLOCK_BOOTTIME)}
        return {"start": w0, "end": w1, "steps": steps}

    med = lambda x: sorted(x)[len(x) // 2]
    for d in depths:
        RES["points"][str(d)] = {}
    def decode_points(P, hashes=None, R_=None, ids_rows=None):
        """decode and verify rows at depth d into P; hashes (arm runs) gets the md5 of the first step's last-row logits"""
        R_ = R_ or R
        ids_rows = rows if ids_rows is None else ids_rows
        for label, ids, n_rows in (("decode", one, 1), ("verify", ids_rows, R_)):
            ver = label == "verify"
            fresh(d)
            timed_steps(ids, d, 3, ver)                  # graph capture / JIT / allocator at this length
            fresh(d)
            if hashes is not None:
                hashes[label] = hashlib.md5(fwd(ids, d, verify=ver)[0, -1].float().cpu().numpy().tobytes()).hexdigest()[:12]
                fresh(d)
            ts = timed_steps(ids, d, args.steps, ver)
            P[label] = {"ms": round(1e3 * med(ts), 3), "ms_min": round(1e3 * min(ts), 3), "rows": n_rows,
                        "tok_s_per_row": round(n_rows / med(ts), 2)}
            if args.kernels:
                fresh(d)
                P[label]["window"] = window(lambda: [fwd(ids, d + k, verify=ver) for k in range(4)], 4)

    if not args.no_decode and not args.arm and not ndt_scan:
        for d in depths:
            P = RES["points"][str(d)]
            decode_points(P)
            if os.environ.get("DP_HASH") == "1": print(f"[hash] depth {d} {HASH.hexdigest()}", flush=True)
            print(f"[dec] depth {d}: decode {P['decode']['ms']} ms, verify({R} rows) {P['verify']['ms']} ms", flush=True)
            guard(f"decode@{d}"); save()
    if not args.no_decode and args.arm:
        from exllamav3.util import knobs
        arms = {}
        for spec in args.arm:
            name, _, kv = spec.partition(":")
            arms[name] = {k: (None if v in ("", "default") else v) for k, _, v in (i.partition("=") for i in kv.split(",") if i)}
        order = args.order or "".join(arms)
        names = {k for a in arms.values() for k in a}
        RES["arms"] = {a: {"knobs": arms[a], "visits": []} for a in arms}
        for n, a in enumerate(order):
            knobs.set_knobs({k: arms[a].get(k) for k in sorted(names)})        # one load: graphs and caches are dropped when a value changes
            visit = {"n": n, "points": {}}
            for d in depths:
                P, H = {}, {}
                decode_points(P, H)
                P["md5"] = H
                visit["points"][str(d)] = P
                print(f"[dec] visit {n} arm {a} depth {d}: decode {P['decode']['ms']} ms ({H['decode']}), verify({R} rows) {P['verify']['ms']} ms ({H['verify']})", flush=True)
                guard(f"decode@{d}"); save()
            RES["arms"][a]["visits"].append(visit)
        knobs.set_knobs({k: None for k in names})
        for a in arms:
            for d in depths:
                vs = [v["points"][str(d)] for v in RES["arms"][a]["visits"]]
                print(f"[arm {a}] depth {d}: decode {med([v['decode']['ms'] for v in vs])} ms  verify {med([v['verify']['ms'] for v in vs])} ms  "
                      f"md5 {'stable' if all(v['md5'] == vs[0]['md5'] for v in vs) else 'UNSTABLE'} {vs[0]['md5']}", flush=True)
    if ndt_scan and not args.no_decode:
        RES["rows_scan"] = {str(n + 1): {} for n in ndt_scan}
        for rep in range(args.reps):                 # visits interleaved over row counts: one CPU window for all of them
            for n in ndt_scan:
                r = n + 1
                for d in depths:
                    P = {}
                    decode_points(P, None, r, rows_for(r))
                    RES["rows_scan"][str(r)].setdefault(str(d), []).append(P)
                    print(f"[rows] rep {rep} rows {r} depth {d}: decode {P['decode']['ms']} ms, "
                          f"verify {P['verify']['ms']} ms (min {P['verify']['ms_min']})", flush=True)
                save(); guard(f"rows@{r}")
        for n in ndt_scan:
            r = n + 1
            for d in depths:
                vs = RES["rows_scan"][str(r)][str(d)]
                print(f"[rows {r}] depth {d}: verify med {med([v['verify']['ms'] for v in vs])} ms "
                      f"(min {min(v['verify']['ms_min'] for v in vs)}, max {max(v['verify']['ms'] for v in vs)}), "
                      f"decode med {med([v['decode']['ms'] for v in vs])} ms over {len(vs)} visits", flush=True)
        save()
    if not args.no_prefill:
        fresh(depths[0])
        for _ in range(1):
            fwd(chunk_ids, depths[0], prefill=True); torch.cuda.synchronize()     # warm
        acc = {d: [] for d in depths}
        for rep in range(args.reps):
            for d in depths:
                fresh(d)
                torch.cuda.synchronize(); a = time.perf_counter()
                fwd(chunk_ids, d, prefill=True)
                torch.cuda.synchronize(); acc[d].append(time.perf_counter() - a)
                guard(f"prefill@{d}")
        for d in depths:
            P = RES["points"][str(d)]
            P["prefill"] = {"ms": round(1e3 * med(acc[d]), 2), "all_ms": [round(1e3 * x, 2) for x in acc[d]], "chunk": args.chunk,
                            "tok_s": round(args.chunk / med(acc[d]), 1)}
            if args.kernels:
                fresh(d)
                P["prefill"]["window"] = window(lambda: fwd(chunk_ids, d, prefill=True), 1)
            print(f"[pf] depth {d}: {P['prefill']['ms']} ms/chunk = {P['prefill']['tok_s']} tok/s", flush=True)
            save()
    save()


def main():
    # the cache tensors are created by the engine under inference mode: updating them needs it too
    import torch
    with torch.inference_mode():
        _main()


if __name__ == "__main__":
    main()
