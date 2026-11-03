#!/usr/bin/env python
"""In-process multi-sequence step scanner: ms per decode/verify step vs token rows, and row exactness.

  tools/depth/rows_scan.py MODEL_DIR --serve glm|qwen -o out.json [--depth 4096] [--configs 1x1,1x2,...,4x8]
        [--groups] [--exact] [--steps 12] [--kernels]

A config BxW is one step of B sequences (jobs), W token rows each (W = 1 + drafted tokens), B * W rows in all: the
shape the generator verifies when B requests speculate together. The model is loaded once with the SERVED environment
(same tables as depth_profile.py, plus the multi-session defaults of the server). Every cache tensor is filled with
synthetic data once; each sequence owns its pages, at position --depth.
  --modes    comma list of plain (one forward of all rows, as the generator runs a step without groups), perseq (the same single
             forward with params["row_inv_per_seq"]: attention layers run one sequence at a time), groups (the generator's
             EXL3_ROW_INV_GROUPS=1 step: forwards of at most EXL3_ROW_INV_GROUP_ROWS (8) rows, jobs kept whole, time = their sum)
  --exact    compare the logits of every sequence in the batched step with the same sequence decoded alone with the same W
             rows (W <= 8: the exact paths), and row 0 with the plain one-token decode; bitwise (torch.equal) and max |diff|
  --kernels  stamp a time window around 4 steps of each config (tools/depth/kernels.py joins them with a rocprofv3 trace)
Output JSON is rewritten after every config. Needs ~110 GiB free for GLM.
"""
import argparse, ast, json, os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PAGE = 256


def served_env(which):
    path, name = {"qwen": ("tools/qwen/serve.py", "SERVE_ENV"), "glm": ("tools/glm/serve.py", "SPEED_ENV")}[which]
    tree = ast.parse(open(os.path.join(ROOT, path)).read()).body
    out = {}
    for node in tree:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) in (name, "MULTI_SESSION_ENV") for t in node.targets):
            out.update(dict(ast.literal_eval(node.value)))
    if not out:
        raise SystemExit(f"{name} not found in {path}")
    return out


def avail_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1048576
    return -1.0


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--serve", choices=["glm", "qwen", "none"], default="none")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--depth", type=int, default=4096)
    ap.add_argument("--configs", default=",".join(f"{b}x{w}" for b in (1, 2, 4) for w in range(1, 9)))
    ap.add_argument("--modes", default="plain", help="plain = one forward of all rows; groups = forwards of <= EXL3_ROW_INV_GROUP_ROWS rows; comma list")
    ap.add_argument("--exact", action="store_true")
    ap.add_argument("--kernels", action="store_true")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--min-avail-gb", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE applied before the engine import (overrides the served table)")
    args = ap.parse_args()
    if args.serve != "none":
        for k, v in served_env(args.serve).items():
            os.environ.setdefault(k, v)
        os.environ.setdefault("EXL3_MOE_UNION_V2", "1")
    for kv in args.env:
        k, v = kv.split("=", 1)
        os.environ[k] = v
    os.environ.setdefault("TQDM_DISABLE", "1")
    sys.path.insert(0, ROOT)
    import torch
    from exllamav3 import Config, Model, Cache
    from exllamav3.generator.row_groups import plan_groups, group_rows_cap
    torch.set_grad_enabled(False)

    cfgs = [tuple(int(x) for x in c.split("x")) for c in args.configs.split(",")]
    Bmax = max(b for b, _ in cfgs)
    Wmax = max(w for _, w in cfgs)
    PP = (args.depth + args.steps + Wmax + 40 + PAGE - 1) // PAGE + 16       # pages per sequence
    PP = (PP + 15) // 16 * 16
    cap = Bmax * PP * PAGE
    RES = {"model": args.model, "args": vars(args), "env": {k: v for k, v in os.environ.items() if k.startswith(("EXL3_", "MPW", "QWSPEC"))},
           "points": {}}

    def save():
        tmp = args.out + ".tmp"
        json.dump(RES, open(tmp, "w"), indent=1)
        os.replace(tmp, args.out)

    t0 = time.time()
    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=cap, max_history=max(3, Wmax), max_batch_size=Bmax)
    model.load(device="cuda:0", progressbar=False)
    RES["load_s"] = round(time.time() - t0, 1)
    RES["avail_after_load_gb"] = round(avail_gb(), 1)
    RES["row_inv_groups_cap"] = model.caps.get("row_inv_groups")
    print(f"loaded {RES['load_s']}s, cache {cap} tokens, avail {RES['avail_after_load_gb']} GiB", flush=True)

    g = torch.Generator(device="cuda:0"); g.manual_seed(args.seed)
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
    save()

    bt_all = torch.arange(Bmax * PP, dtype=torch.int32).view(Bmax, PP)
    V = config.vocab_size if hasattr(config, "vocab_size") else 100000
    gen = torch.Generator().manual_seed(args.seed)
    ids_all = torch.randint(1000, min(V, 100000), (Bmax, Wmax + 64), generator=gen)

    states = []

    def fresh(n, pos):
        for s in states:
            s.free()
        states.clear()
        for _ in range(n):
            states.append(cache.get_test_state(pos))

    def fwd(seqs, w, pos, step=0, group_rows=None, perseq=False):
        """One step of the sequences `seqs` (indices), w rows each, at position pos."""
        B = len(seqs)
        ids = ids_all[seqs][:, :w]
        pages = (pos + w + PAGE - 1) // PAGE
        own = (pages + 15) // 16 * 16
        bt = bt_all[seqs][:, :min(own, PP)]
        verify = B * w > 1
        base = {"attn_mode": "flash_attn", "cache": cache}
        if verify:
            base.update({"recurrent_history": True, "dflash_verify": True})
        cs = torch.full((B,), pos, dtype=torch.int32)
        if group_rows is None:
            p = dict(base, block_table=bt, cache_seqlens=cs, recurrent_states=[states[i] for i in seqs])
            if perseq and B > 1:
                p["row_inv_per_seq"] = True
            if verify:
                p["dflash_table_pages"] = [own] * B
            return model.forward(ids, p)
        out, r0 = [], 0
        for j0, j1 in plan_groups([1] * B, w, group_rows):
            r1 = j1
            p = dict(base, block_table=bt[r0:r1], cache_seqlens=cs[r0:r1], recurrent_states=[states[seqs[i]] for i in range(j0, j1)])
            if (r1 - r0) * w > 1:
                p["dflash_verify"] = True
                p["recurrent_history"] = True
                p["dflash_table_pages"] = [own] * (r1 - r0)
            if r1 - r0 > 1:
                p["row_inv_per_seq"] = True
            if len(plan_groups([1] * B, w, group_rows)) > 1:
                p["pinned_staging"] = False
            out.append(model.forward(ids[r0:r1], p))
            r0 = r1
        return torch.cat(out, dim=0)

    def timed(B, w, grp, perseq):
        seqs = list(range(B))
        fresh(B, args.depth)
        for k in range(3):
            lg = fwd(seqs, w, args.depth + k, group_rows=grp, perseq=perseq)
            _ = lg[0, -1].argmax().item()
        fresh(B, args.depth)
        ts = []
        for k in range(args.steps):
            torch.cuda.synchronize(); a = time.perf_counter()
            lg = fwd(seqs, w, args.depth + k, group_rows=grp, perseq=perseq)
            _ = lg[:, -1].argmax(dim=-1).tolist()          # the served step ends with the token readback
            ts.append(time.perf_counter() - a)
        return ts

    med = lambda x: sorted(x)[len(x) // 2]
    for mode, (B, w) in [(m, c) for m in args.modes.split(",") for c in cfgs]:
        grp = group_rows_cap() if mode == "groups" else None
        perseq = mode == "perseq"
        key = f"{mode}:{B}x{w}"
        P = {"B": B, "w": w, "rows": B * w, "mode": mode}
        try:
            ts = timed(B, w, grp, perseq)
            P.update({"ms": round(1e3 * med(ts), 3), "ms_min": round(1e3 * min(ts), 3), "tok_s_rows": round(B * w / med(ts), 1)})
            if args.kernels:
                fresh(B, args.depth)
                torch.cuda.synchronize()
                w0 = {"mono": time.monotonic_ns(), "boot": time.clock_gettime_ns(time.CLOCK_BOOTTIME)}
                for k in range(4):
                    fwd(list(range(B)), w, args.depth + k, group_rows=grp, perseq=perseq)
                torch.cuda.synchronize()
                w1 = {"mono": time.monotonic_ns(), "boot": time.clock_gettime_ns(time.CLOCK_BOOTTIME)}
                P["window"] = {"start": w0, "end": w1, "steps": 4}
            if args.exact and B * w > 1:
                seqs = list(range(B))
                fresh(B, args.depth)
                lb = fwd(seqs, w, args.depth, group_rows=grp, perseq=perseq).float().clone()
                solo_eq, solo_dmax, r0_eq = [], 0.0, []
                for b in seqs:
                    fresh(Bmax, args.depth)
                    ls = fwd([b], w, args.depth).float()
                    solo_eq.append(bool(torch.equal(lb[b], ls[0])))
                    solo_dmax = max(solo_dmax, (lb[b] - ls[0]).abs().max().item())
                    if w > 1:
                        fresh(Bmax, args.depth)
                        l1 = fwd([b], 1, args.depth).float()
                        r0_eq.append(bool(torch.equal(lb[b, 0], l1[0, 0])))
                P["exact_vs_solo_w"] = solo_eq
                P["max_abs_vs_solo_w"] = solo_dmax
                if r0_eq:
                    P["row0_eq_plain"] = r0_eq
            print(f"{key:>5s} rows {B*w:3d}: {P['ms']:8.2f} ms  ({P['tok_s_rows']:7.1f} rows/s)"
                  + (f"  exact_vs_solo {P.get('exact_vs_solo_w')} row0 {P.get('row0_eq_plain')}" if args.exact and B * w > 1 else ""), flush=True)
        except Exception as e:                                  # a config the engine rejects is a data point, not a crash
            P["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            print(f"{key:>5s} rows {B*w:3d}: ERROR {P['error']}", flush=True)
        RES["points"][key] = {"step": P}      # nested like depth_profile.py so tools/depth/kernels.py joins the windows
        a = avail_gb()
        if a < args.min_avail_gb:
            RES["abort"] = f"MemAvailable {a:.1f} GiB < {args.min_avail_gb} after {key}"
            save(); print(RES["abort"], flush=True); sys.exit(3)
        save()
    save()


def main():
    import torch
    with torch.inference_mode():
        _main()


if __name__ == "__main__":
    main()
