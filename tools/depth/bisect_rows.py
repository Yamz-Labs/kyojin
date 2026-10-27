#!/usr/bin/env python
"""First module where a batched BxW verify step differs from the same sequences decoded alone (rows_scan's setup, same arguments).

  tools/depth/bisect_rows.py MODEL_DIR --serve glm|qwen -o out.json --configs BxW --bis BxW [--subblock N]
Wraps the forward of every top-level module and of the sub-modules of block N (attention, MLP, hyperconnections, projections), records
inputs and outputs of the batched step and of each solo step, and prints the mismatches per sequence (bitwise)."""
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
    ap.add_argument("--exact", action="store_true"); ap.add_argument("--bis", default="4x3"); ap.add_argument("--subblock", type=int, default=2)
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


    B, w = [int(x) for x in args.bis.split("x")]
    rec = {}
    cur = [None]
    def wrap(i, m):
        f = m.forward
        def g(*a, **k):
            y = f(*a, **k)
            if i >= 1000 and torch.is_tensor(a[0]): rec.setdefault(cur[0], {})[i + 5000] = (type(m).__name__ + '.IN', getattr(m, 'key', ''), a[0].detach().float().clone())
            t = y[0] if isinstance(y, tuple) else y
            if torch.is_tensor(t):
                rec.setdefault(cur[0], {})[i] = (type(m).__name__, getattr(m, "key", ""), t.detach().float().clone())
            return y
        m.forward = g
    for i, m in enumerate(model.modules):
        wrap(i, m)
    blk = model.modules[args.subblock]
    for j, nm in enumerate(("attn_norm", "attn", "attn_post_norm", "mlp_norm", "mlp", "mlp_post_norm", "attn_hc", "mlp_hc")):
        sm = getattr(blk, nm, None)
        if sm is not None: wrap(1000 + j, sm)
    def wrapm(i, obj, name):
        f = getattr(obj, name)
        def g(*a, **k):
            if torch.is_tensor(a[0]): rec.setdefault(cur[0], {})[i + 5000] = (name + ".IN", "", a[0].detach().float().clone())
            y = f(*a, **k)
            for n, t in enumerate(y if isinstance(y, tuple) else (y,)):
                if torch.is_tensor(t): rec.setdefault(cur[0], {})[i + n] = (name + f".out{n}", "", t.detach().float().clone())
            return y
        setattr(obj, name, g)
    for j, hn in enumerate(("attn_hc", "mlp_hc")):
        hc = getattr(blk, hn, None)
        if hc is None: continue
        for q, mn in enumerate(("mix", "mix_norm", "apply_")):
            if hasattr(hc, mn): wrapm(3000 + 100 * j + 10 * q, hc, mn)
    at = getattr(blk, "attn", None)
    if at is not None:
        for j, (nm, ch) in enumerate([(k, v) for k, v in vars(at).items() if hasattr(v, "forward") and hasattr(v, "key") and v is not at]):
            print("wrap attn child", 4000 + j, nm, type(ch).__name__, flush=True); wrap(4000 + j, ch)
    mlp = getattr(blk, "mlp", None)
    for j, nm in enumerate(("gates", "ups", "downs")):
        l = getattr(mlp, nm, None)
        if l: wrap(2000 + j, l[0]); inn = getattr(l[0], "inner", None); inn is not None and wrap(2100 + j, inn)
    seqs = list(range(B))
    fresh(B, args.depth); cur[0] = "b"; fwd(seqs, w, args.depth)
    for b in seqs:
        fresh(Bmax, args.depth); cur[0] = f"s{b}"; fwd([b], w, args.depth)
    for b in seqs:
        n = 0
        for i in sorted(rec["b"]):
            nm, key, tb = rec["b"][i]; ts = rec[f"s{b}"][i][2]
            if tb.dim() >= 2 and tb.shape[0] == B and ts.shape[0] == 1 and tb.shape[1] == w: x = tb[b]; y = ts[0]
            elif tb.shape[0] == B * w and ts.shape[0] == w: x = tb[b*w:(b+1)*w]; y = ts
            else:
                if i >= 1000: print(f"seq {b}: sub {i} {nm} shape {tuple(tb.shape)} not comparable", flush=True)
                continue
            if not torch.equal(x, y):
                if n < 3 or i >= 1000: print(f"seq {b}: mismatch module {i} {nm} {key} shape {tuple(tb.shape)} maxabs {(x - y).abs().max().item():.4g}", flush=True)
                n += 1
            elif i >= 1000: print(f"seq {b}: sub {i} {nm} equal", flush=True)
        if n == 0: print(f"seq {b}: all modules equal", flush=True)

def main():
    import torch
    with torch.inference_mode():
        _main()

if __name__ == "__main__":
    main()
