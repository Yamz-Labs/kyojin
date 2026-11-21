"""KV bytes per token, MEASURED (runner job 50_k3_e_kvbytes, GPU, one boot per context, no request): loads the engine at K3_CTXS (default 65536,131072,262144), reads
torch.cuda.memory_allocated() after load (GraphDecoder.prepare allocates every static K / indexer buffer at the full context) and walks the buffers by component.
SLOPE = (alloc(c2) - alloc(c1)) / (cap2 - cap1) in bytes / token must match tools/hy4/kv_bytes.py. 'FIRES kv <n> mtp <0|1>' counts the layers that hold an Fp8KV cache (trunk) and the MTP draft cache.
parent: python kv_bytes_probe.py (flags in the environment); it re-runs itself with K3_CHILD=<ctx> for each context (a fresh process frees the GPU)."""
import json, os, subprocess, sys, time

E = os.environ
here = os.path.dirname(os.path.abspath(__file__))


def nb(t):
    if t is None:
        return 0
    if hasattr(t, "nbytes") and callable(t.nbytes):
        return int(t.nbytes())
    return int(t.numel() * t.element_size())


def touch(eng, alloc, sync, mem, ae, IXF, RR):
    """a 2048-token prefill + 2 verify calls of 4 rows, then the lazy rope tables / row evaluations read back -> list of failure strings. KVTOUCH carries the allocation AFTER the touch
    (the parent prints SLOPE_TOUCH from it: it shows the lazy per-token tables that the load-only slope cannot see)"""
    ids = [(i * 7919) % 50000 + 1000 for i in range(2048 + 8)]
    eng._t0 = time.perf_counter()               # prefill() reads it (qual_llc sets it the same way)
    eng.reset()
    eng.prefill(ids[:2048])
    eng.phase("decode")
    for i in range(2048, 2056, 4):
        eng.GR.step_rows(ids[i:i + 4])
    sync()
    rows_attn = sum(t.shape[0] for t in ae._TAB.values())
    rows_ix = sum(cs[0].shape[0] for cs in IXF.FusedKPoolIndexer._TABS.values())
    print("KVTOUCH", json.dumps({"rope_table_rows_attn": rows_attn, "rope_table_rows_ix": rows_ix, "rope_rows_evals": dict(RR.STATS), "alloc_before_B": alloc, "alloc_after_B": mem()}), flush=True)
    if RR.ENABLED and (rows_attn or rows_ix):
        return [f"HY4_ROPE_POS on but a full-context rope table exists (attn rows {rows_attn}, ix rows {rows_ix})"]
    return []


def child(ctx):
    sys.path.insert(0, here)
    tree = os.path.dirname(os.path.dirname(here))
    sys.path.insert(0, tree)
    import serve
    serve.apply_env(ctx)
    from engine import Hy4Engine
    import torch
    eng = Hy4Engine(tree, E["GGUF"], ctx=ctx, k=3, chunk=8192, prime_rows=4096)
    eng.load(lambda s: None)
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated()
    from exllamav3.loader import step_nosync as ns, kv_fp8, attn_eager as ae, rope_rows as RR, indexer_fused as IXF
    st = ns.state(eng.M)
    comp = {"kv_full_fp8": 0, "kv_full_fp32": 0, "kv_window": 0, "mtp_k": 0, "ix_pool": 0, "ix_vgb": 0, "tok_ar": nb(st.tok) + nb(st.ar), "rope_attn": sum(nb(t) for t in ae._TAB.values()),
            "rope_ix": sum(nb(t) for cs in IXF.FusedKPoolIndexer._TABS.values() for t in cs)}
    fires = 0
    for i, b in st.kv.items():
        if isinstance(b, kv_fp8.Fp8KV):
            comp["kv_full_fp8"] += nb(b)
            fires += 1
        elif eng.M._layer(i).mla.swa:
            comp["kv_window"] += nb(b)
        else:
            comp["kv_full_fp32"] += nb(b)
    mt = eng.MT
    comp["mtp_k"] = nb(mt.kbuf)
    for ix in list(eng.M.ixf.values()) + [mt.ix]:
        comp["ix_pool"] += nb(ix.pool)
        comp["ix_vgb"] += nb(getattr(ix, "vgb", None))
    out = {"ctx": ctx, "cap": st.cap, "alloc_B": alloc, "alloc_gib": round(alloc / 2**30, 3), "comp_B": comp, "fires_kv": fires, "mtp_fp8": int(isinstance(mt.kbuf, kv_fp8.Fp8KV)),
           "env": {k: v for k, v in E.items() if k.startswith("HY4_KV") or k in ("HY4_SWA_RING", "HY4_LC_HB")}}
    pools = [ix.pool for ix in list(eng.M.ixf.values()) + [mt.ix]]
    out["pool16"] = sum(p.dtype == torch.float16 for p in pools)
    out["pools"] = len(pools)
    print("KVBYTES", json.dumps(out), flush=True)
    print(f"FIRES kv {fires} mtp {out['mtp_fp8']}", flush=True)
    bad = []
    if E.get("K3_TOUCH", "1") == "1":                 # one short prefill + a few verify steps: what the lazy rope tables / rows do once real requests run (the load-only numbers above are taken BEFORE this)
        bad += touch(eng, alloc, torch.cuda.synchronize, torch.cuda.memory_allocated, ae, IXF, RR)
    n16 = out["pool16"]
    evals = RR.STATS["attn"] + RR.STATS["ix"]
    print(f"FIRES kv10_pool16 {n16} of {out['pools']}", flush=True)
    print(f"FIRES kv10_rope {evals}", flush=True)
    flagged = ([n16] if IXF.POOL16 else []) + ([evals] if RR.ENABLED and E.get("K3_TOUCH", "1") == "1" else [])
    if flagged:
        print(f"FIRES kv10 {min(flagged)}", flush=True)           # fewest firings of any flagged lever (0 = a lever did not fire: the arm FAILS)
    if IXF.POOL16 and n16 != out["pools"]:
        bad.append(f"HY4_IX_POOL16 on but {n16} of {out['pools']} pools are fp16")
    if flagged and min(flagged) == 0:
        bad.append("FIRES kv10 0: a flagged kv10 lever did not fire")
    if bad:
        print("FAIL " + "; ".join(bad), flush=True)
        sys.exit(3)


def parent():
    """K3_ARMS = 'name:ctx,ctx,ctx:K=V+K=V;name2:...' (default: the 1.9.1 default at 16K / 32K / 64K, compact = HY4_KV_COMPACT=1 with the kv10 levers off (bcfb9e5 bytes) and kv10 = HY4_KV_COMPACT=1 as shipped now, both at 64K / 128K / 256K; the fp32 cache does not fit 256K)"""
    arms = E.get("K3_ARMS", "default:16384,32768,65536:;compact:65536,131072,262144:HY4_KV_COMPACT=1+HY4_IX_POOL16=0+HY4_ROPE_POS=0;kv10:65536,131072,262144:HY4_KV_COMPACT=1")
    allres, failed = {}, []
    for arm in arms.split(";"):
        name, ctxs, kv = arm.split(":", 2)
        env = {**E, **dict(x.split("=", 1) for x in kv.replace("+", " ").split())}          # flags of an arm: K=V joined by "+" (a job XENV cannot hold spaces)
        res = []
        for c in ctxs.split(","):
            t0 = time.time()
            r = subprocess.run([sys.executable, "-u", os.path.abspath(__file__)], env={**env, "K3_CHILD": c}, capture_output=True, text=True, timeout=1200)
            sys.stdout.write("".join(f"[{name}] {ln}\n" for ln in r.stdout.splitlines() if ln.startswith(("KVBYTES", "FIRES"))))
            sys.stdout.write("".join(f"[{name}] {ln}\n" for ln in r.stdout.splitlines() if ln.startswith(("KVTOUCH", "FAIL"))))
            kvl = [ln for ln in r.stdout.splitlines() if ln.startswith("KVBYTES")]
            if r.returncode:
                print(f"[{name}] ctx {c} rc={r.returncode} {r.stderr[-600:]}", flush=True)
                failed.append((name, c, r.returncode))
            if not kvl:
                continue
            d = json.loads(kvl[-1][8:])
            tl = [ln for ln in r.stdout.splitlines() if ln.startswith("KVTOUCH")]
            if tl:
                d["touch"] = json.loads(tl[-1][8:])
            res.append(d)                          # a failed touch / FIRES check keeps the load-only measurement
            print(f"[{name}] ctx {c} done {time.time() - t0:.0f}s", flush=True)
        for a, b in zip(res, res[1:]):
            dc = b["cap"] - a["cap"]
            print(f"SLOPE [{name}] ctx {a['ctx']}->{b['ctx']}: total {(b['alloc_B'] - a['alloc_B']) / dc:.0f} B/token; " + ", ".join(f"{k} {(b['comp_B'][k] - a['comp_B'][k]) / dc:.0f}" for k in a["comp_B"])
                  + f", unlisted {((b['alloc_B'] - sum(b['comp_B'].values())) - (a['alloc_B'] - sum(a['comp_B'].values()))) / dc:.0f}", flush=True)
        for a, b in zip(res, res[1:]):
            if "touch" in a and "touch" in b:
                print(f"SLOPE_TOUCH [{name}] ctx {a['ctx']}->{b['ctx']}: {(b['touch']['alloc_after_B'] - a['touch']['alloc_after_B']) / (b['cap'] - a['cap']):.0f} B/token after the 2048-token touch (lazy rope tables included)", flush=True)
        allres[name] = res
    json.dump(allres, open(E.get("K3_OUT", "/tmp/kvbytes.json"), "w"))
    if failed:
        print("DRIVER_FAIL", failed, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    child(int(E["K3_CHILD"])) if "K3_CHILD" in E else parent()
