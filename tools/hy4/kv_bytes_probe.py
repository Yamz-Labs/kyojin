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
    from exllamav3.loader import step_nosync as ns, kv_fp8, attn_eager as ae
    st = ns.state(eng.M)
    comp = {"kv_full_fp8": 0, "kv_full_fp32": 0, "kv_window": 0, "mtp_k": 0, "ix_pool": 0, "ix_vgb": 0, "tok_ar": nb(st.tok) + nb(st.ar), "rope_attn": sum(nb(t) for t in ae._TAB.values())}
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
    print("KVBYTES", json.dumps(out), flush=True)
    print(f"FIRES kv {fires} mtp {out['mtp_fp8']}", flush=True)


def parent():
    """K3_ARMS = 'name:ctx,ctx,ctx:K=V K=V;name2:...' (default: the 1.9.1 default at 16K / 32K / 64K and HY4_KV_COMPACT=1 at 64K / 128K / 256K; the fp32 cache does not fit 256K)"""
    arms = E.get("K3_ARMS", "default:16384,32768,65536:;compact:65536,131072,262144:HY4_KV_COMPACT=1")
    allres = {}
    for arm in arms.split(";"):
        name, ctxs, kv = arm.split(":", 2)
        env = {**E, **dict(x.split("=", 1) for x in kv.split())}
        res = []
        for c in ctxs.split(","):
            t0 = time.time()
            r = subprocess.run([sys.executable, "-u", os.path.abspath(__file__)], env={**env, "K3_CHILD": c}, capture_output=True, text=True, timeout=1200)
            sys.stdout.write("".join(f"[{name}] {ln}\n" for ln in r.stdout.splitlines() if ln.startswith(("KVBYTES", "FIRES"))))
            if r.returncode:
                print(f"[{name}] ctx {c} rc={r.returncode} {r.stderr[-600:]}", flush=True)
                continue
            res.append(json.loads([ln for ln in r.stdout.splitlines() if ln.startswith("KVBYTES")][-1][8:]))
            print(f"[{name}] ctx {c} done {time.time() - t0:.0f}s", flush=True)
        for a, b in zip(res, res[1:]):
            dc = b["cap"] - a["cap"]
            print(f"SLOPE [{name}] ctx {a['ctx']}->{b['ctx']}: total {(b['alloc_B'] - a['alloc_B']) / dc:.0f} B/token; " + ", ".join(f"{k} {(b['comp_B'][k] - a['comp_B'][k]) / dc:.0f}" for k in a["comp_B"])
                  + f", unlisted {((b['alloc_B'] - sum(b['comp_B'].values())) - (a['alloc_B'] - sum(a['comp_B'].values()))) / dc:.0f}", flush=True)
        allres[name] = res
    json.dump(allres, open(E.get("K3_OUT", "/tmp/kvbytes.json"), "w"))


if __name__ == "__main__":
    child(int(E["K3_CHILD"])) if "K3_CHILD" in E else parent()
