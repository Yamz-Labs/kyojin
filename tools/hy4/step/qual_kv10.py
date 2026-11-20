"""qual_kv10.py (qual_llc.py + the kv10 gates): paired quality of HY4_IX_POOL16 / HY4_ROPE_POS against the compact arm, run inside kv10_driver.py (cwd = tree).
Mode A (ref = HY4_KV_COMPACT=1 with HY4_IX_POOL16=0 HY4_ROPE_POS=0, fp32 pool): also records, on the REAL decode queries of the eager warm-up bodies of every graph capture, the
overlap of the top-k group SET chosen from the fp32 pool with the set chosen from the same pool rounded to fp16 (the pool an fp16-pool arm stores): `IXOVL rows n mean_overlap x min_overlap y exact_rows z`.
Mode B (cand = HY4_KV_COMPACT=1 as shipped: fp16 pool + rope rows): prints FIRES kv10_pool16 / kv10_rope and exits 1 when a flagged lever did not fire.
qual_llc.py: short-context quality gate of the lane long-context config (run inside the job driver, cwd = tree).
Teacher-forced NLL of the decode path (graph step, R = 4 verify rows per call) on a FIXED text (bench corpus, first 400k chars, tokenised by the model vocab), two windows:
  w4k   2048 rows of context (prefill), then 2048 scored rows        -> a 4096-row window
  w32k  31744 rows of context (prefill), then 1024 scored rows       -> depth 31744 .. 32768
Allocation flags (HY4_SWA_RING / HY4_KV_FP8 / HY4_IX_POOL16 ...) cannot be toggled inside one load, so the two arms are two processes:
  LLC_MODE=A  ref arm + 3 noise nulls (1e-6 embedding noise, seeds 1234 / 77 / 5) on the SAME config; saves the ref log-probs (fp16) + null stats to LLC_REF
  LLC_MODE=B  one arm (the env of this process) scored against the saved ref: ppl, paired dNLL, KL(ref||arm), top-1 agreement, KL / noise-null max; deletes LLC_REF
env: LLC_CTX (server -c of this arm), LLC_MODE, LLC_REF (file). Prints NLL / KL numbers only, never text."""
import json, os, sys, time
import numpy as np
tree = os.getcwd()
sys.path.insert(0, os.path.join(tree, "tools", "hy4"))
import serve
import serve_core as C
CTX = int(os.environ["LLC_CTX"]); MODE = os.environ["LLC_MODE"]; REFF = os.environ["LLC_REF"]
assert MODE in ("A", "B")
serve.apply_env(CTX)
import torch
from engine import Hy4Engine
GGUF = os.environ["GGUF"]
e = Hy4Engine(tree, GGUF, ctx=CTX, k=3, chunk=8192, prime_rows=64)       # the draft cache is not read here: fill the last 64 rows only
e.load(print)
from exllamav3.loader import lc_sparse as LS, rope_rows as RR, indexer_fused as IXF, attn_eager as AE
OVL = {"rows": 0, "sum": 0.0, "min": 1.0, "exact": 0}
if MODE == "A" and os.environ.get("HY4_IX_POOL16", "0") != "1":
    _sel = LS.select_groups

    def _sel_ovl(iq, wt, pool, pos, Pb, n_top, kp, tail):
        idx, nsel = _sel(iq, wt, pool, pos, Pb, n_top, kp, tail)
        if pool.dtype == torch.float32 and not torch.cuda.is_current_stream_capturing():
            idx16, nsel16 = _sel(iq, wt, pool.half().contiguous(), pos, Pb, n_top, kp, tail)
            a, b, na, nb = idx.cpu(), idx16.cpu(), nsel.cpu(), nsel16.cpu()
            for t in range(a.shape[0]):
                sa, sb = set(a[t, :int(na[t])].tolist()), set(b[t, :int(nb[t])].tolist())
                ov = len(sa & sb) / max(len(sa), 1)
                OVL["rows"] += 1; OVL["sum"] += ov; OVL["min"] = min(OVL["min"], ov); OVL["exact"] += sa == sb
        return idx, nsel
    LS.select_groups = _sel_ovl          # step_graph calls lc_sparse.select_groups through the module: the hook sees every eager (non-captured) selection
SG, GR, dev = e.SG, e.GR, e.dev
R = 4
text = open(os.environ["LLC_CORPUS"]).read(400000)
ids = C.Vocab(e.meta).encode(text)
WINS = [("w4k", ids[0:2048 + 2048 + 1], 2048, 2048), ("w32k", ids[4096:4096 + 31744 + 1024 + 1], 31744, 1024)]
assert all(len(w[1]) == w[2] + w[3] + 1 for w in WINS), "corpus too short"
print("LLC_TEXT tokens", len(ids), "mode", MODE, "ctx", CTX, flush=True)


def run_arm(seed=None, ref=None):
    """-> {window: (nll [n], kl list, same list, lps fp16 [n, V] or None, prefill_s)}"""
    SG.PERT = None if seed is None else (torch.randn(GR.cap, e.M.D, generator=torch.Generator().manual_seed(seed)) * 1e-6).to(dev)
    out = {}
    GR.clear()          # graphs bake the PERT table: fresh pool per arm (once per arm, not per window)
    for name, wid, nctx, nsc in WINS:
        e._t0 = time.perf_counter(); e.reset()
        t = time.perf_counter(); e.prefill(wid[:nctx]); torch.cuda.synchronize(); pf = time.perf_counter() - t
        e.phase("decode")
        nl, kl, same, lps = [], [], [], []
        for i in range(0, nsc, R):
            lg = GR.step_rows(wid[nctx + i:nctx + i + R])[0]
            lp = torch.log_softmax(lg.reshape(R, -1).float(), -1)
            tgt = torch.tensor(wid[nctx + i + 1:nctx + i + R + 1], device=dev)
            nl.extend((-lp[torch.arange(R, device=dev), tgt]).tolist())
            if ref is None:
                lps.append(lp.half().cpu())
            else:
                lr = ref[name][len(nl) - R:len(nl)].to(dev).float()
                kl.extend((lr.exp() * (lr - lp)).sum(-1).tolist())
                same.extend((lr.argmax(-1) == lp.argmax(-1)).tolist())
        out[name] = (np.array(nl), kl, same, torch.cat(lps) if ref is None else None, pf)
    SG.PERT = None
    GR.clear()
    return out


def stats(arm, name, r, base_nl, nctx):
    nl, kl, same, _, pf = r
    r_ = dict(arm=arm, win=name, n=len(nl), ppl=round(float(np.exp(nl.mean())), 4), prefill_tps=round(nctx / pf, 1))
    if base_nl is not None:
        d = nl - base_nl
        r_.update(d_nll=round(float(d.mean()), 5), d_se=round(float(d.std(ddof=1) / np.sqrt(len(d))), 5), kl_mean=round(float(np.mean(kl)), 6),
                  kl_p99=round(float(np.percentile(kl, 99)), 5), top1_same=round(float(np.mean(same)), 4))
    return r_


NCTX = {w[0]: w[2] for w in WINS}
if MODE == "A":
    ref = run_arm()
    save = dict(lps={k: v[3] for k, v in ref.items()}, nll={k: v[0] for k, v in ref.items()}, nulls={})
    save["ref"] = {}
    for k, v in ref.items():
        save["ref"][k] = stats("ref", k, v, None, NCTX[k])
        print("QRES", json.dumps(save["ref"][k]), flush=True)
    refl = save["lps"]
    for nm, seed in (("noise1234", 1234), ("noise77", 77), ("noise5", 5)):
        res = run_arm(seed, refl)
        for k, v in res.items():
            s = stats(nm, k, v, ref[k][0], NCTX[k])
            save["nulls"].setdefault(k, {})[nm] = s
            print("QRES", json.dumps(s), flush=True)
    torch.save(save, REFF)
    for k in save["nulls"]:
        print("NULL_BAND", k, "kl max", max(s["kl_mean"] for s in save["nulls"][k].values()), "min", min(s["kl_mean"] for s in save["nulls"][k].values()), flush=True)
else:
    saved = torch.load(REFF)
    res = run_arm(None, saved["lps"])
    for k, v in res.items():
        s = stats("cand", k, v, saved["nll"][k], NCTX[k])
        print("QRES", json.dumps(s), flush=True)
        nz = max(x["kl_mean"] for x in saved["nulls"][k].values())
        nd = max(abs(x["d_nll"]) for x in saved["nulls"][k].values())
        print("QKL_VS_NOISE", k, "kl", s["kl_mean"], "x noise-null max", round(s["kl_mean"] / max(nz, 1e-12), 2), "| |dNLL|", abs(s["d_nll"]), "vs null max", nd,
              "| prefill tok/s ref", saved["ref"][k]["prefill_tps"], "cand", s["prefill_tps"], flush=True)
    if os.environ.get("LLC_KEEP") != "1":
        os.remove(REFF)
pools = [ix.pool for ix in list(e.M.ixf.values()) + [e.MT.ix]]
n16 = sum(p.dtype == torch.float16 for p in pools)
rows_attn = sum(t.shape[0] for t in AE._TAB.values())
rows_ix = sum(cs[0].shape[0] for cs in IXF.FusedKPoolIndexer._TABS.values())
if OVL["rows"]:
    print(f"IXOVL rows {OVL['rows']} mean_overlap {OVL['sum'] / OVL['rows']:.6f} min_overlap {OVL['min']:.4f} exact_rows {OVL['exact']}", flush=True)
    print(f"FIRES kv10_ovl {OVL['rows']}", flush=True)
print(f"FIRES kv10_pool16 {n16} of {len(pools)}", flush=True)
print(f"FIRES kv10_rope {RR.STATS['attn'] + RR.STATS['ix']} (rope table rows left: attn {rows_attn} ix {rows_ix})", flush=True)
bad = []
flagged = ([n16] if IXF.POOL16 else []) + ([RR.STATS["attn"] + RR.STATS["ix"]] if RR.ENABLED else [])
if flagged:
    print(f"FIRES kv10 {min(flagged)}", flush=True)           # fewest firings of any flagged lever; 0 fails the arm
    if min(flagged) == 0:
        bad.append("FIRES kv10 0: a flagged kv10 lever did not fire")
if MODE == "B" and IXF.POOL16 and n16 != len(pools):
    bad.append("HY4_IX_POOL16 on but the pools are not fp16")
if MODE == "B" and RR.ENABLED and (not (RR.STATS["attn"] + RR.STATS["ix"]) or rows_attn or rows_ix):
    bad.append("HY4_ROPE_POS on but no row evaluation fired or a full-context rope table exists")
if MODE == "A" and not OVL["rows"]:
    bad.append("no top-k overlap row recorded (the sparse selection never ran eagerly)")
print("QUAL_DONE", flush=True)
if bad:
    print("FAIL " + "; ".join(bad), flush=True)
    sys.exit(1)
