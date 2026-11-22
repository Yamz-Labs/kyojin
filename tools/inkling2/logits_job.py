"""Inkling2 logits match: Inkling2Forward on one GGUF vs the Mac CPU reference (HF transformers PR 51050 loop model, fp32, base weights).

  logits_job.py --gguf G --ref REFDIR [--device cuda|cpu] [--out DIR] [--cos 0.99]

REFDIR (made by the Mac agent): tokens.npy [T], hiddens.npy [48, T, 2304] (output of every application), final_hidden.npy [T, 2304],
logits.npy [T, 200058], topk.npy [48, T, 8] (router picks), meta.json. One prefill of the T tokens, then per application: cosine and
max abs / max |ref| of l_out; expert-set overlap of the router picks; per position: cosine of the logits, top-1 agreement, top-5 overlap.
The GGUF is a quantised copy of the base weights (Q8_0 attention / NVFP4 experts / BF16 head...), so the bar is statistical, not bitwise:
cosine >= --cos on every application and on the logits, top-1 equal on >= T - 1 positions. Output DIR/logits_<tag>.json, `FIRES logits <rows>`;
exit 7 when a bar fails (or no row), 0 otherwise."""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cpu_env  # noqa: E402

TREE = os.path.abspath(os.path.join(HERE, "..", ".."))


def cos(a, b):
    a, b = a.reshape(-1).astype(np.float64), b.reshape(-1).astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="ink_logits_out")
    ap.add_argument("--cos", type=float, default=0.99)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--tag", default="")
    ap.add_argument("--save", "--dump", dest="save", default="", help="write every application's l_out + the logits of THIS run to this .npz")
    ap.add_argument("--vs", "--ref2", dest="vs", default="", help="npz of a torch-path run (INK_BACKEND=torch, same GGUF / tokens): the kernel path is judged against it (--vs-cos), the reference rows against --cos")
    ap.add_argument("--vs-cos", "--cos2", dest="vs_cos", type=float, default=0.999)
    ap.add_argument("--chunk", type=int, default=256, help="prefill chunk (the kernel paths run <= 8 rows per chunk; use 2 to exercise T = 2 / 1 and the cache continuity)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    F = cpu_env.install(TREE)
    from exllamav3.loader.gguf_collection import GGUFCollection
    toks = np.load(os.path.join(a.ref, "tokens.npy")).reshape(-1).tolist()
    hid = np.load(os.path.join(a.ref, "hiddens.npy"))
    ref_logits = np.load(os.path.join(a.ref, "logits.npy"))
    topk = np.load(os.path.join(a.ref, "topk.npy"))
    rec = {}

    def hook(name, il, t):
        if name in ("l_out", "router_sel", "result_norm"):
            rec.setdefault((name, il), []).append(t.detach().cpu().numpy())

    m = F.Inkling2Forward(GGUFCollection(a.gguf), device=a.device, hook=hook, chunk=a.chunk)
    assert m.n_app == hid.shape[0] and len(toks) == hid.shape[1], (m.n_app, hid.shape, len(toks))
    t0 = time.time()
    lg = m.step(toks, all_pos=True).float().cpu().numpy()
    dt = time.time() - t0
    rec = {k: np.concatenate(v, 0) for k, v in rec.items()}       # one array per (hook, application) even when the prefill ran in several chunks
    hids = np.stack([rec[("l_out", il)] for il in range(m.n_app)])
    if a.save:
        os.makedirs(os.path.dirname(os.path.abspath(a.save)), exist_ok=True)
        np.savez(a.save, hiddens=hids, logits=lg)
    r2 = np.load(a.vs) if a.vs and os.path.exists(a.vs) else None
    if a.vs and r2 is None:
        print(f"WARNING: --vs {a.vs} missing: only the reference rows (--cos) are compared", flush=True)
    if r2 is not None:
        r2 = {"hiddens": r2["hiddens"] if "hiddens" in r2 else np.stack([r2[f"l{il}"] for il in range(m.n_app)]), "logits": r2["logits"]}
    rows, bad = [], 0
    for il in range(m.n_app):
        got, ref = rec[("l_out", il)], hid[il]
        c, rel = cos(got, ref), float(np.abs(got - ref).max() / (np.abs(ref).max() + 1e-12))
        sel, rs = rec.get(("router_sel", il)), topk[il]
        ov = float(np.mean([len(set(sel[t].tolist()) & set(rs[t].tolist())) / rs.shape[1] for t in range(len(toks))])) if sel is not None else None
        ok = c >= a.cos
        c2 = cos(got, r2["hiddens"][il]) if r2 is not None else None
        ok = ok and (c2 is None or c2 >= a.vs_cos)
        bad += not ok
        rows.append(dict(app=il, cos=round(c, 6), cos_vs_torch=None if c2 is None else round(c2, 6), maxrel=round(rel, 4), expert_overlap=None if ov is None else round(ov, 3), ok=ok))
        print(f"app {il:2d}: cos {c:.6f} vs_torch {c2 if c2 is None else round(c2, 6)} maxrel {rel:.3e} experts {ov if ov is None else round(ov, 3)}{'' if ok else '  <-- FAIL'}", flush=True)
    vbad = 0
    if r2 is not None:
        for il in range(m.n_app):
            cv = cos(rec[("l_out", il)], r2["hiddens"][il])
            rows[il]["cos_vs_torch"] = round(cv, 6)
            vbad += cv < a.vs_cos
        t1 = sum(int(np.argmax(lg[t])) == int(np.argmax(r2["logits"][t])) for t in range(len(toks)))
        vbad += t1 != len(toks)
        print(f"top-1 equal to the torch path: {t1}/{len(toks)}  logits cos {[round(cos(lg[t], r2['logits'][t]), 6) for t in range(len(toks))]}", flush=True)
    top1 = [int(np.argmax(lg[t])) == int(np.argmax(ref_logits[t])) for t in range(len(toks))]
    lrows = []
    for t in range(len(toks)):
        o5 = len(set(np.argsort(-lg[t])[:5].tolist()) & set(np.argsort(-ref_logits[t])[:5].tolist()))
        lc2 = cos(lg[t], r2["logits"][t]) if r2 is not None else None
        t2 = None if r2 is None else int(np.argmax(lg[t])) == int(np.argmax(r2["logits"][t]))
        lrows.append(dict(pos=t, cos_vs_torch=None if lc2 is None else round(lc2, 6), top1_vs_torch=t2, cos=round(cos(lg[t], ref_logits[t]), 6), top1=top1[t], top5_overlap=o5))
        print(f"logits pos {t}: cos {lrows[-1]['cos']:.6f} top1 {top1[t]} top5 {o5}/5", flush=True)
    lbad = sum(r["cos"] < a.cos for r in lrows) + (len(toks) - sum(top1) > 1)
    if r2 is not None:
        lbad += sum(r["cos_vs_torch"] < a.vs_cos or not r["top1_vs_torch"] for r in lrows)
    res = dict(kstats=m.kstats, glue=(sorted(m.gl.on) if m.gl else None), glue_fires=(dict(m.gl.fires) if m.gl else None), chunk=m.chunk, gguf=os.path.basename(a.gguf), device=a.device, seconds=round(dt, 1), apps=rows, logits=lrows, bad_apps=bad, bad_logits=int(lbad))
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, f"logits_{a.tag or os.path.basename(a.gguf)[:-5]}.json"), "w"), indent=1)
    df = m.dev_fires()
    print("KSTATS", json.dumps(m.kstats), flush=True)
    import ink2_glue
    if m.gl is not None:
        gbad = ink2_glue.fires_check(m)
        print("GLUE", sorted(m.gl.on), json.dumps(dict(m.gl.fires)), "fires-check:", gbad or "ok", flush=True)
        kern_ok = not gbad
    else:
        kern_ok = m.backend != "cuda" or m.kstats["nvfp4_gap"] > 0 or m.kstats.get("xk_nvfp4", 0) > 0 or not any(t == 40 for t, _d, _o in m.g.infos.values())
        if "conv" in F.FUSE and m.backend == "cuda":
            import ink2_glue_conv as _G
            kern_ok = kern_ok and m.kstats["conv_fused"] > 0 and m.kstats["conv_torch"] == 0 and _G.FIRES["conv"] > 0
        if "router" in F.RTX and m.backend == "cuda":
            kern_ok = kern_ok and m.kstats["router_k"] > 0 and m.kstats["torch_router"] == 0 and df["router_blocks"] > 0
        if m.backend == "cuda" and {"mix", "control", "norm"} & F.FUSE:      # the Triton glue fusions fired (python counters advance while capturing / in eager)
            import ink2_glue_misc as _M
            for _f, _k in (("mix", "mix"), ("control", "control"), ("norm", "rmsw")):
                if _f in F.FUSE:
                    kern_ok = kern_ok and _M.FIRES[_k] > 0
            print("MISC_FIRES", _M.FIRES, flush=True)
        if "dense" in F.RTX and m.backend == "cuda":
            kern_ok = kern_ok and m.kstats["dense_q8"] > 0
        if "attn" in F.RTX and m.backend == "cuda":
            kern_ok = kern_ok and m.kstats["ak_gkv"] > 0 and m.kstats["ak_lattn"] > 0 and m.kstats["torch_attn"] == 0
        if "experts" in F.RTX and m.backend == "cuda":      # our kernels wired: they fired, the torch MoE path did not (T <= 8 positions)
            kern_ok = kern_ok and m.kstats["xk_nvfp4"] > 0 and m.kstats["torch_moe"] == 0 and min(df["xk_gate_up_entries_down_blocks"]) > 0
    if not kern_ok:
        print("FIRES-CHECK FAILED: no NVFP4 kernel call on the GPU")
    print(f"FIRES logits {len(rows) + len(lrows)}")
    print(f"VERDICT apps below cos {a.cos}: {bad}, logits bars failed: {int(lbad)}, parity vs torch path failed: {int(vbad)} -> {'PASS' if bad == 0 and lbad == 0 and vbad == 0 else 'FAIL'}")
    sys.exit(0 if bad == 0 and lbad == 0 and vbad == 0 and kern_ok else 7)


if __name__ == "__main__":
    main()
