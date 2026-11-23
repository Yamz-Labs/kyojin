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
            rec[(name, il)] = t.detach().cpu().numpy()

    m = F.Inkling2Forward(GGUFCollection(a.gguf), device=a.device, hook=hook)
    assert m.n_app == hid.shape[0] and len(toks) == hid.shape[1], (m.n_app, hid.shape, len(toks))
    t0 = time.time()
    lg = m.step(toks, all_pos=True).float().cpu().numpy()
    dt = time.time() - t0
    rows, bad = [], 0
    for il in range(m.n_app):
        got, ref = rec[("l_out", il)], hid[il]
        c, rel = cos(got, ref), float(np.abs(got - ref).max() / (np.abs(ref).max() + 1e-12))
        sel, rs = rec.get(("router_sel", il)), topk[il]
        ov = float(np.mean([len(set(sel[t].tolist()) & set(rs[t].tolist())) / rs.shape[1] for t in range(len(toks))])) if sel is not None else None
        ok = c >= a.cos
        bad += not ok
        rows.append(dict(app=il, cos=round(c, 6), maxrel=round(rel, 4), expert_overlap=None if ov is None else round(ov, 3), ok=ok))
        print(f"app {il:2d}: cos {c:.6f} maxrel {rel:.3e} experts {ov if ov is None else round(ov, 3)}{'' if ok else '  <-- FAIL'}", flush=True)
    top1 = [int(np.argmax(lg[t])) == int(np.argmax(ref_logits[t])) for t in range(len(toks))]
    lrows = []
    for t in range(len(toks)):
        o5 = len(set(np.argsort(-lg[t])[:5].tolist()) & set(np.argsort(-ref_logits[t])[:5].tolist()))
        lrows.append(dict(pos=t, cos=round(cos(lg[t], ref_logits[t]), 6), top1=top1[t], top5_overlap=o5))
        print(f"logits pos {t}: cos {lrows[-1]['cos']:.6f} top1 {top1[t]} top5 {o5}/5", flush=True)
    lbad = sum(r["cos"] < a.cos for r in lrows) + (len(toks) - sum(top1) > 1)
    res = dict(kstats=m.kstats, gguf=os.path.basename(a.gguf), device=a.device, seconds=round(dt, 1), apps=rows, logits=lrows, bad_apps=bad, bad_logits=int(lbad))
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, f"logits_{a.tag or os.path.basename(a.gguf)[:-5]}.json"), "w"), indent=1)
    print("KSTATS", json.dumps(m.kstats), flush=True)
    kern_ok = a.device != "cuda" or m.kstats["nvfp4_gap"] > 0 or not any(t == 40 for t, _d, _o in m.g.infos.values())
    if not kern_ok:
        print("FIRES-CHECK FAILED: no NVFP4 kernel call on the GPU")
    print(f"FIRES logits {len(rows) + len(lrows)}")
    print(f"VERDICT apps below cos {a.cos}: {bad}, logits bars failed: {int(lbad)} -> {'PASS' if bad == 0 and lbad == 0 else 'FAIL'}")
    sys.exit(0 if bad == 0 and lbad == 0 and kern_ok else 7)


if __name__ == "__main__":
    main()
