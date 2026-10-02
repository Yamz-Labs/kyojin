#!/usr/bin/env python
"""glm-mtp step 8 unit test: exl3_dec_gemv_r(_multi) row r vs the batch-1 launch on x[r], bitwise,
on real GLM-5.3 td205 MLA weights, for both codebooks.

Groups mirror MLAttention's decode routing: q_a_proj + kv_a_proj_with_mqa share one launch
(_dec_proj_pair -> exl3_dec_gemv_multi / gemv_r_multi), q_b_proj, o_proj and indexer.wq_b run alone.
td205 is all mul1; mcg=True re-reads the same trellis bits with the mcg codebook (no mcg GLM pack on
disk), which exercises the new CB = 1 gemv_r path with real bit statistics. Bits are compared as int16
views (NaN-safe). Control: kv_a alone through gemv_r vs the pair's kv_a row (pick_ktw sees the pair's
strip count, so grouping alone changes the reduction). Prints one RESULT line per case and DONE.
"""
import os, sys, json, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from safetensors import safe_open
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.quant.exl3 import dec_workspace

MODEL = sys.argv[1] if len(sys.argv) > 1 else "~/models/glm53-exl3-td205"
LAYERS = (3, 43, 45)
dev = torch.device("cuda:0")
assert "mcg" in (ext.exl3_dec_gemv_r.__doc__ or ""), "stale .so: exl3_dec_gemv_r has no mcg"
wm = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]


def load(key):
    out = {}
    for t in ("trellis", "suh", "svh"):
        k = f"{key}.{t}"
        with safe_open(os.path.join(MODEL, wm[k]), "pt", device = "cuda:0") as f:
            out[t] = f.get_tensor(k)
    out["K"] = out["trellis"].shape[2] / 16.0
    return out


def run_b1(ws, x, mcg):
    scratch, counters = dec_workspace(dev)
    outs = [torch.empty((1, w["trellis"].shape[1] * 16), dtype = torch.half, device = dev) for w in ws]
    ext.exl3_dec_gemv_multi(x, [w["trellis"] for w in ws], [w["suh"] for w in ws], [w["svh"] for w in ws],
                            outs, scratch, counters, [w["K"] for w in ws], mcg)
    return outs


def run_r(ws, x, mcg):
    scratch, counters = dec_workspace(dev)
    R = x.shape[0]
    outs = [torch.empty((R, w["trellis"].shape[1] * 16), dtype = torch.half, device = dev) for w in ws]
    if len(ws) == 1:
        w = ws[0]
        ext.exl3_dec_gemv_r(x, w["trellis"], w["suh"], w["svh"], outs[0], scratch, counters, w["K"], mcg)
    else:
        ext.exl3_dec_gemv_r_multi(x, [w["trellis"] for w in ws], [w["suh"] for w in ws], [w["svh"] for w in ws],
                                  outs, scratch, counters, [w["K"] for w in ws], mcg)
    return outs


def bits(t):
    return t.contiguous().view(torch.int16)


torch.manual_seed(0)
fails = 0
for L in LAYERS:
    pre = f"model.language_model.layers.{L}.self_attn."
    groups = {"q_a+kv_a": ["q_a_proj", "kv_a_proj_with_mqa"], "q_b": ["q_b_proj"], "o_proj": ["o_proj"],
              "idx_wq_b": ["indexer.wq_b"]}
    for gname, names in groups.items():
        if not all(pre + n + ".trellis" in wm for n in names):
            print("RESULT", json.dumps({"layer": L, "group": gname, "skip": "absent"}), flush = True)
            continue
        ws = [load(pre + n) for n in names]
        Kin = ws[0]["trellis"].shape[0] * 16
        for mcg in (False, True):
            for R in (2, 3, 4):
                x = (torch.randn(R, Kin, device = dev) * 0.5).half()
                yr = run_r(ws, x, mcg)
                y1 = [run_b1(ws, x[r : r + 1].contiguous(), mcg) for r in range(R)]
                neq = [int(sum(int((bits(yr[i][r]) != bits(y1[r][i][0])).sum()) for r in range(R))) for i in range(len(ws))]
                ok = all(n == 0 for n in neq)
                fails += not ok
                rec = {"layer": L, "group": gname, "mcg": mcg, "R": R, "K": ws[0]["K"],
                       "N": [w["trellis"].shape[1] * 16 for w in ws], "neq": neq, "bit_equal": ok,
                       "finite": all(bool(torch.isfinite(y.float()).all()) for y in yr)}
                if len(ws) == 2:
                    # control: kv_a alone (other strip count -> other ktw) vs the pair's kv_a row
                    ya = run_r(ws[1:], x, mcg)[0]
                    rec["ctrl_kv_a_alone_neq"] = int((bits(ya) != bits(yr[1])).sum())
                print("RESULT", json.dumps(rec), flush = True)
print("DONE fails", fails, flush = True)
