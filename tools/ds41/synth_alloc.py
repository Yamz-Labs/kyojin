"""Synthetic allocation tables with the schema the conversion writes (alloc/<unit>.json), for tests and for what-if
studies of the optimizer before real routing statistics exist. NOTHING here is measured on DS4.1: routing skew, error
spread and energies are parameters. Only the K-scaling of the proxy error (x4.07 per bit) and the K=2 level (0.0489)
come from the real quantizer on synthetic Hessians (NOTEBOOK.md), and the byte sizes are exact (plan.exl3_bytes)."""

import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plan as P  # noqa: E402
import split as SP  # noqa: E402

IN, MID = 5120, 2304
ATTN = {"attn.wq_a": (5120, 1280), "attn.wkv": (5120, 512), "attn.wq_b": (1280, 32768), "attn.wo_b": (8192, 5120)}


def synth_tables(out_dir, n_layers=40, n_exp=384, zipf=1.0, seed=0, layer_bits="0-7:3,8-27:2,28-39:3",
                 alt_ks="2,3", up=0.25, down=0.25, attn_primary=4, attn_alt=(3, 5), eps2=0.0489, sigma=0.25,
                 n_tokens=262144, topk=6):
    """Write alloc/layers.L.json for every layer. Candidate availability follows split.choose_alts (up/down fractions),
    so `up=down=1.0` is the full dual-K measurement and `up=down=0` the primary-only pack."""
    rnd = random.Random(seed)
    lb = P.parse_layer_bits(layer_bits)
    aks = [int(x) for x in alt_ks.split(",")] if alt_ks else []
    os.makedirs(os.path.join(out_dir, "alloc"), exist_ok=True)
    ln = lambda s: math.exp(rnd.gauss(0.0, s))
    for L in range(n_layers):
        unit = f"layers.{L}"
        # routing counts: Zipf over a per-layer permutation, layer-dependent steepness
        order = list(range(n_exp))
        rnd.shuffle(order)
        z = zipf * (0.8 + 0.4 * rnd.random())
        w = [0.0] * n_exp
        for r, e in enumerate(order):
            w[e] = 1.0 / (r + 2.0) ** z
        tot = sum(w)
        counts = [int(round(x / tot * n_tokens * topk)) for x in w]
        g2 = [(1.0 / topk) ** 2 * ln(0.15) for _ in range(n_exp)]
        w2sum = [c * g for c, g in zip(counts, g2)]
        prim = {e: lb[L] for e in range(n_exp)}
        alts = SP.choose_alts(counts, prim, aks, up, down) if aks else {}
        experts = {}
        Ew = [ln(0.4) for _ in range(n_exp)]
        sh_E = 3.0 * ln(0.1)
        tab_y = 0.0
        for e in range(n_exp):
            tab_y += w2sum[e] * Ew[e] / n_tokens
        Y = (tab_y + sh_E) * 1.1
        for e in range(n_exp):
            k_ = {}
            base = [eps2 * ln(sigma) for _ in range(3)]       # K=2 relative errors of w1, w3, w2
            jit = {k: ln(0.08) for k in (2, 3, 4)}              # drawn for every K: candidate availability never shifts the stream
            for k in [lb[L]] + alts.get(e, []):
                f = 4.07 ** (2 - k) * jit[k]
                k_[str(k)] = {"proxy": {w_: round(b * f, 7) for w_, b in zip(("w1", "w3", "w2"), base)},
                              "bytes": sum(P.exl3_bytes(i, o, k) for i, o in ((IN, MID), (IN, MID), (MID, IN))),
                              "where": "shard" if k == lb[L] else "cand"}
            experts[str(e)] = {"count": counts[e], "w2sum": w2sum[e], "E": {"w1": 1.0, "w3": 1.0, "w2": Ew[e]},
                               "k": k_}
        lin = {}
        for name, (i, o) in ATTN.items():
            base = eps2 * ln(sigma)
            jj = {k: ln(0.08) for k in range(2, 7)}
            ks = [attn_primary] + [k for k in attn_alt if k != attn_primary]
            lin[f"{unit}.{name}"] = {"kind": "attn", "in": i, "out": o, "E": 1.0, "k": {
                str(k): {"proxy": base * 4.07 ** (2 - k) * jj[k], "bytes": P.exl3_bytes(i, o, k),
                         "where": "shard" if k == attn_primary else "cand"} for k in ks}}
        for s_ in range(8):
            base = eps2 * ln(sigma)
            jj = {k: ln(0.08) for k in range(2, 7)}
            ks = [attn_primary] + [k for k in attn_alt if k != attn_primary]
            lin[f"{unit}.attn.wo_a.slice.{s_}"] = {"kind": "attn", "in": 4096, "out": 1024, "E": 1.0, "k": {
                str(k): {"proxy": base * 4.07 ** (2 - k) * jj[k], "bytes": P.exl3_bytes(4096, 1024, k),
                         "where": "shard" if k == attn_primary else "cand"} for k in ks}}
        for w_, (i, o) in (("w1", (IN, MID)), ("w3", (IN, MID)), ("w2", (MID, IN))):
            base = eps2 * ln(sigma)
            jj = {k: ln(0.08) for k in range(2, 7)}
            ks = [attn_primary] + [k for k in attn_alt if k != attn_primary]
            lin[f"{unit}.ffn.shared_experts.{w_}"] = {"kind": "shared", "in": i, "out": o, "E": sh_E, "k": {
                str(k): {"proxy": base * 4.07 ** (2 - k) * jj[k], "bytes": P.exl3_bytes(i, o, k),
                         "where": "shard" if k == attn_primary else "cand"} for k in ks}}
        json.dump({"unit": unit, "stats": {"n_tokens": n_tokens, "topk": topk, "moe_out_energy": Y},
                   "experts": experts, "linears": lin}, open(os.path.join(out_dir, "alloc", f"{unit}.json"), "w"))
