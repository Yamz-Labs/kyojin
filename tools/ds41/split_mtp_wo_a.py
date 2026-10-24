#!/usr/bin/env python3
"""Pack fixups after partial_pack (idempotent; rerun after every assemble):
1. Split the calibrated MTP wo_a (one EXL3 linear [in=gw, out=G*r]) into the G per-group slices the serving model loads
   (attn.wo_a.slice.g, out = r each). Exact: trellis tiles are [in/16, out/16, words], so slice g = out tiles
   [g*r/16, (g+1)*r/16); svh is per output (sliced), suh per input (shared), mul1 is a per-tensor scalar (copied).
2. Copy the raw Engram projections (q_weight, k_weight, wkv.weight/scale) from the official source shards into
   the layer's own shard when the conversion ran without the Engram source shards.
3. Rewrite the index from the shard headers of these files."""
import json, os, sys
from safetensors import safe_open
from safetensors.torch import save_file

pack = sys.argv[1]; src = sys.argv[2] if len(sys.argv) > 2 else None; G = 8
idx_p = os.path.join(pack, "model.safetensors.index.json"); idx = json.load(open(idx_p)); wm = idx["weight_map"]

def reindex(fn):
    with safe_open(os.path.join(pack, fn), "pt") as f: keys = list(f.keys())
    stem = fn[:-len(".safetensors")]
    for k in [k for k, v in wm.items() if v == fn]: del wm[k]
    for k in keys: wm[k] = fn
    return keys

for j in range(3):
    unit = f"mtp.{j}"; fn = f"{unit}.safetensors"; p = os.path.join(pack, fn); base = f"{unit}.attn.wo_a"
    if not os.path.exists(p): continue
    with safe_open(p, "pt") as f:
        meta = f.metadata(); sliced = f"{base}.slice.0.trellis" in f.keys(); t = None if sliced else {k: f.get_tensor(k) for k in f.keys()}
    if not sliced:
        tr, svh, suh, mul1 = (t.pop(f"{base}.{n}") for n in ("trellis", "svh", "suh", "mul1"))
        out = svh.shape[0]; r = out // G
        assert out % G == 0 and r % 16 == 0 and tr.shape[1] * 16 == out, (tr.shape, svh.shape)
        for g in range(G):
            s = f"{base}.slice.{g}"
            t[f"{s}.trellis"] = tr[:, g * r // 16:(g + 1) * r // 16].contiguous()
            t[f"{s}.svh"] = svh[g * r:(g + 1) * r].contiguous()
            t[f"{s}.suh"] = suh.clone(); t[f"{s}.mul1"] = mul1.clone()
        tmp = p + ".tmp"; save_file(t, tmp, metadata=meta); os.replace(tmp, p)
        print(unit, "split into", G, "slices, r =", r)
    for k in [k for k in wm if k.startswith(base + ".") and ".slice." not in k]: del wm[k]
    reindex(fn)

if src:
    sidx = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
    want = [k for k in sidx if ".engram." in k and k.endswith(("q_weight", "k_weight", "wkv.weight", "wkv.scale"))]
    stale = os.path.join(pack, "engram_proj.safetensors")
    if os.path.exists(stale):
        for k in [k for k, v in wm.items() if v == "engram_proj.safetensors"]: del wm[k]
        os.remove(stale)
    for fn in sorted({f"{k.split('.engram.')[0]}.safetensors" for k in want}):
        p = os.path.join(pack, fn)
        with safe_open(p, "pt") as f: have = set(f.keys()); meta = f.metadata()
        need = [k for k in want if k.startswith(fn[:-len("safetensors")]) and k not in have]
        if need:   # stored in the unit's own shard (the SSD tier and the validator read per-layer files)
            with safe_open(p, "pt") as f: t = {k: f.get_tensor(k) for k in f.keys()}
            for shard in sorted({sidx[k] for k in need}):
                with safe_open(os.path.join(src, shard), "pt") as f:
                    for k in need:
                        if sidx[k] == shard: t[k] = f.get_tensor(k)
            save_file(t, p + ".tmp", metadata=meta); os.replace(p + ".tmp", p); print(fn, "+", len(need), "raw Engram projections")
        reindex(fn)
    files = sorted(set(wm.values()))
    idx.setdefault("metadata", {})["total_size"] = sum(os.path.getsize(os.path.join(pack, f)) for f in files)
tmp = idx_p + ".tmp"; json.dump(idx, open(tmp, "w"), indent=2); os.replace(tmp, idx_p)
