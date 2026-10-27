#!/usr/bin/env python
"""Selection equivalence: indexer top-k of the kernel path vs the torch path (same synthetic pools, same inputs), per layer.
  ds41_sel_equiv.py PACK [--depths 8192,131072,1000000] [--chunk 128]"""
import argparse, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools", "ds41"))
ap = argparse.ArgumentParser(); ap.add_argument("pack"); ap.add_argument("--depths", default="8192,131072,1000000")
ap.add_argument("--chunk", type=int, default=128); a = ap.parse_args()
import torch
from ds41_serving import build_model
from exllamav3.dsv41 import quant
torch.set_grad_enabled(False)
dev = "cuda:0"; depths = [int(x) for x in a.depths.split(",")]
m = build_model(a.pack, device=dev, dtype="float16", mtp=False, moe="auto", prefill_chunk=a.chunk)
top = max(depths) + a.chunk + 64
st = m.new_state(1, top); st.ensure(top)
g = torch.Generator(device=dev).manual_seed(7)
for l in st._kv:
    for pool, fp4 in ((st._kv[l], False), (st._ik[l], True)):
        t = pool.t
        for s in range(0, t.size(1), 1 << 16):
            blk = torch.randn(t[:, s:s + (1 << 16)].shape, generator=g, device=dev, dtype=torch.float16)
            if fp4: quant.fp4_act_quant_(blk, 32)
            t[:, s:s + blk.size(1)] = blk
st.repack_idx()
ids = torch.randint(0, m.cfg.vocab_size - 1, (1, a.chunk), device=dev, generator=g)
bad = 0
snap = lambda: ({l: t.clone() for l, t in st.win.items()}, {l: t.clone() for l, t in st.comp_kv_state.items()}, {l: t.clone() for l, t in st.comp_score_state.items()})


def restore(sn):
    for dct, src in zip((st.win, st.comp_kv_state, st.comp_score_state), sn):
        for l, t in src.items():
            dct[l].copy_(t)


S0 = snap()
for d in depths:
    for L in (1, a.chunk):
        res = {}
        for mode in ("kernel", "torch"):
            m._idx_hip = None if mode == "kernel" else False
            taps = {}
            restore(S0)
            m._forward(ids[:, :L], d, st, taps, "last")
            res[mode] = {k: v.clone() for k, v in taps.items() if k.endswith(".topk")}
        for k in sorted(res["kernel"]):
            x, y = res["kernel"][k], res["torch"][k]
            same = torch.equal(x, y)
            if not same:
                bad += 1
                diff = (x != y).any(-1).sum().item() if x.shape == y.shape else -1
                print(f"MISMATCH depth {d} rows {L} {k}: rows differing {diff}", flush=True)
            else:
                print(f"ok depth {d} rows {L} {k}", flush=True)
print("EQUIV", "PASS" if not bad else f"FAIL ({bad})")
sys.exit(1 if bad else 0)
