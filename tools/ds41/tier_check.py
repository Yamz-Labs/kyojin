"""SSD expert tier exactness: logits with EXL3_EXPERT_BUDGET_GIB set must equal the resident run bit for bit.

usage: tier_check.py PACK LAYERS REF.pt [PREFILL=256] [STEPS=12]
  first run (no EXL3_EXPERT_BUDGET_GIB): writes REF.pt;  second run (EXL3_EXPERT_BUDGET_GIB=<gib>): compares, prints the store stats.
Exit 1 when tokens or logits differ.
"""
import os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ds41_serving import build_model

pack, n, ref = sys.argv[1], int(sys.argv[2]), sys.argv[3]
PF = int(sys.argv[4]) if len(sys.argv) > 4 else 256
STEPS = int(sys.argv[5]) if len(sys.argv) > 5 else 12
tier = os.environ.get("EXL3_EXPERT_BUDGET_GIB", "") != ""
t0 = time.time()
m = build_model(pack, device="cuda:0", dtype="float16", layers=n, mtp=False, moe="auto", prefill_chunk=PF)
torch.cuda.synchronize()
print(f"load {time.time() - t0:.1f}s tier={'on' if tier else 'off'} tiered_blocks={sum(getattr(b, 'tier', None) is not None for b in m.src.blocks.values()) if hasattr(m, 'src') else '?'}", flush=True)
g = torch.Generator(device="cuda:0").manual_seed(7)
ids = torch.randint(0, m.cfg.vocab_size - 1, (1, PF), device="cuda:0", generator=g)
st = m.new_state(1, PF + STEPS + 8)
lg, _ = m.prefill(ids, st, 0, chunk=PF)
out = {"prefill": lg.float().cpu()}
tok, pos, toks, lgs = lg.argmax(-1).view(1, 1), PF, [], []
for _ in range(STEPS):
    lg, _ = m.forward(tok, pos, st)
    pos += 1
    tok = lg.argmax(-1).view(1, 1)
    toks.append(int(tok)); lgs.append(lg.float().cpu())
out["toks"], out["lgs"] = toks, torch.stack(lgs)
if tier:
    from exllamav3.modules.expert_store import get_store
    cfg = [b for b in m.pack_modules[0].modules if getattr(b, "tier", None) is not None][0].config
    print("store stats:", get_store(cfg).stats(), flush=True)
if not tier:
    torch.save(out, ref); print("saved", ref); sys.exit(0)
o = torch.load(ref)
same = o["toks"] == toks and torch.equal(o["prefill"], out["prefill"]) and torch.equal(o["lgs"], out["lgs"])
print("tokens equal", o["toks"] == toks, "| prefill bitwise", torch.equal(o["prefill"], out["prefill"]),
      "| decode logits bitwise", torch.equal(o["lgs"], out["lgs"]), "| max|d|", (o["lgs"] - out["lgs"]).abs().max().item())
print("TIER EXACT" if same else "TIER MISMATCH")
sys.exit(0 if same else 1)
