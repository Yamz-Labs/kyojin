"""End-to-end check of head_hessian.py on the tiny model: real DSV41Model.forward taps (final.in / final.pre) -> fp32 row
state -> head input must reproduce the model's own logits, and H must satisfy sum(logits^2) == tr(W H W^T).

    TESTS_DIR=<engine tests/dsv41 dir with harness.py> py.sh tools/ds41/test_head_hessian_tiny.py [--work DIR]
(pytest is only imported by the test helpers; a stub module is enough.)"""
import os, sys, json, shutil
HERE_ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE_)
sys.path.insert(0, os.environ["TESTS_DIR"])
import numpy as np, torch
from harness import *
from test_dsv41_e2e import setup, build_mine
from exllamav3.dsv41 import ops
import head_hessian as HH
cfg, tok, tmap, tables, sd = setup(False, 3)
mine = build_mine(cfg, sd, tables, tmap)
n = cfg.num_hidden_layers
rows, L = 3, 37
work = sys.argv[sys.argv.index("--work") + 1] if "--work" in sys.argv else os.path.join(HERE_, "_tiny_head_state"); shutil.rmtree(work, ignore_errors=True)
os.makedirs(f"{work}/ho/L{n-1}")
hs, pms, last = [], [], []
for i in range(rows):
    ids = torch.randint(0, cfg.vocab_size, (1, L), generator=torch.Generator().manual_seed(10+i))
    st = mine.new_state(1, L); taps = {}
    lg, _ = mine.forward(ids, 0, st, taps)
    hs.append(taps["final.in"]); pm = taps["final.pre"]; pms.append(pm); last.append(lg)
    torch.save({"pre_mix": {n-1: pm}}, f"{work}/ho/L{n-1}/row{i}.pt")
h = torch.cat(hs)
mm = np.memmap(f"{work}/rowstate.fp32.mmap", dtype=np.float32, mode="w+", shape=tuple(h.shape)); mm[:] = h.numpy(); mm.flush()
json.dump({"next_layer": n, "layer": None, "rows": rows, "L": L}, open(f"{work}/rowstate.json", "w"))
nw = mine.norm_w
xs = list(HH.head_inputs(work, cfg, nw, None, torch.device("cpu"), group=2))
# A: tool tail reproduces the model's own forward logits (last position of each row)
x = torch.cat(xs).view(rows, L, -1)
e = max(float((mine.head_logits(x[i, -1]) - last[i][0]).abs().max() / last[i].abs().max()) for i in range(rows))
print("A: head logits from tool input vs model.forward logits, max rel:", e); assert e <= 1e-5
# B: Hessian vs sum of squared logits over all tokens of the real forward tail
H, c, nt, inf = HH.accumulate(iter(xs), cfg.hidden_size, torch.device("cpu"))
W = mine.head_w.double()
lhs = float(sum(mine.head_logits(xi).double().square().sum() for xi in xs))
rhs = float(((W @ H) * W).sum())
print("B: sum logits^2", lhs, "tr(WHW^T)", rhs, "rel", abs(lhs-rhs)/lhs); assert abs(lhs-rhs)/lhs <= 1e-5
shutil.rmtree(work, ignore_errors=True)
print("OK")
