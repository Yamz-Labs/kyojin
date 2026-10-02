#!/usr/bin/env python
"""Microbench the grouped-MoE prefill kernel (24% of device time) in isolation on real expert
tensors, sweeping the knobs that don't need a kernel rewrite: EXL3_HIP_PREFILL_MIN_ROWS
(prefill vs decode grouped kernel at the MTP row counts) and, via the layer forward, the
whole MoE block. Prints us/forward per (rows, path)."""
import os, sys, subprocess
py = os.path.expanduser("~/kyojin/.venv/bin/python")
WORKER = r'''
import os, sys, torch
sys.path.insert(0, os.path.expanduser("~/kyojin"))
from exllamav3 import Config, Model
import exllamav3.modules.block_sparse_mlp as bsm
config = Config.from_directory(os.path.expanduser("~/models/Qwen3.8-Flash-Next-EXL3"))
model = Model.from_config(config)
def find(mod):
    if isinstance(mod, bsm.BlockSparseMLP): return mod
    for s in getattr(mod, "modules", []) or []:
        r = find(s)
        if r is not None: return r
mlp = find(model.modules[12]); mlp.load(torch.device("cuda:0"))
torch.manual_seed(0)
xs = {R: torch.randn(1, R, 2560, device="cuda", dtype=torch.half) for R in (1, 2, 3, 4)}
junk = torch.empty(96 << 20, device="cuda", dtype=torch.uint8)
out = []
for R, x in xs.items():
    for _ in range(5): mlp.forward(x, {"attn_mode": "flash_attn"})
    torch.cuda.synchronize()
    st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True); tot = 0.0
    for _ in range(40):
        junk.zero_(); st.record(); mlp.forward(x, {"attn_mode": "flash_attn"}); en.record(); en.synchronize(); tot += st.elapsed_time(en)
    out.append(f"R={R} {tot/40*1000:7.1f}us")
print("RES " + "  ".join(out))
'''
import sys
SETS = {
    "minrows": [
        ("MIN_ROWS=2 CFG=2 (current)", {"EXL3_HIP_PREFILL_MIN_ROWS": "2", "EXL3_MOE_CFG": "2"}),
        ("MIN_ROWS=17 CFG=2 (decode kernel)", {"EXL3_HIP_PREFILL_MIN_ROWS": "17", "EXL3_MOE_CFG": "2"}),
        ("MIN_ROWS=17 CFG=1", {"EXL3_HIP_PREFILL_MIN_ROWS": "17", "EXL3_MOE_CFG": "1"}),
        ("MIN_ROWS=17 CFG=0", {"EXL3_HIP_PREFILL_MIN_ROWS": "17", "EXL3_MOE_CFG": "0"}),
        ("MIN_ROWS=3 CFG=2", {"EXL3_HIP_PREFILL_MIN_ROWS": "3", "EXL3_MOE_CFG": "2"}),
    ],
    "prefillcfg": [
        ("PREFILL_CFG=2 (current)", {"EXL3_HIP_PREFILL_MIN_ROWS": "2", "EXL3_MOE_PREFILL_CFG": "2"}),
        ("PREFILL_CFG=1", {"EXL3_HIP_PREFILL_MIN_ROWS": "2", "EXL3_MOE_PREFILL_CFG": "1"}),
        ("PREFILL_CFG=0", {"EXL3_HIP_PREFILL_MIN_ROWS": "2", "EXL3_MOE_PREFILL_CFG": "0"}),
    ],
}
for label, env in SETS[sys.argv[1] if len(sys.argv) > 1 else "minrows"]:
    r = subprocess.run([py, "-c", WORKER], capture_output=True, text=True, env=dict(os.environ, **env), timeout=900)
    res = [l for l in r.stdout.splitlines() if l.startswith("RES ")]
    print(f"{label:36} {res[0][4:] if res else 'FAIL ' + r.stderr[-300:]}")
