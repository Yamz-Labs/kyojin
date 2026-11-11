"""Batch-1 decode step split for a MoE model on the HIP engine: every BlockSparseMLP forward (1 row, routed + shared experts, chained,
20 reps) vs the measured step. usage: split_moe_hip.py MODEL_DIR STEP_MS"""
import sys, time, statistics as st
import torch
from exllamav3 import Config, Model
from exllamav3.modules.block_sparse_mlp import BlockSparseMLP

config = Config.from_directory(sys.argv[1]); model = Model.from_config(config); model.load(progressbar=False)
mlps, stack = [], list(model.modules)
while stack:
    m = stack.pop()
    if isinstance(m, BlockSparseMLP):
        mlps.append(m)
        continue
    stack.extend(getattr(m, "modules", []) or [])
K = mlps[0].hidden_size if hasattr(mlps[0], "hidden_size") else mlps[0].gates[0].in_features
x = torch.randn((1, 1, K), device="cuda", dtype=torch.half) * 0.1
def run(env=None):
    for m in mlps:
        m.forward(x, {})
for _ in range(3):
    run()
torch.cuda.synchronize()
ts = []
for _ in range(20):
    t = time.perf_counter(); run(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
moe = st.median(ts) * 1e3
step = float(sys.argv[2])
print(f"moe layers {len(mlps)} -> {moe:.1f} ms ({moe / len(mlps) * 1e3:.0f} us/layer); step {step:.1f} ms -> outside MoE {step - moe:.1f} ms ({(step - moe) / step * 100:.0f} %)")
m = mlps[0]
for nm, sub in (("shared", getattr(m, "shared_experts", None)),):
    if sub is not None:
        for _ in range(3):
            sub.forward(x, {})
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(50):
            sub.forward(x, {})
        torch.cuda.synchronize()
        print(f"  {nm} expert alone {(time.perf_counter() - t) / 50 * 1e6:.0f} us/layer")
