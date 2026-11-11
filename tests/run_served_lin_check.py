"""Load a GGUF model dir on the HIP engine; for one ForeignLinear of each format, compare module.forward on R rows to x @ dequant (C++ / gap_formats) and the fp16 ref rows in tests/iq2xxs_ref.npz."""
import sys, collections, numpy as np, torch
from exllamav3 import Config, Model, Cache
from exllamav3.modules.foreign_linear import ForeignLinear
d = sys.argv[1]
config = Config.from_directory(d); model = Model.from_config(config); cache = Cache(model, max_num_tokens=8192, max_batch_size=1); model.load()
seen = collections.Counter(); stack = list(model.modules); mods = []
while stack:
    m = stack.pop(); stack.extend(getattr(m, "modules", []) or [])
    if isinstance(m, ForeignLinear):
        seen[m.format] += 1; mods.append(m)
print("formats", dict(seen), flush=True)
e = model.modules[0]; print("embed", type(e).__name__, {k: (tuple(v.shape), v.dtype) for k, v in e.get_tensors().items()} if hasattr(e, "get_tensors") else None)
ids = torch.tensor([[9707, 11, 1879]]); emb = e.forward(ids, {}).float(); print("embed out", tuple(emb.shape), float(emb.abs().mean()), float(emb.std()))
z = np.load("tests/iq2xxs_ref.npz")
done = set()
for m in mods:
    if m.format in done: continue
    done.add(m.format)
    N, K = m.out_features, m.in_features
    w = m._dequant_rows(0, min(N, 4096)).float()
    for R in (1, 4):
        x = (torch.randn((1, R, K), device="cuda") * 0.5).half()
        y = m.forward(x, {}).float().reshape(R, -1)[:, :w.shape[0]]
        msg = f"{m.format:9s} {m.key} N {N} K {K} R{R}"
        if w is not None:
            ref = x.reshape(R, K).float() @ w.t(); msg += f" rel vs own dequant {float((y - ref).norm() / ref.norm()):.2e}"
        print(msg, flush=True)
for m in mods:
    if m.key == "model.layers.0.self_attn.k_proj":
        x = torch.randn((1, 4, m.in_features), device="cuda").half()
        y = m.forward(x, {}).float().reshape(4, -1)[:, :16]
        ref = x.reshape(4, -1).float() @ torch.from_numpy(z["k4096_ref"].astype(np.float32)).cuda().t()
        print("k_proj served vs mac ref (16 rows) rel", float((y - ref).norm() / ref.norm()))
