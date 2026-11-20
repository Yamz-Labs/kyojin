# r8_test.py MODEL_DIR: (1) bit-exactness of the R-row gm_ext decode ops against R one-row calls (qkv, MoE layer), (2) T-row decode forward: ms per forward
# and teacher-forced PPL on 1024 wiki tokens for T = 1, 2, 3, 4, 6, 8 rows per forward (verify rows). Arm = env (EXL3_GM=0 stock rows).
import os, sys, time, math, json
import torch
sys.path.insert(0, os.environ.get("EXL3_ROOT", os.getcwd()))
torch.set_grad_enabled(False)
from exllamav3 import Config, Model, Cache
from exllamav3.modules.attn import Attention
from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
d, ids_json = sys.argv[1], sys.argv[2]
config = Config.from_directory(d); model = Model.from_config(config)
cache = Cache(model, max_num_tokens=2048 + 256); model.load(device="cuda:0", progressbar=False)
dev = "cuda:0"
def walk(ms, cls):
    for m in ms:
        if isinstance(m, cls): yield m
        else: yield from walk(getattr(m, "modules", []) or [], cls)
if os.environ.get("EXL3_GM", "1") != "0":
    torch.manual_seed(3)
    for a in list(walk(model.modules, Attention))[:4]:
        K = a.q_proj.in_features; x = (torch.randn(4, 1, K, device=dev) * 0.5).half()
        r4 = a._qkv_gm3(x, {}, 4, 1)
        if r4 is None: print("qkv gm3 declined", a.key); continue
        ok = True
        for i in range(4):
            r1 = a._qkv_gm3(x[i:i + 1], {}, 1, 1)
            ok &= all(torch.equal(u[i:i + 1], w) for u, w in zip(r4, r1))
        print("qkv R=4 bit-exact vs 4 x R=1:", ok, a.key, flush=True)
    seen = set()
    for m in walk(model.modules, BlockSparseMLP):
        f = m._fq_grouped_kq_ok(torch.zeros(4, 2048, dtype=torch.half, device=dev))
        if f in seen: continue
        seen.add(f); K = 2048
        y = (torch.randn(4, 1, K, device=dev) * 0.5).half()
        r4 = m.forward(y, {}).clone()
        ok = all(torch.equal(r4[i:i + 1], m.forward(y[i:i + 1], {})) for i in range(4))
        print("moe R=4 bit-exact vs 4 x R=1:", ok, f, m.key, flush=True)
ids = torch.tensor([json.load(open(ids_json))[:1024]], dtype=torch.long)
pages = cache.max_num_tokens // 256
bt = torch.arange(pages, dtype=torch.int32, device=dev).view(1, pages)
res = {}
for T in (1, 2, 3, 4, 6, 8):
    sl = torch.zeros(1, dtype=torch.int32, device=dev); nll = 0.0; cnt = 0; tms = []
    for i in range(0, 1024 - T, T):
        x = ids[:, i:i + T].to(dev)
        torch.cuda.synchronize(); t0 = time.time()
        lg = model.forward(x, {"attn_mode": "flash_attn", "block_table": bt, "cache": cache, "cache_seqlens": sl})
        torch.cuda.synchronize(); tms.append(time.time() - t0)
        sl += T
        if i >= 32:
            lp = torch.log_softmax(lg[0].float(), dim=-1)
            tg = ids[0, i + 1:i + T + 1].to(dev)
            nll += -lp[torch.arange(len(tg)), tg].sum().item() if len(tg) == T else 0.0; cnt += len(tg) if len(tg) == T else 0
    ms = sorted(tms[16:])[len(tms[16:]) // 2] * 1000
    res[T] = (round(ms, 2), round(math.exp(nll / cnt), 4)); print(f"T={T} median {ms:.2f} ms/forward  ppl {math.exp(nll / cnt):.4f}", flush=True)
print("RESULT", json.dumps(res), flush=True)
