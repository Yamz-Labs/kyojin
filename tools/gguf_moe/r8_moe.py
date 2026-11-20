# r8_moe.py MODEL_DIR: gm_moe_kq_decode on R = 4 rows vs 4 one-row calls, same explicit routing, on a Q4_K-down and a Q6_K-down layer.
import os, sys, torch
sys.path.insert(0, os.environ.get("EXL3_ROOT", os.getcwd()))
torch.set_grad_enabled(False)
from exllamav3 import Config, Model
from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
from exllamav3 import gm_ext
config = Config.from_directory(sys.argv[1]); model = Model.from_config(config); model.load(device="cuda:0", progressbar=False)
dev = "cuda:0"; m_ext = gm_ext.get()
def walk(ms):
    for m in ms:
        if isinstance(m, BlockSparseMLP): yield m
        else: yield from walk(getattr(m, "modules", []) or [])
mods = list(walk(model.modules)); K, I, E, S = 2048, 768, 128, 8
seen = set()
for m in mods:
    f = m._fq_grouped_kq_ok(torch.zeros(4, K, dtype=torch.half, device=dev))
    if f in seen: continue
    seen.add(f)
    mk = lambda ms: torch.tensor([x.packed.data_ptr() for x in ms], dtype=torch.int64, device=dev)
    g, u, d = mk(m.gates), mk(m.ups), mk(m.downs)
    fd = 0 if f[1] == "q4_k" else 2
    gs = (K // 256) * 144; ds = (I // 256) * (144 if fd == 0 else 210)
    torch.manual_seed(5)
    h = (torch.randn(4, K, device=dev) * 0.5).half()
    sel = torch.stack([torch.randperm(E, device=dev)[:S] for _ in range(4)]).long()
    w = torch.softmax(torch.randn(4, S, device=dev), -1).half()
    def run(hh, ss, ww):
        R = hh.shape[0]
        act = torch.empty(8 * S, I, dtype=torch.half, device=dev); qb = torch.empty(8 * K // 32 * 36, dtype=torch.uint8, device=dev)
        part = torch.empty(8 * S, K, dtype=torch.float, device=dev); out = torch.empty(R, K, dtype=torch.float, device=dev)
        m_ext.gm_moe_kq_decode(hh.contiguous(), ss.reshape(-1).contiguous(), ww.reshape(-1).contiguous(), g, u, d, gs, gs, ds, fd, act, qb, part, out, torch.empty(0, dtype=torch.float, device=dev))
        return out
    o4 = run(h, sel, w)
    o1 = torch.cat([run(h[i:i + 1], sel[i:i + 1], w[i:i + 1]) for i in range(4)])
    print("gm_moe_kq_decode R=4 vs 4 x R=1 bit-exact:", torch.equal(o4, o1), f, "max diff", (o4 - o1).abs().max().item(), flush=True)
