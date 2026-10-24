"""Decode / verify host-overhead bench on a (synthetic) pack: median of 5. usage: bench_decode.py PACK LAYERS [save|cmp FILE]"""
import sys, time, statistics, torch
sys.path.insert(0, "tools/ds41")
from ds41_serving import build_model
pack, n = sys.argv[1], int(sys.argv[2])
mode = sys.argv[3] if len(sys.argv) > 3 else "none"; ref = sys.argv[4] if len(sys.argv) > 4 else None
PF, N, R = 512, 24, 5
m = build_model(pack, device="cuda:0", dtype="float16", layers=n, mtp=False, moe="auto", prefill_chunk=PF)
g = torch.Generator(device="cuda:0").manual_seed(5)
ids = torch.randint(0, m.cfg.vocab_size - 1, (1, PF), device="cuda:0", generator=g)
def fresh():
    st = m.new_state(1, PF + 256); lg, _ = m.prefill(ids, st, 0, chunk=PF); torch.cuda.synchronize(); return st, lg
def step(st, tok, pos):
    return m.forward(tok, pos, st)[0]
res, cpu = [], []
for r in range(R + 1):
    st, lg = fresh(); pos = PF; tok = lg.argmax(-1).view(1, 1)
    torch.cuda.synchronize(); t = time.time(); tc = 0.0
    for _ in range(N):
        t1 = time.time(); lg = step(st, tok, pos); tc += time.time() - t1
        pos += 1; tok = lg.argmax(-1).view(1, 1)
    torch.cuda.synchronize(); dt = (time.time() - t) / N
    if r: res.append(dt * 1e3); cpu.append(tc / N * 1e3)
print(f"decode layers={n}: median {statistics.median(res):.2f} ms/tok  runs {[round(x,2) for x in res]}  host-issue {statistics.median(cpu):.2f} ms")
# verify block of 4 rows
vr = []
for r in range(R + 1):
    st, lg = fresh(); vt = torch.randint(0, m.cfg.vocab_size - 1, (1, 4), device="cuda:0", generator=g)
    torch.cuda.synchronize(); t = time.time()
    for k in range(8):
        st.begin_block(PF + 4 * k, 4); m.forward_block(vt, PF + 4 * k, st); st.rollback(2)
    torch.cuda.synchronize()
    if r: vr.append((time.time() - t) / 8 * 1e3)
print(f"verify N=4 layers={n}: median {statistics.median(vr):.2f} ms/block  runs {[round(x,2) for x in vr]}")
# exactness trace: greedy 12 tokens + logits, and one verify block
st, lg = fresh(); pos = PF; tok = lg.argmax(-1).view(1, 1); toks, lgs = [], []
for _ in range(12):
    lg = step(st, tok, pos); pos += 1; tok = lg.argmax(-1).view(1, 1); toks.append(int(tok)); lgs.append(lg.float().cpu())
st.begin_block(pos, 4); bl, _ = m.forward_block(torch.tensor([toks[:4]], device="cuda:0"), pos, st); st.rollback(1)
lg = step(st, tok, pos + 1)
out = {"toks": toks, "lgs": torch.stack(lgs), "blk": bl.float().cpu(), "after": lg.float().cpu()}
if mode == "save": torch.save(out, ref); print("saved")
elif mode == "cmp":
    o = torch.load(ref)
    print("tokens equal", o["toks"] == toks)
    for k in ("lgs", "blk", "after"):
        d = (o[k] - out[k]).abs().max().item(); print(k, "max|d|", d, "rel", d / o[k].abs().max().item())
