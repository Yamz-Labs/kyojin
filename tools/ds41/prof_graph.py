import sys, time, torch, os
sys.path.insert(0, "tools/ds41")
from ds41_serving import build_model
m = build_model(sys.argv[1], device="cuda:0", dtype="float16", layers=3, mtp=False, moe="auto", prefill_chunk=512)
ids = torch.randint(0, m.cfg.vocab_size - 1, (1, 512), device="cuda:0")
st = m.new_state(1, 800); lg, _ = m.prefill(ids, st, 0); tok = lg.argmax(-1).view(1, 1); pos = 512
for i in range(6): lg, _ = m.forward(tok, pos, st); pos += 1; tok = lg.argmax(-1).view(1, 1)
torch.cuda.synchronize()
for i in range(4): lg, _ = m.forward(tok, pos, st); pos += 1; tok = lg.argmax(-1).view(1, 1)
torch.cuda.synchronize(); t = time.time()
for i in range(8): lg, _ = m.forward(tok, pos, st); pos += 1; tok = lg.argmax(-1).view(1, 1)
torch.cuda.synchronize(); print("ms/step", (time.time() - t) / 8 * 1e3)
