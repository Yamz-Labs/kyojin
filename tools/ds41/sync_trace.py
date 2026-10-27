import sys, torch, warnings, traceback
sys.path.insert(0, "tools/ds41")
from ds41_serving import build_model
m = build_model(sys.argv[1], device="cuda:0", dtype="float16", layers=3, mtp=False, moe="auto", prefill_chunk=512)
ids = torch.randint(0, m.cfg.vocab_size - 1, (1, 512), device="cuda:0")
st = m.new_state(1, 800); lg, _ = m.prefill(ids, st, 0); tok = lg.argmax(-1).view(1, 1)
for i in range(3): lg, _ = m.forward(tok, 512 + i, st); tok = lg.argmax(-1).view(1, 1)
torch.cuda.synchronize()
import collections; C = collections.Counter()
def show(message, category, filename, lineno, file=None, line=None):
    st_ = traceback.extract_stack()
    fr = [f"{f.filename.split('/')[-1]}:{f.lineno}" for f in st_ if "ds41g" in f.filename and "sync_trace" not in f.filename]
    C[" < ".join(fr[-3:][::-1])] += 1
warnings.showwarning = show
torch.cuda.set_sync_debug_mode("warn")
lg, _ = m.forward(tok, 515, st)
torch.cuda.set_sync_debug_mode(0)
for k, v in C.items(): print(v, k)
