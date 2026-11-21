"""CPU test (TRITON_INTERPRET=1, no GPU): HY4_MLA_ROWS2 fires counter. Flag on: dispatches count as trunk, the r2_draft decorator counts as draft and restores the role (also on
an exception), decode-size fallbacks carry a reason, the HY4_MLA_ROWS2_FIRES file holds the line. Flag off: the counter stays at 0 and the old path runs."""
import os, sys, tempfile, importlib.util, torch
os.environ["TRITON_INTERPRET"] = "1"
path = os.path.join(tempfile.mkdtemp(), "fires.txt")
os.environ["HY4_MLA_ROWS2"] = "1"; os.environ["HY4_MLA_ROWS2_FIRES"] = path
spec = importlib.util.spec_from_file_location("ref_fused", "exllamav3/loader/ref_fused.py"); rf = importlib.util.module_from_spec(spec); spec.loader.exec_module(rf)
H, R, P, T, N = 16, 64, 16, 2, 64
torch.manual_seed(0)
kc = torch.randn(N, R + P); ql = torch.randn(T, H, R) * 0.2; qp = torch.randn(T, H, P) * 0.2; qpos = torch.tensor([N - 2, N - 1]); sinks = torch.randn(H); kpos = torch.arange(N)
call = lambda **k: rf.mla_sink_attend(ql, qp, kc[:, :R], kc[:, R:], qpos, kpos, 0.3, sinks, 0, None, **k)
call(kpos_is_arange=True)
assert rf.R2_FIRES["calls"] == 1 and rf.R2_BY_ROLE["trunk"]["calls"] == 1 and rf.R2_BY_ROLE["draft"]["calls"] == 0
rf.r2_draft(lambda: call(kpos_is_arange=True))()
assert rf.R2_BY_ROLE["draft"]["calls"] == 1 and rf.R2_ROLE[0] == "trunk" and rf.R2_FIRES["calls"] == 2
def boom(): raise RuntimeError("x")
try: rf.r2_draft(boom)()
except RuntimeError: pass
assert rf.R2_ROLE[0] == "trunk"
call(kpos_is_arange=False)                                  # fallback reason: kpos not arange
assert rf.R2_FIRES["fallback"] == {"kpos not arange": 1} and rf.R2_BY_ROLE["trunk"]["fallback"] == {"kpos not arange": 1}, rf.R2_FIRES
line = open(path).read().strip()
assert line.startswith("FIRES mla2 2 trunk 1 draft 1 fallback {"), line
rf.R2_ROLE[0] = "trunk"; rf.ROWS2 = False; before = dict(rf.R2_FIRES); call(kpos_is_arange=True)
assert rf.R2_FIRES == before                                 # flag off: no note, no mla2
print(line); print("FIRES_CPU_TEST_OK")
