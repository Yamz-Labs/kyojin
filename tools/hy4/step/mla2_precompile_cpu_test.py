"""CPU test (TRITON_INTERPRET=1, no GPU): mla2_precompile launches the real kernels; plan dedupe; the result of _mla2_launch matches a torch reference."""
import os, sys, torch
os.environ.setdefault("TRITON_INTERPRET", "1")
os.environ["HY4_MLA_ROWS2"] = "1"; os.environ["HY4_MLA_ROWS2_TRACE"] = "1"
sys.path.insert(0, os.getcwd())
import importlib.util
spec = importlib.util.spec_from_file_location("ref_fused", "exllamav3/loader/ref_fused.py"); rf = importlib.util.module_from_spec(spec); spec.loader.exec_module(rf)
H, R, P = 16, 64, 16
# plan checks (host only)
for T in (1, 2, 4, 8):
    for N in (512, 1024, 8192):
        for win in (0, 128):
            RG, nrg, ngr, S = rf._mla2_plan(T, H, N, win)
            assert S >= 1 and nrg * RG >= T and ngr == H // rf.R2_BH, (T, N, win, S)
calls = []
orig = rf._mla2_launch
def spy(*a, **k): calls.append((a[0].shape[0], k.get("S"), a[7], a[8])); return orig(*a, **k)
rf._mla2_launch = spy
n = rf.mla2_precompile([(H, R, P, 0, 0), (H, R, P, 128, 0), (H, R, P, 128, 192)], [1, 2], [512, 1024], "cpu")
assert n == len(calls) and n > 0, (n, calls)
assert len(set(calls)) == len(calls)       # one launch per distinct key
print("precompile launches", n, calls)
# numerical check of the real launch vs torch (T=2, N=64, no window)
torch.manual_seed(0)
T, N = 2, 64
kc = torch.randn(N, R + P); ql = torch.randn(T, H, R) * 0.2; qp = torch.randn(T, H, P) * 0.2; qpos = torch.tensor([N - 2, N - 1]); sinks = torch.randn(H)
out = rf._mla2_launch(ql, qp, kc[:, :R], kc[:, R:], qpos, sinks, 0.3, 0, 0, torch.empty(T, H, R))
ref = rf.mla_sink_attend_ref(ql, qp, kc[:, :R], kc[:, R:], qpos, torch.arange(N), 0.3, sinks, 0, None)
print("max abs err vs torch ref", float((out - ref).abs().max()))
assert float((out - ref).abs().max()) < 2e-2
print("CPU_TEST_OK")
