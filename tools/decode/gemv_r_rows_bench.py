#!/usr/bin/env python3
"""T80 microbench: exl3_dec_gemv_r(_multi) R=1..8, synthetic trellis (random int16 = valid bits), kb2=10 (5 bpw).
check: row j of the R-row call == batch-1 call on row j (bitwise). time: us per launch."""
import os, sys, time, argparse
ap = argparse.ArgumentParser()
ap.add_argument("--rows", default="1,2,3,4,5,6,7,8"); ap.add_argument("--reps", type=int, default=60)
ap.add_argument("--shapes", default="qkv,z,o,sh"); ap.add_argument("--noverify", action="store_true"); ap.add_argument("--arms", default="D1"); ap.add_argument("--kb", type=float, default=5.0)
a = ap.parse_args()
sys.path.insert(0, os.environ["EXL3_ROOT"])
import torch
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.quant.exl3 import dec_workspace
dev = torch.device("cuda:0")
g = torch.Generator(device=dev).manual_seed(1)
KB = a.kb
SH = {  # name: (K, [N...])
  "qkv": (2560, [10240]), "z": (2560, [6144]), "o": (6144, [2560]), "sh": (2560, [640, 640]), "qkvz": (2560, [10240, 6144]), "lm": (2560, [248320]),
}
def mk(K, Ns):
    kb2 = int(KB * 2)
    tr = [torch.randint(-32768, 32767, (K // 16, N // 16, kb2 * 8), dtype=torch.int16, device=dev, generator=g) for N in Ns]
    suh = [(torch.rand(K, device=dev, generator=g) * 2 - 1).half() for _ in Ns]
    svh = [(torch.rand(N, device=dev, generator=g) * 2 - 1).half() for N in Ns]
    return tr, suh, svh
scratch1, counters1 = dec_workspace(dev)
ARMS = {
 "D1": {"EXL3_GEMV_R_DEC1": "1"},
 "D1c4": {"EXL3_GEMV_R_DEC1": "1", "EXL3_GEMV_R_RPB": "4"},
 "D1c4f": {"EXL3_GEMV_R_DEC1": "1", "EXL3_GEMV_R_RPB": "4", "EXL3_GEMV_R_CHUNK_FAST": "1"},
 "D0": {},
}
KEYS = ["EXL3_GEMV_R_DEC1", "EXL3_GEMV_R_RPB", "EXL3_GEMV_R_CHUNK_FAST", "EXL3_GEMV_R_RPB78"]
def setarm(name):
    for k in KEYS: os.environ.pop(k, None)
    os.environ.update(ARMS[name])
for name in a.shapes.split(","):
    K, Ns = SH[name]
    tr, suh, svh = mk(K, Ns)
    Ks = [KB] * len(Ns)
    for R in [int(v) for v in a.rows.split(",")]:
        x = (torch.randn(R, K, device=dev, generator=g) * 0.5).half().contiguous()
        outs = [torch.empty(R, N, dtype=torch.half, device=dev) for N in Ns]
        scr = torch.zeros(R * 8 * sum(Ns) * 2, dtype=torch.float, device=dev); ctr = torch.zeros(8192, dtype=torch.int32, device=dev)
        def call():
            if len(Ns) == 1: ext.exl3_dec_gemv_r(x, tr[0], suh[0], svh[0], outs[0], scr, ctr, KB, False)
            else: ext.exl3_dec_gemv_r_multi(x, tr, suh, svh, outs, scr, ctr, Ks, False)
        arms = a.arms.split(',')
        for arm in arms:
          setarm(arm)
          call(); torch.cuda.synchronize()
          bad = 0
          if not a.noverify:
              for j in range(R):
                  o1 = [torch.empty(1, N, dtype=torch.half, device=dev) for N in Ns]
                  if len(Ns) == 1: ext.exl3_dec_gemv(x[j:j+1].contiguous(), tr[0], suh[0], svh[0], o1[0], scratch1, counters1, KB, False)
                  else: ext.exl3_dec_gemv_multi(x[j:j+1].contiguous(), tr, suh, svh, o1, scratch1, counters1, Ks, False)
                  for i in range(len(Ns)):
                      bad += int((o1[i][0].view(torch.int16) != outs[i][j].view(torch.int16)).sum().item())
          for _ in range(5): call()
          torch.cuda.synchronize()
          ts = []
          for _ in range(3):
              e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
              e0.record()
              for _ in range(a.reps): call()
              e1.record(); torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1) * 1000 / a.reps)
          print(f"{name:5s} {arm:6s} R={R} us/launch min {min(ts):7.1f} all {[round(t,1) for t in ts]} mismatch_elems_vs_batch1={bad}", flush=True)
