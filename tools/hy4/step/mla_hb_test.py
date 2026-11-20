"""mla_hb_test.py: head-batched decode attention vs the shipped per-(row, head) kernel: error + time, real Hy4 shapes (R 512, P 64, H 64 / 32 window layers). Run in job.sh."""
import os, sys, time, statistics
import torch
tree = os.getcwd()
import importlib.util
spec = importlib.util.spec_from_file_location("ref_fused", os.path.join(tree, "exllamav3", "loader", "ref_fused.py"))
rf = importlib.util.module_from_spec(spec); spec.loader.exec_module(rf)
dev = os.environ.get("DEV", "cuda")
CPU = dev == "cpu"
NS = (200, 700) if CPU else (1000, 4000)
R, P = 512, 64
torch.manual_seed(0)
sync = (lambda: None) if CPU else torch.cuda.synchronize


def bench(fn, n=30):
    for _ in range(3):
        fn()
    sync()
    ts = []
    for _ in range(n):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); fn(); e1.record(); sync(); ts.append(e0.elapsed_time(e1) * 1000)
    return statistics.median(ts)


for H, window, ring in ((64, 0, 0), (32, 512, 0), (32, 512, 1024)):
    for N in NS:
        cap = N + 16
        K = (torch.randn(cap, R + P, device=dev) * 1.5)
        sinks = torch.randn(H, device=dev)
        for T in ((1, 3) if CPU else (1, 2, 6)):
            q = torch.randn(T, H, R, device=dev) * 0.5
            qp = torch.randn(T, H, P, device=dev) * 0.5
            qpos = torch.arange(N - T, N, device=dev)
            kpos = torch.arange(cap, device=dev)
            scale = (R + P) ** -0.5 * 1.7
            if ring:
                Kr = torch.zeros(ring, R + P, device=dev)
                for j in range(max(0, N - ring), N):
                    Kr[j % ring] = K[j]
                kc, Nk = Kr, N
            else:
                kc, Nk = K, N
            def call(hb):
                rf.HB = hb
                return rf.mla_sink_attend(q, qp, kc[:, :R], kc[:, R:], qpos, kpos[:Nk], scale, sinks, window, None, kpos_is_arange=True, ring=ring)
            a = call(False)
            ta = None if CPU else bench(lambda: call(False))
            for bn, nw, sp in (((16, 4, 1), (16, 4, 4)) if CPU else ((16, 4, 1), (16, 8, 1), (32, 4, 1), (16, 4, 0), (16, 8, 0), (16, 4, 4), (16, 4, 8), (16, 4, 16), (32, 4, 0))):
                rf.HB_BN, rf.HB_NW, rf.HB_SPLIT = bn, nw, sp
                try:
                    b = call(True)
                except Exception as e:
                    print(f"H {H} win {window} ring {ring} N {N} T {T} BN {bn} NW {nw} S {sp}: FAIL {type(e).__name__} {str(e)[:80]}", flush=True)
                    continue
                err = (a - b).abs().max().item(); rel = ((a - b).norm() / a.norm()).item()
                tb = None if CPU else bench(lambda: call(True))
                print(f"H {H} win {window} ring {ring} N {N} T {T} BN {bn} NW {nw} S {sp}: max_abs {err:.2e} rel {rel:.2e} old {ta} us new {tb} us", flush=True)
print("MLA_HB_DONE", flush=True)
