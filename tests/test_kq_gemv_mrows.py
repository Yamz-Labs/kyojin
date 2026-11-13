#!/usr/bin/env python3
"""Q4_K / Q6_K multi-row decode GEMV (x-rows 2..16): output row m must equal the one-row kernel's output for x-row m BIT FOR BIT,
for every row count R = 1..16, every launch shape (output rows per block, x-rows per pass), half and float output, row-slice views,
and the fallbacks (K not a multiple of 256 is not representable here; a base pointer off by one byte falls back to the v1 kernel).
Also checks that BOLT_FQ_KQ_MROWS=0 (old per-row launch) gives the same values, so batch composition never changes an output.
Mode "wmma" (gfx11 HIP): BOLT_FQ_KQ_WMMA=1 runs the WMMA kernel (fq_kq_wmma.cuh) for every eligible case (Q4_K, K >= 4096, aligned); every other case
falls back to the generic multi-row kernel inside the same launch, so all cases must still match the one-row kernel. Mode "auto" leaves the
variable unset (the default dispatch rule picks WMMA or the generic kernel per shape). The "mrows" mode pins BOLT_FQ_KQ_WMMA=0 so that it keeps
testing the generic kernel.
usage: python tests/test_kq_gemv_mrows.py            (needs a CUDA / HIP GPU and the built extension)"""
import os, subprocess, sys

BB = {"q4_k": 144, "q6_k": 210}
# (fmt, N, K): odd N (N % WROWS != 0), small K (nt padded to 32), K = 12288 (3 chunks per lane), K = 4096, lm_head-like N
CASES = [("q4_k", 1, 256), ("q4_k", 3, 512), ("q4_k", 37, 1024), ("q4_k", 1000, 4096), ("q4_k", 4096, 12288), ("q4_k", 5, 2048),
         ("q4_k", 12288, 4096), ("q4_k", 1024, 4096),
         ("q6_k", 1, 256), ("q6_k", 3, 512), ("q6_k", 37, 1024), ("q6_k", 1000, 4096), ("q6_k", 4096, 12288), ("q6_k", 5, 2048),
         ("q6_k", 151936, 4096)]
# shapes of the 8B decode GEMVs and the WMMA edge cases (K = 6144: 1.5 windows per lane; N = 4096 x 4096 / 12288 x 4096 / 4096 x 12288; lm_head)
WMMA_CASES = [("q4_k", 4096, 4096), ("q4_k", 100, 6144), ("q4_k", 151936, 4096), ("q6_k", 4096, 4096), ("q6_k", 12288, 4096),
              ("q4_k", 33, 4096), ("q4_k", 1024, 4096), ("q4_k", 3000, 8192), ("q6_k", 100, 6144)]
CONFIGS = [(0, 0, 0), (1, 2, 1), (1, 4, 1), (1, 8, 1), (1, 16, 1), (2, 2, 1), (2, 4, 1), (2, 8, 1), (4, 2, 1), (4, 4, 1),
           (1, 2, 2), (1, 4, 2), (2, 2, 2), (2, 4, 2)]   # (BOLT_FQ_KQ_MR_ROWS, BOLT_FQ_KQ_MR_RT, BOLT_FQ_KQ_MR_WT), 0 = default
RS = list(range(1, 17))


def child(mode):
    import torch
    from exllamav3.ext import exllamav3_ext as ext
    if mode in ("wmma", "auto") and not torch.version.hip:
        print(f"child {mode}: skipped (not a HIP build)", flush=True); sys.exit(0)
    bad = [0]; n = [0]

    def gemv(fmt, w, x, dt):
        R = x.shape[0]
        qb = ext.fq_q8_scratch(0, R, x.shape[1])
        ext.fq_quant_q8_1_rows(x, qb)
        out = torch.empty((R, w.shape[0]), device="cuda", dtype=dt)
        getattr(ext, f"fq_gemv_{fmt}_q8")(qb, w, out)
        return out

    def check(name, got, ref):
        n[0] += 1
        if not torch.equal(got, ref):
            print("MISMATCH", name, flush=True); bad[0] += 1

    # the test is evidence only if the multi-row kernel really runs (every fallback gives the same values): look at the launched kernels
    from torch.profiler import profile, ProfilerActivity
    for fmt in ("q4_k", "q6_k"):
        w = torch.randint(0, 256, (256, 16 * BB[fmt]), dtype=torch.uint8, device="cuda")
        w.view(256, 16, BB[fmt])[:, :, 0:4 if fmt == "q4_k" else 0] = 0
        x = torch.randn((8, 4096), device="cuda").half()
        gemv(fmt, w, x, torch.half); torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            gemv(fmt, w, x, torch.half); torch.cuda.synchronize()
        names = [e.key for e in prof.key_averages()]
        ran = any("fq_gemv_kq_mrows_kernel" in k for k in names)
        ran_wmma = any("fq_gemv_kq_wmma" in k for k in names)
        if not names:                                      # torch.profiler sees no kernel at all (ROCm): nothing to check here
            print(f"kernel check {fmt} mode {mode}: skipped, the profiler returned no kernel (check with rocprofv3 --kernel-trace)", flush=True)
            continue
        n[0] += 1
        if ran != (mode == "mrows") or (mode == "wmma" and not ran_wmma):
            print(f"MISMATCH kernel check {fmt} mode {mode}: multi-row kernel ran = {ran}, wmma ran = {ran_wmma}; kernels {names}", flush=True); bad[0] += 1
        else:
            print(f"kernel check {fmt} mode {mode}: multi-row kernel ran = {ran}, wmma ran = {ran_wmma}", flush=True)
    for fmt, N, K in (CASES + WMMA_CASES if mode in ("wmma", "auto") else CASES):
        nb = K // 256
        for variant in ("aligned", "slice", "odd"):
            if variant == "odd" and N > 5000:
                continue
            off = 3 if variant == "slice" else 0
            g = torch.Generator(device="cuda").manual_seed(N * 7919 + K + off)
            w = torch.randint(0, 256, (N + off, nb, BB[fmt]), dtype=torch.uint8, device="cuda", generator=g)
            h = (torch.rand((N + off, nb, 2), device="cuda", generator=g) * 0.002 + 0.0001).half().view(torch.uint8)
            if fmt == "q4_k":
                w[:, :, 0:4] = h
            else:
                w[:, :, 208:210] = h[:, :, 0:2]
            wp = w.reshape(N + off, nb * BB[fmt])[off:]
            if variant == "odd":                                   # base pointer 1 byte off: every kernel falls back to v1 (same values)
                buf = torch.empty(wp.numel() + 1, dtype=torch.uint8, device="cuda")
                buf[1:] = wp.reshape(-1)
                wp = buf[1:].reshape(N, nb * BB[fmt])
                assert wp.data_ptr() % 2 == 1
            xs = (torch.randn((16, K), device="cuda", generator=g) * 3).half()
            xs[5] = 0                                              # all-zero row
            xs[9] *= 40                                            # large magnitudes
            xs[11, ::7] = 0
            for dt in (torch.half, torch.float):
                tag = f"{fmt}|N{N}|K{K}|{variant}|{dt}"
                ref = torch.cat([gemv(fmt, wp, xs[m:m + 1], dt) for m in range(16)])      # one x-row per call: the one-row kernel
                if dt == torch.float and not torch.isfinite(ref).all():   # half output may overflow to inf on the x40 row (equal bit for bit all the same)
                    print("NONFINITE", tag); bad[0] += 1
                if mode in ("wmma", "auto"):
                    for R in RS:
                        check(f"{tag}|R{R}|{mode}", gemv(fmt, wp, xs[:R].contiguous(), dt), ref[:R])
                elif mode == "mrows":
                    cfgs = CONFIGS if N <= 5000 else [(0, 0, 0), (1, 8, 1), (2, 4, 2)]
                    for wr, rt, wt in cfgs:
                        os.environ["BOLT_FQ_KQ_MR_ROWS"] = str(wr); os.environ["BOLT_FQ_KQ_MR_RT"] = str(rt); os.environ["BOLT_FQ_KQ_MR_WT"] = str(wt)
                        for R in RS:
                            check(f"{tag}|R{R}|wr{wr}|rt{rt}|wt{wt}", gemv(fmt, wp, xs[:R].contiguous(), dt), ref[:R])
                    os.environ.pop("BOLT_FQ_KQ_MR_ROWS", None); os.environ.pop("BOLT_FQ_KQ_MR_RT", None); os.environ.pop("BOLT_FQ_KQ_MR_WT", None)
                else:                                              # BOLT_FQ_KQ_MROWS=0: old launch, one block per (x-row, output row) for M >= 2
                    for R in RS:
                        check(f"{tag}|R{R}|off", gemv(fmt, wp, xs[:R].contiguous(), dt), ref[:R])
    print(f"child {mode}: {n[0]} comparisons, {bad[0]} bad", flush=True)
    sys.exit(1 if bad[0] else 0)


def main():
    rc = 0
    for mode, extra in (("mrows", {"BOLT_FQ_KQ_WMMA": "0"}), ("wmma", {"BOLT_FQ_KQ_WMMA": "1"}), ("auto", {}), ("off", {"BOLT_FQ_KQ_MROWS": "0"})):
        env = dict(os.environ, **extra)
        env.pop("BOLT_FQ_KQ_ROWS", None)
        if mode == "auto":
            env.pop("BOLT_FQ_KQ_WMMA", None)
        if mode != "off":
            env.pop("BOLT_FQ_KQ_MROWS", None)
        rc |= subprocess.run([sys.executable, os.path.abspath(__file__), "child", mode], env=env).returncode
    print("test_kq_gemv_mrows:", "PASS (multi-row and WMMA == one-row bitwise, R 1..16, %d launch shapes, half/float out, slices, misaligned fallback, old path)" % len(CONFIGS) if rc == 0 else "FAIL")
    sys.exit(rc)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "child":
        child(sys.argv[2])
    else:
        main()
