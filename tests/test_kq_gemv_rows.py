#!/usr/bin/env python3
"""Q4_K / Q6_K decode GEMV v2 ("rows" kernel) must equal the v1 kernel BIT FOR BIT (float summation order is kept).
Runs the same synthetic problems with BOLT_FQ_KQ_ROWS = 0 (v1), 1, 2, 4, 8 (v2) in child processes and compares the outputs with torch.equal.
usage: python tests/test_kq_gemv_rows.py            (needs a CUDA / HIP GPU and the built extension)"""
import os, subprocess, sys, tempfile

BB = {"q4_k": 144, "q6_k": 210}
# (fmt, N, K): odd N (rows % ROWS != 0), small K (nt padded to 32), K = 12288 (3 chunks per lane), K = 4096, a row-slice view
CASES = [("q4_k", 1, 256), ("q4_k", 3, 512), ("q4_k", 37, 1024), ("q4_k", 1000, 4096), ("q4_k", 4096, 12288), ("q4_k", 5, 2048),
         ("q6_k", 1, 256), ("q6_k", 3, 512), ("q6_k", 37, 1024), ("q6_k", 1000, 4096), ("q6_k", 4096, 12288), ("q6_k", 5, 2048)]


def child(out_path):
    import torch
    from exllamav3.ext import exllamav3_ext as ext
    res = {}
    for fmt, N, K in CASES:
        nb = K // 256
        for slice_off in (0, 3):
            g = torch.Generator(device="cuda").manual_seed(N * 7919 + K + slice_off)
            w = torch.randint(0, 256, (N + slice_off, nb, BB[fmt]), dtype=torch.uint8, device="cuda", generator=g)
            h = (torch.rand((N + slice_off, nb, 2), device="cuda", generator=g) * 0.002 + 0.0001).half().view(torch.uint8)
            if fmt == "q4_k":
                w[:, :, 0:4] = h
            else:
                w[:, :, 208:210] = h[:, :, 0:2]
            wp = w.reshape(N + slice_off, nb * BB[fmt])[slice_off:]            # slice: base pointer at row `slice_off`
            for R in (1, 3):
                x = (torch.randn((R, K), device="cuda", generator=g) * 3).half()
                for dt in (torch.half, torch.float):
                    qb = ext.fq_q8_scratch(0, R, K)
                    ext.fq_quant_q8_1_rows(x, qb)
                    out = torch.empty((R, N), device="cuda", dtype=dt)
                    getattr(ext, f"fq_gemv_{fmt}_q8")(qb, wp, out)
                    res[f"{fmt}|N{N}|K{K}|sl{slice_off}|R{R}|{dt}"] = out.cpu()
    torch.save(res, out_path)


def main():
    import torch
    tmp = tempfile.mkdtemp()
    outs = {}
    for rows in (0, 1, 2, 4, 8, "default"):
        p = os.path.join(tmp, f"r{rows}.pt")
        env = dict(os.environ, BOLT_FQ_KQ_ROWS=str(rows), BOLT_FQ_KQ_MROWS="0")   # R=3: v2 one-row vs v1 per x-row (multi-row has its own test)
        if rows == "default":
            env.pop("BOLT_FQ_KQ_ROWS")
        subprocess.run([sys.executable, os.path.abspath(__file__), "child", p], check=True, env=env)
        outs[rows] = torch.load(p)
    ref = outs[0]
    bad = 0
    for rows in (1, 2, 4, 8, "default"):
        for k, v in ref.items():
            if not torch.equal(v, outs[rows][k]) or not torch.isfinite(v.float()).all():
                print("MISMATCH rows", rows, k); bad += 1
    print("test_kq_gemv_rows:", "PASS (%d cases x 5 row settings, bitwise equal to v1)" % len(ref) if not bad else "FAIL %d" % bad)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "child":
        child(sys.argv[2])
    else:
        main()
