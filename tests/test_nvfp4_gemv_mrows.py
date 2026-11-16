#!/usr/bin/env python3
"""NVFP4 multi-row q8 decode GEMV (x-rows 2..16, BOLT_FQ_NVFP4_MROWS): output row m must equal the one-row kernel's output for x-row m
BIT FOR BIT, for R = 1..16 (row groups of 8: R = 9..16 runs two groups), half and float output, several (N, K) shapes. Also checks that
BOLT_FQ_NVFP4_MROWS=0 (the old per-(row, column) launch) gives the same bytes, so batch composition never changes an output.
usage: python tests/test_nvfp4_gemv_mrows.py            (needs a CUDA / HIP GPU and the built extension)"""
import os, subprocess, sys

CASES = [(1, 32), (37, 1024), (512, 2048), (1000, 2048), (2048, 4096), (4096, 512), (5, 12288)]   # (N, K)


def child(tag):
    import torch
    from exllamav3.ext import exllamav3_ext as ext
    torch.manual_seed(0)
    bad = n = 0
    outs = []
    for N, K in CASES:
        packed = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda")
        scales = torch.randint(0x28, 0x48, (N, K // 16), dtype=torch.uint8, device="cuda")      # positive E4M3 scales
        g = 0.0123
        for R in range(1, 17):
            x = torch.randn(R, K, device="cuda", dtype=torch.half)
            qb = torch.empty(R, K // 32 * 36, device="cuda", dtype=torch.uint8)
            ext.fq_quant_q8_1_rows(x, qb)
            for dt in (torch.half, torch.float):
                out = torch.empty(R, N, device="cuda", dtype=dt)
                ext.fq_gemv_nvfp4_q8(qb, packed, scales, g, out)
                for m in range(R):
                    o1 = torch.empty(1, N, device="cuda", dtype=dt)
                    ext.fq_gemv_nvfp4_q8(qb[m:m + 1].contiguous(), packed, scales, g, o1)
                    n += 1
                    if not torch.equal(out[m], o1[0]):
                        bad += 1
                        print(f"{tag}: MISMATCH N {N} K {K} R {R} row {m} {dt}", flush=True)
                outs.append(out.view(torch.uint8).flatten().cpu())
    torch.save(torch.cat(outs), sys.argv[3])
    print(f"{tag}: {n} row checks, {bad} mismatches", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(sys.argv[2])
    import tempfile
    import torch
    with tempfile.TemporaryDirectory() as td:
        rc = 0
        for tag, val in (("mrows", "1"), ("per_row", "0")):
            env = dict(os.environ, BOLT_FQ_NVFP4_MROWS=val)
            rc |= subprocess.run([sys.executable, __file__, "--child", tag, f"{td}/{tag}.pt"], env=env).returncode
        same = torch.equal(torch.load(f"{td}/mrows.pt"), torch.load(f"{td}/per_row.pt"))
        print("BOLT_FQ_NVFP4_MROWS=1 vs 0: " + ("identical bytes" if same else "DIFFERENT"))
        print("PASS" if rc == 0 and same else "FAIL")
        sys.exit(0 if rc == 0 and same else 1)
