# Bandwidth of the grouped K-quant dequant: packed bytes read + fp16 written per second.
import os, sys, time, torch
from exllamav3.ext import exllamav3_ext as ext
dev = "cuda"
for name, fmt, bpb in (("q4_k", 0, 144), ("q5_k", 1, 176), ("q6_k", 2, 210)):
    n, k, ec = 512, 2048, 64
    kb = k // 256
    packed = torch.randint(0, 256, (ec, n * kb * bpb), dtype=torch.uint8, device=dev)
    if fmt == 2: packed.view(ec, n * kb, bpb)[:, :, 208:210] = torch.tensor([0x00, 0x2C], dtype=torch.uint8, device=dev)
    if fmt == 0 or fmt == 1:
        packed.view(ec, n * kb, bpb)[:, :, :4] = torch.tensor([0x00, 0x2C, 0x00, 0x2C], dtype=torch.uint8, device=dev)  # sane d/dmin
    ptrs = torch.tensor([packed[e].data_ptr() for e in range(ec)], dtype=torch.int64, device=dev)
    out = torch.empty(ec, n, k, dtype=torch.half, device=dev)
    def grouped(): ext.fq_dequant_kq_grouped(ptrs, out, 0, ec, n, k, fmt)
    fn = getattr(ext, "fq_dequant_" + name)
    out2 = torch.empty(ec, n, k, dtype=torch.half, device=dev)
    def perexp():
        for e in range(ec): fn(packed[e].view(n, kb * bpb), out2[e])
    for lab, f in (("grouped", grouped), ("scalar/exp", perexp)):
        f(); torch.cuda.synchronize(); t = time.time()
        for _ in range(5): f()
        torch.cuda.synchronize(); dt = (time.time() - t) / 5
        gb = ec * n * kb * (bpb + 512) / 1e9
        print(f"{name} {lab:10s} {dt*1e3:7.2f} ms  {gb/dt:6.1f} GB/s (read+write)")
    print("  match:", bool((out == out2).all()))
