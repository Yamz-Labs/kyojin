#!/usr/bin/env python3
"""Phase 1 trivial two-GPU sanity check.

Allocates 8 GiB on each device, runs a matmul on each, then cross-copies a
10 KB tensor in both directions through pinned host memory and verifies it.
"""
import time

import torch


def arch_of(props):
    arch = getattr(props, "gcnArchName", None) or getattr(props, "gcn_arch_name", "")
    return str(arch).lower().split(":")[0]


def main():
    print(f"torch {torch.__version__}  HIP {torch.version.hip}  python {__import__('sys').version.split()[0]}")
    print(f"arch list: {torch.cuda.get_arch_list()}")
    devs = {}
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        devs[arch_of(p)] = torch.device(f"cuda:{i}")
        print(f"cuda:{i} {p.name} {arch_of(p)} {p.total_memory / 2**30:.2f} GiB")
    assert {"gfx1151", "gfx1201"} <= set(devs), sorted(devs)
    A, B = devs["gfx1151"], devs["gfx1201"]

    hold = []
    for d in (A, B):
        hold.append(torch.empty(8 * 2**30, dtype=torch.uint8, device=d))
        x = torch.randn(4096, 4096, dtype=torch.float16, device=d)
        t0 = time.perf_counter()
        y = x @ x
        torch.cuda.synchronize(d)
        dt = time.perf_counter() - t0
        print(f"{d} matmul 4096^2 fp16 ok: {dt * 1e3:.1f} ms, y[0,0]={y[0, 0].item():.3f}, "
              f"alloc {torch.cuda.memory_allocated(d) / 2**30:.2f} GiB")
        del x, y
    del hold

    n = 10 * 1024
    pin = torch.empty(n, dtype=torch.uint8, device="cpu").pin_memory()
    for src, dst in ((A, B), (B, A)):
        src_t = torch.randint(0, 256, (n,), dtype=torch.uint8, device=src)
        dst_t = torch.empty(n, dtype=torch.uint8, device=dst)
        src_s = torch.cuda.Stream(device=src)
        dst_s = torch.cuda.Stream(device=dst)

        def hop():
            with torch.cuda.stream(src_s):
                pin.copy_(src_t, non_blocking=True)
                ev = src_s.record_event()
            dst_s.wait_event(ev)
            with torch.cuda.stream(dst_s):
                dst_t.copy_(pin, non_blocking=True)
            dst_s.synchronize()

        for _ in range(20):
            hop()
        t0 = time.perf_counter()
        for _ in range(200):
            hop()
        dt = (time.perf_counter() - t0) / 200
        ok = torch.equal(dst_t.cpu(), src_t.cpu())
        print(f"cross 10 KB {src} -> {dst}: {dt * 1e6:.1f} us, bit-identical: {ok}")
        assert ok


if __name__ == "__main__":
    main()
