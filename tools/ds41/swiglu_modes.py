import torch, sys
sys.path.insert(0, "tools/ds41")
from exllamav3.modules.dsv41_fn import dsv41_fuse as FZ
from exllamav3.dsv41 import ops
g = torch.Generator(device="cuda:0").manual_seed(3)
for mode in range(6):
    bad = tot = 0
    for sc in (0.05, 1.0, 6.0, 40.0):
        for _ in range(8):
            a = (torch.randn(1, 1 << 20, device="cuda:0", generator=g) * sc).half(); b = (torch.randn(1, 1 << 20, device="cuda:0", generator=g) * sc).half()
            ref = ops.swiglu(a, b, 10.0).half(); o = FZ.swiglu_h(a, b, 10.0, mode)
            bad += int((o.view(torch.int16) != ref.view(torch.int16)).sum()); tot += a.numel()
    print(f"MODE {mode}: {bad} / {tot} mismatches", flush=True)
