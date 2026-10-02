#!/usr/bin/env python
"""Score teacher-forced log-prob files against a reference: tfcmp.py REF A [B ...] (KLD(REF||X), top-1)."""
import sys
import torch
ref = torch.load(sys.argv[1]).float()
for f in sys.argv[2:]:
    q = torch.load(f).float()
    n = min(ref.shape[0], q.shape[0])
    P, Q = ref[:n], q[:n]
    kld = (P.exp() * (P - Q)).sum(-1)
    top1 = (P.argmax(-1) == Q.argmax(-1)).float().mean().item()
    print(f"{sys.argv[1].split('/')[-1]} || {f.split('/')[-1]}: n={n} KLD mean {kld.mean().item():.3e} "
          f"median {kld.median().item():.3e} max {kld.max().item():.3e} top1 {top1*100:.2f}%")
