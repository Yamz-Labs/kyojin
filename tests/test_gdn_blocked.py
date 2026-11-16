#!/usr/bin/env python3
"""test_gdn_blocked.py (ROCm gfx11.5, needs the HIP HF kernel): chunk_gated_delta_rule_pf(blocked=True) on the conv-output row (q / k with H heads, blocked pairing)
vs the old caller-side duplication cat([q]*r + [k]*r + [v]) then blocked=False. Output and final state must be torch.equal."""
import os, sys, torch
sys.path.insert(0, os.getcwd())
from exllamav3.vendor.fla import chunk_gated_delta_rule_pf
from exllamav3.vendor.fla.hip import gdn_hf_hip
torch.manual_seed(0)
dev = "cuda"
ok_all = True
for (H, HV, T) in ((8, 16, 4096), (16, 32, 8192), (16, 48, 2048), (8, 16, 777)):
    K = V = 128; r = HV // H
    C = 2 * H * K + HV * V
    row = (torch.randn(1, T, C, device=dev) * 0.5).to(torch.bfloat16)
    g = (-torch.rand(1, T, HV, device=dev) * 0.2).float()
    beta = torch.sigmoid(torch.randn(1, T, HV, device=dev)).to(torch.bfloat16)
    s0 = (torch.randn(1, HV, K, V, device=dev) * 0.05).float()
    q = row[..., :H * K].view(1, T, H, K); k = row[..., H * K:2 * H * K].view(1, T, H, K); v = row[..., 2 * H * K:].view(1, T, HV, V)
    o1, f1 = chunk_gated_delta_rule_pf(q, k, v, g, beta, initial_state=s0.clone(), output_final_state=True, blocked=True)
    dup = torch.cat([row[..., :H * K]] * r + [row[..., H * K:2 * H * K]] * r + [row[..., 2 * H * K:]], dim=-1).contiguous()
    q2 = dup[..., :HV * K].view(1, T, HV, K); k2 = dup[..., HV * K:2 * HV * K].view(1, T, HV, K); v2 = dup[..., 2 * HV * K:].view(1, T, HV, V)
    o2, f2 = chunk_gated_delta_rule_pf(q2, k2, v2, g, beta, initial_state=s0.clone(), output_final_state=True, blocked=False)
    torch.cuda.synchronize()
    eo, es = torch.equal(o1, o2), torch.equal(f1, f2)
    ok_all &= eo and es
    print(f"H {H} HV {HV} T {T}: out equal {eo} state equal {es} max|do| {(o1.float() - o2.float()).abs().max().item():.3g} "
          f"hf calls {gdn_hf_hip._state.get('calls')}", flush=True)
    # timing: blocked vs dup incl. the cat copy
    for name, fn in (("blocked", lambda: chunk_gated_delta_rule_pf(q, k, v, g, beta, initial_state=s0.clone(), output_final_state=True, blocked=True)),
                     ("dup+cat", lambda: chunk_gated_delta_rule_pf(*(lambda d: (d[..., :HV * K].view(1, T, HV, K), d[..., HV * K:2 * HV * K].view(1, T, HV, K), d[..., 2 * HV * K:].view(1, T, HV, V)))(
                         torch.cat([row[..., :H * K]] * r + [row[..., H * K:2 * H * K]] * r + [row[..., 2 * H * K:]], dim=-1)), g, beta, initial_state=s0.clone(), output_final_state=True))):
        for _ in range(2): fn()
        torch.cuda.synchronize(); e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True); e0.record()
        for _ in range(5): fn()
        e1.record(); torch.cuda.synchronize(); print(f"   {name}: {e0.elapsed_time(e1) / 5:.2f} ms", flush=True)
print("PASS" if ok_all else "FAIL")
