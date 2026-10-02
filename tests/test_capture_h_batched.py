"""capture_H deferred batched flush vs the legacy per-call path.

Checks, on synthetic activations shaped like a capture forward:
  - finite-row counts identical (legacy int count vs deferred count_dev)
  - H matches the legacy addmm_ accumulation to fp32 rounding (and reports whether
    it is bit-identical)
  - non-finite rows are excluded from H in both paths (zero-fill == drop)
  - count_dev resolution in finalize_capture_H

Run: python tests/test_capture_h_batched.py   (needs one GPU)
"""
import os
import sys

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav3.modules import linear as lin


def legacy_accumulate(hd, x):
    """Replica of the pre-batch capture_H accumulation body."""
    rows = np.prod(x.shape[:-1])
    dim = x.shape[-1]
    x = x.view((rows, dim)).to(torch.float, copy=True)
    finite = torch.isfinite(x).all(dim=1)
    if not finite.all():
        x = x[finite]
        rows = x.shape[0]
    hd["H"].addmm_(x.T, x)
    hd["count"] += rows


def make_hd(device, dim):
    return {
        "H": torch.zeros(dim, dim, dtype=torch.float32, device=device),
        "first_key": "k",
        "count": 0,
        "finalized": False,
        "num_total": 0,
        "inf_nan": torch.zeros(2, dtype=torch.long, device=device),
        "device": device,
    }


def run_case(device, dims_rows, n_forwards, inject_bad, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    legacy = {d: make_hd(device, d) for d, _ in dims_rows}
    new = {d: make_hd(device, d) for d, _ in dims_rows}

    for _f in range(n_forwards):
        # legacy path: per-call accumulate
        params_l = {"capture": legacy}
        entries = []
        for d, r in dims_rows:
            x = torch.randn(r, d, device=device, dtype=torch.half)
            entries.append((d, x))
        for d, x in entries:
            legacy_accumulate(legacy[d], x.clone())
        # new path: queue views, flush once
        params_n = {"capture": new, "_cap_pending": []}
        for d, x in entries:
            xn = x.clone()
            params_n["_cap_pending"].append((new[d], xn))
        lin.flush_capture_H(params_n)
        assert "_cap_pending" not in params_n or params_n["_cap_pending"] == []

    if inject_bad:
        # one forward with inf/nan rows in the big-dim tensor
        d, r = dims_rows[0]
        x = torch.randn(r, d, device=device, dtype=torch.half)
        x[0, 0] = float("inf")
        x[1, 1] = float("nan")
        x[2, :] = float("-inf")
        legacy_accumulate(legacy[d], x.clone())
        params_n = {"capture": new, "_cap_pending": [(new[d], x)]}
        lin.flush_capture_H(params_n)

    report = []
    for d, _ in dims_rows:
        Hl, Hn = legacy[d]["H"], new[d]["H"]
        cnt_l, cnt_n = legacy[d]["count"], int(new[d].get("count_dev", torch.tensor(0, device=device)).item())
        denom = Hl.abs().max().clamp_min(1e-12)
        rel = ((Hn - Hl).abs().max() / denom).item()
        bitwise = torch.equal(Hl, Hn)
        report.append((d, cnt_l, cnt_n, rel, bitwise))
    return report


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    # homogeneous batch (like the 256 down projections) + singleton big dims
    dims_rows = [(1024, 512)] * 8 + [(4096, 512)]
    ok = True
    for name, kwargs in [
        ("clean x3", dict(n_forwards=3, inject_bad=False)),
        ("clean x2 seed1", dict(n_forwards=2, inject_bad=False, seed=1)),
        ("with bad rows", dict(n_forwards=2, inject_bad=True, seed=2)),
    ]:
        rep = run_case(device, dims_rows, **kwargs)
        print(f"[{name}]")
        for d, cl, cn, rel, bitwise in rep:
            flag = "OK" if (cl == cn and rel <= 1e-5) else "FAIL"
            if flag == "FAIL":
                ok = False
            print(f"  dim={d}: count {cl} vs {cn}  max_rel_diff={rel:.3e}  bitwise={bitwise}  {flag}")

    # finalize resolves count_dev (dim must be a multiple of the Hadamard block, 128)
    hd = make_hd(device, 128)
    hd["count_dev"] = torch.tensor(7, dtype=torch.long, device=device)
    from exllamav3.modules.quant.exl3_lib.quantize import finalize_capture_H
    hd["H"].add_(torch.eye(128, device=device))
    qf, H, L, su, diag = finalize_capture_H(hd, {"K": 3, "seed": 0, "sigma_reg": 0.025}, False)
    ok = ok and hd["count"] == 7
    print(f"finalize count_dev resolution: count={hd['count']} (expect 7) {'OK' if hd['count']==7 else 'FAIL'}")

    print("ALL_OK" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
