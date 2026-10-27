"""Self-test of head_hessian.py on a synthetic row state (real config, real final norm and head weights).

    py.sh tools/ds41/test_head_hessian.py --config config.json --map tensor_map.json --work DIR [--device cpu]

Checks (rel err <= 1e-5):
  1. H against an independent fp64 re-implementation of the tail (hc combine + RMS norm, written by hand);
  2. H against the model's own tail code path (ops.hc_pre / ops.rmsnorm, as in model._forward), through the logits:
       sum(logits^2) with logits = F.linear(x, head_w)   ==   tr(W H W^T)   (the quantity the quantizer minimizes);
  3. the spool file round-trips through split.unpack_hessian and is accepted as a HessianEntry by the quantizer lookup.
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import head_hessian as HH  # noqa: E402


def make_state(work, cfg, rows, L, seed=0):
    d = os.path.join(work, "state")
    shutil.rmtree(d, ignore_errors=True)
    n = cfg.num_hidden_layers
    os.makedirs(os.path.join(d, "ho", f"L{n - 1}"))
    g = torch.Generator().manual_seed(seed)
    scale = torch.exp(torch.randn(cfg.hidden_size, generator=g) * 1.5)     # outlier channels like a real residual
    h = (torch.randn(rows, L, cfg.hc_mult, cfg.hidden_size, generator=g) * scale).to(torch.bfloat16)
    h[0, 3, 1, :] *= 40.0                                                   # a massive-activation token
    mm = np.memmap(os.path.join(d, "rowstate.bf16.mmap"), dtype=np.uint16, mode="w+", shape=tuple(h.shape))
    mm[:] = h.view(torch.int16).numpy().view(np.uint16)
    mm.flush()
    pms = []
    for i in range(rows):
        pm = torch.sigmoid(torch.randn(1, L, cfg.hc_mult, generator=g)) + 1e-6
        pms.append(pm)
        torch.save({"pre_mix": {n - 1: pm, n - 2: pm * 0}, "pool": {}, "idxk": {}, "topk": {}, "cand": None,
                    "main": {}}, os.path.join(d, "ho", f"L{n - 1}", f"row{i}.pt"))
    json.dump({"next_layer": n, "layer": None, "rows": rows, "L": L}, open(os.path.join(d, "rowstate.json"), "w"))
    return d, h, torch.cat(pms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--len", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-head", action="store_true", help="skip the logits check (no real head weight)")
    a = ap.parse_args()
    import fetch as F_
    import split as SP
    from exllamav3.dsv41.config import DSV41Config
    from exllamav3.dsv41 import ops
    cfg = DSV41Config.from_file(a.config)
    tm = json.load(open(a.map))
    fetcher = F_.TensorFetcher(tm)
    norm_w = fetcher.fetch("norm.weight")
    state, h, pm = make_state(a.work, cfg, a.rows, a.len)
    dev = torch.device(a.device)

    # --- tool
    spool = os.path.join(a.work, "spool")
    shutil.rmtree(spool, ignore_errors=True)
    xs = HH.head_inputs(state, cfg, norm_w, None, dev, group=2)
    H, count, nt, infnan = HH.accumulate(xs, cfg.hidden_size, dev)
    path, nbytes = HH.write_spool(spool, "global", H.cpu(), count, nt, infnan)
    N = a.rows * a.len
    assert count == N and infnan == [0, 0], (count, infnan)

    # --- 1. independent fp64 reference of the tail
    hd = h.double()
    comb = (pm.double().unsqueeze(-1) * hd).sum(2)                         # [rows, L, d]  hc_pre
    comb = comb.to(torch.bfloat16).double() if False else comb
    nwd = norm_w.double()
    xr = comb * torch.rsqrt(comb.square().mean(-1, keepdim=True) + cfg.rms_norm_eps) * nwd
    xr = xr.reshape(-1, cfg.hidden_size)
    Hr = xr.T @ xr
    e1 = float((H - Hr).norm() / Hr.norm())
    print(f"1. rel err vs hand-written fp64 tail: {e1:.2e}")
    assert e1 <= 1e-5, e1

    # --- 2. model tail code path + head logits
    xm = []
    for i in range(a.rows):
        x = ops.hc_pre(h[i:i + 1].float(), pm[i:i + 1])
        xm.append(ops.rmsnorm(x, norm_w.float(), cfg.rms_norm_eps).reshape(-1, cfg.hidden_size))
    xm = torch.cat(xm)
    Hm = (xm.double().T @ xm.double())
    e2 = float((H - Hm).norm() / Hm.norm())
    print(f"2a. rel err vs model ops tail (fp64 accumulation): {e2:.2e}")
    assert e2 <= 1e-5, e2

    # --- 3. spool round trip + quantizer lookup
    sp = SP.Spool(spool)
    assert sp.has("global", "lin.head") and sp.capture_done("global")
    ent = sp.get("global", "lin.head")
    hdd = ent.load()
    Hs = hdd["H"].double()
    e3 = float((Hs - Hm).norm() / Hm.norm())
    print(f"3. rel err of the spool matrix after pack/unpack vs reference: {e3:.2e}  (count {hdd['count']}, "
          f"num_total {hdd['num_total']}, {nbytes / 1e6:.0f} MB)")
    assert e3 <= 1e-5 and hdd["count"] == N

    if not a.no_head:
        W = fetcher.fetch("head.weight").float()                           # [vocab, d], model.head_w
        lg = F.linear(xm, W)                                               # model._forward: F.linear(x.float(), head_w)
        lhs = float(lg.double().square().sum())
        Wh = W @ Hs.float()
        rhs = float((Wh * W).double().sum())
        e4 = abs(lhs - rhs) / lhs
        print(f"2b. sum(logits^2) = {lhs:.6e}  tr(W H W^T) from the spool = {rhs:.6e}  rel err {e4:.2e}")
        assert e4 <= 1e-5, e4
    print("OK")


if __name__ == "__main__":
    main()
