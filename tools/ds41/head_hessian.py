"""Calibration Hessian of the DS4.1 output head from the streamed row state left behind after the last decoder layer.

    py.sh tools/ds41/head_hessian.py --state <work>/ckpt --config config.json --map tensor_map.json \
        --spool <spool dir> [--unit global] [--rows N] [--device cpu|cuda:0]

Input: `rowstate.bf16.mmap` (rows, L, hc_mult, hidden) after layer num_hidden_layers-1 (rowstate.json next_layer ==
num_hidden_layers) and the last block's pending hyper-connection pre-mix `ho["pre_mix"][n-1]`, which streamcal keeps in
`ho/L<n-1>/row<i>.pt`. The head input is EXACTLY the model's tail (model._forward):

    x = ops.rmsnorm(ops.hc_pre(h, pre_mix[n-1]), norm_w, rms_norm_eps)     # then logits = head_w @ x

H = sum over all tokens of x^T x (not normalized), accumulated per chunk in fp32 then summed in fp64, written as
`<spool>/<unit>/lin.head.h` (split.pack_hessian, tri_f32), the name HessQuantizer looks up for the plan entry `head`.
`--x-dtype bf16` rounds x to bf16 first (what serving feeds the head); default fp32 = what the capture uses for every
other linear. Non-finite tokens are dropped and counted like calib.h_data_add.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def row_pre_mix(state, i, layer):
    """Pending pre-mix [1, L, hc] fp32 of block `layer`, from the row's hand-off file."""
    p = os.path.join(state, "ho", f"L{layer}", f"row{i}.pt")
    try:
        ho = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
    except (RuntimeError, ValueError):
        ho = torch.load(p, map_location="cpu", weights_only=False)
    pm = ho["pre_mix"][layer]
    return pm.float().reshape(1, -1, pm.shape[-1])


def head_inputs(state, cfg, norm_w, rows, device, group=4, x_dtype="fp32", log=print):
    """Yield x [group*L, hidden] fp32 (the head input) for consecutive groups of rows."""
    from exllamav3.dsv41 import ops
    n = cfg.num_hidden_layers
    pr = json.load(open(os.path.join(state, "rowstate.json")))
    assert pr["next_layer"] == n and not pr.get("commit"), f"row state is at next_layer {pr['next_layer']}, need {n}"
    L = pr["L"]
    bf = os.path.exists(os.path.join(state, "rowstate.bf16.mmap"))
    path = os.path.join(state, "rowstate.bf16.mmap" if bf else "rowstate.fp32.mmap")
    shape_all = (pr["rows"], L, cfg.hc_mult, cfg.hidden_size)
    sz = os.path.getsize(path)
    per_row = L * cfg.hc_mult * cfg.hidden_size * (2 if bf else 4)
    assert sz >= pr["rows"] * per_row, f"row state file too small ({sz} < {pr['rows'] * per_row})"
    n_rows = pr["rows"] if rows is None else min(rows, pr["rows"])
    mm = np.memmap(path, dtype=np.uint16 if bf else np.float32, mode="r", shape=shape_all)
    nw = norm_w.float().to(device)
    for a in range(0, n_rows, group):
        b = min(n_rows, a + group)
        h = torch.from_numpy(np.array(mm[a:b]))                                                       # [r, L, hc, d]
        h = (h.view(torch.bfloat16) if bf else h).to(device).float()
        pm = torch.cat([row_pre_mix(state, i, n - 1) for i in range(a, b)]).to(device)                # [r, L, hc]
        x = ops.rmsnorm(ops.hc_pre(h, pm), nw, cfg.rms_norm_eps)
        if x_dtype == "bf16":
            x = x.to(torch.bfloat16)
        yield x.reshape(-1, cfg.hidden_size).float()


def accumulate(xs, n, device, log=print):
    """H (fp64, [n, n]) = sum x^T x over the finite rows of the chunks `xs`; also count / num_total / inf_nan."""
    H = torch.zeros(n, n, dtype=torch.float64, device=device)
    count = num_total = 0
    inf_nan = [0, 0]
    t0, done = time.time(), 0
    for x in xs:
        num_total += x.numel()
        inf_nan[0] += int(torch.isinf(x).sum())
        inf_nan[1] += int(torch.isnan(x).sum())
        fin = torch.isfinite(x).all(dim=1)
        if not bool(fin.all()):
            x = x[fin]
        if x.shape[0]:
            H += (x.T @ x).double()
        count += x.shape[0]
        done += 1
        if done % 10 == 0:
            log(f"  {count} tokens, {time.time() - t0:.0f}s")
    return H, count, num_total, inf_nan


def write_spool(spool_root, unit, H, count, num_total, inf_nan, name="lin.head", group=None, done=True):
    import split as SP
    sp = SP.Spool(spool_root)
    blob = SP.pack_hessian(H.float(), count, num_total, inf_nan, group or f"{unit}.head")
    sp.put(unit, name, blob)
    if done:
        sp.mark_capture_done(unit)
    return sp.path(unit, name), len(blob)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", required=True, help="ckpt dir holding rowstate.json, rowstate.bf16.mmap, ho/")
    ap.add_argument("--config", required=True, help="model config.json (DSV41Config)")
    ap.add_argument("--map", help="tensor_map.json; the final norm weight 'norm.weight' is fetched through it "
                                  "(DS41_LOCAL_DIR = local shard dir avoids HTTP)")
    ap.add_argument("--norm-file", help="alternative to --map: torch .pt file with the [hidden] norm weight")
    ap.add_argument("--spool", required=True)
    ap.add_argument("--unit", default="global")
    ap.add_argument("--name", default="lin.head", help="spool entry name (lin.<plan key>)")
    ap.add_argument("--rows", type=int, help="use only the first N rows (default: all in the state)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--group", type=int, default=4, help="rows per GEMM chunk")
    ap.add_argument("--x-dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--expect-tokens", type=int, help="fail (nothing written) unless exactly this many finite tokens")
    ap.add_argument("--no-capture-done", action="store_true", help="do not write CAPTURE_DONE in the unit's spool dir")
    a = ap.parse_args()
    from exllamav3.dsv41.config import DSV41Config
    cfg = DSV41Config.from_file(a.config)
    if a.norm_file:
        norm_w = torch.load(a.norm_file)
    else:
        import fetch as F
        norm_w = F.TensorFetcher(json.load(open(a.map))).fetch("norm.weight")
    assert norm_w.shape == (cfg.hidden_size,), norm_w.shape
    t0 = time.time()
    xs = head_inputs(a.state, cfg, norm_w, a.rows, torch.device(a.device), a.group, a.x_dtype)
    H, count, num_total, inf_nan = accumulate(xs, cfg.hidden_size, torch.device(a.device), print)
    print(f"head Hessian: {count} tokens (non-finite dropped: {num_total // cfg.hidden_size - count}), "
          f"{time.time() - t0:.0f}s")
    assert a.expect_tokens is None or count == a.expect_tokens, f"{count} tokens, expected {a.expect_tokens}"
    assert inf_nan == [0, 0], f"non-finite head inputs: {inf_nan}"
    p, nbytes = write_spool(a.spool, a.unit, H.cpu(), count, num_total, inf_nan, a.name, done=not a.no_capture_done)
    print(f"wrote {p} ({nbytes / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
