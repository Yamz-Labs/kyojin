#!/usr/bin/env python3
"""export_spec.py -- CPU: build an ablit5 "arm A" edit_spec (single projected direction) from
collected directions + kernel parameters. No model, no GPU.

  export_spec.py --dirs DIR --arch {auto,glm,mimo_v2} [--model DIR] --kernel params.json --out OUTDIR
    params.json: {"attn": {max_w, peak, min_w, width}, "mlp": {...}}  (an existing edit_spec.json works)
    --scale-kernel: rescale peak/width of a GLM-tuned kernel by relative depth (non-GLM archs)

Writes OUTDIR/edit_spec.json + edit_spec.safetensors ({r}), same layout the runtime hook
(EXL3_ABLIT_RUNTIME) and search_ablit5 use. Direction = interp(R, proj_f) minus the harmless-mean
component at pca_layer, renormalised (search_ablit5 arm A).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arch as arch_mod  # noqa: E402
from hook_ablate import build_single_edit, projected_direction  # noqa: E402


def export(dirs, out, kernel, arch="auto", model=None, scale_kernel=False, tag="provisional", proj_f=None,
           strength=1.0):
    """Write OUT/edit_spec.json + edit_spec.safetensors. strength multiplies both kernels' max_w and min_w
    (1.0 = the kernel as given); proj_f overrides the interpolation index of the direction."""
    import numpy as np
    from safetensors.numpy import load_file, save_file
    A = arch_mod.get(arch, model)
    n = A["n_layers"]
    st = load_file(os.path.join(dirs, "directions.safetensors"))
    R = np.stack([st[f"L{i:02d}.r"].astype(np.float64) for i in range(n)])
    assert R.shape == (n, A["hidden"]), (R.shape, n, A["hidden"])
    mu = st[f"L{A['pca_layer']:02d}.mean_safe"].astype(np.float64)
    pf = A["proj_f"] if proj_f is None else proj_f
    v = projected_direction(R, pf, mu)
    k = json.load(open(kernel))
    attn, mlp = dict(k["attn"]), dict(k["mlp"])
    if scale_kernel:
        for p in (attn, mlp):
            p["peak"] *= A["scale"]
            p["width"] *= A["scale"]
    attn = {x: attn[x] for x in ("max_w", "peak", "min_w", "width")}
    mlp = {x: mlp[x] for x in ("max_w", "peak", "min_w", "width")}
    for p in (attn, mlp):
        p["max_w"] *= strength
        p["min_w"] *= strength
    e = build_single_edit(v, attn, mlp, n_layers=n)
    os.makedirs(out, exist_ok=True)
    spec = {"model": model or "", "hidden": A["hidden"], "n_layers": n, "arm": "A",
            "readout": (f"interp r f={pf:.2f} minus harmless-mean(L{A['pca_layer']}) "
                        f"component, renormalised; single dir all layers"),
            "candidate": tag, "direction_mode": "single-projected",
            "direction_index": pf, "strength": strength, "attn": attn, "mlp": mlp,
            "attn_w": [float(x) for x in e["attn_w"]],
            "mlp_w": [float(x) for x in e["mlp_w"]],
            "norm_preserve": False,
            "apply": ("W_writer <- W - w_comp(L) * r_hat r_hat^T W, fp32 workspace; writers = "
                      + A["writers"])}
    json.dump(spec, open(os.path.join(out, "edit_spec.json"), "w"), indent=1)
    save_file({"r": v.astype(np.float32)}, os.path.join(out, "edit_spec.safetensors"))
    print(f"wrote {out}/edit_spec.json + .safetensors  arch={A['name']} n_layers={n} "
          f"nonzero attn={sum(1 for x in spec['attn_w'] if x)} mlp={sum(1 for x in spec['mlp_w'] if x)}")
    return spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", required=True)
    ap.add_argument("--arch", default="auto", choices=["auto"] + list(arch_mod.ARCHS))
    ap.add_argument("--model", default=None)
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--scale-kernel", action="store_true")
    ap.add_argument("--tag", default="provisional")
    ap.add_argument("--proj-f", type=float, default=None)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    export(a.dirs, a.out, a.kernel, a.arch, a.model, a.scale_kernel, a.tag, a.proj_f, a.strength)


if __name__ == "__main__":
    main()
