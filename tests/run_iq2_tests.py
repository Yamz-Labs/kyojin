"""gap_formats.dequant for q3_k / iq2_xs / iq2_s vs the kyojin-mlx reference (bit-equal to ggml) on real rows (iq2_ref.npz)."""
import numpy as np, torch
from exllamav3.foreign_quant.gap_formats import dequant

z = np.load("/home/yaniss/hermes-work/hipfp8/tests/iq2_ref.npz")
ok = bad = 0
for t, f in (("Q3_K", "q3_k"), ("IQ2_XS", "iq2_xs"), ("IQ2_S", "iq2_s")):
    raw = torch.from_numpy(z[t + "_raw"]).cuda()
    ref = z[t + "_ref"].astype(np.float32)
    out = dequant(f, raw).float().cpu().numpy()
    refh = ref.astype(np.float16).astype(np.float32)          # gap dequant returns fp16
    err = np.abs(out - refh).max()
    if out.shape == ref.shape and err <= 1e-3 * np.abs(ref).max():
        ok += 1
        print(f"{t}: equal (max |diff| {err:.2e}, |ref| max {np.abs(ref).max():.3f})")
    else:
        bad += 1
        print(f"{t}: DIFF shape {out.shape} vs {ref.shape} max {err:.3e}; first {out[0, :8]} ref {ref[0, :8]}")
print(f"{ok} passed, {bad} failed")
