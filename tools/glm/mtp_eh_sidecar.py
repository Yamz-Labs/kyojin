"""Extract the unquantized GLM MTP eh_proj weight into a sidecar for EXL3_MTP_EH_FP16.

usage: python tools/glm/mtp_eh_sidecar.py <hf checkpoint dir> <out.safetensors>
Reads model.safetensors.index.json, copies the single "*.layers.<n>.eh_proj.weight" tensor
(BF16 in the GLM-5.3 FP8 release) as-is. ~64 MiB.
"""
import json, os, sys
from safetensors import safe_open
from safetensors.torch import save_file

src, out = sys.argv[1], sys.argv[2]
wm = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
keys = [k for k in wm if k.endswith("eh_proj.weight")]
assert len(keys) == 1, keys
with safe_open(os.path.join(src, wm[keys[0]]), "pt") as f:
    w = f.get_tensor(keys[0])
assert w.dtype.is_floating_point and w.element_size() == 2, f"{keys[0]}: {w.dtype} (fp8 source needs dequant)"
save_file({keys[0]: w.contiguous()}, out)
print(keys[0], tuple(w.shape), w.dtype, "->", out)
