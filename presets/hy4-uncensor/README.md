# Hy4-flash: decensor preset (on by default)

A run-time edit of the Hy4 forward. It edits no weights and no quantised data. The server loads it by default; `HY4_ABLIT_SPEC=0` switches it off.

```
HY4_ABLIT_SPEC=0 python tools/hy4/serve.py --model ./hy4/Hy4-flash-Q3_K-00001-of-00002.gguf ...
```

`edit_spec.safetensors` must sit next to the JSON. `HY4_ABLIT_SPEC=0`, an empty value or `off` runs the unmodified forward and installs no edit; `HY4_ABLIT_SPEC=<path>/edit_spec.json` loads another spec.

What it does: after the attention output projection and after the post-norm MLP / MoE output of the 42 trunk layers, the output `y` becomes `y - w * ((r . y) - mu) * r`, with a per-layer unit direction `r` and per-layer weights `w`, `mu`. Layers with `w = 0` are not touched. The MTP draft block is never edited. Code: `exllamav3/loader/ablit_hy4.py`; the post-norm MLP site is fused into the residual kernel (`exllamav3/loader/hc_fused.py`).

Development-build numbers (Strix Halo box, Q3_K, same settings with the spec on and off); to be replaced by the release-engine gate results before publication:

| | spec off | spec on |
|---|---|---|
| Fixed gate set of 17 requests, answers judged as declined | 17 | 4 |
| GSM, 30 problems | 28 | 29 |
| Wikitext-2 slice, change in NLL | 0 | -0.046 (inside the noise band) |
| Decode speed | reference | +1.4 % (inside noise) |

Whoever uses a spec is responsible for it. SHA256SUMS lists the two files.
