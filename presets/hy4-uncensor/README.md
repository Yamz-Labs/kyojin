# Hy4-flash: decensor preset (opt-in)

A run-time edit of the Hy4 forward. It edits no weights and no quantised data. Off by default; `HY4_ABLIT_SPEC=on` switches it on.

```
HY4_ABLIT_SPEC=on python tools/hy4/serve.py --model ./hy4/Hy4-flash-Q3_K-00001-of-00002.gguf ...
```

`edit_spec.safetensors` must sit next to the JSON. Unset, `HY4_ABLIT_SPEC=0`, an empty value or `off` runs the unmodified forward and installs no edit; `HY4_ABLIT_SPEC=<path>/edit_spec.json` loads another spec. Validated in no_think mode; in the default thinking mode some answers run long (2 of 17 hit the 4096-token cap in our check), a fix is in progress.

What it does: after the attention output projection and after the post-norm MLP / MoE output of the 42 trunk layers, the output `y` becomes `y - w * ((r . y) - mu) * r`, with a per-layer unit direction `r` and per-layer weights `w`, `mu`. Layers with `w = 0` are not touched. The MTP draft block is never edited. Code: `exllamav3/loader/ablit_hy4.py`; the post-norm MLP site is fused into the residual kernel (`exllamav3/loader/hc_fused.py`).

Development-build numbers (Strix Halo box, Q3_K, same settings with the spec on and off); to be replaced by the release-engine gate results before publication:

| | spec off | spec on |
|---|---|---|
| Fixed gate set of 17 requests, answers judged as declined | 17 | 4 |
| GSM, 30 problems | 28 | 29 |
| Wikitext-2 slice, change in NLL | 0 | -0.046 (inside the noise band) |
| Decode speed | reference | +1.4 % (inside noise) |

Whoever uses a spec is responsible for it. SHA256SUMS lists the two files.
