# DS4.1 per-expert bitrate inside a layer (hot-3 / cold-2)

A routed-expert layer (384 experts, top-6) may mix trellis bitrates, for example the hot experts of the layer at
3 bits and the cold ones at 2 bits (~2.5 bpw average). Code: `exllamav3/modules/moe_mixedk.py`, wired into
`BlockSparseMLP.load_local` / `forward`. Tests: `tests/test_moe_mixedk_cpu.py`.

## Loading and storage

The loader reads K per tensor from the trellis shape (`K = trellis.shape[-1] / 16`), as before. A layer whose
experts are not all the same `(K_gate = K_up, K_down)` takes the mixed route:

* experts are grouped by `(K_gu, K_down)`, ascending, ascending by expert id inside a group;
* each group owns one `MultiLinear` triple (nine contiguous int64 pointer tables, its "stack");
* `kmap[e]` (group), `slot[e]` (position in the group's tables) and `lut[g, e]` (`slot[e]` or `-1`) live on the device;
* the experts' tensors are not copied or repacked. `MultiLinear` still asserts one K per table; only the store
  builds several.

A layer where every expert shares one K does not build a store: it is the existing single-K code, unchanged.

Gate and up of one expert must share K (they run in one grouped launch). The decode kernel also needs
`K_down == K_gate_up`; prefill takes both. A layer the kernels cannot take (K outside 2/2.5/3/4/5/6, bias, mcg
codebook for decode, `tp_mode`, CPU split...) falls to the dense per-expert path, which is K-agnostic.
For a mixed layer `support_quant_paths`, `support_bc_bszn`, `support_dec_moe`, `support_fused`, the BC graph and
the batched-reconstruct tier are off (they are single-K).

## Decode (rows 1..16)

`exl3_moe_gfx12_k3` already zero-fills picks outside `[0, experts)`. Group g runs on `lut[g][selected]`
(one device gather; foreign picks become `-1`), same weights tensor, its own tables and its own K. One kernel set
per K group, no host sync. The fp32 group outputs are added in ascending `(K_gu, K_down)` order.

Notes: every group launches on every call (whether a token picked it is only known on the device), a foreign
pick costs an early-out block, so a two-group layer is about 9 extra tiny launches. Each group's partial sum is in
the kernel's expert-sorted order and foreign picks add exact `+0.0`; a token whose experts span both groups sums
`partial_K2 + partial_K3`, not one expert-ordered chain (differs from a hypothetical single-K sum by fp32
rounding only). A one-group layer is the legacy call argument for argument (bit-identical, tested).

## Prefill (`exl3_moe_prefill_wmma`)

That kernel has no masked picks (its reduce reads `svh_table[e]` for every pick), so groups are compacted:
picks are stably sorted by group, each group's picks become `top_k = 1` pseudo rows (token rows gathered once),
one launch set per group with its K, and the per-pick results are summed per token in routing-slot order (the order
of the kernel's own reduce) into one fp32 buffer; a shared-expert fp16 output is added last. Cost: one host read per
mixed layer and chunk (the group sizes), one `[rows * top_k, hidden]` fp32 buffer, and the GEMM work equals a
single-K layer's. Chunks above `EXL3_MOE_ALIGN_CHUNK` (4096) rows are processed chunk by chunk. A masked-pick
guard in the prefill kernels (gather, act, reduce) would remove the sync and the buffer; not done (needs a rebuild
and the device).

## Pack config: `quantization_config.expert_bits`

The converter writes, next to the per-tensor `bits_per_weight` in `quantization_config.json` (and the
`quantization_config` of `config.json`):

```json
"expert_bits": {
    "layers.3.ffn": [3, 2, 2, 3, 2, ...],    // one integer per routed expert, index = expert id, length n_routed_experts (384)
    "layers.4.ffn": [2, 2, 3, ...]
}
```

* key: module path of the MoE block as in the tensor names (`layers.<L>.ffn`); backbone routed experts only (the MTP
  experts are uniform and may be omitted); a layer that is absent means "read K from the tensors";
* value: the EXL3 `bits_per_weight` (K) of the expert; `w1`, `w2`, `w3` of one expert always carry the same bits;
* the tensors stay authoritative: the loader compares the field with the K it found and refuses to load on a
  mismatch (`check_expert_bits`); it is also what a planner can read to size a layer before loading it.

Helpers in `moe_mixedk.py`: `expert_bits_from_plan(plan["entries"])` (converter side, plan entries of
`tools/ds41/plan.py`), `parse_expert_bits`, `check_expert_bits`.

Converter change (branch `feature/ds41-convert`, `tools/ds41/stream_convert.py:finalize`): add to `qcfg` before it is
written

```python
qcfg["expert_bits"] = expert_bits_from_plan(plan["entries"])   # {"layers.L.ffn": [bits per expert]}
```

`tools/ds41kern/converter_expert_bits.patch` is that change (and the standalone `expert_bits_from_plan`) as a patch
against `70639f0`; this branch does not touch `tools/ds41`.
