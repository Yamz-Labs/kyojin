"""Tiny random Inkling2 twin GGUF (same arch / tensor names / conventions as the real file, hidden 64) for CPU conformance tests.

  python make_twin.py OUT.gguf --src REAL.gguf [--loops 3] [--seed 0]

Config: 8 physical layers, attn-res block 2, core 2..6 (span 4), num_loops (default 3) -> 8 + 4 (loops - 1) applications, global layers
phys % 4 == 3, window / relative extent 8, 8 experts top 2, clips low enough that they bite. Tokenizer KVs come from --src (the llama.cpp
loader checks the vocabulary size). Mixed storage on purpose: Q8_0 attention on odd layers (F32 on even), NVFP4 experts (+ per-expert .scale)
on layers 0..5, BF16 experts on 6..7, BF16 embedding, Q8_0 head (--f32: everything F32). gguf-py comes from the llama.cpp clone (INK_GGUFPY, default ~/scratch/inkling_llama/gguf-py).
"""
import argparse
import os
import sys

import numpy as np

H = os.environ.get("INK_H", os.path.expanduser("~"))
sys.path.insert(0, os.environ.get("INK_GGUFPY", f"{H}/scratch/inkling_llama/gguf-py"))
import gguf  # noqa: E402
from gguf import GGMLQuantizationType as Q  # noqa: E402

D, NH, NKV, HD, NREL, NCONV, E, TOPK, FF = 64, 4, 2, 16, 4, 4, 8, 2, 64
NPHYS, BLK, CORE0, CORE1, WINDOW, RANK = 8, 2, 2, 6, 8, 8
VOCAB_PAD, VOCAB = 201024, 200058


def build(out, src, loops, seed, f32_only=False):
    rng = np.random.default_rng(seed)
    n_extra = (loops - 1) * (CORE1 - CORE0) if loops > 1 else 0
    n_app = NPHYS + n_extra

    def layer_of(il):
        if loops < 2 or il < CORE0:
            return il
        if il < CORE0 + loops * (CORE1 - CORE0):
            return CORE0 + (il - CORE0) % (CORE1 - CORE0)
        return il - n_extra

    phys_swa = [i % 4 != 3 for i in range(NPHYS)]
    pattern = [phys_swa[layer_of(il)] for il in range(n_app)]
    rd = gguf.GGUFReader(src)
    w = gguf.GGUFWriter(out, "inkling2")
    for k, f in rd.fields.items():
        if k.startswith("tokenizer.") or k in ("general.quantization_version",):
            w.add_key_value(k, f.contents(), f.types[0], f.types[-1] if f.types[0] == gguf.GGUFValueType.ARRAY else None)
    w.add_name("inkling2-twin")
    P = "inkling2."
    w.add_uint32(P + "block_count", n_app)
    w.add_uint32(P + "context_length", 4096)
    w.add_uint32(P + "embedding_length", D)
    w.add_uint32(P + "feed_forward_length", FF)
    w.add_array(P + "attention.head_count", [NH] * n_app)
    w.add_array(P + "attention.head_count_kv", [NKV] * n_app)
    w.add_float32(P + "attention.layer_norm_rms_epsilon", 1e-6)
    w.add_uint32(P + "expert_count", E)
    w.add_uint32(P + "expert_used_count", TOPK)
    w.add_uint32(P + "attention.key_length", HD)
    w.add_uint32(P + "attention.value_length", HD)
    w.add_uint32(P + "attention.key_length_swa", HD)
    w.add_uint32(P + "attention.value_length_swa", HD)
    w.add_uint32(P + "expert_feed_forward_length", FF)
    w.add_uint32(P + "rope.dimension_count", 0)
    w.add_uint32(P + "vocab_size", VOCAB_PAD)
    w.add_uint32(P + "vocab_size_unpadded", VOCAB)
    w.add_uint32(P + "attention.sliding_window", WINDOW)
    w.add_array(P + "attention.sliding_window_pattern", pattern)
    w.add_uint32(P + "shortconv.l_cache", NCONV)
    w.add_float32(P + "expert_weights_scale", 2.5)
    w.add_float32(P + "activation.situ_beta", 4.0)
    w.add_float32(P + "activation.situ_linear_beta", 25.0)
    w.add_uint32(P + "attn_res.block_size", BLK)
    w.add_float32(P + "final_logit_softcapping", 30.0)
    w.add_uint32(P + "attention.relative_length", NREL)
    w.add_uint32(P + "attention.relative_extent", WINDOW)
    w.add_float32(P + "attention.qk_clip", 3.0)
    w.add_float32(P + "attention.relative_clip", 1.5)
    w.add_uint32(P + "attention.log_scale_floor", 8)
    w.add_float32(P + "attention.log_scale_alpha", 0.5)
    if loops > 1:
        w.add_uint32(P + "num_loops", loops)
        w.add_uint32(P + "loop.core_start", CORE0)
        w.add_uint32(P + "loop.core_end", CORE1)
        w.add_uint32(P + "loop.gate_rank", RANK)

    def f32(*shape_ggml, scale=1.0, kind="n"):
        shp = tuple(reversed(shape_ggml))
        a = rng.standard_normal(shp).astype(np.float32) * scale if kind == "n" else np.ones(shp, np.float32)
        return a

    def put(name, arr, qt=None):
        if f32_only:
            qt = None
        if qt is None or qt == Q.F32:
            w.add_tensor(name, arr.astype(np.float32))
        elif qt == Q.BF16:
            u = np.ascontiguousarray(arr.astype(np.float32)).view(np.uint32)
            w.add_tensor(name, (u >> 16).astype(np.uint16).view(np.uint8).reshape(*arr.shape[:-1], arr.shape[-1] * 2), raw_dtype=Q.BF16)
        else:
            w.add_tensor(name, gguf.quants.quantize(arr.astype(np.float32), qt), raw_dtype=qt)

    def lin(name, k_in, n_out, qt):
        put(name, f32(k_in, n_out, scale=1.0 / np.sqrt(k_in)), qt)

    put("token_embd.weight", f32(D, VOCAB_PAD, scale=1.0), Q.BF16)
    put("token_embd_norm.weight", 1.0 + 0.1 * f32(D))
    put("output.weight", f32(D, VOCAB_PAD, scale=0.5), Q.Q8_0)
    put("output_norm.weight", 1.0 + 0.1 * f32(D))
    put("output_res_query.weight", f32(D, scale=0.5))
    for i in range(NPHYS):
        L = f"blk.{i}"
        swa = phys_swa[i]
        aq = Q.Q8_0 if i % 2 else None
        lin(f"{L}.attn_q.weight", D, NH * HD, aq)
        lin(f"{L}.attn_k.weight", D, NKV * HD, aq)
        lin(f"{L}.attn_v.weight", D, NKV * HD, aq)
        lin(f"{L}.attn_output.weight", NH * HD, D, aq)
        lin(f"{L}.attn_gate.weight", D, NH * HD, Q.BF16 if i % 3 == 0 else None)
        put(f"{L}.attn_q_clip.weight", 1.0 + 0.3 * f32(HD, NH))
        put(f"{L}.attn_sink_proj.weight", f32(D, NH, scale=1.0 / np.sqrt(D)))
        lin(f"{L}.attn_rel.weight", D, NREL * NH, None)
        put(f"{L}.attn_rel_bias.weight", f32(NREL, WINDOW, scale=1.0))
        put(f"{L}.attn_rel_clip.weight", 1.0 + 0.3 * f32(NREL, NH))
        put(f"{L}.attn_k_conv.weight", f32(NCONV, NKV * HD, scale=0.5))
        put(f"{L}.attn_v_conv.weight", f32(NCONV, NKV * HD, scale=0.5))
        put(f"{L}.attn_post_conv.weight", f32(NCONV, D, scale=0.5))
        put(f"{L}.ffn_post_conv.weight", f32(NCONV, D, scale=0.5))
        if swa:
            put(f"{L}.attn_norm.weight", 1.0 + 0.1 * f32(D))
        else:
            for c in "qkv":
                put(f"{L}.attn_{c}_inp_norm.weight", 1.0 + 0.1 * f32(D))
            put(f"{L}.attn_k_conv_proj.weight", f32(D, NCONV * NKV, scale=0.3 / np.sqrt(D)), Q.BF16)
            put(f"{L}.attn_v_conv_proj.weight", f32(D, NCONV * NKV, scale=0.3 / np.sqrt(D)), Q.BF16)
        if i > 0:
            put(f"{L}.attn_res_query.weight", f32(D, 1 if swa else 3, scale=0.7) if not swa else f32(D, scale=0.7))
        put(f"{L}.ffn_res_query.weight", f32(D, scale=0.7))
        put(f"{L}.ffn_norm.weight", 1.0 + 0.1 * f32(D))
        put(f"{L}.ffn_gate_inp.weight", f32(D, E, scale=1.0 / np.sqrt(D)))
        put(f"{L}.exp_probs_b.bias", 0.1 * f32(E))
        if i < 6 and not f32_only:
            for nm, k_in, n_out in (("ffn_gate_up_exps", D, 2 * FF), ("ffn_down_exps", FF, D)):
                nb = k_in // 64 * 36
                raw = rng.integers(0, 256, size=(E, n_out, nb), dtype=np.uint8)
                blk = raw.reshape(E, n_out, k_in // 64, 36)
                blk[..., 0:4] = rng.integers(0x18, 0x30, size=blk[..., 0:4].shape, dtype=np.uint8)     # UE4M3 scale bytes 0x18..0x2f
                w.add_tensor(f"{L}.{nm}.weight", blk.reshape(E, n_out, nb), raw_dtype=Q.NVFP4)
                put(f"{L}.{nm}.scale", (0.5 + rng.random(E)).astype(np.float32))
        else:
            put(f"{L}.ffn_gate_up_exps.weight", f32(D, 2 * FF, E, scale=1.0 / np.sqrt(D)), Q.BF16)
            put(f"{L}.ffn_down_exps.weight", f32(FF, D, E, scale=1.0 / np.sqrt(FF)), Q.BF16)
        if loops > 1 and CORE0 <= i < CORE1:
            for m in ("attn", "k", "v", "mlp"):
                put(f"{L}.loop_{m}_down.weight", f32(2 * D, RANK, scale=1.0 / np.sqrt(2 * D)))
                put(f"{L}.loop_{m}_gain.weight", f32(RANK, D, scale=1.0 / np.sqrt(RANK)), Q.BF16)
                put(f"{L}.loop_{m}_forget.weight", f32(RANK, D, scale=1.0 / np.sqrt(RANK)))
    if loops > 1:
        put("loop_res_query.weight", f32(D, 2 * (CORE1 - CORE0), scale=0.7))
        put("loop_kv_res_query.weight", f32(D, 2 * (CORE1 - CORE0), scale=0.7))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print("wrote", out, "apps", n_app, "pattern", "".join("S" if s else "G" for s in pattern))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--src", default=f"{H}/models/inkling2-mini/gguf/Inkling-2-Mini-UD-Q4_K_XL.gguf")
    ap.add_argument("--loops", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--f32", action="store_true", help="every tensor F32 (no NVFP4 / Q8_0 / BF16): exact oracle, llama.cpp CPU quantises activations for the quantised types")
    a = ap.parse_args()
    build(a.out, a.src, a.loops, a.seed, a.f32)
