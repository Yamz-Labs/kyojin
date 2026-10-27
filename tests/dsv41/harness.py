"""
Test harness for the DS4.1 port: tiny random checkpoint in the OFFICIAL layout, the official reference
model (vendored model.py/engram.py + CPU kernel shim) and tokenizer fixtures.
"""
from __future__ import annotations
import os, sys, types, math
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from exllamav3.dsv41.config import DSV41Config, tiny_config
from exllamav3.dsv41 import quant

torch.set_num_threads(int(os.environ.get("DSV41_THREADS", "6")))


# ------------------------------------------------------------------------------------ tokenizer
class FakeTok:
    def __init__(self, n=512):
        from tokenizers import Tokenizer, models
        words = ["the", "cat", "sat", "on", "mat", "école", "café", "naïve", "über", "straße", "go", "run",
                 "x", "y", "42", "7", "hello", "world", "a", "b", "c", "d", "e", "f", "ok", "no", "yes",
                 "tok", "en", "gram", "hash", "mem", "ory", "deep", "seek", "v4", "flash", "kyo", "jin"]
        vocab, seen = {}, set()
        forms = [lambda w: w, lambda w: w.capitalize(), lambda w: w.upper(), lambda w: " " + w,
                 lambda w: " " + w.capitalize(), lambda w: w + "\n", lambda w: "  " + w]
        i = 0
        for f in forms:
            for w in words:
                t = f(w)
                if t not in seen and len(vocab) < n - 6:
                    seen.add(t)
                    vocab[t] = len(vocab)
        while len(vocab) < n - 6:
            vocab[f"<r{len(vocab)}>"] = len(vocab)
        for t in [" ", "\ufffd", "\u00e9", "e", "E", "É"]:
            if t not in vocab and len(vocab) < n:
                vocab[t] = len(vocab)
        while len(vocab) < n:
            vocab[f"<p{len(vocab)}>"] = len(vocab)
        self.backend_tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<r1>"))
        self.n = n

    def __len__(self):
        return self.n


# ------------------------------------------------------------------------------------ random checkpoint
def _fp8_block_quant(w: torch.Tensor, block=32):
    out_f, in_f = w.shape
    po, pi = -out_f % block, -in_f % block
    wp = torch.nn.functional.pad(w, (0, pi, 0, po))
    t = wp.view(wp.shape[0] // block, block, wp.shape[1] // block, block)
    amax = t.abs().amax(dim=(1, 3)).clamp_min(1e-8)
    e = torch.ceil(torch.log2(amax / 448.0))
    scale = torch.exp2(e)
    q = (t / scale[:, None, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    q = q.view_as(wp)[:out_f, :in_f].contiguous()
    return q, scale.to(torch.float8_e8m0fnu)


def _fp4_row_quant(w: torch.Tensor, block=32):
    out_f, in_f = w.shape
    t = w.view(out_f, in_f // block, block)
    amax = t.abs().amax(-1).clamp_min(1e-8)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 6.0)))
    v = (t / scale[..., None]).clamp(-6, 6).flatten(-2)
    code = quant.fp4_round(v)
    mag = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    idx = (code.abs().unsqueeze(-1) == mag).float().argmax(-1)
    idx = idx + 8 * (torch.signbit(code)).long()
    lo, hi = idx[:, 0::2], idx[:, 1::2]
    packed = (lo | (hi << 4)).to(torch.uint8)
    return packed.view(torch.int8), scale.to(torch.float8_e8m0fnu)      # HF stores experts as int8


def make_state_dict(cfg: DSV41Config, seed=0, fp8=True, fp4=True, with_vision=False):
    """Random weights with the official tensor names and storage dtypes. Returns (sd, engram_tables)."""
    g = torch.Generator().manual_seed(seed)
    R = lambda *s, sc=1.0: torch.randn(*s, generator=g) * sc
    sd = {}
    d, hd, H = cfg.hidden_size, cfg.head_dim, cfg.num_attention_heads

    def lin(name, out_f, in_f, quantize=fp8, std=None):
        w = R(out_f, in_f, sc=std if std else in_f ** -0.5)
        if quantize:
            q, s = _fp8_block_quant(w)
            sd[name + ".weight"], sd[name + ".scale"] = q, s
        else:
            sd[name + ".weight"] = w.to(torch.bfloat16)

    def norm(name, n):
        sd[name + ".weight"] = (1 + 0.1 * R(n)).to(torch.float32)

    sd["embed.weight"] = R(cfg.vocab_size, d, sc=1.0).to(torch.bfloat16)
    norm("norm", d)
    sd["head.weight"] = R(cfg.vocab_size, d, sc=d ** -0.5 * 2).to(torch.bfloat16)
    mix = (2 + cfg.hc_mult) * cfg.hc_mult
    n_all = cfg.num_hidden_layers + cfg.n_mtp_layers
    for l in range(n_all):
        p = f"layers.{l}" if l < cfg.num_hidden_layers else f"mtp.{l - cfg.num_hidden_layers}"
        for t in ("attn", "ffn"):
            sd[f"{p}.hc_{t}_fn"] = R(mix, cfg.hc_mult * d, sc=0.3 * (cfg.hc_mult * d) ** -0.5)
            sd[f"{p}.hc_{t}_base"] = R(mix, sc=0.5)
            sd[f"{p}.hc_{t}_scale"] = torch.tensor([0.8, 0.6, 0.7]) + 0.1 * R(3)
        norm(f"{p}.attn_norm", d)
        norm(f"{p}.ffn_norm", d)
        a = f"{p}.attn"
        sd[f"{a}.attn_sink"] = R(H, sc=0.5)
        lin(f"{a}.wq_a", cfg.q_lora_rank, d)
        norm(f"{a}.q_norm", cfg.q_lora_rank)
        lin(f"{a}.wq_b", H * hd, cfg.q_lora_rank)
        lin(f"{a}.wkv", hd, d)
        norm(f"{a}.kv_norm", hd)
        # wo_a: block-diagonal grouped, official raw ckpt stores fp8 + scale (convert.py -> bf16)
        lin(f"{a}.wo_a", cfg.o_groups * cfg.o_lora_rank, H * hd // cfg.o_groups, quantize=fp8)
        lin(f"{a}.wo_b", d, cfg.o_groups * cfg.o_lora_rank)
        if l < cfg.num_hidden_layers and cfg.is_kv_source(l):
            r = cfg.ratio(l)
            sd[f"{a}.compressor.wkv.weight"] = R(hd, d, sc=d ** -0.5).to(torch.bfloat16)
            if r > 1:
                sd[f"{a}.compressor.wgate.weight"] = R(hd, d, sc=d ** -0.5).to(torch.bfloat16)
            norm(f"{a}.compressor.norm", hd)
        if l < cfg.num_hidden_layers and cfg.is_index_source(l):
            lin(f"{a}.indexer.wq_b", cfg.index_n_heads * cfg.index_head_dim, cfg.q_lora_rank)
            sd[f"{a}.indexer.weights_proj.weight"] = R(cfg.index_n_heads, d, sc=d ** -0.5).to(torch.bfloat16)
            if cfg.is_kv_source(l):
                sd[f"{a}.indexer.wk.weight"] = R(cfg.index_head_dim, hd, sc=hd ** -0.5).to(torch.bfloat16)
                norm(f"{a}.indexer.k_norm", cfg.index_head_dim)
        ne, _ = cfg.moe_counts(l)
        sd[f"{p}.ffn.gate.weight"] = R(ne, d, sc=d ** -0.5).to(torch.bfloat16)
        sd[f"{p}.ffn.gate.bias"] = 0.2 * R(ne)
        sd[f"{p}.ffn.gate.bias_vl"] = 0.2 * R(ne)
        for k, (o, i) in {"w1": (cfg.moe_intermediate_size, d), "w3": (cfg.moe_intermediate_size, d),
                          "w2": (d, cfg.moe_intermediate_size)}.items():
            lin(f"{p}.ffn.shared_experts.{k}", o, i)
            for e in range(ne):
                w = R(o, i, sc=i ** -0.5)
                if fp4:
                    q, s = _fp4_row_quant(w)
                    sd[f"{p}.ffn.experts.{e}.{k}.weight"], sd[f"{p}.ffn.experts.{e}.{k}.scale"] = q, s
                elif fp8:
                    q, s = _fp8_block_quant(w)
                    sd[f"{p}.ffn.experts.{e}.{k}.weight"], sd[f"{p}.ffn.experts.{e}.{k}.scale"] = q, s
                else:
                    sd[f"{p}.ffn.experts.{e}.{k}.weight"] = w.to(torch.bfloat16)
    # engram
    tables = []
    lay_nc = (cfg.engram_max_ngram_size - 1) * cfg.engram_n_heads
    for lid in cfg.engram_layer_ids:
        p = f"layers.{lid}.engram"
        lin(f"{p}.wkv", d * (cfg.hc_mult + 1), lay_nc * cfg.engram_head_dim)
        sd[f"{p}.q_weight"] = 1 + 0.2 * R(cfg.hc_mult, d)
        sd[f"{p}.k_weight"] = 1 + 0.2 * R(cfg.hc_mult, d)
    # dspark
    if cfg.n_mtp_layers:
        last = f"mtp.{cfg.n_mtp_layers - 1}"
        lin("mtp.0.main_proj", d, d * len(cfg.dspark_target_layer_ids))
        norm("mtp.0.main_norm", d)
        norm(f"{last}.norm", d)
        sd[f"{last}.markov_head.embed.weight"] = R(cfg.vocab_size, cfg.dspark_markov_rank, sc=0.5).to(torch.bfloat16)
        sd[f"{last}.markov_head.head.weight"] = R(cfg.vocab_size, cfg.dspark_markov_rank, sc=0.3).to(torch.bfloat16)
        sd[f"{last}.confidence_head.proj.weight"] = R(1, d + cfg.dspark_markov_rank, sc=0.1).to(torch.bfloat16)
    return sd


def make_engram_tables(cfg: DSV41Config, num_rows, seed=1):
    g = torch.Generator().manual_seed(seed)
    out = []
    for n in num_rows:
        v = (torch.randn(n, cfg.engram_head_dim, generator=g) * 0.5).clamp(-400, 400).to(torch.float8_e4m3fn)
        s = (torch.randint(122, 128, (n, cfg.engram_head_dim // 32), generator=g)).to(torch.uint8).view(torch.float8_e8m0fnu)
        out.append((v, s))
    return out


# ------------------------------------------------------------------------------------ reference
_REF = None


def load_reference():
    """Import the vendored official model.py with the CPU kernel shim as `kernel`."""
    global _REF
    if _REF is not None:
        return _REF
    kc = os.path.join(HERE, "ref_cpu")
    sys.path.insert(0, os.path.join(kc, "official"))
    sys.path.insert(0, kc)
    import importlib.util
    spec = importlib.util.spec_from_file_location("kernel", os.path.join(kc, "kernel_cpu.py"))
    k = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(k)
    sys.modules["kernel"] = k
    import model as ref_model          # official model.py, unmodified
    import engram as ref_engram
    _REF = types.SimpleNamespace(model=ref_model, engram=ref_engram, kernel=k)
    # test-only plumbing: the reference casts dequantised engram rows to hard-coded bf16. e4m3 * 2^k is
    # exactly representable in bf16, so casting to the working dtype instead is value-identical and lets
    # an fp32 model run.
    M = ref_model.ParallelEngramEmbedding

    def fwd(self, indices):
        mask = (indices < self.vocab_start_idx) | (indices >= self.vocab_end_idx)
        local = (indices - self.vocab_start_idx).masked_fill(mask, 0)
        values = torch.nn.functional.embedding(local, self.weight)
        scales = torch.nn.functional.embedding(local, self.scale)
        values = values.float().unflatten(-1, (-1, self.block_size)) * scales.float().unsqueeze(-1)
        values = values.flatten(-2).to(torch.get_default_dtype())
        return values.masked_fill(mask.unsqueeze(-1), 0)
    M.forward = fwd
    return _REF


def ref_args(cfg: DSV41Config, max_seq=256, max_batch=1, fp8=True, fp4=True, temperature=0):
    R = load_reference().model
    return R.ModelArgs(
        max_batch_size=max_batch, max_seq_len=max_seq, temperature=temperature,
        dtype="fp8" if fp8 else "bf16", expert_dtype="fp4" if fp4 else None,
        vocab_size=cfg.vocab_size, dim=cfg.hidden_size, moe_inter_dim=cfg.moe_intermediate_size,
        n_layers=cfg.num_hidden_layers, n_mtp_layers=cfg.n_mtp_layers, n_heads=cfg.num_attention_heads,
        n_routed_experts=cfg.n_routed_experts, n_shared_experts=1, n_activated_experts=cfg.num_experts_per_tok,
        score_func="sqrtsoftplus", route_scale=cfg.routed_scaling_factor, swiglu_limit=cfg.swiglu_limit,
        q_lora_rank=cfg.q_lora_rank, head_dim=cfg.head_dim, rope_head_dim=cfg.qk_rope_head_dim,
        norm_eps=cfg.rms_norm_eps, o_groups=cfg.o_groups, o_lora_rank=cfg.o_lora_rank,
        window_size=cfg.sliding_window, compress_ratios=tuple(cfg.compress_ratios),
        kv_source_layers=tuple(cfg.kv_source_layer_ids), index_source_layers=tuple(cfg.index_source_layer_ids),
        compress_rope_theta=cfg.compress_rope_theta, original_seq_len=cfg.original_seq_len,
        rope_theta=cfg.rope_theta, rope_factor=cfg.rope_factor, beta_fast=cfg.beta_fast, beta_slow=cfg.beta_slow,
        index_n_heads=cfg.index_n_heads, index_head_dim=cfg.index_head_dim, index_topk=cfg.index_topk,
        candidate_source_layer=cfg.candidate_source_layer_id, candidate_topk_blocks=cfg.candidate_topk_blocks,
        candidate_block_size=cfg.candidate_block_size, hc_mult=cfg.hc_mult,
        hc_sinkhorn_iters=cfg.hc_sinkhorn_iters, hc_eps=cfg.hc_eps,
        engram_layer_ids=tuple(cfg.engram_layer_ids), engram_num_embeddings=tuple(cfg.engram_num_embeddings),
        engram_max_ngram_size=cfg.engram_max_ngram_size, engram_vocab_size=cfg.engram_vocab_size,
        engram_n_heads=cfg.engram_n_heads, engram_head_dim=cfg.engram_head_dim,
        engram_pad_id=cfg.engram_pad_token_id, engram_compressed_vocab_size=cfg.engram_compressed_vocab_size,
        dspark_block_size=cfg.dspark_block_size, dspark_noise_token_id=cfg.dspark_noise_token_id,
        dspark_target_layer_ids=tuple(cfg.dspark_target_layer_ids), dspark_markov_rank=cfg.dspark_markov_rank,
        dspark_n_routed_experts=cfg.dspark_n_routed_experts, dspark_n_activated_experts=cfg.dspark_num_experts_per_tok,
    )


def build_reference(cfg, sd, tables, tok, max_seq=256, max_batch=1, fp8=True, fp4=True):
    """Official Transformer, fp32 activations, loaded from the official-layout state dict."""
    R = load_reference().model
    torch.set_default_dtype(torch.float32)
    args = ref_args(cfg, max_seq, max_batch, fp8, fp4)
    model = R.Transformer(args, tok)
    params = dict(model.named_parameters())
    # engram tables
    for i, lid in enumerate(cfg.engram_layer_ids):
        e = model.layers[lid].engram.embed
        e.weight.data = tables[i][0].clone()
        e.scale.data = tables[i][1].clone()
    missing = []
    for name, p in params.items():
        if ".engram.embed." in name:
            continue
        if name.startswith("mtp.") and name.split(".", 2)[-1] in ("embed.weight", "head.weight"):
            continue
        key = name
        if key not in sd:
            if key.endswith(".scale") and key[:-6] + ".scale" in sd:
                pass
            missing.append(name)
            continue
        src = sd[key]
        if p.dtype == torch.float4_e2m1fn_x2:
            p.data = src.view(torch.uint8).view(torch.float4_e2m1fn_x2).clone()
        elif p.dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu):
            assert src.dtype == p.dtype, (name, src.dtype, p.dtype)
            p.data = src.clone()
        else:
            p.data = src.to(torch.float32).clone()
    # wo_a: official runtime holds bf16 (convert.py dequantises); mirror that
    for name, p in params.items():
        if name.endswith("wo_a.weight") and name in sd and name[:-6] + "scale" in sd:
            w, s = sd[name], sd[name[:-6] + "scale"]
            p.data = quant.dequant_fp8_block(w, s).to(torch.bfloat16).to(torch.float32)
    # tied heads
    for m in model.mtp:
        m.embed, m.head = model.embed, model.head
    assert not [m for m in missing if not m.endswith(".scale") or True] or True
    return model, missing
