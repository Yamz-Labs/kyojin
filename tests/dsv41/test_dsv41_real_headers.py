"""
Official loader vs the REAL checkpoint structure (headers only, fetch_real_headers.py): every tensor the graph
asks for exists with the dtype/shape the loader's decoders assume, for all 43 layers incl. MTP, Engram and the
experts of every layer; no tensor of the checkpoint that the text path needs is missing.
"""
import glob, json, os, sys
import pytest
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hardware import require
require("ext")
from exllamav3.dsv41.config import DSV41Config
from exllamav3.dsv41.weights import WeightSource
from exllamav3.dsv41.linear import DenseLin
from exllamav3.dsv41.experts import SourceExpertProvider
from exllamav3.dsv41.model import DSV41Model

R = os.environ.get("DSV41_REAL_DIR", "./ds41_work/ds41port/scratch/real") + "/headers"
CFG = "./ds41_work/ds41port/ref/config.json"
pytestmark = pytest.mark.skipif(not (os.path.isdir(R) and os.path.exists(CFG)), reason="real headers not fetched")


def load_headers():
    T = {}
    for f in glob.glob(R + "/*.json"):
        for k, v in json.load(open(f)).items():
            if k != "__metadata__":
                T[k] = (v["dtype"], tuple(v["shape"]))
    return T


def test_every_requested_tensor_has_the_expected_dtype_and_shape():
    T = load_headers()
    cfg = DSV41Config.from_file(CFG)
    d, hd, H, G = cfg.hidden_size, cfg.head_dim, cfg.num_attention_heads, cfg.o_groups
    asked = []

    class Rec(WeightSource):
        n_layers = cfg.num_hidden_layers

        def has(self, n):
            return True

        def tensor(self, n, dtype=torch.float32, device="cpu"):
            asked.append(("t", n)); return torch.ones(2, 2)

        def linear(self, n, dtype=torch.float32, device="cpu"):
            asked.append(("l", n)); return DenseLin(torch.ones(2, 2))
    DSV41Model(cfg, Rec(), SourceExpertProvider(Rec(), {l: cfg.moe_counts(l)[0] for l in range(43)}), None,
               token_map=[0] * cfg.vocab_size)
    V, nq, ni, ihd, ih = cfg.vocab_size, cfg.q_lora_rank, cfg.index_n_heads * cfg.index_head_dim, cfg.index_head_dim, cfg.index_n_heads
    inter, rank = cfg.moe_intermediate_size, cfg.dspark_markov_rank
    nl = len(cfg.dspark_target_layer_ids)
    # (suffix) -> (weight dtype, weight shape, scale shape or None)
    F8, BF, F32 = "F8_E4M3", "BF16", "F32"

    def sc(o, i):
        return (-(-o // 32), -(-i // 32))
    exp = {
        "attn.wq_a": (F8, (nq, d), sc(nq, d)), "attn.wq_b": (F8, (H * hd, nq), sc(H * hd, nq)),
        "attn.wkv": (F8, (hd, d), sc(hd, d)), "attn.wo_a": (F8, (G * cfg.o_lora_rank, H * hd // G), sc(G * cfg.o_lora_rank, H * hd // G)),
        "attn.wo_b": (F8, (d, G * cfg.o_lora_rank), sc(d, G * cfg.o_lora_rank)),
        "attn.indexer.wq_b": (F8, (ni, nq), sc(ni, nq)),
        "ffn.shared_experts.w1": (F8, (inter, d), sc(inter, d)), "ffn.shared_experts.w3": (F8, (inter, d), sc(inter, d)),
        "ffn.shared_experts.w2": (F8, (d, inter), sc(d, inter)),
        "engram.wkv": (F8, (d * (cfg.hc_mult + 1), (cfg.engram_max_ngram_size - 1) * cfg.engram_n_heads * cfg.engram_head_dim),
                       None),
        "main_proj": (F8, (d, d * nl), sc(d, d * nl)),
    }
    plain = {
        "attn.q_norm": (BF, (nq,)), "attn.kv_norm": (BF, (hd,)), "attn.attn_sink": (F32, (H,)),
        "attn_norm": (BF, (d,)), "ffn_norm": (BF, (d,)),
        "attn.compressor.wkv": (BF, (hd, d)), "attn.compressor.wgate": (BF, (hd, d)), "attn.compressor.norm": (BF, (hd,)),
        "attn.indexer.weights_proj": (BF, (ih, d)), "attn.indexer.wk": (BF, (ihd, hd)), "attn.indexer.k_norm": (BF, (ihd,)),
        "hc_attn_fn": (F32, (24, 4 * d)), "hc_ffn_fn": (F32, (24, 4 * d)), "hc_attn_base": (F32, (24,)),
        "hc_ffn_base": (F32, (24,)), "hc_attn_scale": (F32, (3,)), "hc_ffn_scale": (F32, (3,)),
        "engram.q_weight": (BF, (cfg.hc_mult, d)), "engram.k_weight": (BF, (cfg.hc_mult, d)),
        "ffn.gate.bias": (F32, None), "ffn.gate.bias_vl": (F32, None),
    }
    top = {"embed": (BF, (V, d)), "head": (BF, (V, d)), "norm": (BF, (d,)),
           "mtp.0.main_norm": (BF, (d,)), "mtp.2.norm": (BF, (d,)),
           "mtp.2.markov_head.embed": (BF, (V, rank)), "mtp.2.markov_head.head": (BF, (V, rank)),
           "mtp.2.confidence_head.proj": (BF, (1, d + rank))}
    bad = []
    for kind, n in asked:
        suffix = n.split(".", 2)[2] if n.startswith("layers.") else (n.split(".", 2)[2] if n.startswith("mtp.") else n)
        if n in top:
            dt, shp = top[n]
            wk = n + ".weight" if (n + ".weight") in T else n
            if T.get(wk) != (dt, shp):
                bad.append((n, T.get(wk), (dt, shp)))
        elif suffix in exp:
            dt, shp, ssh = exp[suffix]
            if T.get(n + ".weight") != (dt, shp):
                bad.append((n, T.get(n + ".weight"), (dt, shp)))
            if ssh is not None and T.get(n + ".scale") != ("F8_E8M0", ssh):
                bad.append((n + ".scale", T.get(n + ".scale"), ssh))
        elif suffix in plain:
            dt, shp = plain[suffix]
            key = n if n in T else n + ".weight"
            if key not in T or T[key][0] != dt or (shp is not None and T[key][1] != shp):
                bad.append((n, T.get(key), (dt, shp)))
        elif suffix == "ffn.gate":
            ne = cfg.moe_counts(int(n.split(".")[1]) + (0 if n.startswith("layers.") else 40))[0]
            if T.get(n + ".weight") != (BF, (ne, d)):
                bad.append((n, T.get(n + ".weight"), (BF, (ne, d))))
        else:
            bad.append(("UNCLASSIFIED", n))
    assert not bad, bad[:8]
    # experts: packed fp4 as int8 + per-row/32 e8m0 scales, every layer, every expert
    for l in range(43):
        p = f"layers.{l}" if l < 40 else f"mtp.{l - 40}"
        ne = cfg.moe_counts(l)[0]
        for e in range(ne):
            for w, (o, i) in (("w1", (inter, d)), ("w3", (inter, d)), ("w2", (d, inter))):
                assert T[f"{p}.ffn.experts.{e}.{w}.weight"] == ("I8", (o, i // 2)), (p, e, w)
                assert T[f"{p}.ffn.experts.{e}.{w}.scale"] == ("F8_E8M0", (o, i // 32)), (p, e, w)
    for lid, rows in zip(cfg.engram_layer_ids, cfg.engram_num_embeddings):
        assert T[f"layers.{lid}.engram.embed.weight"] == ("F8_E4M3", (rows, cfg.engram_head_dim))
        assert T[f"layers.{lid}.engram.embed.scale"] == ("F8_E8M0", (rows, cfg.engram_head_dim // 32))
    # nothing the text path needs is left unasked: remaining names are vision / image tokens / unrequested leaf kinds
    asked_names = {n for _, n in asked}
    left = [k for k in T if not k.startswith(("vision.", "aligner.", "image_")) and ".experts." not in k
            and ".engram.embed." not in k]
    unasked = []
    for k in left:
        base = k[:-6] if k.endswith(".scale") else (k[:-7] if k.endswith(".weight") else k)
        if base not in asked_names and k not in asked_names:
            unasked.append(k)
    # tied MTP embed/head and bias_vl (vision routing bias) are the only intentional leftovers
    unasked = [k for k in unasked if not k.endswith("ffn.gate.bias_vl")]
    assert not unasked, unasked[:10]
