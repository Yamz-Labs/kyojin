"""CPU checks of the ideal-curve handlers (no model, no GPU): mock modules named like the engine's classes.
  python tools/depth/test_ideal.py"""
import os, sys, types
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ideal


def mk(cls, **kw):
    return type(cls, (), kw)()


HW = {"bw_gbs": 236.0, "gemm_tflops": 30.0}


def test_dense_gqa_scales_linearly():
    m = mk("Attention", head_dim=128, v_head_dim=128, num_q_heads=32, num_kv_heads=8, sliding_window=-1, qsa_indexer=None)
    a, b = ideal.mix_attention(m, 1000, 2.0, 1), ideal.mix_attention(m, 2000, 2.0, 1)
    assert abs(b["dec_bytes"] / a["dec_bytes"] - 2) < 1e-9 and a["dec_bytes"] == 1000 * 8 * 256 * 2


def test_qsa_sparse_bounded():
    q = mk("QSAIndexer", token_budget=2048, compress_ratio=4, n_heads=4, head_dim=128, block_topk=512, sparse_threshold=staticmethod(lambda: 2051))
    m = mk("Attention", head_dim=256, v_head_dim=256, num_q_heads=24, num_kv_heads=2, sliding_window=-1, qsa_indexer=q)
    a, b = ideal.mix_attention(m, 100000, 2.0, 1), ideal.mix_attention(m, 400000, 2.0, 1)
    assert a["dec_bytes"] == b["dec_bytes"]                                # gathered keys do not grow with depth
    assert abs(b["idx_flops"] / a["idx_flops"] - 4) < 1e-9                 # indexer scoring is linear in depth


def test_mla_dsa_matches_flat1_numbers():
    m = mk("MLAttention", kv_lora_rank=512, qk_rope_head_dim=0, num_q_heads=64, index_topk=2048, indexer_mode="full",
           idx_plane_dim=256, index_kpool=4, index_n_heads=32, index_head_dim=128)
    r = ideal.mix_mla(m, 250000, 2.0, 1)
    assert abs(r["pf_flops"] - 0.268e9) / 0.268e9 < 0.01                   # 2.95 GFLOP/row over 11 DSA layers (flat1 step 1)
    assert abs(r["idx_flops"] - 8192 * 250000 / 4) < 1                     # 8192 FLOP per (row, pooled key)


def test_dsv4_types():
    kw = dict(head_dim=512, num_q_heads=64, sliding_window=128, compress_rate=4, index_n_heads=32, index_head_dim=128, index_topk=512)
    sw = ideal.mix_dsv4(mk("DSV4Attention", layer_type="sliding", **kw), 500000, 2.0, 1)
    hca = ideal.mix_dsv4(mk("DSV4Attention", layer_type="hca", **{**kw, "compress_rate": 128}), 500000, 2.0, 1)
    csa = ideal.mix_dsv4(mk("DSV4Attention", layer_type="csa", **kw), 500000, 2.0, 1)
    assert sw["dec_bytes"] == 128 * 512 * 2 and hca["dec_bytes"] > sw["dec_bytes"] and csa["idx_flops"] > 0


def test_unknown_mixer_raises():
    root = mk("Model", modules=[mk("FancyNewMixer", modules=[])])
    try:
        ideal.collect_mixers(root)
    except NotImplementedError as e:
        assert "FancyNewMixer" in str(e)
        return
    raise AssertionError("unknown class priced silently")


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
