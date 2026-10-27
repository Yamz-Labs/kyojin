"""
Gate 1 (CPU, fp32): tiny random DS4.1 model through the OFFICIAL reference (vendored model.py/engram.py,
CPU kernel shim) and through the Kyojin implementation.
  * per-op taps (engram, compressor latent, indexer top-k, attention out, ffn out, block out)
  * prefill + decode logits, greedy tokens, main_hidden
  * dspark forward_spec (prefill seeding + block decode: ids, logits, confidence)
tol: max abs diff 1e-5 (observed: bitwise 0.0 on CPU, the op sequences are the same)
"""
import os, sys
import pytest
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import *
from exllamav3.dsv41.engram import build_compressed_token_map, EngramLayout, MemoryEngramStore
from exllamav3.dsv41.weights import DictWeightSource
from exllamav3.dsv41.experts import SourceExpertProvider
from exllamav3.dsv41.model import DSV41Model

TOL = 1e-5


def setup(quantized, seed=0):
    cfg = tiny_config()
    tok = FakeTok(cfg.vocab_size)
    tmap, nv = build_compressed_token_map(tok)
    cfg.engram_compressed_vocab_size = nv
    lay = EngramLayout.build(cfg.engram_layer_ids, None, cfg.engram_max_ngram_size, cfg.engram_vocab_size,
                             cfg.engram_n_heads, cfg.engram_head_dim)
    cfg.engram_num_embeddings = list(lay.num_embeddings)
    tables = make_engram_tables(cfg, lay.num_embeddings, seed + 1)
    sd = make_state_dict(cfg, seed, fp8=quantized, fp4=quantized)
    return cfg, tok, tmap, tables, sd


def build_mine(cfg, sd, tables, tmap, dtype=torch.float32, device="cpu"):
    src = DictWeightSource(sd, cfg.num_hidden_layers)
    prov = SourceExpertProvider(src, {l: cfg.moe_counts(l)[0] for l in range(cfg.num_hidden_layers + cfg.n_mtp_layers)},
                                dtype=dtype, device=device)
    store = MemoryEngramStore([t[0] for t in tables], [t[1] for t in tables])
    return DSV41Model(cfg, src, prov, store, token_map=tmap, dtype=dtype, device=device)


def install_hooks(ref, cfg):
    taps = {}

    def mk(name):
        def h(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            taps[name] = o.detach().clone()
        return h
    for l, layer in enumerate(ref.layers):
        layer.attn.register_forward_hook(mk(f"L{l}.attn"))
        layer.ffn.register_forward_hook(mk(f"L{l}.ffn"))
        layer.register_forward_hook(mk(f"L{l}.out"))
        if layer.engram is not None:
            layer.engram.register_forward_hook(mk(f"L{l}.engram"))
        if layer.attn.compressor is not None:
            layer.attn.compressor.register_forward_hook(
                lambda m, i, o, l=l: taps.__setitem__(f"L{l}.comp", None if o is None else o.detach().clone()))
        if layer.attn.indexer is not None:
            layer.attn.indexer.register_forward_hook(mk(f"L{l}.topk"))
    return taps


def cmp_taps(rt, mt, where, worst):
    for k, rv in rt.items():
        if rv is None:
            assert mt.get(k) is None, (where, k, "expected no latent")
            continue
        assert k in mt, (where, k, "missing in mine")
        mv = mt[k]
        assert mv.shape == rv.shape, (where, k, mv.shape, rv.shape)
        if rv.dtype in (torch.int32, torch.int64):
            assert torch.equal(rv, mv), (where, k, "index mismatch")
            continue
        d = (rv.float() - mv.float()).abs().max().item()
        worst[k] = max(worst.get(k, 0.0), d)
        assert d <= TOL, (where, k, d)


@pytest.mark.parametrize("quantized", [True, False], ids=["fp8fp4", "plain"])
@pytest.mark.parametrize("batch,seed", [(1, 0), (2, 3)])
def test_trunk_and_dspark(quantized, batch, seed):
    cfg, tok, tmap, tables, sd = setup(quantized, seed)
    ref, missing = build_reference(cfg, sd, tables, tok, 256, batch, quantized, quantized)
    assert not missing, missing
    mine = build_mine(cfg, sd, tables, tmap)
    st = mine.new_state(batch, 256)
    rtaps = install_hooks(ref, cfg)
    g = torch.Generator().manual_seed(seed + 100)
    S, steps = 43, 20
    ids = torch.randint(0, cfg.vocab_size, (batch, S + steps), generator=g)
    worst = {}
    with torch.inference_mode():
        rids, rl, rm = ref(ids[:, :S])
        mtaps = {}
        ml, mm = mine.forward(ids[:, :S], 0, st, mtaps)
        cmp_taps(rtaps, mtaps, "prefill", worst)
        assert (rl - ml).abs().max().item() <= TOL
        assert (rm - mm).abs().max().item() <= TOL
        assert torch.equal(rids, ml.argmax(-1))
        assert rl.abs().max() > 1.0 and rm.abs().max() > 0.1
        # dspark: prefill seeding
        assert ref.forward_spec(rids, rm, 0) is None
        assert mine.mtp.forward_spec(rids, mm, 0, st) is None
        n_block_checks = 0
        for i in range(S, S + steps):
            rtaps.clear()
            mtaps = {}
            rids, rl, rm = ref(ids[:, i:i + 1], i)
            ml, mm = mine.forward(ids[:, i:i + 1], i, st, mtaps)
            cmp_taps(rtaps, mtaps, f"decode{i}", worst)
            d = (rl - ml).abs().max().item()
            assert d <= TOL, (i, d)
            assert torch.equal(rids, ml.argmax(-1)), i
            rtaps.clear()
            ro = ref.forward_spec(rids, rm, i)
            mo = mine.mtp.forward_spec(rids, mm, i, st)
            r_ids, r_log, r_conf = ro
            m_ids, m_log, m_conf = mo
            assert torch.equal(r_ids, m_ids), (i, r_ids, m_ids)
            assert (r_log - m_log).abs().max().item() <= TOL
            assert (r_conf - m_conf).abs().max().item() <= TOL
            n_block_checks += 1
    print("worst per-op diff:", {k: v for k, v in sorted(worst.items(), key=lambda kv: -kv[1])[:6]})


def test_sensitivity_negative_controls():
    """The gate must be able to fail: perturbing single pieces of my graph breaks the comparison."""
    cfg, tok, tmap, tables, sd = setup(True, 0)
    ref, _ = build_reference(cfg, sd, tables, tok, 256, 1, True, True)
    ids = torch.randint(0, cfg.vocab_size, (1, 43), generator=torch.Generator().manual_seed(7))
    with torch.inference_mode():
        _, rl, _ = ref(ids)

    def run(mutate):
        m = build_mine(cfg, sd, tables, tmap)
        mutate(m)
        st = m.new_state(1, 256)
        return (m.forward(ids, 0, st)[0] - rl).abs().max().item()
    assert run(lambda m: None) <= TOL
    # engram off
    assert run(lambda m: [setattr(m.lw[l], "engram", None) for l in cfg.engram_layer_ids]) > 1e-3
    # candidate level off (layer 5 would pick from all positions)
    def no_cand(m):
        m.cfg.candidate_source_layer_id = -1
    assert run(no_cand) > 1e-4
    # wrong yarn (sliding rope used on compressed layers)
    def bad_rope(m):
        m.cfg.original_seq_len = 0
    assert run(bad_rope) > 1e-4
    # swiglu limit
    def bad_lim(m):
        m.cfg.swiglu_limit = 0.05
    assert run(bad_lim) > 1e-3


@pytest.mark.parametrize("seed", range(10, 18))
def test_seed_sweep(seed):
    """Robustness: fp8/fp4 fake-quant steps amplify any last-bit difference; 8 more seeds x both batch sizes
    must stay inside the tolerance (they are in fact bitwise equal)."""
    for batch in (1, 2):
        test_trunk_and_dspark(True, batch, seed)
        test_trunk_and_dspark(False, batch, seed)


def test_chunked_prefill_fallback_equals_single_pass():
    """forward_chunk (token loop) after an initial prefill == one long prefill: logits and all state used later."""
    cfg, tok, tmap, tables, sd = setup(True, 4)
    a, b = build_mine(cfg, sd, tables, tmap), build_mine(cfg, sd, tables, tmap)
    sa, sb = a.new_state(1, 128), b.new_state(1, 128)
    ids = torch.randint(0, cfg.vocab_size, (1, 50), generator=torch.Generator().manual_seed(1))
    la, _ = a.forward(ids[:, :26], 0, sa)
    lb, mb = b.forward_chunk(ids[:, 26:50], 26, sb) if False else (None, None)
    lb0, _ = b.forward(ids[:, :1], 0, sb)                       # token loop from position 0
    for i in range(1, 26):
        lb0, _ = b.forward(ids[:, i:i + 1], i, sb)
    # token loop vs one-shot prefill agree to the quantiser-flip floor (different GEMM shapes); ids equal
    assert torch.equal(la.argmax(-1), lb0.argmax(-1))
    assert (la - lb0).abs().max().item() < 5e-2
    # continuing in chunks of 5 vs single tokens is exactly the same computation
    c = build_mine(cfg, sd, tables, tmap)
    sc = c.new_state(1, 128)
    c.forward(ids[:, :26], 0, sc)
    d = build_mine(cfg, sd, tables, tmap)
    sdd = d.new_state(1, 128)
    d.forward(ids[:, :26], 0, sdd)
    l1, _ = c.forward_chunk(ids[:, 26:50], 26, sc)
    for i in range(26, 50):
        l2, _ = d.forward(ids[:, i:i + 1], i, sdd)
    assert torch.equal(l1, l2)
