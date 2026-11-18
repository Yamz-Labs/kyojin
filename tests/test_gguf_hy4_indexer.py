"""Hy4 k-pool DSA indexer (row-list form): predicate, maths vs an independent float64 numpy reference, chunked == one-shot
(prefill / decode), and layer-0 runs on the oracle prompts. The oracle (llama.cpp, hy4_oracle) dumps NO indexer tensor (only
hc_pre, attn_kqv, attn_gated, attn_out, ffn_moe_out, l_out, engram_out, hc_head, result_output), so there is no oracle value
for scores or indices: the real-prompt runs check selection structure and the numpy reference, nothing against llama.cpp."""
import os, sys, json, types
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
import test_gguf_hy4_hc as H                                   # noqa: E402  (stubs + GGUF + oracle helpers)
from exllamav3.modules.kpool_indexer import KPoolIndexer, rms, rope_tail, q8_act   # noqa: E402
from exllamav3.loader.gguf_blocks import indexer as IB          # noqa: E402
from exllamav3.loader.gguf_registry import REGISTRY             # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.exists(H.GGUF1), reason="Hy4 GGUF absent")
EPS = 9.999999747378752e-06
BAR = 1e-3
INDEXER_LAYERS = [0, 1] + list(range(5, 42, 4)) + [42]
_W = {}


def weights(layer=0):
    if layer not in _W:
        _W[layer] = IB.load_weights(H.gg()[0], layer)
    return _W[layer]


def np_ref(x, qr, W, top_k, kpool=4, tail=True, rope_dim=64, base=1e7, H_=16, d=128, eps=EPS):
    """independent float64 reference: complex-number rope, per-group loops, per-query sorted selection"""
    f = lambda a: np.asarray(a, np.float64)
    x, qr, W = f(x), f(qr), {k: f(v) for k, v in W.items()}
    rmsn = lambda a, w: a / np.sqrt((a ** 2).mean(-1, keepdims=True) + eps) * w
    inv = base ** (-np.arange(0, rope_dim, 2) / rope_dim)
    def rope(a, p):
        z = (a[..., d - rope_dim::2] + 1j * a[..., d - rope_dim + 1::2]) * np.exp(1j * p * inv)
        o = a.copy(); o[..., d - rope_dim::2], o[..., d - rope_dim + 1::2] = z.real, z.imag
        return o
    n = x.shape[0]
    ik = rmsn(x @ W["attn_k"].T, W["k_norm"]); val, gate = ik @ W["value"].T, ik @ W["gate"].T
    pooled = []
    for g in range(n // kpool):
        s = slice(g * kpool, (g + 1) * kpool)
        e = np.exp(gate[s] + W["ape"]); e /= e.sum(0)
        pooled.append(rope(rmsn((e * val[s]).sum(0), W["pool_norm"]), g * kpool))
    pooled = np.array(pooled).reshape(-1, d)
    iq = (qr @ W["q_b"].T).reshape(n, H_, d)
    wt = x @ W["proj"].T / np.sqrt(d * H_)
    rows, scores = [], np.full((n, len(pooled)), -np.inf)
    for t in range(n):
        iqt = rope(iq[t], t)
        for g in range(len(pooled)):
            if g * kpool + kpool - 1 < t if tail else g * kpool + kpool - 1 <= t:
                scores[t, g] = (np.maximum(iqt @ pooled[g], 0) * wt[t]).sum()
        vis = [g for g in range(len(pooled)) if np.isfinite(scores[t, g])]
        pick = sorted(vis, key=lambda g: (-scores[t, g], g))[:top_k // kpool - int(tail)]
        r = {g * kpool + j for g in pick for j in range(kpool)}
        r |= set(range(t - t % kpool, t + 1)) if tail else set(range(t - (t + 1) % kpool + 1, t + 1))
        rows.append(sorted(r))
    return rows, scores


def compare(sel, rows, scores):
    set_eq = all(a.tolist() == b for a, b in zip(sel.rows, rows))
    fin = np.isfinite(scores)
    assert (torch.isfinite(sel.scores).numpy() == fin).all()
    got = sel.scores.double().numpy()
    err = np.abs(got[fin] - scores[fin]).max() / max(np.abs(scores[fin]).max(), 1e-30) if fin.any() else 0.0
    return set_eq, float(err)


def mk(W, **kw):
    return KPoolIndexer(W, eps=EPS, **kw)


def test_predicate_and_plan():
    g, plan = H.gg()
    for i in range(43):
        names = plan.layers[i]["attn_mod"]
        if i in INDEXER_LAYERS:
            assert "dsa_kpool_indexer" in names and "dsa_indexer" not in names, (i, names)
            assert plan.claimed[f"blk.{i}.indexer.attn_k.weight"] == "dsa_kpool_indexer"
            assert plan.claimed[f"blk.{i}.indexer_compressor_value.weight"] == "dsa_kpool_indexer"
        else:
            assert "dsa_kpool_indexer" not in names and "dsa_indexer" not in names, (i, names)
    c = plan.configs[("dsa_kpool_indexer", 0)]
    assert (c["n_head"], c["head_dim"], c["top_k"], c["kpool"], c["select_tail"], c["rope_dim"], c["rope_base"]) == (16, 128, 2048, 4, True, 64, 1e7)
    assert c["variant"] == "kpool_value" and c["k_norm"] == "rms_weight" and not c["swa_layer"]
    assert not any("dsa_kpool_indexer" in g_ for g_ in plan.gaps) and not any("dsa_indexer" in g_ for g_ in plan.gaps)
    m = IB._build(g, plan, 5)
    assert m.n_top == 511 and m.w["q_b"].shape == (2048, 1024) and m.w["ape"].shape == (4, 128) and m.q8 == {"attn_k", "q_b", "value", "gate"}


def _fake(has, keys):
    return types.SimpleNamespace(has=lambda i, s: s in has, key=lambda k, d=None: keys.get(k, d), feats={}, infos={})


def test_other_gguf_unchanged():
    """deepseek-style dsa indexer (no pooling) and glm5-next (kpool_gate, LayerNorm bias): stay on dsa_indexer"""
    keys = {"attention.indexer.top_k": 2048, "attention.indexer.kpool": 4, "attention.indexer.head_count": 16, "attention.indexer.key_length": 128}
    base = {"indexer.attn_k.weight", "indexer.attn_q_b.weight", "indexer.proj.weight", "indexer.k_norm.weight"}
    dsa = _fake(base, {k: v for k, v in keys.items() if k.endswith(("top_k", "head_count", "key_length"))})
    glm = _fake(base | {"indexer.k_norm.bias", "indexer_compressor_gate.weight", "indexer_compressor_ape.weight", "indexer_compressor_norm.weight"}, keys)
    for ctx in (dsa, glm):
        assert not IB.is_kpool(ctx, 0) and REGISTRY["dsa_indexer"].select(ctx, 0)
    hy = _fake(base | {"indexer_compressor_value.weight", "indexer_compressor_gate.weight", "indexer_compressor_ape.weight", "indexer_compressor_norm.weight"}, keys)
    assert IB.is_kpool(hy, 0) and not REGISTRY["dsa_indexer"].select(hy, 0)


@pytest.mark.parametrize("tail", [True, False])
def test_synthetic_small_topk(tail):
    """selection actually exercised: top_k 16 (3 groups + tail, or 4 groups), 45 tokens, real layer-0 weights, random activations"""
    torch.manual_seed(1)
    W, n = weights(0), 45
    x, qr = torch.randn(n, 3072), torch.randn(n, 1024)
    sel = mk(W, top_k=16, select_tail=tail).forward(x, qr)
    rows, sc = np_ref(x.numpy(), qr.numpy(), {k: v.numpy() for k, v in W.items()}, 16, tail=tail)
    eq, err = compare(sel, rows, sc)
    print(f"synthetic tail={tail}: rows set-equal {eq}, score max|d|/max|ref| {err:.2e}, max rows {max(len(r) for r in rows)}")
    assert eq and err <= BAR
    assert max(len(r) for r in rows) == (16 if tail else 19) and len(rows[n - 1]) == (12 if tail else 16) + ((n - 1) % 4 + 1 if tail else n % 4)


def test_chunked_equals_oneshot():
    """prefill in chunks then decode token by token == one call; also pooled keys carried across calls"""
    torch.manual_seed(2)
    W, n = weights(0), 61
    x, qr = torch.randn(n, 3072), torch.randn(n, 1024)
    a = mk(W, top_k=16).forward(x, qr)
    m, parts, o = mk(W, top_k=16), [], 0
    for c in (7, 16, 1, 1, 1, 13, 1, 1, 20):
        parts.append(m.forward(x[o:o + c], qr[o:o + c])); o += c
    assert o == n
    rows = [r for p in parts for r in p.rows]
    assert all(torch.equal(r1, r2) for r1, r2 in zip(a.rows, rows))
    P = a.scores.shape[1]
    for p, s0 in zip(parts, np.cumsum([0, 7, 16, 1, 1, 1, 13, 1, 1])):
        k = p.scores.shape[0]
        ref = a.scores[s0:s0 + k, :p.scores.shape[1]]
        assert torch.equal(torch.isfinite(ref), torch.isfinite(p.scores)) and (ref[torch.isfinite(ref)] - p.scores[torch.isfinite(p.scores)]).abs().max() < 1e-4


def test_rope_and_q8():
    x = torch.randn(3, 2, 128)
    pos = torch.tensor([0, 5, 977])
    y = rope_tail(x, pos, 64, 1e7)
    assert torch.equal(y[..., :64], x[..., :64]) and torch.equal(y[0], x[0])
    assert torch.allclose(y.pow(2).sum(-1), x.pow(2).sum(-1), rtol=1e-5)
    q = q8_act(x.reshape(-1, 128))
    assert (q - x.reshape(-1, 128)).abs().max() < x.abs().max() / 127


def real_inputs(ids):
    """layer 0 attention inputs from the prompt tokens: the four streams are copies of the embedding, so the HC collapse is a
    scalar times it and the RMSNorm of attn_norm removes the scalar: cur = rms(emb) * attn_norm.weight; qr = rms(cur Wq_a^T) * q_a_norm"""
    g, _ = H.gg()
    emb = H.emb_rows(ids)
    an = g.get_tensor("blk.0.attn_norm.weight").float().reshape(-1)
    cur = rms(emb, an, EPS)
    ty, dims, _ = g.infos["blk.0.attn_q_a.weight"]
    wqa = IB._w(g, "blk.0.attn_q_a.weight", ty, dims)
    qn = g.get_tensor("blk.0.attn_q_a_norm.weight").float().reshape(-1)
    return cur, rms(cur @ wqa.T, qn, EPS)


def run_prompt(prompt, ndec):
    idx = json.load(open(os.path.join(H.ORACLE, prompt, "index.json")))
    ids = idx["prompt_ids"] + idx["gen_ids"][:ndec]
    n0 = idx["n_prompt_tokens"]
    assert len(idx["prompt_ids"]) == n0
    cur, qr = real_inputs(ids)
    out = {}
    for q8 in (False, True):
        m = KPoolIndexer(weights(0), eps=EPS, act_q8=q8)
        pre = m.forward(cur[:n0], qr[:n0])
        dec = [m.forward(cur[n0 + j:n0 + j + 1], qr[n0 + j:n0 + j + 1]) for j in range(ndec)]
        out[q8] = (pre, dec)
    return n0, cur, qr, out


@pytest.mark.parametrize("prompt,ndec", [("p1_short_en", 4), ("p4_long_n700", 4), ("p5_long_n4000", 4)])
def test_oracle_prompts_layer0(prompt, ndec):
    n0, cur, qr, out = run_prompt(prompt, ndec)
    pre, dec = out[False]
    full = [torch.arange(t + 1, dtype=torch.int32) for t in range(n0)]
    exercised = n0 > 2048 + 4
    if not exercised:
        # fewer than top_k rows: every causal row selected (the trivial case)
        assert all(torch.equal(a, b) for a, b in zip(pre.rows, full))
        assert all(torch.equal(d.rows[0], torch.arange(n0 + j + 1, dtype=torch.int32)) for j, d in enumerate(dec))
    else:
        lens = torch.tensor([len(r) for r in pre.rows])
        assert int(lens.max()) <= 2048 and (lens[:2048] == torch.arange(1, 2049)).all()
        t = n0 - 1
        assert len(pre.rows[t]) == 4 * 511 + t % 4 + 1 and (pre.rows[t] != torch.arange(t + 1, dtype=torch.int32)[:len(pre.rows[t])]).any()
        assert all(len(d.rows[0]) == 4 * 511 + (n0 + j) % 4 + 1 for j, d in enumerate(dec))
        # full numpy reference on a prefix covering the first selection (all groups visible beyond 511)
        m = 2400
        rows, sc = np_ref(cur[:m].numpy(), qr[:m].numpy(), {k: v.numpy() for k, v in weights(0).items()}, 2048)
        sel = KPoolIndexer(weights(0), eps=EPS).forward(cur[:m], qr[:m])
        eq, err = compare(sel, rows, sc)
        print(f"{prompt} L0 prefix {m}: rows set-equal {eq}, score rel {err:.2e}")
        assert eq and err <= BAR
        nsel = sum(1 for r, t in zip(rows, range(m)) if len(r) < t + 1)
        print(f"  queries with a real selection (rows < causal): {nsel} of {m}")
        assert nsel > 0
    # q8 activation emulation: scores and picks vs the float activations
    pre8 = out[True][0]
    f = torch.isfinite(pre.scores)
    rel = float((pre.scores[f] - pre8.scores[f]).abs().max() / pre.scores[f].abs().max())
    same = sum(torch.equal(a, b) for a, b in zip(pre.rows, pre8.rows))
    print(f"{prompt} L0 n={n0}+{ndec} dec: selection exercised={exercised}; q8-act vs fp32-act score rel {rel:.2e}, "
          f"rows identical for {same}/{n0} queries")
    assert rel < 2e-2


if __name__ == "__main__":
    sys.exit(pytest.main(["-q", "-s", __file__]))
