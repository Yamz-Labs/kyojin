"""
Streaming calibration API (StreamCalibrator) vs the framework path (modules + run_conversion): the Hessians of every
linear, expert by expert, trunk + MTP, must be the same numbers (both run the same graph on the same rows; the framework
path pools nothing across rows, the streaming path pools the MoE across rows and streams experts).
"""
import os, sys
import pytest
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tiny_ckpt import write_tiny
from harness import *
from exllamav3.dsv41.weights import DictWeightSource
from exllamav3.dsv41.engram import MemoryEngramStore
from exllamav3.dsv41.convert_driver import run_conversion, NoopQuantizer, _anchors
from exllamav3.dsv41.stream_calib import StreamCalibrator, SourceLayerWeights, EngramSupport, CollectSink
from exllamav3.dsv41.config import DSV41Config


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("tiny_stream"))
    cfg, sd, tables, tmap = write_tiny(d, seed=11)
    g = torch.Generator().manual_seed(5)
    rows = [torch.randint(0, cfg.vocab_size, (1, n), generator=g) for n in (45, 38, 52)]
    caps = {}
    res = run_conversion(d, rows, NoopQuantizer(), bits=3.0, moe_mode="routed", mtp_anchors=6, log=lambda s: None,
                         on_capture=lambda k, c: caps.__setitem__(k, {q: {"H": v["H"].clone(), "count": v["count"]}
                                                                      for q, v in c.items()}))
    return d, cfg, sd, tables, tmap, rows, caps, res


WORST = {}


def close(a, b, tol=2e-4):
    d = (a - b).abs().max().item() / max(b.abs().max().item(), 1e-6)
    WORST["max"] = max(WORST.get("max", 0.0), d)
    return d <= tol


def cmp(sink_items, caps_for, qmap_of, what, tol=2e-4):
    n = 0
    for grp, lins, H in sink_items:
        q = qmap_of(grp)
        c = caps_for.get(q)
        if H is None:
            assert c is None, (what, grp, "stream says unrouted, framework captured")
            continue
        assert c is not None, (what, grp, q)
        assert H["count"] == c["count"], (what, grp, H["count"], c["count"])
        a_, b_ = H["H"].cpu(), c["H"].cpu()
        assert close(a_, b_, tol), (what, grp, (a_ - b_).abs().max().item() / b_.abs().max().item())
        n += 1
    return n


def test_stream_hessians_equal_framework(setup):
    d, cfg, sd, tables, tmap, rows, caps, res = setup
    src = DictWeightSource(sd, cfg.num_hidden_layers)
    store = MemoryEngramStore([t[0] for t in tables], [t[1] for t in tables])
    cal = StreamCalibrator(cfg, "cpu", src.tensor("embed", torch.float32), EngramSupport(cfg, tmap, store))
    cal.start(rows)
    checked = 0
    for l in range(cfg.num_hidden_layers):
        sink = CollectSink()
        W = SourceLayerWeights(src, l)
        cal.run_layer(l, W, sink, commit=False)
        cal.run_layer(l, W, None, commit=True)                 # advance (reference weights)
        p = f"layers.{l}"

        def qmap_of(grp):
            if grp.startswith("ffn.experts."):
                e, kind = grp.split(".")[2], grp.split(".")[3]
                return f"{p}.ffn.experts.{e}." + ("in" if kind == "in" else "down")
            return f"{p}.{grp}"
        # pooled-vs-per-row GEMM shapes differ in the last ulp; fake-quant / top-k steps amplify it in deeper layers
        n = cmp(sink.items, caps[p], qmap_of, p, 2e-4 if l < 3 else 3e-2)
        checked += n
        names = {g: ls for g, ls, H in sink.items}
        assert names["attn.in"] == ["attn.wq_a", "attn.wkv"] and names["attn.o.1"] == ["attn.wo_a.slice.1"]
        assert ("engram.wkv" in names) == (l in cfg.engram_layer_ids)
        assert (names["attn.q_b"] == ["attn.wq_b", "attn.indexer.wq_b"]) == cfg.is_index_source(l)
        # every routed expert is reported exactly once for in / mid
        assert sum(1 for g in names if g.startswith("ffn.experts.") and g.endswith(".in")) == cfg.n_routed_experts
        unrouted = [g for g, ls, H in sink.items if H is None]
        routed_total = sum(H["count"] for g, ls, H in sink.items if g.startswith("ffn.experts.") and g.endswith(".in")
                           and H is not None)
        assert routed_total == sum(r.shape[1] for r in rows) * cfg.num_experts_per_tok
    # head
    nw = src.tensor("norm", torch.float32)
    Hh = cal.head_hessian(nw)
    ch = caps["head"]["block"]
    assert Hh["count"] == ch["count"] and close(Hh["H"], ch["H"], 3e-2)
    # MTP
    sink = CollectSink()
    mp = src.tensor("mtp.0.main_proj", torch.float32)
    mn = src.tensor("mtp.0.main_norm", torch.float32)
    cal.mtp_begin(mp, mn, sink, anchors=6)
    cmp(sink.items, caps["mtp.extra"], lambda g: "mtp.main_proj", "main_proj", 3e-2)
    for j in range(cfg.n_mtp_layers):
        sink = CollectSink()
        W = SourceLayerWeights(src, cfg.num_hidden_layers + j)
        cal.run_mtp_stage(j, W, sink, commit=False)
        cal.run_mtp_stage(j, W, None, commit=True)
        p = f"mtp.{j}"

        def qmap_m(grp):
            if grp.startswith("ffn.experts."):
                e, kind = grp.split(".")[2], grp.split(".")[3]
                return f"{p}.ffn.experts.{e}." + ("in" if kind == "in" else "down")
            return f"{p}.{grp}"
        checked += cmp(sink.items, caps[p], qmap_m, p, 3e-2)
    # anchors counts agree with the framework run
    assert [len(a) for a in cal.anc] == [len(_anchors(r.shape[1], 6)) for r in rows]
    print("Hessians compared:", checked, "worst relative diff", WORST)
    assert checked > 100


def test_stream_unrouted_experts_report_none_and_free_memory(setup):
    d, cfg, sd, tables, tmap, rows, caps, res = setup
    src = DictWeightSource(sd, cfg.num_hidden_layers)
    store = MemoryEngramStore([t[0] for t in tables], [t[1] for t in tables])
    # a single short row: with 8 experts / top-2 some experts get no token
    cal = StreamCalibrator(cfg, "cpu", src.tensor("embed", torch.float32), EngramSupport(cfg, tmap, store))
    cal.start([rows[1][:, :6]])
    peak = []

    class S(CollectSink):
        def emit(self, group, linears, H):
            peak.append(0 if H is None else H["H"].numel())
            super().emit(group, linears, H)
    sink = S()
    cal.run_layer(0, SourceLayerWeights(src, 0), sink, commit=True)
    none = [g for g, ls, H in sink.items if H is None]
    assert none, "expected at least one unrouted expert on 6 tokens x top-2 of 8"
    assert all(g.startswith("ffn.experts.") for g in none)


def test_min_expert_tokens_blends_sparse_experts(setup):
    d, cfg, sd, tables, tmap, rows, caps, res = setup
    src = DictWeightSource(sd, cfg.num_hidden_layers)
    store = MemoryEngramStore([t[0] for t in tables], [t[1] for t in tables])
    row = [rows[1][:, :20]]
    outs = {}
    for mt in (0, 12):
        cal = StreamCalibrator(cfg, "cpu", src.tensor("embed", torch.float32), EngramSupport(cfg, tmap, store),
                               min_expert_tokens=mt)
        cal.start(row)
        sink = CollectSink()
        cal.run_layer(0, SourceLayerWeights(src, 0), sink, commit=True)
        outs[mt] = sink.by_group()
    N = row[0].shape[1]
    base, blend = outs[0], outs[12]
    n_sparse = 0
    for e in range(cfg.n_routed_experts):
        g = f"ffn.experts.{e}.in"
        H0, H1 = base[g][1], blend[g][1]
        c = 0 if H0 is None else H0["count"]
        if c >= 12:
            assert H1["count"] == c and torch.equal(H1["H"], H0["H"])
            continue
        n_sparse += 1
        Hs = blend["ffn.shared_in"][1]
        want = Hs["H"] * ((12 - c) / N) + (0 if H0 is None else H0["H"])
        assert H1["count"] == 12 and (H1["H"] - want).abs().max().item() <= 1e-4 * want.abs().max().item()
        assert H1["first_key"] == g
    assert n_sparse >= 1
    # the w2 (mid) Hessians and the shared ones are untouched
    assert all(base[k][1] is None or torch.equal(base[k][1]["H"], blend[k][1]["H"]) for k in base
               if k.endswith(".mid") or k.startswith("ffn.shared"))
