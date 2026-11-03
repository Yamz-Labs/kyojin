"""Memmap streamed capture vs the reference StreamCalibrator on the tiny DS4.1 checkpoint (CPU).
Needs the pinned engine: BIGCONV_ENGINE=<dir containing exllamav3/>."""
import pytest, torch, importlib.util

if not __import__("os").environ.get("BIGCONV_ENGINE") or importlib.util.find_spec("exllamav3") is None:
    pytest.skip("pinned DS4.1 engine not on path (set BIGCONV_ENGINE)", allow_module_level=True)
from tiny_ckpt import write_tiny
from exllamav3.dsv41.weights import DictWeightSource
from exllamav3.dsv41.engram import MemoryEngramStore
from exllamav3.dsv41.stream_calib import StreamCalibrator, SourceLayerWeights, EngramSupport, CollectSink
from bigconv.capture_stream import StreamedCapture
from bigconv.runners.ds41 import DS41Runner


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("tiny"))
    cfg, sd, tables, tmap = write_tiny(d, seed=11)
    g = torch.Generator().manual_seed(5)
    rows = [torch.randint(0, cfg.vocab_size, (1, n), generator=g) for n in (45, 38, 52)]
    src = DictWeightSource(sd, cfg.num_hidden_layers)
    store = MemoryEngramStore([t[0] for t in tables], [t[1] for t in tables])
    return cfg, src, store, tmap, rows, tmp_path_factory


def run(setup, chunk, mode, min_tok=0, layers=None):
    cfg, src, store, tmap, rows, tpf = setup
    emb = src.tensor("embed", torch.float32)
    eng = EngramSupport(cfg, tmap, store)
    layers = range(cfg.num_hidden_layers) if layers is None else layers
    out = {}
    if mode == "ref":
        cal = StreamCalibrator(cfg, "cpu", emb, eng, min_expert_tokens=min_tok)
        cal.start(rows)
        for l in layers:
            s = CollectSink()
            cal.run_layer(l, SourceLayerWeights(src, l), s, commit=True)
            out[l] = s.by_group()
        if len(layers) == cfg.num_hidden_layers:
            out["head"] = cal.head_hessian(src.tensor("norm", torch.float32))
        out["h"] = [h.clone() for h in cal.h]
    else:
        r = DS41Runner(cfg, "cpu", emb, eng)
        cap = StreamedCapture(r, str(tpf.mktemp("work")), chunk=chunk, min_expert_tokens=min_tok)
        cap.start(rows)
        for l in layers:
            s = CollectSink()
            cap.run_layer(l, SourceLayerWeights(src, l), s, commit=True)
            out[l] = s.by_group()
        if len(layers) == cfg.num_hidden_layers:
            out["head"] = cap.head_hessian(src.tensor("norm", torch.float32))
        out["h"] = [cap.store.load_h(i) for i in range(len(rows))]
        out["peak_chunk"] = cap.peak_chunk_tokens
        cap.close()
    return out


def compare(a, b, exact, tol=0):
    for l in [k for k in a if isinstance(k, int)]:
        assert a[l].keys() == b[l].keys()
        for g, (lins, Ha) in a[l].items():
            Hb = b[l][g][1]
            assert b[l][g][0] == lins
            if Ha is None:
                assert Hb is None, (l, g)
                continue
            assert Ha["count"] == Hb["count"], (l, g)
            if exact:
                assert torch.equal(Ha["H"], Hb["H"]), (l, g)
            else:
                rel = (Ha["H"] - Hb["H"]).norm() / Ha["H"].norm().clamp_min(1e-12)
                assert rel < tol, (l, g, float(rel))
    for x, y in zip(a["h"], b["h"]):
        assert torch.equal(x, y) if exact else (x - y).norm() / x.norm() < tol
    if "head" not in a:
        return
    Ha, Hb = a["head"], b["head"]
    assert torch.equal(Ha["H"], Hb["H"]) if exact else (Ha["H"] - Hb["H"]).norm() / Ha["H"].norm() < tol


def test_single_chunk_is_bitwise_reference(setup):
    ref = run(setup, 0, "ref")
    got = run(setup, 1 << 20, "mem")
    compare(ref, got, exact=True)


def test_small_chunks_within_tolerance_and_bounded(setup):
    # per-layer drift: one layer -> ~1e-7 (summation order only); it grows slowly through the 8 layers
    ref1 = run(setup, 0, "ref", layers=[0])
    got1 = run(setup, 7, "mem", layers=[0])
    compare(ref1, got1, exact=False, tol=1e-6)
    ref = run(setup, 0, "ref")
    got = run(setup, 7, "mem")
    compare(ref, got, exact=False, tol=5e-2)   # random tiny net amplifies fp32 reorder noise (router flips); logic proven by 1-layer 1e-6 + bitwise single-chunk
    assert got["peak_chunk"] <= 7


def test_min_expert_tokens_blend_matches(setup):
    ref = run(setup, 0, "ref", min_tok=12, layers=[0])
    got = run(setup, 1 << 20, "mem", min_tok=12, layers=[0])
    for g, (_, Ha) in ref[0].items():
        Hb = got[0][g][1]
        if Ha is not None:
            assert torch.equal(Ha["H"], Hb["H"]) and Ha["count"] == Hb["count"], g
