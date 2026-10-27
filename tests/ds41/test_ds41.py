"""CPU-only tests for tools/ds41 (network needed: Range reads against the public repo)."""

import json
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools", "ds41"))

import common as C  # noqa: E402
import dequant as D  # noqa: E402
import fetch as F  # noqa: E402
import plan as P  # noqa: E402
import stream_convert as S  # noqa: E402
import tensor_map as TM  # noqa: E402
import ref_oracle as R  # noqa: E402


@pytest.fixture(scope="session")
def tmap(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("map") / "tensor_map.json")
    try:
        m = TM.build(path, log=lambda *_: None)
    except OSError as e:
        pytest.skip(f"network unavailable: {e}")
    return m


@pytest.fixture(scope="session")
def fetcher(tmap):
    return F.TensorFetcher(tmap)


# ---------------------------------------------------------------- map

def test_map_complete(tmap):
    assert len(tmap["tensors"]) == C.EXPECTED_TENSORS == 96085
    assert len(tmap["shards"]) == C.EXPECTED_SHARDS
    assert sum(t["length"] for t in tmap["tensors"].values()) == C.EXPECTED_TOTAL_SIZE
    for t in tmap["tensors"].values():
        assert t["length"] == C.expected_nbytes(t["dtype"], t["shape"])
        assert t["offset"] + t["length"] <= tmap["shards"][t["file"]]["file_size"]


def test_map_resume(tmap, tmp_path):
    out = str(tmp_path / "m.json")
    part = {k: v for k, v in tmap.items() if k not in ("shards", "tensors")}
    first = sorted(tmap["shards"])[:2]
    part["shards"] = {f: tmap["shards"][f] for f in first}
    part["tensors"] = {n: t for n, t in tmap["tensors"].items() if t["file"] in first}
    json.dump(part, open(out, "w"))
    logs = []
    full = TM.build(out, log=logs.append)
    assert logs[0].startswith("2/48")
    assert len(full["tensors"]) == 96085


# ---------------------------------------------------------------- fetch

def test_fetch_resume_and_sha(fetcher, tmp_path):
    name = "layers.0.attn_norm.weight"
    raw = bytes(fetcher.fetch_bytes(name))
    assert len(raw) == 5120 * 2
    import hashlib
    sha = hashlib.sha256(raw).hexdigest()
    dest = str(tmp_path / "t.bin")
    open(dest + ".part", "wb").write(raw[:1000])  # interrupted download
    fetcher.fetch_to_file(name, dest, sha)
    assert open(dest, "rb").read() == raw
    with pytest.raises(ValueError):
        fetcher.fetch_bytes(name, "0" * 64)


def test_fetch_chunked_equals_single(tmap):
    small = F.TensorFetcher(tmap, chunk=100_000)
    big = F.TensorFetcher(tmap)
    name = "layers.0.attn.wkv.weight"
    assert bytes(small.fetch_bytes(name)) == bytes(big.fetch_bytes(name))


# ---------------------------------------------------------------- dequant

FP4_TENSORS = ["layers.0.ffn.experts.0.w1", "layers.7.ffn.experts.100.w2"]
FP8_TENSORS = ["layers.3.attn.wq_a", "layers.0.attn.wo_a"]


@pytest.mark.parametrize("base", FP4_TENSORS)
def test_dequant_fp4_bit_exact(fetcher, base):
    w = fetcher.fetch(base + ".weight")
    s = fetcher.fetch(base + ".scale")
    assert w.dtype == torch.int8 and s.dtype == torch.float8_e8m0fnu
    got = D.dequant(w, s)
    assert got.dtype == torch.bfloat16 and got.shape == (w.shape[0], w.shape[1] * 2)
    ref = R.ref_fp4_dequant(w, s)
    assert torch.equal(got.view(torch.int16), ref.view(torch.int16))
    rows = [0, 1, w.shape[0] // 2, w.shape[0] - 1]
    slow = R.slow_fp4_dequant(w, s, rows)
    assert torch.equal(got[rows].view(torch.int16), slow.view(torch.int16))


@pytest.mark.parametrize("base", FP8_TENSORS)
def test_dequant_fp8_bit_exact(fetcher, base):
    w = fetcher.fetch(base + ".weight")
    s = fetcher.fetch(base + ".scale")
    assert w.dtype == torch.float8_e4m3fn
    got = D.dequant(w, s)
    ref = R.ref_fp8_dequant(w, s)
    assert torch.equal(got.view(torch.int16), ref.view(torch.int16))
    assert torch.isfinite(got.float()).all()


def test_scale_decode(fetcher):
    s = fetcher.fetch("layers.0.attn.wq_a.scale")
    assert s.dtype == torch.float8_e8m0fnu and s.shape == (40, 160)
    e = s.view(torch.uint8).int()
    assert (e < 255).all()  # 255 would be NaN
    assert torch.equal(s.float(), torch.pow(2.0, (e - 127).float()))


def test_norm_passthrough(fetcher):
    n = fetcher.fetch("layers.0.attn_norm.weight")
    assert n.dtype == torch.bfloat16 and n.shape == (5120,)
    assert torch.equal(D.dequant(n, None).view(torch.int16), n.view(torch.int16))
    assert torch.isfinite(n.float()).all()


# ---------------------------------------------------------------- plan

def _covered(plan, tmap):
    names = set()
    for e in plan["entries"]:
        if e["action"] == "quant":
            names.add(e["weight"])
            if e["scale"]:
                names.add(e["scale"])
        else:
            names.add(e["name"])
    return names


@pytest.mark.parametrize("target", [2.0, 2.3, 2.6])
def test_plan_totals(tmap, target):
    plan = P.build_plan(tmap, target)
    assert _covered(plan, tmap) == set(tmap["tensors"])
    s = plan["summary"]
    assert abs(s["routed_avg_bpw"] - target) < 0.01
    routed = [e for e in plan["entries"] if e["kind"] == "routed"]
    assert len(routed) == 40 * 384 * 3
    assert {e["bits"] for e in routed} <= {2, 3}
    assert all(e["bits"] == 4 for e in plan["entries"] if e["kind"] == "mtp")
    assert all(e["action"] == "copy_raw" for e in plan["entries"] if e["kind"] == "engram")
    eng = sum(e["out_bytes"] for e in plan["entries"] if e["kind"] == "engram")
    assert eng == sum(t["length"] for n, t in tmap["tensors"].items() if ".engram.embed." in n)
    print(f"\nPLAN target={target}: {s['total_gb']:.1f} GB  groups={ {k: round(v, 2) for k, v in s['groups_gb'].items()} }")


def test_plan_uniform_spread_and_freq(tmap, tmp_path):
    plan = P.build_plan(tmap, 2.3)
    for layer in (0, 17, 39):
        hot = {e["key"] for e in plan["entries"] if e["kind"] == "routed" and e["bits"] == 3
               and e["key"].startswith(f"layers.{layer}.ffn.") and e["key"].endswith(".w1")}
        assert len(hot) == round(0.3 * 384)
    # frequency file: counts favour experts 10..(10+n)
    freq = {str(l): {str(e): (1000 if 10 <= e < 10 + 115 else 1) for e in range(384)} for l in range(40)}
    fp = str(tmp_path / "f.json")
    json.dump(freq, open(fp, "w"))
    p2 = P.build_plan(tmap, 2.3, P.load_freq(fp))
    hot = {int(e["key"].split(".")[4]) for e in p2["entries"]
           if e["kind"] == "routed" and e["bits"] == 3 and e["key"].startswith("layers.5.")}
    assert hot == set(range(10, 125))
    assert abs(p2["summary"]["total_bytes"] - plan["summary"]["total_bytes"]) < 1e6


# ---------------------------------------------------------------- stream converter

def test_writer_roundtrip(tmp_path):
    from safetensors import safe_open
    a = torch.arange(10, dtype=torch.float32)
    b = (torch.arange(6) % 255).to(torch.uint8)
    items = [("a", "F32", [10], 40, iter([a.numpy().tobytes()[:7], a.numpy().tobytes()[7:]])),
             ("b", "U8", [2, 3], 6, iter([b.numpy().tobytes()]))]
    p = str(tmp_path / "x.safetensors")
    S.write_streamed(p, items)
    with safe_open(p, "pt") as f:
        assert torch.equal(f.get_tensor("a"), a)
        assert torch.equal(f.get_tensor("b"), b.reshape(2, 3))


def test_dry_run_layers_0_1(tmap, tmp_path):
    mp = str(tmp_path / "map.json")
    json.dump(tmap, open(mp, "w"))
    out = str(tmp_path / "out")
    args = ["--map", mp, "--out", out, "--target", "2.3", "--layers", "0-1", "--dry-run", "--max-experts", "4"]
    S.main(args)
    rep = json.load(open(os.path.join(out, "dry_run_report.json")))
    assert sorted(rep["units"]) == ["layers.0", "layers.1"]
    plan = json.load(open(os.path.join(out, "plan.json")))
    for u in ("layers.0", "layers.1"):
        ents = [e for e in plan["entries"] if e["unit"] == u]
        want_bf16 = sum(e["in"] * e["out"] * 2 for e in ents if e["action"] == "quant") + \
            sum(e["out_bytes"] for e in ents if e["action"] == "copy")
        assert rep["units"][u]["bf16_bytes"] == want_bf16
        assert rep["units"][u]["planned_bytes"] == sum(e["out_bytes"] for e in ents)
    assert rep["units"]["layers.0"]["peak_rss_gb"] < 8
    # resumable: a second run skips finished units
    mtimes = {u: os.path.getmtime(os.path.join(out, "dry_run", f"{u}.json")) for u in rep["units"]}
    S.main(args)
    assert mtimes == {u: os.path.getmtime(os.path.join(out, "dry_run", f"{u}.json")) for u in rep["units"]}
    assert not any(f.endswith(".safetensors") for f in os.listdir(out))


def test_quantize_path_with_stub(tmap, tmp_path):
    """Non-dry path with a stub quantizer (no GPU): shards, markers, resume."""
    from safetensors import safe_open

    def stub(key, w, bits):
        out_f, in_f = w.shape
        return {"trellis": torch.zeros(in_f // 16, out_f // 16, 16 * bits, dtype=torch.int16),
                "suh": torch.zeros(in_f, dtype=torch.float16), "svh": torch.zeros(out_f, dtype=torch.float16),
                "mul1": torch.tensor([0x83DCD12D - (1 << 32)], dtype=torch.int32)}

    plan = P.build_plan(tmap, 2.3)
    plan["entries"] = [e for e in plan["entries"]
                       if e["kind"] != "routed" or int(e["key"].split(".")[4]) < 2]
    ents = [e for e in plan["entries"] if e["unit"] == "layers.0"]
    out = str(tmp_path)
    st = S.process_unit("layers.0", ents, F.TensorFetcher(tmap), out, False, stub, 4, lambda *_: None)
    assert st["planned_bytes"] == sum(e["out_bytes"] for e in ents)
    assert S.unit_done("layers.0", out, False)
    path = os.path.join(out, "layers.0.safetensors")
    assert os.path.getsize(path) == json.load(open(os.path.join(out, "layers.0.done.json")))["size"]
    with safe_open(path, "pt") as f:
        keys = set(f.keys())
        assert "layers.0.attn.wq_a.trellis" in keys and "layers.0.attn.wo_a.slice.7.suh" in keys
        assert "layers.0.ffn.experts.1.w2.trellis" in keys and "layers.0.attn_norm.weight" in keys
        t = f.get_tensor("layers.0.attn.wq_a.trellis")
        assert t.shape == (5120 // 16, 1280 // 16, 16 * 5)
        total = sum(f.get_tensor(k).numel() * f.get_tensor(k).element_size() for k in keys)
    assert total == st["planned_bytes"]
