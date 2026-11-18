"""Hy4-flash whole-model slow path (loader/hy4_forward.py) vs the llama.cpp oracle (CPU, torch only, no extension).
  - Q3_K / Q4_K torch dequant vs a literal python port of ggml dequantize_row_q3_K / q4_K, and vs foreign_quant.gap_formats (q3_k)
  - is_hy4_plan: true for the Hy4 plan, false for a plan with other blocks
  - truncated depth (embedding + layers 0..2, prompt p1): attention vs the oracle dumps (layers 0 / 1, whose input is exact) >= 0.9999,
    layer outputs l_out >= 0.98, engram_out-1 >= 0.999
  - full depth (HY4_FULL=1 and MemAvailable >= 40 GiB): final logits argmax / top-20 overlap vs result_output of call 0, 3 greedy ids
Skips when the GGUF / oracle dumps are absent. Prints counts and cosines only."""
import json, os, sys, types, importlib.util
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
import test_gguf_hy4_hc as T                      # stub packages + registry
from test_gguf_hy4_hc import ROOT, ORACLE, gg
from test_gguf_hy4_attn import mem_avail_gib
from exllamav3.loader import hy4_forward as HF

pytestmark = pytest.mark.skipif(not (os.path.isdir(ORACLE) and os.path.exists(T.GGUF1)), reason="oracle dumps / Hy4 GGUF absent")


# ---------------------------------------------------------------- dequant
def _q3k_c(blk):
    """literal port of ggml dequantize_row_q3_K for one 110-byte block"""
    hm, qs, sc, d = blk[0:32], blk[32:96], blk[96:108].copy(), np.frombuffer(blk[108:110].tobytes(), np.float16)[0].astype(np.float32)
    aux = np.zeros(4, np.uint32)
    aux[:3] = np.frombuffer(sc.tobytes(), np.uint32)
    k1, k2 = np.uint32(0x03030303), np.uint32(0x0f0f0f0f)
    tmp = aux[2]
    aux[2] = ((aux[0] >> 4) & k2) | (((tmp >> 4) & k1) << 4)
    aux[3] = ((aux[1] >> 4) & k2) | (((tmp >> 6) & k1) << 4)
    aux[0] = (aux[0] & k2) | (((tmp >> 0) & k1) << 4)
    aux[1] = (aux[1] & k2) | (((tmp >> 2) & k1) << 4)
    scales = np.frombuffer(aux.tobytes(), np.int8)
    y, q, m, is_ = [], 0, 1, 0
    for n in range(0, 256, 128):
        shift = 0
        for j in range(4):
            for half in range(2):
                dl = d * (int(scales[is_]) - 32)
                is_ += 1
                for l in range(16):
                    ll = l + 16 * half
                    y.append(dl * (int((qs[q + ll] >> shift) & 3) - (0 if hm[ll] & m else 4)))
            shift += 2
            m <<= 1
        q += 32
    return np.array(y, np.float32)


def test_q3k_matches_c_port_and_gap_formats():
    rng = np.random.default_rng(3)
    raw = rng.integers(0, 256, size=(4, 110), dtype=np.uint8)
    raw[:, 108:110] = np.frombuffer(np.array([0.013, 0.07, 0.002, 0.1], np.float16).tobytes(), np.uint8).reshape(4, 2)
    ref = np.stack([_q3k_c(b) for b in raw])
    got = HF.dq_q3k(torch.from_numpy(raw), 4, 256).numpy()
    assert np.abs(got - ref).max() <= 1e-6 * np.abs(ref).max()
    sp = importlib.util.spec_from_file_location("exllamav3.foreign_quant.gap_formats", os.path.join(ROOT, "exllamav3", "foreign_quant", "gap_formats.py"))
    pk = types.ModuleType("exllamav3.foreign_quant"); pk.__path__ = [os.path.join(ROOT, "exllamav3", "foreign_quant")]
    sys.modules["exllamav3.foreign_quant"] = pk
    gf = importlib.util.module_from_spec(sp); gf.__package__ = "exllamav3.foreign_quant"; sys.modules[sp.name] = gf
    sp.loader.exec_module(gf)
    gap = gf.dequant("q3_k", torch.from_numpy(raw)).float().numpy()
    assert np.abs(gap - ref).max() <= 2e-3 * np.abs(ref).max()


def test_q4k_matches_existing_reference():
    import test_gguf_hy4_moe as M
    rng = np.random.default_rng(4)
    raw = rng.integers(0, 256, size=(3, 144), dtype=np.uint8)
    raw[:, 0:4] = np.frombuffer(np.array([[0.02, 0.01]] * 3, np.float16).tobytes(), np.uint8).reshape(3, 4)
    assert torch.equal(HF.dq_q4k(torch.from_numpy(raw), 3, 256), M.dq_q4k(raw, 3, 256))


# ---------------------------------------------------------------- registry predicate
def test_is_hy4_plan_only_for_hy4():
    g, plan = gg()
    assert HF.is_hy4_plan(plan)
    other = types.SimpleNamespace(layers=[{}] * 2, glob={"head_mod": []}, blocks_at=lambda i: ["gqa", "residual", "dense"])
    assert not HF.is_hy4_plan(other)


# ---------------------------------------------------------------- oracle
def _oracle(prompt):
    idx = json.load(open(os.path.join(ORACLE, prompt, "index.json")))
    by = {(t["name"], t["layer"], t["call"]): t for t in idx["tensors"] if t.get("occ", 0) == 0}

    def get(name, layer, call=0):
        t = by.get((name, layer, call))
        if t is None:
            return None
        return torch.from_numpy(np.fromfile(os.path.join(ORACLE, prompt, t["file"]), dtype="<f4")).reshape(tuple(reversed(t["shape_ggml"])))
    return idx, get


def _cos(a, b):
    a, b = a.float().reshape(a.shape[0], -1), b.float().reshape(b.shape[0], -1)
    return float(torch.nn.functional.cosine_similarity(a, b, dim=1).min())


def _run(prompt, layers, gen=0):
    idx, get = _oracle(prompt)
    cmp = {}

    def hook(name, i, t):
        o = get(name, i)
        if o is not None and o.numel() == t.numel():
            cmp[(name, i)] = _cos(t.cpu(), o)
    g, _ = gg()
    M = HF.Hy4Forward(g, n_layers=layers, hook=hook)
    ids = idx["prompt_ids"]
    if gen:
        first, out, _t = M.generate(ids, gen)
        return idx, get, cmp, first, out
    return idx, get, cmp, M.step(ids), []


def test_truncated_depth_vs_oracle():
    idx, get, cmp, _, _ = _run("p1_short_en", 3)
    print({f"{k[0]}-{k[1]}": round(v, 6) for k, v in sorted(cmp.items(), key=lambda kv: (kv[0][1], kv[0][0]))})
    for L in (0, 1):                        # exact oracle inputs: layer 0 (embedding), layer 1 (engram_out is checked below)
        for n in ("attn_kqv", "attn_gated", "attn_out"):
            assert cmp[(n, L)] >= 0.9999, (n, L, cmp[(n, L)])
    assert cmp[("engram_out", 1)] >= 0.999
    for L in (0, 1, 2):
        assert cmp[("l_out", L)] >= 0.98
        assert cmp[("attn_out", L)] >= 0.999


@pytest.mark.skipif(os.environ.get("HY4_FULL") != "1" or mem_avail_gib() < 40, reason="full depth: HY4_FULL=1 and MemAvailable >= 40 GiB")
def test_full_depth_p1():
    idx, get, cmp, first, out = _run("p1_short_en", None, gen=3)
    lo = get("result_output", -1)
    lo = lo.reshape(-1, first.shape[0])[-1]
    ot, tt = set(torch.topk(lo, 20).indices.tolist()), set(torch.topk(first, 20).indices.tolist())
    print("argmax", int(first.argmax()), int(lo.argmax()), "overlap", len(ot & tt), "greedy", out, idx["gen_ids"][:3])
    assert int(first.argmax()) == int(lo.argmax()) and len(ot & tt) >= 14


if __name__ == "__main__":
    sys.exit(pytest.main(["-q", "-s", __file__]))
