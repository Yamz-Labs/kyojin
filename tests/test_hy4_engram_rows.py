"""Hy4-flash registry blocks `f8_rows` (ggml F8_E4M3 row type) and `engram_hy4` (n-gram rows + gate), CPU only, no extension.
Loaded by path under stub packages (as test_gguf_generic_cpu.py). Run:
  HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= nice -n 19 python -m pytest tests/test_hy4_engram_rows.py -q -s
Oracle part (needs the Hy4 Q3_K parts + the llama.cpp dump, else skipped; env HY4_GGUF, HY4_ORACLE override the paths;
HY4_PROMPTS=p1_short_en,... and HY4_MAX_CALLS cap the work):
  layer 1 (the only dumped Engram layer): input = c<N>_l_out-0 (stream after layer 0), tokens = prompt_ids + gen_ids,
  target = c<N>_engram_out-1, same call, same positions. Bar: max abs / max|ref| < 1e-3 (the _rel of tests/test_glm_dec2.py:91,
  the only kyo_reg bar on a decoded-vs-reference tensor; no GLM-vs-llama.cpp-dump bar exists in tests/)."""
import importlib.util, json, os, sys, types
import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GGUF = os.environ.get("HY4_GGUF", "/path/to/models/hy4-flash-gguf/Hy4-flash-Q3_K-00001-of-00002.gguf")
ORACLE = os.environ.get("HY4_ORACLE", "/path/to/oracle_out")
BAR = 1e-3


def _stub():
    for name, sub in (("exllamav3", ""), ("exllamav3.loader", "loader")):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = [os.path.join(ROOT, "exllamav3", sub)]
            sys.modules[name] = m


def _load(modname, rel):
    _stub()
    if modname in sys.modules:
        return sys.modules[modname]
    spec = importlib.util.spec_from_file_location(modname, os.path.join(ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = modname.rsplit(".", 1)[0]
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


def rows_mod():
    return _load("exllamav3.loader.engram_rows", "exllamav3/loader/engram_rows.py")


def mem_avail_gib():
    for ln in open("/proc/meminfo"):
        if ln.startswith("MemAvailable"):
            return int(ln.split()[1]) / 2 ** 20
    return 0.0


# ---------------------------------------------------------------- E4M3 / E8M0 exactness
def test_e4m3_table_vs_torch_reference():
    R = rows_mod()
    ref = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()     # torch's own e4m3fn, independent of the table code
    t = R.E4M3
    assert torch.equal(torch.isnan(t), torch.isnan(ref)) and int(torch.isnan(ref).sum()) == 2      # 0x7f, 0xff
    ok = ~torch.isnan(ref)
    assert torch.equal(t[ok].view(torch.int32), ref[ok].view(torch.int32))           # bit-exact incl. -0.0
    assert float(t[0x7E]) == 448.0 and float(t[1]) == 2.0 ** -9


def test_e8m0_table():
    R = rows_mod()
    for e in (0, 1, 100, 127, 128, 254):
        assert float(R.E8M0[e]) == float(np.ldexp(1.0, e - 127))
    assert torch.isnan(R.E8M0[255])


# ---------------------------------------------------------------- layout choice + lookup on synthetic tensors
class FakeG:
    """minimal collection: one file, tensors at given offsets, header base 0"""
    def __init__(self, path, infos):
        self.fds, self.bases, self.infos = [os.open(path, os.O_RDONLY)], [0], infos

    def _pread(self, off, n):
        fi, off = off >> 48, off & ((1 << 48) - 1)
        return np.frombuffer(os.pread(self.fds[fi], n, self.bases[fi] + off), dtype=np.uint8)


def _fake(tmp_path, nbytes, width, rows, tail=0, payload=None):
    p = tmp_path / "f.bin"
    data = payload if payload is not None else np.random.default_rng(0).integers(0, 256, nbytes, dtype=np.uint8)
    p.write_bytes(bytes(data) + bytes(tail))
    return FakeG(str(p), {"per_layer_token_embd.weight": (43, (width, rows), 0)}), data


def _nonan(a):
    a = a.copy(); a[(a & 127) == 127] = 0x38; return a


def test_plain_layout_rows(tmp_path):
    R = rows_mod()
    w, n = 96, 50
    g, data = _fake(tmp_path, w * n, w, n, payload=_nonan(np.random.default_rng(1).integers(0, 256, w * n, dtype=np.uint8)))
    t = R.F8Rows(g)
    assert t.layout == "plain" and t.row_bytes == 96
    ids = torch.tensor([[3, 49], [0, 3]])
    ref = torch.from_numpy(data).view(torch.float8_e4m3fn).float().reshape(n, w)
    out = t.lookup(ids)
    assert out.shape == (2, 2, 96) and torch.equal(out, ref[ids])
    with pytest.raises(IndexError):
        t.lookup([50])


def test_blocked_layout_rows(tmp_path):
    R = rows_mod()
    w, n = 96, 40
    rng = np.random.default_rng(2)
    rows = np.concatenate([_nonan(rng.integers(0, 256, (n, w), dtype=np.uint8)), rng.integers(110, 140, (n, w // 32), dtype=np.uint8)], 1)
    g, _ = _fake(tmp_path, 0, w, n, payload=rows.reshape(-1))
    t = R.F8Rows(g)
    assert t.layout == "blocked" and t.row_bytes == 99
    # independent reference: tree dequant_rows (torch e4m3fn / e8m0fnu dtypes), exllamav3/dsv41/engram.py:393 formula
    v = torch.from_numpy(rows[:, :w].copy()).view(torch.float8_e4m3fn).float().unflatten(-1, (-1, 32))
    s = torch.from_numpy(rows[:, w:].copy()).view(torch.float8_e8m0fnu).float()
    ref = (v * s.unsqueeze(-1)).flatten(-2)
    ids = torch.tensor([0, 7, 39, 7])
    assert torch.equal(t.lookup(ids), ref[ids])
    # real DS4.1 width: 256 values + 8 scales = 264 B
    r2 = np.concatenate([_nonan(rng.integers(0, 256, (5, 256), dtype=np.uint8)), rng.integers(120, 135, (5, 8), dtype=np.uint8)], 1)
    g2, _ = _fake(tmp_path, 0, 256, 5, payload=r2.reshape(-1))
    t2 = R.F8Rows(g2)
    assert t2.layout == "blocked" and t2.row_bytes == 264


def test_layout_padding_and_refusal(tmp_path):
    R = rows_mod()
    # ggml pads the tensor to 32 B: 96*50 = 4800 (already a multiple), 33*... use an odd count
    assert R.f8_layout(96 * 5, 480, 96) == "plain"
    assert R.f8_layout(96 * 5, 495, 96) == "blocked" and R.f8_layout(96 * 5, 512, 96) == "blocked"
    for bad in (479, 500, 1000):
        with pytest.raises(R.F8LayoutError, match="neither plain"):
            R.f8_layout(96 * 5, bad, 96)
    assert R.f8_layout(256 * 10, 264 * 10, 256) == "blocked"             # the DS4.1 264 B / 256-element row
    with pytest.raises(R.F8LayoutError, match="neither plain"):
        R.f8_layout(256 * 10, 265 * 10, 256)
    w, n = 96, 10
    g, _ = _fake(tmp_path, w * n + 40, w, n)
    with pytest.raises(R.F8LayoutError, match="refusing"):
        R.F8Rows(g)


def test_registry_and_loader_wiring():
    cfgm = _load("exllamav3.loader.gguf_config", "exllamav3/loader/gguf_config.py")
    assert cfgm.GGML_FMT[43] == "f8_e4m3"
    reg = _load("exllamav3.loader.gguf_registry", "exllamav3/loader/gguf_registry.py")
    importlib.import_module("exllamav3.loader.gguf_blocks.engram")      # registers engram / engram_hy4 / f8_rows
    B = reg.REGISTRY

    class G:  # header view
        def __init__(self, hy4):
            self.meta = {"general.architecture": "x", "x.block_count": 3, "x.ple.layers": [1], "x.ple.ngram_size": 3, "x.ple.heads_per_ngram": 8,
                         "x.embedding_length_per_layer_input": 96}
            names = ["blk.1.engram_kv.weight", "blk.1.engram_gate.weight"] + (["blk.1." + s for s in ("engram_kv_b.weight", "engram_lookup_norm.weight",
                     "engram_pre_gamma.weight", "engram_hc_scale.weight")] if hy4 else [])
            self.infos = {n: (0, (4, 4), 0) for n in names}
    for hy4 in (True, False):
        ctx = reg.Ctx(G(hy4))
        assert B["engram_hy4"].select(ctx, 1) is hy4 and B["engram"].select(ctx, 1) is (not hy4)
        assert not B["engram_hy4"].select(ctx, 0) and not B["engram"].select(ctx, 2)
    assert B["engram_hy4"].build is not None and B["f8_rows"].build is not None and B["engram"].build is None


# ---------------------------------------------------------------- oracle
def _prompts():
    sel = os.environ.get("HY4_PROMPTS", "p1_short_en,p4_long_n700,p5_long_n4000").split(",")
    return [s for s in sel if os.path.isfile(f"{ORACLE}/{s}/index.json")]


def _have_oracle():
    return os.path.isfile(GGUF) and os.path.isdir(ORACLE)


def run_oracle(prompt, R, eng, g, max_calls, tag=""):
    idx = json.load(open(f"{ORACLE}/{prompt}/index.json"))
    tokens = list(idx["prompt_ids"]) + list(idx["gen_ids"])
    by = {(t["name"], t["layer"], t["call"]): t for t in idx["tensors"]}
    rows = []
    calls = sorted({c for (n, l, c) in by if n == "engram_out" and l == 1})[:max_calls]
    for c in calls:
        te, tl = by[("engram_out", 1, c)], by[("l_out", 0, c)]
        assert te["positions"] == tl["positions"] and te["shape_ggml"] == tl["shape_ggml"], (prompt, c)
        n = len(te["positions"])
        rd = lambda t: np.fromfile(f"{ORACLE}/{prompt}/{t['file']}", dtype=np.float32).reshape(n, 4, 3072)
        x, ref = torch.from_numpy(rd(tl)), torch.from_numpy(rd(te))
        out = torch.empty_like(ref)
        for k, pos in enumerate(te["positions"]):                      # non-contiguous positions: one token at a time
            out[k:k + 1] = eng(x[k:k + 1], tokens, pos)
        ma = float((out - ref).abs().max())
        rel = ma / float(ref.abs().max())
        dl = float((ref - x).abs().max())
        rows.append((prompt + tag, c, n, ma, rel, ma / dl))
    return rows


@pytest.mark.skipif(not _have_oracle(), reason="Hy4 GGUF / oracle dump not on this host")
def test_engram_layer1_vs_oracle():
    if mem_avail_gib() < 14:
        pytest.skip(f"MemAvailable {mem_avail_gib():.1f} GiB < 14")
    R = rows_mod()
    gc = _load("exllamav3.loader.gguf_collection", "exllamav3/loader/gguf_collection.py")
    g = gc.GGUFCollection(GGUF)
    table = R.F8Rows(g)
    assert table.layout == "plain" and table.row_bytes == 96 and table.rows == 660006416
    eng = R.Hy4Engram(g, 1, table=table)
    engq = R.Hy4Engram(g, 1, table=table, q8_act=True)
    mc = int(os.environ.get("HY4_MAX_CALLS", "40"))
    allrows, qrows = [], []
    for p in _prompts():
        allrows += run_oracle(p, R, eng, g, mc)
        qrows += run_oracle(p, R, engq, g, mc, tag="+q8act")
    print()
    print("prompt call ntok max_abs rel(maxabs/max|ref|) rel_to_delta bar PASS/FAIL  [layer 1: l_out-0 -> engram_out-1]")
    for p, c, n, ma, rel, rd in allrows + qrows:
        print(f"{p} c{c:03d} n={n} {ma:.3e} {rel:.3e} {rd:.3e} {BAR:.0e} {'PASS' if rel < BAR else 'FAIL'}")
    print("worst exact-fp32 rel %.3e, worst q8-act-emulated rel %.3e" % (max(r[4] for r in allrows), max(r[4] for r in qrows)))
    # llama.cpp CPU quantises the activation of a Q8_0 mul_mat to Q8_0: the engine's exact-fp32 path differs from the dump by that
    # (~1e-3 of max|ref|); the emulated path must meet the bar, the exact path is only bounded (5e-3)
    assert qrows and all(r[4] < BAR for r in qrows)
    assert all(r[4] < 5e-3 for r in allrows)
