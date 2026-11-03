"""
Framework registration of DeepSeek-V4.1 (exllamav3 Config / Model / modules) and the calibrated conversion path, CPU.

  * Config.from_directory resolves the architecture, Model.from_config builds trunk + MTP components, the bit
    allocation (create_q_strategy) sees every quantizable linear, qmaps share Hessians where inputs are shared
  * every checkpoint tensor (but the engram tables) is claimed by a module and survives the conversion
  * the framework blocks equal the verified torch graph block by block (teacher forced), Hessians captured through
    the framework path equal independently computed ones
  * run_conversion (stub quantizer) runs trunk + MTP end to end; counts of captured rows are the expected ones
"""
import json, os, sys
import pytest
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hardware import require
require("ext")
from tiny_ckpt import write_tiny, fw_config
from harness import *
from exllamav3.dsv41 import ops
from exllamav3.dsv41.weights import DictWeightSource
from exllamav3.dsv41.experts import SourceExpertProvider
from exllamav3.dsv41.engram import MemoryEngramStore
from exllamav3.dsv41.model import DSV41Model
from exllamav3.dsv41.convert_driver import run_conversion, StubQuantizer
from exllamav3.model.config import Config
from exllamav3.model.model import Model
from exllamav3.modules import Linear
from exllamav3.conversion.allocation import create_q_strategy
from exllamav3.architecture.architectures import get_architectures


class NoAQ(DictWeightSource):
    """Reference source without the official fp8 activation fake-quant (EXL3 serves fp16 activations)."""

    def linear(self, name, dtype=torch.float32, device="cpu", act_quant=False):
        return super().linear(name, dtype, device, act_quant=False)


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("tiny_ds41"))
    cfg, sd, tables, tmap = write_tiny(d, seed=2)
    return d, cfg, sd, tables, tmap


def ref_model(cfg, sd, tables, tmap):
    src = NoAQ(sd, cfg.num_hidden_layers)
    prov = SourceExpertProvider(src, {l: cfg.moe_counts(l)[0] for l in range(cfg.num_hidden_layers + cfg.n_mtp_layers)})
    store = MemoryEngramStore([t[0] for t in tables], [t[1] for t in tables])
    return DSV41Model(cfg, src, prov, store, token_map=tmap)


def test_registration_and_layout(ckpt):
    d, cfg, *_ = ckpt
    assert "DeepseekV41ForCausalLM" in get_architectures()
    c = Config.from_directory(d)
    assert type(c).__name__ == "DeepseekV41Config" and c.dsv41.num_hidden_layers == 8 and c.dsv41_n_mtp == 2
    assert c.vocab_size == 512 and c.dsv41_moe_mode == "all"
    m = Model.from_config(c)
    keys = [x.key for x in m.modules]
    assert keys == ["embed", "hc_expand"] + [f"layers.{i}" for i in range(8)] + ["norm", "head"]
    assert m.first_block_idx == 2 and m.logit_layer_idx == len(keys) - 1 and m.calibration_all_experts
    mt = Model.from_config(c, component="mtp")
    assert [x.key for x in mt.modules] == ["mtp.0", "mtp.1", "mtp.extra"]
    # layer structure follows the config: kv / index sources own compressor / indexer tensors
    blk = {x.key: x for x in m.modules if x.key.startswith("layers.")}
    assert blk["layers.2"].is_kv and blk["layers.2"].is_idx and blk["layers.3"].idx_wq_b is None
    assert blk["layers.5"].is_idx and not blk["layers.5"].is_kv and blk["layers.1"].engram_pos == 0
    assert blk["layers.5"].engram_pos == 1 and blk["layers.0"].eg_wkv is None
    # qmaps: Hessian sharing where the input is shared
    b2 = blk["layers.2"]
    assert b2.wq_a.qmap == b2.wkv.qmap and b2.wq_b.qmap == b2.idx_wq_b.qmap
    assert len({l.qmap for l in b2.wo_a}) == 2 and b2.ex_w1[0].qmap == b2.ex_w1[3].qmap == b2.ex_w3[1].qmap
    assert len({l.qmap for l in b2.ex_w2}) == b2.n_exp
    # allocation sees every quantizable linear exactly once, with head / mtp bits
    strat, bpw = create_q_strategy(m, mt, c, 3.0, 6, 4, False)
    lins = [x for x in list(m.modules) + list(mt.modules) for x in x if isinstance(x, Linear) and x.qmap]
    assert set(strat) == {l.key for l in lins} and len(strat) == len(lins)
    assert strat["head"] == 6 and strat["mtp.0.attn.wq_a"] >= 4 and strat["mtp.0.main_proj"] >= 4
    assert strat["layers.2.attn.wq_a"] >= 3 and strat["layers.2.ffn.experts.0.w1"] >= 3
    c2 = Config.from_directory(d, dsv41_moe_mode="routed")
    b = [x for x in Model.from_config(c2).modules if x.key == "layers.2"][0]
    assert b.ex_w1[0].qmap != b.ex_w1[1].qmap and b.ex_w1[0].qmap == b.ex_w3[0].qmap


def test_blocks_equal_graph_teacher_forced(ckpt):
    d, cfg, sd, tables, tmap = ckpt
    ref = ref_model(cfg, sd, tables, tmap)
    L = 44
    ids = torch.randint(0, cfg.vocab_size, (1, L), generator=torch.Generator().manual_seed(5))
    taps = {}
    ref.forward(ids, 0, ref.new_state(1, L), taps)
    c = Config.from_directory(d)
    m = Model.from_config(c)
    dev = torch.device("cpu")
    pres = {}
    worst = {}
    for mod in m.modules:
        if not mod.key.startswith("layers."):
            continue
        l = int(mod.key.split(".")[1])
        mod.load(dev)
        x = taps.get(f"L{l}.engram_in", taps[f"L{l}.in"])
        params = {"input_ids": ids, "quant_preserve": pres, "capture": {}, "activate_all_experts": True}
        if l > 0:
            pres.setdefault("dsv41", {"pre_mix": {}, "pool": {}, "idxk": {}, "topk": {}, "cand": None, "main": {}})
            pres["dsv41"]["pre_mix"][l - 1] = taps[f"L{l}.pre"]
        out = mod.forward(x, params)
        mod.unload()
        r = taps[f"L{l}.out"]
        sc = r.abs().max().item()
        diff = (out - r).abs()
        worst[l] = (diff.max().item() / sc, (diff > 1e-4 * sc).float().mean().item())
        assert diff.max().item() <= 2e-2 * sc, (l, diff.max().item(), sc)
        assert (diff > 1e-3 * sc).float().mean().item() <= 0.02, l
    print("worst rel diff / frac > 1e-4 per block:", worst)


def test_framework_hessians_independent(ckpt):
    d, cfg, sd, tables, tmap = ckpt
    ref = ref_model(cfg, sd, tables, tmap)
    L = 44
    ids = torch.randint(0, cfg.vocab_size, (1, L), generator=torch.Generator().manual_seed(8))
    taps = {}
    ref.forward(ids, 0, ref.new_state(1, L), taps)
    c = Config.from_directory(d)
    m = Model.from_config(c)
    dev = torch.device("cpu")
    pres = {"dsv41": {"pre_mix": {}, "pool": {}, "idxk": {}, "topk": {}, "cand": None, "main": {}}}
    ne = cfg.rms_norm_eps
    for mod in m.modules:
        if not mod.key.startswith("layers."):
            continue
        l = int(mod.key.split(".")[1])
        mod.load(dev)
        x = taps.get(f"L{l}.engram_in", taps[f"L{l}.in"])
        cap = {}
        if l > 0:
            pres["dsv41"]["pre_mix"][l - 1] = taps[f"L{l}.pre"]
        mod.forward(x, {"input_ids": ids, "quant_preserve": pres, "capture": cap, "activate_all_experts": True})
        w = ref.lw[l]
        xin = ops.rmsnorm(ops.hc_pre(taps[f"L{l}.in"], taps[f"L{l}.pre"]), w.attn_norm, ne).reshape(-1, cfg.hidden_size)
        Hx = xin.T @ xin
        H = cap[f"layers.{l}.attn.in"]
        assert H["count"] == L and H["first_key"] == f"layers.{l}.attn.wq_a"
        assert (H["H"] - Hx).abs().max().item() <= 2e-3 * Hx.abs().max().item(), l
        assert cap[f"layers.{l}.attn.q_b"]["count"] == L
        assert cap[f"layers.{l}.attn.o.0"]["count"] == L and cap[f"layers.{l}.attn.o_b"]["count"] == L
        assert cap[f"layers.{l}.ffn.shared_in"]["count"] == L
        # 'all' mode: every expert's down projection and the shared gate/up Hessian saw every token
        assert cap[f"layers.{l}.ffn.experts_in"]["count"] == L
        assert all(cap[f"layers.{l}.ffn.experts.{e}.down"]["count"] == L for e in range(mod.n_exp))
        assert ("layers.%d.engram.wkv" % l in cap) == (mod.engram_pos is not None)
        mod.unload()


def _claimed(names):
    out = set()
    for n in names:
        n = n.replace(".scale", ".weight") if n.endswith(".scale") else n
        n = n.replace("wo_a.slice.0", "wo_a")
        out.add(n)
    return out


@pytest.mark.parametrize("moe_mode", ["all", "routed"])
def test_run_conversion_stub_end_to_end(ckpt, moe_mode, tmp_path):
    d, cfg, sd, tables, tmap = ckpt
    g = torch.Generator().manual_seed(3)
    rows = [torch.randint(0, cfg.vocab_size, (1, n), generator=g) for n in (40, 47, 36)]
    logs = []
    res = run_conversion(d, rows, StubQuantizer(), bits=4.0, head_bits=6, mtp_bits=4, moe_mode=moe_mode,
                         mtp_anchors=5, out_dir=str(tmp_path), log=logs.append)
    rep = {r["module"]: r for r in res["report"]}
    n = cfg.num_hidden_layers
    assert list(rep) == ["embed", "hc_expand"] + [f"layers.{i}" for i in range(n)] + ["norm", "head", "mtp.0.main_proj" if False else "mtp.extra", "mtp.0", "mtp.1"]
    toks = sum(r.shape[1] for r in rows)
    # Hessian row counts: every row position reaches the attention input; head sees every position
    assert rep["layers.3"]["hessian_counts"]["layers.3.attn.in"] == toks
    assert rep["head"]["hessian_counts"]["block"] == toks
    if moe_mode == "all":
        assert all(v == toks for k, v in rep["layers.3"]["hessian_counts"].items() if ".experts." in k)
    else:
        # routed: expert inputs are the router assignment, top-k per token overall
        hc = rep["layers.3"]["hessian_counts"]
        assert sum(v for k, v in hc.items() if k.endswith(".in") and ".experts." in k) == toks * cfg.num_experts_per_tok
        assert sum(v for k, v in hc.items() if k.endswith(".down")) == toks * cfg.num_experts_per_tok
    # MTP: draft blocks of dspark_block_size rows per anchor; main_proj sees every target position
    from exllamav3.dsv41.convert_driver import _anchors
    A = [len(_anchors(r.shape[1], 5)) for r in rows]
    blk = cfg.dspark_block_size
    assert rep["mtp.0"]["hessian_counts"]["mtp.0.attn.in"] == sum(A) * blk
    assert rep["mtp.0"]["hessian_counts"]["mtp.0.attn.wkv"] == sum(A) * blk + toks
    assert rep["mtp.extra"]["hessian_counts"]["mtp.main_proj"] == toks
    assert all(r["state_rel_err"] < 1.0 for r in res["report"]), [(r["module"], r["state_rel_err"]) for r in res["report"]]
    assert all(r["proxy_mean"] is None or (0 <= r["proxy_mean"] < 1.0) for r in res["report"])
    # tensor coverage: everything the checkpoint holds (but engram tables) comes back
    out_names = set()
    for k, t in res["tensors"].items():
        out_names |= set(t)
    out_claim = _claimed(out_names)
    ck_claim = _claimed(n for n in sd if ".engram.embed." not in n)
    # slices: out has wo_a.slice.0..G-1 -> collapsed to wo_a.weight; checkpoint has wo_a.weight
    out_claim = {c if ".wo_a.slice." not in c else c.split(".slice.")[0] + ".weight" for c in out_claim}
    missing = ck_claim - out_claim
    extra = out_claim - ck_claim
    assert not missing, sorted(missing)[:10]
    assert not extra, sorted(extra)[:10]
    assert os.path.exists(os.path.join(str(tmp_path), "layers.3.safetensors"))
    assert os.path.exists(os.path.join(str(tmp_path), "driver_report.json"))
    print("\n".join(logs[-6:]))


# ---------------------------------------------------------------------------------------- MXFP4 engram rows
def test_dequant_rows_mxfp4_vs_ml_dtypes():
    """MXFP4 rows: low nibble first, e2m1 value table of ml_dtypes, e8m0 scale per 32 values."""
    import ml_dtypes, numpy as np
    from exllamav3.dsv41.engram import dequant_rows
    g = torch.Generator().manual_seed(1)
    hd = 256
    raw = torch.randint(0, 256, (7, hd // 2 + hd // 32), generator=g, dtype=torch.uint8)
    raw[:, hd // 2:] = torch.randint(118, 130, (7, hd // 32), generator=g, dtype=torch.uint8)
    got = dequant_rows(raw, torch.float32, "mxfp4")
    lut = np.arange(16, dtype=np.uint8).view(ml_dtypes.float4_e2m1fn).astype(np.float32)
    assert lut.tolist() == [0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6]
    b = raw[:, :hd // 2].numpy()
    nib = np.stack([b & 15, b >> 4], -1).reshape(7, hd)
    sc = (2.0 ** (raw[:, hd // 2:].numpy().astype(np.float32) - 127.0))
    ref = (lut[nib].reshape(7, hd // 32, 32) * sc[:, :, None]).reshape(7, hd)
    assert np.array_equal(got.numpy(), ref)


def test_engram_mxfp4_framework_block_and_rawbin_store(tmp_path):
    from exllamav3.dsv41.engram import RawBinEngramStore
    from exllamav3.dsv41.safetensors_io import SafetensorsWeightSource
    d = str(tmp_path / "ckpt")
    cfg, sd, tables, tmap = write_tiny(d, seed=6, engram_fmt="mxfp4")
    assert sd["layers.1.engram.embed.weight"].dtype == torch.uint8 and \
        sd["layers.1.engram.embed.weight"].shape[1] == cfg.engram_head_dim // 2
    # reference graph on the same MXFP4 rows
    src = NoAQ(sd, cfg.num_hidden_layers)
    prov = SourceExpertProvider(src, {l: cfg.moe_counts(l)[0] for l in range(cfg.num_hidden_layers + cfg.n_mtp_layers)})
    store = MemoryEngramStore([t[0] for t in tables], [t[1] for t in tables], fmt="mxfp4")
    ref = DSV41Model(cfg, src, prov, store, token_map=tmap)
    L = 40
    ids = torch.randint(0, cfg.vocab_size, (1, L), generator=torch.Generator().manual_seed(3))
    taps = {}
    ref.forward(ids, 0, ref.new_state(1, L), taps)
    # store 1: from the checkpoint shards (U8 [N, hd/2] detected as mxfp4); store 2: plain .bin files
    bins = tmp_path / "bins"
    bins.mkdir()
    for i, lid in enumerate(cfg.engram_layer_ids):
        tables[i][0].view(torch.uint8).numpy().tofile(str(bins / f"layers.{lid}.engram.embed.weight.bin"))
        tables[i][1].view(torch.uint8).numpy().tofile(str(bins / f"layers.{lid}.engram.embed.scale.bin"))
    st_bin = RawBinEngramStore.open(str(bins), cfg.engram_layer_ids, cfg.engram_head_dim, "mxfp4", allow_buffered=True)
    st_sf = SafetensorsWeightSource(d, cfg.num_hidden_layers, cfg.engram_layer_ids, True).engram_store()
    assert st_sf.fmt == "mxfp4" and st_bin.fmt == "mxfp4"
    rows = torch.tensor([0, 3, 17, tables[0][0].shape[0] - 1])
    mem = store.fetch(0, rows)
    assert torch.equal(st_bin.fetch(0, rows), mem) and torch.equal(st_sf.fetch(0, rows), mem)
    # framework block of the engram layer, teacher forced, both store routes
    for route in ("shards", "bin"):
        if route == "bin":
            os.environ["EXL3_DSV41_ENGRAM_DIR"] = str(bins)
        try:
            c = Config.from_directory(d)
            m = Model.from_config(c)
            blk = [x for x in m.modules if x.key == "layers.1"][0]
            blk.load(torch.device("cpu"))
            pres = {"dsv41": {"pre_mix": {0: taps["L1.pre"]}, "pool": {}, "idxk": {}, "topk": {}, "cand": None, "main": {}}}
            out = blk.forward(taps["L1.engram_in"], {"input_ids": ids, "quant_preserve": pres, "capture": {}})
            r = taps["L1.out"]
            assert (out - r).abs().max().item() <= 1e-3 * r.abs().max().item(), route
            blk.unload()
        finally:
            os.environ.pop("EXL3_DSV41_ENGRAM_DIR", None)


def test_engram_dir_autodetect_fp8_shards_and_mxfp4_bins(tmp_path):
    from safetensors.torch import save_file
    from exllamav3.dsv41.engram import RawBinEngramStore, dequant_rows, detect_engram_format
    g = torch.Generator().manual_seed(2)
    n, hd = 300, 256
    w = torch.randint(0, 120, (n, hd), generator=g, dtype=torch.uint8).view(torch.float8_e4m3fn)
    s = torch.randint(120, 130, (n, hd // 32), generator=g, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    d = tmp_path / "fp8"
    d.mkdir()
    save_file({"layers.1.engram.embed.weight": w, "layers.1.engram.embed.scale": s, "other": torch.zeros(3)},
              str(d / "model-00001-of-00001.safetensors"))
    assert detect_engram_format(str(d)) == "safetensors"
    st = RawBinEngramStore.open(str(d), [1], hd, allow_buffered=True)
    rows = torch.tensor([0, 5, 299, 17])
    assert st.fmt == "fp8"
    got = dequant_rows(st.fetch(0, rows), torch.float32, st.fmt)
    ref = (w.float().view(n, 8, 32) * s.float().unsqueeze(-1)).view(n, hd)[rows]
    assert torch.equal(got, ref)
    b = tmp_path / "bin"
    b.mkdir()
    (b / "layers.1.engram.embed.weight.bin").write_bytes(bytes(n * hd // 2))
    (b / "layers.1.engram.embed.scale.bin").write_bytes(bytes(n * hd // 32))
    assert detect_engram_format(str(b)) == "mxfp4"
    assert RawBinEngramStore.open(str(b), [1], hd, allow_buffered=True).fmt == "mxfp4"
