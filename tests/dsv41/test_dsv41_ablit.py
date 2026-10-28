"""
Runtime refusal-direction hook on DS4.1 (CPU, tiny random model with the real structure: mHC streams, Engram, dspark).

The hook projects the attention output and the MoE output (routed + shared) right before the mHC post/comb mix:
    y <- y - w_L r (r . y)
which is linear in y, so it must equal editing the writers  W <- (I - w_L r r^T) W  of the same layer
(wo_b, shared w2, every routed expert w2). Proofs below:
  * reference graph (DSV41Model) with the hook == the same graph with edited weights, prefill + decode
  * serving graph (DSV41Serving) with the hook == reference with edited weights (prefill + 16 decode steps)
  * all-zero spec: no hook object at all, logits bit-identical to hook off
  * MTP / dspark layers (>= n_layers) never hooked; the spec n_layers bound is honoured
  * off switch, bundled spec resolution (uncensor_spec.json in the model dir)
  * arch table / detect / chat rendering for deepseek_v41
"""
import json
import os
import sys

import pytest
import torch
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "tools", "ablit"))
sys.path.insert(0, os.path.join(ROOT, "tools", "ds41"))
from hardware import require  # noqa: E402
require("ext")
from test_dsv41_serving import make_world, build_ref, build_serving, ids_of  # noqa: E402
from exllamav3.modules import ablit_runtime  # noqa: E402
import arch as A  # noqa: E402


@pytest.fixture
def smooth(monkeypatch):
    """The fp8 / fp4 activation grids flip on a last-bit difference (the edit-in-weights and the hook round in a
    different order), which compounds over layers of a random tiny model; identity grids make the comparison smooth."""
    from exllamav3.dsv41 import quant
    monkeypatch.setattr(quant, "act_quant_fp8_", lambda x, block = 32: x)
    monkeypatch.setattr(quant, "fp4_act_quant_", lambda x, block = 32, e4m3_scale = False: x)


@pytest.fixture(autouse=True)
def _no_triton(monkeypatch):
    monkeypatch.setattr(ablit_runtime, "_USE_TRITON", False)
    monkeypatch.delenv("EXL3_ABLIT_RUNTIME", raising=False)
    ablit_runtime._cache.clear()


@pytest.fixture(scope="module")
def world():
    return make_world(3)


def write_spec(tmp_path, cfg, attn_w, mlp_w, n_layers=None, seed=7, name="spec"):
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(cfg.hidden_size, generator=g, dtype=torch.float64)
    save_file({"r": r.contiguous()}, str(tmp_path / f"{name}.safetensors"))
    spec = {"hidden": cfg.hidden_size, "n_layers": n_layers or cfg.num_hidden_layers,
            "attn_w": attn_w, "mlp_w": mlp_w}
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps(spec))
    return str(p), (r / r.norm())


def edited(sd, cfg, r, attn_w, mlp_w):
    """The dense weight edit W <- (I - w r r^T) W of every writer of the trunk layers."""
    r = r.float()
    out = dict(sd)

    def ed(key, w):
        if w == 0.0:
            return
        W = out[key].float()
        out[key] = W - w * torch.outer(r, r @ W)
    ne = None
    for l in range(cfg.num_hidden_layers):
        ed(f"layers.{l}.attn.wo_b.weight", attn_w[l])
        ne, _ = cfg.moe_counts(l)
        ed(f"layers.{l}.ffn.shared_experts.w2.weight", mlp_w[l])
        for e in range(ne):
            ed(f"layers.{l}.ffn.experts.{e}.w2.weight", mlp_w[l])
    return out


def weights(cfg):
    n = cfg.num_hidden_layers
    return [0.9 - 0.15 * l for l in range(n)], [0.6 + 0.1 * l for l in range(n)]


def test_reference_hook_equals_weight_edit(world, tmp_path, monkeypatch, smooth):
    cfg, sd, tables, tmap = world
    aw, mw = weights(cfg)
    path, r = write_spec(tmp_path, cfg, aw, mw)
    ids = ids_of(cfg, 24)
    ref_edit = build_ref(cfg, edited(sd, cfg, r, aw, mw), tables, tmap)
    monkeypatch.setenv("EXL3_ABLIT_RUNTIME", path)
    hooked = build_ref(cfg, sd, tables, tmap)
    assert hooked.ablit is not None and len(hooked.ablit) == cfg.num_hidden_layers
    monkeypatch.delenv("EXL3_ABLIT_RUNTIME")
    plain = build_ref(cfg, sd, tables, tmap)
    a, b, c = ref_edit.new_state(1, 64), hooked.new_state(1, 64), plain.new_state(1, 64)
    la, _ = ref_edit.forward(ids[:, :20], 0, a)
    lb, _ = hooked.forward(ids[:, :20], 0, b)
    lc, _ = plain.forward(ids[:, :20], 0, c)
    assert (la - lc).abs().max() > 0.05                       # the edit changes the model
    assert (la - lb).abs().max().item() < 1e-4 * la.abs().max().item()
    for i in range(4):
        la, _ = ref_edit.forward(ids[:, 20 + i:21 + i], 20 + i, a)
        lb, _ = hooked.forward(ids[:, 20 + i:21 + i], 20 + i, b)
        assert (la - lb).abs().max().item() < 1e-4 * la.abs().max().item(), i


def test_serving_hook_equals_weight_edit_16_decode_steps(world, tmp_path, monkeypatch, smooth):
    cfg, sd, tables, tmap = world
    aw, mw = weights(cfg)
    path, r = write_spec(tmp_path, cfg, aw, mw)
    L = 43
    ids = ids_of(cfg, L + 16)
    ref_edit = build_ref(cfg, edited(sd, cfg, r, aw, mw), tables, tmap)
    monkeypatch.setenv("EXL3_ABLIT_RUNTIME", path)
    srv = build_serving(cfg, sd, tables, tmap)
    assert srv.ablit is not None
    a, b = ref_edit.new_state(1, 128), srv.new_state(1, 128)
    lr, hr = ref_edit.forward(ids[:, :L], 0, a)
    ls, hs = srv.forward(ids[:, :L], 0, b)
    assert (lr - ls).abs().max().item() < 1e-4 * lr.abs().max().item()
    for i in range(16):
        lr, _ = ref_edit.forward(ids[:, L + i:L + i + 1], L + i, a)
        ls, _ = srv.forward(ids[:, L + i:L + i + 1], L + i, b)
        assert (lr - ls).abs().max().item() < 1e-4 * lr.abs().max().item(), i
        assert int(lr.argmax()) == int(ls.argmax())


def test_zero_spec_is_no_hook_and_bit_identical(world, tmp_path, monkeypatch):
    cfg, sd, tables, tmap = world
    z = [0.0] * cfg.num_hidden_layers
    path, _ = write_spec(tmp_path, cfg, z, z)
    ids = ids_of(cfg, 20)
    off = build_serving(cfg, sd, tables, tmap)
    monkeypatch.setenv("EXL3_ABLIT_RUNTIME", path)
    zero = build_serving(cfg, sd, tables, tmap)
    assert zero.ablit is None
    lo, _ = off.forward(ids, 0, off.new_state(1, 64))
    lz, _ = zero.forward(ids, 0, zero.new_state(1, 64))
    assert torch.equal(lo, lz)


def test_spec_bound_and_mtp_untouched(world, tmp_path, monkeypatch):
    cfg, sd, tables, tmap = world
    n = cfg.num_hidden_layers
    path, _ = write_spec(tmp_path, cfg, [0.5] * n, [0.5] * n, n_layers=n - 1)   # last trunk layer not covered
    monkeypatch.setenv("EXL3_ABLIT_RUNTIME", path)
    srv = build_serving(cfg, sd, tables, tmap)
    assert len(srv.ablit) == n and srv.ablit[n - 1] is None and all(srv.ablit[l] for l in range(n - 1))
    # the dspark stages never go through the hook
    src = open(os.path.join(ROOT, "exllamav3", "dsv41", "dspark.py")).read()
    assert "ablit" not in src and "_abl(" not in src
    assert cfg.n_mtp_layers > 0 and all(l < n for l in range(len(srv.ablit)))


def test_off_switch_and_bundled(world, tmp_path, monkeypatch):
    cfg, sd, tables, tmap = world
    aw, mw = weights(cfg)
    path, _ = write_spec(tmp_path, cfg, aw, mw, name="uncensor_spec")
    os.replace(tmp_path / "uncensor_spec.safetensors", tmp_path / "uncensor_direction.st")
    m = ablit_runtime.prepare_dsv41(cfg.num_hidden_layers, "cpu", str(tmp_path))
    assert m is not None and m[0] is not None                      # bundled spec picked up from the model dir
    monkeypatch.setenv("EXL3_ABLIT_RUNTIME", "off")
    assert ablit_runtime.prepare_dsv41(cfg.num_hidden_layers, "cpu", str(tmp_path)) is None
    monkeypatch.delenv("EXL3_ABLIT_RUNTIME")
    assert ablit_runtime.prepare_dsv41(cfg.num_hidden_layers, "cpu", str(tmp_path / "nothing")) is None


def test_arch_detect_and_chat(tmp_path):
    a = A.get("deepseek_v41")
    assert (a["n_layers"], a["hidden"], a["readout"]) == (40, 5120, "hc_mean")
    assert a["pca_layer"] == round(29 * 39 / 44) and abs(a["proj_f"] - 29.344821947743544 * 39 / 44) < 1e-9
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "deepseek_v41",
                                                      "architectures": ["DeepseekV41ForCausalLM"]}))
    assert A.detect(tmp_path) == "deepseek_v41"
    assert A.detect(tmp_path / "missing") == "glm" and A.get("glm")["n_layers"] == 45
    t = A.chat_wrap("How do I bake bread?", a)
    assert t.startswith("<｜begin▁of▁sentence｜>") and "<｜User｜>How do I bake bread?<｜Assistant｜><think>" in t
    tok = os.path.expanduser("~/ds41_work/ds41ref/data/hf/tokenizer.json")
    if os.path.exists(tok):
        from tokenizers import Tokenizer
        e = Tokenizer.from_file(tok).encode(t, add_special_tokens=False)
        assert e.ids[0] == 0 and e.tokens[-2:] == ["<｜Assistant｜>", "<think>"] and e.ids.count(128804) == 1


def test_mock_directions_and_spec_export_40x5120(tmp_path):
    import subprocess
    T = os.path.join(ROOT, "tools", "ablit")
    run = lambda *x: subprocess.run([sys.executable, *map(str, x)], capture_output=True, text=True, cwd=ROOT)
    r = run(os.path.join(T, "collect_dirs.py"), "--mock", "--arch", "deepseek_v41", "--mock-layers", 40,
            "--mock-hidden", 5120, "--mock-n", 16, "--out", tmp_path / "d")
    assert r.returncode == 0, r.stdout[-500:] + r.stderr[-500:]
    r = run(os.path.join(T, "export_spec.py"), "--dirs", tmp_path / "d", "--arch", "deepseek_v41", "--kernel",
            os.path.join(ROOT, "tools", "ablit", "kernel_glm_ablit5.json"), "--scale-kernel", "--out", tmp_path / "s")
    assert r.returncode == 0, r.stdout[-500:] + r.stderr[-500:]
    s = json.load(open(tmp_path / "s" / "edit_spec.json"))
    assert s["n_layers"] == 40 and s["hidden"] == 5120 and len(s["attn_w"]) == 40 and "dspark" in s["apply"]


def test_search_plumbing_on_tiny_model(world, tmp_path, monkeypatch):
    """search_ds41.run_trials: hook swapped per trial without reload, base == hook off, hook changes the output."""
    import search_ds41 as S
    cfg, sd, tables, tmap = world
    aw, mw = weights(cfg)
    path, _ = write_spec(tmp_path, cfg, aw, mw)
    zero, _ = write_spec(tmp_path, cfg, [0.0] * cfg.num_hidden_layers, [0.0] * cfg.num_hidden_layers, name="zero")
    srv = build_serving(cfg, sd, tables, tmap)
    V = cfg.vocab_size
    encode = lambda s: [1 + (ord(c) % (V - 3)) for c in s][:30]
    decode = lambda ids: " ".join(map(str, ids))
    monkeypatch.setattr(S.E, "is_refusal", lambda t: int(t.split()[0]) % 2 == 0)
    prompts = [{"text": f"how do I do thing number {i} please", "source": "t"} for i in range(6)]
    res = S.run_trials(srv, encode, decode, prompts, prompts[:3], [("zero", zero), ("strong", path)], eos_id=-1,
                       max_new=5, stop_on_pass=False)
    assert [r["name"] for r in res] == ["base", "zero", "strong"]
    assert res[1]["hook_active"] is False and res[1]["kl"] == 0.0 and res[1]["rate"] == res[0]["rate"]
    assert res[2]["hook_active"] is True and res[2]["kl"] > 1e-4
    assert srv.ablit is None                                   # left with the hook off


def test_search_resumable_matches_one_shot(world, tmp_path, monkeypatch):
    """search_ds41.run_resumable: a zero budget stops after the base unit (rc 75), reruns resume and give the same numbers."""
    import search_ds41 as S
    cfg, sd, tables, tmap = world
    aw, mw = weights(cfg)
    path, _ = write_spec(tmp_path, cfg, aw, mw)
    zero, _ = write_spec(tmp_path, cfg, [0.0] * cfg.num_hidden_layers, [0.0] * cfg.num_hidden_layers, name="zero")
    srv = build_serving(cfg, sd, tables, tmap)
    V = cfg.vocab_size
    encode = lambda s: [1 + (ord(c) % (V - 3)) for c in s][:30]
    decode = lambda ids: " ".join(map(str, ids))
    monkeypatch.setattr(S.E, "is_refusal", lambda t: int(t.split()[0]) % 2 == 0)
    prompts = [{"text": f"how do I do thing number {i} please", "source": "t"} for i in range(6)]
    specs = [("zero", zero), ("strong", path)]
    one = S.run_trials(srv, encode, decode, prompts, prompts[:3], specs, eos_id=-1, max_new=5, stop_on_pass=False)
    st = str(tmp_path / "resume")
    calls, rc = 0, 75
    while rc == 75 and calls < 6:
        calls += 1
        res, rc = S.run_resumable(srv, encode, decode, prompts, prompts[:3], specs, -1, st, budget=1e-9, max_new=5,
                                  target_rate=-1.0)
    assert rc == 0 and calls == 3                              # base, zero, strong: one unit per call
    assert [r["name"] for r in res] == ["base", "zero", "strong"]
    assert [round(r["kl"], 6) for r in res] == [round(r["kl"], 6) for r in one]
    assert [r["rate"] for r in res] == [r["rate"] for r in one]
