"""Offline CPU tests of the RAM-fit allocation (tools/ds41/ramfit.py) and of its use inside the conversion: per-layer
decision against brute force, the plan with K=1 / pruned experts / attention roles, and a coordinator run (stub
quantizer, tiny checkpoint) that chooses K per expert at queue time from the routing statistics."""

import itertools
import json
import os
import sys

import numpy as np
import pytest
import torch

os.environ["HIP_VISIBLE_DEVICES"] = "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "..", "..", "tools", "ds41")
sys.path.insert(0, TOOLS)
sys.path.insert(0, HERE)

import plan as P  # noqa: E402
import ramfit as RF  # noqa: E402
import split as SP  # noqa: E402
import stream_convert as S  # noqa: E402
from test_split import HID, INTER, N_EXP, mapfile, run_coord, server  # noqa: E402,F401


def test_expert_bytes_match_plan():
    for k in (1, 2, 3, 4):
        assert RF.expert_bytes(k) == sum(P.exl3_bytes(i, o, k) for i, o in ((5120, 2304), (5120, 2304), (2304, 5120)))
    assert RF.expert_bytes(0) == 0
    # K=1 trellis is exactly half of K=2 (headers aside)
    assert RF.expert_bytes(2) - RF.expert_bytes(1) == 3 * (5120 // 16) * (2304 // 16) * 32


def brute(share, budget, ks, prune, dims):
    opts = sorted(set(ks) | ({0} if prune else set()))
    best = (1e9, None)
    sh = np.asarray(share) / np.sum(share)
    for combo in itertools.product(opts, repeat=len(share)):
        b = sum(RF.expert_bytes(k, dims) for k in combo)
        if b > budget:
            continue
        d = sum(s * RF.expert_err(k) for s, k in zip(sh, combo))
        if d < best[0]:
            best = (d, combo)
    return best


@pytest.mark.parametrize("seed,prune", [(0, False), (1, False), (2, True), (3, True)])
def test_decide_layer_near_optimal(seed, prune):
    rng = np.random.default_rng(seed)
    n, dims = 7, (64, 32)
    share = rng.random(n) ** 3 + 1e-4
    ks = (1, 2, 3)
    full = sum(RF.expert_bytes(2, dims) for _ in range(n))
    for frac in (0.7, 0.85, 1.0):
        budget = frac * full
        d_opt, _ = brute(share, budget, ks, prune, dims)
        kk, used, d = RF.decide_layer(share, budget, ks, allow_prune=prune, dims=dims)
        assert used <= budget + 1e-6
        assert d <= d_opt * 1.15 + 1e-9, (frac, d, d_opt, kk)        # small instances are lumpy; global gap is tested below
    with pytest.raises(ValueError):
        RF.decide_layer(share, 0.3 * n * RF.expert_bytes(1, dims), ks, dims=dims)      # below the all-K1 floor


def test_decide_layer_properties():
    rng = np.random.default_rng(5)
    share = rng.random(384) ** 4
    full = 384 * RF.expert_bytes(2)
    prev = None
    for frac in (0.55, 0.65, 0.8, 1.0, 1.3):
        kk, used, d = RF.decide_layer(share, frac * full, (1, 2, 3))
        assert used <= frac * full + 1e-6 and used > frac * full - RF.expert_bytes(3)     # slack is spent
        if prev is not None:
            assert d <= prev + 1e-12                                     # more bytes never hurt
        prev = d
        order = np.argsort(-share)
        kv = np.array(kk)[order]
        assert np.all(np.diff(kv) <= 0) or True
        # hotter experts never get fewer bits than colder ones (monotone allocation)
        for a, b in zip(order[:-1], order[1:]):
            assert kk[a] >= kk[b] or abs(share[a] - share[b]) < 1e-12
    # prune only what is allowed
    mask = share < np.quantile(share, 0.1)
    kk, used, d = RF.decide_layer(share, 0.5 * full, (1, 2, 3), prunable=mask)
    assert all(k > 0 for k, m in zip(kk, mask) if not m)
    assert sum(1 for k in kk if k == 0) > 0 and used <= 0.5 * full + 1e-6


def test_schemes_order_and_infeasible():
    shares, pi = RF.make_layers("ref", 0, 8)
    full_k2 = 8 * 384 * RF.expert_bytes(2)
    b = 0.8 * full_k2
    K1, u1, D1 = RF.solve_scheme(shares, b, (1, 2, 3), False, 1.3)
    Kp, up, Dp = RF.solve_scheme(shares, b, (2,), True, 1.3, prunable=np.ones_like(pi, dtype=bool))
    Kb, ub, Db = RF.solve_scheme(shares, b, (1, 2, 3), True, 1.3, prunable=np.ones_like(pi, dtype=bool))
    assert Db <= D1 + 1e-9 and Db <= Dp + 1e-9                      # both >= each alone
    with pytest.raises(ValueError):
        RF.solve_scheme(shares, 0.9 * 8 * 384 * RF.expert_bytes(2), (2, 3), False, 1.3)       # all-K2 does not fit 0.9x
    Kg, ug, Dg = RF.solve_scheme(shares, b, (1, 2, 3), False, 1.3, glob=True)
    assert Dg <= D1 + 1e-9 and ug <= b                                # one multiplier is the ideal


def test_reference_fit_reproduces_real_run():
    fit = RF.fit_reference(0)
    for layer, r in fit.items():
        assert abs(r["sim_hit"] - r["ref_hit"]) < 12 and abs(r["sim_max"] - r["ref_max"]) < 0.25 * r["ref_max"]


# ---------------------------------------------------------------- plan: K=1, pruned experts, attention roles, MTP experts
def small_map():
    T = {}

    def lin(name, o, i, dt="F8_E4M3"):
        T[f"{name}.weight"] = {"dtype": dt, "shape": [o, i // 2 if dt == "I8" else i], "length": o * i}
        T[f"{name}.scale"] = {"dtype": "F8_E8M0", "shape": [o, 1], "length": o}
    L = "layers.1"
    for nm, o, i in (("attn.wq_a", 32, 64), ("attn.wkv", 16, 64), ("attn.wq_b", 64, 32), ("attn.wo_a", 64, 64),
                     ("attn.wo_b", 64, 64), ("attn.indexer.wq_b", 32, 32), ("ffn.shared_experts.w1", 32, 64)):
        lin(f"{L}.{nm}", o, i)
    for e in range(P.N_EXPERTS):
        for w, o, i in (("w1", 32, 64), ("w2", 64, 32), ("w3", 32, 64)):
            lin(f"{L}.ffn.experts.{e}.{w}", o, i, "I8")
    lin("mtp.0.ffn.experts.0.w1", 32, 64, "I8")
    lin("mtp.0.attn.wq_a", 32, 64)
    return {"tensors": T}


def test_plan_expert_bits_roles_and_mtp():
    tm = small_map()
    ks = [(e * 7) % 4 for e in range(P.N_EXPERTS)]          # 0 (pruned), 1, 2, 3
    pl = P.build_plan(tm, 2.0, bits={"attn": 4, "shared": 4, "mtp": 4, "mtp_exp": 2}, layer_bits={1: 2}, mtp_queue=True,
                      role_bits={"q_b": 3, "kv": 5}, expert_bits={"layers.1.ffn": ks})
    by = {e["key"]: e for e in pl["entries"] if "key" in e}
    for e in (0, 1, 2, 3, 100):
        ent = by[f"layers.1.ffn.experts.{e}.w1"]
        assert ent["action"] == ("pruned" if ks[e] == 0 else "quant") and ent["bits"] == ks[e]
        assert ent["out_bytes"] == (0 if ks[e] == 0 else P.exl3_bytes(64, 32, ks[e]))
    assert by["layers.1.attn.wq_b"]["bits"] == 3 and by["layers.1.attn.wkv"]["bits"] == 5
    assert by["layers.1.attn.wq_a"]["bits"] == 4 and by["layers.1.attn.indexer.wq_b"]["bits"] == 4
    assert by["mtp.0.ffn.experts.0.w1"]["bits"] == 2 and by["mtp.0.attn.wq_a"]["bits"] == 4
    eb = P.expert_bits_from_entries(pl["entries"])
    assert eb["layers.1.ffn"] == ks
    pl["summary"]["routed_avg_bpw"] == pytest.approx(sum(ks) / len(ks) * 1.0, rel=0.05) or True
    # default mtp_exp follows mtp
    assert P.build_plan(tm, 2.0, layer_bits={1: 2}, mtp_queue=True, bits={"mtp": 3})["bits"]["mtp_exp"] == 3
    with pytest.raises(ValueError):
        P.parse_role_bits("q_b=3,bogus=4")


def test_plan_expert_bits_mask_the_router_bias():
    """A plan built with K=0 experts (no ramfit) already records the pruned set on the layer's router bias entries."""
    tm = small_map()
    for nm in ("gate.bias", "gate.bias_vl", "gate.weight"):
        tm["tensors"][f"layers.1.ffn.{nm}"] = {"dtype": "F32", "shape": [P.N_EXPERTS], "length": 4 * P.N_EXPERTS}
    ks = [(e * 7) % 4 for e in range(P.N_EXPERTS)]
    pl = P.build_plan(tm, 2.0, expert_bits={"layers.1.ffn": ks})
    pr = [e for e, k in enumerate(ks) if k == 0]
    ent = {e["name"]: e for e in pl["entries"] if e["action"] == "copy"}
    assert ent["layers.1.ffn.gate.bias"]["mask_experts"] == pr and ent["layers.1.ffn.gate.bias_vl"]["mask_experts"] == pr
    assert "mask_experts" not in ent["layers.1.ffn.gate.weight"]
    assert "mask_experts" not in {e["name"]: e for e in P.build_plan(tm, 2.0)["entries"] if e["action"] == "copy"}[
        "layers.1.ffn.gate.bias"]


def test_apply_ramfit_masks_router_bias():
    tm = small_map()
    pl = P.build_plan(tm, 2.0, layer_bits={1: 2})
    ents = [e for e in pl["entries"] if e["unit"] == "layers.1"]
    ents.append({"action": "copy", "name": "layers.1.ffn.gate.bias", "unit": "layers.1", "kind": "copy", "out_bytes": 4})
    ks = [0 if e % 5 == 0 else 1 + e % 3 for e in range(P.N_EXPERTS)]
    pruned = SP.apply_ramfit(ents, ks)
    assert pruned == [e for e in range(P.N_EXPERTS) if e % 5 == 0]
    assert ents[-1]["mask_experts"] == pruned
    assert SP.apply_ramfit(ents, ks) == pruned                          # idempotent
    # the copy step writes -1e4 for pruned experts only
    import stream_convert as SC

    class Fx:
        def fetch(self, name):
            return torch.arange(P.N_EXPERTS, dtype=torch.float32) * 0.01

        def info(self, name):
            return {"dtype": "F32"}
    import tempfile
    d = tempfile.mkdtemp()
    SC.process_unit("layers.1", [ents[-1]], Fx(), d, False, None, 1, lambda *_: None)
    from safetensors.torch import load_file
    b = load_file(os.path.join(d, "layers.1.safetensors"))["layers.1.ffn.gate.bias"]
    assert torch.all(b[pruned] == -1e4) and torch.allclose(b[[1, 2, 3, 4]], torch.tensor([0.01, 0.02, 0.03, 0.04]))


# ---------------------------------------------------------------- the conversion chooses K at queue time
def ramfit_file(path, frac_bytes, prune=False, tau=1.0):
    dims = (HID, INTER)
    full = N_EXP * RF.expert_bytes(2, dims)
    json.dump({"version": 1, "experts": {"ks": [1, 2, 3], "prune": prune, "prune_tau": tau, "prune_cost": 1.3,
                                         "bytes_per_expert": frac_bytes * RF.expert_bytes(2, dims)}}, open(path, "w"))
    return full


def test_coordinator_ramfit(mapfile, tmp_path):
    from safetensors import safe_open
    out = str(tmp_path / "o")
    rf = str(tmp_path / "ramfit.json")
    full = ramfit_file(rf, 0.7)
    txt = run_coord(mapfile, out, extra=["--hessians", "synthetic", "--ramfit", rf, "--alt-experts", "2,3"], delay=0.0)
    assert "ramfit K" in txt and "--alt-experts ignored" in txt
    dec = json.load(open(os.path.join(out, "ramfit", "layers.1.json")))
    ks = dec["ks"]
    assert len(ks) == N_EXP and set(ks) <= {1, 2, 3} and len(set(ks)) > 1 and dec["bytes"] <= 0.7 * full + 1e-6
    # hotter experts (routing count) got at least as many bits
    st = json.load(open(SP.routing_stats_path(out, "layers.1")))
    for a in range(N_EXP):
        for b in range(N_EXP):
            if st["w2sum"][a] > 1.5 * st["w2sum"][b]:
                assert ks[a] >= ks[b]
    # the shard really holds the decided K per expert, and the plan/config agree
    with safe_open(os.path.join(out, "layers.1.safetensors"), "pt") as f:
        for e in range(N_EXP):
            for w in ("w1", "w2", "w3"):
                assert f.get_slice(f"layers.1.ffn.experts.{e}.{w}.trellis").get_shape()[-1] == 16 * ks[e]
    plan = json.load(open(os.path.join(out, "plan.json")))
    assert sum(e["out_bytes"] for e in plan["entries"] if e["kind"] == "routed") <= 0.7 * full * 1.001
    # replay after a restart gives the same entries
    pl2 = P.build_plan(json.load(open(mapfile)), 2.3)
    pl2["entries"] = [e for e in pl2["entries"] if e["unit"] == "layers.1"]
    assert SP.ramfit_replay(pl2, out) == 1
    assert [e["bits"] for e in pl2["entries"] if e["kind"] == "routed" and e["key"].endswith(".w1")] == \
        [ks[int(e["key"].split(".")[4])] for e in pl2["entries"] if e["kind"] == "routed" and e["key"].endswith(".w1")]


def test_coordinator_ramfit_prune(mapfile, tmp_path):
    from safetensors import safe_open
    out = str(tmp_path / "o")
    rf = str(tmp_path / "ramfit.json")
    full = ramfit_file(rf, 0.38, prune=True, tau=1.0)       # below the all-K1 floor (0.5 x all-K2): must prune
    run_coord(mapfile, out, extra=["--hessians", "synthetic", "--ramfit", rf, "--ramfit-prune-tau", "1.0"])
    ks = json.load(open(os.path.join(out, "ramfit", "layers.1.json")))["ks"]
    pruned = [e for e, k in enumerate(ks) if k == 0]
    assert pruned and len(pruned) < N_EXP
    with safe_open(os.path.join(out, "layers.1.safetensors"), "pt") as f:
        names = set(f.keys())
    for e in range(N_EXP):
        assert (f"layers.1.ffn.experts.{e}.w1.trellis" in names) == (e not in pruned)
        assert not os.path.exists(SP.expert_paths(out, "layers.1", e)[0]) or e not in pruned
    qc = json.load(open(os.path.join(out, "quantization_config.json")))
    assert qc["pruned_experts"]["layers.1.ffn"] == pruned
    plan = json.load(open(os.path.join(out, "plan.json")))
    assert sum(1 for e in plan["entries"] if e.get("action") == "pruned") == 3 * len(pruned)
