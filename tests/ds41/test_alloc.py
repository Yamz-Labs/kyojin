"""Offline CPU tests of the allocation pipeline: routing statistics, dual-K candidates through the coordinator/worker
split (stub quantizer), the optimizer (against brute force and the LP relaxation) and the re-assembly of a pack from
the chosen candidates."""

import itertools
import json
import os
import random
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

import assemble_pack as AP  # noqa: E402
import capture as CAP  # noqa: E402
import optimize_alloc as OA  # noqa: E402
import plan as P  # noqa: E402
import split as SP  # noqa: E402
import synth_alloc as SA  # noqa: E402
from test_split import N_EXP, common_args, mapfile, run_coord, server  # noqa: E402,F401


# ---------------------------------------------------------------- capture side
def test_routing_stats_counts_and_weights():
    g = torch.Generator().manual_seed(1)
    idx = torch.stack([torch.randperm(10, generator=g)[:3] for _ in range(50)])
    wts = torch.rand(50, 3, generator=g)
    st = CAP.routing_stats(wts, idx, 10, 50)
    assert sum(st["counts"]) == 150 and st["topk"] == 3 and st["n_tokens"] == 50
    for e in range(10):
        m = idx == e
        assert st["counts"][e] == int(m.sum())
        assert st["w2sum"][e] == pytest.approx(float((wts[m].double() ** 2).sum()))


def test_routing_recorder_hooks(tmp_path):
    rec = CAP.RoutingRecorder(str(tmp_path))
    idx = torch.tensor([[0, 1], [1, 2], [1, 3]])
    wts = torch.full((3, 2), 0.5)
    route = rec.wrap_route(lambda cfg, x, gw, gb, k: (wts, idx))
    moe = rec.wrap_moe(lambda *a: torch.ones(3, 4))
    route(None, torch.zeros(3, 8), torch.zeros(4, 8), None, 2)          # no unit set: nothing written
    assert not os.listdir(rec.dir)
    rec.begin("layers.5")
    assert route(None, torch.zeros(3, 8), torch.zeros(4, 8), None, 2) == (wts, idx)
    st = json.load(open(rec.path("layers.5")))
    assert st["counts"] == [1, 3, 1, 1] and st["moe_out_energy"] is None
    moe()
    assert json.load(open(rec.path("layers.5")))["moe_out_energy"] == pytest.approx(4.0)
    rec.end()
    rec.begin("layers.5", skip_existing=True)                            # replay of an assembled layer: keep the file
    route(None, torch.zeros(3, 8), torch.zeros(4, 8), None, 2)
    assert json.load(open(rec.path("layers.5")))["moe_out_energy"] == pytest.approx(4.0)


def test_energy_formula():
    from quant import Exl3Quantizer
    q = object.__new__(Exl3Quantizer)
    q.device = "cpu"
    n, o, cnt = 24, 10, 7
    g = torch.Generator().manual_seed(0)
    X = torch.randn(cnt, n, generator=g)
    W = torch.randn(o, n, generator=g)

    import types
    H = types.SimpleNamespace(raw=(X.T @ X, cnt))
    e = q._energy("k", W, H)
    assert e == pytest.approx(float(((X @ W.T) ** 2).sum()) / cnt, rel=1e-5)
    assert q._energy("k", W, None) is None


def test_choose_alts():
    counts = [100 - i for i in range(20)]            # expert 0 hottest
    prim = {e: 2 for e in range(20)}
    a = SP.choose_alts(counts, prim, [2, 3], up_frac=0.25, down_frac=0.25)
    assert sorted(a) == [0, 1, 2, 3, 4] and all(v == [3] for v in a.values())
    prim3 = {e: 3 for e in range(20)}
    b = SP.choose_alts(counts, prim3, [2, 3], up_frac=0.25, down_frac=0.25)
    assert sorted(b) == [15, 16, 17, 18, 19] and all(v == [2] for v in b.values())
    assert SP.choose_alts(counts, prim, [2, 3], 0.0, 0.0) == {}
    assert SP.choose_alts(counts, prim, [2, 3], 1.0, 0.0).keys() == set(range(20))


# ---------------------------------------------------------------- optimizer
def random_items(n, rnd, n_opt=3):
    it = OA.Items()
    for i in range(n):
        base = rnd.uniform(0.5, 3.0)
        opts = []
        for o in range(n_opt):
            size = rnd.uniform(1, 4) * (o + 1)
            tm = size * rnd.uniform(0.2, 1.0)
            opts.append((f"K{o + 2}:ram", base * 4.0 ** (-o) * rnd.uniform(0.7, 1.3), size, size, tm))
        it.add(f"i{i}", "expert", opts, {"p": 0.1})
    return it.finish()


def brute(it, caps):
    best, bc = float("inf"), None
    for ch in itertools.product(range(it.O), repeat=it.N):
        c = np.array(ch)
        u = OA.usage(it, c)
        if np.all(u <= caps * (1 + 1e-12)):
            d = OA.total_D(it, c)
            if d < best:
                best, bc = d, c
    return best, bc


@pytest.mark.parametrize("seed", range(6))
def test_solver_vs_bruteforce(seed):
    rnd = random.Random(seed)
    it = random_items(9, rnd)
    lo = sum(min(r) for r in it.ram)
    hi = sum(max(r) for r in it.ram)
    caps = np.array([lo + 0.45 * (hi - lo), np.inf, sum(min(r) for r in it.time) + 0.5 *
                     (sum(max(r) for r in it.time) - sum(min(r) for r in it.time))])
    d_opt, _ = brute(it, caps)
    ch, ok = OA.solve(it, caps)
    assert ok
    assert np.all(OA.usage(it, ch) <= caps * (1 + 1e-9))
    assert OA.total_D(it, ch) <= d_opt * 1.25     # tiny instances are lumpy; the LP test below is the tight one


def test_solver_vs_lp_relaxation():
    linprog = pytest.importorskip("scipy.optimize").linprog
    rnd = random.Random(3)
    it = random_items(1500, rnd)
    lo = sum(min(r) for r in it.ram)
    hi = sum(max(r) for r in it.ram)
    tlo = sum(min(r) for r in it.time)
    thi = sum(max(r) for r in it.time)
    caps = np.array([lo + 0.5 * (hi - lo), np.inf, tlo + 0.55 * (thi - tlo)])
    ch, ok = OA.solve(it, caps)
    assert ok
    n, o = it.N, it.O
    c = it.D_.reshape(-1)
    A_eq = np.zeros((n, n * o))
    for i in range(n):
        A_eq[i, i * o:(i + 1) * o] = 1
    A_ub = np.stack([it.A[0].reshape(-1), it.A[2].reshape(-1)])
    r = linprog(c, A_ub=A_ub, b_ub=[caps[0], caps[2]], A_eq=A_eq, b_eq=np.ones(n), bounds=(0, 1), method="highs")
    assert r.status == 0
    assert OA.total_D(it, ch) <= r.fun * 1.01


def test_solver_reports_infeasible():
    it = random_items(6, random.Random(0))
    ch, ok = OA.solve(it, np.array([1e-3, np.inf, np.inf]))
    assert not ok


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("synth"))
    SA.synth_tables(d, n_layers=6, n_exp=64, layer_bits="0-1:3,2-3:2,4-5:3", up=0.5, down=0.5)
    return d


PRM = dict(nu=0.5, w_attn=1.0, w_shared=1.0, layer_weights={}, ks=[], attn_ks=[], bw_ram=200e9, bw_ssd=5e9,
           ssd_hit=0.0, rows=1, smooth=0.05, ram_gib=0.9, runtime_gib=0.0, engram_cache_gib=0.0)


def fixed0(prm):
    return {"disk_gb": 0.0, "ram_gib": 0.0, "read_gb": 0.0, "time_ms": 0.0, "prm": prm}


def test_optimizer_monotone_and_beats_plan(synth):
    tabs = OA.load_tables(synth)
    prm = dict(PRM)
    fx = fixed0(prm)
    base_it = OA.build_items(tabs, prm)
    ram_cap = prm["ram_gib"] * OA.GiB
    base = OA.describe(base_it, OA.prior_choice(base_it, ram_cap, OA.primary_ks(tabs, base_it)), fx)
    res = []
    for gib in (0.5, 0.9, 1.6):
        p = dict(prm, ram_gib=gib)
        it, ch, caps = OA.optimize(tabs, p, fixed0(p), None, None)
        res.append(OA.describe(it, ch, fixed0(p)))
        assert res[-1]["ram_gib_total"] <= gib + 1e-6
    assert res[0]["D"] >= res[1]["D"] >= res[2]["D"] - 1e-12     # more RAM never hurts
    assert res[1]["D"] <= base["D"] + 1e-12                      # the optimum is at least the plan (same RAM)
    # speed constraint binds: lower ceiling target => fewer bytes read
    it, ch, caps = OA.optimize(tabs, prm, fx, None, None)
    t0 = OA.describe(it, ch, fx)["time_ms"]
    it, ch, caps = OA.optimize(tabs, prm, fx, 1.0 / (0.8 * t0 * 1e-3), None)
    r2 = OA.describe(it, ch, fx)
    assert r2["time_ms"] <= 0.8 * t0 * (1 + 1e-6) and r2["D"] >= res[1]["D"] - 1e-12


def test_hot_experts_get_bits_first(synth):
    """Under RAM pressure the optimizer spends K=3 on the hot experts and demotes the cold ones."""
    tabs = OA.load_tables(synth)
    prm = dict(PRM, ram_gib=0.8)
    it, ch, caps = OA.optimize(tabs, prm, fixed0(prm), None, None)
    pk = {}
    for i in range(it.N):
        if it.kind[i] == "expert" and it.meta[i]["unit"] == "layers.2":
            pk[it.meta[i]["e"]] = (it.meta[i]["opts"][ch[i]], it.meta[i]["p"])
    k3 = [p for n, p in pk.values() if n.startswith("K3")]
    k2 = [p for n, p in pk.values() if n.startswith("K2")]
    if k3 and k2:
        assert min(k3) >= np.median(k2)


# ---------------------------------------------------------------- end to end: dual-K through the split, assemble
EXTRA = ["--hessians", "synthetic", "--alt-experts", "2,3", "--alt-up-frac", "0.5", "--alt-down-frac", "0.5",
         "--alt-attn", "3,5", "--alt-shared", "3", "--keep-experts"]


@pytest.fixture(scope="module")
def dual(mapfile, tmp_path_factory):
    out = str(tmp_path_factory.mktemp("dual"))
    run_coord(mapfile, out, extra=EXTRA, delay=0.01)
    return out


def test_candidates_and_tables(mapfile, dual, tmp_path):
    plain = str(tmp_path)
    run_coord(mapfile, plain, workers=("A",), extra=["--hessians", "synthetic"])
    # the primary shard is not touched by the candidates
    assert open(os.path.join(plain, "layers.1.safetensors"), "rb").read() == \
        open(os.path.join(dual, "layers.1.safetensors"), "rb").read()
    plan = json.load(open(os.path.join(dual, "plan.json")))
    prim = {int(e["key"].split(".")[4]): e["bits"] for e in plan["entries"]
            if e["kind"] == "routed" and e["key"].endswith(".w1")}
    st = json.load(open(SP.routing_stats_path(dual, "layers.1")))
    want = SP.choose_alts(st["counts"], prim, [2, 3], 0.5, 0.5)
    assert want
    have = {}
    for e in range(N_EXP):
        for k in (2, 3):
            if os.path.exists(SP.cand_paths(dual, "layers.1", e, k)[1]):
                have.setdefault(e, []).append(k)
    assert have == want
    tab = json.load(open(os.path.join(dual, "alloc", "layers.1.json")))
    assert len(tab["experts"]) == N_EXP and tab["stats"]["n_tokens"] > 0
    for e_s, r in tab["experts"].items():
        e = int(e_s)
        assert set(r["k"]) == {str(prim[e])} | {str(k) for k in want.get(e, [])}
        assert r["count"] == st["counts"][e] and set(r["E"]) == {"w1", "w3", "w2"}
        for k, kr in r["k"].items():
            assert set(kr["proxy"]) == {"w1", "w3", "w2"} and kr["bytes"] > 0
        if len(r["k"]) == 2:       # stub proxies scale like the real quantizer: ~4x per bit
            ks = sorted(r["k"], key=int)
            assert r["k"][ks[0]]["proxy"]["w2"] > 2.5 * r["k"][ks[1]]["proxy"]["w2"]
    # attention / shared candidates
    lin = tab["linears"]
    assert lin["layers.1.attn.wq_a"]["k"].keys() == {"3", "5"}     # plan K 5 + candidate 3 (5 is the primary)
    assert "3" in lin["layers.1.ffn.shared_experts.w1"]["k"]
    assert os.path.exists(os.path.join(dual, "cands", "layers.1", "lin.k3.safetensors"))


def test_optimize_assemble(mapfile, dual, tmp_path):
    from safetensors.torch import load_file
    tabs = OA.load_tables(dual)
    prm = dict(PRM, ram_gib=0.9, runtime_gib=0.0)
    # force a mixed outcome: some experts promoted/demoted, attention variants chosen
    it, ch, caps = OA.optimize(tabs, prm, fixed0(prm), None, None)
    res = OA.describe(it, ch, fixed0(prm))
    alloc = OA.write_alloc(it, ch, tabs, res, prm, str(tmp_path / "alloc.json"))
    eb = alloc["expert_bits"]["layers.1.ffn"]
    plan = json.load(open(os.path.join(dual, "plan.json")))
    pd = {int(e["key"].split(".")[4]): e["bits"] for e in plan["entries"] if e["kind"] == "routed" and e["key"].endswith(".w1")}
    prim = [pd[i] for i in range(N_EXP)]
    # force at least one expert and one module away from the primary so the swap path is exercised
    st = json.load(open(SP.routing_stats_path(dual, "layers.1")))
    prim_d = {i: k for i, k in enumerate(prim)}
    alts = SP.choose_alts(st["counts"], prim_d, [2, 3], 0.5, 0.5)
    e0 = sorted(alts)[0]
    eb[e0] = alts[e0][0]
    alloc["module_bits"]["layers.1.attn.wq_a"] = 3     # plan K is 5 (default attn bits), candidate 3
    dst = str(tmp_path / "pack")
    AP.assemble(dual, alloc, dst, link=True, log=lambda *_: None)
    t = load_file(os.path.join(dst, "layers.1.safetensors"))
    for i, k in enumerate(eb):
        w1 = t[f"layers.1.ffn.experts.{i}.w1.trellis"]
        assert w1.shape[-1] == 16 * k, (i, k, w1.shape)
        assert t[f"layers.1.ffn.experts.{i}.w2.trellis"].shape[-1] == 16 * k
    assert t["layers.1.attn.wq_a.trellis"].shape[-1] == 16 * 3
    m = json.load(open(os.path.join(dst, "layers.1.done.json")))
    assert m["stored"]["layers.1.attn.wq_a"]["bits"] == 3 and m["stored"][f"layers.1.ffn.experts.{e0}.w1"]["bits"] == eb[e0]
    assert m["size"] == os.path.getsize(os.path.join(dst, "layers.1.safetensors"))
    # the source shard is untouched, the index and config describe the result
    assert load_file(os.path.join(dual, "layers.1.safetensors"))["layers.1.attn.wq_a.trellis"].shape[-1] == 16 * 5
    idx = json.load(open(os.path.join(dst, "model.safetensors.index.json")))
    assert idx["weight_map"]["layers.1.attn.wq_a.trellis"] == "layers.1.safetensors"
    qc = json.load(open(os.path.join(dst, "quantization_config.json")))
    assert qc["tensor_storage"]["layers.1.attn.wq_a"]["bits_per_weight"] == 3
    # chosen candidates must exist: asking for a K that was not measured is an error, not a silent fallback
    bad = json.loads(json.dumps(alloc))
    cold = [e for e in range(N_EXP) if e not in alts][0]
    bad["expert_bits"]["layers.1.ffn"][cold] = 6
    with pytest.raises(FileNotFoundError):
        AP.assemble(dual, bad, str(tmp_path / "bad"), log=lambda *_: None)


def test_expert_bits_in_plan_config():
    entries = []
    for e in range(P.N_EXPERTS):
        entries.append({"action": "quant", "kind": "routed", "weight": f"layers.3.ffn.experts.{e}.w1.weight",
                        "bits": 3 if e % 2 else 2})
    eb = P.expert_bits_from_entries(entries)
    assert list(eb) == ["layers.3.ffn"] and eb["layers.3.ffn"][:4] == [2, 3, 2, 3]
    assert P.expert_bits_from_entries(entries[:-1]) == {}


def test_inplace_swap_reversible(mapfile, dual, tmp_path):
    import shutil
    from safetensors.torch import load_file
    work = str(tmp_path / "work")
    shutil.copytree(dual, work, ignore=shutil.ignore_patterns("experts", "spool"))
    sha = lambda d: {f: os.path.getsize(os.path.join(d, "cands", "layers.1", f))
                     for f in os.listdir(os.path.join(d, "cands", "layers.1")) if f.startswith("e")}
    plan = json.load(open(os.path.join(work, "plan.json")))
    pd = {int(e["key"].split(".")[4]): e["bits"] for e in plan["entries"] if e["kind"] == "routed" and e["key"].endswith(".w1")}
    prim = [pd[i] for i in range(N_EXP)]
    st = json.load(open(SP.routing_stats_path(work, "layers.1")))
    alts = SP.choose_alts(st["counts"], dict(enumerate(prim)), [2, 3], 0.5, 0.5)
    new = list(prim)
    for e, ks in alts.items():
        new[e] = ks[0]
    orig = load_file(os.path.join(work, "layers.1.safetensors"))
    n_cands = len(sha(work))
    a1 = {"expert_bits": {"layers.1.ffn": new}, "module_bits": {"layers.1.attn.wq_a": 3, "layers.1.ffn.shared_experts.w1": 3}}
    AP.assemble(work, a1, work, log=lambda *_: None)
    t1 = load_file(os.path.join(work, "layers.1.safetensors"))
    for e, k in enumerate(new):
        assert t1[f"layers.1.ffn.experts.{e}.w3.trellis"].shape[-1] == 16 * k
    assert t1["layers.1.ffn.shared_experts.w1.trellis"].shape[-1] == 16 * 3
    assert len(sha(work)) == n_cands                                   # the displaced variants took the candidates' place
    m = json.load(open(os.path.join(work, "layers.1.done.json")))
    assert m["size"] == os.path.getsize(os.path.join(work, "layers.1.safetensors"))
    assert m["stored"]["layers.1.ffn.shared_experts.w1"]["bits"] == 3
    AP.assemble(work, a1, work, log=lambda *_: None)                  # idempotent: nothing left to swap
    a0 = {"expert_bits": {"layers.1.ffn": prim}, "module_bits": {"layers.1.attn.wq_a": 5, "layers.1.ffn.shared_experts.w1": 5}}
    AP.assemble(work, a0, work, log=lambda *_: None)
    t0 = load_file(os.path.join(work, "layers.1.safetensors"))
    assert set(t0) == set(orig)
    assert all(torch.equal(t0[n], orig[n]) for n in orig)             # back to the conversion's own pack, bit for bit
