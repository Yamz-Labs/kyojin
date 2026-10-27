"""CPU test of the scan-resistant admission of the expert store (EXL3_EXPERT_PROTECT).

A decode phase builds a hot set; long prefill-sized scans (every expert of every layer) follow; decode resumes. Checks:
  * the decode-hot set survives the scans (protect on: every hot expert is still resident at the first decode step after each scan);
  * bitwise identity: every picked expert is resident with the right bytes after its call, protect on or off (the policy only
    decides WHERE bytes live);
  * budget accounting: free + owned slots == pool, owner maps agree with the layer maps, the stream region never exceeds one
    layer union (largest group), stream slots are never counted as stable;
  * scan bytes read stay within a few percent of the unprotected policy; a decode hit promotes a stream slot.
Run: python tests/test_expert_store_protect_cpu.py
"""
import importlib.util
import os
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(__file__)
T = importlib.util.spec_from_file_location("t_es_cpu", os.path.join(HERE, "test_expert_store_cpu.py"))
base = importlib.util.module_from_spec(T)
T.loader.exec_module(base)
es, E, LAYERS = base.es, base.E, base.LAYERS
POOL, HOT, ROWS_PF, ROWS_DEC = 66, 8, 80, 4


def check_bytes(st, lt, stc, data, picks):
    for e in np.unique(picks):
        s = int(lt.smap[e])
        assert s >= 0, "current layer expert evicted"
        for proj, t0 in (("gate_proj", 0), ("up_proj", 3), ("down_proj", 6)):
            for pj, part in enumerate(es.PARTS):
                key = f"{lt.key}.experts.{e}.{proj}.{part}"
                fn = stc.tensor_file_map[key]
                b0, b1 = stc.file_headers[fn][key]["data_offsets"]
                got = base.read_mem(int(lt.tab[(t0 + pj) * E + e]), b1 - b0)
                assert got == data[fn][1][b0:b1], f"bytes differ {key}"


def accounting(st, lts):
    cls = lts[0].cls
    owned = int((cls.owner_layer >= 0).sum())
    assert owned + len(cls.free) == cls.nslots, (owned, len(cls.free), cls.nslots)
    assert len(set(cls.free)) == len(cls.free)
    for s in np.flatnonzero(cls.owner_layer >= 0):
        lt = st.layers[int(cls.owner_layer[s])]
        assert int(lt.smap[int(cls.owner_expert[s])]) == s, "owner map and layer map disagree"
    assert sum(int((l.smap >= 0).sum()) for l in lts) == owned
    if cls.scr_t is not None:
        strm = int((np.isfinite(cls.scr_t) & (cls.owner_layer >= 0)).sum())
        assert strm <= E, f"stream region {strm} exceeds one layer union {E}"


def run_arm(protect, seed = 5, scans = 3, extra_env = None, gap = 0):
    rng = np.random.default_rng(seed)
    with tempfile.TemporaryDirectory(dir = os.path.join(HERE, "..", "scratch") if os.path.isdir(os.path.join(HERE, "..", "scratch")) else None) as d:
        base.rng = np.random.default_rng(99)
        stc, lkeys, data = base.build_files(d)
        cfg = base.SimpleNamespace(stc = stc)
        env = {"EXL3_EXPERT_BUDGET_GIB": "0", "EXL3_EXPERT_RING": "24", "EXL3_EXPERT_RESERVE": "3", "EXL3_EXPERT_CHUNK_KIB": "8",
               "EXL3_EXPERT_READERS": "4", "EXL3_EXPERT_AHEAD_ALL_ROWS": "64", "EXL3_EXPERT_HALFLIFE": "16",
               "EXL3_EXPERT_PROTECT": "1" if protect else "0", "EXL3_EXPERT_HOT_HALFLIFE": "64", **(extra_env or {})}
        os.environ.update(env)
        st = es.get_store(cfg)
        mods = [base.fake_module(stc, lk) for lk in lkeys]
        for m in mods:
            m.routing_cfg = base.SimpleNamespace(gate_tensor = None, router_bias = None, e_score_correction_bias = None, num_experts_per_tok = 10)
        lts = [st.register(m, "cpu", shells = False) for m in mods]
        cls = lts[0].cls
        st.budget = POOL * cls.slot_bytes
        for lt in lts:
            lt.build_tables(torch.device("cpu"))
        st.finalize()
        hot = [rng.permutation(E)[:HOT] for _ in lts]

        def decode_step():
            hits = uses = 0
            for li, lt in enumerate(lts):
                if li == 0:
                    pass
                picks = np.stack([np.concatenate([rng.choice(hot[li], 3, replace = False), rng.choice(E, 1)]) for _ in range(ROWS_DEC)])
                u = np.unique(picks)
                h0, u0 = st.st_hits, st.st_uses
                st.prepare(lt, torch.from_numpy(picks), {})
                hits += st.st_hits - h0
                uses += st.st_uses - u0
                check_bytes(st, lt, stc, data, picks)
                accounting(st, lts)
            return hits / uses

        for _ in range(60):
            decode_step()
        warm_hot = [int(sum(lt.smap[e] >= 0 for e in hot[li])) for li, lt in enumerate(lts)]
        after, scan_bytes = [], []
        for _ in range(scans):
            st.step += gap          # idle forwards between decode and the scan: the overall counters decay, the hot set is stale
            b0 = st.st_bytes
            for li, lt in enumerate(lts):
                picks = np.stack([rng.choice(E, 10, replace = False) for _ in range(ROWS_PF)])
                st.prepare(lt, torch.from_numpy(picks), {})
                check_bytes(st, lt, stc, data, picks)
                accounting(st, lts)
            scan_bytes.append(st.st_bytes - b0)
            after.append(sum(int(sum(lt.smap[e] >= 0 for e in hot[li])) for li, lt in enumerate(lts)))
            first = decode_step()
            for _ in range(5):
                decode_step()
        out = dict(warm_hot = sum(warm_hot), after_scan_hot = after, scan_bytes = sum(scan_bytes), first_dec_hit = first,
                   stream = int((np.isfinite(cls.scr_t) & (cls.owner_layer >= 0)).sum()) if cls.scr_t is not None else 0)
        for lt in lts:
            st.unregister(lt)
        for k in env:
            os.environ.pop(k, None)
        return out


def test_promote():
    # a decode hit on a stream slot makes it stable
    rng = np.random.default_rng(1)
    with tempfile.TemporaryDirectory(dir = os.path.join(HERE, "..", "scratch") if os.path.isdir(os.path.join(HERE, "..", "scratch")) else None) as d:
        base.rng = np.random.default_rng(99)
        stc, lkeys, data = base.build_files(d)
        cfg = base.SimpleNamespace(stc = stc)
        os.environ.update({"EXL3_EXPERT_BUDGET_GIB": "0", "EXL3_EXPERT_RING": "24", "EXL3_EXPERT_RESERVE": "3", "EXL3_EXPERT_CHUNK_KIB": "8",
                           "EXL3_EXPERT_READERS": "4", "EXL3_EXPERT_PROTECT": "1"})
        st = es.get_store(cfg)
        mods = [base.fake_module(stc, lk) for lk in lkeys]
        for m in mods:
            m.routing_cfg = base.SimpleNamespace(gate_tensor = None, router_bias = None, e_score_correction_bias = None, num_experts_per_tok = 10)
        lts = [st.register(m, "cpu", shells = False) for m in mods]
        st.budget = POOL * lts[0].cls.slot_bytes
        for lt in lts:
            lt.build_tables(torch.device("cpu"))
        st.finalize()
        cls = lts[0].cls
        lt = lts[0]
        picks = np.stack([rng.choice(E, 10, replace = False) for _ in range(ROWS_PF)])
        st.prepare(lt, torch.from_numpy(picks), {})
        s = int(lt.smap[int(picks[0, 0])])
        assert np.isfinite(cls.scr_t[s]), "scan miss must be a stream slot"
        st.prepare(lt, torch.from_numpy(picks[:1, :3]), {})
        assert not np.isfinite(cls.scr_t[s]), "decode hit must promote the stream slot"
        for lt in lts:
            st.unregister(lt)
        for k in ("EXL3_EXPERT_BUDGET_GIB", "EXL3_EXPERT_RING", "EXL3_EXPERT_RESERVE", "EXL3_EXPERT_CHUNK_KIB", "EXL3_EXPERT_READERS", "EXL3_EXPERT_PROTECT"):
            os.environ.pop(k, None)


def main():
    test_promote()
    # no scratch ring and no protection: a scan flushes the decode-hot set (what the policy exists to prevent)
    flush = run_arm(False, scans = 6, extra_env = {"EXL3_EXPERT_PF_SCRATCH": "0", "EXL3_EXPERT_HALFLIFE": "4"}, gap = 40)
    print("no ring, no protect:", flush)
    off = run_arm(False)
    on = run_arm(True)
    on6 = run_arm(True, scans = 6, extra_env = {"EXL3_EXPERT_HALFLIFE": "4"}, gap = 40)
    assert on6["after_scan_hot"] == [HOT * LAYERS] * 6, on6
    print("protect off:", off)
    print("protect on :", on)
    assert on["after_scan_hot"] == [on["warm_hot"]] * len(on["after_scan_hot"]) and on["warm_hot"] == HOT * LAYERS, "hot set evicted by a scan"
    assert on["scan_bytes"] <= off["scan_bytes"] * 1.05, (on["scan_bytes"], off["scan_bytes"])
    print("expert store protect CPU test OK")


if __name__ == "__main__":
    main()
