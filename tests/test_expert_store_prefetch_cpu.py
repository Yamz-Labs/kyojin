"""CPU test of the expert store prefetch: the two-queue IO pool (demand first, prefetch cap, cancel, promote) and
exactness of the store with prefetch on/off: the same routing sequence must leave the same resident map after every
step and every picked expert must hold the file bytes, whatever the predictor (good, forced-wrong, wide, deep) and even
when reads are slow enough that prefetches are cancelled mid-queue.

Run: python tests/test_expert_store_prefetch_cpu.py
"""
import importlib.util
import os
import sys
import tempfile
import threading
import time

import numpy as np
import torch

HERE = os.path.dirname(__file__)
T = importlib.util.spec_from_file_location("t_es_cpu", os.path.join(HERE, "test_expert_store_cpu.py"))
base = importlib.util.module_from_spec(T)
T.loader.exec_module(base)
es, E, LAYERS = base.es, base.E, base.LAYERS


def test_pool():
    pool = es.PrioPool(3, 1)
    gate = threading.Event()
    order, running, peak = [], [0], [0]
    lk = threading.Lock()

    def job(name, hold):
        with lk:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
            order.append(name)
        if hold:
            gate.wait(5)
        with lk:
            running[0] -= 1
        return name

    # three prefetch tasks: only pf_max=1 may run, the other two threads stay free for demand
    pf = [pool.submit(job, f"pf{i}", True, prio = 1) for i in range(3)]
    time.sleep(0.1)
    assert order == ["pf0"], order
    d = pool.submit(job, "demand", False)
    assert d.result() == "demand"          # runs at once on a free thread, not behind the prefetch queue
    # cancel a queued prefetch, promote another: the promoted one runs without waiting for pf0
    assert pf[1].cancel() and pf[1].done()
    pf[2].promote()
    time.sleep(0.1)
    assert "pf2" in order and "pf1" not in order, order
    assert not pf[0].cancel()              # running: cannot be cancelled
    gate.set()
    assert pf[0].result() == "pf0" and pf[2].result() == "pf2"
    try:
        pf[1].result()
        raise AssertionError("cancelled result")
    except es.CancelledError:
        pass
    assert peak[0] <= 3
    pool.shutdown()
    print("PrioPool OK")


def run_arm(arm, steps = 300, seed = 7):
    """Returns per-step fingerprints (resident map + bytes of the picked experts) and the store stats."""
    rng = np.random.default_rng(seed)
    with tempfile.TemporaryDirectory(dir = os.path.join(HERE, "..", "scratch") if os.path.isdir(os.path.join(HERE, "..", "scratch")) else None) as d:
        base.rng = np.random.default_rng(99)       # identical files in every arm
        stc, lkeys, data = base.build_files(d)
        cfg = base.SimpleNamespace(stc = stc)
        env = {"EXL3_EXPERT_BUDGET_GIB": "0", "EXL3_EXPERT_RING": "14", "EXL3_EXPERT_RESERVE": "3",
               "EXL3_EXPERT_CHUNK_KIB": "8", "EXL3_EXPERT_READERS": "4", "EXL3_EXPERT_PF_READERS": "2",
               "EXL3_EXPERT_AHEAD": "1" if arm["ahead"] else "0", "EXL3_EXPERT_AHEAD_DEPTH": str(arm.get("depth", 1)),
               "EXL3_EXPERT_AHEAD_K": arm.get("k", "0"), "EXL3_EXPERT_AHEAD_EARLY": "0" if arm.get("late") else "1"}
        os.environ.update(env)
        st = es.get_store(cfg)
        mods = [base.fake_module(stc, lk) for lk in lkeys]
        HID = 32
        wr = np.random.default_rng(5)
        for m in mods:
            m.routing_cfg = base.SimpleNamespace(gate_tensor = torch.tensor(wr.standard_normal((HID, E)), dtype = torch.float32),
                                                 router_bias = None, e_score_correction_bias = None, num_experts_per_tok = 10)
        lts = [st.register(m, "cpu", shells = False) for m in mods]
        cls = lts[0].cls
        st.budget = 60 * cls.slot_bytes
        for lt in lts:
            lt.build_tables(torch.device("cpu"))
        st.finalize()
        if arm.get("wrong"):
            orig = st._predict
            prng = np.random.default_rng(3)

            def wrong(ts, z, nxt = None, k = 0):
                r = orig(ts, z, nxt, k)
                if r is None:
                    return None
                return torch.from_numpy(prng.integers(0, E, r.shape)).to(r.dtype)
            st._predict = wrong
        if arm.get("slow"):
            orig_read = st._read1

            def slow(fid, off, n, dst):
                time.sleep(0.004)
                return orig_read(fid, off, n, dst)
            st._read1 = slow
        hot = [rng.permutation(E)[:8] for _ in lts]
        fp = []
        for it in range(steps):
            li = it % LAYERS
            lt = lts[li]
            rows = int(rng.choice([1, 1, 1, 4, 16]))
            picks = np.empty((rows, 10), dtype = np.int64)
            for r in range(rows):
                picks[r] = np.concatenate([rng.choice(hot[li], 6, replace = False), rng.choice(E, 4, replace = False)])
            st.prepare(lt, torch.from_numpy(picks), {}, torch.from_numpy(rng.standard_normal((rows, HID)).astype(np.float32)))
            # exactness: bytes behind every picked expert's table entries equal the file bytes
            bs = []
            for e in np.unique(picks):
                s = int(lt.smap[e])
                assert s >= 0
                for proj, t0 in (("gate_proj", 0), ("up_proj", 3), ("down_proj", 6)):
                    for pj, part in enumerate(es.PARTS):
                        key = f"{lt.key}.experts.{e}.{proj}.{part}"
                        fn = stc.tensor_file_map[key]
                        b0, b1 = stc.file_headers[fn][key]["data_offsets"]
                        got = base.read_mem(int(lt.tab[(t0 + pj) * E + e]), b1 - b0)
                        assert got == data[fn][1][b0:b1], f"bytes differ {key} step {it}"
                        bs.append(hash(got))
            fp.append((it, tuple(int(x) for x in lts[0].smap), tuple(int(x) for x in lts[1].smap),
                       tuple(int(x) for x in lts[2].smap), tuple(bs)))
            if it == 150:
                hot = [rng.permutation(E)[:8] for _ in lts]
        stats = st.stats()
        for lt in lts:
            st.unregister(lt)
        for k in env:
            os.environ.pop(k, None)
        return fp, stats


def main():
    test_pool()
    ref, s0 = run_arm({"ahead": False})
    arms = {"top-k depth 1": {"ahead": True}, "late issue (v1 order)": {"ahead": True, "late": True, "depth": 2}, "depth 3 wide k": {"ahead": True, "depth": 3, "k": "14,12,10"},
            "forced-wrong": {"ahead": True, "depth": 2, "k": "16", "wrong": True},
            "slow reads (cancels)": {"ahead": True, "depth": 3, "k": "16,16,16", "slow": True}}
    for name, arm in arms.items():
        fp, s = run_arm(arm)
        assert [x[:4] for x in fp] == [x[:4] for x in ref], f"resident map differs from no-prefetch: {name}"
        assert [x[4] for x in fp] == [x[4] for x in ref], f"slot bytes differ: {name}"
        assert s["hits"] == s0["hits"] and s["misses"] == s0["misses"], (name, s["hits"], s0["hits"])
        assert s["ahead_issued"] > 0, name
        assert s["ahead_used"] + s["ahead_wasted"] <= s["ahead_issued"] + 0, (name, s)
        assert s["ahead_peak"] <= 14, s
        print(f"{name}: identical to no-prefetch; issued {s['ahead_issued']} used {s['ahead_used']} wasted {s['ahead_wasted']} "
              f"cancelled {s['ahead_cancelled']} peak {s['ahead_peak']} per-depth issued {s['ahead_issued_d']} used {s['ahead_used_d']}")
    print("expert store prefetch CPU test OK")


if __name__ == "__main__":
    main()
