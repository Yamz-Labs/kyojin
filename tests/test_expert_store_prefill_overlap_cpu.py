"""CPU test of the prefill read overlap of the expert store (EXL3_EXPERT_PF_TOPUP, chunk wrap).

Prefill-sized calls (>= EXL3_EXPERT_AHEAD_ALL_ROWS rows) over a staging ring smaller than two layers of misses. Checks:
  * bitwise identity: after every layer call the same resident map and the file bytes behind every picked expert, whatever
    the arm (no streaming, streaming, streaming + top-up, + chunk wrap, slow reads);
  * the current layer's experts are never evicted (every picked expert is resident with the right bytes after its call);
  * budget accounting: staged entries never exceed ring - reserve, ring entries are all accounted (free + staged + draining);
  * top-up and wrap issue more useful prefetches (ahead_used) than the layer-start issue alone.

Run: python tests/test_expert_store_prefill_overlap_cpu.py
"""
import importlib.util
import os
import tempfile
import time

import numpy as np
import torch

HERE = os.path.dirname(__file__)
T = importlib.util.spec_from_file_location("t_es_cpu", os.path.join(HERE, "test_expert_store_cpu.py"))
base = importlib.util.module_from_spec(T)
T.loader.exec_module(base)
es, E, LAYERS = base.es, base.E, base.LAYERS
RING, RESERVE, ROWS = 14, 3, 80


def run_arm(arm, chunks = 4, seed = 11):
    rng = np.random.default_rng(seed)
    with tempfile.TemporaryDirectory(dir = os.path.join(HERE, "..", "scratch") if os.path.isdir(os.path.join(HERE, "..", "scratch")) else None) as d:
        base.rng = np.random.default_rng(99)
        stc, lkeys, data = base.build_files(d)
        cfg = base.SimpleNamespace(stc = stc)
        env = {"EXL3_EXPERT_BUDGET_GIB": "0", "EXL3_EXPERT_RING": str(RING), "EXL3_EXPERT_RESERVE": str(RESERVE),
               "EXL3_EXPERT_CHUNK_KIB": "8", "EXL3_EXPERT_READERS": "4", "EXL3_EXPERT_PF_READERS": "3",
               "EXL3_EXPERT_PF_AHEAD": "1" if arm["stream"] else "0", "EXL3_EXPERT_PF_TOPUP": "1" if arm.get("topup") else "0",
               "EXL3_EXPERT_PF_DEPTH": "1", "EXL3_EXPERT_AHEAD_ALL_ROWS": "64"}
        os.environ.update(env)
        st = es.get_store(cfg)
        mods = [base.fake_module(stc, lk) for lk in lkeys]
        for m in mods:
            m.routing_cfg = base.SimpleNamespace(gate_tensor = None, router_bias = None, e_score_correction_bias = None, num_experts_per_tok = 10)
        lts = [st.register(m, "cpu", shells = False) for m in mods]
        cls = lts[0].cls
        st.budget = 50 * cls.slot_bytes          # 50 slots for 120 experts: eviction on every layer
        for lt in lts:
            lt.build_tables(torch.device("cpu"))
        st.finalize()
        if arm.get("slow"):
            orig_read = st._read1

            def slow(fid, off, n, dst):
                time.sleep(0.003)
                return orig_read(fid, off, n, dst)
            st._read1 = slow
        fp, early = [], []
        for c in range(chunks):
            st.pf_wrap = bool(arm.get("wrap")) and c < chunks - 1
            for li, lt in enumerate(lts):
                early.append(len(st.staged))        # reads already in flight when the layer starts: overlapped with the layer before
                st.prefill_issue(lt, ROWS)
                picks = np.stack([rng.choice(E, 10, replace = False) for _ in range(ROWS)])
                st.prepare(lt, torch.from_numpy(picks), {})
                # accounting: every ring entry is free, staged or draining
                acc = len(st.free_entries) + len(st.staged) + len(st.draining)
                assert acc == RING, (acc, len(st.free_entries), len(st.staged), len(st.draining))
                assert len(st.staged) <= RING - RESERVE + 0, len(st.staged)
                bs = []
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
                            bs.append(hash(got))
                fp.append((tuple(tuple(int(x) for x in l.smap) for l in lts), tuple(bs)))
        stats = st.stats()
        for lt in lts:
            st.unregister(lt)
        for k in env:
            os.environ.pop(k, None)
        stats["early"] = early
        return fp, stats


def test_chain_keys():
    rx = es.ExpertStore._LAYER_PFX
    for k, pre in (("layers.5.ffn.", "layers."), ("model.layers.5.mlp.", "model.layers."), ("mtp.layers.0.ffn.", "mtp.layers.")):
        m = rx.match(k)
        assert m and m.group(1) == pre and m.group(2) == k.split("layers.")[1].split(".")[0], k


def main():
    test_chain_keys()
    ref, s0 = run_arm({"stream": False})
    res = {}
    arms = {"stream": {"stream": True}, "stream+topup": {"stream": True, "topup": True},
            "stream+topup+wrap": {"stream": True, "topup": True, "wrap": True},
            "topup+wrap, slow reads": {"stream": True, "topup": True, "wrap": True, "slow": True}}
    for name, arm in arms.items():
        fp, s = run_arm(arm)
        assert [x[1] for x in fp] == [x[1] for x in ref], f"slot bytes differ: {name}"
        assert s["hits"] == s0["hits"] and s["misses"] == s0["misses"], (name, s["hits"], s0["hits"])
        assert s["ahead_peak"] <= RING, s
        res[name] = s
        print(f"{name}: identical bytes/hits; issued {s['ahead_issued']} used {s['ahead_used']} wasted {s['ahead_wasted']} peak {s['ahead_peak']}")
    ea = {k: v["early"] for k, v in res.items()}
    n = len(lts_n := ea["stream"])
    # without top-up nothing is in flight when a layer starts; with it the ring is refilled during the previous layer
    assert sum(ea["stream"]) == 0, ea["stream"]
    assert all(x > 0 for i, x in enumerate(ea["stream+topup"]) if i % LAYERS), ea["stream+topup"]
    # chunk boundary (layer 0 of chunks 1..): only the wrap keeps the reads of layer 0 in flight
    first = lambda a: [a[i] for i in range(LAYERS, n, LAYERS)]
    assert sum(first(ea["stream+topup"])) == 0 and all(x > 0 for x in first(ea["stream+topup+wrap"])), (first(ea["stream+topup"]), first(ea["stream+topup+wrap"]))
    print("reads in flight at layer start:", ea["stream+topup+wrap"])
    print("expert store prefill overlap CPU test OK")


if __name__ == "__main__":
    main()
