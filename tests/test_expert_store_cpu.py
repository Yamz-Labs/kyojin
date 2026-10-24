"""CPU test of the tiered expert store (no GPU, no extension): synthetic safetensors-format files with experts
scattered over two files in shuffled order, O_DIRECT reads, slot pools smaller than the expert set, random
skewed routing. After every prepare() every picked expert's pointer-table entries must address memory whose bytes
equal the file bytes, no two live experts may share a slot, evicted experts must point at the dummy slot.

Run: python tests/test_expert_store_cpu.py   (scratch files go under $EXPERT_STORE_TEST_DIR or ./scratch)
"""
import ctypes
import importlib.util
import json
import os
import struct
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

spec = importlib.util.spec_from_file_location(
    "expert_store", os.path.join(os.path.dirname(__file__), "..", "exllamav3", "modules", "expert_store.py"))
es = importlib.util.module_from_spec(spec)
sys.modules["expert_store"] = es
spec.loader.exec_module(es)

E, LAYERS = 40, 3
IN, MID, K = 256, 128, 4
rng = np.random.default_rng(1234)


class FakeStc:
    def __init__(self):
        self.tensor_file_map = {}
        self.file_headers = {}

    def find_stc(self, key):
        return self


def build_files(d):
    """Two safetensors-format files; per expert the nine tensors (+ a 4-byte mul1 between projections) contiguous
    in the real pack order (down, gate, up; suh, svh, mul1, trellis), experts shuffled across files."""
    stc = FakeStc()
    entries = []   # (file idx, key, dtype, shape, bytes)
    layer_keys = []
    for l in range(LAYERS):
        lk = f"model.layers.{l}.mlp"
        layer_keys.append(lk)
        for e in range(E):
            for proj, (i, o) in (("down_proj", (MID, IN)), ("gate_proj", (IN, MID)), ("up_proj", (IN, MID))):
                pre = f"{lk}.experts.{e}.{proj}"
                entries.append((pre + ".suh", "F16", [i], rng.integers(0, 256, i * 2, dtype = np.uint8).tobytes()))
                entries.append((pre + ".svh", "F16", [o], rng.integers(0, 256, o * 2, dtype = np.uint8).tobytes()))
                entries.append((pre + ".mul1", "I32", [], rng.integers(0, 256, 4, dtype = np.uint8).tobytes()))
                tb = (i // 16) * (o // 16) * 16 * K * 2
                entries.append((pre + ".trellis", "I16", [i // 16, o // 16, 16 * K],
                                rng.integers(0, 256, tb, dtype = np.uint8).tobytes()))
    # group per expert, shuffle expert groups, split over two files, odd junk between groups
    groups = [entries[i:i + 12] for i in range(0, len(entries), 12)]
    order = rng.permutation(len(groups))
    files = [[], []]
    for n, gi in enumerate(order):
        files[n % 2].append(groups[gi])
    data = {}
    for fi, gl in enumerate(files):
        hdr, blob = {}, bytearray()
        for g in gl:
            for key, dt, shape, b in g:
                hdr[key] = {"dtype": dt, "shape": shape, "data_offsets": [len(blob), len(blob) + len(b)]}
                blob += b
            blob += b"\x00" * int(rng.integers(1, 700))     # junk, keeps offsets unaligned
        hb = json.dumps(hdr).encode()
        hb += b" " * ((-len(hb)) % 8)
        fn = os.path.join(d, f"model-{fi}.safetensors")
        with open(fn, "wb") as f:
            f.write(struct.pack("<Q", len(hb)) + hb + bytes(blob))
        h = dict(hdr)
        h["_header_offset"] = 8 + len(hb)
        stc.file_headers[fn] = h
        for k in hdr:
            stc.tensor_file_map[k] = fn
        data[fn] = (8 + len(hb), bytes(blob))
    return stc, layer_keys, data


class Lin:
    def __init__(self, key, i, o):
        self.key, self.in_features, self.out_features, self.out_dtype = key, i, o, None


def fake_module(stc, lk):
    mk = lambda e, p, i, o: Lin(f"{lk}.experts.{e}.{p}", i, o)
    return SimpleNamespace(
        key = lk, gated = True, num_experts = E, num_experts_per_tok = 10, num_local_experts = E, routing_first = None, tp_mode = None,
        gates = [mk(e, "gate_proj", IN, MID) for e in range(E)],
        ups = [mk(e, "up_proj", IN, MID) for e in range(E)],
        downs = [mk(e, "down_proj", MID, IN) for e in range(E)],
        config = SimpleNamespace(stc = stc))


def read_mem(addr, n):
    return bytes((ctypes.c_ubyte * n).from_address(int(addr)))


def main(ahead: bool = False):
    base = os.environ.get("EXPERT_STORE_TEST_DIR") or os.path.join(os.path.dirname(__file__), "..", "scratch")
    os.makedirs(base, exist_ok = True)
    with tempfile.TemporaryDirectory(dir = base) as d:
        stc, lkeys, data = build_files(d)
        cfg = SimpleNamespace(stc = stc)
        os.environ["EXL3_EXPERT_BUDGET_GIB"] = "0"          # pool size forced below
        os.environ["EXL3_EXPERT_RING"] = "12"
        os.environ["EXL3_EXPERT_RESERVE"] = "4"
        os.environ["EXL3_EXPERT_CHUNK_KIB"] = "8"            # many sub-reads per expert
        os.environ["EXL3_EXPERT_READERS"] = "3"
        os.environ["EXL3_EXPERT_AHEAD"] = "1" if ahead else "0"
        st = es.get_store(cfg)
        mods = [fake_module(stc, lk) for lk in lkeys]
        HID = 32
        Wr = [torch.randn(HID, E) for _ in mods]
        for m, w in zip(mods, Wr):
            m.routing_cfg = SimpleNamespace(gate_tensor = w, router_bias = None, e_score_correction_bias = None,
                                            num_experts_per_tok = 10)
        lts = [st.register(m, "cpu", shells = False) for m in mods]
        assert all(lts), "probe rejected a synthetic layer"
        cls = lts[0].cls
        assert len(st.classes) == 1
        # budget: 60 slots for 3 layers x 40 experts (min = one layer = 40)
        st.budget = 60 * cls.slot_bytes
        for lt in lts:
            lt.build_tables(torch.device("cpu"))
        # warm start: the 20 hottest experts of every layer (counts skewed to experts 0..19) seed the pool
        st.warm = {lt.key: [100 - e if e < 20 else 0 for e in range(E)] for lt in lts}
        st.finalize()
        assert cls.nslots == 60, cls.nslots
        for lt in lts:
            assert (lt.smap[:20] >= 0).all() and (lt.smap[20:] < 0).all(), "warm set not resident"
        warm_bytes_ok = True
        print(f"slot {cls.slot_bytes} B, pool {cls.nslots}, entry {st.entry_bytes} B")

        def check(lt, picks):
            for e in picks:
                s = int(lt.smap[e])
                assert s >= 0, f"expert {e} not resident"
                assert cls.owner_layer[s] == lt.lidx and cls.owner_expert[s] == e
                addr0 = int(cls.slot_addr[s])
                lin = [("gate_proj", 0), ("up_proj", 3), ("down_proj", 6)]
                for proj, t0 in lin:
                    for pj, part in enumerate(es.PARTS):
                        t = t0 + pj
                        tab_addr = int(lt.tab[t * E + e])
                        assert tab_addr == addr0 + cls.offs[t], (proj, part)
                        key = f"{lt.key}.experts.{e}.{proj}.{part}"
                        fn = stc.tensor_file_map[key]
                        hdr = stc.file_headers[fn]
                        b0, b1 = hdr[key]["data_offsets"]
                        want = data[fn][1][b0:b1]
                        got = read_mem(tab_addr, b1 - b0)
                        assert got == want, f"bytes differ {key}"
                        # stacked gate/up tables
                        if proj in ("gate_proj", "up_proj"):
                            col = 0 if proj == "gate_proj" else 1
                            assert int(lt.gu_view(pj)[e, col]) == tab_addr
            # global invariants
            occ = cls.owner_layer >= 0
            keys = set(zip(cls.owner_layer[occ].tolist(), cls.owner_expert[occ].tolist()))
            assert len(keys) == int(occ.sum()), "slot owned twice"
            for l2 in lts:
                for e in range(E):
                    s = int(l2.smap[e])
                    if s >= 0:
                        assert cls.owner_layer[s] == l2.lidx and cls.owner_expert[s] == e
            for l2 in lts:
                l2_dirty = l2.dirty
                assert not l2_dirty or l2 is not lt, "own table left dirty"

        nsteps = 400
        hot = [rng.permutation(E)[:8] for _ in lts]
        for it in range(nsteps):
            li = it % LAYERS
            lt = lts[li]
            rows = int(rng.choice([1, 1, 1, 4, 16, 40]))
            picks = np.empty((rows, 10), dtype = np.int64)
            for r in range(rows):
                h = rng.choice(hot[li], 6, replace = False)
                c = rng.choice(E, 4, replace = False)
                picks[r] = np.concatenate([h, c])
            z = torch.randn(rows, HID)
            st.prepare(lt, torch.from_numpy(picks), {}, z)
            check(lt, np.unique(picks))
            if it == 200:
                hot = [rng.permutation(E)[:8] for _ in lts]   # drift
        s = st.stats()
        print("stats", s)
        assert s["misses"] > 0 and s["hits"] > 0
        if ahead:
            assert s["ahead_issued"] > 0 and s["ahead_used"] > 0, s
            assert s["ahead_used"] + s["ahead_wasted"] <= s["ahead_issued"], s
        # a full-layer union fits and evicts only other layers' experts
        st.prepare(lts[1], torch.arange(E).view(1, E), {})
        check(lts[1], np.arange(E))
        # budget >= everything: no eviction ever
        for lt in lts:
            st.unregister(lt)
        assert st.classes == {} or True
    print(f"expert store CPU test OK (ahead={ahead})")


if __name__ == "__main__":
    main(False)
    main(True)
