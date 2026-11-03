"""CPU test of the tiered expert store on PER-EXPERT-K layers (DeepSeek-V4.1: hot experts K=3, cold K=2, 384 experts
top-6). Real ExpertStore + real moe_mixedk.MixedKStore / MultiLinear over shell linears; only LinearEXL3 (needs the
extension) is replaced by a stand-in. Synthetic safetensors-format files with experts shuffled over two files.

Checks after every prepare(): for every picked expert the pointer-table entry of ITS K group (the nine tables
MixedKStore hands to the kernels, group-local index) addresses a slot whose bytes equal the file bytes; K groups of the
tier and of MixedKStore agree; slots are never shared; evicted experts point at their class's dummy slot; pinned
experts survive churn; prefill-sized unions (every expert of every group) fit; router-ahead and prefill-ahead run.

Run: python tests/test_expert_store_mixedk_cpu.py   (scratch files under $EXPERT_STORE_TEST_DIR or ./scratch)
"""
import ctypes, json, os, struct, sys, tempfile
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from hardware import require
require("ext")
import exllamav3.modules.quant.exl3 as exl3mod
from exllamav3.modules import expert_store as es
from exllamav3.modules.moe_mixedk import MixedKStore, expert_ks


class FakeLinearEXL3:
    def __init__(self, config, in_features, out_features, scale, su, sv, suh, svh, trellis, mcg, mul1, bias,
                 out_dtype, transformers_fix = False, key = None):
        self.suh, self.svh, self.trellis = suh, svh, trellis
        self.K = trellis.shape[-1] / 16
        self.mcg, self.mul1 = mcg is not None, mul1 is not None
        self.bias = None


exl3mod.LinearEXL3 = FakeLinearEXL3

E, LAYERS = 48, 3
IN, MID = 256, 128
TOPK = 6
HID = 32
rng = np.random.default_rng(77)


def layer_ks(l):
    """(K_gate_up, K_down) per expert: ~1/4 hot (3,3), a few (2,3), the rest (2,2); layer 2 is single-K (2,2)."""
    if l == 2:
        return [(2, 2)] * E
    ks = []
    for e in range(E):
        ks.append((3, 3) if e % 4 == 0 else ((2, 3) if e % 11 == 1 else (2, 2)))
    return ks


class FakeStc:
    def __init__(self):
        self.tensor_file_map, self.file_headers = {}, {}

    def find_stc(self, key):
        return self


def build_files(d):
    stc = FakeStc()
    entries, lkeys = [], []
    for l in range(LAYERS):
        lk = f"model.layers.{l}.mlp"
        lkeys.append(lk)
        for e, (kg, kd) in enumerate(layer_ks(l)):
            for proj, (i, o), K in (("down_proj", (MID, IN), kd), ("gate_proj", (IN, MID), kg), ("up_proj", (IN, MID), kg)):
                pre = f"{lk}.experts.{e}.{proj}"
                entries.append((pre + ".suh", "F16", [i], rng.integers(0, 256, i * 2, dtype = np.uint8).tobytes()))
                entries.append((pre + ".svh", "F16", [o], rng.integers(0, 256, o * 2, dtype = np.uint8).tobytes()))
                entries.append((pre + ".mul1", "I32", [], rng.integers(0, 256, 4, dtype = np.uint8).tobytes()))
                tb = (i // 16) * (o // 16) * 16 * K * 2
                entries.append((pre + ".trellis", "I16", [i // 16, o // 16, 16 * K],
                                rng.integers(0, 256, tb, dtype = np.uint8).tobytes()))
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
            blob += b"\x00" * int(rng.integers(1, 700))
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
        data[fn] = bytes(blob)
    return stc, lkeys, data


class Lin:
    def __init__(self, key, i, o):
        self.key, self.in_features, self.out_features, self.out_dtype = key, i, o, None
        self.softcap, self.post_scale, self.quant_type, self.inner = None, 1.0, None, None


def fake_module(stc, lk):
    mk = lambda e, p, i, o: Lin(f"{lk}.experts.{e}.{p}", i, o)
    return SimpleNamespace(
        key = lk, gated = True, num_experts = E, num_experts_per_tok = TOPK, num_local_experts = E,
        routing_first = None, tp_mode = None,
        gates = [mk(e, "gate_proj", IN, MID) for e in range(E)],
        ups = [mk(e, "up_proj", IN, MID) for e in range(E)],
        downs = [mk(e, "down_proj", MID, IN) for e in range(E)],
        config = SimpleNamespace(stc = stc))


def read_mem(addr, n):
    return bytes((ctypes.c_ubyte * n).from_address(int(addr)))


def main():
    base = os.environ.get("EXPERT_STORE_TEST_DIR") or os.path.join(os.path.dirname(__file__), "..", "scratch")
    os.makedirs(base, exist_ok = True)
    with tempfile.TemporaryDirectory(dir = base) as d:
        stc, lkeys, data = build_files(d)
        cfg = SimpleNamespace(stc = stc)
        os.environ.update({"EXL3_EXPERT_BUDGET_GIB": "0", "EXL3_EXPERT_RING": "40", "EXL3_EXPERT_RESERVE": "4",
                           "EXL3_EXPERT_CHUNK_KIB": "8", "EXL3_EXPERT_READERS": "3", "EXL3_EXPERT_AHEAD": "1",
                           "EXL3_EXPERT_AHEAD_ALL_ROWS": "24", "EXL3_EXPERT_PIN_FRAC": "0.3"})
        st = es.get_store(cfg)
        mods = [fake_module(stc, lk) for lk in lkeys]
        for m in mods:
            m.routing_cfg = SimpleNamespace(gate_tensor = torch.randn(HID, E), router_bias = None,
                                            e_score_correction_bias = None, num_experts_per_tok = TOPK)
        tss = [st.register(m, "cpu", shells = True) for m in mods]
        assert all(tss), "probe rejected a synthetic layer"
        # groups: layer 0/1 mixed (3 groups: (2,2) (2,3) (3,3)), layer 2 single (2,2)
        assert [len(t.groups) for t in tss] == [3, 3, 1], [len(t.groups) for t in tss]
        assert tss[0].mixed and not tss[2].mixed
        # (2,2) of layers 0,1,2 share one class; (2,3) and (3,3) are own classes; tier classes == distinct layouts
        assert len(st.classes) == 3, len(st.classes)
        # the real mixedk store over the shells
        mixed = []
        for ts, m in zip(tss, mods):
            ks = expert_ks(m.gates, m.ups, m.downs)
            if ts.mixed:
                mk = MixedKStore(torch.device("cpu"), m.gates, m.ups, m.downs)
                ts.bind_mixed(mk)
                mixed.append(mk)
            else:
                ts.build_tables(torch.device("cpu"))
                mixed.append(None)
        # warm start: experts 0..11 hot in every layer; pin 30 % of each pool
        # tools/ds41/plan.py frequency file format: {"layers.N": {expert: count}} (keys looked up by layer number)
        st.warm = st._norm_warm({f"layers.{l}": {str(e): 100 - e for e in range(12)} for l in range(LAYERS)})
        # budget: every class gets ~2/3 of its experts, never below one group
        tot = sum(c.total_bytes for c in st.classes.values())
        st.budget = int(tot * 0.6)
        st.finalize()
        for c in st.classes.values():
            assert c.nslots >= c.n_experts_max
        for ts in tss:      # the 12 hottest experts of every layer are resident after the warm start
            for e in range(12):
                assert ts.groups[int(ts.kmap[e])].smap[int(ts.lslot[e])] >= 0, (ts.key, e)
        npin = int(sum(int(c.pinned.sum()) for c in st.classes.values()))
        assert npin > 0, "nothing pinned"
        pinned_owner = {c.cid: set(zip(c.owner_layer[c.pinned].tolist(), c.owner_expert[c.pinned].tolist()))
                        for c in st.classes.values()}
        print("classes:", {c.cid: (c.nslots, c.total_bytes // c.slot_bytes, int(c.pinned.sum())) for c in st.classes.values()})

        def check(ts, mk, picks):
            for e in picks:
                e = int(e)
                gi, li = int(ts.kmap[e]), int(ts.lslot[e])
                g = ts.groups[gi]
                if mk is not None:
                    assert mk.groups[gi].experts[li] == e
                    tabs = mk.groups[gi].tables()
                else:
                    tabs = [g.view(t) for t in range(es.NT)]
                s = int(g.smap[li])
                assert s >= 0, f"expert {e} not resident"
                cls = g.cls
                assert cls.owner_layer[s] == g.lidx and cls.owner_expert[s] == li
                addr0 = int(cls.slot_addr[s])
                for pi, proj in enumerate(("gate_proj", "up_proj", "down_proj")):
                    for pj, part in enumerate(es.PARTS):
                        t = pi * 3 + pj
                        a = int(tabs[t][li])
                        assert a == addr0 + cls.offs[t], (e, proj, part)
                        key = f"{ts.key}.experts.{e}.{proj}.{part}"
                        fn = stc.tensor_file_map[key]
                        b0, b1 = stc.file_headers[fn][key]["data_offsets"]
                        assert read_mem(a, b1 - b0) == data[fn][b0:b1], f"bytes differ {key}"
            for t2 in tss:
                for g in t2.groups:
                    occ = g.smap >= 0
                    for li in np.nonzero(~occ)[0]:     # evicted -> dummy of the group's class
                        assert int(g.host[0 * g.E + li]) == g.cls.dummy.data_ptr() + g.cls.offs[0] or g.dirty
            for c in st.classes.values():
                occ = c.owner_layer >= 0
                keys = set(zip(c.owner_layer[occ].tolist(), c.owner_expert[occ].tolist()))
                assert len(keys) == int(occ.sum()), "slot owned twice"

        for it in range(300):
            li = it % LAYERS
            ts, mk = tss[li], mixed[li]
            rows = int(rng.choice([1, 1, 1, 4, 16, 40]))
            hot = np.array([0, 4, 8, 12, 1, 2, 3, 5])
            picks = np.empty((rows, TOPK), dtype = np.int64)
            for r in range(rows):
                h = rng.choice(hot, 4, replace = False)
                c = rng.choice(E, 2, replace = False)
                picks[r] = np.concatenate([h, c])
            st.prepare(ts, torch.from_numpy(picks), {}, torch.randn(rows, HID))
            check(ts, mk, np.unique(picks))
        for c in st.classes.values():                       # pinned experts survived all the churn
            now = set(zip(c.owner_layer[c.pinned].tolist(), c.owner_expert[c.pinned].tolist()))
            assert now == pinned_owner[c.cid], "pinned expert moved"
        s = st.stats()
        print("stats", {k: s[k] for k in ("uses", "hits", "misses", "hit_rate", "ahead_issued", "ahead_used", "ahead_wasted")})
        assert s["misses"] > 0 and s["hits"] > 0 and s["ahead_issued"] > 0
        # prefill-sized union: every expert of every group of one layer
        st.prepare(tss[0], torch.arange(E).view(1, E).repeat(30, 1)[:, :TOPK].contiguous(), {})
        st.prepare(tss[1], torch.arange(E).view(1, E), {})
        check(tss[1], mixed[1], np.arange(E))
        for ts in tss:
            st.unregister(ts)
    print("expert store mixed-K CPU test OK")


if __name__ == "__main__":
    main()
