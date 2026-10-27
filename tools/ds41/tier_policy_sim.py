"""Offline replay of an expert-store trace (EXL3_EXPERT_TRACE_OUT of a session run) through the REAL policy code of
exllamav3/modules/expert_store.py (ExpertStore._serve_set / _serve_group / _alloc_slots / _warm_fill), with the reads stubbed.
The hit rate of a cache policy depends only on the sequence of (layer, unique experts, rows) calls, which is the same for every
policy (greedy output is policy independent), so one GPU trace ranks any number of policies on the CPU in seconds.

usage: tier_policy_sim.py TRACE.npz [--warm warm.json] [--arm name:ENV=V,ENV=V ...] [--json out.json]
Prints per turn: prefill hit / GB, decode hit / misses. A turn starts at the first prefill-sized call after a decode call."""
import argparse, json, os, sys, importlib.util
from types import SimpleNamespace
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def load_store_module():
    spec = importlib.util.spec_from_file_location("expert_store_sim", os.path.join(ROOT, "exllamav3", "modules", "expert_store.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class SC:                                   # size class
    def __init__(self, cid, nslots, slot_bytes):
        self.cid, self.nslots, self.slot_bytes = cid, nslots, slot_bytes
        self.layers = []
        self.n_experts_max = 0
        self.owner_layer = np.full(nslots, -1, dtype=np.int32)
        self.owner_expert = np.full(nslots, -1, dtype=np.int32)
        self.sval = np.zeros(nslots); self.st = np.zeros(nslots)
        self.dval = np.zeros(nslots); self.dst = np.zeros(nslots)
        self.pinned = np.zeros(nslots, dtype=bool)
        self.scr_t = None
        self.free = list(range(nslots - 1, -1, -1))


class SL:                                   # layer tier group
    def __init__(self, lidx, key, ids, cls, k):
        self.lidx, self.key, self.cls = lidx, key, cls
        self.ids = np.asarray(ids, dtype=np.int64)
        self.E = len(self.ids)
        self.smap = np.full(self.E, -1, dtype=np.int32)
        self.hval = np.zeros(self.E); self.ht = np.zeros(self.E)
        self.hdv = np.zeros(self.E); self.hdt = np.zeros(self.E)
        self.counts = np.zeros(self.E, dtype=np.int64)
        self.module = SimpleNamespace(num_experts_per_tok=k)
        self.set = None

    def set_expert_ptrs(self, e, addr): pass
    def flush(self): pass


class SS:                                   # tier set
    def __init__(self, sidx, key, E, groups, kmap, lslot):
        self.sidx, self.key, self.E, self.groups, self.kmap, self.lslot = sidx, key, E, groups, kmap, lslot
        for g in groups:
            g.set = self

    @property
    def single(self): return len(self.groups) == 1


def build(es, layout):
    classes = {int(c): SC(int(c), v["nslots"], v["slot_bytes"]) for c, v in layout["classes"].items()}
    sets, layers = [], []
    for d in layout["sets"]:
        groups = []
        for gi, cid in enumerate(d["cls"]):
            cls = classes[cid]
            if "kmap" in d:
                kmap = np.array(d["kmap"]); ids = np.flatnonzero(kmap == gi)
            else:
                ids = np.array(d["ids"])
            lt = SL(len(layers), d["key"], ids, cls, d["k"])
            layers.append(lt); cls.layers.append(lt); groups.append(lt)
            cls.n_experts_max = max(cls.n_experts_max, lt.E)
        kmap = np.array(d.get("kmap", [0] * d["E"])); lslot = np.array(d.get("lslot", list(range(d["E"]))))
        sets.append(SS(d["sidx"], d["key"], d["E"], groups, kmap, lslot))
    return classes, sets, layers


def run_arm(es, tr, layout, env, warm):
    env = {"EXL3_EXPERT_BUDGET_GIB": "1", **env}
    keep = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        st = es.ExpertStore(None)
    finally:
        for k, v in keep.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v
    classes, sets, layers = build(es, layout)
    st.classes, st.sets, st.layers = classes, sets, layers

    def _load(lt, experts, slots):
        lt.smap[experts] = slots
        lt.cls.owner_layer[slots] = lt.lidx
        lt.cls.owner_expert[slots] = experts
        st.st_bytes += len(experts) * lt.cls.slot_bytes
    st._load = _load
    st.is_cuda = False
    if warm:
        st.warm = st._norm_warm(json.load(open(warm)))
        st._warm_fill(list(classes.values()))
    st.reset_stats()
    st.st_bytes = 0
    rows_a, off, ids, step, lay = tr["rows"], tr["off"], tr["ids"], tr["step"], tr["layer"]
    tail_of = {int(k): tr["tail_rows"][j] for j, k in enumerate(tr["tail_idx"])} if len(tr.get("tail_rows", [])) else {}
    turns, cur, prev_scan = [], None, False
    for i in range(len(rows_a)):
        rows = int(rows_a[i]); scan = rows >= st.ahead_all_rows
        if scan and not prev_scan:
            cur = {"pre": [0, 0, 0], "dec": [0, 0, 0]}; turns.append(cur)
        prev_scan = scan
        if cur is None:
            continue
        st.step = int(step[i])
        u = ids[off[i]:off[i + 1]].astype(np.int64)
        u0, h0, m0, b0 = st.st_uses, st.st_hits, st.st_misses, st.st_bytes
        tail = None
        if scan and st.tail_rows and i in tail_of:
            tail = np.unique(tail_of[i][-st.tail_rows:]).astype(np.int64)
            tail = tail[(tail >= 0) & (tail < sets[int(lay[i])].E)]
        st._serve_set(sets[int(lay[i])], u, rows, float(st.step), tail)
        a = cur["pre" if scan else "dec"]
        a[0] += st.st_uses - u0; a[1] += st.st_hits - h0; a[2] += st.st_bytes - b0
    return turns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace"); ap.add_argument("--warm"); ap.add_argument("--arm", action="append", default=[])
    ap.add_argument("--json"); ap.add_argument("--scale", type=float, default=1.0)
    a = ap.parse_args()
    es = load_store_module()
    z = np.load(a.trace, allow_pickle=False)
    tr = {k: z[k] for k in ("rows", "off", "ids", "step", "layer")}
    if "tail_idx" in z.files:
        tr["tail_idx"], tr["tail_rows"] = z["tail_idx"], z["tail_rows"]
    layout = json.loads(str(z["layout"]))
    for v in layout["classes"].values():
        v["nslots"] = int(v["nslots"] * a.scale)
    arms = [("base", {})]
    for s in a.arm:
        name, _, kv = s.partition(":")
        arms.append((name, dict(x.split("=", 1) for x in kv.split(",") if x)))
    out = {}
    for name, env in arms:
        turns = run_arm(es, tr, layout, env, a.warm)
        out[name] = [{"pre_hit": t["pre"][1] / max(t["pre"][0], 1), "pre_GB": t["pre"][2] / 1e9,
                      "dec_hit": t["dec"][1] / max(t["dec"][0], 1), "dec_miss_GB": t["dec"][2] / 1e9} for t in turns]
        d = out[name]
        print(f"{name:28s} pre_GB {sum(x['pre_GB'] for x in d):7.1f}  dec_hit " + " ".join(f"{x['dec_hit']:.3f}" for x in d) +
              f"  | mean {np.mean([x['dec_hit'] for x in d[1:]]):.4f}  dec_GB {sum(x['dec_miss_GB'] for x in d):.1f}", flush=True)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
