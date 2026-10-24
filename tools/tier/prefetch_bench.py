#!/usr/bin/env python
"""Store-level decode replay on a synthetic expert pack at real DS4.1 expert size (K=2: 8.85 MiB), real NVMe O_DIRECT reads,
real pinned staging, real pool in device memory. Layer compute is emulated (GPU spin kernel of the modelled duration);
routing and the prefetch guesses are the REAL ones of pred_probe.py (replayed through a patched _predict).

  make:  python tools/tier/prefetch_bench.py make --dir D --layers 6
  run:   python tools/tier/prefetch_bench.py run  --dir D --trace OUTA OUTB --calib-json J --layer0 3 --c1 0.7 --out r.json
Each arm: same steps, prefetch off / on variants. Exactness: a device checksum over the bytes of every picked expert slot,
per step, must be identical in all arms (the store never changes what a kernel reads). Reports steps/s, stall, NVMe volume.
"""
import argparse, importlib.util, json, os, sys, time, hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tests"))
IN, MID, K, E = 5120, 2304, 2, 128   # reduced pack (disk); trace expert ids are folded mod E


def load_store():
    spec = importlib.util.spec_from_file_location("expert_store", os.path.join(HERE, "..", "..", "exllamav3", "modules", "expert_store.py"))
    es = importlib.util.module_from_spec(spec)
    sys.modules["expert_store"] = es
    spec.loader.exec_module(es)
    return es


def tensors_of(e_rng, l, e):
    """the 12 tensors of one expert in pack order: per projection suh, svh, mul1, trellis"""
    out = []
    for proj, (i, o) in (("down_proj", (MID, IN)), ("gate_proj", (IN, MID)), ("up_proj", (IN, MID))):
        pre = f"model.layers.{l}.mlp.experts.{e}.{proj}"
        out.append((pre + ".suh", "F16", [i], i * 2))
        out.append((pre + ".svh", "F16", [o], o * 2))
        out.append((pre + ".mul1", "I32", [], 4))
        out.append((pre + ".trellis", "I16", [i // 16, o // 16, 16 * K], (i // 16) * (o // 16) * 16 * K * 2))
    return out


def make(a):
    os.makedirs(a.dir, exist_ok=True)
    rng = np.random.default_rng(1)
    nfile = 2
    hdrs = [dict() for _ in range(nfile)]
    sizes = [0] * nfile
    plan = []       # (file, key, dtype, shape, nbytes, offset)
    for l in range(a.layers):
        for e in range(E):
            f = (l * E + e) % nfile
            for key, dt, shape, nb in tensors_of(rng, l, e):
                plan.append((f, key, dt, shape, nb, sizes[f]))
                hdrs[f][key] = {"dtype": dt, "shape": shape, "data_offsets": [sizes[f], sizes[f] + nb]}
                sizes[f] += nb
            sizes[f] += int(rng.integers(1, 700))
    for f in range(nfile):
        hb = json.dumps(hdrs[f]).encode()
        hb += b" " * ((-len(hb)) % 8)
        fn = os.path.join(a.dir, f"model-{f}.safetensors")
        with open(fn, "wb") as fh:
            import struct
            fh.write(struct.pack("<Q", len(hb)) + hb)
            base = 8 + len(hb)
            pos = 0
            for (ff, key, dt, shape, nb, off) in plan:
                if ff != f:
                    continue
                if off > pos:
                    fh.write(b"\0" * (off - pos))
                    pos = off
                # distinct bytes per tensor, cheap: random block xor a per-tensor stamp
                blk = rng.integers(0, 256, nb, dtype=np.uint8) if nb < 1 << 16 else None
                if blk is None:
                    if not hasattr(make, "pool"):
                        make.pool = rng.integers(0, 256, 1 << 26, dtype=np.uint8)
                    s = int(rng.integers(0, (1 << 26) - nb))
                    blk = make.pool[s:s + nb].copy()
                    blk[:8] = np.frombuffer(hashlib.sha1(key.encode()).digest()[:8], np.uint8)
                fh.write(blk.tobytes())
                pos += nb
            fh.flush()
            os.fsync(fh.fileno())
            os.posix_fadvise(fh.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        json.dump({"base": base}, open(fn + ".meta", "w"))
    print("pack written", a.dir, [s / 2**30 for s in sizes])


class FakeStc:
    def __init__(self, d, nfile=2):
        import struct
        self.tensor_file_map, self.file_headers = {}, {}
        for f in range(nfile):
            fn = os.path.join(d, f"model-{f}.safetensors")
            with open(fn, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                h = json.loads(fh.read(n))
            h["_header_offset"] = 8 + n
            self.file_headers[fn] = h
            for k in h:
                if k != "_header_offset":
                    self.tensor_file_map[k] = fn

    def find_stc(self, key):
        return self


class Lin:
    def __init__(self, key, i, o):
        self.key, self.in_features, self.out_features, self.out_dtype = key, i, o, None


def module(stc, l):
    lk = f"model.layers.{l}.mlp"
    from types import SimpleNamespace
    mk = lambda e, p, i, o: Lin(f"{lk}.experts.{e}.{p}", i, o)
    return SimpleNamespace(key=lk, gated=True, num_experts=E, num_experts_per_tok=6, num_local_experts=E, routing_first=None,
                           tp_mode=None, gates=[mk(e, "gate_proj", IN, MID) for e in range(E)],
                           ups=[mk(e, "up_proj", IN, MID) for e in range(E)], downs=[mk(e, "down_proj", MID, IN) for e in range(E)],
                           config=SimpleNamespace(stc=stc))


def run(a):
    import torch
    sys.path.insert(0, os.path.join(HERE, "..", "tier"))
    import prefetch_model as PM
    es = load_store()
    dev = torch.device(a.device)
    lay, rid, rowlen = PM.load(a.trace)
    rs = json.load(open(a.calib_json))["row_sources"]
    dom = [PM.DOM[rs[i]] for i in rid]
    order = PM.interleave(dom)
    L = a.layers
    layers = list(range(a.layer0, a.layer0 + L))
    T = 1 if a.N == 0 else a.N + 1
    adv = 1 if a.N == 0 else max(1, int(round(PM.accepted(a.N, a.accept_p))))
    steps = [(r, t) for r in order for t in range(0, rowlen - T + 1, adv)]
    if a.max_steps:
        steps = steps[:a.max_steps]
    ebytes = es_slot = None
    # emulated compute: spin kernel
    if dev.type == "cuda":
        torch.cuda._sleep(1 << 20); torch.cuda.synchronize()
        t0 = time.perf_counter(); torch.cuda._sleep(1 << 24); torch.cuda.synchronize()
        cyc_per_s = (1 << 24) / (time.perf_counter() - t0)

        def spin(sec):
            torch.cuda._sleep(int(sec * cyc_per_s))
    else:
        def spin(sec):
            end = time.perf_counter() + sec
            while time.perf_counter() < end:
                pass
    c1 = a.c1 * 1e-3
    s = 0.18 + (0.45 - 0.18) * max(0.0, min(1.0, (2.4 - a.c1) / (2.4 - 0.67)))
    te, att = c1 * s / 6, c1 * (1 - s) * (1 + 0.04 * (T - 1))
    arms = {
        "off": {"EXL3_EXPERT_AHEAD": "0"},
        "X1k6 late (store v1 order)": {"EXL3_EXPERT_AHEAD": "1", "EXL3_EXPERT_AHEAD_EARLY": "0", "EXL3_EXPERT_AHEAD_K": "0", "EXL3_EXPERT_AHEAD_DEPTH": "1"},
        "X1k6 early": {"EXL3_EXPERT_AHEAD": "1", "EXL3_EXPERT_AHEAD_K": "0", "EXL3_EXPERT_AHEAD_DEPTH": "1"},
        "X1k12 early": {"EXL3_EXPERT_AHEAD": "1", "EXL3_EXPERT_AHEAD_K": "12", "EXL3_EXPERT_AHEAD_DEPTH": "1"},
        "X1k16 early": {"EXL3_EXPERT_AHEAD": "1", "EXL3_EXPERT_AHEAD_K": "16", "EXL3_EXPERT_AHEAD_DEPTH": "1"},
        "X12 k12 early": {"EXL3_EXPERT_AHEAD": "1", "EXL3_EXPERT_AHEAD_K": "12", "EXL3_EXPERT_AHEAD_DEPTH": "2"},
        "X123 k12 early": {"EXL3_EXPERT_AHEAD": "1", "EXL3_EXPERT_AHEAD_K": "12", "EXL3_EXPERT_AHEAD_DEPTH": "3"},
    }
    sel = a.arms.split(";") if a.arms else list(arms)
    results, ref_sums = {}, None
    for name in sel:
        env = dict(arms[name], EXL3_EXPERT_BUDGET_GIB="0", EXL3_EXPERT_RING=str(a.ring), EXL3_EXPERT_RESERVE=str(a.ring // 3),
                   EXL3_EXPERT_READERS=str(a.readers), EXL3_EXPERT_CHUNK_KIB="1024")
        old = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        from types import SimpleNamespace
        stc = FakeStc(a.dir)
        cfg = SimpleNamespace(stc=stc)
        st = es.get_store(cfg)
        mods = [module(stc, l) for l in range(L)]
        for m in mods:
            m.routing_cfg = SimpleNamespace(gate_tensor=torch.zeros(1, E), router_bias=None, e_score_correction_bias=None, num_experts_per_tok=6)
        lts = [st.register(m, str(dev), shells=False) for m in mods]
        assert all(lts)
        cls = lts[0].cls
        nslots = int(a.cap * L * E)
        st.budget = nslots * cls.slot_bytes
        for lt in lts:
            lt.build_tables(dev)
        st.finalize()
        cur = {"a": 0, "b": 0}

        def predict(ts, z, nxt=None, k=0):
            nxt = nxt or ts.next
            if nxt is None:
                return None
            d = nxt.sidx - ts.sidx
            key = f"X{d}"
            arr = lay[layers[nxt.sidx]].get(key)
            if arr is None:
                return None
            kk = max(k, 6)
            return torch.from_numpy((arr[cur["a"]:cur["b"], :kk] % E).astype(np.int64)).to(dev)
        st._predict = predict
        if st.ahead:
            pass
        acc = torch.zeros(1, dtype=torch.int64, device=dev)
        sums = []
        zdummy = torch.zeros(T, 4, device=dev)
        stall = 0.0
        st.reset_stats()
        warm_n = int(len(steps) * a.warm)
        t_meas = None
        for si, (r, t0) in enumerate(steps):
            if si == warm_n:
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                st.reset_stats(); t_meas = time.perf_counter(); stall = 0.0
            a0, b0 = r * rowlen + t0, r * rowlen + t0 + T
            cur["a"], cur["b"] = a0, b0
            acc.zero_()
            for i, l in enumerate(layers):
                spin(att)
                picks = torch.from_numpy((lay[l]["idx"][a0:b0] % E).astype(np.int64)).to(dev)
                tp = time.perf_counter()
                st.prepare(lts[i], picks, {}, zdummy)
                stall += time.perf_counter() - tp
                u = np.unique(lay[l]["idx"][a0:b0] % E)
                # checksum over the bytes of the picked experts as the kernel would read them
                for e in u.tolist():
                    sl = int(lts[i].smap[e])
                    ch = cls.chunks[sl // cls.spc]
                    so = (sl % cls.spc) * cls.slot_bytes
                    acc += ch[so:so + cls.slot_bytes // 8 * 8].view(torch.int64)[::997].sum()
                spin(te * len(u))
            sums.append(acc.clone())
        if dev.type == "cuda":
            torch.cuda.synchronize()
        wall = time.perf_counter() - t_meas
        digest = hashlib.sha256(torch.cat(sums).cpu().numpy().tobytes()).hexdigest()
        sts = st.stats()
        n = len(steps) - warm_n
        results[name] = dict(steps_s=n / wall, tok_s=n / wall * (PM.accepted(a.N, a.accept_p) if a.N else 1.0),
                             step_ms=wall / n * 1e3, stall_ms=stall / len(steps) * 1e3, digest=digest, hit=sts["hit_rate"],
                             miss_per_step=sts["misses"] / n, gbps_wall=sts["gbps_wall"], bytes_per_step=sts["bytes"] / n,
                             ahead_issued=sts["ahead_issued"] / n, ahead_used=sts["ahead_used"] / n, ahead_wasted=sts["ahead_wasted"] / n,
                             ahead_cancelled=sts["ahead_cancelled"] / n, ahead_peak=sts["ahead_peak"])
        r_ = results[name]
        print(f"{name:<28} tok/s {r_['tok_s']:6.1f} step {r_['step_ms']:6.1f} ms  stall {r_['stall_ms']:6.1f}  hit {r_['hit']:.3f}  miss/step {r_['miss_per_step']:5.1f} "
              f"pfI {r_['ahead_issued']:5.1f} pfU {r_['ahead_used']:5.1f} pfW {r_['ahead_wasted']:5.1f} cancel {r_['ahead_cancelled']:5.1f} NVMe {r_['bytes_per_step'] * r_['steps_s'] / 1e9:4.2f} GB/s  sum {digest[:12]}", flush=True)
        for lt in lts:
            st.unregister(lt)
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    digs = {v["digest"] for v in results.values()}
    print("EXACT: all arms read identical expert bytes at every step" if len(digs) == 1 else f"MISMATCH: {len(digs)} different digests")
    if a.out:
        json.dump(dict(args=vars(a), results=results, exact=len(digs) == 1), open(a.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make"); m.add_argument("--dir", required=True); m.add_argument("--layers", type=int, default=3)
    r = sub.add_parser("run")
    r.add_argument("--dir", required=True); r.add_argument("--layers", type=int, default=3); r.add_argument("--layer0", type=int, default=3)
    r.add_argument("--trace", nargs="+", required=True); r.add_argument("--calib-json", required=True)
    r.add_argument("--device", default="cuda:0"); r.add_argument("--c1", type=float, default=0.7); r.add_argument("--N", type=int, default=0)
    r.add_argument("--accept-p", type=float, default=0.70); r.add_argument("--cap", type=float, default=0.65)
    r.add_argument("--ring", type=int, default=48); r.add_argument("--readers", type=int, default=8)
    r.add_argument("--max-steps", type=int, default=0); r.add_argument("--warm", type=float, default=0.25)
    r.add_argument("--arms"); r.add_argument("--out")
    a = ap.parse_args()
    make(a) if a.cmd == "make" else run(a)


if __name__ == "__main__":
    main()
