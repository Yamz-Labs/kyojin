"""Inkling2 HIP-graph step jobs (G2b): the wired eager step (INK_GLUE=all) captured once per T bucket (1, 2, 4, 8) with torch.cuda.CUDAGraph.

  graph_job.py --gguf G --mode capture|parity|decode [--depths 1024,8192] [--steps 8] [--sched 1,2,4,8,1,2,4,8] [--out DIR]

capture: capture the four buckets, report the time and the first error (full traceback per bucket).
parity : eager schedule (logits of every row + hidden), reset in place, graph schedule on the same ids, reset, graph schedule again. Pass = per row cos >= 0.99999 and top-1 equal
         (bit-identical is reported); fires: graph_replays == 2 x schedule length, graph_eager 0, every glue stage fired in warm-up / capture.
decode : per depth (synthetic KV like decode_job.py): 20 s settle + one warm step, then for T = 1, 2, 4, 8: eager arm and graph arm, `--steps` timed steps each at FIXED depth (n rewound
         before every step), median ms / step, spread rule (> 5 % = rerun once, then VOID). Output DIR/graph_<mode>.json, `FIRES graph <n>`; exit 7 when a fires-check or bar fails."""
import argparse
import json
import os
import statistics
import sys
import time
import traceback

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cpu_env  # noqa: E402

TREE = os.path.abspath(os.path.join(HERE, "..", ".."))
H = os.environ.get("INK_H", os.path.expanduser("~"))
if os.environ.get("INK_GUARD_DIR"):
    sys.path.insert(0, os.environ["INK_GUARD_DIR"])      # optional runner-side drv_guard; default = the one next to this file
import drv_guard as guard  # noqa: E402
BUCKETS = (1, 2, 4, 8)


def cosine(a, b):
    a, b = a.reshape(-1).astype(np.float64), b.reshape(-1).astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def build(a):
    F = cpu_env.install(TREE)
    from exllamav3.loader.gguf_collection import GGUFCollection
    m = F.Inkling2Forward(GGUFCollection(a.gguf), device=a.device)
    assert m.gl is not None and m.gl.on >= set(m.gl.STAGES_ALL), "run with INK_GLUE=all"
    assert not m.graphs, "INK_GLUE_GRAPH must be unset: the job captures explicitly"
    return F, m


def capture(m, buckets=BUCKETS):
    t0 = time.time()
    if m.gl.has("experts"):
        m.gl.preload()
        print(f"experts resident in {time.time() - t0:.1f} s", flush=True)
    t0 = time.time()
    m.capture_graphs(buckets)
    return time.time() - t0


def mode_capture(a, m):
    ok = {}
    for T in BUCKETS:
        try:
            m.capture_graphs((T,))
            ok[T] = "ok"
        except Exception:      # noqa: BLE001
            ok[T] = traceback.format_exc()[-1800:]
            print(f"CAPTURE FAILED T={T}\n{ok[T]}", flush=True)
    return dict(buckets=ok), all(v == "ok" for v in ok.values())


def mode_parity(a, m):
    ids = [(1009 * i * i + 7919 * i + 17) % 150000 + 5 for i in range(200)]
    if a.ids:                                                                   # real text (file:class of the teacher ids): router margins are healthy, no chaotic near-tie rows
        f, c = a.ids.split(":")
        ids = [int(x) for x in json.load(open(f))["classes"][c]["ids"][:200]]
        print(f"ids from {a.ids}: {ids[:6]}", flush=True)
    sched = [int(x) for x in a.sched.split(",")]
    assert all(T in BUCKETS for T in sched)

    def run():
        out, hid, p = [], [], 0
        for T in sched:
            lg = m.step(ids[p:p + T], all_pos=True)
            out.append(lg.float().cpu().numpy().copy())
            hid.append(m.last_h.float().cpu().numpy().copy())
            p += T
        return out, hid
    m.reset()
    eager, eh = run()
    m.reset()
    eager2, _ = run()                                                           # eager vs eager: the noise floor of the kernels themselves (atomics / split order)
    ee = [cosine(x, y) for e1, e2 in zip(eager, eager2) for x, y in zip(e1, e2)]
    print(f"eager vs eager (same inputs, reset between): min cos {min(ee):.8f}, rows below 0.99999: {sum(c < 0.99999 for c in ee)}", flush=True)
    m.reset()
    secs = capture(m)
    print(f"graphs captured in {secs:.1f} s", flush=True)
    rows, bad, graph_runs = [], 0, []
    for rep in (1, 2):
        m.reset()
        m.kstats["graph_replays"] = 0
        got, gh = run()
        graph_runs.append(got)
        assert m.kstats["graph_replays"] == len(sched), m.kstats
        p = 0
        for i, T in enumerate(sched):
            for r in range(T):
                e, g = eager[i][r], got[i][r]
                c = cosine(e, g)
                t1 = int(np.argmax(e)) == int(np.argmax(g))
                hc = cosine(eh[i][r], gh[i][r])
                ident = bool(np.array_equal(e, g))
                ok = c >= 0.99999 and t1
                bad += not ok
                rows.append(dict(rep=rep, step=i, T=T, row=r, pos=p + r, cos=round(c, 8), cos_hidden=round(hc, 8), top1=t1, bit_identical=ident, maxabs=float(np.abs(e - g).max()), ok=ok))
                if not ok or r == T - 1:
                    print(f"rep {rep} step {i} T={T} row {r} pos {p + r}: cos {c:.8f} hidden {hc:.8f} top1 {t1} bit-identical {ident} maxabs {rows[-1]['maxabs']:.3e}{'' if ok else '  <-- FAIL'}", flush=True)
            p += T
    gg = [cosine(x, y) for g1, g2 in zip(graph_runs[0], graph_runs[1]) for x, y in zip(g1, g2)]
    print(f"graph rep 1 vs graph rep 2: min cos {min(gg):.8f}, rows below 0.99999: {sum(c < 0.99999 for c in gg)}", flush=True)
    nid = sum(r["bit_identical"] for r in rows)
    print(f"parity rows {len(rows)}: bit-identical {nid}, below bar {bad}; min cos {min(r['cos'] for r in rows):.8f}", flush=True)
    return dict(rows=rows, capture_s=round(secs, 1), bit_identical=nid, n_rows=len(rows), sched=sched, eager_vs_eager_min_cos=min(ee), graph_vs_graph_min_cos=min(gg)), bad == 0


def mode_determinism(a, m):
    """eager step schedule twice (reset between), hooks record l_out / router picks / attn_out / ffn_out per application: the FIRST (step, application, stage) where the two runs differ"""
    ids = [(1009 * i * i + 7919 * i + 17) % 150000 + 5 for i in range(200)]
    if a.ids:
        f, c = a.ids.split(":")
        ids = [int(x) for x in json.load(open(f))["classes"][c]["ids"][:200]]
    sched = [int(x) for x in a.sched.split(",")]
    rec = {}

    def hook(name, il, t):
        rec.setdefault((name, il), []).append(t.detach().float().cpu().numpy().copy())
    m.hook = hook
    runs = []
    for _ in range(3):
        m.reset()
        rec.clear()
        p = 0
        for T in sched:
            m.step(ids[p:p + T], all_pos=True)
            p += T
        runs.append(dict(rec))
    print("zero_reset:", os.environ.get("INK_GLUE_ZERO_RESET"), os.environ.get("INK_GLUE_ZERO_ONLY"), flush=True)
    def cells(x, y):
        n = 0
        for k in runs[x]:
            for st in range(len(sched)):
                n += not np.array_equal(runs[x][k][st], runs[y][k][st])
        return n
    print(f"differing cells: run0 vs run1 {cells(0, 1)}, run0 vs run2 {cells(0, 2)}, run1 vs run2 {cells(1, 2)}", flush=True)
    order = ["layer_inp", "attn_out", "ffn_out", "router_sel", "l_out"]
    for r in (1, 2):
        first = None
        n_diff = 0
        for step in range(len(sched)):
            for il in range(m.n_app):
                for nm in order:
                    k = (nm, il)
                    if k not in runs[0]:
                        continue
                    x, y = runs[0][k][step], runs[r][k][step]
                    if not np.array_equal(x, y):
                        n_diff += 1
                        if first is None:
                            first = (step, il, nm, float(np.abs(x.astype(np.float64) - y).max()), int(np.sum(x != y)))
        print(f"run 0 vs run {r}: differing (step, app, stage) cells {n_diff}; first {first}", flush=True)
    return dict(sched=sched), True


def mode_decode(a, m):
    dev = m.dev
    secs = capture(m)
    print(f"graphs captured in {secs:.1f} s", flush=True)
    g = torch.Generator(device="cpu").manual_seed(7)
    graphs = m.graphs
    res, void_any = [], False
    sync = torch.cuda.synchronize
    for depth in [int(x) for x in a.depths.split(",")]:
        m.reset()
        m.gl.fill_synthetic(depth, g)
        m.n = depth
        tok = 1000
        m.graphs = graphs
        for _ in range(2):
            tok = int(torch.argmax(m.step([tok])))
        time.sleep(20)                                                          # settle (measuring rule), then one warm decode
        m.n = depth
        tok = int(torch.argmax(m.step([tok])))

        def arm(T, use_graph):
            m.graphs = graphs if use_graph else {}
            ids = [tok + 1 + j for j in range(T)]

            def block():
                ts = []
                for _ in range(a.steps):
                    m.n = depth                                              # fixed depth: rewind the position (the cache rows are overwritten in place)
                    sync()
                    t0 = time.time()
                    lg = m.step(ids)
                    int(torch.argmax(lg))                                    # the host sees the token: the step ends here, like a decode loop
                    sync()
                    ts.append(time.time() - t0)
                return ts
            m.n = depth
            m.step(ids)
            ts = block()
            spread = (max(ts) - min(ts)) / statistics.median(ts) * 100
            rerun = False
            if spread > 5.0:
                time.sleep(20)
                ts, rerun = block(), True
                spread = (max(ts) - min(ts)) / statistics.median(ts) * 100
            return dict(depth=depth, T=T, path="graph" if use_graph else "eager", ms=round(statistics.median(ts) * 1e3, 2), spread_pct=round(spread, 1), void=spread > 5.0, reran=rerun,
                        steps_ms=[round(t * 1e3, 1) for t in ts])
        rep0 = m.kstats["graph_replays"]
        for T in BUCKETS:
            for use_graph in (False, True):
                r = arm(T, use_graph)
                void_any |= r["void"]
                res.append(r)
                print("ROW", json.dumps(r), flush=True)
        m.graphs = graphs
        print(f"depth {depth}: graph replays {m.kstats['graph_replays'] - rep0}", flush=True)
    return dict(rows=res, capture_s=round(secs, 1)), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--mode", required=True, choices=("capture", "parity", "decode", "determinism"))
    ap.add_argument("--depths", default="1024,8192")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--ids", default="", help="parity: FILE:CLASS of teacher_ids_ink2_v1.json (default: hashed ids)")
    ap.add_argument("--sched", default="1,2,4,8,1,2,4,8")
    ap.add_argument("--out", default="ink_graph_out")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    F, m = build(a)
    try:
        res, ok = {"capture": mode_capture, "parity": mode_parity, "decode": mode_decode, "determinism": mode_determinism}[a.mode](a, m)
    except Exception:      # noqa: BLE001
        print("JOB FAILED\n" + traceback.format_exc(), flush=True)
        sys.exit(7)
    import ink2_glue
    gbad = ink2_glue.fires_check(m) if a.mode != "capture" else []
    print("KSTATS", json.dumps(m.kstats), flush=True)
    print("GLUE", sorted(m.gl.on), json.dumps(dict(m.gl.fires)), "fires-check:", gbad or "ok", flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(res, kstats=m.kstats, glue_fires=dict(m.gl.fires), fires_check=gbad), open(os.path.join(a.out, f"graph_{a.mode}.json"), "w"), indent=1)
    print(f"FIRES graph {m.kstats['graph_replays']}")
    print("VERDICT", "PASS" if ok and not gbad else "FAIL")
    sys.exit(0 if ok and not gbad else 7)


if __name__ == "__main__":
    guard.main_guard(main)
