#!/usr/bin/env python
"""Per-dispatch decode census from a rocprofv3 --kernel-trace CSV.

  census.py TRACE.csv [--steps N] [--top 40] [--json out.json] [--dump out.tsv]

Reports, for one decode step (totals divided by --steps):
  * every dispatch: class, kernel, duration, and the gap before it (start_i - the
    latest end seen so far, so a dispatch queued behind another stream still shows
    the true hole in the GPU timeline);
  * per class: dispatches, busy ms, mean us/call, and the gap mass before the class;
  * the biggest gaps, attributed to the kernel that precedes them;
  * tiny-op chains: runs of consecutive dispatches shorter than --tiny us, i.e. the
    candidates a capture would have to cover for a graph to pay.

Timestamps are rocprofv3 ns on the mono or the boot clock; both are handled (the
clock that yields a positive timeline inside the trace is used).
"""
import argparse
import collections
import csv
import json
import re
import sys

# class of a kernel, by name. First match wins; order matters (norm before add, hc before norm).
CLASSES = [
    ("hc",        r"\bhc_|hc_mix|hc_apply|hc_pf|gr_dots|gr_finalize|gr_g1|gr_g2|_gr_"),
    ("router",    r"router|topk|top_k|softmax|sinkhorn|moe_gate|gate_softmax|\bgating\b"),
    ("norm",      r"rms_norm|layernorm|\bnorm\b|gated_rms|normalize|ple_norm"),
    ("rotary",    r"rope|rotary|freqs"),
    ("dequant",   r"dequant|exl3_dec|\bdq\b|_dq_|reconstruct|unpack|scales"),
    ("copy",      r"copy|memcpy|cat|_cast|cast_|contiguous|transpose|permute|fill|zero|slice|view_"),
    ("residual",  r"\badd\b|fused_add|residual|elementwise|_elt_|eltwise|mul_|fma"),
    ("attn",      r"attn|attention|qdec|qsa|dsa|mla|paged|kv_update|plane_update|stage_kernel|deinterleave|pool_update"),
    ("moe",       r"moe_|mpw|mf9|grouped|expert|union"),
    ("gemv",      r"gemv|hgemm|Cijk|blas|skinny|gemm"),
    ("gdn",       r"gdn|kda|recurrent|conv1d|delta_rule|chunk|wy_|inter_solve"),
    ("lm_head",   r"lm_head|_head|logits|sampl|argmax"),
    ("other",     r"."),
]
CLASSES = [(c, re.compile(p, re.I)) for c, p in CLASSES]


def klass(name):
    for c, rx in CLASSES:
        if rx.search(name):
            return c
    return "other"


def short(n):
    n = n.replace("(anonymous namespace)::", "")
    n = re.sub(r"^void\s+", "", n)
    return re.sub(r"\(.*$", "", n)[:96]


def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                s = int(r["Start_Timestamp"]); e = int(r["End_Timestamp"])
            except (KeyError, ValueError):
                continue
            rows.append((s, e, r.get("Kernel_Name", "?"), r.get("Stream_Id", "")))
    rows.sort()
    return rows


def census(rows, steps, tiny_us):
    if not rows:
        raise SystemExit("no rows")
    t0 = rows[0][0]
    # timeline with a running max end: the gap before a dispatch is the hole it waits in
    hi = rows[0][1]
    disp = []
    for s, e, n, st in rows:
        gap = max(0, s - hi)
        hi = max(hi, e)
        disp.append(dict(t_ns=s - t0, name=short(n), full=n, cls=klass(n), dur_us=(e - s) / 1e3,
                         gap_us=gap / 1e3, stream=st))
    wall_ms = (hi - t0) / 1e6
    busy_ns = sum(e - s for s, e, _, _ in rows)
    per_cls = collections.defaultdict(lambda: dict(n=0, us=0.0, gap_us=0.0, n_gap=0))
    for d in disp:
        c = per_cls[d["cls"]]
        c["n"] += 1; c["us"] += d["dur_us"]; c["gap_us"] += d["gap_us"]
    per_k = collections.defaultdict(lambda: dict(n=0, us=0.0, gap_us=0.0, cls=None))
    for d in disp:
        k = per_k[d["name"]]
        k["n"] += 1; k["us"] += d["dur_us"]; k["gap_us"] += d["gap_us"]; k["cls"] = d["cls"]
    # chains of tiny dispatches
    chains, cur = [], []
    for d in disp:
        if d["dur_us"] <= tiny_us:
            cur.append(d)
        else:
            if len(cur) >= 3:
                chains.append(cur)
            cur = []
    if len(cur) >= 3:
        chains.append(cur)
    return dict(disp=disp, wall_ms=wall_ms, busy_ms=busy_ns / 1e6, steps=steps, per_cls=per_cls,
                per_k=per_k, chains=chains)


def report(cc, top, tiny_us):
    s = cc["steps"]
    print(f"dispatches {len(cc['disp'])} ({len(cc['disp'])/s:.1f}/step)  wall {cc['wall_ms']/s:.3f} ms/step  "
          f"busy {cc['busy_ms']/s:.3f}  gaps {cc['wall_ms']/s - cc['busy_ms']/s:.3f}")
    print(f"\n{'class':10s} {'n/step':>8s} {'ms/step':>8s} {'us/call':>8s} {'gap ms':>7s} {'gap/call us':>11s} {'%busy':>6s}")
    for c, v in sorted(cc["per_cls"].items(), key=lambda kv: -kv[1]["us"]):
        print(f"{c:10s} {v['n']/s:8.1f} {v['us']/1e3/s:8.3f} {v['us']/v['n']:8.2f} "
              f"{v['gap_us']/1e3/s:7.3f} {v['gap_us']/v['n']:11.2f} {100*v['us']/cc['busy_ms']/1e3:6.2f}")
    print(f"\ntop kernels ({top})")
    print(f"{'kernel':46s} {'cls':9s} {'n/step':>7s} {'us/call':>8s} {'ms/step':>8s} {'gap/call':>9s}")
    for k, v in sorted(cc["per_k"].items(), key=lambda kv: -kv[1]["us"])[:top]:
        print(f"{k[:46]:46s} {v['cls']:9s} {v['n']/s:7.1f} {v['us']/v['n']:8.2f} {v['us']/1e3/s:8.3f} {v['gap_us']/v['n']:9.2f}")
    gaps = sorted(cc["disp"], key=lambda d: -d["gap_us"])[:top]
    print(f"\nbiggest gaps, attributed to the kernel before them")
    for i, d in enumerate(cc["disp"]):
        pass
    idx = {id(d): i for i, d in enumerate(cc["disp"])}
    for d in gaps:
        i = idx[id(d)]
        prev = cc["disp"][i - 1]["name"] if i else "(start)"
        nxt = cc["disp"][i + 1]["name"] if i + 1 < len(cc["disp"]) else "(end)"
        print(f"  {d['gap_us']:8.2f} us  after {prev[:40]:40s} before {d['name'][:40]:40s}")
    tot = sum(d["gap_us"] for d in gaps)
    print(f"  top-{top} gaps carry {tot/1e3/cc['steps']:.3f} of {cc['wall_ms']/cc['steps'] - cc['busy_ms']/cc['steps']:.3f} ms/step")
    print(f"\ntiny-op chains (>=3 consecutive dispatches <= {tiny_us} us)")
    allc = sorted(cc["chains"], key=lambda c: -sum(d["dur_us"] + d["gap_us"] for d in c))[:top]
    for c in allc:
        span = sum(d["dur_us"] + d["gap_us"] for d in c)
        names = " + ".join(collections.Counter(d["name"] for d in c).most_common(4)[0][0] for _ in [0])
        print(f"  len {len(c):2d} span {span:8.2f} us busy {sum(d['dur_us'] for d in c):7.2f}  "
              f"{', '.join(f'{k}x{v}' for k, v in collections.Counter(d['name'] for d in c).most_common(5))}")


def window_rows(path, rows):
    """Rows inside the biggest timed window of a depth_profile.py --kernels JSON (mono or boot clock)."""
    P = json.load(open(path))
    best = []
    for d, pt in P.get("points", {}).items():
        for kind, v in pt.items():
            w = v.get("window") if isinstance(v, dict) else None
            if not w:
                continue
            for clk in ("mono", "boot"):
                sel = [r for r in rows if w["start"][clk] <= r[0] <= w["end"][clk]]
                if len(sel) > len(best):
                    best, best_k = sel, (d, kind, w.get("steps", 1))
    if not best:
        raise SystemExit(f"no window in {path} contains any kernel")
    print(f"window {best_k[0]}/{best_k[1]}: {len(best)} dispatches / {best_k[2]} steps")
    return best, best_k[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--tiny", type=float, default=20.0)
    ap.add_argument("--prof", help="depth_profile.py JSON: restrict to its timed decode window")
    ap.add_argument("--json")
    ap.add_argument("--dump")
    a = ap.parse_args()
    rows = load(a.trace)
    if a.prof:
        rows, steps = window_rows(a.prof, rows)
        a.steps = steps
    cc = census(rows, a.steps, a.tiny)
    report(cc, a.top, a.tiny)
    if a.json:
        json.dump(dict(wall_ms=cc["wall_ms"], busy_ms=cc["busy_ms"], steps=cc["steps"],
                       per_cls={k: v for k, v in cc["per_cls"].items()},
                       per_k={k: v for k, v in cc["per_k"].items()},
                       disp=cc["disp"]), open(a.json, "w"), indent=1)
    if a.dump:
        with open(a.dump, "w") as f:
            f.write("i\tcls\tus\tgap_us\tname\n")
            for i, d in enumerate(cc["disp"]):
                f.write(f"{i}\t{d['cls']}\t{d['dur_us']:.3f}\t{d['gap_us']:.3f}\t{d['name']}\n")


if __name__ == "__main__":
    main()
