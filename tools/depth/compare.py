#!/usr/bin/env python
"""Compare the ideal curve (ideal.py --json) with a measured profile (profile.py -o) and gate it.

  compare.py IDEAL.json PROFILE.json [--tol 0.05] [--floor-ms 0.3] [--top 12]

Per step type (decode, verify, prefill chunk) and depth D:
    ideal growth   g(D) = ideal_ms(D) - ideal_ms(D0)        (what physics adds with depth)
    measured       m(D)                                      (D0 = shallowest profiled depth)
    allowed        (1 + tol) * (m(D0) + g(D)) + floor_ms     (anchored at the measured shallow cost)
    excess         m(D) - (m(D0) + g(D))                     (non-inherent depth cost, ms)
PASS when m(D) <= allowed at every depth for every step type. Anchoring at m(D0) keeps the gate about DEPTH
(a model whose absolute speed is far from the roofline is a different gate); floor_ms absorbs timer noise.
With kernel tables (profile.py --kernels) it also ranks kernels by growth D0 -> Dmax and shows GPU busy vs
wall (the gap is host / launch / sync cost). Exit code 0 = PASS, 1 = FAIL.
"""
import argparse, json, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ideal"); ap.add_argument("profile")
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--floor-ms", type=float, default=0.3)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--json")
    a = ap.parse_args()
    I, P = json.load(open(a.ideal)), json.load(open(a.profile))
    ib = {b["depth"]: b for b in I["breakdown"]}
    depths = sorted(int(d) for d in P["points"])
    kinds = (("decode", lambda b: b["decode_ms"], "decode"), ("verify", lambda b: b["verify_ms"], "verify"),
             ("prefill", lambda b: b["prefill_chunk_ms"], "prefill"))
    fail, rep = [], {}
    for name, ideal_ms, key in kinds:
        pts = {d: P["points"][str(d)].get(key) for d in depths if P["points"][str(d)].get(key)}
        if not pts:
            continue
        dd = sorted(pts); d0 = dd[0]
        if any(d not in ib for d in dd):
            sys.exit(f"ideal.json lacks depth(s) {[d for d in dd if d not in ib]}: rerun ideal.py --depths {','.join(map(str, dd))}")
        # the ideal at the profiled chunk size only matters for prefill: ideal.py --chunk must equal the profile's chunk
        if name == "prefill" and I["chunk"] != pts[d0]["chunk"]:
            sys.exit(f"chunk mismatch: ideal {I['chunk']} vs profile {pts[d0]['chunk']}")
        m0, g0 = pts[d0]["ms"], ideal_ms(ib[d0])
        print(f"\n{name}: measured {m0:.2f} ms at {d0} (ideal {g0:.2f} ms = {100*g0/m0:.0f}% of measured)")
        print(f"{'depth':>8} {'meas ms':>9} {'ideal d':>8} {'excess':>8} {'allowed':>9}  verdict")
        rep[name] = []
        for d in dd:
            m, g = pts[d]["ms"], ideal_ms(ib[d]) - g0
            allowed = (1 + a.tol) * (m0 + g) + a.floor_ms
            ok = m <= allowed
            rep[name].append(dict(depth=d, meas_ms=m, ideal_growth_ms=round(g, 3), excess_ms=round(m - m0 - g, 3), allowed_ms=round(allowed, 3), ok=ok))
            print(f"{d:>8} {m:9.2f} {g:+8.2f} {m-m0-g:+8.2f} {allowed:9.2f}  {'ok' if ok else 'FAIL'}")
            if not ok:
                fail.append((name, d))
    # kernel growth ranking
    for key in ("decode", "verify", "prefill"):
        dk = [d for d in depths if "kernels" in P["points"][str(d)].get(key, {})]
        if len(dk) < 2:
            continue
        d0 = dk[0]; lo, hi = P["points"][str(d0)][key], P["points"][str(dk[-1])][key]
        if "kernels" not in lo or "kernels" not in hi:
            continue
        k0 = {k: (v, c) for k, v, c in lo["kernels"]["top"]}; k1 = {k: (v, c) for k, v, c in hi["kernels"]["top"]}
        rows = sorted(((k, k0.get(k, (0, 0))[0], k1.get(k, (0, 0))[0], k0.get(k, (0, 0))[1], k1.get(k, (0, 0))[1]) for k in set(k0) | set(k1)),
                      key=lambda r: r[2] - r[1], reverse=True)
        gap = lambda p: p["kernels"]["wall_us"] - p["kernels"]["busy_us"]
        print(f"\n{key}: kernels at {d0} -> {dk[-1]}: busy {lo['kernels']['busy_us']/1e3:.2f} -> {hi['kernels']['busy_us']/1e3:.2f} ms, "
              f"wall-busy gap {gap(lo)/1e3:.2f} -> {gap(hi)/1e3:.2f} ms, launches {lo['kernels']['kernels_per_step']:.0f} -> {hi['kernels']['kernels_per_step']:.0f}")
        print(f"  {'kernel':<70} {'us@D0':>9} {'us@Dmax':>9} {'growth':>9} {'calls':>11}")
        for k, u0, u1, c0, c1 in rows[: a.top]:
            print(f"  {k[:70]:<70} {u0:9.0f} {u1:9.0f} {u1-u0:+9.0f} {c0:5.1f}->{c1:5.1f}")
        rep.setdefault("kernels", {})[key] = [dict(kernel=r[0], us_d0=r[1], us_dmax=r[2]) for r in rows[: a.top]]
    print("\nGATE", "PASS" if not fail else f"FAIL {fail}", f"(tol {a.tol:.0%}, floor {a.floor_ms} ms)")
    if a.json:
        json.dump(dict(gate=not fail, tol=a.tol, rep=rep), open(a.json, "w"), indent=1)
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
