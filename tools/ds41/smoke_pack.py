"""GPU smoke on a small synthetic DS4.1 pack: grouped-vs-loop exactness + decode/prefill timing table.

usage: smoke_pack.py PACK [--layers 1,2,3] [--prefill 2048] [--reps 5] [--rel 1e-3] [--json OUT]
Exit code 1 when argmax differs, rel > --rel, or logits are not finite.
"""
import argparse, gc, json, os, statistics, sys, time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from ds41_serving import build_model


def med(xs):
    return statistics.median(xs)


def run(pack, n, moe, pf, reps):
    t0 = time.time()
    m = build_model(pack, device="cuda:0", dtype="float16", layers=n, mtp=False, moe=moe, prefill_chunk=pf)
    torch.cuda.synchronize()
    load_s = time.time() - t0
    g = torch.Generator(device="cuda:0").manual_seed(5)
    ids = torch.randint(0, m.cfg.vocab_size - 1, (1, pf), device="cuda:0", generator=g)
    st = m.new_state(1, pf + 128 + 16 * reps)
    m.prefill(ids[:, :64], st, 0, chunk=64)
    torch.cuda.synchronize()
    tp = []
    lg = None
    for _ in range(reps):
        st = m.new_state(1, pf + 128 + 16 * reps)
        torch.cuda.synchronize()
        t = time.time()
        lg, _ = m.prefill(ids, st, 0, chunk=pf)
        torch.cuda.synchronize()
        tp.append(time.time() - t)
    pos = pf
    tok = lg.argmax(-1).view(1, 1)
    for _ in range(3):
        lg, _ = m.forward(tok, pos, st)
        pos += 1
        tok = lg.argmax(-1).view(1, 1)
    first = None
    td = []
    fin = True
    N = 16
    for r in range(reps):
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(N):
            lg, _ = m.forward(tok, pos, st)
            pos += 1
            tok = lg.argmax(-1).view(1, 1)
            if first is None:
                first = lg.float().cpu()
        torch.cuda.synchronize()
        td.append((time.time() - t) / N)
        fin = fin and bool(torch.isfinite(lg).all())
    kinds = sorted({type(b).__name__ for b in m.moe_backends.values()}) if hasattr(m, "moe_backends") else []
    row = dict(layers=n, moe=moe, backends=kinds, load_s=round(load_s, 1), prefill_ms=round(med(tp) * 1e3, 1),
               prefill_tok_s=round(pf / med(tp)), decode_ms=round(med(td) * 1e3, 2), finite=fin,
               vram_gib=round(torch.cuda.memory_allocated() / 2**30, 1))
    del m, st
    gc.collect()
    torch.cuda.empty_cache()
    return row, first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pack")
    ap.add_argument("--layers", default="1,2,3")
    ap.add_argument("--prefill", type=int, default=2048)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--rel", type=float, default=1e-3)
    ap.add_argument("--json")
    a = ap.parse_args()
    Ns = [int(x) for x in a.layers.split(",")]
    rows, firsts = [], {}
    for n in Ns:
        r, f = run(a.pack, n, "auto", a.prefill, a.reps)
        rows.append(r)
        firsts[n] = f
        print(json.dumps(r), flush=True)
    nl = Ns[-1]
    ref_row, ref = run(a.pack, nl, "loop", a.prefill, 2)
    print(json.dumps(ref_row), flush=True)
    d = (firsts[nl] - ref).abs().max().item()
    rel = d / ref.abs().max().item()
    argmax_eq = bool(firsts[nl].argmax() == ref.argmax())
    ok = argmax_eq and rel <= a.rel and all(r["finite"] for r in rows)
    print(f"EXACT grouped_vs_loop layers={nl} max_abs={d:.3e} rel={rel:.3e} argmax_equal={argmax_eq} limit={a.rel:g} -> {'PASS' if ok else 'FAIL'}")
    print("SPEED layers | prefill%d ms (tok/s) | decode ms/tok (tok/s)" % a.prefill)
    for r in rows:
        print(f"SPEED {r['layers']:>6} | {r['prefill_ms']:>8} ({r['prefill_tok_s']:>5}) | {r['decode_ms']:>7} ({1000/r['decode_ms']:.1f})")
    if len(rows) >= 2:
        a0, a1 = rows[0], rows[-1]
        dn = a1["layers"] - a0["layers"]
        pl = (a1["decode_ms"] - a0["decode_ms"]) / dn
        pp = (a1["prefill_ms"] - a0["prefill_ms"]) / dn
        fixed = a0["decode_ms"] - pl * a0["layers"]
        ext = fixed + 40 * pl
        print(f"SPEED slope: decode {pl:.2f} ms/layer + {fixed:.1f} ms fixed -> 40 layers ~{ext:.0f} ms ({1000/ext:.1f} tok/s); prefill {pp:.0f} ms/layer -> ~{a.prefill/(40*pp/1e3):.0f} tok/s")
    if a.json:
        json.dump(dict(rows=rows, loop=ref_row, rel=rel, argmax_equal=argmax_eq, ok=ok), open(a.json, "w"), indent=1)
    sys.exit(0 if ok else 1)


main()
