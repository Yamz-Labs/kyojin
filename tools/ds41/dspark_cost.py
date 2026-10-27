"""DSpark cost model: decode tok/s against draft depth N and accept rate, for a bandwidth-bound target.

    python3 dspark_cost.py                      # table + optimal N per accept rate + tokens/round needed for 25 tok/s
    python3 dspark_cost.py --measure            # also time draft / verify / decode on the CPU tiny model (proxy only)

Model (bytes moved per forward divided by effective bandwidth; compute and launch overhead are in `--overhead-ms`):

  decode step      = dense + routed(1)                      dense = shared expert, attention, hc, head, norms
  verify (n rows)  = dense + routed(n)                      routed(n) = 40 layers * E * (1 - (1 - 6/E)^n) * expert bytes
                                                            (n tokens pick 6 of E experts each; `--overlap` shrinks the
                                                             union when consecutive tokens share experts)
  draft pass       = 3 stages of (dense layer + routed(block))  = ~3/40 of a trunk pass at block rows
  round(N)         = verify(N+1) + passes(N) * draft pass,  passes = ceil(N / block)
  tokens / round   = sum_{i=0..N} prod_{j<=i} a_j           a_j = accept prob. of slot j (a for j < block, a * chain_decay after)
  tok/s            = tokens/round / round time

Numbers from the DS4.1 facts: ~16B active parameters, 384 routed experts top-6 + 1 shared, hidden 5120, moe_inter 2304, 40
layers, routed experts ~2.5 bpw, ~7.2 GB read per token in total, ~200 GB/s on the box. The routed share is derived
(40 * 6 * 3 * 5120 * 2304 * 2.5 / 8 = 2.65 GB), the rest (4.55 GB) is dense. All of them are flags.
"""
from __future__ import annotations

import argparse
import math
import sys
import time

LAYERS, E, TOPK = 40, 384, 6


class Cost:
    def __init__(self, gb_per_token=7.2, bw=200.0, eff=0.9, bpw_routed=2.5, hidden=5120, inter=2304, block=5,
                 stages=3, overlap=0.0, overhead_ms=0.0, chain_decay=0.85):
        routed = LAYERS * TOPK * 3 * hidden * inter * bpw_routed / 8 / 1e9        # GB per token
        self.routed_tok, self.dense = routed, gb_per_token - routed
        self.expert_gb = 3 * hidden * inter * bpw_routed / 8 / 1e9                # one expert of one layer
        self.bw, self.eff, self.block, self.stages = bw, eff, block, stages
        self.overlap, self.overhead, self.chain_decay = overlap, overhead_ms / 1000, chain_decay

    def routed(self, n: int, layers: int = LAYERS) -> float:
        union = E * (1 - (1 - TOPK / E) ** n)
        union = TOPK + (union - TOPK) * (1 - self.overlap)
        return layers * union * self.expert_gb

    def t(self, gb: float) -> float:
        return gb / (self.bw * self.eff) + self.overhead

    def decode_s(self) -> float:
        return self.t(self.dense + self.routed(1))

    def verify_s(self, rows: int) -> float:
        return self.t(self.dense + self.routed(rows))

    def draft_pass_s(self) -> float:
        per_layer_dense = (self.dense - 0.0) / LAYERS
        return self.t(self.stages * (per_layer_dense + self.routed(self.block, 1)))

    def round_s(self, n: int, drafted: int | None = None) -> float:
        """Round with n drafts sent to verify; `drafted` (default n) tokens were drafted, a confidence stop drops some after the fact."""
        d = n if drafted is None else drafted
        return self.verify_s(n + 1) + math.ceil(d / self.block) * self.draft_pass_s()

    def tokens_per_round(self, n: int, a: float) -> float:
        tot, p = 1.0, 1.0
        for j in range(n):
            p *= a if j < self.block else a * self.chain_decay
            tot += p
        return tot

    def tok_s(self, n: int, a: float) -> float:
        if n == 0:
            return 1 / self.decode_s()
        return self.tokens_per_round(n, a) / self.round_s(n)

    def best(self, a: float, nmax: int = 10) -> tuple[int, float]:
        return max(((n, self.tok_s(n, a)) for n in range(0, nmax + 1)), key=lambda x: x[1])

    def tokens_needed(self, n: int, target: float = 25.0) -> float:
        return target * (self.decode_s() if n == 0 else self.round_s(n))


class Measured(Cost):
    """Times measured on the box GPU with the synthetic 3-layer pack + 3 dspark stages (dspark_gpu_bench.py, dspark_gpu_layers.py),
    extrapolated to 40 layers. verify(n) = fixed(n) + 40 * per_layer(n), both from a least-squares fit over the 1/2/3-layer runs
    (`layers`: {L: {rows: ms}}), interpolated linearly in the row count; draft pass = measured ms (it does not scale with the
    trunk depth); `routed_scale` rescales the routed part when the real pack differs (1.0 = the synthetic mix)."""

    def __init__(self, data: dict, block=5, chain_decay=0.85, trunk_layers=LAYERS):
        self.block, self.chain_decay, self.stages = block, chain_decay, 3
        self.trunk, self.draft_ms = trunk_layers, float(data["draft_pass_ms"])
        rows = sorted({int(r) for v in data["layers"].values() for r in v})
        Ls = sorted(int(l) for l in data["layers"])
        self.fixed, self.per = {}, {}
        for r in rows:
            xs, ys = Ls, [float(data["layers"][str(l)][str(r)]) for l in Ls]
            n = len(xs)
            mx, my = sum(xs) / n, sum(ys) / n
            den = sum((x - mx) ** 2 for x in xs) or 1.0
            b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
            self.per[r], self.fixed[r] = b, my - b * mx
        self.rows = rows

    def _interp(self, tab, n):
        r = self.rows
        if n <= r[0]:
            return tab[r[0]]
        for a, b in zip(r, r[1:]):
            if n <= b:
                return tab[a] + (tab[b] - tab[a]) * (n - a) / (b - a)
        return tab[r[-1]] + (tab[r[-1]] - tab[r[-2]]) / (r[-1] - r[-2]) * (n - r[-1])

    def verify_s(self, rows):
        return (self._interp(self.fixed, rows) + self.trunk * self._interp(self.per, rows)) / 1000

    def decode_s(self):
        return self.verify_s(1)

    def draft_pass_s(self):
        return self.draft_ms / 1000

    def tok_s(self, n, a):
        return 1 / self.decode_s() if n == 0 else self.tokens_per_round(n, a) / self.round_s(n)


def report(c: Cost, target: float, nmax: int) -> str:
    out = []
    if hasattr(c, "dense"):
        out.append(f"dense {c.dense:.2f} GB, routed {c.routed_tok:.2f} GB/token; effective {c.bw * c.eff:.0f} GB/s")
    else:
        out.append("measured model (GPU timings of the synthetic pack, extrapolated to %d layers)" % c.trunk)
    out.append(f"plain decode {c.decode_s() * 1000:.1f} ms = {c.tok_s(0, 0):.1f} tok/s; draft pass {c.draft_pass_s() * 1000:.2f} ms "
               f"({c.draft_pass_s() / c.decode_s() * 100:.0f} % of a decode step)")
    out.append("")
    out.append("verify time (ms) and tokens/round needed for %.0f tok/s:" % target)
    out.append("  N  round_ms  verify_ms  draft_ms  tokens/round needed  min accept rate a (iid)")
    for n in range(1, nmax + 1):
        need = c.tokens_needed(n, target)
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if c.tokens_per_round(n, mid) < need else (lo, mid)
        feas = c.tokens_per_round(n, 1.0) >= need
        out.append(f"{n:3d}  {c.round_s(n) * 1000:7.1f}  {c.verify_s(n + 1) * 1000:9.1f}  "
                   f"{math.ceil(n / c.block) * c.draft_pass_s() * 1000:8.1f}  {need:19.2f}  "
                   f"{hi:6.2f}" + ("" if feas else "  (unreachable)"))
    out.append("")
    out.append("tok/s by N (rows) and per-slot accept rate a (columns); '*' = best N for that a")
    rates = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    out.append("   N " + " ".join(f"a={a:.1f}" for a in rates))
    best = {a: c.best(a, nmax)[0] for a in rates}
    for n in range(0, nmax + 1):
        out.append(f"{n:4d} " + " ".join(f"{c.tok_s(n, a):5.1f}{'*' if best[a] == n else ' '}" for a in rates))
    out.append("")
    out.append("optimal N: " + ", ".join(f"a={a:.1f}: N={best[a]} ({c.tok_s(best[a], a):.1f} tok/s)" for a in rates))
    return "\n".join(out)


def measure(args) -> str:
    """Per-step times on the CPU tiny model (serving graph). A proxy for ratios only: CPU is not bandwidth bound."""
    import os
    import torch
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    sys.path[:0] = [os.path.join(root, "tests", "dsv41"), here, root]
    torch.set_num_threads(min(8, args.threads))
    import test_dsv41_serving as S
    import test_dsv41_spec as T
    from exllamav3.dsv41.spec import PortSpec
    cfg, sd, tb, tm = S.make_world(2, sliding_window=16)
    m = S.build_serving(cfg, sd, tb, tm)
    spec = PortSpec(m, 256, max_depth=10)
    prompt = T.prompt_ids(cfg, 12, 31)
    spec.target.prefill(prompt)
    pos = len(prompt)

    def tm_(f, n=5):
        f(); t0 = time.perf_counter()
        for _ in range(n):
            f()
        return (time.perf_counter() - t0) / n

    out = spec.target.prefill(prompt)
    hid = out.hidden[-1:]
    d5 = tm_(lambda: PortSpecDraft(spec, 5, hid, pos))
    d10 = tm_(lambda: PortSpecDraft(spec, 10, hid, pos))
    spec.target.prefill(prompt)
    dec = tm_(lambda: _step(spec, 1, pos))
    res = [f"CPU tiny model (proxy): decode {dec * 1000:.1f} ms", f"draft pass depth 5: {d5 * 1000:.1f} ms ({d5 / dec:.2f} of a decode step), "
           f"depth 10 (2 passes): {d10 * 1000:.1f} ms ({d10 / dec:.2f})"]
    for k in (2, 4, 6, 10):
        v = tm_(lambda k=k: _step(spec, k + 1, pos))
        res.append(f"verify {k + 1} rows: {v * 1000:.1f} ms ({v / dec:.2f} of a decode step); draft5/verify = {d5 / v:.2f}")
    return "\n".join(res)


def PortSpecDraft(spec, depth, hid, pos):
    spec.drafter.max_depth = spec.drafter.block_size = depth
    spec.drafter._next = pos
    return spec.drafter.draft(1, hid, pos)


def _step(spec, rows, pos):
    t = spec.target
    t.s.length = pos
    out = t.verify([1] * rows, pos)
    t.truncate(pos)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gb-per-token", type=float, default=7.2)
    ap.add_argument("--bw", type=float, default=200.0, help="GB/s peak")
    ap.add_argument("--eff", type=float, default=0.9, help="achieved fraction of peak (180 GB/s = 25 tok/s plain)")
    ap.add_argument("--bpw-routed", type=float, default=2.5)
    ap.add_argument("--overlap", type=float, default=0.0, help="expert overlap between consecutive tokens, 0..1")
    ap.add_argument("--overhead-ms", type=float, default=0.0, help="fixed ms per forward (launch, sync)")
    ap.add_argument("--chain-decay", type=float, default=0.85, help="accept factor of slots beyond the first block")
    ap.add_argument("--target", type=float, default=25.0)
    ap.add_argument("--nmax", type=int, default=10)
    ap.add_argument("--measured", metavar="JSON", help="use the GPU measurements in this file (dspark_measured.json) instead of the bandwidth model")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    if a.measured:
        import json
        c = Measured(json.load(open(a.measured)), chain_decay=a.chain_decay)
        print(report(c, a.target, a.nmax))
        return
    c = Cost(a.gb_per_token, a.bw, a.eff, a.bpw_routed, overlap=a.overlap, overhead_ms=a.overhead_ms, chain_decay=a.chain_decay)
    print(report(c, a.target, a.nmax))
    if a.measure:
        print()
        print(measure(a))


if __name__ == "__main__":
    main()
