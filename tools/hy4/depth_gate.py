"""Draft-depth gate for the Hy4 spec loop (S7, perf/hy4-strix-spec2). Pure Python, no torch.

Idea: the base draft graph runs kb steps (default 2); a tail graph runs `ext` more steps (default 1) from the base chain's last state. After the base graph the
host knows the kb tokens and their confidences (max softmax of the draft head). The gate decides whether the tail step + its verify row are worth their price.

Chain model: the extension row is accepted only if all kb base drafts were accepted AND the extension draft is right. Its marginal tokens are
P3(s) = P(acc >= kb + ext | s) with s = product of the base confidences (probability that the base chain is right if the head were calibrated). Extend iff
    P3(s) * (tokens gained) > rate * extra_ms          rate = recent tokens per ms of this gate's own rounds (running means, same unit as the cost table)
    extra_ms = verify(1 + kb + ext) - verify(1 + kb) + ext * step_ms + gap_ms   (gap_ms = the host sync between the base and the tail graph)
P3 is learnt online per s-bin from the rounds that extended (prior: s * prior_q, weight prior_n); every PROBE-th round extends whatever the bin says, so a bin
that stopped extending can still be re-measured (no selection bias: the probe rounds are not chosen by s).

Determinism (S7b): the decision must depend only on the logits of THIS request. Three modes (HY4_SPEC2_MODE):
  static (default): extend iff s >= theta (HY4_SPEC2_THETA, default 0.6, from the pol_conf replay under the -2 % tau rule). No state at all.
  learn: the online rule above, reset() at every request start (fresh bins, fresh rate, probe counter from 0).
  carry: the online rule with NO reset (the job 17 behaviour: state of the previous requests changes the decisions, so text differs between identical reps). Kept as a negative control.
Lifetime counters (life_rounds / life_ext) survive reset(): the server log differences them per request. `digest` = crc32 of the decision bits of the current request.
"""
import os
import zlib

NBIN = 10
THETA = 0.6
MODE_DEFAULT = "learn"      # S7 arm O: the online rule with a reset at every request. static (a pure threshold, no state) and carry (no reset, negative control) are the other modes


def _table(s):
    t = {}
    for kv in s.split(","):
        a, b = kv.split(":")
        t[int(a)] = float(b)
    return t


class DepthGate:
    def __init__(self, verify_ms, step_ms=3.0, gap_ms=0.3, kb=2, ext=1, prior_n=6.0, prior_q=0.5, probe=16, alpha=0.02, tok0=2.3, mode="learn", theta=THETA):
        assert mode in ("static", "learn", "carry"), mode
        self.vt, self.step_ms, self.gap_ms, self.kb, self.ext = dict(verify_ms), step_ms, gap_ms, kb, ext
        self.prior_n, self.prior_q, self.probe, self.alpha, self.tok0 = prior_n, prior_q, probe, alpha, tok0
        self.mode, self.theta = mode, theta
        self.life_rounds = self.life_ext = 0                # lifetime counters (never reset): the server log differences them per request
        self.life_hist = {}                                 # lifetime histogram of the chosen draft depth {depth: rounds}
        self.life_changed = 0                               # lifetime rounds where the gate CHANGED the depth: it cut the chain below the shipped depth kb + ext
        self.reset(force=True)

    def reset(self, force=False):
        """start of a request: drop every adaptive state so the decisions depend on this request only (carry mode keeps the old behaviour)"""
        if self.mode == "carry" and not force:
            return
        self.hits = [[0.0, 0.0] for _ in range(NBIN)]       # per s-bin: [extension accepted, extension rounds]
        self.n_round = self.n_ext = 0
        self.tok_m = self.tok0                              # running mean tokens per round / ms per round (the rate is their ratio)
        self.ms_m = self.round_ms(False)
        self.dig = 0
        self.req_hist = {}                                  # same two counters, this request only
        self.req_changed = 0

    @classmethod
    def from_env(cls, kb, ext):
        t = _table(os.environ.get("HY4_ROW_MS", "1:61.8,2:76.7,3:90.3,4:104.0,5:117.3,6:130.6,7:143.8,8:157.0"))       # b2 arm C 4K graph replay, T3 / T5+ interpolated (spec2 job measures them)
        return cls(t, float(os.environ.get("HY4_SPEC2_STEP_MS", "3.0")), float(os.environ.get("HY4_SPEC2_GAP_MS", "0.3")), kb, ext,
                   float(os.environ.get("HY4_SPEC2_PRIOR_N", "6")), float(os.environ.get("HY4_SPEC2_PRIOR_Q", "0.5")), int(os.environ.get("HY4_SPEC2_PROBE", "16")),
                   mode=os.environ.get("HY4_SPEC2_MODE", MODE_DEFAULT), theta=float(os.environ.get("HY4_SPEC2_THETA", str(THETA))))

    def verify_ms(self, T):
        if T in self.vt:
            return self.vt[T]
        ks = sorted(self.vt)
        lo = max([x for x in ks if x < T], default=ks[0])
        hi = min([x for x in ks if x > T], default=ks[-1])
        if lo == hi:
            return self.vt[lo] + (T - lo) * (self.vt[ks[-1]] - self.vt[ks[-2]]) / max(1, ks[-1] - ks[-2])
        return self.vt[lo] + (self.vt[hi] - self.vt[lo]) * (T - lo) / (hi - lo)

    def round_ms(self, extended):
        n = self.ext if extended else 0
        return self.verify_ms(1 + self.kb + n) + self.step_ms * (self.kb + n) + (self.gap_ms if extended else 0.0)

    @staticmethod
    def sbin(confs):
        s = 1.0
        for c in confs:
            s *= min(1.0, max(0.0, c))
        return min(NBIN - 1, int(s * NBIN))

    def p_ext(self, b):
        a, n = self.hits[b]
        return (a + self.prior_n * self.prior_q * (b + 0.5) / NBIN) / (n + self.prior_n)

    def extra_ms(self):
        return self.round_ms(True) - self.round_ms(False)

    def decide(self, confs):
        """confs: confidences of the kb base drafts -> (extend?, bin)"""
        self.n_round += 1
        self.life_rounds += 1
        b = self.sbin(confs)
        s = 1.0
        for c in confs:
            s *= min(1.0, max(0.0, c))
        if self.mode == "static":
            go = s >= self.theta
        else:
            go = self.p_ext(b) * 1.0 > (self.tok_m / self.ms_m) * self.extra_ms()
            if self.probe and self.n_round % self.probe == 0:
                go = True
        self.life_ext += go
        depth = self.kb + (self.ext if go else 0)
        self.life_hist[depth] = self.life_hist.get(depth, 0) + 1
        self.req_hist[depth] = self.req_hist.get(depth, 0) + 1
        if not go:                                          # the shipped depth is kb + ext: a cut round is a round where the gate changed the depth
            self.life_changed += 1
            self.req_changed += 1
        self.dig = zlib.crc32(b"\x01" if go else b"\x00", self.dig)
        return go, b

    def update(self, b, extended, acc):
        """acc = accepted drafts of this round (clamped to the drafts verified by the caller)"""
        tokens = 1 + acc
        if self.mode == "static":
            self.n_ext += extended
            return
        if extended:
            self.n_ext += 1
            self.hits[b][1] += 1.0
            if acc >= self.kb + self.ext:
                self.hits[b][0] += 1.0
        self.tok_m += self.alpha * (tokens - self.tok_m)
        self.ms_m += self.alpha * (self.round_ms(extended) - self.ms_m)

    def stats(self):
        return {"rounds": self.life_rounds, "ext_rounds": self.life_ext, "req_rounds": self.n_round, "req_ext": self.n_ext, "mode": self.mode, "digest": f"{self.dig:08x}", "depth_hist": dict(sorted(self.life_hist.items())), "changed": self.life_changed, "req_depth_hist": dict(sorted(self.req_hist.items())), "req_changed": self.req_changed, "rate_tok_s": round(1000 * self.tok_m / self.ms_m, 2),
                "bins": [[round(a, 1), round(n, 1)] for a, n in self.hits]}

    def fires_line(self):
        """one line for the server log / an A/B driver: depth chosen per round (lifetime and this request) and the rounds where the gate changed the depth. changed = 0 means the arm did not fire."""
        return (f"FIRES gate mode={self.mode} depth_hist={dict(sorted(self.life_hist.items()))} changed={self.life_changed}/{self.life_rounds} "
                f"req_depth_hist={dict(sorted(self.req_hist.items()))} req_changed={self.req_changed}/{self.n_round}")


def draft_with_gate(gate, mg, out, kb, j):
    """out = host list of the base graph [kb tokens + kb confidences (fixed point 2^-20)]; j = head position of the round. Returns (drafts, extended, gate bin).
    The tail graph (mg.run_tail) is launched only when the gate says so: one extra draft step + one extra verify row."""
    d, confs = list(out[:kb]), [c / 1048576.0 for c in out[kb:2 * kb]]
    go, b = gate.decide(confs)
    if go:
        d += list(mg.run_tail(j).tolist()[:gate.ext])
    return d, go, b
