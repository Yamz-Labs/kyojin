"""Cost-aware draft length per round (generic: nothing model specific). Inputs: the drafter's confidence of each chained draft token and a verify-cost table
(ms of one verify step with T rows, measured on the machine). Chain model: draft i is accepted with probability p_i = cal(conf_i) given all earlier drafts were; the expected
tokens of a round that verifies L drafts are E(L) = 1 + sum_{i<=L} prod_{j<=i} p_j, its cost is overhead + verify_ms(1 + L); pick the L with the best E / cost (L = 0 allowed).
cal() = per-confidence-bin acceptance, learnt online (prior = the confidence itself, weight PRIOR_N), updated from every verified draft; every PROBE-th round verifies the whole
chain so the bins of long drafts keep learning. Same design as the Mac policy (expected accepted tokens x per-row cost table): only the cost table differs by machine."""
import os

NBIN = 10


def _table(s):
    t = {}
    for kv in s.split(","):
        a, b = kv.split(":")
        t[int(a)] = float(b)
    return t


class CostPolicy:
    def __init__(self, verify_ms, overhead_ms, prior_n=6.0, probe=16):
        self.vt = dict(verify_ms)
        self.ovh = overhead_ms
        self.prior_n, self.probe = prior_n, probe
        self.reset_stats()
        self.hits = [[0.0, 0.0] for _ in range(NBIN)]       # per bin: [accepted, total] observed (prior added in cal)

    @classmethod
    def from_env(cls):
        t = _table(os.environ.get("HY4_ROW_MS", "1:57,2:66,3:75,4:84,5:93,6:102,7:111,8:120"))
        return cls(t, float(os.environ.get("HY4_ROUND_OVH_MS", "12")), float(os.environ.get("HY4_POLICY_PRIOR_N", "6")), int(os.environ.get("HY4_POLICY_PROBE", "16")))

    def reset_stats(self):
        self.n_round = 0
        self.cut = 0                # rounds where fewer drafts than the chain length were verified
        self.rows_saved = 0
        self.lhist = {}

    def verify_ms(self, T):
        if T in self.vt:
            return self.vt[T]
        ks = sorted(self.vt)
        lo = max([x for x in ks if x < T], default=ks[0])
        hi = min([x for x in ks if x > T], default=ks[-1])
        if lo == hi:
            return self.vt[lo] + (T - lo) * (self.vt[ks[-1]] - self.vt[ks[-2]]) / max(1, ks[-1] - ks[-2]) if len(ks) > 1 else self.vt[lo]
        return self.vt[lo] + (self.vt[hi] - self.vt[lo]) * (T - lo) / (hi - lo)

    def cal(self, conf):
        b = min(NBIN - 1, int(conf * NBIN))
        a, n = self.hits[b]
        c = (b + 0.5) / NBIN
        return (a + self.prior_n * c) / (n + self.prior_n)

    def choose(self, confs, min_l=0):
        self.n_round += 1
        k = len(confs)
        if self.probe and self.n_round % self.probe == 0:
            best = k
        else:
            best, bs, surv, e = 0, 1.0 / (self.ovh + self.verify_ms(1)), 1.0, 1.0
            for L in range(1, k + 1):
                surv *= self.cal(confs[L - 1])
                e += surv
                sc = e / (self.ovh + self.verify_ms(1 + L))
                if sc > bs:
                    best, bs = L, sc
            best = max(best, min_l)
        if best < k:
            self.cut += 1
            self.rows_saved += k - best
        self.lhist[best] = self.lhist.get(best, 0) + 1
        return best

    def update(self, confs, nd, acc):
        """nd drafts were verified, the first `acc` of them accepted"""
        for i in range(min(nd, acc + 1, len(confs))):
            b = min(NBIN - 1, int(confs[i] * NBIN))
            self.hits[b][1] += 1.0
            if i < acc:
                self.hits[b][0] += 1.0

    def stats(self):
        return {"rounds": self.n_round, "cut_rounds": self.cut, "rows_saved": self.rows_saved, "lhist": dict(self.lhist),
                "bins": [[round(a, 1), round(n, 1)] for a, n in self.hits]}
