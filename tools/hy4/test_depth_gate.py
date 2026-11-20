"""CPU test of the HY4_SPEC2_GATE logic with a fake engine (pure Python, no torch / GPU): python3 tools/hy4/test_depth_gate.py
Fake trunk = deterministic next-token function f; fake draft graphs = f with noise, confidence correlated with correctness. The round loop copies the engine's
accept / rewind code (engine.generate). Checks: (1) the emitted stream equals the greedy stream of f for ANY gate decision (fixed depth 2, 3, random, DepthGate),
(2) the tail graph is launched only when the gate extends and at head position j + kb, (3) fw.n / tokens stay consistent after every rewind, rows <= TM,
(4) the gate learns: chat-like source (low confidence) -> few extensions, code-like -> many, and its tokens / ms beats the worse fixed depth."""
import os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from depth_gate import DepthGate, draft_with_gate

V, TM = 1000, 8
f = lambda a, b: (a * 31 + b * 17 + 7) % V
ROW = {1: 61.8, 2: 76.7, 3: 90.3, 4: 104.0, 5: 117.3, 6: 130.6, 7: 143.8, 8: 157.0}


class FakeGR:                                   # the verify graph decoder: rows by position, rewind = a length
    def __init__(self, ids):
        self.tokens, self.n = list(ids), len(ids)

    def step_rows(self, toks):
        assert 1 <= len(toks) <= TM
        start = self.n
        self.tokens.extend(toks)
        self.n += len(toks)
        assert len(self.tokens) == self.n
        return [self._nxt(start + i) for i in range(len(toks))]

    def _nxt(self, pos):                         # argmax after row at `pos`: f of the two latest tokens of the sequence
        return f(self.tokens[pos], self.tokens[pos - 1])

    def rewind(self, n):
        assert n <= self.n
        del self.tokens[n:]
        self.n = n


class Out:
    def __init__(self, v):
        self.v = v

    def tolist(self):
        return list(self.v)


class FakeMG:                                    # base graph (kb steps) + tail graph; head rows are written by absolute position
    def __init__(self, gr, kb, ext, p_ok, rng, conf_noise=0.15):
        self.gr, self.k, self.ext, self.p_ok, self.rng, self.cn = gr, kb, ext, p_ok, rng, conf_noise
        self.last, self.j, self.tail_calls, self.tail_pos = None, None, 0, []

    def _step(self, a, b):
        ok = self.rng.random() < self.p_ok
        t = f(a, b) if ok else (f(a, b) + 1 + self.rng.randrange(V - 1)) % V
        conf = min(1.0, max(0.0, (0.9 if ok else 0.35) + self.rng.uniform(-self.cn, self.cn)))
        return t, conf

    def run(self, y, j):
        seq = self.gr.tokens[:self.gr.n]            # the trunk sequence up to and including y
        assert seq[-1] == y and j == self.gr.n - 1
        toks, confs, a, b = [], [], y, seq[-2]
        for i in range(self.k):
            t, c = self._step(a, b)
            toks.append(t), confs.append(int(c * 1048576))
            a, b = t, a
        self.last, self.j = (a, b), j
        return Out(toks + confs)

    def run_tail(self, j):
        assert self.j == j, "tail launched for another round than the base graph"
        self.tail_calls += 1
        self.tail_pos.append(j + self.k)
        toks, confs, a, b = [], [], *self.last
        for i in range(self.ext):
            t, c = self._step(a, b)
            toks.append(t), confs.append(int(c * 1048576))
            a, b = t, a
        return Out(toks + confs)


def generate(policy, p_ok, n_new=400, seed=1, kfix=3, gate=None):
    """policy: 'fixed' (kfix drafts, no gate) or 'gate'. Returns (emitted tokens, ms, rounds, gr, mg)"""
    rng = random.Random(seed)
    ids = [5, 9, 11]
    gr = FakeGR(ids)
    y = f(ids[-1], ids[-2])
    gr.tokens.append(y)
    gr.n += 1                                    # y emitted, sits at position n - 1 (the engine: fw.n counts it after the first step; here the same relation j = n - 1)
    out, rounds, ms = [y], 0, 0.0
    kb = gate.kb if gate else kfix
    mg = FakeMG(gr, kb, gate.ext if gate else 0, p_ok, rng)
    while len(out) < n_new:
        j = gr.n - 1
        dd = mg.run(y, j).tolist()
        if gate:
            d, gext, gb = draft_with_gate(gate, mg, dd, kb, j)
        else:
            d, gext, gb = dd[:kb], False, None
        toks = [y] + d
        gr.rewind(gr.n - 1)                      # row 0 = y is re-verified at its own position, like step_rows(toks) from fw.n - 1 + 1
        start = gr.n
        amh = gr.step_rows(toks)
        acc = 0
        nd = len(toks) - 1
        while acc < nd and toks[acc + 1] == amh[acc]:
            acc += 1
        new = toks[1:acc + 1] + [amh[acc]]
        gr.rewind(start + 1 + acc)               # keep y and the accepted drafts; the next y is amh[acc] appended below
        y = amh[acc]
        gr.tokens.append(y)
        gr.n += 1
        assert len(gr.tokens) == gr.n and gr.tokens[start:start + 1] == [toks[0]]
        if gate:
            gate.update(gb, gext, min(acc, len(d)))
            ms += gate.round_ms(gext)
        else:
            ms += ROW[1 + kb] + 3.0 * kb
        out.extend(new)
        rounds += 1
    return out[:n_new], ms, rounds, gr, mg


def truth(n_new):
    s = [5, 9, 11]
    y = f(s[-1], s[-2])
    s.append(y)
    out = [y]
    while len(out) < n_new:
        y = f(s[-1], s[-2])
        s.append(y)
        out.append(y)
    return out


def main():
    ref = truth(400)
    for p_ok in (0.55, 0.9):
        for kfix in (1, 2, 3):
            o, ms, r, gr, mg = generate("fixed", p_ok, kfix=kfix)
            assert o == ref, ("fixed", kfix, p_ok)
        for seed in (1, 2, 3):
            g = DepthGate(ROW, 3.0, 0.3, 2, 1, probe=4)
            o, ms, r, gr, mg = generate("gate", p_ok, seed=seed, gate=g)
            assert o == ref, ("gate", p_ok, seed)
            assert mg.tail_calls == g.n_ext, (mg.tail_calls, g.n_ext)
            assert all(isinstance(x, int) for x in mg.tail_pos)
        # adversarial: a gate that flips a coin every round (any decision sequence must keep the stream exact)
        class Coin(DepthGate):
            def decide(self, confs):
                self.n_round += 1
                return random.random() < 0.5, 0
            def update(self, b, extended, acc):
                self.n_ext += extended
        random.seed(7)
        g = Coin(ROW, 3.0, 0.3, 2, 1)
        o, ms, r, gr, mg = generate("gate", p_ok, gate=g)
        assert o == ref and mg.tail_calls == g.n_ext
    # the gate learns: low-acceptance source extends rarely, high-acceptance source often; gate tok/ms >= the worse fixed depth
    res = {}
    for name, p in (("chat-like", 0.62), ("code-like", 0.93)):
        g = DepthGate(ROW, 3.0, 0.3, 2, 1)
        o, ms, r, gr, mg = generate("gate", p, n_new=3000, gate=g)
        rates = {k: (lambda r: len(r[0]) / r[1])(generate("fixed", p, n_new=3000, kfix=k)) for k in (2, 3)}
        res[name] = (mg.tail_calls / r, len(o) / ms, rates)
        print(f"{name}: ext rate {mg.tail_calls / r:.2f} gate {1000 * len(o) / ms:.2f} tok/s fixed k2 {1000 * rates[2]:.2f} k3 {1000 * rates[3]:.2f}")
        assert len(o) / ms >= 0.97 * min(rates.values())
    assert res["chat-like"][0] < res["code-like"][0] - 0.2, res
    print("test_depth_gate OK")


if __name__ == "__main__":
    main()
