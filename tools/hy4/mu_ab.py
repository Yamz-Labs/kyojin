"""mu_ab.py: ONE load, graph-replay A/B of the batched step at T = 1 / 2 / 4 / 8 rows (rows of different sessions), flags toggled at run time between arms
(captured graphs are dropped on every toggle): stock vs HY4_GM_MOE + HY4_ROW_TILES (cherry-picked from perf/hy4-strix-spec 961625969). ABAB order, median of 3 x 12 replays.
Also eager synchronised class timers (routed experts class) and the arg-max agreement of the two arms."""
import json
import os
import statistics
import sys
import time

tree = os.getcwd()
sys.path.insert(0, os.path.join(tree, "tools", "hy4"))
import serve  # noqa: E402

CTX = 4096
serve.apply_env(CTX)
import torch  # noqa: E402
import serve_core as C  # noqa: E402
from mu_engine import MuEngine  # noqa: E402

GGUF = "/path/to/models/hy4-flash-gguf/Hy4-flash-Q3_K-00001-of-00002.gguf"
PROMPTS = ["Explain in a few short paragraphs why the sky is blue and why sunsets are red.",
           "Ecris une courte histoire sur un chat qui apprend a naviguer sur un petit bateau.",
           "Write a Python function that merges two sorted lists, then explain its complexity.",
           "List five practical tips to keep a sourdough starter healthy, with one sentence of reason each."]
e = MuEngine(tree, GGUF, ctx=CTX, k=3, slots=4)
e.load(print)
V = C.Vocab(e.meta)
GR, fw, hf = e.GR, e.M, e.hf
import importlib  # noqa: E402
BG = importlib.import_module("exllamav3.loader.mla_bgemv")
HB = importlib.import_module("exllamav3.loader.head_bf16")
print("flags available: GM_MOE", hasattr(hf, "GM_MOE"), "ROW_TILES bgemv", hasattr(BG, "ROW_TILES"), "head", hasattr(HB, "ROW_TILES"), flush=True)
sess = []
e.phase("prefill")
for i in range(4):
    s = GR.new_session(i)
    ids = V.encode(V.template.render([{"role": "user", "content": PROMPTS[i]}], None, True, reasoning_effort="no_think"))
    e.prefill_begin(s, ids, 1024)
    while not e.prefill_step(s):
        pass
    sess.append(s)
e.phase("decode")


def set_arm(on):
    hf.GM_MOE = on
    BG.ROW_TILES = on
    HB.ROW_TILES = on
    GR.clear()
    fw.stats["gm_moe_calls"] = 0


def rows_for(T):
    if T <= 4:
        return [(sess[i], [sess[i].y]) for i in range(T)]
    return [(sess[i], [sess[i].y, sess[i].y]) for i in range(4)]


def replay_ms(T, reps=12):
    rows = rows_for(T)
    n0 = [s.n for s, _ in rows]
    for _ in range(3):                                    # capture + warm
        GR.step_slots(rows)
        for (s, _), n in zip(rows, n0):
            GR.rewind(s, n)
    torch.cuda.synchronize()
    ms = []
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(reps):
            _lg, am, _x, _h = GR.step_slots(rows)
            for (s, _), n in zip(rows, n0):
                GR.rewind(s, n)
        torch.cuda.synchronize()
        ms.append((time.perf_counter() - t0) / reps * 1000)
    _lg, am, _x, _h = GR.step_slots(rows)
    toks = am.reshape(-1).tolist()
    for (s, _), n in zip(rows, n0):
        GR.rewind(s, n)
    return statistics.median(ms), toks


def eager_ms(T):
    rows = rows_for(T)
    gs = GR.gs
    tk, ps, bs = [], [], []
    for s, toks in rows:
        s.tokens.extend(toks)
        tk += toks
        ps += list(range(s.n, s.n + len(toks)))
        bs += [s.slot * GR.cap] * len(toks)
    Tn = len(tk)
    gs.T, gs.Nb = Tn, 512
    GR.hin[0, :Tn] = torch.tensor(tk)
    GR.hin[1, :Tn] = torch.tensor(ps)
    GR.hin[2, :Tn] = torch.tensor(bs)
    gs.inp.copy_(GR.hin)
    fw._gs = gs
    try:
        for _ in range(2):
            fw.tm.on = True
            fw.tm.reset()
            GR.body()
            torch.cuda.synchronize()
            fw.tm.on = False
    finally:
        fw._gs = None
        for s, toks in rows:
            del s.tokens[s.n:]
    return fw.tm.acc.get("routed experts", 0) * 1000, sum(fw.tm.acc.values()) * 1000


res, toks_by = {}, {}
for arm in ("stock", "gm+tiles", "stock", "gm+tiles"):
    set_arm(arm != "stock")
    for T in (1, 2, 4, 8):
        ms, toks = replay_ms(T)
        ex, tot = eager_ms(T)
        res.setdefault(arm, {}).setdefault(T, []).append((round(ms, 2), round(ex, 1), round(tot, 1)))
        toks_by.setdefault((arm, T), toks)
        print("AB", arm, "T", T, "graph ms/step", round(ms, 2), "eager experts ms", round(ex, 1), "eager total", round(tot, 1), "gm_moe_calls", fw.stats.get("gm_moe_calls"), flush=True)
agree = {T: toks_by[("stock", T)] == toks_by[("gm+tiles", T)] for T in (1, 2, 4, 8)}
print("AB argmax agree stock vs gm+tiles", agree, flush=True)
print("AB_DONE", json.dumps({a: {str(T): v for T, v in d.items()} for a, d in res.items()}), flush=True)
