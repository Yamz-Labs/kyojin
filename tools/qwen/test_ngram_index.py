# proof: NgramIndex / SeqFollower == the old ngram_match, CPU only.
#   python tools/qwen/test_ngram_index.py [--fuzz 10000] [--docs DIR]   (HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= nice -n 19)
import os, sys, json, random, time, argparse
os.environ.setdefault("HIP_VISIBLE_DEVICES", ""); os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np, torch
torch.set_num_threads(1)
import spec_policy as sp
from exllamav3.util.tensor import SeqTensor

BR = {}
def hit(k): BR[k] = BR.get(k, 0) + 1

def check(idx, ctx, nmin, K, where):
    ref = sp.ngram_match(ctx, nmin, K); got = idx.match(K)
    assert ref == got, f"MISMATCH {where} n={len(ctx)} nmin={nmin} K={K}: old {ref} new {got}"
    L, c = ref
    if L == 0: hit("none")
    elif L >= 16: hit("L>=16"); hit("L=32") if L == 32 else None
    else: hit("L5-15" if L >= 5 else "L<5")
    if L and len(c) < K: hit("short_cont")
    if L and len(ctx) - len(c) - 0 >= 0:
        pass
    return ref

def same_state(a, b, where):
    assert a.toks == b.toks, where
    for m in range(1, a.lmax + 1):
        assert a.key[m] == b.key[m], f"{where}: key level {m}"
        if m >= a.nmin:
            assert a.prev[m] == b.prev[m], f"{where}: prev level {m}"
            assert a.latest[m] == b.latest[m], f"{where}: latest level {m}"

def gen_stream(rng, n):
    kind = rng.choice(["small", "blocks", "mixed", "zipf", "period"])
    if kind == "small":
        a = rng.randint(2, 6); return [rng.randrange(a) for _ in range(n)]
    if kind == "period":
        p = rng.randint(1, 20); base = [rng.randrange(1000) for _ in range(p)]
        s = [base[i % p] for i in range(n)]
        for _ in range(rng.randint(0, 3)):
            if n: s[rng.randrange(n)] = rng.randrange(1000)
        return s
    if kind == "blocks":
        blocks = [[rng.randrange(50) for _ in range(rng.randint(3, 40))] for _ in range(rng.randint(2, 8))]
        s = []
        while len(s) < n:
            b = rng.choice(blocks); s += b if rng.random() < 0.8 else b[:rng.randint(1, len(b))] + [rng.randrange(50)]
        return s[:n]
    if kind == "zipf":
        w = [1 / (i + 1) for i in range(300)]; return rng.choices(range(300), w, k=n)
    s = []
    while len(s) < n:
        if s and rng.random() < 0.5:
            i = rng.randrange(len(s)); l = rng.randint(2, 40); s += s[i:i + l]
        else: s += [rng.randrange(8) for _ in range(rng.randint(1, 10))]
    return s[:n]

def fuzz(cases, seed=1):
    rng = random.Random(seed); t0 = time.time()
    for c in range(cases):
        nmin = rng.choice([1, 2, 2, 2, 3, 4]); K = rng.choice([1, 2, 3, 3, 3, 5])
        n = rng.choice([0, 1, 2, 3, 5, 10, 40, 100, 300, 1000])
        stream = gen_stream(rng, n + 400)
        idx = sp.NgramIndex(nmin); ctx = []; src = 0
        sid = SeqTensor((1, 1), torch.long, -1, init_cap=64)
        fol = sp.SeqFollower(nmin)
        def sid_to(lst):
            sid.clear(); sid.append(torch.tensor([lst], dtype=torch.long)) if lst else None
        for step in range(rng.randint(5, 40)):
            op = rng.random()
            if op < 0.55 or not ctx:     # accept 1-4 tokens
                k = rng.randint(1, 4); new = stream[src:src + k] or [rng.randrange(5)]
                src += len(new); ctx += new
                sid.append(torch.tensor([new], dtype=torch.long)); hit("append")
            elif op < 0.75:              # rollback 1-4
                k = min(rng.randint(1, 4), len(ctx)); ctx = ctx[:len(ctx) - k]; sid.truncate(len(ctx)); hit("trunc")
            elif op < 0.85:              # deep rollback then different continuation
                k = rng.randint(0, len(ctx)); ctx = ctx[:k]; sid.truncate(k)
                new = [rng.randrange(8) for _ in range(rng.randint(0, 30))]; ctx += new
                if new: sid.append(torch.tensor([new], dtype=torch.long))
                hit("deep")
            elif op < 0.92:              # truncate then append SAME LENGTH different tokens (low-water mark case)
                k = min(rng.choice([rng.randint(1, 6), rng.randint(17, 80)]), len(ctx)); ctx = ctx[:len(ctx) - k]; sid.truncate(len(ctx))
                new = [rng.randrange(8) for _ in range(k)]; ctx += new; sid.append(torch.tensor([new], dtype=torch.long)); hit("same_len_swap")
            elif op < 0.94 and len(ctx) > 40:   # middle swap: tokens differ below the last 16 (the tail belt cannot see it)
                k = rng.randint(17, min(60, len(ctx))); keep = ctx[len(ctx) - 16:]; head = ctx[:len(ctx) - k]
                ctx = head + [rng.randrange(8) for _ in range(k - 16)] + keep
                sid.truncate(len(head)); sid.append(torch.tensor([ctx[len(head):]], dtype=torch.long)); hit("middle_swap")
            elif op < 0.96:              # set (clear + append)
                lst = ctx[:rng.randint(0, len(ctx))] + [rng.randrange(8) for _ in range(rng.randint(0, 50))]
                ctx = lst; sid.set(torch.tensor([lst], dtype=torch.long)) if lst else sid.clear(); hit("set")
            else:                        # big append (bulk path)
                new = stream[src:src + rng.randint(48, 300)]; src += len(new); ctx += new
                if new: sid.append(torch.tensor([new], dtype=torch.long)); hit("bulk")
            got = fol.sync(sid)
            assert len(got) == len(ctx) and list(got.toks) == ctx, f"case {c}: follower mirror differs"
            check(got, ctx, nmin, K, f"case {c} step {step}")
            if rng.random() < 0.05:      # new SeqTensor sharing a prefix (next request)
                k = rng.randint(0, len(ctx)); ctx = ctx[:k] + [rng.randrange(8) for _ in range(rng.randint(0, 20))]
                sid = SeqTensor((1, 1), torch.long, -1, init_cap=64)
                if ctx: sid.append(torch.tensor([ctx], dtype=torch.long))
                hit("new_seq")
        # state equality: bulk/follower index == pure scalar pushes of the same tokens
        if c % 25 == 0:
            fol.sync(sid)
            ref = sp.NgramIndex(nmin)
            for t in ctx: ref.push(t)
            same_state(fol.idx, ref, f"case {c} state")
            hit("state_checked")
    print(f"fuzz {cases} cases OK in {time.time() - t0:.1f}s")

def streams_from_docs(dd):
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(os.path.expanduser("~/models/qwen38-yamz-v2-bal/tokenizer.json"))
    out = {}
    for L in (4096, 65536, 196608):
        d = json.load(open(f"{dd}/doc{L}.json")); out[L] = tk.encode(d["doc"], add_special_tokens=False).ids
    return out

def recorded(dd, rounds=300, seed=3):
    """Rounds of 1-4 accepted tokens (the shipped spec step) over real token streams, with 5% rollbacks of 1-4
    tokens (rejected / checkpoint restore) and a second 'turn' reusing the prefix through a new SeqTensor."""
    rng = random.Random(seed)
    for L, ids in streams_from_docs(dd).items():
        t0 = time.time(); nmin = 2; K = 3
        base = len(ids) - 2000                    # prompt = ids[:base]; rounds continue through the document tail
        sid = SeqTensor((1, 1), torch.long, -1, init_cap=base + 4096)
        sid.append(torch.tensor([ids[:base]], dtype=torch.long))
        fol = sp.SeqFollower(nmin); ctx = ids[:base]; pos = base
        tb = time.time(); fol.sync(sid); cold = time.time() - tb
        n_hits = 0; times = []
        for r in range(rounds):
            if r % 20 == 19:
                k = rng.randint(1, 4); ctx = ctx[:-k]; sid.truncate(len(ctx)); pos -= k
            k = rng.randint(1, 4); new = ids[pos:pos + k]; pos += len(new); ctx = ctx + new
            sid.append(torch.tensor([new], dtype=torch.long))
            t1 = time.perf_counter(); got = fol.sync(sid); res = got.match(K); times.append(time.perf_counter() - t1)
            ref = sp.ngram_match(ctx, nmin, K); assert ref == res, f"recorded L={L} round {r}: old {ref} new {res}"
            n_hits += res[0] >= 5
        # second turn: new tensor = the whole context + new tokens (multi-turn reuse)
        ctx2 = ctx + [int(x) for x in np.random.RandomState(5).randint(0, 1000, 300)]
        sid2 = SeqTensor((1, 1), torch.long, -1, init_cap=len(ctx2) + 64); sid2.append(torch.tensor([ctx2], dtype=torch.long))
        tb = time.time(); got = fol.sync(sid2); turn2 = time.time() - tb
        assert sp.ngram_match(ctx2, nmin, K) == got.match(K)
        ref = sp.NgramIndex(nmin); ref.extend(np.array(ctx2, dtype=np.int64)); same_state(got, ref, "turn2 vs bulk rebuild")
        ref2 = sp.NgramIndex(nmin)
        for t in ctx2[:3000]: ref2.push(t)
        b = sp.NgramIndex(nmin); b.extend(np.array(ctx2[:3000], dtype=np.int64)); same_state(b, ref2, "bulk vs push (3000)")
        t_old = []
        for _ in range(20): t0o = time.perf_counter(); sp.ngram_match(sid.torch_slice(0, len(ctx)).flatten().tolist(), nmin, K); t_old.append(time.perf_counter() - t0o)
        print(f"recorded {L:>6}: ctx {len(ctx)} {rounds} rounds equal ({n_hits} with L>=5); cold build {cold:.2f}s, turn2 resync {turn2*1e3:.1f} ms, "
              f"new sync+match median {1e6*sorted(times)[len(times)//2]:.0f} us (max {1e6*max(times):.0f}), old tolist+scan median {1e3*sorted(t_old)[10]:.2f} ms; fallbacks {fol.idx.fallbacks}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fuzz", type=int, default=10000); ap.add_argument("--docs", default=os.path.expanduser("$WORK_DIR"))
    ap.add_argument("--skip-recorded", action="store_true"); a = ap.parse_args()
    fuzz(a.fuzz)
    print("branches:", json.dumps(BR, sort_keys=True))
    if not a.skip_recorded: recorded(a.docs)
    print("ALL OK")
