#!/usr/bin/env python
"""Tiered expert store gate + measurement driver (one process = one load = one arm).

  gate:     python tests/tier_gate.py -m <pack> --out run.json [--mtp] [--prompts 3] [-n 48]
            records greedy tokens and a sha256 of EVERY Model.forward output (target and MTP draft) in call order.
  compare:  python tests/tier_gate.py --compare ref.json run.json   (exit 1 unless tokens AND all hashes are equal)
  measure:  python tests/tier_gate.py -m <pack> --out run.json --measure --ctx 8192 -n 64 [--mtp]
            prefill an ~ctx-token prompt, then time greedy decode; reports tok/s plus store stats per phase.

The store is switched on by the environment (EXL3_EXPERT_BUDGET_GIB), exactly as in production; unset = resident.
"""
import argparse, hashlib, json, os, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default=os.path.expanduser("~/models/Qwen3.8-Flash-Next-EXL3"))
ap.add_argument("--out")
ap.add_argument("--compare", nargs = 2)
ap.add_argument("--mtp", action = "store_true")
ap.add_argument("--prompts", type = int, default = 3)
ap.add_argument("-n", "--num_tokens", type = int, default = 48)
ap.add_argument("-c", "--cache", type = int, default = 4096)
ap.add_argument("-ndt", "--num_draft_tokens", type = int, default = 4)
ap.add_argument("--measure", action = "store_true")
ap.add_argument("--ctx", type = int, default = 8192)
ap.add_argument("--no-hash", action = "store_true")
args = ap.parse_args()

PROMPTS = [
    "The capital of France is",
    "Explain gradient descent in two sentences:",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "Write a haiku about the sea.\n",
    "List three prime numbers greater than 100:",
]


def compare(a, b):
    ra, rb = json.load(open(a)), json.load(open(b))
    ok = True
    for i, (pa, pb) in enumerate(zip(ra["prompts"], rb["prompts"])):
        t_ok = pa["tokens"] == pb["tokens"]
        h_ok = pa["hashes"] == pb["hashes"]
        print(f"prompt {i}: tokens {'==' if t_ok else '!='} ({len(pa['tokens'])}/{len(pb['tokens'])}), "
              f"forward hashes {'==' if h_ok else '!='} ({len(pa['hashes'])}/{len(pb['hashes'])})")
        if not h_ok:
            for j, (x, y) in enumerate(zip(pa["hashes"], pb["hashes"])):
                if x != y:
                    print(f"   first differing forward #{j}: {x[:2]} vs {y[:2]}")
                    break
        ok = ok and t_ok and h_ok and len(ra["prompts"]) == len(rb["prompts"])
    print("BITWISE IDENTICAL" if ok else "MISMATCH")
    return ok


if args.compare:
    sys.exit(0 if compare(*args.compare) else 1)

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler

HASHES = []
if not args.no_hash:
    def _wrap(name):
        orig = getattr(Model, name)

        def f(self, input_ids, params = None):
            y = orig(self, input_ids, params)
            h = hashlib.sha256()
            def feed(t):
                if isinstance(t, torch.Tensor):
                    h.update(str((t.dtype, tuple(t.shape))).encode())
                    h.update(t.contiguous().view(torch.uint8).cpu().numpy().tobytes())
                elif isinstance(t, (list, tuple)):
                    for x in t: feed(x)
                elif isinstance(t, dict):
                    for k in sorted(t): feed(t[k])
            feed(y)
            HASHES.append((f"{name}:{getattr(self, 'component', 'text')}:{tuple(input_ids.shape)}", h.hexdigest()))
            return y
        setattr(Model, name, f)
    _wrap("forward")
    _wrap("prefill")


def main():
    t0 = time.time()
    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    tokenizer = Tokenizer.from_config(config)
    NDT = args.num_draft_tokens
    cache = Cache(model, max_num_tokens = args.cache, max_history = NDT if args.mtp else 0)
    model.load(progressbar = False)
    draft_model = draft_cache = None
    if args.mtp:
        draft_model = Model.from_config(config, component = "mtp")
        draft_cache = Cache(draft_model, max_num_tokens = args.cache, max_history = NDT)
        draft_model.load(progressbar = False)
    load_s = time.time() - t0
    store = getattr(config, "_expert_store", None)
    print(f"loaded in {load_s:.1f}s  store={'ON' if store else 'off'}  mem {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush = True)
    gen = Generator(model = model, cache = cache, tokenizer = tokenizer, draft_model = draft_model,
                    draft_cache = draft_cache, **({"num_draft_tokens": NDT} if args.mtp else {}))

    def run(ids, n):
        HASHES.clear()
        job = Job(input_ids = ids, max_new_tokens = n, sampler = GreedySampler())
        gen.enqueue(job)
        toks, t_first, t_start = [], None, time.time()
        acc = rej = 0
        while gen.num_remaining_jobs():
            for r in gen.iterate():
                if r.get("error") is not None:
                    raise RuntimeError(r["error"])
                if "token_ids" in r and r["token_ids"] is not None:
                    toks += r["token_ids"].flatten().tolist()
                    if t_first is None and toks:
                        t_first = time.time()
                if "accepted_draft_tokens" in r:
                    acc += r["accepted_draft_tokens"]; rej += r["rejected_draft_tokens"]
        t_end = time.time()
        return toks, t_start, t_first, t_end, acc, rej

    out = {"mtp": args.mtp, "env": {k: v for k, v in os.environ.items() if k.startswith("EXL3_EXPERT")},
           "load_s": load_s, "prompts": []}
    if not args.measure:
        for p in PROMPTS[:args.prompts]:
            ids = tokenizer.encode(p, add_bos = True)
            toks, *_ = run(ids, args.num_tokens)
            out["prompts"].append({"prompt": p, "tokens": toks, "hashes": list(HASHES)})
            print(f"{p[:30]!r}: {len(toks)} tokens, {len(HASHES)} forwards, hash0 {HASHES[0][1][:12] if HASHES else '-'}", flush = True)
            print("   ", tokenizer.decode(torch.tensor([toks]))[0][:100].replace("\n", " "), flush = True)
    else:
        base = tokenizer.encode("The quick brown fox jumps over the lazy dog. " * 4, add_bos = True)
        ids = base
        while ids.shape[-1] < args.ctx:
            ids = torch.cat([ids, tokenizer.encode(f" Section {ids.shape[-1]}: " + PROMPTS[ids.shape[-1] % len(PROMPTS)], add_bos = False)], dim = -1)
        ids = ids[:, :args.ctx]
        if store: store.reset_stats()
        # prefill cost: a 1-token generation over the long prompt
        toks, t_s, t_f, t_e, _, _ = run(ids, 1)
        pf_s = t_e - t_s
        st_pf = store.stats() if store else None
        if store: store.reset_stats()
        # decode: prompt is now cached (prefix reuse) so this mostly times decode
        ids2 = torch.cat([ids, tokenizer.encode(" Continue:", add_bos = False)], dim = -1)
        toks, t_s, t_f, t_e, acc, rej = run(ids2, args.num_tokens)
        st_dec = store.stats() if store else None
        dec_tps = (len(toks) - 1) / (t_e - t_f) if t_f and len(toks) > 1 and t_e > t_f else float("nan")
        out.update(ctx = args.ctx, prefill_s = pf_s, prefill_stats = st_pf, decode_tokens = len(toks),
                   decode_tps = dec_tps, decode_ttft_s = (t_f - t_s) if t_f else None, decode_stats = st_dec,
                   accepted = acc, rejected = rej)
        print(f"ctx {args.ctx}: prefill {pf_s:.1f}s  decode {dec_tps:.2f} tok/s ({len(toks)} tok, acc {acc} rej {rej})", flush = True)
        print("prefill stats", st_pf, flush = True)
        print("decode  stats", st_dec, flush = True)
    out["mem_gib"] = torch.cuda.memory_allocated() / 2**30
    if store:
        out["store_stats_total"] = store.stats()
    if args.out:
        json.dump(out, open(args.out, "w"))
    if store and os.environ.get("EXL3_EXPERT_STATS_OUT"):
        store.dump_counts(os.environ["EXL3_EXPERT_STATS_OUT"])


if __name__ == "__main__":
    import multiprocessing
    try:
        multiprocessing.set_start_method("spawn", force = False)
    except RuntimeError:
        pass
    main()
