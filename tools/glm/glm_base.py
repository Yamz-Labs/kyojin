#!/usr/bin/env python
"""GLM-5.3-Flash EXL3 baseline on Strix Halo.

Modes (one model load each; run on an otherwise idle GPU):
  fast  smoke (3 greedy prompts, saves token ids + logits), ext/triton call census, decode@2K x3,
        prefill@4K/16K x3, wikitext-2 PPL on a fixed slice. JSON rewritten after every stage.
  ref   same smoke prompts teacher-forced through whatever path the env knobs select (the generic
        path when run with the fast kernels disabled), logits compared against the fast run;
        plus its own greedy output and the first PPL rows.
  prof  short decode + one 4K prefill, meant to run under rocprofv3 --kernel-trace --stats.
"""
import argparse, json, math, os, statistics, sys, time
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["fast", "ref", "prof"])
ap.add_argument("-m", "--model", default="~/models/glm53-exl3-td205")
ap.add_argument("-o", "--out", required=True)
ap.add_argument("--smoke-tokens", type=int, default=96)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--decode-depth", type=int, default=2048)
ap.add_argument("--decode-tokens", type=int, default=128)
ap.add_argument("--prefill-depths", default="4096,16384")
ap.add_argument("--ppl-rows", type=int, default=20)
ap.add_argument("--ppl-len", type=int, default=2048)
ap.add_argument("--ref-ppl-rows", type=int, default=4)
ap.add_argument("--corpus", default="~/bench/ppl/wiki.test.raw")
ap.add_argument("--fast-json", default=None, help="ref mode: the fast run's JSON (for its smoke tokens/logits)")
ap.add_argument("--gtt-max-gb", type=float, default=105.0)
ap.add_argument("--prefill-only", action="store_true", help="fast mode: warmup + prefill only (no census/decode/PPL)")
ap.add_argument("--ab-env", default=None, help="prefill A/B in one process, reps interleaved: VAR=a,b (kernel reads VAR per launch)")
ap.add_argument("--ab-sets", default=None, help="prefill-only: composite variants, reps interleaved, one untimed warmup each: 'name:VAR=v,VAR2=v;name2:...'. PREFILL_CHUNK=n sets gen.max_chunk_size")
ap.add_argument("--prof-depths", default=None, help="ab-sets mode, after the A/B: one census-free prefill window per depth (launch env), for rocprofv3 slicing")
ap.add_argument("--ab-ppl", type=int, nargs="?", const=-1, default=0, help="ab-sets mode: PPL on this many rows of --ppl-len under each set (one load); bare flag = --ppl-rows")
ap.add_argument("--depth-check", default=None, help="ab-sets mode: 'LEN:setA,setB' same prompt under two sets (page table + recurrent cache reset between), 32 greedy tokens, ids/logits compared")
args = ap.parse_args()

PROMPTS = [
    "What is the capital of France? Answer in one sentence, then name two famous landmarks there.",
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
    "Explain in three short sentences why the sky is blue.",
]

RES = {"mode": args.mode, "model": args.model, "env": {k: v for k, v in os.environ.items() if k.startswith("EXL3_")}}


def save():
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(RES, f, indent=1)
    os.replace(tmp, args.out)


def gtt_gb():
    best = 0
    import glob
    for p in glob.glob("/sys/class/drm/card*/device/mem_info_gtt_used"):
        try:
            best = max(best, int(open(p).read()))
        except Exception:
            pass
    return best / 1e9


def gtt_guard(tag):
    g = gtt_gb()
    RES.setdefault("gtt_gb", {})[tag] = round(g, 2)
    print(f"[gtt] {tag}: {g:.1f} GB", flush=True)
    if g > args.gtt_max_gb:
        RES["abort"] = f"GTT {g:.1f} GB > {args.gtt_max_gb} at {tag}"
        save()
        sys.exit(3)


# ---------------------------------------------------------------- call census
class Census:
    """Counts exllamav3_ext entry points and Triton kernel launches while active."""

    def __init__(self):
        from exllamav3.ext import exllamav3_ext as ext
        self.ext = ext
        self.orig = {}
        self.counts = {}
        self.triton_orig = None

    def start(self):
        self.counts = {}
        for name in dir(self.ext):
            if name.startswith("_"):
                continue
            fn = getattr(self.ext, name)
            if not callable(fn) or isinstance(fn, type):
                continue
            self.orig[name] = fn

            def wrap(*a, n_=name, f_=fn, **k):
                self.counts[n_] = self.counts.get(n_, 0) + 1
                return f_(*a, **k)
            setattr(self.ext, name, wrap)
        try:
            from triton.runtime.jit import JITFunction
            self.triton_orig = JITFunction.run
            orig = self.triton_orig
            counts = self.counts

            def run(jf, *a, **k):
                n = "triton:" + str(getattr(jf, "__name__", None) or getattr(jf, "fn", "?"))
                counts[n] = counts.get(n, 0) + 1
                return orig(jf, *a, **k)
            JITFunction.run = run
        except Exception as e:
            print("[census] triton hook failed:", e, flush=True)

    def stop(self):
        for n, f in self.orig.items():
            setattr(self.ext, n, f)
        self.orig = {}
        if self.triton_orig is not None:
            from triton.runtime.jit import JITFunction
            JITFunction.run = self.triton_orig
            self.triton_orig = None
        return dict(sorted(self.counts.items(), key=lambda kv: -kv[1]))


# ---------------------------------------------------------------- helpers
def chat(tok, prompt):
    p = f"[gMASK]<sop><|user|>\n{prompt}<|assistant|>\n"
    return tok.encode(p, encode_special_tokens=True)


def run_job(gen, ids, n, stop=True, return_logits=False, constrain=None):
    from exllamav3 import Job
    from exllamav3.generator.sampler import GreedySampler
    job = Job(input_ids=ids, max_new_tokens=n, sampler=GreedySampler(),
              stop_conditions=(gen.model.config.eos_token_id_list if stop else []),
              return_logits=return_logits)
    gen.enqueue(job)
    if constrain is not None:
        job.constrain_output_now(constrain.view(1, -1).contiguous())
    t0 = time.perf_counter()
    ttft = None
    toks, logits, text, new_tokens = [], [], "", 0
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("error"):
                raise RuntimeError(f"job error: {r['error']}")
            tid = r.get("token_ids")
            if tid is not None and tid.numel():
                if ttft is None:
                    torch.cuda.synchronize()
                    ttft = time.perf_counter() - t0
                toks.append(tid.cpu())
            if r.get("logits") is not None:
                logits.append(r["logits"].float().cpu())
            text += r.get("text", "")
            if r.get("eos"):
                new_tokens = r.get("new_tokens", new_tokens)
    torch.cuda.synchronize()
    total = time.perf_counter() - t0
    t = torch.cat(toks, dim=-1).flatten() if toks else torch.zeros(0, dtype=torch.long)
    L = torch.cat(logits, dim=1)[0] if logits else None
    return {"ids": t, "text": text, "ttft": ttft, "total": total, "ntok": int(t.numel()), "logits": L}


def corpus_ids(tok, need):
    text = open(args.corpus, encoding="utf-8").read()
    ids = tok.encode(text, add_bos=False)
    reps = 1
    while ids.shape[-1] < need:
        reps += 1
        ids = torch.cat([ids, ids], dim=-1)
    return ids


def ppl_rows(tok, rows, L):
    ids = tok.encode(open(args.corpus, encoding="utf-8").read(), add_bos=False)
    n = ids.shape[-1] // L
    assert n >= rows, (n, rows)
    return ids[0, : rows * L].view(rows, L), int(ids.shape[-1])


@torch.inference_mode()
def eval_ppl(model, tok, rows, L, tag):
    data, ntok = ppl_rows(tok, rows, L)
    nll, cnt, per_row = 0.0, 0, []
    t0 = time.perf_counter()
    for r in range(rows):
        x = data[r : r + 1]
        logits = model.forward(x, {"attn_mode": "flash_attn_nc"})
        lp = torch.log_softmax(logits[0, :-1, : tok.actual_vocab_size].float(), dim=-1)
        tgt = x[0, 1:].to(lp.device)
        row_nll = -lp.gather(-1, tgt.unsqueeze(-1)).sum().item()
        nll += row_nll
        cnt += tgt.numel()
        per_row.append(math.exp(row_nll / tgt.numel()))
        del logits, lp
        print(f"[ppl {tag}] row {r+1}/{rows} running {math.exp(nll/cnt):.4f}", flush=True)
    return {"rows": rows, "len": L, "scored": cnt, "corpus_tokens": ntok, "ppl": math.exp(nll / cnt),
            "per_row_ppl": [round(p, 4) for p in per_row], "seconds": time.perf_counter() - t0,
            "protocol": f"wiki.test.raw tokenized once add_bos=False; first {rows} non-overlapping rows of {L}; "
                        f"score positions 1..{L-1} of every row; cacheless forward attn_mode=flash_attn_nc"}


def main():
    torch.set_grad_enabled(False)
    from exllamav3 import Config, Model, Cache, Tokenizer, Generator
    import exllamav3_ext as X

    t0 = time.time()
    config = Config.from_directory(args.model)
    model = Model.from_config(config)  # text component only: no MTP, no vision
    tok = Tokenizer.from_config(config)
    max_depth = max([args.decode_depth + args.decode_tokens] + [int(d) + 8 for d in args.prefill_depths.split(",")])
    cache_tokens = ((max_depth + 1023) // 1024 + 1) * 1024
    cache = Cache(model, max_num_tokens=cache_tokens)
    model.load(device="cuda:0", progressbar=False)
    RES["load_s"] = time.time() - t0
    RES["cache_tokens"] = cache_tokens
    try:
        RES["storage_info"] = [float(x) if isinstance(x, (int, float)) else str(x) for x in model.get_storage_info()]
    except Exception as e:
        RES["storage_info"] = repr(e)
    RES["torch_mem_alloc_gb"] = torch.cuda.memory_allocated() / 1e9
    print(f"loaded in {RES['load_s']:.1f}s, torch alloc {RES['torch_mem_alloc_gb']:.1f} GB", flush=True)
    gtt_guard("after_load")
    save()

    gen = Generator(model=model, cache=cache, tokenizer=tok)
    census = Census()

    # ---------------- smoke
    smoke = []
    fast = json.load(open(args.fast_json)) if args.fast_json else None
    for i, p in enumerate(PROMPTS):
        ids = chat(tok, p)
        census.start() if i == 0 else None
        r = run_job(gen, ids, args.smoke_tokens, stop=False, return_logits=True)
        if i == 0:
            RES["census_smoke0"] = census.stop()
        ent = {"prompt": p, "text": r["text"], "ids": r["ids"].tolist(), "ttft": r["ttft"], "ntok": r["ntok"]}
        torch.save(r["logits"], args.out + f".smoke{i}.logits.pt")
        if fast is not None:
            ref_ids = torch.tensor(fast["smoke"][i]["ids"])
            ent["greedy_match_prefix"] = int(next((k for k, (a, b) in enumerate(zip(ref_ids.tolist(), r["ids"].tolist())) if a != b), min(len(ref_ids), r["ntok"])))
            # teacher-force the fast run's sequence through this path
            rf = run_job(gen, ids, ref_ids.numel(), stop=False, return_logits=True, constrain=ref_ids)
            la = torch.load(args.fast_json + f".smoke{i}.logits.pt")
            lb = rf["logits"]
            n = min(la.shape[0], lb.shape[0])
            V = min(la.shape[-1], lb.shape[-1], tok.actual_vocab_size)
            la, lb = la[:n, :V], lb[:n, :V]
            d = (la - lb).abs()
            ta, tb = la.argmax(-1), lb.argmax(-1)
            flips = (ta != tb).nonzero().flatten().tolist()
            top2 = la.topk(2, dim=-1).values
            gaps = [(top2[q, 0] - top2[q, 1]).item() for q in flips]
            pa = torch.log_softmax(la, -1); pb = torch.log_softmax(lb, -1)
            kld = (pa.exp() * (pa - pb)).sum(-1)
            ent["tf_compare"] = {"positions": n, "max_dlogit": d.max().item(), "mean_dlogit": d.mean().item(),
                                 "argmax_flips": len(flips), "flip_gaps": [round(g, 4) for g in gaps[:20]],
                                 "kld_mean": kld.mean().item(), "kld_max": kld.max().item()}
        smoke.append(ent)
        print(f"[smoke {i}] {r['ntok']} tok: {r['text'][:300]!r}", flush=True)
        if "tf_compare" in ent:
            print(f"[smoke {i}] tf_compare {ent['tf_compare']}", flush=True)
        RES["smoke"] = smoke
        save()
    gtt_guard("after_smoke")

    if args.mode == "ref":
        try:
            RES["ppl_ref_rows"] = eval_ppl(model, tok, args.ref_ppl_rows, args.ppl_len, "ref")
        except Exception as e:
            RES["ppl_ref_rows"] = {"error": repr(e)}
        save()
        return

    need = (args.decode_depth * (args.reps + 2) + sum(int(d) * (args.reps + 1) for d in args.prefill_depths.split(",")) + 50000)
    cids = corpus_ids(tok, need)
    cur = [0]

    def nxt(n):
        s = cids[:, cur[0] : cur[0] + n].contiguous()
        cur[0] += n + 37
        return s

    # ---------------- warmup (Triton JIT, graph capture, arena growth)
    run_job(gen, nxt(args.decode_depth), 16, stop=False)
    run_job(gen, nxt(4096), 1, stop=False)
    RES["warmup"] = "decode@2K x16 + prefill@4K"
    save()

    if args.mode == "prof":
        # decode window in both clocks rocprofv3 may stamp with (the trace is filtered on it);
        # no census wrappers inside the window so kernel times are not skewed
        torch.cuda.synchronize()
        w0 = {"mono": time.monotonic_ns(), "boot": time.clock_gettime_ns(time.CLOCK_BOOTTIME)}
        r = run_job(gen, nxt(args.decode_depth), args.decode_tokens, stop=False)
        torch.cuda.synchronize()
        w1 = {"mono": time.monotonic_ns(), "boot": time.clock_gettime_ns(time.CLOCK_BOOTTIME)}
        RES["prof_decode_window"] = {"start": w0, "end": w1}
        # census-free 4K prefill window for per-kernel prefill profiling
        p0 = {"mono": time.monotonic_ns(), "boot": time.clock_gettime_ns(time.CLOCK_BOOTTIME)}
        rp = run_job(gen, nxt(4096), 1, stop=False)
        torch.cuda.synchronize()
        p1 = {"mono": time.monotonic_ns(), "boot": time.clock_gettime_ns(time.CLOCK_BOOTTIME)}
        RES["prof_prefill_window"] = {"start": p0, "end": p1, "ttft": rp["ttft"]}
        census.start()
        run_job(gen, nxt(args.decode_depth), 16, stop=False)
        RES["census_decode"] = census.stop()
        census.start()
        r2 = run_job(gen, nxt(4096), 1, stop=False)
        RES["census_prefill4k"] = census.stop()
        RES["prof_decode"] = {"ttft": r["ttft"], "total": r["total"], "ntok": r["ntok"]}
        RES["prof_prefill"] = {"ttft": r2["ttft"]}
        save()
        return

    if args.prefill_only and args.ab_sets:
        sets = {}
        for spec in args.ab_sets.split(";"):
            name, kv = spec.split(":", 1)
            sets[name] = dict(x.split("=", 1) for x in kv.split(",") if x)
        chunk0 = gen.max_chunk_size
        chunks = []
        for fn in ("prefill", "forward"):
            orig = getattr(gen.model, fn)
            def wrap(*a, _o=orig, **k):
                ids = k.get("input_ids", a[0] if a else None)
                if ids is not None and ids.shape[-1] > 1: chunks.append(int(ids.shape[-1]))
                return _o(*a, **k)
            setattr(gen.model, fn, wrap)
        # Every variant starts from the launch env: a key set by one variant must not leak
        # into the next (it did: LAST_PAGE=0 from one set silently applied to all later ones)
        env0 = {k: os.environ.get(k) for s_ in sets.values() for k in s_ if k != "PREFILL_CHUNK"}
        def apply(env):
            gen.max_chunk_size = int(env.get("PREFILL_CHUNK", chunk0))
            for k, v in env0.items():
                if v is None: os.environ.pop(k, None)
                else: os.environ[k] = v
            for k, v in env.items():
                if k != "PREFILL_CHUNK": os.environ[k] = v
        names = list(sets)
        RES["prefill"] = {"ab_sets": sets}
        for d in [int(x) for x in args.prefill_depths.split(",")]:
            for n in names:  # untimed warmup per variant (JIT for new shapes/specializations)
                apply(sets[n]); run_job(gen, nxt(d), 1, stop=False)
            runs = {n: [] for n in names}
            for i in range(args.reps):
                for n in (names if i % 2 == 0 else names[::-1]):
                    apply(sets[n]); chunks.clear()
                    torch.cuda.synchronize(); a0 = torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
                    r = run_job(gen, nxt(d), 1, stop=False)
                    pk = (torch.cuda.max_memory_allocated() - a0) / 2**30   # transient peak above the resident set
                    ma = int(open("/proc/meminfo").read().split("MemAvailable:")[1].split()[0]) // 1024
                    runs[n].append({"tok_s": d / r["ttft"], "ttft": r["ttft"], "chunks": list(chunks), "mem_avail_mb": ma, "peak_extra_gb": round(pk, 2)})
                    print(f"[prefill@{d}] {n} rep{i}: {d / r['ttft']:.1f} tok/s (ttft {r['ttft']:.2f}s) chunks {chunks} peak+{pk:.2f} GB MemAvail {ma} MB", flush=True)
                    gtt_guard(f"prefill{d}_{n}_rep{i}")
                    RES["prefill"].setdefault("partial", []).append({"depth": d, "set": n, "rep": i, **runs[n][-1]}); save()
            RES["prefill"][str(d)] = {n: {"runs": rr, "median_tok_s": statistics.median(x["tok_s"] for x in rr),
                                          "median_ttft": statistics.median(x["ttft"] for x in rr)} for n, rr in runs.items()}
            for n, rr in runs.items():
                print(f"[prefill@{d}] {n} median {statistics.median(x['tok_s'] for x in rr):.1f} tok/s", flush=True)
            save()
        if args.depth_check:
            L, pair = args.depth_check.split(":")
            ids = nxt(int(L))
            outs = {}
            for n in pair.split(","):
                gen.pagetable.reset_page_table()
                if gen.recurrent_cache is not None: gen.recurrent_cache.clear(); gen.recurrent_cache.current_size = 0
                apply(sets[n]); chunks.clear()
                r = run_job(gen, ids, 32, stop=False, return_logits=True)
                outs[n] = r; print(f"[depth-check {L}] {n} chunks {chunks} ttft {r['ttft']:.2f}s", flush=True)
            a, b = (outs[n] for n in pair.split(","))
            n_ = min(a["logits"].shape[0], b["logits"].shape[0])
            V_ = min(a["logits"].shape[-1], tok.actual_vocab_size)  # padded vocab columns are -inf (inf - inf = nan)
            la, lb = a["logits"][:n_, :V_], b["logits"][:n_, :V_]
            pa, pb = torch.log_softmax(la, -1), torch.log_softmax(lb, -1)
            RES["depth_check"] = {"len": int(L), "sets": pair, "ids_equal": a["ids"].tolist() == b["ids"].tolist(),
                                  "match_prefix": int(next((k for k, (x, y) in enumerate(zip(a["ids"].tolist(), b["ids"].tolist())) if x != y), n_)),
                                  "logits_bitwise": bool(torch.equal(la, lb)), "max_dlogit": (la - lb).abs().max().item(),
                                  "kld_mean": (pa.exp() * (pa - pb)).sum(-1).mean().item()}
            print(f"[depth-check] {RES['depth_check']}", flush=True); save()
        if args.ab_ppl:
            RES["ab_ppl"] = {}
            for n in names:
                apply(sets[n])
                RES["ab_ppl"][n] = eval_ppl(model, tok, args.ppl_rows if args.ab_ppl < 0 else args.ab_ppl, args.ppl_len, n)
                print(f"[ab-ppl] {n} ppl {RES['ab_ppl'][n]['ppl']:.6f} ({RES['ab_ppl'][n]['seconds']:.0f} s)", flush=True); save()
        if args.prof_depths:
            apply({})
            RES["prof_windows"] = {}
            for d in [int(x) for x in args.prof_depths.split(",")]:
                torch.cuda.synchronize(); chunks.clear()
                p0 = {"mono": time.monotonic_ns(), "boot": time.clock_gettime_ns(time.CLOCK_BOOTTIME)}
                rp = run_job(gen, nxt(d), 1, stop=False)
                torch.cuda.synchronize()
                p1 = {"mono": time.monotonic_ns(), "boot": time.clock_gettime_ns(time.CLOCK_BOOTTIME)}
                RES["prof_windows"][str(d)] = {"start": p0, "end": p1, "ttft": rp["ttft"], "chunks": list(chunks)}
                print(f"[prof@{d}] ttft {rp['ttft']:.2f}s chunks {chunks}", flush=True); save()
        return

    if args.prefill_only:
        ab_var, ab_vals = (args.ab_env.split("=", 1)[0], args.ab_env.split("=", 1)[1].split(",")) if args.ab_env else (None, [None])
        RES["prefill"] = {}
        for d in [int(x) for x in args.prefill_depths.split(",")]:
            runs = {v: [] for v in ab_vals}
            for i in range(args.reps):
                for v in (ab_vals if i % 2 == 0 else ab_vals[::-1]):
                    if ab_var:  # VAR1+VAR2=a+b,c+d sets several knobs per arm
                        for kv, vv in zip(ab_var.split("+"), v.split("+")): os.environ[kv] = vv
                    r = run_job(gen, nxt(d), 1, stop=False)
                    runs[v].append({"tok_s": d / r["ttft"], "ttft": r["ttft"]})
                    print(f"[prefill@{d}] {ab_var}={v} rep{i}: {d / r['ttft']:.1f} tok/s (ttft {r['ttft']:.2f}s)", flush=True)
                    gtt_guard(f"prefill{d}_{v}_rep{i}")
            RES["prefill"][str(d)] = {str(v): {"runs": rr, "median_tok_s": statistics.median(x["tok_s"] for x in rr)} for v, rr in runs.items()}
            for v, rr in runs.items():
                print(f"[prefill@{d}] {ab_var}={v} median {statistics.median(x['tok_s'] for x in rr):.1f} tok/s", flush=True)
            save()
        return

    # ---------------- call census (separate pass; wrappers cost time, so never during timed runs)
    census.start()
    run_job(gen, nxt(args.decode_depth), 16, stop=False)
    RES["census_decode2k"] = census.stop()
    census.start()
    run_job(gen, nxt(4096), 1, stop=False)
    RES["census_prefill4k"] = census.stop()
    save()

    # ---------------- decode @2K
    dec = []
    for i in range(args.reps):
        r = run_job(gen, nxt(args.decode_depth), args.decode_tokens, stop=False)
        tps = (r["ntok"] - 1) / (r["total"] - r["ttft"])
        dec.append({"tok_s": tps, "ntok": r["ntok"], "ttft": r["ttft"], "total": r["total"]})
        print(f"[decode@{args.decode_depth}] rep{i}: {tps:.2f} tok/s ({r['ntok']} tok)", flush=True)
    RES["decode"] = {"depth": args.decode_depth, "runs": dec, "median_tok_s": statistics.median(x["tok_s"] for x in dec)}
    save()

    # ---------------- prefill
    RES["prefill"] = {}
    for d in [int(x) for x in args.prefill_depths.split(",")]:
        runs = []
        for i in range(args.reps):
            r = run_job(gen, nxt(d), 1, stop=False)
            runs.append({"tok_s": d / r["ttft"], "ttft": r["ttft"]})
            print(f"[prefill@{d}] rep{i}: {d / r['ttft']:.1f} tok/s (ttft {r['ttft']:.2f}s)", flush=True)
            gtt_guard(f"prefill{d}_rep{i}")
        RES["prefill"][str(d)] = {"runs": runs, "median_tok_s": statistics.median(x["tok_s"] for x in runs)}
        save()

    # ---------------- PPL
    try:
        RES["ppl"] = eval_ppl(model, tok, args.ppl_rows, args.ppl_len, "fast")
    except Exception as e:
        import traceback
        RES["ppl"] = {"error": repr(e), "tb": traceback.format_exc()[-2000:]}
    gtt_guard("after_ppl")
    save()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback
        RES["fatal"] = traceback.format_exc()[-4000:]
        save()
        raise
    print("DONE", flush=True)
