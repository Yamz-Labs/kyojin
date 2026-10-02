#!/usr/bin/env python
"""BRIEF-14 — DFlash speculative decoding for MiMo-V2.6 EXL3 on gfx1151, one load, one process.

Everything the brief asks for happens inside a single model load, because the box serialises
full-model runs behind an exclusive GPU lock with a <20 min budget per window:

  * decode tok/s + acceptance with and without the drafter, 3 prompt types x 3 reps (unique
    prompts per rep), 2K context, 128 new tokens, greedy;
  * token-by-token equality of the greedy outputs with and without the drafter (speculative
    decoding is lossless under greedy);
  * draft-length sweep 2/4/6/8;
  * where the time goes: the drafter's own forward, the trunk verify forward at R = 1 + ndt, and
    the trunk's MoE share at those row counts (BlockSparseMLP.forward is event-timed by row count).

  bash -lc 'source ~/kyojin/tools/strix_halo/env.sh && \
      EXL3_REPO=~/kyojin/dflash PYTHONPATH=$EXL3_REPO \
      python ~/kyojin/dflash/scripts/dflash-bench.py \
        -m ~/models/mimo26-exl3 -dm ~/kyojin/drafter/mimo-dflash-draft --out /tmp/dflash.json'
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.environ.get("EXL3_REPO", "~/kyojin"))

import torch  # noqa: E402

from exllamav3 import Cache, CacheLayer_fp16, Generator, GreedySampler, Job, model_init  # noqa: E402

CORPUS = {
    "prose": os.path.expanduser("~/bench/ppl/wiki.test.raw"),
    "code": os.path.join(os.environ.get("EXL3_REPO", "~/kyojin"), "exllamav3/conversion/standard_cal_data/code.utf8"),
}
# The "chat" prompt type is the model's own chat template rendered around a unique long user turn,
# so it carries the <|im_start|> markers a served conversation has, with content that differs per rep.
CHAT_TEMPLATE = os.path.expanduser("~/models/mimo26-exl3/chat_template.jinja")
PROMPT_TOKENS = 2048
NEW_TOKENS = 128


def load_corpus(path: str) -> str:
    with open(os.path.expanduser(path), "r", encoding = "utf-8") as f:
        return f.read()


def slice_tokens(tokenizer, text: str, n: int, offset: int) -> torch.Tensor:
    ids = tokenizer.encode(text, add_bos = False)
    if ids.shape[-1] < offset + n:
        raise RuntimeError(f"corpus tokenizes to {ids.shape[-1]}, need {offset + n}")
    return ids[:, offset:offset + n].contiguous()


def chat_tokens(tokenizer, text: str, n: int, offset: int) -> torch.Tensor:
    # the rendered template used to be cut at n tokens, which dropped the assistant
    # generation prompt, so "chat" was really raw wiki continuation (degenerate "= = =" output).
    # Now: a real user request over a unique ~n-token wiki slice, template kept whole.
    import jinja2
    env = jinja2.Environment(trim_blocks = False, lstrip_blocks = False)
    tmpl = env.from_string(open(CHAT_TEMPLATE, "r", encoding = "utf-8").read())
    body_ids = tokenizer.encode(text, add_bos = False)[:, offset:offset + n - 64]
    body = tokenizer.decode(body_ids)[0]
    user = ("Here is an excerpt from an encyclopedia article:\n\n" + body +
            "\n\nExplain in your own words, in a friendly tone, what this excerpt is about "
            "and what the most interesting facts in it are.")
    rendered = tmpl.render(messages = [{"role": "user", "content": user}],
                           add_generation_prompt = True, bos_token = "<|begin_of_text|>")
    return tokenizer.encode(rendered, add_bos = False, encode_special_tokens = True).contiguous()


def prompt_ids(tokenizer, corpora, kind: str, n: int, offset: int) -> torch.Tensor:
    if kind == "chat":
        return chat_tokens(tokenizer, corpora["prose"], n, offset)
    return slice_tokens(tokenizer, corpora[kind], n, offset)


def run_job(generator, prompt_ids, max_new_tokens, want_logits: bool = False) -> dict:
    job = Job(input_ids = prompt_ids, max_new_tokens = max_new_tokens,
              stop_conditions = generator.model.config.eos_token_id_list,
              sampler = GreedySampler(), return_logits = want_logits)
    generator.enqueue(job)
    t0 = time.perf_counter()
    ttft, ntok, out_ids, top2 = None, 0, [], None
    while generator.num_remaining_jobs():
        for res in generator.iterate():
            if res.get("error"):
                raise RuntimeError(f"job error: {res['error']}")
            if res.get("token_ids") is not None:
                out_ids = [int(t) for t in res["token_ids"][0].tolist()]
            if want_logits and res.get("logits") is not None:
                # top-2 per position is all the tie analysis needs; the full logits are
                # (1, n, 152576) fp32 = ~78 MB per run
                top2 = res["logits"][0].topk(2, dim = -1).values.float().cpu()
            if res.get("text"):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                ntok += 1
    total = time.perf_counter() - t0
    stats = list(getattr(job, "draft_stats", []) or [])
    # The authoritative output is the job's own sequence (the streamed "token_ids" payload is
    # incremental per step and silently collapsed to the last token when accumulated that way).
    seq = job.sequences[0]
    t = seq.sequence_ids.torch()
    all_ids = (t[0] if t.dim() > 1 else t).tolist()
    prompt_len = len(seq.input_ids)
    out_ids = all_ids[prompt_len:]
    return dict(ids = out_ids, ntok = len(out_ids), prompt_len = prompt_len, ttft = ttft,
                total = total, top2 = top2,
                accepted = sum(s[2] for s in stats), window = sum(s[1] for s in stats),
                steps = len(stats))


def divergence(a: dict, b: dict) -> dict:
    """First differing position plus the base path's top-2 logit gap there (tie_check style)."""
    ia, ib = a["ids"], b["ids"]
    n = min(len(ia), len(ib))
    first = next((i for i in range(n) if ia[i] != ib[i]), None)
    if first is None and len(ia) != len(ib):
        first = n
    out = dict(first_diff = first, len_base = len(ia), len_draft = len(ib),
               n_diff = sum(1 for x, y in zip(ia, ib) if x != y))
    if first is not None and a.get("top2") is not None and first < a["top2"].shape[0]:
        t = a["top2"][first]
        out["base_top1"] = float(t[0]); out["base_top2"] = float(t[1])
        out["base_gap"] = float(t[0] - t[1])
        out["base_argmax"] = int(t[0] == t[0]) and ia[first]
        out["draft_token"] = ib[first] if first < len(ib) else None
    return out


def tps(r: dict) -> float:
    decode_time = r["total"] - (r["ttft"] or 0.0)
    return (r["ntok"] - 1) / decode_time if decode_time > 0 and r["ntok"] > 1 else 0.0


def med(xs):
    return statistics.median(xs) if xs else 0.0


def gtt_gib() -> float:
    """Device memory in use (GTT) -- the number the 22:03 incident was about."""
    import glob
    best = 0
    for p in glob.glob("/sys/class/drm/card*/device/mem_info_gtt_used"):
        try:
            best = max(best, int(open(p).read().strip()))
        except Exception:                                        # noqa: BLE001
            pass
    return best / 2**30


def mem_avail_gib() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1048576
    return 0.0


def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev = False)
    model_init.add_args(ap, cache = True, add_draft_model_args = True)
    ap.add_argument("--reps", type = int, default = 3)
    ap.add_argument("--sweep", default = "2,4,6,8")
    ap.add_argument("--kinds", default = "prose,code,chat",
                    help = "which prompt types to run (short GTT check: --kinds prose)")
    ap.add_argument("--new-tokens", type = int, default = NEW_TOKENS)
    ap.add_argument("--out", default = "scratch/dflash-bench.json")
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    print(f"torch {torch.__version__} hip {torch.version.hip}", flush = True)
    t0 = time.time()
    model, config, cache, tokenizer, draft_model, draft_config, draft_cache = model_init.init(args)

    # ONE draft length per process (lead, 22:03): the box died under memory pressure because a
    # --sweep plan allocated several multi-GB caches up front. The caches below are the only ones
    # this process creates; the draft cache is the one init() built (it must predate the drafter's
    # load), and -cs is kept minimal for 2K context + 128 new tokens.
    ndt = int(args.num_draft_tokens or 4)
    cs = args.cache_size or 2304
    caches = {
        ndt: dict(
            base = Cache(model, max_num_tokens = cs, layer_type = CacheLayer_fp16,
                         max_history = max(4, ndt)),
            base2 = Cache(model, max_num_tokens = cs, layer_type = CacheLayer_fp16,
                          max_history = max(4, ndt)),
            draft = Cache(model, max_num_tokens = cs, layer_type = CacheLayer_fp16,
                          max_history = max(4, ndt)),
            draft_kv = draft_cache,
        )
    }
    model.load()
    print(f"load {time.time() - t0:.1f}s  drafter={'yes' if draft_model else 'no'}  ndt={ndt}  "
          f"cs={cs}", flush = True)
    print(f"gtt {gtt_gib():.1f} GiB  host_avail {mem_avail_gib():.1f} GiB", flush = True)

    # ---- instrumentation (installed only for the profiling phase) ---------------------------
    timers = collections.defaultdict(float)
    calls = collections.Counter()
    skipped = collections.Counter()
    import exllamav3.modules.block_sparse_mlp as bsm
    orig_mlp_fwd = bsm.BlockSparseMLP.forward
    dflash_cls = type(draft_model) if draft_model is not None else None
    orig_draft_fwd = getattr(dflash_cls, "forward", None) if dflash_cls is not None else None

    def timed(orig, key):
        def inner(*a, **kw):
            e0 = torch.cuda.Event(enable_timing = True); e1 = torch.cuda.Event(enable_timing = True)
            e0.record(); r = orig(*a, **kw); e1.record()
            calls[key] += 1
            try:
                timers[key] += e0.elapsed_time(e1)
            except RuntimeError:            # CUDA-graph capture: events never complete, skip
                skipped[key] += 1
            return r
        return inner

    def mlp_fwd(self, x, params, *a, **kw):
        rows = x.shape[0] * x.shape[1] if x.dim() == 3 else x.shape[0]
        e0 = torch.cuda.Event(enable_timing = True); e1 = torch.cuda.Event(enable_timing = True)
        e0.record(); r = orig_mlp_fwd(self, x, params, *a, **kw); e1.record()
        calls[f"moe_R{rows}"] += 1
        try:
            timers[f"moe_R{rows}"] += e0.elapsed_time(e1)
        except RuntimeError:
            skipped[f"moe_R{rows}"] += 1
        return r

    def install_profiling():
        bsm.BlockSparseMLP.forward = mlp_fwd
        if dflash_cls is not None and orig_draft_fwd is not None:
            dflash_cls.forward = timed(orig_draft_fwd, "drafter_forward")

    def remove_profiling():
        bsm.BlockSparseMLP.forward = orig_mlp_fwd
        if dflash_cls is not None and orig_draft_fwd is not None:
            dflash_cls.forward = orig_draft_fwd

    def make_gen(use_draft: bool, ndt: int, which: str = "draft"):
        c = caches[ndt]
        return Generator(
            model = model,
            cache = c[which],
            tokenizer = tokenizer,
            draft_model = draft_model if use_draft else None,
            draft_cache = c["draft_kv"] if use_draft else None,
            num_draft_tokens = ndt, dynamic_draft_tokens = args.dynamic_draft,
            draft_confidence = args.draft_confidence,
            record_draft_stats = True,
        )

    corpora = {k: load_corpus(v) for k, v in CORPUS.items()}
    out: dict = {"drafter": args.draft_model_dir, "model": args.model_dir, "runs": {},
                 "equality": {}, "sweep": {}, "profile": {}, "warnings": []}

    def guarded(label, fn):
        """Phase-level guard: a late failure must not throw away the phases that already ran."""
        try:
            return fn()
        except Exception as e:                                   # noqa: BLE001
            out["warnings"].append(f"{label}: {type(e).__name__}: {e}")
            print(f"  !! {label} failed: {type(e).__name__}: {e}", flush = True)
            return None

    # ---- 1. with vs without, 3 types x reps --------------------------------------------------
    def phase_types():
        for kind in args.kinds.split(","):
            base_gen = make_gen(False, args.num_draft_tokens, "base")
            base2_gen = make_gen(False, args.num_draft_tokens, "base2")
            draft_gen = make_gen(True, args.num_draft_tokens, "draft")
            base_tps, draft_tps, acc, win, same = [], [], 0, 0, 0
            detail = []
            # Discarded warm-up leg (JIT/graph capture), on a slice outside the timed reps
            wids = prompt_ids(tokenizer, corpora, kind, PROMPT_TOKENS, args.reps * PROMPT_TOKENS)
            run_job(base_gen, wids, 32); run_job(base2_gen, wids, 8); run_job(draft_gen, wids, 32)
            for rep in range(args.reps):
                ids = prompt_ids(tokenizer, corpora, kind, PROMPT_TOKENS, rep * PROMPT_TOKENS)
                a = run_job(base_gen, ids, args.new_tokens, want_logits = True)
                b = run_job(draft_gen, ids, args.new_tokens)
                if rep == 0:
                    # determinism controls: base vs base, and drafted vs drafted (fork methodology)
                    c = run_job(base2_gen, ids, args.new_tokens)
                    ctrl_same = a["ids"] == c["ids"]
                    d2 = run_job(draft_gen, ids, args.new_tokens)
                    ctrl_draft_same = b["ids"] == d2["ids"]
                    out.setdefault("control", {})[kind] = dict(
                        base_identical = ctrl_same,
                        draft_identical = ctrl_draft_same,
                        base = divergence(a, c),
                        draft = divergence(b, d2))
                    print(f"  {kind} control: base-vs-base (fresh cache) identical={ctrl_same} | "
                          f"draft-vs-draft (same generator) identical={ctrl_draft_same}", flush = True)
                base_tps.append(tps(a)); draft_tps.append(tps(b))
                acc += b["accepted"]; win += b["window"]
                detail.append(dict(rep = rep, base = dict(tps = tps(a), ntok = a["ntok"],
                                                          total = a["total"]),
                                   draft = dict(tps = tps(b), ntok = b["ntok"],
                                                total = b["total"], steps = b["steps"],
                                                accepted = b["accepted"], window = b["window"],
                                                tokens_per_step = (b["ntok"] / b["steps"]
                                                                   if b["steps"] else None))))
                eq = a["ids"] == b["ids"]
                same += int(eq)
                if not eq:
                    d = divergence(a, b)
                    # Targeted probe: re-run both paths for exactly first_diff+1 tokens so the
                    # logits payload covers the divergence and the tie/no-tie question can be
                    # answered (fork methodology: a summation-order flip is legitimate, a flip
                    # with a multi-logit gap is a bug).
                    k = max(1, min((d["first_diff"] or 0) + 1, args.new_tokens))
                    try:
                        a2 = run_job(base_gen, ids, k, want_logits = True)
                        b2 = run_job(draft_gen, ids, k, want_logits = True)
                        for tag, r in (("base", a2), ("draft", b2)):
                            d[f"{tag}_last"] = (r["ids"][-1] if r["ids"] else None)
                            d[f"{tag}_top2"] = (r["top2"][-1].tolist()
                                                if r["top2"] is not None else None)
                        d["same_token"] = d.get("base_last") == d.get("draft_last")
                    except Exception as e:                           # noqa: BLE001
                        d["tie_probe_error"] = f"{type(e).__name__}: {e}"
                    try:
                        lo = max(0, (d["first_diff"] or 0) - 8)
                        d["base_text"] = tokenizer.decode(torch.tensor([a["ids"][lo:lo + 16]]))[0]
                        d["draft_text"] = tokenizer.decode(torch.tensor([b["ids"][lo:lo + 16]]))[0]
                    except Exception:                                # noqa: BLE001
                        pass
                    out.setdefault("divergence", {}).setdefault(kind, []).append(dict(rep = rep, **d))
                    out["warnings"].append(f"{kind} rep{rep}: greedy divergence at token "
                                           f"{d['first_diff']} (n_diff={d['n_diff']})")
                print(f"  {kind} rep{rep}: base {tps(a):5.2f} tok/s | draft {tps(b):5.2f} tok/s | "
                      f"acc {100*b['accepted']/max(b['window'],1):4.1f}% | identical={eq}"
                      f" | first_diff={0 if eq else divergence(a, b)['first_diff']}", flush = True)
            out["runs"][kind] = dict(
                base_tps = base_tps, draft_tps = draft_tps,
                base_median = med(base_tps), draft_median = med(draft_tps),
                speedup = med(draft_tps) / med(base_tps) if med(base_tps) else None,
                acceptance_pct = 100 * acc / win if win else None,
                identical_reps = same, reps = args.reps,
                detail = detail)
            del base_gen, base2_gen, draft_gen
            torch.cuda.empty_cache()

    # ---- 2. draft-length sweep -----------------------------------------------------------------
    # One draft length per process (lead, 22:03): the sweep is now three processes with different
    # -ndt, and each process reports its own prose numbers from phase 1.
    def phase_sweep():
        print(f"  sweep: this process covers ndt={ndt} only (one draft length per process); "
              f"requested {args.sweep}", flush = True)
        out["sweep"] = {str(ndt): dict(tps_median = out.get("runs", {}).get("prose", {})
                                       .get("draft_median"),
                                       acceptance_pct = out.get("runs", {}).get("prose", {})
                                       .get("acceptance_pct"),
                                       note = "per-process run; prose medians from phase 1")}

    # ---- 3. where the time goes (instrumentation installed only here) -------------------------
    def phase_profile():
        timers.clear(); calls.clear(); skipped.clear()
        install_profiling()
        try:
            gen = make_gen(True, args.num_draft_tokens)
            for rep in range(2):
                ids = prompt_ids(tokenizer, corpora, "prose", PROMPT_TOKENS, rep * PROMPT_TOKENS)
                run_job(gen, ids, args.new_tokens)
        finally:
            remove_profiling()
        def per_call(key):
            n = calls.get(key, 0) - skipped.get(key, 0)
            return round(1e3 * timers[key] / n, 1) if n > 0 else None

        out["profile"] = {
            "seconds": {k: round(v, 4) for k, v in sorted(timers.items())},
            "calls": dict(sorted(calls.items())),
            "skipped_event_samples": dict(sorted(skipped.items())),
            "note": "the decode loop runs under CUDA-graph capture, so event timings are only "
                    "available for the calls outside the graph; call counts are complete",
            "draft_forward_per_call_us": per_call("drafter_forward"),
            "moe_calls_by_rows": {k: calls[k] for k in sorted(calls) if k.startswith("moe_R")},
            "moe_us_per_call_by_rows": {k: per_call(k) for k in sorted(calls)
                                        if k.startswith("moe_R")},
        }
        print("  profile:", json.dumps(out["profile"], indent = 1)[:1200], flush = True)

    gtt_trace = []

    def note_mem(label: str):
        g, m = gtt_gib(), mem_avail_gib()
        gtt_trace.append(dict(at = label, gtt_gib = round(g, 1), host_avail_gib = round(m, 1)))
        print(f"  mem[{label}]: gtt {g:.1f} GiB  host_avail {m:.1f} GiB", flush = True)

    note_mem("after load")
    guarded("phase 1 (with/without)", phase_types)
    note_mem("after phase 1")
    guarded("phase 2 (sweep)", phase_sweep)
    if os.environ.get("BENCH_PROFILE", "0") != "0":
        guarded("phase 3 (profile)", phase_profile)
    note_mem("end")
    out["gtt_trace"] = gtt_trace
    out["gtt_peak_gib"] = max(t["gtt_gib"] for t in gtt_trace)
    out["min_host_avail_gib"] = min(t["host_avail_gib"] for t in gtt_trace)

    json.dump(out, open(args.out, "w"), indent = 1)
    print(f"\nwrote {args.out}  gtt_peak {out['gtt_peak_gib']:.1f} GiB  "
          f"min_host_avail {out['min_host_avail_gib']:.1f} GiB", flush = True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
