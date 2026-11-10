#!/usr/bin/env python3
"""Phase 4 correctness + first-decode-numbers harness for the contiguous layer split.

Runs the same fixed prompt battery through either the split (default) or a single-device
baseline (--baseline), saves the greedy token ids to JSON, and optionally compares two
saved runs token-for-token.  It also measures decode t/s under --num-draft and can run a
long-context needle probe.

Usage (from the repo root, after `source tools/strix_halo/env.sh`):

    python tools/strix_halo/split_eval.py -m <pack> --out /tmp/split.json
    python tools/strix_halo/split_eval.py -m <pack> --baseline --no-graphs --out /tmp/base.json
    python tools/strix_halo/split_eval.py --compare /tmp/base.json /tmp/split.json
    python tools/strix_halo/split_eval.py -m <pack> --needle-depth 120000 --out /tmp/needle.json

The split run forces the graph paths off before exllamav3 is imported (Phase 4 owns
correctness; per-segment capture is Phase 6).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time


# Kept in sync with exllamav3/serve_split.py:GRAPH_ENV_OFF; the consistency test
# asserts the key sets match so a new graph flag cannot silently bypass this gate.
GRAPH_ENV_OFF = (
    "EXL3_BLOCK_GRAPH", "EXL3_BLOCK_GRAPH_MLA", "EXL3_BLOCK_GRAPH_VERIFY",
    "EXL3_BLOCK_GRAPH_DRAFT", "EXL3_BLOCK_GRAPH_ATTN", "EXL3_BLOCK_GRAPH_DEC_MOE",
    "EXL3_CUDA_GRAPH", "EXL3_HIP_GRAPH",
)

PROMPTS = [
    "What is the capital of France? Answer in one sentence.",
    "Write a Python function that reverses a linked list, with comments.",
    "Explain in three short sentences why the sky is blue.",
    "List five prime numbers between 50 and 100, separated by commas.",
    "A train leaves at 14:35 and arrives at 17:20. How long is the trip? Show your reasoning.",
    "Summarize the plot of Romeo and Juliet in exactly two sentences.",
]
PERF_PROMPTS = [
    "Explain in detail how the attention mechanism in a transformer works, including "
    "queries, keys, values and multi-head attention.",
    "Write a Python implementation of a binary heap with push, pop and heapify, with comments.",
    "Write a detailed summary of the causes and consequences of the Industrial Revolution.",
]
NEEDLE = "KX-4817-QM"
FILLER = (
    "The archive room was quiet except for the soft hum of the ventilation and the slow "
    "turning of a clock on the far wall. Shelves of ledgers ran from floor to ceiling, each "
    "volume labelled in a hand that had not been seen in decades. Visitors rarely came this "
    "far, and those who did usually left with more questions than they had brought. "
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--model", default="/data/models/NicSir--GLM-5.3-Flash-EXL3-2.72bpw")
    p.add_argument("--baseline", action="store_true", help="single device (cuda:0), no layer split")
    p.add_argument("--front-layers", type=int, default=None)
    p.add_argument("--front-device", default="auto")
    p.add_argument("--endcaps", choices=("on", "off"), default="on")
    p.add_argument("--front-reserve-gb", type=float, default=1.2)
    p.add_argument("--main-reserve-gb", type=float, default=0.0)
    p.add_argument("-c", "--max-ctx", type=int, default=131072)
    p.add_argument("--num-draft", type=int, default=2)
    p.add_argument("--no-vision", action="store_true")
    p.add_argument("--no-graphs", action="store_true", help="force block/whole-forward graphs off (both modes)")
    p.add_argument("--no-budget", action="store_true", help="do not cap the per-device allocator (diagnostic: isolates reserve effects)")
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--perf-tokens", type=int, default=256)
    p.add_argument("--perf-reps", type=int, default=1)
    p.add_argument("--needle-depth", type=int, default=0, help="run one needle probe at this many prompt tokens")
    p.add_argument("--out", default=None, help="write the full JSON here")
    p.add_argument("--compare", nargs=2, metavar=("BASE", "SPLIT"),
                   help="compare two saved runs: prompt ids and decode numbers")
    p.add_argument("--no-perf", action="store_true", help="skip the 3 decode timing prompts")
    args = p.parse_args()
    if args.front_layers is None:
        # N=6 is measured to OOM the 16 GiB front above 65K (gfx1201 KDA workspace).
        args.front_layers = 5 if args.max_ctx > 65536 else 6
    return args


def configure_env(args: argparse.Namespace) -> None:
    """Set everything exllamav3 reads at import time, before the first import."""
    if args.no_graphs or not args.baseline:
        for k in GRAPH_ENV_OFF:
            os.environ[k] = "0"
    os.environ.setdefault("EXL3_MOE_CFG", "2")
    os.environ.setdefault("EXL3_HIP_PREFILL_MIN_ROWS", "2")
    # serve.py's SPEED_ENV: no dense-GEMM screening, so every run takes the same
    # (untuned, default) path instead of a per-process autotune winner.
    os.environ.setdefault("EXL3_DENSE_GEMM_TUNE", "0")
    os.environ.setdefault("EXL3_MOE_UNION_V2", "1")
    # Same optional unquantized MTP eh_proj as serve.py: raises draft acceptance, never output.
    if "EXL3_MTP_EH_FP16" not in os.environ:
        for path in (os.path.join(args.model, "mtp_eh_proj.st"),
                     os.path.expanduser("~/models/glm53-mtp-eh-proj-bf16.safetensors")):
            if os.path.isfile(path):
                os.environ["EXL3_MTP_EH_FP16"] = path
                break


def ids_of(text: str, tokenizer) -> torch.Tensor:
    p = f"[gMASK]<sop><|user|>\n{text}<|assistant|>\n"
    return tokenizer.encode(p, encode_special_tokens = True)


def run_job(gen, job_cls, ids, n: int):
    import torch
    from exllamav3.generator.sampler import GreedySampler
    job = job_cls(input_ids = ids, max_new_tokens = n, sampler = GreedySampler(),
                  stop_conditions = gen.model.config.eos_token_id_list)
    gen.enqueue(job)
    t0 = time.perf_counter()
    first = last = None
    toks = []
    text = ""
    accepted = rejected = 0
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("error"):
                raise RuntimeError(f"job error: {r['error']}")
            tid = r.get("token_ids")
            if tid is not None and tid.numel():
                if first is None:
                    first = time.perf_counter()
                last = time.perf_counter()
                toks.append(tid.cpu())
            text += r.get("text", "")
            accepted += int(r.get("accepted_draft_tokens", 0))
            rejected += int(r.get("rejected_draft_tokens", 0))
    ttft = (first - t0) if first is not None else None
    ntok = sum(t.numel() for t in toks)
    return {
        "ids": torch.cat(toks, dim = -1).flatten().tolist() if toks else [],
        "text": text,
        "ttft_s": ttft,
        "decode_s": (last - first) if first is not None and last is not None and ntok > 1 else None,
        "tokens": ntok,
        "accepted": accepted,
        "rejected": rejected,
    }


def build_needle(depth: int, tokenizer):
    """Prompt of about `depth` tokens with the passkey in the middle."""
    import torch
    head = "Read the document and answer the question that follows.\n\n"
    tail = "\n\nQuestion: What is the secret passkey in the document above? Reply with the passkey only.\n"
    p = tokenizer.encode(f"[gMASK]<sop><|user|>\n{head}", encode_special_tokens = True).flatten()
    q = tokenizer.encode(tail + "<|assistant|>\n", encode_special_tokens = True).flatten()
    filler_ids = tokenizer.encode(FILLER, encode_special_tokens = False).flatten()
    target = max(64, depth - int(p.numel() + q.numel()))
    reps = target // max(1, filler_ids.numel()) + 1
    ids = filler_ids.repeat(reps)[:target].clone()
    marker = tokenizer.encode(f"\nThe secret passkey is {NEEDLE}.\n", encode_special_tokens = False).flatten()
    mid = ids.numel() // 2
    ids = torch.cat((ids[:mid], marker, ids[mid:]), dim = -1)
    return torch.cat((p, ids, q), dim = -1).unsqueeze(0), NEEDLE


def main() -> None:
    args = parse_args()
    if args.compare:
        compare(args.compare[0], args.compare[1])
        return

    configure_env(args)

    import torch
    from exllamav3 import Cache, CacheLayer_quant, Config, Generator, Job, Model, Tokenizer
    from exllamav3.generator import generator as GM

    torch.set_grad_enabled(False)

    config = Config.from_directory(args.model)
    tokenizer = Tokenizer.from_config(config)
    model = Model.from_config(config)

    qkw = dict(layer_type = CacheLayer_quant, k_bits = 6, v_bits = 6)
    from exllamav3.cache.tq import CacheLayer_tq, supported as tq_supported
    if tq_supported(512, 6):
        qkw = dict(layer_type = CacheLayer_tq, k_bits = 6, v_bits = 6)
    max_history = max(args.num_draft, 1)
    cache = Cache(model, max_num_tokens = args.max_ctx + 4096, max_history = max_history,
                  max_batch_size = 1, **qkw)

    plan = None
    text_map = None
    if not args.baseline:
        from exllamav3.device_map import SplitPlan, resolve_devices
        devices = resolve_devices(args.front_device)
        plan = SplitPlan(
            devices, endcaps = args.endcaps == "on", front_layers = args.front_layers,
            front_reserve_gb = 0.0 if args.no_budget else args.front_reserve_gb,
            main_reserve_gb = args.main_reserve_gb)
        plan.add("text", model, config = config, cache = cache, load_order = 0)
        plan.build()
        text_map = plan.map("text")
        if args.front_layers > 0 and (text_map is None or not text_map.enabled):
            raise SystemExit("split map not enabled (no front device)")
        plan.install_bridges()
        plan.engage_budget()

    t0 = time.perf_counter()
    if text_map is not None:
        model.load(device_map = text_map, progressbar = True, max_chunk_size = 4096)
    else:
        model.load(device = "cuda:0", progressbar = True)
    load_s = time.perf_counter() - t0
    if plan is not None:
        plan.record_load("text", load_s)

    draft = draft_cache = None
    if args.num_draft > 0:
        draft = Model.from_config(config, component = "mtp")
        draft_cache = Cache(draft, max_num_tokens = args.max_ctx + 4096, max_history = max_history,
                            max_batch_size = 1, **qkw)
        t1 = time.perf_counter()
        if plan is not None:
            plan.add("mtp", draft, config = config, cache = draft_cache, load_order = 1)
            draft.load(device_map = plan.map("mtp"), progressbar = True, max_chunk_size = 4096)
            plan.record_load("mtp", time.perf_counter() - t1)
        else:
            draft.load(device = "cuda:0", progressbar = True)
        print(f"[eval] drafter loaded in {time.perf_counter() - t1:.1f} s", flush = True)

    vision = None
    if not args.no_vision:
        try:
            vision = Model.from_config(config, component = "vision")
            if plan is not None:
                plan.add("vision", vision, config = config, load_order = 2)
                t1 = time.perf_counter()
                vision.load(device_map = plan.map("vision"), progressbar = True)
                plan.record_load("vision", time.perf_counter() - t1)
            else:
                vision.load(device = "cuda:0", progressbar = True)
        except Exception as exc:                    # noqa: BLE001
            print(f"[eval] vision not loaded: {type(exc).__name__}: {exc}", flush = True)
            vision = None

    if plan is not None:
        extra = [("mtp", draft, draft_cache)] if draft is not None else []
        if vision is not None:
            extra.append(("vision", vision, None))
        plan.report(model, cache, extra = extra, ctx = args.max_ctx)
        plan.release_budget()

    GM.MTP_FUSE_CATCHUP = 2
    gen = Generator(model = model, cache = cache, tokenizer = tokenizer,
                    draft_model = draft, draft_cache = draft_cache,
                    num_draft_tokens = args.num_draft or None,
                    record_draft_stats = True)

    result = {
        "mode": "baseline" if args.baseline else "split",
        "model": args.model,
        "front_layers": None if args.baseline else args.front_layers,
        "endcaps": None if args.baseline else args.endcaps,
        "ctx": args.max_ctx,
        "num_draft": args.num_draft,
        "graphs_off": args.no_graphs or not args.baseline,
        "load_s": load_s,
        "prompts": [],
        "perf": [],
    }

    for prompt in PROMPTS:
        ids = ids_of(prompt, tokenizer)
        r = run_job(gen, Job, ids, args.tokens)
        r["prompt"] = prompt
        r["prompt_tokens"] = int(ids.numel())
        result["prompts"].append(r)
        print(f"[eval] {prompt[:44]!r}: {r['tokens']} tokens"
              + (f", ttft {r['ttft_s']:.2f} s" if r["ttft_s"] else "")
              + (f", decode {r['tokens'] / r['decode_s']:.2f} t/s"
                 if r["decode_s"] and r["tokens"] > 1 else ""), flush = True)

    if not args.no_perf:
        for i, prompt in enumerate(PERF_PROMPTS):
            ids = ids_of(prompt, tokenizer)
            reps = []
            for rep in range(max(1, args.perf_reps)):
                r = run_job(gen, Job, ids, args.perf_tokens)
                tps = (r["tokens"] - 1) / r["decode_s"] if r["decode_s"] and r["tokens"] > 1 else None
                reps.append({"tokens": r["tokens"], "ttft_s": r["ttft_s"], "decode_s": r["decode_s"],
                             "decode_tps": tps, "accepted": r["accepted"], "rejected": r["rejected"]})
                print(f"[eval] perf {i}.{rep}: ttft {r['ttft_s']:.2f} s, decode {tps:.2f} t/s"
                      f" (acc {r['accepted']} / rej {r['rejected']})", flush = True)
            good = [x["decode_tps"] for x in reps if x["decode_tps"]]
            result["perf"].append({
                "prompt": prompt, "prompt_tokens": int(ids.numel()), "reps": reps,
                "decode_tps": (sum(good) / len(good)) if good else None,
            })

    if args.needle_depth > 0:
        ids, needle = build_needle(min(args.needle_depth, args.max_ctx - 1024), tokenizer)
        r = run_job(gen, Job, ids, 24)
        found = needle.lower() in r["text"].lower()
        result["needle"] = {
            "depth": int(ids.numel()), "needle": needle, "found": found,
            "text": r["text"], "ttft_s": r["ttft_s"], "tokens": r["tokens"],
        }
        print(f"[eval] needle depth {ids.numel()}: {'FOUND' if found else 'MISSING'}"
              f" (ttft {r['ttft_s']:.1f} s) text={r['text'][:120]!r}", flush = True)

    bridge = getattr(config, "device_bridge", None)
    if bridge is not None:
        print(bridge.summary(), flush = True)
        result["bridge"] = bridge.stats

    if args.out:
        tmp = args.out + ".tmp"
        with open(tmp, "w", encoding = "utf8") as f:
            json.dump(result, f, indent = 1)
        os.replace(tmp, args.out)
        print(f"[eval] wrote {args.out}", flush = True)


def compare(base_path: str, split_path: str) -> None:
    base = json.load(open(base_path, encoding = "utf8"))
    split = json.load(open(split_path, encoding = "utf8"))
    n = min(len(base.get("prompts", [])), len(split.get("prompts", [])))
    same = 0
    for i in range(n):
        a = base["prompts"][i]["ids"]
        b = split["prompts"][i]["ids"]
        ok = a == b
        same += ok
        print(f"[compare] prompt {i}: {'IDENTICAL' if ok else 'DIVERGED'} "
              f"({len(a)} vs {len(b)} ids)")
        if not ok:
            d = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            print(f"[compare]   first divergence at {d}: "
                  f"{a[d:d + 8]} vs {b[d:d + 8]}")
    print(f"[compare] {same}/{n} greedy prompts identical")
    if base.get("perf") and split.get("perf"):
        for i, (a, b) in enumerate(zip(base["perf"], split["perf"])):
            if a.get("decode_tps") and b.get("decode_tps"):
                print(f"[compare] perf {i}: baseline {a['decode_tps']:.2f} t/s, "
                      f"split {b['decode_tps']:.2f} t/s "
                      f"({b['decode_tps'] / a['decode_tps']:.3f}x)")
    if "needle" in split:
        print(f"[compare] needle: {'found' if split['needle']['found'] else 'MISSING'}")


if __name__ == "__main__":
    main()
