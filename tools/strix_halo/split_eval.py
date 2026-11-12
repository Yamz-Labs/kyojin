#!/usr/bin/env python3
"""Phase 4 correctness + first-decode-numbers harness for the contiguous layer split.

Runs the same fixed prompt battery through either the split (default) or a single-device
baseline (--baseline), saves the greedy token ids to JSON, and optionally compares two
saved runs token-for-token.  It also measures decode t/s under --num-draft and can run a
long-context needle probe.

The split works on any two supported devices; use --part1-device/--part0-device to put
either device in either position.  The context-dependent --part0-layers default is
harness tuning for a small part0 card, not a property of the split; pass --part0-layers
to choose N explicitly.

Usage (from the repo root, after `source tools/strix_halo/env.sh`):

    python tools/strix_halo/split_eval.py -m <pack> --out /tmp/split.json
    python tools/strix_halo/split_eval.py -m <pack> --baseline --no-graphs --out /tmp/base.json
    python tools/strix_halo/split_eval.py --compare /tmp/base.json /tmp/split.json
    python tools/strix_halo/split_eval.py -m <pack> --needle-depth 120000 --out /tmp/needle.json
    python tools/strix_halo/split_eval.py -m <pack> --kld-corpus /path/corpus.txt --kld-out /tmp/A.kld
    python tools/strix_halo/split_eval.py --kld-compare /tmp/A.kld /tmp/B.kld

The short battery also scores programmatic task checks, and the split/one-device
comparison reports them.  --kld-out teacher-forces corpus slices through
Model.forward; --kld-compare reports KLD in both directions, top-1 agreement and PPL.

The split run forces the graph paths off before exllamav3 is imported (Phase 4 owns
correctness; per-segment capture is Phase 6).
"""
from __future__ import annotations

import argparse
import json
import os
import re
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
# Long-form instructions for --bench-acceptance.  A passkey answer is only ~20 tokens,
# which is too few drafted tokens to compare acceptance between configurations; these
# ask for a long answer so the rate is measured over a useful sample.
# The passkey answer is too short for an acceptance comparison; a summary-set arm
# reaches several hundred drafted tokens.  Keep the compare output honest below this.
MIN_ACCEPTANCE_SAMPLES = 100
SUMMARY_INSTRUCTIONS = (
    "Summarize the document in detail, at least 200 words.",
    "Explain the main trade-offs and risks described above in detail.",
    "Write a technical analysis of the document above, at least 200 words.",
)
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
    p.add_argument("--part0-layers", type=int, default=None)
    p.add_argument("--part1-device", default="auto")
    p.add_argument("--part0-device", default="auto")
    p.add_argument("--long-ctx-threshold", type=int, default=65536,
                   help="context above which the harness default part0-layer count changes")
    p.add_argument("--long-ctx-part0-layers", type=int, default=5,
                   help="harness default for contexts above --long-ctx-threshold; measured on a "
                        "small part0 card, not a property of the split")
    p.add_argument("--short-ctx-part0-layers", type=int, default=6,
                   help="harness default at or below --long-ctx-threshold")
    p.add_argument("--endcaps", choices=("on", "off"), default="on")
    p.add_argument("--part0-reserve-gb", type=float, default=1.2)
    p.add_argument("--part1-reserve-gb", type=float, default=0.0)
    p.add_argument("-c", "--max-ctx", type=int, default=131072)
    p.add_argument("--num-draft", type=int, default=2)
    p.add_argument("--no-vision", action="store_true")
    p.add_argument("--vision-image", default=None,
                   help="run this image through the split vision tower once (twice timed) and "
                        "record the first-image latency; the vision tower must be loaded "
                        "(cannot be combined with --no-vision)")
    p.add_argument("--no-graphs", action="store_true", help="force block/whole-forward graphs off (both modes)")
    p.add_argument("--no-budget", action="store_true", help="do not cap the per-device allocator (diagnostic: isolates reserve effects)")
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--perf-tokens", type=int, default=256)
    p.add_argument("--perf-reps", type=int, default=1)
    p.add_argument("--needle-depth", type=int, default=0, help="run one needle probe at this many prompt tokens")
    p.add_argument("--bench-depth", type=int, default=0,
                   help="also decode --perf-tokens at this prompt depth and report prefill/decode/acceptance")
    p.add_argument("--bench-prompt", choices=("passkey", "summary"), default="passkey",
                   help="long-depth prompt: passkey (needle) or a long-form summary instruction")
    p.add_argument("--bench-acceptance", action="store_true",
                   help="run all long-form summary instructions at --bench-depth and report the "
                        "aggregate draft acceptance with a 95%% interval")
    p.add_argument("--kld-corpus", default=os.environ.get("KYOJIN_KLD_CORPUS"),
                   help="plain-text corpus for --kld-out (or set KYOJIN_KLD_CORPUS)")
    p.add_argument("--kld-slices", type=int, default=2, help="corpus slices for --kld-out")
    p.add_argument("--kld-len", type=int, default=512, help="tokens per KLD slice")
    p.add_argument("--kld-out", default=None,
                   help="teacher-force the corpus and save log-probs/PPL for --kld-compare")
    p.add_argument("--kld-compare", nargs=2, metavar=("BASE", "SPLIT"),
                   help="compare two --kld-out files: KLD both directions, top-1 agreement, PPL")
    p.add_argument("--out", default=None, help="write the full JSON here")
    p.add_argument("--compare", nargs=2, metavar=("BASE", "SPLIT"),
                   help="compare two saved runs: prompt ids and decode numbers")
    p.add_argument("--no-perf", action="store_true", help="skip the 3 decode timing prompts")
    args = p.parse_args()
    if args.bench_acceptance and args.bench_depth <= 0:
        p.error("--bench-acceptance needs --bench-depth N (the prompt depth to measure)")
    if args.vision_image and args.no_vision:
        p.error("--vision-image needs the vision tower; remove --no-vision")
    if args.kld_out and not args.kld_corpus:
        p.error("--kld-out needs --kld-corpus (a plain-text file) or KYOJIN_KLD_CORPUS")
    if args.kld_out and (args.kld_slices < 1 or args.kld_len < 2):
        p.error("--kld-slices must be >= 1 and --kld-len >= 2")
    args.part0_layers_defaulted = args.part0_layers is None
    if args.part0_layers is None:
        # Harness tuning for memory, not a split property: a small part0 card cannot
        # hold N=6 plus the long-context workspace.  Both values are explicit flags.
        args.part0_layers = (args.long_ctx_part0_layers if args.max_ctx > args.long_ctx_threshold
                             else args.short_ctx_part0_layers)
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
    # GLM serve's safe vision default: on gfx12 the MIOpen F.conv2d path can fail with
    # "invalid device function", while the GEMM spatial-merge path is the measured fallback.
    os.environ.setdefault("EXL3_CONV_GEMM", "1")
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


def build_context_prompt(depth: int, tokenizer, instruction: str):
    """~``depth`` tokens of filler followed by ``instruction``, for long-form acceptance."""
    import torch
    head = "Read the document below and follow the instruction that follows.\n\n"
    tail = f"\n\nInstruction: {instruction}\n"
    p = tokenizer.encode(f"[gMASK]<sop><|user|>\n{head}", encode_special_tokens = True).flatten()
    q = tokenizer.encode(tail + "<|assistant|>\n", encode_special_tokens = True).flatten()
    filler_ids = tokenizer.encode(FILLER, encode_special_tokens = False).flatten()
    target = max(64, depth - int(p.numel() + q.numel()))
    reps = target // max(1, filler_ids.numel()) + 1
    ids = filler_ids.repeat(reps)[:target].clone()
    return torch.cat((p, ids, q), dim = -1).unsqueeze(0)


def acceptance_ci95(accepted: int, rejected: int) -> list[float] | None:
    """Wilson score interval for the acceptance rate, in percent (None without samples)."""
    import math
    total = int(accepted) + int(rejected)
    if total <= 0:
        return None
    z = 1.96
    p = int(accepted) / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denom
    return [round(max(0.0, center - half) * 100.0, 2), round(min(1.0, center + half) * 100.0, 2)]


def score_task(index: int, text: str) -> dict:
    """Coarse programmatic check for the short battery: task accuracy, not just id equality.

    Greedy ids already show determinism; this adds a signal that a run is still *correct*
    when a mixed-architecture pair legitimately diverges token-for-token.
    """
    t = (text or "").lower()
    expected, passed = "", False
    if index == 0:                                  # capital of France
        expected, passed = "paris", "paris" in t
    elif index == 1:                                # reverse a linked list
        expected, passed = "def + next", "def " in t and "next" in t
    elif index == 2:                                # why the sky is blue
        expected, passed = "scatter/rayleigh", any(w in t for w in ("scatter", "rayleigh", "wavelength"))
    elif index == 3:                                # five primes between 50 and 100
        expected = ">=3 primes 50-100"
        primes = re.findall(r"\b(?:53|59|61|67|71|73|79|83|89|97)\b", t)
        passed = len(set(primes)) >= 3
    elif index == 4:                                # 14:35 -> 17:20 is 2 h 45 min
        expected = "2 h 45 min"
        passed = bool(re.search(
            r"\b2\s*(?:h|hrs?|hours?)\s*(?:and\s*)?45\b|\b2:45\b|two hours (?:and )?forty[- ]five", t))
    elif index == 5:                                # Romeo and Juliet
        expected, passed = "verona/love", "verona" in t or ("love" in t and ("famil" in t or "feud" in t))
    return {"task": index, "expected": expected, "passed": bool(passed)}


def kld_stats(lp_a, lp_b) -> dict:
    """Compare two (positions, vocab) log-probability matrices.

    Returns mean KLD in both directions, top-1 agreement and the positions compared.  The
    first argument is the reference (baseline) arm.
    """
    import torch
    n = min(int(lp_a.shape[0]), int(lp_b.shape[0]))
    if n <= 0:
        return {"positions": 0}
    a = lp_a[:n].float()
    b = lp_b[:n].float()
    return {
        "positions": n,
        "kld_base_to_split": float((a.exp() * (a - b)).sum(-1).mean()),
        "kld_split_to_base": float((b.exp() * (b - a)).sum(-1).mean()),
        "top1_agreement_pct": float((a.argmax(-1) == b.argmax(-1)).float().mean() * 100.0),
    }


def corpus_ids(path: str, tokenizer, need: int):
    """At least ``need`` ids from a plain-text corpus (repeats the text if it is short)."""
    text = open(path, encoding="utf8", errors="replace").read()
    ids = tokenizer.encode(text, encode_special_tokens = False).flatten()
    if ids.numel() < need:
        reps = need // max(1, ids.numel()) + 1
        ids = ids.repeat(reps)
    return ids


def run_kld(model, ids):
    """One cacheless teacher-forced forward; returns per-position log-probs and mean NLL."""
    import torch
    logits = model.forward(ids.unsqueeze(0), {"attn_mode": "flash_attn_nc"})
    lp = torch.log_softmax(logits[0].float(), -1)
    n = min(lp.shape[0] - 1, ids.numel() - 1)
    tgt = ids[1:1 + n].to(lp.device)
    nll = float(-lp[:n].gather(1, tgt[:, None]).sum())
    return lp[:n].half().cpu(), nll, n


def kld_compare(base_path: str, split_path: str) -> None:
    """Report KLD/PPL from two --kld-out files (baseline first)."""
    import torch
    base = torch.load(base_path, map_location = "cpu", weights_only = True)
    split = torch.load(split_path, map_location = "cpu", weights_only = True)
    if base.get("ids") != split.get("ids"):
        raise SystemExit("KLD files do not use the same corpus slices")
    scored = kld_pos = 0
    kld_ab = kld_ba = agree = 0.0
    nll_a = nll_b = 0.0
    for i, (la, lb) in enumerate(zip(base["logprobs"], split["logprobs"])):
        st = kld_stats(la, lb)
        n = st["positions"]
        if n == 0:
            continue
        kld_ab += st["kld_base_to_split"] * n
        kld_ba += st["kld_split_to_base"] * n
        agree += st["top1_agreement_pct"] * n
        kld_pos += n
        nll_a += base["nll"][i]
        nll_b += split["nll"][i]
        scored += base["scored"][i]
        print(f"[kld] slice {i}: KLD {st['kld_base_to_split']:.3e} / {st['kld_split_to_base']:.3e} "
              f"(base|split, split|base), top1 {st['top1_agreement_pct']:.2f}%", flush = True)
    if kld_pos == 0 or scored == 0:
        raise SystemExit("KLD files have no scored positions")
    import math
    print(f"[kld] KLD mean {kld_ab / kld_pos:.3e} (base|split) / {kld_ba / kld_pos:.3e} (split|base); "
          f"top1 agreement {agree / kld_pos:.2f}%; "
          f"PPL base {math.exp(nll_a / scored):.4f}, split {math.exp(nll_b / scored):.4f} "
          f"({kld_pos} positions)", flush = True)


def main() -> None:
    args = parse_args()
    if args.part0_layers_defaulted and not args.baseline:
        print(f"[harness] no --part0-layers given: using N={args.part0_layers} at -c "
              f"{args.max_ctx} (harness tuning, not a split property; override with "
              f"--part0-layers N or --long-ctx-part0-layers N)", flush=True)
    if args.compare:
        compare(args.compare[0], args.compare[1])
        return
    if args.kld_compare:
        kld_compare(args.kld_compare[0], args.kld_compare[1])
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
        devices = resolve_devices(args.part0_device, args.part1_device)
        plan = SplitPlan(
            devices, endcaps = args.endcaps == "on", part0_layers = args.part0_layers,
            part0_reserve_gb = 0.0 if args.no_budget else args.part0_reserve_gb,
            part1_reserve_gb = args.part1_reserve_gb)
        plan.add("text", model, config = config, cache = cache, load_order = 0)
        plan.build()
        text_map = plan.map("text")
        if args.part0_layers > 0 and (text_map is None or not text_map.enabled):
            raise SystemExit("split map not enabled (no part0 device)")
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

    devs = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    after_load = {str(d): round(torch.cuda.memory_allocated(d) / 2**30, 2) for d in devs}
    device_info = {}
    for i, d in enumerate(devs):
        p = torch.cuda.get_device_properties(i)
        device_info[str(d)] = {
            "name": p.name, "arch": getattr(p, "gcnArchName", ""),
            "total_gib": round(p.total_memory / 2**30, 2),
        }

    GM.MTP_FUSE_CATCHUP = 2
    gen = Generator(model = model, cache = cache, tokenizer = tokenizer,
                    draft_model = draft, draft_cache = draft_cache,
                    num_draft_tokens = args.num_draft or None,
                    record_draft_stats = True)

    result = {
        "mode": "baseline" if args.baseline else "split",
        "model": args.model,
        "part0_layers": None if args.baseline else args.part0_layers,
        "part0_device": None if args.baseline else args.part0_device,
        "part1_device": None if args.baseline else args.part1_device,
        "part0_reserve_gb": None if args.baseline else args.part0_reserve_gb,
        "endcaps": None if args.baseline else args.endcaps,
        "ctx": args.max_ctx,
        "cache_tokens": args.max_ctx + 4096,
        "num_draft": args.num_draft,
        "graphs_off": args.no_graphs or not args.baseline,
        "load_s": load_s,
        "devices": device_info,
        "after_load_gib": after_load,
        "prompts": [],
        "perf": [],
        "vision": None,
    }

    if args.vision_image and vision is None:
        raise SystemExit("--vision-image was requested but the vision tower did not load")
    if vision is not None and args.vision_image:
        from PIL import Image
        image = Image.open(args.vision_image).convert("RGB")
        times = []
        emb = None
        for _ in range(2):
            t0 = time.perf_counter()
            emb = vision.get_image_embeddings(tokenizer = tokenizer, image = image)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        result["vision"] = {
            "image": args.vision_image, "mm_length": int(emb.mm_length),
            "embedding_shape": list(emb.embeddings.shape),
            "embedding_device": str(emb.embeddings.device),
            "first_s": times[0], "second_s": times[1],
        }
        print(f"[eval] vision {args.vision_image}: {emb.mm_length} tokens, "
              f"first {times[0]:.2f} s, second {times[1]:.2f} s, "
              f"embeddings {tuple(emb.embeddings.shape)} on {emb.embeddings.device}", flush = True)

    for i, prompt in enumerate(PROMPTS):
        ids = ids_of(prompt, tokenizer)
        r = run_job(gen, Job, ids, args.tokens)
        r["prompt"] = prompt
        r["prompt_tokens"] = int(ids.numel())
        r["task_check"] = score_task(i, r["text"])
        result["prompts"].append(r)
        print(f"[eval] {prompt[:44]!r}: {r['tokens']} tokens"
              + (f", ttft {r['ttft_s']:.2f} s" if r["ttft_s"] else "")
              + (f", decode {r['tokens'] / r['decode_s']:.2f} t/s"
                 if r["decode_s"] and r["tokens"] > 1 else "")
              + (f", task {'PASS' if r['task_check']['passed'] else 'FAIL'}"
                 if r["task_check"]["expected"] else ""), flush = True)
    checks = [p["task_check"] for p in result["prompts"]]
    result["tasks"] = {"passed": sum(t["passed"] for t in checks), "total": len(checks)}
    print(f"[eval] tasks {result['tasks']['passed']}/{result['tasks']['total']} passed", flush = True)

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

    if args.bench_depth > 0:
        depth = min(args.bench_depth, args.max_ctx - 1024)
        if args.bench_prompt == "summary" or args.bench_acceptance:
            instructions = SUMMARY_INSTRUCTIONS if args.bench_acceptance else SUMMARY_INSTRUCTIONS[:1]
            runs = []
            for instruction in instructions:
                ids = build_context_prompt(depth, tokenizer, instruction)
                r = run_job(gen, Job, ids, args.perf_tokens)
                tps = (r["tokens"] - 1) / r["decode_s"] if r["decode_s"] and r["tokens"] > 1 else None
                runs.append({
                    "instruction": instruction,
                    "depth": int(ids.numel()), "ids": r["ids"],
                    "ttft_s": r["ttft_s"], "tokens": r["tokens"],
                    "decode_s": r["decode_s"], "decode_tps": tps,
                    "accepted": r["accepted"], "rejected": r["rejected"],
                    "acceptance": round(100 * r["accepted"] / (r["accepted"] + r["rejected"]), 2)
                                  if r["accepted"] + r["rejected"] else None,
                    "text": r["text"],
                })
                if r["ttft_s"] is None or tps is None:
                    print(f"[eval] long summary depth {ids.numel()}: incomplete run "
                          f"({r['tokens']} tokens, ttft={r['ttft_s']}, decode_tps={tps}, "
                          f"accept {r['accepted']}/{r['rejected']})", flush = True)
                else:
                    print(f"[eval] long summary depth {ids.numel()}: "
                          f"ttft {r['ttft_s']:.1f} s, decode {tps:.2f} t/s, "
                          f"accept {r['accepted']}/{r['rejected']}", flush = True)
            acc = sum(x["accepted"] for x in runs)
            rej = sum(x["rejected"] for x in runs)
            base = runs[0]
            result["long"] = {
                "prompt": "summary-set" if args.bench_acceptance else "summary",
                "depth": base["depth"], "ids": base["ids"],
                "ttft_s": base["ttft_s"], "tokens": base["tokens"],
                "decode_s": base["decode_s"], "decode_tps": base["decode_tps"],
                "accepted": acc, "rejected": rej,
                "acceptance": round(100 * acc / (acc + rej), 2) if acc + rej else None,
                "acceptance_ci95": acceptance_ci95(acc, rej),
                "runs": runs, "text": base["text"],
            }
            if acc + rej:
                lo, hi = result["long"]["acceptance_ci95"]
                print(f"[eval] acceptance {acc}/{acc + rej} = {result['long']['acceptance']:.2f}% "
                      f"(95% CI {lo:.2f}-{hi:.2f}%, {acc + rej} drafted tokens)", flush = True)
        else:
            ids, _ = build_needle(depth, tokenizer)
            r = run_job(gen, Job, ids, args.perf_tokens)
            tps = (r["tokens"] - 1) / r["decode_s"] if r["decode_s"] and r["tokens"] > 1 else None
            acc, rej = r["accepted"], r["rejected"]
            result["long"] = {
                "prompt": "passkey",
                "depth": int(ids.numel()), "ids": r["ids"],
                "ttft_s": r["ttft_s"], "tokens": r["tokens"],
                "decode_s": r["decode_s"], "decode_tps": tps,
                "accepted": acc, "rejected": rej,
                "acceptance": round(100 * acc / (acc + rej), 2) if acc + rej else None,
                "acceptance_ci95": acceptance_ci95(acc, rej),
                "text": r["text"],
            }
            if r["ttft_s"] is None or tps is None:
                print(f"[eval] long depth {ids.numel()}: incomplete run "
                      f"({r['tokens']} tokens, ttft={r['ttft_s']}, decode_tps={tps}, "
                      f"accept {acc}/{rej})", flush = True)
            else:
                print(f"[eval] long depth {ids.numel()}: ttft {r['ttft_s']:.1f} s, "
                      f"prefill {ids.numel() / r['ttft_s']:.2f} t/s, decode {tps:.2f} t/s, "
                      f"accept {acc}/{rej}", flush = True)

    if args.kld_out:
        import math
        need = args.kld_slices * args.kld_len + 4096
        all_ids = corpus_ids(args.kld_corpus, tokenizer, need)
        scored = kld_pos = 0
        nll_sum = 0.0
        entries = []
        start = min(4096, max(0, all_ids.numel() - args.kld_slices * (args.kld_len + 37)))
        for s in range(args.kld_slices):
            off = start + s * (args.kld_len + 37)
            ids = all_ids[off:off + args.kld_len].clone()
            lp, nll, n = run_kld(model, ids)
            entries.append({"ids": ids.tolist(), "logprobs": lp, "nll": nll, "scored": n})
            nll_sum += nll
            scored += n
            print(f"[kld] slice {s}: {n} positions, NLL mean {nll / max(1, n):.4f}, "
                  f"PPL {math.exp(nll / max(1, n)):.4f}", flush = True)
        payload = {
            "model": args.model, "mode": result["mode"],
            "corpus": args.kld_corpus, "slices": args.kld_slices, "len": args.kld_len,
            "ids": [e["ids"] for e in entries], "nll": [e["nll"] for e in entries],
            "scored": [e["scored"] for e in entries],
            "logprobs": [e["logprobs"] for e in entries],
        }
        torch.save(payload, args.kld_out)
        result["kld"] = {
            "corpus": args.kld_corpus, "slices": args.kld_slices, "len": args.kld_len,
            "out": args.kld_out, "scored": scored,
            "ppl": math.exp(nll_sum / scored) if scored else None,
        }
        print(f"[kld] wrote {args.kld_out}: PPL {result['kld']['ppl']:.4f} "
              f"({scored} positions)", flush = True)

    for d in devs:
        result["devices"][str(d)].update({
            "allocated_gib": round(torch.cuda.memory_allocated(d) / 2**30, 2),
            "peak_gib": round(torch.cuda.max_memory_allocated(d) / 2**30, 2),
            "reserved_gib": round(torch.cuda.memory_reserved(d) / 2**30, 2),
        })

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
    for label, run in (("base", base), ("split", split)):
        if "tasks" in run:
            t = run["tasks"]
            print(f"[compare] tasks {label}: {t['passed']}/{t['total']} passed")
    if base.get("perf") and split.get("perf"):
        for i, (a, b) in enumerate(zip(base["perf"], split["perf"])):
            if a.get("decode_tps") and b.get("decode_tps"):
                print(f"[compare] perf {i}: baseline {a['decode_tps']:.2f} t/s, "
                      f"split {b['decode_tps']:.2f} t/s "
                      f"({b['decode_tps'] / a['decode_tps']:.3f}x)")
    if "needle" in split:
        print(f"[compare] needle: {'found' if split['needle']['found'] else 'MISSING'}")
    if base.get("long") and split.get("long"):
        a, b = base["long"], split["long"]
        runs_a, runs_b = a.get("runs"), b.get("runs")
        if runs_a is not None or runs_b is not None:
            if not runs_a or not runs_b:
                print("[compare] long runs: cannot compare (one side has no per-run ids)")
            else:
                if len(runs_a) != len(runs_b):
                    print(f"[compare] long runs: {len(runs_a)} vs {len(runs_b)} runs")
                differing = [
                    i for i, (ra, rb) in enumerate(zip(runs_a, runs_b))
                    if ra.get("ids") != rb.get("ids")
                ]
                if differing:
                    i = differing[0]
                    print(f"[compare] long run {i} DIVERGED: {runs_a[i].get('instruction')!r}")
                else:
                    print("[compare] long runs: IDENTICAL")
        elif a.get("ids") is not None and b.get("ids") is not None:
            same_long = a["ids"] == b["ids"]
            print(f"[compare] long prompt ids: {'IDENTICAL' if same_long else 'DIVERGED'}")
        if a.get("acceptance") is not None and b.get("acceptance") is not None:
            a_n = a["accepted"] + a["rejected"]
            b_n = b["accepted"] + b["rejected"]
            if a.get("prompt") == "passkey" or b.get("prompt") == "passkey":
                print(f"[compare] acceptance: skipped (passkey sample {a_n}/{b_n} drafted tokens; "
                      "use --bench-prompt summary)")
            elif min(a_n, b_n) < MIN_ACCEPTANCE_SAMPLES:
                print(f"[compare] acceptance: sample too small ({a_n}/{b_n} drafted tokens; "
                      f"need >= {MIN_ACCEPTANCE_SAMPLES})")
            else:
                print(f"[compare] acceptance: first {a['acceptance']:.2f}% "
                      f"({a['accepted']}/{a_n}), second {b['acceptance']:.2f}% "
                      f"({b['accepted']}/{b_n})")


if __name__ == "__main__":
    main()
