#!/usr/bin/env python3
"""Plain greedy reference for the msrv parity check: the same prompt the server gets, greedy, but
through the plain dflash-bench.py path (no drafter, no SpecGate), so served DFlash+SpecGate output
can be compared token by token.

  python tools/mimo/plain_greedy.py --out scratch/msrv/plain_code.json --kind code
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
os.environ["EXL3_SPEC_GATE"] = "0"      # plain path: the gate only exists with a drafter

import torch  # noqa: E402

from exllamav3 import Generator, GreedySampler, Job, model_init  # noqa: E402

from accept import chat_prompt, code_prompt, load_corpus, CODE_CORPUS  # noqa: E402
from serve import render_prompt  # noqa: E402

N_PARITY = 64


def main() -> int:
    default_model = os.path.expanduser("~/models/mimo26-exl3")
    ap = argparse.ArgumentParser(allow_abbrev=False)
    model_init.add_args(ap, cache=True)          # provides -m/--model_dir (required)
    ap.add_argument("--kind", default="code", choices=["code", "chat"])
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--tokens", type=int, default=N_PARITY)
    ap.add_argument("--out", default="scratch/msrv/plain_code.json")
    # -cs is fixed at 2304 like dflash-bench.py; -m defaults to the local MiMo pack.
    argv = sys.argv[1:]
    if not any(a == "-m" or a.startswith("--model_dir") for a in argv):
        argv = ["-m", default_model] + argv
    if not any(a == "-cs" or a.startswith("--cache_size") for a in argv):
        argv = ["-cs", "2304"] + argv
    args = ap.parse_args(argv)
    model_path = args.model_dir
    torch.set_grad_enabled(False)
    model, config, cache, tokenizer = model_init.init(args)
    model.load()
    generator = Generator(model=model, cache=cache, tokenizer=tokenizer)

    template = open(model_path + "/chat_template.jinja", encoding="utf-8").read()
    if args.kind == "chat":
        messages = chat_prompt(load_corpus(os.path.expanduser("~/bench/ppl/wiki.test.raw")),
                               tokenizer, args.offset)
    else:
        messages = [{"role": "user", "content": code_prompt(load_corpus(CODE_CORPUS), tokenizer,
                                                           args.offset)}]
    prompt = render_prompt(template, messages, None)
    encoded = tokenizer.encode(prompt, encode_special_tokens=True)
    ids = encoded[0] if isinstance(encoded, tuple) else encoded   # encode() may return (ids, mask)
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)      # Job wants (batch, seq), like dflash-bench.py's prompt_ids
    job = Job(input_ids=ids, max_new_tokens=args.tokens, sampler=GreedySampler(),
              stop_conditions=model.config.eos_token_id_list)
    generator.enqueue(job)
    t0 = time.perf_counter()
    while generator.num_remaining_jobs():
        list(generator.iterate())
    total = time.perf_counter() - t0
    seq = job.sequences[0]
    t = seq.sequence_ids.torch()
    all_ids = (t[0] if t.dim() > 1 else t).tolist()
    out_ids = all_ids[len(seq.input_ids):]
    decoded = tokenizer.decode(torch.tensor(out_ids, dtype=torch.long))
    text = decoded[0] if isinstance(decoded, tuple) else decoded
    out = dict(kind=args.kind, offset=args.offset, prompt_len=int(len(seq.input_ids)),
               ids=out_ids, text=text, seconds=total, tps=len(out_ids) / total)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"plain greedy {args.kind}: {len(out_ids)} tok in {total:.2f}s "
          f"({out['tps']:.2f} tok/s) prompt_len={out['prompt_len']} -> {args.out}")
    print("first ids:", out_ids[:16])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
