#!/usr/bin/env python
"""Free-run coherence gate for a dense EXL3 model.

Not CPU CI: this loads a real model and generates on the visible device. It is the
manual whole-model integration gate for the ROCm/gfx12 decode path: eight prompts
covering factual, arithmetic, code, instruction-following and long-form output, run
greedily and with the engine's default sampler. For each run it records the stop
reason, the loop-guard verdicts used by tools/qwen/serve.py, the worst 20-gram
repetition count and whether EOS was reached.

  python tools/strix_halo/coherence_suite.py -m <model>
  HIP_VISIBLE_DEVICES=1 python tools/strix_halo/coherence_suite.py -m <model> -n 512

The model must fit the visible device (use the layer-split sibling for larger packs).

"""
import argparse
import collections
import os
import sys

p = argparse.ArgumentParser()
p.add_argument("-m", "--model", default=os.environ.get("EXL3_TEST_MODEL", ""))
p.add_argument("-n", "--max-new-tokens", type=int, default=256)
p.add_argument("-c", "--cache-tokens", type=int, default=4096)
p.add_argument("--modes", default="greedy,default")
args = p.parse_args()
if not args.model:
    raise SystemExit("--model or EXL3_TEST_MODEL is required")

import torch
from exllamav3 import Cache, Config, Generator, Job, Model, Tokenizer
from exllamav3.generator.sampler import DefaultSampler, GreedySampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loopguard import answer_text_loop, text_loop  # noqa: E402

PROMPTS = [
    ("factual", "What is the capital of France?", ["paris"]),
    ("factual", "Who wrote the novel Pride and Prejudice?", ["austen"]),
    ("arithmetic", "A shop sells pens for 3 dollars each. How much do 7 pens cost? Answer with the number.", ["21"]),
    ("code", "Write a Python function called is_even that returns True for even integers. Reply with code only.", ["def", "is_even"]),
    ("instruction", "List exactly three primary colours of light, one per line, and nothing else.", ["red", "blue", "green"]),
    ("instruction", "Reply with the single word: BANANA", ["banana"]),
    ("long-form", "Explain in two short paragraphs why the sky appears blue during the day.", []),
    ("long-form", "Describe the water cycle in three sentences.", []),
]


def _worst_ngram_repeat(text, n=20):
    words = text.split()
    if len(words) < n:
        return 0
    counts = collections.Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
    return max(counts.values())


def _render(tokenizer, prompt):
    """Render the pack's own Jinja chat template (no transformers dependency)."""
    from jinja2 import BaseLoader
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    path = os.path.join(tokenizer.config.directory, "chat_template.jinja")
    if not os.path.isfile(path):
        return prompt
    env = ImmutableSandboxedEnvironment(loader=BaseLoader(), trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(ValueError(msg))
    template = env.from_string(open(path, encoding="utf-8").read())
    messages = [{"role": "user", "content": prompt}]
    try:
        return template.render(messages=messages, add_generation_prompt=True,
                               enable_thinking=False)
    except Exception:
        return template.render(messages=messages, add_generation_prompt=True)


def _tokenize(tokenizer, prompt):
    ids = tokenizer.encode(_render(tokenizer, prompt), add_bos=True,
                           encode_special_tokens=True)
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    return ids.contiguous()


def run(gen, model, tokenizer, prompt, mode, max_new_tokens):
    sampler = GreedySampler() if mode == "greedy" else DefaultSampler()
    job = Job(input_ids=_tokenize(tokenizer, prompt),
              max_new_tokens=max_new_tokens,
              stop_conditions=list(model.config.eos_token_id_list),
              return_logits=False,
              sampler=sampler)
    gen.enqueue(job)
    text = ""
    eos_reason = None
    while gen.num_remaining_jobs():
        for res in gen.iterate():
            text += res.get("text", "")
            if res.get("eos"):
                eos_reason = res.get("eos_reason")
    return text, eos_reason


def main():
    pr = torch.cuda.get_device_properties(0)
    print(f"gpu {pr.name} / {pr.gcnArchName} / {pr.total_memory / 2**30:.1f} GiB", flush=True)
    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    tokenizer = Tokenizer.from_config(config)
    cache = Cache(model, max_num_tokens=args.cache_tokens, max_batch_size=1)
    model.load(progressbar=False)
    gen = Generator(model=model, cache=cache, tokenizer=tokenizer)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    failures = []
    try:
        for mode in modes:
            print(f"\n=== sampling: {mode} ===", flush=True)
            for kind, prompt, keywords in PROMPTS:
                text, eos_reason = run(gen, model, tokenizer, prompt, mode, args.max_new_tokens)
                low = text.lower()
                missing = [k for k in keywords if k not in low]
                ngram = _worst_ngram_repeat(text)
                guard = answer_text_loop(text) or text_loop(text)
                eos = eos_reason in ("stop_token", "stop_string")
                ok = not missing and not guard and ngram <= 2 and eos
                if not ok:
                    failures.append((mode, kind, prompt, missing, guard, ngram, eos_reason))
                print(f"[{'PASS' if ok else 'FAIL'}] {mode:7} {kind:12} "
                      f"eos={eos_reason!s:14} guard={int(guard)} max20={ngram} "
                      f"missing={missing} chars={len(text)}", flush=True)
                print(f"  {text[:220]!r}", flush=True)
    finally:
        model.unload()

    print(f"\n{len(failures)} failure(s)")
    for f in failures:
        print("  ", f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
