#!/usr/bin/env python
"""Follow-up: does real bsz=1 decode actually call exl3_dec_gemv_multi (qkv),
exl3_dec_gemv_strided (o_proj) and exl3_dec_gemv (lm_head) -- the kernels gemv_r_probe.py's
reference used? Spy on ext.* during one real generation (no draft model) and report call
counts vs the number of decode steps, so a per-step rate confirms which kernel actually fires.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.environ.get("EXL3_REPO", "~/kyojin"))
import torch  # noqa: E402
from exllamav3 import Cache, Generator, GreedySampler, Job, Config, Model  # noqa: E402
import exllamav3.modules.attn as attn_mod  # noqa: E402

DEV = "cuda:0"
ext = attn_mod.ext


def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("-m", "--model_dir", default=os.path.expanduser("~/models/mimo26-exl3"))
    ap.add_argument("--new-tokens", type=int, default=16)
    ap.add_argument("-dm", "--draft_model_dir", default=None)
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    config = Config.from_directory(args.model_dir)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=4096)
    model.load(progressbar=False)

    counts = {}

    def spy(name):
        real = getattr(ext, name)

        def wrapped(*a, **k):
            counts[name] = counts.get(name, 0) + 1
            return real(*a, **k)
        setattr(ext, name, wrapped)

    for name in ["exl3_dec_gemv", "exl3_dec_gemv_multi", "exl3_dec_gemv_strided",
                 "exl3_dec_gemv_r", "exl3_dec_gemv_r_multi",
                 "exl3_dec_moe", "exl3_dec_moe_union", "exl3_moe_gfx12_k3",
                 "exl3_dec_router_norm", "exl3_dec_router", "exl3_mgemm"]:
        if hasattr(ext, name):
            spy(name)

    draft_model = draft_cache = None
    if args.draft_model_dir:
        from exllamav3 import Config as _Cfg
        draft_config = _Cfg.from_directory(args.draft_model_dir)
        draft_model = Model.from_config(draft_config)
        draft_cache = Cache(draft_model, max_num_tokens=4096)
        draft_model.load(progressbar=False)

    from exllamav3.tokenizer import Tokenizer
    tok = Tokenizer.from_config(config)
    ids = tok.encode("The capital of France is")
    job = Job(input_ids=ids, max_new_tokens=args.new_tokens,
              stop_conditions=model.config.eos_token_id_list, sampler=GreedySampler())
    gen = Generator(model=model, cache=cache, tokenizer=tok,
                     draft_model=draft_model, draft_cache=draft_cache)
    gen.enqueue(job)
    n_steps = 0
    while gen.num_remaining_jobs():
        for res in gen.iterate():
            if res.get("error"):
                raise RuntimeError(res["error"])
            if res.get("text"):
                n_steps += 1

    print(f"decode steps (new tokens generated): {n_steps}")
    print("ext.* call counts over the whole run (prefill + n_steps decode steps):")
    for name, c in sorted(counts.items()):
        per_step = c / max(n_steps, 1)
        print(f"  {name}: {c}  ({per_step:.2f}/decode-step)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
