#!/usr/bin/env python
"""Why do drafted greedy outputs differ from unassisted ones? Separate numerics from rollback.

One load, one draft length. For each prompt:
  A        unassisted greedy decode (R=1 steps)                                  -> reference tokens
  oracle   the DFlash generator, but the drafter's block is replaced by the reference tokens
           themselves: every draft is accepted, every verify runs at R = 1 + ndt, no rollback
  wrong    same, but the last `--wrong` drafted tokens are deliberately wrong: every step rejects
           them, so the sliding-window ring and the KV cache rewind after every verify

Reading: oracle == A and wrong != A  -> rollback bug;  oracle != A -> R>1 numerics differ from R=1.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

sys.path.insert(0, os.environ.get("EXL3_REPO", "~/kyojin/dflash"))
import torch  # noqa: E402
from exllamav3 import Cache, CacheLayer_fp16, Generator, model_init  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "dfb", os.path.join(os.path.dirname(os.path.abspath(__file__)), "dflash-bench.py"))
dfb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dfb)


def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev = False)
    model_init.add_args(ap, cache = True, add_draft_model_args = True)
    ap.add_argument("--kinds", default = "prose,chat")
    ap.add_argument("--reps", type = int, default = 2)
    ap.add_argument("--new-tokens", type = int, default = 96)
    ap.add_argument("--wrong", type = int, default = 2)
    ap.add_argument("--out", default = "/tmp/lossless-probe.json")
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    model, config, cache, tokenizer, draft_model, draft_config, draft_cache = model_init.init(args)
    ndt = int(args.num_draft_tokens or 4)
    cs = args.cache_size or 2304
    c_base = Cache(model, max_num_tokens = cs, layer_type = CacheLayer_fp16, max_history = max(4, ndt))
    c_draft = Cache(model, max_num_tokens = cs, layer_type = CacheLayer_fp16, max_history = max(4, ndt))
    model.load()
    print(f"loaded ndt={ndt} gtt {dfb.gtt_gib():.1f} GiB avail {dfb.mem_avail_gib():.1f} GiB", flush = True)

    base_gen = Generator(model = model, cache = c_base, tokenizer = tokenizer)
    gen = Generator(model = model, cache = c_draft, tokenizer = tokenizer, draft_model = draft_model,
                    draft_cache = draft_cache, num_draft_tokens = ndt, record_draft_stats = True)

    state = {"ref": None, "wrong": 0}

    def fake_draft(self, results):
        jobs = [j for j in self.active_jobs if j.is_prefill_done()]
        if not jobs:
            return None
        seq = jobs[0].sequences[0]
        t = seq.sequence_ids.torch()
        g = (t.shape[-1]) - len(seq.input_ids)          # tokens generated so far
        ref = state["ref"]
        d = [ref[min(g + i, len(ref) - 1)] for i in range(ndt)]
        for i in range(ndt - state["wrong"], ndt):
            d[i] = (d[i] + 1) % config.vocab_size       # guaranteed wrong
        self.draft_ids_pinned[0, :ndt].copy_(torch.tensor(d, dtype = self.draft_ids_pinned.dtype))
        return self.draft_ids_pinned[:, :ndt]

    Generator.iterate_draftmodel_dflash_gen = fake_draft

    corpora = {k: dfb.load_corpus(v) for k, v in dfb.CORPUS.items()}
    out = {"ndt": ndt, "wrong": args.wrong, "cases": []}
    for kind in args.kinds.split(","):
        for rep in range(args.reps):
            ids = dfb.prompt_ids(tokenizer, corpora, kind, dfb.PROMPT_TOKENS, rep * dfb.PROMPT_TOKENS)
            a = dfb.run_job(base_gen, ids, args.new_tokens, want_logits = True)
            state["ref"] = a["ids"]
            case = {"kind": kind, "rep": rep}
            for mode, nw in (("oracle", 0), ("wrong", args.wrong)):
                state["wrong"] = nw
                b = dfb.run_job(gen, ids, args.new_tokens)
                dv = dfb.divergence(a, b)
                case[mode] = dict(dv, steps = b["steps"], accepted = b["accepted"], window = b["window"])
                print(f"{kind} rep{rep} {mode:6s} first_diff={dv['first_diff']} n_diff={dv['n_diff']} "
                      f"gap={dv.get('base_gap')} steps={b['steps']} acc={b['accepted']}/{b['window']}",
                      flush = True)
            out["cases"].append(case)
    out["gtt_end_gib"] = dfb.gtt_gib()
    with open(args.out, "w") as f:
        json.dump(out, f, indent = 1)
    print(f"wrote {args.out}  gtt {dfb.gtt_gib():.1f} GiB", flush = True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
