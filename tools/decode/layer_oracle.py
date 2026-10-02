#!/usr/bin/env python
"""Per-layer oracle (advisor-directed): find the first TransformerBlock whose row-0 output
differs bitwise between a plain batch-1 decode step and row 0 of a perfect-draft verify round
at the same absolute position. Monkeypatches TransformerBlock.forward to clone x[:, 0:1] (the
residual stream row-0 output) after every q_len<=8 call, tagged by a run label, then diffs the
two captures layer by layer (int32 view, NaN/-inf safe).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.environ.get("EXL3_REPO", "~/kyojin"))
import torch  # noqa: E402
from exllamav3 import Cache, Config, Generator, GreedySampler, Job, Model, Tokenizer  # noqa: E402
import exllamav3.modules.transformer as transformer_mod  # noqa: E402

PROMPT = ("Write the opening three paragraphs of a short story about a lighthouse "
          "keeper who discovers a message in a bottle, in a literary, descriptive style.")

CAPTURES = {}   # label -> {layer_idx: tensor}
CURRENT_LABEL = [None]
CAPTURE_COUNT = [0]  # only capture the FIRST qualifying call per layer per label


def install_hook():
    real_forward = transformer_mod.TransformerBlock.forward

    def hooked_forward(self, x, params, out_dtype=None):
        y = real_forward(self, x, params, out_dtype)
        label = CURRENT_LABEL[0]
        if label is not None and y.dim() == 3 and 1 <= y.shape[0] * y.shape[1] <= 16:
            # Key by self.key, not layer_idx: the draft model is ALSO built from
            # TransformerBlock instances with layer_idx 0..4, colliding with the target
            # model's layer_idx 0..4 if keyed numerically. self.key disambiguates
            # ("model.layers.N" for MiMo's target vs "layers.N" for the dflash draft).
            d = CAPTURES.setdefault(label, {})
            if self.key not in d:
                d[self.key] = y[:1, :1].detach().clone().cpu()
        return y

    transformer_mod.TransformerBlock.forward = hooked_forward


def run_one_step(generator, tok, prompt, label, max_new_tokens=1):
    ids = tok.encode(prompt)
    job = Job(input_ids=ids, max_new_tokens=max_new_tokens, stop_conditions=[], sampler=GreedySampler())
    generator.enqueue(job)
    CURRENT_LABEL[0] = label
    while generator.num_remaining_jobs():
        for res in generator.iterate():
            if res.get("error"):
                raise RuntimeError(res["error"])
    CURRENT_LABEL[0] = None


def main():
    torch.set_grad_enabled(False)
    install_hook()
    m = os.path.expanduser("~/models/mimo26-exl3")
    dm = os.path.expanduser("~/models/mimo26-exl3/dflash")
    config = Config.from_directory(m)
    model = Model.from_config(config)
    cache_plain = Cache(model, max_num_tokens=4096)
    cache_b = Cache(model, max_num_tokens=4096)
    model.load(progressbar=False)
    tok = Tokenizer.from_config(config)
    draft_config = Config.from_directory(dm)
    draft_model = Model.from_config(draft_config)
    draft_cache = Cache(draft_model, max_num_tokens=4096)
    draft_model.load(progressbar=False)

    # Plain: prefill + first single-token decode step (q_len=1). The hook captures every
    # layer's row-0 output at that first decode step (prefill itself is q_len=28, skipped by
    # the <=16 gate).
    gen_plain = Generator(model=model, cache=cache_plain, tokenizer=tok)
    run_one_step(gen_plain, tok, PROMPT, "plain")

    # Arm B analog: prefill + first verify round with a REAL (not perfect) draft is fine here --
    # we only need row 0 of the round, and row 0 always matches ground truth token-for-token in
    # every trace so far (the first mismatch across all repros was at i>=1), so whatever the
    # real drafter proposes, row 0's INPUT context is identical to plain's (both are just the
    # prompt). If row 0 itself differs, that's the answer regardless of what row 1+ do.
    gen_b = Generator(model=model, cache=cache_b, tokenizer=tok,
                       draft_model=draft_model, draft_cache=draft_cache)
    run_one_step(gen_b, tok, PROMPT, "draft", max_new_tokens=8)

    la = CAPTURES.get("plain", {})
    lb = CAPTURES.get("draft", {})
    # Only compare keys present in both runs. Keys unique to "draft" are the draft model's own
    # TransformerBlocks (self.key = "layers.N", no "model." prefix) -- not a target-model layer,
    # nothing to compare against in "plain".
    shared = sorted(set(la) & set(lb), key=lambda k: int(k.rsplit(".", 1)[-1]))
    print(f"plain-only keys: {sorted(set(la) - set(lb))}")
    print(f"draft-only keys (expected: the draft model's own blocks): {sorted(set(lb) - set(la))}")
    print(f"compared keys: {shared}")
    first_diff = None
    for li in shared:
        a, b = la[li].float(), lb[li].float()
        if a.shape != b.shape:
            print(f"layer {li}: SHAPE MISMATCH {a.shape} vs {b.shape}")
            continue
        bitdiff = (a.view(torch.int32) != b.view(torch.int32))
        n = bitdiff.sum().item()
        if n == 0:
            print(f"layer {li}: bitwise EXACT")
        else:
            finite = ~(torch.isnan(a) | torch.isnan(b) | torch.isinf(a) | torch.isinf(b))
            maxd = (a[finite] - b[finite]).abs().max().item() if finite.any() else float("nan")
            print(f"layer {li}: {n} elements differ, max|d|={maxd:.6f}")
            if first_diff is None:
                first_diff = li
    print(f"\nFIRST DIFFERING LAYER: {first_diff}")


if __name__ == "__main__":
    main()
