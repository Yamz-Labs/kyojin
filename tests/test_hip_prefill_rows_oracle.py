"""Opt-in full Qwen3.8-Flash-Next oracle for the gfx12 prefill route at 1024/2048 rows.

Validates the Stage-A change: a prefill chunk with >512 rows must stay on the fast
route (exl3_moe_gfx12_k3_prefill) and produce the same output as the generic
fallback. Intentionally fails before MOE_PREFILL_MAX_ROWS/_HIP_PREFILL_MAX_ROWS are
raised to 2048 (the binding TORCH_CHECK rejects R>512), and passes after.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from exllamav3 import Cache, Config, Generator, GreedySampler, Job, Model, Tokenizer
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.block_sparse_mlp import BlockSparseMLP

MODEL = Path(os.environ.get(
    "EXL3_FLASH_TEST_MODEL", "~/Models/Qwen3.8-Flash-Next-exl3"
)).expanduser()
RUN_ORACLE = os.environ.get("EXL3_RUN_PREFILL_ROWS_ORACLE") == "1"
BASE_PROMPT = "Q: The capital of France is\nA:"
STEP_OUT = 4
# One full 2048-row chunk plus a tail >= 1024 rows forces both new row sizes onto the route.
PROMPT_TOKENS = 2300


def _gfx12_devices():
    if not (torch.version.hip and torch.cuda.is_available()):
        return []
    return [
        index for index in range(torch.cuda.device_count())
        if getattr(torch.cuda.get_device_properties(index), "gcnArchName", "").split(":", 1)[0]
        in ("gfx1200", "gfx1201")
    ]


def _pad_ids(tokenizer, length):
    base = tokenizer.encode(BASE_PROMPT, add_bos=True)
    base = base.reshape(-1).long()  # encode() already returns a tensor; ensure 1-D
    reps = -(-length // len(base))
    return base.repeat(reps)[:length]


def _generate(model, cache, tokenizer, ids, forced=None):
    gen = Generator(model=model, cache=cache, tokenizer=tokenizer)
    job = Job(
        input_ids=ids,
        max_new_tokens=STEP_OUT + 1,
        stop_conditions=[],
        return_logits=True,
        sampler=GreedySampler(),
    )
    gen.enqueue(job)
    if forced is not None:
        job.constrain_output_now(forced.contiguous())
    token_chunks, logit_chunks = [], []
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if "token_ids" in r:
                token_chunks.append(r["token_ids"].cpu())
                logit_chunks.append(r["logits"].cpu())
    tokens = torch.cat(token_chunks, dim=-1)
    logits = torch.cat(logit_chunks, dim=1)
    return tokens, logits


@pytest.mark.skipif(not RUN_ORACLE, reason="set EXL3_RUN_PREFILL_ROWS_ORACLE=1")
@torch.inference_mode()
def test_gfx12_prefill_rows_1024_2048_oracle(monkeypatch):
    devices = _gfx12_devices()
    if len(devices) < 2:
        pytest.skip(f"prefill-rows oracle requires two gfx12 devices, found {devices}")
    if not MODEL.is_dir():
        pytest.skip(f"Flash model not found: {MODEL}")

    model = Model.from_config(Config.from_directory(str(MODEL)))
    oracle_cache = Cache(model, max_num_tokens=4096, max_batch_size=1)
    budgets = [0.0] * torch.cuda.device_count()
    budgets[devices[0]] = float(os.environ.get("EXL3_FLASH_ORACLE_GPU0_GB", "25"))
    budgets[devices[1]] = float(os.environ.get("EXL3_FLASH_ORACLE_GPU1_GB", "31"))
    tokenizer = Tokenizer.from_config(model.config)
    ids = _pad_ids(tokenizer, PROMPT_TOKENS)

    real_prefill = ext.exl3_moe_gfx12_k3_prefill
    mode = {"name": None}
    prefill_rows = {"fallback": [], "fast": []}
    max_seen = {"rows": 0, "device": None}

    def prefill_spy(*args, **kwargs):
        r = args[0].shape[0]
        if mode["name"] in prefill_rows:
            prefill_rows[mode["name"]].append(r)
        if mode["name"] == "fast" and r > max_seen["rows"]:
            max_seen["rows"] = r
            max_seen["device"] = args[0].device.index
        return real_prefill(*args, **kwargs)

    monkeypatch.setenv("EXL3_GEMV", "2")
    try:
        model.load(use_per_device=budgets, max_chunk_size=2048, max_batch_size=1)
        ext.exl3_moe_gfx12_k3_prefill = prefill_spy

        # Reference: generic fallback (fast route disabled).
        mode["name"] = "fallback"
        monkeypatch.setenv("EXL3_HIP_GROUPED_MOE_PREFILL", "0")
        ftokens, flogits = _generate(model, oracle_cache, tokenizer, ids)

        # Fast route at 2048-row prefill chunks.
        mode["name"] = "fast"
        monkeypatch.setenv("EXL3_HIP_GROUPED_MOE_PREFILL", "1")
        ftokens.set_()  # ensure reference tokens drive nothing here (no forcing)
        ttokens, tlogits = _generate(model, oracle_cache, tokenizer, ids, forced=ftokens)

        # The fast route must have handled a full 2048-row prefill chunk.
        assert prefill_rows["fast"], "fast route never invoked"
        assert max_seen["rows"] >= 2048, \
            f"expected a >=2048-row fast prefill call, max was {max_seen['rows']}"
        # And it must not have emitted any >2048 rows (envelope cap).
        assert max(prefill_rows["fast"]) <= 2048
        # A real (not tail-only) 2048-row call ran on one of the gfx12 devices.
        assert max_seen["device"] in devices

        assert torch.equal(ttokens, ftokens), "forced tokens diverged from reference"

        for lg in (flogits, tlogits):
            assert not torch.isnan(lg).any() and not torch.isposinf(lg).any()

        top1 = int((flogits.argmax(dim=-1) == tlogits.argmax(dim=-1)).sum())
        finite = torch.isfinite(flogits) & torch.isfinite(tlogits)
        delta = (flogits.float() - tlogits.float()).abs()[finite]
        print(f"prefill-rows fast calls: {prefill_rows['fast']}")
        print(f"max fast rows={max_seen['rows']} device={max_seen['device']}")
        print(f"top1 agree={top1}/{STEP_OUT} max_diff={float(delta.max()):.4f} mean={float(delta.mean()):.6f}")
        # Generic fallback vs fast route are the same math; difference is fp32 rounding noise.
        assert top1 == STEP_OUT
        assert float(delta.max()) <= 1e-3
    finally:
        mode["name"] = None
        ext.exl3_moe_gfx12_k3_prefill = real_prefill
        model.unload()
        torch.cuda.empty_cache()
