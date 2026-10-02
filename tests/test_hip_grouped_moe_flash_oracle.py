"""Opt-in full Qwen3.8-Flash-Next oracle for the grouped routed-expert path."""
from __future__ import annotations

from collections import Counter, defaultdict
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
RUN_ORACLE = os.environ.get("EXL3_RUN_FLASH_GROUPED_ORACLE") == "1"
STEPS = 16
PROMPT = "Q: The capital of France is\nA:"


def _gfx12_devices():
    if not (torch.version.hip and torch.cuda.is_available()):
        return []
    return [
        index for index in range(torch.cuda.device_count())
        if getattr(torch.cuda.get_device_properties(index), "gcnArchName", "").split(":", 1)[0]
        in ("gfx1200", "gfx1201")
    ]


def _run_forced_context(model, cache, tokenizer, forced_tokens=None):
    generator = Generator(model=model, cache=cache, tokenizer=tokenizer)
    job = Job(
        input_ids=tokenizer.encode(PROMPT, add_bos=True),
        max_new_tokens=STEPS + 1,
        stop_conditions=[],
        return_logits=True,
        sampler=GreedySampler(),
    )
    generator.enqueue(job)
    if forced_tokens is not None:
        job.constrain_output_now(forced_tokens.contiguous())

    token_chunks, logit_chunks = [], []
    while generator.num_remaining_jobs():
        for result in generator.iterate():
            if "token_ids" in result:
                token_chunks.append(result["token_ids"].cpu())
                logit_chunks.append(result["logits"].cpu())
    tokens = torch.cat(token_chunks, dim=-1)
    logits = torch.cat(logit_chunks, dim=1)
    assert tokens.shape == (1, STEPS)
    assert logits.shape[:2] == (1, STEPS)
    return tokens, logits


@pytest.mark.skipif(not RUN_ORACLE, reason="set EXL3_RUN_FLASH_GROUPED_ORACLE=1")
@torch.inference_mode()
def test_flash_grouped_full_forced_context_oracle(monkeypatch):
    devices = _gfx12_devices()
    if len(devices) < 2:
        pytest.skip(f"full grouped oracle requires two gfx12 devices, found {devices}")
    if not MODEL.is_dir():
        pytest.skip(f"Flash model not found: {MODEL}")
    assert hasattr(ext, "exl3_moe_gfx12_k3")

    model = Model.from_config(Config.from_directory(str(MODEL)))
    # One attached cache gives every pass identical tensor addresses as well as the same
    # loaded weights and layer placement. Completed jobs release their recurrent slot.
    oracle_cache = Cache(model, max_num_tokens=512, max_batch_size=1)
    first_gb = float(os.environ.get("EXL3_FLASH_ORACLE_GPU0_GB", "25"))
    second_gb = float(os.environ.get("EXL3_FLASH_ORACLE_GPU1_GB", "31"))
    budgets = [0.0] * torch.cuda.device_count()
    budgets[devices[0]], budgets[devices[1]] = first_gb, second_gb

    route_mode = {"name": None}
    active_decode_mlp = {"key": None}
    route_counts = {
        "fallback": {"grouped": Counter(), "k3": Counter()},
        "grouped": {"grouped": Counter(), "k3": Counter()},
    }
    layer_captures = {
        "fallback": defaultdict(list),
        "grouped": defaultdict(list),
    }
    wrapped_mlps = []
    real_grouped = ext.exl3_moe_gfx12_k3
    real_gemv = ext.exl3_gemv

    def grouped_spy(*args, **kwargs):
        mode = route_mode["name"]
        if mode in route_counts:
            route_counts[mode]["grouped"][args[0].device.index] += 1
        return real_grouped(*args, **kwargs)

    def gemv_spy(*args, **kwargs):
        mode = route_mode["name"]
        if mode in route_counts and active_decode_mlp["key"] is not None \
                and args[1].shape[-1] // 16 == 3:
            route_counts[mode]["k3"][args[0].device.index] += 1
        return real_gemv(*args, **kwargs)

    monkeypatch.setenv("EXL3_GEMV", "2")
    try:
        model.load(
            use_per_device=budgets,
            max_chunk_size=256,
            max_batch_size=1,
        )
        tokenizer = Tokenizer.from_config(model.config)
        grouped_mlps = [
            module for module in model
            if isinstance(module, BlockSparseMLP) and module.support_hip_grouped
        ]
        assert grouped_mlps
        placed_devices = {torch.device(module.device).index for module in grouped_mlps}
        assert set(devices[:2]) <= placed_devices

        for mlp in grouped_mlps:
            original = mlp.forward

            def capture_forward(x, params, out_dtype=None, _mlp=mlp, _forward=original):
                is_decode = x.numel() == 2560
                if is_decode:
                    active_decode_mlp["key"] = _mlp.key
                try:
                    output = _forward(x, params, out_dtype)
                finally:
                    active_decode_mlp["key"] = None
                mode = route_mode["name"]
                if mode in layer_captures and is_decode:
                    layer_captures[mode][_mlp.key].append(output.detach().float().cpu().clone())
                return output

            mlp.forward = capture_forward
            wrapped_mlps.append((mlp, original))

        ext.exl3_moe_gfx12_k3 = grouped_spy
        ext.exl3_gemv = gemv_spy

        route_mode["name"] = "fallback"
        monkeypatch.setenv("EXL3_HIP_GROUPED_MOE", "0")
        fallback_tokens, fallback_logits = _run_forced_context(
            model, oracle_cache, tokenizer)

        # The recurrent model amplifies run-to-run fp32 noise. Repeat the established route
        # under the same forced context to establish the control envelope.
        route_mode["name"] = None
        control_tokens, control_logits = _run_forced_context(
            model, oracle_cache, tokenizer, fallback_tokens)

        route_mode["name"] = "grouped"
        monkeypatch.setenv("EXL3_HIP_GROUPED_MOE", "1")
        grouped_tokens, grouped_logits = _run_forced_context(
            model, oracle_cache, tokenizer, fallback_tokens)
        route_mode["name"] = None

        assert torch.equal(control_tokens, fallback_tokens)
        assert torch.equal(grouped_tokens, fallback_tokens)
        assert sum(route_counts["fallback"]["grouped"].values()) == 0
        assert sum(route_counts["fallback"]["k3"].values()) > 0
        assert set(route_counts["grouped"]["grouped"]) == set(devices[:2])
        grouped_calls = sum(route_counts["grouped"]["grouped"].values())
        assert grouped_calls >= len(grouped_mlps) * STEPS
        assert sum(route_counts["grouped"]["k3"].values()) == 0

        for logits in (fallback_logits, control_logits, grouped_logits):
            assert not torch.isnan(logits).any()
            assert not torch.isposinf(logits).any()

        def comparison(candidate):
            top1 = int((fallback_logits.argmax(dim=-1) == candidate.argmax(dim=-1)).sum())
            baseline_top5 = torch.topk(fallback_logits, 5, dim=-1).indices
            candidate_top5 = torch.topk(candidate, 5, dim=-1).indices
            top5 = (baseline_top5.unsqueeze(-1) == candidate_top5.unsqueeze(-2)) \
                .any(dim=-1).sum(dim=-1)
            finite = torch.isfinite(fallback_logits) & torch.isfinite(candidate)
            delta = (fallback_logits.float() - candidate.float()).abs()[finite]
            return top1, int(top5.min()), float(delta.max()), float(delta.mean())

        control_metrics = comparison(control_logits)
        grouped_metrics = comparison(grouped_logits)
        top1_agree, top5_min, max_diff, mean_diff = grouped_metrics

        first_drift = None
        for step in range(STEPS):
            for mlp in grouped_mlps:
                fallback_rows = layer_captures["fallback"][mlp.key]
                grouped_rows = layer_captures["grouped"][mlp.key]
                if len(fallback_rows) < STEPS or len(grouped_rows) < STEPS:
                    continue
                frow = fallback_rows[-STEPS + step]
                grow = grouped_rows[-STEPS + step]
                if not torch.equal(frow, grow):
                    delta = (frow - grow).abs()
                    first_drift = (step, mlp.key, float(delta.max()), float(delta.mean()))
                    break
            if first_drift is not None:
                break

        print(f"grouped route counts: {route_counts}")
        print(f"first grouped drift: {first_drift}")
        print(f"fallback-repeat metrics: {control_metrics}")
        print(f"grouped metrics: {grouped_metrics}")
        assert first_drift is not None
        # The first grouped difference is fp32 rounding noise at the first routed MLP. Ground
        # the recurrently amplified end-to-end envelope against the fallback repeat.
        assert first_drift[2] <= 1e-5 and first_drift[3] <= 1e-7
        assert top1_agree >= STEPS - 4
        assert top5_min >= 2
        assert max_diff <= max(6.0, control_metrics[2] * 1.5)
        assert mean_diff <= max(0.6, control_metrics[3] * 1.5)

        completion = Generator(model=model, cache=oracle_cache, tokenizer=tokenizer).generate(
            prompt="The capital of France is",
            stop_conditions=[],
            max_new_tokens=8,
            completion_only=True,
            add_bos=True,
        )
        print(f"grouped free-run: {completion!r}")
        assert completion and "Paris" in completion
    finally:
        route_mode["name"] = None
        active_decode_mlp["key"] = None
        ext.exl3_moe_gfx12_k3 = real_grouped
        ext.exl3_gemv = real_gemv
        for mlp, original in wrapped_mlps:
            mlp.forward = original
        model.unload()
        torch.cuda.empty_cache()
