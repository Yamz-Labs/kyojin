"""CPU checks for the job groups of a row-invariant multi-job decode step (generator/row_groups.py).

The arithmetic is proven on the GPU (tools/qwen/batch_inv.py); this file pins the pure logic: how jobs are packed
into forwards, and that the grouped forward feeds every group its own slice of the batch and puts the pieces back in
batch order.
"""
import types

import pytest
import torch

from exllamav3.generator import row_groups as RG
from exllamav3.generator.generator import Generator
from exllamav3.util.row_inv import ROW_INV_MAX_ROWS


def test_cap_default_and_override(monkeypatch):
    monkeypatch.delenv("EXL3_ROW_INV_GROUP_ROWS", raising=False)
    assert RG.group_rows_cap() == ROW_INV_MAX_ROWS == (32 if torch.version.hip else 8)
    monkeypatch.setenv("EXL3_ROW_INV_GROUP_ROWS", "1")
    assert RG.group_rows_cap() == 1
    monkeypatch.setenv("EXL3_ROW_INV_GROUP_ROWS", "0")
    assert RG.group_rows_cap() == 1
    monkeypatch.setenv("EXL3_ROW_INV_GROUP_ROWS", "")
    assert RG.group_rows_cap() == ROW_INV_MAX_ROWS


@pytest.mark.parametrize("value,expect", [(None, False), ("", False), ("0", False), ("1", True)])
def test_grouped_step_is_opt_in(value, expect):
    import os
    import subprocess
    import sys
    env = {k: v for k, v in os.environ.items() if k != "EXL3_ROW_INV_GROUPS"}
    if value is not None:
        env["EXL3_ROW_INV_GROUPS"] = value
    out = subprocess.run([sys.executable, "-c", "from exllamav3.generator.row_groups import GROUPS; print(GROUPS['on'])"],
                         env=env, capture_output=True, text=True, check=True).stdout.strip()
    assert out == str(expect)


def test_plain_decode_packs_eight_jobs_per_forward():
    assert RG.plan_groups([1] * 4, 1, 8) == [(0, 4)]
    assert RG.plan_groups([1] * 8, 1, 8) == [(0, 8)]
    assert RG.plan_groups([1] * 9, 1, 8) == [(0, 8), (8, 9)]
    assert RG.plan_groups([1] * 20, 1, 8) == [(0, 8), (8, 16), (16, 20)]


def test_verify_rows_split_by_width():
    # four jobs of 2, 3, 4 and 8 token rows
    assert RG.plan_groups([1] * 4, 2, 8) == [(0, 4)]
    assert RG.plan_groups([1] * 4, 3, 8) == [(0, 2), (2, 4)]
    assert RG.plan_groups([1] * 4, 4, 8) == [(0, 2), (2, 4)]
    assert RG.plan_groups([1] * 4, 8, 8) == [(0, 1), (1, 2), (2, 3), (3, 4)]
    assert RG.plan_groups([1] * 3, 5, 8) == [(0, 1), (1, 2), (2, 3)]


def test_groups_cover_every_job_once_in_order():
    for n in range(0, 12):
        for width in range(1, 9):
            for cap in (1, 2, 5, 8, 16):
                spans = RG.plan_groups([1] * n, width, cap)
                flat = [j for a, b in spans for j in range(a, b)]
                assert flat == list(range(n)), (n, width, cap, spans)
                for a, b in spans:
                    assert b > a
                    assert (b - a) * width <= max(cap, width), (n, width, cap, spans)


def test_multi_sequence_jobs_stay_whole():
    # a job with 3 sequences and width 2 carries 6 rows; the next job (2 rows) does not fit in the same group of 8
    assert RG.plan_groups([3, 1, 1], 2, 8) == [(0, 2), (2, 3)]
    # a job larger than the cap is its own group
    assert RG.plan_groups([5, 1], 2, 8) == [(0, 1), (1, 2)]


class _Job:
    def __init__(self, nseq=1):
        self.sequences = [object() for _ in range(nseq)]


def _fake_model(log):
    """Logits row = f(token id, cache_seqlen, group size): any slicing mistake or reordering shows."""
    def forward(input_ids, params):
        bsz, width = input_ids.shape
        assert params["block_table"].shape[0] == bsz
        assert params["cache_seqlens"].shape[0] == bsz
        assert len(params["recurrent_states"]) == bsz
        if "dflash_table_pages" in params:
            assert len(params["dflash_table_pages"]) == bsz
        log.append((bsz, width, bool(params.get("dflash_verify"))))
        out = (input_ids.float() * 1000 + params["cache_seqlens"].float().view(-1, 1)).unsqueeze(-1).repeat(1, 1, 4)
        params["export_states"] = [out[..., :2].clone(), out[..., 2:].clone() + 0.5]
        return out
    return types.SimpleNamespace(forward=forward)


def _run(n, width, draft, cap=8):
    jobs = [_Job() for _ in range(n)]
    ids = torch.arange(n * width, dtype=torch.long).view(n, width) + 10
    params = {
        "block_table": torch.arange(n * 6, dtype=torch.int32).view(n, 6),
        "cache_seqlens": torch.arange(n, dtype=torch.int32) * 100 + 7,
        "recurrent_states": [object() for _ in range(n)],
        "positions": None,
        "dflash_verify": draft,
    }
    own = [16 + i for i in range(n)] if draft else None
    log = []
    gen = types.SimpleNamespace(model=_fake_model(log))
    spans = RG.plan_groups([1] * n, width, cap)
    out = Generator._forward_groups(gen, ids, params, jobs, spans, own)
    whole_params = dict(params)
    ref = _fake_model([]).forward(ids, whole_params)
    return out, ref, params, whole_params, log


@pytest.mark.parametrize("n,width,draft", [(4, 1, False), (4, 3, True), (3, 8, True), (9, 1, False), (5, 2, True)])
def test_grouped_forward_equals_the_whole_batch(n, width, draft):
    out, ref, params, whole_params, log = _run(n, width, draft)
    assert torch.equal(out, ref)
    for a, b in zip(params["export_states"], whole_params["export_states"]):
        assert torch.equal(a, b)
    assert sum(b for b, _, _ in log) == n
    assert all(b * w <= 8 or b == 1 for b, w, _ in log)


def test_plain_group_of_several_rows_takes_the_verify_kernels():
    _, _, _, _, log = _run(4, 1, False)
    assert log == [(4, 1, True)]
    _, _, _, _, log = _run(9, 1, False)
    assert log == [(8, 1, True), (1, 1, False)]       # a lone one-row job keeps the one-row path


def test_several_forwards_do_not_share_the_pinned_embedding_staging():
    seen = []
    gen = types.SimpleNamespace(model=types.SimpleNamespace(forward=lambda input_ids, params: (
        seen.append(params.get("pinned_staging")) or torch.zeros(input_ids.shape[0], input_ids.shape[1], 2))))
    for n, expect in ((4, [True]), (9, [False, False])):
        del seen[:]
        jobs = [_Job() for _ in range(n)]
        params = {"block_table": torch.zeros(n, 3, dtype=torch.int32), "cache_seqlens": torch.zeros(n, dtype=torch.int32),
                  "recurrent_states": [object() for _ in range(n)], "pinned_staging": True}
        Generator._forward_groups(gen, torch.zeros(n, 1, dtype=torch.long), params, jobs, RG.plan_groups([1] * n, 1, 8), None)
        assert seen == expect, (n, seen)


def test_cap_one_runs_every_job_alone():
    _, _, _, _, log = _run(3, 3, True, cap=1)
    assert [b for b, _, _ in log] == [1, 1, 1]
    assert all(v for _, w, v in log)                 # width 3: still the verify rows of the solo path


def test_group_of_several_sequences_runs_attention_per_sequence():
    seen = []
    gen = types.SimpleNamespace(model=types.SimpleNamespace(forward=lambda input_ids, params: (
        seen.append((input_ids.shape[0], bool(params.get("row_inv_per_seq")))) or torch.zeros(input_ids.shape[0], input_ids.shape[1], 2))))
    jobs = [_Job() for _ in range(9)]
    params = {"block_table": torch.zeros(9, 3, dtype=torch.int32), "cache_seqlens": torch.zeros(9, dtype=torch.int32),
              "recurrent_states": [object() for _ in range(9)]}
    Generator._forward_groups(gen, torch.zeros(9, 1, dtype=torch.long), params, jobs, RG.plan_groups([1] * 9, 1, 8), None)
    # the eight-sequence group splits its attention per sequence, the lone job keeps the plain path
    assert seen == [(8, True), (1, False)]


def test_split_params_by_sequence_slices_every_per_sequence_input():
    from exllamav3.modules.attn import split_params_by_sequence
    seqlens = torch.tensor([5, 9, 300], dtype=torch.int32)
    params = {
        "block_table": torch.arange(12, dtype=torch.int32).view(3, 4),
        "cache_seqlens": seqlens,
        "positions": seqlens,                  # what prepare_flash_attn does for a plain decode: an alias
        "dev_cache": {"cache_seqlens": object()},
        "causal": True,
    }
    parts = split_params_by_sequence(params, 3)
    assert len(parts) == 3
    for b, p in enumerate(parts):
        assert torch.equal(p["block_table"], params["block_table"][b:b + 1])
        assert torch.equal(p["cache_seqlens"], seqlens[b:b + 1])
        assert p["positions"] is p["cache_seqlens"]
        assert "dev_cache" not in p and p["causal"] is True
    # separate positions (a caller that set them) are sliced on their own
    params["positions"] = seqlens + 1
    parts = split_params_by_sequence(params, 3)
    assert [int(p["positions"]) for p in parts] == [6, 10, 301]
    # none: stays none
    params["positions"] = None
    assert all(p["positions"] is None for p in split_params_by_sequence(params, 3))
    # per-sequence position ids are not sliced here: refuse instead of running wrong
    params["position_ids"] = torch.zeros(3, 1, dtype=torch.long)
    with pytest.raises(AssertionError):
        split_params_by_sequence(params, 3)
