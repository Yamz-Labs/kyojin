import atexit
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from exllamav3 import Cache, Config, Generator, GreedySampler, Job, Model, Tokenizer
from exllamav3.ext import exllamav3_ext as ext

DEFAULT_MODEL = os.path.expanduser("~/Models/Qwen3.8-27B-exl3")
MODEL = os.environ.get("EXL3_TEST_MODEL", DEFAULT_MODEL)


def _roc_available():
    if not (torch.version.hip and torch.cuda.is_available()):
        return False
    return True


def _require_gfx12():
    if not _roc_available():
        raise SystemExit("ROCm build / device not available")
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    # gfx12 (RDNA4) and gfx11.5 (RDNA3.5 / Strix Halo) both have a WMMA GEMV.
    # Ask the extension rather than hardcoding an arch allowlist.
    _wmma_archs = ("gfx1200", "gfx1201", "gfx1150", "gfx1151", "gfx1152")
    if arch.split(":", 1)[0] not in _wmma_archs:
        raise SystemExit(f"HIP GEMV oracle requires a WMMA arch {_wmma_archs}, got {arch or 'unknown'}")
    assert hasattr(ext, "exl3_gemv"), \
        "gfx12 target build is missing the required ext.exl3_gemv binding"
    assert hasattr(ext, "exl3_gemv_supported"), \
        "gfx12 target build is missing the required ext.exl3_gemv_supported binding"
    assert ext.exl3_gemv_supported(0), \
        f"ext.exl3_gemv_supported rejected gfx12 device 0 ({arch})"


STEPS = 16
PROMPT = "The capital of France is"


def main():
    if not _roc_available():
        raise SystemExit("ROCm build / device not available")
    _require_gfx12()
    if not os.path.isdir(MODEL):
        raise SystemExit(f"Test model not found: {MODEL}")

    mode = sys.argv[1]
    if mode not in {"fallback", "gemv"}:
        raise SystemExit(f"invalid mode: {mode}")
    output_path = Path(sys.argv[2])
    force_tokens_path = Path(sys.argv[3]) if len(sys.argv) > 3 else None

    config = Config.from_directory(MODEL)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=2048, max_batch_size=1)
    model.load(device="cuda")
    unload_model = model.unload
    atexit.register(unload_model)
    tokenizer = Tokenizer.from_config(model.config)
    generator = Generator(model=model, cache=cache, tokenizer=tokenizer)

    job = Job(
        input_ids=tokenizer.encode(PROMPT, add_bos=True),
        max_new_tokens=STEPS + 1,
        stop_conditions=[],
        return_logits=True,
        sampler=GreedySampler(),
    )
    generator.enqueue(job)
    if mode == "gemv":
        if force_tokens_path is None:
            raise SystemExit("gemv mode requires a forced-token path")
        baseline_token_ids = torch.load(force_tokens_path, map_location="cpu")
        if baseline_token_ids.dim() == 1:
            baseline_token_ids = baseline_token_ids.unsqueeze(0)
        job.constrain_output_now(baseline_token_ids.contiguous())

    real_gemv = ext.exl3_gemv
    real_reconstruct = ext.reconstruct
    real_reconstruct_slice = getattr(ext, "reconstruct_slice", None)
    real_reconstruct_had_slice = getattr(ext, "reconstruct_had_slice", None)
    real_hgemm = ext.hgemm
    counts = {"gemv": 0, "reconstruct": 0, "reconstruct_slice": 0, "reconstruct_had_slice": 0, "hgemm": 0}

    def gemv_spy(*args, **kwargs):
        counts["gemv"] += 1
        return real_gemv(*args, **kwargs)

    def reconstruct_spy(*args, **kwargs):
        counts["reconstruct"] += 1
        return real_reconstruct(*args, **kwargs)

    def reconstruct_slice_spy(*args, **kwargs):
        counts["reconstruct_slice"] += 1
        return real_reconstruct_slice(*args, **kwargs)

    def reconstruct_had_slice_spy(*args, **kwargs):
        counts["reconstruct_had_slice"] += 1
        return real_reconstruct_had_slice(*args, **kwargs)

    def hgemm_spy(*args, **kwargs):
        counts["hgemm"] += 1
        return real_hgemm(*args, **kwargs)

    ext.exl3_gemv = gemv_spy
    ext.reconstruct = reconstruct_spy
    if real_reconstruct_slice is not None:
        ext.reconstruct_slice = reconstruct_slice_spy
    if real_reconstruct_had_slice is not None:
        ext.reconstruct_had_slice = reconstruct_had_slice_spy
    ext.hgemm = hgemm_spy
    try:
        token_chunks = []
        logit_chunks = []
        while generator.num_remaining_jobs():
            for result in generator.iterate():
                if "token_ids" in result:
                    assert result["token_ids"].shape[-1] == result["logits"].shape[1]
                    token_chunks.append(result["token_ids"].cpu())
                    logit_chunks.append(result["logits"].cpu())
    finally:
        ext.exl3_gemv = real_gemv
        ext.reconstruct = real_reconstruct
        if real_reconstruct_slice is not None:
            ext.reconstruct_slice = real_reconstruct_slice
        if real_reconstruct_had_slice is not None:
            ext.reconstruct_had_slice = real_reconstruct_had_slice
        ext.hgemm = real_hgemm

    if not token_chunks or not logit_chunks:
        raise AssertionError("worker produced no token/logit results")
    token_ids = torch.cat(token_chunks, dim=-1)
    logits = torch.cat(logit_chunks, dim=1)
    if token_ids.shape[-1] != STEPS or logits.shape[1] != STEPS:
        raise AssertionError(
            f"expected exactly {STEPS} steps, got {token_ids.shape[-1]} tokens and {logits.shape[1]} logits")
    torch.save({"token_ids": token_ids, "logits": logits, "counts": counts}, output_path)
    try:
        unload_model()
    finally:
        atexit.unregister(unload_model)


if __name__ == "__main__":
    main()
