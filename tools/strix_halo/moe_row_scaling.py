#!/usr/bin/env python
"""Does the grouped-MoE kernel amortize its weight read across rows?

This is THE question for MTP throughput. Verification runs m = 1 + ndt rows
through the same experts; if the kernel reads each expert's weights once and
does m rows of math, MTP is nearly free. If cost scales with m, weights are
being re-read and that is where the 34%-of-peak comes from.

Measured by calling ext.exl3_moe_gfx12_k3 directly at several row counts,
reusing the live model's real expert pointer tables (captured from a forward),
so the trellis state, shapes and codebooks are genuine.
"""
import os, sys, time, collections
import torch

sys.path.insert(0, os.path.expanduser("~/kyojin"))
import exllamav3_ext as X
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler

MODEL = os.path.expanduser("~/models/Qwen3.8-Flash-Next-EXL3")


def main():
    config = Config.from_directory(MODEL)
    model = Model.from_config(config)
    tokenizer = Tokenizer.from_config(config)
    cache = Cache(model, max_num_tokens=2048)
    model.load(progressbar=False)

    # Capture one real call's arguments from a live forward.
    captured = {}
    real = X.exl3_moe_gfx12_k3

    def spy(*a, **k):
        if "args" not in captured:
            captured["args"] = a
        return real(*a, **k)

    X.exl3_moe_gfx12_k3 = spy
    gen = Generator(model=model, cache=cache, tokenizer=tokenizer)
    gen.enqueue(Job(input_ids=tokenizer.encode("hello world", add_bos=True),
                    max_new_tokens=2, sampler=GreedySampler()))
    while gen.num_remaining_jobs():
        for _ in gen.iterate():
            pass
    X.exl3_moe_gfx12_k3 = real

    if "args" not in captured:
        print("no grouped-MoE call captured")
        return
    a = list(captured["args"])
    print(f"captured {len(a)} args")
    for i, v in enumerate(a[:8]):
        if torch.is_tensor(v):
            print(f"  arg{i}: {tuple(v.shape)} {v.dtype}")
        else:
            print(f"  arg{i}: {type(v).__name__} {v if not hasattr(v,'__len__') else ''}")

    # arg0 = A [R, 2560] fp16, arg1 = output [R, 2560] fp32,
    # arg2 = selected [R, 10] int64, arg3 = weights [R, 10] fp16
    A, out, sel, wts = a[0], a[1], a[2], a[3]
    R0 = A.shape[0]
    print(f"\nnative rows R={R0}")

    print(f"\n{'rows':>5} {'ms/call':>9} {'ms/row':>8} {'vs m=1':>8}  interpretation")
    print("-" * 64)
    base = None
    for m in (1, 2, 3, 4, 8, 16):
        if m > A.shape[0]:
            # widen by repeating the captured row
            A2 = A[:1].repeat(m, 1).contiguous()
            o2 = out[:1].repeat(m, 1).contiguous()
            s2 = sel[:1].repeat(m, 1).contiguous()
            w2 = wts[:1].repeat(m, 1).contiguous()
        else:
            A2, o2, s2, w2 = (A[:m].contiguous(), out[:m].contiguous(),
                              sel[:m].contiguous(), wts[:m].contiguous())
        args = [A2, o2, s2, w2] + a[4:]
        try:
            for _ in range(3):
                real(*args)
            torch.cuda.synchronize()
            reps = 20
            t0 = time.perf_counter()
            for _ in range(reps):
                real(*args)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / reps
        except Exception as e:
            print(f"{m:>5}  FAILED: {type(e).__name__}: {str(e)[:60]}")
            continue
        if base is None:
            base = dt
        ratio = dt / base
        interp = ("FLAT - weights amortized" if ratio < 1.3 * 1 and m > 1 and ratio < m * 0.5
                  else "scales with rows" if m > 1 and ratio > 0.7 * m else "")
        print(f"{m:>5} {dt*1000:>9.3f} {dt*1000/m:>8.3f} {ratio:>8.2f}x  {interp}")

    print()
    print("FLAT cost => verification is cheap, MTP bounded by something else.")
    print("LINEAR cost => weights re-read per row; batching the MoE is the win.")


if __name__ == "__main__":
    import multiprocessing
    try:
        multiprocessing.set_start_method("spawn", force=False)
    except RuntimeError:
        pass
    main()
