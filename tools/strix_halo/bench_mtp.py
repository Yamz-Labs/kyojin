#!/usr/bin/env python
"""Benchmark MTP speculative decoding on gfx1151.

This model (Qwen3.8-Flash-Next) ships 6,200 mtp.* tensors and its architecture
declares caps["mtp_draft"] = True, so the MTP head can act as a self-draft
model: it proposes N tokens per step and the trunk verifies them in ONE
multi-row forward. Accepted tokens are free, which is exactly the lever that
gets past a per-token-launch-bound decode wall.

Reports tok/s plus the acceptance rate, because speculative decode only pays
off when acceptance is high -- a low rate means wasted verification work.
"""
import argparse, os, sys, time
import torch

sys.path.insert(0, os.path.expanduser("~/kyojin"))

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default=os.path.expanduser("~/models/Qwen3.8-Flash-Next-EXL3"))
ap.add_argument("-p", "--prompt", default="Explain gradient descent in two sentences:")
ap.add_argument("-n", "--num_tokens", type=int, default=64)
ap.add_argument("-c", "--cache", type=int, default=4096)
ap.add_argument("-ndt", "--num_draft_tokens", type=int, default=None)
ap.add_argument("--no-mtp", action="store_true")
ap.add_argument("-dds", "--dynamic", action="store_true")
ap.add_argument("-dc", "--draft_confidence", type=float, default=0.4)
ap.add_argument("-g", "--greedy", action="store_true",
                help="Greedy sampling: removes acceptance-rate variance between runs")
ap.add_argument("-mcs", "--moe_cpu_split", type=int, default=0,
                help="Run the TAIL N routed experts per layer on the CPU, overlapped")
args = ap.parse_args()


def main():
    import exllamav3_ext as X
    from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job

    print(f"family={X.exl3_gemv_wmma_family(0)}  mtp={'OFF' if args.no_mtp else 'ON'}"
          f"  ndt={args.num_draft_tokens}  dynamic={args.dynamic}", flush=True)

    t0 = time.time()
    config = Config.from_directory(args.model)
    if args.moe_cpu_split:
        # Per-layer expert split: the TAIL N routed experts of every eligible MoE
        # layer run on the CPU worker, overlapping that layer's GPU expert compute.
        # Requires mul1-codebook experts (this model qualifies) and layer-split mode.
        config.infer_params.moe_cpu_split = args.moe_cpu_split
    model = Model.from_config(config)
    tokenizer = Tokenizer.from_config(config)
    # Recurrent models (gated delta net) require max_history == num_draft_tokens so
    # the recurrent state has room to roll back rejected draft tokens. Without it:
    #   RuntimeError: recurrent_state must be [num_slots, max_history + 1, ...]
    NDT = args.num_draft_tokens if args.num_draft_tokens is not None else 4
    MH = 0 if args.no_mtp else NDT
    cache = Cache(model, max_num_tokens=args.cache, max_history=MH)
    model.load(progressbar=False)

    draft_model = draft_cache = None
    if not args.no_mtp:
        # MTP: the draft "model" is the same directory / same config, loaded as the
        # MTP component. Mirrors model_init.init() with args.mtp = True.
        draft_config = config
        draft_model = Model.from_config(draft_config, component="mtp")
        draft_cache = Cache(draft_model, max_num_tokens=args.cache, max_history=MH)
        draft_model.load(progressbar=False)
        caps = getattr(draft_model, "caps", {})
        print(f"draft caps mtp_draft={caps.get('mtp_draft')}", flush=True)

    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    kw = {}
    if args.num_draft_tokens is not None:
        kw["num_draft_tokens"] = args.num_draft_tokens
    if args.dynamic:
        kw["dynamic_draft_tokens"] = True
        kw["draft_confidence"] = args.draft_confidence

    gen = Generator(model=model, cache=cache, tokenizer=tokenizer,
                    draft_model=draft_model, draft_cache=draft_cache, **kw)
    print(f"generator.mtp_draft = {getattr(gen, 'mtp_draft', None)}", flush=True)

    ids = tokenizer.encode(args.prompt, add_bos=True)
    job_kw = {}
    if args.greedy:
        from exllamav3.generator.sampler import GreedySampler
        job_kw["sampler"] = GreedySampler()
    t0 = time.time()
    gen.enqueue(Job(input_ids=ids, max_new_tokens=args.num_tokens, **job_kw))
    out, ttft, ntok = "", None, 0
    acc = rej = 0
    err_seen = False
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("error") is not None or r.get("eos_reason") == "error":
                print("JOB ERROR:", repr(r.get("error")), "eos_reason=", repr(r.get("eos_reason")), flush=True); err_seen = True
            c = r.get("text", "")
            if c:
                if ttft is None:
                    ttft = time.time() - t0
                out += c
                ntok += 1
            if "accepted_draft_tokens" in r:
                acc += r["accepted_draft_tokens"]
                rej += r["rejected_draft_tokens"]
    total = time.time() - t0

    print()
    print(f"output:  {out[:220]!r}")
    print(f"ttft:    {ttft:.2f}s" if ttft else "ttft: n/a")
    if ttft and ntok > 1:
        print(f"decode:  {(ntok-1)/(total-ttft):.2f} tok/s  ({ntok} chunks, {total:.1f}s)")
    if acc + rej:
        print(f"draft:   accepted={acc} rejected={rej} "
              f"acceptance={100*acc/(acc+rej):.1f}%")
    else:
        print("draft:   no draft stats reported (speculation inactive)")
    print(f"mem:     {torch.cuda.memory_allocated()/2**30:.1f} GiB")


if __name__ == '__main__':
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=False)
    except RuntimeError:
        pass
    main()
