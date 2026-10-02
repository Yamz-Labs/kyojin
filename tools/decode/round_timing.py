#!/usr/bin/env python
"""synced wall-clock split of one DFlash round: drafter (iterate_draftmodel_dflash_gen),
target verify forward (model.forward at R rows), everything else (sampling/accept/bookkeeping),
vs a plain batch-1 decode step. 2K-token code prompt, warm-up discarded."""
from __future__ import annotations

import collections
import os
import sys
import time

sys.path.insert(0, os.environ.get("EXL3_REPO", "~/kyojin"))
import torch  # noqa: E402
from exllamav3 import Cache, Config, Generator, GreedySampler, Job, Model, Tokenizer  # noqa: E402

REPO = os.environ.get("EXL3_REPO", "~/kyojin")
CODE = os.path.join(REPO, "exllamav3/conversion/standard_cal_data/code.utf8")
NDTS = [int(x) for x in os.environ.get("NDTS", "3,4,7").split(",")]
N = 128


def main():
    torch.set_grad_enabled(False)
    m = os.path.expanduser("~/models/mimo26-exl3")
    config = Config.from_directory(m)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=2560)
    model.load(progressbar=False)
    tok = Tokenizer.from_config(config)
    dconf = Config.from_directory(os.environ.get("DRAFT_DIR", os.path.join(m, "dflash")))
    dmodel = Model.from_config(dconf)
    dcache = Cache(dmodel, max_num_tokens=2560)
    dmodel.load(progressbar=False)
    text = open(CODE, encoding="utf-8").read()
    all_ids = tok.encode(text, add_bos=False)

    T = collections.defaultdict(float)
    C = collections.Counter()
    orig_fwd = model.forward

    def fwd(*a, **kw):
        ids = kw.get("input_ids", a[0] if a else None)
        rows = ids.shape[-1] if ids is not None and ids.dim() == 2 else -1
        torch.cuda.synchronize(); t0 = time.perf_counter()
        r = orig_fwd(*a, **kw)
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        key = f"target_R{rows}" if rows <= 16 else "target_prefill"
        T[key] += dt; C[key] += 1
        return r

    model.forward = fwd

    if os.environ.get("MODPROF"):
        from exllamav3.modules.transformer import TransformerBlock

        def wrap(obj, kind):
            orig = obj.forward

            def f(x, params, *a, **kw):
                rows = x.shape[0] * x.shape[1] if x.dim() == 3 else x.shape[0]
                torch.cuda.synchronize(); t0 = time.perf_counter()
                r = orig(x, params, *a, **kw)
                torch.cuda.synchronize()
                key = f"  {kind}_{type(obj).__name__}_R{rows}"
                T[key] += time.perf_counter() - t0; C[key] += 1
                return r
            obj.forward = f

        for mod in model.modules:
            if isinstance(mod, TransformerBlock):
                if mod.attn is not None:
                    wrap(mod.attn, "attn")
                if mod.mlp is not None:
                    wrap(mod.mlp, "mlp")

    def run(gen, ids, n):
        job = Job(input_ids=ids, max_new_tokens=n, stop_conditions=[], sampler=GreedySampler())
        gen.enqueue(job)
        torch.cuda.synchronize(); t0 = time.perf_counter(); ttft = None; k = 0
        while gen.num_remaining_jobs():
            for r in gen.iterate():
                if r.get("text") and ttft is None:
                    ttft = time.perf_counter() - t0
                if r.get("eos"):
                    k = r.get("new_tokens", 0); last = r
        tot = time.perf_counter() - t0
        return k, tot - ttft, last

    def one(tag, gen, off):
        # warm-up
        run(gen, all_ids[:, 50000:52048].contiguous(), 32)
        T.clear(); C.clear()
        ids = all_ids[:, off:off + 2048].contiguous()
        k, dec, last = run(gen, ids, N)
        acc = last.get("accepted_draft_tokens"); rej = last.get("rejected_draft_tokens")
        print(f"[{tag}] {k} tok decode {dec:.3f}s = {(k - 1) / dec:.2f} tok/s acc={acc} rej={rej}")
        for key in sorted(T):
            print(f"   {key:16s} calls={C[key]:4d} total={T[key]:.3f}s per_call={1e3 * T[key] / C[key]:.2f} ms")
        sys.stdout.flush()

    run_cfg = run_cfg_factory(model, cache, tok, dmodel, dcache, one, T, C)
    configs = [c for c in os.environ.get("CONFIGS", "base:").split(";") if c]
    for cfg in configs:
        name, _, assigns = cfg.partition(":")
        for kv in [a for a in assigns.split(",") if a]:
            k, v = kv.split("=")
            os.environ[k] = v
        print(f"##### config {name}: {assigns}", flush=True)
        run_cfg(name)
        for kv in [a for a in assigns.split(",") if a]:
            os.environ.pop(kv.split("=")[0], None)


def run_cfg_factory(model, cache, tok, dmodel, dcache, one, T, C):
    def run_cfg(name):
      gp = Generator(model=model, cache=cache, tokenizer=tok)
      one(f"{name} plain", gp, 10000)
      del gp
      for ndt in NDTS:
          gd = Generator(model=model, cache=cache, tokenizer=tok, draft_model=dmodel, draft_cache=dcache,
                         num_draft_tokens=ndt)
          orig_draft = gd.iterate_draftmodel_dflash_gen

          def dr(results, orig_draft=orig_draft):
              torch.cuda.synchronize(); t0 = time.perf_counter()
              r = orig_draft(results)
              torch.cuda.synchronize(); T["drafter"] += time.perf_counter() - t0; C["drafter"] += 1
              return r

          gd.iterate_draftmodel_dflash_gen = dr
          one(f"{name} dflash ndt={ndt}", gd, 10000)
          del gd

    return run_cfg

if __name__ == "__main__":
    main()
