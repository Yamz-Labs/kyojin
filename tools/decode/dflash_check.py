#!/usr/bin/env python
"""Diagnosis: plain greedy vs real-DFlash greedy on code/chat/prose.
Per prompt: first token divergence, plain top1-top2 margin there (near-tie < 0.05 => numerics),
drafted-run margin there, max|dlogit| over the positions before the divergence (numeric noise
floor of the R-row verify), smallest plain margin before the divergence, acceptance rate.
Optional PERFECT=1 also runs a perfect-draft arm (drafter proposes plain's continuation)."""
from __future__ import annotations

import collections
import importlib.util
import os
import sys
import time

sys.path.insert(0, os.environ.get("EXL3_REPO", "~/kyojin"))
import torch  # noqa: E402
from exllamav3 import Cache, Config, Generator, GreedySampler, Job, Model, Tokenizer  # noqa: E402

N = int(os.environ.get("NTOK", "128"))
PROMPTS = [
    ("code", "Write a Python function `is_prime(n: int) -> bool` that returns True if n "
             "is prime, using trial division up to sqrt(n). Then write a second function "
             "`primes_up_to(limit: int) -> list[int]` that uses it. Show both with docstrings."),
    ("chat", "Give me a friendly, detailed three-paragraph explanation of how a bicycle "
             "stays upright while moving, aimed at a curious 12-year-old."),
    ("prose", "Write the opening three paragraphs of a short story about a lighthouse "
              "keeper who discovers a message in a bottle, in a literary, descriptive style."),
]


def run(gen, tok, prompt, n):
    ids = tok.encode(prompt) if isinstance(prompt, str) else prompt
    job = Job(input_ids=ids, max_new_tokens=n, stop_conditions=[], sampler=GreedySampler(),
              return_logits=True)
    gen.enqueue(job)
    toks, logits, last = [], [], {}
    t0 = time.perf_counter()
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("error"):
                raise RuntimeError(r["error"])
            if r.get("token_ids") is not None:
                toks.append(r["token_ids"].cpu())
            if r.get("logits") is not None:
                logits.append(r["logits"].cpu())
            if r.get("eos"):
                last = r
    last = dict(last); last["draft_stats"] = list(getattr(job, "draft_stats", []) or [])
    dt = time.perf_counter() - t0
    t = torch.cat(toks, dim=-1).flatten().tolist()
    L = torch.cat(logits, dim=1).float()[0]
    return t, L, last, dt


def margin(row):
    v = row.topk(2).values
    return (v[0] - v[1]).item()


def compare(name, arm, pt, pl, dt_, dl, last, ref=None):
    P = min(len(pt), len(dt_), pl.shape[0], dl.shape[0])
    V = min(pl.shape[-1], dl.shape[-1])
    div = next((i for i in range(P) if pt[i] != dt_[i]), None)
    upto = div if div is not None else P
    fin = torch.isfinite(pl[:upto, :V]) & torch.isfinite(dl[:upto, :V])
    d = (pl[:upto, :V] - dl[:upto, :V]).abs().where(fin, torch.zeros(()))
    maxd = d.max().item() if upto else 0.0
    per_pos = d.amax(dim=-1) if upto else torch.zeros(0)
    minm = min((margin(pl[i, :V]) for i in range(upto)), default=float("nan"))
    acc = last.get("accepted_draft_tokens", 0)
    rej = last.get("rejected_draft_tokens", 0)
    rate = acc / max(1, acc + rej)
    print(f"[{name}/{arm}] first_div={div} maxd_before={maxd:.5f} "
          f"min_plain_margin_before={minm:.5f} accepted={acc} rejected={rej} accept_rate={rate:.3f}")
    if div is not None:
        mp = margin(pl[div, :V])
        md = margin(dl[div, :V])
        dd = (pl[div, :V] - dl[div, :V]).abs()
        dd = dd[torch.isfinite(dd)].max().item()
        verdict = "NEAR-TIE" if mp < 0.05 else "CLEAR"
        print(f"[{name}/{arm}] at div: plain_tok={pt[div]} draft_tok={dt_[div]} plain_margin={mp:.5f} "
              f"drafted_margin={md:.5f} maxd_at_div={dd:.5f} plain_logit[draft_tok]-plain_top1="
              f"{(pl[div, dt_[div]] - pl[div, pt[div]]).item():.5f} -> {verdict}")
    rowof = {}
    for end, w, a in last.get("draft_stats", []):
        st = end - (a + 1)
        for i in range(a + 1):
            rowof[st + i] = i
    big = [(i, rowof.get(i), round(per_pos[i].item(), 3), round(margin(pl[i, :V]), 3),
            int(pl[i, :V].argmax() != dl[i, :V].argmax()),
            round((dl[i, pt[i]] - pl[i, pt[i]]).item(), 3)) for i in range(upto) if per_pos[i] > 0.25]
    print(f"[{name}/{arm}] big |dlogit|>0.25 (pos,row,maxd,plain_margin,argmax_flip,d_top1logit): {big[:40]}")
    rows = collections.defaultdict(list)
    for i in range(upto):
        rows[rowof.get(i)].append(per_pos[i].item())
    print(f"[{name}/{arm}] mean max|dlogit| by row-in-round: "
          + " ".join(f"r{k}:{sum(v)/len(v):.3f}(n{len(v)})" for k, v in sorted(rows.items(), key=lambda kv: -1 if kv[0] is None else kv[0])))
    if ref is not None and upto:
        V2 = min(V, ref.shape[-1])
        dr = (ref[:upto, :V2] - dl[:upto, :V2]).abs().nan_to_num(0, 0, 0).amax(-1)
        print(f"[{name}/{arm}] drafted-vs-prefillref max|dlogit| per pos (first 48): "
              + " ".join(f"{x:.3f}" for x in dr[:48].tolist()))
    nz = (per_pos > 0).nonzero().flatten().tolist() if upto else []
    print(f"[{name}/{arm}] positions with any |dlogit|>0 before div: {len(nz)}/{upto}; first={nz[:8]}")
    if upto:
        print(f"[{name}/{arm}] per-pos max|dlogit| (first 48): "
              + " ".join(f"{x:.3f}" for x in per_pos[:48].tolist()))


def main():
    torch.set_grad_enabled(False)
    m = os.path.expanduser("~/models/mimo26-exl3")
    config = Config.from_directory(m)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=2560)
    cache_d = Cache(model, max_num_tokens=2560)
    model.load(progressbar=False)
    tok = Tokenizer.from_config(config)
    dconf = Config.from_directory(os.path.join(m, "dflash"))
    dmodel = Model.from_config(dconf)
    dcache = Cache(dmodel, max_num_tokens=2560)
    dmodel.load(progressbar=False)

    gen_p = Generator(model=model, cache=cache, tokenizer=tok)
    gen_d = Generator(model=model, cache=cache_d, tokenizer=tok, draft_model=dmodel, draft_cache=dcache,
                      num_draft_tokens=int(os.environ.get("NDT", "7")), record_draft_stats=True)
    prompts = PROMPTS
    if os.environ.get("PSRC") == "bench":
        # the 2K-token prompts dflash-bench.py uses (kind, rep)
        spec = importlib.util.spec_from_file_location("db", os.path.join(
            os.environ.get("EXL3_REPO", "."), "scripts/dflash-bench.py"))
        db = importlib.util.module_from_spec(spec); spec.loader.exec_module(db)
        corp = {k: db.load_corpus(v) for k, v in db.CORPUS.items()}
        prompts = []
        for kr in os.environ.get("BENCH_PROMPTS", "chat:0,chat:2,code:0,prose:0").split(","):
            k, r = kr.split(":")
            prompts.append((f"{k}{r}", db.prompt_ids(tok, corp, k, 2048, int(r) * 2048)))
    print(f"flags: EXL3_DEC_MOE_UNION={os.environ.get('EXL3_DEC_MOE_UNION')} ndt={gen_d.num_draft_tokens}")
    only = os.environ.get("ONLY")
    for name, prompt in prompts:
        if only and name not in only.split(","):
            continue
        pt, pl, _, tp = run(gen_p, tok, prompt, N)
        dt_, dl, last, td = run(gen_d, tok, prompt, N)
        print(f"[{name}] plain {len(pt)} tok {tp:.2f}s | dflash {len(dt_)} tok {td:.2f}s (incl prefill, cold-ish)")
        ref = None
        if os.environ.get("REF"):
            pids = tok.encode(prompt) if isinstance(prompt, str) else prompt
            full = torch.cat([pids.view(1, -1), torch.tensor([pt[:-1]], dtype=torch.long)], dim=-1)
            out = model.forward(full, {"attn_mode": "flash_attn_nc"})
            out = out["logits"] if isinstance(out, dict) else out
            ref = out[0, pids.shape[-1] - 1:].float().cpu()
            del out
            V = min(ref.shape[-1], pl.shape[-1])
            dp = (ref[:, :V] - pl[:len(pt), :V]).abs().nan_to_num(0, 0, 0).amax(-1)
            print(f"[{name}/ref] plain-vs-prefillref max|dlogit| per pos (first 48): "
                  + " ".join(f"{x:.3f}" for x in dp[:48].tolist()))
        compare(name, "real", pt, pl, dt_, dl, last, ref)
        if os.environ.get("PERFECT"):
            gt = torch.tensor(pt + [0] * 16, dtype=torch.long)
            real_fn = gen_d.iterate_draftmodel_dflash_gen

            def perfect(results, real_fn=real_fn, gt=gt):
                r = real_fn(results)
                if r is None:
                    return None
                s = gen_d.active_jobs[0].new_tokens
                w = r.shape[-1]
                if s + w > len(pt):
                    return r
                p = r.clone()
                p[0, :w] = gt[s:s + w].to(r.device)
                return p

            gen_d.iterate_draftmodel_dflash_gen = perfect
            bt, bl, blast, _ = run(gen_d, tok, prompt, N)
            del gen_d.iterate_draftmodel_dflash_gen
            compare(name, "perfect", pt, pl, bt, bl, blast, ref)
        if os.environ.get("SAVE"):
            dd = {"pt": pt, "pl": pl.half(), "dt": dt_, "dl": dl.half(), "dstats": last.get("draft_stats"),
                  "ref": ref.half() if ref is not None else None}
            if os.environ.get("PERFECT"):
                dd.update(bt=bt, bl=bl.half(), bstats=blast.get("draft_stats"))
            torch.save(dd, os.path.join(os.environ["SAVE"], f"lg-{name}.pt"))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
