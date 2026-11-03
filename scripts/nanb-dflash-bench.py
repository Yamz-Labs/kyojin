#!/usr/bin/env python
"""DFlash2 drafter on a Llama-architecture target (Nanbeige4.3-33B EXL3): decode N=1 tok/s, acceptance, losslessness.

One process, one load of target + drafter. Per prompt kind (code / chat / agentic), per prompt: a plain greedy decode
and a drafted greedy decode on the same rendered prompt, then the token-by-token comparison and the first divergence
with the plain path's top-2 logit gap there (a near-tie is the documented caveat of R-row verify numerics).

  PYTHONPATH=$PWD python scripts/nanb-dflash-bench.py -m ~/models/nanbeige43-33b-exl3-4.0 \
      -dm ~/models/drafters/nanb-dflash2-exl3br --template <chat_template.jinja> --out /tmp/nanb_dflash.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "qwen"))
import serve  # noqa: E402  (the server's own template render and serving environment)

# the environment the server sets before exllamav3 is imported (verify-row kernels, graphs): the caller's environment wins
if os.environ.get("NANB_BENCH_SERVE_ENV", "1") != "0":
    for _k, _v in serve.SERVE_ENV + serve.CUDA_SERVE_ENV:
        if _k not in serve.HIP_ONLY_SERVE_ENV:
            os.environ.setdefault(_k, _v)
render_prompt = serve.render_prompt

import torch  # noqa: E402

from exllamav3 import Cache, CacheLayer_fp16, CacheLayer_quant, Config, Generator, GreedySampler, Job, Model, Tokenizer  # noqa: E402

TOOLS = [
    {"type": "function", "function": {"name": "terminal", "description": "Run a shell command and return its output.",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read a text file.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Write a text file.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["path", "content"]}}},
]
SYSTEM = "You are a coding agent. Use the tools to inspect and edit files in the working directory, then answer briefly."
CSV_SRC = ("import csv, sys\n\ndef load(path):\n    rows = []\n    with open(path) as f:\n        for r in csv.reader(f):\n"
           "            rows.append(r)\n    return rows\n\ndef total(rows, col):\n    return sum(float(r[col]) for r in rows)\n\n"
           "if __name__ == '__main__':\n    rows = load(sys.argv[1])\n    print(total(rows, 2))\n")

CODE = [
    "Write a Python class LRUCache with get and put in O(1), plus a short pytest file for it.",
    "Implement a thread-safe bounded blocking queue in Go with Put, Get and Close, and explain the locking.",
    "Write a bash script that rotates the files of a directory by timestamp, keeping the last 7, with a --dry-run flag.",
    "Write a Rust function that parses a CSV line with quoted fields and escaped quotes, with unit tests.",
]
CHAT = [
    "Explique-moi simplement la différence entre un processus et un thread, avec une analogie du quotidien.",
    "What are the main trade-offs between renting and buying a home in a high-interest-rate environment?",
    "Donne-moi un plan de trois jours pour visiter Lisbonne sans voiture, avec les quartiers à ne pas manquer.",
    "Write a short story, about 300 words, about a lighthouse keeper who finds a message in a bottle.",
]


def agentic_prompts():
    u = "Fix the bug in total.py: it crashes when a row is empty. Look at the file first."
    call = lambda i, n, a: {"role": "assistant", "content": "", "tool_calls": [  # noqa: E731
        {"id": f"c{i}", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}]}
    res = lambda i, t: {"role": "tool", "tool_call_id": f"c{i}", "content": t}  # noqa: E731
    sys_m = {"role": "system", "content": SYSTEM}
    return [
        [sys_m, {"role": "user", "content": u}],
        [sys_m, {"role": "user", "content": u}, call(1, "read_file", {"path": "total.py"}), res(1, CSV_SRC)],
        [sys_m, {"role": "user", "content": u}, call(1, "read_file", {"path": "total.py"}), res(1, CSV_SRC),
         call(2, "write_file", {"path": "total.py", "content": CSV_SRC.replace("for r in csv.reader(f):\n            rows.append(r)",
                                                                              "for r in csv.reader(f):\n            if r:\n                rows.append(r)")}),
         res(2, "ok"), call(3, "terminal", {"command": "python total.py data.csv"}), res(3, "Traceback (most recent call last):\n  File \"total.py\", line 14, in <module>\n    print(total(rows, 2))\nIndexError: list index out of range")],
        [sys_m, {"role": "user", "content": "List the python files under src/, show the line count of each, and tell me which is the biggest."}],
    ]


def run_job(gen, ids, new, want_top2=False):
    # want_top2: also keep the full logits rows (cpu) for the plain-vs-drafted noise floor
    job = Job(input_ids=ids, max_new_tokens=new, stop_conditions=gen.model.config.eos_token_id_list,
              sampler=GreedySampler(), return_logits=want_top2)
    gen.enqueue(job)
    t0 = time.perf_counter()
    t_first = None
    top2 = []
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("error"):
                raise RuntimeError(r["error"])
            if want_top2 and r.get("logits") is not None:
                top2.append(r["logits"][0].float().cpu())   # one chunk per emit
            if r.get("token_ids") is not None and t_first is None and r["token_ids"].numel():
                t_first = time.perf_counter()
    total = time.perf_counter() - t0
    seq = job.sequences[0]
    t = seq.sequence_ids.torch()
    out = (t[0] if t.dim() > 1 else t).tolist()[len(seq.input_ids):]
    stats = list(getattr(job, "draft_stats", []) or [])
    dec = total - ((t_first - t0) if t_first else 0.0)
    return dict(ids=out, n=len(out), total=total, ttft=(t_first - t0) if t_first else None,
                tps=(len(out) - 1) / dec if dec > 0 and len(out) > 1 else 0.0, logits=torch.cat(top2) if top2 else None,
                rounds=len(stats), accepted=sum(s[2] for s in stats), window=sum(s[1] for s in stats))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-m", required=True)
    p.add_argument("-dm", required=True)
    p.add_argument("--template", required=True)
    p.add_argument("--kv", default="6,6")
    p.add_argument("--cache", type=int, default=12288)
    p.add_argument("--ndt", type=int, default=7)
    p.add_argument("--new", type=int, default=256)
    p.add_argument("--think", action="store_true", help="render with enable_thinking (default off: answers right away)")
    p.add_argument("--kinds", default="code,chat,agentic")
    p.add_argument("--paths", action="store_true", help="count which LinearEXL3 kernel path each row count takes in a drafted job, then exit")
    p.add_argument("--noise", action="store_true", help="where plain and drafted decode split: compare each against a prefill reference (teacher-forced), then exit")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    kb, vb = (int(x) for x in a.kv.split(","))
    ndt = a.ndt

    t0 = time.time()
    config = Config.from_directory(a.m)
    model = Model.from_config(config)
    ckw = dict(layer_type=CacheLayer_fp16) if kb == 16 else dict(layer_type=CacheLayer_quant, k_bits=kb, v_bits=vb)
    cache = Cache(model, max_num_tokens=a.cache, max_history=max(4, ndt), **ckw)
    cache2 = Cache(model, max_num_tokens=a.cache, max_history=max(4, ndt), **ckw)
    model.load(progressbar=False)
    tok = Tokenizer.from_config(config)
    dconf = Config.from_directory(a.dm)
    draft = Model.from_config(dconf)
    dcache = Cache(draft, max_num_tokens=a.cache, max_history=max(4, ndt))
    draft.load(progressbar=False)
    draft.attach_to(model)
    print(f"load {time.time() - t0:.1f}s  vram {torch.cuda.memory_allocated() / 2**30:.2f} GiB  tap_shift={dconf.tap_shift}", flush=True)

    base = Generator(model=model, cache=cache, tokenizer=tok)
    drafted = Generator(model=model, cache=cache2, tokenizer=tok, draft_model=draft, draft_cache=dcache,
                        num_draft_tokens=ndt, dynamic_draft_tokens=False, record_draft_stats=True)
    template = open(a.template, encoding="utf-8").read()
    kw = {"enable_thinking": bool(a.think)}

    def enc(messages, tools):
        text = render_prompt(template, messages, tools, **kw)
        return tok.encode(text, add_bos=False, encode_special_tokens=True).contiguous()

    sets = {
        "code": [enc([{"role": "user", "content": q}], None) for q in CODE],
        "chat": [enc([{"role": "user", "content": q}], None) for q in CHAT],
        "agentic": [enc(m, TOOLS) for m in agentic_prompts()],
    }
    if a.noise:
        out = []
        for kind in a.kinds.split(","):
            for i, ids in enumerate(sets[kind]):
                r0 = run_job(base, ids, a.new, want_top2=True)
                r1 = run_job(drafted, ids, a.new, want_top2=True)
                n = min(len(r0["ids"]), len(r1["ids"]))
                first = next((j for j in range(n) if r0["ids"][j] != r1["ids"][j]), None)
                if first is None:
                    print(f"NOISE {kind}[{i}] identical", flush=True)
                    continue
                e0, e1, e01, gaps = [], [], [], []
                for pos in range(1, first + 1, 3):
                    ctx = torch.cat([ids, torch.tensor([r0["ids"][:pos]], dtype=ids.dtype)], dim=-1)
                    ref = run_job(base, ctx, 1, want_top2=True)["logits"][0].float()   # next-token logits after the context, one prefill
                    a0 = r0["logits"][pos]
                    a1 = r1["logits"][pos]
                    idx = ref.topk(20).indices
                    e0.append(float((a0[idx] - ref[idx]).abs().max()))
                    e1.append(float((a1[idx] - ref[idx]).abs().max()))
                    e01.append(float((a0[idx] - a1[idx]).abs().max()))
                    t2 = ref.topk(2).values
                    gaps.append(float(t2[0] - t2[1]))
                med = lambda v: sorted(v)[len(v) // 2]   # noqa: E731
                rec = dict(kind=kind, i=i, first=first, positions=len(e0), plain_vs_ref_med=med(e0), plain_vs_ref_max=max(e0),
                           draft_vs_ref_med=med(e1), draft_vs_ref_max=max(e1), plain_vs_draft_max=max(e01), min_ref_gap=min(gaps))
                out.append(rec)
                print("NOISE", json.dumps(rec), flush=True)
        json.dump(out, open(a.out, "w"), indent=1)
        return
    if a.paths:
        import collections
        from exllamav3.modules.quant.exl3 import LinearEXL3
        cnt = collections.Counter()
        def hook(name):
            orig = getattr(LinearEXL3, name)
            def f(self, x, *args, **kw):
                rows = x.numel() // x.shape[-1] if name != "dec_gemv_r" else args[0]
                cnt[(name, rows, str(self.key).split(".")[-1] if self.key else "?")] += 1
                return orig(self, x, *args, **kw)
            setattr(LinearEXL3, name, f)
        for n in ("dec_gemv", "dec_gemv_r", "reconstruct_hgemm"):
            hook(n)
        run_job(drafted, sets["code"][0], 48)
        for k, v in sorted(cnt.items()):
            print("PATH", k, v)
        return
    warm = sets["code"][0]
    run_job(base, warm, 24)
    run_job(drafted, warm, 24)

    res = {"model": a.m, "drafter": a.dm, "kv": a.kv, "ndt": ndt, "new": a.new, "think": a.think, "kinds": {}}
    for kind in a.kinds.split(","):
        rows = []
        for i, ids in enumerate(sets[kind]):
            r0 = run_job(base, ids, a.new, want_top2=True)
            r1 = run_job(drafted, ids, a.new, want_top2=True)
            n = min(len(r0["ids"]), len(r1["ids"]))
            first = next((j for j in range(n) if r0["ids"][j] != r1["ids"][j]), None)
            if first is None and len(r0["ids"]) != len(r1["ids"]):
                first = n
            gap = noise = dgap = None
            L0, L1 = r0["logits"], r1["logits"]
            if first is not None and L0 is not None and L1 is not None and 0 < first < min(L0.shape[0], L1.shape[0]):
                t2 = L0[first].topk(2).values
                gap = float(t2[0] - t2[1])                       # plain top-1 - top-2 where the texts split
                d2 = L1[first].topk(2).values
                dgap = float(d2[0] - d2[1])                      # same in the drafted run (its own logits at that step)
                dd = (L0[:first] - L1[:first]).abs()
                noise = float(dd[torch.isfinite(dd)].max())   # largest logit difference on the shared prefix
            row = dict(i=i, prompt_tokens=int(ids.shape[-1]), plain_tps=r0["tps"], draft_tps=r1["tps"], n_plain=r0["n"], n_draft=r1["n"],
                       identical=first is None, first_diff=first, plain_gap=gap, draft_gap=dgap, prefix_dlogit_max=noise, rounds=r1["rounds"],
                       accepted=r1["accepted"], window=r1["window"],
                       mean_len=(r1["n"] / r1["rounds"]) if r1["rounds"] else None)
            rows.append(row)
            print(f"{kind}[{i}] ptok={row['prompt_tokens']} plain {r0['tps']:.1f} dflash {r1['tps']:.1f} tok/s "
                  f"({r1['tps'] / max(r0['tps'], 1e-9):.2f}x) mean_len {row['mean_len']:.2f} identical={row['identical']} "
                  f"first_diff={first} gap={gap} draft_gap={dgap} prefix_dlogit_max={noise}", flush=True)
        res["kinds"][kind] = dict(rows=rows, plain_tps=statistics.mean(r["plain_tps"] for r in rows),
                                  draft_tps=statistics.mean(r["draft_tps"] for r in rows),
                                  mean_len=statistics.mean(r["mean_len"] for r in rows if r["mean_len"]),
                                  identical=sum(r["identical"] for r in rows), n=len(rows))
    json.dump(res, open(a.out, "w"), indent=1)
    for k, v in res["kinds"].items():
        print(f"SUMMARY {k}: plain {v['plain_tps']:.1f} dflash {v['draft_tps']:.1f} tok/s ({v['draft_tps'] / v['plain_tps']:.2f}x) "
              f"mean accepted length {v['mean_len']:.2f} identical {v['identical']}/{v['n']}", flush=True)


if __name__ == "__main__":
    main()
