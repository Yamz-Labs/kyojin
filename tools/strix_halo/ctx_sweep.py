#!/usr/bin/env python
"""Context sweep: how far does the pack go on gfx1151, and what does decode cost at depth?

For each configured cache size, load the model + MTP head, prefill a synthetic prompt of
PROMPT_TOKENS, then decode NTOK greedy tokens. Reports GPU memory after load, prefill tok/s,
TTFT, decode tok/s and MTP acceptance. Sizes that fail to load are reported as OOM.

  CS=65536 PROMPTS=1024,8192,32768,60000 python ctx_sweep.py
"""
import os, sys, time, json
import torch
sys.path.insert(0, os.path.expanduser("~/kyojin"))
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler

MODEL = os.path.expanduser("~/models/Qwen3.8-Flash-Next-EXL3")
CS = int(os.environ.get("CS", "65536"))
PROMPTS = [int(x) for x in os.environ.get("PROMPTS", "1024,8192,32768").split(",")]
NTOK = int(os.environ.get("NTOK", "128"))
NDT = int(os.environ.get("NDT", "3")); DC = float(os.environ.get("DC", "0.6"))
CHUNK = int(os.environ.get("CHUNK", "2048"))
CQ = os.environ.get("CQ")
RANDOM_PROMPT = os.environ.get("RANDOM_PROMPT", "0") == "1"   # random token ids: defeats prefix caching, cold prefill
FILL = os.environ.get("FILL", "0") == "1"                     # add one prompt of exactly CS - NTOK - NDT - 8 tokens          # e.g. "4" or "6,4": quantized KV cache (k_bits[,v_bits]) on both caches
from exllamav3.cache import CacheLayer_quant
def mk_cache(m):
    if CQ:
        bits = [int(b) for b in CQ.split(",")]; kb = bits[0]; vb = bits[-1]
        return Cache(m, max_num_tokens=CS, max_history=NDT, layer_type=CacheLayer_quant, k_bits=kb, v_bits=vb)
    return Cache(m, max_num_tokens=CS, max_history=NDT)

config = Config.from_directory(MODEL)
model = Model.from_config(config); tok = Tokenizer.from_config(config)
try:
    cache = mk_cache(model); model.load(progressbar=False)
    draft = Model.from_config(config, component="mtp")
    dcache = mk_cache(draft); draft.load(progressbar=False)
except Exception as e:
    print(json.dumps({"cs": CS, "cq": CQ, "status": "OOM" if "VRAM" in str(e) or "memory" in str(e).lower() else "ERR", "err": str(e)[:200]}))
    sys.exit(0)
torch.cuda.synchronize()
mem = torch.cuda.memory_allocated() / 2**30
print(json.dumps({"cs": CS, "cq": CQ, "status": "loaded", "mem_gib": round(mem, 1),
                  "total_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)}), flush=True)

gen = Generator(model=model, cache=cache, tokenizer=tok, draft_model=draft, draft_cache=dcache,
                num_draft_tokens=NDT, dynamic_draft_tokens=True, draft_confidence=DC, max_chunk_size=CHUNK)

# Synthetic long document: varied prose so the n-gram / MTP path sees realistic text, not a loop
base = open(os.path.join(os.path.dirname(__file__), "..", "README.md")).read() if os.path.exists(os.path.join(os.path.dirname(__file__), "..", "README.md")) else ""
if len(base) < 2000:
    base = "The quick brown fox jumps over the lazy dog. " * 50
filler_ids = tok.encode(base, add_bos=False).flatten()
question = tok.encode("\n\nSummarize the document above in three sentences.", add_bos=False).flatten()

if FILL:
    PROMPTS.append(CS - NTOK - NDT - 8)
for P in PROMPTS:
    if P + NTOK + NDT + 8 > CS:
        print(json.dumps({"prompt": P, "status": "skip", "reason": f"exceeds cache {CS}"})); continue
    need = P - question.numel() - 1
    if RANDOM_PROMPT:
        g = torch.Generator().manual_seed(P)
        body = torch.randint(1000, 100000, (need,), generator=g, dtype=torch.long)
    else:
        reps = (need + filler_ids.numel() - 1) // filler_ids.numel()
        body = filler_ids.repeat(reps)[:need]
    ids = torch.cat([tok.encode("", add_bos=True).flatten()[:1], body, question]).view(1, -1)
    gen.clear_queue()
    try:
        cache.reset() if hasattr(cache, "reset") else None
    except Exception:
        pass
    gen.enqueue(Job(input_ids=ids, max_new_tokens=NTOK, sampler=GreedySampler()))
    t0 = time.perf_counter(); ttft = None; n = 0; acc = rej = 0; err = None
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("error"): err = repr(r["error"])[:160]
            if r.get("text"):
                if ttft is None: ttft = time.perf_counter() - t0
                n += 1
            if "accepted_draft_tokens" in r:
                acc += r["accepted_draft_tokens"]; rej += r["rejected_draft_tokens"]
    torch.cuda.synchronize(); total = time.perf_counter() - t0
    out = {"prompt": int(ids.numel()), "status": "ok" if err is None else "err", "ttft_s": round(ttft, 2) if ttft else None,
           "prefill_tps": round(ids.numel() / ttft) if ttft else None,
           "decode_tps": round((n - 1) / (total - ttft), 2) if ttft and n > 1 else None,
           "acceptance": round(100 * acc / max(acc + rej, 1), 1), "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
    if err: out["err"] = err
    print(json.dumps(out), flush=True)
