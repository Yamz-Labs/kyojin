# DFlash2 smoke (GPU, run by hand): the target alone vs target + DFlash2 drafter, greedy, num_draft 7.
# The two token streams must be identical for every prompt; prints tok/s and the drafter's gain.
# usage: python -I smoke_dflash2.py ENGINE_DIR TARGET_DIR DRAFT_DIR OUT.json [NEW_TOKENS] [NUM_DRAFT]
# Needs a prompt_long.txt next to this file or at PROMPT_FILE (third prompt); falls back to a repeated paragraph.
# Exit code 0 = all streams identical, 1 = mismatch.
import sys, os, time, json
eng, mdir, ddir, out = sys.argv[1:5]
NEW = int(sys.argv[5]) if len(sys.argv) > 5 else 256
ND = int(sys.argv[6]) if len(sys.argv) > 6 else 7
sys.path.insert(0, eng)
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import ArgmaxSampler
import exllamav3

CTX = int(os.environ.get("SMOKE_CTX", 8192))
res = {"engine": exllamav3.__file__, "target": mdir, "draft": ddir, "num_draft": ND, "runs": []}

# Drafter first (as a server loads it): its cache holds the same number of tokens as the target's
dcfg = Config.from_directory(ddir); dm = Model.from_config(dcfg)
assert "dflash_draft" in dm.caps and dm.caps.get("attach_target"), "not a DFlash2 drafter"
dc = Cache(dm, max_num_tokens = CTX); dm.load()
cfg = Config.from_directory(mdir); model = Model.from_config(cfg)
# The target is hybrid (recurrent): a rejected draft rolls its state back from max_history snapshots
hist = max(dm.caps.get("default_draft_size", 4), ND)
cache = Cache(model, max_num_tokens = CTX, max_history = hist, max_batch_size = 1); model.load()
tok = Tokenizer.from_config(cfg)
S = ArgmaxSampler()

pf = os.environ.get("PROMPT_FILE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt_long.txt"))
base = open(pf).read() if os.path.exists(pf) else "The quick brown fox jumps over the lazy dog near the river bank. " * 120
tail = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
prompts = [
    "<|im_start|>user\nWrite a short story about a lighthouse keeper who finds a message in a bottle." + tail,
    "<|im_start|>user\nWrite a Python function that parses an ISO 8601 date string without using datetime, with tests." + tail,
    "<|im_start|>user\nSummarise the following text in five bullet points:\n" + base + tail,
]

def run(gen, ids, new):
    job = Job(input_ids = ids, max_new_tokens = new, sampler = S, stop_conditions = [])
    torch.cuda.synchronize(); t0 = time.time(); tf = None; o = []
    gen.enqueue(job)
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("stage") == "streaming":
                t = r.get("token_ids")
                if t is not None and t.numel():
                    if tf is None: torch.cuda.synchronize(); tf = time.time()
                    o += t.flatten().tolist()
    torch.cuda.synchronize()
    return o, tf - t0, time.time() - tf

def bench(gen, label):
    run(gen, tok.encode("Hello, warm-up."), 16)
    outs = []
    for p in prompts:
        ids = tok.encode(p)
        o, tp, td = run(gen, ids, NEW)
        r = {"mode": label, "n_in": ids.shape[-1], "ttft_s": tp, "n_out": len(o), "decode_s": td,
             "decode_tps": (len(o) - 1) / td, "ids": o}
        res["runs"].append(r); outs.append(o)
        print(json.dumps({k: v for k, v in r.items() if k != "ids"}), flush = True)
    return outs

plain = bench(Generator(model = model, cache = cache, tokenizer = tok), "plain")
gen = Generator(model = model, cache = cache, tokenizer = tok, draft_model = dm, draft_cache = dc, num_draft_tokens = ND)
assert gen.dflash_draft, "generator did not pick the DFlash path"
spec = bench(gen, "dflash2")

ok = True
for i, (a, b) in enumerate(zip(plain, spec)):
    n = min(len(a), len(b)); first = next((j for j in range(n) if a[j] != b[j]), None)
    same = first is None and len(a) == len(b)
    ok &= same
    tp = res["runs"][i]["decode_tps"]; ts = res["runs"][3 + i]["decode_tps"]
    print(f"prompt {i}: identical={same} first_diff={first} plain={tp:.1f} tok/s dflash2={ts:.1f} tok/s x{ts / tp:.2f}", flush = True)
res["identical"] = ok
json.dump(res, open(out, "w"), indent = 1)
print("RESULT", "OK identical" if ok else "MISMATCH", flush = True)
sys.exit(0 if ok else 1)
