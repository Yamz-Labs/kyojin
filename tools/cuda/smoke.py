# Kyojin CUDA smoke (tools/cuda/README.md): load an EXL3 model with the engine at ENGINE (sys.path), greedy plain decode, time prefill and decode.
# usage: python -I tools/cuda/smoke.py ENGINE_DIR MODEL_DIR OUT.json [NEW_TOKENS]
import sys, os, time, json
eng, mdir, out = sys.argv[1], sys.argv[2], sys.argv[3]; NEW = int(sys.argv[4]) if len(sys.argv) > 4 else 256
sys.path.insert(0, eng)
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator
from exllamav3.cache import CacheLayer_quant
from exllamav3.generator.sampler import ArgmaxSampler
import exllamav3
res = {"engine": exllamav3.__file__, "version": exllamav3.__version__ if hasattr(exllamav3, "__version__") else None,
       "model": mdir, "env": {k: v for k, v in os.environ.items() if k.startswith("EXL3_")}, "runs": []}
t0 = time.time()
MTP = int(os.environ.get("SMOKE_MTP", "0")); POL = os.environ.get("SMOKE_POLICY", "fixed")   # MTP drafts (0 = plain), fixed | mix
cfg = Config.from_directory(mdir); model = Model.from_config(cfg)
KV = os.environ.get("SMOKE_KV", "16")   # 16 or K,V bits of the quantized attention cache (e.g. 4,4)
qkw = {} if KV == "16" else dict(layer_type=CacheLayer_quant, k_bits=int(KV.split(",")[0]), v_bits=int(KV.split(",")[1]))
cache = Cache(model, max_num_tokens=8192, **qkw, **({"max_history": max(3, MTP), "max_batch_size": 1} if MTP else {})); model.load(); tok = Tokenizer.from_config(cfg)
kw = {}
DFL = os.environ.get("SMOKE_DFLASH")   # DFlash2 drafter dir (upstream 1.6.0 only): a served speculative config
if DFL:
    dcfg = Config.from_directory(DFL); dm = Model.from_config(dcfg); dc = Cache(dm, max_num_tokens=16 * 256); dm.load()
    dc.max_num_tokens = 8192   # ring cache as in a served drafter: the generator only compares sizes
    kw = dict(draft_model=dm, draft_cache=dc, num_draft_tokens=MTP or 7)
if MTP and not DFL:
    dm = Model.from_config(cfg, component="mtp"); dc = Cache(dm, max_num_tokens=8192, max_history=max(3, MTP)); dm.load()
    kw = dict(draft_model=dm, draft_cache=dc, num_draft_tokens=MTP)
res["load_s"] = time.time() - t0; res["vram_gib"] = torch.cuda.max_memory_allocated() / 2**30
gen = Generator(model=model, cache=cache, tokenizer=tok, **kw)
res["mtp"] = MTP; res["policy"] = POL; res["kv"] = KV
if MTP and POL == "mix":
    sys.path.insert(0, os.path.join(eng, "tools", "qwen")); import spec_policy
    thf, thv, maxd = spec_policy.env_defaults(MTP)
    spec_policy.install(gen, dm, model, thf, thv, maxd, (2, 3, 5), {}, rule=spec_policy.rule_from_env(thf, thv, maxd), fixed=None)
S = ArgmaxSampler()
base = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt_long.txt")).read()
prompts = [
  "<|im_start|>user\nWrite a short story about a lighthouse keeper who finds a message in a bottle.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
  "<|im_start|>user\nWrite a Python function that parses an ISO 8601 date string without using datetime, with tests.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
  "<|im_start|>user\nSummarise the following text in five bullet points:\n" + base + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
]
from exllamav3 import Job
def run(ids, new):
    job = Job(input_ids=ids, max_new_tokens=new, sampler=S, stop_conditions=[])
    torch.cuda.synchronize(); t0 = time.time(); tf = None; out = []
    gen.enqueue(job)
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("stage") == "streaming":
                t = r.get("token_ids")
                if t is not None and t.numel():
                    if tf is None: torch.cuda.synchronize(); tf = time.time()
                    out += t.flatten().tolist()
    torch.cuda.synchronize(); te = time.time()
    return out, tf - t0, te - tf
run(tok.encode("Hello, warm-up."), 16)  # warm-up (graphs, tuners)
for p in prompts:
    ids = tok.encode(p); n_in = ids.shape[-1]
    oids, tp, td = run(ids, NEW)
    r = {"n_in": n_in, "ttft_s": tp, "prefill_tps": n_in / tp, "decode_s": td, "decode_tps": (len(oids) - 1) / td,
         "n_out": len(oids), "ids": oids, "text": tok.decode(torch.tensor([oids]))[0][:600] if oids else ""}
    res["runs"].append(r); print(json.dumps({k: v for k, v in r.items() if k not in ("text", "ids")}), flush=True)
res["max_vram_gib"] = torch.cuda.max_memory_allocated() / 2**30
json.dump(res, open(out, "w"), indent=1)
