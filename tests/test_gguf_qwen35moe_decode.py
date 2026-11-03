"""Decode a plain qwen35moe GGUF through the generator: grouped mixed-format MoE kernel vs per-expert path, tok/s each.
Text is meaningless on a truncated cut; judge the kernel by token agreement and speed.

    FMOE_GGUF=<file.gguf> python tests/test_gguf_qwen35moe_decode.py
"""
import json, os, sys, time
import torch

ENGINE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ENGINE)
from exllamav3 import Model, Config, Cache, Generator, Job, GreedySampler, Tokenizer
from exllamav3.loader.gguf_config import write_config_dir


src = os.environ["FMOE_GGUF"]
d = write_config_dir(src, os.path.join(os.environ.get("FMOE_CFG", "/tmp"), "qwen35moe_cfg"))
config = Config.from_directory(d)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens=4096)
model.load()
tok = Tokenizer.from_config(config)
gen = Generator(model=model, cache=cache, tokenizer=tok)
ids = tok.encode("The capital of France is a city that")
N = 64
res = {}
for grouped in ("1", "0", "1"):
    os.environ["MCPM_FQ_MOE_GROUPED"] = grouped
    gen.generate("The capital of France", max_new_tokens=4, completion_only=True)  # warm
    torch.cuda.synchronize(); t = time.time()
    job = Job(input_ids=ids, max_new_tokens=N, sampler=GreedySampler(), stop_conditions=[])
    gen.enqueue(job)
    toks = []
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r["stage"] == "streaming" and "token_ids" in r:
                toks += r["token_ids"].flatten().tolist()
    torch.cuda.synchronize(); dt = time.time() - t
    res[grouped] = toks
    print(("grouped " if grouped == "1" else "per-expert"), f"{len(toks)} tok in {dt:.2f}s = {len(toks) / dt:.1f} tok/s")
n = min(len(res["1"]), len(res["0"]))
agree = sum(a == b for a, b in zip(res["1"], res["0"]))
print(f"token agreement {agree}/{n}")
if os.environ.get("FMOE_TOKOUT"):
    json.dump(res["1"], open(os.environ["FMOE_TOKOUT"], "w"))
print("PASS")
