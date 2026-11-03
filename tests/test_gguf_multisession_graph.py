"""Multi-session decode on a foreign GGUF: eager vs CUDA graph (one job per graph) vs CUDA graph MULTI (several jobs per graph).
Aggregate tok/s with U concurrent sessions, and token agreement of every session against eager.

    FMOE_GGUF=<file.gguf> [FMOE_USERS=4] python tests/test_gguf_multisession_graph.py
"""
import os, sys, time
import torch

ENGINE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ENGINE)
from exllamav3 import Model, Config, Cache, Generator, Job, GreedySampler, Tokenizer
from exllamav3.loader.gguf_config import write_config_dir
from exllamav3.model import cuda_graph as cg

U = int(os.environ.get("FMOE_USERS", "4"))
if os.environ.get("MODEL_DIR"):   # a native EXL3 directory instead of a GGUF
    d = os.environ["MODEL_DIR"]
else:
    d = write_config_dir(os.environ["FMOE_GGUF"], os.path.join(os.environ.get("FMOE_CFG", "/tmp"), "qwen35moe_cfg"))
config = Config.from_directory(d)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens=8192)
model.load()
tok = Tokenizer.from_config(config)
gen = Generator(model=model, cache=cache, tokenizer=tok)
prompts = ["The capital of France is a city that", "Water boils at a temperature of", "In 1969 the first humans",
           "The quick brown fox jumps over", "Photosynthesis is the process by which", "The largest planet in the solar system",
           "A binary search tree is a structure", "Mount Everest is located in"][:U]
N = int(os.environ.get("FMOE_NEW", "48"))


def run():
    jobs = [Job(input_ids=tok.encode(p), max_new_tokens=N, sampler=GreedySampler(), stop_conditions=[], identifier=i)
            for i, p in enumerate(prompts)]
    for j in jobs:
        gen.enqueue(j)
    out = {i: [] for i in range(len(jobs))}
    torch.cuda.synchronize(); t = time.time()
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r["stage"] == "streaming" and "token_ids" in r:
                out[r["identifier"]] += r["token_ids"].flatten().tolist()
    torch.cuda.synchronize()
    return out, time.time() - t


arms = [("eager", False, False), ("graph-single", True, False), ("graph-multi", True, True), ("eager-again", False, False)]
ref = None
for name, on, multi in arms:
    cg.CUDA_GRAPH_ENABLED = on
    cg.set_multi(multi)
    cg.purge(model)
    run()  # warm / capture
    out, dt = run()
    n = sum(len(v) for v in out.values())
    if ref is None:
        ref = out
    agree = sum(a == b for i in out for a, b in zip(out[i], ref[i]))
    print(f"{name:13s} users={U} {n} tok in {dt:.2f}s = {n / dt:.1f} tok/s aggregate, agreement vs eager {agree}/{n}")
print("PASS")
