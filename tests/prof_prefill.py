"""torch-profiler op-class split of one prefill. MODEL_DIR | FMOE_GGUF, PREFILL_L (default 8192)."""
import os, sys
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from exllamav3 import Model, Config, Cache, Generator, Job, GreedySampler, Tokenizer
from exllamav3.loader.gguf_config import write_config_dir
from torch.profiler import profile, ProfilerActivity
L = int(os.environ.get("PREFILL_L", "8192"))
d = os.environ.get("MODEL_DIR") or write_config_dir(os.environ["FMOE_GGUF"], os.path.join(os.environ.get("FMOE_CFG", "/tmp"), "qwen35moe_cfg"))
config = Config.from_directory(d); model = Model.from_config(config)
cache = Cache(model, max_num_tokens=(L + 1024) // 256 * 256 + 256); model.load()
tok = Tokenizer.from_config(config); gen = Generator(model=model, cache=cache, tokenizer=tok)
g = torch.Generator().manual_seed(2)
def run(seed):
    ids = torch.randint(1000, 100000, (1, L), generator=g)
    gen.enqueue(Job(input_ids=ids, max_new_tokens=1, sampler=GreedySampler(), stop_conditions=[]))
    while gen.num_remaining_jobs(): gen.iterate()
run(0); torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as p:
    run(1); torch.cuda.synchronize()
rows = sorted(p.key_averages(), key=lambda e: -e.device_time_total)[:14]
tot = sum(e.device_time_total for e in p.key_averages())
print(f"total device time {tot / 1e3:.0f} ms")
for e in rows: print(f"{100 * e.device_time_total / tot:5.1f}%  {e.device_time_total / 1e3:8.1f} ms  n={e.count:5d}  {e.key[:90]}")
