"""Prefill throughput vs the GEMM roof. Prints, per prompt length: tok/s, the GEMM TFLOPs that rate implies (2 x active params
per token; attention FLOPs not counted, so the gap at 32K includes them) and its share of the device's measured fp16 matmul peak.

    MODEL_DIR=<exl3 dir> | FMOE_GGUF=<file.gguf>  [PREFILL_LENS=8192,32768] python tests/bench_prefill_roof.py
"""
import os, sys, time
import torch

ENGINE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ENGINE)
from exllamav3 import Model, Config, Cache, Generator, Job, GreedySampler, Tokenizer
from exllamav3.loader.gguf_config import write_config_dir
from exllamav3.modules import BlockSparseMLP

lens = [int(v) for v in os.environ.get("PREFILL_LENS", "8192,32768").split(",")]
if os.environ.get("MODEL_DIR"):
    d = os.environ["MODEL_DIR"]
else:
    d = write_config_dir(os.environ["FMOE_GGUF"], os.path.join(os.environ.get("FMOE_CFG", "/tmp"), "qwen35moe_cfg"))
config = Config.from_directory(d)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens=(max(lens) + 1024) // 256 * 256 + 256)
model.load()
tok = Tokenizer.from_config(config)
gen = Generator(model=model, cache=cache, tokenizer=tok)


def active_params(mod, scale=1.0):
    n = 0.0
    if isinstance(mod, BlockSparseMLP):
        k = mod.num_experts_per_tok / mod.num_experts
        keep = {id(getattr(mod, a, None)) for a in ("shared_experts", "shared_gate", "routing_gate", "latent_in", "latent_out")}
        for c in mod.modules:
            n += active_params(c, scale if id(c) in keep else scale * k)
        return n
    if hasattr(mod, "in_features") and hasattr(mod, "out_features") and not mod.modules:
        return scale * mod.in_features * mod.out_features
    for c in mod.modules:
        n += active_params(c, scale)
    return n


dev = torch.device("cuda")
a = torch.randn(8192, 8192, device=dev, dtype=torch.half)
b = torch.randn(8192, 8192, device=dev, dtype=torch.half)
for _ in range(3): a @ b
torch.cuda.synchronize(); t = time.time()
for _ in range(20): a @ b
torch.cuda.synchronize()
peak = 20 * 2 * 8192 ** 3 / (time.time() - t) / 1e12
del a, b
act = sum(active_params(m) for m in model.modules)
print(f"device fp16 matmul peak {peak:.1f} TFLOPs; active params/token {act / 1e9:.2f} B -> GEMM roof {peak * 1e12 / (2 * act):.0f} tok/s")

g = torch.Generator().manual_seed(1)
vocab = min(config.vocab_size, 100000)
for L in [1024] + lens:
    for rep in range(2 if L == 1024 else 1):
        ids = torch.randint(1000, vocab, (1, L), generator=g)   # random ids: no prefix-cache hit
        job = Job(input_ids=ids, max_new_tokens=1, sampler=GreedySampler(), stop_conditions=[])
        gen.enqueue(job)
        torch.cuda.synchronize(); t = time.time()
        while gen.num_remaining_jobs():
            gen.iterate()
        torch.cuda.synchronize(); dt = time.time() - t
    if L != 1024:
        r = L / dt
        print(f"prefill L={L}: {dt:.2f}s = {r:.0f} tok/s, {2 * act * r / 1e12:.1f} TFLOPs-equiv = {100 * 2 * act * r / 1e12 / peak:.0f} % of matmul peak")
