"""Multi-user decode on the HIP engine: NB concurrent greedy requests in one batch, aggregate tok/s = all new tokens / wall of the batch.
usage: mu_hip.py MODEL_DIR [NB] [max_new]     A/B env: BOLT_FQ_ROWS_TRITON=0|1"""
import sys, time
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator

d = sys.argv[1]
nb = int(sys.argv[2]) if len(sys.argv) > 2 else 4
new = int(sys.argv[3]) if len(sys.argv) > 3 else 200
config = Config.from_directory(d)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens=8192 * nb, max_batch_size=nb)
model.load(progressbar=False)
tok = Tokenizer.from_config(config)
print("alloc GiB", round(torch.cuda.memory_allocated() / 2 ** 30, 1), flush=True)
gen = Generator(model=model, cache=cache, tokenizer=tok)
base = ["Explain how a hash map handles collisions, with a short Python example.",
        "Écris un court paragraphe en français sur l'histoire du phare d'Alexandrie.",
        "Write a Python function that parses an ISO 8601 date string without external libraries.",
        "Describe the water cycle for a ten-year-old.",
        "List five differences between TCP and UDP.",
        "Summarize the plot of Hamlet in one paragraph.",
        "How does a transistor work?",
        "Give three tips for writing clean Python code."]
fmt = lambda p: f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
gen.generate(prompt=[fmt("Hi")] * nb, max_new_tokens=8, completion_only=True)            # warm-up, dropped
for N in (1, nb):
    ps = [fmt(base[i % len(base)]) for i in range(N)]
    t = time.time()
    outs = gen.generate(prompt=ps, max_new_tokens=new, completion_only=True, stop_conditions=[])
    dt = time.time() - t
    outs = outs if isinstance(outs, list) else [outs]
    n = sum(tok.encode(o).shape[-1] for o in outs)
    print(f"N{N} agg {n / dt:.2f} tok/s  n {n}  wall {dt:.1f}s  head0 {outs[0][:80]!r}", flush=True)
