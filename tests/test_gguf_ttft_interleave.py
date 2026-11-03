"""Multi-user TTFT with a big prefill in flight: a short request arrives while a long prompt prefills. Reports the short job's
time to first token and the long job's total time. Run once per EXL3_SCHED_CHUNK_IL value (0/1, optionally EXL3_CIL_DECODE_PRIO=1).

    FMOE_GGUF=<file.gguf> [FMOE_LONG=24000] EXL3_SCHED_CHUNK_IL=1 python tests/test_gguf_ttft_interleave.py
"""
import hashlib, os, sys, time
import torch

ENGINE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ENGINE)
from exllamav3 import Model, Config, Cache, Generator, Job, GreedySampler, Tokenizer
from exllamav3.loader.gguf_config import write_config_dir

L = int(os.environ.get("FMOE_LONG", "24000"))
if os.environ.get("MODEL_DIR"):   # a native EXL3 directory instead of a GGUF
    d = os.environ["MODEL_DIR"]
else:
    d = write_config_dir(os.environ["FMOE_GGUF"], os.path.join(os.environ.get("FMOE_CFG", "/tmp"), "qwen35moe_cfg"))
config = Config.from_directory(d)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens=(L + 4096) // 256 * 256 + 256)
model.load()
tok = Tokenizer.from_config(config)
gen = Generator(model=model, cache=cache, tokenizer=tok)
short_ids = tok.encode("What is two plus two?")


def run(seed):
    # a fresh prompt per run: an identical one would hit the prefix cache and prefill nothing
    long_ids = tok.encode(f"Story {seed}. " + "The history of the Roman empire is long. " * (L // 8))[:, :L]
    t0 = time.time()
    ja = Job(input_ids=long_ids, max_new_tokens=8, sampler=GreedySampler(), stop_conditions=[], identifier="long")
    jb = Job(input_ids=short_ids, max_new_tokens=16, sampler=GreedySampler(), stop_conditions=[], identifier="short")
    first = {}
    toks = {"long": [], "short": []}
    done = {}
    sent = False
    gen.enqueue(ja)
    gen.iterate()   # first chunk of the long prefill is in flight, then the short request arrives
    while gen.num_remaining_jobs():
        if not sent and time.time() - t0 > 0.0:
            gen.enqueue(jb); sent = True; tb = time.time()
        results = gen.iterate()
        for r in results:
            i = r["identifier"]
            if r["stage"] == "streaming" and "token_ids" in r and i not in first:
                first[i] = time.time()
            if r["stage"] == "streaming" and "token_ids" in r:
                toks[i] += r["token_ids"].flatten().tolist()
            if r.get("eos"):
                done[i] = time.time()
    torch.cuda.synchronize()
    h = " ".join(hashlib.md5(repr(toks[k]).encode()).hexdigest()[:6] + f"({k})" for k in ("long", "short")) + " L=" + str(toks["long"]) + " S=" + str(toks["short"])
    return first["short"] - tb, done["long"] - t0, h


run(1)
a, b, h = run(2)
print(f"CHUNK_IL={os.environ.get('EXL3_SCHED_CHUNK_IL', '0')} DPRIO={os.environ.get('EXL3_CIL_DECODE_PRIO', '0')} long={L}: short TTFT {a:.2f}s, long total {b:.2f}s, tokens {h}")
