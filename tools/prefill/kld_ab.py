"""End-to-end A/B of the grouped WMMA MoE prefill on the full MiMo model, one model load.

Teacher-forced: each 2048-token wiki slice runs through model.forward twice, EXL3_MOE_WMMA=0
(per-expert reconstruct/GEMV path = baseline build) then =1, and the full-vocab distributions
are compared: mean KLD(base || new), top-1 agreement, NLL of both. Then greedy generation on
three prompts with the new path (one long prompt so the prefill actually takes it).

  big-gpu-run.sh tools/prefill/env-run.sh python tools/prefill/kld_ab.py
"""
import argparse, os, sys, time
import torch
from exllamav3 import Generator, Job, GreedySampler, model_init

p = argparse.ArgumentParser()
model_init.add_args(p, cache = True)
p.add_argument("--corpus", default = "~/bench/ppl/wiki.test.raw")
p.add_argument("--slices", type = int, default = 2)
p.add_argument("--len", type = int, default = 2048)
p.add_argument("--no-gen", action = "store_true")
p.add_argument("--variants", nargs = "+", default = ["base:EXL3_MOE_WMMA=0", "new:EXL3_MOE_WMMA=1"],
               help = "name:ENV=v+ENV=v ... ; every variant is compared to the first")
args = p.parse_args()
torch.set_grad_enabled(False)
import exllamav3_ext
print("ext", exllamav3_ext.__file__, flush = True)

model, config, cache, tokenizer = model_init.init(args)[:4]
text = open(args.corpus, encoding = "utf-8").read()

VARS = []
for v in args.variants:
    name, _, envs = v.partition(":")
    VARS.append((name, dict(e.split("=", 1) for e in envs.split("+") if e)))
ALL_KEYS = sorted({k for _, e in VARS for k in e})

def run(ids, env):
    for k in ALL_KEYS:
        os.environ.pop(k, None)
    os.environ.update(env)
    torch.cuda.synchronize(); t0 = time.time()
    logits = model.forward(ids, {"attn_mode": "flash_attn_nc"})
    torch.cuda.synchronize()
    return logits[0].float(), time.time() - t0

stats = {name: {"kld": 0.0, "agree": 0.0, "nll": 0.0, "t": 0.0} for name, _ in VARS}
n_pos = 0
for s in range(args.slices):
    off = 700000 + s * 40000
    ids = tokenizer.encode(text[off: off + args.len * 8])[:, :args.len]
    tgt = ids[0, 1:].cuda()
    lp0 = None
    for name, env in VARS:
        lg, dt = run(ids, env)
        lp = torch.log_softmax(lg, -1)
        st = stats[name]
        st["t"] += dt
        st["nll"] += -lp[:-1].gather(1, tgt[:, None]).sum().item()
        if lp0 is None:
            lp0, am0 = lp, lg.argmax(-1)
        else:
            kld = (lp0.exp() * (lp0 - lp)).sum(-1)
            st["kld"] += kld.sum().item()
            st["agree"] += (lg.argmax(-1) == am0).float().sum().item()
            print(f"slice {s} {name}: KLD mean {kld.mean().item():.3e} max {kld.max().item():.3e}  "
                  f"top1 {(lg.argmax(-1) == am0).float().mean().item()*100:.2f}%  forward {dt:.2f}s", flush = True)
        del lg, lp
    n_pos += args.len
    del lp0
n_tgt = args.slices * (args.len - 1)
for name, _ in VARS:
    st = stats[name]
    print(f"TOTAL {name}: KLD mean {st['kld'] / n_pos:.3e}  top1 {st['agree'] / n_pos * 100:.2f}%  "
          f"PPL {torch.exp(torch.tensor(st['nll'] / n_tgt)).item():.4f}  forward {st['t']:.2f}s", flush = True)

if not args.no_gen:
    os.environ["EXL3_MOE_WMMA"] = "1"
    gen = Generator(model = model, cache = cache, tokenizer = tokenizer)
    passage = text[900000: 900000 + 6000]
    prompts = [
        "Write a haiku about the ocean.",
        "Explain in three sentences why the sky is blue.",
        "Summarize the following text in two sentences.\n\n" + passage,
    ]
    for pr in prompts:
        ids = tokenizer.hf_chat_template(messages = [{"role": "user", "content": pr}],
                                         add_generation_prompt = True, enable_thinking = False)
        job = Job(input_ids = ids, max_new_tokens = 96, sampler = GreedySampler())
        gen.enqueue(job)
        txt = ""
        while gen.num_remaining_jobs():
            for r in gen.iterate():
                txt += r.get("text", "")
        print(f"\n=== prompt ({ids.shape[-1]} tok): {pr[:60]!r}\n{txt}", flush = True)
