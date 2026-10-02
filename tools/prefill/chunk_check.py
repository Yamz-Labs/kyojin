"""Prefill chunk-size / MoE-path consistency through the Generator (the path the bench times).
For N long prompts, the first-token distribution (logits right after prefill) is taken under each
variant "name:chunk:EXL3_MOE_WMMA"; every variant is compared to the first: mean KLD, top-1 agreement.
The control pair (baseline MoE path at two chunk sizes) sets the noise floor of re-chunking."""
import argparse, os, torch
from exllamav3 import Generator, Job, GreedySampler, model_init
p = argparse.ArgumentParser(); model_init.add_args(p, cache = True)
p.add_argument("--corpus", default = "~/bench/ppl/wiki.test.raw")
p.add_argument("--len", type = int, default = 5000)
p.add_argument("--prompts", type = int, default = 6)
p.add_argument("--variants", nargs = "+",
               default = ["base2048:2048:0", "base4096:4096:0", "new2048:2048:1", "new4096:4096:1"])
args = p.parse_args(); torch.set_grad_enabled(False)
model, config, cache, tokenizer = model_init.init(args)[:4]
text = open(args.corpus, encoding = "utf-8").read()
prompts = [tokenizer.encode(text[1000000 + i * 60000: 1000000 + i * 60000 + args.len * 8])[:, :args.len]
           for i in range(args.prompts)]
res = {}
for v in args.variants:
    name, ch, wm = v.split(":")
    os.environ["EXL3_MOE_WMMA"] = wm
    gen = Generator(model = model, cache = cache, tokenizer = tokenizer, max_chunk_size = int(ch))
    out = []
    for i, ids in enumerate(prompts):
        gen.enqueue(Job(input_ids = ids, max_new_tokens = 1, sampler = GreedySampler(),
                        return_logits = True, identifier = i))
        lg = None
        while gen.num_remaining_jobs():
            for r in gen.iterate():
                if r.get("logits") is not None and lg is None:
                    lg = r["logits"].view(-1)[:config.vocab_size].float()
        out.append(lg)
    res[name] = torch.stack(out)
    del gen
    gen = None
    torch.cuda.empty_cache()
ref_name = args.variants[0].split(":")[0]
lp0 = torch.log_softmax(res[ref_name], -1)
for name in res:
    if name == ref_name: continue
    lp = torch.log_softmax(res[name], -1)
    kld = (lp0.exp() * (lp0 - lp)).sum(-1)
    agree = (res[name].argmax(-1) == res[ref_name].argmax(-1)).float().mean().item()
    print(f"{name} vs {ref_name}: KLD mean {kld.mean().item():.3e} max {kld.max().item():.3e} "
          f"top1 {agree * 100:.1f}% ({args.prompts} prompts x {args.len} tok)", flush = True)
