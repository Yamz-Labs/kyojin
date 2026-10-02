"""Record real per-expert routing counts of one prefill forward (EXL3_MOE_WMMA_STATS=1)."""
import argparse, os, sys, torch
os.environ["EXL3_MOE_WMMA_STATS"] = "1"
from exllamav3 import model_init
import exllamav3.modules.block_sparse_mlp as bsm
p = argparse.ArgumentParser(); model_init.add_args(p, cache = True)
p.add_argument("--corpus", default = "~/bench/ppl/wiki.test.raw")
p.add_argument("--len", type = int, nargs = "+", default = [2048, 4096])
p.add_argument("--out", required = True)
args = p.parse_args(); torch.set_grad_enabled(False)
model, config, cache, tokenizer = model_init.init(args)[:4]
text = open(args.corpus, encoding = "utf-8").read()
res = {}
for L in args.len:
    bsm._HIP_WMMA_STATS.clear()
    ids = tokenizer.encode(text[800000: 800000 + L * 8])[:, :L]
    model.forward(ids, {"attn_mode": "flash_attn_nc"})
    res[L] = torch.stack([c for _, c in bsm._HIP_WMMA_STATS])
    c = res[L].float()
    t128 = ((c + 127) // 128).sum(1).mean().item(); t256 = ((c + 255) // 256).sum(1).mean().item()
    print(f"len {L}: layers {c.shape[0]}  active/layer {(c > 0).sum(1).float().mean():.1f}  max {c.max():.0f}  "
          f"tiles128/layer {t128:.1f}  tiles256/layer {t256:.1f}  ideal128 {L * 8 / 128:.1f}", flush = True)
torch.save(res, args.out)
for L, c in res.items():
    for i, row in enumerate(c.tolist()):
        print(f"COUNTS {L} {i} " + " ".join(map(str, row)))
