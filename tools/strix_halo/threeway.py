#!/usr/bin/env python
"""Three-way accuracy test: WMMA gemv vs reconstruct-hgemm vs fp32 golden.

Both paths are individually bit-exact, so the cross-path logit delta (max 0.85)
is a systematic rounding/accumulation difference, not nondeterminism. The
question that matters: WHICH IS CLOSER TO THE TRUTH?

Layer discovery: LinearEXL3 instances are reached via Linear.forward
(linear.py:622 -> exl3.py:178), not by walking .modules. So we capture the live
`self` objects from inside a spy on LinearEXL3.hip_gemv during one real forward.

Golden per layer, all fp32:
    xh32 = had_r_128(x32, suh32); w32 = reconstruct(trellis).float()
    y32  = had_r_128(xh32 @ w32, svh32)
"""
import os, sys
import torch

sys.path.insert(0, os.path.expanduser("~/kyojin"))
import exllamav3_ext as X
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
import exllamav3.modules.quant.exl3 as exl3mod
from exllamav3.modules.quant.exl3 import LinearEXL3

MODEL = os.path.expanduser("~/models/Qwen3.8-Flash-Next-EXL3")
print(f"family={X.exl3_gemv_wmma_family(0)}", flush=True)

config = Config.from_directory(MODEL)
model = Model.from_config(config)
tokenizer = Tokenizer.from_config(config)
cache = Cache(model, max_num_tokens=2048)
model.load(progressbar=False)

# --- capture live LinearEXL3 instances from a real forward ---
captured = {}
orig_hip = LinearEXL3.hip_gemv


def spy(self, x, out_dtype):
    key = (self.K, self.in_features, self.out_features, bool(self.mcg), bool(self.mul1))
    if key not in captured:
        captured[key] = self
    return orig_hip(self, x, out_dtype)


LinearEXL3.hip_gemv = spy
gen = Generator(model=model, cache=cache, tokenizer=tokenizer)
gen.enqueue(Job(input_ids=tokenizer.encode("hello world", add_bos=True), max_new_tokens=2))
while gen.num_remaining_jobs():
    for _ in gen.iterate():
        pass
LinearEXL3.hip_gemv = orig_hip

print(f"captured {len(captured)} distinct LinearEXL3 shapes", flush=True)
if not captured:
    print("no layers captured -- aborting")
    sys.exit(1)

torch.manual_seed(7)


def golden(lin, x):
    x32 = x.float()
    xh32 = torch.empty_like(x32)
    X.had_r_128(x32, xh32, lin.suh.float(), None, 1.0)
    w = torch.empty((lin.in_features, lin.out_features), dtype=torch.half,
                    device=lin.trellis.device)
    X.reconstruct(w, lin.trellis, lin.K, lin.mcg, lin.mul1)
    y32 = xh32 @ w.float()
    out = torch.empty_like(y32)
    X.had_r_128(y32, out, None, lin.svh.float(), 1.0)
    return out


print()
hdr = (f"{'K':>2} {'shape':>14} {'mcg':>3} {'mul1':>4} {'m':>3} "
       f"{'err_fast':>10} {'err_ref':>10} {'ratio':>7}  verdict")
print(hdr)
print("-" * len(hdr))

wins = {"fast": 0, "ref": 0, "tie": 0}
skipped = []
for key, lin in sorted(captured.items()):
    for m in (1, 2, 4, 8, 16):
        x = (torch.randn(m, lin.in_features, dtype=torch.half, device="cuda") * 0.05)
        try:
            g = golden(lin, x)
            exl3mod._hip_gemv_support_cache.clear()
            y_fast = lin.forward(x.clone(), {}, out_dtype=torch.float)
            saved = exl3mod._hip_gemv_supported
            exl3mod._hip_gemv_supported = lambda dev: False
            try:
                y_ref = lin.forward(x.clone(), {}, out_dtype=torch.float)
            finally:
                exl3mod._hip_gemv_supported = saved
        except Exception as e:
            skipped.append(f"K={lin.K} m={m}: {type(e).__name__}: {str(e)[:60]}")
            continue

        yf = y_fast.float().reshape(g.shape)
        yr = y_ref.float().reshape(g.shape)
        ef = (yf - g).abs().mean().item()
        er = (yr - g).abs().mean().item()
        ratio = ef / (er + 1e-12)
        if abs(ef - er) / max(er, 1e-12) < 0.02:
            verdict, k = "tie", "tie"
        elif ef < er:
            verdict, k = "FAST closer", "fast"
        else:
            verdict, k = "ref closer", "ref"
        wins[k] += 1
        print(f"{lin.K:>2} {str((lin.in_features, lin.out_features)):>14} "
              f"{int(bool(lin.mcg)):>3} {int(bool(lin.mul1)):>4} {m:>3} "
              f"{ef:>10.6f} {er:>10.6f} {ratio:>7.3f}  {verdict}")

print("-" * len(hdr))
print(f"fast closer: {wins['fast']}   ref closer: {wins['ref']}   tie(<2%): {wins['tie']}")
for s in skipped[:6]:
    print("  skipped:", s)
print()
tot = wins["fast"] + wins["ref"] + wins["tie"]
if tot and wins["fast"] + wins["tie"] >= wins["ref"]:
    print("CONCLUSION: the WMMA kernel is as accurate as (or more accurate than) the")
    print("reconstruct+hgemm fallback measured against fp32. The cross-path logit")
    print("delta is two equally-valid roundings; the fork's 27B-calibrated absolute")
    print("thresholds do not transfer to this MoE model.")
else:
    print("CONCLUSION: the fallback is closer to fp32 -- investigate accumulation order.")
