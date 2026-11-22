"""Inkling2 plain decode tok/s at 1K / 8K depth with OUR forward (loader/inkling2_forward.py on the GPU through torch: the step-0 reference path,
no fused kernels yet: experts are read raw + dequantised per picked expert, attention is torch). The number is the BASELINE the op agents beat.

  decode_job.py --gguf G [--depths 1024,8192] [--steps 12] [--out DIR] [--device cuda]

The KV state of every application is filled with synthetic K / V (random, normalised like real keys) to the requested depth instead of a prefill:
decode timing at depth, not quality. Fires-check (exit 7): the NVFP4 expert path must have run (counter of experts dequantised from type 40), the
loop must have run 48 applications per step, and tok/s must be > 0. Output DIR/decode_<model>.json + `FIRES decode <n>`."""
import argparse
import json
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cpu_env  # noqa: E402

TREE = os.path.abspath(os.path.join(HERE, "..", ".."))
H = os.environ.get("INK_H", os.path.expanduser("~"))
if os.environ.get("INK_GUARD_DIR"):
    sys.path.insert(0, os.environ["INK_GUARD_DIR"])      # optional runner-side drv_guard; default = the one next to this file
import drv_guard as guard  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--depths", default="1024,8192")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--out", default="ink_decode_out")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    F = cpu_env.install(TREE)
    from exllamav3.loader.gguf_collection import GGUFCollection
    counts = {"apps": 0, "experts_nvfp4": 0, "experts": 0}
    orig_read_expert = F.read_expert

    def counting_read_expert(g, name, e, dev=None):
        counts["experts"] += 1
        counts["experts_nvfp4"] += g.infos[name][0] == 40
        return orig_read_expert(g, name, e, dev)
    F.read_expert = counting_read_expert

    def hook(name, il, t):
        if name == "l_out":
            counts["apps"] += 1
    m = F.Inkling2Forward(GGUFCollection(a.gguf), device=a.device, hook=hook)
    dev = torch.device(a.device)
    res = []
    g = torch.Generator(device="cpu").manual_seed(7)
    for depth in [int(x) for x in a.depths.split(",")]:
        m.reset()
        for il, slot in enumerate(m.slots):
            n = min(depth, m.window) if m.pattern[il] else depth
            k = torch.randn(m.n_kv, n, m.hd, generator=g)
            k = (k / k.norm(dim=-1, keepdim=True) * 8.0).to(dev)
            slot.k, slot.v = k, torch.randn(m.n_kv, n, m.hd, generator=g).to(dev)
            slot.start = depth - n
            slot.hist = torch.zeros(m.n_conv - 1, m.n_chan, device=dev)
        m.n = depth
        tok = 1000
        for _ in range(2):                                          # warm-up: first use dequantises + caches the dense weights
            tok = int(torch.argmax(m.step([tok])))
        counts.update(apps=0, experts=0, experts_nvfp4=0)
        ts = []
        for _ in range(a.steps):
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            tok = int(torch.argmax(m.step([tok])))
            if dev.type == "cuda":
                torch.cuda.synchronize()
            ts.append(time.time() - t0)
        ms = statistics.median(ts) * 1e3
        r = dict(depth=depth, ms_per_token=round(ms, 1), tok_s=round(1e3 / ms, 3), steps=a.steps, apps_per_step=counts["apps"] / a.steps,
                 experts_per_step=counts["experts"] / a.steps, nvfp4_experts_per_step=counts["experts_nvfp4"] / a.steps, path="torch reference (per-expert dequant, eager attention), synthetic KV")
        print("ROW", json.dumps(r), flush=True)
        res.append(r)
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(gguf=os.path.basename(a.gguf), device=a.device, rows=res), open(os.path.join(a.out, f"decode_{os.path.basename(a.gguf)[:-5]}.json"), "w"), indent=1)
    print(f"FIRES decode {len(res)}")
    bad = [r for r in res if r["tok_s"] <= 0 or r["apps_per_step"] != m.n_app]
    nv = any(r["nvfp4_experts_per_step"] > 0 for r in res)
    print("KSTATS", json.dumps(m.kstats), flush=True)
    kern = m.dev.type != "cuda" or (m.kstats["nvfp4_gap"] > 0 and m.kstats["expert_fallback"] == 0)
    if bad or not nv or not kern:
        print("FIRES-CHECK FAILED", bad, "nvfp4 experts seen:", nv, "kernel path (gap calls > 0, fallbacks 0):", kern)
        sys.exit(7)


if __name__ == "__main__":
    guard.main_guard(main)
