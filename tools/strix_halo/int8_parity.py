#!/usr/bin/env python3
"""Parity test for the HIP port of the fused int8-activation GEMV (exl3_gemv_int8).

Driver mode (no --worker): spawns one worker process per EXL3_INT8_GEMV mode {0, 2, 1}, each
of which loads the model, runs the selected mul1 Linear modules on fixed random inputs (m=1 and
m=2), and saves outputs to /tmp/int8_parity_<mode>.pt. Then compares the int8 modes against
mode 0 (fp16 WMMA path) and prints a table. The env var is read once by C++, hence the
separate processes. The worker sets EXL3_INT8_GEMV_TRACE=1 so the C++ side prints one line per
int8 call, proving the route was actually taken.
"""
import os, sys, subprocess, time, zlib

MODEL = os.path.expanduser("~/models/Qwen3.8-Flash-Next-EXL3")
OUT = "/tmp/int8_parity_{}.pt"
NOISE = ["UserWarning", "amdgpu.ids", "NumPy", "_POSIX_C_SOURCE", "features.h", "hip_runtime",
         "triton_launcher", "In file included", "warning", "^\\s*[0-9]+ \\|", "^\\s*\\^~", "from /"]


def worker(mode):
    import torch, re
    from exllamav3 import Config, Model
    from exllamav3.modules import Linear
    torch.manual_seed(1234)
    cfg = Config.from_directory(MODEL)
    model = Model.from_config(cfg)
    model.load(progressbar=False)

    picks = []
    seen_keys = set()
    want = {"q_proj": 1, "k_proj": 1, "v_proj": 1, "o_proj": 1, "lm_head": 1, "gate_proj": 1, "up_proj": 1, "down_proj": 1}
    # Also one K=3 mul1 tensor of each shape class (the routed experts), keyed on K
    want_k3 = 3
    # Walk the module tree
    stack = [model]
    while stack:
        mod = stack.pop()
        if isinstance(mod, Linear):
            inner = getattr(mod, "inner", None)
            if inner is not None and getattr(inner, "mul1", False):
                base = mod.key.split(".")[-1]
                if base in want and want[base] > 0:
                    want[base] -= 1
                    picks.append(mod)
                elif inner.K == 3 and want_k3 > 0:
                    want_k3 -= 1
                    picks.append(mod)
        for c in getattr(mod, "modules", []):
            stack.append(c)
    picks.sort(key=lambda m: m.key)
    print(f"[worker mode={mode}] picked {len(picks)} mul1 modules", flush=True)

    results = {}
    for mod in picks:
        inner = mod.inner
        k = mod.in_features
        n = mod.out_features
        K = inner.K
        # NOT hash(): str hashes are salted per process, and the workers must see identical inputs
        g = torch.Generator(device="cpu").manual_seed(zlib.crc32(mod.key.encode()))
        for m in (1, 2):
            x = (torch.randn(m, k, generator=g) * 0.5).to(torch.half).cuda()
            y = inner.forward(x, {})
            torch.cuda.synchronize()
            results[(mod.key, K, k, n, m)] = y.float().cpu()
            print(f"[worker mode={mode}] {mod.key} K={K} k={k} n={n} m={m} out={tuple(y.shape)} dtype={y.dtype}", flush=True)
    torch.save(results, OUT.format(mode))
    print(f"[worker mode={mode}] saved {OUT.format(mode)}", flush=True)


def run_worker(mode):
    env = dict(os.environ)
    env["EXL3_INT8_GEMV"] = str(mode)
    env["EXL3_INT8_GEMV_TRACE"] = "1"
    env.setdefault("EXL3_MOE_CFG", "2")
    env.setdefault("EXL3_HIP_PREFILL_MIN_ROWS", "2")
    log = f"/tmp/int8_parity_worker_{mode}.log"
    for attempt in range(3):
        with open(log, "w") as f:
            rc = subprocess.call([sys.executable, __file__, "--worker", str(mode)], env=env, stdout=f, stderr=subprocess.STDOUT)
        txt = open(log).read()
        if rc == 0 and os.path.exists(OUT.format(mode)):
            break
        if "out of memory" in txt:
            print(f"[driver] mode {mode}: transient OOM, retrying in 8 s", flush=True)
            time.sleep(8)
            continue
        print(f"[driver] mode {mode} FAILED rc={rc}; tail of {log}:")
        print("\n".join(txt.splitlines()[-30:]))
        sys.exit(1)
    n_trace = sum(1 for l in txt.splitlines() if l.startswith("[exl3_gemv_int8]"))
    n_fail = sum(1 for l in txt.splitlines() if "coop launch FAILED" in l)
    print(f"[driver] mode {mode}: rc={rc} int8 trace lines={n_trace} coop-fails={n_fail}", flush=True)
    return n_trace


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(int(sys.argv[2]))
        return
    import torch
    traces = {}
    for mode in (0, 2, 1):
        if os.path.exists(OUT.format(mode)):
            os.remove(OUT.format(mode))
        traces[mode] = run_worker(mode)
        time.sleep(6)
    ref = torch.load(OUT.format(0))
    print()
    print(f"{'module':<48} {'K':>2} {'k':>6} {'n':>7} {'m':>2} {'mode':>4} {'max_abs':>10} {'max_rel':>10} {'rms_rel':>10} {'argmax':>8}")
    for mode in (2, 1):
        res = torch.load(OUT.format(mode))
        for key in sorted(ref):
            a = ref[key]; b = res[key]
            d = (a - b).abs()
            max_abs = d.max().item()
            denom = a.abs().max().item()
            max_rel = max_abs / max(denom, 1e-12)
            rms_rel = (d.pow(2).mean().sqrt() / max(a.pow(2).mean().sqrt().item(), 1e-12)).item()
            agree = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
            name, K, k, n, m = key
            print(f"{name:<48} {K:>2} {k:>6} {n:>7} {m:>2} {mode:>4} {max_abs:>10.4g} {max_rel:>10.4g} {rms_rel:>10.4g} {agree:>8.2f}")
    print()
    print("int8 trace lines per mode:", traces)
    # sanity: mode 0 must have zero trace lines, others > 0
    if traces[0] != 0 or traces[2] == 0 or traces[1] == 0:
        print("ROUTE CHECK FAILED", file=sys.stderr)
        sys.exit(2)
    print("route check OK: int8 path ran only in modes 1/2")


if __name__ == "__main__":
    main()
