"""Inkling2 plain decode tok/s at 1K / 8K depth with our forward (loader/inkling2_forward.py). The op kernels are picked by the device
(CUDA on NVIDIA, HIP glue on AMD, torch otherwise: see the module docstring of the forward); the torch step-0 path (experts read raw +
dequantised per picked expert, eager attention) is the baseline both kernel sets are measured against, with INK_BACKEND=torch.

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


def warm_spin(m, tok, secs=3.0):
    """clock-ramp warm decode: keep the GPU busy for `secs` (one decode step is ~20 ms; the clocks need ~1 s of load to settle: reps drifted 18.2 -> 16.3 ms without it)"""
    t0 = time.time()
    while time.time() - t0 < secs:
        tok = int(torch.argmax(m.step([tok])))
    return tok


def kernel_profile(m, tok, n=3, top=28):
    """GPU time per kernel name over n steps (torch.profiler / CUPTI, works on graph replays): the step split under graphs, where the python methods are not called"""
    from torch.profiler import profile, ProfilerActivity
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            tok = int(torch.argmax(m.step([tok])))
        torch.cuda.synchronize()
    rows = {}
    for e in prof.key_averages():
        us = getattr(e, "device_time_total", None)
        if us is None:
            us = getattr(e, "cuda_time_total", 0)
        if us > 0 and e.key and not e.key.startswith(("cuda", "Memcpy", "Memset")):
            rows[e.key] = (us / n / 1e3, e.count / n)
    tot = sum(v[0] for v in rows.values())
    out = {"total_kernel_ms_per_token": round(tot, 2), "n_kernels_per_token": round(sum(v[1] for v in rows.values()))}
    out["top"] = [(k[:90], round(v[0], 3), round(v[1], 1)) for k, v in sorted(rows.items(), key=lambda kv: -kv[1][0])[:top]]
    return out


def step_split(F, m, dev, tok, n):
    """ms per decoded token by kernel class (a separate arm AFTER the timed one: every wrapped call syncs, so the sum is longer than the plain step): the methods of the step are wrapped, the rest = glue / small torch ops"""
    acc = {}

    def wrap(name):
        fn = getattr(F.Inkling2Forward, name)

        def w(self, *a, **k):
            torch.cuda.synchronize()
            t0 = time.time()
            r = fn(self, *a, **k)
            torch.cuda.synchronize()
            acc[name] = acc.get(name, 0.0) + time.time() - t0
            return r
        setattr(F.Inkling2Forward, name, w)
        return fn
    names = [nm for nm in ["_moe", "_moe_kernel", "_attend", "_attend_rtx", "mm", "head", "mix", "control", "conv"] if hasattr(F.Inkling2Forward, nm)]
    orig = {nm: wrap(nm) for nm in names}
    try:
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n):
            tok = int(torch.argmax(m.step([tok])))
        torch.cuda.synchronize()
        tot = time.time() - t0
    finally:
        for nm, fn in orig.items():
            setattr(F.Inkling2Forward, nm, fn)
    out = {nm.strip("_"): round(acc.get(nm, 0.0) / n * 1e3, 1) for nm in names}
    out["total_synced"] = round(tot / n * 1e3, 1)
    out["other_glue_ms"] = round(out["total_synced"] - sum(out[k] for k in ("moe", "attend", "attend_rtx", "head", "mix", "control", "conv") if k in out), 1)
    out["note"] = "every wrapped call syncs (~20 us each, ~700 calls: ~10-15 ms of the total is sync cost); moe_kernel is a subset of moe (rest of moe = router + torch ops), attend_rtx = rel + append + attend launches; other_glue = dense projections (W dequant + matmul), norms, embedding"
    return out


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
    if os.environ.get("INK_GRAPH") == "1":
        m.capture_graphs()
    res = []
    want_prof = m.gl is not None and m.gl.prof
    g = torch.Generator(device="cpu").manual_seed(7)
    for depth in [int(x) for x in a.depths.split(",")]:
        m.reset()
        for il, slot in enumerate(m.slots):
            n = min(depth, m.window) if m.pattern[il] else depth
            if slot.hist is None:
                slot.hist = torch.zeros(m.n_conv - 1, m.n_chan, device=dev)      # in place when it exists: captured graphs keep the address (reset() already zeroed it)
            if m.gl is not None and m.gl.has("lattn" if m.pattern[il] else "gkv"):
                continue                                            # this slot lives in the kernel cache (fill_synthetic below), no torch K / V
            k = torch.randn(m.n_kv, n, m.hd, generator=g)
            k = (k / k.norm(dim=-1, keepdim=True) * 8.0).to(dev)
            m.seed_kv(il, k, torch.randn(m.n_kv, n, m.hd, generator=g).to(dev), depth - n)      # NVIDIA: the kernel caches, else the torch slots
            if slot.hist is None:
                slot.hist = torch.zeros(m.n_conv - 1, m.n_chan, device=dev)      # in place when it exists: captured graphs keep the address (reset() already zeroed it)
            else:
                slot.hist.zero_()
        if m.gl is not None:
            m.gl.fill_synthetic(depth, g)                           # kernel caches (gkv / lattn) get the same kind of synthetic rows
        m.n = depth
        tok = 1000
        if m.gl is not None and m.gl.has("experts"):
            t0 = time.time()
            m.gl.preload()                                          # resident raw expert bytes of the 24 physical layers (counted outside the timed steps)
            print(f"experts resident in {time.time() - t0:.1f} s", flush=True)
        for _ in range(2):                                          # warm-up: first use dequantises + caches the dense weights
            tok = int(torch.argmax(m.step([tok])))
        want_prof = m.gl is not None and m.gl.prof
        if m.gl is not None:
            m.gl.prof = False                                       # no event spans in the timed steps
        REPS = 3
        void = False
        for attempt in range(2):                                    # measuring rule: 20 s idle + warm decode before the arm; reps spread > 5 % = VOID, rerun once after a new settle + warm
            if dev.type == "cuda":
                guard.settle_and_warm(lambda: warm_spin(m, tok), label=f"depth {depth} try {attempt + 1}")
            counts.update(apps=0, experts=0, experts_nvfp4=0)
            rep_ms = []
            for _ in range(REPS):
                ts = []
                for _ in range(a.steps):
                    if dev.type == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.time()
                    tok = int(torch.argmax(m.step([tok])))
                    if dev.type == "cuda":
                        torch.cuda.synchronize()
                    ts.append(time.time() - t0)
                rep_ms.append(statistics.median(ts) * 1e3)
            sp = (max(rep_ms) - min(rep_ms)) / statistics.median(rep_ms)
            void = sp > 0.05
            print(f"MEASURE_RULE depth {depth} try {attempt + 1}: reps {[round(x, 1) for x in rep_ms]} spread {sp * 100:.1f} %{' VOID' if void else ''}", flush=True)
            if not void:
                break
        if void:
            print(f"FIRES-CHECK FAILED: depth {depth} arm VOID twice (spread {sp * 100:.1f} %): no number reported", flush=True)
            sys.exit(7)
        ms = statistics.median(rep_ms)
        a_steps = a.steps * REPS
        prof = {}
        if want_prof:                                               # ms / token per kernel class, a separate (synced) arm after the timed one
            m.gl.prof = True
            m.gl.prof_report()
            for _ in range(3):
                tok = int(torch.argmax(m.step([tok])))
            prof = {k: [round(v[0] / 3, 3), v[1] // 3] for k, v in m.gl.prof_report().items()}
            m.gl.prof = False
        path = f"{'CUDA graph' if m.graphs else 'eager'} step, backend {m.backend}"
        if m.gl is not None:
            path += ", glue " + ",".join(sorted(m.gl.on))
        else:
            path += f", INK_RTX={','.join(sorted(F.RTX)) or 'none'} (torch for the rest)"
        r = dict(depth=depth, glue=(sorted(m.gl.on) if m.gl else None), backend=m.backend, kernel_ms=prof, launches_per_token=-1,
                 ms_per_token=round(ms, 1), tok_s=round(1e3 / ms, 3), steps=a.steps, reps=[round(x, 1) for x in rep_ms], steps_ms=[round(ts[-1] * 1e3, 1)],
                 spread_pct=round(sp * 100, 1), void=void, reran=attempt > 0, apps_per_step=counts["apps"] / a_steps,
                 experts_per_step=counts["experts"] / a_steps, nvfp4_experts_per_step=counts["experts_nvfp4"] / a_steps, path=path + ", synthetic KV")
        if os.environ.get("INK_SPLIT", "1") == "1" and not m.graphs:              # AFTER the counters of the timed arm are read: the split arm runs more (synced) steps
            r["split_ms"] = step_split(F, m, dev, tok, 3)
        if os.environ.get("INK_PROFILE") == "1":
            r["kernel_profile"] = kernel_profile(m, tok)
        print("ROW", json.dumps(r), flush=True)
        res.append(r)
        if m.gl is not None:                                        # launch count LAST (the ROCm profiler sees no kernels: aten ops + Triton launches + ctypes kernel launches are counted instead)
            try:
                import ink2_glue
                lc = ink2_glue.count_launches(m, lambda: m.step([tok]))
                r["launches_per_token"] = lc["total"]
                r["launch_split"] = lc
                print("LAUNCHES", depth, json.dumps(lc), flush=True)
            except Exception as e:      # noqa: BLE001
                print("launch count unavailable:", repr(e)[:200], flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(gguf=os.path.basename(a.gguf), device=a.device, rows=res), open(os.path.join(a.out, f"decode_{os.path.basename(a.gguf)[:-5]}.json"), "w"), indent=1)
    print(f"FIRES decode {len(res)}")
    bad = [r for r in res if r["tok_s"] <= 0 or (r["apps_per_step"] != m.n_app and not m.graphs)]
    df = m.dev_fires()
    print("KSTATS", json.dumps(m.kstats), flush=True)
    nv = any(r["nvfp4_experts_per_step"] > 0 for r in res) or m.kstats["nvfp4_gap"] > 0 or m.kstats.get("xk_nvfp4", 0) > 0      # the kernel path never calls read_expert: the kernel counter is the evidence
    if m.gl is not None:      # AMD: the glue fires-check covers every stage the run enabled (the CUDA counters above stay 0 on purpose)
        import ink2_glue
        gbad = ink2_glue.fires_check(m)
        print("GLUE", sorted(m.gl.on), json.dumps(dict(m.gl.fires)), "fires-check:", gbad or "ok", flush=True)
        bad = bad + gbad
        kern = not gbad
        nv = nv or m.gl.has("experts")
        if m.gl.has("experts"):
            nv = True
    else:
        if "experts" in F.RTX:         # our kernels: fired, and the torch path never did
            kern = m.backend != "cuda" or (m.kstats["xk_nvfp4"] > 0 and m.kstats["torch_moe"] == 0 and m.kstats["nvfp4_gap"] == 0 and m.kstats["expert_fallback"] == 0)
        else:
            kern = m.backend != "cuda" or (m.kstats["nvfp4_gap"] > 0 and m.kstats["expert_fallback"] == 0)
        if "experts" in F.RTX and m.backend == "cuda":
            kern = kern and min(df["xk_gate_up_entries_down_blocks"]) > 0
        if "conv" in F.FUSE and m.backend == "cuda":     # the fused conv kernel ran at every call site and the torch conv never did
            import ink2_glue_conv as _G
            kern = kern and m.kstats["conv_fused"] > 0 and m.kstats["conv_torch"] == 0 and _G.FIRES["conv"] > 0
            print("CONV_FIRES", _G.FIRES, flush=True)
        if m.backend == "cuda" and {"mix", "control", "norm"} & F.FUSE:      # the Triton glue fusions fired (python counters advance while capturing / in eager)
            import ink2_glue_misc as _M
            for _f, _k in (("mix", "mix"), ("control", "control"), ("norm", "rmsw")):
                if _f in F.FUSE:
                    kern = kern and _M.FIRES[_k] > 0
            print("MISC_FIRES", _M.FIRES, flush=True)
        if "router" in F.RTX and m.backend == "cuda":
            kern = kern and m.kstats["router_k"] > 0 and m.kstats["torch_router"] == 0 and df["router_blocks"] > 0
        if "dense" in F.RTX and m.backend == "cuda":     # the Q8_0 GEMV kernel took the dense projections + the head
            kern = kern and m.kstats["dense_q8"] > 0
        if "attn" in F.RTX and m.backend == "cuda":     # the attention kernels fired (device counters: attend / combine / append / rel blocks) and the torch attention did not
            kern = kern and m.kstats["ak_gkv"] > 0 and m.kstats["ak_lattn"] > 0 and m.kstats["torch_attn"] == 0 and min(df["ak_gkv_lattn_combine_append_rel_blocks"][:2] + df["ak_gkv_lattn_combine_append_rel_blocks"][3:]) > 0
    if m.graphs:      # graph arm: every timed step was a replay (python counters only advance while capturing: the device counters are the evidence)
        kern = kern and m.kstats["graph_replays"] > 0 and m.kstats["graph_eager"] == 0
    if bad or not nv or not kern:
        print("FIRES-CHECK FAILED", bad, "nvfp4 experts seen:", nv, "kernel path (gap calls > 0, fallbacks 0):", kern)
        sys.exit(7)


if __name__ == "__main__":
    guard.main_guard(main)
