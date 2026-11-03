#!/usr/bin/env python
"""Interleaved served-style A/B for the HIP glue levers, one model load.

  glueab.py MODEL --serve glm|qwen [--depths 4096,32768] [--steps 12] [--reps 3]
             [--knob KYOJIN_HIP_GLUE] [--gen 64] [-o out.json]

Loads the model once with the server's EXL3 defaults, fills one cache with synthetic pages
(the same construction as tools/depth/depth_profile.py), then runs the arms in ABAB order,
visit by visit, at every depth:
  * decode: --steps single-token steps per arm and visit at position D, ms/step (mean +- SE
    over visits of the per-visit median), after one dropped warm-up step per arm/visit;
  * exactness: md5 over the float logits of every timed step, and md5 over the greedy ids of
    a --gen-token generation from one fixed prompt at depth 0.
Arms are flipped with exllamav3.util.knobs.set_knobs inside the process, so build/load state
is identical between them and the only variable is the knob.
"""
import argparse, hashlib, json, os, statistics, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PAGE = 256


def served_env(which):
    import ast
    path, name = {"qwen": ("tools/qwen/serve.py", "SERVE_ENV"), "glm": ("tools/glm/serve.py", "SPEED_ENV")}[which]
    for node in ast.parse(open(os.path.join(ROOT, path)).read()).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return dict(ast.literal_eval(node.value))
    raise SystemExit(f"{name} not found in {path}")


def fill_cache(cache, seed):
    """Fill every cache tensor with synthetic data (random fp16 K/V and packed pages), inside
    inference_mode: the cache tensors are inference tensors and may only be mutated there."""
    import torch
    with torch.inference_mode():
        g = torch.Generator(device="cuda:0"); g.manual_seed(seed)
        for layer in cache.layers.values():
            for name, v in vars(layer).items():
                if name.startswith("_") or not (isinstance(v, torch.Tensor) and v.is_cuda):
                    continue
                f = v.view(-1)
                scale_like = name in ("sk", "sv", "pool_s")
                for i in range(0, f.numel(), 1 << 26):
                    sl = f[i: i + (1 << 26)]
                    if sl.dtype == torch.half:
                        x = torch.randn(sl.shape, generator=g, device=sl.device, dtype=torch.float32)
                        sl.copy_((x.abs().mul_(0.05) if scale_like else x.mul_(0.5)).to(torch.half))
                    elif sl.dtype in (torch.int32, torch.int16, torch.uint8, torch.int8):
                        sl.copy_(torch.randint(-2**31, 2**31 - 1, sl.shape, generator=g, device=sl.device, dtype=torch.int64).to(sl.dtype))


def avail_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1048576
    return -1.0


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--serve", choices=["glm", "qwen"], required=True)
    ap.add_argument("--depths", default="4096,32768")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--reps", type=int, default=3, help="visits per arm")
    ap.add_argument("--knob", default="KYOJIN_HIP_GLUE", help="run-time knob flipped per arm; none = do not touch a knob")
    ap.add_argument("--arm-attr", default="", help="module:ATTR to flip instead of --knob (arm on = True, off = False)")
    ap.add_argument("--purge", default="", help="module:function called between arms (e.g. a graph cache purge)")
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--gen-reps", type=int, default=2, help="generation passes per arm, interleaved (ABAB)")
    ap.add_argument("--gen-only", action="store_true", help="skip the timed phases: greedy exactness only (generator uses the cache it was created with)")
    ap.add_argument("--cache-bits", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--min-avail-gb", type=float, default=5.0)
    ap.add_argument("-o", "--out", default="glueab.json")
    a = ap.parse_args()

    for k, v in served_env(a.serve).items():
        os.environ.setdefault(k, v)
    os.environ.setdefault("TQDM_DISABLE", "1")
    sys.path.insert(0, ROOT)
    import torch
    from exllamav3 import Config, Model, Cache
    from exllamav3.util.knobs import set_knobs, knob
    from exllamav3.cache import CacheLayer_quant
    torch.set_grad_enabled(False)

    depths = [int(x) for x in a.depths.split(",")]
    top = max(depths) + 2 * a.steps + 256
    cap = ((top + PAGE - 1) // PAGE + 1) * PAGE
    R = {"model": a.model, "args": vars(a), "points": {}, "md5": {}, "gen": {}}

    def save():
        json.dump(R, open(a.out + ".tmp", "w"), indent=1)
        os.replace(a.out + ".tmp", a.out)

    t0 = time.time()
    config = Config.from_directory(a.model)
    model = Model.from_config(config)
    qkw = dict(layer_type=CacheLayer_quant, k_bits=a.cache_bits, v_bits=a.cache_bits) if a.cache_bits < 16 else {}
    cache = Cache(model, max_num_tokens=cap, max_history=4, max_batch_size=1, **qkw)
    # a generator needs its own page table (the timed phases map one by hand), and caches must be
    # constructed before model.load
    gcap = 2048 if a.gen else 0
    gcache = Cache(model, max_num_tokens=gcap, max_history=2, max_batch_size=1, **qkw) if gcap else None
    model.load(device="cuda:0", progressbar=False)
    av = avail_gb()
    R["load_s"] = round(time.time() - t0, 1); R["avail_after_load_gb"] = round(av, 1)
    print(f"loaded {R['load_s']}s avail {av:.1f} GiB knob {a.knob}={knob(a.knob)}", flush=True)
    if av < a.min_avail_gb:
        R["abort"] = f"MemAvailable {av:.1f} < {a.min_avail_gb}"; save(); sys.exit(3)

    fill_cache(cache, a.seed)

    bt_full = torch.arange(cap // PAGE, dtype=torch.int32).unsqueeze(0)
    st = {"s": cache.get_new_state()}
    V = config.vocab_size if hasattr(config, "vocab_size") else 100000
    gg = torch.Generator().manual_seed(a.seed)
    one = torch.randint(1000, min(V, 100000), (1, 1), generator=gg)

    def fresh(pos):
        st["s"].free(); st["s"] = cache.get_test_state(pos)

    def fwd(ids, pos):
        pages = (pos + ids.shape[-1] + PAGE - 1) // PAGE
        params = {"attn_mode": "flash_attn", "block_table": bt_full[:, :pages], "cache": cache,
                  "cache_seqlens": torch.tensor([pos], dtype=torch.int32), "recurrent_states": [st["s"]]}
        return model.forward(ids, params)

    arms = ["off", "on"]

    ARM = {}
    if a.arm_attr:
        import importlib
        mod_name, _, attr = a.arm_attr.partition(":")
        ARM["mod"] = importlib.import_module(mod_name)
        ARM["attr"] = attr
    PURGE = None
    if a.purge:
        import importlib
        m, _, f = a.purge.partition(":")
        PURGE = getattr(importlib.import_module(m), f)

    def set_arm(arm):
        if a.knob and a.knob != "none" and not a.arm_attr:
            set_knobs({a.knob: "1" if arm == "on" else "0"})
        if a.arm_attr:
            setattr(ARM["mod"], ARM["attr"], arm == "on")
        if PURGE is not None:
            try:
                PURGE()
            except TypeError:
                PURGE(None)

    def run_arm(arm, d, step_seed):
        set_arm(arm)
        fresh(d)
        ids = torch.randint(1000, min(V, 100000), (1, 1), generator=torch.Generator().manual_seed(step_seed))
        fwd(ids, d)                      # dropped warm-up step
        torch.cuda.synchronize()
        ts, h = [], hashlib.md5()
        for k in range(a.steps):
            torch.cuda.synchronize(); t = time.perf_counter()
            lg = fwd(ids, d + 1 + k)
            tok = int(lg[0, -1].argmax().item())
            ts.append((time.perf_counter() - t) * 1e3)
            h.update(lg.float().cpu().numpy().tobytes())
        return ts, h.hexdigest()

    for d in ([] if a.gen_only else depths):
        pt = R["points"][str(d)] = {arm: [] for arm in arms}
        R["md5"][str(d)] = {arm: None for arm in arms}
        for v in range(a.reps):
            for arm in (arms if v % 2 == 0 else arms[::-1]):
                ts, h = run_arm(arm, d, a.seed + d)
                pt[arm].append(round(statistics.median(ts), 3))
                if R["md5"][str(d)][arm] is None:
                    R["md5"][str(d)][arm] = h
                elif R["md5"][str(d)][arm] != h:
                    R["md5"][str(d)][arm] += f"!{h}"
                print(f"  d={d} visit={v} arm={arm} median {statistics.median(ts):.3f} ms  [{min(ts):.2f}..{max(ts):.2f}]", flush=True)
        save()

    # greedy generation exactness at depth 0, one prompt, both arms
    if a.gen:
        from exllamav3 import Tokenizer, Generator, Job
        from exllamav3.generator.sampler import GreedySampler
        tok = Tokenizer.from_config(config)
        prompt = "Write one paragraph about the speed of light in a vacuum, then a short list of its consequences."
        ids = tok.encode(prompt, add_bos=True, encode_special_tokens=True)
        out = {arm: [] for arm in arms}
        for rep in range(a.gen_reps):
            for arm in (arms if rep % 2 == 0 else arms[::-1]):
                set_arm(arm)
                gen = Generator(model=model, cache=gcache, tokenizer=tok)
                gen.enqueue(Job(input_ids=ids, max_new_tokens=a.gen, sampler=GreedySampler()))
                toks = []
                while gen.num_remaining_jobs():
                    for r in gen.iterate():
                        for t in (r.get("token_ids") or []):
                            toks.append(int(t))
                out[arm].append(dict(md5=hashlib.md5(",".join(map(str, toks)).encode()).hexdigest(), n=len(toks), ids=toks[:8]))
                print(f"  gen rep={rep} arm={arm} {out[arm][-1]['md5']} n={out[arm][-1]['n']} head={out[arm][-1]['ids']}", flush=True)
        R["gen"] = out
        same_arm = all(len({d["md5"] for d in v}) == 1 for v in out.values())
        cross = len({v[0]["md5"] for v in out.values()}) == 1
        print(f"gen: within-arm stable {same_arm}, off==on {cross}", flush=True)
        save()

    for d in ([] if a.gen_only else depths):
        o, n = R["points"][str(d)]["off"], R["points"][str(d)]["on"]
        mo, mn = statistics.mean(o), statistics.mean(n)
        se = statistics.stdev([x - y for x, y in zip(o, n)]) / max(1, len(o)) ** 0.5 if len(o) > 1 else 0.0
        print(f"depth {d}: off {mo:.3f} ms ({1000/mo:.2f} tok/s)  on {mn:.3f} ({1000/mn:.2f})  "
              f"delta {100*(mn-mo)/mo:+.2f} % +- {100*se/mo:.2f}  md5 {'EQUAL' if R['md5'][str(d)]['off']==R['md5'][str(d)]['on'] else 'DIFFER'}")
    if R["gen"]:
        print("greedy md5", "EQUAL" if R["gen"]["off"][0]["md5"] == R["gen"]["on"][0]["md5"] else "DIFFER")
    save()


def main():
    # the cache tensors are created by the engine under inference mode: updating them needs it too
    # (same wrapper as tools/depth/depth_profile.py)
    import torch
    with torch.inference_mode():
        _main()


if __name__ == "__main__":
    main()
