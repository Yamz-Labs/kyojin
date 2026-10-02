# D1-2 acceptaudit v2: independent teacher-forced MTP top-1/top-4 vs served accept (DESIGN-D1 §3c).
#
# Phase 1 (served path): per prompt, greedy generation through the served draft loop (serve.py
#   SPEED_ENV, num_draft_tokens=1, MTP_FUSE_CATCHUP=2, UNION_V2) -> served accept from
#   job.draft_stats and the continuation ids. Prompts are rendered through the model's own
#   chat_template.jinja (ends in <think>), exactly as serve.py does.
# Phase 2 (teacher-forced, independent of the generator): one cacheless target forward over
#   prompt+continuation exporting BOTH the pre-norm (hc_head mean collapse) and post-norm
#   (model.norm) states, then one cacheless MTP forward per arm over all positions.
#   Row p takes (emb x_p, h_{p-1}) and predicts x_{p+1} (exl3 pairing, same as job.py prefill).
#   Scored on continuation tokens x_q, q >= P+1 (x_P is the first sample, never drafted).
#   Arms: tap {post, pre} x eh_proj {exl3 2-bit, fp16 from the FP8 checkpoint (BF16 tensor)}.
# Phase 3: served accept again with the best TF arm (target ids must stay equal).
#
# usage: python tools/glm/acceptaudit_tf2.py <out.json>
# eh_proj sidecar: scratch/acceptaudit/eh_proj_45_fp16.pt (BF16 layers.45.eh_proj.weight from the FP8 checkpoint, as fp16)
import json, math, os, statistics, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools", "glm"))
import ast
_src = open(os.path.join(ROOT, "tools", "glm", "serve.py")).read()
SPEED_ENV = next(ast.literal_eval(n.value) for n in ast.parse(_src).body
                 if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "SPEED_ENV")
for k, v in SPEED_ENV.items(): os.environ.setdefault(k, v)
os.environ.setdefault("EXL3_MOE_UNION_V2", "1")
os.environ["EXL3_MTP_PRENORM_H"] = "0"   # tap is switched in-process below

import torch

out = sys.argv[1]
NPROMPT = int(os.environ.get("AA_NPROMPT", "8"))
NTOK = int(os.environ.get("AA_NTOK", "400"))
PHASE3 = os.environ.get("AA_PHASE3", "1") == "1"
MODEL = f"{os.path.expanduser('~')}/models/glm53-exl3-td205"
EH_FP16 = os.environ.get("AA_EH_FP16", f"{os.path.expanduser('~')}/models/glm53-mtp-eh-proj-bf16.safetensors")

# Prompt 0 of each class = the serve_accept.py DECODE_PROMPTS entry (the served numbers).
PROMPTS = {
    "prose": [
        "Write a 300-word story about a lighthouse.",
        "Describe a busy harbour at dawn in about 250 words, as the opening of a novel.",
        "Write a short essay on why old libraries feel quiet even when they are full.",
        "Tell the story of a village that floods every spring, from the point of view of the river.",
        "Write a 300-word piece about the first snowfall in a mountain town.",
        "Write a letter from a lighthouse keeper to his daughter who lives in the city.",
        "Describe an orchestra tuning up before a concert, in vivid prose.",
        "Write a short story about a clockmaker who repairs a clock that runs backwards.",
    ],
    "chat": [
        "Explain how a hash map works, with its time complexity, in about 300 words.",
        "My train was delayed by two hours and I still have a meeting when I arrive. What would you do in my place?",
        "I have been trying to make a simple tomato sauce for years and it always tastes flat. What am I most likely getting wrong?",
        "I want to start running in my forties. I can walk for an hour without trouble. Where should I begin?",
        "We are choosing between two apartments: one closer to work and darker, the other further and brighter. How would you decide?",
        "Explain, as if I know nothing about it, why bridges have expansion joints.",
        "What is the difference between a virus and a bacterium, and why do antibiotics only work on one?",
        "How does compound interest work? Give a simple example with numbers.",
    ],
    "code": [
        "Write a Python function that parses an ISO-8601 duration string into seconds, with tests.",
        "Write a Python function that reads a CSV file, groups the rows by the value in the second column, and returns a dict of lists.",
        "I have a race condition in a C program where two threads increment a shared counter. Show me the correct fix.",
        "Write a SQL query that returns, for each customer, the number of orders placed in 2024 and in 2023, including customers with no orders.",
        "Write a bash script that watches a directory for new files and moves them into subdirectories based on their extension.",
        "Implement an LRU cache in Python with O(1) get and put, and explain the design.",
        "Write a JavaScript debounce function and a small usage example.",
        "Write a Rust function that returns the n-th Fibonacci number using iteration, with unit tests.",
    ],
}


def render(template, prompt):
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = lambda v, ensure_ascii=False, indent=None, separators=None, sort_keys=False: \
        json.dumps(v, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
    def raise_exception(msg): raise ValueError(msg)
    env.globals["raise_exception"] = raise_exception
    return env.from_string(template).render(messages=[{"role": "user", "content": prompt}], tools=[],
                                            add_generation_prompt=True)


class FP16Inner:
    """Wraps the EXL3_MTP_EH_FP16 inner (LinearFP16, unquantized eh_proj) to optionally record
    cos(exl3 2-bit output, fp16 output) on the real MTP input."""
    def __init__(self, inner):
        self.inner = inner
        self.probe = None      # set to the exl3 inner to record cos(exl3, fp16) on the real input
        self.cos = []
    def forward(self, x, params, out_dtype=None):
        y = self.inner.forward(x, params, out_dtype)
        if self.probe is not None:
            yq = self.probe.forward(x, params, torch.float)
            self.cos.append(torch.nn.functional.cosine_similarity(
                yq.float().flatten(1), y.float().flatten(1), dim=-1).mean().item())
            self.probe = None
        return y


def main():
    torch.set_grad_enabled(False)
    from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
    from exllamav3.generator import generator as GM
    from exllamav3.generator.sampler import GreedySampler
    from exllamav3.modules import block_graph as BG
    from exllamav3.modules.hyperconnections import HyperHead
    GM.MTP_FUSE_CATCHUP = 2

    config = Config.from_directory(MODEL)
    model = Model.from_config(config)
    tok = Tokenizer.from_config(config)
    cache = Cache(model, max_num_tokens=8192, max_history=1)
    model.load(device="cuda:0", progressbar=False)
    draft = Model.from_config(config, component="mtp")
    draft_cache = Cache(draft, max_num_tokens=8192, max_history=1)
    draft.load(device="cuda:0", progressbar=False)
    gen = Generator(model=model, cache=cache, tokenizer=tok, draft_model=draft, draft_cache=draft_cache,
                    num_draft_tokens=1, record_draft_stats=True)
    template = open(os.path.join(MODEL, "chat_template.jinja")).read()
    eos = list(config.eos_token_id_list)

    li = model.logit_layer_idx
    lm = model.modules[li]
    post_key = model.modules[li - 1].key
    pre_key = next(m.norm.key for m in model.modules[li::-1] if isinstance(m, HyperHead) and m.mean)
    vocab = config.vocab_size
    fc = draft.input_layer.fc
    q_inner = fc.inner
    assert draft.load_eh_proj_sidecar(EH_FP16)      # the EXL3_MTP_EH_FP16 code path
    f_inner = FP16Inner(fc.inner)
    fc.inner = q_inner

    def set_arm(tap, eh):
        draft.draft_verifier_params = {"export_state_norm_keys": {pre_key if tap == "pre" else post_key}}
        fc.inner = f_inner.inner if eh == "fp16" else q_inner
        BG.purge()

    R = {"env": {k: os.environ.get(k) for k in list(SPEED_ENV) + ["EXL3_MOE_UNION_V2"]}, "ntok": NTOK,
         "keys": {"pre": pre_key, "post": post_key}, "prompts": {}, "summary": {}}

    def save():
        with open(out + ".tmp", "w") as f: json.dump(R, f, indent=1)
        os.replace(out + ".tmp", out)

    def served(ids):
        job = Job(input_ids=ids, max_new_tokens=NTOK, sampler=GreedySampler(), stop_conditions=eos)
        gen.enqueue(job)
        got = []
        t0 = time.perf_counter()
        while gen.num_remaining_jobs():
            for r in gen.iterate():
                if r.get("error"): raise RuntimeError(r["error"])
                t = r.get("token_ids")
                if t is not None and t.numel(): got += t.flatten().tolist()
        dt = time.perf_counter() - t0
        ds = list(job.draft_stats)
        den = sum(w for _, w, _ in ds)
        acc = sum(a for _, _, a in ds) / den if den else None
        return got, acc, den, len(got) / dt

    def tf(ids, cont):
        full = torch.cat((ids, torch.tensor([cont], dtype=torch.long)), dim=1)
        P, L = ids.shape[1], full.shape[1]
        params = {"attn_mode": "flash_attn_nc", "export_state_norm_keys": {pre_key, post_key}}
        logits = model.forward(full, params)
        st = params["export_states"]
        assert len(st) == 2, f"expected 2 exported states, got {len(st)}"
        pre, post = st[0], st[1]
        # sanity: target cacheless argmax reproduces the served greedy continuation
        tgt = logits[0, P - 1:L - 1, :vocab].argmax(-1).cpu()
        tgt_match = (tgt == full[0, P:L]).float().mean().item()
        del logits
        info = {"tgt_match": tgt_match, "pre_finite": bool(torch.isfinite(pre).all()),
                "pre_absmax": pre.float().abs().max().item(), "pre_rms": pre.float().pow(2).mean().sqrt().item(),
                "post_rms": post.float().pow(2).mean().sqrt().item()}
        # rows j = 0..L-2: ids x_{j+1}, hidden h_j, predict x_{j+2}; score q = j+2 >= P+1
        x_in = full[:, 1:]
        tgt_ids = full[0, 2:]
        j0 = P - 1
        res = {}
        for tap, eh in ARMS_ORDER:
            h = (pre if tap == "pre" else post)[:, :-1, :].contiguous()
            fc.inner = f_inner if eh == "fp16" else q_inner
            if eh == "fp16": f_inner.probe = q_inner
            s = draft.forward(x_in, {"attn_mode": "flash_attn_nc", "target_hidden": h})
            lg = lm.forward(lm.prepare_for_device(s, {}), {})[0, :L - 2, :vocab].float()
            top = lg.topk(4, dim=-1).indices.cpu()
            y = tgt_ids.view(-1, 1)
            t1 = (top[:, :1] == y).any(-1)[j0:]
            t4 = (top == y).any(-1)[j0:]
            res[f"{tap}/{eh}"] = {"top1": t1.float().mean().item(), "top4": t4.float().mean().item(),
                                  "n": int(t1.numel())}
            del lg, s
        fc.inner = q_inner
        info["eh_cos_real"] = f_inner.cos[-2:]
        return res, info

    arms = [("post", "q2"), ("pre", "q2"), ("post", "fp16"), ("pre", "fp16")]
    global ARMS_ORDER
    # eh_proj cosine check on one real input (catches a transposed/garbage override)
    xin = torch.randn(1, 8, 8192, device="cuda:0", dtype=torch.half) * 0.5
    yq = q_inner.forward(xin, {}, torch.float)
    yf = f_inner.forward(xin, {}, torch.float)
    R["eh_cos_random"] = torch.nn.functional.cosine_similarity(yq.flatten(), yf.flatten(), dim=0).item()
    for name, sl in (("e_half", slice(0, 4096)), ("h_half", slice(4096, 8192))):
        xh = torch.zeros_like(xin); xh[..., sl] = xin[..., sl]
        a, b = q_inner.forward(xh, {}, torch.float), f_inner.forward(xh, {}, torch.float)
        R[f"eh_cos_{name}"] = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
        R[f"eh_norm_ratio_{name}"] = (a.norm() / b.norm()).item()
    print("eh_proj halves: " + json.dumps({k: round(v, 4) for k, v in R.items() if k.startswith("eh_")}), flush=True)
    print(f"eh_proj cos(exl3 2-bit, fp16) on random input = {R['eh_cos_random']:.4f}", flush=True)

    set_arm("post", "q2")
    for cls, plist in PROMPTS.items():
        for i, p in enumerate(plist[:NPROMPT]):
            key = f"{cls}/{i}"
            ids = tok.encode(render(template, p), encode_special_tokens=True)
            cont, acc, rounds, tps = served(ids)
            # rotate the arm order per prompt (off/on/on/off pattern across prompts)
            k = (len(R["prompts"])) % 4
            ARMS_ORDER = arms[k:] + arms[:k]
            if len(R["prompts"]) % 2: ARMS_ORDER = ARMS_ORDER[::-1]
            tfr, info = tf(ids, cont)
            R["prompts"][key] = {"P": int(ids.shape[1]), "ncont": len(cont), "served_accept": acc,
                                 "served_rounds": rounds, "served_tps": tps, "tf": tfr, "info": info,
                                 "cont_hash": hash(tuple(cont))}
            print(f"[{key}] P={ids.shape[1]} n={len(cont)} served={acc:.4f} tps={tps:.2f} tgt_match={info['tgt_match']:.4f} "
                  + " ".join(f"{a}:{v['top1']:.4f}/{v['top4']:.4f}" for a, v in tfr.items())
                  + f" pre_absmax={info['pre_absmax']:.1f} rms pre/post={info['pre_rms']:.3f}/{info['post_rms']:.3f}",
                  flush=True)
            save()

    def ms(v):
        m = statistics.mean(v)
        se = statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else float("nan")
        return {"mean": m, "se": se, "n": len(v)}
    classes = list(PROMPTS)
    S = R["summary"]
    for grp in classes + ["all"]:
        ps = [v for k, v in R["prompts"].items() if grp == "all" or k.startswith(grp + "/")]
        S[grp] = {"served_accept": ms([p["served_accept"] for p in ps])}
        for a in [f"{t}/{e}" for t, e in arms]:
            S[grp][a + " top1"] = ms([p["tf"][a]["top1"] for p in ps])
            S[grp][a + " top4"] = ms([p["tf"][a]["top4"] for p in ps])
            if a != "post/q2":
                S[grp][a + " dtop1"] = ms([p["tf"][a]["top1"] - p["tf"]["post/q2"]["top1"] for p in ps])
        S[grp]["gap_served_vs_tf_base"] = ms([p["served_accept"] - p["tf"]["post/q2"]["top1"] for p in ps])
    for grp in classes + ["all"]:
        print(f"SUMMARY {grp}: " + json.dumps({k: round(v['mean'], 4) for k, v in S[grp].items()}), flush=True)
    save()

    if PHASE3:
        best = max((a for a in arms if a != ("post", "q2")),
                   key=lambda a: S["all"][f"{a[0]}/{a[1]} top1"]["mean"])
        R["phase3_arm"] = "/".join(best)
        if S["all"][f"{best[0]}/{best[1]} dtop1"]["mean"] > 0.005:
            set_arm(*best)
            R["phase3_env"] = {"EXL3_MTP_EH_FP16": EH_FP16 if best[1] == "fp16" else "0",
                               "EXL3_MTP_PRENORM_H": "1" if best[0] == "pre" else "0"}
            P3 = {}
            for cls, plist in PROMPTS.items():
                for i, p in enumerate(plist[:NPROMPT]):
                    key = f"{cls}/{i}"
                    ids = tok.encode(render(template, p), encode_special_tokens=True)
                    cont, acc, rounds, tps = served(ids)
                    P3[key] = {"served_accept": acc, "served_tps": tps,
                               "ids_equal": hash(tuple(cont)) == R["prompts"][key]["cont_hash"]}
                    print(f"[P3 {R['phase3_arm']} {key}] served={acc:.4f} (base {R['prompts'][key]['served_accept']:.4f}) "
                          f"ids_equal={P3[key]['ids_equal']}", flush=True)
                    R["phase3"] = P3
                    save()
            d = [P3[k]["served_accept"] - R["prompts"][k]["served_accept"] for k in P3]
            R["phase3_summary"] = {"daccept": ms(d), "ids_equal": sum(v["ids_equal"] for v in P3.values()),
                                   "n": len(P3)}
            for cls in classes:
                R["phase3_summary"][cls] = ms([P3[k]["served_accept"] for k in P3 if k.startswith(cls + "/")])
            print("PHASE3 " + json.dumps(R["phase3_summary"]), flush=True)
            set_arm("post", "q2")
        save()
    print("DONE", flush=True)


ARMS_ORDER = []
if __name__ == "__main__":
    main()
