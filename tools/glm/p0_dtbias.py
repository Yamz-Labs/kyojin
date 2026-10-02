# Decode-path dt_bias precision check (run with EXL3_KDA_DT_F32=1 so the graph buffer is fp32).
# Arms switch in place: "bf16" = buffer holds bf16-rounded dt_bias (bit-identical to the old bf16
# kernel input), "fp32" = exact fp32 dt_bias (what prefill and HF use). One load:
#   1) short smoke: the glm_base 3 chat prompts, 96 greedy tokens via Generator, per arm
#   2) PPL 1 x 2048 (cacheless forward) per arm (expected bit-identical: prefill path)
#   3) curve doc (frozen ids) prefilled to S = E-512; reference logits = one 512-token prefill chunk;
#      then per arm roll back to S and decode the same 512 tokens one at a time (block graph):
#      NLL, KL(prefill || decode), top-1 agreement vs the prefill logits
#   4) per arm: greedy 64 tokens from E (the 128K repetition loop)
# Usage: p0_dtbias.py <model_dir> <out.json>
import json, math, os, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
E = int(os.environ.get("P0_END", "131072"))
N = int(os.environ.get("P0_TF", "512"))
ARMS = os.environ.get("P0_ARMS", "bf16;fp32;bf16").split(";")
model_dir, out = sys.argv[1], sys.argv[2]
sys.argv = [sys.argv[0], "fast", "-m", model_dir, "-o", out + ".gb", "--corpus", os.path.expanduser("~/bench/ppl/wiki.test.raw")]
sys.path.insert(0, HERE)
import glm_base as gb                                              # noqa: E402  (argparse at import)
RES = {"mode": "p0_dtbias", "end": E, "tf": N, "arms": ARMS, "env": {k: v for k, v in os.environ.items() if k.startswith("EXL3_")},
       "smoke": {}, "ppl": {}, "tf_res": {}, "greedy": {}}


def save():
    with open(out, "w") as f:
        json.dump(RES, f, indent=1)


@torch.inference_mode()
def main():
    from exllamav3 import Config, Model, Cache, Tokenizer, Generator
    import exllamav3.modules.gated_delta_net as gdn
    assert gdn._KDA_DT_F32, "run with EXL3_KDA_DT_F32=1"
    PAGE, C = 256, 2048
    ids = torch.load(os.path.join(HERE, "data", "curve_ids_140k.pt"))
    import hashlib
    RES["curve_sha_E1"] = hashlib.sha1(ids[0, : E + 1].numpy().tobytes()).hexdigest()
    cap = ((E + 128 + PAGE - 1) // PAGE + 1) * PAGE
    t0 = time.time()
    config = Config.from_directory(model_dir)
    model = Model.from_config(config)
    tok = Tokenizer.from_config(config)
    cache = Cache(model, max_num_tokens=cap, max_batch_size=1)
    small = Cache(model, max_num_tokens=6144)                      # must exist before load
    model.load(device="cuda:0", progressbar=False)
    V = tok.actual_vocab_size
    RES["load_s"] = round(time.time() - t0, 1)

    def diag(tag):
        g = [m for m in model if isinstance(m, gdn.GatedDeltaNet)]
        d0 = g[0] if g else None
        info = {"n_gdn": len(g), "n_kda": sum(bool(m.kda) for m in g), "n_bc_split": sum(bool(getattr(m, "bc_split", False)) for m in g)}
        if d0 is not None:
            q = lambda x: None if x is None else getattr(x, "quant_type", type(x).__name__)
            info["m0"] = {"key": d0.key, "bc": type(getattr(d0, "bc", None)).__name__,
                          "qkv": q(d0.qkv_proj), "o": q(d0.o_proj),
                          "small": [q(getattr(d0, n, None)) for n in ("b_proj", "f_a_proj", "f_b_proj", "g_a_proj", "g_b_proj")],
                          "conv_flat": str(getattr(getattr(d0, "conv1d_weight_flat", None), "dtype", None)),
                          "conv_bias": str(getattr(getattr(d0, "conv1d_bias", None), "dtype", None)),
                          "dt_bias": str(getattr(d0.dt_bias, "dtype", None)),
                          "dt_bias_bc": str(getattr(getattr(d0, "dt_bias_bc", None), "dtype", None))}
        RES.setdefault("diag", {})[tag] = info
        print(f"[diag] {tag} {json.dumps(info)}", flush=True)
        return [m for m in g if m.kda and getattr(m, "bc_split", False)]

    diag("load")
    print(f"loaded {RES['load_s']}s", flush=True)

    def set_arm(arm):
        for m in kdas:
            assert m.dt_bias_bc.dtype == torch.float
            src = m.dt_bias.float()
            m.dt_bias_bc.copy_(src.bfloat16().float() if arm.startswith("bf16") else src)
            assert m.ba_weight_filled, "BC buffers not filled yet"
        torch.cuda.synchronize()

    # 1) + 2) smoke and PPL on a small generator cache (first forward fills the BC buffers, then arms override)
    gen = Generator(model=model, cache=small, tokenizer=tok)
    gb.run_job(gen, gb.chat(tok, "Hi"), 2, stop=False)          # deferred fill happens here
    kdas = diag("warm")
    RES["n_kda_bc"] = len(kdas)
    save()
    if not kdas:
        print("[short] NO_BC: decode does not use the BC gate kernel", flush=True)
        return
    for k, arm in enumerate(ARMS):
        set_arm(arm)
        tag = f"{arm}#{k}"
        RES["smoke"][tag] = [gb.run_job(gen, gb.chat(tok, p), 96, stop=False)["ids"].tolist() for p in gb.PROMPTS]
        RES["ppl"][tag] = gb.eval_ppl(model, tok, 1, 2048, tag)["ppl"]
        print(f"[short] {tag} ppl {RES['ppl'][tag]:.6f} ids {[r[:6] for r in RES['smoke'][tag]]}", flush=True)
        save()
    del gen

    # 3) teacher-forced decode vs prefill at depth
    bt = torch.arange(cap // PAGE, dtype=torch.int32).unsqueeze(0)
    st = cache.get_new_state()
    S = E - N

    def fwd(x, pos, prefill_only=False):
        T = x.shape[-1]
        p = {"attn_mode": "flash_attn", "block_table": bt[:, : (pos + T + PAGE - 1) // PAGE], "cache": cache,
             "cache_seqlens": torch.tensor([pos], dtype=torch.int32), "recurrent_states": [st]}
        return model.prefill(x, p) if prefill_only else model.forward(x, p)

    a = time.time()
    for p0 in range(0, S, C):
        fwd(ids[:, p0 : min(p0 + C, S)], p0, prefill_only=True)
        if (p0 + C) % 32768 == 0:
            print(f"[pre] {p0 + C} {time.time() - a:.0f}s", flush=True)
    RES["prefill_s"] = round(time.time() - a, 1)
    base = st.stash()
    tgt = ids[0, S + 1 : E + 1]
    ref = {}
    for arm in dict.fromkeys(ARMS):                               # prefill reference per distinct arm (must be identical)
        set_arm(arm)
        st.position = S; st.unstash(base)
        ref[arm] = torch.log_softmax(fwd(ids[:, S:E], S)[0, :, :V].float(), dim=-1)
    arms_u = list(ref)
    RES["prefill_ref_arm_maxdiff"] = (ref[arms_u[0]] - ref[arms_u[-1]]).abs().max().item()
    lpr = ref[arms_u[0]]
    RES["prefill_ref_nll"] = -lpr.gather(-1, tgt.to(lpr.device).unsqueeze(-1)).mean().item()
    print(f"[ref] prefill nll {RES['prefill_ref_nll']:.4f} arm maxdiff {RES['prefill_ref_arm_maxdiff']}", flush=True)
    save()
    for k, arm in enumerate(ARMS):
        set_arm(arm)
        st.position = S; st.unstash(base)
        a = time.time()
        nl, kl, top1, klmax = 0.0, 0.0, 0, 0.0
        for i in range(N):
            lp = torch.log_softmax(fwd(ids[:, S + i : S + i + 1], S + i)[0, -1, :V].float(), dim=-1)
            r = lpr[i]
            nl += -lp[int(tgt[i])].item()
            d = (r.exp() * (r - lp)).sum().item()
            kl += d; klmax = max(klmax, d)
            top1 += int(lp.argmax().item() == r.argmax().item())
        res = {"nll": round(nl / N, 5), "kl_mean": round(kl / N, 6), "kl_max": round(klmax, 5),
               "top1_agree": round(top1 / N, 4), "s": round(time.time() - a, 1)}
        RES["tf_res"][f"{arm}#{k}"] = res
        print(f"[tf] {arm}#{k} {json.dumps(res)}", flush=True)
        save()

    # 4) greedy from E, per arm
    set_arm(arms_u[0])
    st.position = S; st.unstash(base)
    lg = fwd(ids[:, S:E], S)
    first = int(lg[0, -1, :V].argmax().item()); del lg
    baseE = st.stash()
    for k, arm in enumerate(ARMS):
        set_arm(arm)
        st.position = E; st.unstash(baseE)
        nxt, outt = first, [first]
        for j in range(63):
            nxt = int(fwd(torch.tensor([[nxt]], dtype=torch.long), E + j)[0, -1, :V].argmax().item())
            outt.append(nxt)
        txt = tok.decode(torch.tensor([outt]))
        txt = (txt[0] if isinstance(txt, list) else txt)
        RES["greedy"][f"{arm}#{k}"] = {"ids": outt, "text": txt[:300]}
        print(f"[greedy] {arm}#{k} {txt[:160]!r}", flush=True)
        save()
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
