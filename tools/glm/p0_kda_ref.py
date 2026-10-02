# P0 step 2: is the KDA prefill kernel (vendored fla chunk_kda) the cause of GLM depth loss?
#
# p0_bisect.py showed zeroing the KDA state at 122880 recovers the late-token NLL (3.87 -> 1.41) while
# restricting DSA to a short window does not (3.64). This run prefills the depth_curve document to E
# with every KDA prefill call ALSO computed by a torch fp32 chunked reference (the HF GLM5.3
# chunk_kimi_delta_attention math). Per call it logs the engine kernel's output/state error vs the
# reference from the SAME initial state (local error), then, with P0_KDA=replace, feeds the reference
# output and state forward instead (P0_KDA=compare keeps the engine's). Scores chunk NLL throughout and
# the ctrl targets (last 4096 before E).
import json, os, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
E = int(os.environ.get("P0_END", "131072"))
MODE = os.environ.get("P0_KDA", "replace")
model_dir, out = sys.argv[1], sys.argv[2]
sys.argv = [sys.argv[0], "ctrl", "-m", model_dir, "-o", out, "--total", "512000"]
sys.path.insert(0, HERE)
import depth_curve as dc                                          # noqa: E402

_walk = os.walk
SKIP = {"p0_bisect.py", "p0_kda_ref.py"}
def _walk_skip(top, *a, **k):
    for dp, dn, fn in _walk(top, *a, **k):
        yield dp, dn, [f for f in fn if f not in SKIP]
dc.os.walk = _walk_skip
RES = dc.RES
RES.update(mode=f"p0_kda_{MODE}", end=E, calls=[], chunks=[])


def l2norm(x):
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + 1e-6)


def ref_kda(q, k, v, g, beta, state, cs=64, sub=512):
    """fp32 chunked KDA (HF Glm5Next chunk_kimi_delta_attention). q/k/v (1,s,h,d), g (1,s,h,dk),
    beta (1,s,h); state (1,h,dk,dv) fp32, updated in place. Returns (1,s,h,dv) fp32."""
    q, k, v, g, beta = [x.transpose(1, 2).float() for x in (q, k, v, g, beta)]
    q = l2norm(q) * q.shape[-1] ** -0.5
    k = l2norm(k)
    s = q.shape[2]
    assert s % cs == 0
    outs = []
    mask = torch.triu(torch.ones(cs, cs, dtype=torch.bool, device=q.device), 0)
    smask = torch.triu(torch.ones(cs, cs, dtype=torch.bool, device=q.device), 1)
    eye = torch.eye(cs, device=q.device)
    for a in range(0, s, sub):
        b_ = min(a + sub, s)
        sh = lambda x: x[:, :, a:b_].reshape(x.shape[0], x.shape[1], -1, cs, x.shape[-1])
        qc, kc, vc, gc = sh(q), sh(k), sh(v), sh(g)
        bc = beta[:, :, a:b_].reshape(beta.shape[0], beta.shape[1], -1, cs)
        kb, vb = kc * bc.unsqueeze(-1), vc * bc.unsqueeze(-1)
        gc = gc.cumsum(-2)
        dm = (gc.unsqueeze(-2) - gc.unsqueeze(-3)).masked_fill(smask[..., None], float("-inf")).exp()
        attn = -(kb.unsqueeze(-2) * kc.unsqueeze(-3) * dm).sum(-1).masked_fill(mask, 0)
        for i in range(1, cs):
            row = attn[..., i, :i].clone(); sb = attn[..., :i, :i].clone()
            attn[..., i, :i] = row + (row.unsqueeze(-1) * sb).sum(-2)
        attn = attn + eye
        val = attn @ vb
        kcd = attn @ (kb * gc.exp())
        o = torch.empty_like(val)
        for i in range(val.shape[2]):
            qi, ki, gi = qc[:, :, i], kc[:, :, i], gc[:, :, i]
            inter = (qi * gi.exp()) @ state
            intra = (qi.unsqueeze(-2) * ki.unsqueeze(-3) * dm[:, :, i]).sum(-1).masked_fill(smask, 0)
            vn = val[:, :, i] - kcd[:, :, i] @ state
            o[:, :, i] = inter + intra @ vn
            state.mul_(gi[:, :, -1].exp().unsqueeze(-1)).add_((ki * (gi[:, :, -1:] - gi).exp()).transpose(-1, -2) @ vn)
        outs.append(o.reshape(o.shape[0], o.shape[1], -1, o.shape[-1]))
    return torch.cat(outs, 2).transpose(1, 2)


def rel(a, b):
    return ((a.float() - b.float()).norm() / (b.float().norm() + 1e-12)).item()


@torch.inference_mode()
def main():
    from exllamav3 import Config, Model, Cache, Tokenizer
    import exllamav3.modules.gated_delta_net as gdn
    PAGE, C = 256, 2048
    cap = ((E + 64 + PAGE - 1) // PAGE + 1) * PAGE
    config = Config.from_directory(model_dir)
    model = Model.from_config(config)
    tok = Tokenizer.from_config(config)
    cache = Cache(model, max_num_tokens=cap, max_batch_size=1)
    model.load(device="cuda:0", progressbar=False)
    V = tok.actual_vocab_size
    dc.mem_guard("after_load")
    ids = dc.build_doc(tok, 512001)
    RES["doc_match_curve"] = RES["doc"]["sha1_first_need"] == "2dd044b6f3dac87dd1793463973a8beb5b8ac1a6"
    print(f"loaded, doc_match {RES['doc_match_curve']} mode {MODE}", flush=True)
    orig = gdn.gated_delta_rule_fn
    call = {"errs": []}

    def hooked(mixed_qkv, beta, g, recurrent_state, recurrent_slots, history, save_state,
               num_k_heads, num_v_heads, k_dim, v_dim, k_head_dim, v_head_dim, params=None, channelwise_g=False):
        bsz, seqlen, _ = mixed_qkv.shape
        if not (channelwise_g and seqlen >= num_v_heads and not history and recurrent_state is not None and bsz == 1):
            return orig(mixed_qkv, beta, g, recurrent_state, recurrent_slots, history, save_state, num_k_heads,
                        num_v_heads, k_dim, v_dim, k_head_dim, v_head_dim, params, channelwise_g)
        st = recurrent_state[0, 0].unsqueeze(0)
        s0 = st.clone()
        o_e = orig(mixed_qkv, beta, g, recurrent_state, recurrent_slots, history, save_state, num_k_heads,
                   num_v_heads, k_dim, v_dim, k_head_dim, v_head_dim, params, channelwise_g)
        q, k, v = torch.split(mixed_qkv, [k_dim, k_dim, v_dim], dim=-1)
        sr = s0.clone()
        o_r = ref_kda(q.reshape(1, seqlen, -1, k_head_dim), k.reshape(1, seqlen, -1, k_head_dim),
                      v.reshape(1, seqlen, -1, v_head_dim), g, beta, sr)
        call["errs"].append((rel(o_e, o_r), rel(st, sr), sr.norm().item()))
        if MODE == "replace":
            st.copy_(sr)
            return o_r.to(torch.bfloat16).contiguous()
        return o_e

    gdn.gated_delta_rule_fn = hooked
    bt = torch.arange(cap // PAGE, dtype=torch.int32).unsqueeze(0)
    st = cache.get_new_state()
    a = time.time()
    tot, n = 0.0, 0
    for p0 in range(0, E, C):
        p = {"attn_mode": "flash_attn", "block_table": bt[:, : (p0 + C + PAGE - 1) // PAGE], "cache": cache,
             "cache_seqlens": torch.tensor([p0], dtype=torch.int32), "recurrent_states": [st]}
        call["errs"] = []
        lg = model.forward(ids[:, p0 : p0 + C], p)
        tgt = ids[0, p0 + 1 : p0 + C + 1].to(lg.device)
        lp = torch.log_softmax(lg[0, :, :V].float(), dim=-1)
        s = -lp.gather(-1, tgt.unsqueeze(-1)).sum().item()
        del lp, lg
        if p0 >= E - 4096:
            tot += s; n += C
        e = call["errs"]
        row = {"end": p0 + C, "nll": round(s / C, 4), "n_kda": len(e),
               "out_rel_max": round(max(x[0] for x in e), 5) if e else None,
               "state_rel_max": round(max(x[1] for x in e), 5) if e else None,
               "state_norm_max": round(max(x[2] for x in e), 2) if e else None}
        RES["chunks"].append(row)
        if (p0 + C) % 16384 == 0:
            print(f"[chunk] {json.dumps(row)} t {time.time()-a:.0f}s", flush=True)
            dc.mem_guard(f"@{p0+C}")
            dc.save()
    RES["nll_last4k"] = round(tot / n, 5)
    RES["s"] = round(time.time() - a, 1)
    dc.save()
    print(f"[result] mode {MODE} E {E} nll_last4k {RES['nll_last4k']} ({RES['s']}s)", flush=True)


if __name__ == "__main__":
    main()
