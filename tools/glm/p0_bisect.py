# P0 bisect: which depth-dependent component breaks GLM quality past 64-80K?
#
# One load, one prefill of the depth_curve document to S = E - 8192, then arms over tokens S..E that
# each score the ctrl targets (chunks p0 >= E - 4096), rolling back to S between arms (KDA stash +
# attention cache rewind by seqlen):
#   A  normal                                   (reproduces curve: ~3.9 at E = 131072)
#   B  KDA/conv state zeroed at S, DSA full     (recovers -> KDA long state is the cause)
#   C  KDA full, DSA pools restricted to >= S   (recovers -> long DSA selection is the cause)
#   D  both short                               (reproduces ctrl: ~1.2)
#   E  dense latent MLA over the full cache, KDA full (DSA off; tells selection vs attention)
#   A2 normal again                             (rollback sanity)
# During prefill, every 16K it logs the per-layer KDA recurrent-state norm and max-abs.
import json, os, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
E = int(os.environ.get("P0_END", "131072"))
# Arms separated by ";". Zi-j zeroes the KDA state of the i..j-th recurrent layers only (0-based, in
# cache order), e.g. "A;Z0-16;Z17-33;Z5"
ARMS = os.environ.get("P0_ARMS", "A;B;C;D;E;A2").split(";")
model_dir, out = sys.argv[1], sys.argv[2]
sys.argv = [sys.argv[0], "ctrl", "-m", model_dir, "-o", out, "--total", "512000"]
sys.path.insert(0, HERE)
import depth_curve as dc                                          # noqa: E402  (argparse at import)

_walk = os.walk
def _walk_skip(top, *a, **k):                                     # keep the doc identical to curve.json
    for dp, dn, fn in _walk(top, *a, **k):
        yield dp, dn, [f for f in fn if f not in ("p0_bisect.py", "p0_kda_ref.py")]
dc.os.walk = _walk_skip
RES = dc.RES
RES.update(mode="p0_bisect", end=E, arms={}, kda=[])


@torch.inference_mode()
def main():
    from exllamav3 import Config, Model, Cache, Tokenizer
    import exllamav3.modules.attention_fn.dsa_triton as dt
    import exllamav3.modules.mla_attn as ma
    PAGE, C, S = 256, 2048, E - 8192
    cap = ((E + 64 + PAGE - 1) // PAGE + 1) * PAGE
    t0 = time.time()
    config = Config.from_directory(model_dir)
    model = Model.from_config(config)
    tok = Tokenizer.from_config(config)
    cache = Cache(model, max_num_tokens=cap, max_batch_size=1)
    model.load(device="cuda:0", progressbar=False)
    V = tok.actual_vocab_size
    RES["load_s"] = round(time.time() - t0, 1)
    dc.mem_guard("after_load")
    ids = dc.build_doc(tok, 512001)
    RES["doc_match_curve"] = RES["doc"]["sha1_first_need"] == "2dd044b6f3dac87dd1793463973a8beb5b8ac1a6"
    print(f"loaded {RES['load_s']}s doc_match {RES['doc_match_curve']}", flush=True)
    # KDA gate check: every fused kda_gate call is re-derived with the HF ForgetGate + sigmoid(b) in fp32
    import exllamav3.modules.gated_delta_net as gdn
    gchk = {"calls": 0, "dg": 0.0, "dbeta": 0.0, "beta_scales": set()}

    class _ExtProxy:
        def __init__(self, inner):
            self._inner = inner
        def __getattr__(self, n):
            return getattr(self._inner, n)
        def kda_gate(self, b, f, dt_bias, a_log, beta_scale, lb, has_lb, beta, g):
            self._inner.kda_gate(b, f, dt_bias, a_log, beta_scale, lb, has_lb, beta, g)
            H = a_log.numel()
            gf = (f.float() + dt_bias.float().view(1, 1, -1)).view(*f.shape[:2], H, -1)
            dec = torch.exp(a_log.float()).view(1, 1, H, 1)
            g_ref = lb * torch.sigmoid(dec * gf) if has_lb else -dec * torch.nn.functional.softplus(gf)
            gchk["calls"] += 1
            gchk["dg"] = max(gchk["dg"], (g.float().view_as(g_ref) - g_ref).abs().max().item())
            gchk["dbeta"] = max(gchk["dbeta"], (beta.float() - torch.sigmoid(b.float())).abs().max().item())
            gchk["beta_scales"].add(float(beta_scale))
    gdn.ext = _ExtProxy(gdn.ext)
    RES["fuse_kda_gate"] = bool(gdn._FUSE_KDA_GATE)
    bt = torch.arange(cap // PAGE, dtype=torch.int32).unsqueeze(0)
    st = cache.get_new_state()
    mlas = [m for m in model if isinstance(m, ma.MLAttention)]
    RES["n_mla"] = len(mlas)

    def fwd(x, pos):
        T = x.shape[-1]
        p = {"attn_mode": "flash_attn", "block_table": bt[:, : (pos + T + PAGE - 1) // PAGE], "cache": cache,
             "cache_seqlens": torch.tensor([pos], dtype=torch.int32), "recurrent_states": [st]}
        return model.forward(x, p)

    def nll(lg, p0, T):
        tgt = ids[0, p0 + 1 : p0 + T + 1].to(lg.device)
        lp = torch.log_softmax(lg[0, :, :V].float(), dim=-1)
        return -lp.gather(-1, tgt.unsqueeze(-1)).sum().item()

    def kda_stats(stash, pos):
        rows = []
        for k, v in stash.items():
            if isinstance(v, tuple):
                s = v[0].float()
                rows.append([round(s.norm().item(), 2), round(s.abs().max().item(), 3),
                             bool(torch.isfinite(s).all()), round(v[1].float().abs().max().item(), 2)])
        norms = [r[0] for r in rows]
        RES["kda"].append({"pos": pos, "layers": rows})
        print(f"[kda] pos {pos} norm min/med/max {min(norms):.1f}/{sorted(norms)[len(norms)//2]:.1f}/{max(norms):.1f} "
              f"maxabs {max(r[1] for r in rows):.2f} finite {all(r[2] for r in rows)}", flush=True)

    # Prefill 0..S (chunk NLL logged to compare with curve.json)
    a = time.time()
    pre = []
    for p0 in range(0, S, C):
        lg = fwd(ids[:, p0 : p0 + C], p0)
        pre.append(round(nll(lg, p0, C) / C, 4)); del lg
        if (p0 + C) % 16384 == 0:
            kda_stats(st.stash(), p0 + C)
            dc.mem_guard(f"prefill@{p0+C}")
    RES["prefill_s"] = round(time.time() - a, 1)
    RES["gate_check"] = {"calls": gchk["calls"], "max_abs_dg": gchk["dg"], "max_abs_dbeta": gchk["dbeta"],
                         "beta_scales": sorted(gchk["beta_scales"]), "fused": RES["fuse_kda_gate"]}
    print(f"[gate] {json.dumps(RES['gate_check'])}", flush=True)
    RES["prefill_chunk_nll_tail"] = pre[-8:]
    dc.save()
    print(f"prefill to {S} in {RES['prefill_s']}s; last chunks {pre[-4:]}", flush=True)
    base = st.stash()
    zero = {k: (tuple(t.clone().zero_() for t in v) if isinstance(v, tuple) else v) for k, v in base.items()}
    kda_keys = [k for k, v in base.items() if isinstance(v, tuple)]
    RES["kda_keys"] = [str(k) for k in kda_keys]
    orig_scores = dt.dsa_indexer_scores

    def restricted(*a, **k):                                      # single tile at E <= 131072: col = pool id
        sc = orig_scores(*a, **k)
        sc[:, : S // 4] = -float("inf")
        return sc

    for arm in ARMS:
        st.position = S
        if arm.startswith("Z"):
            lo, hi = (int(x) for x in (arm[1:] + "-" + arm[1:]).split("-")[:2])
            part = dict(base)
            for i, k in enumerate(kda_keys):
                if lo <= i <= hi:
                    part[k] = zero[k]
            st.unstash(part)
        else:
            st.unstash(zero if arm in ("B", "D") else base)
        if arm in ("C", "D"):
            dt.dsa_indexer_scores = restricted
        if arm == "E":
            for m in mlas:
                m._p0_topk = m.index_topk; m.index_topk = 1 << 30
            ma._prefill_mode = "latent"
        a = time.time()
        tot, n, err = 0.0, 0, None
        try:
            for p0 in range(S, E, C):
                lg = fwd(ids[:, p0 : p0 + C], p0)
                if p0 >= E - 4096:
                    tot += nll(lg, p0, C); n += C
                del lg
        except Exception as e:
            err = repr(e)[:300]
        finally:
            dt.dsa_indexer_scores = orig_scores
            if arm == "E":
                for m in mlas:
                    m.index_topk = m._p0_topk
                ma._prefill_mode = "mha"
            st.position = S + 8192 if err is None else st.position
        RES["arms"][arm] = {"nll": round(tot / n, 5) if n else None, "tokens": n, "s": round(time.time() - a, 1), "err": err}
        print(f"[arm] {arm} {json.dumps(RES['arms'][arm])}", flush=True)
        dc.save()
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
