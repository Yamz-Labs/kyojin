"""Decode-faithful KL of MLA latent cache formats on one GLM pack, one model load.

  tq_mla_kl.py PACK DOCS.pt OUT.json [N_PREFIX] [N_DECODE] [formats...]

Formats: fp16 (reference), k8 / k6 / k5 (CacheLayer_MLA_quant), fq_tq6 / fq_tq5 (fake TurboQuant: rotate, Lloyd-Max encode, decode,
un-rotate, store fp16: exact TurboQuant numerics with no kernel), tq6 / tq5 (CacheLayer_MLA_tq, the real layer).
All caches are attached before the load; the prefix is prefilled into each in 2048-row chunks, then N_DECODE teacher-forced decode
steps run through the engine's decode path. KL is against the fp16 cache's log-probs.
Run with EXL3_BC_ATTN=0 for the fake formats (the graph path appends through its own kernel and bypasses update_kv_direct)."""
import gc
import json
import os
import sys
import time

import torch


def fake_tq(bits, blocks=1, gamma=False):
    from exllamav3.cache import CacheLayer_MLA_fp16, CacheLayer_fp16
    from exllamav3.modules.mla_attn import MLAttention
    orig = MLAttention.cache_layer_type
    if not getattr(orig, 'patched', False):
        def clt(self, default, kwargs):
            mc = getattr(default, 'mla_cls', None)
            return (mc, {}) if mc else orig(self, default, kwargs)
        clt.patched = True
        MLAttention.cache_layer_type = clt
    from exllamav3.cache.tq import TQState, encode_ref, decode_ref

    class FakeTQ(CacheLayer_MLA_fp16):
        def update_kv_direct(self, cache_seqlens, block_table, k, v, length):
            from exllamav3.modules.attention_fn.mla_triton import mla_kv_append
            full = self.kv_lora_rank
            dim = full // blocks
            st = TQState.get(dim, bits, bits, k.device)
            x = k.reshape(-1, full).float()
            if gamma:
                g = self.attention.kv_a_layernorm.weight.data.float().to(k.device)
                x = x / g
            x = x.reshape(-1, dim).half()
            y = st.rotate(x)
            words, sc = encode_ref(y, st.cbk, st.midk, bits)
            xr = st.unrotate(decode_ref(words, sc, st.cbk, bits, dim)).reshape(-1, full)
            if gamma:
                xr = (xr.float() * g).half()
            mla_kv_append(xr.reshape(k.shape[0], length, full), v.reshape(v.shape[0], length, self.qk_rope_head_dim),
                          self.k, self.v, block_table, cache_seqlens)
    class Req(CacheLayer_fp16):
        mla_cls = FakeTQ
    return Req


def main(pack, docs, out, n_prefix=16384, n_dec=64, fmts=None):
    from exllamav3 import Cache, CacheLayer_quant, Config, Model
    fmts = fmts or ["fp16", "k8", "k6", "fq_tq6", "fq_tq5"]
    ids = torch.load(docs)["ids"][0][: n_prefix + n_dec]
    T = len(ids)
    e = T - n_dec
    m = Model.from_config(Config.from_directory(pack))
    caches = {}
    for f in fmts:
        kw = dict(max_num_tokens=(T + 255) // 256 * 256 + 256, max_batch_size=2, max_history=3)
        if f == "fp16":
            caches[f] = Cache(m, **kw)
        elif f in ("k8", "k6", "k5"):
            b = int(f[1:])
            caches[f] = Cache(m, layer_type=CacheLayer_quant, k_bits=b, v_bits=b, **kw)
        elif f.startswith("fq_tq"):
            # fq_tq6 | fq_tq6g (gamma-normalized) | fq_tq6b4 (4 blocks of 128) | fq_tq6gb4
            import re
            mt = re.match(r"fq_tq(\d)(g?)(?:b(\d))?$", f)
            caches[f] = Cache(m, layer_type=fake_tq(int(mt[1]), int(mt[3] or 1), bool(mt[2])), **kw)
        elif f.startswith("tq"):
            from exllamav3.cache.tq import CacheLayer_tq
            b = int(f[2:])
            caches[f] = Cache(m, layer_type=CacheLayer_tq, k_bits=b, v_bits=b, **kw)
    m.load(progressbar=False)
    x = torch.tensor([ids])
    bt = torch.arange((T + 256) // 256, dtype=torch.int32)[None]
    lps, res = {}, {}
    import os
    refp = os.path.join(os.path.expanduser(os.environ.get("TQ_MLA_KL_REF_DIR", "~/.cache/kyojin/tq_mla_kl")), f"ref_fp16_{n_prefix}_{n_dec}.pt")   # fp16 reference logprobs (written by a run that includes the fp16 format)
    prof = os.environ.get("PROF") == "1"       # PROF=1: kernel time table of 4 decode steps, no KL
    if "fp16" not in fmts and not prof:
        lps["fp16"] = torch.load(refp)
    for f in fmts:
        cache = caches[f]
        with torch.inference_mode():
            st = cache.get_new_state() if cache.recurrent_state_cls is not None else None

            def par(pos, t):
                p = {"attn_mode": "flash_attn", "cache": cache, "block_table": bt[:, : (pos + t + 255) // 256], "cache_seqlens": torch.tensor([pos], dtype=torch.int32)}
                if st is not None:
                    p["recurrent_states"] = [st]
                return p
            for a in range(0, e, 2048):
                m.prefill(x[:, a: min(a + 2048, e)], par(a, min(a + 2048, e) - a))
            rows = []
            steps = []
            if prof:
                from exllamav3.modules.mla_attn import MLAttention
                import collections
                acc = collections.defaultdict(float)

                def timed(cls, name):
                    f = getattr(cls, name)

                    def w(*a, **k):
                        torch.cuda.synchronize(); t = time.time()
                        r = f(*a, **k)
                        torch.cuda.synchronize(); acc[name] += time.time() - t
                        return r
                    setattr(cls, name, w)
                for nm in ("_attend_pre", "_indexer_topk_kpool", "_indexer_topk", "_attend_sparse", "_attend_split", "decode_flash_attn"):
                    timed(MLAttention, nm)
                for j in range(4):
                    m.forward(x[:, e + j: e + j + 1], par(e + j, 1))
                acc.clear()
                for j in range(4, 8):
                    m.forward(x[:, e + j: e + j + 1], par(e + j, 1))
                print("PROF ms/step (sync'd)", {k: round(v / 4 * 1000, 2) for k, v in acc.items()}, flush=True)
                return
            for j in range(n_dec):
                t0 = time.time()
                lg = m.forward(x[:, e + j: e + j + 1], par(e + j, 1))
                r = torch.log_softmax(lg.float().reshape(-1, lg.shape[-1])[-1], -1).cpu()      # .cpu() syncs the step
                steps.append(time.time() - t0)
                rows.append(r)
            dt = sorted(steps[4:] or steps)[len(steps[4:] or steps) // 2]      # median step, first 4 (Triton compiles) dropped
            if st is not None:
                st.free()
        lps[f] = torch.stack(rows)
        if f == "fp16":
            os.makedirs(os.path.dirname(refp), exist_ok=True)
            torch.save(lps[f], refp)
        if f != "fp16":
            r = lps["fp16"]
            kl = (r.exp() * (r - lps[f]).nan_to_num(0)).sum(-1)
            res[f] = dict(kl_vs_fp16=kl.mean().item(), top1=(r.argmax(-1) == lps[f].argmax(-1)).float().mean().item(), s_per_tok=dt)
        else:
            res[f] = dict(s_per_tok=dt)
        print(f, res[f], flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], a[3], int(a[4]) if len(a) > 4 else 16384, int(a[5]) if len(a) > 5 else 64, a[6:] or None)
