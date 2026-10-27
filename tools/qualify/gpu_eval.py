"""Pack-side GPU measurements, run with the backend python (torch + exllamav3 on PYTHONPATH). CLI only; stages parse its JSON.
  gpu_eval.py parity PACK DOCS.pt REFDIR OUT.json            KL / top-1 / dNLL per doc (fp16 KV) against the streamed reference logits
  gpu_eval.py kv     PACK DOCS.pt REFDIR OUT.json B1 B2 ...  tail KL of symmetric quantized KV (k<b>v<b>) at the long doc, vs reference and vs k8v8
  gpu_eval.py kvmode PACK DOCS.pt OUT.json [N]               decode-faithful KL vs k8v8 at the long doc of the KV formats k6v6 (quant), kw6 (k6v6 + sink/window ring), tq6, tq5 (TurboQuant + ring)
KL maths and the real-quantizer forward are adapted from nanb kv_kl.py (drill 0)."""
import gc
import json
import sys
import time

import torch


def stats(a, b, tgt):
    """a = reference logits, b = test logits, both [T,V]; tgt = next tokens [T]. -> KL(a||b), top-1 agreement, dNLL."""
    la, lb = torch.log_softmax(a.float(), -1), torch.log_softmax(b.float(), -1)
    kl = (la.exp() * (la - lb)).sum(-1).mean().item()
    t1 = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    dn = (-lb.gather(1, tgt[:, None]).mean() + la.gather(1, tgt[:, None]).mean()).item()
    return kl, t1, dn


def load(pack, T, kv=None):
    from exllamav3 import Cache, CacheLayer_quant, Config, Model
    m = Model.from_config(Config.from_directory(pack))
    cache = Cache(m, max_num_tokens=T + 256, layer_type=CacheLayer_quant, k_bits=kv[0], v_bits=kv[1]) if kv else Cache(m, max_num_tokens=T + 256)
    m.load(progressbar=False)
    return m, cache


def forward_tail(m, cache, ids, tail):
    T = len(ids)
    x = torch.tensor([ids])
    bt = torch.arange((T + 256) // 256, dtype=torch.int32)[None]
    with torch.inference_mode():
        for a in range(0, T - tail, 2048):
            e = min(a + 2048, T - tail)
            m.prefill(x[:, a:e], {"attn_mode": "flash_attn", "cache": cache, "block_table": bt, "cache_seqlens": torch.tensor([a], dtype=torch.int32)})
        lg = m.forward(x[:, T - tail:], {"attn_mode": "flash_attn", "cache": cache, "block_table": bt, "cache_seqlens": torch.tensor([T - tail], dtype=torch.int32), "last_tokens_only": tail})
    return lg[0].float()


def free(m, cache):
    m.unload(); del cache, m; gc.collect(); torch.cuda.empty_cache()


def parity(pack, docs, ref, out):
    d = torch.load(docs)
    res = {}
    m, cache = load(pack, max(len(i) for i in d["ids"]))
    for n, ids in zip(d["names"], d["ids"]):
        r = torch.load(f"{ref}/logits_{n}.pt").float().cuda()       # [T,V]; row i predicts ids[i+1]
        T = len(ids)
        lg = forward_tail(m, cache, ids, T)
        kl, t1, dn = stats(r[:-1], lg[:-1], torch.tensor(ids[1:]).cuda())
        res[n] = dict(kl=kl, top1=t1, dnll=dn, tokens=T - 1)
        print(n, res[n], flush=True)
    free(m, cache)
    json.dump(res, open(out, "w"), indent=1)


def kv(pack, docs, ref, out, bits):
    d = torch.load(docs)
    ids = d["ids"][0]
    r = torch.load(f"{ref}/logits_{d['names'][0]}.pt").float().cuda()
    tail = r.shape[0]
    tgt = torch.tensor(ids[-tail + 1:]).cuda(); r = r[:-1]
    res, lgs = {}, {}
    for b in [8] + [x for x in bits if x != 8]:
        m, cache = load(pack, len(ids), (b, b))
        lg = forward_tail(m, cache, ids, tail)[:-1]
        free(m, cache)
        kl, t1, dn = stats(r, lg, tgt)
        res[b] = dict(kl_ref=kl, top1_ref=t1, dnll=dn)
        lgs[b] = lg.half()
        if b != 8:
            k2, t2, _ = stats(lgs[8], lg, tgt)
            res[b].update(kl_vs_k8=k2, top1_vs_k8=t2)
        print(f"k{b}v{b}", res[b], flush=True)
    json.dump(res, open(out, "w"), indent=1)


def kvmode(pack, docs, out, n=256):
    """Prefill ids[:T-n] in 2048-row chunks (quantize-on-write, the KVWIN ring bound for the tq formats), then n teacher-forced decode steps
    through the engine's own decode path, once per KV format; KL of each format's log-probs against k8v8's. No BF16 reference needed.
    One model load per format: a 33B pack on a 24 GB card has no room for several 32K caches at once."""
    import os
    from exllamav3 import Cache, CacheLayer_quant, Config, Model
    from exllamav3.cache import tq
    ids = torch.load(docs)["ids"][0]
    T = len(ids)
    e = T - n
    cfg = Config.from_directory(pack)
    hd = cfg.head_dim
    fmts = [("k8v8", "quant", 8), ("k6v6", "quant", 6), ("kw6", "quant", 6)] + [(f"tq{b}", "tq", b) for b in (6, 5) if hd > 0 and tq.supported(hd, b)]
    x = torch.tensor([ids])
    npg = (T + 256) // 256
    bt = torch.arange(npg, dtype=torch.int32)[None]
    res, lps = {}, {}
    for name, kind, b in fmts:
        m = Model.from_config(Config.from_directory(pack))
        os.environ.pop("EXL3_KVWIN", None)
        if kind == "tq":
            cache = Cache(m, max_num_tokens=T + 256, max_batch_size=2, max_history=3, layer_type=tq.CacheLayer_tq, k_bits=b, v_bits=b)
        else:
            os.environ["EXL3_KVWIN"] = "4:128" if name == "kw6" else "0"     # the production formats at >= 6 bits run without the window; kw6 = the same cache with the ring
            cache = Cache(m, max_num_tokens=T + 256, max_batch_size=2, max_history=3, layer_type=CacheLayer_quant, k_bits=b, v_bits=b)
            os.environ.pop("EXL3_KVWIN")
        m.load(progressbar=False)
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
            t_dec = time.time()
            for j in range(n):
                lg = m.forward(x[:, e + j: e + j + 1], par(e + j, 1))
                rows.append(torch.log_softmax(lg.float().reshape(-1, lg.shape[-1])[-1], -1).half().cpu())
            t_dec = (time.time() - t_dec) / n
            if st is not None:
                st.free()
        lps[name] = torch.stack(rows).float()
        free(m, cache)
        if name != "k8v8":
            r8 = lps["k8v8"]
            kl = (r8.exp() * (r8 - lps[name]).nan_to_num(0)).sum(-1)
            res[name] = dict(kl_vs_k8=kl.mean().item(), top1_vs_k8=(r8.argmax(-1) == lps[name].argmax(-1)).float().mean().item(),
                             bits_per_val=(b + 0.5) if kind == "quant" else b + 16 / hd, decode_s_per_tok=t_dec)
            print(name, res[name], flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "parity":
        parity(*sys.argv[2:6])
    elif cmd == "kv":
        kv(*sys.argv[2:6], [int(x) for x in sys.argv[6:]])
    elif cmd == "kvmode":
        kvmode(*sys.argv[2:5], *[int(x) for x in sys.argv[5:6]])
