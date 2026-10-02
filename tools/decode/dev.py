#!/usr/bin/env python
"""Decode-path dev harness for MiMo-V2.6 EXL3 on gfx1151. One model load per invocation.

Modes (comma-separated, run in order on one load):
  tf      teacher-forced decode: prefill --tf-prefill tokens of a wikitext slice, then feed the next
          --tf-steps reference tokens one at a time through the batch-1 decode path, recording
          log-softmax per step. --tf-save PATH writes them; --tf-ref PATH compares (KLD, top-1).
  greedy  3 chat prompts, greedy, --greedy-tokens each, printed.
  prof    2K prompt, then --prof-tokens decode steps bracketed by marker kernels (fill of
          MARKER_NUMEL floats) so a rocprofv3 kernel trace can be cut to the decode window.
  speed   decode tok/s at --ctx, median of --reps, unique prompt per rep.
  --ref32 replaces every batch-1 EXL3 GEMV and the routed MoE sum by an fp32 computation on the
          exact codebook weights (Hadamards applied to the activations in fp32), for a
          higher-precision tf reference that both decode paths can be scored against.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.environ.get("EXL3_REPO", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import torch

from exllamav3 import Generator, GreedySampler, Job, model_init

MARKER_NUMEL = 7777777
CORPUS = next((c for c in ("~/bench/ppl/wiki.test.raw", "~/bench/ppl/wiki.test.raw") if os.path.exists(c)), "")
PROMPTS = [
    "What is the capital of Australia, and why is it not Sydney? Answer in three sentences.",
    "Write a Python function `is_prime(n: int) -> bool` that returns True if n is prime, and nothing else.",
    "Explique en trois phrases pourquoi le ciel est bleu.",
]


class ForcedSampler(GreedySampler):
    """Records log-softmax of each decode step's logits and returns the reference token."""

    def __init__(self, ref_ids, vocab):
        super().__init__()
        self.ref = ref_ids
        self.vocab = vocab
        self.rec = []

    def forward(self, logits, *args, **kwargs):
        i = len(self.rec)
        lp = torch.log_softmax(logits.view(-1, logits.shape[-1])[-1, :self.vocab].float(), dim = -1)
        self.rec.append(lp.half().cpu())
        tok = int(self.ref[i]) if i < len(self.ref) else 0
        return torch.full(logits.shape[:-1], tok, dtype = torch.long, device = logits.device)


class RecordingSampler(GreedySampler):
    """Greedy, and records the token ids it picks so two arms can be diffed exactly."""

    def __init__(self):
        super().__init__()
        self.ids = []

    def forward(self, logits, *args, **kwargs):
        tok = super().forward(logits, *args, **kwargs)
        self.ids.append(int(tok.view(-1)[0]))
        return tok


def run_job(generator, ids, n, sampler):
    job = Job(input_ids = ids, max_new_tokens = n, stop_conditions = [], sampler = sampler)
    generator.enqueue(job)
    t0 = time.time()
    ttft = None
    ntok = 0
    text = ""
    while generator.num_remaining_jobs():
        for res in generator.iterate():
            if res.get("error"):
                raise RuntimeError(res["error"])
            if res.get("stage") == "streaming" or "text" in res:
                if res.get("text") is not None or res.get("token_ids") is not None:
                    if ttft is None:
                        torch.cuda.synchronize()
                        ttft = time.time() - t0
                    ntok += 1
                    text += res.get("text", "") or ""
    torch.cuda.synchronize()
    return text, ttft, ntok, time.time() - t0


def rel_l2(a, b):
    a = a.float(); b = b.float()
    return ((a - b).norm() / (b.norm() + 1e-30)).item()


_HAD = {}


def gemv32(inner, x):
    """y = x @ W in fp32 for one EXL3 inner Linear: exact codebook weights (reconstruct), both
    Hadamards applied to the activations in fp32."""
    import math
    from exllamav3.util.hadamard import get_hadamard_dt
    from exllamav3.ext import exllamav3_ext as real_ext
    dev = x.device
    if dev not in _HAD:
        _HAD[dev] = get_hadamard_dt(128, dev, torch.float, 1 / math.sqrt(128))
    H = _HAD[dev]
    suh = (inner.unpack_bf(inner.su) if inner.su is not None else inner.suh).float()
    svh = (inner.unpack_bf(inner.sv) if inner.sv is not None else inner.svh).float()
    a = ((x.float().reshape(1, -1) * suh).view(-1, 128) @ H).view(1, -1)
    n = inner.out_features
    z = torch.empty((1, n), dtype = torch.float, device = dev)
    CH = 16384
    for c0 in range(0, n, CH):
        c1 = min(n, c0 + CH)
        tr = inner.trellis[:, c0 // 16:c1 // 16].contiguous()
        w = torch.empty((inner.in_features, c1 - c0), dtype = torch.half, device = dev)
        real_ext.reconstruct(w, tr, inner.K, inner.mcg, inner.mul1)
        z[:, c0:c1] = a @ w.float()
    y = (z.view(-1, 128) @ H).view(1, n) * svh
    if inner.bias is not None:
        y = y + inner.bias.float()
    return y


def install_ref32(model):
    """Batch-1 decode in fp32: LinearEXL3.dec_gemv and exl3_dec_moe become fp32 torch code."""
    import math
    from exllamav3.modules.quant.exl3 import LinearEXL3
    from exllamav3.modules import block_sparse_mlp as bsm
    from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
    from exllamav3.util.hadamard import get_hadamard_dt
    from exllamav3.ext import exllamav3_ext as real_ext
    def dec_gemv(self, x, out_dtype):
        y = gemv32(self, x)
        return y.to(out_dtype or self.default_out_dtype).view(x.shape[:-1] + (self.out_features,))
    LinearEXL3.dec_gemv = dec_gemv

    by_ptr = {}
    for m in model.modules:
        for sm in [m] + list(getattr(m, "modules", [])):
            if isinstance(sm, BlockSparseMLP) and sm.multi_gate is not None:
                by_ptr[sm.multi_gate.ptrs_trellis.data_ptr()] = sm

    def dec_moe(y, out, sel, wts, gtr, *rest):
        mod = by_ptr[gtr.data_ptr()]
        acc = torch.zeros((1, out.shape[-1]), dtype = torch.float, device = y.device)
        for e, w in zip(sel.view(-1).tolist(), wts.view(-1).float().tolist()):
            g = gemv32(mod.gates[e].inner, y)
            u = gemv32(mod.ups[e].inner, y)
            acc += w * gemv32(mod.downs[e].inner, torch.nn.functional.silu(g) * u)
        if rest[-1]:
            out.view(1, -1).add_(acc)
        else:
            out.view(1, -1).copy_(acc)

    class Proxy:
        def __getattr__(self, k):
            return dec_moe if k == "exl3_dec_moe" else getattr(real_ext, k)
    bsm.ext = Proxy()
    print(f"ref32: fp32 batch-1 GEMV + MoE installed ({len(by_ptr)} MoE layers)", flush = True)


def w32(lin):
    """Dense fp32 weight [in, out] of an EXL3 Linear, Hadamards applied in fp32."""
    from exllamav3.modules.quant.exl3_lib.quantize import preapply_had_l, preapply_had_r
    inner = lin.inner
    suh = (inner.unpack_bf(inner.su) if inner.su is not None else inner.suh).float().unsqueeze(1)
    svh = (inner.unpack_bf(inner.sv) if inner.sv is not None else inner.svh).float().unsqueeze(0)
    w = inner.get_inner_weight_tensor().float()
    w = preapply_had_l(w, 128) * suh
    w = preapply_had_r(w, 128) * svh
    return w


def gate32(model, generator, tokenizer, all_ids):
    """Every layer vs an fp32 reference (gemv32) on inputs captured from one real decode step:
    rel-L2 of the exl3_dec path (new) and of the pre-existing path (old)."""
    from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
    blocks = sorted((m for m in model.modules if hasattr(m, "mlp") and hasattr(m, "layer_idx")), key = lambda m: m.layer_idx)
    head = model.modules[model.logit_layer_idx]
    want = {}
    for b in blocks:
        li = b.layer_idx
        if b.mlp is not None: want[f"L{li}.mlp"] = b.mlp
        if b.attn is not None:
            want[f"L{li}.attn"] = b.attn
            want[f"L{li}.o"] = b.attn.o_proj
    want["head"] = head
    caps, origs = {}, {}
    for name, mod in want.items():
        orig = mod.forward
        origs[name] = orig
        def wrapped(x, params, *a, _n = name, _o = orig, **k):
            if _n not in caps and x.numel() == x.shape[-1]:
                caps[_n] = x.detach().clone()
            return _o(x, params, *a, **k)
        mod.forward = wrapped
    run_job(generator, all_ids[:, 70000:70000 + 256], 3, GreedySampler())
    for name, mod in want.items():
        mod.forward = origs[name]

    def off(lins):
        oks = [l.inner.dec_ok for l in lins]
        for l in lins: l.inner.dec_ok = False
        return oks

    def on(lins, oks):
        for l, o in zip(lins, oks): l.inner.dec_ok = o

    agg = {}
    def rec(kind, name, n, o):
        agg.setdefault(kind, []).append((n, o, name))

    for name, mod in want.items():
        if name not in caps: continue
        x = caps[name]
        if isinstance(mod, BlockSparseMLP):
            y = x.view(-1, mod.hidden_size)
            s0, w0 = mod.routing_fn(1, mod.routing_cfg, y, {})
            s0, w0 = s0.clone(), w0.clone()
            out_new = mod.forward(x, {}).clone().float().view(1, -1)
            keep = mod.support_dec_moe
            mod.support_dec_moe = False
            el = [l for l in list(mod.gates) + list(mod.ups) + list(mod.downs) if getattr(l, "quant_type", None) == "exl3"]
            oks = off(el)
            out_old = mod.forward(x, {}).clone().float().view(1, -1)
            on(el, oks)
            mod.support_dec_moe = keep
            y32 = y.float()
            ref = torch.zeros_like(out_new)
            for e, w in zip(s0[0].tolist(), w0[0].float().tolist()):
                a = torch.nn.functional.silu(gemv32(mod.gates[e].inner, y32)) * gemv32(mod.ups[e].inner, y32)
                ref += w * gemv32(mod.downs[e].inner, a)
            rec("moe", name, rel_l2(out_new, ref), rel_l2(out_old, ref))
        elif hasattr(mod, "q_proj"):
            lins = [l for l in (mod.q_proj, mod.k_proj, mod.v_proj) if l is not None]
            q1, k1, v1, _ = mod.project_qkv(x, {})
            q1, k1, v1 = q1.clone(), k1.clone(), v1.clone()
            keep = mod._dec_qkv_linears()
            mod._dec_qkv = None
            oks = off(lins)
            q0, k0, v0, _ = mod.project_qkv(x, {})
            on(lins, oks)
            mod._dec_qkv = keep
            for tag, l, t1, t0 in (("q", mod.q_proj, q1, q0), ("k", mod.k_proj, k1, k0), ("v", mod.v_proj, v1, v0)):
                if tag == "v" and getattr(mod, "v_norm", None) is not None: continue
                r = gemv32(l.inner, x)
                rec(tag, name, rel_l2(t1.reshape(1, -1).float(), r), rel_l2(t0.reshape(1, -1).float(), r))
        else:
            lins = [m for m in ([mod] + list(getattr(mod, "gates", [])) + list(getattr(mod, "ups", [])) + list(getattr(mod, "downs", [])))
                    if m is not None and getattr(m, "quant_type", None) == "exl3"]
            if not lins and hasattr(mod, "modules"):
                lins = [m for m in mod.modules if getattr(m, "quant_type", None) == "exl3"]
            out_new = mod.forward(x, {}).clone().float().reshape(1, -1)
            oks = off(lins)
            out_old = mod.forward(x, {}).clone().float().reshape(1, -1)
            on(lins, oks)
            if len(lins) == 1:
                ref = gemv32(lins[0].inner, x)
                kind = "head" if name == "head" else "o"
            else:
                g = [l for l in lins if "gate" in l.key][0]; u = [l for l in lins if "up" in l.key][0]; d = [l for l in lins if "down" in l.key][0]
                ref = gemv32(d.inner, torch.nn.functional.silu(gemv32(g.inner, x)) * gemv32(u.inner, x))
                kind = "dense"
            rec(kind, name, rel_l2(out_new, ref), rel_l2(out_old, ref))
    for kind, rows in agg.items():
        n = [r[0] for r in rows]; o = [r[1] for r in rows]
        worse = [r for r in rows if r[0] > 1.25 * r[1]]
        print(f"gate32 {kind:<5} n={len(rows):<3} new mean {sum(n)/len(n):.2e} max {max(n):.2e} | old mean {sum(o)/len(o):.2e} "
              f"max {max(o):.2e} | new worse(>1.25x) in {len(worse)}", flush = True)
        for r in sorted(rows, key = lambda r: -r[0] / max(r[1], 1e-12))[:3]:
            print(f"   {r[2]:<8} new {r[0]:.2e} old {r[1]:.2e}", flush = True)


def layer_gate(model, generator, tokenizer, all_ids):
    """Per-layer check on real weights: capture the inputs of selected modules during one decode
    step, then run each module with the exl3_dec path on and off and compare (rel-L2)."""
    from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
    blocks = {m.layer_idx: m for m in model.modules if hasattr(m, "mlp") and hasattr(m, "layer_idx")}
    head = model.modules[model.logit_layer_idx]
    want = {}
    for li in (0, 1, 5, 12, 40, 47):
        b = blocks[li]
        want[f"L{li}.mlp"] = b.mlp
        if li in (1, 5):
            want[f"L{li}.attn"] = b.attn
            want[f"L{li}.o_proj"] = b.attn.o_proj
    want["head"] = head
    caps = {}
    origs = {}
    for name, mod in want.items():
        orig = mod.forward
        origs[name] = orig
        def wrapped(x, params, *a, _n = name, _o = orig, **k):
            if _n not in caps and x.numel() == x.shape[-1]:
                caps[_n] = (x.detach().clone(), {kk: vv for kk, vv in params.items()})
            return _o(x, params, *a, **k)
        mod.forward = wrapped
    ids = all_ids[:, 70000:70000 + 256]
    run_job(generator, ids, 3, GreedySampler())
    for name, mod in want.items():
        mod.forward = origs[name]

    def env(k, v):
        if v is None: os.environ.pop(k, None)
        else: os.environ[k] = v

    for name, mod in want.items():
        if name not in caps:
            print(f"layers {name}: not captured"); continue
        x, params = caps[name]
        if isinstance(mod, BlockSparseMLP):
            p = {}
            cfg = mod.routing_cfg
            y = x.view(-1, mod.hidden_size)
            env("EXL3_DEC_ROUTER", "1")
            s1, w1 = mod.routing_fn(1, cfg, y, p)
            s1 = s1.clone(); w1 = w1.clone()
            out_dec = mod.forward(x, p).clone()
            env("EXL3_DEC_ROUTER", "0")
            s0, w0 = mod.routing_fn(1, cfg, y, p)
            s0 = s0.clone(); w0 = w0.clone()
            keep = mod.support_dec_moe
            mod.support_dec_moe = False
            elins = [l for l in list(mod.gates) + list(mod.ups) + list(mod.downs)
                     if getattr(l, "quant_type", None) == "exl3"]
            eoks = [l.inner.dec_ok for l in elins]
            for l in elins: l.inner.dec_ok = False
            out_ref = mod.forward(x, p).clone()
            for l, o in zip(elins, eoks): l.inner.dec_ok = o
            mod.support_dec_moe = keep
            env("EXL3_DEC_ROUTER", None)
            y32 = y.float()
            ref32 = torch.zeros((1, mod.hidden_size), dtype = torch.float, device = y.device)
            amax = 0.0
            for e, w in zip(s0[0].tolist(), w0[0].float().tolist()):
                g = y32 @ w32(mod.gates[e]); u = y32 @ w32(mod.ups[e])
                a = torch.nn.functional.silu(g) * u
                amax = max(amax, a.abs().max().item(), g.abs().max().item(), u.abs().max().item())
                ref32 += w * (a @ w32(mod.downs[e]))
            print(f"layers {name:<10} MoE vs fp32: dec {rel_l2(out_dec, ref32):.2e}  old {rel_l2(out_ref, ref32):.2e}  "
                  f"max|g,u,act| {amax:.1f}", flush = True)
            sel_same = sorted(s1[0].tolist()) == sorted(s0[0].tolist())
            d1 = dict(zip(s1[0].tolist(), w1[0].float().tolist()))
            werr = max(abs(d1.get(e, 0.0) - w) for e, w in zip(s0[0].tolist(), w0[0].float().tolist()))
            print(f"layers {name:<10} MoE dec={keep}: rel-L2 {rel_l2(out_dec, out_ref):.2e}  router selection "
                  f"{'same' if sel_same else 'DIFF'} max w err {werr:.1e}  K={mod.multi_up.K if mod.multi_up else '-'}", flush = True)
        elif hasattr(mod, "q_proj"):
            p = {}
            q1, k1, v1, _ = mod.project_qkv(x, p)
            q1, k1, v1 = q1.clone(), k1.clone(), v1.clone()
            keep = mod._dec_qkv_linears() if hasattr(mod, "_dec_qkv_linears") else None
            lins = [l for l in (mod.q_proj, mod.k_proj, mod.v_proj, getattr(mod, "g_proj", None))
                    if l is not None and getattr(l, "quant_type", None) == "exl3"]
            oks = [l.inner.dec_ok for l in lins]
            mod._dec_qkv = None
            for l in lins: l.inner.dec_ok = False
            q0, k0, v0, _ = mod.project_qkv(x, p)
            mod._dec_qkv = keep
            for l, o in zip(lins, oks): l.inner.dec_ok = o
            print(f"layers {name:<10} qkv multi={keep is not None}: rel-L2 q {rel_l2(q1, q0):.2e} k {rel_l2(k1, k0):.2e} "
                  f"v {rel_l2(v1, v0):.2e}", flush = True)
        else:
            # Linear (o_proj, head) or dense GatedMLP (layer 0): toggle dec_ok on every EXL3 Linear inside
            lins = [m for m in ([mod] + list(getattr(mod, "gates", [])) + list(getattr(mod, "ups", [])) + list(getattr(mod, "downs", [])))
                    if m is not None and getattr(m, "quant_type", None) == "exl3"]
            if not lins and hasattr(mod, "modules"):
                lins = [m for m in mod.modules if getattr(m, "quant_type", None) == "exl3"]
            p = {}
            out_dec = mod.forward(x, p).clone()
            oks = [l.inner.dec_ok for l in lins]
            for l in lins: l.inner.dec_ok = False
            out_ref = mod.forward(x, p).clone()
            for l, o in zip(lins, oks): l.inner.dec_ok = o
            print(f"layers {name:<10} linears {len(lins)} dec={any(oks)}: rel-L2 {rel_l2(out_dec, out_ref):.2e}", flush = True)
            if len(lins) == 1 and mod is lins[0] and x.shape[-1] == lins[0].in_features and lins[0].out_features <= 65536:
                ref32 = x.view(1, -1).float() @ w32(lins[0])
                print(f"layers {name:<10} vs fp32: dec {rel_l2(out_dec, ref32):.2e}  old {rel_l2(out_ref, ref32):.2e}", flush = True)


def main():
    p = argparse.ArgumentParser(allow_abbrev = False)
    model_init.add_args(p, cache = True)
    p.add_argument("--modes", default = "greedy")
    p.add_argument("--tf-prefill", type = int, default = 1536)
    p.add_argument("--tf-steps", type = int, default = 512)
    p.add_argument("--tf-offset", type = int, default = 20000)
    p.add_argument("--tf-save", default = None)
    p.add_argument("--tf-ref", default = None)
    p.add_argument("--greedy-tokens", type = int, default = 128)
    p.add_argument("--prof-tokens", type = int, default = 32)
    p.add_argument("--ctx", type = str, default = "2048")
    p.add_argument("--decode-tokens", type = int, default = 64)
    p.add_argument("--reps", type = int, default = 3)
    p.add_argument("--ref32", action = "store_true")
    p.add_argument("--graph", type = int, default = 0, help = "capture the batch-1 decode step in a HIP graph")
    p.add_argument("--graph-warmups", type = int, default = 3)
    p.add_argument("--graph-bisect", default = None,
                   help = "comma-separated fwd_modules prefixes; capture+replay each, smallest first")
    p.add_argument("--ids-save", default = None)
    args = p.parse_args()
    if args.ref32:
        os.environ["EXL3_DEC_QKV"] = "0"
    torch.set_grad_enabled(False)

    import exllamav3_ext
    print("ext", exllamav3_ext.__file__, flush = True)
    t0 = time.time()
    model, config, cache, tokenizer, *_ = model_init.init(args)
    print(f"load {time.time() - t0:.1f}s", flush = True)
    if args.ref32:
        install_ref32(model)
    generator = Generator(model = model, cache = cache, tokenizer = tokenizer)
    grapher = None
    if args.graph or args.graph_bisect:
        from step_graph import StepGraph
        scopes = [int(s) for s in args.graph_bisect.split(",")] if args.graph_bisect else None
        grapher = StepGraph(model, warmups = args.graph_warmups, bisect = scopes)
        grapher.install()
        print(f"graph: ON (warmups {args.graph_warmups}, bisect {scopes})", flush = True)
    corpus = open(CORPUS, encoding = "utf-8").read()
    all_ids = tokenizer.encode(corpus, add_bos = False)
    vocab = tokenizer.actual_vocab_size

    for mode in args.modes.split(","):
        if mode == "greedy":
            for prompt in PROMPTS:
                ids = tokenizer.hf_chat_template(messages = [{"role": "user", "content": prompt}],
                                                 add_generation_prompt = True, enable_thinking = False)
                text, ttft, n, tot = run_job(generator, ids, args.greedy_tokens, GreedySampler())
                print(f"\n=== greedy {n} tok {tot:.1f}s ===\n{text}\n", flush = True)

        elif mode == "gids":
            # greedy, 128 tokens, ids saved so two arms can be diffed exactly
            allrec = []
            for prompt in PROMPTS:
                ids = tokenizer.hf_chat_template(messages = [{"role": "user", "content": prompt}],
                                                 add_generation_prompt = True, enable_thinking = False)
                s = RecordingSampler()
                text, ttft, n, tot = run_job(generator, ids, args.greedy_tokens, s)
                allrec.append(s.ids)
                print(f"gids {n} tok {tot:.1f}s first {s.ids[:8]}", flush = True)
            if args.ids_save:
                torch.save(allrec, args.ids_save)
                print(f"gids saved {args.ids_save}", flush = True)

        elif mode == "tf":
            o = args.tf_offset
            ids = all_ids[:, o:o + args.tf_prefill + args.tf_steps + 1]
            prompt = ids[:, :args.tf_prefill]
            ref = ids[0, args.tf_prefill:args.tf_prefill + args.tf_steps].tolist()
            s = ForcedSampler(ref, vocab)
            _, _, _, tot = run_job(generator, prompt, args.tf_steps, s)
            rec = torch.stack(s.rec[:args.tf_steps])
            print(f"tf: {rec.shape[0]} steps in {tot:.1f}s", flush = True)
            if args.tf_save:
                torch.save(rec, args.tf_save)
                print("tf saved", args.tf_save, flush = True)
            if args.tf_ref:
                refrec = torch.load(args.tf_ref)
                n = min(refrec.shape[0], rec.shape[0])
                P = refrec[:n].float()
                Q = rec[:n].float()
                kld = (P.exp() * (P - Q)).sum(-1)
                top1 = (P.argmax(-1) == Q.argmax(-1)).float().mean().item()
                nll_p = -P[torch.arange(n), torch.tensor(ref[:n])].mean().item()
                nll_q = -Q[torch.arange(n), torch.tensor(ref[:n])].mean().item()
                print(f"tf-compare n={n} KLD mean {kld.mean().item():.3e} max {kld.max().item():.3e} "
                      f"top1 {top1*100:.2f}%  nll ref {nll_p:.4f} new {nll_q:.4f}", flush = True)

        elif mode == "prof":
            # decode-only window: markers after the first generated token and after the last
            ids = all_ids[:, 60000:60000 + 2048]
            m = torch.empty(MARKER_NUMEL, device = "cuda")
            job = Job(input_ids = ids, max_new_tokens = args.prof_tokens + 2, stop_conditions = [], sampler = GreedySampler())
            generator.enqueue(job)
            n = 0
            t0 = None
            while generator.num_remaining_jobs():
                for res in generator.iterate():
                    if res.get("text") is not None or res.get("token_ids") is not None:
                        n += 1
                        if n == 1:
                            torch.cuda.synchronize()
                            m.fill_(1.0)
                            torch.cuda.synchronize()
                            t0 = time.time()
            torch.cuda.synchronize()
            m.fill_(2.0)
            torch.cuda.synchronize()
            print(f"prof: {n - 1} decode tokens in window, {(n - 1) / (time.time() - t0):.2f} tok/s", flush = True)

        elif mode == "tprof":
            # which Python call sites still emit small torch ops per decode step (dispatch-level log)
            import collections, traceback
            from torch.utils._python_dispatch import TorchDispatchMode
            counts = collections.Counter()
            class Log(TorchDispatchMode):
                def __torch_dispatch__(self, func, types, args = (), kwargs = None):
                    name = str(func.overloadpacket.__name__)
                    if name not in ("view", "_unsafe_view", "as_strided", "t", "detach", "alias",
                                    "_reshape_alias", "unsqueeze", "squeeze", "select", "slice",
                                    "expand", "permute", "transpose", "empty", "empty_strided",
                                    "empty_like", "reshape", "is_same_size"):
                        fr = [f for f in traceback.extract_stack()[:-1] if "/exllamav3/" in f.filename]
                        site = " <- ".join(f"{f.filename.split('/')[-1]}:{f.lineno}" for f in fr[-3:][::-1])
                        counts[(name, site)] += 1
                    return func(*args, **(kwargs or {}))
            ids = all_ids[:, 60000:60000 + 2048]
            job = Job(input_ids = ids, max_new_tokens = 8, stop_conditions = [], sampler = GreedySampler())
            generator.enqueue(job)
            n = 0
            mode_ = None
            while generator.num_remaining_jobs():
                for res in generator.iterate():
                    if res.get("text") is not None or res.get("token_ids") is not None:
                        n += 1
                        if n == 3 and mode_ is None:
                            mode_ = Log(); mode_.__enter__(); n0 = n
                if mode_ is not None and n >= 6:
                    break
            mode_.__exit__(None, None, None)
            while generator.num_remaining_jobs():
                for res in generator.iterate(): pass
            steps = max(n - n0, 1)
            print(f"tprof: {steps} decode steps, {sum(counts.values()) / steps:.0f} ops/tok", flush = True)
            for (name, site), c in counts.most_common(45):
                print(f"  {c / steps:7.1f}/tok  {name:<18} {site}", flush = True)

        elif mode == "normcheck":
            # every rms_norm / rms_norm_res_in call of a short prefill + decode: native vs torch fallback
            import collections
            from exllamav3.ext import exllamav3_ext as E
            from exllamav3 import ext_fallbacks as FB
            nat_n, nat_r = E.rms_norm, E.rms_norm_res_in
            stats = collections.defaultdict(lambda: [0, 0, 0.0, 0])   # calls, mismatching elems, max abs diff, elems
            def key(x, w, y, r, mode, extra):
                return (mode, tuple(x.shape[-1:]), x.shape[0] if x.dim() > 1 else 1, str(x.dtype)[6:], str(y.dtype)[6:],
                        None if w is None else str(w.dtype)[6:], None if r is None else str(r.dtype)[6:], extra)
            def chk(k, a, b):
                st = stats[k]
                d = (a.float() - b.float()).abs()
                st[0] += 1; st[1] += int((d > 0).sum()); st[2] = max(st[2], d.max().item()); st[3] += d.numel()
            def rn(x, w, y, eps, cb, cs, span, addr, wg = 1):
                y2 = y.clone()
                FB.rms_norm(x, w, y2, eps, cb, cs, span, addr, wg)
                nat_n(x, w, y, eps, cb, cs, span, addr, wg)
                chk(key(x, w, y, None, "add" if addr else "plain", (span, wg, cb, cs)), y, y2)
            def rr(x, w, y, r, eps, cb, cs):
                y2 = y.clone(); r2 = r.clone()
                FB.rms_norm_res_in(x, w, y2, r2, eps, cb, cs)
                nat_r(x, w, y, r, eps, cb, cs)
                chk(key(x, w, y, r, "res_in.y", (cb, cs)), y, y2)
                chk(key(x, w, y, r, "res_in.r", (cb, cs)), r, r2)
            E.rms_norm, E.rms_norm_res_in = rn, rr
            ids = all_ids[:, 80000:80000 + 300]
            run_job(generator, ids, 4, GreedySampler())
            E.rms_norm, E.rms_norm_res_in = nat_n, nat_r
            for k, st in sorted(stats.items(), key = lambda kv: -kv[1][1]):
                print(f"normcheck {k}: calls {st[0]} mismatch {st[1]}/{st[3]} ({100 * st[1] / max(st[3], 1):.3f} %) max|d| {st[2]:.3e}", flush = True)

        elif mode == "gate32":
            gate32(model, generator, tokenizer, all_ids)

        elif mode == "layers":
            layer_gate(model, generator, tokenizer, all_ids)

        elif mode == "speed":
            for ctx in (int(c) for c in str(args.ctx).split(",")):
                r = []
                for i in range(args.reps):
                    o = 100000 + i * (ctx + 500)
                    ids = all_ids[:, o:o + ctx]
                    _, ttft, n, tot = run_job(generator, ids, args.decode_tokens, GreedySampler())
                    r.append((n - 1) / (tot - ttft))
                    print(f"speed ctx {ctx} rep{i}: {r[-1]:.2f} tok/s (ttft {ttft:.2f}s)", flush = True)
                print(f"speed ctx {ctx} median {statistics.median(r):.2f} tok/s", flush = True)

    if grapher is not None:
        print(grapher.summary(), flush = True)


if __name__ == "__main__":
    main()
