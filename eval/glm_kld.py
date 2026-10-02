"""
Streamed KLD instrument for GLM-5.3-Flash (and any model whose unquantized source does not fit in memory).

  rows   build the fixed eval set once (token ids, variable row lengths) -> OUT/rows.safetensors
  ref    stream the reference (FP8 source) module by module over every row; save final logits (fp16) per row,
         the fp32 residual (hc streams) at every block boundary for the attribution rows, and optionally
         (--pack) the local per-group output error of the pack's block on the reference input
  eval   load a pack whole (resident), forward every row, KLD(ref || pack) per row + top-1 agreement + PPL
  curve  load a pack whole, restart the forward at every block boundary from the stored reference state
         (layers < L reference, layers >= L quantized) -> KLD(L). KLD(L) - KLD(L+1) = marginal cost of layer L

Everything is written under OUT (default ~/exl3-glm-kld/ref). Rows are processed one at a time; the
final three modules (hc_head, norm, lm_head) run in 2048-token chunks so a long row never materializes full fp32
logits.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, json, math, time, glob, resource
import torch
from safetensors.torch import save_file, load_file
from exllamav3 import Config, Model, Tokenizer
from exllamav3.util.memory import free_mem
from exllamav3.util.measures import compute_kl_div

HEAD_CHUNK = 2048
OUT_DEFAULT = os.path.expanduser("~/exl3-glm-kld/ref")


def check_disk(min_gb = 60):
    """Abort long jobs before they fill the shared disk (lead's guard, 24/09)."""
    import shutil
    free = shutil.disk_usage(os.path.expanduser("~")).free / 1e9
    if free < min_gb:
        raise SystemExit(f"ABORT: only {free:.0f} GB free on the home disk (< {min_gb} GB)")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush = True)


def peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576


def gtt_gb():
    v = 0
    for p in glob.glob("/sys/class/drm/card*/device/mem_info_gtt_used"):
        try:
            v = max(v, int(open(p).read()))
        except Exception:
            pass
    return v / 1e9


# ---------------------------------------------------------------------------------------------------------------
# Eval rows

def build_rows(args):
    """Fixed eval set. Domains: wiki (wikitext-2 test, contiguous from offset 0), agent (coding and curl agent
    conversations rendered through the chat template, windows from the conversation tail), prose (eval_texts
    novels, mid-book windows), code (this repo's Python sources, never in the bundled calibration corpus),
    long (two 16K agent rows: DSA indexer selects top-2048 of 16K keys, KDA state at depth)."""
    from transformers import AutoTokenizer
    here = os.path.dirname(os.path.abspath(__file__))
    config = Config.from_directory(args.tok_dir)
    tok = Tokenizer.from_config(config)
    hft = AutoTokenizer.from_pretrained(args.tok_dir)
    L = 2048
    rows, names = [], []

    def enc(text):
        return tok.encode(text, add_bos = False).flatten()

    # wiki
    from model_diff import get_test_tokens
    w = get_test_tokens(tok, args.wiki_rows, L, L)
    for i in range(w.shape[0]):
        rows.append(w[i]); names.append(f"wiki{i}")

    # agent: render each conversation, take windows from the tail (the head is a shared system prompt)
    agent_files = sorted(glob.glob(os.path.join(here, "prompts", "agentic_*.json")))
    rendered = {}
    for f in agent_files:
        d = json.load(open(f))
        for msg in d["messages"]:
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", tc)
                if isinstance(fn.get("arguments"), str):
                    try:
                        fn["arguments"] = json.loads(fn["arguments"])
                    except Exception:
                        fn["arguments"] = {"raw": fn["arguments"]}
        try:
            text = hft.apply_chat_template(d["messages"], tools = d.get("tools"), tokenize = False)
        except Exception as e:
            log(f"chat template failed on {f}: {e}; using raw json")
            text = json.dumps(d["messages"], ensure_ascii = False)
        rendered[os.path.basename(f)] = enc(text)
    long_src = ["agentic_curl_16.json", "agentic_code_29.json"]
    n_agent = 0
    for name, ids in rendered.items():
        if name in long_src or n_agent >= args.agent_rows:
            continue
        if ids.shape[0] < L:
            continue
        rows.append(ids[-L:]); names.append(f"agent:{name}"); n_agent += 1

    # prose
    for fn in ["pride_prejudice_mod.txt", "variable_man_mod.txt", "illustrious_client.txt"]:
        ids = enc(open(os.path.join(here, "eval_texts", fn)).read())
        for k in range(args.prose_per_file):
            a = (ids.shape[0] // (args.prose_per_file + 1)) * (k + 1) - L // 2
            rows.append(ids[a:a + L]); names.append(f"prose:{fn}:{a}")

    # code
    root = os.path.dirname(here)
    srcs = ["exllamav3/modules/mla_attn.py", "exllamav3/conversion/convert_model.py", "exllamav3/modules/linear.py",
            "exllamav3/modules/block_sparse_mlp.py", "exllamav3/generator/generator.py", "exllamav3/cache/cache.py",
            "exllamav3/modules/gated_delta_net.py", "exllamav3/model/model.py"]
    n_code = 0
    for s in srcs:
        p = os.path.join(root, s)
        if not os.path.isfile(p) or n_code >= args.code_rows:
            continue
        ids = enc(open(p).read())
        if ids.shape[0] < L:
            continue
        rows.append(ids[:L]); names.append(f"code:{s}"); n_code += 1

    # long
    for name in long_src:
        ids = rendered.get(name)
        if ids is None or ids.shape[0] < args.long_len:
            log(f"long source {name} too short ({None if ids is None else ids.shape[0]})")
            continue
        rows.append(ids[-args.long_len:]); names.append(f"long:{name}")

    os.makedirs(args.out, exist_ok = True)
    t = {f"row{i:03d}": r.to(torch.long).contiguous() for i, r in enumerate(rows)}
    save_file(t, os.path.join(args.out, "rows.safetensors"), metadata = {"names": json.dumps(names)})
    ntok = sum(r.shape[0] for r in rows)
    log(f"{len(rows)} rows, {ntok} tokens -> {args.out}/rows.safetensors")
    for i, n in enumerate(names):
        print(f"  row{i:03d} {rows[i].shape[0]:6d} {n}")


def load_rows(out, rows_file = None):
    from safetensors import safe_open
    p = rows_file or os.path.join(out, "rows.safetensors")
    with safe_open(p, "pt") as f:
        names = json.loads(f.metadata()["names"])
        rows = [f.get_tensor(f"row{i:03d}") for i in range(len(names))]
    return rows, names


def attrib_rows(args, n):
    idx = [int(v) for v in args.attrib_idx.split(",")] if args.attrib_idx else list(range(args.attrib_rows))
    return [i for i in idx if i < n]


def domain(name):
    return name.split(":")[0].rstrip("0123456789")


# ---------------------------------------------------------------------------------------------------------------
# Forward helpers

def fwd(module, state, params):
    state = module.prepare_for_device(state, params)
    return module.forward(state, params).clone()


def head_logits(head_mods, state, params_proto):
    """Run the final modules over a (1, seq, ...) state in HEAD_CHUNK slices; yields (a, b, logits fp16)."""
    seq = state.shape[1]
    for a in range(0, seq, HEAD_CHUNK):
        b = min(seq, a + HEAD_CHUNK)
        x = state[:, a:b].contiguous()
        p = dict(params_proto)
        for m in head_mods:
            x = fwd(m, x, p)
        yield a, b, x.view(b - a, -1)


def kld_rows(logits_q, logits_r, vocab):
    """Per-token KL(ref || q) on the device, fp32."""
    return compute_kl_div(logits_q, logits_r, vocab).float().flatten()


def logprob_of(logits, targets, vocab):
    l = logits[:, :vocab].float()
    return l.gather(-1, targets.view(-1, 1)).flatten() - l.logsumexp(-1)


# ---------------------------------------------------------------------------------------------------------------
# Reference pass (streamed)

@torch.inference_mode()
def ref_pass(args):
    device = torch.device(args.device)
    rows, names = load_rows(args.out, args.rows_file)
    if args.max_rows:
        rows, names = rows[:args.max_rows], names[:args.max_rows]
    attrib = set(attrib_rows(args, len(rows)))
    # --bnd_layers: save the input state of these module indices for EVERY row (calibration states for
    # single-layer quantization experiments); --stop_after: stop streaming after that module (no logits)
    bnd_all = {int(v) for v in args.bnd_layers.split(",")} if args.bnd_layers else set()
    config = Config.from_directory(args.model)
    config.override_dynamic_seq_len(max(r.shape[0] for r in rows))
    if getattr(args, "override", None):
        apply_overrides(config, args.override)
    model = Model.from_config(config)
    vocab = Tokenizer.from_config(config).actual_vocab_size

    pack_model = None
    if args.pack:
        cfg_p = Config.from_directory(args.pack)
        cfg_p.override_dynamic_seq_len(max(r.shape[0] for r in rows))
        pack_model = Model.from_config(cfg_p)

    mods = model.modules
    n = len(mods)
    head_start = model.logit_layer_idx - 2
    first_block = model.first_block_idx
    params = [{} for _ in rows]
    states = [model.prepare_inputs(r.view(1, -1), params[i]) for i, r in enumerate(rows)]
    bnd_dir = os.path.join(args.out, "bnd")
    log_dir = os.path.join(args.out, "logits")
    os.makedirs(bnd_dir, exist_ok = True)
    os.makedirs(log_dir, exist_ok = True)
    local = {}
    t0 = time.time()

    for idx in range(head_start):
        m = mods[idx]
        check_disk()
        if idx in bnd_all:
            for i in range(len(rows)):
                p = os.path.join(bnd_dir, f"m{idx:03d}_row{i:03d}.safetensors")
                if not os.path.exists(p):
                    save_file({"x": states[i].to(torch.bfloat16).cpu().contiguous()}, p)
        if args.stop_after is not None and idx > args.stop_after:
            log(f"stop after m{args.stop_after:03d}")
            return
        # Boundary state = input of module idx (blocks only), attribution rows, fp32 as the stream carries it
        if idx >= first_block and args.save_bnd:
            for i in attrib:
                p = os.path.join(bnd_dir, f"m{idx:03d}_row{i:03d}.safetensors")
                if not os.path.exists(p):
                    save_file({"x": states[i].to(torch.bfloat16).cpu().contiguous()}, p)
        tl = time.time()
        config.stc.begin_deferred_load()
        m.load(device if not m.caps.get("prefer_cpu") else "cpu")
        config.stc.end_deferred_load()
        tload = time.time() - tl

        mq = None
        if pack_model is not None and idx >= first_block:
            mq = pack_model.modules[idx]
            pack_model.config.stc.begin_deferred_load()
            mq.load(device)
            pack_model.config.stc.end_deferred_load()

        for i in range(len(rows)):
            # Modules mutate the residual in place: keep a pristine copy of the input for the local swaps
            x_in = states[i].to(device)
            x_keep = x_in.clone() if (mq is not None and i in attrib) else None
            y = fwd(m, x_in, params[i])
            x_in = x_keep
            if mq is not None and i in attrib:
                local.setdefault(idx, {})
                for g, y_q in local_group_outputs(m, mq, x_in, params[i]).items():
                    d = (y_q.float() - y.float())
                    e = local[idx].setdefault(g, [0.0, 0.0, 0])
                    e[0] += d.pow(2).sum().item()
                    e[1] += (y.float() - x_in.float()).pow(2).sum().item()
                    e[2] += d[0].numel()
            states[i] = y
        m.unload(); config.stc.close(); free_mem()
        if mq is not None:
            mq.unload(); pack_model.config.stc.close(); free_mem()
        log(f"m{idx:03d} {m.key:40s} load {tload:6.1f}s  total {time.time() - t0:7.0f}s  "
            f"rss {peak_rss_gb():5.1f}G gtt {gtt_gb():5.1f}G")
        if idx in local:
            log("     local " + "  ".join(f"{g}: rel {v[0] / max(v[1], 1e-30):.3e}" for g, v in local[idx].items()))

    if args.save_bnd:
        for i in attrib:
            p = os.path.join(bnd_dir, f"m{head_start:03d}_row{i:03d}.safetensors")
            if not os.path.exists(p):
                save_file({"x": states[i].to(torch.bfloat16).cpu().contiguous()}, p)

    # Head: logits per row, fp16
    head = mods[head_start:]
    for m in head:
        config.stc.begin_deferred_load(); m.load(device); config.stc.end_deferred_load()
    lp_sum, lp_cnt = 0.0, 0
    for i, r in enumerate(rows):
        chunks = []
        tgt = r.to(device)
        for a, b, lg in head_logits(head, states[i].to(device), params[i]):
            lg = lg[:, :vocab].half()
            chunks.append(lg.cpu())
            bb = min(b, r.shape[0] - 1)
            if bb > a:
                lp = logprob_of(lg[:bb - a], tgt[a + 1:bb + 1], vocab)
                lp_sum += lp.sum().item(); lp_cnt += lp.numel()
        save_file({"logits": torch.cat(chunks, 0)}, os.path.join(log_dir, f"row{i:03d}.safetensors"))
    for m in head:
        m.unload()
    res = {"model": args.model, "rows": len(rows), "ppl": math.exp(-lp_sum / lp_cnt),
           "local": {str(k): v for k, v in local.items()}, "seconds": time.time() - t0,
           "peak_rss_gb": peak_rss_gb()}
    json.dump(res, open(os.path.join(args.out, "ref.json" if not args.pack else "ref_local.json"), "w"), indent = 1)
    log(f"ref done: ppl {res['ppl']:.4f}  {res['seconds']:.0f}s  peak rss {peak_rss_gb():.1f}G")


def local_group_outputs(m_ref, m_q, x, params):
    """Block output with one tensor group taken from the pack, the rest reference. Groups: attn, routed (routed
    experts + router, reference shared expert), shared, mlp (dense layers), all (whole pack block, by swapping)
    and pack (the pack's own block forward: must equal 'all', control)."""
    ref_mlp = m_ref.mlp
    ref_shared = getattr(ref_mlp, "shared_experts", None)
    q_shared = getattr(m_q.mlp, "shared_experts", None)
    swaps = {"attn": [(m_ref, "attn", m_q.attn)],
             "all": [(m_ref, "attn", m_q.attn), (m_ref, "mlp", m_q.mlp)]}
    if ref_shared is not None:
        swaps["routed"] = [(m_ref, "mlp", m_q.mlp), (m_q.mlp, "shared_experts", ref_shared)]
        swaps["shared"] = [(ref_mlp, "shared_experts", q_shared)]
    else:
        swaps["mlp"] = [(m_ref, "mlp", m_q.mlp)]
    out = {}
    for g, sw in swaps.items():
        saved = []
        try:
            for obj, attr, val in sw:
                saved.append((obj, attr, getattr(obj, attr)))
                setattr(obj, attr, val)
            out[g] = fwd(m_ref, x.clone(), dict(params))
        finally:
            for obj, attr, val in reversed(saved):
                setattr(obj, attr, val)
    out["pack"] = fwd(m_q, x.clone(), dict(params))
    return out


# ---------------------------------------------------------------------------------------------------------------
# Resident evaluation

def apply_overrides(config, spec):
    """spec: list of 'glob=dir' (tensor-key glob, e.g. '*.layers.40.mlp.experts.*') or a YAML file in model_diff's
    -or format. Later entries win."""
    if not spec:
        return
    from exllamav3.loader import SafetensorsCollection, VariantSafetensorsCollection
    pairs = []
    for s_ in spec:
        if s_.endswith(".yaml") or s_.endswith(".yml"):
            import yaml
            comp = yaml.safe_load(open(s_))
            src = {x["id"]: x["model_dir"] for x in comp["sources"]}
            pairs += [(o["key"], src[o["source"]]) for o in comp["overrides"]]
        else:
            k, d = s_.split("=", 1)
            pairs.append((k, d))
    vstc = VariantSafetensorsCollection(config.stc)
    by_dir = {}
    for k, d in pairs:
        by_dir.setdefault(d, []).append(k)
    for d, ks in by_dir.items():
        log(f"override from {d}: {ks[:4]}{' ...' if len(ks) > 4 else ''} ({len(ks)} globs)")
        vstc.add_stc(ks, SafetensorsCollection(d))
    config.stc = vstc


def load_resident(pack, max_len, device, overrides = None):
    config = Config.from_directory(pack)
    config.override_dynamic_seq_len(max_len)
    apply_overrides(config, overrides)
    model = Model.from_config(config)
    model.load(device = device)
    return config, model


@torch.inference_mode()
def eval_pack(args):
    device = torch.device(args.device)
    rows, names = load_rows(args.out)
    if args.max_rows:
        rows, names = rows[:args.max_rows], names[:args.max_rows]
    # Long rows never go through the resident model: an ~82 GB pack plus a 16K cacheless forward thrashed GTT
    # and hung the box twice (24/09). They are evaluated streamed (ref -m <pack> on the long-row subset)
    keep = [i for i, r in enumerate(rows) if r.shape[0] <= args.max_len]
    row_ids = keep
    rows, names = [rows[i] for i in keep], [names[i] for i in keep]
    config, model = load_resident(args.pack, max(r.shape[0] for r in rows), device, args.override)
    vocab = Tokenizer.from_config(config).actual_vocab_size
    log(f"loaded {args.pack}: gtt {gtt_gb():.1f}G")
    mods = model.modules
    head_start = model.logit_layer_idx - 2
    per_row = []
    t0 = time.time()
    for i, r in enumerate(rows):
        params = {}
        x = model.prepare_inputs(r.view(1, -1), params)
        for m in mods[:head_start]:
            x = fwd(m, x, params)
        ref = load_file(os.path.join(args.out, "logits", f"row{row_ids[i]:03d}.safetensors"))["logits"]
        tgt = r.to(device)
        kl, agree, lpq, lpr, cnt = [], 0, 0.0, 0.0, 0
        for a, b, lg in head_logits(mods[head_start:], x, params):
            lr = ref[a:b].to(device)
            lq = lg[:, :vocab]
            kl.append(kld_rows(lq, lr, vocab).cpu())
            agree += (lq.argmax(-1) == lr.argmax(-1)).sum().item()
            bb = min(b, r.shape[0] - 1)
            if bb > a:
                lpq += logprob_of(lq[:bb - a], tgt[a + 1:bb + 1], vocab).sum().item()
                lpr += logprob_of(lr[:bb - a], tgt[a + 1:bb + 1], vocab).sum().item()
                cnt += bb - a
        kl = torch.cat(kl)
        # KLD by position bucket inside the row: where in the row the error grows (depth / DSA tail)
        nb = args.buckets
        edges = [(kl.numel() * i) // nb for i in range(nb + 1)]
        buckets = [kl[edges[i]:edges[i + 1]].mean().item() for i in range(nb) if edges[i + 1] > edges[i]]
        per_row.append({"row": row_ids[i], "name": names[i], "n": int(kl.numel()), "kld": kl.mean().item(),
                        "kld_p90": kl.quantile(0.9).item() if kl.numel() < 16_000_000 else None,
                        "kld_by_bucket": buckets,
                        "top1": agree / kl.numel(), "lp_q": lpq, "lp_r": lpr, "cnt": cnt})
        if args.verbose:
            log(f"row{i:03d} {names[i][:40]:40s} kld {per_row[-1]['kld']:.5f} top1 {per_row[-1]['top1']:.4f}")
    res = summarize(per_row)
    res.update({"pack": args.pack, "seconds": time.time() - t0, "per_row": per_row, "gtt_gb": gtt_gb()})
    tag = args.tag or os.path.basename(os.path.normpath(args.pack))
    os.makedirs(os.path.join(args.out, "evals"), exist_ok = True)
    json.dump(res, open(os.path.join(args.out, "evals", f"{tag}.json"), "w"), indent = 1)
    log(f"{tag}: KLD {res['kld']:.5f} (tok-weighted {res['kld_tok']:.5f})  top1 {res['top1']:.4f}  "
        f"ppl q {res['ppl_q']:.4f} r {res['ppl_r']:.4f}  buckets " +
        " ".join(f"{b:.4f}" for b in res["kld_by_bucket"]) + "  by domain " +
        " ".join(f"{d}={v:.4f}" for d, v in res["by_domain"].items()))


def summarize(per_row):
    n = sum(p["n"] for p in per_row)
    by = {}
    for p in per_row:
        by.setdefault(domain(p["name"]), []).append(p["kld"])
    nb = max(len(p.get("kld_by_bucket") or []) for p in per_row)
    buckets = [sum(p["kld_by_bucket"][i] * p["n"] for p in per_row if len(p.get("kld_by_bucket") or []) > i) /
               max(sum(p["n"] for p in per_row if len(p.get("kld_by_bucket") or []) > i), 1) for i in range(nb)]
    return {
        "kld": sum(p["kld"] for p in per_row) / len(per_row),
        "kld_tok": sum(p["kld"] * p["n"] for p in per_row) / n,
        "kld_by_bucket": buckets,
        "top1": sum(p["top1"] * p["n"] for p in per_row) / n,
        "ppl_q": math.exp(-sum(p["lp_q"] for p in per_row) / sum(p["cnt"] for p in per_row)),
        "ppl_r": math.exp(-sum(p["lp_r"] for p in per_row) / sum(p["cnt"] for p in per_row)),
        "by_domain": {d: sum(v) / len(v) for d, v in by.items()},
    }


@torch.inference_mode()
def curve(args):
    """Restart at stored reference boundary states: layers before L reference, from L on the pack."""
    device = torch.device(args.device)
    rows, names = load_rows(args.out)
    attrib = attrib_rows(args, len(rows))
    config, model = load_resident(args.pack, max(rows[i].shape[0] for i in attrib), device, args.override)
    vocab = Tokenizer.from_config(config).actual_vocab_size
    mods = model.modules
    head_start = model.logit_layer_idx - 2
    starts = list(range(model.first_block_idx, head_start + 1, args.stride))
    if starts[-1] != head_start:
        starts.append(head_start)
    refs = {i: load_file(os.path.join(args.out, "logits", f"row{i:03d}.safetensors"))["logits"] for i in attrib}
    res = {"pack": args.pack, "rows": attrib, "curve": {}}
    t0 = time.time()
    for L in starts:
        tot, cnt = 0.0, 0
        per = []
        for i in attrib:
            params = {}
            model.prepare_inputs(rows[i].view(1, -1), params)
            x = load_file(os.path.join(args.out, "bnd", f"m{L:03d}_row{i:03d}.safetensors"))["x"].to(device).float()
            for m in mods[L:head_start]:
                x = fwd(m, x, params)
            s = 0.0
            for a, b, lg in head_logits(mods[head_start:], x, params):
                s += kld_rows(lg[:, :vocab], refs[i][a:b].to(device), vocab).sum().item()
            per.append(s / rows[i].shape[0])
        if per:
            res["curve"][L] = {"kld": sum(per) / len(per), "per_row": per}
            log(f"L={L:3d} ({mods[L].key}) kld {res['curve'][L]['kld']:.5f}  {time.time() - t0:.0f}s")
    tag = args.tag or os.path.basename(os.path.normpath(args.pack))
    os.makedirs(os.path.join(args.out, "curves"), exist_ok = True)
    json.dump(res, open(os.path.join(args.out, "curves", f"{tag}.json"), "w"), indent = 1)


@torch.inference_mode()
def long_eval(args):
    """Rows longer than --max_len, streamed through the pack one module at a time (low GTT), KLD vs reference."""
    import copy, shutil
    rows, names = load_rows(args.out)
    ids = [i for i, r in enumerate(rows) if r.shape[0] > args.max_len]
    tag = args.tag or os.path.basename(os.path.normpath(args.pack))
    tmp = os.path.join(args.out, "tmp_long_" + tag)
    os.makedirs(tmp, exist_ok = True)
    save_file({f"row{j:03d}": rows[i] for j, i in enumerate(ids)}, os.path.join(tmp, "rows.safetensors"),
              metadata = {"names": json.dumps([names[i] for i in ids])})
    a2 = copy.copy(args)
    a2.model, a2.out, a2.pack, a2.save_bnd, a2.rows_file, a2.max_rows = args.pack, tmp, None, 0, None, 0
    a2.bnd_layers, a2.stop_after = None, None
    ref_pass(a2)
    device = torch.device(args.device)
    vocab = Tokenizer.from_config(Config.from_directory(args.pack)).actual_vocab_size
    per_row = []
    for j, i in enumerate(ids):
        q = load_file(os.path.join(tmp, "logits", f"row{j:03d}.safetensors"))["logits"]
        r = load_file(os.path.join(args.out, "logits", f"row{i:03d}.safetensors"))["logits"]
        kl, agree = [], 0
        for a in range(0, q.shape[0], HEAD_CHUNK):
            lq, lr = q[a:a + HEAD_CHUNK].to(device), r[a:a + HEAD_CHUNK].to(device)
            kl.append(kld_rows(lq, lr, vocab).cpu()); agree += (lq.argmax(-1) == lr.argmax(-1)).sum().item()
        kl = torch.cat(kl)
        # KLD of the last 2048 positions: where DSA selects top-2048 out of > 2048 keys
        per_row.append({"row": i, "name": names[i], "n": int(kl.numel()), "kld": kl.mean().item(),
                        "kld_tail2k": kl[-2048:].mean().item(), "top1": agree / kl.numel()})
        log(f"row{i:03d} {names[i]} kld {per_row[-1]['kld']:.5f} tail2k {per_row[-1]['kld_tail2k']:.5f} top1 {per_row[-1]['top1']:.4f}")
    os.makedirs(os.path.join(args.out, "evals"), exist_ok = True)
    json.dump({"pack": args.pack, "per_row": per_row}, open(os.path.join(args.out, "evals", f"{tag}_long.json"), "w"), indent = 1)
    shutil.rmtree(tmp)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices = ["rows", "ref", "eval", "curve", "leval"])
    ap.add_argument("-o", "--out", default = OUT_DEFAULT)
    ap.add_argument("-m", "--model", default = os.path.expanduser("~/models/glm53-fp8"))
    ap.add_argument("-p", "--pack", default = None)
    ap.add_argument("--tok_dir", default = os.path.expanduser("~/models/glm53-exl3-td205"))
    ap.add_argument("--wiki_rows", type = int, default = 12)
    ap.add_argument("--agent_rows", type = int, default = 7)
    ap.add_argument("--prose_per_file", type = int, default = 2)
    ap.add_argument("--code_rows", type = int, default = 7)
    ap.add_argument("--long_len", type = int, default = 16384)
    ap.add_argument("--attrib_rows", type = int, default = 8)
    ap.add_argument("--attrib_idx", default = "0,1,12,13,19,21,25,26", help = "attribution rows (boundary states saved)")
    ap.add_argument("--max_rows", type = int, default = 0)
    ap.add_argument("--save_bnd", type = int, default = 1)
    ap.add_argument("--stride", type = int, default = 1)
    ap.add_argument("--rows_file", default = None)
    ap.add_argument("--max_len", type = int, default = 4096, help = "eval: longest row run resident")
    ap.add_argument("--buckets", type = int, default = 4, help = "eval: KLD buckets per row (equal position spans)")
    ap.add_argument("--bnd_layers", default = None, help = "module indices whose input state is saved for all rows")
    ap.add_argument("--stop_after", type = int, default = None)
    ap.add_argument("--tag", default = None)
    ap.add_argument("-or", "--override", action = "append", default = None, help = "glob=dir or YAML, repeatable")
    ap.add_argument("-d", "--device", default = "cuda:0")
    ap.add_argument("-v", "--verbose", action = "store_true")
    a = ap.parse_args()
    {"rows": build_rows, "ref": ref_pass, "eval": eval_pack, "curve": curve, "leval": long_eval}[a.cmd](a)
