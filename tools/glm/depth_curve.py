# GLM depth curve in ONE pass.
#
#   curve: load once, build one unique ~512K-token document (wikitext-2 test articles + this repo's
#          source files, shuffled at unit level so every depth bucket mixes prose and code), prefill it
#          in chunks through model.forward (timing each chunk -> prefill t/s vs depth; logits -> NLL per
#          depth bucket), and at each checkpoint decode N tokens greedily (timed), then restore the
#          recurrent (KDA) state from a stash and rewind the attention cache by position before going on.
#          The JSON is rewritten after every checkpoint, so a crash keeps the finished points.
#   prof:  fake-depth decode windows for rocprofv3 (test recurrent state at position d, attention cache
#          length d, zero KV): N decode steps at each --prof-depths, windows stamped in monotonic ns.
#          Decode cost is content-independent except the DSA top-k, so the curve's real decode t/s at the
#          same depth cross-checks the fake one.
#   pfprof: fake-depth PREFILL chunk (pfDepth1): test recurrent state at position d, attention/indexer
#          cache filled with random fp16 (so top-k sees real score spread), one --chunk-row forward at
#          each --pf-depths, reps interleaved across depths. Each rep = one plain (uninstrumented) chunk
#          + one instrumented chunk with torch.cuda.Event around every attn/mlp sublayer, the MLA
#          indexer / sparse-attention methods, dsa_indexer_scores and ext.dsa_topk.
import argparse, glob, hashlib, json, math, os, random, re, sys, time
import torch

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["curve", "prof", "ctrl", "pfprof"])
ap.add_argument("-m", "--model", required=True)
ap.add_argument("-o", "--out", required=True)
ap.add_argument("--chunk", type=int, default=int(os.environ.get("EXL3_PREFILL_CHUNK", "2048")))
ap.add_argument("--total", type=int, default=512000)
ap.add_argument("--checkpoints", default="4096,32768,131072,262144,512000")
ap.add_argument("--decode", type=int, default=32)
ap.add_argument("--corpus", default=os.path.expanduser("~/bench/ppl/wiki.test.raw"))
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--prof-depths", default="4096,512000")
ap.add_argument("--prof-steps", type=int, default=8)
ap.add_argument("--ctrl-ends", default="16384,65536,131072,196608,262144,327680,393216,458752,512000")
ap.add_argument("--pf-depths", default="4096,512000")
ap.add_argument("--pf-reps", type=int, default=3)
ap.add_argument("--pf-arms", default="", help="pfDepth2: arm order per rep, e.g. O,A,B,B,A,O "
                "(O: old chunked dsa_topk fallback, tiled; A: fallback fast path, tiled; B: A + EXL3_DSA_PF_NOTILE)")
ap.add_argument("--pf-check", default="", help="pfDepth2: depths for the bitwise indices + 32-token greedy check")
ap.add_argument("--min-avail-gb", type=float, default=10.0)
args = ap.parse_args()
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PAGE = 256
RES = {"mode": args.mode, "model": args.model, "args": vars(args),
       "env": {k: v for k, v in os.environ.items() if k.startswith("EXL3_")}}


def save():
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(RES, f, indent=1)
    os.replace(tmp, args.out)


def avail_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1048576
    return -1.0


def gtt_gb():
    best = 0
    for p in glob.glob("/sys/class/drm/card*/device/mem_info_gtt_used"):
        try:
            best = max(best, int(open(p).read()))
        except Exception:
            pass
    return best / 1e9


def mem_guard(tag):
    a = avail_gb()
    if a < args.min_avail_gb:
        RES["abort"] = f"MemAvailable {a:.1f} GiB < {args.min_avail_gb} at {tag}"
        save()
        print(RES["abort"], flush=True)
        sys.exit(3)
    return a


def build_doc(tok, need):
    text = open(args.corpus, encoding="utf-8").read()
    units, cur = [], []
    for line in text.split("\n"):
        if re.match(r"^ = [^=].* = $", line) and cur:
            units.append("\n".join(cur)); cur = []
        cur.append(line)
    units.append("\n".join(cur))
    n_wiki = len(units)
    seen, n_src = set(), 0
    for sub in ("exllamav3", "tools", "doc", "examples", "eval", "tests"):
        for dp, dn, fn in sorted(os.walk(os.path.join(ROOT, sub))):
            dn.sort()
            if "__pycache__" in dp:
                continue
            for f in sorted(fn):
                if not f.endswith((".py", ".cu", ".cuh", ".h", ".cpp", ".md")) or "_hip." in f or "_rocm" in f:
                    continue
                s = open(os.path.join(dp, f), encoding="utf-8", errors="replace").read()
                h = hashlib.sha1(s.encode()).hexdigest()
                if h in seen or len(s) < 200:
                    continue
                seen.add(h)
                units.append(f"# file: {os.path.relpath(os.path.join(dp, f), ROOT)}\n{s}")
                n_src += 1
    random.Random(args.seed).shuffle(units)
    ids = tok.encode("\n\n".join(units), add_bos=False)
    RES["doc"] = {"wiki_units": n_wiki, "src_units": n_src, "tokens_available": int(ids.shape[-1]),
                  "sha1_first_need": hashlib.sha1(ids[0, :need].numpy().tobytes()).hexdigest()}
    assert ids.shape[-1] >= need, f"document too short: {ids.shape[-1]} < {need} (no repetition allowed)"
    return ids[:, :need].contiguous()


def dec_text(tok, out):
    try:
        d = tok.decode(torch.tensor([out]))
        return (d[0] if isinstance(d, list) else d)[:160]
    except Exception as e:
        return repr(e)


def cache_bytes(cache):
    tot = 0
    for layer in cache.layers.values():
        for v in vars(layer).values():
            if isinstance(v, torch.Tensor) and v.is_cuda:
                tot += v.numel() * v.element_size()
    return tot


@torch.inference_mode()
def main():
    torch.set_grad_enabled(False)
    from exllamav3 import Config, Model, Cache, Tokenizer

    cps = [int(x) for x in args.checkpoints.split(",")] if args.mode == "curve" else []
    pdep = [int(x) for x in args.prof_depths.split(",")] if args.mode == "prof" else []
    pfd = [int(x) for x in args.pf_depths.split(",")] if args.mode == "pfprof" else []
    top = max(cps + [args.total]) if args.mode == "curve" else max(pdep) if args.mode == "prof" else \
        max(pfd) + args.chunk if args.mode == "pfprof" else 8192
    cap = ((top + args.decode + 64 + PAGE - 1) // PAGE + 1) * PAGE
    t0 = time.time()
    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    tok = Tokenizer.from_config(config)
    cache = Cache(model, max_num_tokens=cap, max_batch_size=1)
    model.load(device="cuda:0", progressbar=False)
    V = tok.actual_vocab_size
    kvb = cache_bytes(cache)
    RES.update(load_s=round(time.time() - t0, 1), cache_tokens=cap, kv_bytes_alloc=kvb,
               kv_bytes_per_token=kvb / cap, gtt_after_load_gb=round(gtt_gb(), 2), avail_after_load_gb=round(avail_gb(), 1))
    print(f"loaded {RES['load_s']}s cache {cap} tok = {kvb/1e9:.2f} GB ({kvb/cap:.0f} B/tok) gtt {RES['gtt_after_load_gb']} "
          f"avail {RES['avail_after_load_gb']}", flush=True)
    mem_guard("after_load")
    save()
    bt_full = torch.arange(cap // PAGE, dtype=torch.int32).unsqueeze(0)
    st = {"s": cache.get_new_state()}

    def fwd(ids, pos, prefill_only=False):
        T = ids.shape[-1]
        params = {"attn_mode": "flash_attn", "block_table": bt_full[:, : (pos + T + PAGE - 1) // PAGE],
                  "cache": cache, "cache_seqlens": torch.tensor([pos], dtype=torch.int32),
                  "recurrent_states": [st["s"]]}
        return model.prefill(ids, params) if prefill_only else model.forward(ids, params)

    def decode(first, pos, n):
        nxt, times, out = first, [], []
        for k in range(n):
            torch.cuda.synchronize(); a = time.perf_counter()
            lg = fwd(torch.tensor([[nxt]], dtype=torch.long), pos + k)
            nxt = int(lg[0, -1, :V].argmax().item())
            times.append(time.perf_counter() - a)
            out.append(nxt)
        return out, times

    def fresh_state(position=0, test=False):
        st["s"].free()
        st["s"] = cache.get_test_state(position) if test else cache.get_new_state()

    if args.mode == "prof":
        RES["prof"] = []
        for d in pdep:
            fresh_state(d, test=True)
            decode(1000, d, 3)                                   # warm: graph capture, JIT at this length
            fresh_state(d, test=True)
            torch.cuda.synchronize()
            w0 = {"mono": time.monotonic_ns(), "boot": time.clock_gettime_ns(time.CLOCK_BOOTTIME)}
            _, times = decode(1000, d, args.prof_steps)
            torch.cuda.synchronize()
            w1 = {"mono": time.monotonic_ns(), "boot": time.clock_gettime_ns(time.CLOCK_BOOTTIME)}
            RES["prof"].append({"depth": d, "steps": args.prof_steps, "start": w0, "end": w1,
                                "ms_per_tok": round(1000 * sum(times) / len(times), 2)})
            print(f"[prof] depth {d}: {1000*sum(times)/len(times):.2f} ms/tok", flush=True)
            save()
        return

    if args.mode == "pfprof":
        pfprof(model, tok, cache, fwd, fresh_state, pfd)
        return

    # ---------------- curve
    need = args.total + 1
    ids = build_doc(tok, need)
    print(f"doc: {RES['doc']}", flush=True)
    C = args.chunk

    if args.mode == "ctrl":
        # Short-context control: same tokens as the curve's last two chunks before each end E,
        # scored with only 4096 tokens of context (window ids[E-8192:E], NLL of targets E-4095..E).
        RES["ctrl"] = []
        for E in [int(x) for x in args.ctrl_ends.split(",")]:
            fresh_state()
            tot, n = 0.0, 0
            for p0 in range(E - 8192, E, C):
                lg = fwd(ids[:, p0 : p0 + C], p0 - (E - 8192))
                if p0 >= E - 4096:
                    tgt = ids[0, p0 + 1 : p0 + C + 1].to(lg.device)
                    lp = torch.log_softmax(lg[0, :, :V].float(), dim=-1)
                    tot += -lp.gather(-1, tgt.unsqueeze(-1)).sum().item(); n += C
                    del lp
                del lg
            RES["ctrl"].append({"end": E, "tokens": n, "nll_ctx4k": round(tot / n, 5)})
            print(f"[ctrl] {json.dumps(RES['ctrl'][-1])}", flush=True)
            save()
        return
    # warmup on a separate text (end of the shuffled doc is not used: take a fixed code-free string),
    # plus the lm_head overhead of forward() vs prefill() at pos 0
    wu = tok.encode(("The quick brown fox jumps over the lazy dog. " * 400), add_bos=False)[:, :C]
    fwd(wu, 0); decode(11, C, 4)
    fresh_state(); torch.cuda.synchronize(); a = time.perf_counter(); fwd(wu, 0, prefill_only=True); torch.cuda.synchronize()
    tp = time.perf_counter() - a
    fresh_state(); torch.cuda.synchronize(); a = time.perf_counter(); fwd(wu, 0); torch.cuda.synchronize()
    tf = time.perf_counter() - a
    fresh_state()
    RES["warmup"] = {"chunk": C, "prefill_s": round(tp, 3), "forward_s": round(tf, 3),
                     "head_overhead_s": round(tf - tp, 3), "note": "chunk timings below use forward() (logits needed); subtract head_overhead_s for prefill()-only rate"}
    print(f"warmup: {RES['warmup']}", flush=True)
    save()

    edges = [0] + cps
    bucket = {f"{edges[i]}-{edges[i+1]}": [0.0, 0] for i in range(len(edges) - 1)}
    chunks, points = [], []
    RES.update(chunks=chunks, checkpoints=points, buckets={})
    t_pass = time.time()
    pos = 0
    while pos < args.total:
        T = min(C, args.total - pos)
        x = ids[:, pos : pos + T]
        torch.cuda.synchronize(); a = time.perf_counter()
        logits = fwd(x, pos)
        torch.cuda.synchronize()
        dt = time.perf_counter() - a
        # NLL of tokens pos+1 .. pos+T (target = next token of each position)
        tgt = ids[0, pos + 1 : pos + T + 1].to(logits.device)
        nll = 0.0
        for s in range(0, T, 512):
            lp = torch.log_softmax(logits[0, s : s + 512, :V].float(), dim=-1)
            g = -lp.gather(-1, tgt[s : s + 512].unsqueeze(-1)).squeeze(-1)
            for i, (k, v) in enumerate(bucket.items()):
                lo, hi = edges[i], edges[i + 1]
                p = torch.arange(pos + 1 + s, pos + 1 + s + g.numel(), device=g.device)
                m = (p > lo) & (p <= hi)
                if m.any():
                    v[0] += g[m].double().sum().item(); v[1] += int(m.sum().item())
            nll += g.double().sum().item()
            del lp, g
        last_tok = int(logits[0, -1, :V].argmax().item())
        del logits
        pos += T
        chunks.append({"end": pos, "T": T, "s": round(dt, 4), "tps": round(T / dt, 1), "nll": round(nll / T, 5)})
        if len(chunks) % 16 == 0:
            a_gb = mem_guard(f"chunk@{pos}")
            print(f"[chunk] end {pos} {T/dt:.1f} t/s nll {nll/T:.4f} avail {a_gb:.1f} GiB elapsed {time.time()-t_pass:.0f}s", flush=True)
            RES["buckets"] = {k: {"tokens": v[1], "ppl": math.exp(v[0] / v[1]) if v[1] else None} for k, v in bucket.items()}
            save()
        if pos in cps:
            stashed = st["s"].stash()
            out, times = decode(last_tok, pos, args.decode)
            st["s"].position = stashed["position"]
            st["s"].unstash(stashed)                                  # KDA state back to pos; attention rewinds by seqlen
            win = [c for c in chunks[-4:]]
            pt = {"depth": pos, "decode_tokens": len(out), "decode_tps": round(len(times) / sum(times), 2),
                  "decode_tps_excl_first": round((len(times) - 1) / sum(times[1:]), 2),
                  "decode_ms_median": round(1000 * sorted(times)[len(times) // 2], 2),
                  "prefill_chunk_tps_last4": round(sum(c["T"] for c in win) / sum(c["s"] for c in win), 1),
                  "kv_gb": round(RES["kv_bytes_per_token"] * pos / 1e9, 2), "gtt_gb": round(gtt_gb(), 2),
                  "avail_gb": round(avail_gb(), 1), "elapsed_s": round(time.time() - t_pass, 1),
                  "decoded_text": dec_text(tok, out)}
            points.append(pt)
            print(f"[checkpoint] {json.dumps(pt)}", flush=True)
            RES["buckets"] = {k: {"tokens": v[1], "ppl": math.exp(v[0] / v[1]) if v[1] else None} for k, v in bucket.items()}
            save()
    RES["pass_s"] = round(time.time() - t_pass, 1)
    RES["buckets"] = {k: {"tokens": v[1], "ppl": math.exp(v[0] / v[1]) if v[1] else None} for k, v in bucket.items()}
    save()
    print(f"[done] pass {RES['pass_s']}s buckets {json.dumps(RES['buckets'])}", flush=True)


def pfprof(model, tok, cache, fwd, fresh_state, pfd):
    import statistics
    from exllamav3.modules.mla_attn import MLAttention
    from exllamav3.modules.gated_delta_net import GatedDeltaNet
    from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
    import exllamav3.modules.attention_fn.dsa_triton as dsat
    from exllamav3.ext import exllamav3_ext as ext

    ON, REC = [False], []

    def wrap(label, fn):
        def w(*a, **k):
            if not ON[0]:
                return fn(*a, **k)
            e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
            c0 = time.perf_counter(); e0.record()
            r = fn(*a, **k)
            e1.record(); REC.append((label, e0, e1, time.perf_counter() - c0))
            return r
        return w

    # sublayers (instance forward) and top-level modules
    for m in model.modules:
        m.forward = wrap("top", m.forward)
        for attr in ("attn", "mlp"):
            sub = getattr(m, attr, None)
            if sub is None:
                continue
            lab = {MLAttention: "mla", GatedDeltaNet: "kda", BlockSparseMLP: "moe"}.get(type(sub), f"{attr}:{type(sub).__name__}")
            sub.forward = wrap(lab, sub.forward)
    for meth in ("_attend_pre", "_indexer_keys", "_update_pool_plane", "_indexer_topk_kpool", "_indexer_topk", "_attend_sparse"):
        setattr(MLAttention, meth, wrap("mla." + meth.strip("_"), getattr(MLAttention, meth)))
    dsat.dsa_indexer_scores = wrap("idx.scores", dsat.dsa_indexer_scores)
    ext.dsa_topk = wrap("idx.dsa_topk", ext.dsa_topk)

    # random fp16 in every attention/indexer cache tensor
    g = torch.Generator(device="cuda:0"); g.manual_seed(1)
    nfill = 0
    for layer in cache.layers.values():
        for v in vars(layer).values():
            if isinstance(v, torch.Tensor) and v.is_cuda and v.dtype == torch.half:
                f = v.view(-1)
                for i in range(0, f.numel(), 1 << 26):
                    sl = f[i: i + (1 << 26)]
                    sl.copy_(torch.randn(sl.shape, generator=g, device=sl.device, dtype=torch.float32).mul_(0.5))
                nfill += v.numel() * 2
    RES["pf_filled_bytes"] = nfill
    C = args.chunk
    text = open(args.corpus, encoding="utf-8").read()[:200000]
    ids = tok.encode(text, add_bos=False)[:, :C].contiguous()
    assert ids.shape[-1] == C

    def run(d, instr):
        fresh_state(d, test=True)
        REC.clear(); ON[0] = instr
        torch.cuda.synchronize(); a = time.perf_counter()
        lg = fwd(ids, d)
        torch.cuda.synchronize(); dt = time.perf_counter() - a
        ON[0] = False
        del lg
        agg = {}
        for lab, e0, e1, cpu in REC:
            x = agg.setdefault(lab, [0.0, 0.0, 0])
            x[0] += e0.elapsed_time(e1); x[1] += cpu * 1e3; x[2] += 1
        REC.clear()
        return dt, agg

    print(f"pfprof: filled {nfill/1e9:.2f} GB, depths {pfd}, chunk {C}", flush=True)
    if args.pf_arms:
        return pfab(pfd, run, ids, fwd, fresh_state, tok)
    for d in pfd:                       # warm: triton JIT per shape, tile counts
        run(d, False); run(d, True)
    RES["pfprof"] = {str(d): {"plain_s": [], "instr_s": [], "agg": []} for d in pfd}
    for rep in range(args.pf_reps):
        for d in pfd:
            r = RES["pfprof"][str(d)]
            dt, _ = run(d, False); r["plain_s"].append(round(dt, 4))
            dt2, agg = run(d, True); r["instr_s"].append(round(dt2, 4))
            r["agg"].append({k: [round(v[0], 3), round(v[1], 3), v[2]] for k, v in agg.items()})
            print(f"[pf] rep {rep} depth {d}: plain {dt*1e3:.1f} ms ({C/dt:.1f} t/s) instr {dt2*1e3:.1f} ms "
                  + " ".join(f"{k}={v[0]:.1f}" for k, v in sorted(agg.items())), flush=True)
            mem_guard(f"pf@{d}")
            save()
    for d in pfd:
        r = RES["pfprof"][str(d)]
        keys = sorted({k for a in r["agg"] for k in a})
        r["median"] = {"plain_ms": round(1e3 * statistics.median(r["plain_s"]), 2),
                       "instr_ms": round(1e3 * statistics.median(r["instr_s"]), 2),
                       **{k: {"gpu_ms": round(statistics.median(a[k][0] for a in r["agg"]), 3),
                              "cpu_ms": round(statistics.median(a[k][1] for a in r["agg"]), 3),
                              "calls": r["agg"][0][k][2]} for k in keys}}
    save()


ARMS = {"O": (False, False), "A": (True, False), "B": (True, True)}


def set_arm(a):
    import exllamav3.ext_fallbacks as fb
    import exllamav3.modules.mla_attn as mm
    fb.DSA_TOPK_ALLVALID_FAST, mm.EXL3_DSA_PF_NOTILE = ARMS[a]


def pfab(pfd, run, ids, fwd, fresh_state, tok):
    """pfDepth2: one load, arms switched in-process. (1) bitwise DSA indices (every kpool call, set
    and order) and chunk logits of each arm vs O at --pf-check depths, plus O vs O; (2) 32-token
    greedy ids per arm at the same depths; (3) timing, arms interleaved in --pf-arms order."""
    import statistics
    from exllamav3.modules.mla_attn import MLAttention
    arms = args.pf_arms.split(",")
    uniq = sorted(set(arms), key=arms.index)
    chk = [int(x) for x in args.pf_check.split(",")] if args.pf_check else []
    CAP = [None]
    inner = MLAttention._indexer_topk_kpool

    def cap_topk(self, *a, **k):
        r = inner(self, *a, **k)
        if CAP[0] is not None:
            CAP[0].append(r.cpu())
        return r
    MLAttention._indexer_topk_kpool = cap_topk
    V = tok.actual_vocab_size

    def capture(d, arm):
        set_arm(arm); fresh_state(d, test=True); CAP[0] = []
        t = time.perf_counter()
        lg = fwd(ids, d)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        out, CAP[0] = CAP[0], None
        nxt = int(lg[0, -1, :V].argmax().item()); gen = [nxt]
        lgc = lg[0, -64:, :V].float().cpu(); del lg          # last 64 rows (host RAM bound)
        for i in range(31):
            l2 = fwd(torch.tensor([[nxt]], dtype=torch.long), d + ids.shape[-1] + i)
            nxt = int(l2[0, -1, :V].argmax().item()); gen.append(nxt)
        return out, lgc, gen, dt

    RES["pfab_check"] = {}
    for d in chk:
        ref, ref_lg, ref_gen, dt = capture(d, "O")
        print(f"[chk] depth {d} arm O: {len(ref)} kpool calls, {dt*1e3:.0f} ms, ids {ref_gen[:8]}", flush=True)
        for arm in ["O"] + [a for a in uniq if a != "O"]:
            out, lg, gen, dt = capture(d, arm)
            same = sum(torch.equal(x, y) for x, y in zip(out, ref))
            sets = sum(torch.equal(x.sort(1).values, y.sort(1).values) for x, y in zip(out, ref))
            rows = sum(int((x != y).any(1).sum()) for x, y in zip(out, ref))
            r = {"calls": len(out), "bitwise_calls": same, "set_equal_calls": sets, "rows_differ": rows,
                 "logits_max_abs": float((lg - ref_lg).abs().max()), "greedy_same": sum(a == b for a, b in zip(gen, ref_gen)),
                 "ids": gen, "ms": round(dt * 1e3, 1)}
            RES["pfab_check"][f"{arm}@{d}"] = r
            print(f"[chk] depth {d} arm {arm} vs O: bitwise {same}/{len(out)} set {sets} rows_differ {rows} "
                  f"logits_max_abs {r['logits_max_abs']:.3g} greedy {r['greedy_same']}/32 ({dt*1e3:.0f} ms)", flush=True)
            save()
    for d in pfd:                        # warm every arm x depth (JIT: NOTILE's pow2 score stride)
        for arm in uniq:
            set_arm(arm); run(d, False)
    P = RES["pfprof"] = {f"{a}@{d}": {"plain_s": [], "instr_s": [], "agg": []} for a in uniq for d in pfd}
    for rep in range(args.pf_reps):
        for arm in arms:
            set_arm(arm)
            for d in pfd:
                r = P[f"{arm}@{d}"]
                dt, _ = run(d, False); r["plain_s"].append(round(dt, 4))
                dt2, agg = run(d, True); r["instr_s"].append(round(dt2, 4))
                r["agg"].append({k: [round(v[0], 3), round(v[1], 3), v[2]] for k, v in agg.items()})
                print(f"[pf] rep {rep} arm {arm} depth {d}: plain {dt*1e3:.1f} instr {dt2*1e3:.1f} "
                      f"topk={agg.get('idx.dsa_topk', [0])[0]:.1f} kpool={agg.get('mla.indexer_topk_kpool', [0])[0]:.1f}", flush=True)
                mem_guard(f"pf@{arm}{d}")
                save()
    set_arm("A")
    for key, r in P.items():
        keys = sorted({k for a in r["agg"] for k in a})
        r["median"] = {"plain_ms": round(1e3 * statistics.median(r["plain_s"]), 2),
                       "instr_ms": round(1e3 * statistics.median(r["instr_s"]), 2),
                       **{k: {"gpu_ms": round(statistics.median(a[k][0] for a in r["agg"]), 3),
                              "cpu_ms": round(statistics.median(a[k][1] for a in r["agg"]), 3),
                              "calls": r["agg"][0][k][2]} for k in keys}}
    save()


if __name__ == "__main__":
    main()
