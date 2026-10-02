#!/usr/bin/env python3
"""
GLM-5.3 depth curve on the SERVED decode path (MTP n1f2), in one pass. depth1.

Why this exists: tools/glm/depth_curve.py measures the depth curve with a hand-rolled raw
prefill/decode loop (plain R=1 decode, no drafter). The served lane is MTP n1f2
(num_draft_tokens=1 + MTP_FUSE_CATCHUP=2 + EXL3_MOE_UNION_V2=1, serve.py:150-170), and the
GOAL target "speed flat to 500K" is a served number. So: same one-pass idea, served path.

modes
  curve  ONE prefill of a ~500K real document (wikitext-2 test articles + this repo's own
         sources, shuffled at unit level, exactly depth_curve.py's build_doc so the document
         matches the earlier curve runs), driven by the Generator itself so the K/V pages, the
         page hash chain and the page-aligned recurrent stashes all exist. Prefill chunks are
         timed by wrapping model.forward (the MTP prefill path, identified by
         params["last_tokens_only"]) and draft_model.prefill -> prefill t/s vs depth.
         At each checkpoint a 64-token MTP job on the same document prefix is enqueued: the
         prefix pages are found by content hash and the recurrent state is restored from the
         page-aligned stash, so the job re-prefills only the gap (`replay_tokens`, reported, and
         never inside the decode window) and the window measures served t/s + accept at that
         depth. Checkpoints ascend, so every replay is bounded by the next checkpoint.
  prof   Fake-depth decode windows for rocprofv3: K/V pages filled with random fp16 and a test
         recurrent state at position d (no prefill at all), N served rounds at each
         --prof-depths, windows stamped in monotonic ns + boottime. Round cost is
         content-independent except the DSA top-k, so the curve's real t/s at the same depth
         cross-checks the fake one; the harness prints both.

Both modes write JSON after every point, so a crash keeps the finished work.
"""
import argparse, ast, glob, hashlib, json, math, os, random, re, statistics, sys, time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["curve", "prof"])
ap.add_argument("-m", "--model", default="")
ap.add_argument("-o", "--out", required=True)
ap.add_argument("--total", type=int, default=500000, help="curve: document tokens (excl. template)")
ap.add_argument("--checkpoints", default="4096,32768,131072,262144,499712",
                help="curve: absolute sequence positions (page-aligned) for the decode windows")
ap.add_argument("--decode", type=int, default=64, help="curve: decode tokens per checkpoint")
ap.add_argument("--corpus", default="")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--prof-depths", default="4096,499712")
ap.add_argument("--prof-rounds", type=int, default=8)
ap.add_argument("--prof-warm", type=int, default=3)
ap.add_argument("--speed-env", type=int, default=1, help="apply tools/glm/serve.py SPEED_ENV")
ap.add_argument("--min-avail-gb", type=float, default=10.0)
args = ap.parse_args()

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
PAGE = 256


def first_existing(*paths):
    for p in paths:
        if p and os.path.exists(os.path.expanduser(p)):
            return os.path.expanduser(p)
    return None


args.model = args.model or first_existing("~/models/glm53-exl3-td205",
                                          "~/models/glm53-exl3-td205") or ""
assert args.model, "model dir not found; pass -m"
args.corpus = args.corpus or first_existing("~/bench/ppl/wiki.test.raw",
                                            "~/bench/ppl/wiki.test.raw") or ""

# ---- served configuration (serve.py SPEED_ENV, ast-imported so there is one source of truth) ----
if args.speed_env:
    _src = open(os.path.join(ROOT, "tools/glm/serve.py")).read()
    for k, v in next(ast.literal_eval(n.value) for n in ast.parse(_src).body
                     if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "SPEED_ENV").items():
        os.environ.setdefault(k, v)
os.environ.setdefault("EXL3_MOE_UNION_V2", "1")

RES = {"mode": args.mode, "args": vars(args), "started": time.strftime("%F %T"),
       "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("EXL3_")}}


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


def cache_bytes(cache):
    tot = 0
    for layer in cache.layers.values():
        for v in vars(layer).values():
            if isinstance(v, torch.Tensor) and v.is_cuda:
                tot += v.numel() * v.element_size()
    return tot


def build_doc(tok, need):
    """depth_curve.py's document: wiki articles + repo sources, shuffled at unit level. Same
    sha1 as the earlier curve runs if the corpus and the tree match."""
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
    RES["doc"] = {"corpus": args.corpus, "wiki_units": n_wiki, "src_units": n_src,
                  "tokens_available": int(ids.shape[-1]),
                  "sha1": hashlib.sha1(ids[0, :need].numpy().tobytes()).hexdigest()}
    assert ids.shape[-1] >= need, f"document too short: {ids.shape[-1]} < {need}"
    return ids[:, :need].contiguous()


def fill_random(cache, seed=1, block_elems=1 << 25):
    """Random fp16 into every attention/indexer cache tensor, so a fake-depth DSA top-k sees a
    realistic score spread (depth_curve.py pfprof).

    One 64 MiB block of randn is drawn and then *tiled* by a broadcasting copy_. Drawing 19 GB of
    randomness element by element measured at ~19 MB/s on the test box (17 min for a 500K cache, host-bound
    inside torch.randn); the tile covers 32M elements = 256K keys of 128 dims, more than the whole
    pooled key set of a 500K cache, so the scores still spread over the full context.
    """
    torch.manual_seed(seed)
    blk = (torch.randn(block_elems, device="cuda:0", dtype=torch.float32) * 0.5).half()
    n, nt, t0 = 0, 0, time.perf_counter()
    for layer in cache.layers.values():
        for v in vars(layer).values():
            if not (isinstance(v, torch.Tensor) and v.is_cuda and v.dtype == torch.half):
                continue
            f = v.view(-1)
            nel = f.numel()
            full = (nel // block_elems) * block_elems
            if full:
                f[:full].view(-1, block_elems).copy_(blk)
            pos = full
            while pos < nel:
                take = min(block_elems, nel - pos)
                f[pos:pos + take].copy_(blk[:take])
                pos += take
            n += f.numel() * 2
            nt += 1
    torch.cuda.synchronize()
    print(f"[fill] {n/1e9:.2f} GB in {nt} tensors, {time.perf_counter()-t0:.1f} s", flush=True)
    return n


def build_chain(ids, n_pages):
    """Page-hash chain of ids[:n_pages*PAGE], the same function the page table and
    SlotStore use, so a hash computed here matches the one the prefill published."""
    from exllamav3.generator.pagetable import tensor_hash_checksum
    prev, out = None, []
    for pi in range(n_pages):
        prev = tensor_hash_checksum(ids[:, pi * PAGE:(pi + 1) * PAGE], prev)
        out.append(prev)
    return out


def find_page(pt, h):
    return pt.referenced_pages.get(h) or pt.unreferenced_pages.get(h)


def register_chain(gen, ids, n_pages):
    """Publish pages 0..n_pages-1 of `ids` to the Generator's PageTable as a fully cached
    prefix, so a job on ids[:n_pages*PAGE] finds them by content hash and skips the prefill.
    The (fake or raw) prefill wrote page i's K/V into physical page i, so the identity block
    table still holds; defrag is off for that reason. Mirrors SlotStore.restore()."""
    from exllamav3.generator.pagetable import tensor_hash_checksum
    pt, prev, hashes = gen.pagetable, None, []
    for pi in range(n_pages):
        chunk = ids[:, pi * PAGE:(pi + 1) * PAGE]
        assert chunk.shape[-1] == PAGE
        h = tensor_hash_checksum(chunk, prev)
        page = pt.all_pages[pi]
        assert page.page_index == pi and page.ref_count == 0, f"page {pi} not free/identity"
        pt.access_serial += 1
        page.add_ref_clear(pt.access_serial, h)          # kv_position=0, prev_hash=None
        page.prev_hash = prev
        page.kv_position = PAGE
        page.sequence[:, :].copy_(chunk)
        page.can_revert = False
        hashes.append(h)
        prev = h
    return hashes


def inject_state(gen, phash, state):
    """Publish a recurrent state as the stash for page hash `phash`, the way
    RecurrentCache.put()/stash_midchunk() do, so a job resuming at that page gets it verbatim."""
    rc = gen.recurrent_cache
    stashed = state.stash()
    rc.put(phash, None, stashed=stashed)
    return stashed["position"], stashed["checkpoint_size"]


def new_cache_state(cache, position, test=False):
    st = cache.get_test_state(position) if test else cache.get_new_state()
    st.last_history = 0
    return st


@torch.inference_mode()
def main():
    torch.set_grad_enabled(False)
    from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
    from exllamav3.generator import generator as GM
    from exllamav3.generator.sampler import GreedySampler

    import exllamav3
    assert exllamav3.__file__.startswith(ROOT), exllamav3.__file__

    cps = [int(x) for x in args.checkpoints.split(",")] if args.mode == "curve" else []
    pdep = [int(x) for x in args.prof_depths.split(",")] if args.mode == "prof" else []
    top = (max(cps + pdep) if (cps or pdep) else 8192) + args.decode + 64 * PAGE
    top = (top + PAGE - 1) // PAGE * PAGE          # Cache asserts a page multiple
    t0 = time.time()
    config = Config.from_directory(args.model)
    tok = Tokenizer.from_config(config)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=top, max_history=1)
    model.load(device="cuda:0", progressbar=False)
    draft_model = Model.from_config(config, component="mtp")
    draft_cache = Cache(draft_model, max_num_tokens=top, max_history=1)
    draft_model.load(device="cuda:0", progressbar=False)
    GM.MTP_FUSE_CATCHUP = 2                                   # n1f2, as serve.py
    gen = Generator(model=model, cache=cache, tokenizer=tok, max_batch_size=1,
                    draft_model=draft_model, draft_cache=draft_cache, num_draft_tokens=1,
                    record_draft_stats=True, enable_defrag=False)
    kvb, dkvb = cache_bytes(cache), cache_bytes(draft_cache)
    RES.update(load_s=round(time.time() - t0, 1), cache_tokens=top,
               kv_bytes_target=kvb, kv_bytes_draft=dkvb, kv_bytes_total=kvb + dkvb,
               gtt_after_load_gb=round(gtt_gb(), 2), avail_after_load_gb=round(avail_gb(), 1))
    print(f"loaded {RES['load_s']}s  target KV {kvb/1e9:.2f} GB + draft {dkvb/1e9:.2f} GB "
          f"= {(kvb+dkvb)/1e9:.2f} GB  gtt {RES['gtt_after_load_gb']}  avail {RES['avail_after_load_gb']}",
          flush=True)
    mem_guard("after_load")
    save()

    # ---- prefill instrumentation: the MTP prefill path is model.forward with
    # params["last_tokens_only"]; the decode path never sets it. Draft prefill is its own call. ----
    CH = {"pos": 0, "cur": None, "on": False, "phase": "", "list": []}
    _fwd, _dpf = model.forward, draft_model.prefill

    def fwd(input_ids=None, params=None, *a, **k):
        p = params or {}
        if not (CH["on"] and p.get("last_tokens_only")):
            return _fwd(input_ids=input_ids, params=params, *a, **k)
        rows = int(input_ids.shape[-1])
        torch.cuda.synchronize(); t = time.perf_counter()
        r = _fwd(input_ids=input_ids, params=params, *a, **k)
        torch.cuda.synchronize(); dt = time.perf_counter() - t
        CH["cur"] = {"phase": CH["phase"], "pos": CH["pos"], "rows": rows,
                     "tgt_s": round(dt, 4), "draft_s": 0.0}
        CH["pos"] += rows
        CH["cur"]["end"] = CH["pos"]
        return r

    def dpf(input_ids=None, params=None, *a, **k):
        if not CH["on"]:
            return _dpf(input_ids=input_ids, params=params, *a, **k)
        torch.cuda.synchronize(); t = time.perf_counter()
        r = _dpf(input_ids=input_ids, params=params, *a, **k)
        torch.cuda.synchronize(); dt = time.perf_counter() - t
        if CH["cur"] is not None:
            CH["cur"]["draft_s"] = round(dt, 4)
            CH["list"].append(CH["cur"])
        return r

    model.forward = fwd
    draft_model.prefill = dpf

    def run_job(job, window_cb=None, ids_out=None):
        """Drive one job to completion.

        iterate() runs ONE prefill chunk per call, so a call is classified by the job's own
        is_prefill_done() *before* it: prefill-phase calls are accumulated as replay_s and never
        enter the decode window. The call that completes the prefill also emits the first token,
        and it is counted as prefill, so the window starts at the second round.
        """
        gen.enqueue(job)
        n_tok, n_round, pf_s, pf_calls = 0, 0, 0.0, 0
        t_start = time.perf_counter()
        while gen.num_remaining_jobs():
            n0 = job.new_tokens
            pre = not job.is_prefill_done()
            torch.cuda.synchronize(); t = time.perf_counter()
            events = gen.iterate()
            torch.cuda.synchronize(); dt = time.perf_counter() - t
            n1 = job.new_tokens
            if pre:
                pf_s += dt
                pf_calls += 1
            else:
                n_round += 1
                if n1 > n0 and window_cb:
                    window_cb(n1 - n0, dt)
            n_tok += n1 - n0
            for e in events:
                if e.get("error"):
                    raise RuntimeError(str(e["error"]))
                tid = e.get("token_ids")
                if ids_out is not None and tid is not None and getattr(tid, "numel", lambda: 0)():
                    ids_out += [int(x) for x in tid.reshape(-1).tolist()]
        return {"tokens": n_tok, "rounds": n_round, "wall_s": round(time.perf_counter() - t_start, 2),
                "replay_s": round(pf_s, 2), "replay_calls": pf_calls}

    if args.mode == "prof":
        return prof(gen, tok, model, cache, draft_cache, pdep, save, mem_guard)

    # =============================== curve =====================================================
    doc = build_doc(tok, args.total)
    tpl = tok.encode("[gMASK]<sop><|user|>\n", encode_special_tokens=True)
    tail = tok.encode("<|assistant|>\n", encode_special_tokens=True)
    prompt = torch.cat([tpl, doc, tail], dim=1).contiguous()
    N = prompt.shape[-1]
    RES["prompt"] = {"template_tokens": int(tpl.shape[-1] + tail.shape[-1]), "length": N,
                     "sha1": hashlib.sha1(prompt[0].numpy().tobytes()).hexdigest()}
    print(f"doc: {RES['doc']}  prompt {N} tok (template {RES['prompt']['template_tokens']})", flush=True)
    assert max(cps) < N, f"checkpoint {max(cps)} >= prompt length {N}"
    assert all(c % PAGE == 0 for c in cps), "checkpoints must be page aligned"

    # ---- phase 1: one prefill of the whole document -------------------------------------------
    jobA = Job(input_ids=prompt, max_new_tokens=1, sampler=GreedySampler(), stop_conditions=[])
    CH.update(on=True, pos=0, phase="prefill")
    print("[prefill] starting", flush=True)
    rA = run_job(jobA)
    CH["on"] = False
    chunks = list(CH["list"])
    RES["replays"] = []
    RES["chunks"] = chunks
    RES["prefill"] = {"tokens": CH["pos"], "s": round(sum(c["tgt_s"] for c in chunks), 2),
                      "draft_s": round(sum(c["draft_s"] for c in chunks), 2),
                      "wall_s": rA["wall_s"], "iterate_calls": rA["rounds"] + rA["replay_calls"],
                      "tps_target": round(CH["pos"] / max(sum(c["tgt_s"] for c in chunks), 1e-9), 1),
                      "tps_served": round(CH["pos"] / max(sum(c["tgt_s"] + c["draft_s"] for c in chunks), 1e-9), 1),
                      "chunks_n": len(chunks)}
    print(f"[prefill] done {json.dumps(RES['prefill'])}", flush=True)
    save()
    mem_guard("after_prefill")

    # ---- phase 2: decode windows, ascending depth ---------------------------------------------
    rc = gen.recurrent_cache
    RES["recurrent_cache"] = {"size_gb": round(rc.current_size / 2**30, 2), "max_gb": round(rc.max_size / 2**30, 2),
                              "entries": len(rc), "metrics": dict(rc.metrics)}
    points = []
    RES["checkpoints"] = points
    pt_ = gen.pagetable
    chain = build_chain(prompt, N // PAGE)
    print(f"[cp] chain of {len(chain)} page hashes built", flush=True)
    for c in cps:
        mem_guard(f"cp@{c}")
        # most recent page-aligned recurrent stash at or before c, keyed by the page hash chain
        stashed = None
        for pi in range(c // PAGE - 1, -1, -1):
            page = find_page(gen.pagetable, chain[pi])
            if page is None:
                continue
            s = rc.get_stashed(page.phash)
            if s is not None and s["position"] == (pi + 1) * PAGE:
                stashed = s
                break
        # two passes: pass 0 also pays JIT/graph/hipBLASLt-class warmup for these shapes, pass 1 is
        # the steady-state window. Both are reported.
        passes = []
        got = []
        for p_i in range(2):
            CH.update(on=True, pos=0, phase=f"replay@{c}")
            CH["list"] = []
            job = Job(input_ids=prompt[:, :c], max_new_tokens=args.decode, sampler=GreedySampler(),
                      stop_conditions=[])
            t_win = []

            def cb(ntok, dt, _t=t_win, _g=got):
                _t.append((ntok, dt))
            r = run_job(job, window_cb=cb, ids_out=got)
            CH["on"] = False
            dec_s = sum(dt for _, dt in t_win)
            dec_tok = sum(n for n, _ in t_win)
            passes.append({"pass": p_i, "rounds": r["rounds"], "window_rounds": len(t_win),
                           "replay_s": r["replay_s"], "replay_calls": r["replay_calls"],
                           "wall_s": r["wall_s"], "tokens": r["tokens"],
                           "decode_tps": round(dec_tok / max(dec_s, 1e-9), 2),
                           "ms_per_round": round(1000 * dec_s / max(len(t_win), 1), 2)})
            if p_i == 0 and r["replay_calls"]:
                rep = [ch for ch in CH["list"] if ch["phase"] == f"replay@{c}"]
                if rep:
                    RES["replays"].append({"depth": c, "chunks": len(rep), "rows": sum(x["rows"] for x in rep),
                                           "s": round(sum(x["tgt_s"] for x in rep), 2),
                                           "tps": round(sum(x["rows"] for x in rep) /
                                                        max(sum(x["tgt_s"] for x in rep), 1e-9), 1)})
            CH["list"] = []
        if rep:
            RES["replays"].append({"depth": c, "chunks": len(rep), "rows": sum(x["rows"] for x in rep),
                                   "s": round(sum(x["tgt_s"] for x in rep), 2),
                                   "tps": round(sum(x["rows"] for x in rep) /
                                                max(sum(x["tgt_s"] for x in rep), 1e-9), 1),
                                   "first_pos": rep[0]["pos"], "last_pos": rep[-1]["pos"]})
        # accept from the steady-state pass only (job.draft_stats accumulates both passes)
        ds = list(job.draft_stats)
        acc = sum(int(x[2]) for x in ds) / max(len(ds), 1) if ds else 0.0
        warm = passes[-1]
        seg = [ch for ch in chunks if ch["pos"] < c <= ch["end"]]
        pt = {"depth": c, "decode_tokens": warm["tokens"], "rounds": warm["rounds"],
              "decode_tps": warm["decode_tps"], "decode_ms_per_round": warm["ms_per_round"],
              "tok_per_round": round(warm["tokens"] / max(warm["rounds"], 1), 3),
              "accept": round(acc, 4), "draft_rounds": len(ds), "passes": passes,
              "replay_from": (stashed["position"] if stashed else 0),
              "replay_tokens": c - (stashed["position"] if stashed else 0),
              "prefill_chunk_tps": round(seg[-1]["rows"] / (seg[-1]["tgt_s"] + seg[-1]["draft_s"]), 1) if seg else None,
              "prefill_chunk_tps_target": round(seg[-1]["rows"] / seg[-1]["tgt_s"], 1) if seg else None,
              "prefill_seg_tps": round(sum(x["rows"] for x in seg) / max(sum(x["tgt_s"] + x["draft_s"] for x in seg), 1e-9), 1) if seg else None,
              "gtt_gb": round(gtt_gb(), 2), "avail_gb": round(avail_gb(), 1)}
        pt["first_tokens"] = got[:8]
        pt["decoded_text"] = dec_text(tok, got[:24])
        points.append(pt)
        print(f"[cp] {json.dumps(pt)}", flush=True)
        save()
    RES["rc_metrics_end"] = dict(rc.metrics)
    RES["done_s"] = round(time.time() - t0, 1)
    save()
    print(table(RES), flush=True)
    return


def dec_text(tok, out):
    try:
        d = tok.decode(torch.tensor(list(out)))
        return (d[0] if isinstance(d, list) else d)[:120]
    except Exception as e:
        return repr(e)


def table(R):
    pts = R.get("checkpoints") or []
    if not pts:
        return "(no checkpoints)"
    base = pts[0]
    out = ["| depth | decode t/s (MTP n1f2) | vs 4K | accept | ms/round | prefill chunk t/s (served) | prefill chunk t/s (target) |",
           "|---|---|---|---|---|---|---|"]
    for p in pts:
        out.append(f"| {p['depth']} | {p['decode_tps']} | {100*(p['decode_tps']/base['decode_tps']-1):+.1f} % | "
                   f"{p['accept']} | {p['decode_ms_per_round']} | {p['prefill_chunk_tps']} | {p['prefill_chunk_tps_target']} |")
    return "\n".join(out)


def prof(gen, tok, model, cache, draft_cache, depths, save, mem_guard):
    """Fake-depth served rounds for rocprofv3. No prefill: random K/V + a test recurrent state at
    position d, pages published to the page table, then --prof-rounds rounds at each depth."""
    from exllamav3 import Job
    from exllamav3.generator.sampler import GreedySampler
    nmax = max(depths) // PAGE
    nbytes = fill_random(cache) + fill_random(draft_cache, seed=2)
    ids = torch.arange(nmax * PAGE, dtype=torch.long, device="cpu").unsqueeze(0) % 1000 + 100
    hashes = register_chain(gen, ids, nmax)
    RES["prof_fill_bytes"] = nbytes
    RES["prof"] = []
    print(f"[prof] filled {nbytes/1e9:.2f} GB of K/V, {nmax} pages, depths {depths}", flush=True)
    save()
    for d in depths:
        pi = d // PAGE - 1
        assert hashes[pi] is not None
        st = new_cache_state(cache, d, test=True)
        pos, csize = inject_state(gen, hashes[pi], st)
        assert pos == d, (pos, d)
        st.free()
        for it in range(args.prof_warm + 1):
            # min_new_tokens = budget: the prompt is synthetic token noise, so the model can emit EOS
            # on the first round and end the job before there is anything to trace
            job = Job(input_ids=ids[:, :d], max_new_tokens=args.prof_rounds,
                      min_new_tokens=args.prof_rounds, sampler=GreedySampler(), stop_conditions=[])
            gen.enqueue(job)
            rounds, pf_calls, t0, t1 = [], 0, None, None

            def stamp():
                return {"mono": time.monotonic_ns(), "boot": time.clock_gettime_ns(time.CLOCK_BOOTTIME)}

            while gen.num_remaining_jobs():
                n0 = job.new_tokens
                pre = not job.is_prefill_done()
                torch.cuda.synchronize(); t = time.perf_counter()
                gen.iterate()
                torch.cuda.synchronize(); dt = time.perf_counter() - t
                if pre:                      # a prefill chunk leaked into the window: must not happen
                    pf_calls += 1
                    continue
                if t0 is None:               # first decode round: boundary marker, not counted
                    t0 = stamp()
                else:
                    t1 = stamp()
                    rounds.append((job.new_tokens - n0, dt))
            if it == args.prof_warm and len(rounds) >= 2:  # last pass = measured window
                acc = [int(x[2]) for x in job.draft_stats]
                RES["prof"].append({"depth": d, "rounds": len(rounds), "start": t0, "end": t1,
                                    "ms_per_round": round(1000 * statistics.median(dt for _, dt in rounds), 3),
                                    "ms_per_round_min": round(1000 * min(dt for _, dt in rounds), 3),
                                    "tok_per_round": round(sum(n for n, _ in rounds) / max(len(rounds), 1), 3),
                                    "accept": round(sum(acc) / max(len(acc), 1), 3),
                                    "prefill_calls_in_window": pf_calls,
                                    "warm_passes": args.prof_warm,
                                    "stashed_state_mb": round(csize / 2**20, 1)})
                print(f"[prof] depth {d}: {json.dumps(RES['prof'][-1])}", flush=True)
                save()
            elif it == args.prof_warm:
                print(f"[prof] depth {d}: only {len(rounds)+1} decode round(s), no window", flush=True)
        mem_guard(f"prof@{d}")
    save()
    print("[prof] " + "; ".join(f"{p['depth']}: {p['ms_per_round']} ms/round" for p in RES["prof"]), flush=True)
    return


if __name__ == "__main__":
    main()
