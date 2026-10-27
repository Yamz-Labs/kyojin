"""DS4.1 pack bench, one stage per call (each fits one gpurun ticket). Results accumulate in --out (JSON).

    ds41_bench.py PACK --stage prefill|decode|dspark|quality|longctx|maketeacher [options]

prefill     prefill tok/s at --lens (default 2048,16384); the first run of a length is cold (tier fill), the second warm
decode      plain greedy decode tok/s after a 2048-token real prompt (--new tokens)
dspark      the same prompt with DSpark (--draft depth): tok/s, accepted tokens per round, speedup over plain
quality     top-1 agreement and KL(teacher || pack) against teacher files (--teacher DIR/teacher/row_XXX.npz), resumable:
            rows already in --out are skipped, the stage stops after --budget seconds (exit 75 = run again)
longctx     needle in a --ctx token real-text haystack (prefill + 12 greedy tokens), pass = the code is in the answer
maketeacher write teacher-format files from THIS pack (self-test of the quality stage: KL 0, top-1 100 %)

Teacher format (recipe teacher.py): row_XXX.npz with ids [L], top_idx int32 [L-1,256], top_val fp16 [L-1,256] (logits),
lse f32 (full-vocab logsumexp), tgt_lp f32 (log-prob of the next token), argmax i32. Rows are data/short.npy (38 x 2048)
then data/long.npy (3 x 16384): row index 0..37 short, 38..40 long.
KL is exact on the teacher top-256 plus one tail bucket (teacher tail mass vs pack tail mass).
"""
import argparse, glob, json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
H = os.path.expanduser("~/ds41_work")


def load_json(p):
    return json.load(open(p)) if os.path.exists(p) else {}


def save_json(p, d):
    tmp = p + ".tmp"
    json.dump(d, open(tmp, "w"), indent=1, default=float)
    os.replace(tmp, p)


def data_rows(a):
    s = np.load(os.path.join(a.data, "short.npy")).astype(np.int64)
    l = np.load(os.path.join(a.data, "long.npy")).astype(np.int64)
    return s, l


def row_ids(a, i):
    s, l = data_rows(a)
    return (s[i] if i < len(s) else l[i - len(s)]).tolist()


def sync():
    torch.cuda.synchronize()


def stage_prefill(m, a, res):
    out = {}
    s, l = data_rows(a)
    for L in [int(x) for x in a.lens.split(",")]:
        row = l[0] if L > 2048 else s[0]
        ids = torch.tensor([row[:L].tolist()], device="cuda:0")
        runs = []
        for rep in range(2 if L <= 4096 else 1):
            st = m.new_state(1, L + 32)
            sync(); t = time.perf_counter()
            m.prefill(ids, st, 0, chunk=a.chunk)
            sync(); runs.append(L / (time.perf_counter() - t))
            del st
        out[str(L)] = {"cold_tok_s": round(runs[0], 1), "warm_tok_s": round(runs[-1], 1)}
        print(f"prefill {L}: {out[str(L)]}", flush=True)
    res["prefill"] = out


def make_engine(m, a, num_draft):
    from ds41_engine import HFTokenizer
    from ds41_model import PortEngine
    return PortEngine(m, HFTokenizer(os.path.join(a.pack, "tokenizer.json")), ctx=a.ctx, num_draft=num_draft, eos_id=None,
                      prefix_cache=False)


def run_gen(eng, ids, n):
    from ds41_model import TopPSampler
    toks = []
    st = eng._run(ids, n, TopPSampler(0.0, 1.0, None), -1, None, lambda t: toks.extend(t))
    return st, toks


def stage_decode(m, a, res):
    ids = row_ids(a, 0)
    eng = make_engine(m, a, 0)
    run_gen(eng, ids, 8)                                   # warm-up (kernel JIT, hot set)
    st, toks = run_gen(eng, ids, a.new)
    r = {"new": st["new_tokens"], "prefill_s": round(st["time_prefill"], 3),
         "decode_tok_s": round((st["new_tokens"] - 1) / st["time_generate"], 2)}
    print("decode plain:", r, flush=True)
    res["decode"] = r


def stage_dspark(m, a, res):
    ids = row_ids(a, 0)
    plain = make_engine(m, a, 0)
    run_gen(plain, ids, 8)
    st0, toks0 = run_gen(plain, ids, a.new)
    eng = make_engine(m, a, a.draft)
    if eng.spec is None:
        raise SystemExit("this pack has no DSpark stages (mtp.*)")
    run_gen(eng, ids, 8)
    st, toks = run_gen(eng, ids, a.new)
    r = {"draft": a.draft, "new": st["new_tokens"], "dspark_tok_s": round((st["new_tokens"] - 1) / st["time_generate"], 2),
         "plain_tok_s": round((st0["new_tokens"] - 1) / st0["time_generate"], 2),
         "greedy_equal_to_plain": toks[:len(toks0)] == toks0[:len(toks)],
         "stats": {k: v for k, v in st.items() if k not in ("time_prefill", "time_generate")}}
    r["speedup"] = round(r["dspark_tok_s"] / r["plain_tok_s"], 3)
    print("dspark:", r, flush=True)
    res["dspark"] = r


def row_logits(m, ids, chunk, keep=None):
    """Teacher-forced logits [L-1, V] (fp32, CPU-free): yields (start, logits) per chunk, position t predicts token t+1."""
    L = len(ids)
    st = m.new_state(1, L + 8)
    t = torch.tensor([ids], device="cuda:0")
    for a0 in range(0, L - 1, chunk):
        b0 = min(L - 1, a0 + chunk)
        lg, _ = m._forward(t[:, a0:b0], a0, st, None, "all", ahead_ids=t[:, b0:b0 + chunk] if b0 < L - 1 else None)
        yield a0, lg[0].float()


def stage_maketeacher(m, a, res):
    os.makedirs(os.path.join(a.teacher, "teacher"), exist_ok=True)
    for i in [int(x) for x in a.rowlist.split(",")]:
        ids = row_ids(a, i)[:a.trunc]
        tv, ti, ls, tl, am = [], [], [], [], []
        for a0, lg in row_logits(m, ids, a.chunk):
            lse = torch.logsumexp(lg, -1)
            v, ix = lg.topk(256, -1)
            tgt = torch.tensor(ids[a0 + 1:a0 + 1 + lg.size(0)], device=lg.device)
            tv.append(v.half().cpu()); ti.append(ix.int().cpu()); ls.append(lse.cpu())
            tl.append((lg.gather(1, tgt[:, None])[:, 0] - lse).cpu()); am.append(lg.argmax(-1).int().cpu())
        np.savez(os.path.join(a.teacher, "teacher", f"row_{i:03d}.npz"), ids=np.array(ids), top_idx=torch.cat(ti).numpy(),
                 top_val=torch.cat(tv).numpy(), lse=torch.cat(ls).numpy(), tgt_lp=torch.cat(tl).numpy(), argmax=torch.cat(am).numpy())
        print("wrote teacher row", i, len(ids), flush=True)


def stage_quality(m, a, res):
    q = res.setdefault("quality", {})
    files = sorted(glob.glob(os.path.join(a.teacher, "teacher", "row_*.npz")))
    if not files:
        raise SystemExit(f"quality: no teacher files in {a.teacher}/teacher")
    t0 = time.time()
    todo = [f for f in files if os.path.basename(f) not in q][:a.rows if a.rows else None]
    for f in todo:
        if time.time() - t0 > a.budget:
            save_json(a.out, res); print("budget reached, run again", flush=True); return 75
        z = np.load(f)
        ids = z["ids"].tolist()
        if a.trunc and len(ids) > a.trunc:
            ids = ids[:a.trunc]
        P = len(ids) - 1
        ti, tv, lse = (torch.from_numpy(z[k][:P]).to("cuda:0") for k in ("top_idx", "top_val", "lse"))
        tam, tlp = torch.from_numpy(z["argmax"][:P]).to("cuda:0"), torch.from_numpy(z["tgt_lp"][:P]).to("cuda:0")
        kl = torch.zeros(P, device="cuda:0"); top1 = torch.zeros(P, device="cuda:0"); nll = torch.zeros(P, device="cuda:0")
        for a0, lg in row_logits(m, ids, a.chunk):
            n = lg.size(0)
            sl = slice(a0, a0 + n)
            ls = torch.log_softmax(lg, -1)
            lst = ls.gather(1, ti[sl].long())
            pt = (tv[sl].float() - lse[sl, None]).exp()
            lpt = tv[sl].float() - lse[sl, None]
            qt = (1 - pt.sum(1)).clamp_min(1e-9)
            qs = (1 - lst.exp().sum(1)).clamp_min(1e-9)
            kl[sl] = (pt * (lpt - lst)).sum(1) + qt * (qt.log() - qs.log())
            top1[sl] = (lg.argmax(-1) == tam[sl]).float()
            tgt = torch.tensor(ids[a0 + 1:a0 + 1 + n], device="cuda:0")
            nll[sl] = -ls.gather(1, tgt[:, None])[:, 0]
        q[os.path.basename(f)] = {"positions": P, "kl_mean": kl.mean().item(), "top1": top1.mean().item(),
                                  "nll_pack": nll.mean().item(), "nll_teacher": (-tlp).mean().item()}
        print(os.path.basename(f), q[os.path.basename(f)], flush=True)
        save_json(a.out, res)
    if q:
        w = np.array([v["positions"] for v in q.values()], dtype=float)
        g = lambda k: float((np.array([v[k] for v in q.values()]) * w).sum() / w.sum())
        res["quality_summary"] = {"rows": len(q), "positions": int(w.sum()), "kl_mean": g("kl_mean"), "top1": g("top1"),
                                  "nll_pack": g("nll_pack"), "nll_teacher": g("nll_teacher")}
        print("quality summary:", res["quality_summary"], flush=True)
    return 0


def stage_longctx(m, a, res):
    from ds41_engine import HFTokenizer
    tk = HFTokenizer(os.path.join(a.pack, "tokenizer.json"))
    s, l = data_rows(a)
    hay = np.concatenate([l[0], l[1], l[2]]).tolist()[:a.ctx - 200]
    code = "48213"
    needle = tk.encode(f"\n\nNote for the reader: the secret access code is {code}. Remember it.\n\n")
    q = tk.encode("\n\nQuestion: what is the secret access code mentioned in the text above?\nAnswer: The secret access code is")
    mid = len(hay) // 2
    ids = hay[:mid] + needle + hay[mid:] + q
    eng = make_engine(m, a, 0)
    st, toks = run_gen(eng, ids, 12)
    txt = tk._tk.decode(toks)
    r = {"prompt_tokens": len(ids), "prefill_s": round(st["time_prefill"], 1), "answer": txt, "pass": code in txt.replace(" ", "")}
    print("longctx:", r, flush=True)
    res["longctx"] = r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pack")
    ap.add_argument("--stage", required=True)
    ap.add_argument("--out", default=os.path.join(H, "ds41ready_scratch", "bench.json"))
    # short.npy / long.npy: ds41ready_data (box) or the recipe bench data (worker)
    ap.add_argument("--data", default=next((d for d in (os.path.join(H, "ds41ready_data"), os.path.join(H, "recipe", "bench", "data"))
                                            if os.path.exists(os.path.join(d, "short.npy"))), os.path.join(H, "ds41ready_data")))
    ap.add_argument("--teacher", default=os.path.join(H, "ds41ready_data", "teacher"))
    ap.add_argument("--layers", type=int)
    ap.add_argument("--mtp", default="auto")
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--lens", default="2048,16384")
    ap.add_argument("--ctx", type=int, default=20480)
    ap.add_argument("--new", type=int, default=64)
    ap.add_argument("--draft", type=int, default=4)
    ap.add_argument("--rows", type=int, default=0, help="quality: at most N new rows this call")
    ap.add_argument("--budget", type=float, default=480)
    ap.add_argument("--trunc", type=int, default=0, help="use only the first N tokens of a row")
    ap.add_argument("--rowlist", default="0,1")
    ap.add_argument("--engram-dir")
    a = ap.parse_args()
    a.pack = os.path.expanduser(a.pack)
    from ds41_serving import build_model
    mtp = a.stage == "dspark" if a.mtp == "auto" else a.mtp == "1"
    t0 = time.time()
    m = build_model(a.pack, device="cuda:0", dtype="float16", layers=a.layers, mtp=mtp, moe="auto",
                    prefill_chunk=a.chunk, engram_dir=a.engram_dir)
    sync()
    print(f"load {time.time() - t0:.1f}s", flush=True)
    res = load_json(a.out)
    rc = {"prefill": stage_prefill, "decode": stage_decode, "dspark": stage_dspark, "quality": stage_quality,
          "longctx": stage_longctx, "maketeacher": stage_maketeacher}[a.stage](m, a, res)
    if os.environ.get("EXL3_EXPERT_BUDGET_GIB"):
        from exllamav3.modules.expert_store import get_store
        blk = [b for b in m.pack_modules[0].modules if getattr(b, "tier", None) is not None]
        if blk:
            res.setdefault("tier_stats", {})[a.stage] = get_store(blk[0].config).stats()
            print("tier stats:", res["tier_stats"][a.stage], flush=True)
    save_json(a.out, res)
    print("saved", a.out)
    return rc if isinstance(rc, int) else 0


if __name__ == "__main__":
    sys.exit(main())
