"""Big-chunk prefill A/B in one process (one model load): for each prompt length and chunk size, prefill, then 16 greedy
tokens. Prints tok/s, peak torch GPU memory above the pre-prefill level, tier bytes read, token ids. Saves last logits.
usage: ds41_chunk_ab.py PACK --lens 32768,65536 --chunks 16384,32768,65536 [--engram-dir D] [--out DIR] [--rows-off]"""
import argparse, hashlib, json, os, sys, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ap = argparse.ArgumentParser()
ap.add_argument("pack"); ap.add_argument("--lens", default="32768"); ap.add_argument("--chunks", default="16384,32768")
ap.add_argument("--engram-dir"); ap.add_argument("--data", default=os.path.expanduser("~/ds41_work/ds41ready_data"))
ap.add_argument("--out", default=os.path.expanduser("~/ds41_work/ds41chunk_out")); ap.add_argument("--new", type=int, default=16)
ap.add_argument("--blocking", default="1,0", help="1 = row blocking on, 0 = off (baseline); one arm per value")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
from ds41_serving import build_model
def mem(): return int([l for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0].split()[1]) / 1048576
m = build_model(a.pack, device="cuda:0", dtype="float16", layers=None, mtp=True, moe="auto", prefill_chunk=4096, engram_dir=a.engram_dir)
blk = [b for b in m.pack_modules[0].modules if getattr(b, "tier", None) is not None]
store = None
if blk:
    from exllamav3.modules.expert_store import get_store
    store = get_store(blk[0].config)
rows = [np.load(os.path.join(a.data, f"{n}.npy")).astype(np.int64) for n in ("long", "short")]
pool = np.concatenate([rows[0].reshape(-1), rows[1].reshape(-1)])
big = (1 << 30)
dflt = (m.pf_attn_rows, m.pf_eng_rows, m.pf_moe_rows, m.idx_score_bytes)
for L in [int(x) for x in a.lens.split(",")]:
    ids = torch.tensor([pool[:L].tolist()], device="cuda:0")
    for bl in a.blocking.split(","):
        m.pf_attn_rows, m.pf_eng_rows, m.pf_moe_rows, m.idx_score_bytes = dflt if bl == "1" else (big, big, big, 1 << 62)
        for ch in [int(x) for x in a.chunks.split(",")]:
            os.environ["EXL3_TIER_PF_CHUNK"] = str(ch)
            torch.cuda.empty_cache(); st = m.new_state(1, L + a.new + 8); torch.cuda.synchronize()
            base = torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
            if store: store.reset_stats()
            minm = mem(); t = time.perf_counter()
            try:
                lg, _ = m.prefill(ids, st, 0); torch.cuda.synchronize()
            except Exception as e:
                print(json.dumps({"L": L, "chunk": ch, "blocking": bl, "error": str(e)[:200]}), flush=True); del st; continue
            dt = time.perf_counter() - t
            peak = (torch.cuda.max_memory_allocated() - base) / 2**30
            s = store.stats() if store else {}
            toks, pos = [], L
            lg = lg.float(); np.save(os.path.join(a.out, f"logits_{L}_{ch}_{bl}.npy"), lg.cpu().numpy())
            for _ in range(a.new):
                tk = int(lg[0].argmax()); toks.append(tk)
                lg, _ = m._forward(torch.tensor([[tk]]), pos, st, None, "last"); lg = lg.float(); pos += 1
            print(json.dumps({"L": L, "chunk": ch, "blocking": bl, "tok_s": round(L / dt, 1), "s": round(dt, 1),
                              "peak_gib": round(peak, 2), "read_gb": round(s.get("bytes", 0) / 1e9, 1), "hit": s.get("hit_rate"),
                              "min_memavail": round(min(minm, mem()), 1), "toks": toks}), flush=True)
            del st
