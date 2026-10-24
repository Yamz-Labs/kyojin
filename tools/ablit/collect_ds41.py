#!/usr/bin/env python3
"""collect_ds41.py -- refusal-direction collection on the DS4.1 FP8/FP4 SOURCE, one layer at a time.

Same stream as the converter (tools/ds41/capture.py): StreamCalibrator rows, ranged fetch of one layer into a
cache (put it in /dev/shm: no SSD writes), exact reference forward (sink=None, commit=True). After each layer the
mHC streams of every prompt are averaged (hc_mean readout, as arch.py "deepseek_v41") and the last-token and
mean-of-last-4-tokens vectors are written to <out>/readouts/L<NN>.npz. `--finish` turns the readouts into
directions.safetensors + manifest.json with the unchanged collect_dirs stats (same keys as GLM / MiMo).

Resumable: the row state is saved to --state after every layer (tmp + rename); a restart continues at the next layer.

  # one step (the driver runs it under `gpurun`): layers LO..HI
  python collect_ds41.py --map M --config C --tokenizer T --engram-dir E --cache /dev/shm/x/cache --state /dev/shm/x/state \
      --out OUT --harmful H.jsonl --harmless S.jsonl --layers 0-1 --device cuda:0
  # aggregate
  python collect_ds41.py --finish --out OUT
"""
import argparse
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "ds41"))
import arch as arch_mod  # noqa: E402

LOG = lambda m: print(f"[{time.strftime('%H:%M:%S')}] [collect_ds41] {m}", flush=True)


def read_prompts(path, limit):
    import collect_dirs as C
    return C.load_prompts(path, limit)


def encode_rows(tok_path, prompts, max_tokens):
    """-> (list of [1, n] int64 tensors, kept prompt indices). A prompt whose rendered row exceeds max_tokens is skipped
    (the sliding window is 128; the compressed-attention path is exercised by the converter's own rows)."""
    import torch
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(tok_path)
    arch = arch_mod.get("deepseek_v41")
    rows, kept = [], []
    for i, p in enumerate(prompts):
        ids = tk.encode(arch_mod.chat_wrap(p, arch), add_special_tokens=False).ids
        if len(ids) <= max_tokens:
            rows.append(torch.tensor(ids, dtype=torch.int64).view(1, -1))
            kept.append(i)
    return rows, kept


def step(a):
    import numpy as np
    import torch
    import capture as CAP
    from exllamav3.dsv41.config import DSV41Config
    from exllamav3.dsv41.stream_calib import StreamCalibrator, SourceLayerWeights, EngramSupport
    from exllamav3.dsv41.safetensors_io import SafetensorsIndex, engram_store_from_index
    from exllamav3.dsv41.engram import build_compressed_token_map
    import fetch as F

    torch.set_grad_enabled(False)
    if a.threads:
        torch.set_num_threads(a.threads)
    cfg = DSV41Config.from_file(a.config)
    arch = arch_mod.get("deepseek_v41")
    assert cfg.num_hidden_layers == arch["n_layers"] and cfg.hidden_size == arch["hidden"]
    tmap = json.load(open(a.map))
    os.makedirs(a.cache, exist_ok=True)
    os.makedirs(a.state, exist_ok=True)
    os.makedirs(os.path.join(a.out, "readouts"), exist_ok=True)
    meta_p = os.path.join(a.out, "rows.json")
    harm = read_prompts(a.harmful, a.limit)
    safe = read_prompts(a.harmless, a.limit)
    prompts = harm + safe
    rows, kept = encode_rows(a.tokenizer, prompts, a.max_tokens)
    label = np.array([0 if k < len(harm) else 1 for k in kept], dtype=np.int8)  # 0 harmful, 1 harmless
    json.dump({"n_harm": int((label == 0).sum()), "n_safe": int((label == 1).sum()), "n_prompts": len(prompts),
               "kept": kept, "max_tokens": a.max_tokens,
               "tokens": int(sum(r.shape[1] for r in rows))}, open(meta_p, "w"))
    np.save(os.path.join(a.out, "labels.npy"), label)
    LOG(f"{len(rows)} rows ({int((label == 0).sum())} harmful, {int((label == 1).sum())} harmless), "
        f"{sum(r.shape[1] for r in rows)} tokens")

    fetcher = F.TensorFetcher(tmap, cache=a.cache)
    src = CAP.make_source(fetcher, cfg.num_hidden_layers, a.device)
    import glob
    files = sorted(glob.glob(os.path.join(a.engram_dir, "*.safetensors")))
    store = engram_store_from_index(SafetensorsIndex(files), cfg.engram_layer_ids, allow_buffered=True)
    from engram_subset import token_map_from_file
    tm, _ = token_map_from_file(a.tokenizer)
    engram = EngramSupport(cfg, tm, store)
    embed = src.tensor("embed", torch.float32, "cpu")
    cal = StreamCalibrator(cfg, a.device, embed, engram, state_device=a.device, min_expert_tokens=0)
    del embed
    cal.start(rows)

    lo, _, hi = a.layers.partition("-")
    lo, hi = int(lo), int(hi or lo)
    cp = os.path.join(a.state, "state.pt")
    start = 0
    if os.path.exists(cp):
        st = torch.load(cp, weights_only=False)
        def mv(o):
            if torch.is_tensor(o):
                return o.to(a.device)
            if isinstance(o, dict):
                return {k: mv(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return type(o)(mv(v) for v in o)
            return o
        cal.h, cal.ho, start = mv(st["h"]), mv(st["ho"]), st["next_layer"]
        del st
        LOG(f"resumed row state at layer {start}")
    if start > lo:
        LOG(f"layers < {start} already done")
        lo = start
    if lo > start:
        raise SystemExit(f"state is at layer {start}, asked to start at {lo}")

    from concurrent.futures import ThreadPoolExecutor
    pool, pf = ThreadPoolExecutor(1), {}

    def ensure_prefetch(l):
        for j in range(l, min(hi, l + a.lookahead) + 1):
            if j not in pf:
                pf[j] = pool.submit(CAP.prefetch_layer, fetcher, tmap, f"layers.{j}", a.fetch_threads, LOG)

    t_start = time.time()
    done = 0
    for l in range(lo, hi + 1):
        t0 = time.time()
        if a.external_prefetch:
            # prefetch_ds41.py fills the cache outside the GPU window and drops <cache>/layers.N.ready when layer N is complete
            if not os.path.exists(os.path.join(a.cache, f"layers.{l}.ready")):
                LOG(f"layer {l} not fetched yet: leaving the window")
                break
            if done and time.time() - t_start > a.budget_s:
                LOG(f"time budget {a.budget_s}s used: leaving the window")
                break
        else:
            ensure_prefetch(l)
            pf[l].result()
            pf.pop(l)
        t1 = time.time()
        W = SourceLayerWeights(src, l, a.device)
        cal.run_layer(l, W, None, commit=True)
        if str(a.device).startswith("cuda"):
            torch.cuda.synchronize()
        last, last4 = [], []
        for h in cal.h:
            v = h.float().mean(dim=2)[0]                       # [seq, hidden]: mean over the 4 mHC streams
            assert torch.isfinite(v).all(), f"non finite state at layer {l}"
            last.append(v[-1].cpu().numpy())
            last4.append(v[-min(4, v.shape[0]):].mean(0).cpu().numpy())
        np.savez(os.path.join(a.out, "readouts", f"L{l:02d}.npz"), last=np.stack(last), last4=np.stack(last4))
        st = {"h": cal.h, "ho": cal.ho, "next_layer": l + 1}
        torch.save(st, cp + ".tmp")
        os.replace(cp + ".tmp", cp)
        if not a.keep_cache:
            shutil.rmtree(os.path.join(a.cache, f"layers.{l}"), ignore_errors=True)
            try:
                os.remove(os.path.join(a.cache, f"layers.{l}.ready"))
            except OSError:
                pass
        open(os.path.join(a.out, "next_layer.txt"), "w").write(str(l + 1))
        done += 1
        nrm = float(np.linalg.norm(np.stack(last), axis=1).mean())
        LOG(f"layer {l}: fetch {t1 - t0:.0f}s compute {time.time() - t1:.0f}s mean |h| {nrm:.1f}")
        del W


def finish(a):
    import numpy as np
    import collect_dirs as C
    arch = arch_mod.get("deepseek_v41")
    n = arch["n_layers"]
    meta = json.load(open(os.path.join(a.out, "rows.json")))
    label = np.load(os.path.join(a.out, "labels.npy"))
    hh, hs, hh4, hs4 = [], [], [], []
    for l in range(n):
        z = np.load(os.path.join(a.out, "readouts", f"L{l:02d}.npz"))
        hh.append(z["last"][label == 0]); hs.append(z["last"][label == 1])
        hh4.append(z["last4"][label == 0]); hs4.append(z["last4"][label == 1])
    stats = C.accumulate_stats(hh, hs, hh4, hs4)
    C.save_dirs(stats, a.out, {"mode": "real", "arch": "deepseek_v41", "model": "DeepSeek-V4.1-Flash-1017 (source, fp8/fp4)",
                               "n_harm": meta["n_harm"], "n_safe": meta["n_safe"], "max_tokens": meta["max_tokens"],
                               "readout": "mean over 4 mHC streams, output of block L (before the next block)",
                               "chat": "ds41_chat.render_chat thinking=on effort=high",
                               "harmful": a.harmful, "harmless": a.harmless})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--finish", action="store_true")
    ap.add_argument("--map")
    ap.add_argument("--config")
    ap.add_argument("--tokenizer")
    ap.add_argument("--engram-dir")
    ap.add_argument("--cache")
    ap.add_argument("--state")
    ap.add_argument("--out", required=True)
    ap.add_argument("--harmful")
    ap.add_argument("--harmless")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--max-tokens", type=int, default=110)
    ap.add_argument("--layers", default="0-39")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fetch-threads", type=int, default=6)
    ap.add_argument("--lookahead", type=int, default=1, help="layers fetched ahead of the one being computed")
    ap.add_argument("--external-prefetch", action="store_true", help="layers are fetched by prefetch_ds41.py (ready markers)")
    ap.add_argument("--budget-s", type=int, default=600, help="external-prefetch: no new layer after this many seconds")
    ap.add_argument("--keep-cache", action="store_true", help="smoke tests: leave the fetched layer for the real run")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    a = ap.parse_args()
    finish(a) if a.finish else step(a)
