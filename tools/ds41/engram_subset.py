"""Extract the Engram rows a calibration set touches into one small file (so the capture machine needs no 98 GB table).

    python engram_subset.py --calib calib.npy --rows 128 --tokenizer tokenizer.json --config config.json \
        --engram-dir ~/models/ds41-1017/engram_mxfp4 --out engram_subset.pt

The capture (capture.py) loads the file as a `SubsetEngramStore`. Same hasher, same rows as the capture.
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch


class _Tok:
    def __init__(self, path):
        from tokenizers import Tokenizer
        self.backend_tokenizer = Tokenizer.from_file(path)

    def __len__(self):
        return self.backend_tokenizer.get_vocab_size(with_added_tokens=True)


def token_map_from_file(path):
    from exllamav3.dsv41.engram import build_compressed_token_map
    return build_compressed_token_map(_Tok(path))


def hashed_rows(cfg, token_map, rows_ids):
    """-> per engram layer position: sorted unique table rows touched by the calibration rows."""
    from exllamav3.dsv41.engram import EngramLayout, NgramHasher, compute_multipliers
    nv = cfg.engram_compressed_vocab_size or int(max(token_map)) + 1
    layout = EngramLayout.build(cfg.engram_layer_ids, cfg.engram_num_embeddings, cfg.engram_max_ngram_size,
                                cfg.engram_vocab_size, cfg.engram_n_heads, cfg.engram_head_dim, None)
    mult = compute_multipliers(layout.layer_ids, layout.max_ngram_size, nv)
    per = [[] for _ in layout.layer_ids]
    for ids in rows_ids:
        h = NgramHasher(layout, token_map, cfg.engram_pad_token_id, nv, mult, 1, ids.shape[1], "cpu")(ids, 0)
        for p in range(len(layout.layer_ids)):
            per[p].append(torch.unique(h[0, :, p, :].reshape(-1)))
    return [torch.unique(torch.cat(x)) for x in per]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True)
    ap.add_argument("--rows", type=int, default=128)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--engram-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    from exllamav3.dsv41.config import DSV41Config
    from exllamav3.dsv41.engram import RawBinEngramStore
    cfg = DSV41Config.from_file(a.config)
    tm, _ = token_map_from_file(a.tokenizer)
    arr = np.load(a.calib)[:a.rows].astype(np.int64)
    rows_ids = [torch.from_numpy(r).view(1, -1) for r in arr]
    t0 = time.time()
    uniq = hashed_rows(cfg, tm, rows_ids)
    print(f"hashed {len(rows_ids)} rows -> unique rows per layer {[int(u.numel()) for u in uniq]} ({time.time()-t0:.0f}s)", flush=True)
    store = RawBinEngramStore.open(a.engram_dir, cfg.engram_layer_ids, cfg.engram_head_dim, "auto")
    os.makedirs(a.out, exist_ok=True)
    import threading
    tl = threading.local()
    for p, u in enumerate(uniq):
        n = u.numel()
        np.save(os.path.join(a.out, f"rows.{p}.npy"), u.numpy())
        chunk = 4096
        parts = [(i, min(n, i + chunk)) for i in range(0, n, chunk)]

        def rd(ab):
            a_, b_ = ab
            # DirectRowReader owns one aligned scratch buffer: one reader per thread (a shared one corrupts rows)
            st = getattr(tl, "store", None)
            if st is None:
                st = tl.store = RawBinEngramStore.open(a.engram_dir, cfg.engram_layer_ids, cfg.engram_head_dim, "auto")
            return st.readers[p].read_rows(u[a_:b_].tolist()).numpy().tobytes()
        t1 = time.time()
        with open(os.path.join(a.out, f"data.{p}.bin.tmp"), "wb") as f, ThreadPoolExecutor(a.threads) as ex:
            futs = []
            for k, ab in enumerate(parts):
                futs.append(ex.submit(rd, ab))
                while len(futs) > a.threads * 3:       # bounded read-ahead (memory)
                    f.write(futs.pop(0).result())
                if k % 200 == 0:
                    print(f"  layer pos {p}: {k}/{len(parts)} chunks {time.time()-t1:.0f}s", flush=True)
            for fu in futs:
                f.write(fu.result())
        os.replace(os.path.join(a.out, f"data.{p}.bin.tmp"), os.path.join(a.out, f"data.{p}.bin"))
        print(f"layer pos {p}: {n} rows read in {time.time()-t1:.0f}s", flush=True)
    store.close()
    open(os.path.join(a.out, "DONE"), "w").write(f"{a.rows}\n")
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
