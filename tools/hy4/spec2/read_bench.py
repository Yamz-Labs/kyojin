"""CPU-only: cost of the Engram row reads of one verify round (4 layers x T rows x 16 columns of 99-byte rows from the 65 GB table in the GGUF), cold random rows,
per strategy. Run from the tree root with the engine venv; no GPU."""
import os, sys, time, types, statistics
import numpy as np
tree = os.getcwd()
for n, sub in (("exllamav3", ""), ("exllamav3.loader", "loader")):
    m = types.ModuleType(n); m.__path__ = [os.path.join(tree, "exllamav3", sub)]; sys.modules[n] = m
from concurrent.futures import ThreadPoolExecutor
from exllamav3.loader.gguf_collection import GGUFCollection
from exllamav3.loader.engram_rows import F8Rows
from exllamav3.loader import engram_stage as ES
g = GGUFCollection("/path/to/models/hy4-flash-gguf/Hy4-flash-Q3_K-00001-of-00002.gguf")
tab = F8Rows(g)
print("table rows", tab.rows, "row_bytes", tab.row_bytes, flush=True)
nE, TM, nc, T = 4, 8, 16, int(os.environ.get("T", "4"))
rng = np.random.default_rng(5)
mods = [types.SimpleNamespace(e=i, n_gram=3, n_head=8, n_cols=16, token_map=[0, 1], pad_id=0, mult=[1, 2, 3], primes=[5] * 16, offs=[0] * 16) for i in range(nE)]
rnp = np.zeros((nE, TM, nc, tab.row_bytes), dtype=np.uint8); vnp = np.zeros((nE, TM, nc), dtype=np.float32)
def q(v, p): s = sorted(v); return s[min(len(s) - 1, int(p * len(s)))]
for name, workers, pool_n, inline in (("pool64 old-like (chunks of 4)", 64, 64, 0), ("4 workers", 4, 8, 0), ("8 workers", 8, 8, 0), ("16 workers", 16, 16, 0), ("32 workers", 32, 32, 0), ("inline", 1, 0, 10 ** 9)):
    pool = ThreadPoolExecutor(max_workers=pool_n) if pool_n else None
    st = ES.EngramStager(mods, tab, rnp, vnp, pool=pool, workers=workers)
    st.inline_max = inline
    ts = []
    for r in range(int(os.environ.get("ROUNDS", "150"))):
        ids = rng.integers(0, tab.rows, size=(nE, T, nc))
        t = time.perf_counter()
        for f in st.submit(ids, np.ones(T, dtype=np.float32), 0):
            f.result()
        ts.append((time.perf_counter() - t) * 1000)
    print(f"{name:32s} cold random rows: med {statistics.median(ts):.2f} p90 {q(ts, .9):.2f} max {max(ts):.2f} ms", flush=True)
    ids = rng.integers(0, tab.rows, size=(nE, T, nc))
    st.submit(ids, np.ones(T, dtype=np.float32), 0)
    ts = []
    for r in range(50):
        t = time.perf_counter()
        for f in st.submit(ids, np.ones(T, dtype=np.float32), 0):
            f.result()
        ts.append((time.perf_counter() - t) * 1000)
    print(f"{name:32s} warm same rows:   med {statistics.median(ts):.2f} p90 {q(ts, .9):.2f} max {max(ts):.2f} ms", flush=True)
