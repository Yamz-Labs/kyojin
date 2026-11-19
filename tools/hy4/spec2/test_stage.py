"""CPU unit + bounds tests of the vectorised Engram staging (exllamav3/loader/engram_stage.py) and the bucket arithmetic of step_graph. No GPU, no model.
Run: python -I tools/hy4/spec2/test_stage.py   (from the tree root; needs torch + numpy only)"""
import importlib.util, os, random, sys, tempfile, types
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
def load(name, fn):
    sp = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "exllamav3", "loader", fn))
    m = importlib.util.module_from_spec(sp); sys.modules[name] = m; sp.loader.exec_module(m); return m
ER = load("engram_rows_t", "engram_rows.py")
ES = load("engram_stage_t", "engram_stage.py")

NG, NH = 3, 8
NC = (NG - 1) * NH
def make_mods(rng, nE, vocab, rows):
    tm = [rng.choice([-1, rng.randrange(0, 5000)]) if rng.random() < 0.1 else rng.randrange(0, 5000) for _ in range(vocab)]
    pad_id = next(i for i, x in enumerate(tm) if x != -1)
    mods = []
    for e in range(nE):
        primes = [rng.randrange(rows // 40, rows // 20) | 1 for _ in range(NC)]
        offs = [0] * NC
        acc = 0
        for c in range(NC):                       # disjoint id ranges like the real head offsets
            offs[c] = acc; acc += primes[c]
        mods.append(types.SimpleNamespace(e=e, n_gram=NG, n_head=NH, n_cols=NC, token_map=tm, pad_id=pad_id,
                                          mult=[rng.getrandbits(64) | 1 for _ in range(NG)], primes=primes, offs=offs))
    return mods, acc

def make_table(path, rows, rb, rng, shard_off=4096):
    data = np.frombuffer(bytes(rng.getrandbits(8) for _ in range(rows * rb)), dtype=np.uint8)
    with open(path, "wb") as f:
        f.write(b"\xAA" * shard_off); f.write(data.tobytes()); f.write(b"\xBB" * 4096)       # bytes around the table: an off-by-one read shows up
    fd = os.open(path, os.O_RDONLY)
    tab = types.SimpleNamespace(off=(0 << 48) | 0, rows=rows, row_bytes=rb, name="synthetic", g=types.SimpleNamespace(fds=[fd], bases=[shard_off]))
    return tab, data.reshape(rows, rb)

def test_hash_matches_reference():
    rng = random.Random(1)
    rows = 20000
    mods, tot = make_mods(rng, 4, 4000, tot_rows := 10 ** 6)
    tab = types.SimpleNamespace(off=0, rows=tot, row_bytes=8, name="x", g=types.SimpleNamespace(fds=[0], bases=[0]))
    TM = 8
    rnp = np.zeros((4, TM, NC, 8), dtype=np.uint8); vnp = np.zeros((4, TM, NC), dtype=np.float32)
    st = ES.EngramStager(mods, tab, rnp, vnp, pool=None)
    for trial in range(300):
        n = rng.randrange(1, 60)
        toks = [rng.choice([rng.randrange(0, 4000), rng.randrange(3990, 4100), -1]) for _ in range(n)]     # includes ids past the token map
        T = rng.randrange(1, TM + 1)
        start = rng.randrange(0, n)
        hist = toks[:start] + [rng.randrange(0, 4000) for _ in range(T)]
        t0 = rng.randrange(0, T)
        ids, valid = st.hash(hist, start, t0, T)
        for e, m in enumerate(mods):
            ref, rv = ER.engram_hash(hist, start + t0, T - t0, m.token_map, m.pad_id, m.mult, m.primes, m.offs, NG, NH)
            assert np.array_equal(ids[e], ref.numpy()), (trial, e)
            assert np.array_equal(np.broadcast_to(valid[:, None], rv.shape), rv.numpy()), (trial, e)
        # the history passed as tokens + extra gives the same hash
        k = rng.randrange(0, min(T, 3) + 1)
        ids2, valid2 = st.hash(hist[:len(hist) - k], start, t0, T, extra=hist[len(hist) - k:]) if start + T - k >= start else (ids, valid)
        assert np.array_equal(ids, ids2) and np.array_equal(valid, valid2)
    print("hash == engram_hash: ok")

def run_reads(workers, inline):
    rng = random.Random(7)
    nE, rb, TM = 4, 99, 8
    mods, tot = make_mods(rng, nE, 3000, 400000)
    with tempfile.TemporaryDirectory() as d:
        tab, data = make_table(os.path.join(d, "t.bin"), tot, rb, rng)
        pool = ThreadPoolExecutor(max_workers=8)
        for trial in range(40):
            rnp = np.full((nE, TM, NC, rb), 0xCD, dtype=np.uint8); vnp = np.full((nE, TM, NC), -7.0, dtype=np.float32)
            st = ES.EngramStager(mods, tab, rnp, vnp, pool=pool, workers=workers)
            st.inline_max = inline
            n = rng.randrange(3, 50)
            hist = [rng.randrange(0, 3000) for _ in range(n)]
            T = rng.randrange(1, TM + 1); start = n - 1; hist = hist + [rng.randrange(0, 3000) for _ in range(T)]
            t0 = rng.randrange(0, T)
            for f in st.stage(hist, start, t0, T):
                f.result()
            ids, valid = st.hash(hist, start, t0, T)
            for e in range(nE):
                for t in range(T - t0):
                    assert np.array_equal(rnp[e, t0 + t], data[ids[e, t]]), (trial, e, t)
                    assert (vnp[e, t0 + t] == valid[t]).all()
            assert (rnp[:, :t0] == 0xCD).all() and (rnp[:, T:] == 0xCD).all()          # canary: rows outside the staged range are untouched
            assert (vnp[:, :t0] == -7.0).all() and (vnp[:, T:] == -7.0).all()
    print(f"reads == table rows (workers {workers}, inline {inline}): ok")

def test_reads():
    run_reads(1, 0); run_reads(4, 0); run_reads(4, 10 ** 9); run_reads(8, 0)

def test_bounds():
    rng = random.Random(3)
    nE, rb, TM = 2, 16, 4
    mods, tot = make_mods(rng, nE, 500, 100000)
    with tempfile.TemporaryDirectory() as d:
        tab, data = make_table(os.path.join(d, "t.bin"), tot, rb, rng)
        rnp = np.full((nE, TM, NC, rb), 0xCD, dtype=np.uint8); vnp = np.zeros((nE, TM, NC), dtype=np.float32)
        st = ES.EngramStager(mods, tab, rnp, vnp, pool=None)
        hist = [rng.randrange(0, 500) for _ in range(20)]
        # a table one row too short: some id is past the end -> IndexError and NO byte staged
        small = types.SimpleNamespace(**{**tab.__dict__, "rows": 5000})
        st2 = ES.EngramStager(mods, small, rnp, vnp, pool=None)
        try:
            st2.stage(hist, 10, 0, 4); raise SystemExit("bounds: out-of-range id was not caught")
        except IndexError:
            pass
        assert (rnp == 0xCD).all(), "bounds: bytes staged before the check"
        # more rows than the staging buffer
        try:
            st.stage(hist + [1] * 8, 10, 2, 10); raise SystemExit("bounds: T past TM not caught")
        except (IndexError, ValueError):
            pass
        # history too short for the window
        try:
            st.hash(hist[:5], 10, 0, 2); raise SystemExit("bounds: short history not caught")
        except IndexError:
            pass
        # last row of the table: read exactly rb bytes, bytes after it never staged
        ids = np.full((nE, 1, NC), tot - 1, dtype=np.int64)
        st.submit(ids, np.ones(1, dtype=np.float32), 0)
        assert (rnp[:, 0] == data[tot - 1]).all() and (rnp[:, 1:] == 0xCD).all()
    print("bounds: ok")

def test_bucket():
    for n, sub in (("exllamav3", ""), ("exllamav3.loader", "loader")):
        m = types.ModuleType(n); m.__path__ = [os.path.join(ROOT, "exllamav3", sub)]; sys.modules[n] = m
    from exllamav3.loader import step_graph as SG
    B = SG.BUCKET
    gd = types.SimpleNamespace(cap=8 * B)
    f = lambda start, T: SG.GraphDecoder.bucket(gd, start, T)
    assert f(0, 1) == B and f(B - 1, 1) == B and f(B, 1) == 2 * B and f(B - 2, 4) == 2 * B and f(8 * B - 2, 8) == 8 * B
    for start in range(0, 8 * B - 8, 37):
        for T in range(1, SG.TM + 1):
            Nb = f(start, T)
            assert Nb % B == 0 and start + T <= Nb <= gd.cap           # the bucket covers the last row of the step and stays inside the static buffers
    print("bucket: ok")

if __name__ == "__main__":
    test_hash_matches_reference(); test_reads(); test_bounds(); test_bucket()
    print("ALL_OK")
