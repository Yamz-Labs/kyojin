import os, sys, time, numpy as np, torch
from exllamav3.dsv41.engram import DirectRowReader
from exllamav3.modules.dsv41_fn.engram_io import ParallelRowReader
d = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/models/ds41-1017/engram_fp8")
from exllamav3.dsv41.engram import detect_engram_format
if detect_engram_format(d) == "safetensors":          # official shards, read in place
    from exllamav3.dsv41.safetensors_io import SafetensorsIndex
    ix = SafetensorsIndex(d)
    (fw, _, sw, ow, _), (fs, _, ss, os_, _) = ix.info("layers.1.engram.embed.weight"), ix.info("layers.1.engram.embed.scale")
    n, segs = sw[0], [(fw, ow, sw[1], sw[1]), (fs, os_, ss[1], ss[1])]
else:
    w, s = f"{d}/layers.1.engram.embed.weight.bin", f"{d}/layers.1.engram.embed.scale.bin"
    n, segs = os.path.getsize(s) // 8, [(w, 0, 128, 128), (s, 0, 8, 8)]
print("dir", d, "rows", n, "segs", segs)
rng = np.random.default_rng(1)
for label, mk in (("serial DirectRowReader", lambda: DirectRowReader(segs)),) + tuple((f"parallel x{t}", (lambda t=t: ParallelRowReader(segs, threads=t))) for t in (4, 16, 32)):
    r = mk(); print(label, "direct", r.direct)
    for cnt in (24, 2048):
        rows = rng.integers(0, n, cnt).tolist()
        t0 = time.time(); reps = 20 if cnt == 24 else 3
        for _ in range(reps): r.read_rows(rows)
        dt = (time.time() - t0) / reps
        print(f"  {cnt:5d} rows: {dt*1e3:8.2f} ms  ({dt/cnt*1e6:6.1f} us/row)")
    r.close()
from exllamav3.modules.dsv41_fn.engram_io import NativeRowReader
for t in (16, 32, 64):
    r = NativeRowReader(segs, threads=t); print("native x%d" % t, r.direct)
    for cnt in (24, 2048, 16384):
        rows = np.sort(rng.integers(0, n, cnt)).tolist()
        t0 = time.time(); reps = 20 if cnt == 24 else 3
        for _ in range(reps): r.read_rows(rows)
        dt = (time.time() - t0) / reps
        print(f"  {cnt:5d} rows: {dt*1e3:8.2f} ms  ({dt/cnt*1e6:6.1f} us/row)")
    r.close()
