import os, mmap, time, random, sys, glob
from concurrent.futures import ThreadPoolExecutor
shards = sorted(glob.glob(os.path.expanduser("~/models/ds41-1017-yamz/*.safetensors")), key=os.path.getsize)[-1:]
f = shards[0]; size = os.path.getsize(f); fd = os.open(f, os.O_RDONLY | os.O_DIRECT)
EXP = 12 * 1024 * 1024
def run(m, chunk, nth, reps=12):
    bufs = [mmap.mmap(-1, EXP) for _ in range(m)]
    ex = ThreadPoolExecutor(nth); lat = []
    for _ in range(reps):
        offs = [(random.randrange(0, size - EXP) // 4096) * 4096 for _ in range(m)]
        t = time.perf_counter(); futs = []
        for b, o in zip(bufs, offs):
            for c in range(0, EXP, chunk):
                futs.append(ex.submit(lambda b=b, o=o, c=c: os.preadv(fd, [memoryview(b)[c:c + chunk]], o + c)))
        for x in futs: x.result()
        lat.append(time.perf_counter() - t)
    ex.shutdown(); lat.sort(); med = lat[len(lat) // 2]
    return med, m * EXP / med / 1e9
for m in (1, 6, 24):
    for nth in (8, 16):
        for ch in (256 << 10, 512 << 10, 1 << 20, 4 << 20):
            med, gb = run(m, ch, nth)
            print(f"experts/batch {m:2d} threads {nth:2d} chunk {ch>>10:5d} KiB  median {med*1e3:7.1f} ms  {gb:5.2f} GB/s", flush=True)
