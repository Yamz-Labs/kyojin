"""Tile sweep of triton_iq._gemv_iq at R1 on the largest real matrix of each IQ type in the two Qwen3-8B UD GGUFs (BN x num_warps).
BN / NW are read at import: run once per config via EXL3_IQ_BN / EXL3_IQ_NW in a subprocess. usage: iq_tiles.py [one]"""
import os, sys, subprocess, time
import torch

if len(sys.argv) > 1 and sys.argv[1] == "one":
    from exllamav3.loader.gguf_collection import GGUFCollection
    from exllamav3.foreign_quant import triton_iq as TI
    TY = {18: "iq3_xxs", 21: "iq3_s", 22: "iq2_s"}
    best = {}
    for m in ("iq3xxs", "iq2m"):
        g = GGUFCollection(f"/home/yaniss/models/foreign/q3_8b_{m}_cfg/model.gguf")
        for k, (ty, dims, off) in g.infos.items():
            f = TY.get(ty)
            if f and len(dims) == 2 and dims[1] * dims[0] > best.get(f, (0,))[0]:
                best[f] = (dims[1] * dims[0], m, k)
    out = []
    for f, (_, m, k) in sorted(best.items()):
        g = GGUFCollection(f"/home/yaniss/models/foreign/q3_8b_{m}_cfg/model.gguf")
        ty, dims, _ = g.infos[k]
        raw = g.get_tensor(k, device="cuda").reshape(dims[1], -1).contiguous()
        x = torch.randn((1, dims[0]), device="cuda").half()
        for _ in range(5): TI.gemv(f, x, raw)
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(60): TI.gemv(f, x, raw)
        torch.cuda.synchronize()
        us = (time.perf_counter() - t) / 60 * 1e6
        out.append(f"{f}:{raw.numel() / us / 1e3:.1f}")
    print("RES", " ".join(out), flush=True)
    sys.exit(0)

res = {}
for bn in (8, 16, 32, 64):
    for nw in (1, 2, 4):
        env = dict(os.environ, EXL3_IQ_BN=str(bn), EXL3_IQ_NW=str(nw))
        r = subprocess.run([sys.executable, "-u", __file__, "one"], env=env, capture_output=True, text=True, timeout=300)
        line = [l for l in r.stdout.splitlines() if l.startswith("RES")]
        if not line:
            print(bn, nw, "FAIL", (r.stderr or "")[-200:].replace("\n", " "), flush=True)
            continue
        for kv in line[0].split()[1:]:
            f, v = kv.split(":")
            res.setdefault(f, []).append((float(v), bn, nw))
        print(bn, nw, line[0], flush=True)
for f, v in sorted(res.items()):
    v.sort(reverse=True)
    cur = [x for x in v if x[1] == 16 and x[2] == 2]
    print(f"{f:8s} best {v[:3]}  current (16, 2) -> {cur[0][0] if cur else 'n/a'} GB/s")
