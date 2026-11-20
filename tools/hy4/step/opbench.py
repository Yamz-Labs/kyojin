"""opbench.py: per-op decode microbench, no server boot, shipped kernel flags, real GGUF tensor bytes (only the tensors of the op are read).

  opbench.py OP [--T 1,2,4,8] [--ctx 1024] [--check] [--no-graph] [--gguf PATH] [--out DIR] [--layers N] [--nrep 200] [--tmax S] [--force]
  OPs: roofs dense_q8 experts router absorb mla_rows head draft_head info all

Method: one CYCLE of calls over distinct per-layer weights / KV (total far above the 32 MB MALL), captured in a HIP graph and replayed;
reported ms = median graph replay / calls per cycle. step_ms = ms per call x calls per decode step (compare with OP_LEDGER_STRIX.md).
pct_roof = GB/s achieved (weight bytes read once per call) / measured read roof of the same format (out/roofs.json from `opbench.py roofs`).
Point cache: every line is keyed by engine sha + flag hash + op + T + ctx + variant; an existing key is skipped unless --force.
Run inside the engine env wrapper (extension .so, HY4_EXT_DIR etc. as for the server) under the GPU locks. The shipped flag set is read from serve.SHIPPED_ENV."""
import os, sys, ast, json, time, math, hashlib, argparse, statistics, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
T_START = time.time()


def shipped_env():
    src = open(os.path.join(TREE, "tools", "hy4", "serve.py")).read()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "SHIPPED_ENV" for t in n.targets):
            return ast.literal_eval(n.value)
    raise RuntimeError("SHIPPED_ENV not found in serve.py")


ENV = shipped_env()
for _k, _v in ENV.items():
    os.environ.setdefault(_k, _v)
FLAGHASH = hashlib.sha1(json.dumps({k: os.environ.get(k) for k in sorted(ENV)}, sort_keys=True).encode()).hexdigest()[:8]

sys.path.insert(0, os.path.join(TREE, "tools", "hy4"))
import numpy as np
import torch
import engine as _eng
_eng._stub_package(TREE)
from exllamav3.loader import hy4_forward as HF, mla_bgemv as BG, router_gemv as RG, head_bf16 as HB, ref_fused as RF
from exllamav3.loader.gguf_collection import GGUFCollection

dev = torch.device("cuda")
GiB = 2 ** 30


def sha():
    try:
        s = subprocess.check_output(["git", "-C", TREE, "rev-parse", "--short=9", "HEAD"], text=True).strip()
        d = subprocess.check_output(["git", "-C", TREE, "diff", "HEAD", "--", "exllamav3", "tools"], text=True)
        return s + ("+" + hashlib.sha1(d.encode()).hexdigest()[:6] if d.strip() else "")
    except Exception:       # noqa: BLE001
        return "nogit"


SHA = sha()


# ---------------------------------------------------------------------------------------------------------------- timing
def bench(fns, nrep=200, budget=6.0, graph=True):
    """fns = one cycle of zero-arg callables. -> (ms per replay median, mode, n, p10, p90). Graph replay when capture works, else eager."""
    for f in fns:
        f()
    torch.cuda.synchronize()
    mode, run = "eager", (lambda: [f() for f in fns])
    if graph:
        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for f in fns:
                    f()
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for f in fns:
                    f()
            mode, run = "graph", g.replay
        except Exception as e:      # noqa: BLE001
            print(f"  (graph capture failed: {type(e).__name__} {str(e)[:70]}; eager)", flush=True)
            torch.cuda.synchronize()
    for _ in range(5):
        run()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts, t0 = [], time.time()
    while len(ts) < nrep and (time.time() - t0 < budget or len(ts) < 30):
        a.record(); run(); b.record(); b.synchronize()
        ts.append(a.elapsed_time(b))
    ts.sort()
    return statistics.median(ts), mode, len(ts), ts[len(ts) // 10], ts[(9 * len(ts)) // 10]


# ---------------------------------------------------------------------------------------------------------------- context / output
class Ctx:
    def __init__(self, a):
        self.a = a
        self.out = a.out
        os.makedirs(self.out, exist_ok=True)
        self.deadline = T_START + a.tmax
        self.g = None
        self.roofs = None
        self.keys = {}

    def gguf(self):
        if self.g is None:
            path = self.a.gguf or os.environ.get("GGUF")
            assert path, "pass --gguf PATH or set GGUF"
            self.g = GGUFCollection(path)
            m = self.g.meta
            self.p = m["general.architecture"] + "."
            self.n_dense = m[self.p + "leading_dense_block_count"]
            self.n_trunk = m[self.p + "block_count"] - m.get(self.p + "nextn_predict_layers", 0)
            self.topk = m[self.p + "expert_used_count"]
        return self.g

    def left(self):
        return self.deadline - time.time()

    def roof(self, fmt):
        if self.roofs is None:
            f = os.path.join(self.out, "roofs.json")
            self.roofs = json.load(open(f)) if os.path.exists(f) else {}
        return self.roofs.get("read_best_gbs")      # hardware streaming read roof; the per-format gemv numbers are in roofs.json (pct_fmt)

    def seen(self, op, key):
        f = os.path.join(self.out, op + ".jsonl")
        if op not in self.keys:
            ks = set()
            if os.path.exists(f):
                for ln in open(f):
                    try:
                        ks.add(json.loads(ln).get("key"))
                    except Exception:       # noqa: BLE001
                        pass
            self.keys[op] = ks
        return key in self.keys[op]

    def emit(self, op, key, rec):
        rec.update(op=op, key=key, sha=SHA, flags=FLAGHASH, time=int(time.time()))
        with open(os.path.join(self.out, op + ".jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        self.keys[op].add(key)
        r = rec
        print(f"{op:10s} {r.get('variant', ''):26s} T {r.get('T', 0)} ctx {r.get('ctx', 0):6d} | {r['ms_call'] * 1000:9.1f} us/call {r['ms_call'] * 1000 / max(r.get('T', 1), 1):8.1f} us/row "
              f"{r.get('gbs', 0):7.1f} GB/s {('%5.0f%%' % r['pct_roof']) if r.get('pct_roof') else '   n/a'} | step {r.get('step_ms', 0):7.2f} ms | err {r.get('err_rel', float('nan')):.1e} | {r['mode']} n={r['n']}", flush=True)


def point(c, op, T, ctx, variant, fns, nbytes, fmt, per_step, err=None, extra=None):
    """time one cycle and emit a line. nbytes = weight / KV bytes one call must read at least; per_step = calls of this class per decode step"""
    key = f"{SHA}|{FLAGHASH}|{op}|{variant}|{T}|{ctx}"
    if c.seen(op, key) and not c.a.force:
        print(f"{op} {variant} T {T} ctx {ctx}: cached", flush=True)
        return
    if c.left() < 10:
        print(f"{op} {variant} T {T} ctx {ctx}: SKIPPED (deadline)", flush=True)
        return
    ms, mode, n, p10, p90 = bench(fns, c.a.nrep, min(6.0, max(1.0, c.left() / 4)), not c.a.no_graph)
    ms_call = ms / len(fns)
    gbs = nbytes / (ms_call * 1e-3) / 1e9
    roof = c.roof(fmt)
    rec = dict(T=T, ctx=ctx, variant=variant, ms_call=ms_call, p10=p10 / len(fns), p90=p90 / len(fns), gbs=gbs, bytes=nbytes, calls_cycle=len(fns), per_step=per_step,
               step_ms=ms_call * per_step, fmt=fmt, roof_gbs=roof, pct_roof=(100 * gbs / roof if roof else None), mode=mode, n=n)
    fr = c.roofs.get("gemv_gbs", {}).get(fmt) if c.roofs else None
    if fr:
        rec["pct_fmt"] = 100 * gbs / fr
    if err:
        rec.update(err_rel=err[0], err_max=err[1])
    if extra:
        rec.update(extra)
    c.emit(op, key, rec)


def relerr(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item(), (a - b).abs().max().item()


def tlist(a):
    return [int(x) for x in a.T.split(",")]


def moe_layers(c):
    return list(range(c.n_dense, c.n_trunk))


# ---------------------------------------------------------------------------------------------------------------- roofs
def op_roofs(c):
    out, R = {}, {}
    out["device"] = torch.cuda.get_device_name(0)
    n = 1 << 30
    src = torch.randint(0, 255, (n,), dtype=torch.uint8, device=dev)
    dst = torch.empty_like(src)
    def tm(fn, k=20):
        fn(); fn(); torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        ts = []
        for _ in range(k):
            a.record(); fn(); b.record(); b.synchronize(); ts.append(a.elapsed_time(b))
        return statistics.median(ts) / 1e3
    t = tm(lambda: dst.copy_(src))
    out["copy_read_gbs"], out["copy_traffic_gbs"] = n / t / 1e9, 2 * n / t / 1e9      # read-only convention (bytes / t) and read+write traffic (2 bytes / t)
    f32 = src.view(torch.float32)
    t = tm(lambda: f32.sum())
    out["read_sum_gbs"] = n / t / 1e9          # pure read: 1 GiB reduction
    del dst
    print(f"roofs: copy {out['copy_read_gbs']:.0f} GB/s (read-equiv) / {out['copy_traffic_gbs']:.0f} GB/s (r+w traffic); read-sum {out['read_sum_gbs']:.0f} GB/s", flush=True)
    # fp16 / bf16 WMMA
    for dt, nm in ((torch.float16, "fp16"), (torch.bfloat16, "bf16")):
        A = torch.randn(8192, 8192, device=dev, dtype=dt)
        B = torch.randn(8192, 8192, device=dev, dtype=dt)
        t = tm(lambda: torch.mm(A, B), 10)
        out[nm + "_tflops"] = 2 * 8192 ** 3 / t / 1e12
        del A, B
    print(f"roofs: torch.mm 8192^3 fp16 {out['fp16_tflops']:.1f} TF, bf16 {out['bf16_tflops']:.1f} TF", flush=True)
    # launch + gap inside a graph
    x = torch.zeros(1, device=dev)
    NK = 1000
    def chain():
        for _ in range(NK):
            x.add_(1)
    ms, mode, nn, _, _ = bench([chain], 100, 3.0, True)
    out["graph_tiny_kernel_us"] = ms * 1e3 / NK
    ms_e, _, _, _, _ = bench([chain], 20, 3.0, False)
    out["eager_tiny_kernel_us"] = ms_e * 1e3 / NK
    big = torch.zeros(1 << 22, device=dev)         # 16 MB fp32
    big2 = torch.zeros_like(big)
    def bigs():
        for _ in range(200):
            big2.copy_(big)
    def bigsmall():
        for _ in range(200):
            big2.copy_(big); x.add_(1)
    tb = bench([bigs], 60, 3.0, True)[0]
    tbs = bench([bigsmall], 60, 3.0, True)[0]
    out["graph_big_to_small_extra_us"] = (tbs - tb) * 1e3 / 200      # one tiny kernel inserted after a big one: its duration + both boundaries
    print(f"roofs: graph chain of tiny kernels {out['graph_tiny_kernel_us']:.2f} us per kernel (eager {out['eager_tiny_kernel_us']:.1f} us); big->small extra {out['graph_big_to_small_extra_us']:.2f} us per inserted kernel", flush=True)
    del src, big, big2
    torch.cuda.empty_cache()
    # GEMV read roofs on the real shapes of each stored format (header only, random bytes: speed is data independent)
    g = c.gguf()
    ext = HF._ext()
    assert ext is not None, "extension .so not loaded (set HY4_EXT_DIR like the server): QLinear / mmvq would silently fall back"
    di = torch.cuda.current_device()
    blk = {8: ("q8_0", 32, 34), 11: ("q3_k", 256, 110), 12: ("q4_k", 256, 144), 13: ("q5_k", 256, 176)}
    gem, shapes = {}, {}
    for ty, (fmt, be, bb) in blk.items():
        cands = {}
        for name, (t_, dims, off) in g.infos.items():
            if t_ == ty and len(dims) >= 2 and dims[0] % be == 0:
                cands[(dims[0], dims[1])] = dims[0] // be * bb * dims[1] * (dims[2] if len(dims) > 2 else 1)      # (K, rows of one matrix) -> bytes of the whole tensor
        if not cands and ty == 13:          # no Q5_K tensor in the GGUF: the Q5_K draft head (K = hidden size, rows stacked) sets the shape
            K_ = int(g.meta[c.p + "embedding_length"])
            cands[(K_, 1)] = (K_ // be * bb) * (96 << 20) // (K_ // be * bb)
        if not cands:
            continue
        best = []
        for (K, Nr), _nb in sorted(cands.items(), key=lambda kv: -kv[1])[:3]:
            rowb = K // be * bb
            Nr = min(_nb // rowb, max(16384, (96 << 20) // rowb))      # stack the experts' rows of a 3-D tensor: a roof needs >= ~64 MB per launch
            ncopy = max(2, math.ceil(1.5 * GiB / (Nr * rowb)))
            bufs = [torch.randint(0, 255, (Nr, rowb), dtype=torch.uint8, device=dev) for _ in range(ncopy)]
            x = torch.randn(1, K, device=dev)
            o = torch.empty(1, Nr, device=dev)
            if ty == 8:
                pk = [b.reshape(Nr, K // 32, 34) for b in bufs]
                fns = [(lambda p=p: ext.fq_gemv_q8_0(x, p, o)) for p in pk]
            else:
                fk = HF.mmvq_fn(ty, K)
                qb = ext.fq_q8_scratch(di, 1, K)
                ext.fq_quant_q8_1_rows(x.half().contiguous(), qb)
                fns = [(lambda b=b: fk(qb[0:1], b, o)) for b in bufs]
            ms = bench(fns, 100, 3.0, True)[0] / ncopy
            g_ = Nr * rowb / (ms * 1e-3) / 1e9
            best.append(g_)
            shapes.setdefault(fmt, []).append(dict(K=K, N=Nr, gbs=g_))
            print(f"roofs: gemv {fmt} K {K} N {Nr} ({Nr * rowb / 1e6:.0f} MB x {ncopy}): {ms * 1e3:.0f} us {g_:.0f} GB/s", flush=True)
            del bufs
        gem[fmt] = max(best)
    out["gemv_gbs"], out["gemv_shapes"] = gem, shapes
    out["read_best_gbs"] = max([out["read_sum_gbs"]] + list(gem.values()))
    out["sha"], out["flags"] = SHA, FLAGHASH
    json.dump(out, open(os.path.join(c.out, "roofs.json"), "w"), indent=1)
    print("ROOFS", json.dumps({k: (round(v, 1) if isinstance(v, float) else v) for k, v in out.items() if k != "gemv_shapes"}), flush=True)


# ---------------------------------------------------------------------------------------------------------------- dense_q8
def op_dense_q8(c):
    g = c.gguf()
    assert HF._ext() is not None, "extension .so not loaded"
    groups = {}          # (suffix, N, K) -> tensor names of the trunk layers (the shape differs between layer classes: head count, dense vs MoE FFN)
    for n, (t_, dims, _o) in g.infos.items():
        if n.startswith("blk.") and t_ == 8 and len(dims) == 2 and int(n.split(".")[1]) < c.n_trunk and not any(s_ in n for s_ in ("exps", "engram", "indexer", "ple")):
            groups.setdefault((n.split(".", 2)[2], dims[1], dims[0]), []).append(n)
    for (suf, N, K), names in sorted(groups.items()):
        names = sorted(names, key=lambda n: int(n.split(".")[1]))
        nb1 = N * (K // 32) * 34
        sel = names[:max(c.a.layers, math.ceil((96 << 20) / nb1))]          # cycle >= 96 MB of distinct weights (MALL is 32 MB)
        qs = []
        for n in sel:
            q = HF.load_linear(g, n, dev)
            assert q is not None, f"load_linear refused {n} (extension missing?)"
            qs.append(q)
        nbytes = N * (K // 32) * 34
        for T in tlist(c.a):
            x = torch.randn(T, K, device=dev)
            err = None
            if c.a.check:
                err = relerr(qs[0](x), x.float() @ qs[0].dense().T)
            point(c, "dense_q8", T, 0, f"{suf}[{N}x{K}]", [(lambda q=q: q(x)) for q in qs], nbytes, "q8_0", len(names), err, dict(layers=len(qs), small_cycle=len(qs) * nbytes < (64 << 20)))
        del qs
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------------------------- experts
def calib_overlap(S, E, target4=47, trials=120):
    rng = np.random.default_rng(0)
    def distinct(p, T=4):
        d = 0
        for _ in range(trials):
            prev, seen = rng.choice(E, S, replace=False), set()
            seen.update(prev.tolist())
            for _t in range(T - 1):
                k = rng.binomial(S, p)
                keep = rng.choice(prev, k, replace=False) if k else np.array([], dtype=int)
                ks = set(keep.tolist())
                rest = [e for e in rng.permutation(E) if e not in ks][:S - k]
                prev = np.concatenate([keep, np.array(rest, dtype=int)])
                seen.update(prev.tolist())
            d += len(seen)
        return d / trials
    lo, hi = 0.0, 0.95
    for _ in range(8):
        mid = (lo + hi) / 2
        if distinct(mid) > target4:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def sample_sel(T, S, E, p, rng):
    rows, prev = [], rng.choice(E, S, replace=False)
    rows.append(prev)
    for _ in range(T - 1):
        k = rng.binomial(S, p)
        keep = rng.choice(prev, k, replace=False) if k else np.array([], dtype=int)
        ks = set(keep.tolist())
        rest = [e for e in rng.permutation(E) if e not in ks][:S - k]
        prev = np.concatenate([keep, np.array(rest, dtype=int)])
        rows.append(prev)
    return np.stack(rows).astype(np.int64)


def load_experts(c, i):
    g = c.gguf()
    info = {n: g.infos[f"blk.{i}.ffn_{n}_exps.weight"] for n in ("gate", "up", "down")}
    f = {n: HF.GG_GROUPED[int(info[n][0])] for n in info}
    rows = {n: (info[n][1][1], info[n][1][0] // f[n][1] * f[n][2], info[n][1][2]) for n in info}
    bufs, ptrs, strides = {}, {}, {}
    for n in ("gate", "up", "down"):
        r, rb, E = rows[n]
        tot = r * rb * E
        buf = torch.empty(tot, dtype=torch.uint8, device=dev)
        for o in range(0, tot, 1 << 27):
            cc = min(1 << 27, tot - o)
            buf[o:o + cc].copy_(torch.from_numpy(np.ascontiguousarray(g._pread(info[n][2] + o, cc))))
        bufs[n], strides[n] = buf, rb
        ptrs[n] = torch.arange(E, dtype=torch.int64, device=dev) * (r * rb) + buf.data_ptr()
    from types import SimpleNamespace
    return SimpleNamespace(bufs=bufs, ptrs=(ptrs["gate"], ptrs["up"], ptrs["down"]), strides=(strides["gate"], strides["up"], strides["down"]), fu=f["gate"][0], fd=f["down"][0],
                           nff=rows["gate"][0], nout=rows["down"][0], E=rows["gate"][2], K=info["gate"][1][0], ebytes=(rows["gate"][0] * rows["gate"][1], rows["down"][0] * rows["down"][1]),
                           fmt=HF.GG_MMVQ[int(info["gate"][0])][0])


def op_experts(c):
    g = c.gguf()
    ext = HF._ext()
    assert ext is not None and hasattr(ext, "fq_moe_kq_grouped_mixed_q8"), "extension .so not loaded"
    m, p = g.meta, c.p
    clamp = list(m[p + "swiglu_clamp_exp"])
    w_scale = float(m[p + "expert_weights_scale"])
    S = c.topk
    first, types = {}, {}
    for i in moe_layers(c):
        ty = int(g.infos[f"blk.{i}.ffn_gate_exps.weight"][0])
        types[ty] = types.get(ty, 0) + 1
        first.setdefault(ty, i)
    E0 = g.infos[f"blk.{c.n_dense}.ffn_gate_exps.weight"][1][2]
    pk = calib_overlap(S, E0)
    print(f"experts: E {E0} topk {S}; layers per gate type {types}; synthetic routing: row overlap p={pk:.2f} calibrated to 47 distinct experts at T=4 (no captured routing sample on disk)", flush=True)
    rng = np.random.default_rng(1)
    for ty, i in first.items():
        rs = load_experts(c, i)
        N = rs.K
        per_exp = 2 * rs.ebytes[0] + rs.ebytes[1]
        lim = float(clamp[i])
        for T in tlist(c.a):
            sels = [sample_sel(T, S, rs.E, pk, rng) for _ in range(c.a.variants)]
            dis = [len(set(s.flatten().tolist())) for s in sels]
            z = (torch.randn(T, N, device=dev) * 0.5).half()
            ws = [(torch.rand(T, S, device=dev) + 0.1) for _ in sels]
            ws = [(w / w.sum(-1, keepdim=True) * w_scale).half() for w in ws]
            seltt = [torch.from_numpy(s).to(dev) for s in sels]
            gpart = torch.empty(HF.MMVQ_MAXR * S, N, dtype=torch.float32, device=dev)
            gout = torch.empty(HF.MMVQ_MAXR, N, dtype=torch.float32, device=dev)
            def call(sel, w):
                ext.fq_moe_kq_set_swiglu_limit(lim)
                ext.fq_moe_kq_grouped_mixed_q8(z, sel, w, *rs.ptrs, *rs.strides, rs.fu, rs.fd, gpart[:T * S], gout[:T])
            err = None
            if c.a.check:
                call(seltt[0], ws[0]); torch.cuda.synchronize()
                err = relerr(gout[:T].clone(), HF.moe_ref(z.float(), seltt[0], ws[0].float(), rs, lim))
            md = sum(dis) / len(dis)
            point(c, "experts", T, 0, f"{rs.fmt}_layer{i}", [(lambda s=s, w=w: call(s, w)) for s, w in zip(seltt, ws)], md * per_exp, rs.fmt, types[ty], err,
                  dict(distinct_mean=md, per_expert_bytes=per_exp, overlap_p=pk, routing="synthetic"))
        del rs
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------------------------- router
def op_router(c):
    g = c.gguf()
    ws, K = [], None
    for i in moe_layers(c):
        nm = f"blk.{i}.ffn_gate_inp.weight"
        if nm not in g.infos:
            continue
        t_, dims, off = g.infos[nm]
        assert t_ == 0, "router tensor is not F32"
        raw = np.ascontiguousarray(g._pread(off, dims[0] * dims[1] * 4)).copy().view(np.float32).reshape(dims[1], dims[0])
        w32 = torch.from_numpy(raw).to(dev)
        ws.append((w32, RG.store(w32)))
        K = dims[0]
    N = ws[0][0].shape[0]
    f16 = ws[0][1].dtype == torch.float16
    nbytes = N * K * ws[0][1].element_size()
    for T in tlist(c.a):
        x = torch.randn(T, K, device=dev)
        err = relerr(RG.mm(x, ws[0][1]), x.double() @ ws[0][0].double().T) if c.a.check else None
        point(c, "router", T, 0, f"rg_{'f16' if f16 else 'f32'}[{N}x{K}]", [(lambda w=w: RG.mm(x, w)) for _, w in ws], nbytes, "read", len(ws), err)
        rb = torch.randn(N, device=dev) * 0.1
        wsc = float(c.g.meta[c.p + "expert_weights_scale"])
        if RF.ENABLED:       # the second router launch per layer: sigmoid + bias + top-k select + weight norm (ref_fused.router_select)
            lg = [RG.mm(x, w) for _, w in ws]
            err2 = None
            if c.a.check:
                pr = torch.sigmoid(lg[0]); sl = torch.sort(pr + rb, dim=-1, descending=True, stable=True).indices[:, :c.topk]
                s_, w_ = RF.router_select(lg[0], rb, c.topk, wsc)
                err2 = (float((s_.sort(-1).values != sl.sort(-1).values).sum()), 0.0)        # (number of differing selected ids, -)
            point(c, "router", T, 0, f"select[{N}]", [(lambda l=l: RF.router_select(l, rb, c.topk, wsc)) for l in lg], N * 4 * T, "read", len(ws), err2, dict(note="err_rel = count of selected ids that differ from the torch sort reference"))


# ---------------------------------------------------------------------------------------------------------------- absorb
def op_absorb(c):
    g = c.gguf()
    groups = {}
    for name, (t_, dims, off) in g.infos.items():
        for suf, tag in (("attn_k_b.weight", "kb"), ("attn_v_b.weight", "vb")):
            if name.endswith(suf) and name.startswith("blk.") and int(name.split(".")[1]) < c.n_trunk:
                groups.setdefault((tag, dims[2], dims[1], dims[0]), []).append(name)
    for (tag, H, N, K), names in sorted(groups.items()):
        ms_ = [BG.load(g, n, H, dev) for n in names[:max(c.a.layers * 4, 16)]]
        nbytes = H * N * K * 34 / 32
        for T in tlist(c.a):
            x = torch.randn(T, H, K, device=dev)
            err = None
            if c.a.check:
                m0 = ms_[0]
                w = m0.q.float().reshape(H, N, K // 32, 32) * m0.d.float()[..., None]
                err = relerr(m0.mv(x).reshape(T, H, N), torch.einsum("hnk,thk->thn", w.reshape(H, N, K), x))
            point(c, "absorb", T, 0, f"{tag}[H{H} N{N} K{K}]", [(lambda m=m: m.mv(x)) for m in ms_], nbytes, "q8_0", len(names), err, dict(layers=len(ms_)))
        del ms_
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------------------------- mla_rows
def op_mla_rows(c):
    g = c.gguf()
    m, p = g.meta, c.p
    R, P = int(m[p + "attention.kv_lora_rank"]), int(m[p + "rope.dimension_count"])
    Hs = [int(h) for h in list(m[p + "attention.head_count"])]
    pat = [bool(x) for x in list(m[p + "attention.sliding_window_pattern"])]
    win = int(m[p + "attention.sliding_window"])
    cfg = [(Hs[i], win if pat[i] else 0) for i in range(c.n_trunk)]
    classes = sorted(set(cfg))
    print(f"mla_rows: R {R} P {P} layer classes (H, window) -> count: " + str({k: cfg.count(k) for k in classes}), flush=True)
    for ctx in [int(x) for x in c.a.ctx.split(",")]:
        N = ctx
        # one KV buffer per layer (distinct, fp32 like the engine); built once per ctx and reused over T
        kvs = [torch.randn(N + 16, R + P, device=dev) * 1.5 for _ in range(c.n_trunk)]
        sinks = [torch.randn(H, device=dev) for H, _w in cfg]
        scale = (R + P) ** -0.5 * 1.7
        kpos = torch.arange(N + 16, device=dev)
        for T in tlist(c.a):
            q = torch.randn(T, max(Hs), R, device=dev) * 0.5
            qp = torch.randn(T, max(Hs), P, device=dev) * 0.5
            qpos = torch.arange(N - T, N, device=dev)
            def call(i):
                H, w = cfg[i]
                kc = kvs[i]
                return RF.mla_sink_attend(q[:, :H], qp[:, :H], kc[:N, :R], kc[:N, R:], qpos, kpos[:N], scale, sinks[i], w, None, kpos_is_arange=True)
            err = None
            if c.a.check:
                e = []
                for i in sorted({cfg.index(k) for k in classes}):
                    H, w = cfg[i]
                    kc = kvs[i]
                    e.append(relerr(call(i), RF.mla_sink_attend_ref(q[:, :H], qp[:, :H], kc[:N, :R], kc[:N, R:], qpos, kpos[:N], scale, sinks[i], w, None)))
                err = (max(x[0] for x in e), max(x[1] for x in e))
            # whole step (all layers) is the cycle; bytes = KV rows each layer must read once
            span = sum(min(N, w) if w else N for _H, w in cfg)
            point(c, "mla_rows", T, ctx, "allL", [(lambda i=i: call(i)) for i in range(c.n_trunk)], span * (R + P) * 4 / c.n_trunk, "read", c.n_trunk, err,
                  dict(note="ms_call is per layer; bytes = KV read once per layer (row-sharing floor); dense attend, no indexer"))
        del kvs
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------------------------- head
def op_head(c):
    g = c.gguf()
    t_, dims, _o = g.infos["output.weight"]
    D, V = dims[0], dims[1]
    W = (torch.randn(V, D, device=dev) * 0.05).to(torch.bfloat16)
    assert HB.ENABLED, "HY4_HEAD_BF16 not on"
    for T in tlist(c.a):
        h = torch.randn(T, D, device=dev)
        err = None
        if c.a.check:
            ref = torch.cat([h.double() @ W[a:a + 16384].double().T for a in range(0, V, 16384)], 1)
            err = relerr(HB.gemv(h, W), ref)
        point(c, "head", T, 0, f"bf16[{V}x{D}]", [lambda: HB.gemv(h, W)], V * D * 2, "read", 1, err, dict(note="shipped bf16 head; Q8 head (dec8) measured 1638 us in head_ab.log HEADOP"))


# ---------------------------------------------------------------------------------------------------------------- draft_head
def op_draft_head(c):
    ext = HF._ext()
    assert ext is not None, "extension .so not loaded"
    f = os.environ.get("HY4_MTP_HEAD_BIN")
    assert f and os.path.exists(f), "set HY4_MTP_HEAD_BIN to the Q5_K draft head file"
    D = int(c.gguf().meta[c.p + "embedding_length"])
    rowb = D // 256 * 176
    raw = np.fromfile(f, dtype=np.uint8)
    V = raw.size // rowb
    W = torch.from_numpy(raw[:V * rowb]).reshape(V, rowb).to(dev)
    fk = HF.mmvq_fn(13, D)
    di = torch.cuda.current_device()
    for T in tlist(c.a):
        x = torch.randn(T, D, device=dev)
        qb = ext.fq_q8_scratch(di, T, D)
        o = torch.empty(T, V, device=dev)
        def call():
            ext.fq_quant_q8_1_rows(x.half().contiguous(), qb)
            for a0 in range(0, T, 4):
                fk(qb[a0:a0 + 4], W, o[a0:a0 + 4])
        point(c, "draft_head", T, 0, f"q5_k[{V}x{D}]", [call], V * rowb, "q5_k", 3, None, dict(note="per draft token; 3 per round; no torch Q5_K reference in tree: check n/a"))


def op_info(c):
    g = c.gguf()
    m = g.meta
    for k, v in m.items():
        if any(s in k for s in ("head_count", "sliding", "kv_lora", "rope.dim", "block_count", "expert_", "embedding_length", "leading", "nextn")) and "tokenizer" not in k:
            print(k, str(v)[:120])
    print("n_dense", c.n_dense, "n_trunk", c.n_trunk, "topk", c.topk, "ext", HF._ext() is not None, "sha", SHA, "flags", FLAGHASH)


OPS = dict(roofs=op_roofs, dense_q8=op_dense_q8, experts=op_experts, router=op_router, absorb=op_absorb, mla_rows=op_mla_rows, head=op_head, draft_head=op_draft_head, info=op_info)
ALL = ["roofs", "experts", "mla_rows", "dense_q8", "absorb", "router", "head", "draft_head"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("op")
    ap.add_argument("--T", default="1,2,4,8")
    ap.add_argument("--ctx", default="1024")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--gguf")
    ap.add_argument("--out", default=os.environ.get("OPBENCH_OUT", "opbench_out"))
    ap.add_argument("--layers", type=int, default=6, help="layers cycled per dense matrix (distinct weights, > MALL)")
    ap.add_argument("--variants", type=int, default=12, help="routing variants cycled per experts point")
    ap.add_argument("--nrep", type=int, default=200)
    ap.add_argument("--tmax", type=float, default=170, help="seconds for the whole run (points past it are skipped)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    c = Ctx(a)
    c.gguf()
    assert HF._ext() is not None or a.op == "info", "extension .so not loaded: set HY4_EXT_DIR"
    for op in (ALL if a.op == "all" else a.op.split(",")):
        t0 = time.time()
        print(f"=== {op} (sha {SHA} flags {FLAGHASH})", flush=True)
        OPS[op](c)
        print(f"=== {op} done {time.time() - t0:.0f} s", flush=True)
    print("OPBENCH_DONE", round(time.time() - T_START), "s", flush=True)


if __name__ == "__main__":
    main()
