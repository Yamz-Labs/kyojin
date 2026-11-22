"""Prefill routed experts v2 (fewer launches): same math as ink2_prefill_moe.py (resident NVFP4 / Q8_0 bytes -> Triton dequant to fp16 with the per-expert scale folded -> cuBLAS fp16 GEMM, fp32 out),
but (1) the dequant of GROUP experts runs in ONE launch into a [GROUP, N, K] fp16 scratch (grid z = expert of the group), (2) SwiGLU + softcap + fp16 cast is ONE Triton kernel, (3) the down GEMM writes straight into
the slot buffer (mm out=), (4) the experts of a layer are walked in sorted order with one host sync per application. ~3 launches per expert instead of ~15."""
import torch
import triton
import triton.language as tl


@triton.jit
def _dq_nvfp4_b(RAW, OUT, S, IDX, N, NB, K, RB, BR: tl.constexpr):
    pr = tl.program_id(0)
    pb = tl.program_id(1)
    pz = tl.program_id(2)
    e = tl.load(IDX + pz).to(tl.int64)
    rows = pr * BR + tl.arange(0, BR)
    c = tl.arange(0, 64)
    g = c // 16
    j = c % 16
    rm = rows[:, None] < N
    base = e * RB + (rows[:, None] * NB + pb) * 36
    sb = tl.load(RAW + base + g[None, :], mask=rm, other=0).to(tl.int32)
    db = tl.load(RAW + base + 4 + (g * 8 + (j % 8))[None, :], mask=rm, other=0).to(tl.int32)
    nib = tl.where((j >= 8)[None, :], db >> 4, db & 15)
    mag = nib & 7
    ex = mag >> 1
    m = (mag & 1).to(tl.float32)
    v = tl.where(ex == 0, m * 0.5, (1.0 + m * 0.5) * tl.exp2((ex - 1).to(tl.float32)))
    v = tl.where((nib >> 3) == 1, -v, v)
    se = (sb >> 3) & 15
    sm = (sb & 7).to(tl.float32)
    sc = tl.where(se == 0, sm * (1.0 / 512.0), (1.0 + sm * 0.125) * tl.exp2((se - 7).to(tl.float32)))
    sc = tl.where((sb == 0) | (sb == 0x7F), 0.0, sc)
    val = v * sc * tl.load(S + e)
    tl.store(OUT + pz.to(tl.int64) * N * K + rows[:, None] * K + pb * 64 + c[None, :], val.to(tl.float16), mask=rm)


@triton.jit
def _dq_q8_b(RAW, OUT, S, IDX, N, NB, K, RB, BR: tl.constexpr):
    pr = tl.program_id(0)
    pb = tl.program_id(1)
    pz = tl.program_id(2)
    e = tl.load(IDX + pz).to(tl.int64)
    rows = pr * BR + tl.arange(0, BR)
    c = tl.arange(0, 64)
    blk = c // 32
    rm = rows[:, None] < N
    base = e * RB + (rows[:, None] * NB + pb * 2 + blk[None, :]) * 34
    d = tl.load((RAW + base).to(tl.pointer_type(tl.float16)), mask=rm, other=0.0).to(tl.float32)
    q = tl.load(RAW + base + 2 + (c % 32)[None, :], mask=rm, other=0).to(tl.int32)
    q = tl.where(q >= 128, q - 256, q).to(tl.float32)
    tl.store(OUT + pz.to(tl.int64) * N * K + rows[:, None] * K + pb * 64 + c[None, :], (d * q * tl.load(S + e)).to(tl.float16), mask=rm)


@triton.jit
def _swiglu(H, OUT, FF, beta, lb, HAS_LB: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    cb = tl.program_id(1)
    cols = cb * BLOCK + tl.arange(0, BLOCK)
    m = cols < FF
    g = tl.load(H + r * 2 * FF + cols, mask=m, other=0.0)
    u = tl.load(H + r * 2 * FF + FF + cols, mask=m, other=0.0)
    t = g / beta
    th = 1.0 - 2.0 / (tl.exp(2.0 * t) + 1.0)
    act = beta * th * (1.0 / (1.0 + tl.exp(-g)))
    if HAS_LB:
        tu = u / lb
        u = lb * (1.0 - 2.0 / (tl.exp(2.0 * tu) + 1.0))
    y = tl.minimum(tl.maximum(act * u, -65504.0), 65504.0)
    tl.store(OUT + r * FF + cols, y.to(tl.float16), mask=m)


def dq_group(raw, out, s, idx, q8):
    """raw uint8 [E, N, nbytes] (resident), idx int32 [G] expert ids, out fp16 [G, N, K] <- dequant x s[e]"""
    G, N, K = out.shape
    BR = 16
    RB = raw.shape[1] * raw.shape[2]
    grid = (triton.cdiv(N, BR), K // 64, G)
    if q8:
        _dq_q8_b[grid](raw, out, s, idx, N, K // 32, K, RB, BR=BR, num_warps=4)
    else:
        _dq_nvfp4_b[grid](raw, out, s, idx, N, K // 64, K, RB, BR=BR, num_warps=4)


class PrefillMoe2:
    GROUP = 8

    def __init__(self, dev, D, ff, topk, beta, lbeta, E):
        self.dev, self.D, self.ff, self.topk, self.beta, self.lbeta = dev, D, ff, topk, beta, lbeta
        self.wgu = torch.empty(self.GROUP, 2 * ff, D, dtype=torch.float16, device=dev)
        self.wdn = torch.empty(self.GROUP, D, ff, dtype=torch.float16, device=dev)
        self.ones = torch.ones(E, dtype=torch.float32, device=dev)
        self.fires = {"apps": 0, "experts": 0, "rows": 0}

    @torch.no_grad()
    def __call__(self, gu, dn, s_gu, s_dn, q8, x, sel, w):
        T, D = x.shape
        E, ff, tk, G = gu.shape[0], self.ff, self.topk, self.GROUP
        flat = sel.reshape(-1).long()
        perm = torch.argsort(flat, stable=True)
        cnt = torch.bincount(flat, minlength=E).tolist()
        tok = perm // tk
        xs = x.clamp(-65504.0, 65504.0).half()[tok]
        ys = torch.empty(T * tk, D, dtype=torch.float32, device=self.dev)
        act = torch.empty(T * tk, ff, dtype=torch.float16, device=self.dev)
        present = [e for e in range(E) if cnt[e] > 0]
        offs, o = {}, 0
        for e in range(E):
            offs[e] = o
            o += cnt[e]
        sg = s_gu if s_gu is not None else self.ones
        sd = s_dn if s_dn is not None else self.ones
        for i in range(0, len(present), G):
            grp = present[i:i + G]
            idx = torch.tensor(grp, dtype=torch.int32, device=self.dev)
            dq_group(gu, self.wgu[:len(grp)], sg, idx, q8)
            dq_group(dn, self.wdn[:len(grp)], sd, idx, q8)
            for k, e in enumerate(grp):
                c, off = cnt[e], offs[e]
                h = torch.mm(xs[off:off + c], self.wgu[k].T, out_dtype=torch.float32)
                _swiglu[(c, triton.cdiv(ff, 1024))](h, act[off:off + c], ff, self.beta, self.lbeta if self.lbeta > 0 else 1.0, HAS_LB=self.lbeta > 0, BLOCK=1024)
                ys[off:off + c] = torch.mm(act[off:off + c], self.wdn[k].T, out_dtype=torch.float32)
            self.fires["experts"] += len(grp)
        out = torch.zeros(T, D, dtype=torch.float32, device=self.dev)
        out.index_add_(0, tok, ys * w.reshape(-1)[perm][:, None])
        self.fires["apps"] += 1
        self.fires["rows"] += T
        return out
