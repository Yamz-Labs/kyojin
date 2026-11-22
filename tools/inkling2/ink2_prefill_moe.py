"""Prefill (T > 8 rows) routed experts of Inkling 2 Mini on CUDA: sort the (token, expert) slots by expert, then per expert
   Triton dequant of the RESIDENT stored bytes (NVFP4 GGUF type 40 / Q8_0, the per-expert fp32 scale folded in) into an fp16 scratch  ->  cuBLAS fp16 GEMM with fp32 output (tensor cores) over the expert rows
   -> SwiGLU (fp32) -> dequant down -> GEMM -> weighted index_add. One resident copy of the weights (the same bytes the T <= 8 decode kernels read), no fp16 copy of the model: the scratch is 2 matrices (N x K fp16) reused per expert.
Semantics = the torch loop of Inkling2Forward._moe (same gate / up split, situ, per-expert scales, router weights); activations are fp16 into the GEMMs (as in the decode kernels), outputs fp32."""
import torch
import triton
import triton.language as tl


@triton.jit
def _dq_nvfp4(RAW, OUT, S, N, NB, K, BR: tl.constexpr):
    pr = tl.program_id(0)
    pb = tl.program_id(1)
    rows = pr * BR + tl.arange(0, BR)
    c = tl.arange(0, 64)
    g = c // 16
    j = c % 16
    rm = rows[:, None] < N
    base = (rows[:, None] * NB + pb) * 36
    sb = tl.load(RAW + base + g[None, :], mask=rm, other=0).to(tl.int32)
    db = tl.load(RAW + base + 4 + (g * 8 + (j % 8))[None, :], mask=rm, other=0).to(tl.int32)
    nib = tl.where((j >= 8)[None, :], db >> 4, db & 15)
    mag = nib & 7
    e = mag >> 1
    m = (mag & 1).to(tl.float32)
    v = tl.where(e == 0, m * 0.5, (1.0 + m * 0.5) * tl.exp2((e - 1).to(tl.float32)))
    v = tl.where((nib >> 3) == 1, -v, v)
    se = (sb >> 3) & 15
    sm = (sb & 7).to(tl.float32)
    sc = tl.where(se == 0, sm * (1.0 / 512.0), (1.0 + sm * 0.125) * tl.exp2((se - 7).to(tl.float32)))
    sc = tl.where((sb == 0) | (sb == 0x7F), 0.0, sc)
    val = v * sc * tl.load(S)
    tl.store(OUT + rows[:, None] * K + pb * 64 + c[None, :], val.to(tl.float16), mask=rm)


@triton.jit
def _dq_q8(RAW, OUT, S, N, NB, K, BR: tl.constexpr):
    pr = tl.program_id(0)
    pb = tl.program_id(1)
    rows = pr * BR + tl.arange(0, BR)
    c = tl.arange(0, 64)
    blk = c // 32
    rm = rows[:, None] < N
    base = (rows[:, None] * NB + pb * 2 + blk[None, :]) * 34
    d = tl.load((RAW + base).to(tl.pointer_type(tl.float16)), mask=rm, other=0.0).to(tl.float32)
    q = tl.load(RAW + base + 2 + (c % 32)[None, :], mask=rm, other=0).to(tl.int32)
    q = tl.where(q >= 128, q - 256, q).to(tl.float32)
    tl.store(OUT + rows[:, None] * K + pb * 64 + c[None, :], (d * q * tl.load(S)).to(tl.float16), mask=rm)


def dq_expert(raw_e, out, scale, q8):
    """raw_e uint8 [N, nbytes] of ONE expert -> out fp16 [N, K] (K % 64 == 0), x scale (device fp32 [1])"""
    N, K = out.shape
    BR = 16
    if q8:
        _dq_q8[(triton.cdiv(N, BR), K // 64)](raw_e, out, scale, N, K // 32, K, BR=BR, num_warps=4)
    else:
        _dq_nvfp4[(triton.cdiv(N, BR), K // 64)](raw_e, out, scale, N, K // 64, K, BR=BR, num_warps=4)


def situ(gate, up, beta, lb):
    act = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if lb > 0:
        up = lb * torch.tanh(up / lb)
    return act * up


class PrefillMoe:
    def __init__(self, dev, D, ff, topk, beta, lbeta):
        self.dev, self.D, self.ff, self.topk, self.beta, self.lbeta = dev, D, ff, topk, beta, lbeta
        self.wgu = torch.empty(2 * ff, D, dtype=torch.float16, device=dev)
        self.wdn = torch.empty(D, ff, dtype=torch.float16, device=dev)
        self.one = torch.ones(1, dtype=torch.float32, device=dev)
        self.fires = {"apps": 0, "experts": 0, "rows": 0}

    @torch.no_grad()
    def __call__(self, gu, dn, s_gu, s_dn, q8, x, sel, w):
        """gu [E, 2ff, rb] uint8, dn [E, D, rb2] uint8 (resident), s_* [E] fp32 or None; x [T, D] fp32, sel [T, topk], w [T, topk] fp32 (router weights incl. the scale) -> [T, D] fp32"""
        T, D = x.shape
        E, ff, tk = gu.shape[0], self.ff, self.topk
        flat = sel.reshape(-1).long()
        perm = torch.argsort(flat, stable=True)
        cnt = torch.bincount(flat, minlength=E).tolist()
        tok = perm // tk
        xs = x.clamp(-65504.0, 65504.0).half()[tok]
        ys = torch.empty(T * tk, D, dtype=torch.float32, device=self.dev)
        off = 0
        for e in range(E):
            c = cnt[e]
            if c == 0:
                continue
            dq_expert(gu[e], self.wgu, s_gu[e:e + 1] if s_gu is not None else self.one, q8)
            h = torch.mm(xs[off:off + c], self.wgu.T, out_dtype=torch.float32)
            a = situ(h[:, :ff], h[:, ff:], self.beta, self.lbeta).clamp(-65504.0, 65504.0).half()
            dq_expert(dn[e], self.wdn, s_dn[e:e + 1] if s_dn is not None else self.one, q8)
            ys[off:off + c] = torch.mm(a, self.wdn.T, out_dtype=torch.float32)
            off += c
            self.fires["experts"] += 1
        out = torch.zeros(T, D, dtype=torch.float32, device=self.dev)
        out.index_add_(0, tok, ys * w.reshape(-1)[perm][:, None])
        self.fires["apps"] += 1
        self.fires["rows"] += T
        return out
