"""Two-stage draft head (opt-in: EXL3_DRAFT2S=<pack.safetensors>, built by tools/qwen/draft2s_pack.py).

A draft step needs one argmax and one max probability over the whole vocabulary. Today it reads the whole EXL3 head
(Qwen 2560 x 248,320: 318 MB, 1.35 ms). This head reads ~70 MB instead:
  tier 1  rotation xr = E @ h (512 axes) and a 4-bit preview score of every row; 4096-bin histogram of the score keys;
          threshold = smallest bin whose suffix count >= M (pack meta "m", default 512; GLM 4096-wide head: 1024);
          compaction of the rows above it (~1.2 M, capped at CAP)
  tier 2  a 3-bit row copy (input Hadamard per 128 group) scores the candidates; top-16 by that score
  tier 3  the EXL3 head itself on the 16 candidates' 128-row blocks (trellis tiles gathered, production exl3_dec_gemv):
          exact logits, bit-identical to the full head's
  argmax = best exact logit among the 16; p = 1 / sum over candidates exp(v - max), v = exact logit for the 16, the
  3-bit score for the others (rows outside the candidates are left out: they hold a median 0.0000 / 90th pct 0.0011 of
  the normaliser on held-out Qwen draft steps, and dropping them changes the replayed served rule by <= 0.12 point).
The target verifies every draft, so this head cannot change the text, only how many drafts are accepted (measured:
argmax = full head on 99.70 % of held-out steps).
All index maths are bounds-checked on the CPU (masked loads, clamped rows)."""
import os
import torch
import triton
import triton.language as tl
from safetensors import safe_open
from safetensors.torch import load_file

NB, M, CAP, TOPN = 4096, 512, 2048, 16


@triton.jit
def _rot_kernel(E, x, xr, had, hq, hist, cnt, A, NA, H: tl.constexpr, G: tl.constexpr, RB: tl.constexpr, NB: tl.constexpr):
    # programs < NA: xr = E @ x (A rows); programs >= NA: hq = per-group Hadamard of x (H rows); program 0 zeroes the histogram
    pid = tl.program_id(0)
    if pid < NA:
        rows = pid * RB + tl.arange(0, RB)
        rm = rows < A
        acc = tl.zeros((RB,), dtype=tl.float32)
        for k0 in tl.static_range(0, H, 512):
            k = k0 + tl.arange(0, 512)
            km = k < H
            e = tl.load(E + rows[:, None] * H + k[None, :], mask=rm[:, None] & km[None, :], other=0.0).to(tl.float32)
            xv = tl.load(x + k, mask=km, other=0.0).to(tl.float32)
            acc += tl.sum(e * xv[None, :], axis=1)
        tl.store(xr + rows, acc, mask=rm)
    else:
        kh = (pid - NA) * RB + tl.arange(0, RB)
        khm = kh < H
        jj = tl.arange(0, G)
        hm = tl.load(had + (kh % G)[:, None] * G + jj[None, :], mask=khm[:, None], other=0.0).to(tl.float32)
        xh = tl.load(x + (kh // G)[:, None] * G + jj[None, :], mask=khm[:, None], other=0.0).to(tl.float32)
        tl.store(hq + kh, tl.sum(hm * xh, axis=1), mask=khm)
    if pid == 0:
        b = tl.arange(0, NB)
        tl.store(hist + b, tl.zeros((NB,), dtype=tl.int32))
        tl.store(cnt + tl.arange(0, 1), tl.zeros((1,), dtype=tl.int32))


@triton.jit
def _key_bin(s):
    u = s.to(tl.uint32, bitcast=True)
    k = tl.where((u >> 31) != 0, ~u, u | 0x80000000)          # order-preserving float -> uint32
    return (k >> 20).to(tl.int32)                              # 4096 bins


@triton.jit
def _preview_kernel(codes, scale, xr, out, hist, V, AW: tl.constexpr, AWP: tl.constexpr, BR: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BR + tl.arange(0, BR)
    rm = rows < V
    wi = tl.arange(0, AWP)
    wm = wi < AW
    w = tl.load(codes + rows[:, None] * AW + wi[None, :], mask=rm[:, None] & wm[None, :], other=0)
    acc = tl.zeros((BR,), dtype=tl.float32)
    for i in tl.static_range(8):
        xi = tl.load(xr + wi * 8 + i, mask=wm, other=0.0)
        acc += tl.sum((((w >> (4 * i)) & 15).to(tl.float32) - 7.5) * xi[None, :], axis=1)
    s = acc * tl.load(scale + rows, mask=rm, other=0.0)
    tl.store(out + rows, s, mask=rm)
    tl.atomic_add(hist + _key_bin(s), 1, mask=rm)


@triton.jit
def _thresh_kernel(hist, thr, M, NB: tl.constexpr):
    b = tl.arange(0, NB)
    h = tl.load(hist + b)
    suf = tl.sum(h) - tl.cumsum(h, axis=0) + h                 # rows with bin >= b
    best = tl.max(tl.where(suf >= M, b, -1))
    tl.store(thr + tl.arange(0, 1), tl.full((1,), 0, tl.int32) + best)


@triton.jit
def _compact_kernel(scores, thr, cnt, idx, V, CAP, BR: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BR + tl.arange(0, BR)
    rm = rows < V
    s = tl.load(scores + rows, mask=rm, other=float("-inf"))
    t = tl.load(thr)
    sel = rm & (_key_bin(s) >= t)
    pos = tl.atomic_add(cnt + tl.zeros((BR,), dtype=tl.int32), 1, mask=sel)
    tl.store(idx + pos, rows, mask=sel & (pos < CAP))


@triton.jit
def _row3_kernel(q3, q3s, hq, idx, cnt, out, CAP, V, D: tl.constexpr, QW: tl.constexpr, QWP: tl.constexpr,
                 G: tl.constexpr, NG: tl.constexpr):
    # one program per candidate slot: score = sum_k (code - 4) * scale[row, k // G] * hq[k], k = 10 w + i
    m = tl.program_id(0)
    n = tl.minimum(tl.load(cnt), CAP)
    live = m < n
    r = tl.load(idx + m, mask=live, other=0)
    r = tl.minimum(tl.maximum(r, 0), V - 1).to(tl.int64)
    wi = tl.arange(0, QWP)
    wm = (wi < QW) & live
    w = tl.load(q3 + r * QW + wi, mask=wm, other=0)
    acc = tl.zeros((QWP,), dtype=tl.float32)
    for i in tl.static_range(10):
        k = wi * 10 + i
        km = wm & (k < D)
        sc = tl.load(q3s + r * NG + k // G, mask=km, other=0.0).to(tl.float32)
        xv = tl.load(hq + k, mask=km, other=0.0).to(tl.float32)
        acc += tl.where(km, (((w >> (3 * i)) & 7).to(tl.float32) - 4.0) * sc * xv, 0.0)
    tl.store(out + m, tl.where(live, tl.sum(acc), float("-inf")))


@triton.jit
def _select_kernel(s2, idx, cnt, rows16, CAP, V, TOPN: tl.constexpr, CAPP: tl.constexpr):
    # top-TOPN candidate slots by tier-2 score (first index on ties); their rows (clamped) -> rows16; their s2 -> -inf so
    # the finish kernel's normaliser counts them once, with their exact logits
    m = tl.arange(0, CAPP)
    n = tl.minimum(tl.load(cnt), CAP)
    live = m < n
    s = tl.load(s2 + m, mask=live, other=float("-inf"))
    for t in tl.static_range(TOPN):
        mv = tl.max(s)
        pos = tl.min(tl.where(s == mv, m, CAPP - 1))
        r = tl.load(idx + pos)
        r = tl.minimum(tl.maximum(r, 0), V - 1).to(tl.int64)
        tl.store(rows16 + t + tl.arange(0, 1), tl.full((1,), 0, tl.int64) + r)
        s = tl.where(m == pos, float("-inf"), s)
    tl.store(s2 + m, s, mask=live)


@triton.jit
def _gather_kernel(tre, svh, rows16, sub, svs, NT, TOPN: tl.constexpr, TW8: tl.constexpr, TW8P: tl.constexpr):
    # program (kt, j): the 8 n-tiles of candidate j's 128-row block (contiguous in the trellis) -> sub[kt, 8 j : 8 j + 8]
    kt = tl.program_id(0).to(tl.int64)
    j = tl.program_id(1)
    blk = tl.minimum(tl.load(rows16 + j) // 128, NT // 8 - 1)
    e = tl.arange(0, TW8P)
    em = e < TW8
    v = tl.load(tre + kt * NT * TW8 // 8 + blk * TW8 + e, mask=em, other=0)
    tl.store(sub + kt * TOPN * TW8 + j * TW8 + e, v, mask=em)
    if kt == 0:
        i = tl.arange(0, 128)
        tl.store(svs + j * 128 + i, tl.load(svh + blk * 128 + i))


@triton.jit
def _finish_kernel(s2, cnt, rows16, y16, ids, pout, conf, CAP, TOPN: tl.constexpr, CAPP: tl.constexpr):
    # exact logits of the TOPN short-listed rows (y16[j * 128 + row % 128]); argmax among them; normaliser over the candidates
    j = tl.arange(0, TOPN)
    rj = tl.load(rows16 + j)
    lj = tl.load(y16 + j * 128 + rj % 128).to(tl.float32)
    mx = tl.max(lj)
    best = tl.min(tl.where(lj == mx, j, TOPN))
    rid = tl.sum(tl.where(j == best, rj, 0))
    n = tl.minimum(tl.load(cnt), CAP)
    m = tl.arange(0, CAPP)
    live = m < n
    s = tl.load(s2 + m, mask=live, other=float("-inf"))
    z = tl.sum(tl.where(live, tl.exp(s - mx), 0.0)) + tl.sum(tl.exp(lj - mx))
    tl.store(ids + tl.arange(0, 1), tl.full((1,), 0, tl.int64) + rid.to(tl.int64))
    tl.store(pout + tl.arange(0, 1), tl.full((1,), 0.0, tl.float32) + 1.0 / z)
    tl.store(conf + tl.arange(0, 1), tl.full((1,), 0.0, tl.float32) + mx)


class Draft2SHead:
    """Bound to one EXL3 head (LinearEXL3 inner: trellis / suh / svh, mul1 codebook). __call__(h [1, D] fp16) -> (ids [1, 1]
    int64, p [1] fp32, conf [1] fp32 = best exact logit). Fixed shapes, no host sync: graph-capturable."""

    def __init__(self, pack_path: str, inner, vocab: int):
        from exllamav3.ext import exllamav3_ext as ext
        from exllamav3.modules.quant.exl3 import dec_workspace
        self.ext = ext
        dev = inner.trellis.device
        P = load_file(pack_path, device=str(dev))
        with safe_open(pack_path, "pt") as fm: self.M = int((fm.metadata() or {}).get("m", M))
        self.E, self.pcodes, self.pscale = P["E"].contiguous(), P["pcodes"].contiguous(), P["pscale"].float().contiguous()
        self.q3, self.q3s, self.had = P["q3"].contiguous(), P["q3s"].contiguous(), P["had"].contiguous()
        self.A, self.D = self.E.shape
        self.V = self.pcodes.shape[0]
        self.G = self.had.shape[0]
        assert self.V == vocab and self.D == inner.in_features and self.D % self.G == 0
        assert self.q3.shape == (self.V, triton.cdiv(self.D, 10)) and self.q3s.shape == (self.V, self.D // self.G)
        self.tre, self.suh, self.svh, self.K = inner.trellis, inner.suh, inner.svh, inner.K
        self.KT, self.NT, self.TW = self.tre.shape
        self.nblk = inner.out_features // 128
        assert self.nblk * 128 >= self.V
        f = lambda *s, dt=torch.float32: torch.empty(*s, dtype=dt, device=dev)
        self.xr, self.scores = f(self.A), f(self.V)
        self.hist, self.cnt, self.thr = f(NB, dt=torch.int32), f(1, dt=torch.int32), f(1, dt=torch.int32)
        self.idx = torch.zeros(CAP, dtype=torch.int32, device=dev)
        self.s2, self.hq, self.rows16 = f(CAP), f(self.D), torch.zeros(TOPN, dtype=torch.long, device=dev)
        self.sub = torch.empty(self.KT, TOPN * 8, self.TW, dtype=self.tre.dtype, device=dev)
        self.svs = torch.empty(TOPN * 128, dtype=self.svh.dtype, device=dev)
        self.y16 = f(1, TOPN * 128, dt=torch.half)
        self.ids, self.p, self.conf = f(1, 1, dt=torch.long), f(1), f(1)
        self.scratch, self.counters = dec_workspace(dev)
        self.AW = self.A // 8
        self.AWP = 1 << (self.AW - 1).bit_length()
        self.QW = triton.cdiv(self.D, 10)              # the row kernel masks k >= D
        self.QWP = 1 << (self.QW - 1).bit_length()
        self.NA = triton.cdiv(self.A, 8)
        self.TW8 = 8 * self.TW
        self.TW8P = 1 << (self.TW8 - 1).bit_length()

    def __call__(self, h: torch.Tensor):
        x = h.reshape(self.D)
        if x.dtype != torch.half: x = x.half()
        x = x.contiguous()
        _rot_kernel[(self.NA + triton.cdiv(self.D, 8),)](self.E, x, self.xr, self.had, self.hq, self.hist, self.cnt, self.A,
                                                          self.NA, H=self.D, G=self.G, RB=8, NB=NB)
        _preview_kernel[(triton.cdiv(self.V, 32),)](self.pcodes, self.pscale, self.xr, self.scores, self.hist, self.V,
                                                    AW=self.AW, AWP=self.AWP, BR=32)
        _thresh_kernel[(1,)](self.hist, self.thr, self.M, NB=NB)
        _compact_kernel[(triton.cdiv(self.V, 1024),)](self.scores, self.thr, self.cnt, self.idx, self.V, CAP, BR=1024)
        _row3_kernel[(CAP,)](self.q3, self.q3s, self.hq, self.idx, self.cnt, self.s2, CAP, self.V, D=self.D, QW=self.QW,
                             QWP=self.QWP, G=self.G, NG=self.D // self.G)
        _select_kernel[(1,)](self.s2, self.idx, self.cnt, self.rows16, CAP, self.V, TOPN=TOPN, CAPP=CAP)
        _gather_kernel[(self.KT, TOPN)](self.tre, self.svh, self.rows16, self.sub, self.svs, self.NT, TOPN=TOPN, TW8=self.TW8,
                                        TW8P=self.TW8P)
        self.ext.exl3_dec_gemv(x.view(1, self.D), self.sub, self.suh, self.svs, self.y16, self.scratch, self.counters, self.K)
        _finish_kernel[(1,)](self.s2, self.cnt, self.rows16, self.y16, self.ids, self.p, self.conf, CAP, TOPN=TOPN, CAPP=CAP)
        return self.ids, self.p, self.conf


_heads = {}


def head_for(lm, vocab):
    """The Draft2SHead for this lm_head (a Linear wrapper) when EXL3_DRAFT2S names a pack and the head qualifies (EXL3 mul1
    inner, no LoRA / bias / logit scaling, pack built for this vocabulary), else None. One per head, built on first use."""
    path = os.environ.get("EXL3_DRAFT2S", "")
    if not path: return None
    key = id(lm)
    if key not in _heads:
        inner = getattr(lm, "inner", None)
        ok = (type(inner).__name__ == "LinearEXL3" and getattr(inner, "mul1", False) and not getattr(inner, "mcg", False)
              and inner.bias is None and not lm.lora_a_tensors and lm.pre_scale == 1.0 and lm.softcap == 0.0
              and lm.post_scale == 1.0 and inner.out_features % 128 == 0)
        _heads[key] = Draft2SHead(path, inner, vocab) if ok else None
        print(f" -- EXL3_DRAFT2S: {'two-stage draft head ON' if ok else 'head not eligible, OFF'} ({path})")
    return _heads[key]
