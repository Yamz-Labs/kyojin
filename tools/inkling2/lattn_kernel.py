"""Inkling 2 Mini local-layer attention core (decode T=1..8 over the 1024-token ring), Triton, gfx1151.

One fused op for the 20 LOCAL layers: QK (GQA 3:1), relative bias lookup (dist < 1024) or bias from r.proj in-kernel, per-head sink,
online softmax, PV, split-K combine, sigmoid output gate. Inputs are the clipped q (and r), the ring K / V already holding the T new rows
(written in place at slot pos % R by the conv op; never a concat). Semantics = tools/inkling2_ref.py `attn` (window 1024, mask dist<0 or dist>=1024,
bias only for 0 <= dist < 1024, sink = extra logit with zero value).
"""
import math
import torch
import triton
import triton.language as tl

G = 3          # q heads per kv head
NH, NKV, HD, WIN, DR = 24, 8, 128, 1024, 16


@triton.jit
def _part_kernel(Q, TAB, RR, PT, K, V, POS, WSA, WSML,
                 T, sk_slot, sk_head, sv_slot, sv_head, st_t, sr_t, stab_t, scale,
                 R: tl.constexpr, WIN: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, HD: tl.constexpr,
                 G: tl.constexpr, NH: tl.constexpr, NKV: tl.constexpr, BPS: tl.constexpr, FUSE_R: tl.constexpr):
    kvh = tl.program_id(0)
    sp = tl.program_id(1)
    pos0 = tl.load(POS)
    m = tl.arange(0, BM)
    t = m // G
    head = kvh * G + (m % G)
    rvalid = m < T * G
    d = tl.arange(0, HD)
    q = tl.load(Q + t[:, None] * st_t + head[:, None] * HD + d[None, :], mask=rvalid[:, None], other=0.0)
    p_lo = pos0 - (WIN - 1)
    nkeys = WIN - 1 + T
    acc = tl.zeros([BM, HD], dtype=tl.float32)
    mi = tl.full([BM], float("-inf"), dtype=tl.float32)
    li = tl.zeros([BM], dtype=tl.float32)
    jn = tl.arange(0, BN)
    for b in range(BPS):
        j = (sp * BPS + b) * BN + jn
        p = p_lo + j
        pv = (p >= 0) & (j < nkeys)
        slot = tl.where(pv, p, 0) % R
        k = tl.load(K + slot[:, None] * sk_slot + kvh * sk_head + d[None, :], mask=pv[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale
        dist = pos0 + t[:, None] - p[None, :]
        ok = rvalid[:, None] & pv[None, :] & (dist >= 0) & (dist < WIN)
        if FUSE_R:
            bias = tl.zeros([BM, BN], dtype=tl.float32)
            for e in tl.static_range(16):
                re = tl.load(RR + t * sr_t + head * 16 + e, mask=rvalid, other=0.0)
                pe = tl.load(PT + e * WIN + dist, mask=ok, other=0.0)
                bias += re[:, None] * pe
        else:
            bias = tl.load(TAB + t[:, None] * stab_t + head[:, None] * WIN + dist, mask=ok, other=0.0)
        s = tl.where(ok, s + bias, float("-inf"))
        mn = tl.maximum(mi, tl.max(s, 1))
        ms = tl.where(mn == float("-inf"), 0.0, mn)
        pr = tl.exp(s - ms[:, None])
        al = tl.exp(mi - ms)
        li = li * al + tl.sum(pr, 1)
        v = tl.load(V + slot[:, None] * sv_slot + kvh * sv_head + d[None, :], mask=pv[:, None], other=0.0)
        acc = acc * al[:, None] + tl.dot(pr.to(v.dtype), v)
        mi = mn
    base = (sp * NKV + kvh) * BM + m
    tl.store(WSA + base[:, None] * HD + d[None, :], acc)
    tl.store(WSML + base * 2, mi)
    tl.store(WSML + base * 2 + 1, li)


@triton.jit
def _comb_kernel(WSA, WSML, SINK, GATE, OUT, NSPLIT, BM: tl.constexpr, HD: tl.constexpr, G: tl.constexpr,
                 NH: tl.constexpr, NKV: tl.constexpr, NSP2: tl.constexpr):
    pid = tl.program_id(0)
    t = pid // NH
    head = pid % NH
    kvh = head // G
    m = t * G + head % G
    sp = tl.arange(0, NSP2)
    msk = sp < NSPLIT
    base = (sp * NKV + kvh) * BM + m
    mi = tl.load(WSML + base * 2, mask=msk, other=float("-inf"))
    li = tl.load(WSML + base * 2 + 1, mask=msk, other=0.0)
    sink = tl.load(SINK + t * NH + head)
    M = tl.maximum(tl.max(mi, 0), sink)
    w = tl.exp(mi - M)
    den = tl.sum(li * w, 0) + tl.exp(sink - M)
    d = tl.arange(0, HD)
    a = tl.load(WSA + base[:, None] * HD + d[None, :], mask=msk[:, None], other=0.0)
    o = tl.sum(a * w[:, None], 0) / den
    g = tl.load(GATE + t * NH * HD + head * HD + d).to(tl.float32)
    o = o * (1.0 / (1.0 + tl.exp(-g)))
    tl.store(OUT + t * NH * HD + head * HD + d, o.to(OUT.dtype.element_ty))


class LAttn:
    """Workspace + launcher (no allocation in __call__: HIP-graph safe). pos = int32 device tensor [1] = absolute position of query row 0."""

    def __init__(self, device, ring_rows=1088, bn=64, nsplit=None, fuse_r=False, num_warps=4, max_t=8, out_dtype=torch.bfloat16):
        self.R, self.bn, self.fuse_r, self.nw = ring_rows, bn, fuse_r, num_warps
        self.max_t, self.out_dtype = max_t, out_dtype
        self.nsplit_arg = nsplit
        self.ws = {}
        self.dev = device

    def _ws(self, T):
        BM = 16 if G * T <= 16 else 32
        nb = math.ceil((WIN - 1 + T) / self.bn)
        ns = self.nsplit_arg or nb
        ns = min(ns, nb)
        bps = math.ceil(nb / ns)
        ns = math.ceil(nb / bps)
        key = (T,)
        if key not in self.ws:
            self.ws[key] = dict(BM=BM, ns=ns, bps=bps,
                                acc=torch.empty(ns, NKV, BM, HD, device=self.dev, dtype=torch.float32),
                                ml=torch.empty(ns, NKV, BM, 2, device=self.dev, dtype=torch.float32),
                                out=torch.empty(T, NH * HD, device=self.dev, dtype=self.out_dtype))
        return self.ws[key]

    def __call__(self, q, kring, vring, pos, sink, gate, tab=None, r=None, projE=None, out=None):
        """q [T,NH,HD] bf16; kring/vring: tensors [R,NKV,HD] or [NKV,R,HD] (strides read from the tensors; slot dim size R);
        sink [T,NH] f32; gate [T,NH*HD] (pre-sigmoid); tab [T,NH,WIN] f32 (bias lookup) OR r [T,NH,16] f32 + projE [16,WIN] f32 (e-major: coalesced along the key axis) (fused)."""
        T = q.shape[0]
        w = self._ws(T)
        slot_dim = 0 if kring.shape[1] == NKV else 1                                                # [R,NKV,HD] or [NKV,R,HD]
        head_dim = 1 - slot_dim
        o = out if out is not None else w["out"]
        dummy = q
        _part_kernel[(NKV, w["ns"])](
            q, tab if tab is not None else dummy, r if r is not None else dummy, projE if projE is not None else dummy,
            kring, vring, pos, w["acc"], w["ml"],
            T, kring.stride(slot_dim), kring.stride(head_dim), vring.stride(slot_dim), vring.stride(head_dim),
            q.stride(0), (r.stride(0) if r is not None else 0), (tab.stride(0) if tab is not None else 0), HD ** -0.5,
            R=self.R, WIN=WIN, BM=w["BM"], BN=self.bn, HD=HD, G=G, NH=NH, NKV=NKV, BPS=w["bps"], FUSE_R=self.fuse_r,
            num_warps=self.nw)
        _comb_kernel[(T * NH,)](w["acc"], w["ml"], sink, gate, o, w["ns"], BM=w["BM"], HD=HD, G=G, NH=NH, NKV=NKV,
                                NSP2=triton.next_power_of_2(w["ns"]), num_warps=4)
        return o


def lattn_ref(q, kall, vall, pos0, sink, gate, tab=None, r=None, proj=None):
    """float32 torch reference over ABSOLUTE positions (no ring): kall/vall [P,NKV,HD] hold positions 0..P-1 (P >= pos0 + T).
    Same math as tools/inkling2_ref.py attn (local layer): bias for 0 <= dist < 1024, mask dist < 0 or >= 1024, sink logit, sigmoid gate."""
    T = q.shape[0]
    q, kall, vall = q.float(), kall.float(), vall.float()
    P = kall.shape[0]
    kr, vr = kall.repeat_interleave(G, 1), vall.repeat_interleave(G, 1)                          # [P,NH,HD]
    kp = torch.arange(P, device=q.device)
    qp = pos0 + torch.arange(T, device=q.device)
    dist = qp[:, None] - kp[None, :]                                                               # [T,P]
    if tab is None:
        tab = torch.einsum("thd,de->the", r.float(), proj.float())                                 # [T,NH,WIN]
    di = dist.clamp(0, WIN - 1)
    bias = torch.gather(tab.float(), 2, di[:, None, :].expand(T, NH, P))
    inb = (dist >= 0) & (dist < WIN)
    bias = torch.where(inb[:, None, :], bias, torch.zeros_like(bias))
    sc = torch.einsum("thd,phd->thp", q, kr) * HD ** -0.5 + bias
    sc = sc.masked_fill(~inb[:, None, :], float("-inf"))
    sc = torch.cat([sc, sink.float()[:, :, None]], -1)
    p = torch.softmax(sc, -1)[..., :-1]
    o = torch.einsum("thp,phd->thd", p, vr).reshape(T, NH * HD)
    return o * torch.sigmoid(gate.float())


def fill_ring(kall, ring, R, upto):
    """write absolute positions [max(0,upto-R), upto) of kall [P,NKV,HD] into ring [R,NKV,HD] at slot p % R (in place, what the conv op does per token)."""
    for p in range(max(0, upto - R), upto):
        ring[p % R] = kall[p]


# ---------------------------------------------------------------------------------------------------- prefill tiles
@triton.jit
def _prefill_kernel(Q, RR, PT, K, V, SINK, GATE, OUT, SCR, T, hist, scale,
                    WIN: tl.constexpr, BQ: tl.constexpr, BN: tl.constexpr, HD: tl.constexpr, G: tl.constexpr, NH: tl.constexpr, NKV: tl.constexpr, BIAS: tl.constexpr, W: tl.constexpr):
    head = tl.program_id(0)
    tq = tl.program_id(1)
    kvh = head // G
    i = tq * BQ + tl.arange(0, BQ)               # query row (position pos0 + i; key row index of its own K = hist + i)
    rvalid = i < T
    d = tl.arange(0, HD)
    q = tl.load(Q + i[:, None] * (NH * HD) + head * HD + d[None, :], mask=rvalid[:, None], other=0.0)
    acc = tl.zeros([BQ, HD], dtype=tl.float32)
    mi = tl.full([BQ], float("-inf"), dtype=tl.float32)
    li = tl.zeros([BQ], dtype=tl.float32)
    row_hi = hist + tq * BQ + BQ                 # exclusive bound of key rows any row of this tile can see
    row_lo = tl.maximum(hist + tq * BQ - (WIN - 1), 0)
    row_lo = (row_lo // BN) * BN
    jn = tl.arange(0, BN)
    il = tl.arange(0, BQ)
    if BIAS == 2:
        ee = tl.arange(0, 16)
        rt = tl.load(RR + i[:, None] * (NH * 16) + head * 16 + ee[None, :], mask=rvalid[:, None], other=0.0)   # [BQ,16]
        ww = tl.arange(0, W)
        scr = SCR + (head * tl.num_programs(1) + tq) * (BQ * W)
    for j0 in range(row_lo, row_hi, BN):
        j = j0 + jn
        dist = (hist + i)[:, None] - j[None, :]
        ok = rvalid[:, None] & (dist >= 0) & (dist < WIN) & (j[None, :] < hist + T)
        k = tl.load(K + j[:, None] * (NKV * HD) + kvh * HD + d[None, :], mask=(j < hist + T)[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale
        if BIAS == 1:
            bias = tl.zeros([BQ, BN], dtype=tl.float32)
            for e in tl.static_range(16):
                re = tl.load(RR + i * (NH * 16) + head * 16 + e, mask=rvalid, other=0.0)
                pe = tl.load(PT + e * WIN + dist, mask=ok, other=0.0)
                bias += re[:, None] * pe
            s = s + bias
        if BIAS == 3:                                                  # PT = precomputed table [T,NH,WIN] f32 (glue: r x proj per sub-chunk), one load per score
            s = s + tl.load(PT + i[:, None] * (NH * WIN) + head * WIN + dist, mask=ok, other=0.0)
        if BIAS == 2:
            # Toeplitz tile: L[i, w] = r_i . proj[:, dd0 + w] on a band of distances (MMA, f32), spilled to a per-program scratch (L2) and re-read along the diagonals
            dd0 = (hist + tq * BQ) - j0 - (BN - 1)
            dw = dd0 + ww
            pet = tl.load(PT + ee[:, None] * WIN + dw[None, :], mask=((dw >= 0) & (dw < WIN))[None, :], other=0.0)    # [16,W]
            lt = tl.dot(rt, pet, input_precision="ieee")                                                           # [BQ,W]
            tl.debug_barrier()
            tl.store(scr + il[:, None] * W + ww[None, :], lt)
            tl.debug_barrier()
            idx = il[:, None] - jn[None, :] + (BN - 1)
            bias = tl.load(scr + il[:, None] * W + idx, cache_modifier=".cg")
            tl.debug_barrier()
            s = s + bias
        s = tl.where(ok, s, float("-inf"))
        mn = tl.maximum(mi, tl.max(s, 1))
        ms = tl.where(mn == float("-inf"), 0.0, mn)
        pr = tl.exp(s - ms[:, None])
        al = tl.exp(mi - ms)
        li = li * al + tl.sum(pr, 1)
        v = tl.load(V + j[:, None] * (NKV * HD) + kvh * HD + d[None, :], mask=(j < hist + T)[:, None], other=0.0)
        acc = acc * al[:, None] + tl.dot(pr.to(v.dtype), v)
        mi = mn
    sink = tl.load(SINK + i * NH + head, mask=rvalid, other=0.0)
    M = tl.maximum(mi, sink)
    w = tl.exp(mi - M)
    den = li * w + tl.exp(sink - M)
    o = acc * (w / den)[:, None]
    g = tl.load(GATE + i[:, None] * (NH * HD) + head * HD + d[None, :], mask=rvalid[:, None], other=0.0).to(tl.float32)
    o = o * (1.0 / (1.0 + tl.exp(-g)))
    tl.store(OUT + i[:, None] * (NH * HD) + head * HD + d[None, :], o.to(OUT.dtype.element_ty), mask=rvalid[:, None])


def lattn_prefill(q, r, projE, kbuf, vbuf, sink, gate, hist, out=None, bq=16, bn=32, num_warps=4, bias=1):
    """Prefill tile: T query rows at positions pos0..pos0+T-1 over linear key buffers kbuf / vbuf [hist + T, NKV, HD] (row i = position pos0 - hist + i:
    the last `hist` (<= 1023) ring rows in time order, then the K / V rows of the chunk itself). Bias from r [T,NH,16] x projE [16,WIN] in-kernel; sink + gate fused."""
    assert bias != 2 or bn == 32, "Toeplitz mode is validated for bn == 32 only (bn 64 deviates 1.1e-3)"
    T = q.shape[0]
    out = out if out is not None else torch.empty(T, NH * HD, device=q.device, dtype=torch.bfloat16)
    ntq = triton.cdiv(T, bq)
    W = triton.next_power_of_2(bq + bn)
    scr = torch.empty(NH * ntq * bq * W if bias == 2 else 1, device=q.device, dtype=torch.float32)
    _prefill_kernel[(NH, ntq)](q, r, projE, kbuf, vbuf, sink, gate, out, scr, T, hist, HD ** -0.5,
                                              WIN=WIN, BQ=bq, BN=bn, HD=HD, G=G, NH=NH, NKV=NKV, BIAS=int(bias), W=W, num_warps=num_warps)
    return out


# ---------------------------------------------------------------------------------------------------- in-place ring append
@triton.jit
def _append_kernel(KN, VN, KR, VR, POS, T, skr_slot, skr_head, svr_slot, svr_head, R: tl.constexpr, NKV: tl.constexpr, HD: tl.constexpr):
    t = tl.program_id(0)
    pos0 = tl.load(POS)
    x = tl.arange(0, NKV * HD)
    h = x // HD
    c = x % HD
    slot = (pos0 + t) % R
    live = (t + R) >= T                            # prefill chunk longer than the ring: only the last R rows are written (no two rows share a slot in one launch)
    kv = tl.load(KN + t * (NKV * HD) + x)
    vv = tl.load(VN + t * (NKV * HD) + x)
    tl.store(KR + slot * skr_slot + h * skr_head + c, kv, mask=live)
    tl.store(VR + slot * svr_slot + h * svr_head + c, vv, mask=live)


def lattn_append(k_new, v_new, kring, vring, pos, ring_rows=1088):
    """Write the T new K / V rows ([T,8,128] bf16, after conv + clip / rms) at slots (pos0 + t) % R of the ring, in place; pos0 = device int32 tensor (graph safe).
    Spec rollback = reset pos0 (never read beyond). Prefill chunks longer than R write only their last R rows."""
    T = k_new.shape[0]
    sd = 0 if kring.shape[1] == NKV else 1
    _append_kernel[(T,)](k_new, v_new, kring, vring, pos, T, kring.stride(sd), kring.stride(1 - sd), vring.stride(sd), vring.stride(1 - sd),
                         R=ring_rows, NKV=NKV, HD=HD, num_warps=4)
