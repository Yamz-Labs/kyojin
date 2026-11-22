"""Inkling2 attention-projection weight formats (pure torch, CPU or GPU): the load-time quantisers / repacks and their exact dequant.

  Q8   int8 data [N, K] (natural order) + fp16 scales [N, K/32]        == GGUF type 8 (Q8_0) bytes split in two planes (34 B per 32 values)
  NV   nibbles u8 [N, K/2] (unit order, see ink2_attnproj.hip) + UE4M3 scale bytes u8 [N, K/16] + one fp32 `gs` per tensor   (0.5625 B per value)
       value = sign * E2M1(mag) * ue4m3(scale) * gs;  E2M1 magnitudes 0 .5 1 1.5 2 3 4 6; ue4m3 = fp8 e4m3fn positive half (0x7f = NaN is never produced)
  BF   bf16 as stored

All functions take / return torch tensors; `dq_*` are the references the kernel is checked against (float64 exact decode of the stored planes).
`fake_quant(w, fmt)` = dq(quant(w)): the whole-model quality arm (INK2_ATTN_FQ) swaps the dense attention weights for it."""
import torch

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float64)
E2M1_MID = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float64)


# ------------------------------------------------------------------------------------------------------------------------------ Q8_0
def quant_q8(w):
    """fp32 [N, K] -> (data int8 [N, K], scales fp16 [N, K/32]): ggml Q8_0 (d = amax / 127 stored fp16, q = round(w / d))"""
    N, K = w.shape
    g = w.float().reshape(N, K // 32, 32)
    d = (g.abs().amax(-1) / 127.0).half()
    df = d.float()
    q = torch.where(df[..., None] > 0, g / df[..., None].clamp_min(1e-30), torch.zeros_like(g)).round().clamp(-127, 127).to(torch.int8)
    return q.reshape(N, K).contiguous(), d.contiguous()


def split_q8_gguf(raw):
    """GGUF Q8_0 bytes uint8 [N, nb*34] -> (data int8 [N, K], scales fp16 [N, nb]); a pure byte shuffle, the values are identical"""
    N = raw.shape[0]
    b = raw.reshape(N, -1, 34)
    d = b[..., 0:2].contiguous().view(torch.float16).reshape(N, -1)
    q = b[..., 2:].contiguous().view(torch.int8).reshape(N, -1)
    return q.contiguous(), d.contiguous()


def dq_q8(q, d):
    N, K = q.shape
    return (q.double().reshape(N, K // 32, 32) * d.double()[..., None]).reshape(N, K)


# ------------------------------------------------------------------------------------------------------------------------------ NVFP4
def ue4m3(b):
    """UE4M3 bytes (uint8, sign bit ignored) -> float64"""
    b = b.long() & 0x7F
    ex, ma = (b >> 3) & 15, b & 7
    v = torch.where(ex == 0, ma.double() / 8.0 * 2.0 ** -6, (1.0 + ma.double() / 8.0) * torch.pow(2.0, (ex - 7).double()))
    return torch.where(b == 0x7F, torch.full_like(v, float("nan")), v)


def _nearest_mag(a):
    """|value| / scale (float64) -> E2M1 magnitude index 0..7 (ties go up)"""
    return torch.bucketize(a, E2M1_MID.to(a.device), right=True)


def quant_nvfp4(w, search=True):
    """fp32 [N, K] -> (nib u8 [N, K/2], scale u8 [N, K/16], gs float, wq float32 [N, K] = the exact reconstruction).
    gs = amax / (6 * 448) so the largest group scale is 448; per 16-group the UE4M3 scale is the nearest of {rn, up, down} minimising the group squared error."""
    N, K = w.shape
    g = w.double().reshape(N, K // 16, 16)
    amax = g.abs().amax(-1)
    gmax = float(amax.max())
    gs = gmax / (6.0 * 448.0) if gmax > 0 else 1.0
    target = (amax / (6.0 * gs)).clamp(max=448.0)
    s8 = target.float().to(torch.float8_e4m3fn).view(torch.uint8)
    cands = [s8]
    if search:
        cands += [torch.clamp(s8.int() + 1, max=0x7E).to(torch.uint8), torch.clamp(s8.int() - 1, min=0).to(torch.uint8)]
    best = None
    for sb in cands:
        sval = ue4m3(sb) * gs
        sv = sval[..., None]
        mag = _nearest_mag(torch.where(sv > 0, g.abs() / sv.clamp_min(1e-300), torch.zeros_like(g)))
        rec = torch.sign(g) * E2M1.to(g.device)[mag] * sv
        err = (rec - g).square().sum(-1)
        if best is None:
            best = [err, sb, mag, rec]
        else:
            m = err < best[0]
            best = [torch.where(m, err, best[0]), torch.where(m, sb, best[1]), torch.where(m[..., None], mag, best[2]), torch.where(m[..., None], rec, best[3])]
    _, sb, mag, rec = best
    neg = (g < 0) & (mag > 0)
    code = (mag + 8 * neg.long()).reshape(N, K)                      # 4-bit codes, natural order
    return pack_nibbles(code), sb.reshape(N, K // 16).contiguous(), gs, rec.reshape(N, K).float()


def pack_nibbles(code):
    """codes int64 [N, K] (0..15) -> u8 [N, K/2] in the kernel's unit order: per 32-value unit, dword d (4 per unit) holds value e = 8d + 2o + m at bit 4o + 16m"""
    N, K = code.shape
    c = code.reshape(N, K // 32, 4, 4, 2)                            # [N, unit, d, o, m]  (e = 8 d + 2 o + m)
    o = torch.arange(4).reshape(1, 1, 1, 4, 1)
    m = torch.arange(2).reshape(1, 1, 1, 1, 2)
    dw = (c << (4 * o + 16 * m)).sum((-1, -2))                       # [N, unit, d] int64 (< 2^32)
    dw = dw.to(torch.int64)
    b = torch.stack([(dw >> (8 * i)) & 0xFF for i in range(4)], -1)  # little endian bytes [N, unit, d, 4]
    return b.reshape(N, K // 2).to(torch.uint8).contiguous()


def unpack_nibbles(nib):
    """inverse of pack_nibbles: u8 [N, K/2] -> codes int64 [N, K] (natural order)"""
    N, kh = nib.shape
    b = nib.long().reshape(N, kh // 16, 4, 4)                        # [N, unit, d, byte]
    dw = b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16) | (b[..., 3] << 24)
    o = torch.arange(4).reshape(1, 1, 1, 4, 1)
    m = torch.arange(2).reshape(1, 1, 1, 1, 2)
    c = (dw[..., None, None] >> (4 * o + 16 * m)) & 15               # [N, unit, d, o, m]
    return c.reshape(N, kh * 2)


def dq_nvfp4(nib, scale, gs):
    N, kh = nib.shape
    K = kh * 2
    code = unpack_nibbles(nib)
    mag = E2M1.to(nib.device)[code & 7]
    val = torch.where((code & 8) > 0, -mag, mag)
    sv = ue4m3(scale)
    return (val.reshape(N, K // 16, 16) * sv[..., None]).reshape(N, K) * gs


def gguf_nvfp4_to_planes(raw):
    """GGUF type 40 bytes [N, nb*36] -> (nib u8 [N, K/2] in the unit order above, scale u8 [N, K/16], gs = 1 * 0.5 folded as 0.5 scale factor).
    ggml: value = ue4m3(scale) * 0.5 * kvalues[nibble] with kvalues = 2 x E2M1, i.e. ue4m3(scale) * E2M1: gs = 1."""
    N = raw.shape[0]
    b = raw.reshape(N, -1, 36)
    scale = b[..., :4].reshape(N, -1).contiguous()
    qs = b[..., 4:].reshape(N, -1, 4, 8)                             # 16-group g: 8 bytes, low nibble = value j, high = value j + 8
    code = torch.cat([qs.long() & 15, qs.long() >> 4], -1).reshape(N, -1)
    # ggml kvalues index order is the E2M1 code order (sign bit 3): the nibble IS the E2M1 code
    return pack_nibbles(code), scale, 1.0


# ------------------------------------------------------------------------------------------------------------------------------ whole-model arm
def fake_quant(w, fmt):
    """dense fp32 weight [N, K] -> the weight the kernel would read (fp32)"""
    if fmt == "q8":
        q, d = quant_q8(w)
        return dq_q8(q, d).float()
    if fmt == "nvfp4":
        return quant_nvfp4(w)[3]
    return w
