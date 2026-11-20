"""HY4_KV_FP8=1: latent K cache of the full-attention (DSA) layers stored as fp8 e4m3 (512 latent dims, ONE fp32 scale per token) + bf16 rope dims
(DeepSeek V3.2 style): 512 + 4 + 128 = 644 B per token and layer instead of 2304 (fp32). Dequantised on gather: the sparse decode kernel reads the bytes and
decodes e4m3 itself (integer maths, no fp8 hardware / Triton fp8 types), prefill dequantises the keys of the call to fp16 / fp32.
Acts like a [n, R+P] fp32 K buffer for the few things callers do with one: .shape, row slices, write rows, zero_."""
import torch

E4M3_MAX = 448.0


def quant_rows(kn, R):
    """kn [T, R+P] fp32 -> (lat8 [T, R] uint8 (e4m3 bits), sc [T] fp32, pe [T, P] bf16)"""
    lat = kn[:, :R].float()
    sc = (lat.abs().amax(-1) / E4M3_MAX).clamp_min(1e-20)
    q = (lat / sc[:, None]).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).view(torch.uint8)
    return q, sc, kn[:, R:].to(torch.bfloat16)


def dq8(u8):
    """uint8 e4m3fn bits -> fp32 (same maths as the Triton kernel)"""
    b = u8.to(torch.int32)
    s, e, m = (b >> 7) & 1, (b >> 3) & 15, b & 7
    v = torch.where(e == 0, m.float() * 2.0 ** -9, (8 + m).float() * torch.exp2((e - 10).float()))
    return torch.where(s == 1, -v, v)


class Fp8KV:
    dtype = torch.float32          # what callers think they hold

    def __init__(self, cap, R, P, dev, lat=None, sc=None, pe=None):
        self.R, self.P = R, P
        self.lat = lat if lat is not None else torch.zeros(cap, R, dtype=torch.uint8, device=dev)
        self.sc = sc if sc is not None else torch.zeros(cap, dtype=torch.float32, device=dev)
        self.pe = pe if pe is not None else torch.zeros(cap, P, dtype=torch.bfloat16, device=dev)
        self.device = self.lat.device

    @property
    def shape(self):
        return (self.lat.shape[0], self.R + self.P)

    def __len__(self):
        return self.lat.shape[0]

    def __getitem__(self, sl):
        assert isinstance(sl, slice), "Fp8KV supports row slices only"
        return Fp8KV(0, self.R, self.P, self.device, self.lat[sl], self.sc[sl], self.pe[sl])

    def nbytes(self):
        return self.lat.numel() + 4 * self.sc.numel() + 2 * self.pe.numel()

    def zero_(self):
        self.lat.zero_(); self.sc.zero_(); self.pe.zero_()
        return self

    def write(self, start, kn):
        q, sc, pe = quant_rows(kn, self.R)
        T = kn.shape[0]
        self.lat[start:start + T] = q; self.sc[start:start + T] = sc; self.pe[start:start + T] = pe

    def write_pos(self, pos, kn):
        """pos [T] long device (graph-capturable)"""
        q, sc, pe = quant_rows(kn, self.R)
        self.lat.index_copy_(0, pos, q); self.sc.index_copy_(0, pos, sc); self.pe.index_copy_(0, pos, pe)

    def load_f32(self, b):
        """fill from an fp32 buffer [n, R+P]"""
        self.write(0, b)

    def dequant32(self):
        return torch.cat([dq8(self.lat) * self.sc[:, None], self.pe.float()], -1)

    def dequant16(self):
        """(lat16 [n, R], pe16 [n, P]) fp16, clamped like the fp32 -> fp16 conversion of the CK path"""
        return (dq8(self.lat) * self.sc[:, None]).clamp(-65504.0, 65504.0).half().contiguous(), self.pe.float().clamp(-65504.0, 65504.0).half().contiguous()

    def dequant16_cat(self, chunk=65536):
        """the whole cache as ONE fp16 [n, R+P] tensor (latent * scale, clamped like the fp32 -> fp16 cast; rope as is): the sparse prefill kernel reads it.
        torch fp8 cast (exact, same values as dq8), chunked so the fp32 temporaries stay small at 1M rows."""
        n, R = self.lat.shape[0], self.R
        out = torch.empty(n, R + self.P, dtype=torch.float16, device=self.device)
        for s in range(0, n, chunk):
            e = min(n, s + chunk)
            out[s:e, :R] = (self.lat[s:e].view(torch.float8_e4m3fn).float() * self.sc[s:e, None]).clamp(-65504.0, 65504.0).half()
            out[s:e, R:] = self.pe[s:e].float().clamp(-65504.0, 65504.0).half()
        return out
