"""Torch dequant of the ggml block formats the CUDA/HIP fq_* kernels do not cover: Q4_0, Q5_0, IQ4_NL, IQ3_XXS, IQ3_S.

Input is the raw GGUF payload as uint8 [N, nb * block_bytes] (row-major GGUF order, K along the row, same shape the
GGUFCollection serves for the K-quants); output is fp16 [N, K]. Layouts follow ggml-quants.h and are checked
against the gguf-py reference in tests/test_foreign_gap_formats_cpu.py. Runs on any torch device.
"""
from __future__ import annotations

import torch

from .gguf_tables import IQ3_S_GRID, IQ3_XXS_GRID, IQ4_NL_VALUES, KSIGNS

# name -> (elements per block, bytes per block)
BLOCKS = {"q8_0": (32, 34), "q4_0": (32, 18), "q5_0": (32, 22), "q5_1": (32, 24), "iq4_nl": (32, 18), "iq3_xxs": (256, 98), "iq3_s": (256, 110), "iq2_xs": (256, 74), "iq2_s": (256, 82), "q3_k": (256, 110), "q2_k": (256, 84),
          "mxfp4": (32, 17), "nvfp4_gguf": (64, 36)}
GGML_TYPE_IDS = {"q8_0": 8, "q4_0": 2, "q5_0": 6, "q5_1": 7, "iq3_xxs": 18, "iq4_nl": 20, "iq3_s": 21, "mxfp4": 39, "nvfp4_gguf": 40, "iq2_xs": 17, "iq2_s": 22, "q3_k": 11, "q2_k": 10}
# MXFP4 / NVFP4 nibble -> value in half units (the scale carries the other factor 0.5)
_FP4_HALF = (0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12)

_TABLES: dict = {}


def _table(name: str, device) -> torch.Tensor:
    key = (name, str(device))
    if key not in _TABLES:
        if name == "iq4_nl":
            t = torch.tensor(IQ4_NL_VALUES, dtype=torch.float32)
        elif name == "fp4":
            t = torch.tensor(_FP4_HALF, dtype=torch.float32)
        elif name == "iq3_xxs":
            t = torch.frombuffer(bytearray(IQ3_XXS_GRID), dtype=torch.uint8).reshape(256, 4).float()
        elif name == "iq3_s":
            t = torch.frombuffer(bytearray(IQ3_S_GRID), dtype=torch.uint8).reshape(512, 4).float()
        elif name in ("iq2_xs", "iq2_s"):
            from .iq_grids_gen import iq2_grid
            t = torch.from_numpy(iq2_grid(name[4:]).copy()).float()          # [512 | 1024, 8]
        else:
            t = torch.frombuffer(bytearray(KSIGNS), dtype=torch.uint8).long()
        _TABLES[key] = t.to(device)
    return _TABLES[key]


def _f16(b: torch.Tensor) -> torch.Tensor:
    return b.contiguous().view(torch.float16).float()          # [..., 2] bytes -> [..., 1]


def _bits(x: torch.Tensor, n: int) -> torch.Tensor:
    """[...] integer -> [..., n] bits, least significant first."""
    sh = torch.arange(n, device=x.device, dtype=torch.int64)
    return (x.long().unsqueeze(-1) >> sh) & 1


def _blocks(raw: torch.Tensor, fmt: str) -> torch.Tensor:
    qk, bb = BLOCKS[fmt]
    n, w = raw.shape
    assert raw.dtype == torch.uint8 and w % bb == 0, (fmt, tuple(raw.shape))
    return raw.reshape(n * (w // bb), bb)


def dequant(fmt: str, raw: torch.Tensor) -> torch.Tensor:
    """raw uint8 [N, nb*bb] -> fp16 [N, nb*qk]."""
    qk, bb = BLOCKS[fmt]
    n = raw.shape[0]
    b = _blocks(raw, fmt)
    nb = b.shape[0]
    if fmt == "q8_0":       # [fp16 d][32 x int8]; torch fallback for builds without the fq_* extension kernels (ROCm)
        d = b[:, 0:2].contiguous().view(torch.float16).float()
        return (d * b[:, 2:34].contiguous().view(torch.int8).float()).reshape(n, -1).half()
    if fmt == "mxfp4":      # [e8m0 scale][16 bytes: elements 0..15 low nibbles, 16..31 high nibbles]
        e = b[:, 0:1].float()
        d = torch.exp2(e - 128.0)                                # ggml_e8m0_to_fp32_half
        qs = b[:, 1:17]
        v = _table("fp4", raw.device)[torch.cat([qs & 15, qs >> 4], dim=1).long()]
        return (d * v).reshape(n, -1).half()
    if fmt == "nvfp4_gguf":  # 64 values: [4 x UE4M3 scale (one per 16)][4 x 8 bytes: low nibbles = first 8, high = last 8]
        x = b[:, 0:4].long()
        ex = (x >> 3) & 15
        mt = (x & 7).float()
        sc = torch.where(ex == 0, mt * 2.0 ** -9, (1.0 + mt / 8.0) * torch.exp2(ex.float() - 7.0))
        sc = torch.where((x == 0) | (x == 0x7F), torch.zeros_like(sc), sc * 0.5)          # [nb, 4]
        qs = b[:, 4:36].reshape(nb, 4, 8)
        v = _table("fp4", raw.device)[torch.cat([qs & 15, qs >> 4], dim=2).long()]         # [nb, 4, 16]
        return (sc.unsqueeze(-1) * v).reshape(n, -1).half()
    if fmt == "q2_k":       # [scales 16 (low 4 scale, high 4 min)][qs 64][d fp16][dmin fp16] (ggml dequantize_row_q2_K)
        d, dm = _f16(b[:, 80:82]).unsqueeze(-1), _f16(b[:, 82:84]).unsqueeze(-1)       # [nb, 1, 1, 1]
        sc = b[:, 0:16].long().reshape(nb, 2, 4, 2).repeat_interleave(16, dim=3)          # [nb, n, j, 32]: scale of 16-weight half
        q = (b[:, 16:80].long().reshape(nb, 2, 1, 32) >> (2 * torch.arange(4, device=raw.device)).reshape(1, 1, 4, 1)) & 3
        return (d.unsqueeze(-1) * (sc & 15).float() * q.float() - dm.unsqueeze(-1) * (sc >> 4).float()).reshape(n, -1).half()
    if fmt == "q3_k":       # [hmask 32][qs 64][scales 12][d fp16] (ggml dequantize_row_q3_K)
        d = _f16(b[:, 108:110])                                 # [nb, 1]
        hm = b[:, 0:32].long()
        qs = b[:, 32:96].long()
        a3 = b[:, 96:108].contiguous().view(torch.int32).long() & 0xFFFFFFFF   # [nb, 3]
        k1, k2 = 0x03030303, 0x0F0F0F0F
        t = a3[:, 2]
        aux = torch.stack([(a3[:, 0] & k2) | (((t >> 0) & k1) << 4), (a3[:, 1] & k2) | (((t >> 2) & k1) << 4),
                           ((a3[:, 0] >> 4) & k2) | (((t >> 4) & k1) << 4), ((a3[:, 1] >> 4) & k2) | (((t >> 6) & k1) << 4)], 1)
        sc = ((aux.unsqueeze(-1) >> (torch.arange(4, device=raw.device) * 8)) & 255).reshape(nb, 16)
        sc = (sc - 256 * (sc >= 128).long() - 32).float()       # int8 - 32
        out = torch.empty((nb, 256), dtype=torch.float32, device=raw.device)
        i = 0
        for hh in range(2):
            q = qs[:, 32 * hh:32 * hh + 32]
            for j in range(4):
                sh, m = 2 * j, 1 << (4 * hh + j)
                for half in range(2):
                    sl = slice(16 * half, 16 * half + 16)
                    v = ((q[:, sl] >> sh) & 3) - torch.where((hm[:, sl] & m) != 0, 0, 4)
                    o = 128 * hh + 32 * j + 16 * half
                    out[:, o:o + 16] = d * sc[:, i:i + 1] * v.float()
                    i += 1
        return out.reshape(raw.shape[0], -1).half()
    if fmt == "iq2_xs":     # [d fp16][qs 32 x uint16: 9-bit grid index | 7-bit sign index][scales 8]
        d = _f16(b[:, 0:2])
        q = b[:, 2:66].contiguous().view(torch.int16).long() & 0xFFFF       # [nb, 32]
        scb = b[:, 66:74].long()                                # [nb, 8]
        sc = torch.stack([scb & 15, scb >> 4], -1).reshape(nb, 16).float()  # one per 16 = per pair of groups of 8
        db = d * (0.5 + sc) * 0.25                              # [nb, 16]
        grid = _table("iq2_xs", raw.device)[q & 511]            # [nb, 32, 8]
        sgn = 1.0 - 2.0 * _bits(_table("ksigns", raw.device)[q >> 9], 8).float()
        out = (db.repeat_interleave(2, 1).unsqueeze(-1) * grid * sgn).reshape(nb, 256)
        return out.reshape(raw.shape[0], -1).half()
    if fmt == "iq2_s":      # [d fp16][qs 32 grid low bytes][signs 32][qh 8][scales 8]
        d = _f16(b[:, 0:2])
        ql = b[:, 2:34].long()
        sg = b[:, 34:66]
        qh = b[:, 66:74].long()                                 # [nb, 8]
        scb = b[:, 74:82].long()
        sc = torch.stack([scb & 15, scb >> 4], -1).reshape(nb, 16).float()
        db = d * (0.5 + sc) * 0.25                              # [nb, 16]: groups l 0,1 -> low nibble, l 2,3 -> high
        l = torch.arange(4, device=raw.device)
        hi = ((qh.unsqueeze(-1) << (8 - 2 * l)) & 0x300).reshape(nb, 32)
        grid = _table("iq2_s", raw.device)[ql | hi]             # [nb, 32, 8]
        sgn = 1.0 - 2.0 * _bits(sg, 8).float()                  # [nb, 32, 8]
        out = (db.repeat_interleave(2, 1).unsqueeze(-1) * grid * sgn).reshape(nb, 256)
        return out.reshape(raw.shape[0], -1).half()
    d = _f16(b[:, 0:2])                                         # [nb, 1]
    if fmt == "q5_1":       # [d fp16][m fp16][qh u32][qs 16]: x = d * (low nibble | high bit << 4) + m
        m = _f16(b[:, 2:4])
        qs = b[:, 8:24]
        q = torch.cat([qs & 15, qs >> 4], dim=1).long()
        qh = b[:, 4:8].contiguous().view(torch.int32).long() & 0xFFFFFFFF
        hb = _bits(qh.reshape(nb), 32)
        out = d * (q | (hb << 4)).float() + m
        return out.reshape(raw.shape[0], -1).half()
    if fmt in ("q4_0", "iq4_nl", "q5_0"):
        qs = b[:, 2:18] if fmt != "q5_0" else b[:, 6:22]
        q = torch.cat([qs & 15, qs >> 4], dim=1).long()         # elements 0..15 low nibbles, 16..31 high nibbles
        if fmt == "q4_0":
            v = (q - 8).float()
        elif fmt == "iq4_nl":
            v = _table("iq4_nl", raw.device)[q]
        else:
            qh = b[:, 2:6].contiguous().view(torch.int32).long() & 0xFFFFFFFF   # [nb, 1]
            hb = _bits(qh.reshape(nb), 32)                      # [nb, 32]
            v = ((q | (hb << 4)) - 16).float()
        out = d * v
    elif fmt == "iq3_xxs":
        qs = b[:, 2:66].long()                                  # [nb, 64] grid indices
        sc = b[:, 66:98].contiguous().view(torch.int32).long() & 0xFFFFFFFF   # [nb, 8]
        db = d.reshape(nb, 1) * (0.5 + (sc >> 28).float()) * 0.5                # [nb, 8]
        sh = torch.arange(4, device=raw.device) * 7     # on-device (no host copy inside a graph capture)
        sidx = (sc.unsqueeze(-1) >> sh) & 0x7F                  # [nb, 8, 4]
        sbits = _bits(_table("ksigns", raw.device)[sidx], 8)    # [nb, 8, 4, 8]
        sgn = 1.0 - 2.0 * sbits.float()
        grid = _table("iq3_xxs", raw.device)[qs].reshape(nb, 8, 4, 8)
        out = (db.reshape(nb, 8, 1, 1) * grid * sgn).reshape(nb, 256)
    else:   # iq3_s
        qs = b[:, 2:66].long()
        qh = b[:, 66:74].long()                                 # [nb, 8]
        sg = b[:, 74:106]                                       # [nb, 32]
        scb = b[:, 106:110]                                     # [nb, 4]
        sc = torch.stack([(scb & 15), (scb >> 4)], dim=-1).reshape(nb, 8).float()
        db = d.reshape(nb, 1) * (1.0 + 2.0 * sc)
        sgn = (1.0 - 2.0 * _bits(sg, 8).float()).reshape(nb, 8, 4, 8)
        idx = qs | (_bits(qh, 8).reshape(nb, 64) << 8)
        grid = _table("iq3_s", raw.device)[idx].reshape(nb, 8, 4, 8)
        out = (db.reshape(nb, 8, 1, 1) * grid * sgn).reshape(nb, 256)
    return out.reshape(n, -1).half()
