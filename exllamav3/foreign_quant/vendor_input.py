"""Vendor checkpoint INPUT formats for the EXL3 converters (numpy only, no torch, one file).

Reads the quantized checkpoints vendors ship and dequantizes them to float32 exactly per the
vendor spec, so a converter (CUDA/ROCm tools/bigconv + Linear loader, Mac conv_mlx) can requantize
to EXL3 from them. This file is self-contained and is copied verbatim to kyojin-mlx
(`kyojin_mlx/vendor_input.py`): keep it free of any exllamav3 import.

Formats (value = decoded payload x scale, all in float32):

  dense   BF16 / F16 / F32 `P.weight`.
  fp8     `P.weight` F8_E4M3 (E5M2 too) with a scale tensor, multiplied (never divided):
            `P.weight_scale_inv`  DeepSeek / Qwen3-FP8 / GLM / MiniMax: 2D grid, block 128x128, ceil-div
            `P.weight_scale`      per-tensor ([] or [1]), per-channel ([out] / [out,1]) or 2D grid
            `P.scale`             DeepSeek-V4: F8_E8M0 grid (32x32 in that checkpoint)
          a U8 / F8_E8M0 scale on FP8 weights is MXFP8 (MiniMax-M3): E8M0 exponent per 32 inputs, [out, in/32]
          no scale tensor -> scale 1 (plain cast).
  mxfp4   OCP microscaling E2M1 + E8M0 per 32 inputs, two nibbles per byte, LOW nibble first:
            linear:  `P.weight` I8/U8 [out,in/2] + `P.scale` F8_E8M0 or `P.weight_scale` U8/E8M0 [out,in/32]
                     (DeepSeek-V4, MiMo-V2), or compressed-tensors `P.weight_packed` + `P.weight_scale`
            blocks:  gpt-oss `P_blocks` U8 [...,G,16] + `P_scales` U8 [...,G] (fused expert tensors)
  nvfp4   E2M1 + E4M3 scale per 16 inputs + one fp32 global scale:
            ModelOpt:           `P.weight` U8 [out,in/2], `P.weight_scale` F8_E4M3 [out,in/16],
                                `P.weight_scale_2` F32 [] ; w = (e2m1 * scale) * weight_scale_2
            compressed-tensors: `P.weight_packed`, `P.weight_scale`, `P.weight_global_scale` F32 [1];
                                w = (e2m1 * scale) / weight_global_scale
          `input_scale` / `input_global_scale` (activation scales) are ignored.

Codebooks are built from the format definitions (OCP MX v1.0 / OCP FP8) and checked against
ml_dtypes and torch in the tests. E2M1 x E4M3 / E8M0 products are exact in fp32, so every format
rounds at most once (the global scale of NVFP4, or an fp32 block scale of FP8).
"""
from __future__ import annotations

import json
import mmap
import os
import struct

import numpy as np

# ----------------------------------------------------------------------------------------------
# codebooks
# ----------------------------------------------------------------------------------------------

E2M1_LUT = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                    dtype=np.float32)


def _build_fp8_lut(ebits: int, mbits: int, bias: int, e4m3fn: bool) -> np.ndarray:
    lut = np.zeros(256, dtype=np.float64)
    for b in range(256):
        s = -1.0 if b & 0x80 else 1.0
        e = (b >> mbits) & ((1 << ebits) - 1)
        m = b & ((1 << mbits) - 1)
        emax = (1 << ebits) - 1
        if e4m3fn:
            if e == emax and m == (1 << mbits) - 1:
                v = float("nan")
            elif e == 0:
                v = m * 2.0 ** (1 - bias - mbits)
            else:
                v = (1.0 + m / (1 << mbits)) * 2.0 ** (e - bias)
        else:                                            # IEEE-like (E5M2): top exponent = inf / nan
            if e == emax:
                v = float("inf") if m == 0 else float("nan")
            elif e == 0:
                v = m * 2.0 ** (1 - bias - mbits)
            else:
                v = (1.0 + m / (1 << mbits)) * 2.0 ** (e - bias)
        lut[b] = s * v if v == v else v
    return lut.astype(np.float32)


E4M3FN_LUT = _build_fp8_lut(4, 3, 7, True)      # OCP FP8 E4M3 (finite, NaN only at 0x7F / 0xFF), max 448
E5M2_LUT = _build_fp8_lut(5, 2, 15, False)
E8M0_LUT = np.array([np.float32(2.0 ** (b - 127)) if b < 255 else np.float32("nan") for b in range(256)],
                    dtype=np.float32)            # b = 0 -> 2^-127 is a float32 subnormal, exact


def e4m3_to_f32(u8):
    return E4M3FN_LUT[np.asarray(u8, dtype=np.uint8)]


def e5m2_to_f32(u8):
    return E5M2_LUT[np.asarray(u8, dtype=np.uint8)]


def e8m0_to_f32(u8):
    return E8M0_LUT[np.asarray(u8, dtype=np.uint8)]


def bf16_to_f32(u16):
    return (np.asarray(u16, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16_bits(x):
    """float32 -> bf16 bit pattern (uint16), round to nearest even, NaN kept."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    r = ((u >> 16) & 1) + np.uint32(0x7FFF)
    out = ((u + r) >> 16).astype(np.uint16)
    nan = np.isnan(x)
    if nan.any():
        out[nan] = 0x7FC0
    return out


def unpack_e2m1(packed):
    """uint8/int8 [..., n/2] -> float32 [..., n], LOW nibble first."""
    p = np.ascontiguousarray(packed).view(np.uint8)
    out = np.empty(p.shape + (2,), dtype=np.float32)
    out[..., 0] = E2M1_LUT[p & 0xF]
    out[..., 1] = E2M1_LUT[p >> 4]
    return out.reshape(p.shape[:-1] + (p.shape[-1] * 2,))


# ----------------------------------------------------------------------------------------------
# safetensors (mmap, raw bytes: F8 / BF16 are not numpy dtypes)
# ----------------------------------------------------------------------------------------------

_NP = {"F64": np.float64, "F32": np.float32, "F16": np.float16, "BF16": np.uint16, "I64": np.int64, "I32": np.int32,
       "I16": np.int16, "I8": np.int8, "U8": np.uint8, "BOOL": np.uint8, "U16": np.uint16, "U32": np.uint32,
       "U64": np.uint64, "F8_E4M3": np.uint8, "F8_E5M2": np.uint8, "F8_E8M0": np.uint8}
MAX_HEADER = 100 * 1024 ** 2


class SafetensorsFile:
    def __init__(self, path):
        self.path = path
        self._f = open(path, "rb")
        size = os.fstat(self._f.fileno()).st_size
        n = struct.unpack("<Q", self._f.read(8))[0]
        if n > MAX_HEADER or 8 + n > size:
            raise ValueError(f"{path}: bad safetensors header length {n}")
        hdr = json.loads(self._f.read(n))
        self.meta = hdr.pop("__metadata__", {})
        self.base = 8 + n
        self.header = hdr
        for k, m in hdr.items():
            b, e = m["data_offsets"]
            if m["dtype"] in _NP and e - b != int(np.prod(m["shape"], dtype=np.int64)) * np.dtype(_NP[m["dtype"]]).itemsize:
                raise ValueError(f"{path}:{k}: {m['dtype']}{m['shape']} does not span {e - b} bytes")
            if self.base + e > size:
                raise ValueError(f"{path}:{k}: data past end of file")
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ) if size else None

    def close(self):
        # numpy views of the map keep it alive; dropping our reference is enough
        self._mm = None
        self._f.close()

    def info(self):
        return {k: (m["dtype"], tuple(m["shape"])) for k, m in self.header.items()}

    def raw(self, name):
        """(dtype tag, shape, numpy view in the storage dtype: uint8 for F8_*, uint16 for BF16). Zero-copy."""
        m = self.header[name]
        b, e = m["data_offsets"]
        a = np.frombuffer(self._mm, dtype=_NP[m["dtype"]], count=(e - b) // np.dtype(_NP[m["dtype"]]).itemsize,
                          offset=self.base + b)
        return m["dtype"], tuple(m["shape"]), a.reshape(m["shape"])


def scale_to_f32(tag, a):
    """Decode a scale tensor: U8 scales (MXFP8 `weight_scale_inv`, MXFP4 `weight_scale`) are E8M0 exponent bytes."""
    return e8m0_to_f32(a) if tag == "U8" else to_f32(tag, a)


def to_f32(tag, a):
    """Decode a raw storage array of safetensors dtype `tag` to float32."""
    if tag == "F32":
        return np.asarray(a, dtype=np.float32)
    if tag == "BF16":
        return bf16_to_f32(a)
    if tag == "F8_E4M3":
        return e4m3_to_f32(a)
    if tag == "F8_E5M2":
        return e5m2_to_f32(a)
    if tag == "F8_E8M0":
        return e8m0_to_f32(a)
    if tag in ("F16", "F64"):
        return a.astype(np.float32)
    raise ValueError(f"cannot decode {tag} as a float")


class Checkpoint:
    """A directory of *.safetensors (+ config.json). Header-only until a tensor is read."""

    def __init__(self, path):
        self.dir = path
        self.config = json.load(open(os.path.join(path, "config.json"))) if os.path.exists(os.path.join(path, "config.json")) else {}
        idx = os.path.join(path, "model.safetensors.index.json")
        files = sorted({f for f in json.load(open(idx))["weight_map"].values()}) if os.path.exists(idx) \
            else sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))
        self.files = {f: SafetensorsFile(os.path.join(path, f)) for f in files}
        self.where = {k: f for f, st in self.files.items() for k in st.header}
        self._info = None

    def info(self):
        if self._info is None:
            self._info = {k: v for st in self.files.values() for k, v in st.info().items()}
        return self._info

    def has(self, name):
        return name in self.where

    def raw(self, name):
        return self.files[self.where[name]].raw(name)

    def f32(self, name):
        tag, _, a = self.raw(name)
        return to_f32(tag, a)

    def scale_f32(self, name):
        tag, _, a = self.raw(name)
        return scale_to_f32(tag, a)

    def close(self):
        for st in self.files.values():
            st.close()


# ----------------------------------------------------------------------------------------------
# dequant kernels (float32 in, float32 out)
# ----------------------------------------------------------------------------------------------

def _axis_block(n, s, block):
    """Block length along an axis of n elements covered by s scales."""
    if block and -(-n // block) == s:
        return block
    if n % s == 0:
        return n // s
    raise ValueError(f"scale grid {s} does not cover {n} elements (block {block})")


def apply_scales(w, s, block=None):
    """w float32 [out,in] x scale float32 (scalar / [out] / [out,1] / [sr,sc] grid, ceil-div blocks) -> float32.

    `block` = (bo, bi) from the config (`weight_block_size`) when the grid is not exactly divisible."""
    w = np.asarray(w, dtype=np.float32)
    s = np.asarray(s, dtype=np.float32)
    out, in_ = w.shape
    if s.size == 1:
        return w * s.reshape(())
    if s.ndim == 1:
        if s.shape[0] == out:
            return w * s[:, None]
        if s.shape[0] == in_:
            return w * s[None, :]
        raise ValueError(f"1D scale {s.shape} fits neither out={out} nor in={in_}")
    if s.ndim != 2:
        raise ValueError(f"scale rank {s.ndim}")
    sr, sc = s.shape
    bo = _axis_block(out, sr, block[0] if block else None)
    bi = _axis_block(in_, sc, block[1] if block else None)
    if out % bo == 0 and in_ % bi == 0:
        r = w.reshape(sr, bo, sc, bi) * s[:, None, :, None]
        return r.reshape(out, in_)
    full = np.repeat(np.repeat(s, bo, axis=0), bi, axis=1)[:out, :in_]
    return w * full


def dequant_fp8(w_u8, w_tag, scale_f32, block=None):
    wf = to_f32(w_tag, w_u8)
    return wf if scale_f32 is None else apply_scales(wf, scale_f32, block)


def dequant_mxfp4_linear(packed, scale_f32):
    """packed uint8/int8 [out,in/2], scale float32 [out,in/32] (decoded E8M0) -> float32 [out,in]."""
    return apply_scales(unpack_e2m1(packed), scale_f32)


def dequant_mxfp4_blocks(blocks, scales_u8):
    """gpt-oss: blocks uint8 [...,G,16], scales uint8 [...,G] (E8M0 bytes) -> float32 [..., G*32]."""
    lead = blocks.shape[:-2]
    g = blocks.shape[-2]
    w = unpack_e2m1(blocks).reshape(lead + (g, 32))
    w = w * e8m0_to_f32(scales_u8)[..., None]
    return w.reshape(lead + (g * 32,))


def dequant_nvfp4(packed, scale_e4m3_u8, global_scale, inverse_global=False):
    """packed uint8 [out,in/2], scale E4M3 bytes [out,in/16], global scale (scalar)
    ModelOpt (inverse_global=False): (e2m1 * scale) * g;  compressed-tensors (True): (e2m1 * scale) / g.
    e2m1 * scale is exact in fp32, so the result is the exact product correctly rounded once (same order as
    foreign_quant.nvfp4.dequant_nvfp4)."""
    g = np.float32(np.asarray(global_scale, dtype=np.float32).reshape(-1)[0])
    w = apply_scales(unpack_e2m1(packed), e4m3_to_f32(scale_e4m3_u8))
    return w / g if inverse_global else w * g


# ----------------------------------------------------------------------------------------------
# detection (names + dtypes + shapes only; works on a header map without any weights)
# ----------------------------------------------------------------------------------------------

FP8_TAGS = ("F8_E4M3", "F8_E5M2")
DENSE_TAGS = {"BF16": "bf16", "F16": "f16", "F32": "f32"}


def _grid_kind(wshape, sshape):
    if int(np.prod(sshape)) == 1:
        return "tensor"
    if len(sshape) == 1 or (len(sshape) == 2 and sshape[1] == 1 and sshape[0] == wshape[0]):
        return "channel"
    return "block"


def classify_linear(info, prefix):
    """info: {name: (dtype tag, shape)}. Returns a dict describing how linear `prefix` is stored, or None.
    keys: fmt (dense_bf16|dense_f16|dense_f32|fp8_tensor|fp8_channel|fp8_block|fp8_noscale|mxfp4|mxfp4_blocks|nvfp4),
          weight, scale, scale2, inverse_global, detail."""
    def get(k):
        return info.get(prefix + k)

    if prefix + "_blocks" in info and prefix + "_scales" in info:
        return dict(fmt="mxfp4_blocks", weight=prefix + "_blocks", scale=prefix + "_scales", detail="gpt-oss blocks/scales")
    wname = ".weight" if get(".weight") else (".weight_packed" if get(".weight_packed") else None)
    if wname is None:
        return None
    tag, shape = get(wname)
    if tag in DENSE_TAGS:
        if len(shape) < 2:
            return None
        return dict(fmt="dense_" + DENSE_TAGS[tag], weight=prefix + wname, detail=tag)
    if tag in FP8_TAGS:
        for sfx in (".weight_scale_inv", ".weight_scale", ".scale"):
            sc = get(sfx)
            if sc:
                kind = _grid_kind(shape, sc[1])
                return dict(fmt="fp8_" + kind, weight=prefix + wname, scale=prefix + sfx,
                            detail=f"{tag} x {sc[0]}{list(sc[1])} ({sfx[1:]})")
        return dict(fmt="fp8_noscale", weight=prefix + wname, detail=tag)
    if tag in ("U8", "I8") and len(shape) == 2:
        ws, sc2 = get(".weight_scale"), get(".scale")
        if ws and ws[0] == "F8_E4M3":
            for g in (".weight_scale_2", ".weight_global_scale"):
                if get(g):
                    return dict(fmt="nvfp4", weight=prefix + wname, scale=prefix + ".weight_scale", scale2=prefix + g,
                                inverse_global=(g == ".weight_global_scale"),
                                detail=f"group {2 * shape[1] // max(1, ws[1][-1])}, global {g[1:]}")
            return dict(fmt="nvfp4", weight=prefix + wname, scale=prefix + ".weight_scale", scale2=None,
                        detail="no global scale tensor")
        for sfx, s in ((".scale", sc2), (".weight_scale", ws)):
            if s and s[0] in ("F8_E8M0", "U8") and len(s[1]) == 2:
                return dict(fmt="mxfp4", weight=prefix + wname, scale=prefix + sfx,
                            detail=f"group {2 * shape[1] // max(1, s[1][-1])}, {sfx[1:]} {s[0]}")
    return None


def linear_prefixes(info):
    """Candidate linear prefixes of a header map (2D+ weights, fused *_blocks)."""
    out = set()
    for n, (tag, shape) in info.items():
        if n.endswith("_blocks") and n[:-7] + "_scales" in info:
            out.add(n[:-7])
        elif n.endswith((".weight", ".weight_packed")) and len(shape) >= 2:
            out.add(n.rsplit(".", 1)[0])
    return sorted(out)


def _config_block(config):
    q = (config or {}).get("quantization_config") or (config or {}).get("text_config", {}).get("quantization_config") or {}
    b = q.get("weight_block_size")
    if b is None:
        cg = q.get("config_groups") or {}
        for g in cg.values():
            b = (g.get("weights") or {}).get("block_structure")
            if b:
                break
    if isinstance(b, str):
        b = [int(x) for x in b.split("x")]
    return tuple(b) if b and len(b) == 2 else None


def detect_checkpoint(info, config=None):
    """Header map (+ config.json) -> report of the input format. Never reads a weight."""
    counts, examples, unscaled = {}, {}, []
    for p in linear_prefixes(info):
        c = classify_linear(info, p)
        if c is None:
            wk = [k for k in (p + ".weight", p + ".weight_packed") if k in info]
            if wk and info[wk[0]][0] in ("U8", "I8", "I32") and len(info[wk[0]][1]) == 2:
                unscaled.append(p)                         # integer payload no reader here understands (GPTQ / AWQ ...)
            continue
        counts[c["fmt"]] = counts.get(c["fmt"], 0) + 1
        examples.setdefault(c["fmt"], dict(prefix=p, detail=c["detail"]))
    q = (config or {}).get("quantization_config") or (config or {}).get("text_config", {}).get("quantization_config") or {}
    quant = {k: q[k] for k in ("quant_method", "fmt", "weight_block_size", "quant_algo", "format") if k in q}
    nq = {f: n for f, n in counts.items() if not f.startswith("dense_")}
    if unscaled and not nq:
        primary = "unsupported-integer-quant"
    elif nq:
        primary = max(nq, key=nq.get) if len(nq) == 1 else "mixed:" + "+".join(sorted(nq))
    else:
        primary = max(counts, key=counts.get) if counts else "unknown"
    return dict(primary=primary, counts=counts, examples=examples, quantization_config=quant,
                block_size=_config_block(config), unsupported_integer_linears=len(unscaled), unsupported_examples=unscaled[:3])


# ----------------------------------------------------------------------------------------------
# one-call loader
# ----------------------------------------------------------------------------------------------

def load_linear(ck, prefix, block=None):
    """float32 [out,in] (or [E,out,in] for gpt-oss blocks) of linear `prefix` of Checkpoint `ck`: one linear at a
    time, so a converter streams layer by layer and its peak is a few fp32 copies of the largest linear."""
    c = classify_linear(ck.info(), prefix)
    if c is None:
        raise KeyError(f"{prefix}: not a recognised linear")
    block = block or _config_block(ck.config)
    fmt = c["fmt"]
    tag, shape, w = ck.raw(c["weight"])
    if fmt.startswith("dense_"):
        return to_f32(tag, w)
    if fmt == "mxfp4_blocks":
        return dequant_mxfp4_blocks(w, ck.raw(c["scale"])[2])
    if fmt.startswith("fp8_"):
        s = None if fmt == "fp8_noscale" else ck.scale_f32(c["scale"])
        return dequant_fp8(w, tag, s, block)
    if fmt == "mxfp4":
        return dequant_mxfp4_linear(w, ck.scale_f32(c["scale"]))
    if fmt == "nvfp4":
        if c["scale2"] is None:
            raise ValueError(f"{prefix}: NVFP4 without a global scale tensor")
        return dequant_nvfp4(w, ck.raw(c["scale"])[2], ck.f32(c["scale2"]), c["inverse_global"])
    raise AssertionError(fmt)
