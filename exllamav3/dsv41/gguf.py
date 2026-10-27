"""
GGUF (antirez DeepSeek-V4.1-Flash Q2 layout, general.architecture = deepseek41) reader and weight source.

  * own parser, pread based: no mmap of model bytes anywhere
  * decoders in torch for F32 / F16 / BF16 / I8 / Q8_0 / Q2_K / IQ2_XXS (ggml block layouts)
  * GGUFWeightSource maps the official tensor names to the GGUF names, slices routed experts out of the
    3-D `ffn_*_exps` tensors by byte range, hands Engram rows to an O_DIRECT row store
Spec for the layouts: ggml-common.h block_q8_0 / block_q2_K / block_iq2_xxs (dequantize_row_*),
antirez/ds4 ds4.c tensor names, header facts in ds41probe_parse.py.
"""
from __future__ import annotations
import json
import os
import struct
import numpy as np
import torch
from .config import DSV41Config
from .linear import DenseLin
from .experts import ExpertHandle
from .weights import WeightSource
from .engram import DirectRowReader, FileEngramStore, ROW_BYTES
from . import iq2_tables

# ggml_type -> (name, block elements, block bytes)
GGML_TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 8: ("Q8_0", 32, 34), 10: ("Q2_K", 256, 84),
    16: ("IQ2_XXS", 256, 66), 24: ("I8", 1, 1), 30: ("BF16", 1, 2),
}
_KV_SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_KV_FMT = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
_NP = {0: np.uint8, 1: np.int8, 2: np.uint16, 3: np.int16, 4: np.uint32, 5: np.int32, 6: np.float32,
       10: np.uint64, 11: np.int64, 12: np.float64}


class TensorInfo:
    __slots__ = ("name", "dims", "gtype", "offset")

    def __init__(self, name, dims, gtype, offset):
        self.name, self.dims, self.gtype, self.offset = name, dims, gtype, offset

    @property
    def type_name(self):
        return GGML_TYPES.get(self.gtype, (f"t{self.gtype}",))[0]

    @property
    def torch_shape(self):
        return tuple(reversed(self.dims))

    def row_bytes(self):
        _, be, bb = GGML_TYPES[self.gtype]
        assert self.dims[0] % be == 0, (self.name, self.dims, be)
        return self.dims[0] // be * bb

    def nbytes(self):
        n = 1
        for d in self.dims[1:]:
            n *= d
        return self.row_bytes() * n


class _Rd:
    """Sequential buffered reader over a file (header parsing only)."""

    def __init__(self, fd, size_hint=1 << 22):
        self.fd, self.pos = fd, 0
        self.buf, self.base = b"", 0
        self.chunk = size_hint

    def read(self, n):
        end = self.pos + n
        if self.pos < self.base or end > self.base + len(self.buf):
            self.base = self.pos
            self.buf = os.pread(self.fd, max(n, self.chunk), self.pos)
            if len(self.buf) < n:
                raise EOFError(f"GGUF header truncated at {self.pos}")
        o = self.pos - self.base
        self.pos = end
        return self.buf[o:o + n]

    def u32(self):
        return struct.unpack("<I", self.read(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.read(8))[0]

    def string(self):
        return self.read(self.u64()).decode("utf8", "replace")


class GGUFFile:
    def __init__(self, path: str, parse_tokens: bool = False):
        self.path = path
        self.fd = os.open(path, os.O_RDONLY)
        self.size = os.fstat(self.fd).st_size
        r = _Rd(self.fd)
        if r.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        self.version = r.u32()
        nt, nkv = r.u64(), r.u64()
        self.kv = {}
        for _ in range(nkv):
            k = r.string()
            t = r.u32()
            self.kv[k] = self._val(r, t, parse_tokens or "tokenizer" not in k)
        self.tensors = {}
        for _ in range(nt):
            name = r.string()
            nd = r.u32()
            dims = tuple(r.u64() for _ in range(nd))
            gt = r.u32()
            off = r.u64()
            self.tensors[name] = TensorInfo(name, dims, gt, off)
        align = int(self.kv.get("general.alignment", 32))
        self.alignment = align
        self.data_start = (r.pos + align - 1) // align * align

    @staticmethod
    def _val(r, t, keep=True):
        if t == 8:
            return r.string()
        if t == 9:
            et = r.u32()
            n = r.u64()
            if et == 8:
                v = [r.string() for _ in range(n)]
                return v if keep else v[:1]
            raw = r.read(_KV_SZ[et] * n)
            return np.frombuffer(raw, dtype=_NP[et]).copy() if keep else None
        return struct.unpack("<" + _KV_FMT[t], r.read(_KV_SZ[t]))[0]

    def close(self):
        os.close(self.fd)

    def has(self, name):
        return name in self.tensors

    def read_bytes(self, name, start=0, n=None) -> np.ndarray:
        t = self.tensors[name]
        n = t.nbytes() - start if n is None else n
        buf = os.pread(self.fd, n, self.data_start + t.offset + start)
        if len(buf) != n:
            raise EOFError(f"{name}: short read {len(buf)}/{n}")
        return np.frombuffer(buf, dtype=np.uint8)

    def tensor(self, name, dtype=torch.float32) -> torch.Tensor:
        t = self.tensors[name]
        return decode(torch.from_numpy(self.read_bytes(name).copy()), t.gtype, t.torch_shape).to(dtype)

    def expert_slice(self, name, e, dtype=torch.float32) -> torch.Tensor:
        """Decode expert e of a 3-D [in, out, n_exp] tensor -> [out, in]."""
        t = self.tensors[name]
        assert len(t.dims) == 3
        per = t.dims[1] * t.row_bytes()
        raw = torch.from_numpy(self.read_bytes(name, e * per, per).copy())
        return decode(raw, t.gtype, (t.dims[1], t.dims[0])).to(dtype)


# --------------------------------------------------------------------------------------- decoders
def _f16(b):  # bytes [..., 2] -> float32
    return b.contiguous().view(torch.float16).float()


def dq_q8_0(raw: torch.Tensor) -> torch.Tensor:
    b = raw.view(-1, 34)
    return (b[:, 2:].contiguous().view(torch.int8).float() * _f16(b[:, :2]))


def dq_q2_k(raw: torch.Tensor) -> torch.Tensor:
    b = raw.view(-1, 84)
    nb = b.size(0)
    sc, qs = b[:, :16], b[:, 16:80]
    d, dmin = _f16(b[:, 80:82]), _f16(b[:, 82:84])
    dl = (d * (sc & 0xF).float()).view(nb, 16, 1)
    ml = (dmin * (sc >> 4).float()).view(nb, 16, 1)
    shift = torch.tensor([0, 2, 4, 6], dtype=torch.uint8).view(1, 1, 4, 1)
    q = ((qs.view(nb, 2, 1, 32) >> shift) & 3).reshape(nb, 16, 16).float()
    return (dl * q - ml).reshape(nb, 256)


_IQ2 = {}


def dq_iq2_xxs(raw: torch.Tensor) -> torch.Tensor:
    if not _IQ2:
        _IQ2["grid"], _IQ2["ks"] = iq2_tables.grid_table(), iq2_tables.ksigns_table().long()
    b = raw.view(-1, 66)
    nb = b.size(0)
    d = _f16(b[:, :2])                                           # [nb,1]
    q = b[:, 2:].reshape(nb, 8, 8)
    gidx = q[..., :4].long()                                     # four 8-value grid rows per 32 weights
    w1 = q[..., 4:].contiguous().view(torch.int32).squeeze(-1)    # [nb,8] scale (top 4 bits) + 4 x 7-bit signs
    db = d * (0.5 + ((w1 >> 28) & 0xF).float()) * 0.25            # [nb,8]
    sh = torch.tensor([0, 7, 14, 21], dtype=torch.int32).view(1, 1, 4)
    sidx = ((w1.unsqueeze(-1) >> sh) & 127).long()               # [nb,8,4]
    signs = _IQ2["ks"][sidx]                                     # [nb,8,4]
    bits = (signs.unsqueeze(-1) >> torch.arange(8)) & 1          # [nb,8,4,8]
    sgn = torch.where(bits == 0, 1.0, -1.0)
    grid = _IQ2["grid"][gidx]                                    # [nb,8,4,8]
    return (db.view(nb, 8, 1, 1) * grid * sgn).reshape(nb, 256)


def decode(raw: torch.Tensor, gtype: int, shape) -> torch.Tensor:
    """uint8 bytes of a whole tensor/slice -> float32 tensor of `shape` (torch order)."""
    name = GGML_TYPES[gtype][0]
    if name == "F32":
        return raw.view(torch.float32).reshape(shape)
    if name == "F16":
        return raw.view(torch.float16).float().reshape(shape)
    if name == "BF16":
        return raw.view(torch.bfloat16).float().reshape(shape)
    if name == "I8":
        return raw.view(torch.int8).float().reshape(shape)
    if name == "Q8_0":
        return dq_q8_0(raw).reshape(shape)
    if name == "Q2_K":
        return dq_q2_k(raw).reshape(shape)
    if name == "IQ2_XXS":
        return dq_iq2_xxs(raw).reshape(shape)
    raise NotImplementedError(name)


# --------------------------------------------------------------------------------------- source
_LAYER_MAP = {
    "attn.wq_a": "attn_q_a.weight", "attn.wq_b": "attn_q_b.weight", "attn.wkv": "attn_kv.weight",
    "attn.wo_a": "attn_output_a.weight", "attn.wo_b": "attn_output_b.weight",
    "attn.q_norm": "attn_q_a_norm.weight", "attn.kv_norm": "attn_kv_a_norm.weight",
    "attn.attn_sink": "attn_sinks.weight", "attn_norm": "attn_norm.weight", "ffn_norm": "ffn_norm.weight",
    "attn.compressor.wkv": "attn_compressor_kv.weight", "attn.compressor.wgate": "attn_compressor_gate.weight",
    "attn.compressor.norm": "attn_compressor_norm.weight",
    "attn.indexer.wq_b": "indexer.attn_q_b.weight", "attn.indexer.weights_proj": "indexer.proj.weight",
    "attn.indexer.wk": "indexer.attn_k.weight", "attn.indexer.k_norm": "indexer.k_norm.weight",
    "ffn.gate": "ffn_gate_inp.weight", "ffn.gate.bias": "exp_probs_b.bias", "ffn.gate.bias_vl": "exp_probs_b_vl.bias",
    "ffn.shared_experts.w1": "ffn_gate_shexp.weight", "ffn.shared_experts.w3": "ffn_up_shexp.weight",
    "ffn.shared_experts.w2": "ffn_down_shexp.weight",
    "hc_attn_fn": "hc_attn_fn.weight", "hc_attn_base": "hc_attn_base.weight", "hc_attn_scale": "hc_attn_scale.weight",
    "hc_ffn_fn": "hc_ffn_fn.weight", "hc_ffn_base": "hc_ffn_base.weight", "hc_ffn_scale": "hc_ffn_scale.weight",
    "engram.wkv": "engram_kv.weight", "engram.q_weight": "engram_q_norm.weight",
    "engram.k_weight": "engram_k_norm.weight",
}
_TOP_MAP = {"embed": "token_embd.weight", "norm": "output_norm.weight", "head": "output.weight"}
_EXPERT_MAP = {"w1": "ffn_gate_exps.weight", "w3": "ffn_up_exps.weight", "w2": "ffn_down_exps.weight"}


class GGUFWeightSource(WeightSource):
    def __init__(self, gg: GGUFFile, n_layers: int, allow_buffered_engram: bool = False):
        self.gg, self.n_layers = gg, n_layers
        self.allow_buffered = allow_buffered_engram
        self._store = None

    def _gname(self, name: str):
        if name in _TOP_MAP:
            return _TOP_MAP[name]
        if name.startswith("layers."):
            _, l, rest = name.split(".", 2)
            if rest in _LAYER_MAP:
                return f"blk.{l}.{_LAYER_MAP[rest]}"
        if name.startswith("mtp."):
            return None                      # the antirez GGUF has no MTP/dspark tensors
        raise KeyError(name)

    def has(self, name):
        try:
            g = self._gname(name)
        except KeyError:
            return False
        return g is not None and self.gg.has(g)

    def tensor(self, name, dtype=torch.float32, device="cpu"):
        g = self._gname(name)
        if g is None or not self.gg.has(g):
            raise KeyError(f"{name} ({g}) not in GGUF")
        return self.gg.tensor(g, dtype).to(device)

    def linear(self, name, dtype=torch.float32, device="cpu"):
        return DenseLin(self.tensor(name, dtype, device))

    def expert(self, layer, e, dtype=torch.float32, device="cpu"):
        w = {k: DenseLin(self.gg.expert_slice(f"blk.{layer}.{g}", e, dtype).to(device)) for k, g in _EXPERT_MAP.items()}
        return ExpertHandle(layer, e, w["w1"], w["w3"], w["w2"])

    # ---- engram
    def engram_layer_tensors(self, layer_ids):
        return [f"blk.{l}.engram_embd.weight" for l in layer_ids]

    def engram_store(self):
        if self._store is None:
            names = self.engram_layer_tensors(self.meta_layer_ids())
            readers, rows = [], []
            for n in names:
                t = self.gg.tensors[n]
                rb = t.dims[0]                       # head_dim e4m3 bytes + head_dim/32 e8m0 bytes (264 for 256)
                assert t.gtype == 24 and rb * 32 % 33 == 0 and (rb * 32 // 33) % 32 == 0, (n, t.dims, t.type_name)
                readers.append(DirectRowReader([(self.gg.path, self.gg.data_start + t.offset, rb, rb)],
                                               allow_buffered=self.allow_buffered))
                rows.append(t.dims[1])
            self._store = FileEngramStore(readers, rows)
        return self._store

    # ---- metadata
    def meta_layer_ids(self):
        return [int(v) for v in self.gg.kv["deepseek41.engram.layer_ids"]]

    def config(self) -> DSV41Config:
        cfg = DSV41Config.from_hf_dict(json.loads(self.gg.kv["deepseek41.config"]))
        if not self.has("mtp.0.attn.wq_a"):
            cfg.dspark_block_size = 0                  # the antirez GGUF carries no MTP/dspark tensors
        return cfg

    def engram_meta(self, cfg: DSV41Config):
        kv = self.gg.kv
        ids = self.meta_layer_ids()
        rows = [int(v) for v in kv["deepseek41.engram.rows"]]
        nh, ng = cfg.engram_n_heads, cfg.engram_max_ngram_size - 1
        primes = np.asarray(kv["deepseek41.engram.primes"], dtype=np.int64).reshape(len(ids), ng, nh)
        mult = np.asarray(kv["deepseek41.engram.multipliers"], dtype=np.uint64).astype(np.int64).reshape(len(ids), -1)
        return dict(layer_ids=ids, rows=rows, primes=primes.tolist(), multipliers=mult,
                    token_map=np.asarray(kv["deepseek41.engram.token_map"], dtype=np.int64),
                    compressed_vocab=int(kv["deepseek41.engram.compressed_vocab_size"]),
                    pad_id=int(kv["deepseek41.engram.pad_id"]))


def load_gguf_model(path: str, dtype=torch.float32, device="cpu", cache_experts: bool = True,
                    allow_buffered_engram: bool = False, cfg: DSV41Config = None):
    """Build a DSV41Model from a V4.1 GGUF alone (config, token map, primes, multipliers are in it)."""
    from .model import DSV41Model
    from .experts import SourceExpertProvider
    gg = GGUFFile(path)
    n_probe = int(json.loads(gg.kv["deepseek41.config"]).get("text_config", {}).get("num_hidden_layers", 0))
    src = GGUFWeightSource(gg, n_probe, allow_buffered_engram)
    cfg = cfg or src.config()
    src.n_layers = cfg.num_hidden_layers
    em = src.engram_meta(cfg)
    cfg.engram_compressed_vocab_size = em["compressed_vocab"]
    cfg.engram_num_embeddings = em["rows"]
    counts = {l: cfg.moe_counts(l)[0] for l in range(cfg.num_hidden_layers + cfg.n_mtp_layers)}
    prov = SourceExpertProvider(src, counts, cache=cache_experts, dtype=dtype, device=device)
    model = DSV41Model(cfg, src, prov, src.engram_store(), token_map=em["token_map"], multipliers=em["multipliers"],
                       primes=em["primes"], dtype=dtype, device=device)
    return model, src
