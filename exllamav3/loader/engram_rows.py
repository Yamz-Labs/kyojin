"""Row-lookup pieces of the Hy4-flash registry blocks `f8_rows` and `engram` (design HY4_STRIX_DESIGN.md §2, §4).

Pure torch / numpy, no extension, no relative imports (tests load it by path).
  * E4M3 / E8M0 decode tables (exact), F8Rows: row lookup in a ggml type-43 tensor, PLAIN or BLOCKED layout chosen by byte span.
  * engram_hash: llama.cpp hy_v4_flash engram_hash (M:5175-5221) for a whole token sequence.
  * Hy4Engram: build_engram (M:5292-5353) on CPU fp32. Weights are dequantised once (Q8_0 / F32 / F16); the engine's
    ForeignLinear path is a separate step, this is the maths + the row source.
"""
from __future__ import annotations
import os
import numpy as np
import torch

GGML_F8_E4M3 = 43
QB = 32                      # elements per BLOCKED block
MASK64 = (1 << 64) - 1


# ------------------------------------------------------------------------------------------------ decode tables
def _e4m3_table() -> torch.Tensor:
    """OCP E4M3FN: bias 7, no inf, S.1111.111 = NaN, subnormals e=0 -> m/8 * 2^-6. Built from the bit fields, not from torch."""
    t = torch.empty(256, dtype=torch.float32)
    for b in range(256):
        s, e, m = b >> 7, (b >> 3) & 15, b & 7
        if e == 15 and m == 7:
            v = float("nan")
        elif e == 0:
            v = (m / 8.0) * 2.0 ** -6
        else:
            v = (1.0 + m / 8.0) * 2.0 ** (e - 7)
        t[b] = -v if s else v
    return t


def _e8m0_table() -> torch.Tensor:
    t = torch.tensor([2.0 ** (e - 127) for e in range(255)] + [float("nan")], dtype=torch.float64)
    return t.to(torch.float32)   # 2^-127 is subnormal in fp32 but exact (fp32 subnormal min 2^-149)


E4M3 = _e4m3_table()
E8M0 = _e8m0_table()


def decode_e4m3(u8: torch.Tensor) -> torch.Tensor:
    return E4M3[u8.long()]


# ------------------------------------------------------------------------------------------------ F8 row type
class F8LayoutError(ValueError):
    pass


def tensor_span(g, name: str) -> int:
    """Bytes the tensor may occupy: distance to the next tensor of the same file, else to the end of the file."""
    ty, dims, off = g.infos[name]
    fi, o = off >> 48, off & ((1 << 48) - 1)
    nxt = [x & ((1 << 48) - 1) for _, _, x in g.infos.values() if (x >> 48) == fi and (x & ((1 << 48) - 1)) > o]
    end = min(nxt) if nxt else os.fstat(g.fds[fi]).st_size - g.bases[fi]
    return end - o


def f8_layout(n_elem: int, span: int, width: int) -> str:
    """'plain' (1 B/element) or 'blocked' (33 B per 32 elements; per row: width e4m3 bytes then width/32 e8m0 scale bytes, the DS4.1
    form), by the tensor byte span. NB the span cannot tell this row form from a 33 B interleaved [scale|32 values] block form
    (same size): only the tree's row form is decoded.
    ggml pads a tensor to 32 B, so a span is accepted when it equals the exact size or the exact size rounded up to 32."""
    al = lambda n: (n + 31) // 32 * 32
    plain = {n_elem, al(n_elem)}
    blocked = {n_elem // QB * 33, al(n_elem // QB * 33)} if n_elem % QB == 0 and width % QB == 0 else set()
    hp, hb = span in plain, span in blocked
    if hp and hb:
        raise F8LayoutError(f"F8_E4M3 tensor span {span} B for {n_elem} elements fits both the plain and the 33 B-per-32 blocked layout")
    if hp:
        return "plain"
    if hb:
        return "blocked"
    raise F8LayoutError(f"F8_E4M3 tensor: {span} B for {n_elem} elements is neither plain (1 B/element = {n_elem} B) nor blocked "
                        f"(33 B per 32 elements = {n_elem // QB * 33 if n_elem % QB == 0 else 'n/a'} B); refusing")


class F8Rows:
    """Row lookup of a ggml F8_E4M3 tensor (row-lookup-only: never a Linear, never resident). dims = (width, rows)."""

    def __init__(self, g, name: str = "per_layer_token_embd.weight"):
        ty, dims, off = g.infos[name]
        if ty != GGML_F8_E4M3:
            raise F8LayoutError(f"{name}: ggml type {ty} is not F8_E4M3 (43)")
        if len(dims) != 2:
            raise F8LayoutError(f"{name}: expected a 2-D table, got dims {dims}")
        self.g, self.name = g, name
        self.width, self.rows = int(dims[0]), int(dims[1])
        self.layout = f8_layout(self.width * self.rows, tensor_span(g, name), self.width)
        self.row_bytes = self.width if self.layout == "plain" else self.width + self.width // QB
        self.off = off

    def raw_row(self, r: int) -> np.ndarray:
        if not 0 <= r < self.rows:
            raise IndexError(f"{self.name}: row {r} outside 0..{self.rows - 1}")
        return self.g._pread(self.off + r * self.row_bytes, self.row_bytes)

    def decode_raw(self, raw: np.ndarray) -> torch.Tensor:
        u = torch.from_numpy(np.ascontiguousarray(raw))
        if self.layout == "plain":
            return decode_e4m3(u)
        # DS4.1 row form (dsv41_hip.hip ds41_fp8_decode, dsv41/engram.py dequant_rows): width e4m3 values, then width/32 e8m0 scales
        w = self.width
        return (decode_e4m3(u[:w]).reshape(-1, QB) * E8M0[u[w:].long()][:, None]).reshape(-1)

    def lookup(self, ids) -> torch.Tensor:
        """ids (any shape of ints) -> float32 [*shape, width]; every distinct row is read once."""
        ids = torch.as_tensor(ids, dtype=torch.int64)
        flat = ids.reshape(-1)
        uniq, inv = torch.unique(flat, return_inverse=True)
        rows = torch.stack([self.decode_raw(self.raw_row(int(r))) for r in uniq])
        return rows[inv].reshape(*ids.shape, self.width)


# ------------------------------------------------------------------------------------------------ engram hash
def engram_hash(tokens, start, n_tokens, token_map, pad_id, mult, primes, offs, n_gram, n_head):
    """Row ids for positions start..start+n_tokens-1 of the token list `tokens` (the whole history).
    mult [n_gram] u64, primes / offs [n_cols] for ONE module. Returns (ids int64 [n, n_cols], valid float32 [n, n_cols]).
    Token not in token_map -> dead (pad in the hash, valid 0); position before the sequence start -> pad, still valid (M:5207)."""
    n_cols = (n_gram - 1) * n_head
    tm = list(token_map)
    pad = int(tm[pad_id])
    cid = lambda t: -1 if t < 0 or t >= len(tm) else int(tm[t])
    ids = np.zeros((n_tokens, n_cols), dtype=np.int64)
    valid = np.zeros((n_tokens, n_cols), dtype=np.float32)
    for k in range(n_tokens):
        p = start + k
        c = cid(tokens[p])
        dead = c == -1
        toks = [pad if dead else c]
        for s in range(1, n_gram):
            c = -1 if p - s < 0 else cid(tokens[p - s])
            toks.append(pad if c == -1 else c)
        rolling = (toks[0] * int(mult[0])) & MASK64
        for o in range(1, n_gram):
            rolling ^= (toks[o] * int(mult[o])) & MASK64
            for h in range(n_head):
                col = (o - 1) * n_head + h
                ids[k, col] = rolling % int(primes[col]) + int(offs[col])
                valid[k, col] = 0.0 if dead else 1.0
    return torch.from_numpy(ids), torch.from_numpy(valid)


# ------------------------------------------------------------------------------------------------ weights
def _dequant_q8_0(raw: np.ndarray, K: int, N: int) -> torch.Tensor:
    b = np.ascontiguousarray(raw).reshape(N, K // 32, 34)
    d = b[:, :, :2].copy().view(np.float16).astype(np.float32)          # [N, nb, 1]
    q = b[:, :, 2:].view(np.int8).astype(np.float32)
    return torch.from_numpy((q * d).reshape(N, K))


def load_dense(g, name: str) -> torch.Tensor:
    """a small GGUF tensor as float32 in storage orientation [dims[1], dims[0]] (F32 / F16 / BF16 / Q8_0)."""
    ty, dims, off = g.infos[name]
    if ty == 8:
        K, N = dims[0], dims[1]
        return _dequant_q8_0(g._pread(off, N * (K // 32) * 34), K, N)
    if ty in (0, 1, 30):
        t = g.get_tensor(name)
        return t.float()
    raise ValueError(f"{name}: ggml type {ty} not handled by the Hy4 engram loader")


# ------------------------------------------------------------------------------------------------ the block
def q8_0_roundtrip(x: torch.Tensor) -> torch.Tensor:
    """quantize_row_q8_0 + dequantize (ggml-quants.c): per 32 values d = amax/127 (stored fp16), q = round-half-away(x / d)."""
    b = x.reshape(*x.shape[:-1], -1, 32)
    d = (b.abs().amax(-1, keepdim=True) / 127.0).half().float()    # ggml: d = amax/127 as fp32, id = 1/d, stored d as fp16
    d32 = b.abs().amax(-1, keepdim=True) / 127.0
    idv = torch.where(d32 != 0, 1.0 / d32, torch.zeros_like(d32))
    v = b * idv
    q = torch.round(v)   # AVX2/NEON quantize_row_q8_0 rounds to nearest-even (_mm256_round_ps); the scalar ref uses roundf (half away)
    return (q * d).reshape(x.shape)


def rms(x: torch.Tensor, eps: float) -> torch.Tensor:
    """ggml_rms_norm: x / sqrt(mean(x^2) + eps) over the last dim."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class Hy4Engram:
    """One Engram module of hy_v4_flash (the layer-`layer` pre-hook), fp32 CPU. Config from the GGUF header, never a model name."""

    def __init__(self, g, layer: int, eps: float = 1.1920929e-07, table: F8Rows | None = None, q8_act: bool = False):
        m = g.meta
        p = (m["general.architecture"]) + "."
        pl = list(m[p + "ple.layers"])
        self.layer, self.e = layer, pl.index(layer)
        self.n_gram, self.n_head = m[p + "ple.ngram_size"], m[p + "ple.heads_per_ngram"]
        self.n_cols = (self.n_gram - 1) * self.n_head
        e, nc, ng = self.e, self.n_cols, self.n_gram
        self.mult = list(m[p + "ple.layer_multipliers"])[e * ng:(e + 1) * ng]
        self.primes = list(m[p + "ple.head_vocab_sizes"])[e * nc:(e + 1) * nc]
        self.offs = list(m[p + "ple.head_offsets"])[e * nc:(e + 1) * nc]
        self.token_map, self.pad_id = m[p + "engram.token_map"], m[p + "engram.pad_token_id"]
        self.head_dim = m[p + "embedding_length_per_layer_input"]
        self.n_embd, self.hc = m[p + "embedding_length"], m[p + "hyper_connection.count"]
        self.eps = eps
        self.q8_act = q8_act      # emulate llama.cpp CPU mul_mat(Q8_0 weight): the activation is quantised to Q8_0 first (vec_dot_type)
        b = f"blk.{layer}."
        self.kv = load_dense(g, b + "engram_kv.weight")                        # [(hc+1)*n_embd, n_cols*head_dim]
        self.kv_b = load_dense(g, b + "engram_kv_b.weight")                    # [hc+1, n_embd]
        self.lookup_norm = load_dense(g, b + "engram_lookup_norm.weight")      # [n_cols, head_dim]
        self.gate = load_dense(g, b + "engram_gate.weight")                    # [hc, n_embd]
        self.hc_scale = load_dense(g, b + "engram_hc_scale.weight")            # [hc, n_embd]
        self.pre_gamma = load_dense(g, b + "engram_pre_gamma.weight")          # [n_embd]
        assert self.kv.shape == ((self.hc + 1) * self.n_embd, nc * self.head_dim), self.kv.shape
        self.table = table if table is not None else F8Rows(g)
        assert self.table.width == self.head_dim

    def row_ids(self, tokens, start, n):
        return engram_hash(tokens, start, n, self.token_map, self.pad_id, self.mult, self.primes, self.offs, self.n_gram, self.n_head)

    def apply(self, x: torch.Tensor, rows: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """x [T, hc, n_embd] fp32 stream, rows [T, n_cols, head_dim] decoded table rows, valid [T, n_cols] -> x + delta (M:5292-5353)."""
        if not self.q8_act and x.is_cuda and x.dtype == torch.float32 and os.environ.get("HY4_ENGRAM_FUSED", "0") == "1":     # two fused launches around the GEMM (engram_fused.py)
            from . import engram_fused as _EF
            return _EF.apply(self, x, rows, valid)
        T, eps = x.shape[0], self.eps
        r = rms(rows, eps) * self.lookup_norm * valid[:, :, None]              # delta 1
        r = r.reshape(T, -1)
        if self.q8_act:
            r = q8_0_roundtrip(r)
        kv = (r @ self.kv.T).reshape(T, self.hc + 1, self.n_embd) + self.kv_b      # delta 2
        key, value = kv[:, :self.hc], kv[:, self.hc]
        dot = (rms(x, eps) * rms(key, eps) * self.gate).sum(-1) / self.n_embd ** 0.5               # [T, hc]
        mag = dot.abs().clamp_min(1e-6).sqrt()
        sign = 1.0 - 2.0 * (-dot > 0).float()                                  # ggml_step(-dot): dot < 0 -> -1, dot >= 0 -> +1
        gate = torch.sigmoid(mag * sign)
        signal = rms(value, eps) * self.pre_gamma                              # delta 3
        scale = gate[:, :, None] + self.hc_scale                               # delta 4
        delta = signal[:, None, :] * scale
        anyv = (valid.sum(-1) > 0).float()                                     # whole-token mask
        return x + delta * anyv[:, None, None]

    def __call__(self, x: torch.Tensor, tokens, start: int = 0) -> torch.Tensor:
        ids, valid = self.row_ids(tokens, start, x.shape[0])
        return self.apply(x, self.table.lookup(ids), valid)
