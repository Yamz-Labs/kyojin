#!/usr/bin/env python
"""Ideal (physics) speed-vs-depth curve of a model on this box, from config + pack alone.

  python tools/depth/ideal.py MODEL_DIR [--cache-bits 16|8|N] [--ndt 2] [--chunk 4096]
         [--depths 8192,32768,...] [--bpw 2.1] [--hw tools/depth/hw.json] [--json out.json]

No weights are loaded and no GPU is needed: the module tree comes from Model.from_config (meta), the
bytes per tensor from the safetensors headers of the pack (or --bpw when no pack exists yet).

Roofline per step, with  t = max(FLOPs / peak_gemm, bytes / peak_bandwidth):
  decode  (1 row, and a verify step of ndt+1 rows): active weight bytes (dense + shared + the routed
          experts a row touches; the union of experts for R rows) + the per-layer depth terms read at
          depth D (KV / latent / pooled indexer keys / score round trip / recurrent state).
  prefill (chunk of C rows at offset D): all weights read once per chunk, FLOPs of the active weights
          plus the attention / indexer / linear-attention terms at the chunk's mean context D + C/2.
Output: tok/s vs D for decode, verify (per row), prefill marginal (rate of a chunk at offset D) and
prefill average over [0, D] (what a served cold prompt of length D reports), plus the inherent drop
8K -> 500K and a per-component breakdown (weights / attention / indexer / recurrent).

Depth terms come from the module classes (one handler each, below). An unknown mixer class RAISES: it
is never priced as zero. Add a handler to MIXERS when a new architecture brings a new mixer.
"""
import argparse, glob, json, math, os, struct, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HW_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hw.json")
DEPTHS = [8192, 32768, 65536, 131072, 262144, 500000]
QTILE = 128          # query rows that share one pass over K/V in a flash-style prefill kernel
SCALE_B = 2 / 32     # fp16 group scale per 32 quantized values (CacheLayer_quant / dsa pool_s)

_ST_BYTES = {"F64": 8, "F32": 4, "I64": 8, "I32": 4, "F16": 2, "BF16": 2, "I16": 2, "U16": 2, "I8": 1, "U8": 1,
             "F8_E4M3": 1, "F8_E5M2": 1, "BOOL": 1}


def read_headers(model_dir):
    """tensor name -> (nbytes, shape) from every *.safetensors header (reads a few KB per file)."""
    out = {}
    for f in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            a, b = v["data_offsets"]
            out[k] = (b - a, tuple(v["shape"]))
    return out


def walk(mod):
    yield mod
    for s in getattr(mod, "modules", []) or []:
        yield from walk(s)


def cache_elem_bytes(cache_bits):
    return 2.0 if not cache_bits or cache_bits >= 16 else cache_bits / 8 + SCALE_B


# ----------------------------------------------------------------------------------------- weights
class Weights:
    """Per-model weight accounting. bytes_dense: read for every row; moe: list of (bytes_all_experts, E, k)."""

    def __init__(self, model, headers, bpw):
        self.params_dense = 0            # multiply-accumulates per row outside routed experts
        self.bytes_dense = 0.0
        self.moe = []                    # (expert_bytes_total, E, k, flops_per_row_active)
        keys = {m.key for m in walk(model) if getattr(m, "key", None)}
        owner = {}
        if headers:
            for name, (nb, _) in headers.items():
                k = name
                while k and k not in keys:
                    k = k.rpartition(".")[0]
                owner.setdefault(k, []).append((name, nb))
        self.pack = bool(headers)
        mods = [m for m in walk(model) if getattr(m, "key", None)]
        moes = [m for m in mods if type(m).__name__ == "BlockSparseMLP"]
        exp_bytes = {}
        for m in moes:
            tot = 0.0
            for k in keys:
                if k.startswith(m.key + ".experts.") and k in owner:
                    tot += sum(nb for _, nb in owner[k])
            E = m.num_experts
            if not tot:                  # no pack: 3 matrices per expert at --bpw
                tot = E * 3 * m.hidden_size * m.intermediate_size * bpw / 8
            self.moe.append((tot, E, m.num_experts_per_tok, 2.0 * 3 * m.hidden_size * m.intermediate_size * m.num_experts_per_tok))
            exp_bytes[m.key] = tot
        # lookup tables (token / n-gram embeddings) read a few rows per token: no streamed bytes
        skip_prefix = tuple(m.key + ".experts." for m in moes) + \
            tuple(m.key + "." for m in mods if type(m).__name__ in ("Embedding", "NGramEmbedding"))
        for m in mods:
            nm = type(m).__name__
            if nm in ("Embedding", "NGramEmbedding") or any(m.key.startswith(p) for p in skip_prefix):
                continue
            if nm == "Linear":
                self.params_dense += m.in_features * m.out_features
                nb = sum(b for _, b in owner.get(m.key, [])) if headers else m.in_features * m.out_features * bpw / 8
                self.bytes_dense += nb
            elif headers and m.key in owner and nm not in ("BlockSparseMLP",):
                self.bytes_dense += sum(b for _, b in owner[m.key])      # norms, conv, A_log, hyper-connection mixers

    def decode_bytes(self, rows):
        b = self.bytes_dense
        for tot, E, k, _ in self.moe:
            touched = E * (1 - (1 - k / E) ** rows) if rows > 0 else 0
            b += tot * touched / E
        return b

    def all_bytes(self):
        return self.bytes_dense + sum(t for t, *_ in self.moe)

    def flops_row(self):
        return 2.0 * self.params_dense + sum(f for *_, f in self.moe)


# ----------------------------------------------------------------------------------------- mixers
# Each handler returns for context length d: dict(
#   dec_bytes, dec_flops: one decode row reading context d;  pf_flops, pf_bytes: per prefill row of a chunk
#   whose rows see about d keys;  state_bytes: depth independent recurrent state traffic (read+write) per row)
# plus a "kind" label for the breakdown.
def _attn_sparse_terms(d, n_keys_dense, sel, cache_B_per_key, flops_per_key, idx):
    """Shared by QSA / DSA / DSV4-CSA. idx = None or (n_pool, idx_heads, idx_dim)."""
    r = dict(dec_bytes=0.0, dec_flops=0.0, pf_flops=0.0, pf_bytes=0.0, idx_bytes=0.0, idx_row_bytes=0.0, idx_flops=0.0)
    if idx and d > sel:
        n, nh, dim = idx
        n_pool = d / n
        keys = sel
        r["idx_flops"] = 2.0 * nh * dim * n_pool
        r["idx_bytes"] = n_pool * dim * 2                          # pooled keys fp16 (shared by the rows of a chunk)
        r["idx_row_bytes"] = 4 * n_pool                            # score matrix row: fp16 write by the scorer + read by top-k
    else:
        keys = n_keys_dense
    r["dec_bytes"] = keys * cache_B_per_key
    r["dec_flops"] = keys * flops_per_key
    r["pf_flops"] = keys * flops_per_key
    r["pf_bytes"] = (keys * cache_B_per_key) / (1 if keys == sel and idx and d > sel else QTILE)   # gathered keys miss; dense streams
    return r


def mix_attention(m, d, cb, C):
    hd, vd, Hq, Hkv = m.head_dim, getattr(m, "v_head_dim", m.head_dim), m.num_q_heads, m.num_kv_heads
    w = getattr(m, "sliding_window", -1)
    keys = min(d, w) if w and w > 0 else d
    per_key_B = Hkv * (hd + vd) * cb
    per_key_F = 2.0 * Hq * (hd + vd)
    q = getattr(m, "qsa_indexer", None)
    if q is not None:
        sel = q.token_budget + 32                                  # block_topk blocks + tail block (Kpad)
        r = _attn_sparse_terms(d, keys, sel, per_key_B, per_key_F,
                               (q.compress_ratio, q.n_heads, q.head_dim) if d > q.sparse_threshold() else None)
        r["kind"] = "attn+qsa"
    else:
        r = _attn_sparse_terms(d, keys, 0, per_key_B, per_key_F, None)
        r["pf_bytes"] = keys * per_key_B / QTILE
        r["kind"] = "attn"
    return r


def mix_mla(m, d, cb, C):
    lat = m.kv_lora_rank + m.qk_rope_head_dim
    per_key_B = (m.kv_lora_rank * cb) + (m.qk_rope_head_dim * 2)
    per_key_F = 2.0 * m.num_q_heads * (lat + m.kv_lora_rank)     # absorbed: q.latent + p.latent
    sel = getattr(m, "index_topk", 0) or 0
    idx = None
    if sel and getattr(m, "indexer_mode", "full") == "full" and getattr(m, "idx_plane_dim", 0):
        idx = (m.index_kpool or 1, m.index_n_heads, m.index_head_dim)
    r = _attn_sparse_terms(d, d, sel, per_key_B, per_key_F, idx)
    if not idx or d <= sel:
        r["pf_bytes"] = d * per_key_B / QTILE
    r["kind"] = "mla+dsa" if idx else "mla"
    return r


def mix_gdn(m, d, cb, C):
    H, dk, dv = m.num_v_heads, m.k_head_dim, m.v_head_dim
    st = H * dk * dv
    # chunked scan: state update + readout (4 st) + intra-chunk (64 rows) attention-like term
    f = 4.0 * st + 2.0 * H * 64 * (dk + dv)
    return dict(kind="recurrent", dec_bytes=0.0, dec_flops=f, pf_flops=f, pf_bytes=0.0, state_bytes=2.0 * st * 4,
                idx_bytes=0.0, idx_row_bytes=0.0, idx_flops=0.0)


def mix_dsv4(m, d, cb, C):
    hd = m.head_dim
    per_key_B = hd * cb
    per_key_F = 2.0 * m.num_q_heads * 2 * hd
    win = min(d, m.sliding_window or d)
    if m.layer_type == "sliding":
        keys, idx = win, None
    elif m.layer_type == "hca":
        keys, idx = win + d / m.compress_rate, None
    else:                                                          # csa
        keys = win + (m.index_topk if d / m.compress_rate > m.index_topk else d / m.compress_rate)
        idx = (m.compress_rate, m.index_n_heads, m.index_head_dim) if d / m.compress_rate > m.index_topk else None
    r = dict(kind=f"dsv4.{m.layer_type}", dec_bytes=keys * per_key_B, dec_flops=keys * per_key_F, pf_flops=keys * per_key_F,
             pf_bytes=keys * per_key_B / QTILE, idx_bytes=0.0, idx_row_bytes=0.0, idx_flops=0.0)
    if idx:
        n_pool = d / idx[0]
        r["idx_flops"] = 2.0 * idx[1] * idx[2] * n_pool
        r["idx_bytes"] = n_pool * idx[2] * 2
        r["idx_row_bytes"] = 4 * n_pool
    return r


MIXERS = {"Attention": mix_attention, "MLAttention": mix_mla, "GatedDeltaNet": mix_gdn, "DSV4Attention": mix_dsv4}
# module classes known to carry no depth-dependent cost and no unpriced weights beyond Linear children
NO_DEPTH = {"TransformerBlock", "RMSNorm", "LayerNorm", "GatedRMSNorm", "HyperConnection", "HyperHead", "GatedResidual",
            "ExpandStreams", "GatedMLP", "MLP", "BlockSparseMLP", "Linear", "Embedding", "PLELayer", "NGramEmbedding",
            "QSAIndexer", "OutputGather", "PosEmbedding", "Conv", "ShortConv"}
# classes nested under a mixer that the mixer handler already prices
NESTED_OK = {"QSAIndexer"}


def collect_mixers(model):
    out, unknown = [], set()
    for m in walk(model):
        nm = type(m).__name__
        if nm in MIXERS:
            out.append(m)
        elif nm not in NO_DEPTH and nm != type(model).__name__:
            unknown.add(nm)
    if unknown:
        raise NotImplementedError(f"no depth handler for module class(es) {sorted(unknown)}: add one to ideal.py MIXERS "
                                  f"(or to NO_DEPTH if it has no depth-dependent cost)")
    return out


# ----------------------------------------------------------------------------------------- curves
def step_time(flops, nbytes, hw):
    tf, tb = flops / (hw["gemm_tflops"] * 1e12), nbytes / (hw["bw_gbs"] * 1e9)
    return max(tf, tb), tf, tb


def decode_cost(W, mixers, D, rows, cb, hw):
    parts = dict(weights=W.decode_bytes(rows), attn=0.0, indexer=0.0, recurrent=0.0)
    flops = W.flops_row() * rows
    for m in mixers:
        r = MIXERS[type(m).__name__](m, D, cb, 1)
        gathered = rows if ("dsa" in r["kind"] or "qsa" in r["kind"] or "csa" in r["kind"]) else 1   # each row gathers its own keys
        parts["attn"] += r["dec_bytes"] * gathered
        parts["indexer"] += r["idx_bytes"] + r["idx_row_bytes"] * rows
        parts["recurrent"] += r.get("state_bytes", 0.0) * (1 + rows) / 2 if r["kind"] == "recurrent" else 0.0
        flops += (r["dec_flops"] + r["idx_flops"]) * rows
    t, tf, tb = step_time(flops, sum(parts.values()), hw)
    return t, parts, tf, tb


def prefill_chunk(W, mixers, D, C, cb, hw):
    mid = D + C / 2
    flops = W.flops_row() * C
    nbytes = W.all_bytes()
    parts = dict(weights=W.all_bytes(), attn=0.0, indexer=0.0, recurrent=0.0)
    fl_attn = fl_idx = fl_rec = 0.0
    for m in mixers:
        r = MIXERS[type(m).__name__](m, mid, cb, C)
        if r["kind"] == "recurrent":
            fl_rec += r["pf_flops"] * C
            continue
        fl_attn += r["pf_flops"] * C
        fl_idx += r["idx_flops"] * C
        parts["attn"] += r["pf_bytes"] * C
        parts["indexer"] += (r["idx_bytes"] / QTILE + r["idx_row_bytes"]) * C
    flops += fl_attn + fl_idx + fl_rec
    nbytes = sum(parts.values())
    t, tf, tb = step_time(flops, nbytes, hw)
    # per-component time shares (FLOP part only for compute-bound prefill)
    comp = dict(weights=W.flops_row() * C, attn=fl_attn, indexer=fl_idx, recurrent=fl_rec)
    return t, comp, tf, tb


def curves(W, mixers, depths, rows_verify, cb, C, hw, probe=500000):
    res = dict(depths=depths, decode=[], verify=[], prefill=[], prefill_avg=[], breakdown=[])
    # prefill average over [0, D]: integrate chunk times on a 16-point grid per interval
    grid = list(range(0, probe + 1, 8192))
    tt = [prefill_chunk(W, mixers, g, C, cb, hw)[0] for g in grid]

    def avg_rate(D):
        tot, pos = 0.0, 0.0
        for i, g in enumerate(grid):
            if g >= D:
                break
            span = min(8192, D - g)
            tot += tt[i] * span / C
            pos += span
        return pos / tot if tot else float("nan")
    for D in depths:
        t1, p1, f1, b1 = decode_cost(W, mixers, D, 1, cb, hw)
        tv, pv, fv, bv = decode_cost(W, mixers, D, rows_verify, cb, hw)
        tp, comp, fp, bp = prefill_chunk(W, mixers, D, C, cb, hw)
        res["decode"].append(1 / t1)
        res["verify"].append(rows_verify / tv)
        res["prefill"].append(C / tp)
        res["prefill_avg"].append(avg_rate(D))
        res["breakdown"].append(dict(depth=D, decode_ms=1e3 * t1, decode_bytes_MB={k: v / 1e6 for k, v in p1.items()},
                                     verify_ms=1e3 * tv, prefill_chunk_ms=1e3 * tp, prefill_flops_G=({k: v / 1e9 for k, v in comp.items()}),
                                     prefill_bound="compute" if fp >= bp else "memory", decode_bound="compute" if f1 >= b1 else "memory"))
    return res


def drop(a):
    return 100 * (a[-1] / a[0] - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--cache-bits", type=int, default=16, help="K/V bits of the served cache (16 = fp16, 8 = int8 pages)")
    ap.add_argument("--ndt", type=int, default=2, help="drafted tokens per verify step (rows = ndt + 1)")
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--depths", default=",".join(map(str, DEPTHS)))
    ap.add_argument("--bpw", type=float, default=2.1, help="bits per weight when the model dir has no safetensors")
    ap.add_argument("--hw", default=HW_DEFAULT)
    ap.add_argument("--json")
    args = ap.parse_args()
    sys.path.insert(0, ROOT)
    from exllamav3 import Config, Model
    hw = json.load(open(args.hw))
    cfg = Config.from_directory(args.model)
    model = Model.from_config(cfg)
    headers = read_headers(args.model)
    W = Weights(model, headers, args.bpw)
    mixers = collect_mixers(model)
    cb = cache_elem_bytes(args.cache_bits)
    depths = [int(x) for x in args.depths.split(",")]
    res = curves(W, mixers, depths, args.ndt + 1, cb, args.chunk, hw, probe=max(depths))
    res.update(model=args.model, hw=hw, cache_bits=args.cache_bits, ndt=args.ndt, chunk=args.chunk, pack=W.pack,
               weight_GB_all=W.all_bytes() / 1e9, weight_GB_decode_row=W.decode_bytes(1) / 1e9, weight_GFLOP_row=W.flops_row() / 1e9,
               mixers={n: sum(1 for m in mixers if type(m).__name__ == n) for n in MIXERS})
    res["drop_pct"] = {k: drop(res[k]) for k in ("decode", "verify", "prefill", "prefill_avg")}
    print(f"{os.path.basename(args.model.rstrip('/'))}: pack={'yes' if W.pack else f'bpw {args.bpw}'} weights {res['weight_GB_all']:.1f} GB "
          f"(decode row {res['weight_GB_decode_row']:.2f} GB, {res['weight_GFLOP_row']:.1f} GFLOP/row) mixers {res['mixers']}")
    print(f"hw: {hw['bw_gbs']:.0f} GB/s, {hw['gemm_tflops']:.1f} TFLOPS fp16")
    print("depth    " + "".join(f"{d:>9}" for d in depths))
    for k, label in (("decode", "decode"), ("verify", "verify/r"), ("prefill", "pf marg."), ("prefill_avg", "pf avg")):
        print(f"{label:<9}" + "".join(f"{v:9.1f}" for v in res[k]) + f"   drop {res['drop_pct'][k]:+.1f}%")
    if args.json:
        json.dump(res, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
