from __future__ import annotations
from typing_extensions import override
import os as _os
import torch
from ..util.device_copy import to_device
from ..ext import exllamav3_ext as ext
from ..model.config import Config
from ..util.rope import RopeSettings, RoPE
from ..util.tensor import get_for_device, to2, host_cuts
from . import Module, Linear, RMSNorm
from .layernorm import LayerNorm
from ..model.model_tp_alloc import TPAllocation
from .attention_fn.mla_triton import (
    mla_attn_triton_decode,
    mla_attn_triton_prefill,
    mla_attn_triton_prefill_mha,
    mla_absorb,
    mla_unfold,
)
from .attention_fn.bc_attn import MAX_BSZ as _bc_max_bsz
import os

# Prefill strategy: "mha" (default) up-projects past tiles from the compressed cache and attends
# in MHA form. ~2.8x fewer FLOPs than running the absorbed form over the whole context. "absorbed"
# restores the single-kernel absorbed prefill for A/B testing
_prefill_mode = os.environ.get("EXL3_MLA_PREFILL", "mha")

# Query lengths at or below this use the flash-decoding kernel (kv split across programs);
# above it, the long-query kernel (q split across programs) wins
MAX_DECODE_QLEN = 16

# Width of one indexer-scoring tile (keys per kernel call) in the eager top-k path. Bounds
# the score transient at (256 rows, tile) regardless of context length; must be a multiple
# of the page size (256)
_score_tile = int(os.environ.get("EXL3_DSA_SCORE_TILE", 32768))
_kpool_eager_rows = int(os.environ.get("EXL3_DSA_KPOOL_EAGER_ROWS", 1024))  # 0 = one shot (old path)
assert _score_tile % 256 == 0 and _score_tile > 0

# Decode/verify has few query rows.  When enabled, score the whole visible slab in one
# call instead of running the per-tile top-k merge below.  Prefill remains tiled.
EXL3_DSA_DEC_NOTILE = _os.environ.get("EXL3_DSA_DEC_NOTILE", "1") == "1"

# fuse the decode/verify DSA indexer torch glue (pool expand + tail
# append, ~15 launches/layer, ~0.73 ms/round) into one Triton kernel. Default off.
EXL3_DSA_GLUE_FUSE = _os.environ.get("EXL3_DSA_GLUE_FUSE", "0") == "1"

# Prefill analogue (pfDepth2): score a whole 256-row slab over all visible entries in one call,
# one dsa_topk, then reorder score-descending like the tiled merge; drops the per-tile torch
# gather/cat/topk. Bounded: only when the (rows, pow2 stride) fp16 transient fits in
# EXL3_DSA_PF_NOTILE_MB (default 256 MB = 1M-entry rows); longer rows stay tiled.
EXL3_DSA_PF_NOTILE = _os.environ.get("EXL3_DSA_PF_NOTILE", "0") == "1"
EXL3_DSA_PF_NOTILE_MB = int(_os.environ.get("EXL3_DSA_PF_NOTILE_MB", "256"))


def _pf_notile_ok(rows, t):
    return EXL3_DSA_PF_NOTILE and rows * (1 << max(t - 1, 1).bit_length()) * 2 <= EXL3_DSA_PF_NOTILE_MB << 20

# Batch-1 decode: the MLA projections run through the exl3_dec GEMV kernels instead of
# LinearEXL3.forward's own route. LinearEXL3.forward does reach exl3_dec_gemv for a single row
# when its dec_ok holds, but that gate is mul1-only, so on a GLM-5.3-Flash pack (all mcg) every
# projection here falls back to the generic path; this wiring decodes either codebook and folds
# q_a_proj + kv_a_proj_with_mqa into one launch. EXL3_MLA_DEC=0 restores the plain forwards.
EXL3_MLA_DEC = os.environ.get("EXL3_MLA_DEC", "1") != "0"

# Verify batch-invariance (glm-mtp step 7): at 2..EXL3_MLA_DEC_ROWLOOP_MAX rows of an MTP verify
# forward, run the MLA projections as one exl3_dec GEMV launch PER ROW (the exact batch-1 kernel)
# instead of LinearEXL3.forward's M-row route (bc / hip GEMV, a different reduction), so row i of
# an R-row verify equals the plain R = 1 decode bitwise. 0 restores the M-row route
EXL3_MLA_DEC_ROWLOOP_MAX = int(os.environ.get("EXL3_MLA_DEC_ROWLOOP_MAX", "4"))

# Same idea for the dense MLA decode kernel: a short verify pass runs as R one-row decodes
EXL3_MLA_ATTN_ROWLOOP = os.environ.get("EXL3_MLA_ATTN_ROWLOOP", "1") != "0"
# Projection half (glm-mtp step 8): "gemv_r" (default) = one exl3_dec_gemv_r(_multi) launch over the R
# rows, each row reduced exactly like the batch-1 exl3_dec_gemv (mcg included), so verify row i is
# bit-equal to plain decode at no extra launch. "loop" = step 7's R batch-1 launches (same bits, +2.9 ms
# per R=2 verify), "off" = LinearEXL3.forward's M-row route (rows drift at ulp level, ids flip at
# near-ties). Both halves exact = MTP greedy ids equal plain greedy ids (step 7: 32/32 cases).
EXL3_MLA_PROJ_R = os.environ.get("EXL3_MLA_PROJ_R", "gemv_r")

# K*2 values the exl3_dec GEMV kernels are instantiated for (quant/exl3.py: _DEC_KB2)
_DEC_KB2 = (4, 5, 6, 8, 10, 12)


def dec_proj_ok(lin) -> bool:
    """True when one MLA projection can run the dense exl3_dec GEMV on a one-row input.

    Looser than LinearEXL3.dec_ok on the codebook (mcg decodes too, and GLM-5.3-Flash's packs are
    entirely mcg) and on K 2.5; equally strict on shapes; stricter on the wrappers, because the
    fused launch replaces a bare exl3 matmul: LoRA, a bias, pre/post scales, a softcap or padded
    output columns all disqualify it.
    """
    from ..ext import exllamav3_ext as ext
    inner = getattr(lin, "inner", None)
    if inner is None or not hasattr(inner, "K"):   # LinearFP16 (test models) has no EXL3 trellis
        return False
    k2 = inner.K * 2
    return (
        EXL3_MLA_DEC and bool(torch.version.hip) and hasattr(ext, "exl3_dec_gemv") and
        getattr(lin, "quant_type", None) == "exl3" and (inner.mul1 or inner.mcg) and
        int(round(k2)) in _DEC_KB2 and k2 == int(round(k2)) and
        inner.in_features % 512 == 0 and inner.out_features % 128 == 0 and
        tuple(inner.trellis.shape[:2]) == (inner.in_features // 16, inner.out_features // 16) and
        inner.suh is not None and inner.svh is not None and inner.bias is None and
        not getattr(lin, "lora_a_tensors", None) and
        getattr(lin, "pre_scale", 1.0) == 1.0 and getattr(lin, "post_scale", 1.0) == 1.0 and
        getattr(lin, "softcap", 0.0) == 0.0 and
        lin.out_features == lin.out_features_unpadded and
        inner.default_out_dtype in (torch.half, torch.float)
    )


def dec_proj_shapes(lins) -> bool:
    """True when lins can share one exl3_dec_gemv_multi launch (same input width, K and codebook)."""
    return (
        len(lins) >= 1 and all(
            l.inner.in_features == lins[0].inner.in_features and
            int(round(l.inner.K * 2)) == int(round(lins[0].inner.K * 2)) and
            bool(l.inner.mcg) == bool(lins[0].inner.mcg)
            for l in lins
        )
    )


_GEMV_R_MCG = []

def dec_gemv_r_mcg() -> bool:
    """True when this build's exl3_dec_gemv_r takes the mcg codebook (glm-mtp step 8). Cached."""
    if not _GEMV_R_MCG:
        from ..ext import exllamav3_ext as ext
        _GEMV_R_MCG.append("mcg" in (getattr(getattr(ext, "exl3_dec_gemv_r", None), "__doc__", "") or ""))
    return _GEMV_R_MCG[0]


def dec_proj_multi(lins, x, rows = 1):
    """One exl3_dec_gemv_multi launch for lins over the one-row input x: (1, out_features) each.
    rows > 1: one exl3_dec_gemv_r_multi launch over x as (rows, in_features), row r bit-equal to
    the one-row launch on x[r]."""
    from .quant.exl3 import dec_workspace
    from ..ext import exllamav3_ext as ext
    scratch, counters = dec_workspace(x.device)
    outs = [torch.empty((rows, l.inner.out_features), dtype = l.inner.default_out_dtype,
                        device = x.device) for l in lins]
    fn = ext.exl3_dec_gemv_r_multi if rows > 1 else ext.exl3_dec_gemv_multi
    fn(x.reshape(rows, lins[0].inner.in_features).contiguous(),
       [l.inner.trellis for l in lins], [l.inner.suh for l in lins], [l.inner.svh for l in lins],
       outs, scratch, counters, [float(l.inner.K) for l in lins], bool(lins[0].inner.mcg))
    return outs


def dec_proj_single(lin, x, rows = 1):
    """One exl3_dec_gemv launch for one projection over the one-row input x (rows > 1:
    exl3_dec_gemv_r over (rows, in_features), row r bit-equal to the one-row launch)."""
    from .quant.exl3 import dec_workspace
    from ..ext import exllamav3_ext as ext
    inner = lin.inner
    y = torch.empty((rows, inner.out_features), dtype = inner.default_out_dtype, device = x.device)
    scratch, counters = dec_workspace(x.device)
    fn = ext.exl3_dec_gemv_r if rows > 1 else ext.exl3_dec_gemv
    fn(x.reshape(rows, inner.in_features).contiguous(), inner.trellis, inner.suh,
       inner.svh, y, scratch, counters, float(inner.K), bool(inner.mcg))
    return y



def _host_seqlens(params: dict, cache_seqlens: torch.Tensor) -> list:
    """Host copy of cache_seqlens, once per forward (shared across layers via the params
    dict). The generator supplies cache_seqlens as a host (pinned staging) tensor, in which
    case this is a free read; only harnesses that pass device tensors pay a sync."""
    host = params.get("_mla_host_seqlens")
    if host is None:
        src = params.get("cache_seqlens")
        if isinstance(src, torch.Tensor) and src.device.type == "cpu":
            host = src.tolist()
        else:
            host = cache_seqlens.cpu().tolist()
        params["_mla_host_seqlens"] = host
    return host

def _check_block_table(block_table: torch.Tensor, host_seqlens: list, seqlen: int):
    """A block table shorter than the context makes the paged writes (latent, indexer keys) and
    the indexer/attention reads index past its end: garbage page ids, then an HSA hardware
    exception / hipErrorLaunchFailure reported later at an unrelated call (moeA4). Host-only
    check, no device sync, no numeric effect."""
    from ..constants import PAGE_SIZE
    need = -(-(max(host_seqlens) + seqlen) // PAGE_SIZE)
    assert block_table.shape[-1] >= need, \
        f"block_table has {block_table.shape[-1]} pages, context of " \
        f"{max(host_seqlens) + seqlen} tokens needs {need}"


class MLAttention(Module):
    """
    Multi-head latent attention (DeepSeek-V2/V3, Kimi-Linear).

    Attention runs in absorbed form end to end. The cache holds only the compressed latent and the
    shared RoPE key: 576 values per token, against 81920 for the equivalent expanded K/V of a
    128-head model. Per-head K and V are never materialized, in prefill or decode. What makes
    that possible is folding the kv_b up-projection into the query and the output instead:

        scores = (q_nope @ W_UK) . c_kv  +  q_pe . k_pe
        o      = (softmax(scores) @ c_kv) @ W_UV

    Both folds are batched GEMMs over the head axis, so the queries and the attention output are
    kept head-major throughout and the two kv_lora_rank-wide tensors are never permuted.

    W_UK and W_UV stay unquantized. They are pure weight streaming (measured at 76-81% of memory
    peak), the absorb is a bmm rather than a GEMM so the exl3 kernels do not apply, and folding
    W_UK into q_b_proj instead would triple that projection (strictly worse than the 33 MB per
    layer this costs on a 128-head model.)
    """

    def __init__(
        self,
        config: Config | None,
        key: str,
        layer_idx: int,
        hidden_size: int,
        num_q_heads: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        rope_settings: RopeSettings | None,
        q_lora_rank: int | None = None,
        sm_scale: float | None = None,
        rms_norm_eps: float = 1e-6,
        qmap: str | None = None,
        out_dtype: torch.dtype | None = None,
        key_q: str = "q_proj",
        key_q_a: str = "q_a_proj",
        key_q_b: str = "q_b_proj",
        key_q_a_norm: str = "q_a_layernorm",
        key_kv_a: str = "kv_a_proj_with_mqa",
        key_kv_a_norm: str = "kv_a_layernorm",
        key_kv_b: str = "kv_b_proj",
        key_o: str = "o_proj",
        qbits_key: str = "bits",
        select_hq_bits: int = 0,
        indexer_mode: str | None = None,
        index_n_heads: int = 0,
        index_head_dim: int = 0,
        index_topk: int = 0,
        index_norm_eps: float = 1e-6,
        index_kpool: int = 0,
        index_kpool_tail: bool = True,
        key_indexer: str = "indexer",
    ):
        super().__init__(config, key, None)

        self.q_priority = 2 + select_hq_bits
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.num_q_heads = num_q_heads
        self.num_kv_heads = 1
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.rope_settings = rope_settings
        self.rope = None
        # Llama-4 position scale (Mistral-Small-4): set in load_local from the RoPE instance
        self.l4_beta = 0.0
        self.l4_original = 1
        self.out_dtype = out_dtype
        self.key_kv_b = key_kv_b
        self.key_indexer = key_indexer
        self.norm_eps = rms_norm_eps

        # The softmax scale follows the *unabsorbed* head dim: absorption does not change the
        # scores, only how they are computed. Architectures with YaRN mscale pass their own
        self.sm_scale = sm_scale if sm_scale is not None else self.qk_head_dim ** -0.5

        # head_dim is what the rest of the stack reads for reporting and allocation; the latent
        # width is what actually lands in the cache
        self.head_dim = self.qk_head_dim

        qmap_in = qmap + ".input" if qmap is not None else None
        qmap_o = qmap + ".o" if qmap is not None else None

        # Query path: either a direct projection or a LoRA-style pair with a norm between
        if q_lora_rank is None:
            self.q_a_proj = None
            self.q_a_layernorm = None
            self.q_b_proj = None
            self.q_proj = Linear(
                config, f"{key}.{key_q}", hidden_size, num_q_heads * self.qk_head_dim,
                qmap = qmap_in, out_dtype = torch.half, trim_padded_out = True,
                select_hq_bits = select_hq_bits, qbits_key = qbits_key,
            )
            self.register_submodule(self.q_proj)
        else:
            self.q_a_proj = Linear(
                config, f"{key}.{key_q_a}", hidden_size, q_lora_rank,
                qmap = qmap_in, out_dtype = torch.half, trim_padded_out = True,
                select_hq_bits = select_hq_bits, qbits_key = qbits_key,
            )
            self.q_a_layernorm = RMSNorm(
                config, f"{key}.{key_q_a_norm}", rms_norm_eps, out_dtype = torch.half,
            )
            self.q_b_proj = Linear(
                config, f"{key}.{key_q_b}", q_lora_rank, num_q_heads * self.qk_head_dim,
                qmap = qmap + ".q_a" if qmap is not None else None,
                out_dtype = torch.half, trim_padded_out = True,
                select_hq_bits = select_hq_bits, qbits_key = qbits_key,
            )
            self.q_proj = self.q_b_proj
            self.register_submodule(self.q_a_proj)
            self.register_submodule(self.q_a_layernorm)
            self.register_submodule(self.q_b_proj)

        # Latent path. The output of this projection goes straight into the cache, so it is the
        # one place where quantization error compounds over the whole context
        self.kv_a_proj_with_mqa = Linear(
            config, f"{key}.{key_kv_a}", hidden_size, kv_lora_rank + qk_rope_head_dim,
            qmap = qmap_in, out_dtype = torch.half, trim_padded_out = True,
            select_hq_bits = select_hq_bits, qbits_key = qbits_key,
        )
        self.kv_a_layernorm = RMSNorm(
            config, f"{key}.{key_kv_a_norm}", rms_norm_eps, out_dtype = torch.half,
        )
        self.register_submodule(self.kv_a_proj_with_mqa)
        self.register_submodule(self.kv_a_layernorm)

        self.o_proj = Linear(
            config, f"{key}.{key_o}", num_q_heads * v_head_dim, hidden_size,
            qmap = qmap_o, out_dtype = out_dtype, trim_padded_out = True,
            select_hq_bits = select_hq_bits, qbits_key = qbits_key,
        )
        self.register_submodule(self.o_proj)

        # DSA lightning indexer (GLM-5.2 / DeepSeek-V3.2-on-MLA). "full" layers score and select
        # index_topk tokens per query and publish the selection; "shared" layers reuse the
        # nearest preceding full layer's selection. None = plain dense MLA
        assert indexer_mode in (None, "full", "shared")
        self.indexer_mode = indexer_mode
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        # GLM5.3-style k-pool compression: keys are pooled index_kpool at a time (softmax over
        # cached per-token gate scores + a learned in-pool APE), scoring and selection run over
        # pools, and selections expand back to raw token indices (plus the incomplete tail
        # pool's tokens, appended raw, when index_kpool_tail). The cached plane then holds
        # [k || gate_scores] per token
        self.index_kpool = index_kpool
        self.index_kpool_tail = index_kpool_tail
        self.idx_kpool_ape = None
        self.idx_kpool_gate = None
        # Cache layers allocate a per-token indexer-key plane for layers that score selections
        self.idx_plane_dim = (
            None if indexer_mode != "full" else
            index_head_dim * 2 if index_kpool else index_head_dim
        )
        if indexer_mode == "full":
            assert q_lora_rank is not None, "DSA indexer queries project from the q_a latent"
            self.idx_wq_b = Linear(
                config, f"{key}.{key_indexer}.wq_b", q_lora_rank, index_n_heads * index_head_dim,
                qmap = qmap + ".q_a" if qmap is not None else None,
                out_dtype = torch.half, trim_padded_out = True,
                select_hq_bits = select_hq_bits, qbits_key = qbits_key,
            )
            # Key head and per-head scoring weights are router-like: tiny, and selection noise
            # is coherent across every layer sharing it, so they stay unquantized
            self.idx_wk = Linear(
                config, f"{key}.{key_indexer}.wk", hidden_size, index_head_dim,
                qmap = None, out_dtype = torch.half, pad_to = 1,
            )
            self.idx_k_norm = LayerNorm(
                config, f"{key}.{key_indexer}.k_norm", index_norm_eps, out_dtype = torch.half,
            )
            self.idx_weights = Linear(
                config, f"{key}.{key_indexer}.weights_proj", hidden_size, index_n_heads,
                qmap = None, out_dtype = torch.half, pad_to = 1,
            )
            self.register_submodule(self.idx_wq_b)
            self.register_submodule(self.idx_wk)
            self.register_submodule(self.idx_k_norm)
            self.register_submodule(self.idx_weights)
        else:
            self.idx_wq_b = None
            self.idx_wk = None
            self.idx_k_norm = None
            self.idx_weights = None

        self.caps.update({
            "kv_cache": True
        })

        self.cache_layers = []
        self.tp_cache_lookup = {}
        self.has_split_cache = False
        self.dispatch_cache = {}
        # Batch-1 fused GEMV wiring (see dec_proj_ok)
        self.dec_gemv = EXL3_MLA_DEC and bool(torch.version.hip)

        # kv_b_proj, stored ONLY in the flattened (kv_lora_rank, H * dim) form: the prefill
        # up-projection GEMMs consume it directly, and the decode absorb/unfold run as Triton
        # kernels that read per-head column blocks out of the same layout - one resident copy
        # serves every path, and no cuBLAS batched GEMM is involved anywhere in the module
        self.w_uk_flat = None   # (kv_lora_rank, H * qk_nope_head_dim)
        self.w_uv_flat = None   # (kv_lora_rank, H * v_head_dim)
        self._scratch = {}


    def cache_layer_type(self, default, kwargs: dict):
        """MLA stores a latent instead of per-head K/V, so it overrides the cache layer the Cache
        was constructed with. A quantized cache request maps to the packed-latent layer: k_bits
        sets the latent width, the shared rope key stays fp16 (v_bits is accepted but unused -
        there is no separate V)."""
        from ..cache import CacheLayer_fp16, CacheLayer_MLA_fp16, CacheLayer_quant, CacheLayer_MLA_quant
        if issubclass(default, CacheLayer_quant):
            return CacheLayer_MLA_quant, kwargs
        if issubclass(default, CacheLayer_fp16):
            return CacheLayer_MLA_fp16, {}
        raise NotImplementedError(
            f"{default.__name__} is not supported for MLA layers; use CacheLayer_fp16 or "
            f"CacheLayer_quant"
        )


    @override
    def optimizer_targets(self):
        q = (self.q_a_proj.optimizer_targets() if self.q_a_proj else []) + \
            self.q_proj.optimizer_targets() + \
            (self.idx_wq_b.optimizer_targets() if self.idx_wq_b else [])
        kv = self.kv_a_proj_with_mqa.optimizer_targets()
        o = self.o_proj.optimizer_targets()
        return [[q, kv, o]]


    def load_local(self, device, **kwargs):
        for cl in self.cache_layers:
            cl.alloc(device)

        if self.index_kpool:
            stc = self.config.stc
            self.idx_kpool_ape = stc.get_tensor(
                f"{self.key}.{self.key_indexer}.index_kpool_compress_ape", device,
                no_defer = True, allow_bf16 = True).float()
            self.idx_kpool_gate = stc.get_tensor(
                f"{self.key}.{self.key_indexer}.index_kpool_compress_gate", device,
                no_defer = True, allow_bf16 = True).to(torch.half)

        if self.rope_settings:
            self.rope = RoPE(device, self.rope_settings)
            # The Llama-4 scale multiplies the FULL query post-rope (HF semantics, q only). Only
            # the q_pe/k_pe slices pass through the rope kernel here, which would miss q_nope, so
            # take the beta out of the kernel and scale the whole query in _attend instead
            self.l4_beta = self.rope.llama_4_scaling_beta
            self.l4_original = self.rope.llama_4_scaling_original
            self.rope.llama_4_scaling_beta = 0.0

        # kv_b_proj maps the latent to per-head K-nope and V. Attention never applies it as that
        # GEMM; the halves fold into the query/output (decode) or up-project past tiles (prefill)
        w = self.config.stc.get_tensor(f"{self.key}.{self.key_kv_b}.weight", device, no_defer = True)
        if w.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            # fp8 checkpoint: this tensor is read raw rather than through a Linear, so apply the
            # inverse weight scale here (scalar per-tensor or block grid)
            si = self.config.stc.get_tensor(
                f"{self.key}.{self.key_kv_b}.weight_scale_inv", device, optional = True, no_defer = True
            )
            wf = w.float()
            if si is not None:
                si = si.float()
                if si.dim() == 2:
                    r, c = wf.shape
                    sr, sc = si.shape
                    wf = (wf.view(sr, r // sr, sc, c // sc) * si.view(sr, 1, sc, 1)).view(r, c)
                else:
                    wf = wf * si
            w = wf.half()
        assert w.shape == (self.num_q_heads * (self.qk_nope_head_dim + self.v_head_dim),
                           self.kv_lora_rank), \
            f"{self.key}.{self.key_kv_b}: unexpected shape {tuple(w.shape)}"
        H, nope, v, D_c = self.num_q_heads, self.qk_nope_head_dim, self.v_head_dim, self.kv_lora_rank
        w = w.view(H, nope + v, D_c)
        self.w_uk_flat = torch.empty((D_c, H * nope), dtype = torch.half, device = device)
        self.w_uk_flat.view(D_c, H, nope).copy_(w[:, :nope, :].permute(2, 0, 1))
        self.w_uv_flat = torch.empty((D_c, H * v), dtype = torch.half, device = device)
        self.w_uv_flat.view(D_c, H, v).copy_(w[:, nope:, :].permute(2, 0, 1))


    @override
    def load(self, device: torch.Device, **kwargs):
        super().load(device, **kwargs)
        self.load_local(device, **kwargs)


    @override
    def get_tensors(self):
        # kv_b stays unquantized, so it is carried into the converted model; reconstruct the
        # checkpoint layout from the flats (bf16 -> fp16 is value-exact at weight magnitudes)
        t = {}
        if self.w_uk_flat is not None:
            H, nope, v = self.num_q_heads, self.qk_nope_head_dim, self.v_head_dim
            D_c = self.kv_lora_rank
            uk = self.w_uk_flat.view(D_c, H, nope).permute(1, 2, 0)
            uv = self.w_uv_flat.view(D_c, H, v).permute(1, 2, 0)
            t[f"{self.key}.{self.key_kv_b}.weight"] = \
                torch.cat([uk, uv], dim = 1).reshape(H * (nope + v), D_c).contiguous()
        # K-pool compression parameters are raw (unquantized) module tensors, carried as-is
        if self.idx_kpool_ape is not None:
            t[f"{self.key}.{self.key_indexer}.index_kpool_compress_ape"] = \
                self.idx_kpool_ape.contiguous()
            t[f"{self.key}.{self.key_indexer}.index_kpool_compress_gate"] = \
                self.idx_kpool_gate.contiguous()
        return t


    @override
    def unload(self):
        super().unload()
        for cl in self.cache_layers:
            cl.free()
        self.rope = None
        self.w_uk_flat = None
        self.w_uv_flat = None
        self.idx_kpool_ape = None
        self.idx_kpool_gate = None
        self._idx_kpool_gate_t = None
        self._scratch = {}
        self.dispatch_cache = {}


    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        attn_mode = params.get("attn_mode", "flash_attn_nc")
        match attn_mode:
            case "flash_attn":
                x = self.decode_flash_attn(x, bsz, seqlen, params)
            case "flash_attn_nc":
                x = self.decode_flash_attn_nc(x, bsz, seqlen, params)
            case _:
                raise ValueError(f"Unknown attn_mode: {attn_mode}")
        if x is None:
            return None                          # EXL3_PF_SKIP: kv_only, see _attend
        return to2(x, out_dtype, self.out_dtype)


    def project_q(self, x: torch.Tensor, params: dict, return_resid: bool = False, fused: dict | None = None):
        if self.q_a_proj is None:
            q = self._dec_proj(self.q_proj, x, params, fused)
            return (q, None) if return_resid else q
        q_resid = self._dec_proj(self.q_a_proj, x, params, fused)
        q_resid = self.q_a_layernorm.forward(q_resid, params, out_dtype = torch.half)
        q = self._dec_proj(self.q_b_proj, q_resid, params)
        return (q, q_resid) if return_resid else q


    def _dec_proj(self, lin, x: torch.Tensor, params: dict, fused: dict | None = None) -> torch.Tensor:
        """One MLA projection at batch-1 decode: the fused exl3_dec GEMV when it qualifies, else
        lin.forward. `fused` carries outputs already produced by the shared launch
        (_dec_proj_pair), so a projection computed there is not recomputed."""
        if fused:
            y = fused.get(id(lin))
            if y is not None:
                return y
        if (
            self.dec_gemv and x.dtype == torch.half and x.numel() == lin.in_features and
            "capture" not in params and "ovr" not in params and dec_proj_ok(lin)
        ):
            return to2(dec_proj_single(lin, x), None, lin.out_dtype)
        rows = self._dec_rowloop_rows(x, lin.in_features, params)
        if rows and dec_proj_ok(lin):
            if self._proj_gemv_r():
                y = dec_proj_single(lin, x, rows)
            else:
                x2 = x.reshape(rows, lin.in_features)
                y = torch.cat([dec_proj_single(lin, x2[r : r + 1]) for r in range(rows)], dim = 0)
            return to2(y.view(x.shape[:-1] + (lin.out_features,)), None, lin.out_dtype)
        return lin.forward(x, params)


    @staticmethod
    def _proj_gemv_r() -> bool:
        return EXL3_MLA_PROJ_R == "gemv_r" and dec_gemv_r_mcg()


    def _dec_rowloop_rows(self, x: torch.Tensor, in_features: int, params: dict) -> int:
        """Row count when an R-row verify projection should run the batch-1 exl3_dec arithmetic per
        row (one gemv_r launch or R batch-1 launches, bit-identical to plain decode, see
        EXL3_MLA_PROJ_R / EXL3_MLA_DEC_ROWLOOP_MAX), else 0."""
        if not (EXL3_MLA_PROJ_R in ("gemv_r", "loop") and self.dec_gemv and x.dtype == torch.half and params.get("dflash_verify") and
                "capture" not in params and "ovr" not in params):
            return 0
        rows = x.numel() // in_features
        if 1 < rows <= EXL3_MLA_DEC_ROWLOOP_MAX and rows * in_features == x.numel():
            return rows
        return 0


    def _dec_proj_pair(self, x: torch.Tensor, params: dict) -> dict:
        """q_a_proj and kv_a_proj_with_mqa in ONE exl3_dec_gemv_multi launch: at batch 1 both read
        x and (on GLM) share the trellis bitrate, which is what LinearEXL3's own route pays two
        launches for. Returns {id(lin): output}, empty when the pair cannot share a launch."""
        if self.q_a_proj is None:
            return {}
        rows = self._dec_rowloop_rows(x, self.q_a_proj.in_features, params)
        if (
            not rows and (not self.dec_gemv or x.dtype != torch.half or
            "capture" in params or "ovr" in params or x.numel() != self.q_a_proj.in_features)
        ):
            return {}
        lins = (self.q_a_proj, self.kv_a_proj_with_mqa)
        if not all(dec_proj_ok(l) for l in lins) or not dec_proj_shapes(lins):
            return {}
        if rows:
            # Verify rows: the batch-1 fused launch's arithmetic per row, bitwise equal to plain decode
            if self._proj_gemv_r():
                cat = dec_proj_multi(lins, x, rows)
            else:
                x2 = x.reshape(rows, self.q_a_proj.in_features)
                per = [dec_proj_multi(lins, x2[r : r + 1]) for r in range(rows)]
                cat = [torch.cat([p[i] for p in per], dim = 0) for i in range(len(lins))]
            return {id(l): to2(cat[i].view(x.shape[:-1] + (l.out_features,)),
                               None, l.out_dtype) for i, l in enumerate(lins)}
        outs = dec_proj_multi(lins, x)
        return {id(l): to2(o, None, l.out_dtype) for l, o in zip(lins, outs)}


    def _indexer_rope_(self, x4, position, positions, position_ids, inv_freq):
        """In-place partial rope on the leading qk_rope_head_dim dims of a (bsz, seqlen,
        heads, D) tensor, via a strided trailing-slice view: the kernel reads through the
        head stride, so no slice copy or writeback is made (the eager mirror of the BC
        path's rope_gr on its kidx4/qidx4 narrow views)."""
        from ..ext import exllamav3_ext as ext
        if self.rope is None or self.qk_rope_head_dim == 0:
            return
        rope = self.rope
        v = x4[..., : self.qk_rope_head_dim]
        ext.rope(
            v, v, None, None,
            rope.inv_freq if inv_freq is None else inv_freq,
            position,
            positions.contiguous() if positions is not None else None,
            position_ids.contiguous() if position_ids is not None else None,
            int(rope.rope_settings.rope_style),
            rope.attn_factor,
            None, None, self.norm_eps, 0.0,
            rope.llama_4_scaling_beta, rope.llama_4_scaling_original,
            rope.rope_settings.rotate_dims, 0,
        )

    def _indexer_keys(self, x, params, position, positions, position_ids, inv_freq):
        """Roped indexer keys for the current chunk, (bsz, seqlen, index_head_dim). Only the
        first qk_rope_head_dim dims rotate (interleaved pairing, same table as the main
        attention); k_norm is a biased LayerNorm applied before the rotation."""
        k = self.idx_wk.forward(x, params)
        k = self.idx_k_norm.forward(k, params, out_dtype = torch.half).contiguous()
        bsz, seqlen, D_i = k.shape
        k = k.view(bsz, seqlen, 1, D_i)
        self._indexer_rope_(k, position, positions, position_ids, inv_freq)
        k = k.view(bsz, seqlen, D_i)
        if self.index_kpool:
            # Pack the per-token pool-gate scores next to the key: pooled keys are rebuilt
            # from this plane (a pool's members' gate scores feed the in-pool softmax)
            xh = x.to(torch.half)
            if xh.numel() <= 8 * xh.shape[-1] and torch.version.hip and \
                    _os.environ.get("EXL3_HIP_F16_SMALL_M_HGEMM", "1") != "0":
                # decode: multi-CU skinny GEMV (ext.hgemm) instead of a 1-workgroup hipBLAS tile
                gt = getattr(self, "_idx_kpool_gate_t", None)
                if gt is None or gt.device != xh.device:
                    gt = self._idx_kpool_gate_t = self.idx_kpool_gate.T.contiguous()
                gate = torch.empty(xh.shape[:-1] + (gt.shape[1],), dtype = torch.half, device = xh.device)
                ext.hgemm(xh.reshape(-1, xh.shape[-1]), gt, gate.view(-1, gt.shape[1]))
            else:
                gate = torch.nn.functional.linear(xh, self.idx_kpool_gate)
            k = torch.cat((k, gate), dim = -1)
        return k


    def _indexer_topk(self, x, params, q_resid, bsz, seqlen, host_seqlens,
                      position, positions, position_ids, inv_freq,
                      k_idx_chunk = None, idx_pool = None, block_table = None):
        """Lightning-indexer scoring + top-k selection, per batch row. Keys come either from
        the current chunk (cache-less path, k_idx_chunk) or the paged indexer plane (idx_pool +
        block_table). Returns -1-padded int32 indices, (bsz * seqlen, K_pad); selection is per
        query token, shared by all attention heads."""
        from .attention_fn.dsa_triton import dsa_indexer_scores
        from ..ext import exllamav3_ext as ext

        H_i, D_i = self.index_n_heads, self.index_head_dim
        q_idx = self._dec_proj(self.idx_wq_b, q_resid, params).view(bsz, seqlen, H_i, D_i).contiguous()
        self._indexer_rope_(q_idx, position, positions, position_ids, inv_freq)
        # Raw head weights; the scoring kernel folds in the D_i**-0.5 and H_i**-0.5 scales and
        # runs the relu-weighted reduction in fp32, matching the reference
        w = self.idx_weights.forward(x, params)

        t_max = max(host_seqlens) + seqlen
        k_pad = -(-min(self.index_topk, t_max) // 32) * 32
        indices = torch.empty((bsz * seqlen, k_pad), dtype = torch.int32, device = x.device)
        # Row slabs bound the transient score matrix in one direction, tiles over the visible
        # context bound it in the other: nothing here scales with T, so the reference forward
        # at load time is a true worst case for any later context length. Selection is
        # per-row and per-tile top-k merge is exact (any global top-k member is in its own
        # tile's top-k), so neither slabbing nor tiling changes membership
        slab = 256
        t_tile = _score_tile
        if idx_pool is not None:
            # Tiles slice the block table whole pages at a time
            epp = idx_pool.shape[1]
            t_tile = max(epp, t_tile // epp * epp)
        from ..util.tensor import g_tensor_cache
        for b in range(bsz):
            for r0 in range(0, seqlen, slab):
                r1 = min(r0 + slab, seqlen)
                rows = r1 - r0
                pos0 = host_seqlens[b] + r0
                t_slab = host_seqlens[b] + r1
                k_sel = min(self.index_topk, t_slab)
                out_slab = indices[b * seqlen + r0 : b * seqlen + r1]

                def tile_scores(t0, t1, scores_out = None):
                    if k_idx_chunk is not None:
                        return dsa_indexer_scores(
                            q_idx[b, r0:r1], w[b, r0:r1], k_idx_chunk[b][t0:t1],
                            pos0 - t0, 1, t1 - t0, scores = scores_out,
                        )
                    epp = idx_pool.shape[1]
                    bt = block_table[b]
                    if t0:
                        bt = bt[t0 // epp : -(-t1 // epp)]
                    return dsa_indexer_scores(
                        q_idx[b, r0:r1], w[b, r0:r1], idx_pool.view(-1, D_i),
                        pos0 - t0, 1, t1 - t0, scores = scores_out,
                        block_table = bt, epp = epp,
                    )

                dev = x.device
                # One fixed score-row stride (the tile width) for every slab and tile: the
                # stride is a kernel constexpr, so letting it follow the visible length
                # recompiled the scoring kernel on every new 128-key boundary
                s_stride = -(-t_tile // 128) * 128
                s_backing = g_tensor_cache.get(dev, (slab * s_stride,), torch.half, "dsa_stile")

                if t_slab <= t_tile:
                    sc = tile_scores(0, t_slab, s_backing[: rows * s_stride].view(rows, s_stride))
                    ext.dsa_topk(sc, out_slab, k_sel, None, 0)
                    continue

                if (EXL3_DSA_DEC_NOTILE and seqlen <= 8) or _pf_notile_ok(rows, t_slab):
                    # Decode/verify (and prefill under EXL3_DSA_PF_NOTILE): one full-slab score transient and one dsa_topk call.
                    # No backing: dsa_indexer_scores rounds the row stride (a kernel constexpr)
                    # up to a power of two. A (rows, t_slab) backing failed its stride % 8 assert
                    # and would recompile the scorer on every new visible length
                    sc = tile_scores(0, t_slab)
                    ext.dsa_topk(sc, out_slab, k_sel, None, 0)
                    # Score-descending order, as the tiled merge emits (see _indexer_topk_kpool)
                    sel = out_slab[:, :k_sel]
                    _, o = sc.gather(1, sel.long()).topk(k_sel, dim = 1)
                    out_slab[:, :k_sel] = sel.gather(1, o)
                    continue

                # Tiled path: fixed-size score/index backings, running (score, index) top-k
                # candidate set merged tile by tile
                i_backing = g_tensor_cache.get(dev, (slab * k_pad,), torch.int32, "dsa_itile")
                run_scr = torch.full((rows, k_sel), -float("inf"), dtype = torch.half, device = dev)
                run_idx = torch.full((rows, k_sel), -1, dtype = torch.int32, device = dev)
                for t0 in range(0, t_slab, t_tile):
                    t1 = min(t0 + t_tile, t_slab)
                    sc = tile_scores(t0, t1, s_backing[: rows * s_stride].view(rows, s_stride))
                    k_t = min(k_sel, t1 - t0)
                    kp_t = -(-k_t // 32) * 32
                    ti = i_backing[: rows * kp_t].view(rows, kp_t)
                    ext.dsa_topk(sc, ti, k_t, None, 0)
                    t_scr = sc.gather(1, ti.clamp_min(0).long())
                    t_scr = t_scr.masked_fill(ti < 0, -float("inf"))
                    t_idx = torch.where(ti >= 0, ti + t0, ti)
                    cand_scr = torch.cat((run_scr, t_scr), dim = 1)
                    cand_idx = torch.cat((run_idx, t_idx), dim = 1)
                    run_scr, sel = cand_scr.topk(k_sel, dim = 1)
                    run_idx = cand_idx.gather(1, sel)
                out_slab.fill_(-1)
                out_slab[:, :k_sel] = torch.where(
                    run_scr > -float("inf"), run_idx, run_idx.new_full((), -1))
        return indices


    def _update_pool_plane(self, idx_layer, block_table, host_seqlens, seqlen):
        """
        Incrementally extend the pooled-key plane with the pools completed by this chunk's
        append. A pool's key is the (gate_scores + APE)-softmax-weighted mean of its
        index_kpool members' keys; members are read back from the packed per-token plane
        (already appended), so pools straddling chunk boundaries resolve correctly. Complete
        pools are immutable, so each is written exactly once.
        """
        P, D_i = self.index_kpool, self.index_head_dim
        plane = idx_layer.get_idx()
        epp = plane.shape[1]
        flat = plane.view(-1, 2 * D_i)
        ape = self.idx_kpool_ape
        for b in range(block_table.shape[0] if block_table.dim() == 2 else 1):
            pos0 = host_seqlens[b]
            first_pool = pos0 // P
            last_pool = (pos0 + seqlen) // P
            n_new = last_pool - first_pool
            if n_new <= 0:
                continue
            bt = block_table[b]
            tok = torch.arange(first_pool * P, last_pool * P, device = plane.device)
            rows = bt[tok // epp].long() * epp + tok % epp
            members = flat[rows].view(n_new, P, 2 * D_i).float()
            keys, gates = members[..., :D_i], members[..., D_i:]
            probs = torch.softmax(gates + ape.unsqueeze(0), dim = 1)
            pool_keys = (probs * keys).sum(dim = 1).to(torch.half)
            # torch.full, not torch.tensor(device=): a pageable H2D copy blocks the host until
            # the GPU queue drains (one full-queue stall per completed pool per DSA layer)
            idx_layer.update_pool_direct(
                torch.full((1,), first_pool, dtype = torch.int32, device = plane.device),
                bt.unsqueeze(0) if bt.dim() == 1 else bt,
                pool_keys.unsqueeze(0),
            )


    def _indexer_topk_kpool(self, x, params, q_resid, bsz, seqlen, host_seqlens,
                            k_idx_chunk = None, idx_pool = None, pool_plane = None,
                            block_table = None):
        """
        K-pool indexer selection (GLM5.3): scoring and top-k run over pooled keys
        (index_kpool consecutive tokens per entry), selections expand back to raw indices,
        and the query's incomplete tail pool is appended as raw tokens. Paged path scores
        the incrementally maintained pooled plane through the tiled scorer (compress_rate
        gives the per-query causal pool bound); the cache-less path (k_idx_chunk, packed
        [k || gate] rows) builds pools eagerly. Returns -1-padded int32 indices.
        """
        import torch.nn.functional as F
        from .attention_fn.dsa_triton import dsa_indexer_scores
        from ..ext import exllamav3_ext as ext
        from ..util.tensor import g_tensor_cache

        H_i, D_i, P = self.index_n_heads, self.index_head_dim, self.index_kpool
        q_idx = self._dec_proj(self.idx_wq_b, q_resid, params).view(bsz, seqlen, H_i, D_i).contiguous()
        w = self.idx_weights.forward(x, params)

        t_max = max(host_seqlens) + seqlen
        sel_pools_max = min(self.index_topk // P, t_max // P)
        out_w = max(sel_pools_max * P + (P - 1 if self.index_kpool_tail else 0), 1)
        k_pad = -(-out_w // 32) * 32
        indices = torch.full((bsz * seqlen, k_pad), -1, dtype = torch.int32, device = x.device)

        def append_tail(b, base_cols):
            if not self.index_kpool_tail or P <= 1:
                return
            q_pos = host_seqlens[b] + torch.arange(seqlen, device = x.device)
            vis = q_pos + 1
            tail_count = vis.remainder(P)
            tail_start = vis - tail_count
            toff = torch.arange(P - 1, device = x.device)
            tail = tail_start.unsqueeze(1) + toff.unsqueeze(0)
            tail = torch.where(toff.unsqueeze(0) < tail_count.unsqueeze(1),
                               tail, tail.new_full((), -1))
            rows = indices[b * seqlen : (b + 1) * seqlen]
            rows[:, base_cols : base_cols + P - 1] = tail.int()

        if pool_plane is not None:
            # Paged pooled plane: same slab/tile structure as _indexer_topk, entries are
            # pools. dsa_indexer_scores' causal bound (q_pos0 + r + 1) // compress_rate is
            # exactly the visible-complete-pool count
            wf = w  # raw head weights, scales folded in-kernel
            epp = pool_plane.shape[1]
            slab = 256
            t_tile = max(epp, _score_tile // epp * epp)
            for b in range(bsz):
                pools_b = (host_seqlens[b] + seqlen) // P
                base_cols = 0
                fused_tail = False
                for r0 in range(0, seqlen, slab):
                    r1 = min(r0 + slab, seqlen)
                    rows = r1 - r0
                    pos0 = host_seqlens[b] + r0
                    pools_slab = (host_seqlens[b] + r1) // P
                    k_sel = min(self.index_topk // P, pools_slab)
                    if k_sel <= 0:
                        continue
                    out_slab = indices[b * seqlen + r0 : b * seqlen + r1]

                    def tile_scores(t0, t1, scores_out = None):
                        bt = block_table[b]
                        if t0:
                            bt = bt[t0 // epp : -(-t1 // epp)]
                        return dsa_indexer_scores(
                            q_idx[b, r0:r1], wf[b, r0:r1], pool_plane.view(-1, D_i),
                            pos0 - t0 * P, P, t1 - t0, scores = scores_out,
                            block_table = bt, epp = epp,
                        )

                    dev = x.device
                    # Fixed score-row stride (see _indexer_topk): a constexpr in the kernel
                    s_stride = -(-t_tile // 128) * 128
                    s_backing = g_tensor_cache.get(dev, (slab * s_stride,), torch.half, "dsa_stile")
                    kp_sel = -(-k_sel // 32) * 32

                    if pools_slab <= t_tile:
                        sc = tile_scores(0, pools_slab,
                                         s_backing[: rows * s_stride].view(rows, s_stride))
                        pool_idx = torch.empty((rows, kp_sel), dtype = torch.int32, device = dev)
                        ext.dsa_topk(sc, pool_idx, k_sel, None, 0)
                        pool_idx = pool_idx[:, :k_sel]
                    elif (EXL3_DSA_DEC_NOTILE and seqlen <= 8) or _pf_notile_ok(rows, pools_slab):
                        sc = tile_scores(0, pools_slab)   # pow2 stride, see _indexer_topk
                        pool_idx = torch.empty((rows, kp_sel), dtype = torch.int32, device = dev)
                        ext.dsa_topk(sc, pool_idx, k_sel, None, 0)
                        pool_idx = pool_idx[:, :k_sel]
                        # Same score-descending order as the tiled merge's final topk: the sparse
                        # attention accumulates in index order, so order is not free
                        _, o = sc.gather(1, pool_idx.long()).topk(k_sel, dim = 1)
                        pool_idx = pool_idx.gather(1, o)
                    else:
                        i_backing = g_tensor_cache.get(dev, (slab * kp_sel,), torch.int32, "dsa_itile")
                        run_scr = torch.full((rows, k_sel), -float("inf"), dtype = torch.half, device = dev)
                        run_idx = torch.full((rows, k_sel), -1, dtype = torch.int32, device = dev)
                        for t0 in range(0, pools_slab, t_tile):
                            t1 = min(t0 + t_tile, pools_slab)
                            sc = tile_scores(t0, t1, s_backing[: rows * s_stride].view(rows, s_stride))
                            k_t = min(k_sel, t1 - t0)
                            kp_t = -(-k_t // 32) * 32
                            ti = i_backing[: rows * kp_t].view(rows, kp_t)
                            ext.dsa_topk(sc, ti, k_t, None, 0)
                            t_scr = sc.gather(1, ti.clamp_min(0).long())
                            t_scr = t_scr.masked_fill(ti < 0, -float("inf"))
                            t_idx = torch.where(ti >= 0, ti + t0, ti)
                            cand_scr = torch.cat((run_scr, t_scr), dim = 1)
                            cand_idx = torch.cat((run_idx, t_idx), dim = 1)
                            run_scr, sel = cand_scr.topk(k_sel, dim = 1)
                            run_idx = cand_idx.gather(1, sel)
                        pool_idx = torch.where(
                            run_scr > -float("inf"), run_idx, run_idx.new_full((), -1))

                    # Expand pools to raw token indices (one kernel when set;
                    # single-slab only so the tail offset is this slab's expand width)
                    if EXL3_DSA_GLUE_FUSE and seqlen <= 256:
                        from .attention_fn.dsa_triton import dsa_glue_expand
                        base_cols = max(base_cols, dsa_glue_expand(
                            pool_idx, out_slab, P, host_seqlens[b] + r0,
                            bool(self.index_kpool_tail and P > 1)))
                        fused_tail = True
                    else:
                        exp = (pool_idx.unsqueeze(-1) * P +
                               torch.arange(P, dtype = torch.int32, device = dev)).flatten(1)
                        exp = torch.where(
                            pool_idx.unsqueeze(-1).expand(-1, -1, P).flatten(1) >= 0,
                            exp, exp.new_full((), -1))
                        out_slab[:, : exp.shape[1]] = exp
                        base_cols = max(base_cols, exp.shape[1])
                if not fused_tail:
                    append_tail(b, base_cols)
            return indices

        # Cache-less eager path (parity/compare): pools built from the packed chunk rows
        for b in range(bsz):
            t_total = host_seqlens[b] + seqlen
            plane = k_idx_chunk[b][:t_total]
            keys, gates = plane[:, :D_i].float(), plane[:, D_i:].float()

            n_pools = t_total // P
            base_cols = 0
            if n_pools:
                gk = keys[: n_pools * P].view(n_pools, P, D_i)
                gg = gates[: n_pools * P].view(n_pools, P, D_i)
                probs = torch.softmax(gg + self.idx_kpool_ape.unsqueeze(0), dim = 1)
                pool_keys = (probs * gk).sum(dim = 1)

                q_pos = host_seqlens[b] + torch.arange(seqlen, device = x.device)
                pool_end = torch.arange(n_pools, device = x.device) * P + (P - 1)
                select_k = min(self.index_topk // P, n_pools)
                qf, wb = q_idx[b], w[b]
                # Query-chunked: the [rows, H_i, n_pools] fp32 score cube is 8 GiB per copy at 16K tokens.
                # Rows are independent, so every chunk computes the same per-row values.
                step = _kpool_eager_rows if _kpool_eager_rows > 0 else seqlen
                top_sc_l, top_pool_l = [], []
                for r0 in range(0, seqlen, step):
                    r1 = min(r0 + step, seqlen)
                    sc = torch.einsum("shd,pd->shp", qf[r0:r1].float(), pool_keys)
                    sc = F.relu(sc * (D_i ** -0.5))
                    sc = torch.einsum("sh,shp->sp", wb[r0:r1].float() * (H_i ** -0.5), sc)
                    sc = sc.masked_fill(pool_end.unsqueeze(0) > q_pos[r0:r1].unsqueeze(1), -float("inf"))
                    ts, tp = sc.topk(select_k, dim = -1)
                    del sc
                    top_sc_l.append(ts); top_pool_l.append(tp)
                top_sc = top_sc_l[0] if len(top_sc_l) == 1 else torch.cat(top_sc_l)
                top_pool = top_pool_l[0] if len(top_pool_l) == 1 else torch.cat(top_pool_l)
                exp = (top_pool.unsqueeze(-1) * P +
                       torch.arange(P, device = x.device)).view(seqlen, -1)
                exp = torch.where(
                    (top_sc > -float("inf")).unsqueeze(-1).expand(-1, -1, P).reshape(seqlen, -1),
                    exp, exp.new_full((), -1))
                indices[b * seqlen : (b + 1) * seqlen, : exp.shape[1]] = exp.int()
                base_cols = exp.shape[1]
            append_tail(b, base_cols)

        return indices

    def _l4_scale(self, bsz, seqlen, position, positions, position_ids, device) -> torch.Tensor:
        """Llama-4 query scale, 1 + beta * ln(1 + pos // original_max), as (R, 1, 1) fp16.
        Position resolution mirrors the rope kernel: position_ids > positions > position."""
        if position_ids is not None:
            pos = position_ids.view(bsz, -1)[:, :seqlen].float()
        elif positions is not None:
            pos = positions.view(bsz, 1).float() + torch.arange(seqlen, device = device, dtype = torch.float)
        else:
            pos = (position + torch.arange(seqlen, device = device, dtype = torch.float)).expand(bsz, seqlen)
        scale = 1.0 + self.l4_beta * torch.log1p(torch.floor(pos / self.l4_original))
        return scale.to(torch.half).reshape(bsz * seqlen, 1, 1)


    def _attend_pre(self, x, bsz, seqlen, params, position, positions, position_ids, inv_freq,
                    use_mha = False, absorb_row0 = 0):
        """Context-length-independent head of _attend: q/kv projections, norms, rope and W_UK
        absorption. Returns (q, q_resid, q_pe, ckv, k_pe, q_lat, q_pe_hm); q_lat/q_pe_hm are None
        on the MHA path. Graph-capturable when rope and the l4 scale are off (no position use)."""
        from .attention_fn.mla_triton import _dbg_sync
        H = self.num_q_heads
        R = bsz * seqlen
        q_lat = q_pe_hm = None
        # Queries
        _dbg_sync("attend-entry (upstream modules)", x.device)
        # Batch-1 decode: q_a_proj and kv_a_proj_with_mqa in one exl3_dec_gemv_multi launch (the
        # fused path is a no-op above one row, so this is computed unconditionally)
        fused = self._dec_proj_pair(x, params)
        q, q_resid = self.project_q(x, params, return_resid = True, fused = fused)
        q = q.view(R, H, self.qk_head_dim)
        if self.l4_beta:
            # Whole query (nope + pe), before the split so the absorbed and MHA paths both
            # inherit it; a per-token scalar commutes with the rotation
            q *= self._l4_scale(bsz, seqlen, position, positions, position_ids, x.device)
        _dbg_sync("project_q", x.device)
        q_nope = q[:, :, :self.qk_nope_head_dim]
        q_pe = q[:, :, self.qk_nope_head_dim:].reshape(bsz, seqlen, H, self.qk_rope_head_dim)

        # Latent K/V. The normalized latent is what gets cached, matching the reference order
        # (kv_a_layernorm is applied before kv_b_proj would be)
        ckv_kpe = self._dec_proj(self.kv_a_proj_with_mqa, x, params, fused)
        _dbg_sync("kv_a_proj", x.device)
        ckv = self.kv_a_layernorm.forward(
            ckv_kpe[..., :self.kv_lora_rank].contiguous(), params, out_dtype = torch.half
        )
        k_pe = ckv_kpe[..., self.kv_lora_rank:].reshape(bsz, seqlen, 1, self.qk_rope_head_dim).contiguous()
        _dbg_sync("kv_a_layernorm+slice", x.device)

        if self.rope is not None:
            q_pe, k_pe = self.rope.apply(
                q_pe, k_pe, position, positions, position_ids, True,
                None, None, self.norm_eps, 0.0, inv_freq,
            )
            _dbg_sync("rope", x.device)

        if not use_mha:
            # Absorb W_UK into the queries, per head, straight from the flat layout. This runs as
            # a Triton kernel rather than a cuBLAS batched GEMM: the strided-batched fp16 form
            # made cuBLAS pick an SM120 nvjet TMA kernel that intermittently MMU-faults (caught
            # with cuda-gdb after presenting as flaky illegal-memory-access crashes whose
            # incidence tracked allocation layout), and the kernel reads any strides for free
            # absorb_row0 > 0 (dense-head split, see _attend): rows below it take the MHA path
            # and never read q_lat, so only the sparse tail rows are absorbed
            r0 = absorb_row0
            q_lat = mla_absorb(q.view(R, H, self.qk_head_dim)[r0:], self.w_uk_flat, H, self.qk_nope_head_dim)
            q_pe_hm = q_pe.reshape(R, H, self.qk_rope_head_dim)[r0:].permute(1, 0, 2).contiguous()
        return q, q_resid, q_pe, ckv, k_pe, q_lat, q_pe_hm


    def _attend(self, x, bsz, seqlen, params, ckv_cache, kpe_cache, block_table, cache_seqlens,
                append, qc = None, host_seqlens = None, idx_layer = None):
        """Projections, absorption, attention and o_proj, shared by the cached and cache-less
        paths. `append` writes the new latent/rope rows into the supplied page tensors.
        `idx_layer` is the cache layer holding the paged indexer-key plane (full-indexer layers
        on the cached path only)."""
        position = params.get("position", 0)
        positions = get_for_device(params, "positions", self.device, None)
        position_ids = get_for_device(params, "position_ids", self.device, None)
        inv_freq = get_for_device(params, "inv_freq", self.device, None)
        causal = params.get("causal", True)

        H = self.num_q_heads
        R = bsz * seqlen

        from .attention_fn.mla_triton import _dbg_sync

        # Sparse DSA applies once the visible context exceeds the selection budget; below that,
        # top-k selection is all-inclusive and the dense path is bit-equivalent
        if self.indexer_mode is not None:
            assert causal, "DSA indexer layers are causal-only"
            assert host_seqlens is not None
            sparse = max(host_seqlens) + seqlen > self.index_topk
        else:
            sparse = False

        use_mha = seqlen > MAX_DECODE_QLEN and _prefill_mode == "mha" and causal and not sparse
        # EXL3_BIG_TAIL_DENSE=1: a sparse prefill chunk that starts below index_topk (the big first
        # chunk of EXL3_PREFILL_BIG_TAIL) splits at the first row whose causal span exceeds the
        # budget. Rows [0, head) see <= index_topk keys, where top-k selects everything, so they
        # take the dense MHA path exactly as the 2048-row policy's first chunk does; rows
        # [head, seqlen) keep DSA with the chunk's full indexer selection
        head = 0
        if sparse and bsz == 1 and seqlen > MAX_DECODE_QLEN and _prefill_mode == "mha" and \
                "_mla_pre" not in params and not params.get("_mla_defer_post") and \
                os.environ.get("EXL3_BIG_TAIL_DENSE", "0") == "1":
            head = self.index_topk - int(host_seqlens[0])
            if not (MAX_DECODE_QLEN < head < seqlen):
                head = 0
        # Piecewise block graph (block_graph.py) runs the projections as a captured segment and
        # hands their outputs in; otherwise compute them here
        pre = params.pop("_mla_pre", None)
        if pre is None or use_mha:
            pre = self._attend_pre(x, bsz, seqlen, params, position, positions, position_ids,
                                   inv_freq, use_mha, absorb_row0 = head)
        q, q_resid, q_pe, ckv, k_pe, q_lat, q_pe_hm = pre

        append(ckv, k_pe)

        # Full-indexer layers compute this chunk's indexer keys unconditionally: on the cached
        # path they must land in the paged plane even while the context is still dense, so the
        # selection has complete history once it activates
        k_idx = None
        if self.indexer_mode == "full":
            k_idx = self._indexer_keys(x, params, position, positions, position_ids, inv_freq)
            if idx_layer is not None:
                idx_layer.update_idx_direct(cache_seqlens, block_table, k_idx, seqlen)
                if self.index_kpool:
                    self._update_pool_plane(idx_layer, block_table, host_seqlens, seqlen)

        # EXL3_PF_SKIP: kv_only -- the caller wants the K/V rows just appended (and the indexer
        # key/pool planes written above) and nothing else. The MTP draft's prompt prefill is the
        # one such caller: the next draft step only attends over this cache, and prefill_ls
        # discards the block's own output anyway. Returning None here skips the indexer top-k,
        # the attention, o_proj and, in TransformerBlock, the whole MoE half.
        if params.get("kv_only"):
            return None

        if sparse:
            if self.indexer_mode == "full":
                if self.index_kpool:
                    indices = self._indexer_topk_kpool(
                        x, params, q_resid, bsz, seqlen, host_seqlens,
                        k_idx_chunk = k_idx if idx_layer is None else None,
                        pool_plane = idx_layer.get_pool() if idx_layer is not None else None,
                        block_table = block_table,
                    )
                else:
                    indices = self._indexer_topk(
                        x, params, q_resid, bsz, seqlen, host_seqlens,
                        position, positions, position_ids, inv_freq,
                        k_idx_chunk = k_idx if idx_layer is None else None,
                        idx_pool = idx_layer.get_idx() if idx_layer is not None else None,
                        block_table = block_table,
                    )
                params["dsa_topk_indices"] = indices
            else:
                indices = params.get("dsa_topk_indices")
                assert indices is not None, \
                    "shared-indexer DSA layer found no top-k selection in params"
                indices = to_device(indices, x.device)
            if head:
                return self._attend_split(q, q_pe, q_lat, head, seqlen, params, ckv_cache,
                                          kpe_cache, block_table, host_seqlens, indices, qc)
            return self._attend_sparse(
                q_lat, q_pe, bsz, seqlen, params, ckv_cache, kpe_cache, block_table, indices, qc,
                pool_len = max(host_seqlens) + seqlen,
            )

        if use_mha:
            # MHA-form prefill: everything (past and current chunk) is read back from the cache
            # and attended over per-head up-projections. RoPE produced q_pe as a copy (the strided
            # slice cannot reshape into a view), so fold it back into q's pe columns for the
            # kernel's packed [nope | pe] per-head rows
            q = q.view(R, H, self.qk_head_dim)
            q[:, :, self.qk_nope_head_dim:] = q_pe.reshape(R, H, self.qk_rope_head_dim)
            o = mla_attn_triton_prefill_mha(
                q,
                self.w_uk_flat, self.w_uv_flat,
                ckv_cache, kpe_cache, block_table, host_seqlens,
                bsz, seqlen, self.v_head_dim, self.qk_nope_head_dim, self.sm_scale,
                pre_appended_len = seqlen,
                qc = qc,
            )
            o = o.reshape(bsz, seqlen, H * self.v_head_dim)
            return self._dec_proj(self.o_proj, o, params)

        kernel = mla_attn_triton_decode if seqlen <= MAX_DECODE_QLEN else mla_attn_triton_prefill
        extra = {}
        if (bsz == 1 and 1 < seqlen <= EXL3_MLA_DEC_ROWLOOP_MAX and EXL3_MLA_ATTN_ROWLOOP and
                causal and params.get("dflash_verify")):
            # Verify rows as R one-row decodes (row r sees cache + r + 1 keys): the q_len = 2 tile
            # layout (block_h 8 x block_m 2, other split count) rounds row 0 differently from the
            # plain q_len = 1 decode; this keeps every row bitwise equal to plain decode
            o_lat = torch.empty_like(q_lat)
            for r in range(seqlen):
                kw = {}
                if host_cuts():
                    # The kernel allocates its own out per row otherwise (one at::empty per
                    # row per layer per round). Same shape, same dtype, fully overwritten by
                    # the split/combine kernel pair, so a persistent per-layer buffer is
                    # indistinguishable from the fresh one.
                    from ..util.tensor import g_tensor_cache
                    shape = (q_lat.shape[0], bsz, q_lat.shape[-1])
                    kw["out"] = g_tensor_cache.get(
                        q_lat.device, shape, q_lat.dtype, f"mla_row_out_{self.layer_idx}")
                o_r = kernel(
                    q_lat[:, r : r + 1].contiguous(), q_pe_hm[:, r : r + 1].contiguous(),
                    ckv_cache, kpe_cache, block_table, cache_seqlens,
                    bsz = 1, q_len = 1, causal = causal, softmax_scale = self.sm_scale,
                    pre_appended_len = r + 1, qc = qc,
                    **kw,
                )
                o_lat[:, r : r + 1].copy_(o_r)
            if params.get("_mla_defer_post"):
                return o_lat
            return self.attend_post(o_lat, bsz, seqlen, params)
        o_lat = kernel(
            q_lat, q_pe_hm, ckv_cache, kpe_cache, block_table, cache_seqlens,
            bsz = bsz, q_len = seqlen,
            causal = causal, softmax_scale = self.sm_scale,
            pre_appended_len = seqlen,
            qc = qc,
            **extra,
        )

        from .attention_fn.mla_triton import _debug_sync
        if _debug_sync:
            # NaN/Inf in the attention output would reach the MoE router of the next block, and a
            # top-k over non-finite logits can select garbage expert ids -> wild pointer loads
            if not torch.isfinite(o_lat).all():
                bad = (~torch.isfinite(o_lat)).sum().item()
                raise RuntimeError(
                    f"MLA debug: non-finite attention output, layer {self.layer_idx}, "
                    f"{bad}/{o_lat.numel()} elements, bsz={bsz} seqlen={seqlen} dev={o_lat.device}")

        if params.get("_mla_defer_post"):
            return o_lat
        return self.attend_post(o_lat, bsz, seqlen, params)


    def attend_post(self, o_lat, bsz, seqlen, params):
        """Unfold W_UV per head from the flat layout (the kernel emits token-major output, so it
        feeds o_proj without a permute), then o_proj. Context-length independent; the piecewise
        block graph captures it with the MLP half."""
        o = mla_unfold(o_lat, self.w_uv_flat, self.v_head_dim)
        o = o.reshape(bsz, seqlen, self.num_q_heads * self.v_head_dim)
        return self._dec_proj(self.o_proj, o, params)


    def _attend_split(self, q, q_pe, q_lat, head, seqlen, params, ckv_cache, kpe_cache,
                      block_table, host_seqlens, indices, qc):
        """Batch-1 chunk split at `head` (see _attend): dense MHA prefill for rows [0, head)
        against the first host_seqlens[0] + head keys, DSA for the rest (q_lat already holds
        only the tail rows). The two halves meet before one o_proj."""
        from .attention_fn.dsa_triton import dsa_attn
        H = self.num_q_heads
        D_r = self.qk_rope_head_dim
        R = seqlen
        tail = seqlen - head
        q = q.view(R, H, self.qk_head_dim)
        q_pe = q_pe.reshape(R, H, D_r)
        qh = q[:head]
        qh[:, :, self.qk_nope_head_dim:] = q_pe[:head]
        o = torch.empty((R, H, self.v_head_dim), dtype = torch.half, device = q.device)
        o[:head] = mla_attn_triton_prefill_mha(
            qh,
            self.w_uk_flat, self.w_uv_flat,
            ckv_cache, kpe_cache, block_table, host_seqlens,
            1, head, self.v_head_dim, self.qk_nope_head_dim, self.sm_scale,
            pre_appended_len = head,
            qc = qc,
        )
        o_lat = dsa_attn(
            q_lat, ckv_cache, kpe_cache, block_table,
            indices = indices[head:], k_len = indices.shape[1],
            scale = self.sm_scale, page_size = ckv_cache.shape[1],
            q_pe = q_pe[head:], out_latent = True,
            qc = qc,
            pool_len = int(host_seqlens[0]) + seqlen,
        )
        o[head:] = mla_unfold(o_lat, self.w_uv_flat, self.v_head_dim).view(tail, H, self.v_head_dim)
        o = o.view(1, seqlen, H * self.v_head_dim)
        return self._dec_proj(self.o_proj, o, params)


    def _attend_sparse(self, q_lat, q_pe, bsz, seqlen, params, ckv_cache, kpe_cache,
                       block_table, indices, qc, pool_len = 0):
        """Gathered attention over the top-k selected latent rows (V3.2-on-MLA form of
        dsa_attn: no window, no sinks, V is the latent). The chunk's own rows are already in
        the paged pool (fp16 or packed-quantized; the packed form is dequantized online by the
        gather kernel) and the indexer's causal bound keeps the selection causal, so the
        kernel needs no mask of its own. The kernel reads the head-major absorbed queries and
        the token-major rope queries directly and emits the head-major latent output the
        unfold consumes: no packed query copy, no output slice/transpose, and the rope half
        of the weighted sum is never accumulated."""
        from .attention_fn.dsa_triton import dsa_attn

        H = self.num_q_heads
        R = bsz * seqlen
        D_r = self.qk_rope_head_dim

        # A single-row block table is shared by every query row inside dsa_attn (stride-0
        # lookup), so bsz 1 never materializes the (R, pages) expansion which would other-
        # wise be the one sparse-path transient that grows with context (pages)
        bt = block_table if bsz == 1 or seqlen == 1 \
            else block_table.repeat_interleave(seqlen, dim = 0)
        o_lat = dsa_attn(
            q_lat, ckv_cache, kpe_cache, bt,
            indices = indices, k_len = indices.shape[1],
            scale = self.sm_scale, page_size = ckv_cache.shape[1],
            q_pe = q_pe.reshape(R, H, D_r), out_latent = True,
            qc = qc,   # packed latent pages read online (scales, bits), or staged for prefill
            pool_len = pool_len,   # entries the selection can reference (context, not pool)
        )
        if params.get("_mla_defer_post"):
            return o_lat
        return self.attend_post(o_lat, bsz, seqlen, params)


    def decode_flash_attn(
        self,
        x: torch.Tensor,
        bsz: int,
        seqlen: int,
        params: dict,
    ):
        cache = params.get("cache")
        if self.has_split_cache:
            cache = self.tp_cache_lookup[cache]
        block_table = get_for_device(params, "block_table", self.device)
        cache_seqlens = get_for_device(params, "cache_seqlens", self.device)
        assert params.get("non_causal_spans") is None, \
            "MLAttention does not support non-causal spans"

        layer = cache if not hasattr(cache, "layers") else \
            cache.layers[self.layer_idx, params.get("layer_instance") or 0]

        # Graph-captured C++ path for the whole decode block (projections through o_proj as one
        # replayed CUDA graph). Falls back to the dispatch path for unsupported configurations,
        # including per-step declines (missing shared selection)
        if (
            seqlen <= MAX_DECODE_QLEN and bsz <= _bc_max_bsz and
            params.get("causal", True) and params.get("inv_freq") is None and
            # piecewise block graph (block_graph.py) wants the latent output, not o_proj's
            not params.get("_mla_defer_post")
        ):
            y = self.bc_mla_step(x, params, layer, block_table, cache_seqlens)
            if y is not None:
                return y

        from ..cache import CacheLayer_MLA_quant
        if isinstance(layer, CacheLayer_MLA_quant):
            # Packed latent feeds the kernels directly (online dequant, rotated domain); the rope
            # key pages are fp16 either way
            ckv_cache, sk, kpe_cache, bits = layer.get_qc()
            qc = (sk, bits)
        else:
            ckv_cache, kpe_cache = layer.get_kv(cache_seqlens, block_table)
            qc = None

        # Host-side lengths for the tiled prefill and the sparse-DSA decision, shared across
        # layers via the params dict; free when the generator passes host cache_seqlens
        if seqlen > MAX_DECODE_QLEN or self.indexer_mode is not None:
            host_seqlens = _host_seqlens(params, cache_seqlens)
            _check_block_table(block_table, host_seqlens, seqlen)
        else:
            host_seqlens = None

        return self._attend(
            x, bsz, seqlen, params, ckv_cache, kpe_cache, block_table, cache_seqlens,
            append = lambda ckv, k_pe:
                layer.update_kv_direct(cache_seqlens, block_table, ckv, k_pe, seqlen),
            qc = qc,
            host_seqlens = host_seqlens,
            idx_layer = layer if self.idx_plane_dim else None,
        )


    def bc_mla_step(self, x, params, layer, block_table, cache_seqlens):
        """Graph-captured decode block, or None when the module/cache-layer pair is not
        supported (caller falls back to the dispatch path)."""
        key = ("bcm", id(layer))
        bcm = self.dispatch_cache.get(key)
        if bcm is None:
            from .attention_fn.bc_mla import build_bc_mla
            bcm = self.dispatch_cache[key] = (build_bc_mla(self, layer) or False)
        if bcm is False:
            return None
        if self.indexer_mode is not None:
            # Host lengths for the dense/sparse regime decision; same key the dispatch path
            # uses, free when the generator passes host cache_seqlens
            _host_seqlens(params, cache_seqlens)
        position = params.get("position", 0)
        positions = get_for_device(params, "positions", self.device, None)
        position_ids = get_for_device(params, "position_ids", self.device, None)
        return bcm.step(
            x.contiguous(), params, cache_seqlens, block_table, position, positions, position_ids
        )


    def decode_flash_attn_nc(
        self,
        x: torch.Tensor,
        bsz: int,
        seqlen: int,
        params: dict,
    ):
        """Cache-less attention over the current chunk only, used by the quantization calibration
        pass. The chunk's own latent/rope rows go into a scratch page pool with an identity block
        table, so this runs the same kernels as the cached path rather than a separate variant."""
        from ..constants import PAGE_SIZE
        from .attention_fn.mla_triton import mla_kv_append

        assert params.get("cache") is None, "Cache provided for attn_mode: flash_attn_nc"

        pages = (seqlen + PAGE_SIZE - 1) // PAGE_SIZE
        dev = x.device
        # Only rows below seqlen are ever read (the kernel derives its bound from cache_seqlens +
        # pre_appended_len), so the page tail does not need initializing
        ckv_cache = torch.empty((bsz * pages, PAGE_SIZE, 1, self.kv_lora_rank),
                                dtype = torch.half, device = dev)
        kpe_cache = torch.empty((bsz * pages, PAGE_SIZE, 1, self.qk_rope_head_dim),
                                dtype = torch.half, device = dev)
        block_table = torch.arange(bsz * pages, dtype = torch.int32, device = dev).view(bsz, pages)
        cache_seqlens = torch.zeros((bsz,), dtype = torch.int32, device = dev)

        return self._attend(
            x, bsz, seqlen, params, ckv_cache, kpe_cache, block_table, cache_seqlens,
            append = lambda ckv, k_pe:
                mla_kv_append(
                    ckv.reshape(bsz, seqlen, self.kv_lora_rank),
                    k_pe.reshape(bsz, seqlen, self.qk_rope_head_dim),
                    ckv_cache, kpe_cache, block_table, cache_seqlens,
                ),
            host_seqlens = [0] * bsz,
        )


    def autosplit_extra_measure(self, params):
        """
        The (1, chunk)-at-context-0 pass this follows is NOT this module's memory worst case:
        sparse DSA replaces the MHA prefill with a different transient set once the context
        exceeds index_topk, and the BC decode slots allocate their statics only when a decode
        shape first occurs.

        Both are exercised here so an OoM lands where the loader advances to the next
        device, rather than after deployment. Outputs are discarded; only allocation shapes
        matter. The BC slots are configured but never run, so nothing is graph-captured at
        load time and the end-of-load tensor-cache drop leaves no baked pointers behind.
        """

        if os.environ.get("EXL3_AUTOSPLIT_WORSTCASE", "1") == "0":
            return
        cache = params.get("cache")
        if cache is None or self.device is None:
            return
        from ..cache import CacheLayer_MLA_quant, CacheLayer_MLA_fp16
        layer = cache if not hasattr(cache, "layers") else \
            cache.layers[self.layer_idx, params.get("layer_instance") or 0]
        quant = isinstance(layer, CacheLayer_MLA_quant)
        if not quant and not isinstance(layer, CacheLayer_MLA_fp16):
            return
        chunk = params["batch_shape"][1]

        # Decode statics: every buffer the (bsz <= MAX_BSZ, q_len <= 16) slot family can
        # request, both regimes. Backings are bucketed and shared across slots and layers,
        # so configuring the largest and smallest shapes bounds the whole family
        key = ("bcm", id(layer))
        bcm = self.dispatch_cache.get(key)
        if bcm is None:
            from .attention_fn.bc_mla import build_bc_mla
            bcm = build_bc_mla(self, layer)
        if bcm:
            from .attention_fn.bc_attn import MAX_BSZ
            regimes = (0, 1) if self.indexer_mode is not None else (0,)
            for b, q in ((1, 1), (MAX_BSZ, MAX_DECODE_QLEN)):
                for rg in regimes:
                    bcm._configure(b, q, rg)

        # Sparse prefill at maximum context. Synthetic state: every block-table entry aliases
        # page 0, zeroed so the math stays finite
        if self.indexer_mode is None:
            return
        from ..constants import PAGE_SIZE
        num_pages = (layer.qk if quant else layer.k).shape[0]
        t_syn = num_pages * PAGE_SIZE - chunk
        if t_syn + chunk <= self.index_topk:
            return   # cache too small to ever reach the sparse regime
        if quant:
            layer.qk[0].zero_()
            layer.sk[0].zero_()
        else:
            layer.k[0].zero_()
        layer.v[0].zero_()
        if self.idx_plane_dim:
            layer.get_idx()[0].zero_()
            if self.index_kpool:
                layer.get_pool()[0].zero_()
        p2 = {k2: v2 for k2, v2 in params.items() if k2 not in
              ("dev_cache", "_mla_host_seqlens", "positions", "position_ids")}
        p2["cache_seqlens"] = torch.tensor([t_syn], dtype = torch.int32)
        p2["block_table"] = torch.zeros((1, num_pages), dtype = torch.int32)
        p2["position"] = t_syn
        # Selections thread from full to shared layers exactly as in a real forward, via
        # the loader's shared params dict
        ind = params.get("_as_dsa_indices")
        if ind is not None:
            p2["dsa_topk_indices"] = ind
        x = torch.zeros((1, chunk, self.hidden_size), dtype = torch.half, device = self.device)
        self.forward(x, p2)
        if self.indexer_mode == "full":
            params["_as_dsa_indices"] = p2.get("dsa_topk_indices")


    def make_tp_allocation(self, options: dict) -> list[TPAllocation]:
        raise NotImplementedError()


    def tp_export(self, plan, producer):
        raise NotImplementedError("Tensor-parallel inference is not implemented for MLA layers")
