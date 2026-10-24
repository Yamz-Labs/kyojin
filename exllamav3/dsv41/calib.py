"""
Calibration forward of DeepSeek-V4.1-Flash for the EXL3 quantizer: the activations every linear sees, per block.

Everything here is plain torch (CPU and HIP) and reuses the verified graph math of `DSV41Graph`
(model.py), so the inputs the quantizer sees are the inputs of the real model, not a re-derivation.

Pieces (the framework modules in modules/dsv41.py and the standalone streaming API both use them):

  HData / h_data_new / h_data_add     Hessian accumulator in the exact dict format `quantize_exl3` takes
  CalibGraph.run_block(...)           one trunk block on one calibration row (start_pos 0, any length),
                                      cross-layer hand-offs (pre_mix, compressed pools, indexer keys,
                                      top-k, candidate blocks) live in a per-row dict `ho`
  moe_calibrate(...)                  router assignment + per-expert input rows / intermediate activations
  expert_ffn(...)                     one routed/shared expert with Hessian hooks on its two inputs
  DSparkCalib                         MTP (dspark) stages: decode-style draft blocks anchored on row positions
  RowEmbed                            embedding + hc expansion of a row, engram injection

Row state: h is [1, L, hc_mult, dim] fp32 (hyper-connection streams). The only other per-row state is `ho`:
    ho["pre_mix"][l]        [1, L, hc] fp32, pre-mix produced by block l (consumed by block l + 1)
    ho["pool"][src]         compressed-KV pool of kv source layer `src`
    ho["idxk"][src]         indexer keys pool of kv source layer `src`
    ho["topk"][src]         top-k indices produced by index source `src`
    ho["cand"]              candidate-block mask of the candidate source layer
    ho["main"][l]           [1, L, dim] fp16/fp32 stream mean at the input of dspark target layer l (optional)
Every write is keyed by the block that makes it, so running a block twice (capture pass, then reference pass,
then state advance after quantization) is idempotent: it overwrites its own entries and reads only entries of
earlier blocks.
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from . import ops, quant
from .config import DSV41Config
from .model import DSV41Graph, SharedRt


# ---------------------------------------------------------------------------------------------- Hessians
def h_data_new(n: int, device, key: str = "") -> dict:
    """Empty Hessian record in the format of `Linear.init_H_data` (what `quantize_exl3` consumes)."""
    return {
        "H": torch.zeros(n, n, dtype=torch.float32, device=device),
        "first_key": key,
        "count": 0,
        "finalized": False,
        "num_total": 0,
        "inf_nan": torch.zeros(2, dtype=torch.long, device=device),
        "device": device,
    }


def h_data_add(hd: dict, x: torch.Tensor) -> None:
    """H += x^T x over the finite rows of x [..., n]. Non-finite rows are dropped and counted."""
    x = x.reshape(-1, x.shape[-1])
    xf = x.float()
    fin = torch.isfinite(xf).all(dim=1)
    hd["num_total"] += x.numel()
    hd["inf_nan"][0] += torch.isinf(xf).sum()
    hd["inf_nan"][1] += torch.isnan(xf).sum()
    if not bool(fin.all()):
        xf = xf[fin]
    if xf.shape[0]:
        hd["H"].addmm_(xf.T, xf)
    hd["count"] += xf.shape[0]


def h_data_merge(a: dict, b: dict) -> dict:
    """a += b (two device shards of the same Hessian; the proxy is a plain sum over rows)."""
    a["H"] += b["H"].to(a["H"].device)
    a["count"] += b["count"]
    a["num_total"] += b["num_total"]
    a["inf_nan"] += b["inf_nan"].to(a["inf_nan"].device)
    return a


class HessianSet:
    """Named Hessians of one block: name -> H_data. `add(name, x)` creates on first use (n from x)."""

    def __init__(self, device="cpu", enabled: bool = True):
        self.device = device
        self.enabled = enabled
        self.data: dict = {}

    def add(self, name: str, x: torch.Tensor):
        if not self.enabled:
            return
        hd = self.data.get(name)
        if hd is None:
            hd = self.data[name] = h_data_new(x.shape[-1], self.device, name)
        h_data_add(hd, x.to(self.device))

    def counts(self) -> dict:
        return {k: v["count"] for k, v in self.data.items()}


class RecordingLin:
    """Wraps a Lin / callable: records the input into a HessianSet, then applies it."""

    def __init__(self, fn, hset: HessianSet, name: str):
        self.fn, self.hset, self.name = fn, hset, name
        for a in ("out_features", "in_features"):
            if hasattr(fn, a):
                setattr(self, a, getattr(fn, a))

    def __call__(self, x):
        if self.name is not None:          # None: shares the Hessian of a sibling linear (same input)
            self.hset.add(self.name, x)
        return self.fn(x)


def wo_a_recorder(w_wo_a: torch.Tensor, cfg: DSV41Config, hset: HessianSet, name: str):
    """wo_a_fn for a dense [G*r, in] wo_a tensor that records one Hessian per group (`name.g`)."""
    G = cfg.o_groups
    wg = w_wo_a.view(G, cfg.o_lora_rank, -1)

    def fn(o):
        for g in range(G):
            hset.add(f"{name}.{g}", o[:, :, g])
        return torch.einsum("bsgd,grd->bsgr", o, wg)
    return fn


# ---------------------------------------------------------------------------------------------- states
class _Ring(dict):
    """dict of zero tensors created on demand: sliding-window rings per layer."""

    def __init__(self, shape, dtype, device):
        super().__init__()
        self._a = (shape, dtype, device)

    def __missing__(self, k):
        t = torch.zeros(self._a[0], dtype=self._a[1], device=self._a[2])
        self[k] = t
        return t


class RowState:
    """The slice of DSV41State one block needs (batch 1, start_pos 0). Pools of kv sources are created for the
    block that owns them; readers get the pool through `rt` from the hand-off dict."""

    def __init__(self, cfg: DSV41Config, max_seq: int, dtype, device, batch: int = 1):
        self.batch, self.max_seq = batch, max_seq
        self.win = _Ring((batch, cfg.sliding_window, cfg.head_dim), dtype, device)
        self.comp_kv, self.idx_k, self.comp_kv_state, self.comp_score_state = {}, {}, {}, {}
        self.hasher = None
        self._cfg, self._dt, self._dv = cfg, dtype, device

    def prepare_source(self, l: int):
        cfg = self._cfg
        r = cfg.ratio(l)
        B, hd = self.batch, cfg.head_dim
        self.comp_kv[l] = torch.zeros(B, self.max_seq // r, hd, dtype=self._dt, device=self._dv)
        self.idx_k[l] = torch.zeros(B, self.max_seq // r, cfg.index_head_dim, dtype=self._dt, device=self._dv)
        if r > 1:
            self.comp_kv_state[l] = torch.zeros(B, r, hd, dtype=torch.float32, device=self._dv)
            self.comp_score_state[l] = torch.full((B, r, hd), -torch.inf, dtype=torch.float32, device=self._dv)


def new_handoff() -> dict:
    return {"pre_mix": {}, "pool": {}, "idxk": {}, "topk": {}, "cand": None, "main": {}}


def identity_pre_mix(h: torch.Tensor, hc: int) -> torch.Tensor:
    pm = h.new_zeros(h.size(0), h.size(1), hc, dtype=torch.float32)
    pm[:, :, 0] = 1.0
    return pm


class RowEmbed:
    """Embedding + hc expansion of a row and the engram injection that precedes an engram layer."""

    def __init__(self, cfg: DSV41Config, embed_w: torch.Tensor):
        self.cfg, self.embed_w = cfg, embed_w

    def embed(self, input_ids: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
        assert 0 <= int(input_ids.min()) and int(input_ids.max()) < self.embed_w.shape[0], \
            "token id outside the embedding table (an out-of-range gather faults the GPU)"
        h = F.embedding(input_ids.to(self.embed_w.device), self.embed_w).to(dtype)
        return h.unsqueeze(2).repeat(1, 1, self.cfg.hc_mult, 1)


# ---------------------------------------------------------------------------------------------- graph
class CalibGraph(DSV41Graph):
    """DSV41Graph whose MoE is supplied as a callable `moe_fn(l, x[N,d]) -> y[N,d]` (fp32)."""

    def __init__(self, cfg: DSV41Config, dtype=torch.float32, device="cpu"):
        super().__init__(cfg.validate(), dtype, device)
        self.moe_fn = None

    def _moe(self, l, x, taps=None):
        shape = x.size()
        y = self.moe_fn(l, x.view(-1, self.cfg.hidden_size))
        return y.type_as(x).view(shape)

    def _row_setup(self, l: int, h: torch.Tensor, ho: dict):
        cfg = self.cfg
        L = h.size(1)
        st = RowState(cfg, L, h.dtype if h.dtype != torch.float16 else torch.float32, self.device, h.size(0))
        rt = SharedRt()
        ks, is_ = cfg.kv_source_of(l), cfg.index_source_of(l)
        if cfg.ratio(l):
            if ks is not None and ks != l:
                rt.compress_kv = ho["pool"][ks]
                rt.index_k = ho["idxk"][ks]
            if is_ is not None and is_ != l and is_ in ho["topk"]:
                rt.topk = ho["topk"][is_]
            rt.candidates = ho["cand"]
        if cfg.is_kv_source(l):
            st.prepare_source(l)
        pre_mix = identity_pre_mix(h, cfg.hc_mult) if l == 0 else ho["pre_mix"][l - 1]
        if l in cfg.dspark_target_layer_ids:
            ho["main"][l] = h.mean(dim=2)
        return st, rt, pre_mix

    def _row_commit(self, l: int, ho: dict, st, rt, f_pre):
        cfg = self.cfg
        ho["pre_mix"][l] = f_pre
        if cfg.is_kv_source(l):
            ho["pool"][l] = st.comp_kv[l]
            ho["idxk"][l] = st.idx_k[l]
        if cfg.is_index_source(l) and cfg.ratio(l):
            ho["topk"][l] = rt.topk
        if l == cfg.candidate_source_layer_id:
            ho["cand"] = rt.candidates

    def block_pre(self, l: int, h: torch.Tensor, ho: dict, taps: dict = None):
        """Attention half of trunk block l on a full row. Returns (x_ffn [1,L,d], ctx). The row's `ho` is
        committed by block_post."""
        st, rt, pre_mix = self._row_setup(l, h, ho)
        x, ctx = self._block_pre(l, h, 0, pre_mix, st, rt, taps)
        return x, (ctx, st, rt)

    def block_post(self, l: int, ctx, y: torch.Tensor, ho: dict, taps: dict = None) -> torch.Tensor:
        """FFN half: y [1,L,d] is the MoE + shared output for this row. Returns the next h."""
        c, st, rt = ctx
        out, f_pre = self._block_post(l, c, y.type_as(c[0]), taps)
        self._row_commit(l, ho, st, rt, f_pre)
        return out

    def run_block(self, l: int, h: torch.Tensor, ho: dict, taps: dict = None) -> torch.Tensor:
        """Trunk block `l` on a full row with `moe_fn`. h [1, L, hc, dim] -> next h. Updates `ho`."""
        x, ctx = self.block_pre(l, h, ho, taps)
        y = self._moe(l, x, taps)
        return self.block_post(l, ctx, y, ho, taps)

    def final_hidden(self, h: torch.Tensor, ho: dict) -> torch.Tensor:
        """hc_pre of the last block's output with its pending pre_mix (input of the final norm)."""
        return ops.hc_pre(h, ho["pre_mix"][self.cfg.num_hidden_layers - 1])


# ---------------------------------------------------------------------------------------------- MoE
def expert_ffn(x: torch.Tensor, w1, w3, w2, limit: float, hook_in=None, hook_mid=None, weights=None) -> torch.Tensor:
    """One expert on rows x [n, d]: w2(swiglu(w1 x, w3 x)). w* are callables (Linear modules / Lin) or dense
    [out, in] tensors. hook_in(x) sees the input of w1/w3, hook_mid(h) the UNSCALED input of w2 (what the EXL3
    kernels feed w2; the Hessians of the three linears). `weights` [n,1] (routing weight) is applied before w2 like
    the reference (model.py:843), which keeps the forward bitwise equal to it."""
    f = lambda w, t: F.linear(t, w.to(t.dtype)) if isinstance(w, torch.Tensor) else w(t)
    if hook_in is not None:
        hook_in(x)
    h = ops.swiglu(f(w1, x), f(w3, x), limit)
    if hook_mid is not None:
        hook_mid(h)
    if weights is not None:
        h = weights * h
    return f(w2, h.to(x.dtype))


def moe_route(cfg: DSV41Config, x: torch.Tensor, gate_w: torch.Tensor, gate_b: torch.Tensor, topk: int):
    """Router of the reference (model.py:811): (weights [N,k] fp32 incl. routed scaling, indices [N,k])."""
    return ops.route(x, gate_w, gate_b, topk, cfg.scoring_func, cfg.norm_topk_prob, cfg.routed_scaling_factor,
                     cfg.gate_temp)


class ExpertAssignment:
    """Which tokens of a layer go to which routed expert. `rows(e)` -> (token index, routing weight)."""

    def __init__(self, wts: torch.Tensor, idx: torch.Tensor, n_experts: int):
        self.wts, self.idx, self.n = wts, idx, n_experts
        flat = idx.flatten()
        self.counts = torch.bincount(flat, minlength=n_experts)
        order = torch.argsort(flat, stable=True)
        self._tok = order // idx.shape[1]
        self._wt = wts.flatten()[order]
        self._off = [0] + self.counts.cumsum(0).tolist()

    def rows(self, e: int):
        a, b = self._off[e], self._off[e + 1]
        return self._tok[a:b], self._wt[a:b]


def moe_calibrate(
    cfg: DSV41Config,
    x: torch.Tensor,
    gate_w: torch.Tensor,
    gate_b: torch.Tensor,
    topk: int,
    n_experts: int,
    expert_fn,
    shared_fn,
    mode: str = "routed",
    on_expert=None,
):
    """Full MoE of one layer on x [N, d] with calibration hooks.

    expert_fn(e, capture) -> (w1, w3, w2, hook_in, hook_mid): the expert's three linears and its Hessian hooks
    (callables). capture=False must return callables that record nothing (hooks None). Called per expert and
    dropped afterwards (streaming friendly).
    shared_fn(x) -> y  : shared expert (applies its own hooks)
    mode "routed": expert e sees only the tokens the router sent to it (the deployment distribution);
         experts without tokens are skipped (no Hessian -> the quantizer falls back to uncalibrated).
    mode "all":    every expert sees every token for the Hessians (the exllamav3 default, `activate_all_experts`);
         the layer output still only uses the routed contributions.
    Returns (y [N,d] fp32, ExpertAssignment). Output order of operations equals DSV41Model._moe.
    """
    wts, idx = moe_route(cfg, x, gate_w, gate_b, topk)
    asg = ExpertAssignment(wts, idx, n_experts)
    y = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for e in range(n_experts):
        ti, tw = asg.rows(e)
        if mode == "all":
            w1, w3, w2, hi, hm = expert_fn(e, True)
            expert_ffn(x, w1, w3, w2, cfg.swiglu_limit, hi, hm)
        if ti.numel():
            w1, w3, w2, hi, hm = expert_fn(e, mode == "routed")
            out = expert_ffn(x[ti], w1, w3, w2, cfg.swiglu_limit, hi, hm, tw[:, None])
            y.index_add_(0, ti, out.float())
        if on_expert is not None:
            on_expert(e, int(asg.counts[e]))
    y += shared_fn(x).float()
    return y, asg


# ---------------------------------------------------------------------------------------------- dspark
class DSparkCalib:
    """Calibration of the MTP (dspark) stages. The stages are exercised the way they run in serving: a draft
    block of `dspark_block_size` positions anchored at row position p (decode step, start_pos = p), attending
    to the target's K of the last `sliding_window` positions (main_proj/main_norm of the stream means at the
    dspark target layers) plus the draft block itself, no causal mask inside the block (model.py:1034-1160).

    Anchor a (position p): draft tokens [row[p + 1], noise x block]; stage 0 input is their embeddings
    expanded to hc streams; stage j + 1 consumes stage j's output. Anchors need p + 1 < L.

    `lw[l]` for l >= n_layers are layer-weight bundles (same _LW layout as the trunk, wq_a/wq_b/wkv/wo_b
    callables, wo_a tensor [G*r, in], sink, norms, hc params, gate); `main_proj` is a callable.
    """

    def __init__(self, graph: CalibGraph, embed_w: torch.Tensor, main_proj, main_norm: torch.Tensor):
        self.g, self.embed_w, self.main_proj, self.main_norm = graph, embed_w, main_proj, main_norm
        self.cfg = graph.cfg

    # ---- main_x for the whole row, from the three target taps
    def main_x(self, ho: dict) -> torch.Tensor:
        cfg = self.cfg
        taps = [ho["main"][l].float() for l in cfg.dspark_target_layer_ids]
        mh = torch.cat(taps, dim=-1)                                   # [1, L, 3*dim]
        return ops.rmsnorm(self.main_proj(mh), self.main_norm, cfg.rms_norm_eps)

    def anchors(self, L: int, n: int) -> list:
        """n anchor positions spread evenly over [1, L - 2] (decode steps need start_pos > 0 and the next token)."""
        lo, hi = 1, L - 2
        if hi < lo:
            return []
        n = min(n, hi - lo + 1)
        return sorted({lo + int(round(i * (hi - lo) / max(n - 1, 1))) for i in range(n)})

    def stage_mkv(self, l: int, mx: torch.Tensor) -> torch.Tensor:
        """K of the target positions as stage l sees them: wkv(main_x), kv_norm, rope, fp8 fake-quant."""
        cfg, w = self.cfg, self.g.lw[l]
        rd = cfg.qk_rope_head_dim
        L = mx.size(1)
        fc = self.g._rope_for(0, L + cfg.dspark_block_size + 2)
        mkv = ops.rmsnorm(w.wkv(mx), w.kv_norm, cfg.rms_norm_eps)
        ops.rope_apply_(mkv[..., -rd:], fc[:L])
        quant.act_quant_fp8_(mkv, cfg.fp8_block_size)
        return mkv

    def stage_attention(self, l: int, x: torch.Tensor, p: int, ring_kv: torch.Tensor) -> torch.Tensor:
        """Draft-block attention of stage l at anchor p. x [1, blk, dim] (normed), ring_kv [1, win, hd]."""
        cfg, w = self.cfg, self.g.lw[l]
        H, hd, rd, win, blk = cfg.num_attention_heads, cfg.head_dim, cfg.qk_rope_head_dim, cfg.sliding_window, x.size(1)
        fc_t = self.g._rope_for(0, p + blk + 2)
        fc = fc_t[p + 1:p + 1 + blk]
        qr = ops.rmsnorm(w.wq_a(x), w.q_norm, cfg.rms_norm_eps)
        q = w.wq_b(qr).unflatten(-1, (H, hd))
        ops.rope_apply_(q[..., -rd:], fc)
        kv = ops.rmsnorm(w.wkv(x), w.kv_norm, cfg.rms_norm_eps)
        ops.rope_apply_(kv[..., -rd:], fc)
        quant.act_quant_fp8_(kv, cfg.fp8_block_size)
        idx = torch.cat([torch.arange(min(win, p + 1)), win + torch.arange(blk)]).int()
        idx = idx.view(1, 1, -1).expand(1, blk, -1).contiguous().to(x.device)
        kv_all = torch.cat([ring_kv, kv], dim=1)
        o = ops.sparse_attention(q, kv_all, w.sink, idx, hd ** -0.5)
        ops.rope_apply_(o[..., -rd:], fc, True)
        G = cfg.o_groups
        o = o.view(1, blk, G, -1)
        quant.act_quant_fp8_(o, cfg.fp8_block_size)
        o = self.g._wo_a(w, o)
        return w.wo_b(o.flatten(2))

    def ring_for(self, mkv: torch.Tensor, p: int) -> torch.Tensor:
        win = self.cfg.sliding_window
        ring = torch.zeros(1, win, mkv.size(-1), dtype=mkv.dtype, device=mkv.device)
        for q in range(max(0, p - win + 1), p + 1):
            ring[0, q % win] = mkv[0, q]
        return ring

    def stage_pre(self, l: int, h: torch.Tensor, pre_mix: torch.Tensor, mkv: torch.Tensor, anchors: list,
                  taps: dict = None):
        """Attention half of stage l over all anchors (attention runs per anchor: positions differ).
        h [A, blk, hc, dim], pre_mix [A, blk, hc]. Returns (x_ffn [A, blk, dim], ctx)."""
        cfg, w = self.cfg, self.g.lw[l]
        hc, it, he, ne = cfg.hc_mult, cfg.hc_sinkhorn_iters, cfg.hc_eps, cfg.rms_norm_eps
        resid = h
        a_pre, a_post, a_comb = ops.hc_mixes(h, w.hc_attn_fn, w.hc_attn_scale, w.hc_attn_base, hc, it, he, ne)
        x = ops.rmsnorm(ops.hc_pre(h, pre_mix), w.attn_norm, ne)
        xs = [self.stage_attention(l, x[a:a + 1], p, self.ring_for(mkv, p)) for a, p in enumerate(anchors)]
        x = torch.cat(xs, dim=0)
        if taps is not None:
            taps[f"S{l}.attn"] = x.clone()
        h = ops.hc_post(x, resid, a_post, a_comb)
        resid = h
        f_pre, f_post, f_comb = ops.hc_mixes(h, w.hc_ffn_fn, w.hc_ffn_scale, w.hc_ffn_base, hc, it, he, ne)
        x = ops.rmsnorm(ops.hc_pre(h, a_pre), w.ffn_norm, ne)
        return x, (resid, f_pre, f_post, f_comb)

    def stage_post(self, l: int, ctx, y: torch.Tensor, taps: dict = None):
        """FFN half: hc_post of the MoE output y. Returns (h, pre_mix) for the next stage."""
        resid, f_pre, f_post, f_comb = ctx
        if taps is not None:
            taps[f"S{l}.ffn"] = y.clone()
        return ops.hc_post(y, resid, f_post, f_comb), f_pre

    def run_stage(self, l: int, h: torch.Tensor, pre_mix: torch.Tensor, mkv: torch.Tensor, anchors: list,
                  taps: dict = None):
        """One stage over all anchors with the graph's MoE. Returns (h, pre_mix) for the next stage."""
        x, ctx = self.stage_pre(l, h, pre_mix, mkv, anchors, taps)
        y = self.g._moe(l, x)
        return self.stage_post(l, ctx, y, taps)

    def draft_input(self, input_ids: torch.Tensor, anchors: list):
        cfg = self.cfg
        blk = cfg.dspark_block_size
        A = len(anchors)
        assert 0 <= cfg.dspark_noise_token_id < self.embed_w.shape[0] and int(input_ids.max()) < self.embed_w.shape[0], \
            "dspark noise / row token id outside the embedding table (an out-of-range gather faults the GPU)"
        toks = input_ids.new_full((A, blk), cfg.dspark_noise_token_id)
        for a, p in enumerate(anchors):
            toks[a, 0] = input_ids[0, p + 1]
        h = F.embedding(toks.to(self.embed_w.device), self.embed_w).float()
        h = h.unsqueeze(2).repeat(1, 1, cfg.hc_mult, 1)
        return h, identity_pre_mix(h, cfg.hc_mult)
