"""
DSpark draft heads of DeepSeek-V4.1 (the `mtp.*` stages): block drafter that attends to the target
model's hidden states (model.py:1034-1160, 1278).

Stage j is a normal block (hc + MoE with dspark_n_routed_experts) whose attention
  * keeps a sliding ring of the *target's* K (main_x -> wkv/kv_norm), seeded at prefill,
  * for a draft block of `dspark_block_size` positions attends to ring + all block keys (no causal mask).
Stage 0 owns main_proj/main_norm, the last stage owns norm, the markov head and the confidence head.
"""
from __future__ import annotations
import os
import torch
import torch.nn.functional as F
from . import ops, quant
from .linear import DenseLin


class DSparkHeads:
    def __init__(self, m):
        cfg, s, dt, dv = m.cfg, m.src, m.dtype, m.device
        self.m = m
        n = cfg.num_hidden_layers
        self.n_stages = cfg.n_mtp_layers
        p0, pl = f"mtp.0", f"mtp.{self.n_stages - 1}"
        self.main_proj = s.linear(f"{p0}.main_proj", dt, dv)
        self.main_norm = s.tensor(f"{p0}.main_norm", dt, dv)
        self.norm = s.tensor(f"{pl}.norm", dt, dv)
        self.markov_embed = s.tensor(f"{pl}.markov_head.embed", dt, dv)
        self.markov_head = s.tensor(f"{pl}.markov_head.head", torch.float32, dv)
        if dv.type != "cpu":      # streamed 5x per draft pass: fp16 halves the bytes; draft-only, the target verifies everything
            self.markov_head = self.markov_head.half()
        self.conf = s.tensor(f"{pl}.confidence_head.proj", torch.float32, dv)   # [1, dim+rank]
        self.rhead = None                       # RestrictedHead: the draft looks at a vocabulary subset (see draft_head.py)
        self._mh = None                         # draft-only copy of the markov head (restricted rows and / or fp16)
        n_ids = int(os.environ.get("DSV41_DRAFT_HEAD_IDS", "0") or 0)
        fp16 = os.environ.get("DSV41_DRAFT_MARKOV_FP16", "0") == "1"
        if n_ids or fp16:
            self.set_draft_head(n_ids, markov_fp16=fp16)

    def set_draft_head(self, n_ids: int = 0, counts: torch.Tensor = None, markov_fp16: bool = False):
        """Draft-pass tuning that cannot change the text (verify always uses the full head and the full-precision
        weights; the draft only proposes): n_ids > 0 drafts with the first n_ids token ids (or the most frequent
        128-blocks of `counts`), markov_fp16 reads the markov head as fp16 (fp32 accumulate). Both only move the acceptance
        rate. set_draft_head(0) restores the exact draft."""
        from .draft_head import RestrictedHead
        m = self.m
        self.rhead = self._mh = None
        mh = self.markov_head
        if n_ids:
            V = m.cfg.vocab_size
            self.rhead = (RestrictedHead.from_counts(m.head, counts, n_ids) if counts is not None
                          else RestrictedHead.prefix(m.head, V, n_ids))
            mh = mh[self.rhead.ids].contiguous()
        if markov_fp16:
            mh = mh.half()
        if mh is not self.markov_head:
            self._mh = mh

    # -------------------------------------------------------------- attention
    def _attn(self, l, x, start_pos, main_x, st):
        m, cfg, w = self.m, self.m.cfg, self.m.lw[l]
        B, S, _ = main_x.shape
        win, rd, H, hd = cfg.sliding_window, cfg.qk_rope_head_dim, cfg.num_attention_heads, cfg.head_dim
        fc_t = m._rope_for(0, st.max_seq)
        ring = st.win[l]
        mkv = ops.rmsnorm(w.wkv(main_x), w.kv_norm, cfg.rms_norm_eps)
        ops.rope_apply_(mkv[..., -rd:], fc_t[start_pos:start_pos + S])
        quant.act_quant_fp8_(mkv, cfg.fp8_block_size)
        if start_pos == 0:
            if S <= win:
                ring[:B, :S] = mkv
            else:
                cut = S % win
                ring[:B, cut:win], ring[:B, :cut] = mkv[:, -win:].split([win - cut, cut], dim=1)
            return x
        B, blk, _ = x.shape
        fc = fc_t[start_pos + S:start_pos + S + blk]
        qr = ops.rmsnorm(w.wq_a(x), w.q_norm, cfg.rms_norm_eps)
        q = w.wq_b(qr).unflatten(-1, (H, hd))
        ops.rope_apply_(q[..., -rd:], fc)
        kv = ops.rmsnorm(w.wkv(x), w.kv_norm, cfg.rms_norm_eps)
        ops.rope_apply_(kv[..., -rd:], fc)
        quant.act_quant_fp8_(kv, cfg.fp8_block_size)
        ring[:B, start_pos % win] = mkv.squeeze(1)
        n_valid = min(win, start_pos + 1)
        keys = torch.cat([ring[:B, :n_valid], kv], dim=1)            # ring slots written so far + the block's own keys
        o = ops.shared_key_attention(q, keys, w.sink, hd ** -0.5)
        ops.rope_apply_(o[..., -rd:], fc, True)
        G = cfg.o_groups
        o = o.view(B, blk, G, -1)
        quant.act_quant_fp8_(o, cfg.fp8_block_size)
        o = m._wo_a(w, o)
        return w.wo_b(o.flatten(2))

    def _block(self, l, h, start_pos, pre_mix, main_x, st, taps):
        m, cfg, w = self.m, self.m.cfg, self.m.lw[l]
        if start_pos == 0:
            self._attn(l, h, 0, main_x, st)
            return h, pre_mix
        hc, it, he, ne = cfg.hc_mult, cfg.hc_sinkhorn_iters, cfg.hc_eps, cfg.rms_norm_eps
        resid = h
        a_pre, a_post, a_comb = ops.hc_mixes(h, w.hc_attn_fn, w.hc_attn_scale, w.hc_attn_base, hc, it, he, ne)
        x = ops.rmsnorm(ops.hc_pre(h, pre_mix), w.attn_norm, ne)
        x = self._attn(l, x, start_pos, main_x, st)
        if taps is not None:
            taps[f"S{l}.attn"] = x.clone()
        h = ops.hc_post(x, resid, a_post, a_comb)
        resid = h
        f_pre, f_post, f_comb = ops.hc_mixes(h, w.hc_ffn_fn, w.hc_ffn_scale, w.hc_ffn_base, hc, it, he, ne)
        x = ops.rmsnorm(ops.hc_pre(h, a_pre), w.ffn_norm, ne)
        x = m._moe(l, x, taps)
        if taps is not None:
            taps[f"S{l}.ffn"] = x.clone()
        h = ops.hc_post(x, resid, f_post, f_comb)
        return h, f_pre

    def _block_hip(self, l, S, pre, start_pos, main_x, st):
        """`_block` on the fused hyper-connection chain of the serving model (HipStream: hc_mix + hc_cn kernels instead of
        ~20 small torch ops per mix). Same dataflow as the trunk layers; returns the pre-mix of the ffn site."""
        m, w = self.m, self.m.lw[l]
        x = S.cn(pre, w.attn_norm)
        a_pre, a_post, a_comb = S.mix(w.hc_attn_fn, w.hc_attn_scale, w.hc_attn_base)
        S.push(self._attn(l, x, start_pos, main_x, st), a_post, a_comb)
        x = S.cn(a_pre, w.ffn_norm)
        f_pre, f_post, f_comb = S.mix(w.hc_ffn_fn, w.hc_ffn_scale, w.hc_ffn_base)
        S.push(m._moe(l, x), f_post, f_comb)
        return f_pre

    # -------------------------------------------------------------- entry points
    @torch.inference_mode()
    def feed(self, main_hidden: torch.Tensor, start_pos: int, st):
        """Seed the window ring of every stage with the target rows of positions start_pos.. (main_hidden
        [B,S,len(targets)*dim]) and do nothing else. The slot a stage writes depends on main_x alone, so this is
        what S `forward_spec` calls (start_pos+i, outputs discarded) leave behind, for the price of one wkv per
        stage instead of a full draft block with MoE."""
        m, cfg = self.m, self.m.cfg
        n, win, rd = cfg.num_hidden_layers, cfg.sliding_window, cfg.qk_rope_head_dim
        B, S, _ = main_hidden.shape
        main_x = ops.rmsnorm(self.main_proj(main_hidden), self.main_norm, cfg.rms_norm_eps)
        fc = m._rope_for(0, st.max_seq)[start_pos:start_pos + S]
        keep = min(S, win)                                    # position p lives in slot p % win; older rows are overwritten
        slots = torch.tensor([(start_pos + i) % win for i in range(S - keep, S)], device=main_x.device)
        for j in range(self.n_stages):
            w = m.lw[n + j]
            mkv = ops.rmsnorm(w.wkv(main_x), w.kv_norm, cfg.rms_norm_eps)
            ops.rope_apply_(mkv[..., -rd:], fc)
            quant.act_quant_fp8_(mkv, cfg.fp8_block_size)
            st.win[n + j][:B, slots] = mkv[:, S - keep:].to(st.win[n + j].dtype)

    @torch.inference_mode()
    def forward_spec(self, input_ids: torch.Tensor, main_hidden: torch.Tensor, start_pos: int, st, taps=None):
        """input_ids [B] (the token just produced), main_hidden [B,S,len(targets)*dim].
        Prefill returns None (only seeds the rings); decode returns (output_ids [B,blk+1], logits, confidence)."""
        m, cfg = self.m, self.m.cfg
        n = cfg.num_hidden_layers
        input_ids = input_ids.to(m.device)
        main_x = ops.rmsnorm(self.main_proj(main_hidden), self.main_norm, cfg.rms_norm_eps)
        blk = cfg.dspark_block_size
        draft = input_ids.new_full([input_ids.size(0), blk], cfg.dspark_noise_token_id)
        draft[:, 0] = input_ids
        x = F.embedding(draft, m.embed_w)
        h = x.unsqueeze(2).repeat(1, 1, cfg.hc_mult, 1)
        pre_mix = h.new_zeros(h.size(0), h.size(1), cfg.hc_mult, dtype=torch.float32)
        pre_mix[:, :, 0] = 1.0
        if taps is None and start_pos > 0 and getattr(getattr(m, "hc", None), "name", "") == "hip":
            S = m.hc.new(h.float())
            for j in range(self.n_stages):
                pre_mix = self._block_hip(n + j, S, pre_mix, start_pos, main_x, st)
            h = S.apply().view(h.shape)
            return self._head(h, pre_mix, input_ids, taps)
        for j in range(self.n_stages):
            h, pre_mix = self._block(n + j, h, start_pos, pre_mix, main_x, st, taps)
        if start_pos == 0:
            return None
        return self._head(h, pre_mix, input_ids, taps)

    def _head(self, h, pre_mix, input_ids, taps=None):
        cfg = self.m.cfg
        blk = cfg.dspark_block_size
        x = ops.hc_pre(h, pre_mix)
        xn = ops.rmsnorm(x, self.norm, cfg.rms_norm_eps)
        rh = self.rhead
        logits = rh(xn) if rh is not None else self.m.head_logits(xn)
        mh = self._mh if self._mh is not None else self.markov_head
        out = input_ids.new_empty(input_ids.size(0), blk + 1)
        out[:, 0] = input_ids
        embeds = []
        for i in range(blk):
            e = F.embedding(out[:, i], self.markov_embed)
            bias = F.linear(e.to(mh.dtype), mh).float()
            logits[:, i].add_(bias)
            embeds.append(e)
            a = logits[:, i].argmax(-1)
            out[:, i + 1] = a if rh is None else rh.ids[a]
        me = torch.stack(embeds, dim=1)
        conf = F.linear(torch.cat([x, me], dim=-1).float(), self.conf).squeeze(-1)
        return out, logits, conf
