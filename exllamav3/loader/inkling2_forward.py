"""Inkling 2 Mini WHOLE-MODEL forward from a GGUF (llama.cpp #32695 convention), slow but exact: torch only, CPU or any torch device.

GGUF convention (arch `inkling2`): block_count = 48 = one layer index per APPLICATION; `num_loops` (3), `loop.core_start/end` (6 / 18),
`loop.gate_rank` (64). Tensors are stored once per PHYSICAL layer (blk.0 .. blk.23); the looped core (physical 6..17) runs `num_loops`
times: application il uses the weights of physical layer `loop_layer(il)`, `is_swa` comes from the sliding_window_pattern at the
APPLICATION index, block boundaries (`ip % attn_res_block_size == 0`) from the PHYSICAL index.

Maths (checked against modeling_inkling2.py of the HF checkpoint and src/models/inkling2.cpp of the llama.cpp PR):
  embed        rms_norm(tok_embd[id]) * token_embd_norm
  block depth  mix(blocks, cur, query): unweighted rms_norm of [blocks..., cur], softmax(q . s / sqrt(D)) over that set, weighted sum of
               the normed states. attn: 1 query (SWA layers) or 3 (global layers: q, k, v inputs); mlp: 1; head: 1.
  attention    q/k/v/gate/rel/sink projections of the (normed) inputs; k and v go through a depthwise causal conv (k = 4, taps stored
               oldest -> current; global layers add 4 per-token taps per kv head from `attn_{k,v}_conv_proj`), scaled (x + conv) / sqrt(k+1)
               (or sqrt(2k+1) when dynamic); q: per-head scale then L2 clip 22.63; k: L2 clip; v: unweighted rms_norm; r (16 per head): scale then
               L2 clip 8; global layers: q, r *= tau = 1 + 0.1 ln(max(1, (pos+1)/8192)). Bias = r . rel_bias[:, dist] for 0 <= dist < 1024,
               0 elsewhere; softmax over keys plus one sink logit (zero value); window 1024 on SWA layers; out * sigmoid(gate) -> wo -> post conv.
  residual     block start (ip % 6 == 0): blocks += [input]; cur = attn_out. else cur = input + attn_out.
  loop         every pass restarts from the prelude (blocks + cur at the core start); pass p > 0 mixes the PREVIOUS pass's blocks (bank =
               blocks[len(prelude)+1:], bank_last = its output) with 4 learned queries per layer (loop_res_query rows 2k, 2k+1 = attn, mlp;
               loop_kv_res_query = k, v) and merges them into q/k/v/mlp inputs: cur = (1 - tanh(F h)) cur + tanh(G h) msg, h = D [rms(cur); rms(msg)].
  moe          sigmoid router (F32) + selection bias, top-8 of 128, weights renormalised (eps 1e-20) x 8.0, gate_up = [gate; up] (first / second half),
               per-expert `.scale` on the gate_up and down outputs, situ(g, u) = 4 tanh(g/4) sigmoid(g) * 25 tanh(u/25).
  head         mix(output_res_query) -> rms * output_norm -> head -> 30 tanh(x/30) -> first 200058 logits.
Stored types handled: F32, F16, BF16, Q8_0, NVFP4 (ggml type 40). KV and conv state are SEPARATE per application (48 slots); the routed
experts stay raw in the file and are read per call (pread) for the experts the router picked. Activations stay fp32 (no activation quantisation).
"""
from __future__ import annotations

import math
import os
import time

import numpy as np
import torch

T_F32, T_F16, T_Q8_0, T_BF16, T_NVFP4 = 0, 1, 8, 30, 40
_QBLK = {T_Q8_0: (32, 34), T_NVFP4: (64, 36)}
EPS_MIX = 1e-6


def _gap():
    from ..foreign_quant import gap_formats
    return gap_formats


def _u8(g, off, n, dev=None):
    t = torch.from_numpy(np.ascontiguousarray(g._pread(off, n)))
    return t if dev is None else t.to(dev)             # bytes go to the device first: the dequant runs there


def dq_bytes(raw, ty, rows, K):
    """raw uint8 bytes of `rows` rows of K elements, stored as ggml type `ty` -> fp32 [rows, K]"""
    if ty == T_F32:
        return raw.view(torch.float32).reshape(rows, K)
    if ty == T_F16:
        return raw.view(torch.float16).reshape(rows, K).float()
    if ty == T_BF16:
        return raw.view(torch.bfloat16).reshape(rows, K).float()
    if ty == T_Q8_0:
        b = raw.reshape(rows * (K // 32), 34)
        d = b[:, 0:2].contiguous().view(torch.float16).float()
        return (d * b[:, 2:34].contiguous().view(torch.int8).float()).reshape(rows, K)
    if ty == T_NVFP4:
        return _gap().dequant("nvfp4_gguf", raw.reshape(rows, -1)).float().reshape(rows, K)
    raise ValueError(f"ggml type {ty} not handled by the Inkling2 forward")


def row_bytes(ty, K):
    if ty in (T_F32,):
        return 4 * K
    if ty in (T_F16, T_BF16):
        return 2 * K
    qk, bb = _QBLK[ty]
    return K // qk * bb


def nvfp4_gguf_to_modelopt(raw):
    """GGUF type 40 bytes [N, nb*36] (per 64 values: 4 UE4M3 scale bytes, then 4 x 8 bytes: low nibbles = values 0..7 of a 16-group, high = 8..15)
    -> ModelOpt layout read by ext.fq_gemv_nvfp4: packed [N, K/2] (value 2i in the low nibble) + scales [N, K/16] (same E4M3 bytes, sign 0).
    Same real values: ggml scale x 0.5 x kvalues (= 2 x E2M1) == E4M3 scale x E2M1 (global scale 1.0)."""
    N = raw.shape[0]
    b = raw.reshape(N, -1, 36)
    scales = b[..., :4].reshape(N, -1).contiguous()
    qs = b[..., 4:].reshape(N, -1, 4, 8)
    nib = torch.cat([qs & 15, qs >> 4], -1).reshape(N, -1, 4, 8, 2)                  # 16 values per group in order, as pairs
    packed = (nib[..., 0] | (nib[..., 1] << 4)).reshape(N, -1).contiguous()
    return packed, scales


def read_dense(g, name, dev=None):
    """a 1-D / 2-D GGUF tensor as fp32 in storage orientation: [N, K] for dims [K, N] (out x in), [K] for a vector"""
    ty, dims, off = g.infos[name]
    if len(dims) == 1:
        return dq_bytes(_u8(g, off, row_bytes(ty, dims[0]), dev), ty, 1, dims[0])[0]
    K, N = dims[0], dims[1]
    return dq_bytes(_u8(g, off, N * row_bytes(ty, K), dev), ty, N, K)


def read_rows(g, name, rows, dev=None):
    """rows `rows` (list of int) of a 2-D tensor [N, K] (e.g. the embedding) as fp32 [len(rows), K]"""
    ty, dims, off = g.infos[name]
    K = dims[0]
    rb = row_bytes(ty, K)
    raw = torch.cat([_u8(g, off + r * rb, rb) for r in rows])
    return dq_bytes(raw if dev is None else raw.to(dev), ty, len(rows), K)


def read_expert(g, name, e, dev=None):
    """expert e of a [K, N, E] tensor -> fp32 [N, K]"""
    ty, dims, off = g.infos[name]
    K, N = dims[0], dims[1]
    nb = N * row_bytes(ty, K)
    return dq_bytes(_u8(g, off + e * nb, nb, dev), ty, N, K)


def rms(x, eps=EPS_MIX):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)


def l2_clip(x, scale, thr):
    if scale is not None:
        x = x * scale
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True) + 1e-6
    return x * (thr / norm).clamp(0.0, 1.0)


def situ(gate, up, beta, lb):
    act = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if lb > 0:
        up = lb * torch.tanh(up / lb)
    return act * up


class _Slot:
    """KV + conv state of one application: K / V [n_kv, N, hd] (the SWA slots keep the last `window` rows; `start` = absolute position of row 0)."""
    __slots__ = ("k", "v", "start", "hist")

    def __init__(self):
        self.k = self.v = None
        self.start = 0
        self.hist = None      # [n_conv - 1, n_channels]: last inputs of the 4 convs (k | v | attn out | ffn out)


class Inkling2Forward:
    def __init__(self, g, device="cpu", hook=None, chunk=256, cache_weights=True, wdtype=torch.float32):
        m = g.meta
        a = m.get("general.architecture", "inkling2")
        assert a == "inkling2", f"architecture {a} is not inkling2"
        p = a + "."
        self.g, self.dev, self.hook, self.chunk, self.cache_weights = g, torch.device(device), hook, chunk, cache_weights
        self.n_app = int(m[p + "block_count"])
        self.n_loops = int(m.get(p + "num_loops", 1))
        self.core_start = int(m.get(p + "loop.core_start", 0))
        self.core_end = int(m.get(p + "loop.core_end", 0))
        self.n_core = self.core_end - self.core_start if self.n_loops > 1 else 0
        self.n_extra = (self.n_loops - 1) * self.n_core
        self.n_phys = self.n_app - self.n_extra
        self.D = int(m[p + "embedding_length"])
        self.eps = float(m[p + "attention.layer_norm_rms_epsilon"])
        self.n_head = self._scalar(m[p + "attention.head_count"])
        self.n_kv = self._scalar(m[p + "attention.head_count_kv"])
        self.hd = int(m[p + "attention.key_length"])
        self.hd_swa = int(m.get(p + "attention.key_length_swa", self.hd))
        assert self.hd == self.hd_swa and int(m.get(p + "attention.value_length", self.hd)) == self.hd
        self.n_exp = int(m[p + "expert_count"])
        self.topk = int(m[p + "expert_used_count"])
        self.w_scale = float(m.get(p + "expert_weights_scale", 1.0))
        self.window = int(m[p + "attention.sliding_window"])
        self.pattern = [bool(x) for x in m[p + "attention.sliding_window_pattern"]]       # True = SWA, per APPLICATION
        assert len(self.pattern) == self.n_app, (len(self.pattern), self.n_app)
        self.n_conv = int(m[p + "shortconv.l_cache"])
        self.blk = int(m[p + "attn_res.block_size"])
        self.qk_clip = float(m[p + "attention.qk_clip"])
        self.rel_clip = float(m[p + "attention.relative_clip"])
        self.n_rel = int(m[p + "attention.relative_length"])
        self.rel_extent = int(m[p + "attention.relative_extent"])
        self.log_floor = float(m[p + "attention.log_scale_floor"])
        self.log_alpha = float(m[p + "attention.log_scale_alpha"])
        self.beta = float(m[p + "activation.situ_beta"])
        self.lbeta = float(m[p + "activation.situ_linear_beta"])
        self.softcap = float(m.get(p + "final_logit_softcapping", 0.0))
        self.vocab_pad = int(m[p + "vocab_size"])
        self.vocab = int(m.get(p + "vocab_size_unpadded", self.vocab_pad))
        self.kvd = self.n_kv * self.hd
        self.n_chan = 2 * self.kvd + 2 * self.D
        for il in range(self.n_app):
            ip = self.loop_layer(il)
            assert self.pattern[il] == self.pattern[ip], f"sliding_window_pattern differs between application {il} and layer {ip}"
        self._w = {}
        self.slots = [_Slot() for _ in range(self.n_app)]
        self.n = 0                     # tokens processed
        self._il = -1
        self._k = None                 # GPU kernels (triton_gap nvfp4_gguf GEMV, fq_gemv_q8_0), resolved on first use; cpu devices never use them
        self.kstats = {"nvfp4_gap": 0, "q8_gemv": 0, "expert_fallback": 0, "expert_cpu": 0}
        self.stats = {"layer_s": 0.0}
        self.tok_embd_name = "token_embd.weight"

    @staticmethod
    def _scalar(v):
        if isinstance(v, (list, tuple)):
            assert len(set(v)) == 1, f"per-layer head counts {sorted(set(v))} not supported"
            return int(v[0])
        return int(v)

    # ---------------------------------------------------------------- loop maps (llama_hparams::loop_layer / loop_visit)
    def loop_layer(self, il):
        if self.n_loops < 2 or il < self.core_start:
            return il
        if il < self.core_start + self.n_loops * self.n_core:
            return self.core_start + (il - self.core_start) % self.n_core
        return il - self.n_extra

    # ---------------------------------------------------------------- weights
    def W(self, name):
        t = self._w.get(name)
        if t is None:
            t = read_dense(self.g, name, self.dev)
            if self.cache_weights:
                self._w[name] = t
        return t

    def has(self, name):
        return name in self.g.infos

    def reset(self):
        self.slots = [_Slot() for _ in range(self.n_app)]
        self.n = 0

    # ---------------------------------------------------------------- pieces
    def mix(self, blocks, cur, query):
        """block depth attention. blocks: list of [T, D]; cur [T, D]; query [Q, D] -> [Q, T, D]"""
        states = rms(torch.stack(list(blocks) + [cur], 0))                       # [S, T, D]
        scores = torch.einsum("qd,std->qts", query, states) / math.sqrt(self.D)
        return torch.einsum("qts,std->qtd", torch.softmax(scores, -1), states)

    def control(self, cur, msg, L, m):
        """(1 - tanh(F h)) cur + tanh(G h) msg, h = D [rms(cur); rms(msg)]; m: 0 attn(q), 1 k, 2 v, 3 mlp"""
        nm = ("attn", "k", "v", "mlp")[m]
        h = torch.cat([rms(cur), rms(msg)], -1) @ self.W(f"{L}.loop_{nm}_down.weight").T
        gain = torch.tanh(h @ self.W(f"{L}.loop_{nm}_gain.weight").T)
        forget = torch.tanh(h @ self.W(f"{L}.loop_{nm}_forget.weight").T)
        return (1.0 - forget) * cur + gain * msg

    def conv(self, x, w, dyn, slot, off):
        """causal depthwise conv with residual. x [T, C]; w [C, k] taps oldest -> current; dyn [T, heads, k] per-token taps (lag 0 = current) or None."""
        T, C = x.shape
        k = self.n_conv
        hist = slot.hist[:, off:off + C]
        ext = torch.cat([hist, x], 0)                                            # [k-1+T, C]
        slot.hist[:, off:off + C] = ext[-(k - 1):]
        out = torch.zeros_like(x)
        for j in range(k):
            out = out + ext[j:j + T] * w[:, j]
        if dyn is not None:
            nh = dyn.shape[1]
            for lag in range(k):
                win = ext[k - 1 - lag:k - 1 - lag + T].reshape(T, nh, C // nh)
                out = out + (win * dyn[:, :, lag:lag + 1]).reshape(T, C)
        return (x + out) / math.sqrt((2 if dyn is not None else 1) * k + 1)

    def _attend(self, q, k_new, v_new, bias_tab, sinks, slot, swa, pos):
        """q [T, nh, hd]; k_new / v_new [T, nkv, hd]; bias_tab [T, nh, extent]; sinks [T, nh]; pos [T] absolute positions"""
        T = q.shape[0]
        kn, vn = k_new.permute(1, 0, 2), v_new.permute(1, 0, 2)                  # [nkv, T, hd]
        if slot.k is None:
            slot.k, slot.v, slot.start = kn.clone(), vn.clone(), int(pos[0])
        else:
            slot.k, slot.v = torch.cat([slot.k, kn], 1), torch.cat([slot.v, vn], 1)
        N = slot.k.shape[1]
        kpos = slot.start + torch.arange(N, device=q.device)
        dist = pos[:, None] - kpos[None, :]                                      # [T, N]
        valid = dist >= 0
        if swa:
            valid = valid & (dist < self.window)
        g = self.n_head // self.n_kv
        kk = slot.k.repeat_interleave(g, 0)                                      # [nh, N, hd]
        vv = slot.v.repeat_interleave(g, 0)
        s = torch.einsum("thd,hnd->htn", q, kk) / math.sqrt(self.hd)
        inb = (dist >= 0) & (dist < self.rel_extent)
        idx = dist.clamp(0, self.rel_extent - 1)                                 # [T, N]
        b = torch.gather(bias_tab, 2, idx[:, None, :].expand(T, self.n_head, N)) # [T, nh, N]
        b = torch.where(inb[:, None, :], b, torch.zeros_like(b)).permute(1, 0, 2)
        s = s + b
        s = s.masked_fill(~valid[None], float("-inf"))
        s = torch.cat([s, sinks.T[:, :, None]], 2)                               # sink logit: unscaled, zero value
        p = torch.softmax(s, -1)[:, :, :-1]
        o = torch.einsum("htn,hnd->thd", p, vv)
        if swa and N > self.window:                                              # later chunks only need the last `window` keys
            slot.k, slot.v = slot.k[:, -self.window:].contiguous(), slot.v[:, -self.window:].contiguous()
            slot.start += N - self.window
        return o

    def _kern(self):
        """{"ext": extension} on a cuda device, {} otherwise (torch dequant + matmul). triton_gap has NO nvfp4_gguf kernel (q4_0 / iq4_nl / mxfp4 / q5_x only)."""
        if self._k is None:
            self._k = {}
            if self.dev.type == "cuda":
                try:
                    from . import hy4_forward as HF
                    ext = HF._ext()
                    if ext is not None and hasattr(ext, "fq_gemv_q8_0"):
                        self._k["ext"] = ext
                except Exception as e:      # noqa: BLE001
                    print("inkling2: extension unavailable:", repr(e)[:100], flush=True)
        return self._k

    def _expert_mm(self, name, e, x):
        """x [R, K] fp32 @ expert e of the [K, N, E] tensor `name` -> [R, N] fp32: NVFP4 through ext.fq_gemv_nvfp4 (the stored bytes repacked to the ModelOpt
        layout on the device, fp32 activations), Q8_0 through ext.fq_gemv_q8_0, anything else (and prefill widths > 16 rows) torch dequant + matmul"""
        ty, dims, off = self.g.infos[name]
        K, N = dims[0], dims[1]
        R = x.shape[0]
        k = self._kern()
        if self.dev.type != "cuda":
            self.kstats["expert_cpu"] += 1
        elif ty == T_NVFP4 and "ext" in k and hasattr(k["ext"], "fq_gemv_nvfp4") and R <= 16 and K % 64 == 0:
            raw = _u8(self.g, off + e * N * row_bytes(ty, K), N * row_bytes(ty, K), self.dev).reshape(N, -1)
            packed, scales = nvfp4_gguf_to_modelopt(raw)
            out = torch.empty(R, N, dtype=torch.float32, device=self.dev)
            k["ext"].fq_gemv_nvfp4(x.float().contiguous(), packed, scales, 1.0, out, True)
            self.kstats["nvfp4_gap"] += 1
            return out
        elif ty == T_Q8_0 and "ext" in k and R <= 16 and K % 32 == 0:
            packed = _u8(self.g, off + e * N * row_bytes(ty, K), N * row_bytes(ty, K), self.dev).reshape(N, K // 32, 34)
            out = torch.empty(R, N, dtype=torch.float32, device=self.dev)
            k["ext"].fq_gemv_q8_0(x.float().contiguous(), packed, out)
            self.kstats["q8_gemv"] += 1
            return out
        else:
            self.kstats["expert_fallback"] += 1
        return x @ read_expert(self.g, name, e, self.dev).T

    def _moe(self, L, x):
        R = self.W(f"{L}.ffn_gate_inp.weight")                                   # [E, D] fp32
        scores = torch.sigmoid(x @ R.T)
        bias = self.W(f"{L}.exp_probs_b.bias") if self.has(f"{L}.exp_probs_b.bias") else 0.0
        sel = torch.topk(scores + bias, self.topk, dim=-1, sorted=False).indices
        w = scores.gather(1, sel)
        w = w / (w.sum(-1, keepdim=True) + 1e-20) * self.w_scale
        gu_n, dn_n = f"{L}.ffn_gate_up_exps.weight", f"{L}.ffn_down_exps.weight"
        s_gu = self.W(gu_n[:-6] + "scale") if self.has(gu_n[:-6] + "scale") else None
        s_dn = self.W(dn_n[:-6] + "scale") if self.has(dn_n[:-6] + "scale") else None
        self._hk("router_sel", self._il, sel)
        out = torch.zeros_like(x)
        ff = self.g.infos[dn_n][1][0]
        for e in torch.unique(sel).tolist():
            tok, slot = torch.where(sel == e)
            h = self._expert_mm(gu_n, e, x[tok])
            if s_gu is not None:
                h = h * s_gu[e]
            y = self._expert_mm(dn_n, e, situ(h[:, :ff], h[:, ff:], self.beta, self.lbeta))
            if s_dn is not None:
                y = y * s_dn[e]
            out.index_add_(0, tok, y * w[tok, slot][:, None])
        return out

    def _hk(self, name, il, t):
        if self.hook is not None:
            self.hook(name, il, t)

    # ---------------------------------------------------------------- forward over one chunk
    def _chunk(self, ids, start):
        T = len(ids)
        D, nh, nkv, hd = self.D, self.n_head, self.n_kv, self.hd
        pos = torch.arange(start, start + T, device=self.dev)
        cur = read_rows(self.g, self.tok_embd_name, ids, self.dev)
        cur = rms(cur, self.eps) * self.W("token_embd_norm.weight")
        tau = (1.0 + self.log_alpha * torch.log(((pos + 1).float() / self.log_floor).clamp(min=1.0)))[:, None, None]
        blocks, prelude_blocks, prelude_cur, bank, bank_last = [], None, None, [], None
        for il in range(self.n_app):
            ip = self.loop_layer(il)
            L = f"blk.{ip}"
            swa = self.pattern[il]
            msgs = [None] * 4
            if self.n_core and self.core_start <= ip < self.core_end:
                pss = (il - self.core_start) // self.n_core
                if ip == self.core_start:
                    if pss == 0:
                        prelude_blocks, prelude_cur = list(blocks), cur
                    else:
                        bank, bank_last = blocks[len(prelude_blocks) + 1:], cur
                        blocks, cur = list(prelude_blocks), prelude_cur
                if pss > 0:
                    row = 2 * (ip - self.core_start)
                    qa = self.W("loop_res_query.weight")[row:row + 2]
                    qk = self.W("loop_kv_res_query.weight")[row:row + 2]
                    ma, mk = self.mix(bank, bank_last, qa), self.mix(bank, bank_last, qk)
                    msgs = [ma[0], mk[0], mk[1], ma[1]]                          # q(attn), k, v, mlp
            self._il = il
            slot = self.slots[il]
            if slot.hist is None:
                slot.hist = torch.zeros(self.n_conv - 1, self.n_chan, device=self.dev)
            self._hk("layer_inp", il, cur)
            inp_sa = cur
            inputs = [cur, cur, cur]
            if self.has(f"{L}.attn_res_query.weight"):
                q = self.W(f"{L}.attn_res_query.weight").reshape(-1, D)
                mixed = self.mix(blocks, cur, q)
                inputs = [mixed[0 if swa else j] for j in range(3)]
            if msgs[0] is not None:
                inputs = [self.control(inputs[j], msgs[j], L, j) for j in range(3)]
            if swa:
                nw = self.W(f"{L}.attn_norm.weight")
                if msgs[0] is not None:
                    inputs = [rms(x, self.eps) * nw for x in inputs]
                else:
                    inputs = [rms(inputs[0], self.eps) * nw] * 3
            else:
                inputs = [rms(inputs[j], self.eps) * self.W(f"{L}.attn_{c}_inp_norm.weight") for j, c in enumerate("qkv")]
            xq, xk, xv = inputs
            Q = xq @ self.W(f"{L}.attn_q.weight").T
            Kp = xk @ self.W(f"{L}.attn_k.weight").T
            Vp = xv @ self.W(f"{L}.attn_v.weight").T
            R = xq @ self.W(f"{L}.attn_rel.weight").T
            dk = dv = None
            if not swa:
                dk = (xk @ self.W(f"{L}.attn_k_conv_proj.weight").T).reshape(T, nkv, self.n_conv)
                dv = (xv @ self.W(f"{L}.attn_v_conv_proj.weight").T).reshape(T, nkv, self.n_conv)
            Kc = self.conv(Kp, self.W(f"{L}.attn_k_conv.weight"), dk, slot, 0)
            Vc = self.conv(Vp, self.W(f"{L}.attn_v_conv.weight"), dv, slot, self.kvd)
            Q = l2_clip(Q.reshape(T, nh, hd), self.W(f"{L}.attn_q_clip.weight"), self.qk_clip)
            Kc = l2_clip(Kc.reshape(T, nkv, hd), None, self.qk_clip)
            Vc = rms(Vc.reshape(T, nkv, hd), self.eps)
            R = l2_clip(R.reshape(T, nh, self.n_rel), self.W(f"{L}.attn_rel_clip.weight"), self.rel_clip)
            if not swa:
                Q, R = Q * tau, R * tau
            tab = R @ self.W(f"{L}.attn_rel_bias.weight").T                      # [T, nh, extent]
            sinks = xq @ self.W(f"{L}.attn_sink_proj.weight").T                  # [T, nh]
            o = self._attend(Q, Kc, Vc, tab, sinks, slot, swa, pos).reshape(T, nh * hd)
            o = o * torch.sigmoid(xq @ self.W(f"{L}.attn_gate.weight").T)
            o = o @ self.W(f"{L}.attn_output.weight").T
            cur = self.conv(o, self.W(f"{L}.attn_post_conv.weight"), None, slot, 2 * self.kvd)
            self._hk("attn_out", il, cur)
            if ip % self.blk == 0:
                blocks.append(inp_sa)
            else:
                cur = inp_sa + cur
            ffn_inp = cur
            cur = self.mix(blocks, cur, self.W(f"{L}.ffn_res_query.weight").reshape(1, D))[0]
            if msgs[3] is not None:
                cur = self.control(cur, msgs[3], L, 3)
            cur = rms(cur, self.eps) * self.W(f"{L}.ffn_norm.weight")
            cur = self._moe(L, cur)
            cur = self.conv(cur, self.W(f"{L}.ffn_post_conv.weight"), None, slot, 2 * self.kvd + D)
            self._hk("ffn_out", il, cur)
            cur = ffn_inp + cur
            self._hk("l_out", il, cur)
        cur = self.mix(blocks, cur, self.W("output_res_query.weight").reshape(1, D))[0]
        cur = rms(cur, self.eps) * self.W("output_norm.weight")
        self._hk("result_norm", -1, cur)
        return cur

    def head(self, h):
        """h [T, D] -> logits [T, vocab] (softcapped, unpadded)"""
        W = self._w.get("output.weight")
        if W is None:
            W = read_dense(self.g, "output.weight", self.dev)
            if self.cache_weights:
                self._w["output.weight"] = W
        logits = h @ W[:self.vocab].T
        if self.softcap:
            logits = self.softcap * torch.tanh(logits / self.softcap)
        return logits

    @torch.no_grad()
    def step(self, ids, all_pos=False):
        """append `ids` (list[int]) to the context; logits of the last position [vocab] (or of every new position [T, vocab])"""
        t0 = time.time()
        ids = [int(i) for i in ids]
        outs = []
        for s in range(0, len(ids), self.chunk):
            outs.append(self._chunk(ids[s:s + self.chunk], self.n + s))
        h = torch.cat(outs, 0)
        self.n += len(ids)
        logits = self.head(h if all_pos else h[-1:])
        self._hk("result_output", -1, logits)
        self.stats["layer_s"] += time.time() - t0
        return logits if all_pos else logits[0]

    @torch.no_grad()
    def generate(self, ids, n_new):
        out = []
        lg = self.step(ids)
        for _ in range(n_new):
            t = int(torch.argmax(lg))
            out.append(t)
            lg = self.step([t])
        return out
