"""Shared MLA variant maths: per-head attention sinks + sliding window + sigmoid output gate (Hy4-flash attention,
llama.cpp hy-v4-flash.cpp build_attention M:5787-5947). Pure torch, no package imports, fp32, absorbed form.

  scores[h, i, j] = scale * (q_lat[h, i] . ckv[j] + q_pe[h, i] . k_pe[j])     scale = 1/sqrt(nope + rope)  (unabsorbed head dim)
  visible j: kpos[j] <= qpos[i]  and (window == 0 or qpos[i] - kpos[j] < window)         (llama.cpp LLAMA_SWA_TYPE_STANDARD)
  sink s[h] joins the softmax denominator only (no value):  p = exp(scores - m) / (sum exp(scores - m) + exp(s - m)),  m = max(max scores, s)
  o_lat[h, i] = p @ ckv ;  the caller folds W_UV per head afterwards, multiplies by sigmoid(gate_proj(x)) and then applies o_proj.
"""
import torch


def mla_sink_attend(q_lat, q_pe, ckv, k_pe, qpos, kpos, scale, sinks=None, window=0):
    """q_lat [T,H,R], q_pe [T,H,P], ckv [N,R], k_pe [N,P], qpos [T], kpos [N], sinks [H] or None -> o_lat [T,H,R]"""
    s = torch.einsum("thr,nr->htn", q_lat, ckv) + torch.einsum("thp,np->htn", q_pe, k_pe)
    s = s * scale
    vis = kpos[None, :] <= qpos[:, None]
    if window:
        vis = vis & ((qpos[:, None] - kpos[None, :]) < window)
    s = s.masked_fill(~vis[None], float("-inf"))
    if sinks is not None:
        s = torch.cat([s, sinks.float()[:, None, None].expand(-1, s.shape[1], 1)], dim=-1)
    p = torch.softmax(s, dim=-1)
    if sinks is not None:
        p = p[..., :-1]
    return torch.einsum("htn,nr->thr", p, ckv)


def sigmoid_gate(o, gate_logits):
    """the Hy4 output gate: o * sigmoid(gate_logits), before o_proj"""
    return o * torch.sigmoid(gate_logits)

