"""HY4_ROPE_POS=1: rope cos / sin rows computed from the positions of the call, no per-token table (pure torch, no triton).

The 1.9.1 tables are built once to the full context: attn_eager [n, 2, P/2, 2] fp32 per rope theta (512 B per token each, full + window layers) and the indexer
cos / sin [n, P/2] (256 B per token). Rope only ever looks up the positions of the rows in flight (T <= one prefill chunk), so these rows are evaluated from `pos`
instead: fp64 angle = pos * inv_freq (the same expression as the table build, so the fp32 values are the same), then cos / sin -> fp32.

Memo: the rows depend only on (pos values, theta), so every layer of one call shares them. The memo lives on the pos tensor object and is valid for the same
object at the same in-place version (a static graph input is overwritten in place: the version moves). Inside a graph capture only the GraphDecoder.pos view
(marked `_rope_ok`, a fresh object per segment body) is memoised: a replay does not run python, so an older capture's tensor must never be reused. STATS counts evaluations (FIRES)."""
import os

import torch

ENABLED = os.environ.get("HY4_ROPE_POS", "0") == "1"
STATS = {"attn": 0, "ix": 0}
_INV = {}


def _inv(P, base, dev, cpu_build):
    """inv_freq [P/2] fp64 on dev, built like the table it replaces: attn_eager builds on the device, the indexer on the host (then moved)"""
    key = (P, float(base), str(dev), cpu_build)
    v = _INV.get(key)
    if v is None:
        if cpu_build:
            v = (base ** (-torch.arange(0, P, 2, dtype=torch.float64) / P)).to(dev)
        else:
            v = base ** (-2.0 * torch.arange(P // 2, dtype=torch.float64, device=dev) / P)
        _INV[key] = v
    return v


def _memo(pos, key, build):
    if pos.is_cuda and torch.cuda.is_current_stream_capturing() and not getattr(pos, "_rope_ok", False):
        return build()                  # a capture on a position tensor that does not start a fresh object per capture (GraphDecoder.pos does): no memo, evaluate here
    m = pos.__dict__.setdefault("_rope_rows", {})
    hit = m.get(key)
    if hit is not None and hit[0] == pos._version:
        return hit[1]
    val = build()
    m[key] = (pos._version, val)
    return val


def attn_rows(P, base, pos):
    """pos [T] long -> [T, 2, P/2, 2] fp32 = rows (cos, sin | -sin, cos) of the attn_eager table for these positions"""
    def build():
        STATS["attn"] += 1
        th = pos.double()[:, None] * _inv(P, base, pos.device, False)[None, :]
        c, s = th.cos().float(), th.sin().float()
        tab = torch.empty(pos.shape[0], 2, P // 2, 2, dtype=torch.float32, device=pos.device)
        tab[:, 0, :, 0], tab[:, 0, :, 1], tab[:, 1, :, 0], tab[:, 1, :, 1] = c, s, -s, c
        return tab
    return _memo(pos, ("attn", P, float(base)), build)


def ix_rows(base, rope_dim, pos, group=0):
    """pos [T] long -> (cos, sin) [T, rope_dim/2] fp32 contiguous, like indexer_fused._table()[pos]; group > 0: the rows of the group start (pos // group) * group"""
    def build():
        STATS["ix"] += 1
        p = (pos // group) * group if group else pos
        th = p.double()[:, None] * _inv(rope_dim, base, pos.device, True)[None, :]
        return th.cos().float().contiguous(), th.sin().float().contiguous()
    return _memo(pos, ("ix", rope_dim, float(base), group), build)
