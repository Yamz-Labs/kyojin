"""Host-sync-free decode step driver (flag HY4_NOSYNC=1, default off). Part F of the Hy4 Strix forward.

Owns only: the step driver, token history / position / K-cache bookkeeping, embedding lookup, logits -> next id, and the
host read of the expert ids. Generic (stored Q8_0 embedding table, any [rows, R+rope] K rows); nothing here is keyed on a model name.
Every function is only reached when active(fw) is true, so the flag-off path of hy4_forward.py is the old code, untouched.

What it removes from the decode loop (per token):
  * 41x  sel.flatten().tolist()  (router ids to host) when the layer's experts are resident (the result was thrown away)
  * embed: per-token pread + CPU dequant + pageable H2D  ->  resident Q8_0 table on the device, gather + dequant on the device
  * torch.arange(start, ..) per step + torch.arange(N) per layer  ->  slices of one preallocated device arange
  * torch.cat([kc, kn]) per layer per step (alloc + O(N) copy)  ->  preallocated [cap, R+rope] buffer, append in place, view [:N]
  * out[0].cpu() (129k floats D2H + queue drain) and int(lg.argmax()) per token  ->  device argmax, the id stays on the device,
    one host read of all ids at the end of generate()
Syncs that remain here until other parts land: fw.tokens (host list) is fed by one .tolist() per token while the Engram part
(A) still hashes on the host (fw.engram_device_tokens is False); A sets fw.engram_device_tokens = True and reads fw._ns.tok (device history).
"""
import os
import time

import torch


def enabled():
    return os.environ.get("HY4_NOSYNC", "0") == "1"


def active(fw):
    return enabled() and fw.dev.type == "cuda"


class State:
    CAP = int(os.environ.get("HY4_NOSYNC_CAP", "4096"))      # rows of the K buffers / token history; grows (host int check) when exceeded

    def __init__(self, dev):
        self.dev, self.cap = dev, State.CAP
        self.ar = torch.arange(self.cap, device=dev)          # int64, same values / dtype as torch.arange(N, device=dev)
        self.tok = torch.zeros(self.cap, dtype=torch.long, device=dev)      # device token history (for a device-side Engram hash)
        self.kv = {}                                          # layer -> [cap, W] buffer
        self.table = None                                     # resident Q8_0 embedding bytes [V, K//32*34] uint8

    def ensure(self, need):
        if need > self.cap:
            cap = max(need, 2 * self.cap)
            ar = torch.arange(cap, device=self.dev)
            tok = torch.zeros(cap, dtype=torch.long, device=self.dev)
            tok[:self.cap].copy_(self.tok)
            self.ar, self.tok, self.cap = ar, tok, cap


def state(fw):
    st = getattr(fw, "_ns", None)
    if st is None:
        st = fw._ns = State(fw.dev)
    return st


# ---- embedding -------------------------------------------------------------------------------------------------------
def load_table(g, name, dev):
    """whole Q8_0 [V, K] tensor as raw bytes [V, K//32*34] uint8 on dev (stored bytes, ~V*K*1.0625)"""
    import numpy as np
    ty, dims, off = g.infos[name]
    assert ty == 8 and dims[0] % 32 == 0, (name, ty, dims)
    K, V = dims[0], dims[1]
    rb = K // 32 * 34
    raw = torch.from_numpy(np.ascontiguousarray(g._pread(off, V * rb))).reshape(V, rb)
    return raw.to(dev)


def gather_dequant_q8_0(table, ids):
    """table [V, nb*34] uint8, ids [T] int -> fp32 [T, nb*32]; same arithmetic as engram_rows._dequant_q8_0 (fp16 d -> fp32, int8 q -> fp32, one multiply)"""
    T = ids.shape[0]
    rows = table[ids].reshape(T, -1, 34)
    d = rows[..., :2].contiguous().view(torch.float16).float()           # [T, nb, 1]
    q = rows[..., 2:].contiguous().view(torch.int8).float()              # [T, nb, 32]
    return (q * d).reshape(T, -1)


def embed(fw, ids_dev):
    st = state(fw)
    if st.table is None:
        st.table = load_table(fw.g, "token_embd.weight", fw.dev)
    return gather_dequant_q8_0(st.table, ids_dev)


# ---- K cache / positions -----------------------------------------------------------------------------------------------
def append_kv(fw, i, kn, start):
    """K rows of layer i: write kn [T, W] at rows start..start+T-1 of a preallocated buffer, return the view [:start+T] (== torch.cat([old, kn]))"""
    st = state(fw)
    T = kn.shape[0]
    need = start + T
    b = st.kv.get(i)
    if b is None or b.shape[0] < need:
        nb = torch.zeros(max(st.cap, need, 2 * (b.shape[0] if b is not None else 0)), kn.shape[1], dtype=kn.dtype, device=kn.device)
        if b is not None:
            nb[:start].copy_(b[:start])
        elif fw.kc[i] is not None and start > 0:           # cache built by the old path before the flag was switched on
            nb[:start].copy_(fw.kc[i])
        b = st.kv[i] = nb
    b[start:need].copy_(kn)
    return b[:need]


def kpos(fw, N):
    st = state(fw)
    st.ensure(N)
    return st.ar[:N]


# ---- router ids --------------------------------------------------------------------------------------------------------
def skip_host_ids(fw, i, T, maxr):
    """True when the host list of picked experts is never used: resident grouped kernel takes the device ids (hy4_forward._ffn: rs is not None)"""
    return active(fw) and fw.res.get(i) is not None and T <= maxr


# ---- step / generate ---------------------------------------------------------------------------------------------------
def argmax_id(lg):
    return torch.argmax(lg.reshape(-1))        # 0-d device tensor, stays on the device


def step(fw, ids, all_pos=False):
    """Hy4Forward.step with ids = list[int] (prefill, or host-fed decode) or a device int tensor [T] (the id from argmax_id). Returns DEVICE logits."""
    st = state(fw)
    host = not torch.is_tensor(ids)
    if fw.tm.on:
        torch.cuda.synchronize()
        base = sum(fw.tm.acc.values())
    t0 = time.time()
    start = fw.n
    if host:
        fw.tokens += [int(t) for t in ids]
        T = len(ids)
        idt = torch.tensor(list(ids), dtype=torch.long, device=fw.dev)
    else:
        idt = ids.to(fw.dev, torch.long).reshape(-1)
        T = idt.shape[0]
        if not getattr(fw, "engram_device_tokens", False):
            fw.tokens += idt.tolist()                       # the one remaining per-token sync until Engram (A) hashes on the device
    st.ensure(start + T)
    st.tok[start:start + T].copy_(idt)
    pos = st.ar[start:start + T]
    e = embed(fw, idt)
    x = e[:, None, :].expand(-1, fw.hc, -1).contiguous()
    for i in range(fw.n_layers):
        x = fw.layer(i, x, pos, fw.tokens, start)
    fw.n += T
    fw._x = x
    fw._ns_dev = True                                       # _head: keep the logits on the device
    try:
        lg = fw.head(x, all_pos) if fw.n_layers == fw.n_trunk else None
    finally:
        fw._ns_dev = False
    if fw.tm.on:
        torch.cuda.synchronize()
        tot = time.time() - t0
        fw.tm.acc["other"] = fw.tm.acc.get("other", 0.0) + max(0.0, tot - (sum(fw.tm.acc.values()) - base))
        fw.tm.last_total = tot
    fw.stats["layer_s"] += time.time() - t0
    return lg


def generate(step_fn, ids, n_new):
    """greedy loop: step_fn(ids_or_device_tensor) -> logits. The next id never leaves the device; one host read of all ids at the end."""
    t0 = time.time()
    lg = step_fn(ids)
    t1 = time.time()
    first, out = lg, []
    nxt = argmax_id(lg)
    for k in range(n_new):
        out.append(nxt)
        if k + 1 == n_new:
            break
        lg = step_fn(nxt.reshape(1))
        nxt = argmax_id(lg)
    ids_out = torch.stack(out).tolist() if out else []
    t2 = time.time()
    return first, ids_out, {"prefill_s": t1 - t0, "decode_s": t2 - t1, "decode_steps": max(len(ids_out) - 1, 0)}
