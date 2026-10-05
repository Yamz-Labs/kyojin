"""weight prefetch into the infinity cache (MALL, 32 MB, about 700 GB/s on re-reads against 236 GB/s from DRAM) during the
latency-bound kernels of a decode token. Read-only side-stream reads of weights that a later streaming kernel (qkv|z gemv, o gemv, fused
MoE front end) will read; no output of the model is touched, so results are bit-identical by construction. The side stream never runs while
the persistent fused MoE kernel (grid barrier, all blocks must stay resident) is on the device: its launch waits for the side stream's last event.
Switch: EXL3_QWD3_PF (0 off, default 0 until the served A/B); harness arms flip PF["on"] at run time.
Parts: w1 = MB of the qkv trellis read after the previous MoE half (dots / finalize window); o = o_proj trellis; moe = HC mixer codes, router,
shared expert of the same block, both read after the qkv|z launch (skinny / conv / recurrence window)."""
import os, torch

PF = {"on": int(os.environ.get("EXL3_QWD3_PF", "0") != "0"), "w1": 8, "o": 1, "moe": 1, "fired1": 0, "fired2": 0, "waits": 0}
_S = {"side": None, "ev0": None, "ev1": None, "evs": None, "pending": False, "sink": None, "cur": None, "inst": False}
MB = 1 << 20

def _side():
    if _S["side"] is None:
        _S["side"] = torch.cuda.Stream(); _S["ev0"] = torch.cuda.Event(); _S["ev1"] = torch.cuda.Event(); _S["evs"] = torch.cuda.Event()
        _S["sink"] = torch.zeros((), dtype = torch.float32, device = "cuda")
    return _S["side"]

def _f32(t, nbytes = None):
    """Contiguous view of t's first nbytes as fp32 (4-byte aligned prefix), no copy."""
    v = t.reshape(-1).view(torch.uint8) if t.dtype != torch.uint8 else t.reshape(-1)
    n = v.numel() if nbytes is None else min(nbytes, v.numel())
    n -= n % 4
    return v[:n].view(torch.float32) if n else None

def _touch(t, nbytes = None):
    if PF.get("noop"): return
    v = _f32(t, nbytes)
    if v is not None:
        torch.sum(v, dim = 0, out = _S["sink"])

def _is_decode(x, params):
    return x.dim() >= 2 and x.numel() // x.shape[-1] // (x.shape[2] if x.dim() == 4 else 1) == 1 and not params.get("prefill")

def _gdn_tensors(attn):
    ts = getattr(attn, "_qd3_t", None)
    if ts is None:
        ts = (attn.qkv_proj.inner.trellis, attn.z_proj.inner.trellis, attn.o_proj.inner.trellis)
        attn._qd3_t = ts
    return ts

def _moe_tensors(fh):
    ts = getattr(fh, "_qd3_t", None)
    if ts is None:
        fq, fsc, uq, usc = fh.gr.q8_set()
        ts = [fq, uq, fh.router, fh.sgate]
        ts += [t for t in (fh.svt,) if t is not None]
        mlp = fh.mlp
        sh = mlp.shared_experts
        ts += [sh.gates[0].inner.trellis, sh.ups[0].inner.trellis, sh.downs[0].inner.trellis]
        fh._qd3_t = ts
    return ts

def install():
    if _S["inst"] or not torch.cuda.is_available():
        return
    _S["inst"] = True
    from . import transformer, moe_fused, gated_delta_net
    TB = transformer.TransformerBlock
    orig_fwd = TB.forward
    def block_forward(self, x, params, *a, **kw):
        if PF["on"] and isinstance(getattr(self, "attn", None), gated_delta_net.GatedDeltaNet) and _is_decode(x, params):
            _S["cur"] = self
            side = _side(); main = torch.cuda.current_stream()
            _S["ev0"].record(main); side.wait_event(_S["ev0"])
            with torch.cuda.stream(side):
                if PF["w1"]: _touch(_gdn_tensors(self.attn)[0], int(PF["w1"] * MB))
                _S["evs"].record(side)
            _S["pending"] = True; PF["fired1"] += 1
        else:
            _S["cur"] = None
        return orig_fwd(self, x, params, *a, **kw)
    TB.forward = block_forward
    G = gated_delta_net.GatedDeltaNet
    orig_p = G.project_qkvz_dec
    def project_qkvz_dec(self, x, params, bsz, seqlen):
        r = orig_p(self, x, params, bsz, seqlen)
        blk = _S["cur"]
        if PF["on"] and r is not None and blk is not None and blk.attn is self:
            side = _side(); main = torch.cuda.current_stream()
            _S["ev1"].record(main); side.wait_event(_S["ev1"])
            with torch.cuda.stream(side):
                if PF["o"]: _touch(_gdn_tensors(self)[2])
                fh = getattr(blk, "_moe_fused", None)
                if PF["moe"] and fh is not None and fh.supported:
                    for t in _moe_tensors(fh): _touch(t)
                _S["evs"].record(side)
            _S["pending"] = True; PF["fired2"] += 1
        return r
    G.project_qkvz_dec = project_qkvz_dec
    F = moe_fused.FusedMoEHalf
    orig_call = F.__call__
    def call(self, x):
        if _S["pending"]:
            torch.cuda.current_stream().wait_event(_S["evs"]); _S["pending"] = False; PF["waits"] += 1
        return orig_call(self, x)
    F.__call__ = call
