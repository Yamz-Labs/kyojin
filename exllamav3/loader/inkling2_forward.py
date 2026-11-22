"""Inkling 2 Mini WHOLE-MODEL forward from a GGUF (llama.cpp #32695 convention).

One step over the 48 applications; the op kernels are picked by the DEVICE at load time: the CUDA kernels (tools/inkling2/ink2_*_cuda.cu,
INK_RTX) on NVIDIA, the HIP kernels + Triton glue (tools/inkling2/ink2_*.hip, ink2_glue.py, INK_GLUE) on AMD (ROCm), the plain tensor /
dequant path on CPU or any other device. A HIP build is never loaded on an NVIDIA device and vice versa (torch reports both as device
type "cuda": the discriminator is torch.version.hip). INK_BACKEND=cuda|hip|torch forces one, INK_AUTO=0 keeps the old behaviour (only the
kernels named by INK_RTX / INK_GLUE). Both kernel sets are opt-out per stage through the same env flags as before.

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
import sys
import time

import numpy as np
import torch

T_F32, T_F16, T_Q8_0, T_BF16, T_NVFP4 = 0, 1, 8, 30, 40
_QBLK = {T_Q8_0: (32, 34), T_NVFP4: (64, 36)}
EPS_MIX = 1e-6


RESIDENT = os.environ.get("INK_RESIDENT", "0") == "1"       # CUDA 24 GB cards: Q8_0 dense on the device as bytes, experts resident (see _w_resident / _xcache); off = the streaming path of the Strix runs
RES_GB = float(os.environ.get("INK_RES_GB", "16.6"))
FUSE = set(x for x in os.environ.get("INK_GLUE_FUSE", "").split(",") if x)   # glue fusions: conv = the G2a Triton short-conv (+ residual / l2 clip / rms epilogue), one launch per conv call site
def _misc():
    """the Triton glue fusions (ink2_glue_misc.py): INK_GLUE_FUSE = mix, control, norm (or any comma list)"""
    import sys
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools", "inkling2")
    if d not in sys.path:
        sys.path.insert(0, d)
    import ink2_glue_misc
    return ink2_glue_misc


RTX = set(x for x in os.environ.get("INK_RTX", "").split(",") if x)   # our CUDA kernels wired into the eager step: experts (ink2_experts_cuda.cu), attn (ink2_attn_cuda.cu); empty = the torch reference path
CFG_NV = tuple(int(v) for v in os.environ.get("INK_X_CFG_NV", "100:100").split(":"))   # (gate_up, down) launch cfgs of the NVFP4 experts kernel (100 = auto: best measured cfg per format and T, tip 8b3473c65)
CFG_Q8 = tuple(int(v) for v in os.environ.get("INK_X_CFG_Q8", "100:100").split(":"))      # Q8_0 layers (4 of 24)

RTX_ALL = {"experts", "attn", "dense", "router"}              # every CUDA kernel of the step (what an NVIDIA device picks by default)
FUSE_CUDA_ALL = ("conv", "control", "mix", "norm")            # every Triton fusion of the NVIDIA step (ink2_glue_conv / ink2_glue_misc); the HIP set is ink2_glue.FUSE_ALL
AUTO = os.environ.get("INK_AUTO", "1") != "0"                 # device-picked kernels: on unless INK_AUTO=0 (then only INK_RTX / INK_GLUE name them)


def device_kind(dev):
    """the kernel family a device can run: "hip" = AMD (ROCm; torch also calls it "cuda"), "cuda" = NVIDIA, "torch" = CPU / anything else"""
    if dev.type != "cuda":
        return "torch"
    return "hip" if torch.version.hip else "cuda"


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


_AFQ = None


def _attn_fq(name, t):
    """whole-model quality arm of the attention bytes lever (env INK2_ATTN_FQ = "FMT[,proj=FMT...]", FMT in q8 | nvfp4 | none, proj in q gate rel k v output):
    the dense attention projection becomes the weight the fused kernel reads (ink2_attnquant.fake_quant of the BF16 tensor). Small tensors (sink_proj, conv_proj) stay as stored."""
    global _AFQ
    if _AFQ is None:
        spec = os.environ.get("INK2_ATTN_FQ", "")
        _AFQ = {}
        for i, tok in enumerate(x for x in spec.split(",") if x):
            if "=" in tok:
                k, v = tok.split("=")
                _AFQ[k] = v
            elif i == 0:
                _AFQ["*"] = tok
    if not _AFQ or t.dim() != 2 or t.shape[1] % 64 or t.numel() < 50_000:
        return t
    for p in ("q", "gate", "rel", "k", "v", "output"):
        if name.endswith(f".attn_{p}.weight"):
            fmt = _AFQ.get(p, _AFQ.get("*", "none"))
            if fmt in ("none", ""):
                return t
            from . import ink2_attnquant as AQ
            return AQ.fake_quant(t.cpu(), fmt).to(t.device)
    return t


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
        self._xc, self._xc_bytes = {}, 0
        self.slots = [_Slot() for _ in range(self.n_app)]
        self.n = 0                     # tokens processed
        self._il = -1
        self._k = None                 # GPU kernels (triton_gap nvfp4_gguf GEMV, fq_gemv_q8_0), resolved on first use; cpu devices never use them
        self.kstats = {"nvfp4_gap": 0, "q8_gemv": 0, "expert_fallback": 0, "expert_fallback_other": 0, "expert_cpu": 0, "xk_calls": 0, "xk_nvfp4": 0, "xk_q8": 0, "torch_moe": 0, "ak_gkv": 0, "ak_lattn": 0, "torch_attn": 0, "dense_q8": 0, "dense_torch": 0, "conv_fused": 0, "conv_torch": 0, "graph_replays": 0, "graph_eager": 0, "router_k": 0, "torch_router": 0,
                       "torch_head": 0, "torch_experts": 0, "torch_attn": 0, "torch_attn_global": 0, "torch_attn_local": 0, "torch_attnproj": 0}
        self._q8p = {}                 # dense Q8_0 matrices resident as the stored bytes [N, K/32, 34] (INK_RTX=dense: ext.fq_gemv_q8_0 reads them once, no fp32 dequant)
        self.graphs = {}               # T -> (CUDAGraph, static input [T, D], (hidden [T, D], logits [T, vocab])): INK_GRAPH=1, capture_graphs()
        self._ar, self._gin = {}, {}
        self._gtap, self.tap_layers = {}, ()     # T -> static taps [T, len(tap_layers) + 1, D] written INSIDE the captured graph (drafter context rows)
        self._cf = None                # ConvFuse (INK_GLUE_FUSE=conv): per-application position-indexed pre-conv rings
        self._pos0 = None              # device int32 [1]: absolute position of the first row of the chunk
        self._ax = None                # Ink2AttnCuda (INK_RTX=attn)
        self._kvc = {}                 # application -> (K, V, cap) device cache of the attention kernels (persistent across reset())
        self._relT = {}                # physical layer -> rel_bias^T [16, 1024] fp32 contiguous
        self._xk = None                # Ink2ExpertsCuda (INK_RTX=experts), loaded on first use
        self._xl = {}                  # physical layer -> (gu bytes [E, N, rb], dn bytes, s_gu, s_dn, is_q8): the whole layer resident on the device
        self.stats = {"layer_s": 0.0}
        self.tok_embd_name = "token_embd.weight"
        self.gl = None                 # HIP glue (tools/inkling2/ink2_glue.py), off unless the device is AMD or INK_GLUE names stages
        self.last_h = None
        self.backend = self._pick_backend()

    # ---------------------------------------------------------------- backend pick (at load time, from the device)
    def _pick_backend(self):
        """hip = AMD (ROCm), cuda = NVIDIA, torch = everything else. A HIP build is never loaded on an NVIDIA device and the CUDA
        kernels never on an AMD one (both are device type "cuda": torch.version.hip tells them apart). INK_BACKEND forces one, and the
        named stages stay the per-stage switches (INK_RTX / INK_GLUE); with INK_AUTO=0 nothing is picked by the device."""
        kind = device_kind(self.dev)
        forced = os.environ.get("INK_BACKEND", "").strip()
        if forced:
            assert forced in ("hip", "cuda", "torch"), f"INK_BACKEND={forced}: expected hip, cuda or torch"
            kind = forced
        self.backend = kind                      # set before the kernels wire themselves (Glue / graph capture read it)
        if kind == "hip":
            spec = os.environ.get("INK_GLUE") or ("all" if AUTO else "")
            if spec:
                self.enable_glue(spec)
        elif kind == "cuda":
            if AUTO and not RTX and not self._cuda_kernels_built():
                print("inkling2: CUDA kernels not built for this tree: torch path", flush=True)
            elif AUTO and not RTX:
                RTX.update(RTX_ALL)
                FUSE.update(FUSE_CUDA_ALL)
            if "clip" in FUSE:
                print("inkling2: INK_GLUE_FUSE=clip is the HIP launch: dropped on this NVIDIA device", flush=True)
                FUSE.discard("clip")
            if RTX and not self._cuda_kernels_built():
                print("inkling2: INK_RTX set but the CUDA kernels are not built: torch path", flush=True)
                RTX.clear()
        return kind

    def _cuda_kernels_built(self):
        """the three .so files the CUDA kernels of the step load at first use"""
        import importlib
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools", "inkling2")
        if d not in sys.path:
            sys.path.insert(0, d)
        try:
            for mod, cls, so in (("ink2_experts_cuda", "Ink2ExpertsCuda", "INK2_SO"), ("ink2_attn_glue", "Ink2AttnCuda", "INK2_ATTN_SO")):
                c = getattr(importlib.import_module(mod), cls)
                so_file = os.environ.get(so) or os.path.join(d, f"libink2_{'experts' if mod.endswith('experts_cuda') else 'attn'}_cuda.so")
                if not os.path.exists(so_file):
                    return False
                c(so=so_file)
            return True
        except Exception as e:      # noqa: BLE001
            print("inkling2: CUDA kernel probe failed:", repr(e)[:200], flush=True)
            return False

    def enable_glue(self, spec, **kw):
        """wire the HIP op kernels into the eager step: spec = comma list of router, head, experts, gkv, lattn, attn, or `all`.
        Kernels take T <= 8 rows, so the chunk shrinks to 8. Never on an NVIDIA device: those kernels are HIP builds."""
        if device_kind(self.dev) != "hip":
            raise RuntimeError(f"the glue kernels are HIP builds (ink2_glue.py): not loadable on device {self.dev}")
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools", "inkling2")
        if d not in sys.path:
            sys.path.insert(0, d)
        import ink2_glue
        self.gl = ink2_glue.Glue(self, spec, **kw)
        self.chunk = min(self.chunk, 8)
        if os.environ.get("INK_GLUE_GRAPH") == "1":
            self.capture_graphs()
        return self.gl

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
        if RESIDENT and self.backend == "cuda":
            return self._w_resident(name)
        t = self._w.get(name)
        if t is None:
            t = read_dense(self.g, name, self.dev)
            t = _attn_fq(name, t)
            if self.cache_weights:
                self._w[name] = t
        return t

    def _w_resident(self, name):
        """INK_RESIDENT=1 (24 GB cards): 2-D Q8_0 matrices stay on the device as the stored bytes (1.06 B / weight) and dequantise to fp32 per use
        (exact: same d * q values as the fp32 cache, which would need 4 B / weight); everything else is cached as fp32."""
        ty, dims, off = self.g.infos[name]
        if ty == T_Q8_0 and len(dims) == 2:
            raw = self._w.get(name)
            if raw is None:
                raw = _u8(self.g, off, dims[1] * row_bytes(ty, dims[0]), self.dev)
                self._w[name] = raw
            return dq_bytes(raw, ty, dims[1], dims[0])
        t = self._w.get(name)
        if t is None:
            t = self._w[name] = read_dense(self.g, name, self.dev)
        return t

    def has(self, name):
        return name in self.g.infos

    def fuse_conv(self, T):
        """the CUDA fused conv (Triton short-conv + residual / clip / rms epilogue): NVIDIA only, the HIP side has gl.conv_kv"""
        return "conv" in FUSE and self.backend == "cuda" and T <= 8

    def _glue_begin(self, T, start):
        """device pos0 (shared by the attention kernels and the fused conv; set OUTSIDE any captured graph) + lazy ConvFuse"""
        if self.gl is not None:
            self.gl.begin(start, T)                                              # HIP glue: its own device pos0 + KV / ring state
            return
        if self.backend != "cuda" or not ("conv" in FUSE or "attn" in RTX):
            return
        if T > 8:
            raise RuntimeError("INK_GLUE_FUSE=conv / INK_RTX=attn: T <= 8 rows per chunk only")
        if self._pos0 is None:
            self._pos0 = torch.zeros(1, dtype=torch.int32, device=self.dev)
        if "conv" in FUSE and self._cf is None:
            d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools", "inkling2")
            if d not in sys.path:
                sys.path.insert(0, d)
            import ink2_glue_conv as G
            self._cf = G.ConvFuse(self.n_app, self.n_chan, self.dev, self._pos0, self.qk_clip, self.eps)
        self._pos0.fill_(int(start))

    def reset(self):
        """new context. The conv history tensors are zeroed IN PLACE (captured graphs hold their addresses)"""
        if self._cf is not None:
            self._cf.reset()
        old = self.slots
        self.slots = [_Slot() for _ in range(self.n_app)]
        for s, o in zip(self.slots, old):
            if o.hist is not None:
                s.hist = o.hist.zero_()
        self.n = 0
        if self.gl is not None:
            self.gl.reset()

    # ---------------------------------------------------------------- pieces
    def mix(self, blocks, cur, query):
        """block depth attention. blocks: list of [T, D]; cur [T, D]; query [Q, D] -> [Q, T, D]"""
        if "mix" in FUSE and self.backend == "cuda" and cur.shape[0] <= 8 and query.shape[0] <= 3:
            return _misc().mix(torch.stack(list(blocks) + [cur], 0).contiguous(), query.contiguous(), EPS_MIX)       # rms + scores + softmax + weighted sum: 2 launches
        states = rms(torch.stack(list(blocks) + [cur], 0))                       # [S, T, D]
        scores = torch.einsum("qd,std->qts", query, states) / math.sqrt(self.D)
        return torch.einsum("qts,std->qtd", torch.softmax(scores, -1), states)

    def control(self, cur, msg, L, m):
        """(1 - tanh(F h)) cur + tanh(G h) msg, h = D [rms(cur); rms(msg)]; m: 0 attn(q), 1 k, 2 v, 3 mlp"""
        nm = ("attn", "k", "v", "mlp")[m]
        if "control" in FUSE and self.backend == "cuda" and cur.shape[0] <= 8:
            return _misc().control(cur, msg, self.W(f"{L}.loop_{nm}_down.weight").contiguous(), self.W(f"{L}.loop_{nm}_gain.weight").contiguous(),
                                   self.W(f"{L}.loop_{nm}_forget.weight").contiguous(), EPS_MIX)
        h = torch.cat([rms(cur), rms(msg)], -1) @ self.W(f"{L}.loop_{nm}_down.weight").T
        gain = torch.tanh(h @ self.W(f"{L}.loop_{nm}_gain.weight").T)
        forget = torch.tanh(h @ self.W(f"{L}.loop_{nm}_forget.weight").T)
        return (1.0 - forget) * cur + gain * msg

    def conv(self, x, w, dyn, slot, off):
        """causal depthwise conv with residual. x [T, C]; w [C, k] taps oldest -> current; dyn [T, heads, k] per-token taps (lag 0 = current) or None."""
        T, C = x.shape
        k = self.n_conv
        if hasattr(self, "kstats"):
            self.kstats["conv_torch"] += 1
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
        QB = int(os.environ.get("INK_ATTN_QB", "256"))                           # query blocks: the [nh, QB, N] score / bias temporaries stay small at T = 1-2K prefill chunks (bit-identical per row)
        outs = []
        for t0 in range(0, T, QB):
            sl = slice(t0, min(T, t0 + QB))
            outs.append(self._attend_core(q[sl], slot.k, slot.v, kpos, bias_tab[sl], sinks[sl], swa, pos[sl]))
        o = outs[0] if len(outs) == 1 else torch.cat(outs, 0)
        if swa and N > self.window:                                              # later chunks only need the last `window` keys
            slot.k, slot.v = slot.k[:, -self.window:].contiguous(), slot.v[:, -self.window:].contiguous()
            slot.start += N - self.window
        return o

    def _attend_core(self, q, kk_, vv_, kpos, bias_tab, sinks, swa, pos):
        """the attention maths over explicit keys: kk_ / vv_ [nkv, N, hd] at absolute positions kpos [N]; q [T, nh, hd]; pos [T]"""
        T, N = q.shape[0], kk_.shape[1]
        dist = pos[:, None] - kpos[None, :]                                      # [T, N]
        valid = dist >= 0
        if swa:
            valid = valid & (dist < self.window)
        g = self.n_head // self.n_kv
        kk = kk_.repeat_interleave(g, 0)                                         # [nh, N, hd]
        vv = vv_.repeat_interleave(g, 0)
        s = torch.einsum("thd,hnd->htn", q, kk) / math.sqrt(self.hd)
        inb = (dist >= 0) & (dist < self.rel_extent)
        idx = dist.clamp(0, self.rel_extent - 1)                                 # [T, N]
        b = torch.gather(bias_tab, 2, idx[:, None, :].expand(T, self.n_head, N)) # [T, nh, N]
        b = torch.where(inb[:, None, :], b, torch.zeros_like(b)).permute(1, 0, 2)
        s = s + b
        s = s.masked_fill(~valid[None], float("-inf"))
        s = torch.cat([s, sinks.T[:, :, None]], 2)                               # sink logit: unscaled, zero value
        p_ = torch.softmax(s, -1)[:, :, :-1]
        return torch.einsum("htn,hnd->thd", p_, vv)

    def _kern(self):
        """{"ext": extension} on an NVIDIA device, {} otherwise (torch dequant + matmul). triton_gap has NO nvfp4_gguf kernel (q4_0 / iq4_nl / mxfp4 / q5_x only)."""
        if self._k is None:
            self._k = {}
            if self.backend == "cuda":
                try:
                    from . import hy4_forward as HF
                    ext = HF._ext()
                    if ext is not None and hasattr(ext, "fq_gemv_q8_0"):
                        self._k["ext"] = ext
                except Exception as e:      # noqa: BLE001
                    print("inkling2: extension unavailable:", repr(e)[:100], flush=True)
        return self._k

    def _xcache(self, key, val, nbytes):
        """INK_RESIDENT=1: keep the device-side expert bytes (repacked NVFP4 / raw Q8_0) up to INK_RES_GB (default 16.6); later experts stream as before"""
        if RESIDENT and self._xc_bytes + nbytes <= RES_GB * 1e9:
            self._xc[key] = val
            self._xc_bytes += nbytes
            self.kstats["expert_resident"] = len(self._xc)

    def _expert_mm(self, name, e, x):
        """x [R, K] fp32 @ expert e of the [K, N, E] tensor `name` -> [R, N] fp32: NVFP4 through ext.fq_gemv_nvfp4 (the stored bytes repacked to the ModelOpt
        layout on the device, fp32 activations), Q8_0 through ext.fq_gemv_q8_0, anything else (and prefill widths > 16 rows) torch dequant + matmul"""
        ty, dims, off = self.g.infos[name]
        K, N = dims[0], dims[1]
        R = x.shape[0]
        k = self._kern()
        if self.backend != "cuda":
            self.kstats["expert_cpu"] += 1
        elif ty == T_NVFP4 and "ext" in k and hasattr(k["ext"], "fq_gemv_nvfp4") and R <= 16 and K % 64 == 0:
            ck = (name, e)
            if RESIDENT and ck in self._xc:
                packed, scales = self._xc[ck]
            else:
                raw = _u8(self.g, off + e * N * row_bytes(ty, K), N * row_bytes(ty, K), self.dev).reshape(N, -1)
                packed, scales = nvfp4_gguf_to_modelopt(raw)
                self._xcache(ck, (packed, scales), packed.numel() + scales.numel())
            out = torch.empty(R, N, dtype=torch.float32, device=self.dev)
            k["ext"].fq_gemv_nvfp4(x.float().contiguous(), packed, scales, 1.0, out, True)
            self.kstats["nvfp4_gap"] += 1
            return out
        elif ty == T_Q8_0 and "ext" in k and R <= 16 and K % 32 == 0:
            ck = (name, e)
            packed = self._xc.get(ck) if RESIDENT else None
            if packed is None:
                packed = _u8(self.g, off + e * N * row_bytes(ty, K), N * row_bytes(ty, K), self.dev).reshape(N, K // 32, 34)
                self._xcache(ck, packed, packed.numel())
            out = torch.empty(R, N, dtype=torch.float32, device=self.dev)
            k["ext"].fq_gemv_q8_0(x.float().contiguous(), packed, out)
            self.kstats["q8_gemv"] += 1
            return out
        else:      # expert_fallback = an NVFP4 expert that missed the kernel (must stay 0); other types (BF16 / MXFP8 layers) are counted apart
            self.kstats["expert_fallback" if ty == T_NVFP4 else "expert_fallback_other"] += 1
        return x @ read_expert(self.g, name, e, self.dev).T

    def rmsw(self, x, w):
        """rms_norm(x, model eps) * w: one Triton launch with INK_GLUE_FUSE=norm (T <= 8), else torch"""
        if "norm" in FUSE and self.backend == "cuda" and x.shape[0] <= 8:
            return _misc().rmsw(x, w.float().contiguous(), self.eps)
        return rms(x, self.eps) * w

    def _norm_w(self, x, w):
        """rms_norm(eps) * w through the backend that owns a fused form: the HIP glue when its control / norm launch is on (it also
        writes the controller's rms*w), the Triton norm kernel on NVIDIA, else torch"""
        if self.gl is not None:
            return self.gl.rmsw(x, w) if "control" in self.gl.fuse else rms(x, self.eps) * w
        return self.rmsw(x, w)

    def mm(self, x, name):
        """x [T, K] @ W[name].T -> [T, N] fp32. INK_RTX=dense: Q8_0 matrices (T <= 16) go through ext.fq_gemv_q8_0 on the resident stored bytes (1.06 B / weight read once);
        anything else = torch matmul on the fp32 weight"""
        if "dense" in RTX and self.backend == "cuda" and x.shape[0] <= 16:
            ty, dims, off = self.g.infos[name]
            k = self._kern()
            if ty == T_Q8_0 and len(dims) == 2 and dims[0] % 32 == 0 and "ext" in k:
                pk = self._q8p.get(name)
                if pk is None:
                    pk = self._q8p[name] = _u8(self.g, off, dims[1] * row_bytes(ty, dims[0]), self.dev).reshape(dims[1], dims[0] // 32, 34)
                out = torch.empty(x.shape[0], dims[1], dtype=torch.float32, device=self.dev)
                k["ext"].fq_gemv_q8_0(x.float().contiguous(), pk, out)
                self.kstats["dense_q8"] += 1
                return out
            self.kstats["dense_torch"] += 1
        return x @ self.W(name).T

    def _akern(self):
        if self._ax is None:
            d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools", "inkling2")
            if d not in sys.path:
                sys.path.insert(0, d)
            from ink2_attn_glue import Ink2AttnCuda
            assert (self.n_head, self.n_kv, self.hd, self.rel_extent, self.n_rel) == (24, 8, 128, 1024, 16), "the attention kernels are built for 24 / 8 heads, hd 128, rel 16 x 1024"
            assert self.backend == "cuda", "ink2_attn_cuda.so is an NVIDIA build: the AMD attention is gl.attn (ink2_glue.py)"
            self._ax = Ink2AttnCuda()
            if self._pos0 is None:
                self._pos0 = torch.zeros(1, dtype=torch.int32, device=self.dev)
            self._ax.pos0 = self._pos0       # ONE device pos0, set outside the graph by _glue_begin
        return self._ax

    def _kvcache(self, il):
        c = self._kvc.get(il)
        if c is None:
            c = self._kvc[il] = self._akern().new_cache(self.pattern[il], int(os.environ.get("INK_CTX", "34816")), self.dev)
        return c

    def dev_fires(self):
        """device-side path counters of our kernels into kstats (fires-check; synchronous, call outside the timed region)"""
        d = {}
        if self._xk is not None:
            d["xk_gate_up_entries_down_blocks"] = list(self._xk.fires())
        if getattr(self, "_rk", None) is not None:
            d["router_blocks"] = self._rk.fires()
        if self._ax is not None:
            d["ak_gkv_lattn_combine_append_rel_blocks"] = self._ax.fires()
        self.kstats["dev_fires"] = d
        return d

    def rtx_attn(self, T):
        return "attn" in RTX and self.backend == "cuda" and T <= 8

    def seed_kv(self, il, k, v, start):
        """synthetic KV for the decode bench: k / v [n_kv, n, hd] at absolute positions start .. start + n - 1 of application il"""
        if "attn" in RTX and self.backend == "cuda":
            self._akern().seed(self._kvcache(il), k, v, start, self.pattern[il])
        else:
            slot = self.slots[il]
            slot.k, slot.v, slot.start = k, v, start

    def _attend_rtx(self, il, ip, q, kc, vc, r, sinks, gate, start):
        """our kernels for one application: rel table + in-place append + attend (fused combine, sink, gate). q [T, nh, hd], kc / vc [T, nkv, hd], r [T, nh, 16] fp32 -> [T, nh * hd] fp32"""
        T = q.shape[0]
        ax = self._akern()
        pt = self._relT.get(ip)
        if pt is None:
            pt = self._relT[ip] = self.W(f"blk.{ip}.attn_rel_bias.weight").float().T.contiguous()
        swa = self.pattern[il]
        o = ax.step(self._kvcache(il), q.half().contiguous(), kc.float().contiguous(), vc.float().contiguous(), r.float().contiguous(), pt, sinks.float().contiguous(),
                    gate.float().contiguous(), T, swa)
        self.kstats["ak_lattn" if swa else "ak_gkv"] += 1
        return o

    def _xkern(self):
        if self._xk is None:
            import sys
            d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools", "inkling2")
            if d not in sys.path:
                sys.path.insert(0, d)
            from ink2_experts_cuda import Ink2ExpertsCuda
            self._xk = Ink2ExpertsCuda()
        return self._xk

    def _xlayer(self, L, gu_n, dn_n):
        """resident bytes of one physical layer's routed experts + per-expert scales (loaded once; the 24 physical layers = the 48 applications)"""
        v = self._xl.get(L)
        if v is None:
            gty, gd, goff = self.g.infos[gu_n]
            dty, dd, doff = self.g.infos[dn_n]
            assert gty == dty and gty in (T_NVFP4, T_Q8_0) and gd[0] == self.D and gd[1] == 2 * dd[0] and dd[1] == self.D and gd[2] == self.n_exp, (gty, dty, gd, dd)
            E = self.n_exp
            gb, db = gd[1] * row_bytes(gty, gd[0]), dd[1] * row_bytes(dty, dd[0])
            gu = _u8(self.g, goff, E * gb, self.dev).reshape(E, gd[1], -1)
            dn = _u8(self.g, doff, E * db, self.dev).reshape(E, dd[1], -1)
            s1 = gu_n[:-6] + "scale"
            s2 = dn_n[:-6] + "scale"
            v = self._xl[L] = (gu, dn, self.W(s1).float().contiguous() if self.has(s1) else None, self.W(s2).float().contiguous() if self.has(s2) else None, gty == T_Q8_0)
        return v

    def _moe_kernel(self, L, x, sel, w, gu_n, dn_n):
        """OUR grouped experts kernels: plan + gate_up (SwiGLU, per-expert scales) + down in 3 launches, no host sync. x [T, D] fp32, T <= 8"""
        gu, dn, s_gu, s_dn, q8 = self._xlayer(L, gu_n, dn_n)
        cg, cd = CFG_Q8 if q8 else CFG_NV
        out = self._xkern().moe_nvfp4(x.float().contiguous(), sel.to(torch.int32).contiguous(), w.float().contiguous(), gu, dn, s_gu, s_dn,
                                      self.beta, self.lbeta, cfg_gu=cg, cfg_dn=cd, q8=q8)
        self.kstats["xk_calls"] += 1
        self.kstats["xk_q8" if q8 else "xk_nvfp4"] += 1
        return out

    def _rkern(self):
        if getattr(self, "_rk", None) is None:
            import sys
            d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools", "inkling2")
            if d not in sys.path:
                sys.path.insert(0, d)
            from ink2_router_glue import Ink2RouterCuda
            self._rk = Ink2RouterCuda()
        return self._rk

    def _moe(self, L, x):
        """router picks + the selected experts. The backend that owns both (HIP glue router / experts, CUDA router / grouped-expert
        kernels) runs them as launches; the torch path (sigmoid + topk + per-expert dequant) is the reference of both."""
        gl = self.gl
        gu_n, dn_n = f"{L}.ffn_gate_up_exps.weight", f"{L}.ffn_down_exps.weight"
        self._hk("router_in", self._il, x)
        if gl is not None and gl.has("router"):
            sel, w = gl.router(L, x)                                             # one launch: sigmoid + bias + top-8 + renorm x scale
        elif "router" in RTX and "experts" in RTX and self.backend == "cuda" and x.shape[0] <= 8 and self.g.infos[gu_n][0] in (T_NVFP4, T_Q8_0):
            bn = f"{L}.exp_probs_b.bias"      # sigmoid + bias + top-8 + renorm in ONE launch (ink2_router_cuda.cu); the plan + experts kernels read its sel / wt
            sel, w = self._rkern().route(x.float().contiguous(), self.W(f"{L}.ffn_gate_inp.weight").float().contiguous(), self.W(bn).float().contiguous() if self.has(bn) else None, self.w_scale)
            self.kstats["router_k"] += 1
            self._hk("router_sel", self._il, sel)
            return self._moe_kernel(L, x, sel, w, gu_n, dn_n)
        else:
            self.kstats["torch_router"] += 1
            R = self.W(f"{L}.ffn_gate_inp.weight")                               # [E, D] fp32
            scores = torch.sigmoid(x @ R.T)
            bias = self.W(f"{L}.exp_probs_b.bias") if self.has(f"{L}.exp_probs_b.bias") else 0.0
            sel = torch.topk(scores + bias, self.topk, dim=-1, sorted=False).indices
            w = scores.gather(1, sel)
            w = w / (w.sum(-1, keepdim=True) + 1e-20) * self.w_scale
        self._hk("router_sel", self._il, sel)
        if gl is not None and gl.has("experts"):
            return gl.experts(L, x, sel, w)                                      # grouped expert kernels on the resident bytes
        if "experts" in RTX and self.backend == "cuda" and x.shape[0] <= 8 and self.g.infos[gu_n][0] in (T_NVFP4, T_Q8_0):
            return self._moe_kernel(L, x, sel, w, gu_n, dn_n)
        if "prefill" in RTX and self.backend == "cuda" and x.shape[0] > 8 and self.g.infos[gu_n][0] in (T_NVFP4, T_Q8_0):
            out = self._moe_prefill(L, x, sel, w, gu_n, dn_n)
            if os.environ.get("INK_PF_CHECK") == "1":      # parity probe: the torch loop on the same inputs, per-call numbers into kstats["pf_check"]
                ref = self._moe_loop(L, x, sel, w, exact=True)
                d = (out - ref).float()
                self.kstats.setdefault("pf_check", []).append((self._il, round(float(torch.dot(out.flatten(), ref.flatten()) / (out.norm() * ref.norm())), 6), float(d.norm() / ref.norm()), float(ref.abs().max()), float(out.abs().max())))
            return out
        self.kstats["torch_moe" if gl is None else "torch_experts"] += 1
        return self._moe_loop(L, x, sel, w, exact=getattr(self, "pf_exact", False))

    def _moe_loop(self, L, x, sel, w, exact=False):
        """the torch expert loop. exact = fp32 activations @ the dequantised expert (no gemv kernel: the kernels quantise activations to q8_1, ~1 % rel error): the reference of the prefill parity gate"""
        gu_n, dn_n = f"{L}.ffn_gate_up_exps.weight", f"{L}.ffn_down_exps.weight"
        s_gu = self.W(gu_n[:-6] + "scale") if self.has(gu_n[:-6] + "scale") else None
        s_dn = self.W(dn_n[:-6] + "scale") if self.has(dn_n[:-6] + "scale") else None
        out = torch.zeros_like(x)
        ff = self.g.infos[dn_n][1][0]
        mmf = (lambda name, e, a: a.float() @ read_expert(self.g, name, e, self.dev).float().T) if exact else self._expert_mm
        for e in torch.unique(sel).tolist():
            tok, slot = torch.where(sel == e)
            h = mmf(gu_n, e, x[tok])
            if s_gu is not None:
                h = h * s_gu[e]
            y = mmf(dn_n, e, situ(h[:, :ff], h[:, ff:], self.beta, self.lbeta))
            if s_dn is not None:
                y = y * s_dn[e]
            out.index_add_(0, tok, y * w[tok, slot][:, None])
        return out

    def _moe_prefill(self, L, x, sel, w, gu_n, dn_n):
        """T > 8 rows (INK_RTX=prefill): per-expert Triton dequant of the resident bytes + cuBLAS fp16 GEMM (tools/inkling2/ink2_prefill_moe.py)"""
        import sys
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools", "inkling2")
        if d not in sys.path:
            sys.path.insert(0, d)
        if getattr(self, "_pm", None) is None:
            if os.environ.get("INK_PF_V", "1") == "2":
                from ink2_prefill_moe2 import PrefillMoe2
                self._pm = PrefillMoe2(self.dev, self.D, self.g.infos[dn_n][1][0], self.topk, self.beta, self.lbeta, self.n_exp)
            else:
                from ink2_prefill_moe import PrefillMoe
                self._pm = PrefillMoe(self.dev, self.D, self.g.infos[dn_n][1][0], self.topk, self.beta, self.lbeta)
        gu, dn, s_gu, s_dn, q8 = self._xlayer(L, gu_n, dn_n)
        self.kstats["prefill_moe"] = self.kstats.get("prefill_moe", 0) + 1
        return self._pm(gu, dn, s_gu, s_dn, q8, x.float(), sel, w.float())

    def _hk(self, name, il, t):
        if self.hook is not None:
            self.hook(name, il, t)

    # ---------------------------------------------------------------- forward over one chunk
    def _arange(self, T):
        a = self._ar.get(T)
        if a is None:
            a = self._ar[T] = torch.arange(T, device=self.dev)
        return a

    def alloc_state(self):
        """conv history of every application, allocated once (graphs keep its address)"""
        for slot in self.slots:
            if slot.hist is None:
                slot.hist = torch.zeros(self.n_conv - 1, self.n_chan, device=self.dev)

    def embed(self, ids):
        """token embedding rows (host pread) -> [T, D] fp32; never inside a captured graph"""
        cur = read_rows(self.g, self.tok_embd_name, ids, self.dev)
        return rms(cur, self.eps) * self.W("token_embd_norm.weight")

    def _chunk(self, ids, start):
        self._glue_begin(len(ids), start)
        return self._body(self.embed(ids), len(ids), start)

    def _body(self, cur, T, start):
        """the 48 applications over the embedded rows `cur` [T, D]. Every site asks the backend that owns a fused form first: the HIP
        glue (gl, picked on AMD), the CUDA kernels / Triton fusions (RTX / FUSE, picked on NVIDIA), else the torch maths. With a device
        pos0 (attention kernels / fused conv of either backend) positions are read from it: no host value is used, so one captured
        graph serves every position"""
        D, nh, nkv, hd = self.D, self.n_head, self.n_kv, self.hd
        gl = self.gl
        pos = (self.gl.pos0.long() + self._arange(T)) if self.gl is not None else \
              ((self._pos0.long() + self._arange(T)) if self._pos0 is not None else torch.arange(start, start + T, device=self.dev))
        tau = (1.0 + self.log_alpha * torch.log(((pos + 1).float() / self.log_floor).clamp(min=1.0)))[:, None, None]
        fz = gl.fuse if gl is not None else frozenset()                         # HIP glue fusions
        fc, fk, fm = "conv" in fz, "control" in fz, "mix" in fz
        fz2 = self.fuse_conv(T)                                                 # CUDA fused conv (Triton)
        blocks, prelude_blocks, prelude_cur, bank, bank_last = [], None, None, [], None
        nb = prelude_nb = n_msg = 0                                             # fused mix: block rows live in gl.bank [NBMAX, 8, D] (current pass) and gl.mbank (previous pass)
        for il in range(self.n_app):
            ip = self.loop_layer(il)
            L = f"blk.{ip}"
            swa = self.pattern[il]
            msgs = [None] * 4
            if self.n_core and self.core_start <= ip < self.core_end:
                pss = (il - self.core_start) // self.n_core
                if ip == self.core_start:
                    if pss == 0:
                        prelude_blocks, prelude_cur, prelude_nb = list(blocks), cur, nb
                    else:
                        if fm:
                            n_msg = nb - (prelude_nb + 1)
                            if n_msg > 0:
                                gl.mbank[:n_msg, :T].copy_(gl.bank[prelude_nb + 1:nb, :T])
                            bank_last, nb, cur = cur, prelude_nb, prelude_cur
                        else:
                            bank, bank_last = blocks[len(prelude_blocks) + 1:], cur
                            blocks, cur = list(prelude_blocks), prelude_cur
                if pss > 0:
                    row = 2 * (ip - self.core_start)
                    if fm:
                        mo = gl.mix_f(gl.mbank, max(n_msg, 0), bank_last, gl.msg_queries(row))      # [4, T, D]: attn, mlp, k, v queries in one launch
                        msgs = [mo[0], mo[2], mo[3], mo[1]]                      # q(attn), k, v, mlp
                    else:
                        qa = self.W("loop_res_query.weight")[row:row + 2]
                        qk = self.W("loop_kv_res_query.weight")[row:row + 2]
                        ma, mk = self.mix(bank, bank_last, qa), self.mix(bank, bank_last, qk)
                        msgs = [ma[0], mk[0], mk[1], ma[1]]                      # q(attn), k, v, mlp
            self._il = il
            slot = self.slots[il]
            if slot.hist is None:
                slot.hist = torch.zeros(self.n_conv - 1, self.n_chan, device=self.dev)
            self._hk("layer_inp", il, cur)
            inp_sa = cur
            inputs = [cur, cur, cur]
            normed = False                                                       # True once the input norm (rms * w) is already applied
            nws = [self.W(f"{L}.attn_norm.weight")] * 3 if swa else [self.W(f"{L}.attn_{c}_inp_norm.weight") for c in "qkv"]
            if self.has(f"{L}.attn_res_query.weight"):
                q = self.W(f"{L}.attn_res_query.weight").reshape(-1, D)
                if fm:
                    epi = None if msgs[0] is not None else (nws[0] if swa else gl.norm_stack(L))
                    mixed = gl.mix_f(gl.bank, nb, cur, q, epi)               # depth mix + the input norm epilogue in one launch
                    normed = epi is not None
                else:
                    mixed = self.mix(blocks, cur, q)
                inputs = [mixed[0 if swa else j] for j in range(3)]
            if msgs[0] is not None:
                if fk:
                    inputs = [gl.control(inputs[j], msgs[j], L, j, nws[j]) for j in range(3)]      # controller + rms * w
                    normed = True
                else:
                    inputs = [self.control(inputs[j], msgs[j], L, j) for j in range(3)]
            if not normed:
                if swa and msgs[0] is None:
                    inputs = [self._norm_w(inputs[0], nws[0])] * 3
                else:
                    inputs = [self._norm_w(inputs[j], nws[j]) for j in range(3)]
            xq, xk, xv = inputs
            use_ap = gl is not None and gl.has("attn") and gl.ap_ok(L, swa)
            if use_ap:                                                           # one launch: q, gate, rel, sink_proj, k, v (+ conv_proj) from the resident planes
                Q, Kp, Vp, R, sinks, gate, dk, dv = gl.proj(L, swa, xq, xk, xv)
                if not swa:
                    dk, dv = dk.reshape(T, nkv, self.n_conv), dv.reshape(T, nkv, self.n_conv)
            else:
                self.kstats["torch_attnproj"] += 1
                Q = self.mm(xq, f"{L}.attn_q.weight")
                Kp = self.mm(xk, f"{L}.attn_k.weight")
                Vp = self.mm(xv, f"{L}.attn_v.weight")
                R = self.mm(xq, f"{L}.attn_rel.weight")
                dk = dv = None
                if not swa:
                    dk = (self.mm(xk, f"{L}.attn_k_conv_proj.weight")).reshape(T, nkv, self.n_conv)
                    dv = (self.mm(xv, f"{L}.attn_v_conv_proj.weight")).reshape(T, nkv, self.n_conv)
                sinks = self.mm(xq, f"{L}.attn_sink_proj.weight")              # [T, nh]
                gate = self.mm(xq, f"{L}.attn_gate.weight")                    # [T, nh * hd] pre-sigmoid
            if fc:                                                               # conv + l2 clip (K) / rms (V) in one launch each, on the position-indexed rings
                Kc = gl.conv_kv(il, 0, Kp, self.W(f"{L}.attn_k_conv.weight"), dk, "k").reshape(T, nkv, hd)
                Vc = gl.conv_kv(il, self.kvd, Vp, self.W(f"{L}.attn_v_conv.weight"), dv, "v").reshape(T, nkv, hd)
            elif fz2:       # ONE launch per conv: taps + residual + l2 clip (K) / rms (V), position-indexed ring
                self.kstats["conv_fused"] += 4       # K, V (here) + post-attn, post-ffn (below)
                Kc = self._cf.kv(il, 0, Kp.float().contiguous(), self.W(f"{L}.attn_k_conv.weight").float().contiguous(), dk, "k", torch.empty(T, self.kvd, device=self.dev))
                Vc = self._cf.kv(il, self.kvd, Vp.float().contiguous(), self.W(f"{L}.attn_v_conv.weight").float().contiguous(), dv, "v", torch.empty(T, self.kvd, device=self.dev))
                Kc, Vc = Kc.reshape(T, nkv, hd), Vc.reshape(T, nkv, hd)      # already clipped / normalised by the conv kernel
            else:
                Kc = self.conv(Kp, self.W(f"{L}.attn_k_conv.weight"), dk, slot, 0)
                Vc = self.conv(Vp, self.W(f"{L}.attn_v_conv.weight"), dv, slot, self.kvd)
                Kc = l2_clip(Kc.reshape(T, nkv, hd), None, self.qk_clip)
                Vc = rms(Vc.reshape(T, nkv, hd), self.eps)
            if gl is not None and "clip" in fz:                                  # Q / R l2 clip (+ tau on global layers) in one launch
                Q, R = gl.qr_clip(Q, R, self.W(f"{L}.attn_q_clip.weight"), self.W(f"{L}.attn_rel_clip.weight"), None if swa else tau)
            else:
                Q = l2_clip(Q.reshape(T, nh, hd), self.W(f"{L}.attn_q_clip.weight"), self.qk_clip)
                R = l2_clip(R.reshape(T, nh, self.n_rel), self.W(f"{L}.attn_rel_clip.weight"), self.rel_clip)
                if not swa:
                    Q, R = Q * tau, R * tau
            if gl is not None and gl.has("lattn" if swa else "gkv"):
                o = gl.attn(il, L, swa, Q, R, Kc, Vc, sinks, gate)               # cache append + split-K attention + sink + sigmoid gate, [T, nh * hd] fp32
            elif self.rtx_attn(T):
                o = self._attend_rtx(il, ip, Q, Kc, Vc, R, sinks, gate, start)   # rel table, append, attend, sink, sigmoid gate: our kernels
            else:
                if "attn" in RTX and self.backend == "cuda":
                    raise RuntimeError("INK_RTX=attn keeps the KV in the kernel cache: T <= 8 rows per chunk only (use INK_RTX=experts for prefill)")
                self.kstats["torch_attn"] += 1
                self.kstats["torch_attn_local" if swa else "torch_attn_global"] += 1
                tab = R @ self.W(f"{L}.attn_rel_bias.weight").T                  # [T, nh, extent]
                o = self._attend(Q, Kc, Vc, tab, sinks, slot, swa, pos).reshape(T, nh * hd)
                o = o * torch.sigmoid(gate)
            o = gl.out_proj(L, swa, o) if use_ap else self.mm(o, f"{L}.attn_output.weight")
            opener = ip % self.blk == 0
            if fc:                                                               # conv + the block residual in one launch (a block opener has no residual)
                cur = gl.conv_post(il, 2 * self.kvd, o, self.W(f"{L}.attn_post_conv.weight"), None if opener else inp_sa)
                self._hk("attn_out", il, cur)
            elif fz2:       # conv + residual add (inside a block) in one launch; the attn_out hook then sees the post-residual value
                cur = self._cf.post(il, 2 * self.kvd, o.float().contiguous(), self.W(f"{L}.attn_post_conv.weight").float().contiguous(), None if opener else inp_sa, torch.empty(T, D, device=self.dev))
                self._hk("attn_out", il, cur)
            else:
                cur = self.conv(o, self.W(f"{L}.attn_post_conv.weight"), None, slot, 2 * self.kvd)
                self._hk("attn_out", il, cur)
                if not opener:
                    cur = inp_sa + cur
            if opener:
                if fm:
                    gl.bank[nb, :T].copy_(inp_sa)
                    nb += 1
                else:
                    blocks.append(inp_sa)
            ffn_inp = cur
            qf = self.W(f"{L}.ffn_res_query.weight").reshape(1, D)
            nwf = self.W(f"{L}.ffn_norm.weight")
            if fm:
                cur = gl.mix_f(gl.bank, nb, cur, qf, None if msgs[3] is not None else nwf)[0]
            else:
                cur = self.mix(blocks, cur, qf)[0]
            if msgs[3] is not None:
                cur = gl.control(cur, msgs[3], L, 3, nwf) if fk else self._norm_w(self.control(cur, msgs[3], L, 3), nwf)
            elif not fm:
                cur = self._norm_w(cur, nwf)
            cur = self._moe(L, cur)
            if fc:
                cur = gl.conv_post(il, 2 * self.kvd + D, cur, self.W(f"{L}.ffn_post_conv.weight"), ffn_inp)
                self._hk("ffn_out", il, cur)
            elif fz2:
                cur = self._cf.post(il, 2 * self.kvd + D, cur.float().contiguous(), self.W(f"{L}.ffn_post_conv.weight").float().contiguous(), ffn_inp, torch.empty(T, D, device=self.dev))
                self._hk("ffn_out", il, cur)
            else:
                cur = self.conv(cur, self.W(f"{L}.ffn_post_conv.weight"), None, slot, 2 * self.kvd + D)
                self._hk("ffn_out", il, cur)
                cur = ffn_inp + cur
            self._hk("l_out", il, cur)
        qo, nwo = self.W("output_res_query.weight").reshape(1, D), self.W("output_norm.weight")
        if fm:
            cur = gl.mix_f(gl.bank, nb, cur, qo, nwo)[0]                         # depth mix + the output norm in one launch
        else:
            cur = self.mix(blocks, cur, qo)[0]
            cur = self._norm_w(cur, nwo)
        self._hk("result_norm", -1, cur)
        return cur

    def head_raw(self, h):
        """h [T <= 16, D] -> UNcapped logits [T, vocab] through the resident Q8_0 head (the drafter reads the softcap-free base rows; the softcap is monotonic)"""
        pk = self._q8p["output.weight"]
        logits = torch.empty(h.shape[0], self.vocab, dtype=torch.float32, device=self.dev)
        hf = h.float().contiguous()
        for r0 in range(0, h.shape[0], 16):
            self._kern()["ext"].fq_gemv_q8_0(hf[r0:r0 + 16].contiguous(), pk, logits[r0:r0 + 16])
        return logits

    def head(self, h):
        """h [T, D] -> logits [T, vocab] (softcapped, unpadded)"""
        if self.gl is not None and self.gl.has("head"):
            lg = self.gl.head(h)                                               # HIP head kernel (Q8_0 / BF16 rows, softcap inside)
            if lg is not None:
                return lg
        if "dense" in RTX and self.backend == "cuda" and self.g.infos["output.weight"][0] == T_Q8_0:
            ty, dims, off = self.g.infos["output.weight"]
            pk = self._q8p.get("output.weight")
            if pk is None:        # only the unpadded vocab rows are read
                pk = self._q8p["output.weight"] = _u8(self.g, off, self.vocab * row_bytes(ty, dims[0]), self.dev).reshape(self.vocab, dims[0] // 32, 34)
            logits = torch.empty(h.shape[0], self.vocab, dtype=torch.float32, device=self.dev)
            hf = h.float().contiguous()
            for r0 in range(0, h.shape[0], 16):       # blocks of <= 16 rows: the GEMV reads the 0.49 GB head once per block, no fp32 dequant of the head (1.85 GB)
                self._kern()["ext"].fq_gemv_q8_0(hf[r0:r0 + 16].contiguous(), pk, logits[r0:r0 + 16])
            self.kstats["dense_q8"] += 1
            if self.softcap:
                logits = self.softcap * torch.tanh(logits / self.softcap)
            return logits
        self.kstats["torch_head"] += 1
        if RESIDENT and self.backend == "cuda":
            W = self._w_resident("output.weight")
        else:
            W = self._w.get("output.weight")
            if W is None:
                W = read_dense(self.g, "output.weight", self.dev)
                if self.cache_weights:
                    self._w["output.weight"] = W
        logits = h @ W[:self.vocab].T
        if self.softcap:
            logits = self.softcap * torch.tanh(logits / self.softcap)
        return logits

    # ---------------------------------------------------------------- captured graphs (INK_GRAPH=1 / INK_GLUE_GRAPH=1): one step per T bucket
    GRAPH_BUCKETS = (1, 2, 4, 8)

    def _graph_body(self, T):
        """what a graph records: the 48 applications + the head over ALL T rows, from the static input buffer; returns (hidden [T, D], logits [T, vocab])"""
        h = self._body(self._gin[T], T, 0)
        lg = self.head(h)
        assert lg.shape[0] == T and h.shape[0] == T
        return h, lg

    @torch.no_grad()
    def capture_graphs(self, buckets=None, fill_id=1000, taps=()):
        """capture the step once per T bucket (needs a backend that reads no host value: the device pos0, the device plan). On NVIDIA:
        INK_RTX=experts,attn[/dense] and INK_GLUE_FUSE=conv; on AMD: INK_GLUE with all six stages. Per bucket: eager warm-up on a side
        stream (loads weights, JIT-compiles, creates every scratch), then torch.cuda.CUDAGraph capture (a CPU stub re-runs the body).
        Hooks and profile spans are off while capturing; state is reset in place afterwards."""
        gl = self.gl
        if gl is not None:
            assert set(gl.STAGES_ALL) <= gl.on, "graph capture needs the six glue stages (INK_GLUE=all): the torch paths read host values"
            prof, gl.prof = gl.prof, False
        else:
            assert self.backend == "cuda" and {"experts", "attn"} <= RTX and "conv" in FUSE, "graph capture needs INK_RTX=experts,attn[,dense] and INK_GLUE_FUSE=conv (the torch conv keeps per-slot host state)"
            prof = None
        hook, self.hook = self.hook, None
        t0 = time.time()
        self.tap_layers = tuple(taps)
        self.alloc_state()
        cuda = self.dev.type == "cuda"
        try:
            for T in tuple(buckets or self.GRAPH_BUCKETS):
                self._gin[T] = torch.zeros(T, self.D, device=self.dev)
                if self.tap_layers:
                    buf = self._gtap[T] = torch.zeros(T, len(self.tap_layers) + 1, self.D, device=self.dev)
                    def _w(name, il, t, buf=buf):
                        if name == "layer_inp" and il in self.tap_layers:
                            buf[:, self.tap_layers.index(il)].copy_(t)
                        elif name == "result_norm":
                            buf[:, -1].copy_(t)
                    self.hook = _w
                self.reset()
                self._glue_begin(T, 0)
                if cuda:
                    side = torch.cuda.Stream()
                    side.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(side):
                        self._gin[T].copy_(self.embed([fill_id] * T))
                        for _ in range(2):
                            self._graph_body(T)
                    torch.cuda.current_stream().wait_stream(side)
                    torch.cuda.synchronize()
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g, capture_error_mode="thread_local"):
                        out = self._graph_body(T)
                else:                                                            # CPU: a stub that re-runs the body into static buffers (tests the protocol, not the capture)
                    self._gin[T].copy_(self.embed([fill_id] * T))
                    out = self._graph_body(T)
                    g = _StubGraph(self, T, out)
                self.graphs[T] = (g, self._gin[T], out)
                print(f"graph T={T} captured ({time.time() - t0:.1f} s)", flush=True)
        finally:
            self.hook = hook
            self.reset()
            if gl is not None:
                gl.prof = prof
                gl.reset()
        return self.graphs

    @torch.no_grad()
    def prefill_big(self, ids, chunk=1024, taps=None):
        """prompt prefill for the graph engine from an EMPTY state (self.n == 0): chunks of `chunk` rows (> 8) through the torch step (INK_RTX=prefill grouped experts; torch attention with
        per-slot K / V; torch conv with slot.hist: INK_RTX=attn and INK_GLUE_FUSE=conv are switched off for the duration), then the K / V of every application are written into the
        kernel caches (seed_kv: global = all rows, SWA = the last window) and the last 3 pre-conv rows into the position-indexed rings. Returns (logits of the last row [vocab], taps [n, ntap + 1, D] or None).
        taps = layer indices whose layer_inp is returned, plus the final normed hidden."""
        assert self.n == 0 and self.backend == "cuda" and self.gl is None, "prefill_big seeds the CUDA kernel caches from a torch prefill: NVIDIA kernels only"
        ids = [int(i) for i in ids]
        n = len(ids)
        self._glue_begin(1, 0)                                    # creates ConvFuse / pos0 / attention kernels while the flags are still on
        saved = (set(RTX), set(FUSE), self.hook)
        RTX.discard("attn"), FUSE.discard("conv")
        got = {}
        if taps:
            self.hook = lambda name, il, t: (got.setdefault(il if name == "layer_inp" else -1, []).append(t.clone()) if (name == "result_norm" or (name == "layer_inp" and il in taps)) else None)
        try:
            self.reset()
            h = None
            for st in range(0, n, chunk):
                if self._pos0 is not None:
                    self._pos0.fill_(st)
                h = self._chunk(ids[st:st + chunk], self.n)
                self.n += min(chunk, n - st)
            lg = self.head(h[-1:])[0]
        finally:
            RTX.clear(), RTX.update(saved[0]), FUSE.clear(), FUSE.update(saved[1])
            self.hook = saved[2]
        for il, slot in enumerate(self.slots):                    # torch state -> kernel state
            self.seed_kv(il, slot.k, slot.v, slot.start)
            if self._cf is not None:
                hist = slot.hist                                  # [3, n_chan] pre-conv rows n-3 .. n-1
                for i in range(hist.shape[0]):
                    p = n - hist.shape[0] + i
                    if p >= 0:
                        self._cf.rings[il][p % 16].copy_(hist[i])
            slot.k = slot.v = None
        out = None
        if taps:
            out = torch.stack([torch.cat(got[l], 0) for l in (*taps, -1)], 1)
        self.kstats["prefill_big_tokens"] = self.kstats.get("prefill_big_tokens", 0) + n
        return lg, out

    # ---------------------------------------------------------------- speculative decoding entry points (engine adapter of tools/inkling2/spec_loop.py)
    @torch.no_grad()
    def spec_verify(self, ids):
        """append `ids` (T <= 8) through the graph of bucket T (or the next larger bucket, padded with the last id: pad rows only write positions >= n + T, never read) -> (logits [T, vocab], taps [T, ntap + 1, D]) static buffers,
        valid until the next replay. fw.n advances by T; rollback = set fw.n (device pos0 is rewritten at every step, rings are position indexed)."""
        T = len(ids)
        b = T if T in self.graphs else min(k for k in self.graphs if k >= T)
        h, lg = self._step_graph(list(ids) + [ids[-1]] * (b - T))
        self.n -= b - T
        return lg[:T], (self._gtap[b][:T] if self._gtap else None)

    def spec_snapshot(self):
        """state for arm switching (not a rollback): n, conv rings, the SWA rings (global caches only gain rows >= n, which a replay overwrites)"""
        sw = {il: (self._kvc[il][0].clone(), self._kvc[il][1].clone()) for il in self._kvc if self.pattern[il]}
        return dict(n=self.n, rings=[r.clone() for r in self._cf.rings], sw=sw)

    def spec_restore(self, s):
        self.n = s["n"]
        for r, x in zip(self._cf.rings, s["rings"]):
            r.copy_(x)
        for il, (k, v) in s["sw"].items():
            self._kvc[il][0].copy_(k)
            self._kvc[il][1].copy_(v)

    def _step_graph(self, ids):
        T = len(ids)
        g, gin, (h, lg) = self.graphs[T]
        if self.gl is not None:
            assert self.n + T <= self.gl.cap, f"global KV cache full: {self.n + T} > cap {self.gl.cap} (INK_GLUE_CAP)"
        self._glue_begin(T, self.n)
        gin.copy_(self.embed(ids))
        g.replay()
        self.n += T
        self.kstats["graph_replays"] += 1
        return h, lg

    def capture_check(self, T=1, allow=("_moe_loop", "head", "experts", "attn", "router")):
        """DRY capture check (any device, torch emulation of the kernels): trace one `_graph_body(T)` with every host-sync entry point patched to RECORD (item, cpu, tolist, numpy, __bool__,
        __int__, __float__, __index__, nonzero, unique, masked_select) plus host IO (`_u8`). Returns (events, bad): events = sorted unique (api, function, file:line); bad = those whose innermost
        frame in this code is not in `allow` (the torch emulation of the experts / attention / head / router is not what a graph records: the kernels are). Run `step` once first: weights load lazily."""
        ev = {}
        here = ("inkling2_forward.py", "ink2_glue.py")
        orig = {}

        def hit(api):
            f, names, first = sys._getframe(2), [], None
            while f is not None:
                if f.f_code.co_filename.split("/")[-1] in here:
                    names.append(f.f_code.co_name)
                    first = first or (f.f_code.co_name, f"{f.f_code.co_filename.split('/')[-1]}:{f.f_lineno}")
                f = f.f_back
            if first is not None:
                ev[(api,) + first] = ev.get((api,) + first, False) or any(n in allow for n in names)

        def wrap(owner, name, label=None):
            o = getattr(owner, name)
            orig[(owner, name)] = o

            def w(*a, **k):
                hit(label or name)
                return o(*a, **k)
            setattr(owner, name, w)
        mod = sys.modules[type(self).__module__]
        if T not in self._gin:
            self._gin[T] = torch.zeros(T, self.D, device=self.dev)
        self.alloc_state()
        self._gin[T].copy_(self.embed([1000] * T))                                # the embedding is outside the graph: not traced
        self._glue_begin(T, self.n)
        for n in ("item", "cpu", "tolist", "numpy", "__bool__", "__int__", "__float__", "__index__", "nonzero"):
            wrap(torch.Tensor, n)
        wrap(torch, "unique")
        wrap(torch, "nonzero")
        wrap(mod, "_u8", "host_io(_u8)")
        try:
            with torch.no_grad():
                self._graph_body(T)
        finally:
            for (owner, name), o in orig.items():
                setattr(owner, name, o)
        events = sorted(ev)
        return events, [e for e in events if not ev[e]]

    @torch.no_grad()
    def step(self, ids, all_pos=False):
        """append `ids` (list[int]) to the context; logits of the last position [vocab] (or of every new position [T, vocab])"""
        t0 = time.time()
        ids = [int(i) for i in ids]
        if self.graphs:
            if len(ids) in self.graphs:
                h, lg = self._step_graph(ids)
                self.last_h = h
                self.stats["layer_s"] += time.time() - t0
                return lg if all_pos else lg[-1]               # static buffers: valid until the next replay
            self.kstats["graph_eager"] += 1
        outs = []
        for s in range(0, len(ids), self.chunk):
            outs.append(self._chunk(ids[s:s + self.chunk], self.n + s))
        h = torch.cat(outs, 0)
        self.last_h = h
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


class _StubGraph:
    """CPU stand-in of a CUDAGraph: replay() re-runs the body (reading only static buffers + the device pos0) and copies into the captured outputs"""

    def __init__(self, fw, T, out):
        self.fw, self.T, self.out = fw, T, out

    def replay(self):
        h, lg = self.fw._graph_body(self.T)
        self.out[0].copy_(h)
        self.out[1].copy_(lg)
