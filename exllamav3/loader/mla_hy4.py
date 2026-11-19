"""CPU fp32 reference of the Hy4-flash attention layer (registry blocks mla + mla_window_sinks + mla_output_gate; design
HY4_STRIX_DESIGN.md section 2). Maths of llama.cpp hy-v4-flash.cpp build_attention (patch M:5787-5947), config from the GGUF
header keys and the per-layer tensors only. Weights are dequantised once (Q8_0 / F32); the ForeignLinear path is a later step.
Only relative import: loader/engram_rows.py (tests load both by path)."""
from __future__ import annotations
import importlib.util, os
import torch
from .engram_rows import load_dense, _dequant_q8_0, q8_0_roundtrip, rms
from . import ref_fused as _rf
from . import mla_bgemv as _BG
from . import attn_eager as _ae
from . import dense_req as _DR

PF_ATTN = os.environ.get("HY4_PREFILL_ATTN", "0") in ("1", "flash")
PF_ATTN_FLASH = os.environ.get("HY4_PREFILL_ATTN", "0") == "flash"       # 1 = prefill rows (T > PF_ATTN_MIN): flash-style MLA kernel (prefill_attn.py)
PF_ATTN_MIN = 16
def pf_wot():
    return os.environ.get("HY4_PF_WOT", "0") == "1"
#           # HY4_PF_WOT=1: gate output written transposed [H*vd, T] so the wo GEMM reads A column-major (faster vendor kernel at K=8192-16384)
def pf_f16():
    return os.environ.get("HY4_PF_F16", "0") == "1"


# HY4_PF_F16=1 (read live, per call): 1 = prefill attention chain in fp16 end to end (no fp32 [T, H, R] intermediates; same values, fewer passes)
PF_DENSE_ON = os.environ.get("HY4_PREFILL_DENSE", "0") == "1"


def _sinks_mod():
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "modules", "attention_fn", "mla_sinks.py")
    spec = importlib.util.spec_from_file_location("exl_mla_sinks", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _dense3(g, name, h):
    """GGUF 3-D Q8_0 tensor (ne0, ne1, h) -> [h, ne1, ne0] fp32 (ggml_mul_mat(w, x): out[h, j] = sum_k w[h, j, k] x[k])"""
    ty, dims, off = g.infos[name]
    if ty == 30:      # BF16 (the MTP block): stored [ne2][ne1][ne0] row-major
        raw = g._pread(off, dims[0] * dims[1] * dims[2] * 2)
        return torch.from_numpy(raw.copy()).view(torch.bfloat16).float().reshape(h, dims[1], dims[0])
    assert ty == 8 and dims[2] == h, (name, ty, dims)
    return _dequant_q8_0(g._pread(off, dims[1] * dims[2] * (dims[0] // 32) * 34), dims[0], dims[1] * dims[2]).view(h, dims[1], dims[0])


def rope_norm(x: torch.Tensor, pos: torch.Tensor, base: float) -> torch.Tensor:
    """ggml_rope_ext mode NORM (consecutive pairs), n_dims = x.shape[-1], no scaling / yarn: x [T, ..., P], pos [T]"""
    P = x.shape[-1]
    th = pos.double()[:, None] * (base ** (-2.0 * torch.arange(P // 2, dtype=torch.float64, device=pos.device) / P))[None, :]   # [T, P/2]
    c, s = th.cos().float().to(x.device), th.sin().float().to(x.device)
    shp = (x.shape[0],) + (1,) * (x.dim() - 2) + (P // 2,)
    c, s = c.view(shp), s.view(shp)
    x0, x1 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2], out[..., 1::2] = x0 * c - x1 * s, x0 * s + x1 * c
    return out


class Hy4MLA:
    """One attention layer: cur [T, n_embd] (normed collapsed stream) -> kqv / gated / out, with a K cache the caller keeps."""

    def __init__(self, g, layer: int, q8_act: bool = False, f16_kv: bool = False, lin=None):
        m = g.meta
        p = m["general.architecture"] + "."
        b = f"blk.{layer}."
        self.layer, self.q8_act, self.f16_kv = layer, q8_act, f16_kv
        lin = lin or (lambda n: load_dense(g, n))        # lin(name) -> fp32 [N, K] or a QLinear (stored bytes, hy4_forward.py)
        self.H = int(list(m[p + "attention.head_count"])[layer])
        self.hd, self.vd = m[p + "attention.key_length_mla"], m[p + "attention.value_length_mla"]
        self.rope_dim, self.R = m[p + "rope.dimension_count"], m[p + "attention.kv_lora_rank"]
        self.nope = self.hd - self.rope_dim
        pat = m.get(p + "attention.sliding_window_pattern")
        self.swa = bool(list(pat)[layer]) if pat is not None else False
        self.window = int(m[p + "attention.sliding_window"]) if self.swa else 0
        self.theta = float(m[p + ("rope.freq_base_swa" if self.swa else "rope.freq_base")])
        self.eps = float(m[p + "attention.layer_norm_rms_epsilon"])
        self.scale = 1.0 / float(self.hd) ** 0.5
        self.wq_a, self.q_a_norm = lin(b + "attn_q_a.weight"), load_dense(g, b + "attn_q_a_norm.weight")
        self.wq_b = lin(b + "attn_q_b.weight")
        self.wkv_a, self.kv_a_norm = lin(b + "attn_kv_a_mqa.weight"), load_dense(g, b + "attn_kv_a_norm.weight")
        self.wk_b, self.wv_b = _dense3(g, b + "attn_k_b.weight", self.H), _dense3(g, b + "attn_v_b.weight", self.H)
        self.wo, self.wgate = lin(b + "attn_output.weight"), lin(b + "attn_gate.weight")
        pn = b + POST_NORM
        self.post_norm = load_dense(g, pn) if pn in g.infos else None
        sk = b + "attn_sinks.weight"
        self.sinks = load_dense(g, sk) if sk in g.infos else None
        self.S = _sinks_mod()

    tm = type("_NoTm", (), {"t": staticmethod(lambda n: __import__("contextlib").nullcontext())})()      # replaced by Hy4Forward._Tm
    q8_skip: tuple = ()          # matmul tags (wq_a wq_b wkv_a kb vb wgate wo) whose activation is NOT quantised (diagnostics)
    f16_parts: tuple = ("q", "k")

    def _q(self, x, tag=None):      # activation of a Q8_0 mul_mat (llama.cpp CPU vec_dot_type)
        return q8_0_roundtrip(x) if self.q8_act and tag not in self.q8_skip else x

    def _h(self, x, part="k"):
        return x.half().float() if self.f16_kv and part in self.f16_parts else x

    def _w16(self, name):
        c = self.__dict__.setdefault("_w16c", {})
        if name not in c:
            c[name] = getattr(self, name).clamp(-65504.0, 65504.0).half().contiguous()
        return c[name]

    @staticmethod
    def _f16(x):
        return x.clamp(-65504.0, 65504.0).half()

    @staticmethod
    def _bmm_th(w16, x16):
        """w16 [H, Nout, K] fp16, x16 [T, H, K] fp16 -> [T, H, Nout] fp32 ROW-major: the batch-over-heads GEMM writes [H, T, Nout]; the permute-back copy keeps Nout contiguous.
        (Writing the result as a permuted view [T, H, Nout] with T fastest made every later elementwise / cat / clamp on it a stride-T gather: 10-30x slower.)"""
        o = torch.bmm(x16.permute(1, 0, 2), w16.transpose(1, 2))
        return o.permute(1, 0, 2).contiguous().float()

    def kv(self, cur, pos):
        """cur [T, n_embd], pos [T] -> K cache rows [T, R + rope] = [rms(kv_a)*w | rope(k_pe)] (V is the first R columns)"""
        if _ae.ENABLED:       # HY4_ATTN_EAGER: table rope, no cat
            return _ae.kv(self, cur, pos)
        tm = self.tm
        with tm.t("f kv: wkv_a mm"):
            c = self._q(cur, "wkv_a") @ self.wkv_a.T
        with tm.t("f kv: norm+rope"):
            lat = rms(c[:, :self.R], self.eps) * self.kv_a_norm
            pe = rope_norm(c[:, self.R:], pos, self.theta)
            return self._h(torch.cat([lat, pe], -1))

    def q(self, cur, pos, qa=None):
        """-> q_lat [T,H,R], q_pe [T,H,rope] (q_nope absorbed with k_b; rope on the pe part)"""
        if _ae.ENABLED:
            return _ae.q(self, cur, pos)
        T, H = cur.shape[0], self.H
        tm = self.tm
        with tm.t("f q: wq_a mm"):
            if qa is None or self.q8_act:
                qa = self._q(cur, "wq_a") @ self.wq_a.T
        with tm.t("f q: norm"):
            qr = rms(qa, self.eps) * self.q_a_norm
        with tm.t("f q: wq_b mm"):
            q = (self._q(qr, "wq_b") @ self.wq_b.T).view(T, H, self.hd)
        with tm.t("f q: rope"):
            qn, qp = q[..., :self.nope], rope_norm(q[..., self.nope:].contiguous(), pos, self.theta)
        with tm.t("f q: wk_b bmm"):
            if PF_ATTN and T > PF_ATTN_MIN and cur.is_cuda:       # prefill rows: fp16 batched GEMM over heads (fp32 einsum is ~3x slower)
                qlat = self._bmm_th(self._w16("wk_b"), self._f16(qn))
            elif _BG.ENABLED and T <= _BG.MAXT and getattr(self, "bg_kb", None) is not None:
                qlat = self.bg_kb.mv(self._q(qn, "kb"))
            else:
                qlat = torch.einsum("hrk,thk->thr", self.wk_b, self._q(qn, "kb"))
        return self._h(qlat, "q"), self._h(qp, "q")

    _qr_n = None        # normalised q_a of this call's rows (indexer branch, HY4_PF_FUSE=1), consumed by attend_ck
    _qa_pre = None       # raw wq_a output of this call's rows, left by the indexer branch (same expression): reused instead of a second GEMM

    def _f16_ok(self, cur):
        T = cur.shape[0]
        return (pf_f16() and PF_ATTN and not PF_ATTN_FLASH and T > PF_ATTN_MIN and cur.is_cuda and getattr(self, "kpos_arange", False)
                and not self.f16_kv and not self.q8_act and os.environ.get("HY4_PREFILL_DENSE", "0") == "1" and T >= 1024
                and all(getattr(getattr(self, n), "ty", 0) == 8 for n in ("wq_a", "wq_b", "wgate", "wo")))

    def q16(self, cur, pos, qa=None):
        """prefill rows -> qc [H, T, R + rope] fp16 (saturated): [wk_b(q_nope) | rope(q_pe)] laid out per head, the layout the banded attention (ht=True) reads"""
        from . import prefill_kernels as _pk
        T, H = cur.shape[0], self.H
        tm = self.tm
        with tm.t("f q: wq_a mm"):
            if qa is None:
                qa = cur @ self.wq_a.T
        with tm.t("f q: norm"):
            qr = rms(qa, self.eps) * self.q_a_norm
        with tm.t("f q: wq_b mm"):
            q = _DR.raw16(self.wq_b, qr, tm)
        with tm.t("f q: rope"):
            P = self.rope_dim
            rc = pos.__dict__.setdefault("_rope_tabs", {})      # the tables depend only on (positions, theta): one fp64 evaluation per step, not per layer
            if self.theta not in rc:
                th = pos.double()[:, None] * (self.theta ** (-2.0 * torch.arange(P // 2, dtype=torch.float64, device=pos.device) / P))[None, :]
                rc[self.theta] = (th.cos().float().contiguous(), th.sin().float().contiguous())
            xh, qc = _pk.qprep(q, rc[self.theta][0], rc[self.theta][1], H, self.hd, self.nope, P, self.R)
        with tm.t("f q: wk_b bmm"):
            torch.bmm(xh, self._w16("wk_b").transpose(1, 2), out=qc[..., :self.R])
        return qc

    def attend16(self, cur, pos, kcache, allow, qa):
        from . import prefill_attn as _pa, prefill_kernels as _pk
        T, H = cur.shape[0], self.H
        tm = self.tm
        with tm.t("attn proj"):
            qcat = self.q16(cur, pos, qa)
        with tm.t("MLA core + sinks"):
            o = _pa.mla_banded(self.R, None, kcache, kcache.shape[0] - T, self.scale, self.sinks, self.window, allow, qcat=qcat, o16=True, ht=True)      # [H, T, R] fp16
        with tm.t("attn proj"):
            with tm.t("f attend: wv_b bmm"):
                kv = torch.bmm(o, self._w16("wv_b").transpose(1, 2))      # [H, T, vd] fp16
            with tm.t("f gate: wgate mm + sigmoid"):
                gated = _pk.gate16(kv, _DR.raw16(self.wgate, cur, tm))
            with tm.t("f gate: wo mm"):
                out = _DR.full(self.wo, gated, tm)
            with tm.t("f gate: post norm"):
                out = rms(out, self.eps) * self.post_norm if self.post_norm is not None else out
        return None, gated, out

    def _ck_ok(self, cur):
        """un-absorbed CK path (HY4_PF_CK=1): prefill rows >= 1024, fp16 chain, hd == vd == 256 (nope + rope), R == 512, libckmla loadable"""
        if os.environ.get("HY4_PF_CK", "0") != "1" or not self._f16_ok(cur) or cur.shape[0] < int(os.environ.get("HY4_PF_CK_MINT", "1024")):
            return False
        from . import ck_mla
        return self.hd == 256 and self.vd == 256 and self.R == 512 and self.rope_dim == 64 and ck_mla.available()

    def _kvw(self):
        w = self.__dict__.get("_kvw_c")
        if w is None:
            from . import prefill_unabs as _pu
            w = self._kvw_c = _pu.kv_weight(self._w16("wk_b"), self._w16("wv_b"))
        return w

    def attend_ck(self, cur, pos, kcache, allow, qa):
        """prefill attention in the un-absorbed form: K / V decompressed from the latent cache by one GEMM per call, CK fmha (hd 256 / 256) with the
        sink, the sliding window and the indexer mask (fp16 elementwise bias) inside; None when CK declines (caller keeps attend16)"""
        from . import prefill_kernels as _pk, prefill_unabs as _pu, ck_mla
        T, H, R, P = cur.shape[0], self.H, self.R, self.rope_dim
        tm = self.tm
        N = kcache.shape[0]
        with tm.t("attn proj"):
            with tm.t("f q: wq_a mm"):
                if qa is None:
                    qa = cur @ self.wq_a.T
            with tm.t("f q: norm"):
                from . import hc_fused as _hcf
                qr, self._qr_n = self._qr_n, None
                if qr is None or qr.shape[0] != qa.shape[0]:
                    qr = _hcf.rmsw(qa, self.q_a_norm, self.eps, emit16=True) if (_pk.PF_FUSE and _hcf.ENABLED) else rms(qa, self.eps) * self.q_a_norm
            with tm.t("f q: wq_b mm"):
                q = _pk.qdense_raw16(qr, self.wq_b.packed, tm)
            with tm.t("f q: rope"):
                rc = pos.__dict__.setdefault("_rope_tabs", {})
                if self.theta not in rc:
                    th = pos.double()[:, None] * (self.theta ** (-2.0 * torch.arange(P // 2, dtype=torch.float64, device=pos.device) / P))[None, :]
                    rc[self.theta] = (th.cos().float().contiguous(), th.sin().float().contiguous())
                _pu.qrope_inplace(q, rc[self.theta][0], rc[self.theta][1], H, self.hd, self.hd - P, P)
            with tm.t("f ck: kv decompress"):
                klo = max(0, N - T - self.window + 1) if self.window else 0
                kc = kcache[klo:]
                lat16 = kc[:, :R].clamp(-65504.0, 65504.0).half().contiguous()
                pe16 = kc[:, R:].clamp(-65504.0, 65504.0).half().contiguous()
                kk, vv = _pu.kv_decompress(lat16, pe16, self._kvw(), H)
            bias = getattr(allow, "_ck_bias", None) if klo == 0 else None
            if allow is not None and bias is None:
                with tm.t("f ck: allow bias"):
                    Nk = N - klo
                    bias = torch.empty(T, (Nk + 7) // 8 * 8, dtype=torch.float16, device=cur.device)
                    bias[:, :Nk] = torch.where(allow[:, klo:], 0.0, -30000.0).half()
        with tm.t("MLA core + sinks"):
            o = ck_mla.fmha(q.view(T, H, self.hd), kk, vv, self.scale, self.sinks, bias, self.window)
        if o is None:
            return None
        with tm.t("attn proj"):
            with tm.t("f gate: wgate mm + sigmoid"):
                g = _pk.qdense_raw16(cur, self.wgate.packed, tm)
                gated = _pk.gate16t(o.view(T, H * self.vd), g)
            with tm.t("f gate: wo mm"):
                out = _pk.qdense_tm(gated, self.wo.packed, tm)
            with tm.t("f gate: post norm"):
                if self.post_norm is not None:
                    from . import hc_fused as _hf
                    out = _hf.rmsw(out, self.post_norm, self.eps) if _hf.ENABLED else rms(out, self.eps) * self.post_norm
        return None, gated, out

    def attend(self, cur, pos, kcache, kpos, allow=None):
        """cur [T, n_embd] at positions pos; kcache [N, R+rope] at kpos (includes the T new rows). -> (kqv, gated, out)"""
        T, H = cur.shape[0], self.H
        tm = self.tm
        qa, self._qa_pre = self._qa_pre, None
        if self._ck_ok(cur):
            r = self.attend_ck(cur, pos, kcache, allow, qa)
            if r is not None:
                return r
        if self._f16_ok(cur):
            return self.attend16(cur, pos, kcache, allow, qa)
        with tm.t("attn proj"):
            qlat, qpe = self.q(cur, pos, qa)
        with tm.t("MLA core + sinks"):
            if PF_ATTN and T > PF_ATTN_MIN and qlat.is_cuda and getattr(self, "kpos_arange", False):
                from . import prefill_attn as _pa     # prefill rows: flash-style kernel, keys visited only inside the window / causal range
                o = (_pa.mla_flash if PF_ATTN_FLASH else _pa.mla_banded)(qlat, qpe, kcache, kcache.shape[0] - T, self.scale, self.sinks, self.window, allow)
            elif allow.__class__.__name__ == "SparseSel":       # HY4_LC_SPARSE: decode rows, exact selected groups + tail only
                from . import lc_sparse as _LS
                o = _LS.sparse_mla_attend(qlat, qpe, kcache[:, :self.R], kcache[:, self.R:], pos, allow, self.scale, self.sinks)
            elif _rf.ENABLED:       # HY4_REF_FUSED=1: one online-softmax kernel (sinks / window / allow inside)
                o = _rf.mla_sink_attend(qlat, qpe, kcache[:, :self.R], kcache[:, self.R:], pos, kpos, self.scale, self.sinks, self.window, allow,
                                        kpos_is_arange=getattr(self, "kpos_arange", False), ring=getattr(self, "ring_call", 0))
            else:
                o = self.S.mla_sink_attend(qlat, qpe, kcache[:, :self.R], kcache[:, self.R:], pos, kpos, self.scale, self.sinks, self.window, allow)
        with tm.t("attn proj"):
            with tm.t("f attend: wv_b bmm"):
                if PF_ATTN and T > PF_ATTN_MIN and cur.is_cuda:
                    kqv = self._bmm_th(self._w16("wv_b"), self._f16(o)).reshape(T, H * self.vd)
                elif _BG.ENABLED and T <= _BG.MAXT and getattr(self, "bg_vb", None) is not None:
                    kqv = self.bg_vb.mv(self._q(o, "vb")).reshape(T, H * self.vd)
                else:
                    kqv = torch.einsum("hdr,thr->thd", self.wv_b, self._q(o, "vb")).reshape(T, H * self.vd)
            gated, out = self.gate_out(cur, kqv)
        return kqv, gated, out

    def gate_out(self, cur, kqv):
        """isolated tail (gate, then wo, then post norm): oracle attn_kqv + cur -> (gated, out)"""
        tm = self.tm
        with tm.t("f gate: wgate mm + sigmoid"):
            gated = self.S.sigmoid_gate(kqv, self._q(cur, "wgate") @ self.wgate.T)
        with tm.t("f gate: wo mm"):
            out = self._q(gated, "wo") @ self.wo.T
        with tm.t("f gate: post norm"):
            if _ae.RMS and self.post_norm is not None:      # HY4_ATTN_EAGER=2
                return gated, _ae.rmsw(out, self.post_norm, self.eps)
            return gated, (rms(out, self.eps) * self.post_norm if self.post_norm is not None else out)


POST_NORM = "post_attention_norm.weight"


class Hy4HCPre:
    """hc_pre (M:5383-5418) + reduce + attn_norm: stream [T, hc, n_embd] -> normed collapsed cur [T, n_embd] and pre weights [T, hc]"""

    def __init__(self, g, layer: int):
        m = g.meta
        p = m["general.architecture"] + "."
        b = f"blk.{layer}."
        self.hc, self.eps = m[p + "hyper_connection.count"], float(m[p + "attention.layer_norm_rms_epsilon"])
        self.hc_eps = float(m[p + "hyper_connection.epsilon"])
        self.fn, self.base, self.scale = load_dense(g, b + "hc_attn_fn.weight"), load_dense(g, b + "hc_attn_base.weight"), load_dense(g, b + "hc_attn_scale.weight")
        self.norm = load_dense(g, b + "attn_norm.weight")

    def __call__(self, x):
        T = x.shape[0]
        mixes = rms(x.reshape(T, -1), self.eps) @ self.fn.T
        pre = torch.sigmoid(mixes[:, :self.hc] * self.scale[0] + self.base[:self.hc]) + self.hc_eps
        cur = (x * pre[:, :, None]).sum(1)
        return rms(cur, self.eps) * self.norm, pre
