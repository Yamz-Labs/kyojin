"""Glue: wires the Strix op kernels of Inkling 2 Mini into the EAGER decode step of Inkling2Forward (no graph yet).

  INK_GLUE=router,head,experts,gkv,lattn,attn   (or `all` = everything but the attn Q8_0 / NVFP4 modes)   INK_GLUE_PROF=1: per-class GPU event times
  fw.enable_glue(spec) does the same from code. Every kernel takes T = 1..8 rows, so the forward runs chunks of <= 8 tokens when glue is on.

Stages (one merge + wiring commit each, cumulative; each keeps a fires counter + a `torch_*` counter of the old path in fw.kstats):
  router   ink2_routerhead.router: sigmoid + bias + top-8 + renorm x 8.0 (+ expert union plan) in ONE launch instead of ~8 torch ops
  head     ink2_routerhead.head_q8 / head_bf16: resident raw output.weight, softcap fused, no dense fp32 copy
  experts  ink2_experts.moe_nvfp4: plan + gate_up + down = 3 launches per application over the UNION of the picked experts, resident raw bytes per physical layer
  gkv      8 global slots: fixed slot-indexed fp16 cache, in place append, WMMA split-K attention + sink + gate (gkv.cu)
  lattn    40 local slots: bf16 ring (R = window + 64), in place append, Triton split-K attention + sink + gate (lattn_kernel.py)
  attn     fused attention projections (ink2_attnproj, BF16 mode on, Q8_0 / NVFP4 modes behind INK_GLUE_ATTNQ)
On a CPU device (or INK_GLUE_EMU=1) every stage runs a torch emulation with the same buffers / call order, so the wiring is testable on the tiny twin GGUF."""
import collections
import contextlib
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
H = os.environ.get("INK_H", os.path.expanduser("~"))
STAGES = ("router", "head", "experts", "gkv", "lattn", "attn")
T_Q8_0, T_BF16 = 8, 30
FUSE_ALL = ("conv", "control", "mix")        # G2a glue fusions (INK_GLUE_FUSE=conv,control,mix | all): ink2_glue_conv / ink2_glue_ctrl / ink2_glue_mix
NBMAX = 9                                    # block states kept for the depth mixes (physical layers 0..23, block size 6: <= 5 per pass)
HEAD_CFG = {1: 11, 2: 10, 4: 9, 8: 5}          # best head_q8 cfg per T (ops_routerhead.md); cfg 0 elsewhere


def parse(spec):
    spec = (spec or "").replace(" ", "")
    if spec in ("all", "1"):
        return set(STAGES)
    on = {s for s in spec.split(",") if s}
    bad = on - set(STAGES)
    assert not bad, f"unknown glue stage {sorted(bad)}; known {STAGES}"
    return on


class Glue:
    STAGES_ALL = STAGES

    def __init__(self, fw, spec, emu=None, so_dir=None, fuse=None):
        self.fw = fw
        self.on = parse(spec)
        self.F = sys.modules[type(fw).__module__]                       # the forward module (row_bytes, _u8 ...)
        self.emu = (fw.dev.type != "cuda") if emu is None else emu
        if os.environ.get("INK_GLUE_EMU"):
            self.emu = True
        self.so_dir = so_dir or os.environ.get("INK_GLUE_SO") or os.path.join(HERE, "build")
        self.fires = collections.Counter()
        self.prof = bool(os.environ.get("INK_GLUE_PROF")) and fw.dev.type == "cuda"
        self._ev = []
        self._rh = self._ex = None
        self._ew = {}                                                   # physical layer -> (gu, dn, s_gu, s_dn, q8): resident raw expert bytes
        self._hw = None
        self._pos0 = torch.zeros(1, dtype=torch.int32, device=fw.dev)        # absolute position of the chunk's first row (device int32): created once, graphs hold its address
        self._ainit = False
        fz = (os.environ.get("INK_GLUE_FUSE", "") if fuse is None else fuse).replace(" ", "")
        self.fuse = set(FUSE_ALL) if fz in ("all", "1") else {x for x in fz.split(",") if x}
        assert self.fuse <= set(FUSE_ALL), f"unknown fusion {sorted(self.fuse - set(FUSE_ALL))}; known {FUSE_ALL}"
        self._cw, self._nw, self._qm, self._cscr = {}, {}, {}, None
        self.cf = self.bank = self.mbank = None
        self._start = 0
        self._pj = {}
        self._apk = None
        self._aps = {}
        self.fuse_init()

    def has(self, stage):
        return stage in self.on

    # ------------------------------------------------------------------ G2a fusions (conv / control / mix) wired into the step
    def fuse_init(self):
        """buffers of the fusions, created once (graphs keep their addresses): conv rings by position (replace slot.hist), block banks [NBMAX, 8, D] for the depth mixes"""
        if self.cf is not None or self.bank is not None:
            return
        fw, dev = self.fw, self.fw.dev
        if "conv" in self.fuse:
            import ink2_glue_conv as CV
            self.CV = CV
            self.cf = CV.ConvFuse(fw.n_app, fw.n_chan, dev, self._pos0, fw.qk_clip, fw.eps)
        if "control" in self.fuse:
            import ink2_glue_ctrl as CT
            self.CT = CT
            self._cscr = torch.zeros(8, 64, device=dev)
        if "mix" in self.fuse:
            import ink2_glue_mix as MX
            self.MX = MX
            self.bank = torch.zeros(NBMAX, 8, fw.D, device=dev)             # the blocks of the current pass (written in place, never a python list)
            self.mbank = torch.zeros(NBMAX, 8, fw.D, device=dev)            # the previous pass's blocks (messages of the loop passes)

    def conv_kv(self, il, off, x, w, dyn, kind):
        """short conv + l2 clip (K, kind 'k') / rms (V, 'v') per kv head on the ring of application il -> fp32 [T, C]; one launch (torch conv_ref + epilogue on CPU)"""
        fw, cf, CV = self.fw, self.cf, self.CV
        T, C = x.shape
        self.fires["conv"] += 1
        ring = cf.rings[il][:, off:off + C]
        if self.emu:                                                    # same ring semantics, torch (the twin heads are not 128 wide: the epilogue runs per head of fw.hd)
            y = CV.conv_ref(x, ring, w, self._pos0, dyn=dyn).reshape(T, fw.n_kv, fw.hd)
            y = self.F.l2_clip(y, None, fw.qk_clip) if kind == "k" else self.F.rms(y, fw.eps)
            return y.reshape(T, C)
        out = torch.empty(T, C, dtype=torch.float32, device=x.device)
        with self.span("conv"):
            return CV.conv_fused(x, ring, w, self._pos0, out, dyn=dyn, epi=CV.EPI_L2CLIP if kind == "k" else CV.EPI_RMS, thr=fw.qk_clip, eps=fw.eps)

    def conv_post(self, il, off, x, w, res):
        """post-attn / post-ffn conv (+ the residual add when res is given) -> fp32 [T, D]"""
        cf, CV = self.cf, self.CV
        T, C = x.shape
        self.fires["conv"] += 1
        ring = cf.rings[il][:, off:off + C]
        if self.emu:
            y = CV.conv_ref(x, ring, w, self._pos0)
            return y if res is None else res + y
        out = torch.empty(T, C, dtype=torch.float32, device=x.device)
        with self.span("conv"):
            return CV.conv_fused(x, ring, w, self._pos0, out, res=res, epi=CV.EPI_ADD if res is not None else CV.EPI_NONE)

    def _ctrl_w(self, L, nm):
        k = (L, nm)
        if k not in self._cw:
            dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[os.environ.get("INK_GLUE_CTRLW", "bf16")]
            self._cw[k] = tuple(self.CT.to_ctrl_dtype(self.fw.W(f"{L}.loop_{nm}_{p}.weight"), dt) for p in ("down", "gain", "forget"))
        return self._cw[k]

    def control(self, cur, msg, L, m, nw=None):
        """loop controller (1 - tanh(F h)) cur + tanh(G h) msg, followed by rms * nw when nw is given (2 + 1 launches instead of 20 + 6)"""
        fw = self.fw
        self.fires["control"] += 1
        if self.emu:
            y = fw.control(cur, msg, L, m)
            return y if nw is None else self.F.rms(y, fw.eps) * nw
        nm = ("attn", "k", "v", "mlp")[m]
        wd, wg, wf = self._ctrl_w(L, nm)
        out = torch.empty_like(cur)
        with self.span("control"):
            self.CT.control_fused(cur, msg, wd, wg, wf, out, scratch=self._cscr)
            if nw is None:
                return out
            out2 = torch.empty_like(cur)
            return self.CT.rms_w(out, nw, out2, fw.eps)

    def rmsw(self, x, nw):
        """rms(x) * nw in one launch"""
        if self.emu:
            return self.F.rms(x, self.fw.eps) * nw
        self.fires["control"] += 1
        with self.span("control"):
            return self.CT.rms_w(x, nw, torch.empty_like(x), self.fw.eps)

    def norm_stack(self, L):
        """attn_{q,k,v}_inp_norm of a global layer as [3, D] (one norm weight per mix query)"""
        v = self._nw.get(L)
        if v is None:
            v = self._nw[L] = torch.stack([self.fw.W(f"{L}.attn_{c}_inp_norm.weight") for c in "qkv"]).contiguous()
        return v

    def msg_queries(self, row):
        """the 4 message queries of a core layer: loop_res_query rows (attn, mlp) then loop_kv_res_query rows (k, v) -> [4, D]"""
        v = self._qm.get(row)
        if v is None:
            fw = self.fw
            v = self._qm[row] = torch.cat([fw.W("loop_res_query.weight")[row:row + 2], fw.W("loop_kv_res_query.weight")[row:row + 2]], 0).contiguous()
        return v

    def mix_f(self, buf, nb, cur, query, nw=None):
        """depth mix over the nb block rows of `buf` [NBMAX, 8, D] plus the current state `cur` [T, D], queries [Q, D] -> [Q, T, D]; nw (None | [D] | [Q, D]) = fused input norm rms(.) * nw"""
        fw = self.fw
        T = cur.shape[0]
        self.fires["mix"] += 1
        if self.emu:
            o = fw.mix([buf[i, :T] for i in range(nb)], cur, query)
            return o if nw is None else self.F.rms(o, fw.eps) * (nw if nw.dim() == 1 else nw[:, None, :])
        out = torch.empty(query.shape[0], T, fw.D, dtype=torch.float32, device=cur.device)
        with self.span("mix"):
            return self.MX.mix_fused(buf, nb, cur, query, out, nw=nw, eps=self.F.EPS_MIX, eps_n=fw.eps)

    # ------------------------------------------------------------------ profiling (GPU events per kernel class)
    @contextlib.contextmanager
    def span(self, name):
        if not self.prof:
            yield
            return
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        yield
        b.record()
        self._ev.append((name, a, b))

    def prof_report(self, reset=True):
        """{class: [ms, calls]} since the last report (syncs)"""
        if not self._ev:
            return {}
        torch.cuda.synchronize()
        out = {}
        for n, a, b in self._ev:
            r = out.setdefault(n, [0.0, 0])
            r[0] += a.elapsed_time(b)
            r[1] += 1
        if reset:
            self._ev = []
        return out

    # ------------------------------------------------------------------ libraries
    def rh(self):
        if self._rh is None:
            import ink2_routerhead
            self._rh = ink2_routerhead.Ink2RouterHead(os.environ.get("INK2_RH_SO") or f"{self.so_dir}/libink2_rh.so")
        return self._rh

    def ex(self):
        if self._ex is None:
            import ink2_experts
            self._ex = ink2_experts.Ink2Experts(os.environ.get("INK2_SO") or f"{self.so_dir}/libink2_experts.so")
        return self._ex

    # ------------------------------------------------------------------ router
    def router(self, L, x):
        """x [T, D] fp32 -> sel [T, 8] int32 (selection order), w [T, 8] fp32 (renormalised, x route scale)"""
        fw = self.fw
        R = fw.W(f"{L}.ffn_gate_inp.weight")
        bias = fw.W(f"{L}.exp_probs_b.bias") if fw.has(f"{L}.exp_probs_b.bias") else torch.zeros(fw.n_exp, device=fw.dev)
        self.fires["router"] += 1
        with self.span("router"):
            if self.emu:
                scores = torch.sigmoid(x @ R.T)
                sel = torch.topk(scores + bias, fw.topk, dim=-1, sorted=False).indices
                w = scores.gather(1, sel)
                return sel.to(torch.int32), w / (w.sum(-1, keepdim=True) + 1e-20) * fw.w_scale
            sel, wt, _plan = self.rh().router(x.contiguous(), R, bias, fw.w_scale)
        return sel, wt

    # ------------------------------------------------------------------ LM head
    def head(self, h):
        """h [T, D] fp32 -> logits [T, vocab] fp32 (softcapped) from the resident raw output.weight, or None (fall back to torch)"""
        fw = self.fw
        if self.emu or not fw.softcap:
            return None
        ty, dims, off = fw.g.infos["output.weight"]
        K = dims[0]
        if ty not in (T_Q8_0, T_BF16) or K != fw.D:
            return None
        if self._hw is None:
            rb = K // 32 * 34 if ty == T_Q8_0 else K * 2
            raw = self.F._u8(fw.g, off, fw.vocab * rb, fw.dev)
            self._hw = raw.reshape(fw.vocab, rb) if ty == T_Q8_0 else raw.view(torch.bfloat16).reshape(fw.vocab, K)
        rh = self.rh()
        fn = rh.head_q8 if ty == T_Q8_0 else rh.head_bf16
        hc = h.float().contiguous()
        outs = []
        for s in range(0, hc.shape[0], 8):
            xs = hc[s:s + 8]
            cfg = HEAD_CFG.get(xs.shape[0], 0) if ty == T_Q8_0 else 0
            self.fires["head"] += 1
            with self.span("head"):
                try:
                    lg = fn(xs, self._hw, fw.vocab, softcap=fw.softcap, cfg=cfg)[0]
                except NotImplementedError:
                    lg = fn(xs, self._hw, fw.vocab, softcap=fw.softcap, cfg=0)[0]
            outs.append(lg)
        return outs[0] if len(outs) == 1 else torch.cat(outs, 0)

    # ------------------------------------------------------------------ routed experts
    def _expert_bytes(self, L):
        """resident raw GGUF bytes of one physical layer's experts (gu [E, N, rowbytes], dn, per-expert scales), loaded once"""
        ent = self._ew.get(L)
        if ent is None:
            fw, F = self.fw, self.F
            gu_n, dn_n = f"{L}.ffn_gate_up_exps.weight", f"{L}.ffn_down_exps.weight"
            parts = []
            for n in (gu_n, dn_n):
                ty, dims, off = fw.g.infos[n]
                K, N, E = dims[0], dims[1], dims[2]
                rb = F.row_bytes(ty, K)
                parts.append((F._u8(fw.g, off, E * N * rb, fw.dev).reshape(E, N, rb), ty))
            (gu, tg), (dn, td) = parts
            assert tg == td and tg in (T_Q8_0, 40), (L, tg, td)
            s = []
            for n in (gu_n, dn_n):
                sn = n[:-6] + "scale"
                s.append(fw.W(sn).float().contiguous() if fw.has(sn) else None)
            ent = self._ew[L] = (gu, dn, s[0], s[1], tg == T_Q8_0)
        return ent

    def preload(self):
        if self.emu:
            return
        for il in range(self.fw.n_phys):
            self._expert_bytes(f"blk.{il}")

    def experts(self, L, x, sel, w):
        fw = self.fw
        self.fires["experts"] += 1
        if self.emu:
            return fw._moe_loop(L, x, sel, w)
        gu, dn, s_gu, s_dn, q8 = self._expert_bytes(L)
        sel = sel.to(torch.int32).contiguous()
        with self.span("experts"):
            return self.ex().moe_nvfp4(x.float().contiguous(), sel, w.float().contiguous(), gu, dn, s_gu, s_dn, fw.beta, fw.lbeta, q8=q8)

    # ------------------------------------------------------------------ attention caches (global: gkv, local: lattn)
    SMAX, CMIN = 32, 256                     # gkv split policy (ops_gkv.md)

    def reset(self):
        self._start = 0
        self._pos0.zero_()
        if self.cf is not None:
            self.cf.reset()
        if os.environ.get("INK_GLUE_ZERO_RESET"):
            self.zero_scratch()

    def zero_scratch(self, only=None):
        """diagnostic (INK_GLUE_ZERO_RESET=1 [, INK_GLUE_ZERO_ONLY=a,b]): zero every scratch / cache buffer of the glue so a stale-state read after reset() shows up as a difference"""
        want = (lambda n: only is None or n in only)
        t = []
        for n in ("K", "V", "KR", "VR", "gws", "bank", "mbank", "_cscr"):
            b = getattr(self, n, None)
            if b is not None and want(n):
                b.zero_()
                t.append(n)
        if want("la") and getattr(self, "la", None) is not None:
            for w in self.la.ws.values():
                for k in ("acc", "ml", "out"):
                    w[k].zero_()
            t.append("la")
        if want("ap"):
            for r in self._aps.values():
                for sg in (r[0] + [r[1]]) if r else []:
                    if sg.buf is not None:
                        sg.buf.zero_()
            t.append("ap")
        if want("ex") and self._ex is not None:
            for pl, hs in self._ex._plans.values():
                pl.zero_()
                hs.zero_()
            t.append("ex")
        if want("rh") and self._rh is not None:
            for bufs in self._rh._bufs.values():
                for b in bufs:
                    b.zero_()
            t.append("rh")
        return t

    def begin(self, start, T):
        """called at the top of every chunk: the absolute position of its first row, as a device int32 (graph-safe kernels read it there)"""
        self._start = start
        self._pos0.fill_(start)

    @property
    def pos0(self):
        return self._pos0

    def _attn_init(self):
        if self._ainit:
            return
        self._ainit = True
        fw, dev = self.fw, self.fw.dev
        self.gidx = {il: i for i, il in enumerate(il for il in range(fw.n_app) if not fw.pattern[il])}
        self.lidx = {il: i for i, il in enumerate(il for il in range(fw.n_app) if fw.pattern[il])}
        nkv, hd = fw.n_kv, fw.hd
        if not self.emu:
            assert (fw.n_head, nkv, hd, fw.window, fw.rel_extent) == (24, 8, 128, 1024, 1024), "attention kernels are built for the real Inkling 2 Mini shapes"
        self.cap = (int(os.environ.get("INK_GLUE_CAP", 256 if self.emu else 16448)) + 63) // 64 * 64
        self.ring = fw.window + (8 if self.emu else 64)               # lattn kernel: R = 1088 = 1024 + 64 (rollback <= 64, T <= 8 verify rows)
        ng, nl = len(self.gidx), len(self.lidx)
        if self.has("gkv"):
            kdt = torch.float32 if self.emu else torch.float16
            self.K = torch.zeros(ng, nkv, self.cap, hd, dtype=kdt, device=dev)
            self.V = torch.zeros(ng, nkv, self.cap, hd, dtype=kdt, device=dev) if self.emu else torch.zeros(ng, nkv, self.cap // 64, hd, 64, dtype=kdt, device=dev)
            if not self.emu:
                from gkv_lib import Lib
                self.glib = Lib(os.environ.get("GKV_SO") or f"{self.so_dir}/libgkv.so")
                self.gws = torch.empty(self.glib.l.gkv_ws_floats(self.SMAX), dtype=torch.float32, device=dev)
        if self.has("lattn"):
            rdt = torch.float32 if self.emu else torch.bfloat16
            self.KR = torch.zeros(nl, self.ring, nkv, hd, dtype=rdt, device=dev)
            self.VR = torch.zeros(nl, self.ring, nkv, hd, dtype=rdt, device=dev)
            if not self.emu:
                from lattn_kernel import LAttn
                self.la = LAttn(dev, ring_rows=self.ring, bn=32, nsplit=8)

    def _proj(self, L):
        """rel_logits_proj as [16, extent] fp32 (the gkv_rel / lattn operand), per physical layer"""
        p = self._pj.get(L)
        if p is None:
            p = self._pj[L] = self.fw.W(f"{L}.attn_rel_bias.weight").T.contiguous().float()
        return p

    def _append(self, il, swa, Kc, Vc, T):
        """append the T new rows (post conv / clip / rms, [T, nkv, hd] fp32) of application `il` at positions start .. start+T-1, in place"""
        fw, s = self.fw, self._start
        if swa:
            li = self.lidx[il]
            if self.emu:
                rows = (s + torch.arange(T, device=fw.dev)) % self.ring
                self.KR[li][rows], self.VR[li][rows] = Kc, Vc
            else:
                from lattn_kernel import lattn_append
                lattn_append(Kc.to(torch.bfloat16).contiguous(), Vc.to(torch.bfloat16).contiguous(), self.KR[li], self.VR[li], self._pos0, ring_rows=self.ring)
            return
        ci = self.gidx[il]
        assert s + T <= self.cap, f"global KV cache full: {s + T} > cap {self.cap} (INK_GLUE_CAP)"
        if self.emu:
            self.K[ci][:, s:s + T], self.V[ci][:, s:s + T] = Kc.permute(1, 0, 2), Vc.permute(1, 0, 2)
        else:
            kp, vp = self._gptr(ci)
            self.glib.append(kp, vp, None, None, Kc.half().contiguous(), Vc.half().contiguous(), self._pos0, T, self.cap, 0)

    def _gptr(self, ci):
        fw = self.fw
        off = ci * fw.n_kv * self.cap * fw.hd * 2
        return self.K.data_ptr() + off, self.V.data_ptr() + off

    def attn(self, il, L, swa, Q, R, Kc, Vc, sinks, gate):
        """Q [T, nh, hd] (clipped, x tau on global), R [T, nh, 16] (clipped, x tau), Kc / Vc [T, nkv, hd], sinks [T, nh], gate [T, nh * hd] pre-sigmoid -> [T, nh * hd] fp32 = attention out * sigmoid(gate)"""
        fw = self.fw
        self._attn_init()
        T = Q.shape[0]
        s = self._start
        self.fires["lattn" if swa else "gkv"] += 1
        with self.span("attn_local" if swa else "attn_global"):
            self._append(il, swa, Kc, Vc, T)                          # rows first: the kernels read the cache in place up to pos0 + T - 1
            if self.emu:
                pos = s + torch.arange(T, device=fw.dev)
                tab = R @ self._proj(L)
                if swa:
                    lo = max(0, s - (fw.window - 1))
                    kpos = torch.arange(lo, s + T, device=fw.dev)
                    rows = kpos % self.ring
                    kk, vv = self.KR[self.lidx[il]][rows].permute(1, 0, 2), self.VR[self.lidx[il]][rows].permute(1, 0, 2)
                else:
                    ci = self.gidx[il]
                    kpos = torch.arange(s + T, device=fw.dev)
                    kk, vv = self.K[ci][:, :s + T], self.V[ci][:, :s + T]
                o = fw._attend_core(Q, kk, vv, kpos, tab, sinks, swa, pos).reshape(T, -1)
                return o * torch.sigmoid(gate)
            if swa:
                tab = (R @ self._proj(L)).contiguous()                # [T, nh, 1024] fp32 (bias lookup table)
                li = self.lidx[il]
                o = self.la(Q.to(torch.bfloat16).contiguous(), self.KR[li], self.VR[li], self._pos0, sinks.float().contiguous(), gate.float().contiguous(), tab=tab)
                return o.float()
            ci = self.gidx[il]
            tab = torch.empty(T, fw.n_head, fw.rel_extent, dtype=torch.float32, device=fw.dev)
            self.glib.rel(R.float().contiguous(), self._proj(L), tab, T)
            kp, vp = self._gptr(ci)
            self.glib.attend(Q.half().contiguous(), kp, vp, None, None, self._pos0, tab, self.gws, T, self.SMAX, self.CMIN, self.cap, 0)
            out = torch.empty(T, fw.n_head, fw.hd, dtype=torch.float16, device=fw.dev)
            self.glib.combine(self.gws, sinks.float().contiguous(), gate.half().contiguous(), out, self._pos0, T, self.SMAX, self.CMIN)
            return out.float().reshape(T, -1)

    def fill_synthetic(self, depth, gen):
        """decode timing at depth without a prefill: random normalised K (norm 8) / random V rows for positions depth - n .. depth - 1 of every cache"""
        fw = self.fw
        if not (self.has("gkv") or self.has("lattn")):
            return
        self._attn_init()
        for il in range(fw.n_app):
            swa = fw.pattern[il]
            if not self.has("lattn" if swa else "gkv"):
                continue
            n = min(depth, fw.window) if swa else depth
            k = torch.randn(n, fw.n_kv, fw.hd, generator=gen)
            k = (k / k.norm(dim=-1, keepdim=True) * 8.0).to(fw.dev)
            v = torch.randn(n, fw.n_kv, fw.hd, generator=gen).to(fw.dev)
            self.begin(depth - n, n)
            self._append(il, swa, k, v, n)
        self.begin(depth, 0)

    # ------------------------------------------------------------------ fused attention projections (attnbytes)
    AP_MATS = (("q", 0), ("gate", 0), ("rel", 0), ("sink_proj", 0), ("k", 1), ("k_conv_proj", 1), ("v", 2), ("v_conv_proj", 2))

    def _ap(self):
        if self._apk is None:
            import ink2_attnproj
            self._AP = ink2_attnproj
            self._apk = ink2_attnproj.Ink2AttnProj(os.environ.get("INK2_AP_SO") or f"{self.so_dir}/libink2_attnproj_w8_s1.so")      # shipped config WPB 8, STAGE_FIRST 1
        return self._apk

    def _ap_planes(self, name):
        """-> (w, sc, fmt, gs, N, K) of one attention matrix as the kernel reads it, or None (type not handled).
        As stored: Q8_0 bytes split into int8 + fp16 scale planes (identical values), BF16 as is. INK_GLUE_ATTNQ=nvfp4 re-quantises the Q8_0 / BF16 matrices to NVFP4 at load
        (fewer bytes, changes numbers: quality-gated, OFF by default)."""
        fw, F = self.fw, self.F
        ty, dims, off = fw.g.infos[name]
        K, N = dims[0], dims[1]
        AP = self._AP
        if ty == T_Q8_0:
            raw = F._u8(fw.g, off, N * (K // 32) * 34, fw.dev).reshape(N, -1)
            q, d = self._AQ().split_q8_gguf(raw)
            if os.environ.get("INK_GLUE_ATTNQ") == "nvfp4" and N * K >= 50_000:
                nib, sc, gs, _ = self._AQ().quant_nvfp4(self._AQ().dq_q8(q.cpu(), d.cpu()).float())
                return nib.to(fw.dev), sc.to(fw.dev), AP.F_NV, gs, N, K
            return q, d, AP.F_Q8, 1.0, N, K
        if ty == T_BF16:
            return F._u8(fw.g, off, N * K * 2, fw.dev).view(torch.bfloat16).reshape(N, K), None, AP.F_BF, 1.0, N, K
        return None

    def _AQ(self):
        import importlib
        return importlib.import_module("exllamav3.loader.ink2_attnquant")

    def ap_layer(self, L, swa):
        """segment tables of one physical layer: launch 1 = q, gate, rel, sink_proj, k (+ k_conv_proj), v (+ v_conv_proj); launch 2 = output; None when a tensor has an unsupported type"""
        key = (L, swa)
        if key in self._aps:
            return self._aps[key]
        fw = self.fw
        self._ap()
        AP = self._AP
        segs, ok = [], True
        for n, xi in self.AP_MATS:
            nm = f"{L}.attn_{n}.weight"
            if n.endswith("conv_proj") and swa or not fw.has(nm):
                continue
            pl = self._ap_planes(nm)
            ok &= pl is not None and pl[5] in (2304, 3072)
            if pl is None:
                break
            w, sc, fmt, gs, N, K = pl
            segs.append(AP.Seg(w, sc, N, K, xi, fmt, gs, n))
        out = None
        pl = self._ap_planes(f"{L}.attn_output.weight") if ok else None
        if pl is not None and pl[5] == 3072:
            out = AP.Seg(pl[0], pl[1], pl[4], pl[5], 0, pl[2], pl[3], "output")
        res = (segs, out) if ok and out is not None else None
        self._aps[key] = res
        return res

    def ap_ok(self, L, swa):
        return self.emu or self.ap_layer(L, swa) is not None

    def proj(self, L, swa, xq, xk, xv):
        """one launch for q, gate, rel, sink_proj, k, v (+ conv_proj on global layers): fp32 [T, N] outputs (views of the segment buffers, valid until the next call on this layer)"""
        T = xq.shape[0]
        self.fires["attn"] += 1
        if self.emu:                                                  # same outputs from dense torch matmuls (wiring test)
            W, nm = self.fw.W, lambda n: f"{L}.attn_{n}.weight"
            cp = (lambda x, n: x @ W(nm(n)).T) if not swa else (lambda x, n: None)
            return (xq @ W(nm("q")).T, xk @ W(nm("k")).T, xv @ W(nm("v")).T, xq @ W(nm("rel")).T, xq @ W(nm("sink_proj")).T, xq @ W(nm("gate")).T,
                    cp(xk, "k_conv_proj"), cp(xv, "v_conv_proj"))
        segs, _ = self.ap_layer(L, swa)
        with self.span("attnproj"):
            self._ap().run(segs, [xq.contiguous(), xk.contiguous(), xv.contiguous()], T)
        o = {s.name: s.out for s in segs}
        return o["q"], o["k"], o["v"], o["rel"], o["sink_proj"], o["gate"], o.get("k_conv_proj"), o.get("v_conv_proj")

    def out_proj(self, L, swa, o):
        if self.emu:
            return o @ self.fw.W(f"{L}.attn_output.weight").T
        _, seg = self.ap_layer(L, swa)
        with self.span("attnproj"):
            self._ap().run([seg], [o.contiguous()], o.shape[0])
        return seg.out


TORCH_OF = {"router": "torch_router", "head": "torch_head", "experts": "torch_experts", "gkv": "torch_attn_global", "lattn": "torch_attn_local", "attn": "torch_attnproj"}


def fires_check(fw):
    """fires-check of a glue run: every flagged stage fired (> 0) and its old torch path never ran. Returns a list of failures (empty = pass)."""
    gl, bad = fw.gl, []
    if gl is None:
        return bad
    for st in sorted(gl.on):
        if st == "head" and gl.emu:
            continue                                                  # the head kernel has no CPU emulation: torch runs it
        if gl.fires[st] == 0:
            bad.append(f"stage {st}: 0 kernel firings")
        if fw.kstats[TORCH_OF[st]] > 0:
            bad.append(f"stage {st}: old torch path ran {fw.kstats[TORCH_OF[st]]}x")
    for f in sorted(gl.fuse):
        if gl.fires[f] == 0:
            bad.append(f"fusion {f}: 0 firings")
    if fw.graphs:                                                        # graph mode: the per-stage counters above only count warm-up + capture, so require real replays
        if fw.kstats["graph_replays"] == 0:
            bad.append("graph mode: 0 graph replays")
        if fw.kstats["graph_eager"] > 0:
            bad.append(f"graph mode: {fw.kstats['graph_eager']} steps fell back to the eager path")
    return bad


def count_launches(fw, fn):
    """kernel launches of one call of fn() (eager step): aten ops that are not views (TorchDispatchMode) + Triton kernel launches + the ctypes kernel launches of the five wired ops
    (router 1, head 1 per <= 8 rows, experts 3, attnproj 2 per application, gkv append / rel / attend / combine 4 per global application). A lower bound of the HIP launch count (some aten ops launch 2)."""
    from torch.utils._python_dispatch import TorchDispatchMode
    import triton.runtime.jit as tj
    n = {"aten": 0, "triton": 0}

    class Count(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            rets = func._schema.returns
            if not any(r.alias_info is not None and not r.alias_info.is_write for r in rets):
                n["aten"] += 1
            return func(*args, **(kwargs or {}))
    orig = tj.JITFunction.run

    def run(self, *a, **k):
        n["triton"] += 1
        return orig(self, *a, **k)
    gl = fw.gl
    f0 = dict(gl.fires)
    tj.JITFunction.run = run
    try:
        with Count():
            fn()
    finally:
        tj.JITFunction.run = orig
    d = {k: gl.fires[k] - f0.get(k, 0) for k in gl.fires}
    n["ctypes"] = d.get("router", 0) + d.get("head", 0) + 3 * d.get("experts", 0) + 2 * d.get("attn", 0) + 4 * d.get("gkv", 0)
    n["total"] = n["aten"] + n["triton"] + n["ctypes"]
    return n
