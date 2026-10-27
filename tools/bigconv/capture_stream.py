"""Memory-guarded streamed capture: peak RAM independent of the number of calibration tokens.

The reference capture (tools/ds41/capture.py + exllamav3.dsv41.stream_calib.StreamCalibrator) keeps in RAM, for all
N calibration tokens: the row states h [N, hc, d] fp32, the pooled MoE input X [N, d] and the output y [N, d]
(measured 116.6 kB/token on DS4.1, i.e. 245 GB at 2M tokens; the 117 GB seen on 1000 rows was a killed run).

Design (this module):
  RowStore    one np.memmap file per row for the row state h (updated in place), one torch file per row for the
              cross-layer hand-off `ho` (compressed pools, top-k, taps; small).
  pass A      per row: attention half -> x [L, d] written into the X memmap [N, d] (disk). Dense-group Hessians
              (attn inputs, ...) accumulate across rows: they are d x d accumulators, not N-sized.
  routing     router on X chunks (per token) -> idx [N, k] int64 + wts [N, k] fp32 in RAM (12 B per token per slot).
  experts     expert by expert (one expert's weights resident): its tokens are gathered from X in chunks of `chunk`
              tokens, Hessians accumulate chunk by chunk (H += x^T x is additive), weighted outputs are added into the
              Y memmap. The shared expert runs chunked over X the same way.
  pass B      per row: attention half again (hooks off; deterministic recompute instead of spilling the attention
              context to disk) -> block output from Y[row] -> new h written in place.
Peak RAM = one row + one chunk + the Hessian accumulators of one layer + one expert's weights; disk = h + X + Y.

Numerics: the order of floating point sums differs from the pooled reference only inside the Hessian accumulation
(chunked addmm) and in the router/expert GEMM batch shapes; with chunk >= the largest expert token set and a single
router chunk the result is bitwise the reference on CPU (tests/bigconv/test_capture_stream.py).
"""
from __future__ import annotations

import json
import os
import shutil
import time

import numpy as np
import torch


class RowStore:
    def __init__(self, root: str, runner):
        self.root, self.runner = root, runner
        os.makedirs(root, exist_ok=True)
        self.shape, self.lens, self.maps = {}, [], {}

    def _hp(self, i):
        return os.path.join(self.root, f"h.{i}.f32")

    def _op(self, i):
        return os.path.join(self.root, f"ho.{i}.pt")

    def create(self, i, h0: torch.Tensor):
        a = h0.detach().float().cpu().numpy()
        mm = np.memmap(self._hp(i), dtype=np.float32, mode="w+", shape=a.shape)
        mm[:] = a
        mm.flush()
        self.shape[i] = a.shape
        del mm

    def load_h(self, i) -> torch.Tensor:
        return torch.from_numpy(np.array(np.memmap(self._hp(i), dtype=np.float32, mode="r", shape=self.shape[i])))

    def save_h(self, i, h: torch.Tensor):
        mm = np.memmap(self._hp(i), dtype=np.float32, mode="r+", shape=self.shape[i])
        mm[:] = h.detach().float().cpu().numpy()
        mm.flush()
        del mm

    def load_ho(self, i) -> dict:
        p = self._op(i)
        return torch.load(p, weights_only=False) if os.path.exists(p) else self.runner.new_handoff()

    def save_ho(self, i, ho: dict):
        tmp = self._op(i) + ".tmp"
        torch.save(ho, tmp)
        os.replace(tmp, self._op(i))


class MemArray:
    """[N, d] fp32 np.memmap with torch gather / scatter-add helpers."""

    def __init__(self, path, n, d):
        self.mm = np.memmap(path, dtype=np.float32, mode="w+", shape=(n, d))
        self.path = path

    def put(self, a, b, t: torch.Tensor):
        self.mm[a:b] = t.detach().float().cpu().numpy()

    def get(self, a, b) -> torch.Tensor:
        return torch.from_numpy(np.array(self.mm[a:b]))

    def gather(self, idx: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(self.mm[idx.cpu().numpy()])

    def add_rows(self, idx: torch.Tensor, t: torch.Tensor):
        ii = idx.cpu().numpy()
        self.mm[ii] = self.mm[ii] + t.detach().float().cpu().numpy()      # idx unique within one expert

    def close(self):
        self.mm.flush()
        del self.mm
        try:
            os.remove(self.path)
        except OSError:
            pass


def routing_stats(wts, idx, n_experts, n_tokens) -> dict:
    flat = idx.reshape(-1).long()
    counts = torch.bincount(flat, minlength=n_experts)
    w2 = torch.zeros(n_experts, dtype=torch.float64, device=flat.device)
    w2.index_add_(0, flat, (wts.reshape(-1).double()) ** 2)
    return {"n_tokens": int(n_tokens), "topk": int(idx.shape[-1]), "counts": [int(c) for c in counts.tolist()],
            "w2sum": [float(v) for v in w2.tolist()], "moe_out_energy": None}


class StreamedCapture:
    """Generic capture driver. `runner` is an arch ArchRunner (see runners/ds41.py for the interface)."""

    def __init__(self, runner, workdir: str, chunk: int = 16384, min_expert_tokens: int = 0, log=None):
        self.r, self.chunk, self.min_expert_tokens = runner, chunk, min_expert_tokens
        self.work = workdir
        self.store = RowStore(os.path.join(workdir, "rows"), runner)
        self.log = log or (lambda *_: None)
        self.rows = []
        self.on_routing = None          # callback(layer, stats dict) after routing; energy filled later (same dict)
        self.peak_chunk_tokens = 0

    # ------------------------------------------------------------------ rows
    def start(self, rows: list):
        self.rows = [r.clone() for r in rows]
        self.lens = [int(r.shape[1]) for r in self.rows]
        self.off = np.concatenate([[0], np.cumsum(self.lens)]).tolist()
        self.N = self.off[-1]
        for i, r in enumerate(self.rows):
            self.store.create(i, self.r.embed_row(r))
            self.store.save_ho(i, self.r.new_handoff())

    # ------------------------------------------------------------------ one trunk layer
    def run_layer(self, l: int, W, sink=None, commit: bool = True):
        r, N, d = self.r, self.N, self.r.hidden
        ls = r.layer_begin(l, W, capture=sink is not None)
        X = MemArray(os.path.join(self.work, "X.f32"), N, d)
        Y = MemArray(os.path.join(self.work, "Y.f32"), N, d)
        try:
            # ---- pass A: attention half of every row -> X
            for i in range(len(self.rows)):
                h, ho = self.store.load_h(i), self.store.load_ho(i)
                x, _ctx = r.attn_half(ls, h, ho, self.rows[i])
                X.put(self.off[i], self.off[i + 1], x.reshape(-1, d))
            lay = r.dense_layout(ls)
            shared = r.shared_groups
            if sink is not None:
                self._emit_groups(sink, ls, lay, [k for k in lay if k not in shared])
            # ---- routing (per token, chunked)
            E, k = r.n_exp_of(ls), r.topk_of(ls)
            idx = torch.empty(N, k, dtype=torch.int64)
            wts = torch.empty(N, k, dtype=torch.float32)
            for a in range(0, N, self.chunk):
                b = min(N, a + self.chunk)
                w_, i_ = r.route(ls, X.get(a, b))
                wts[a:b], idx[a:b] = w_.float().cpu(), i_.cpu()
            stats = routing_stats(wts, idx, E, N)
            if self.on_routing is not None:
                self.on_routing(l, stats)
            counts = torch.bincount(idx.flatten(), minlength=E)
            order = torch.argsort(idx.flatten(), stable=True)
            tok, wt = order // k, wts.flatten()[order]
            off_e = [0] + counts.cumsum(0).tolist()
            deferred = []
            # ---- experts, one at a time
            for e in range(E):
                a, b = off_e[e], off_e[e + 1]
                ti, tw = tok[a:b], wt[a:b]
                if ti.numel():
                    w = r.expert_weights(ls, e)
                    for c in range(0, ti.numel(), self.chunk):
                        tic, twc = ti[c:c + self.chunk], tw[c:c + self.chunk]
                        self.peak_chunk_tokens = max(self.peak_chunk_tokens, int(tic.numel()))
                        out = r.expert_apply(ls, X.gather(tic).to(r.device), w, e, sink is not None, twc[:, None].to(r.device))
                        Y.add_rows(tic, out.float())
                    del w
                if sink is not None:
                    self._on_expert(sink, ls, e, int(counts[e]), deferred)
            # ---- shared expert over X chunks
            en, nrm = 0.0, 0
            for a in range(0, N, self.chunk):
                b = min(N, a + self.chunk)
                xs = X.get(a, b).to(r.device)
                ys = Y.get(a, b).to(r.device) + r.shared_apply(ls, xs).float()
                Y.put(a, b, ys)
                n_ = torch.linalg.vector_norm(ys.float(), dim=-1).double()
                en += float((n_ * n_).sum())
            stats["moe_out_energy"] = en / N
            if self.on_routing is not None:
                self.on_routing(l, stats)
            if sink is not None:
                self._flush_deferred(sink, ls, deferred)
                self._emit_groups(sink, ls, lay, list(shared))
            # ---- pass B: block output per row
            ls.hs.enabled = False
            for i in range(len(self.rows)):
                h, ho = self.store.load_h(i), self.store.load_ho(i)
                _x, ctx = r.attn_half(ls, h, ho, self.rows[i])
                y = Y.get(self.off[i], self.off[i + 1]).reshape(1, self.lens[i], d).to(r.device)
                out = r.post(ls, ctx, y, ho)
                if commit:
                    self.store.save_h(i, out)
                self.store.save_ho(i, ho)
        finally:
            r.layer_end(ls)
            X.close()
            Y.close()

    # ------------------------------------------------------------------ sink plumbing (same order as StreamCalibrator)
    @staticmethod
    def _emit_groups(sink, ls, layout, groups):
        for grp in groups:
            H = ls.hs.data.pop(grp, None)
            sink.emit(grp, layout[grp], H)

    def _on_expert(self, sink, ls, e, count, deferred):
        hs = ls.hs
        lin_in, lin_mid = [f"ffn.experts.{e}.w1", f"ffn.experts.{e}.w3"], [f"ffn.experts.{e}.w2"]
        Hin = hs.data.pop(f"ffn.experts.{e}.in", None)
        if self.min_expert_tokens and count < self.min_expert_tokens:
            deferred.append((e, lin_in, Hin, count))
        else:
            sink.emit(f"ffn.experts.{e}.in", lin_in, Hin)
        sink.emit(f"ffn.experts.{e}.mid", lin_mid, hs.data.pop(f"ffn.experts.{e}.mid", None))

    def _flush_deferred(self, sink, ls, deferred):
        if not deferred:
            return
        Hs = ls.hs.data["ffn.shared_in"]
        for e, lins, H, c in deferred:
            blend = dict(Hs)
            blend["H"] = Hs["H"] * ((self.min_expert_tokens - c) / Hs["count"])
            blend["inf_nan"] = Hs["inf_nan"].clone()
            blend["first_key"] = f"ffn.experts.{e}.in"
            if H is not None:
                blend["H"] = blend["H"] + H["H"]
            blend["count"] = self.min_expert_tokens
            sink.emit(f"ffn.experts.{e}.in", lins, blend)

    # ------------------------------------------------------------------ head
    def head_hessian(self, norm_w: torch.Tensor):
        """Hessian of the output head over all rows, rows streamed from the store."""
        hd = self.r.head_h_new()
        for i in range(len(self.rows)):
            h, ho = self.store.load_h(i), self.store.load_ho(i)
            self.r.head_h_add(hd, h, ho, norm_w)
        return hd

    def close(self):
        shutil.rmtree(self.store.root, ignore_errors=True)
