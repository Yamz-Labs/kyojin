"""HY4_SPEC2_GATE: the draft chain split in two graphs. Base = the first kb steps (the usual MTPGraph, k = kb) and it also leaves its last token / state in static buffers;
tail = `ext` more steps that continue from those buffers. The host reads the base output (tokens + confidences, it needs them for the verify anyway) and launches the tail only
when DepthGate says the extra draft step + verify row pay (tools/hy4/depth_gate.py). Head KV rows are written by absolute position (pos + i), so a skipped tail leaves nothing
to undo: the rows past the accepted ones are overwritten by the next round exactly as with a shorter chain."""
import torch

from . import graph_bucket as _gb
from .mtp_head import MTPGraph, CONF, CONF_ONE


def _conf(lg):
    l32 = lg.float()
    return (torch.exp(l32.max() - torch.logsumexp(l32, 0)) * CONF_ONE).long().reshape(1)


class MTPGraphGate(MTPGraph):
    def __init__(self, mt, kb, ext=1):
        assert CONF, "MTPGraphGate needs HY4_DRAFT_CONF=1 (the engine sets it before the import)"
        super().__init__(mt, kb)
        self.ext = ext
        self.st_tok = torch.zeros(1, dtype=torch.long, device=mt.dev)
        self.st_h = torch.zeros(1, mt.fw.D, device=mt.dev)
        self.tails = {}

    def body(self, N):
        mt = self.mt
        outs, confs, tok, h, pos = [], [], self.inp[0:1], self.h0, self.inp[1:2]
        for i in range(self.k):
            y, lg = mt.step(tok, h, pos + i if i else pos, N)
            tok = mt.pick(lg)
            outs.append(tok)
            confs.append(_conf(lg))
            h = y
        self.st_tok.copy_(tok)
        self.st_h.copy_(h)
        return torch.cat(outs + confs)

    def tail_body(self, N):
        mt = self.mt
        outs, confs, tok, h, pos = [], [], self.st_tok, self.st_h, self.inp[1:2]
        for i in range(self.ext):
            y, lg = mt.step(tok, h, pos + (self.k + i), N)
            tok = mt.pick(lg)
            outs.append(tok)
            confs.append(_conf(lg))
            h = y
        return torch.cat(outs + confs)

    def _tail_graph(self, N):
        g = self.tails.get(N)
        if g is not None:
            return g
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self.tail_body(N)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        cg = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cg, pool=self.pool):
            out = self.tail_body(N)
        g = self.tails[N] = (cg, out)
        return g

    def precapture(self, ns=None):
        built = super().precapture(ns)
        mt = self.mt
        for N in (sorted(set(ns)) if ns is not None else _gb.buckets(mt.cap, self.BUCKET)):
            if N not in self.tails:
                self._tail_graph(N)
                built += 1
        return built

    def run_tail(self, j):
        """continue the base chain of the round whose head position is j (the base graph replayed last): device [ext tokens + ext confidences]"""
        mt = self.mt
        N = _gb.bucket(j + self.k + self.ext + 1, mt.cap, self.BUCKET)
        cg, out = self._tail_graph(N)
        cg.replay()
        return out
