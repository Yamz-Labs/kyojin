"""
DSpark speculative decoding on the DS4.1 port: the `Target` / `Drafter` objects of the draft/verify controller
(tools/ds41/dspark.py, framework agnostic, structural Protocols) over `DSV41Model`.

  spec = PortSpec(model, max_seq)
  tokens, dec = dspark.decode(spec.target, spec.drafter, prompt, max_tokens, eos_id)

Target
  prefill(tokens)          one forward at start_pos 0 (logits of the last row, hidden of every row)
  verify(tokens, start)    ONE forward over anchor + k drafts (`DSV41Model.forward_block`): logits and target-layer
                           hidden for every position; each query sees what a decode step at its position sees
  truncate(length)         `DSV41State.rollback`: the ring-buffer rows and compressor slots the rejected tail
                           overwrote are put back from the block's undo log, nothing is recomputed. Pools, indexer
                           keys and the engram history are positional and rewritten before they are read again;
                           the hyper-connection mixes never leave a forward.
Drafter
  draft(token, hidden, p)  `hidden` = target rows of the positions the drafter has not seen yet, ending at p. They
                           seed the window rings of the three stages (`DSparkHeads.feed`: one wkv per stage), then
                           one `forward_spec` drafts block_size tokens from [anchor, noise x (block-1)].
                           The drafter is only ever fed accepted positions, so it needs no rollback.

Both share one `DSV41State` (the stage rings are layers n..n+2 of the same state); `prefill` starts a new one.
The returned objects are duck-typed to dspark.TargetOut / dspark.DraftOut (logits, hidden / tokens, q).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Sequence
import torch


@dataclass
class BlockOut:
    logits: Any
    hidden: Any


@dataclass
class DraftTokens:
    tokens: list
    q: Any = None
    conf: Any = None                    # raw confidence-head score per slot (a logit of acceptance), known before the slot's token


class PortSpec:
    def __init__(self, model, max_seq: int = 4096, max_depth: int | None = None):
        if model.mtp is None:
            raise ValueError("the model has no DSpark stages (num_nextn_predict_layers / dspark_block_size = 0)")
        self.model, self.max_seq = model, max_seq
        self.st = None
        self.length = 0
        self.spans = None                   # vision.ImageSpan list for the next prefill (set by the engine)
        self.supports_spans = True
        self.block = None                   # (start_pos, n_tokens) of the last verify
        self.user_id = self.assistant_id = None
        self.keys = None                    # cache keys of the next prefill (token ids; image positions carry the pixel hash)
        self.keys_run = []                  # keys of the prompt being decoded
        self.rows = None                    # last <= sliding_window target rows, ending at position length - 1
        self.cached, self.miss_lcp = 0, 0   # tokens the last prefill took from a snapshot / common prefix when it took none
        self.target = PortTarget(self)
        self.drafter = PortDrafter(self, max_depth)

    def reset(self):
        self.st = self.model.new_state(1, self.max_seq)
        self.length, self.block, self.rows = 0, None, None

    # -- prompt cache (exllamav3.dsv41.prefix_cache)
    def attach_prefix(self, store, user_id=None, assistant_id=None):
        self.user_id, self.assistant_id = user_id, assistant_id      # the store itself arrives per request (begin_request)

    def begin_request(self, keys, store):
        """Per request: the cache keys of the prompt and the store to use (None = no reuse, nothing stored)."""
        self.keys, self.store_now = keys, store

    store_now = None

    def store_reply(self, produced):
        """Snapshot the state after the reply. `produced`: every token the decoder handed out; the cache holds the prompt and
        the first `length - len(prompt)` of them (the last one is never fed)."""
        store = self.store_now
        if store is None or self.st is None or self.rows is None:
            return
        from . import prefix_cache
        keys = (list(self.keys_run) + list(produced))[: self.length]
        if len(keys) == self.length and self.length > len(self.keys_run):
            store.put(prefix_cache.snapshot(self.model.cfg, self.st, self.length, keys, self.rows, "reply"))


class PortTarget:
    def __init__(self, spec: PortSpec):
        self.s = spec

    @property
    def length(self) -> int:
        return self.s.length

    def prefill(self, tokens: Sequence[int]) -> BlockOut:
        s = self.s
        if len(tokens) + 1 > s.max_seq:
            raise ValueError(f"prompt of {len(tokens)} tokens does not fit max_seq {s.max_seq}")
        from . import prefix_cache
        tokens = list(tokens)
        keys = s.keys if s.keys is not None and len(s.keys) == len(tokens) else tokens
        spans = s.spans
        pf = prefix_cache.prefill(s.model, s.store_now, tokens, keys, max_seq=s.max_seq, user_id=s.user_id,
                                  assistant_id=s.assistant_id, spans_fn=(lambda cached: spans) if spans else None,
                                  reuse=s.store_now is not None)
        s.st, s.length, s.block, s.keys_run = pf.st, len(tokens), None, list(keys)
        s.keys = None
        s.cached, s.miss_lcp = pf.cached, pf.lcp
        s.rows = pf.hidden.to(s.model.device) if pf.hidden is not None else None
        s.drafter._next = len(tokens) - (s.rows.size(0) if s.rows is not None else 0)
        return BlockOut(logits=pf.logits, hidden=s.rows)

    def verify(self, tokens: Sequence[int], start_pos: int) -> BlockOut:
        s = self.s
        assert start_pos == s.length, (start_pos, s.length)
        n = len(tokens)
        if start_pos + n > s.max_seq:
            raise ValueError("verify block does not fit max_seq")
        s.st.begin_block(start_pos, n)
        ids = torch.tensor([list(tokens)], dtype=torch.int64)
        logits, hidden = s.model.forward_block(ids, start_pos, s.st)
        s.length, s.block = start_pos + n, (start_pos, n)
        if s.rows is not None:
            s.rows = torch.cat([s.rows, hidden[0]], dim=0)[-s.model.cfg.sliding_window:]
        return BlockOut(logits=logits[0], hidden=hidden[0])

    def truncate(self, length: int) -> None:
        s = self.s
        start, n = s.block
        assert start <= length <= start + n, (start, length, n)
        if length < start + n:
            s.st.rollback(length - start)
        else:
            s.st.end_block()
        if s.rows is not None and length < start + n:
            s.rows = s.rows[: s.rows.size(0) - (start + n - length)]
        s.length, s.block = length, None


class PortDrafter:
    """`max_depth` above the block size chains drafting passes: the next pass takes the last drafted token as its anchor
    (at its position) and the newest target row as a stand-in for the hidden state it has not got. Only the draft
    quality of the extra tokens is affected, never the output (the target verifies everything). The window slot a pass
    writes is saved and restored, so a chain pass leaves no trace in the drafter rings."""

    def __init__(self, spec: PortSpec, max_depth: int | None = None):
        self.s = spec
        self.blk = spec.model.cfg.dspark_block_size
        self.max_depth = int(max_depth or self.blk)
        if self.max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        self.block_size = self.max_depth    # what the controller caps num_draft at
        self.holes = 0                      # positions the window never saw (lowers acceptance, never correctness)
        self._next = 0                      # first position not yet fed

    supports_want = True                # draft(want=k): chain passes only up to k tokens (the first pass is free of it)

    def draft(self, token: int, hidden, start_pos: int, want: int | None = None) -> DraftTokens:
        s = self.s
        limit = self.max_depth if want is None else max(1, min(int(want), self.max_depth))
        heads = s.model.mtp
        n = hidden.shape[0]
        first = start_pos - n + 1
        if start_pos < 1 or first < 0:
            raise ValueError(f"cannot draft at start_pos {start_pos} from {n} rows")
        if first == 0:
            self._next = 0
        elif first > self._next:
            self.holes += first - self._next
        if n > 1:
            heads.feed(hidden[None, : n - 1], first, s.st)
        last = hidden[n - 1][None, None]
        ids, _logits, conf = heads.forward_spec(torch.tensor([token]), last, start_pos, s.st)
        self._next = start_pos + 1
        toks, confs = ids[0, 1:], conf[0]
        sp = start_pos
        while toks.numel() < limit:                            # chained pass: anchor = last drafted token
            sp += self.blk
            cfg = s.model.cfg
            slot = sp % cfg.sliding_window
            n0 = cfg.num_hidden_layers
            saved = [s.st.win[n0 + j][:, slot].clone() for j in range(heads.n_stages)]
            ids2, _l2, conf2 = heads.forward_spec(toks[-1:].clone(), last, sp, s.st)
            for j, v in enumerate(saved):
                s.st.win[n0 + j][:, slot] = v
            toks = torch.cat([toks, ids2[0, 1:]])
            confs = torch.cat([confs, conf2[0]])
        toks, confs = toks[: limit], confs[: limit]
        return DraftTokens(toks.tolist(), conf=confs)           # conf stays a tensor: read only when a threshold is set
