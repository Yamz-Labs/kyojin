"""DSpark speculative decoding controller for DeepSeek-V4.1-Flash-1017 (framework agnostic).

What the official code gives us
-------------------------------
The official `inference/` folder ships the DSpark *forward pass* only (`Transformer.forward_spec`, `DSparkBlock`).
Its README says generation is plain autoregressive and `model.py` says nothing calls `forward_spec`. So there is no
official draft/verify loop, no official acceptance rule and no official cache rollback. This module is that loop,
built on the contract `forward_spec` defines:

* The draft stack (`n_mtp_layers` = 3 `DSparkBlock` layers, run as one network) takes ONE block per call: the token the
  target just chose (the "anchor", at position L) followed by `dspark_block_size - 1` noise tokens
  (`dspark_noise_token_id`). All positions of the block attend to each other and to a sliding window of main-model
  context, projected from `main_hidden`: the attention inputs (mean over the hyper-connection copies) of the target
  layers `dspark_target_layer_ids`, concatenated, at position L-1.
* The output is `block_size` draft tokens for positions L+1 .. L+block_size. Each one is chosen from the stage logits
  plus a rank-`dspark_markov_rank` bigram bias that depends on the previous drafted token (the Markov head), so the
  block is chained. A confidence head (`DSparkConfidenceHead`, one linear layer on [hidden, markov_embed], fp32) also
  returns one raw score per slot. The official code does not say what it means or how to use it. We read it as a
  logit of "this slot's draft will be accepted" and use it only as a stopping rule (see "Early stop" below).
* The stack keeps one window-KV slot per context position, written from `main_hidden` of that position. Each call
  writes exactly one slot, so the drafter must see the hidden state of EVERY accepted position, in order. A skipped
  position leaves a hole in its window and lowers acceptance (it never makes the output wrong).

The loop (greedy) per step, with `tok` at position L not yet in the target cache:

1. `drafts = drafter.draft(tok, hidden_rows, L - 1)` -> up to `block_size` tokens, d1..dK for positions L+1..L+K.
2. `out = target.verify([tok, d1..dK], L)` -> logits and target-layer hidden states for those K+1 positions.
3. Accept d_k while `argmax(out.logits[k-1]) == d_k`. The token after the last accepted one is `argmax` of the next
   logits row (a correction on a mismatch, a free bonus token when all K matched). A step emits a+1 tokens.
4. `target.truncate(L + a + 1)` drops the rejected positions; the new token becomes `tok`; `hidden_rows` are the rows
   of the a+1 kept positions. The drafter needs no rollback: it is only ever fed accepted positions.

Output is token identical to plain greedy decoding, for any drafter.

Sampling. The official sampler (`model.sample`) is the Gumbel-max trick. With a deterministic (argmax) draft the exact
speculative sampling rule is: draw the target token for each position in order with the official rule, accept the draft
if it equals the draw, stop at the first mismatch. That is standard speculative sampling for a one-hot draft law
(accept with probability p(d), residual = p without d), and it consumes one noise draw per emitted token, so under the
same seed the output matches the plain `generate()` loop token for token. When the drafter returns the laws `q` it drew
from (`DraftOut.q`), the standard rule is used instead: accept with probability min(1, p/q), resample from the
normalised max(0, p - q) on rejection; same output distribution, no seed parity with the plain loop.

Truncated sampling. `GumbelSampler(top_k, top_p, min_p)` filters the softmax law (top_k, then top_p, then min_p, then
renormalise) and runs Gumbel-max on it. The filter is a pure function of the logits row, and the noise draw count is
still one per emitted token, so the exactness argument above holds unchanged: same seed, same tokens as the plain loop
(the plain loop uses the same sampler). With draft laws `q`, `p` is the filtered law, so the output law is the filtered one.

Early stop and draft length. How many drafted tokens go to verify (`k`) is a free choice of the controller, as long as
it is made before the target rows of the round are seen. It then only changes cost, never the output: the walk
draws one noise value per emitted token whatever `k` is. Two rules choose `k`:

* Early stop: drafted slot i (0-based) is verified only if the confidence scores of slots 0..i are all at least
  logit(`conf_threshold`). The score of slot i is computed before token i is drawn (the Markov embedding of slot i
  is that of the previous token), so the stop depends on the tokens before i only; with drafts that carry laws `q`
  this keeps the p/q rule exact.
* `DraftTuner` (auto draft length): picks k in 1..4 from the running per-position acceptance, maximising expected
  tokens per round over a cost of 1 + cost_per_position * k.

Plain EOS handling follows `generate.py`: a token counts against `max_tokens` when it is sampled; the EOS token is
not part of the returned completion; the last emitted token is never in the target cache.
"""
from __future__ import annotations

import math
import time
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = ["TargetOut", "DraftOut", "Target", "Drafter", "Sampler", "GreedySampler", "GumbelSampler", "truncated_probs",
           "DraftTuner", "TimedTuner", "NgramLookup", "DSparkStats", "DSparkDecoder", "decode"]


# ---- interfaces ---------------------------------------------------------------------------------------------------


@dataclass
class TargetOut:
    """`logits[i]` is the next-token logit row after input position i; `hidden[i]` is the target-layer hidden state
    (what `Transformer.forward` returns as `main_hidden`) of input position i. Any 2-D indexable and sliceable
    object works (torch tensor, numpy array, list of rows)."""
    logits: Any
    hidden: Any


@dataclass
class DraftOut:
    """Draft tokens for positions L+1.. . `q[i]`, when given, is the probability vector `tokens[i]` was drawn from.
    `conf[i]`, when given, is the raw confidence-head score of slot i (a logit of acceptance), known before token i."""
    tokens: list[int]
    q: Sequence[Any] | None = None
    conf: Sequence[float] | None = None


@runtime_checkable
class Target(Protocol):
    """The big model, with a KV cache that holds the committed positions 0..n-1."""

    def prefill(self, tokens: Sequence[int]) -> TargetOut:
        """Reset the cache and run `tokens` at positions 0..P-1. `logits` needs only the last row (the official prefill
        returns just that one); `hidden` has P rows."""
        ...

    def verify(self, tokens: Sequence[int], start_pos: int) -> TargetOut:
        """Run `tokens` at positions start_pos.. on top of a cache of length start_pos. One logits row and one hidden
        row per token. The cache then holds start_pos + len(tokens) positions."""
        ...

    def truncate(self, length: int) -> None:
        """Make the cache equal to a cache that only ever saw positions 0..length-1. Called with a length that is not
        below the start of the last `verify` and not above its end."""
        ...


@runtime_checkable
class Drafter(Protocol):
    """The DSpark stack."""

    block_size: int       # `dspark_block_size`: the most tokens one call proposes

    def draft(self, token: int, hidden: Any, start_pos: int) -> DraftOut:
        """`token` is the anchor, at position start_pos + 1. `hidden` holds the target-layer hidden rows of the
        positions the drafter has not seen yet, ending at start_pos: rows for positions start_pos-n+1 .. start_pos.
        The first call gets the whole prompt (prefill seeds the window); later calls get the kept positions of the
        last step. Returns up to `block_size` tokens for positions start_pos + 2 .."""
        ...


@runtime_checkable
class Sampler(Protocol):
    stochastic: bool

    def sample(self, logits_row: Any) -> int:
        """One token from a logits row. Consumes one random draw when stochastic."""
        ...

    # only used for drafts that come with their laws `q`
    def probs(self, logits_row: Any) -> Any: ...
    def uniform(self) -> float: ...
    def draw(self, probs: Any) -> int: ...


class GreedySampler:
    stochastic = False

    def sample(self, logits_row: Any) -> int:
        return int(logits_row.argmax())

    def probs(self, logits_row: Any) -> Any:
        raise NotImplementedError("greedy decoding has no probabilities")

    uniform = draw = probs


def truncated_probs(logits_row: Any, temperature: float = 1.0, top_p: float = 1.0, top_k: int = 0, min_p: float = 0.0) -> Any:
    """softmax(logits / temperature) restricted to the top_k most likely tokens, then to the smallest set whose mass
    reaches top_p (the most likely token always stays), then to tokens with p >= min_p * p_max; renormalised. Ties are
    broken by token id (stable sort), so the result is a deterministic function of the row. float32, shape (V,)."""
    import torch
    probs = torch.softmax(logits_row.reshape(-1) / max(temperature, 1e-5), dim=-1, dtype=torch.float32)
    n = probs.numel()
    if not (0 < top_k < n or top_p < 1.0 or min_p > 0.0):
        return probs
    order = torch.sort(probs, descending=True, stable=True).indices
    sp = probs[order]
    keep = torch.ones(n, dtype=torch.bool)
    if 0 < top_k < n:
        keep[top_k:] = False
    if top_p < 1.0:
        before = torch.cumsum(sp, 0) - sp                         # mass of the tokens ranked above
        keep &= before < top_p
    if min_p > 0.0:
        keep &= sp >= min_p * sp[0]
    keep[0] = True
    out = torch.zeros_like(probs)
    out[order[keep]] = sp[keep]
    return out / out.sum()


class GumbelSampler:
    """The official `model.sample` rule: Gumbel-max over softmax(logits / temperature), one exponential draw per call.
    `top_k` (0 = off), `top_p` (1 = off) and `min_p` (0 = off) truncate the law first (see `truncated_probs`).
    Uses torch's global RNG unless a `torch.Generator` is given. temperature 0 is argmax."""
    stochastic = True

    def __init__(self, temperature: float = 1.0, generator: Any = None, *, top_p: float = 1.0, top_k: int = 0,
                 min_p: float = 0.0):
        if not 0 <= top_p <= 1 or top_k < 0 or not 0 <= min_p <= 1:
            raise ValueError("need 0 <= top_p <= 1, top_k >= 0, 0 <= min_p <= 1")
        self.temperature = temperature
        self.generator = generator
        self.top_p, self.top_k, self.min_p = top_p, top_k, min_p

    def sample(self, logits_row: Any) -> int:
        import torch
        if self.temperature == 0:
            return int(logits_row.argmax())
        probs = self.probs(logits_row).reshape(1, -1)
        noise = torch.empty_like(probs).exponential_(1, generator=self.generator)
        return int(probs.div_(noise).argmax(dim=-1))

    def probs(self, logits_row: Any) -> Any:
        return truncated_probs(logits_row, self.temperature, self.top_p, self.top_k, self.min_p)

    def uniform(self) -> float:
        import torch
        return float(torch.rand(1, generator=self.generator))

    def draw(self, probs: Any) -> int:
        import torch
        return int(torch.multinomial(probs, 1, generator=self.generator))


class DraftTuner:
    """Picks the draft length of each round from the running acceptance. The per-position acceptance `a` is a decayed
    count (success / trials, with a prior); a round that drafts k tokens is expected to emit 1 + a + ... + a^k tokens
    and to cost 1 + cost_per_position * k (one verify of k+1 positions, relative to a plain step). The best k in
    [k_min, k_max] wins, ties go to the smaller k. It only ever sees past rounds, so the output stays exact."""

    def __init__(self, k_min: int = 1, k_max: int = 4, *, cost_per_position: float = 0.1, decay: float = 0.9,
                 prior_rate: float = 0.7, prior_weight: float = 2.0):
        if not 1 <= k_min <= k_max:
            raise ValueError("need 1 <= k_min <= k_max")
        self.k_min, self.k_max, self.cost, self.decay = k_min, k_max, cost_per_position, decay
        self.succ, self.trials = prior_rate * prior_weight, prior_weight
        self.last = k_max

    @property
    def rate(self) -> float:
        return self.succ / self.trials

    def choose(self, cap: int | None = None) -> int:
        a = self.rate
        hi = self.k_max if cap is None else max(self.k_min, min(self.k_max, cap))
        best_k, best = self.k_min, -1.0
        for k in range(self.k_min, hi + 1):
            tokens = (k + 1) if a >= 1 else (1 - a ** (k + 1)) / (1 - a)
            score = tokens / (1 + self.cost * k)
            if score > best + 1e-12:
                best_k, best = k, score
        self.last = best_k
        return best_k

    def observe(self, proposed: int, compared: int, accepted: int) -> None:
        """A round sent `proposed` drafts; `compared` of them were checked (up to the first miss); `accepted` matched."""
        if proposed <= 0:
            return
        self.succ = self.succ * self.decay + accepted
        self.trials = self.trials * self.decay + compared


def _cat_rows(a: Any, b: Any) -> Any:
    """Rows of a followed by rows of b (torch tensors, numpy arrays or lists)."""
    if hasattr(a, "cat") or type(a).__module__.startswith("torch"):
        import torch
        return torch.cat([a, b], dim=0)
    if type(a).__module__.startswith("numpy"):
        import numpy as np
        return np.concatenate([a, b], axis=0)
    return list(a) + list(b)


class NgramLookup:
    """Prompt lookup: the continuation that followed the latest earlier occurrence of the last `n` tokens of the context
    (prompt + emitted tokens). Pure Python, exact (a draft only ever proposes; the target verifies). `match` is how many
    tokens before the continuation agree with the context tail (up to `max_back`)."""

    def __init__(self, n: int = 3, max_back: int = 32):
        self.n, self.max_back = n, max_back
        self.ctx: list[int] = []
        self.idx: dict[tuple, int] = {}
        self.done = 0                                   # keys ending at ctx index < done are in idx

    def extend(self, tokens: Sequence[int]) -> None:
        self.ctx.extend(int(t) for t in tokens)

    def find(self, k: int) -> tuple[list[int], int]:
        """(continuation of up to k tokens, match length) or ([], 0)."""
        ctx, n = self.ctx, self.n
        last = len(ctx) - 1                             # index the tail key ends at; earlier keys only
        for j in range(max(self.done, n - 1), last):
            self.idx[tuple(ctx[j - n + 1:j + 1])] = j + 1
        self.done = max(self.done, last)
        if last < n - 1 or k < 1:
            return [], 0
        p = self.idx.get(tuple(ctx[last - n + 1:last + 1]))
        if p is None:
            return [], 0
        m = n
        while m < self.max_back and p - 1 - m >= 0 and ctx[p - 1 - m] == ctx[last - m]:
            m += 1
        return ctx[p:p + k], m


class TimedTuner:
    """Draft length from live per-position acceptance AND the measured cost of a round. Round time is fitted as
    t = a + b*k (decayed least squares over the rounds seen, with two pseudo points on the prior slope b = cost_ratio*a),
    so a tier that stalls on SSD reads (every extra verified position pulls in more experts) makes the tuner draft less, a
    resident pack makes it draft more. Position i has its own success rate (decayed, conditional on reaching it). The best k
    in [k_min, k_max] maximises expected tokens / expected time; k = 0 (a plain step) is allowed. Sees only past rounds:
    the output stays exact."""

    def __init__(self, k_min: int = 0, k_max: int = 4, *, cost_ratio: float = 0.3, decay: float = 0.9, time_decay: float = 0.95,
                 prior_rate: float = 0.7, prior_weight: float = 2.0, prior_points: float = 2.0):
        if not 0 <= k_min <= k_max:
            raise ValueError("need 0 <= k_min <= k_max")
        self.k_min, self.k_max, self.decay, self.tdecay = k_min, k_max, decay, time_decay
        self.cost_ratio, self.pp = cost_ratio, prior_points
        self.succ = [prior_rate * prior_weight] * (k_max + 2)
        self.trials = [prior_weight] * (k_max + 2)
        self.sw = self.sk = self.skk = self.st = self.skt = 0.0
        self.last = k_max

    @property
    def rate(self) -> float:
        return self.succ[1] / self.trials[1]

    def _fit(self) -> tuple[float, float]:
        if self.sw <= 0:
            return 1.0, self.cost_ratio
        mean_t, mean_k = self.st / self.sw, self.sk / self.sw
        a0 = mean_t / (1 + self.cost_ratio * mean_k)
        w, K = self.pp, float(self.k_max)
        pts = ((0.0, a0), (K, a0 * (1 + self.cost_ratio * K)))
        sw, sk, skk, st, skt = self.sw, self.sk, self.skk, self.st, self.skt
        for k, t in pts:
            sw += w; sk += w * k; skk += w * k * k; st += w * t; skt += w * k * t
        den = sw * skk - sk * sk
        b = (sw * skt - sk * st) / den if den > 1e-12 else self.cost_ratio * a0
        a = (st - b * sk) / sw
        return max(a, 1e-9), max(b, 0.0)

    def choose(self, cap: int | None = None) -> int:
        a, b = self._fit()
        hi = self.k_max if cap is None else max(self.k_min, min(self.k_max, cap))
        tokens, run = 1.0, 1.0
        best_k, best = self.k_min, -1.0
        for k in range(0, hi + 1):
            if k > 0:
                run *= min(self.succ[k] / self.trials[k], 1.0)
                tokens += run
            if k < self.k_min:
                continue
            score = tokens / (a + b * k)
            if score > best * (1 + 1e-9):
                best_k, best = k, score
        self.last = best_k
        return best_k

    def observe(self, proposed: int, compared: int, accepted: int) -> None:
        if proposed <= 0:
            return
        self.succ = [x * self.decay for x in self.succ]
        self.trials = [x * self.decay for x in self.trials]
        for i in range(1, min(accepted + 1, proposed) + 1):          # positions reached: accepted ones and the first miss
            self.trials[i] += 1
            if i <= accepted:
                self.succ[i] += 1

    def observe_time(self, k: int, seconds: float) -> None:
        d = self.tdecay
        self.sw, self.sk, self.skk, self.st, self.skt = (x * d for x in (self.sw, self.sk, self.skk, self.st, self.skt))
        self.sw += 1; self.sk += k; self.skk += k * k; self.st += seconds; self.skt += k * seconds


# ---- statistics ---------------------------------------------------------------------------------------------------


@dataclass
class DSparkStats:
    steps: int = 0                  # verify rounds that had at least one draft token
    plain_steps: int = 0            # rounds without a draft (the last token of a budget)
    proposed: int = 0               # draft tokens sent to verify
    accepted: int = 0               # draft tokens equal to the target's choice
    emitted: int = 0                # tokens returned, EOS included
    draft_len_hist: Counter = field(default_factory=Counter)      # draft length -> rounds
    accepted_hist: Counter = field(default_factory=Counter)       # accepted drafts -> rounds
    verify_calls: int = 0           # target.verify calls
    verify_positions: int = 0       # positions run by those calls (the cost of verifying)
    conf_cut_rounds: int = 0        # rounds where the confidence rule sent fewer drafts than the length cap allowed
    conf_dropped: int = 0           # drafted tokens the confidence rule kept out of verify
    ngram_rounds: int = 0           # rounds whose draft came (partly) from the prompt lookup
    tuner_k: int = 0                # last draft length the tuner chose (0 = no tuner)

    @property
    def rejected(self) -> int:
        return self.proposed - self.accepted

    @property
    def accepted_per_step(self) -> float:
        """Mean accepted draft tokens per drafting round (the bonus/correction token not counted)."""
        return self.accepted / self.steps if self.steps else 0.0

    @property
    def tokens_per_step(self) -> float:
        """Mean tokens emitted per drafting round (accepted + the correction or bonus token)."""
        return (self.accepted + self.steps) / self.steps if self.steps else 0.0

    @property
    def draft_len_mean(self) -> float:
        return self.proposed / self.steps if self.steps else 0.0

    def as_engine_stats(self) -> dict[str, Any]:
        """The keys tools/glm/serve.py reads from an engine job (`accepted_draft_tokens`, `rejected_draft_tokens`)
        plus the step count and the per-step mean."""
        return {"accepted_draft_tokens": self.accepted, "rejected_draft_tokens": self.rejected,
                "draft_steps": self.steps, "accepted_per_step": round(self.accepted_per_step, 4),
                "draft_len_hist": dict(sorted(self.draft_len_hist.items())),
                "draft_len_mean": round(self.draft_len_mean, 4), "verify_calls": self.verify_calls,
                "verify_positions": self.verify_positions, "conf_cut_rounds": self.conf_cut_rounds,
                "conf_dropped_tokens": self.conf_dropped, "tuner_k": self.tuner_k, "ngram_rounds": self.ngram_rounds}


# ---- the loop -----------------------------------------------------------------------------------------------------


class DSparkDecoder:
    """Runs one request. `num_draft` caps the draft length (default and maximum: `drafter.block_size`).
    `conf_threshold` (a probability in (0, 1); needs drafts that carry `conf`) stops sending drafts to verify at the
    first slot whose confidence is below it. `tuner` chooses the draft length of each round (capped by `num_draft`).
    Neither changes the output."""

    def __init__(self, target: Target, drafter: Drafter, *, eos_id: int, sampler: Sampler | None = None,
                 num_draft: int | None = None, conf_threshold: float | None = None, tuner: DraftTuner | None = None,
                 ngram: str = "", ngram_n: int = 3, ngram_min_match: int = 6, want_cap: bool = False):
        block = int(drafter.block_size)
        if num_draft is not None and num_draft < 1:
            raise ValueError("num_draft must be >= 1 (use plain decoding for 0)")
        if conf_threshold is not None and not 0 < conf_threshold < 1:
            raise ValueError("conf_threshold must be a probability in (0, 1)")
        self.conf_cut = None if conf_threshold is None else math.log(conf_threshold / (1 - conf_threshold))
        self.tuner = tuner
        self.want_cap = bool(want_cap)                # ask the drafter for only as many tokens as the round may send
        self.ngram_mode = ngram if ngram in ("strong", "extend") else ""
        self.ngram_n, self.ngram_min_match = int(ngram_n), int(ngram_min_match)
        self.target, self.drafter, self.eos_id = target, drafter, eos_id
        self.sampler = sampler or GreedySampler()
        self.num_draft = min(num_draft or block, block)
        self.stats = DSparkStats()

    def _conf_len(self, draft: DraftOut, k: int) -> int:
        """Longest prefix of the first k drafts whose confidence scores all reach the threshold."""
        if self.conf_cut is None or draft.conf is None:
            return k
        for i in range(min(k, len(draft.conf))):
            if float(draft.conf[i]) < self.conf_cut:
                return i
        return k

    # -- one verify round --------------------------------------------------------------------------------------------

    def _walk(self, out: TargetOut, draft: DraftOut, k: int, budget: int) -> tuple[list[int], int]:
        """Pick the tokens of one round from the K+1 logits rows. Returns (tokens, matches): `tokens` stops at the first
        mismatch, at EOS, or when `budget` tokens are out; `matches` counts drafts equal to the chosen token."""
        s = self.sampler
        use_q = s.stochastic and draft.q is not None
        tokens: list[int] = []
        matches = 0
        for i in range(k + 1):
            row = out.logits[i]
            d = draft.tokens[i] if i < k else None
            if use_q and d is not None:
                p, q = s.probs(row), draft.q[i]
                pd, qd = float(p[d]), float(q[d])
                if qd > 0 and s.uniform() * qd <= pd:
                    tok, hit = d, True
                else:
                    resid = (p - q).clamp(min=0) if hasattr(p, "clamp") else (p - q).clip(min=0)
                    tot = float(resid.sum())
                    tok, hit = (s.draw(resid / tot) if tot > 0 else s.draw(p)), False
            else:
                tok = s.sample(row)
                hit = d is not None and tok == d
            tokens.append(tok)
            matches += hit
            if tok == self.eos_id or len(tokens) >= budget or not hit:
                break
        return tokens, matches

    def stream(self, prompt: Sequence[int], max_tokens: int) -> Iterator[list[int]]:
        """Yield the tokens of each round (EOS included as its last element when sampled). Generation ends after
        `max_tokens` tokens or at EOS. The cache is left as a plain loop leaves it: every emitted token but the last."""
        if max_tokens < 1:
            return
        st = self.stats
        n_prompt = len(prompt)
        out = self.target.prefill(prompt)
        tok = self.sampler.sample(out.logits[-1])
        look = NgramLookup(self.ngram_n) if self.ngram_mode else None
        new_tokens: list[int] = []
        if look is not None:
            look.extend(prompt)
            new_tokens = [tok]
        st.emitted += 1
        yield [tok]
        count = 1
        hidden = out.hidden
        pos = n_prompt                                # position of `tok`
        while tok != self.eos_id and count < max_tokens:
            remaining = max_tokens - count
            t_round0 = time.perf_counter()
            # no draft for the last token of the budget, nor from a one-token context (the stack needs start_pos > 0;
            # position 0 then never reaches the drafter, a hole in its window that cannot change the output)
            cap = self.num_draft if self.tuner is None else min(self.num_draft, self.tuner.choose())
            ng, ng_m = ([], 0)
            if self.ngram_mode and remaining > 1:
                look.extend(new_tokens); new_tokens = []
                ng, ng_m = look.find(min(self.num_draft, remaining - 1))
            used_ng = False
            strong = bool(ng) and ng_m >= self.ngram_min_match
            if self.ngram_mode == "strong" and strong:
                want = 1                                  # the drafter still runs (its window needs the rows) but chains nothing
            else:
                want = cap
            skip_drafter = self.want_cap and remaining > 1 and pos >= 2 and (cap == 0 or (self.ngram_mode == "strong" and strong))
            if remaining <= 1 or pos < 2 or skip_drafter:
                draft = DraftOut([])
            elif self.want_cap and getattr(self.drafter, "supports_want", False):
                draft = self.drafter.draft(tok, hidden, pos - 1, want=want)
            else:
                draft = self.drafter.draft(tok, hidden, pos - 1)
            if cap == 0 and not strong:
                pass
            elif strong and remaining > 1 and pos >= 2:
                dt = list(draft.tokens)
                if self.ngram_mode == "strong":
                    draft, cap, used_ng = DraftOut(list(ng)), max(cap, len(ng)), True
                elif ng[:len(dt)] == dt[:len(ng)] and len(ng) > len(dt):      # extend: the lookup agrees with the drafter, then goes on
                    draft, cap, used_ng = DraftOut(dt + list(ng[len(dt):]), conf=draft.conf), max(cap, len(ng)), True
            k = min(len(draft.tokens), cap, remaining - 1)
            if used_ng:
                st.ngram_rounds += 1
            k_conf = self._conf_len(draft, k)
            if k_conf < k:
                st.conf_cut_rounds += 1
                st.conf_dropped += k - k_conf
                k = k_conf
            out = self.target.verify([tok, *draft.tokens[:k]], pos)
            st.verify_calls += 1
            st.verify_positions += k + 1
            tokens, matches = self._walk(out, draft, k, remaining)
            if self.tuner is not None and not used_ng:
                self.tuner.observe(k, min(len(tokens), k), matches)
                st.tuner_k = self.tuner.last
            kept = len(tokens)                        # positions pos .. pos + kept - 1 stay: tok + kept-1 drafts
            if kept < k + 1:
                self.target.truncate(pos + kept)
            if hasattr(self.tuner, "observe_time"):
                self.tuner.observe_time(k, time.perf_counter() - t_round0)
            if k:
                st.steps += 1
                st.proposed += k
                st.accepted += matches
                st.draft_len_hist[k] += 1
                st.accepted_hist[matches] += 1
            else:
                st.plain_steps += 1
            st.emitted += len(tokens)
            count += len(tokens)
            tok = tokens[-1]
            if look is not None:
                new_tokens = new_tokens + tokens
            hidden = _cat_rows(hidden, out.hidden[:kept]) if skip_drafter else out.hidden[:kept]
            pos += kept
            yield tokens


def decode(target: Target, drafter: Drafter, prompt: Sequence[int], max_tokens: int, eos_id: int, *,
           sampler: Sampler | None = None, num_draft: int | None = None, conf_threshold: float | None = None,
           tuner: DraftTuner | None = None) -> tuple[list[int], DSparkDecoder]:
    """Whole completion, EOS cut off like `generate.py`. Returns (tokens, decoder); `decoder.stats` has the numbers."""
    dec = DSparkDecoder(target, drafter, eos_id=eos_id, sampler=sampler, num_draft=num_draft,
                        conf_threshold=conf_threshold, tuner=tuner)
    out: list[int] = []
    for chunk in dec.stream(prompt, max_tokens):
        out.extend(chunk)
    if eos_id in out:
        out = out[:out.index(eos_id)]
    return out, dec
