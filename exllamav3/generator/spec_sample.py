import os
import torch
from .sampler.custom import CustomSampler, SS_Fused, SS_Sample, SS_Normalize, SS_Sort, SS_TopK, SS_TopP, SS_Temperature, \
    SamplingState, SS

# Speculative sampling with a sampled draft (Leviathan et al. 2023, arXiv 2211.17192, Alg. 1; Chen et al. 2023,
# arXiv 2302.01318, Alg. 2). For sampled jobs (T > 0) the MTP drafter draws d ~ q from its own logits under the job's
# sampler law instead of taking its argmax; the verify keeps d with probability min(1, p(d) / q(d)) and otherwise draws
# from norm(max(0, p - q)). The output is distributed exactly as plain sampling from p, for any q; the keep rate is
# sum_x min(p(x), q(x)) instead of p(argmax q). Greedy jobs never take this path. EXL3_SPEC_SAMPLED=0 disables it.
SPEC_SAMPLED = os.environ.get("EXL3_SPEC_SAMPLED", "0") != "0"


# Step-by-step stacks whose law is rebuilt by running them up to (not including) the final SS_Sample
LAW_STEPS = (SS_Normalize, SS_Sort, SS_TopK, SS_TopP, SS_Temperature)


def eligible(job) -> tuple | None:
    """
    How to rebuild the job's sampling law exactly, or None (the job keeps the argmax draft rule). Two cases: the fused
    sampler (temperature, top-K, top-P; no min-P), rebuilt by fused_law(); or a step-by-step stack of LAW_STEPS ending in
    SS_Sample (Gumbel-max over the kept probabilities, i.e. a draw proportional to them), rebuilt by running the same steps
    on the same row. No penalties, masks, filters, forced tokens or returned probabilities.
    """
    if job.filters or job.forced_ids is not None or job.return_probs or job.return_top_tokens or \
            job.device_logit_mask is not None or len(job.sequences) != 1:
        return None
    s = job.sampler
    if not isinstance(s, CustomSampler) or not s.steps:
        return None
    vs = job.generator.tokenizer.actual_vocab_size
    st = s.steps
    if len(st) == 1 and type(st[0]) is SS_Fused:
        f = st[0]
        if f.mode not in (SS_Fused.MODE_SAMPLE, SS_Fused.MODE_SAMPLE_FILTERS):
            return None
        if f.mode == SS_Fused.MODE_SAMPLE_FILTERS and f.filters & SS_Fused.F_MINP:
            return None
        filters = f.filters if f.mode == SS_Fused.MODE_SAMPLE_FILTERS else 0
        return ("fused", (f.inv_temp, f.inv_temp_filter, filters, f.top_k, f.top_p), vs)
    if type(st[-1]) is SS_Sample and all(type(x) in LAW_STEPS for x in st[:-1]):
        return ("steps", st[:-1], vs)
    return None


def law(logits: torch.Tensor, prm: tuple) -> torch.Tensor:
    """Dense sampling law (sums to 1, zero on the padded vocabulary) of one row of logits over the padded vocabulary."""
    kind, x, vs = prm
    row = logits.view(-1)
    if kind == "fused":
        p = torch.zeros(row.numel(), dtype = torch.float, device = row.device)
        p[:vs] = fused_law(row[:vs], x)
        return p
    # Same input as Sampler.forward gives the steps: the full padded row with the padding at -inf
    row = row.float().clone()
    row[vs:] = -float("inf")
    state = SamplingState(rand_u32 = 0, dim = row.numel(), bsz = 1, in_logits = row.view(1, -1))
    for step in x:
        step.run(state)
    match state.state:
        case SS.PROBS_S | SS.PROBS_N_S:
            p = torch.zeros_like(row).scatter_(0, state.indices.view(-1), state.probs.view(-1))
        case SS.PROBS | SS.PROBS_N:
            p = state.probs.view(-1).clone()
        case SS.LOGITS_S:
            p = torch.zeros_like(row).scatter_(0, state.indices.view(-1), torch.softmax(state.logits.view(-1), 0))
        case SS.LOGITS:
            p = torch.softmax(state.logits.view(-1), 0)
        case _:
            p = torch.softmax(row, 0)
    return p / p.sum()


def fused_law(logits: torch.Tensor, prm: tuple) -> torch.Tensor:
    """
    Dense sampling law of the fused sampler (sampling_fused.cu) for one row of logits over the unpadded vocabulary:
    top-K keeps the k best with their exact ties; top-P (mass at the filter temperature, over the top-K set) keeps the
    tokens strictly above the first one whose inclusive cumulative mass exceeds top_p, dropping that token's tie group
    (the top token always kept); then softmax(l / T) over the kept set. No host sync.
    """
    inv_temp, inv_temp_filter, filters, top_k, top_p = prm
    l = logits.float().view(-1)
    n = l.numel()
    sl, si = torch.sort(l, descending = True)
    kept = torch.ones_like(sl, dtype = torch.bool)
    if filters & SS_Fused.F_TOPK and 0 < top_k < n:
        kept = sl >= sl[top_k - 1]
    if filters & SS_Fused.F_TOPP and top_p < 1.0:
        w = torch.exp((sl - sl[0]) * inv_temp_filter) * kept
        cs = torch.cumsum(w, 0)
        # cs[-1] (the top-K mass) exceeds the target, so the crossing index c is always inside the top-K set
        c = (cs <= top_p * cs[-1]).sum().clamp(max = n - 1)
        kept = kept & ((sl > sl.gather(0, c.view(1))) | (sl == sl[0]))
    pw = torch.exp((sl - sl[0]) * inv_temp) * kept
    p = torch.zeros_like(l)
    p.scatter_(0, si, pw / pw.sum())
    return p


def draw(p: torch.Tensor, u: float) -> torch.Tensor:
    """
    Inverse-CDF draw from a dense (unnormalized) law with a uniform u in [0, 1); never returns a zero-mass index, so
    never an id of the padded vocabulary (a law is zero there). The search is bounded by the last index that holds mass:
    that is where the running sum first reaches its total. It matters when u * total rounds up to total in fp32 (the
    search then runs off the end), and for a law without mass (all zero, or NaN from a broken row), which used to end on
    the last column of the row, a padded id. Such a law yields index 0, a real token. No host sync.
    """
    cs = torch.cumsum(p, 0)
    total = cs[-1:]
    idx = torch.searchsorted(cs, total * u, right = True)
    last = (cs < total).sum()
    return torch.minimum(idx, last.view(1))


def verify(logits: torch.Tensor, q: torch.Tensor, d: int, prm: tuple, rng) -> torch.Tensor:
    """
    One verify position: target logits (padded vocabulary), the draft law q and the drafted token d (d ~ q). Returns
    the emitted token as a (1, 1) long tensor: d if kept, else a draw from norm(max(0, p - q)) (never d).
    """
    p = law(logits, prm)
    u1, u2 = rng.random(), rng.random()
    keep = u1 * q[d] < p[d]
    res = (p - q).clamp_(min = 0.0)
    r = draw(torch.where(res.sum() > 0, res, p), u2)   # p == q leaves no residual: draw from p then (keep is true but for rounding)
    return torch.where(keep, torch.full_like(r, d), r).view(1, 1)


# Relaxed ("typical") acceptance, Medusa rule (Cai et al. 2024, arXiv 2401.10774) for sampled jobs that keep the argmax draft:
# the verify keeps draft j when the sampled token equals it (exact) OR when p(draft_j) >= min(eps, delta * exp(-H_j)), p and H
# being the job's sampling law (top-k / top-p cuts, then temperature) at row j. A token cut by the filters has p = 0 and is never
# kept. At the first rejected row the token is the normal sample. EXL3_SPEC_TYPICAL=0 (default): nothing changes, bit for bit.
TYPICAL = os.environ.get("EXL3_SPEC_TYPICAL", "0") != "0"
TYP_EPS = float(os.environ.get("EXL3_SPEC_TYPICAL_EPS", "0.09"))
TYP_DELTA = float(os.environ.get("EXL3_SPEC_TYPICAL_DELTA", "0.3"))
# EXL3_SPEC_TYPICAL_HMAX (nats, default unset = off): entropy gate. A non-matching draft is relaxed only at rows whose entropy H
# is <= HMAX, in addition to the p >= min(eps, delta * exp(-H)) rule; above it the draft is kept by equality only.
_hm = os.environ.get("EXL3_SPEC_TYPICAL_HMAX", "").strip()
TYP_HMAX = float(_hm) if _hm else None
# Read-only probe: EXL3_SPEC_PTAP=<file.jsonl> appends one line per verify round (p_draft, entropy, match per row)
PTAP = os.environ.get("EXL3_SPEC_PTAP") or None
# Test switch for in-server A/B: EXL3_SPEC_TYPICAL_FILE=<path>, the rule is on while the file holds "1" (re-read when its mtime changes)
# EXL3_SPEC_TYPICAL_LOOKUP, how the relaxed rule treats the draft sources:
#   "rep" (default): relaxed for model-head (MTP) drafts and for drafts copied from the prompt / tool outputs, except that a draft
#       token which would complete an n-gram (REP_N tokens) already seen >= 2 times in the job's last REP_W generated tokens is
#       accepted by equality only. Source-independent and keyed on the job's own output: a sampled job cannot be pulled around a
#       loop by relaxed drafts, a copy of the prompt stays relaxed.
#   "0": drafts copied from the context keep the exact rule, relaxed for model-head drafts only (measurement arm).
#   "1": relaxed rule for every draft source, no repetition check (the old behaviour; measurement arm).
TYPICAL_LOOKUP = os.environ.get("EXL3_SPEC_TYPICAL_LOOKUP", "rep").strip().lower()
if TYPICAL_LOOKUP not in ("rep", "0", "1"):
    TYPICAL_LOOKUP = "rep"
REP_N = int(os.environ.get("EXL3_SPEC_TYPICAL_REP_N", "12"))
REP_W = int(os.environ.get("EXL3_SPEC_TYPICAL_REP_W", "2048"))
REP_COUNT = 2
REP_REFUSED = [0]   # relaxed acceptances refused by the repetition check, whole process
_TFILE = os.environ.get("EXL3_SPEC_TYPICAL_FILE") or None
_tstate = [0.0, TYPICAL]


def typical_on() -> bool:
    if _TFILE is None:
        return TYPICAL
    try:
        m = os.stat(_TFILE).st_mtime
        if m != _tstate[0]:
            _tstate[0] = m
            _tstate[1] = open(_TFILE).read().strip() == "1"
    except OSError:
        pass
    return _tstate[1]


def draft_is_copy(gen, draft_tokens: torch.Tensor, row: int) -> bool:
    """
    True when row `row` of this round's draft was copied from the context, not produced by the model head. A drafter round returns a
    slice of gen.draft_ids_pinned; every other draft (lookup, mixed rows) is a tensor of its own, copied by default. A policy that
    mixes sources in one tensor says which rows are copies in gen.draft_copy_rows = (tensor, [bool per row]).
    """
    mark = getattr(gen, "draft_copy_rows", None)
    if mark is not None and mark[0] is draft_tokens:
        return bool(mark[1][row])
    return draft_tokens.data_ptr() != gen.draft_ids_pinned.data_ptr()


# EXL3_SPEC_TYPICAL_PHASE (read per round): "answer" = the relaxed rule applies only once the job's think block is closed (drafts
# inside thinking are verified exactly: free-text loops start there, and relaxed acceptance of copied/argmax drafts feeds them);
# "all" (default) = the relaxed rule everywhere, as before. Same rule as the MLX engine's spec_typical_phase.
def phase_answer_only() -> bool:
    return os.environ.get("EXL3_SPEC_TYPICAL_PHASE", "all").strip().lower() == "answer"


def _think_ids(gen) -> tuple:
    ids = getattr(gen, "_typ_think_ids", None)
    if ids is None:
        def one(t):
            try:
                v = gen.tokenizer.single_id(t)
                return v if isinstance(v, int) and v >= 0 else -1
            except Exception:
                return -1
        ids = (one("<think>"), one("</think>"))
        gen._typ_think_ids = ids
    return ids


def in_think(gen, job) -> bool:
    """True while the last think marker of the job's sequence (prompt + output) is an open <think>. Incremental: reads only the new
    tail each round (state on the job); a shorter sequence (rewind) rescans from the start. False when the model has no think tokens."""
    o, c = _think_ids(gen)
    if o < 0 or c < 0:
        return False
    seq = job.sequences[0].sequence_ids
    n = len(seq)
    lo, state = getattr(job, "_typ_think", (0, False))
    if n < lo:
        lo, state = 0, False
    if n > lo:
        for t in reversed(seq.torch_slice(lo, n).reshape(-1).tolist()):
            if t == o or t == c:
                state = t == o
                break
    job._typ_think = (n, state)
    return state


def relax_allowed(gen, draft_tokens: torch.Tensor, row: int) -> bool:
    """Source gate: False only in arm "0" for a copied draft. The repetition check (rep_refused) is separate."""
    return TYPICAL_LOOKUP != "0" or not draft_is_copy(gen, draft_tokens, row)


# Adaptive loop control for the relaxed rule. A loop is cheap to stop early and expensive once its span is in the context, so a job that
# shows an early loop signal accepts by exact equality (no relaxed rule, MTP and copied drafts alike) for a window of its own generated
# tokens, then relaxes again. Signals: a refused REP_N-gram (rep_refused), and a thinking-text line of >= 40 non-space chars seen twice
# (tools/loopguard.LineWatch, fed by the server). A re-trigger while the window is open, or within the next window, doubles it.
# EXL3_SPEC_TYPICAL_COOLDOWN: first window in tokens (default 512, 0 = off, read per round); cap CD_MAX; one burst of signals counts once.
# EXL3_SPEC_TYPICAL_REWIND: rewinds per reply (default 3, 0 = off, read per round); the server cuts the thinking back to just before the first
# copy of the repeated span and continues from there (see Cooldown.rw_abs). The relaxed rule is never touched while no signal fires.
CD_MAX = 16384
CD_DEBOUNCE = 64
STATS = {"cooldown_triggers": 0, "cooldown_tokens": 0, "rewinds": 0, "rewound_tokens": 0}   # whole process, exposed by the server


def _int_env(name: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(name, str(default)) or 0))
    except ValueError:
        return 0


def cooldown_base() -> int:
    return _int_env("EXL3_SPEC_TYPICAL_COOLDOWN", 512)


def rewind_max() -> int:
    return _int_env("EXL3_SPEC_TYPICAL_REWIND", 3)


class Cooldown:
    """Per-job window state. Positions are the job's own generated-token count (job.new_tokens)."""

    def __init__(self, win: int = 0, end: int = 0):
        self.win = win        # current window (0 = never triggered)
        self.end = end        # exact acceptance while the job's generated count is below end
        self.last = -1        # generated count at the last trigger
        self.rw_abs = None    # absolute sequence index where the first copy of the refused n-gram starts (the server consumes it)

    def armed(self, n: int) -> bool:
        return n < self.end

    def trigger(self, n: int, base: int) -> bool:
        """Early loop signal at generated count n. True when it counted (not a repeat of the same burst)."""
        if self.last >= 0 and n - self.last < CD_DEBOUNCE:
            return False
        win = min(2 * self.win, CD_MAX) if self.win and n < self.end + self.win else base
        self.win, self.end, self.last = win, n + win, n
        STATS["cooldown_triggers"] += 1
        STATS["cooldown_tokens"] += win
        return True


def cd_armed(job) -> bool:
    """True while the job is inside an exact-acceptance window (flag on)."""
    cd = getattr(job, "cd", None)
    return cd is not None and cd.armed(job.new_tokens)


class RepTracker:
    """
    Counts the REP_N-grams of one job's last REP_W generated tokens (a dict from n-gram to count, updated from the new tail of
    the sequence on each call, so O(new tokens) per round and O(REP_N) per check).
    """
    def __init__(self):
        self.toks = []        # generated tokens inside the window
        self.end = 0          # absolute length of the sequence consumed so far
        self.counts = {}

    def _add(self, t):
        self.toks.append(t)
        if len(self.toks) >= REP_N:
            k = tuple(self.toks[-REP_N:])
            self.counts[k] = self.counts.get(k, 0) + 1
        if len(self.toks) > REP_W:   # evict the oldest token and the n-gram that started on it
            k = tuple(self.toks[:REP_N])
            c = self.counts.get(k, 0) - 1
            if c > 0: self.counts[k] = c
            else: self.counts.pop(k, None)
            del self.toks[0]

    def feed(self, seq, ngen: int):
        """seq: the job's token sequence (len, torch_slice), ngen: how many trailing tokens were generated. Reads only the new tail."""
        n = len(seq)
        if self.end and self.end <= n and n - self.end <= REP_W:
            lo = self.end
        else:   # first call, truncation or a long gap: rebuild from the generated tail
            lo = max(n - min(max(ngen, 0), REP_W), 0)
            self.toks, self.counts = [], {}
        for t in (seq.torch_slice(lo, n).reshape(-1).tolist() if n > lo else []):
            self._add(t)
        self.end = n

    def repeated(self, d: int) -> bool:
        if len(self.toks) < REP_N - 1:
            return False
        return self.counts.get(tuple(self.toks[-(REP_N - 1):]) + (d,), 0) >= REP_COUNT

    def first_repeat(self, d: int) -> int:
        """Absolute sequence index of the start of the FIRST occurrence (inside the window) of the n-gram that d would complete; -1 if none."""
        t = self.toks
        if len(t) < REP_N - 1:
            return -1
        gram = t[-(REP_N - 1):] + [d]
        k = len(gram)
        for i in range(len(t) - k + 1):
            if t[i] == gram[0] and t[i:i + k] == gram:
                return self.end - len(t) + i
        return -1


def rep_refused(job, d: int) -> bool:
    """
    True when draft token d, appended to the job's generated output so far, would complete a REP_N-gram already seen
    >= REP_COUNT times in its last REP_W generated tokens. Only meaningful in "rep" mode. Counts the refusal.
    """
    if TYPICAL_LOOKUP != "rep":
        return False
    tr = getattr(job, "_rep_tracker", None)
    if tr is None:
        tr = job._rep_tracker = RepTracker()
    tr.feed(job.sequences[0].sequence_ids, job.new_tokens + getattr(job, "rep_extra", 0))   # rep_extra: kept thinking of a rewound reply
    if tr.repeated(d):
        REP_REFUSED[0] += 1
        job.rep_refused = getattr(job, "rep_refused", 0) + 1
        base = cooldown_base()
        if base:                                  # the refused n-gram is an early loop signal
            cd = getattr(job, "cd", None)
            if cd is None:
                cd = job.cd = Cooldown()
            if cd.trigger(job.new_tokens, base) and rewind_max() > 0:
                at = tr.first_repeat(d)
                if at >= 0:
                    cd.rw_abs = at if cd.rw_abs is None else min(cd.rw_abs, at)
        return True
    return False


def _topk_of(prm: tuple):
    """The top-K cut of the law (None = no cut): lets typical_rows work on K logits per row instead of the whole vocabulary."""
    kind, x, vs = prm
    if kind == "fused":
        return x[3] if x[2] & SS_Fused.F_TOPK and 0 < x[3] < vs else None
    ks = [st.top_k for st in x if type(st) is SS_TopK and 0 < st.top_k < vs]
    return min(ks) if ks else None


def typical_rows(rows: torch.Tensor, drafts: list, prm: tuple, eps: float, delta: float):
    """
    rows: (n, V) verify logits of the n draft rows; drafts: the n drafted ids. Returns (p, H, ok) as python lists, one
    host sync: p = law probability of the draft, H = entropy of the law, ok = p > 0 and p >= min(eps, delta * exp(-H))
    and, when TYP_HMAX is set, H <= TYP_HMAX.
    With a top-K cut in the law only the K best logits of each row are sorted and normalised (the law is zero elsewhere,
    so p and H are those of the dense law).
    """
    ps, hs = [], []
    K = _topk_of(prm)
    if K is not None:
        kind, x, vs = prm
        v, ix = torch.topk(rows[:, :vs].float(), K, dim = -1)
        sub = (kind, x, K)
        dr = torch.tensor(drafts, dtype = ix.dtype, device = ix.device).view(-1, 1)
        hit = (ix == dr)
    for i, d in enumerate(drafts):
        if K is not None:
            pl = law(v[i], sub)
            ps.append((pl * hit[i]).sum())
        else:
            pl = law(rows[i], prm)
            ps.append(pl[d])
        hs.append(-(pl * torch.log(pl.clamp_min(1e-30))).sum())
    p = torch.stack(ps)
    h = torch.stack(hs)
    ok = (p > 0) & (p >= torch.clamp(delta * torch.exp(-h), max = eps))
    if TYP_HMAX is not None:
        ok = ok & (h <= TYP_HMAX)
    out = torch.stack([p, h, ok.float()]).tolist()
    return out[0], out[1], [x > 0.5 for x in out[2]]
