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
    """Inverse-CDF draw from a dense (unnormalized) law with a uniform u in [0, 1); never returns a zero-mass index."""
    cs = torch.cumsum(p, 0)
    idx = torch.searchsorted(cs, (cs[-1:] * u), right = True)
    return idx.clamp(max = p.numel() - 1)


def verify(logits: torch.Tensor, q: torch.Tensor, d: int, prm: tuple, rng) -> torch.Tensor:
    """
    One verify position: target logits (padded vocabulary), the draft law q and the drafted token d (d ~ q). Returns
    the emitted token as a (1, 1) long tensor: d if kept, else a draw from norm(max(0, p - q)) (never d).
    """
    p = law(logits, prm)
    u1, u2 = rng.random(), rng.random()
    keep = u1 * q[d] < p[d]
    r = draw((p - q).clamp_(min = 0.0), u2)
    return torch.where(keep, torch.full_like(r, d), r).view(1, 1)
