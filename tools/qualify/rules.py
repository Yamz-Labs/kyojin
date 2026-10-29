"""Decision rules (pure). Owner rules: KV cache Q6 minimum (25/10 19:55); gate = measured values, advisory verdict."""
from __future__ import annotations

KV_MIN_BITS = 6
KV_KL_MAX = 0.02      # KL(k<b> vs k8v8) accepted for the KV choice (drill 0: Nanbeige k6v6 0.014 ok, k4v4 0.123 refused)
CANDIDATES = (6, 7, 8)


def choose_kv(kl_vs_k8: dict[int, float], kl_max: float = KV_KL_MAX) -> tuple[int, int]:
    """Smallest symmetric bits >= Q6 whose KL against k8v8 is within kl_max; k8v8 when none passes. kl_vs_k8: {bits: KL}."""
    for b in CANDIDATES:
        if b >= 8:
            break
        if b in kl_vs_k8 and kl_vs_k8[b] <= kl_max:
            return b, b
    return 8, 8


KV_MODE_MARGIN = 0.9    # one 256-step doc is noisy (two docs of one model scatter by 10-30 %): a candidate has to beat k6v6 by this factor


KV_TQ_MARGIN = 0.9      # same margin as kw6: a 256-step doc scatters 10-30 %, a tie is a coin flip, so a tq format must clearly beat k6v6 to be chosen
KV_DEFAULT_FORMATS = ("k6v6", "kw6")    # tq6 / tq5 are measured and reported but stay opt-in (QUALIFY_KV_TQ=1): the gate "tq6 >= default on every model" is not met yet
KV_WIN_SPEC = "4:128"                   # kw6 = k6v6 + the fp16 sink-4 / window-128 ring


KV_MODE_MAX_SLOWDOWN = 1.10    # a format whose decode step is > 10 % slower than k6v6's is not a free quality win (Qwen3.5-0.8B on ROCm: kw6 0.0090 vs 0.0068 s/tok)


def choose_kv_mode(kl: dict[str, float], bits_per_val: dict[str, float], base: str = "k6v6", margin: float = KV_MODE_MARGIN,
                   allowed: tuple = KV_DEFAULT_FORMATS, dec_s: dict | None = None, max_slowdown: float = KV_MODE_MAX_SLOWDOWN,
                   tq_margin: float = KV_TQ_MARGIN) -> str:
    """Q6 rule for the KV format: a candidate whose decode-faithful KL vs k8v8 is <= its margin x the k6v6 baseline's qualifies (kw6: `margin`, the
    ring is an extra only when it clearly helps; tq6 / tq5: `tq_margin`, the owner floor of "never worse than k6v6"); the one with the fewest bytes
    per value wins (ties: lower KL), else k6v6. kl / bits_per_val: {format: value}, both holding `base`; a format missing from kl was not measured
    (e.g. the model's attention cannot hold a TurboQuant cache). The wider-must-qualify check looks at every measured format, `allowed` or not."""
    lim = lambda n: (tq_margin if n.startswith("tq") else margin) * kl[base]
    ok = [n for n in kl if n != base and n in bits_per_val and kl[n] <= lim(n)]
    # sanity: more bits can only help, so a format with fewer bits that beats the baseline while a wider tq format does not is measurement noise
    ok = [n for n in ok if all(m in ok for m in kl if m != base and bits_per_val.get(m, 0) > bits_per_val[n])]
    ok = [n for n in ok if n in allowed]
    if dec_s and dec_s.get(base):
        ok = [n for n in ok if not dec_s.get(n) or dec_s[n] <= max_slowdown * dec_s[base]]
    return min(ok, key=lambda n: (bits_per_val[n], kl[n])) if ok else base


DRAFT_CHAT_MIN = 0.97      # the adaptive policy must never leave chat slower than plain decode (box-lead 26/10 08:35 gate)
DRAFT_GAIN_MIN = 1.10      # best kind (code / agentic) must gain at least this to justify the drafter's VRAM (2.2 GiB on the 3090)


def choose_draft(speedup: dict[str, float], chat_min: float = DRAFT_CHAT_MIN, gain_min: float = DRAFT_GAIN_MIN) -> tuple[bool, str]:
    """speedup: {kind: drafted tok/s / plain tok/s} (kinds code / chat / agentic). -> (use_drafter, reason)."""
    if not speedup:
        return False, "no measurement"
    chat = speedup.get("chat")
    if chat is not None and chat < chat_min:
        return False, f"chat {chat:.2f}x < {chat_min}"
    best = max(speedup.values())
    if best < gain_min:
        return False, f"best kind {best:.2f}x < {gain_min}"
    return True, f"best {best:.2f}x, chat {fmt_x(chat)}"


def fmt_x(x) -> str:
    return "-" if x is None else f"{x:.2f}x"


def verdict(q: dict, gates: dict | None = None) -> dict:
    """q: assembled qualify data. Returns {pass, reasons[], unmeasured[]}. Missing measurements are listed, never counted as pass."""
    g = dict(kl_mean=0.15, top1_min=0.75, ctx_min=32768, smoke_min=0.8)
    g.update(gates or {})
    reasons, unmeasured, warnings = [], [], []
    par = q.get("parity")
    if par:
        if par["kl_mean"] > g["kl_mean"]:
            reasons.append(f"mean KL {par['kl_mean']:.3f} > {g['kl_mean']}")
        for d, v in par["by_domain"].items():
            if v["top1"] < g["top1_min"]:
                reasons.append(f"domain {d} top-1 {v['top1']:.3f} < {g['top1_min']}")
    else:
        unmeasured.append("parity")
    kv = q.get("kv")
    if kv:
        if min(kv["chosen"]) < KV_MIN_BITS and not kv.get("q6_equivalent"):
            reasons.append(f"KV below Q6: {kv['chosen']}")
    else:
        unmeasured.append("kv")
    mc = q.get("maxctx")
    if mc:
        c1 = mc.get("1")
        if not c1 or c1 < g["ctx_min"]:
            reasons.append(f"max ctx for 1 session {c1} < {g['ctx_min']}")
    else:
        unmeasured.append("maxctx")
    sv = q.get("serve")
    if sv:
        if not sv.get("sha_self_equal", False):
            nd, tot = sv.get("sha_n_diff"), len(sv.get("sha") or [])
            if nd is not None and tot and 0 < nd < tot:      # the repeat runs on a warm prefix cache and, with a drafter, a different verify batch: near-tie numerics (REPORT-27 class)
                warnings.append(f"greedy sha differs in {nd}/{tot} prompts between the cold run and the repeat (near-tie numerics of the cached-prefix / R-row verify path)")
            else:
                reasons.append("greedy sha differs between two identical runs")
        if sv.get("oom"):
            reasons.append("out of memory in the server log")
        if sv.get("conformance") and not sv["conformance"].get("ok", False):      # the vendor chose its own order and trained on it: a warning, not a gate
            warnings.append("template renders the system text before the tools (" + sv["conformance"].get("detail", "") + "): sessions with different system prompts do not share the tool-schema prefix cache")
    else:
        unmeasured.append("serve")
    sm = q.get("smoke")
    if sm:
        if sm["total"]:       # graded suite: the mean score decides (a partial 0.4 on a trap task is not a failure of the lane); scripted fallback: pass rate
            val = sm["score_mean"] if sm.get("score_mean") is not None else sm["passed"] / sm["total"]
            if val < g["smoke_min"]:
                reasons.append(f"smoke {'mean score' if sm.get('score_mean') is not None else 'pass rate'} {val:.2f} ({sm['passed']}/{sm['total']}) < {g['smoke_min']:.0%}")
    else:
        unmeasured.append("smoke")
    return dict(pass_=not reasons and not unmeasured, reasons=reasons, unmeasured=unmeasured, warnings=warnings)
