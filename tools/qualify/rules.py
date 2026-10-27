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


KV_DEFAULT_FORMATS = ("k6v6", "kw6")    # tq6 / tq5 are measured and reported but stay opt-in (--kv-mode tq) in 1.8.1
KV_WIN_SPEC = "4:128"                   # kw6 = k6v6 + the fp16 sink-4 / window-128 ring


def choose_kv_mode(kl: dict[str, float], bits_per_val: dict[str, float], base: str = "k6v6", margin: float = KV_MODE_MARGIN,
                   allowed: tuple = KV_DEFAULT_FORMATS) -> str:
    """Q6 rule for the KV format: any candidate (kw6, and with `allowed` extended tq6, tq5) whose decode-faithful KL vs k8v8 is <= margin x the
    k6v6 baseline's qualifies; the one with the fewest bytes per value wins (ties: lower KL), else k6v6. kl / bits_per_val: {format: value},
    both holding `base`. The wider-must-qualify check looks at every measured format, `allowed` or not."""
    ok = [n for n in kl if n != base and n in bits_per_val and kl[n] <= margin * kl[base]]
    # sanity: more bits can only help, so a format with fewer bits that beats the baseline while a wider tq format does not is measurement noise
    ok = [n for n in ok if all(m in ok for m in kl if m != base and bits_per_val.get(m, 0) > bits_per_val[n])]
    ok = [n for n in ok if n in allowed]
    return min(ok, key=lambda n: (bits_per_val[n], kl[n])) if ok else base


def verdict(q: dict, gates: dict | None = None) -> dict:
    """q: assembled qualify data. Returns {pass, reasons[], unmeasured[]}. Missing measurements are listed, never counted as pass."""
    g = dict(kl_mean=0.15, top1_min=0.75, ctx_min=32768, smoke_min=0.8)
    g.update(gates or {})
    reasons, unmeasured = [], []
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
            reasons.append("greedy sha differs between two identical runs")
        if sv.get("oom"):
            reasons.append("out of memory in the server log")
        if sv.get("conformance") and not sv["conformance"].get("ok", False):
            reasons.append("template/tool conformance failed: " + sv["conformance"].get("detail", ""))
    else:
        unmeasured.append("serve")
    sm = q.get("smoke")
    if sm:
        if sm["total"] and sm["passed"] / sm["total"] < g["smoke_min"]:
            reasons.append(f"smoke {sm['passed']}/{sm['total']} < {g['smoke_min']:.0%}")
    else:
        unmeasured.append("smoke")
    return dict(pass_=not reasons and not unmeasured, reasons=reasons, unmeasured=unmeasured)
