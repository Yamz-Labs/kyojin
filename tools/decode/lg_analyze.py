#!/usr/bin/env python
"""Offline analysis of dflash_check SAVE dumps: KL and top-k logit deltas between plain decode,
DFlash verify (perfect-draft arm) and the no-cache prefill reference, per position and by
row-in-round. Max|dlogit| over the full vocab is dominated by irrelevant tail logits; KL and the
delta on the reference's top-8 tokens are what decide greedy/acceptance."""
import glob
import sys

import torch


def kl(p_logits, q_logits):
    p, q = p_logits.float(), q_logits.float()
    ok = torch.isfinite(p) & torch.isfinite(q)
    lp = torch.log_softmax(p.masked_fill(~ok, -1e4), -1)
    lq = torch.log_softmax(q.masked_fill(~ok, -1e4), -1)
    return (lp.exp() * (lp - lq)).sum(-1)


def topk_delta(a, b, ref, k=8):
    idx = ref.float().topk(k, -1).indices
    return (a.float().gather(-1, idx) - b.float().gather(-1, idx)).abs().amax(-1)


def rows(stats):
    m = {}
    for end, w, a in stats or []:
        st = end - (a + 1)
        for i in range(a + 1):
            m[st + i] = i
    return m


for f in (sorted(glob.glob(sys.argv[1] + "/lg-*.pt")) if __name__ == "__main__" else []):
    d = torch.load(f)
    name = f.split("lg-")[-1][:-3]
    pl, ref = d["pl"], d["ref"]
    V = min(pl.shape[-1], ref.shape[-1]) if ref is not None else pl.shape[-1]
    n = len(d["pt"])
    print(f"== {name}")
    if ref is not None:
        k_pr = kl(ref[:n, :V], pl[:n, :V])
        t_pr = topk_delta(ref[:n, :V], pl[:n, :V], ref[:n, :V])
        agree = (ref[:n, :V].argmax(-1) == torch.tensor(d["pt"])).float().mean().item()
        print(f"  plain vs ref : KL mean {k_pr.mean():.4f} max {k_pr.max():.4f} | top8 dlogit mean "
              f"{t_pr.mean():.3f} max {t_pr.max():.3f} | ref-argmax==plain tok {agree:.3f}")
    if "bl" in d:
        bl, bt = d["bl"], d["bt"]
        div = next((i for i in range(min(n, len(bt))) if bt[i] != d["pt"][i]), min(n, len(bt)))
        if div == 0:
            continue
        k_pb = kl(pl[:div, :V], bl[:div, :V])
        t_pb = topk_delta(pl[:div, :V], bl[:div, :V], pl[:div, :V])
        print(f"  plain vs verify(perfect) upto div {div}: KL mean {k_pb.mean():.5f} max {k_pb.max():.5f} "
              f"| top8 dlogit mean {t_pb.mean():.3f} max {t_pb.max():.3f}")
        if ref is not None:
            k_rb = kl(ref[:div, :V], bl[:div, :V])
            print(f"  verify vs ref upto div: KL mean {k_rb.mean():.4f} max {k_rb.max():.4f} "
                  f"(plain vs ref same span: {k_pr[:div].mean():.4f} / {k_pr[:div].max():.4f})")
        rm = rows(d.get("bstats"))
        by = {}
        for i in range(div):
            by.setdefault(rm.get(i), []).append((k_pb[i].item(), t_pb[i].item()))
        print("  plain-vs-verify by row: " + " ".join(
            f"r{r}:KL{sum(x for x, _ in v)/len(v):.4f}/t8 {sum(y for _, y in v)/len(v):.3f}(n{len(v)})"
            for r, v in sorted(by.items(), key=lambda kv: -1 if kv[0] is None else kv[0])))
        worst = sorted(range(div), key=lambda i: -k_pb[i].item())[:6]
        print("  worst plain-vs-verify pos (pos,row,KL,top8d,KL plain-ref,KL verify-ref): " + str([
            (i, rm.get(i), round(k_pb[i].item(), 4), round(t_pb[i].item(), 3),
             round(k_pr[i].item(), 4) if ref is not None else None,
             round(kl(ref[i, :V], bl[i, :V]).item(), 4) if ref is not None else None) for i in worst]))
