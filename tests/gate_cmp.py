"""gate_cmp.py REF.pt NEW.pt : REF = qdec.hip off, NEW = qdec.hip on run with --force REF.pt. Prints per prompt: greedy ids identical / first
divergence (+ REF top1-top2 gap there), teacher-forced top-1 agreement and KL(REF||NEW) over the window."""
import sys, torch
R, N = torch.load(sys.argv[1]), torch.load(sys.argv[2])
for i, (r, n) in enumerate(zip(R["res"], N["res"])):
    g, h = r["gen"], n["gen"]
    L = min(len(g), len(h)); div = next((k for k in range(L) if g[k] != h[k]), None)
    msg = f"prompt {i}: {'IDENTICAL' if div is None and len(g) == len(h) else 'DIVERGES at ' + str(div)}"
    if div is not None:
        top2 = r["lp"][div].float().topk(2); msg += f" (ref top1-top2 logprob gap {float(top2.values[0] - top2.values[1]):.3f}; ref tok {g[div]} new tok {h[div]})"
    p, q = r["lp"].float(), n["lp_forced"].float()
    m = min(len(p), len(q)); p, q = p[:m], q[:m]
    pe = p.exp()
    kl = torch.where(pe > 0, pe * (p - q), torch.zeros_like(p)).sum(-1)    # masked vocabulary (-inf in both) contributes 0
    agree = (p.argmax(-1) == q.argmax(-1)).float().mean().item()
    msg += f" | TF top1 agree {agree * 100:.1f}% KL mean {kl.mean().item():.2e} max {kl.max().item():.2e} steps {m}"
    print(msg)
