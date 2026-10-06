"""microbench: served dt DSA decode split vs variants, GLM decode shapes (bench_dsa_split.build).
Each arm = env overrides applied before the call (dsa_triton reads them per call).
Prints median / amortized ms and relL2 vs the torch reference per (ctx, R).
usage: python tools/glm/decattn2_mb.py [--ctx 4096,32768] [--rows 1,2] [--reps 200]
       --arms 'name=ENV:VAL|ENV2:VAL2;name2=...' (empty arm 'base=' = served default)"""
import argparse, os, sys
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "tools/glm"))
import torch
import bench_dsa_split as B
from exllamav3.modules.attention_fn import dsa_triton as D

ap = argparse.ArgumentParser()
ap.add_argument("--ctx", default="4096,32768")
ap.add_argument("--rows", default="1,2")
ap.add_argument("--reps", type=int, default=200)
ap.add_argument("--arms", default="base=")
ap.add_argument("--splits", type=int, default=0)
ap.add_argument("--compress", type=int, default=1, help="pool entries = ctx / compress (served probe: pool_len = ctx -> 1)")
a = ap.parse_args()
B.COMPRESS = a.compress

arms = []
for spec in a.arms.split(";"):
    if not spec.strip():
        continue
    name, _, envs = spec.partition("=")
    arms.append((name.strip(), [tuple(e.split(":", 1)) for e in envs.split("|") if e]))
keys = {k for _, e in arms for k, _ in e}
print(f"dev={torch.cuda.get_device_name(0)} arms={[n for n, _ in arms]}", flush=True)
print(f"{'arm':<22} {'ctx':>6} {'R':>2} {'ms':>7} {'amort':>7} {'relL2':>9} {'ratio':>6}", flush=True)
for ctx in [int(x) for x in a.ctx.split(",")]:
    for R in [int(x) for x in a.rows.split(",")]:
        t = B.build(ctx, R)
        if a.splits:
            t[1]["n_splits"] = a.splits
        ref = B.ref_attn(t)
        base_am = None
        for name, envs in arms:
            for k in keys:
                os.environ.pop(k, None)
            for k, v in envs:
                os.environ[k] = v
            try:
                out = D.dsa_attn(*t[0], **t[1]).float()
                rel = B.rel_l2(out, ref)
                ms, am = B.kernel_ms(t, reps=a.reps)
            except Exception as ex:
                print(f"{name:<22} {ctx:>6} {R:>2} FAIL {type(ex).__name__}: {str(ex)[:120]}", flush=True)
                continue
            base_am = base_am or am
            print(f"{name:<22} {ctx:>6} {R:>2} {ms:>7.3f} {am:>7.3f} {rel:>9.2e} {am / base_am:>6.2f}", flush=True)
        for k in keys:
            os.environ.pop(k, None)
        del t, ref
        torch.cuda.empty_cache()
