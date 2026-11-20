"""kv10_driver.py (cwd = tree): paired quality of the 1.9.2 KV levers HY4_IX_POOL16 (fp16 indexer pool) + HY4_ROPE_POS (rope rows from positions) against the compact arm.
Two processes (allocation flags cannot be toggled inside one load), both with HY4_KV_COMPACT=1:
  A  HY4_IX_POOL16=0 HY4_ROPE_POS=0  ref log-probs + 3 noise nulls + the top-k SET overlap of the fp32 pool vs the same pool in fp16 on the real decode queries (qual_kv10.py)
  B  as shipped (HY4_KV_COMPACT=1 sets both levers)  paired dNLL / KL / top-1 vs A, short window (w4k: 2048 context rows) and long window (w32k: 31744 context rows)
Prints FIRES kv10 / kv10_* from both arms (a flagged lever with 0 firings fails the job: rc 1) and the lines KV10_GATE topk (mean / min overlap of the fp32-pool vs fp16-pool top-k group sets) and KV10_GATE short|long: KL / noise-null max <= 1.0 and |dNLL| <= null max.
env: LLC_CTX (default 65536), LLC_REF (scratch file of ~0.8 GB for the ref log-probs, deleted by arm B), LLC_CORPUS (text file), GGUF, DRV_GUARD_DIR."""
import os, re, subprocess, sys
sys.path.append(os.environ["DRV_GUARD_DIR"])            # directory of the job runner's shared drv_guard.py (dead / zero-sample arm = fail at once)
import drv_guard as guard
guard.install_excepthook()
E, tree = os.environ, os.getcwd()
base = dict(E, LLC_CTX=E.get("LLC_CTX", "65536"), LLC_REF=E["LLC_REF"], HY4_KV_COMPACT="1")


def arm(mode, extra):
    out = []
    p = subprocess.Popen([sys.executable, "-u", tree + "/tools/hy4/step/qual_kv10.py"], env={**base, "LLC_MODE": mode, **extra}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for ln in p.stdout:
        sys.stdout.write(ln)
        sys.stdout.flush()
        out.append(ln.rstrip("\n"))
    p.wait()
    print(f"KV10_ARM {mode} rc={p.returncode}", flush=True)
    return p.returncode, out


rcA, outA = arm("A", {"HY4_IX_POOL16": "0", "HY4_ROPE_POS": "0"})
if rcA:
    print("DRIVER_FAIL arm A", flush=True)
    sys.exit(1)
rcB, outB = arm("B", {})
qres = sum(ln.startswith("QRES") for ln in outB)
fires = {m.group(1): int(m.group(2)) for ln in outA + outB for m in [re.match(r"FIRES (kv10_\w+) (\d+)", ln)] if m}
for ln in outB:
    m = re.match(r"QKL_VS_NOISE (\w+) kl ([\d.eE-]+) x noise-null max ([\d.]+) \| \|dNLL\| ([\d.eE-]+) vs null max ([\d.eE-]+)", ln)
    if m:
        name, x, dn, dmax = {"w4k": "short", "w32k": "long"}.get(m.group(1), m.group(1)), float(m.group(3)), float(m.group(4)), float(m.group(5))
        print(f"KV10_GATE {name} kl_x_null {x} dNLL {dn} null_max {dmax} -> {'PASS' if x <= 1.0 and dn <= dmax else 'FAIL'}", flush=True)
TOPK_MEAN_MIN = float(E.get("KV10_TOPK_MEAN_MIN", "0.99"))        # thresholds anchored on the random-data CPU check (tests/test_kv10.py: mean 0.9998, min 0.998 at 4K .. 64K groups) with margin; a pool in fp16 moves only the groups next to the cut-off
TOPK_ROW_MIN = float(E.get("KV10_TOPK_ROW_MIN", "0.97"))
for ln in outA:
    m = re.match(r"IXOVL rows (\d+) mean_overlap ([\d.]+) min_overlap ([\d.]+)", ln)
    if m:
        mean, mn = float(m.group(2)), float(m.group(3))
        print(f"KV10_GATE topk rows {m.group(1)} mean_overlap {mean} min_overlap {mn} -> {'PASS' if mean >= TOPK_MEAN_MIN and mn >= TOPK_ROW_MIN else 'FAIL'} (mean >= {TOPK_MEAN_MIN}, min >= {TOPK_ROW_MIN})", flush=True)
if rcB:
    print("DRIVER_FAIL arm B (a flagged lever did not fire or the arm crashed)", flush=True)
    sys.exit(1)
guard.require_samples({"qres_B": qres, "fires_kv10": fires.get("kv10", 0), "fires_kv10_pool16": fires.get("kv10_pool16", 0), "fires_kv10_ovl": fires.get("kv10_ovl", 0)}, label="kv10_driver")
print("KV10_DONE", flush=True)
