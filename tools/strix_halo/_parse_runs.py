#!/usr/bin/env python3
import glob, re, os, sys
root = sys.argv[1] if len(sys.argv) > 1 else "/tmp/recipe_validate"
pat = re.compile(
    r"Context: (\d+) new tokens at ([\d.]+) t/s.*Generate: (\d+) tokens at ([\d.]+) t/s.*accepted \(([\d.]+)%\)"
)
for p in sorted(glob.glob(root + "/run_*.raw")):
    t = re.sub(r"\x1b\[[0-9;]*m", "", open(p, errors="replace").read())
    m = list(pat.finditer(t))
    name = os.path.basename(p)
    if m:
        x = m[-1]
        print(f"{x.group(4):>7} t/s  acc={x.group(5):>6}%  prefill={x.group(2):>7}  {name}")
    else:
        oom = "OOM" if "Insufficient" in t else "NO-MATCH"
        print(f"   FAIL {oom:8}  {name}")
