#!/usr/bin/env python3
import json, sys
p = sys.argv[1] if len(sys.argv) > 1 else "/tmp/bt/final.jsonl"
rows = [json.loads(l) for l in open(p) if l.strip()]
print(f"{len(rows)} results")
for r in rows[:4]:
    t = " ".join(r["text"].strip().split())
    print(f"[{r['index']}] tokens={r['tokens']} eos={r.get('eos_reason')}")
    print(f"    Q: {r['prompt'][:70]}")
    print(f"    A: {t[:200]}")
bad = [r for r in rows if not r["text"].strip()]
marks = [r for r in rows if "<|im_" in r["text"] or "<think>" in r["text"]]
print(f"\nempty outputs: {len(bad)}   outputs containing turn/think markers: {len(marks)}")
print(f"eos reasons: {sorted({r.get('eos_reason') for r in rows})}")
