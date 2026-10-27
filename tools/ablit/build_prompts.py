#!/usr/bin/env python3
"""build_prompts.py -- fetch + normalize ablit3 prompt sets (CPU only).

Harmful (~500): AdvBench + HarmBench (public GitHub CSVs) + JailbreakBench
  JBB-Behaviors/harmful (HF, non-gated). The walledai HF mirrors are gated;
  the GitHub CSVs are the same upstream content.
Harmless (~500): XSTest safe rows (GitHub CSV) + Alpaca (HF tatsu-lab/alpaca).

Sampling: deterministic (seed 0), stratified hold-out of 100 per class.
Output (default scratch/ablit/prompts/):
  harmful_collect.jsonl (400) / harmful_heldout.jsonl (100)
  harmless_collect.jsonl (400) / harmless_heldout.jsonl (100)
  SOURCES.md (source URLs + licenses)
Row format: {"text": ..., "source": "advbench|harmbench|jbb|xstest|alpaca", "split": ...}
"""
import argparse
import csv
import io
import json
import os
import random
import urllib.request

URLS = {
    "advbench": "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv",
    "harmbench": "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv",
    "xstest": "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv",
}

SOURCES_MD = """# ablit3 prompt sources

- AdvBench harmful behaviors (`goal` column): llm-attacks/llm-attacks data,
  <https://github.com/llm-attacks/llm-attacks> (repo license MIT).
  URL: <{advbench}>
- HarmBench standard text behaviors (`Behavior` column):
  centerforaisafety/HarmBench,
  <https://github.com/centerforaisafety/HarmBench> (repo license MIT).
  URL: <{harmbench}>
- JailbreakBench JBB-Behaviors harmful split (`Goal` column):
  HF `JailbreakBench/JBB-Behaviors` config `behaviors` split `harmful`
  (project license MIT), fetched via `datasets`.
- XSTest prompts, safe rows only (`prompt` where `label` == safe):
  paul-rottger/xstest, <https://github.com/paul-rottger/xstest>
  (LICENSE = CC-BY-4.0).
  URL: <{xstest}>
- Alpaca instructions (`instruction` + `input`):
  HF `tatsu-lab/alpaca` (dataset card license CC BY-NC 4.0), fetched via `datasets`.
"""


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "ablit3-prompt-builder"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8")


def get_advbench(n):
    rows = list(csv.DictReader(io.StringIO(fetch(URLS["advbench"]))))
    return [{"text": r["goal"].strip(), "source": "advbench"} for r in rows
            if r.get("goal", "").strip()][:n]


def get_harmbench(n):
    rows = list(csv.DictReader(io.StringIO(fetch(URLS["harmbench"]))))
    return [{"text": r["Behavior"].strip(), "source": "harmbench"} for r in rows
            if r.get("Behavior", "").strip()][:n]


def get_jbb(n):
    from datasets import load_dataset
    ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
    return [{"text": r["Goal"].strip(), "source": "jbb"} for r in ds
            if r.get("Goal", "").strip()][:n]


def get_xstest(n):
    rows = list(csv.DictReader(io.StringIO(fetch(URLS["xstest"]))))
    safe = [{"text": r["prompt"].strip(), "source": "xstest"} for r in rows
            if r.get("label") == "safe" and r.get("prompt", "").strip()]
    return safe[:n]


def get_alpaca(n, seed):
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    rng = random.Random(seed)
    idx = rng.sample(range(len(ds)), n * 2)  # oversample, then filter short
    out = []
    for i in idx:
        r = ds[int(i)]
        t = (r.get("instruction", "") + (" " + r["input"] if r.get("input") else "")).strip()
        if len(t.split()) >= 4:
            out.append({"text": t, "source": "alpaca"})
        if len(out) >= n:
            break
    return out


def split_holdout(items, n_hold, seed):
    rng = random.Random(seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    hold = {i for i in idx[:n_hold]}
    return [x for i, x in enumerate(items) if i not in hold], \
           [x for i, x in enumerate(items) if i in hold]


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="scratch/ablit/prompts")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)

    adv = get_advbench(220)
    hb = get_harmbench(220)
    jbb = get_jbb(100)
    rng.shuffle(adv)
    rng.shuffle(hb)
    # 200 adv + 140 harmbench + 60 jbb = 400 collect; holdout 20/60/20... use
    # stratified: hold out 100 total proportional to source counts below.
    harm_pool = adv[:250] + hb[:180] + jbb[:100]  # 530 -> trim
    harm_pool = harm_pool[:500]
    xs = get_xstest(200)
    al = get_alpaca(320, a.seed)
    rng.shuffle(xs)
    rng.shuffle(al)
    safe_pool = (xs[:200] + al[:300])[:500]

    assert len(harm_pool) == 500, len(harm_pool)
    assert len(safe_pool) == 500, len(safe_pool)

    harm_c, harm_h = split_holdout(harm_pool, 100, a.seed)
    safe_c, safe_h = split_holdout(safe_pool, 100, a.seed + 1)

    def tag(rows, split):
        for r in rows:
            r["split"] = split
        return rows

    write_jsonl(f"{a.out}/harmful_collect.jsonl", tag(harm_c, "collect"))
    write_jsonl(f"{a.out}/harmful_heldout.jsonl", tag(harm_h, "heldout"))
    write_jsonl(f"{a.out}/harmless_collect.jsonl", tag(safe_c, "collect"))
    write_jsonl(f"{a.out}/harmless_heldout.jsonl", tag(safe_h, "heldout"))
    with open(f"{a.out}/SOURCES.md", "w") as f:
        f.write(SOURCES_MD.format(**URLS))

    from collections import Counter
    print("harmful collect:", Counter(r["source"] for r in harm_c),
          "heldout:", Counter(r["source"] for r in harm_h))
    print("harmless collect:", Counter(r["source"] for r in safe_c),
          "heldout:", Counter(r["source"] for r in safe_h))
    print(f"wrote 4 jsonl + SOURCES.md under {a.out}")


if __name__ == "__main__":
    main()
