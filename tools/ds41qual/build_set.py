#!/usr/bin/env python3
"""Build the fixed DS4.1 gate eval set: held-out agentic-style rows (tool calls, code, long multi-turn transcripts).

  python3 tools/ds41qual/build_set.py --tok TOKENIZER.json [--prompts DIR] [--calib calib_ds41.npy] [--french-ids eval_ids.npy]

Writes data/gate_ids.npz (row000..: int64 token ids, BOS first) and data/gate_meta.json (names, lengths, sha256).
Rows never come from the calibration mix (public wiki / c4 / code / UltraChat / Hermes tool sets): the 24-gram overlap with
--calib is measured and the build fails above 1 %.
"""
import argparse, glob, hashlib, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools", "ds41"))
BOS = 0

ap = argparse.ArgumentParser()
ap.add_argument("--tok", required=True)
ap.add_argument("--prompts", default=os.path.expanduser("~/ds41_work/rel1510/eval/prompts"))
ap.add_argument("--calib", default=os.path.expanduser("~/ds41_work/ds41calib/data/calib_ds41.npy"))
ap.add_argument("--french-ids", default=os.path.expanduser("~/ds41_work/ds41ref/data/eval_ids.npy"))
ap.add_argument("--french-row", type=int, default=10)
ap.add_argument("--out", default=os.path.join(HERE, "data"))
a = ap.parse_args()

from tokenizers import Tokenizer
import ds41_chat
tk = Tokenizer.from_file(a.tok)


def render(name):
    d = json.load(open(os.path.join(a.prompts, name)))
    msgs = d["messages"]
    for m in msgs:                                   # tool-call arguments arrive as JSON strings or objects
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", tc)
            if isinstance(fn.get("arguments"), str):
                try:
                    fn["arguments"] = json.loads(fn["arguments"])
                except Exception:
                    fn["arguments"] = {"raw": fn["arguments"]}
    text = ds41_chat.render_chat(msgs, d.get("tools"), thinking=False)
    return tk.encode(text, add_special_tokens=False).ids


def tail(ids, n):
    ids = [i for i in ids if i != BOS]
    assert len(ids) >= n - 1, (len(ids), n)
    return [BOS] + ids[-(n - 1):]


rows, names = [], []
def add(name, ids):
    rows.append(np.array(ids, dtype=np.int64)); names.append(name)

for n in ("agentic_code_01", "agentic_curl_05", "agentic_code_10"):
    add(f"agentic:{n}:2048", tail(render(n + ".json"), 2048))
for n in ("agentic_code_20", "agentic_curl_10"):
    add(f"agentic:{n}:4096", tail(render(n + ".json"), 4096))
long_ids = []
for n in ("agentic_curl_16", "agentic_code_29"):
    long_ids += render(n + ".json")
add("agentic_long:curl_16+code_29:8192", tail(long_ids, 8192))
src = ""
for f in ("ds41_model.py", "serve.py"):
    src += open(os.path.join(ROOT, "tools", "ds41", f)).read()
add("code:ds41_serving_sources:2048", tail(tk.encode(src, add_special_tokens=False).ids, 2048))
if os.path.exists(a.french_ids):
    fr = np.load(a.french_ids)[a.french_row].astype(np.int64)
    add(f"french:ds41ref_row{a.french_row}:{len(fr)}", fr.tolist())

# held-out check against the calibration mix (24-gram shingles)
if os.path.exists(a.calib):
    cal = np.load(a.calib)
    G = 24
    seen = set()
    for r in cal:
        for i in range(0, len(r) - G, 4):
            seen.add(hash(r[i:i + G].tobytes()))
    hit = tot = 0
    for r in rows:
        for i in range(0, len(r) - G, 1):
            tot += 1
            hit += hash(r[i:i + G].tobytes()) in seen
    frac = hit / max(tot, 1)
    print(f"24-gram overlap with calibration: {frac:.4%}")
    assert frac < 0.01, "eval rows overlap the calibration mix"

os.makedirs(a.out, exist_ok=True)
np.savez_compressed(os.path.join(a.out, "gate_ids.npz"), **{f"row{i:03d}": r for i, r in enumerate(rows)})
sha = hashlib.sha256(b"".join(r.astype(np.int32).tobytes() for r in rows)).hexdigest()
json.dump({"names": names, "lengths": [len(r) for r in rows], "bos_id": BOS, "start": 64, "sha256": sha,
           "total_tokens": int(sum(len(r) for r in rows))}, open(os.path.join(a.out, "gate_meta.json"), "w"), indent=1)
for n, r in zip(names, rows):
    print(f"{len(r):6d} {n}")
print("total", sum(len(r) for r in rows), "sha", sha[:12])
