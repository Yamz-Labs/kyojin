#!/usr/bin/env python3
"""Answer-phase loop guard gate: 0 false stops on the legit corpus, every loop of the loop corpus stopped.

    loopguard_gate.py qwen=/path/tokenizer.json glm=/path/tokenizer.json [--chunks 1,3,4] [--old]

Each text is replayed through loopguard.LoopMonitor("answer") with the real tokenizer of the model family, a few tokens per
event (speculative decoding delivers several). --old replays with the release-gate rule alone (the pre-fix answer rule).
CPU only. Exit code 1 when the gate fails."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import loopguard_corpus
import serve
import loopguard


def replay(tok, text: str, chunk: int):
    enc = tok.encode(text)
    ids, offs = enc.ids, enc.offsets
    mon = loopguard.LoopMonitor("answer", None)
    cur = 0          # byte-level BPE offsets overlap inside a multi-byte character: stream each character once
    for i in range(0, len(ids), chunk):
        j = min(i + chunk, len(ids))
        end = max(cur, offs[j - 1][1])
        hit = mon.feed(ids[i:j], text[cur:end])
        cur = end
        if hit:
            return hit, len(ids)
    return None, len(ids)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tokenizers", nargs="+", help="name=path/to/tokenizer.json")
    ap.add_argument("--chunks", default="1,3,4")
    ap.add_argument("--old", action="store_true")
    args = ap.parse_args()
    if args.old:
        loopguard.answer_text_loop = loopguard.text_loop
    from tokenizers import Tokenizer
    legit, loops = loopguard_corpus.legit(), loopguard_corpus.loops()
    failed = False
    print(f"{'family':8s} {'chunk':>5s} {'loops stopped':>14s} {'legit false stops':>18s}")
    for spec in args.tokenizers:
        name, path = spec.split("=", 1)
        tok = Tokenizer.from_file(path)
        for chunk in (int(c) for c in args.chunks.split(",")):
            miss = [n for n, t in loops.items() if replay(tok, t, chunk)[0] is None]
            fp = [(n, h["kind"], h["at"]) for n, t in legit.items() if (h := replay(tok, t, chunk)[0])]
            print(f"{name:8s} {chunk:5d} {len(loops) - len(miss):>8d}/{len(loops):<5d} {len(fp):>12d}/{len(legit)}")
            for n in miss:
                print("   not stopped:", n)
            for f in fp:
                print("   FALSE STOP:", f)
            failed |= bool(miss or fp)
    print("GATE", "FAIL" if failed else "PASS")
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
