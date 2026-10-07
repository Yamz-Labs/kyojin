#!/usr/bin/env python
"""Deterministic synthetic English text for the long-context stages of pair_inv.py.

No external data: sentences are drawn from small built-in word lists with a fixed seed, so every
clone builds the same ~180K characters. Usage: make_corpus.py [out_file] [n_chars] [seed]
"""
import random, sys

_SUBJ = ["The harbour authority", "A small research team", "The regional council", "Our maintenance crew",
         "The night shift", "An independent auditor", "The village bakery", "A group of students",
         "The old lighthouse keeper", "The logistics office", "A travelling engineer", "The city archive",
         "The orchestra", "A local farmer", "The river commission", "The software team"]
_VERB = ["reviewed", "repaired", "catalogued", "measured", "rebuilt", "inspected", "compared", "scheduled",
         "documented", "relocated", "translated", "tested", "painted", "counted", "mapped", "restored"]
_OBJ = ["the northern warehouse", "a set of weather records", "the bridge over the canal", "forty wooden crates",
        "an unfinished timetable", "the east gate", "several hand-drawn maps", "the water pumps",
        "a box of old letters", "the main reading room", "three broken lamps", "the harvest figures",
        "a long list of invoices", "the signal tower", "the market square", "an archive of photographs"]
_WHEN = ["before sunrise", "after the first frost", "during the spring fair", "late on Tuesday", "in the third week of March",
         "once the rain stopped", "while the roads were closed", "at the end of the quarter", "on a quiet Sunday",
         "just before the audit", "in the middle of the festival", "when the power came back"]
_TAIL = ["and wrote down every detail", "so that nothing would be lost", "although the budget was tight",
         "because the old records were unclear", "and sent a copy to the council", "without telling anyone",
         "to prepare for the winter", "as the manager had asked", "even though the tools were worn",
         "with help from the neighbours", "and found two small mistakes", "before the deadline passed"]


def build(n_chars=180000, seed=20251007):
    rnd = random.Random(seed)
    out, n = [], 0
    while n < n_chars:
        para = []
        for _ in range(rnd.randrange(4, 9)):
            s = f"{rnd.choice(_SUBJ)} {rnd.choice(_VERB)} {rnd.choice(_OBJ)} {rnd.choice(_WHEN)} {rnd.choice(_TAIL)}."
            if rnd.random() < 0.5:
                s += f" The count came to {rnd.randrange(12, 9800)} items in {rnd.randrange(2, 40)} days."
            para.append(s)
        p = " ".join(para)
        out.append(p); n += len(p) + 2
    return "\n\n".join(out)[:n_chars]


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "corpus.txt"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 180000
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 20251007
    open(path, "w").write(build(n, seed))
