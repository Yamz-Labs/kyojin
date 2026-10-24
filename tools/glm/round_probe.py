"""Opt-in per-round decode log for serve.py. Off unless EXL3_ROUND_LOG names a file.

One JSON line per decode round of a single active request (a round = one Generator.iterate() call
after the prefill, GPU-synced on both sides, so ms is the real round time, the cost of the two
syncs included):

  t         wall clock (time.time) at the round start; gaps between t+ms and the next t are time outside the rounds
  n         sequence length before the round (prompt + generated)
  new       job.new_tokens after the round
  produced  tokens the round added (accepted drafts + 1)
  window    drafted tokens the round verified (rows - 1); None = plain 1-row step
  acc       accepted draft tokens (from job.draft_stats)
  kind      "ng" for an n-gram lookup round, "dr" for the MTP drafter's own round
  ms        round wall time
  prof      EXL3_SPEC_PROF=1 only: the Generator's per-phase times (t_dfwd, t_verify, t_post, t_kvup, R)

A line {"req": ...} marks the start of every request (first decode round of a new job).
Rounds of batched requests (> 1 active) are not logged. The file is line-buffered: a client on the
same box reads it by byte offset between requests (tools/glm/round_replay.py).
"""
import json
import os
import time

import torch


def install(g, path, lookup_stats=None):
    f = open(path, "a", buffering=1)
    orig = g.iterate
    st = {"job": None}

    def sync():
        torch.cuda.synchronize()

    def iterate():
        act = g.active_jobs
        job = act[0] if len(act) == 1 and act[0].is_prefill_done() and len(act[0].sequences) == 1 else None
        if job is None:
            return orig()
        n = len(job.sequences[0].sequence_ids)
        ng0 = (lookup_stats or {}).get("ng", 0)
        ds0 = len(job.draft_stats)
        n0 = job.new_tokens
        if st["job"] is not job:
            st["job"] = job
            f.write(json.dumps({"req": id(job), "t": time.time(), "n": n}) + "\n")
        prof_mod = None
        if os.environ.get("EXL3_SPEC_PROF") == "1":
            from exllamav3.generator import generator as GM
            prof_mod = GM
            GM._PROF_CUR.clear()
        sync()
        t0 = time.perf_counter()
        tw = time.time()
        out = orig()
        sync()
        ms = (time.perf_counter() - t0) * 1e3
        rec = {"t": round(tw, 4), "n": n, "new": job.new_tokens, "produced": job.new_tokens - n0, "ms": round(ms, 3),
               "kind": "ng" if (lookup_stats or {}).get("ng", 0) > ng0 else "dr"}
        if len(job.draft_stats) > ds0:
            _, w, a = job.draft_stats[-1]
            rec["window"], rec["acc"] = int(w), int(a)
        else:
            rec["window"] = rec["acc"] = None
        if prof_mod is not None:
            rec["prof"] = {k: (round(v * 1e3, 3) if isinstance(v, float) else v) for k, v in prof_mod._PROF_CUR.items()}
        f.write(json.dumps(rec) + "\n")
        return out

    g.iterate = iterate
    return f
