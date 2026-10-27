#!/usr/bin/env python3
"""OpenAI-compatible, single-flight server for GLM-5.3 EXL3."""
from __future__ import annotations

import traceback
import argparse
import asyncio
import base64
import contextlib
import functools
import io
import json
import os
import re
import sys
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import aiohttp
from aiohttp import web
from jinja2.sandbox import ImmutableSandboxedEnvironment

sys.path.append(str(Path(__file__).resolve().parent.parent))
import startup_health  # noqa: E402  (GET /health with load progress while the model loads)
from loopguard import AnswerTap, loop_guard_default  # noqa: E402  (answer-phase loop guard shared with the Qwen server)
import serve_compat  # noqa: E402  (vLLM-style token-id prompts, cache_salt, /tokenize, usage details)
from l21_prompt_encode_cache import PromptEncodeCache  # noqa: E402  (incremental prompt encode)
# serve_metrics.py is a symlink to the shared tools/strix_halo/serve_metrics.py
from serve_metrics import Metrics
# serve_reqlog.py is a symlink to the shared tools/strix_halo/serve_reqlog.py: one line per request
# lifecycle event (numbers and event names only, never prompt or reply text) plus a heartbeat while a job
# is queued or prefilling, so "the client went silent" can be lined up with what the server was doing.
# Off unless KYOJIN_REQLOG_DIR names a directory; the server prints which of the two it is at start-up.
from serve_reqlog import Lifecycle, ReqLog, StreamWriter, refusal_text, sanitize
# serve_ping.py is a symlink to the shared tools/strix_halo/serve_ping.py: an SSE comment line every
# KYOJIN_SSE_PING_S while a stream has nothing to send yet (off by default), so the same silence is
# visible to the client and not only in the log file.
from serve_ping import ServePing
from image_cache import ImageEmbeddingCache, digest

DEFAULT_MODEL = "~/models/glm53-exl3-td205"
EH_SIDECAR_DEFAULT = "~/models/glm53-mtp-eh-proj-bf16.safetensors"
# Name inside the model folder. Not *.safetensors, so the weight loader never indexes it as model weights
# (same layout trick as uncensor_direction.st); the file is a normal safetensors file.
EH_SIDECAR_IN_PACK = "mtp_eh_proj.st"
# --sessions > 1 (several requests decode in one step): plain steps take the R-row exact paths (union MoE on the device table, R-row
# GEMV, batched router) instead of per-expert eager loops, and 2..8-row verify blocks replay as graphs. Box 4 requests, same prompts:
# plain 26.1 -> 40.7 tok/s aggregate, speculative 38.1 -> 42.5; one request alone and speculative text equal to the solo run unchanged.
MULTI_SESSION_ENV = {
    "EXL3_PLAIN_ROWS_VERIFY": "1",
    "EXL3_BLOCK_GRAPH_VERIFY": "1",
    "EXL3_DEC_MOE_UNION_DEV": "1",
}
SPEED_ENV = {
    # Vision spatial-merge convolution as one GEMM: no MIOpen solver search, so the first image of a new
    # size costs seconds, not minutes (first 1024x1024: 433 s -> 4 s; answers byte-identical). "0" = F.conv2d.
    "EXL3_CONV_GEMM": "1",
    # No dense GEMM screening: a first start takes about a minute instead of 13 (771 s -> 65 s) and a new image
    # size never stalls on it. Answers are byte-identical. "1" brings the tuner and its start-up pre-tuning
    # (EXL3_SERVE_DTUNE_PRIME_S) back; see doc/EXL3_CONV_GEMM.md for the offline pre-generation.
    "EXL3_DENSE_GEMM_TUNE": "0",
    "EXL3_MOE_CFG": "2",
    "EXL3_HIP_PREFILL_MIN_ROWS": "2",
    "EXL3_BLOCK_GRAPH": "1",
    "EXL3_BLOCK_GRAPH_MLA": "2",
    "EXL3_HC_FN_HALF_PF": "1",
    "EXL3_HC_1PASS_PF": "1",
    # 26/09 gains: DSA decode attention +4.6 % at 4K; prefill chunking -7.3 % at 4K.
    "EXL3_DEC_DSA_FAST": "1",
    "EXL3_MIDCHUNK_CKPT": "2",
    "EXL3_PREFILL_BIG_TAIL": "1",
    "EXL3_BIG_TAIL_DENSE": "1",
    # 16/10: big chunks at every depth (was 32768). Prefill +12.6 % at 128K, +12.1 % at 256K, +10.5 % at 500K;
    # vs the true next tokens NLL -0.012 (95 % CI -0.027..+0.003), next-token accuracy unchanged, 500K needle found,
    # peak memory still fits a 500K context on a 128 GB Strix Halo.
    "EXL3_PREFILL_BIG_CHUNK_MAXPOS": "0",
    # 15/10: 8192-row prefill chunks (an 8K prompt = one forward; 4-slot KDA checkpoints keep every tail boundary).
    # Box 8K cold window -3.8 %, served 8K 697 / 32K 681 tok/s, logits + restore hashes equal to 4096 chunks.
    "EXL3_PREFILL_BIG_CHUNK": "8192",
    "EXL3_NC2_CKPT4": "1",
    # 27/09: unquantized MTP eh_proj (td205 stores it at 2 bpw, cos 0.79):
    # served accept +0.077 +- 0.008 (n=24), target ids 24/24 equal, serve_accept decode +4.8 %.
    # The pack ships the sidecar as mtp_eh_proj.st; loaded by default (see apply_eh_sidecar_default()).
    # 27/09: R-row verify linears in one launch (Hadamards in-kernel):
    # served R=2 round -3.04 +- 0.20 ms, +6.7 % t/s; R=1 bitwise, R=2 greedy 64/64.
    "EXL3_VERIFY_FUSE": "1",
    # calibrated dynamic draft depth for the MTP drafter; served decode geomean x1.070 (structured x1.104, code x1.070, prose x1.036).
    "EXL3_GLM_DYN": "3,0.70",
    # 27/09: one router launch per verify + wide MoE combine; with R=3 (--num-draft 2)
    # served +6.42 +- 1.34 % vs R=2 base (12 prompts/arm, greedy ids 128/128 x 3 equal).
    "EXL3_DEC_ROUTER_ROWS": "2",
    "EXL3_MOE_COMBINE_WIDE": "1",
    # 27/09: R=3 union MoE with RM=3 instead of RM=4 (tighter registers/LDS
    # when every expert serves at most 3 rows); served R=3 round -1.08 ms, +1.46 +- 0.06 % t/s
    # (one box, one load, n=12/arm paired), greedy ids 12x128/128 identical, kernel bitwise.
    "EXL3_MOEDEC1_RM3": "1",
    # NLL n=64 mean +0.092 % (CI incl 0), median -0.003 %, dPPL -0.04 % -> within GOAL PPL <= +0.1 %; union MoE -3 %.
    "EXL3_MOEDEC1_KBS_A": "3",
    "EXL3_GEMV_R_DEC1": "1",
    # 27/09: DSA indexer glue fused (n=27/arm paired): decode 29.53 -> 30.04 t/s,
    # +1.74 +- 0.43 %; greedy ids 3x64 and MTP accept identical. (round2's stack saw -0.08 % at n=12.)
    "EXL3_DSA_GLUE_FUSE": "1",
    # sparse (> index_topk 2048) DSA decode fast path: bsz >= 3 takes one batched call, bsz 1/2 stay
    # byte-identical; long-context decode +4.34 % (3/3 paired reps, N=8 homogeneous ~3.6K-token batch).
    # Staged default: inert until the DSA rowloop knob reads it. 0 = per-job calls.
    "EXL3_MU_DSA_ATTN_ROWLOOP_OFF_BSZ": "3",
    # 27/09: prefill mHC apply + next mix_norm fused; 16K 574.3 -> 579.8
    # (+0.95 +- 0.07 %), 4K +1.46 +- 0.48 % (n=10). Decode neutral. Needs the rebuilt ext.
    "EXL3_PF_HC_FUSE": "1",
    # 27/09: mpw2x MoE grouped GEMM (no register spill, 224 VGPR): prefill 579.8 -> 598.2
    # t/s 4K (+3.24 +- 0.19 %, n=6), 16K +2.62 +- 0.24 % (n=4); ids identical, logits bitwise at 4K.
    "EXL3_MPW2X": "2",
    "EXL3_MLA_PF_FAST": "1",
    "EXL3_KDA_PF_SPLIT": "2",
    "EXL3_PF_NO_TAIL": "1",
    # 15/10: a turn that resumes mid-page (tail checkpoint) and adds >= 1024 tokens no longer pays a separate last-page
    # forward: served tool turn 4.34 -> 3.35 s, 2.58 -> 2.14 s; first-token KLD vs cold within the chunk-size floor
    # Slot files keep the tail checkpoint (slot_store format 2), so a reload does not replay the gap.
    "EXL3_TAIL_MAXGAP": "8192",
    # a prompt that shares cached pages but not their recurrent checkpoint (a new conversation on the same long system prompt)
    # side-writes the state where those pages end, so the next one resumes there: first token after a shared 9K / 12.1K prefix
    # 2.41 -> 1.94 s, 3.84 -> 2.06 s; the writing prompt is bitwise unchanged, KLD vs a cold prefill at the resumed-turn level.
    "EXL3_BRANCH_CKPT": "1",
    # prompt-tail recurrent checkpoints on the last 7 page boundaries before the prompt end: a re-sent prefix whose last page
    # checkpoint lies after the divergence resumes within one page of it. 89.6K prompt, new question at the end: 6.0 -> 1.7 s,
    # 1539 -> 259 rows replayed (one forward pass); about +0.4 GiB host memory, flat over 20 turns. A resume replays a different
    # number of rows than a cold prefill and the GEMM path depends on the row count, so the output is not yet bit-identical to a
    # cold prefill; the fix is in progress. Off on GLM until then (the Qwen preset has it on).
    "EXL3_TAIL_DENSE": "0",
    # sub-page prompt-tail checkpoints on the last 4 multiples of 64 rows: a prompt that diverges in its final tokens replays
    # about 65 rows instead of up to a page (89.6K prompt, new question at the end: 1.7 -> 0.8 s first token); at most
    # EXL3_TAIL_SUB_MAX (8) entries of 142 MiB host memory. Same numerics caveat as the dense checkpoints; off.
    "EXL3_TAIL_SUB": "0",
    # such a resume (page-aligned, off the 2048 chunk grid) with a short remainder keeps the tail in its one chunk (prompt-end tail
    # checkpoint) instead of a separate last-page forward: 2.00 -> 1.42 s, 2.44 -> 1.51 s; KLD vs a cold prefill unchanged in class.
    "EXL3_TAIL_OFFGRID": "1",
    # cold prompts checkpoint every 2048 rows before the tail window too (default grid 32768), so the FIRST new conversation on a
    # long system prompt cached by a conversation with a long first turn resumes near the shared end instead of at token 0: served,
    # 14.7K-token system + 4.5K-token turns, 28.4 -> 8.0 s; later cold prompts equal within noise, the first one per process +~1.8 s.
    "EXL3_CKPT_INTERVAL_PP": "2048",
    # Dense EXL3 GEMM with in-GEMM decode for forwards of 1..1023 rows (the large K 3/4 shapes skip the fp16 weight rebuild).
    # Served GLM first token +5.7..+12.2 % on 200-900-token prompts, cold and follow-up. Quality checked against an fp32
    # reference: top-1 vs fp32 never below the current path over 96 rows. Read per call, not latched. "0" = off.
    "EXL3_DENSE_MPW2": "1023",
    "EXL3_HOST_LEAN": "1",
    "EXL3_HOST_CUTS": "1",
    "EXL3_PF_GLUE": "1",
    "EXL3_PF_MSPLIT": "2048",
    "EXL3_PF_SKIP": "1",
    # recurrent-state stashes (mid-prompt checkpoints, prompt-end tail) copy to pinned host memory without a host sync: bitwise
    # (first-token + tail-resume logits equal, 5 reps x 8K/32K, base vs on); removes ~33 x 0.6 ms of idle at the end of a prefill. 0 = off.
    "EXL3_NC2_ASYNC_STASH": "1",
    # a speculative rewind records the accepted recurrent-state history index in a per-slot base instead of copying
    # it back to index 0 (272 MiB, ~1.3 ms per launch): bit-exact (ids equal off vs on, GLM KDA depth 2). 0 = copy.
    "EXL3_STATE_REBASE": "1",
    # 14/10: verify of 5..8 rows (GLM_NGRAM lookup rounds of 4..7 copied tokens), the Qwen server's switches: MoE row split,
    # one-launch 8-row fused MoE, one-launch 6-row dense GEMV. Bit-identical per row, rows <= 4 untouched (--num-draft 2 rounds).
    # Harness, one load each: 6-row round +6.9 % (RM6), 8-row round +2.6 %, ids equal. 0 = off.
    "EXL3_VERIFY_ROW_SPLIT": "2",
    "EXL3_MOE_R8": "1",
    "EXL3_GEMV_R_RM6": "1",
    # 7..8 rows (4 sessions x 2 draft rows): one decode pass of the dense GEMV instead of two 4-row chunks, bit-identical. 0 = off.
    "EXL3_GEMV_R_WIDE8": "1",
    # 14/10: sampled requests (temperature > 0) draw the MTP drafts from the drafter's own law and keep them with the exact
    # min(1, p / q) rule (exllamav3/generator/spec_sample.py): same output distribution, greedy untouched. One load, T 1 top_p
    # 0.95, 15 prompts x 3 seeds: prose +5.2 % decode, no measured change on chat / code / structured. 0 = argmax drafts.
    "EXL3_SPEC_SAMPLED": "1",
    # 14/10: prefill producers write GEMM-padded rows (KDA o_proj input, mHC norm outputs) and the shared expert is
    # added in the routed-MoE reduce epilogue: no pad copies, no separate widen + fp32 add. Bitwise: first-token and restored-
    # prompt logits equal on vs off (3 x 8K + 1 x 32K, one load, ABBA). 8K prefill +3.7..+6.2 % (these four + the async
    # stash, which alone shows no gain and stays off), 32K +0.3 % (shared fuse) / +2.2 % (pads). 0 = off.
    "EXL3_NC2_SHARED_FUSE": "1",
    "EXL3_NC2_KDA_OPAD": "1",
    "EXL3_NC2_HC_YPAD": "1",
    "EXL3_NC2_HC_YPAD_MLP": "1",
    # 15/10: silu_mul, the MLA unfold and the KDA prefill norm write their GEMM input at the padded
    # pitch; attention / MLA o_proj / the ablit projection keep fp16 output where the residual reads it. Bitwise (one load,
    # ABBA, on top of the four above: first-token and restored-prompt logits equal, 3 x 8K + 1 x 32K); prefill +0.3 % at 8K (iGPU
    # heat), +0.8 % at 32K. 0 = off.
    "EXL3_NC2_SILU_PAD": "1",
    "EXL3_NC2_MLA_OPAD": "1",
    "EXL3_NC2_ATTN_F16": "1",
    "EXL3_NC2_ABLIT_F16": "1",
    "EXL3_NC2_MLA_OPROJ_F16": "1",
    "EXL3_NC2_KDA_NPAD": "1",
    # 17/10: multi-block exact decode top-k (bitwise on 24+ cases, decode R<=1 T>=32768 only; saves 0.13/0.42/0.91 ms per token at 128K/256K/500K). 0 = one-block.
    "EXL3_TOPK_PAR": "1",
}

# Host memory for the decoded image embeddings (--image-cache-mb). One entry per picture is 49.0 MiB for a
# 1568x1568 page (3,136 tokens) and 1 MB for a 224x224 crop, so pictures, not bytes, are what a conversation
# spends: the default holds about 20 of those pages, as upstream's 1 GiB did.
IMAGE_CACHE_MB_DEFAULT = 1024
# Count cap on top of the byte budget, as upstream's shared cache defaults to it (0 = no limit). For big pages
# the byte budget binds first; for small ones this keeps the entry count bounded.
IMAGE_CACHE_ITEMS_DEFAULT = 32
# How the cached rows are stored (--image-cache-format). fp32 is what the vision merger emits; fp16 is exactly
# the same values (the tower computes in fp16) and q8 rounds at one part in 254 of the row's peak. Kept here
# rather than imported from the engine, which serve.py must not import before SPEED_ENV is applied.
IMAGE_CACHE_FORMATS = ("fp32", "fp16", "q8")
IMAGE_CACHE_DIR_DEFAULT = "~/.cache/kyojin/image-embeddings"
# GiB cap on the second-level picture cache directory. 0 means the disk cache is off, which is the default: the
# memory cache above is what a normal session needs, and the disk cache exists for conversations with more
# pictures than that budget holds. A positive value is always a real cap, there is no "unlimited".
IMAGE_CACHE_DIR_MAX_GB = 0.0
# How long a conversation keeps the room its picture cache took on disk (--image-cache-dir-lock-s). The clock
# starts when the files are written and is not refreshed by later requests, so a conversation that grabbed the
# cache cannot keep it for ever.
IMAGE_CACHE_DIR_LOCK_S = 300.0


def make_image_cache(image_cache_bytes: int | None = None, image_cache_items: int = IMAGE_CACHE_ITEMS_DEFAULT,
                     sessions: int = 1, policy: str | None = None) -> ImageEmbeddingCache:
    """The picture cache of the process (--image-cache-mb / --image-cache-items).

    One cache for the whole server, shared by every session exactly like the KV pool `-c` sizes, with the
    eviction policy of the shared cache in tools/strix_halo/image_cache.py: the default gives up the head of a
    prompt a client keeps re-sending (what the servers always did), and `KYOJIN_IMAGE_CACHE_POLICY=tail` gives up
    its tail instead. The policy remembers one prompt order per concurrent request plus one, so a conversation is
    not forgotten between two of its own turns when another session slips in between.
    """
    return ImageEmbeddingCache(max_items=image_cache_items, max_orders=max(2, sessions),
                               **({} if image_cache_bytes is None else {"max_bytes": image_cache_bytes}),
                               **({} if policy is None else {"policy": policy}))


def env_fingerprint_source(model_dir: str | None, config) -> dict:
    """What the rows in the picture cache on disk were computed with (see `make_env_fingerprint`).

    A stored row is only worth keeping while the model and the preprocessing that produced it stay the
    same, and the file name has to say so or a restart with another model would inject rows from the old
    one. What goes in: the pack directory (two packs of the same architecture encode differently), the
    vision settings that decide the resize and the canvas, and the size and mtime of the visual weight
    shards - the cheap stand-in for hashing ~1.5 GB at every start. A pack that is converted again keeps
    its name and its config, so its rows are caught only if its shards change size or date; the disk tier
    is a cache, and a miss costs an encode, not correctness of the answer.
    """
    source: dict = {"model": Path(model_dir).name if model_dir else "?"}
    for attr in ("vision_config", "vision_pp"):
        part = getattr(config, attr, None)
        if part is None:
            continue
        for field in ("patch_size", "merge_size", "temporal_patch_size", "min_image_tokens",
                      "max_image_tokens", "spatial_merge_size", "hidden_size"):
            value = getattr(part, field, None)
            if value is not None:
                source[f"{attr}.{field}"] = value
    if model_dir:
        index = Path(model_dir) / "model.safetensors.index.json"
        try:
            weight_map = json.loads(index.read_text())["weight_map"]
            shards = sorted({name for key, name in weight_map.items()
                             if "visual" in key or "vision" in key})
            for name in shards:
                shard = Path(model_dir) / name
                stat = shard.stat()
                source[f"shard.{name}"] = f"{stat.st_size}:{int(stat.st_mtime)}"
        except (OSError, KeyError, ValueError):
            pass
    return source


def make_env_fingerprint(model_dir: str | None, config) -> str:
    """`env_fingerprint_source` reduced to a short name that can go in a file name.

    Hashed rather than spelled out, so the name stays portable and fixed-length; the source dictionary is
    written into the request log too - `image_cache_env` on the `serve_start` line and `env` inside the
    `disk=` report of `img_cache` - which is what makes a fingerprint that changed between two runs visible
    instead of mysterious.
    """
    import hashlib
    source = env_fingerprint_source(model_dir, config)
    blob = json.dumps(source, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def make_image_disk(image_cache_dir: str | None, image_cache_dir_max_gb: float, image_cache_dir_lock_s: float,
                    start_token: int = 0, end_token: int = 0, env_fingerprint: str = ""):
    """The second-level picture cache, or None when it is off (`--image-cache-dir-max-gb` 0 or less).

    A cap is required: the disk cache is meant for conversations with more pictures than `--image-cache-mb`
    holds, and an unbounded directory is not something a server should be able to create by omission.
    """
    if not image_cache_dir_max_gb or image_cache_dir_max_gb <= 0:
        return None
    from image_cache_disk import ImageEmbeddingDisk      # imports exllamav3: only once it is wanted
    return ImageEmbeddingDisk(image_cache_dir or IMAGE_CACHE_DIR_DEFAULT,
                              int(float(image_cache_dir_max_gb) * 2 ** 30),
                              start_token=int(start_token or 0), end_token=int(end_token or 0),
                              lock_s=max(0.0, float(image_cache_dir_lock_s)),
                              env_fingerprint=env_fingerprint)


def apply_eh_sidecar_default(model_dir: str | None = None) -> str:
    """Pick the optional unquantized MTP eh_proj sidecar and log which source is used.

    Order: explicit EXL3_MTP_EH_FP16 (never touched; "0" = off; a wrong path still fails loudly at load)
    > EH_SIDECAR_IN_PACK inside the model folder > the old default path under ~/models > absent.
    The sidecar only raises draft acceptance, the server runs without it.
    Returns the source: "explicit", "pack", "default" or "absent".
    """
    if "EXL3_MTP_EH_FP16" in os.environ:
        print(f"serve: MTP eh_proj sidecar: EXL3_MTP_EH_FP16={os.environ['EXL3_MTP_EH_FP16']} (explicit)", flush=True)
        return "explicit"
    if model_dir:
        path = Path(model_dir).expanduser() / EH_SIDECAR_IN_PACK
        if path.is_file():
            os.environ["EXL3_MTP_EH_FP16"] = str(path)
            print(f"serve: MTP eh_proj sidecar: {path} (model folder)", flush=True)
            return "pack"
    path = os.path.expanduser(EH_SIDECAR_DEFAULT)
    if os.path.isfile(path):
        os.environ["EXL3_MTP_EH_FP16"] = path
        print(f"serve: MTP eh_proj sidecar: {path} (default path)", flush=True)
        return "default"
    print(f"serve: MTP eh_proj sidecar: none ({EH_SIDECAR_IN_PACK} not in the model folder, nothing at "
          f"{EH_SIDECAR_DEFAULT}): running without it, draft acceptance is lower. To add it: python "
          "tools/glm/mtp_eh_sidecar.py <official GLM-5.3 checkpoint dir> <model folder>/" + EH_SIDECAR_IN_PACK,
          flush=True)
    return "absent"


DRAFT2S_IN_PACK = "draft2s.st"  # safetensors layout, but not *.safetensors so the weight loader never indexes it


def _draft2s_pack_matches(path: Path, model_dir: Path) -> bool:
    """True when the pack header (tools/qwen/draft2s_pack.py metadata) was built for this model's vocabulary and width."""
    import struct
    try:
        with open(path, "rb") as fh:
            meta = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0])).get("__metadata__") or {}
        cfg = json.loads((model_dir / "config.json").read_text())
        cfg = cfg.get("text_config", cfg)
        return meta.get("format") == "draft2s-v1" and int(meta["vocab"]) == cfg["vocab_size"] \
            and int(meta["in_features"]) == cfg["hidden_size"]
    except (OSError, ValueError, KeyError, struct.error):
        return False


def apply_draft2s_default(model_dir: str | None = None) -> str:
    """Pick the optional two-stage draft head pack (EXL3_DRAFT2S, exllamav3/modules/draft2s.py) and log which source is used.

    Order: explicit EXL3_DRAFT2S (never touched; "", "0" or "off" = off) > DRAFT2S_IN_PACK inside the model folder when its
    header matches the model (vocab_size, hidden_size) > absent. The head only changes how the draft argmax is found, never
    the output (every draft is verified). Returns the source: "explicit", "off", "pack", "mismatch" or "absent".
    """
    if "EXL3_DRAFT2S" in os.environ:
        if os.environ["EXL3_DRAFT2S"].strip().lower() in ("", "0", "off"):
            os.environ["EXL3_DRAFT2S"] = ""
            print("serve: two-stage draft head: off (EXL3_DRAFT2S)", flush=True)
            return "off"
        print(f"serve: two-stage draft head: EXL3_DRAFT2S={os.environ['EXL3_DRAFT2S']} (explicit)", flush=True)
        return "explicit"
    if model_dir:
        path = Path(model_dir).expanduser() / DRAFT2S_IN_PACK
        if path.is_file():
            if not _draft2s_pack_matches(path, Path(model_dir).expanduser()):
                print(f"serve: two-stage draft head: {path} does not match this model (vocab / width): not used", flush=True)
                return "mismatch"
            os.environ["EXL3_DRAFT2S"] = str(path)
            print(f"serve: two-stage draft head: {path} (model folder)", flush=True)
            return "pack"
    print(f"serve: two-stage draft head: none ({DRAFT2S_IN_PACK} not in the model folder): full draft head", flush=True)
    return "absent"


# The template deliberately uses a zero-width space so normal prose does not trigger tools.
TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"


# Server-side defaults for fields a client leaves out (set from the CLI in main()).
# The engine's own default stays greedy; the lane passes the model card's sampling.
# Used when a request omits the field. main() replaces the sampling values with the pack's
# generation_config.json (the model card's recommendation); command-line flags win over both.
_UNKNOWN_SALT = object()   # cache namespace after a failed reset: the next request resets again
SERVE_DEFAULTS: dict[str, Any] = {"temperature": 1.0, "top_p": 0.95, "reasoning_effort": "medium", "max_tokens": None}
# The model's template (levels low, high, max) with a medium level added: the model's "high" plus one instruction line.
DEFAULT_TEMPLATE = Path(__file__).resolve().parent.parent / "lanes" / "assets" / "glm53-template-medium.jinja"


def pack_sampling(model_dir: str) -> dict[str, float]:
    """temperature / top_p of the pack's generation_config.json, when it has them."""
    gc = Path(model_dir).expanduser() / "generation_config.json"
    if not gc.is_file():
        return {}
    cfg = json.loads(gc.read_text())
    return {k: float(cfg[k]) for k in ("temperature", "top_p") if cfg.get(k) is not None}


def template_kwargs(body: dict[str, Any]) -> dict[str, Any]:
    """Template variables: chat_template_kwargs (vLLM style) wins over the top-level fields, then the server defaults."""
    ctk = body.get("chat_template_kwargs") or {}
    if not isinstance(ctk, dict):
        raise web.HTTPBadRequest(reason="chat_template_kwargs must be an object")
    pick = lambda key, default=None: ctk[key] if ctk.get(key) is not None else body.get(key, default)
    kw = {"clear_thinking": pick("clear_thinking")}
    effort = pick("reasoning_effort", SERVE_DEFAULTS["reasoning_effort"])
    if effort is not None:
        kw["reasoning_effort"] = effort
    return kw


def _template_env() -> Any:
    """One Jinja environment per process (same settings as transformers' apply_chat_template)."""
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = lambda value, ensure_ascii=False, indent=None, separators=None, sort_keys=False: \
        json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
    def raise_exception(msg): raise ValueError(msg)
    env.globals["raise_exception"] = raise_exception
    return env


_COMPILED_TEMPLATES: dict[str, Any] = {}   # template source -> compiled Template (render is thread-safe)


def _compiled_template(template: str) -> Any:
    """The compiled chat template, cached: compiling it per request costs ~24 ms (8.6 KB GLM template)."""
    hit = _COMPILED_TEMPLATES.get(template)
    if hit is None:
        hit = _template_env().from_string(template)
        _COMPILED_TEMPLATES[template] = hit
    return hit


def render_prompt(template: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None,
                  **kwargs: Any) -> str:
    """Render the model's own Jinja chat template."""
    # Same environment as transformers' apply_chat_template.
    # OpenAI sends tool-call arguments as a JSON string; the GLM template iterates them as a dict.
    fixed = []
    for m in messages:
        if m.get("role") == "developer":                               # OpenAI's newer name for the system role
            m = dict(m, role="system")
        if m.get("tool_calls"):
            m = dict(m, tool_calls=[
                dict(c, function=dict(c["function"], arguments=_json_value(c["function"]["arguments"])))
                if isinstance(c.get("function", {}).get("arguments"), str) else c
                for c in m["tool_calls"]])
        fixed.append(m)
    return _compiled_template(template).render(
        messages=fixed, tools=tools or [], add_generation_prompt=True, **kwargs)


def _json_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def parse_completion(text: str) -> dict[str, Any]:
    """Split GLM think blocks and zero-width GLM tool calls into OpenAI fields."""
    calls: list[dict[str, Any]] = []
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    cursor = 0
    pattern = re.compile(re.escape(TOOL_OPEN) + r"(.*?)" + re.escape(TOOL_CLOSE), re.S)
    for match in pattern.finditer(text):
        visible = text[cursor:match.start()]
        if "<think>" in visible:
            before, after = visible.split("<think>", 1)
            content_parts.append(before)
            thinking, _, visible = after.partition("</think>")
            reasoning_parts.append(thinking)
        content_parts.append(visible)
        body = match.group(1)
        name_match = re.match(r"\s*([^\s<]+)", body)
        if not name_match:
            continue
        args: dict[str, Any] = {}
        for part in re.finditer(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", body, re.S):
            args[part.group(1).strip()] = _json_value(part.group(2).strip())
        calls.append({"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
                      "function": {"name": name_match.group(1), "arguments": json.dumps(args, ensure_ascii=False)}})
        cursor = match.end()
    tail = text[cursor:]
    if "<think>" in tail:
        before, after = tail.split("<think>", 1)
        content_parts.append(before)
        thinking, _, tail = after.partition("</think>")
        reasoning_parts.append(thinking)
    content_parts.append(tail)
    message: dict[str, Any] = {"role": "assistant", "content": "".join(content_parts)}
    reasoning = "".join(reasoning_parts)
    if reasoning:
        message["reasoning_content"] = reasoning
    if calls:
        message["tool_calls"] = calls
        message["finish_reason"] = "tool_calls"
    return message


def token_count(event: dict) -> int:
    t = event.get("token_ids")
    return int(t.numel()) if t is not None else 0


def guard_hit(tap, event: dict) -> dict | None:
    """Feed one generator event to the answer-phase guard; the hit dict when the answer loops."""
    if tap is None or event.get("eos"):
        return None
    t = event.get("token_ids")
    return tap.feed(t.flatten().tolist() if t is not None and t.numel() else [], event.get("text") or "")


def guard_stats(hit: dict, n_new: int) -> dict:
    """The end-of-job report of a request the guard stopped (the job was cancelled, so it has no eos event)."""
    return {"eos_reason": "loop_guard", "new_tokens": n_new, "loop_guard": hit}


def held_tail(text: str, markers: list[str]) -> int:
    """Characters at the end of `text` that could still become one of `markers` (a proper prefix of one). A stream holds back only
    these: holding a marker-length tail delayed the first visible text by several tokens, which under load is several seconds."""
    best = 0
    for m in markers:
        for k in range(min(len(m) - 1, len(text)), best, -1):
            if text.endswith(m[:k]):
                best = k
                break
    return best


def opens_thinking(prompt: str) -> bool:
    """The GLM generation prompt ends in <think>, so the model output starts inside the block."""
    return prompt.rstrip().endswith("<think>")


def stop_text(text: str, stop: list[str]) -> tuple[str, str | None]:
    """Cut at the earliest stop string and return content and matched stop."""
    positions = [(text.find(s), s) for s in stop if s and text.find(s) >= 0]
    if not positions:
        return text, None
    pos, match = min(positions)
    return text[:pos], match


def token_limit(body: dict[str, Any], default: int | None = None) -> int:
    """Reply token budget: max_completion_tokens (newer OpenAI field) wins over max_tokens."""
    for key in ("max_completion_tokens", "max_tokens"):
        value = body.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
            return min(value, body.get("_room") or value)
    default = default or SERVE_DEFAULTS["max_tokens"] or body.get("_room") or 32768  # unset: all the free context
    return min(default, body.get("_room") or default)


def reply_room(engine: Any, prompt_tokens: int) -> int | None:
    """Tokens a reply may still use: server context minus prompt, one slot and the speculative window.
    None when the engine does not report a context (fake engines). A prompt with no room is refused,
    as the cache cannot hold it (the engine would otherwise fail the job with an assertion)."""
    ctx = getattr(engine, "ctx", None)
    if not ctx:
        return None
    room = ctx - prompt_tokens - 1 - getattr(engine, "num_draft", 0)
    if room < 1:
        raise web.HTTPBadRequest(reason=f"context length exceeded: prompt is {prompt_tokens} tokens, the server context is {ctx}")
    return room


def sse(data: dict[str, Any], event: str | None = None) -> bytes:
    prefix = f"event: {event}\n" if event else ""
    return (prefix + f"data: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n").encode()


# Decoding-side default penalties (ComboSampler kwargs), e.g. SERVE_SAMPLER_KW='{"dry_multiplier":0.8}'. Empty = none.
DEFAULT_SAMPLER_KW: dict = json.loads(os.environ.get("SERVE_SAMPLER_KW", "{}") or "{}")


# ---- image input -----------------------------------------------------------------------------------
# The chat template writes one IMAGE_TRIPLE per image part; the engine swaps each for the embedding's alias,
# which the tokenizer expands to <|begin_of_image|> + one slot per vision token + <|end_of_image|>.
IMAGE_TRIPLE = "<|begin_of_image|><|image|><|end_of_image|>"
MAX_IMAGE_BYTES = 32 * 1024**2


def _bad(reason: str) -> web.HTTPBadRequest:
    return web.HTTPBadRequest(reason=reason)



def extract_images(messages: list[dict[str, Any]]) -> list[str]:
    """Image URLs of the messages in prompt order (OpenAI image_url parts). Video and audio parts are refused:
    this server has no such tower, and dropping them silently would let the model invent an answer."""
    urls = []
    for m in messages:
        content = m.get("content") if isinstance(m, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            kind = part.get("type")
            if kind in ("video", "video_url", "audio", "audio_url", "input_audio"):
                raise _bad(f"{kind} input is not supported by this server (images only)")
            if kind in ("image_url", "image"):
                ref = part.get("image_url", part.get("image"))
                url = ref.get("url") if isinstance(ref, dict) else ref
                if not isinstance(url, str) or not url:
                    raise _bad("image_url needs a url")
                urls.append(url)
    return urls


def load_image(url: str) -> tuple[Any, str]:
    """(PIL image, content hash) from a data: URI, a file:// URL or a local file path."""
    from PIL import Image
    if url.startswith("data:"):
        try:
            raw = base64.b64decode(url.split(",", 1)[1], validate=False)
        except Exception as exc:                                      # noqa: BLE001
            raise _bad(f"bad data URI: {exc}") from exc
    elif url.startswith(("http://", "https://")):
        raise _bad("image url must be a data: URI or a local file path (the server does not fetch remote images)")
    else:
        path = Path(url[7:] if url.startswith("file://") else url).expanduser()
        try:
            if path.stat().st_size > MAX_IMAGE_BYTES:
                raise _bad("image too large")
            raw = path.read_bytes()
        except OSError as exc:
            raise _bad(f"cannot read image file: {exc}") from exc
    if len(raw) > MAX_IMAGE_BYTES:
        raise _bad("image too large")
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB"), digest(raw)
    except Exception as exc:                                          # noqa: BLE001
        raise _bad(f"cannot decode image: {exc}") from exc


MAX_SESSIONS = 16   # upper bound of --sessions (the Cache default max_batch_size)



class ResidentEngine:
    """One loaded target model, optional MTP model, and resident paged caches."""
    supports_prefix_reuse = True
    row_budget = 0

    def __init__(self, model_path: str, max_history: int = 1, max_ctx: int = 131072, num_draft: int = 1,
                 vision: bool = True, sessions: int = 1,
                 image_cache_bytes: int | None = None, image_cache_items: int = 0,
                 image_cache_format: str = "fp32",
                 image_cache_dir: str | None = IMAGE_CACHE_DIR_DEFAULT,
                 image_cache_dir_max_gb: float = IMAGE_CACHE_DIR_MAX_GB,
                 image_cache_dir_lock_s: float = IMAGE_CACHE_DIR_LOCK_S):
        import torch
        from exllamav3 import Cache, Config, Generator, Job, Model, Tokenizer
        from exllamav3.generator import generator as GM

        torch.set_grad_enabled(False)
        self.torch = torch
        self.Job = Job
        # EXL3_GLM_DYN = "window,confidence": the Generator's confidence-calibrated draft stop (dynamic_draft_tokens) for the
        # MTP drafter, `window` drafted tokens at most per round, drafting stops when the estimated probability that the next
        # position still contributes falls below `confidence`. Overrides --num-draft. "0" = fixed --num-draft depth
        dyn = os.environ.get("EXL3_GLM_DYN", "0").strip()
        self.dyn = None
        if dyn not in ("", "0") and num_draft > 0:
            w, c = dyn.split(",")
            self.dyn = (int(w), float(c))
            num_draft = self.dyn[0]
            max_history = max(max_history, num_draft)
        self.num_draft = num_draft
        # Concurrent requests batched through the one Generator. The recurrent (KDA) state is allocated per slot and per
        # history level (138 MiB each). The Cache default of 16 slots was allocated whatever --sessions said: 6.6 GiB at
        # --num-draft 2, of which a single session uses 0.4 GiB. The cache now gets one slot per session (the Generator
        # caps its batch at the slot count, the admission gate at --sessions). EXL3_GLM_SLOTS=16 restores the old layout. The paged KV pool (ctx + 4096 tokens) is shared by all sessions.
        if not 1 <= sessions <= MAX_SESSIONS:
            raise ValueError(f"sessions must be 1..{MAX_SESSIONS}")
        self.sessions = sessions
        self.row_budget = 0   # set by --row-budget (sessions > 1)
        self.max_draft = num_draft
        self.config = Config.from_directory(model_path)
        self.model = Model.from_config(self.config)
        self.tokenizer = Tokenizer.from_config(self.config)
        if os.environ.get("KYO_TOKCACHE", "8") == "0":
            self.pec = None
        else:
            self.pec = PromptEncodeCache(self.tokenizer, n=int(os.environ.get("KYO_TOKCACHE", "8") or 0), cut=["<|system|>", "<|user|>", "<|assistant|>", "<|observation|>"], check=os.environ.get("KYO_TOKCACHE_CHECK") == "1", log=lambda m: print(f"serve: {m}", flush=True))
        max_position = int(self.config.config_dict.get("max_position_embeddings") or self.config.config_dict.get("text_config", {}).get("max_position_embeddings") or 32768)
        ctx = min(max_position, max_ctx)
        self.ctx = ctx
        slots = max(sessions, int(os.environ.get("EXL3_GLM_SLOTS", "0") or 0))
        # GLM_NGRAM = "gram,K,match[,cmin]": n-gram lookup of tools/qwen/spec_policy.install_lookup. When the
        # last `match` tokens occurred before, the round verifies the K tokens that followed them instead of the drafter's
        # --num-draft tokens. K rows beyond the first need K history levels (138 MiB each per slot). Default 2,7,16,3 with
        # one session (copy / edit turns +26 %, +0.67 GiB; greedy text may change at an fp16 tie), off with more sessions
        # (lookup rounds need a single active request and the history cost grows per slot). GLM_NGRAM=off turns it off.
        v = (os.environ.get("GLM_NGRAM") or ("2,7,16,3" if sessions == 1 else "off")).strip().lower()
        self.lookup = None if v in ("off", "0", "none") or num_draft < 1 else tuple(int(x) for x in v.split(","))
        if self.lookup:
            if not (3 <= len(self.lookup) <= 4 and 1 <= self.lookup[0] <= 16 and self.lookup[1] >= 1 and self.lookup[2] >= self.lookup[0]):
                raise ValueError(f"GLM_NGRAM={v}")
            max_history = max(max_history, self.lookup[1])
        self.lookup_stats: dict[str, int] = {}
        self.cache = Cache(self.model, max_num_tokens=ctx + 4096,
                           max_history=max_history, max_batch_size=slots)
        from exllamav3.util import jit_log, kernel_seed, tune_cache
        tune_cache.install_pack(model_path)   # <pack>/tuning: GEMM autotune winners from qualify --stage autotune
        kernel_seed.install()   # shipped Triton kernels for this Triton version: nothing to compile on a first start
        jit_log.install()   # EXL3_LOG_JIT=1: one line per Triton compile
        startup_health.stage("target weights")
        self.model.load(device="cuda:0", progressbar=False, callback=startup_health.load_callback())
        if num_draft > 0:
            self.draft_model = Model.from_config(self.config, component="mtp")
            self.draft_cache = Cache(self.draft_model, max_num_tokens=ctx + 4096,
                                     max_history=max_history, max_batch_size=slots)
            startup_health.stage("drafter weights")
            self.draft_model.load(device="cuda:0", progressbar=False, callback=startup_health.load_callback())
        else:   # --num-draft 0: plain decode, no drafter
            self.draft_model = self.draft_cache = None
        self.vision = None
        self.image_cache_format = image_cache_format
        self.emb_cache = make_image_cache(image_cache_bytes, image_cache_items, sessions)
        # Second-level cache: the rows of the pictures the memory cache evicts. Off unless a cap is given
        # (--image-cache-dir-max-gb > 0), because the memory budget above is what a normal session needs.
        self.env_fingerprint = make_env_fingerprint(model_path, self.config)
        self.emb_disk = make_image_disk(image_cache_dir, image_cache_dir_max_gb, image_cache_dir_lock_s,
                                        getattr(self.config, "vision_start_token_id", 0),
                                        getattr(self.config, "vision_end_token_id", 0),
                                        env_fingerprint=self.env_fingerprint)
        if vision:
            self.vision = self._load_vision()
        self.supports_vision = self.vision is not None
        self.vocab_size = int(self.config.vocab_size or self.tokenizer.actual_vocab_size)   # rows of the embedding table
        self.Generator = Generator
        # The loaded model and tokenizer are 1.1 M Python objects and every later gc.collect() walks them all: the Generator
        # rebuild of a cache_salt change pays 0.4 s for it inside time to first token. EXL3_GC_FREEZE=1 keeps them out of the
        # collector, before the first Generator exists (a frozen Generator would never be freed). Four server loads in both
        # orders give +2.1 % first-token rate at 8K and +7.9 % at 2K, texts identical; on by default as in the Qwen server. 0 = off.
        if os.environ.get("EXL3_GC_FREEZE", "1") != "0":
            import gc
            gc.collect()
            gc.freeze()
        self.greedy_generator = self._make_generator()
        # n1f2: one drafted token, fused catch-up variant 2. One Generator only: a second one over the
        # same Cache takes ownership and breaks the first. Sampling also goes through MTP: a draft is
        # kept only when the independently sampled target token equals it, so outputs follow the target.
        GM.MTP_FUSE_CATCHUP = 2
        os.environ.setdefault("EXL3_MOE_UNION_V2", "1")  # the "+v2" of n1f2+v2

    def _make_generator(self):
        lookup = getattr(self, "lookup", None)
        g = self.Generator(
            model=self.model, cache=self.cache, tokenizer=self.tokenizer,
            draft_model=self.draft_model, draft_cache=self.draft_cache,
            num_draft_tokens=max(self.num_draft, lookup[1] if lookup else 0) or None,   # buffers for the widest round
            record_draft_stats=True,
            dynamic_draft_tokens=self.dyn is not None, draft_confidence=self.dyn[1] if self.dyn else 0.6)
        if lookup:
            qdir = str(Path(__file__).resolve().parent.parent / "qwen")
            if qdir not in sys.path:
                sys.path.append(qdir)
            import spec_policy
            spec_policy.install_lookup(g, lookup, self.lookup_stats)
            g.num_draft_tokens = self.num_draft     # the drafter's own depth; lookup rounds bring their K tokens
        if os.environ.get("EXL3_ROUND_LOG"):   # opt-in per-round decode log (tools/glm/round_probe.py), off by default
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import round_probe
            round_probe.install(g, os.environ["EXL3_ROUND_LOG"], self.lookup_stats)
        return g

    def _rebuild_generator(self) -> None:
        """A fresh Generator over the same Cache (one at a time: a second one takes the Cache over). It drops the whole prompt
        cache: page table, recurrent checkpoints, prompt-end checkpoints, CPU tier. The old one is freed first."""
        import gc
        self.greedy_generator = None
        gc.collect()
        self.torch.cuda.empty_cache()
        self.greedy_generator = self._make_generator()
        if getattr(self, "slot_store", None) is not None:
            self.slot_store.generator = self.greedy_generator
            self.slot_store.erase()

    async def settle_reply(self) -> None:
        """EXL3_REPLY_FIRST, single-session path: the previous request's finished job is released in an executor
        thread after its reply went out (see generate's finally); wait for it before anything touches the Generator or Cache."""
        fut = getattr(self, "_reply_first_fut", None)
        if fut is not None:
            self._reply_first_fut = None
            await fut

    # Persistent prompt cache (EXL3_PREFIX_DISK_DIR, exllamav3/generator/prefix_disk.py). Both run with the generator idle.
    async def disk_restore(self, ids) -> None:
        """Before the request is enqueued: put a deeper resume point from the disk (an earlier run, an evicted prefix) back."""
        pd = self.prefix_disk
        gen = self.greedy_generator
        try:
            if self.sessions > 1:
                r = await self._post(lambda: None if gen.num_remaining_jobs() else pd.restore(ids))
            else:
                r = await asyncio.get_running_loop().run_in_executor(None, pd.restore, ids)
        except Exception as exc:                                        # noqa: BLE001 the disk never fails a request
            print(f"serve: prefix disk restore failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            return
        if r:
            print(f"serve: prefix disk restore resume_at={r['resume_at']} (ram {r['ram_at']}) pages={r['pages']} {r['ms']:.0f} ms",
                  flush=True)

    async def disk_save(self) -> None:
        """After a request: persist its prompt (new pages, checkpoints) with a background write."""
        pd = getattr(self, "prefix_disk", None)
        ids = getattr(self, "_disk_ids", None)
        if pd is None or ids is None:
            return
        await self.settle_reply()                                      # the deferred release writes the prompt-end checkpoint
        gen = self.greedy_generator
        try:
            if self.sessions > 1:
                r = await self._post(lambda: None if gen.num_remaining_jobs() else pd.save(ids))
            else:
                r = await asyncio.get_running_loop().run_in_executor(None, pd.save, ids)
        except Exception as exc:                                        # noqa: BLE001
            print(f"serve: prefix disk save failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            return
        if r is not None:
            self._disk_ids = None

    async def reset_prefix_cache(self) -> None:
        """Forget every cached prompt prefix (cache_salt namespace change). The caller has drained the requests."""
        if getattr(self, "prefix_disk", None) is not None:
            self.prefix_disk.clear()
        await self.settle_reply()
        if self.sessions > 1:
            await self._post(self._rebuild_generator)    # on the driver, between two steps
        else:
            await asyncio.get_running_loop().run_in_executor(None, self._rebuild_generator)

    def _load_vision(self):
        """The vision tower (model.visual.*, fp16) next to the text model; None, with a log line, when the pack has none."""
        from exllamav3 import Model
        t0 = time.perf_counter()
        try:
            startup_health.stage("vision tower")
            tower = Model.from_config(self.config, component="vision")
            tower.load(device="cuda:0", progressbar=False, callback=startup_health.load_callback())
        except Exception as exc:                                      # noqa: BLE001
            print(f"serve: vision tower not loaded ({type(exc).__name__}: {exc}); image input disabled", flush=True)
            return None
        print(f"serve: vision tower loaded in {time.perf_counter() - t0:.1f} s", flush=True)
        return tower

    def image_tokens(self, image: Any) -> int:
        """Vision tokens one image takes (the same canvas rule as the tower's preprocessing, no GPU)."""
        from exllamav3.architecture.mm_processing.glm5_next import glm5_vision_canvas
        v, pp = self.config.vision, self.config.vision_pp
        w, h = image.size
        th, tw = glm5_vision_canvas(pp.temporal_patch_size, h, w, pp.temporal_patch_size,
                                    pp.patch_size * pp.merge_size, pp.min_image_tokens, pp.max_image_tokens)
        return (th // pp.patch_size) * (tw // pp.patch_size) // (v.spatial_merge_size ** 2)

    def _embed(self, images: list[tuple[Any, str]], stats: dict[str, Any] | None = None) -> list:
        """One MMEmbedding per image. The same bytes give the same embedding object (same pseudo token ids), so a
        history turn that repeats an image still matches the paged KV cache (see image_cache.py).

        `stats`, when given, receives what the cache did for this request. A hit keeps the object and therefore
        the pseudo ids the KV pages were keyed on; a miss re-encodes the picture and the paged prefix can only
        match up to it, so `first_missing` (index of the first picture not held) is the number to watch. This
        only reports: the cache policy itself is untouched."""
        cache = self.emb_cache
        disk = getattr(self, "emb_disk", None)
        built: set[int] = set()
        evicted: list = []
        mode = getattr(self, "image_cache_format", "fp32")

        def fetch(image, key):
            """One picture back from the second-level cache, or None: its identity comes back with it."""
            return disk.fetch(key, mode) if disk is not None else None

        def build(image):
            t0 = time.perf_counter()
            emb = self.vision.get_image_embeddings(tokenizer=self.tokenizer, image=image)
            if mode != "fp32":     # store the rows compressed; the ids and the text alias are untouched
                # imported here, not at module level: exllamav3 reads some of the speed defaults at import
                # time, and serve.py must not pull it in before SPEED_ENV is applied
                from exllamav3.tokenizer.mm_embedding import quantize_rows
                emb.embeddings = quantize_rows(emb.embeddings, mode)
            built.add(id(emb))     # tells a hit from a miss once get_many() hands the list back in prompt order
            print(f"serve: image {image.size[0]}x{image.size[1]} -> {emb.mm_length} tokens, tower {(time.perf_counter() - t0) * 1000:.0f} ms", flush=True)
            return emb

        mine: set = set()          # the pictures of *this* prompt: the conversation this request serves
        out = cache.get_many(images, build, evicted, fetch, mine)
        if disk is not None:
            # Every picture of this request keeps its identity, so a later restore reproduces the prompt; then
            # the ones that just left memory are written, outside the cache lock. A session that fits in
            # --image-cache-mb writes nothing, one that does not writes each evicted picture once
            # (see image_cache_disk.py). The cache hands over (key, embedding, position-in-this-prompt), the
            # conversations it still remembers and the keys of this one, so the collector can give up an
            # abandoned conversation first and never release the cache of a conversation that is being served.
            for (_, key), emb in zip(images, out):
                disk.remember(key, emb)
            disk.store(evicted, mode, live=cache.live_orders, protect=mine)
            if stats is not None:
                stats["disk"] = disk.report()
        if stats is not None:
            missed = [id(emb) in built for emb in out]
            misses = sum(missed)
            stats.update({
                "imgs": len(out), "hits": len(out) - misses, "misses": misses,
                "first_missing": next((i for i, m in enumerate(missed) if m), None),
                "entries": len(cache), "mb": round(cache.nbytes / 2 ** 20, 1),
                "budget_mb": round(float(getattr(cache, "max_bytes", 0)) / 2 ** 20, 1),
                "item_cap": int(getattr(cache, "max_items", 0) or 0),
                # real evictions, not a difference of counts: a picture that came back from disk also enters
                # the cache, so entries_before + misses - len() would miss those
                "evicted": len(evicted),
            })
        return out

    def count_tokens(self, text: str) -> int:
        return int(self.tokenizer.encode(text, encode_special_tokens=True).numel())

    def prompt_ids(self, prompt: str):
        if self.pec and self.pec.n:
            return self.pec.encode(prompt)[0]
        ids = self.tokenizer.encode(prompt, encode_special_tokens=True)
        ids = ids[0] if isinstance(ids, tuple) else ids
        return ids.reshape(-1)

    def encode_ids(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """POST /tokenize: the ids of text (special-token text is encoded as the special token; add_special_tokens adds BOS)."""
        return [int(t) for t in self.tokenizer.encode(text, add_bos=add_special_tokens, encode_special_tokens=True).flatten().tolist()]

    async def generate(self, prompt: str, *, max_tokens: int, temperature: float,
                       top_p: float, stop: list[str], sampler_kw: dict | None = None,
                       cancel: asyncio.Event | None = None,
                       images: list | None = None, stats: dict | None = None, img_stats: dict | None = None,
                       prompt_ids: list[int] | None = None, ignore_eos: bool = False,
                       loop_guard: bool | None = None) -> AsyncIterator[str]:
        """Yield incremental decoded text. With sessions == 1 the caller serializes access to this method; with
        sessions > 1 several calls may run at once and the final stats of a request land in `stats`.

        `img_stats` (optional) receives the image-embedding cache hit/miss counts of this request; it is kept
        apart from `stats` because `stats` is the Generator's own end-of-job report.

        When `cancel` is set (client gone) the job is cancelled between two generator steps. A step is one
        prefill chunk or one decode step, so a drop during prefill stops at the next chunk boundary."""
        import time
        import torch
        from exllamav3.generator.sampler import ComboSampler, GreedySampler
        t0 = time.perf_counter()
        await self.settle_reply()
        probe = getattr(self, "rss_probe", None)
        embs = []
        if prompt_ids is not None:   # token-id prompt: exactly these ids, no template, no BOS, no re-encoding
            ids = torch.tensor([prompt_ids], dtype=torch.long)
            if getattr(self, "slot_store", None) is not None:
                self.slot_store.note_prompt("", ids)
        elif images:
            if self.sessions > 1:   # the tower must not run next to a Generator step: run it on the driver, between steps
                embs = await self._post(self._embed, images, img_stats)
            else:
                embs = await asyncio.get_running_loop().run_in_executor(None, self._embed, images, img_stats)
            for emb in embs:
                prompt = prompt.replace(IMAGE_TRIPLE, emb.text_alias, 1)
            ids = self.tokenizer.encode(prompt, encode_special_tokens=True, embeddings=embs)
            if getattr(self, "slot_store", None) is not None:
                self.slot_store.erase()   # embedding ids are per process: a saved image slot could never be matched again
            room = reply_room(self, int(ids.numel()))
            max_tokens = min(max_tokens, room) if room else max_tokens
        else:
            ids = self.prompt_ids(prompt).reshape(1, -1)
            if getattr(self, "slot_store", None) is not None:
                self.slot_store.note_prompt(prompt, ids)
        self.last_stats = {}
        if cancel is not None and cancel.is_set():
            return                          # the client left while the request was queued: no prefill
        if getattr(self, "prefix_disk", None) is not None and not embs:
            await self.disk_restore(ids)
            self._disk_ids = ids
        kw = dict(sampler_kw if sampler_kw is not None else DEFAULT_SAMPLER_KW)  # rep_p / pres_p / dry_* (loop2)
        if temperature == 0:
            sampler = ComboSampler(temperature=0.0, **kw) if kw else GreedySampler()
        else:
            sampler = ComboSampler(temperature=max(temperature, 1e-6), top_p=top_p, min_p=0.0, **kw)
        generator = self.greedy_generator
        conditions = ([] if ignore_eos else list(self.config.eos_token_id_list)) + stop
        rq = serve_compat.rq_budget(getattr(self, "ctx", None), self.sessions, int(ids.numel()), max_tokens, getattr(self, "num_draft", 0))
        job = self.Job(input_ids=ids, max_new_tokens=max_tokens, sampler=sampler,
                       stop_conditions=conditions, embeddings=embs or None,
                       **({} if rq is None else {"max_rq_tokens": rq}))
        if self.sessions > 1:
            async for piece in self._run_shared(job, ids, cancel, stats if stats is not None else {}, t0, probe, prompt, ignore_eos, loop_guard):
                yield piece
            return
        loop = asyncio.get_running_loop()
        guard_on = loop_guard_default() if loop_guard is None else bool(loop_guard)   # loop_guard=False in the request turns it off
        tap = AnswerTap(opens_thinking(prompt)) if guard_on and not ignore_eos else None
        seen = 0                            # tokens streamed so far, for the guard stop report
        generator.enqueue(job)
        try:
            while generator.num_remaining_jobs():
                if cancel is not None and cancel.is_set():
                    generator.cancel(job)
                    return
                batch = await loop.run_in_executor(None, lambda: list(generator.iterate()))
                for event in batch:
                    if event.get("error"):
                        raise RuntimeError(str(event["error"]))
                    text = event.get("text", "")
                    if text:
                        yield text
                    hit = guard_hit(tap, event)
                    if hit is not None and not event.get("eos"):
                        generator.cancel(job)
                        seen += token_count(event)
                        self.last_stats = guard_stats(hit, seen)
                        return
                    seen += token_count(event)
                    if event.get("eos"):
                        self.last_stats = event
                        if probe is not None:
                            probe.request(int(ids.numel()), int(event.get("new_tokens", 0)),
                                          time.perf_counter() - t0)
                        return
            torch.cuda.synchronize()
        finally:
            # Whatever way the call ends (error, client gone, cancelled task) no job stays in the Generator:
            # a leftover job would run inside the next request.
            if generator.num_remaining_jobs():
                generator.clear_queue()
            if getattr(generator, "deferred_jobs", None):    # EXL3_REPLY_FIRST: release (reply-end stash) while the reply goes out
                self._reply_first_fut = loop.run_in_executor(None, generator.flush_deferred)

    # Shared mode (sessions > 1). One driver task is the only code that touches the Generator: it runs queued commands
    # (enqueue, cancel, image tower, slot actions) between two iterate() calls and routes the events of each step to the
    # request that owns the job (by serial number).

    def _post(self, fn, *args) -> asyncio.Future:
        loop = asyncio.get_running_loop()
        if getattr(self, "_driver", None) is None or self._driver.get_loop() is not loop:
            self._cmds, self._routes, self._wake = [], {}, asyncio.Event()
            self._driver = loop.create_task(self._drive())
        elif self._driver.done():
            exc = None if self._driver.cancelled() else self._driver.exception()
            if exc is None:
                raise RuntimeError("the engine driver has stopped")
            print(f"serve: WARNING the engine driver crashed ({exc!r}); starting a new one", file=sys.stderr, flush=True)
            self._driver = loop.create_task(self._drive())
        fut = loop.create_future()
        self._cmds.append((fn, args, fut))
        self._wake.set()
        return fut

    async def _drive(self) -> None:
        try:
            await self._drive_loop()
        finally:   # cancelled or crashed: nothing will answer the waiting requests any more, so fail them now
            stopped = RuntimeError("the engine driver has stopped")
            for _, _, fut in self._cmds:
                if not fut.done():
                    fut.set_exception(stopped)
            self._cmds.clear()
            for q in self._routes.values():
                q.put_nowait(stopped)
            self._routes.clear()

    async def _drive_loop(self) -> None:
        loop = asyncio.get_running_loop()
        rf_wait = 0.0
        while True:
            while self._cmds:
                fn, args, fut = self._cmds.pop(0)
                try:
                    res = await loop.run_in_executor(None, fn, *args)
                except asyncio.CancelledError:
                    if not fut.done():
                        fut.set_exception(RuntimeError("the engine driver has stopped"))
                    raise
                except Exception as exc:                              # noqa: BLE001
                    if not fut.done():
                        fut.set_exception(exc)
                else:
                    if not fut.done():
                        fut.set_result(res)
            # read every round: _rebuild_generator (cache_salt change) replaces the Generator between two steps
            generator = self.greedy_generator
            if not generator.num_remaining_jobs():
                # EXL3_REPLY_FIRST: release the finished jobs (reply-end stash) once idle AND their handlers have
                # dropped their routes (forget is the last driver command a request waits for before its reply ends), at most
                # 50 ms after the step; a new request's iterate() releases them first anyway
                dj = getattr(generator, "deferred_jobs", None)
                if dj and not self._cmds:
                    if rf_wait < 0.05 and any(self._routes.get(getattr(j, "serial_number", None)) is not None for j in dj):
                        self._wake.clear()
                        try:
                            await asyncio.wait_for(self._wake.wait(), 0.005)
                        except asyncio.TimeoutError:
                            pass
                        rf_wait += 0.005
                        continue
                    rf_wait = 0.0
                    await loop.run_in_executor(None, generator.flush_deferred)
                    continue
                self._wake.clear()
                if not self._cmds:
                    await self._wake.wait()
                continue
            if self.row_budget and generator.draft_model is not None:
                # verify rows per round = busy sessions x (depth + 1): keep them within the budget (depth >= 1)
                generator.num_draft_tokens = max(1, min(self.max_draft, self.row_budget // generator.num_remaining_jobs() - 1))
            try:
                events = await loop.run_in_executor(None, lambda: list(generator.iterate()))
            except Exception as exc:                                  # noqa: BLE001
                # a failed step leaves every job in an unknown state: fail them all
                for q in self._routes.values():
                    q.put_nowait(exc)
                self._routes.clear()
                generator.clear_queue()
                continue
            by_serial: dict[int, list] = {}
            for ev in events:
                by_serial.setdefault(ev.get("serial"), []).append(ev)
            for serial, evs in by_serial.items():
                q = self._routes.get(serial)
                if q is not None:      # None: cancelled after the step started
                    q.put_nowait(evs)

    async def _run_shared(self, job, ids, cancel, stats: dict, t0: float, probe, prompt: str = "", ignore_eos: bool = False,
                          loop_guard: bool | None = None) -> AsyncIterator[str]:
        """One request's job in the shared Generator: enqueue on the driver, stream its events, and (whatever way this
        ends) remove the job, so no leftover job runs inside another request."""
        q: asyncio.Queue = asyncio.Queue()
        generator = self.greedy_generator
        guard_on = loop_guard_default() if loop_guard is None else bool(loop_guard)
        tap = AnswerTap(opens_thinking(prompt)) if guard_on and not ignore_eos else None
        seen = 0

        def enqueue():
            self._routes[generator.enqueue(job)] = q   # on the driver: no step can emit before the route exists
        await self._post(enqueue)
        try:
            while True:
                if cancel is not None and cancel.is_set():
                    return
                try:
                    item = await asyncio.wait_for(q.get(), 0.25)
                except asyncio.TimeoutError:
                    continue
                if isinstance(item, Exception):
                    raise item
                for event in item:
                    if event.get("error"):
                        raise RuntimeError(str(event["error"]))
                    text = event.get("text", "")
                    if text:
                        yield text
                    hit = guard_hit(tap, event)
                    if hit is not None and not event.get("eos"):
                        stats.update(guard_stats(hit, seen + token_count(event)))
                        self.last_stats = stats
                        return                  # the finally block cancels the job
                    seen += token_count(event)
                    if event.get("eos"):
                        stats.update(event)
                        self.last_stats = stats
                        if probe is not None:
                            probe.request(int(ids.numel()), int(event.get("new_tokens", 0)), time.perf_counter() - t0)
                        return
        finally:
            def forget():
                serial = job.serial_number
                if self._routes.get(serial) is q:
                    del self._routes[serial]
                for j in generator.active_jobs + generator.pending_jobs:
                    if j is job:
                        generator.cancel(j)
            if not self._driver.done():
                await asyncio.shield(self._post(forget))

    async def run_exclusive(self, fn, *args):
        """Run fn on the driver, between steps (slot save/restore; the caller has drained the requests)."""
        return await self._post(fn, *args)


def warmup(engine: Any, template: str, prompt_tokens: int = 4608, decode_tokens: int = 16) -> float:
    """Run one ~4.6K-token prefill (a full 4096-row chunk plus a tail) and a short MTP decode before serving,
    so Triton JIT, allocator growth and decode graph capture do not land on the first user request."""
    base = " ".join(str(i) for i in range(1000))
    per = max(engine.count_tokens(base), 1) / 1000
    words = " ".join(str(i) for i in range(int(prompt_tokens / per)))
    prompt = render_prompt(template, [{"role": "user", "content": words + "\nCount on."}], None)

    async def run() -> None:
        async for _ in engine.generate(prompt, max_tokens=decode_tokens, temperature=0.0, top_p=1.0, stop=[]):
            pass

    t0 = time.perf_counter()
    asyncio.run(run())
    return time.perf_counter() - t0


def warm_indexer(engine) -> int:
    """Compile the DSA indexer scorer at the power-of-two score strides a decode round reaches beyond one 32K tile
    (first decode past 131072 tokens paid two Triton compiles in the user's first round). See
    MLAttention.warm_indexer_strides. One representative attention per distinct (heads, dim, pool, page) config."""
    import torch
    seen, n = set(), 0
    for m in engine.model.modules:
        attn = getattr(m, "attn", None)
        if attn is None or not hasattr(attn, "warm_indexer_strides") or getattr(attn, "indexer_mode", None) != "full":
            continue
        key = (attn.index_n_heads, attn.index_head_dim, attn.index_kpool)
        if key in seen:
            continue
        seen.add(key)
        n += attn.warm_indexer_strides(engine.cache, engine.cache.max_num_tokens)
    torch.cuda.synchronize()
    return n


def mem_line(stage: str) -> str:
    """GTT (whole unified-memory budget, all processes) next to this process's torch allocator."""
    import torch
    try:
        gtt = int(Path("/sys/class/drm/card1/device/mem_info_gtt_used").read_text()) / 2**30
    except OSError:
        gtt = float("nan")
    try:
        rss = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**30
    except OSError:
        rss = float("nan")
    return (f"serve: mem[{stage}] gtt {gtt:.1f} GB, torch alloc {torch.cuda.memory_allocated() / 2**30:.1f} "
            f"reserved {torch.cuda.memory_reserved() / 2**30:.1f} GB, rss {rss:.1f} GB")


def mem_diag(stage: str) -> None:
    """EXL3_SERVE_MEMDIAG=1: what the caching allocator holds (allocated by block size, inactive slack)."""
    import collections
    import torch
    act, ina = collections.Counter(), []
    for seg in torch.cuda.memory_snapshot():
        for blk in seg["blocks"]:
            if blk["state"] == "active_allocated":
                act[blk["size"]] += 1
            else:
                ina.append(blk["size"])
    big = sorted(((sz * n, sz, n) for sz, n in act.items()), reverse=True)[:14]
    small = sum(sz * n for sz, n in act.items() if sz < 2**26)
    print(f"serve: memdiag[{stage}] active >=64MB: " +
          ", ".join(f"{n}x{sz / 2**20:.0f}MB" for _, sz, n in big) +
          f"; active <64MB total {small / 2**30:.2f} GB; inactive total {sum(ina) / 2**30:.2f} GB, "
          f"largest {sorted(ina, reverse=True)[:5]}", flush=True)


def dense_tune_path() -> Path:
    """Same file the C++ dense GEMM tuner (hgemm.cu dtune) reads and appends to."""
    p = os.environ.get("EXL3_DENSE_GEMM_TUNE_FILE")
    return Path(p) if p else Path(os.environ.get("HOME", "/tmp")) / ".cache/exllamav3/dense_gemm_tune.txt"


def prime_dense_tune(budget_s: float = 600.0) -> tuple[int, float]:
    """Tune every dense prefill GEMM shape at every 256-row class before serving.

    The C++ tuner screens all rocBLAS solutions (~2.6 s per shape) the first time a prefill chunk
    lands in a new 256-row class, inside the request: a new tail length cost one ~40 s stall per
    class (15 shapes). Warm-up has just logged every shape it met, so cross those shapes with all
    classes 256..4096 and pay the tuning here. A cached key costs one GEMM. Tuned winners are
    bit-exact with the default path, so outputs do not change. Returns (keys, seconds)."""
    startup_health.stage("dense GEMM tuning")   # entered even when nothing is left to tune: a clean stage list
    path = dense_tune_path()
    if os.environ.get("EXL3_DENSE_GEMM_TUNE", "1") == "0" or not path.exists():
        return 0, 0.0
    # Imported after the early return, so a disabled tuner or a missing cache file needs no built extension.
    import torch
    from exllamav3.ext import exllamav3_ext as ext
    # Only the row classes a prefill chunk can reach (<= 4096); the cache file may hold bigger classes
    # from other runs, and sizing the buffers for them cost ~6 GB of GTT on a 128 GB box.
    max_class = int(os.environ.get("EXL3_SERVE_DTUNE_MAX_CLASS", "4096"))
    shapes, classes, have, tag = set(), set(range(256, max_class + 1, 256)), set(), None
    for line in path.read_text().splitlines():
        f = line.split()
        if len(f) == 7:
            shapes.add((int(f[1]), int(f[2]), int(f[4]), int(f[5])))
            have.add((f[0], int(f[1]), int(f[2]), int(f[3]), int(f[4]), int(f[5])))
            tag = f[0]   # the last line carries the running library's tag (the C++ side appends)
    # Keys the C++ tuner already holds cost nothing to skip: no buffers, no GEMM.
    todo = [(n, k, ldn, f32, mc) for n, k, ldn, f32 in sorted(shapes) for mc in sorted(classes)
            if (tag, n, k, mc, ldn, f32) not in have]
    print(f"serve: dense GEMM prime: {len(todo)} of {len(shapes) * len(classes)} keys missing", flush=True)
    if not todo:
        return 0, 0.0
    t0, done = time.perf_counter(), 0
    gen = torch.Generator(device="cuda").manual_seed(0)
    # One set of buffers sized for the largest missing key, viewed per key (a/b/c per key left one caching
    # allocator block per distinct size, ~38 GB reserved on td205).
    def cbytes(n, ldn, f32, mc):
        return mc * (n if ldn else n + 64) * (4 if f32 else 2)
    a_buf = torch.randn(max(mc * k for n, k, _, _, mc in todo), device="cuda", dtype=torch.half, generator=gen)
    b_buf = torch.empty(max(k * n for n, k, _, _, _ in todo), device="cuda", dtype=torch.half)
    c_buf = torch.empty(max(cbytes(n, ldn, f32, mc) for n, _, ldn, f32, mc in todo), device="cuda", dtype=torch.uint8)
    print(mem_line("prime-bufs"), flush=True)
    last, shown = None, t0
    for n, k, ldn, f32, mc in todo:
        if time.perf_counter() - t0 > budget_s:
            break
        if time.perf_counter() - shown > 30.0:   # the port opens only after this pass: show it is alive
            shown = time.perf_counter()
            print(f"serve: dense GEMM prime: {done} of {len(todo)} keys, {shown - t0:.0f} s", flush=True)
        dt = torch.float32 if f32 else torch.half
        w = n if ldn else n + 64
        if last != (n, k):   # finite data: the tuner keeps only winners whose output equals the default's
            b_buf[:k * n].view(k, n).normal_(0, 0.02, generator=gen)
            last = (n, k)
        a = a_buf[:mc * k].view(mc, k)
        b = b_buf[:k * n].view(k, n)
        c = c_buf[:mc * w * dt.itemsize].view(dt).view(mc, w)
        ext.hgemm_recon(a, b, c if ldn else c[:, :n])
        done += 1
        startup_health.progress(done, len(todo))
    torch.cuda.synchronize()
    del a_buf, b_buf, c_buf
    torch.cuda.empty_cache()
    return done, time.perf_counter() - t0


# Every route of create_app(), for the early /health listener (404 / 405 while loading); a test keeps it in step.
ROUTES = {"/health": ("GET",), "/v1/models": ("GET",), "/metrics": ("GET",), "/slots": ("GET",), "/v1/chat/completions": ("POST",), "/apply-template": ("POST",), "/tokenize": ("POST",), "/completion": ("POST",), "/v1/completions": ("POST",), "/slots/{id}": ("POST",)}


def create_app(engine: Any, model_id: str, template: str) -> web.Application:
    """Build the HTTP layer around a resident engine (also accepts a fake engine in tests)."""
    queue: asyncio.Queue[tuple[dict[str, Any], asyncio.Queue, dict[str, Any]]] = asyncio.Queue()
    lock = asyncio.Lock()  # serializes generation with slot save/restore
    sessions = int(getattr(engine, "sessions", 1))
    gate = asyncio.Semaphore(sessions)   # sessions > 1: one permit per running request; slot actions take them all
    slot_lock = asyncio.Lock()           # sessions > 1: one slot action at a time
    salt_state: dict[str, Any] = {"current": None}   # cache_salt namespace of the prompt cache (None = requests without a salt)
    admit_lock = asyncio.Lock()          # sessions > 1: serializes admission
    processing = 0  # admitted requests; closure, app config is immutable after startup
    running = 0     # sessions > 1: requests inside generate()
    app = web.Application(client_max_size=64 * 1024**2, middlewares=[startup_health.json_errors_middleware()])
    app.update(engine=engine, model_id=model_id, template=template, queue=queue, metrics=Metrics())

    # Request log: off unless KYOJIN_REQLOG_DIR names a directory, in which case every generation request gets a line
    # per lifecycle event (numbers and event names only, never prompt or reply text) plus an 'alive' heartbeat
    # while it is queued or running. `inflight` is what the heartbeat walks; it holds no request content.
    reqlog = ReqLog.from_env(server="glm")
    app.update(reqlog=reqlog)
    inflight: dict[str, dict[str, Any]] = {}
    background: dict[str, Any] = {}      # the heartbeat task, created at start-up (nothing when the log is off)

    # SSE keep-alive: off unless KYOJIN_SSE_PING_S names an interval, in which case a stream that has
    # nothing to send yet (queued, or prefilling a long prompt) writes a ':' comment line every N
    # seconds. The pinger is per request; the switch and the start-up line are shared by the three servers.
    ping = ServePing.from_env(reqlog)
    app.update(ping=ping)

    # The lifecycle a request goes through is logged by one shared implementation (tools/strix_halo/
    # serve_reqlog.py): the request table, the id every line carries, the 'alive' heartbeat and the route
    # wrapper. This server only says when a phase begins; the events it has and the counters it keeps are
    # the ones its own call sites pass. `processing`/`running` are this server's two counts.
    lifecycle = Lifecycle(reqlog, ping=ping,
                          counters=lambda: {"processing": processing, "running": running})

    def reqid() -> str | None:
        """The log id of the request being handled (None outside a logged route)."""
        return lifecycle.reqid()

    writers = lifecycle.writers          # the streaming writer per in-flight request (the heartbeat reads it)

    def track(rid: str | None, phase: str, **fields: Any) -> None:
        lifecycle.track(rid, phase, **fields)

    def salt_of(body: dict[str, Any]) -> str | None:
        try:
            salt = serve_compat.cache_salt(body)
        except serve_compat.CompatError as exc:
            raise web.HTTPBadRequest(reason=str(exc)) from exc
        if salt is not None and not hasattr(engine, "reset_prefix_cache"):
            raise web.HTTPBadRequest(reason="cache_salt is not supported by this engine")
        return salt

    def observe(st: dict[str, Any], prompt_tokens: int) -> None:
        # The MTP window is a fixed num_draft_tokens per round, so rounds = proposed / window
        # (exact except for a final window truncated by max_new_tokens). No generator -> no rounds.
        num_draft = int(getattr(getattr(engine, "greedy_generator", None), "num_draft_tokens", 0) or 0)
        accepted = int(st.get("accepted_draft_tokens", 0))
        rejected = int(st.get("rejected_draft_tokens", 0))
        proposed = accepted + rejected
        app["metrics"].observe(
            prompt_tokens=prompt_tokens, cached=int(st.get("cached_tokens", 0)),
            predicted=int(st.get("new_tokens", 0)),
            prefill_s=float(st.get("time_prefill") or 0.0), generate_s=float(st.get("time_generate") or 0.0),
            drafts=proposed // num_draft if proposed and num_draft else 0,
            draft_tokens=proposed, accepted=accepted)

    async def health(_: web.Request) -> web.Response:
        return web.json_response({"status": "ok", "source": "kyojin", "model": model_id,
                                  "vision": bool(getattr(engine, "supports_vision", False)),
                                  **startup_health.compiler_health()})

    async def models(_: web.Request) -> web.Response:
        return web.json_response({"object": "list", "data": [
            {"id": model_id, "object": "model", "created": 0, "owned_by": "local"}]})

    @contextlib.asynccontextmanager
    async def exclusive():
        """The generator is idle inside: sessions == 1 holds the lock; sessions > 1 waits for every running request."""
        if sessions == 1:
            async with lock:
                yield
            return
        async with slot_lock:
            got = 0
            try:
                for _ in range(sessions):
                    await gate.acquire()
                    got += 1
                yield
            finally:
                for _ in range(got):
                    gate.release()

    @contextlib.asynccontextmanager
    async def admit(salt: str | None):
        """Permission to run one request. Every running request shares one cache_salt namespace: when the salt changes, the
        requests in flight finish, the prompt cache is dropped (engine.reset_prefix_cache) and the new request starts alone.
        The same salt (and traffic with no salt at all) never drops anything."""
        async def reset() -> None:
            salt_state["current"] = _UNKNOWN_SALT            # a failed reset must not leave the old namespace claimed
            await engine.reset_prefix_cache()
            salt_state["current"] = salt

        if sessions == 1:
            async with lock:
                if salt != salt_state["current"]:
                    await reset()
                yield
            return
        async with admit_lock:
            if salt == salt_state["current"]:
                await gate.acquire()
            else:
                async with slot_lock:
                    held = 0
                    try:
                        for _ in range(sessions):
                            await gate.acquire()
                            held += 1
                        await reset()
                    except BaseException:
                        for _ in range(held):
                            gate.release()
                        raise
                    for _ in range(held - 1):                # keep the permit of this request
                        gate.release()
        try:
            yield
        finally:
            gate.release()

    async def worker() -> None:
        nonlocal processing, running
        while True:
            body, out, request_stats = await queue.get()
            cancel = body.get("_cancel")
            rid = body.get("_reqid")
            if cancel is not None and cancel.is_set():   # the client left while the job was queued: no prefill, no decode
                lifecycle.job_skipped(rid, time.perf_counter() - (body.get("_queued_at") or time.perf_counter()))
                out.put_nowait(None)
                queue.task_done()
                continue
            store = getattr(engine, "slot_store", None)
            processing += 1  # admitted: counted from the dequeue, incl. lock and slot restore
            started = time.perf_counter()
            first_token = False             # flipped once, so the log gets exactly one job_first_token line
            try:
                async with admit(body.get("_cache_salt")):
                    running += 1
                    lifecycle.job_started(rid, started - (body.get("_queued_at") or started),
                                          prompt_tokens=body.get("_prompt_tokens"),
                                          images=len(body.get("_images") or ()),
                                          messages=body.get("_messages"), sessions=sessions,
                                          processing=processing, running=running)
                    if store is not None:
                        store.busy = True
                    if sessions == 1:
                        engine.last_stats = {}  # a cancelled or failed job must not re-count the previous one
                    try:
                        async for delta in engine.generate(
                            body["_prompt"],
                            max_tokens=token_limit(body), temperature=body.get("temperature", 0.0),
                            top_p=body.get("top_p", 1.0), stop=body.get("_stop", []),
                            cancel=body.get("_cancel"), img_stats=body.get("_img_stats"),
                            **({"images": body["_images"]} if body.get("_images") else {}),
                            **({"prompt_ids": body["_prompt_ids"]} if body.get("_prompt_ids") is not None else {}),
                            **({"ignore_eos": True} if body.get("_ignore_eos") else {}),
                            **({"loop_guard": body["_loop_guard"]} if body.get("_loop_guard") is not None else {}),
                            **({"stats": request_stats} if sessions > 1 else {})):
                            if not first_token:
                                first_token = True
                                lifecycle.first_token(rid, started,
                                                      prompt_tokens=body.get("_prompt_tokens"))
                            out.put_nowait(delta)
                    finally:
                        running -= 1
                        if sessions > 1 and store is not None:
                            store.busy = running > 0
                    if sessions == 1:
                        request_stats.update(getattr(engine, "last_stats", None) or {})
                    if request_stats:
                        observe(request_stats, body.get("_prompt_tokens") or int(request_stats.get("prompt_tokens", 0)))
                    log_job_done(rid, body, request_stats, started)
                    out.put_nowait(None)
                    if getattr(engine, "prefix_disk", None) is not None:
                        await engine.disk_save()      # the reply is out: the host copy of the new pages holds the lock, the write does not
            except Exception as exc:
                lifecycle.job_error(rid, exc, started)
                out.put_nowait(exc)
            finally:
                processing -= 1
                if store is not None and sessions == 1:
                    store.busy = False
                queue.task_done()

    def log_job_done(rid: str | None, body: dict[str, Any], st: dict[str, Any], started: float) -> None:
        """One line per finished job: what the prompt cache did (cache_n vs prompt tokens) and how long each
        phase took. This is the line that turns 'the client saw nothing for 290 s' into a number."""
        prompt_tokens = int(st.get("prompt_tokens") or body.get("_prompt_tokens") or 0)
        cache_n = int(st.get("cached_tokens") or 0)
        lifecycle.job_done(rid, st, started, prompt_tokens=prompt_tokens)
        if body.get("_img_stats"):
            reqlog.log("img_cache", req=rid, **body["_img_stats"])

    async def start(_: web.Application) -> None:
        app["worker_tasks"] = [asyncio.create_task(worker()) for _ in range(sessions)]
        app["worker_task"] = app["worker_tasks"][0]
        if reqlog.enabled and reqlog.heartbeat_s > 0:
            background["heartbeat"] = asyncio.create_task(lifecycle.heartbeat_forever())

    async def close(app_: web.Application) -> None:
        for t in app_["worker_tasks"]:
            t.cancel()
        beat = background.pop("heartbeat", None)
        if beat is not None:
            beat.cancel()
        reqlog.close()

    async def drain(request: web.Request, cancel: asyncio.Event, out: asyncio.Queue,
                    writer: StreamWriter | None = None, ping: ServePing | None = None,
                    ping_at: float | None = None) -> AsyncIterator[str]:
        """Items the worker produced for one request. Nothing is written to the client while a job is
        queued or prefilling (and never, for a non-streaming reply), so a client that hung up is noticed
        by polling the transport. Raises ConnectionResetError and sets `cancel` when it has left.

        For a streaming reply (`writer`, `ping`, `ping_at` given), the keep-alive is asked before each
        poll whether a comment is due: the client sees that the server is still working on a prompt (or
        on a held-back tool call) it has not answered yet, which is all it can see until then."""
        while True:
            tr = request.transport
            if tr is None or tr.is_closing():
                cancel.set()
                if ping is not None:
                    ping.limit()
                raise ConnectionResetError("client left")
            if ping is not None:
                try:                              # before the wait, so a stall is covered from its start
                    await ping.comment(writer, ping_at or 0.0)
                except ConnectionResetError:
                    cancel.set()
                    raise
            try:
                item = await asyncio.wait_for(out.get(), 0.5)
            except asyncio.TimeoutError:
                continue
            if item is None:
                return
            if isinstance(item, Exception):
                raise item
            yield item

    def client_left(cancel: asyncio.Event, path: str) -> web.Response:
        cancel.set()
        lifecycle.client_left(path)
        print(f"serve: client left {path}, job cancelled", file=sys.stderr, flush=True)
        return web.Response(status=499)

    async def completions(request: web.Request) -> web.StreamResponse:
        try:
            body = await request.json()
            if body.get("model", model_id) != model_id:
                raise web.HTTPBadRequest(reason=f"unknown model; expected {model_id}")
            if not isinstance(body.get("messages"), list) or not body["messages"]:
                raise web.HTTPBadRequest(reason="messages must be a non-empty array")
            if "stream" in body and not isinstance(body["stream"], bool):
                raise web.HTTPBadRequest(reason="stream must be boolean")
            if body.get("loop_guard") is not None and not isinstance(body["loop_guard"], bool):
                raise web.HTTPBadRequest(reason="loop_guard must be boolean")
            body["_loop_guard"] = body.get("loop_guard")   # False: this request runs without the answer-phase loop guard
            stops = body.get("stop", [])
            body["_stop"] = [stops] if isinstance(stops, str) else list(stops)
            urls = extract_images(body["messages"])
            if urls and not getattr(engine, "supports_vision", False):
                raise web.HTTPBadRequest(reason="this server runs without the vision tower (--no-vision or no vision weights in the model folder): it cannot read images")
            prompt = render_prompt(template, body["messages"], body.get("tools"), **template_kwargs(body))
            if len(urls) != prompt.count(IMAGE_TRIPLE):
                raise web.HTTPBadRequest(reason="image parts and image placeholders do not match (the chat template must write one image placeholder per image part)")
            images = [load_image(u) for u in urls]
            body["_images"] = images or None
            prompt_tokens = int(engine.prompt_ids(prompt).numel()) + sum(engine.image_tokens(im) - 1 for im, _ in images)
            body["_prompt_tokens"] = prompt_tokens
            body["_room"] = reply_room(engine, prompt_tokens)
            body["_prompt"] = prompt
            body["_cache_salt"] = salt_of(body)
            for key in ("temperature", "top_p"):
                if body.get(key) is None:
                    body[key] = SERVE_DEFAULTS[key]
        except (json.JSONDecodeError, web.HTTPException, TypeError) as exc:
            if isinstance(exc, web.HTTPException):
                raise
            raise web.HTTPBadRequest(reason=str(exc)) from exc

        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        base = {"id": request_id, "object": "chat.completion.chunk", "created": created, "model": model_id}

        def event(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
            return base | {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

        out: asyncio.Queue = asyncio.Queue()
        request_stats: dict[str, Any] = {}
        cancel = body["_cancel"] = asyncio.Event()
        body["_reqid"] = reqid()
        body["_messages"] = len(body["messages"])
        body["_img_stats"] = {} if images else None
        body["_queued_at"] = time.perf_counter()
        ahead = queue.qsize()                       # requests already waiting when this one arrived
        await queue.put((body, out, request_stats))
        lifecycle.queued(body["_reqid"], "/v1/chat/completions", queued_ahead=ahead,
                         prompt_tokens=prompt_tokens, messages=body["_messages"], images=len(images),
                         stream=bool(body.get("stream")))
        prefix = "<think>" if opens_thinking(prompt) else ""
        # This request's SSE keep-alive, and the moment its wait started: every comment counts from
        # there, which is what the client has actually been waiting for. Off (the default) this costs one
        # comparison per empty poll. A JSON reply has no stream to write a comment to, so `deltas` takes
        # the response per call instead of closing over one that a non-streaming request never sets.
        ping = app["ping"].per_request() if body.get("stream", False) else None
        ping_at = body["_queued_at"]

        def deltas(writer: StreamWriter | None = None) -> AsyncIterator[str]:
            return drain(request, cancel, out, writer, ping, ping_at)

        def usage(raw: str) -> dict[str, int]:
            return serve_compat.usage(prompt_tokens, engine.count_tokens(raw), int(request_stats.get("cached_tokens") or 0))

        def finish_for(message: dict[str, Any], raw: str) -> str:
            # A reply cut by max_tokens is "length", not "stop" (clients use it to tell a truncated answer).
            finish = message.pop("finish_reason", "stop")
            if finish == "stop" and (request_stats.get("eos_reason") == "loop_guard"
                                     or engine.count_tokens(raw) >= token_limit(body)):
                finish = "length"
            return finish

        def timings(raw: str) -> dict[str, Any]:
            # llama.cpp names: the swap orchestrator reads cache_n as the verdict of a KV restore
            st = request_stats
            cache_n = int(st.get("cached_tokens", 0))
            pre, gen = st.get("time_prefill") or 0.0, st.get("time_generate") or 0.0
            n = int(st.get("new_tokens", 0))
            return {"cache_n": cache_n, "prompt_n": max(prompt_tokens - cache_n, 0),
                    "prompt_ms": pre * 1000, "predicted_n": n, "predicted_ms": gen * 1000,
                    "prompt_per_second": (prompt_tokens - cache_n) / pre if pre else 0.0,
                    "predicted_per_second": n / gen if gen else 0.0,
                    **({"loop_guard": st["loop_guard"]} if st.get("loop_guard") else {})}

        if body.get("stream", False):
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
            await response.prepare(request)
            writer = lifecycle.stream_started(body["_reqid"], "/v1/chat/completions", response,
                                              prompt_tokens=prompt_tokens)
            # Reasoning and prose stream as they decode. Tool calls are parsed only from complete
            # XML, so everything from <tool_call> on is held and sent once at the end.
            text, done, mode = prefix, len(prefix), "reasoning" if prefix else "content"
            holds = ["</think>", "<think>", TOOL_OPEN] + [x for x in body["_stop"] if x]
            visible: list[float] = []   # set at the first frame with text in it: the client's first sign of life

            async def flush(final: bool) -> None:
                nonlocal done, mode
                while mode != "tool":
                    tags = ["</think>"] if mode == "reasoning" else ["<think>", TOOL_OPEN]
                    hits = [(text.find(t, done), t) for t in tags if text.find(t, done) >= 0]
                    end = min(hits)[0] if hits else (len(text) if final else max(done, len(text) - held_tail(text, holds)))
                    piece = stop_text(text[done:end], body["_stop"])[0]
                    if piece:
                        if not visible:
                            visible.append(time.perf_counter())
                            lifecycle.first_content(body["_reqid"], body["_queued_at"])
                        key = "reasoning_content" if mode == "reasoning" else "content"
                        await writer.write(sse(event({key: piece})))
                        ping.note_output()   # the client has text: comments wait for the next silence
                    done = end
                    if not hits:
                        return
                    pos, tag = min(hits)
                    done = pos + len(tag)
                    mode = {"</think>": "content", "<think>": "reasoning", TOOL_OPEN: "tool"}[tag]

            try:
                await writer.write(sse(event({"role": "assistant", "content": ""})))
                ping.note_output()
                async for delta in deltas(writer):
                    text += delta
                    await flush(False)
                text, _ = stop_text(text, body["_stop"])
                await flush(True)
                message = parse_completion(text)
                if "tool_calls" in message:
                    for n, call in enumerate(message["tool_calls"]):   # one stream message per call (clients that read one call per message)
                        await writer.write(sse(event({"tool_calls": [dict(call, index=n)]})))
                final = event({}, finish_for(message, text[len(prefix):])) | {"usage": usage(text[len(prefix):]), "timings": timings(text)}
                if request_stats.get("eos_reason") == "loop_guard":
                    final["choices"][0]["stop_reason"] = "loop_guard"
                await writer.write(sse(final))
                await writer.write(b"data: [DONE]\n\n")
                await response.write_eof()
                lifecycle.stream_done(body["_reqid"], int(request_stats.get("new_tokens") or 0))
            except ConnectionResetError:
                client_left(cancel, "stream")                        # client left: nothing to send
                return response
            except asyncio.CancelledError:
                cancel.set()
                raise
            except Exception as exc:                                   # noqa: BLE001
                cancel.set()
                lifecycle.stream_error(body["_reqid"], exc)
                # Fail inside the stream, but still end it the OpenAI way: clients such as OpenCode report
                # "stream ended without finish_reason" and retry when the connection just drops.
                try:
                    await writer.write(sse({"error": {"message": f"{type(exc).__name__}: {exc}"}}))
                    await writer.write(sse(event({}, "error")))
                    await writer.write(b"data: [DONE]\n\n")
                    await response.write_eof()
                except ConnectionError:
                    pass
            return response

        try:
            raw = "".join([d async for d in deltas()])
        except ConnectionResetError:
            return client_left(cancel, "request")
        except asyncio.CancelledError:
            cancel.set()
            raise
        except Exception:
            cancel.set()
            raise
        text, _ = stop_text(prefix + raw, body["_stop"])
        message = parse_completion(text)
        finish = finish_for(message, text[len(prefix):])
        payload = {
            "id": request_id, "object": "chat.completion", "created": created, "model": model_id,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": usage(text[len(prefix):]), "timings": timings(text)}
        if request_stats.get("eos_reason") == "loop_guard":
            payload["choices"][0]["stop_reason"] = "loop_guard"
        return web.json_response(payload)

    async def apply_template(request: web.Request) -> web.Response:
        """llama.cpp's POST /apply-template: render the chat template, generate nothing.

        Same render_prompt() call as /v1/chat/completions, so what the conformance tool reads here is
        byte-for-byte what a request would be sent.
        """
        try:
            body = await request.json()
            if not isinstance(body.get("messages"), list) or not body["messages"]:
                raise web.HTTPBadRequest(reason="messages must be a non-empty array")
            prompt = render_prompt(template, body["messages"], body.get("tools"), **template_kwargs(body))
        except (json.JSONDecodeError, web.HTTPException, TypeError) as exc:
            if isinstance(exc, web.HTTPException):
                raise
            raise web.HTTPBadRequest(reason=str(exc)) from exc
        return web.json_response({"prompt": prompt})

    async def completion(request: web.Request) -> web.Response:
        """Minimal llama.cpp POST /completion over the same queue/generator as /v1/chat/completions.

        The prompt is used verbatim (no chat template, no implicit <think> prefix), and the engine's
        generate() calls slot_store.note_prompt(), so /slots reflects the probe and the slot can be
        saved. "content" is the raw continuation, reasoning block included, as llama.cpp does not
        split <think> for this endpoint.
        """
        try:
            body = await request.json()
            prompt = body.get("prompt")
            if not isinstance(prompt, str) or not prompt:
                raise web.HTTPBadRequest(reason="prompt must be a non-empty string")
            n_predict = int(body.get("n_predict", 128) or 0)
            if n_predict <= 0:
                n_predict = 128
            n_predict = min(n_predict, 8192)
            room = reply_room(engine, engine.count_tokens(prompt))
            if room:
                n_predict = min(n_predict, room)
            stops = body.get("stop", [])
            job = {"_prompt": prompt, "max_tokens": n_predict,
                   "temperature": float(body.get("temperature", SERVE_DEFAULTS["temperature"])),
                   "top_p": float(body.get("top_p", SERVE_DEFAULTS["top_p"])),
                   "_stop": [stops] if isinstance(stops, str) else list(stops)}
        except (json.JSONDecodeError, web.HTTPException, TypeError, ValueError) as exc:
            if isinstance(exc, web.HTTPException):
                raise
            raise web.HTTPBadRequest(reason=str(exc)) from exc

        out: asyncio.Queue = asyncio.Queue()
        request_stats: dict[str, Any] = {}
        cancel = job["_cancel"] = asyncio.Event()
        await queue.put((job, out, request_stats))
        text = ""
        try:
            async for item in drain(request, cancel, out):
                text += item
        except ConnectionResetError:
            return client_left(cancel, "completion")
        except asyncio.CancelledError:
            cancel.set()
            raise
        except Exception:
            cancel.set()
            raise
        text, stopped = stop_text(text, job["_stop"])
        st = request_stats
        return web.json_response({
            "id": f"cmpl-{uuid.uuid4().hex}", "object": "completion", "created": int(time.time()),
            "model": model_id, "content": text,
            "tokens_cached": int(st.get("cached_tokens", 0)),
            "tokens_predicted": engine.count_tokens(text),
            "cache_prompt": bool(body.get("cache_prompt", True)),
            "stop": True, "stopped_eos": bool(st.get("eos")), "stopped_word": stopped,
            "timings": {"prompt_n": int(st.get("cached_tokens", 0)),
                        "predicted_n": int(st.get("new_tokens", 0))}})

    async def tokenize(request: web.Request) -> web.Response:
        """vLLM POST /tokenize: {prompt | messages, add_special_tokens?} -> {tokens, count, max_model_len}."""
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise web.HTTPBadRequest(reason="body must be a JSON object")
            if body.get("model", model_id) != model_id:
                raise web.HTTPBadRequest(reason=f"unknown model; expected {model_id}")
            if not hasattr(engine, "encode_ids"):
                raise web.HTTPNotImplemented(reason="this engine cannot tokenize")
            add_special = body.get("add_special_tokens")
            if add_special is not None and not isinstance(add_special, bool):
                raise web.HTTPBadRequest(reason="add_special_tokens must be boolean")
            if body.get("messages") is not None:
                if not isinstance(body["messages"], list) or not body["messages"]:
                    raise web.HTTPBadRequest(reason="messages must be a non-empty array")
                if extract_images(body["messages"]):
                    raise web.HTTPBadRequest(reason="images cannot be tokenized here")
                text = render_prompt(template, body["messages"], body.get("tools"), **template_kwargs(body))
                add_special = bool(add_special)          # a rendered chat prompt carries its own special tokens
            else:
                text = body.get("prompt")
                if not isinstance(text, str):
                    raise web.HTTPBadRequest(reason="prompt must be a string (or send messages)")
                add_special = True if add_special is None else add_special
            ids = engine.encode_ids(text, add_special)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise web.HTTPBadRequest(reason=str(exc)) from exc
        return web.json_response(serve_compat.tokenize_reply(ids, getattr(engine, "ctx", None)))

    async def v1_completions(request: web.Request) -> web.StreamResponse:
        """OpenAI POST /v1/completions over the same queue/generator as chat: the prompt (text, or a list of token ids) is
        used verbatim (no chat template, no BOS added), one prompt, one choice. Streamed with a final `usage`."""
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise web.HTTPBadRequest(reason="body must be a JSON object")
            if body.get("model", model_id) != model_id:
                raise web.HTTPBadRequest(reason=f"unknown model; expected {model_id}")
            prompt = body.get("prompt")
            try:
                prompt_ids = serve_compat.token_prompt(prompt, getattr(engine, "vocab_size", None))
            except serve_compat.CompatError as exc:
                raise web.HTTPBadRequest(reason=str(exc)) from exc
            if prompt_ids is not None:
                prompt = ""
            else:
                if isinstance(prompt, list) and len(prompt) == 1:
                    prompt = prompt[0]
                if not isinstance(prompt, str) or not prompt:
                    raise web.HTTPBadRequest(reason="prompt must be a non-empty string or a list of token ids (one prompt per request)")
            if body.get("n") not in (None, 1) or body.get("echo") or body.get("best_of") not in (None, 1):
                raise web.HTTPBadRequest(reason="n, best_of and echo are not supported")
            for key in ("stream", "ignore_eos", "loop_guard"):
                if body.get(key) is not None and not isinstance(body[key], bool):
                    raise web.HTTPBadRequest(reason=f"{key} must be boolean")
            max_tokens = 16 if body.get("max_tokens") is None else body["max_tokens"]
            if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
                raise web.HTTPBadRequest(reason="max_tokens must be a positive integer")
            stops = body.get("stop") or []
            stops = [stops] if isinstance(stops, str) else stops
            if not isinstance(stops, list) or not all(isinstance(x, str) for x in stops):
                raise web.HTTPBadRequest(reason="stop must be a string or a list of strings")
            prompt_tokens = len(prompt_ids) if prompt_ids is not None else engine.count_tokens(prompt)
            job = {"_prompt": prompt, "_prompt_ids": prompt_ids, "_ignore_eos": body.get("ignore_eos") is True,
                   "_loop_guard": body.get("loop_guard"),
                   "max_tokens": max_tokens, "_room": reply_room(engine, prompt_tokens), "_stop": list(stops),
                   "temperature": float(SERVE_DEFAULTS["temperature"] if body.get("temperature") is None else body["temperature"]),
                   "top_p": float(SERVE_DEFAULTS["top_p"] if body.get("top_p") is None else body["top_p"]),
                   "_cache_salt": salt_of(body)}
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise web.HTTPBadRequest(reason=str(exc)) from exc

        limit = token_limit(job)
        request_id, created = f"cmpl-{uuid.uuid4().hex}", int(time.time())
        out: asyncio.Queue = asyncio.Queue()
        request_stats: dict[str, Any] = {}
        cancel = job["_cancel"] = asyncio.Event()
        job["_reqid"] = reqid()
        job["_queued_at"] = time.perf_counter()
        ahead = queue.qsize()
        await queue.put((job, out, request_stats))
        lifecycle.queued(job["_reqid"], "/v1/completions", queued_ahead=ahead,
                         prompt_tokens=prompt_tokens, images=0, stream=bool(body.get("stream")))

        def chunk(text: str, finish: str | None) -> dict[str, Any]:
            return {"id": request_id, "object": "text_completion", "created": created, "model": model_id,
                    "choices": [{"text": text, "index": 0, "logprobs": None, "finish_reason": finish}]}

        def finish_and_usage(text: str, stopped: str | None) -> tuple[str, dict[str, Any]]:
            n_new = int(request_stats.get("new_tokens") or engine.count_tokens(text))
            reason = "length" if stopped is None and (request_stats.get("eos_reason") in ("max_new_tokens", "loop_guard") or n_new >= limit) else "stop"
            return reason, serve_compat.usage(prompt_tokens, n_new, int(request_stats.get("cached_tokens") or 0))

        hold = max((len(x) for x in job["_stop"]), default=1) - 1

        if body.get("stream", False):
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
            await response.prepare(request)
            writer = lifecycle.stream_started(job["_reqid"], "/v1/completions", response,
                                              prompt_tokens=prompt_tokens)
            # Same keep-alive as /v1/chat/completions, but this route writes no role chunk: the first
            # thing the client sees is text, so a comment covers the whole wait before it.
            ping = app["ping"].per_request()
            ping_at = job["_queued_at"]
            try:
                text, sent, stopped = "", 0, None
                async for delta in drain(request, cancel, out, writer, ping, ping_at):
                    text += delta
                    cut, stopped = stop_text(text, job["_stop"])
                    if stopped is not None:
                        cancel.set()            # a stop string ended the reply: the rest is not generated
                        text = cut
                        break
                    safe = max(sent, len(text) - hold)
                    if safe > sent:
                        if not sent:
                            lifecycle.first_content(job["_reqid"], job["_queued_at"])
                        await writer.write(sse(chunk(text[sent:safe], None)))
                        ping.note_output()
                        sent = safe
                if len(text) > sent:
                    await writer.write(sse(chunk(text[sent:], None)))
                reason, usage = finish_and_usage(text, stopped)
                last = chunk("", reason) | {"usage": usage}
                if request_stats.get("eos_reason") == "loop_guard":
                    last["choices"][0]["stop_reason"] = "loop_guard"
                    last["timings"] = {"loop_guard": request_stats.get("loop_guard")}
                await writer.write(sse(last))
                await writer.write(b"data: [DONE]\n\n")
                await response.write_eof()
                lifecycle.stream_done(job["_reqid"], int(request_stats.get("new_tokens") or 0))
            except ConnectionResetError:
                client_left(cancel, "stream")
            except asyncio.CancelledError:
                cancel.set()
                raise
            except Exception as exc:                                   # noqa: BLE001
                cancel.set()
                lifecycle.stream_error(job["_reqid"], exc)
                try:
                    await writer.write(sse({"error": {"message": f"{type(exc).__name__}: {exc}"}}))
                    await response.write_eof()
                except ConnectionError:
                    pass
            return response

        try:
            raw = ""
            async for delta in drain(request, cancel, out):
                raw += delta
        except ConnectionResetError:
            return client_left(cancel, "request")
        except asyncio.CancelledError:
            cancel.set()
            raise
        except Exception:
            cancel.set()
            raise
        text, stopped = stop_text(raw, job["_stop"])
        reason, usage = finish_and_usage(text, stopped)
        out_ = chunk(text, reason) | {"usage": usage}
        if request_stats.get("eos_reason") == "loop_guard":
            out_["choices"][0]["stop_reason"] = "loop_guard"
            out_["timings"] = {"loop_guard": request_stats.get("loop_guard")}
        return web.json_response(out_)

    async def slots(_: web.Request) -> web.Response:
        return web.json_response(engine.slot_store.slots())

    async def slot_action(request: web.Request) -> web.Response:
        store = engine.slot_store
        if request.match_info["id"] != "0":
            raise web.HTTPBadRequest(reason="only slot 0 exists")
        action = request.query.get("action")
        body = await request.json() if request.can_read_body else {}
        try:
            async with exclusive():
                if hasattr(engine, "settle_reply"):
                    await engine.settle_reply()
                if action == "erase":
                    return web.json_response(store.erase())
                if action not in ("save", "restore"):
                    raise web.HTTPBadRequest(reason="action must be save, restore or erase")
                fn = getattr(store, action)
                result = await asyncio.get_running_loop().run_in_executor(None, fn, body.get("filename", ""))
                return web.json_response(result)
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            traceback.print_exc()
            return web.json_response({"error": {"code": 400, "message": f"{type(exc).__name__}: {exc}", "type": "invalid_request_error"}},
                                     status=400)

    async def metrics_view(_: web.Request) -> web.Response:
        # used_tokens = pages referenced by active jobs: in-flight tokens, retained cache excluded.
        # get_cache_stats is memoized on the page tables; the except covers fake engines and a
        # page-table mutation in the worker thread mid-walk.
        gen = getattr(engine, "generator", None) or getattr(engine, "greedy_generator", None)
        try:
            cs = gen.get_cache_stats()
            kv = cs["used_tokens"] / cs["max_tokens"] if cs.get("max_tokens") else 0.0
        except Exception:                                        # noqa: BLE001
            kv = 0.0
        return web.Response(body=app["metrics"].render(
            processing=processing, deferred=queue.qsize(), kv_ratio=kv),
            headers={"Content-Type": "text/plain; version=0.0.4; charset=utf-8"})

    if getattr(engine, "slot_store", None) is not None:
        app.router.add_get("/slots", slots)
        app.router.add_post("/slots/{id}", slot_action)
    app.router.add_get("/health", health)
    app.router.add_get("/v1/models", models)
    app.router.add_get("/metrics", metrics_view)
    app.router.add_post("/v1/chat/completions", lifecycle.wrapped(completions))
    app.router.add_post("/apply-template", apply_template)
    app.router.add_post("/tokenize", tokenize)
    app.router.add_post("/v1/completions", lifecycle.wrapped(v1_completions))
    app.router.add_post("/completion", completion)
    app.on_startup.append(start)
    app.on_cleanup.append(close)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--no-vision", action="store_true", help="do not load the vision tower (saves memory; image requests get HTTP 400)")
    parser.add_argument("--no-uncensor", action="store_true", help="ignore a bundled uncensor_spec.json in the model directory (same as EXL3_ABLIT_RUNTIME=off)")
    parser.add_argument("--model-id", default="glm-5.3-exl3")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--exact-prefix-cache", action="store_true",
                        help="a request resumed from a cached page checkpoint gives the same greedy text as its cold prefill: "
                             "plain prefill chunks (EXL3_PREFILL_BIG_TAIL=0), about 5-12 %% slower cold prefill at 7-14K tokens")
    parser.add_argument("--num-draft", type=int, default=2, help="MTP draft tokens per round (2 = R=3 verify, round2 27/09)")
    parser.add_argument("--max-history", type=int, default=None, help="recurrent state history slots; default = --num-draft. Each extra slot costs ~2.15 GiB")
    parser.add_argument("--row-budget", type=int, default=None,
                        help="sessions > 1: cap verify rows per round (busy sessions x (draft depth + 1)) by lowering the draft depth, never below 1; "
                             "default 5 when --sessions > 1, 0 = off")
    parser.add_argument("--sessions", type=int, default=1,
                        help=f"requests decoded together through the one Generator (1..{MAX_SESSIONS}); default 1 = one request at a time. "
                             "The KV pool (-c) is shared by all sessions")
    parser.add_argument("-c", "--max-ctx", type=int, default=131072, help="context window (capped by max_position_embeddings)")
    parser.add_argument("--image-cache-mb", type=int, default=IMAGE_CACHE_MB_DEFAULT,
                        help=f"host memory for the decoded image embeddings, in MiB; 0 = no byte limit. Like -c this "
                             f"is one pool shared by every --sessions request (default {IMAGE_CACHE_MB_DEFAULT}): "
                             "about 49 MiB per 1568x1568 page, 1 MB per 224x224 crop")
    parser.add_argument("--image-cache-items", type=int, default=IMAGE_CACHE_ITEMS_DEFAULT,
                        help=f"cap on how many pictures are held at once; 0 = no limit "
                             f"(default {IMAGE_CACHE_ITEMS_DEFAULT}, as upstream)")
    parser.add_argument("--image-cache-dir", default=IMAGE_CACHE_DIR_DEFAULT,
                        help=f"directory of the second-level picture cache (default {IMAGE_CACHE_DIR_DEFAULT}); "
                             "only used when --image-cache-dir-max-gb is set")
    parser.add_argument("--image-cache-dir-lock-s", type=float, default=IMAGE_CACHE_DIR_LOCK_S,
                        help=f"seconds a conversation keeps its files on disk before another conversation may "
                             f"take the room (default {IMAGE_CACHE_DIR_LOCK_S:g} = 5 min, 0 = no tenure). Tenure "
                             "is what keeps a set of conversations taking turns from evicting each one just "
                             "before it is served again; while it holds, a new picture that would have to push a "
                             "recent conversation out is simply not cached")
    parser.add_argument("--image-cache-dir-max-gb", type=float, default=IMAGE_CACHE_DIR_MAX_GB,
                        help=f"cap on that directory, in GiB; 0 = off, which is the default "
                             f"({IMAGE_CACHE_DIR_MAX_GB:g}). No unlimited setting: a positive value is the cap")
    parser.add_argument("--image-cache-format", choices=IMAGE_CACHE_FORMATS, default="fp32",
                        help="how the cached image embeddings are stored in host memory: fp32 as the vision merger "
                             "emits them (default), fp16 (half the bytes, bit-exact on this pack) or q8 (8 bit with "
                             "one fp16 scale per row, a quarter of the bytes, 0.86 percent error)")
    parser.add_argument("--slot-save-path", default="~/cache/llama-slots",
                        help="directory for /slots save/restore files (llama.cpp compatible)")
    parser.add_argument("--chat-template", default=None,
                        help="Jinja chat template file (default: tools/lanes/assets/glm53-template-medium.jinja; "
                             "<model>/chat_template.jinja is the model's unchanged one)")
    parser.add_argument("--default-temperature", type=float, default=None,
                        help="used when a request omits temperature (default: generation_config.json)")
    parser.add_argument("--default-top-p", type=float, default=None,
                        help="used when a request omits top_p (default: generation_config.json)")
    parser.add_argument("--default-max-tokens", type=int, default=None,
                        help="reply budget when a request omits max_tokens, thinking included (default: all the free context)")
    parser.add_argument("--default-reasoning-effort", choices=("low", "medium", "high", "max"), default=None,
                        help="thinking effort when a request omits it (default: medium)")
    args = parser.parse_args()
    if not 1 <= args.sessions <= MAX_SESSIONS:
        parser.error(f"--sessions must be 1..{MAX_SESSIONS}")
    if args.sessions > 1:
        print(f"serve: --sessions {args.sessions}: up to {args.sessions} requests share the KV pool of -c {args.max_ctx} "
              f"(per-session window = -c / sessions when all are busy); greedy output is not guaranteed equal to the same "
              f"request run alone; slot save/restore waits for every running request and acts on the last prompt noted",
              file=sys.stderr, flush=True)
    if os.environ.get("EXL3_SCHED_CHUNK_IL", "0") == "1":
        # chunked-prefill interleave (exllamav3/generator/chunk_il.py): needs another request running beside the big prefill
        print("serve: chunk_il ON " + ("" if args.sessions > 1 else "(no effect: needs --sessions >= 2)"), file=sys.stderr, flush=True)
    if args.no_uncensor:
        os.environ["EXL3_ABLIT_RUNTIME"] = "off"
    SERVE_DEFAULTS.update(pack_sampling(args.model))
    SERVE_DEFAULTS.update({k: v for k, v in (("temperature", args.default_temperature), ("top_p", args.default_top_p),
                                             ("reasoning_effort", args.default_reasoning_effort),
                                             ("max_tokens", args.default_max_tokens)) if v is not None})
    if args.exact_prefix_cache:
        os.environ["EXL3_PREFILL_BIG_TAIL"] = "0"
    for key, value in SPEED_ENV.items():
        os.environ.setdefault(key, value)
    if args.sessions > 1:
        for key, value in MULTI_SESSION_ENV.items():
            os.environ.setdefault(key, value)
    os.environ.setdefault("EXL3_MOE_UNION_V2", "1")
    apply_eh_sidecar_default(args.model)
    apply_draft2s_default(args.model)
    warm = os.environ.get("EXL3_SERVE_WARMUP", "1") != "0"
    tune = warm and os.environ["EXL3_DENSE_GEMM_TUNE"] != "0" and float(os.environ.get("EXL3_SERVE_DTUNE_PRIME_S", "600")) > 0
    stages = ["target weights", "drafter weights"] + ([] if args.no_vision else ["vision tower"]) + (["warm-up"] if warm else []) + (["dense GEMM tuning"] if tune else [])
    startup_health.check_model_dir("serve", args.model, ("config.json",))
    # A start with the dense GEMM tuning already cached is much shorter than the first one: keep their timings apart.
    startup_health.start("glm", stages, args.host, args.port, routes=ROUTES,
                         variant="tuned" if tune and dense_tune_path().exists() else "untuned")
    startup_health.report_compiler()
    try:
        image_cache_bytes = max(0, int(args.image_cache_mb)) * 2 ** 20
        image_cache_size = (f"{args.image_cache_mb} MiB of host memory (--image-cache-mb)" if image_cache_bytes
                            else "no byte limit (--image-cache-mb 0)")
        engine = ResidentEngine(args.model, max_history=args.max_history or max(args.num_draft, 1), max_ctx=args.max_ctx, num_draft=args.num_draft,
                              vision=not args.no_vision, sessions=args.sessions,
                              image_cache_bytes=image_cache_bytes, image_cache_items=args.image_cache_items,
                              image_cache_format=args.image_cache_format,
                              image_cache_dir=args.image_cache_dir, image_cache_dir_max_gb=args.image_cache_dir_max_gb,
                              image_cache_dir_lock_s=args.image_cache_dir_lock_s)
        engine.row_budget = (5 if args.row_budget is None else args.row_budget) if args.sessions > 1 else 0
        template = Path(args.chat_template or DEFAULT_TEMPLATE).expanduser().read_text(encoding="utf-8")
        print(mem_line("loaded"), flush=True)
        print(f"serve: vision={engine.supports_vision}", flush=True)
        print(f"serve: image cache {image_cache_size}, {'no count cap' if not engine.emb_cache.max_items else str(engine.emb_cache.max_items) + ' pictures max'}, "
              f"{args.image_cache_format} rows, {engine.emb_cache.policy} policy, shared by {args.sessions} session(s)", flush=True)
        if engine.emb_disk is not None:
            print(f"serve: {engine.emb_disk.describe()}", flush=True)
        # Warm-up before the slot store is attached, so the warm-up prompt never shows in /slots.
        # EXL3_SERVE_WARMUP=0 skips it.
        if os.environ.get("EXL3_SERVE_WARMUP", "1") != "0":
            startup_health.stage("warm-up")
            print(f"serve: warm-up {warmup(engine, template):.1f} s", flush=True)
            print(mem_line("warm"), flush=True)
            if os.environ.get("EXL3_WARM_INDEXER", "1") != "0":
                t0 = time.perf_counter()
                try:
                    warmed = warm_indexer(engine)
                except Exception as e:   # a warm-up is an optimisation: never keep the server from starting
                    import traceback; traceback.print_exc()
                    warmed = f"FAILED ({e!r})"
                print(f"serve: indexer stride warm-up {warmed} stride(s) {time.perf_counter() - t0:.1f} s", flush=True)
            if engine.supports_vision:   # first tower pass pays kernel setup and allocator growth: not on the first user image
                from PIL import Image
                t0 = time.perf_counter()
                engine._embed([(Image.new("RGB", (448, 448), (127, 127, 127)), "warmup")])
                engine.emb_cache.clear()
                print(f"serve: vision warm-up {time.perf_counter() - t0:.1f} s", flush=True)
            if os.environ.get("EXL3_SERVE_EMPTY_CACHE", "1") == "1":   # lever (a): drop the allocator's inactive slack
                import torch
                torch.cuda.empty_cache()
                print(mem_line("emptied"), flush=True)
            if os.environ.get("EXL3_SERVE_MEMDIAG") == "1":
                mem_diag("warm")
            # EXL3_SERVE_DTUNE_PRIME_S: time budget for pre-tuning dense GEMM classes (0 = skip)
            budget = float(os.environ.get("EXL3_SERVE_DTUNE_PRIME_S", "600"))
            if budget > 0 and os.environ.get("EXL3_DENSE_GEMM_TUNE", "1") != "0":
                keys, secs = prime_dense_tune(budget)
                print(f"serve: dense GEMM prime {keys} keys {secs:.1f} s", flush=True)
                print(mem_line("primed"), flush=True)
        from exllamav3.generator.slot_store import SlotStore
        engine.slot_store = SlotStore(engine.greedy_generator, [c for c in (engine.cache, engine.draft_cache) if c is not None],
                                      args.slot_save_path, args.model_id)
        from exllamav3.util.disk_store import env_dir
        if env_dir():
            from exllamav3.generator.prefix_disk import PagedPrefixDisk, weights_identity
            engine.prefix_disk = PagedPrefixDisk(engine.slot_store, model_id=args.model_id, extra=weights_identity(args.model))
            print(f"serve: prefix disk {engine.prefix_disk.disk.dir} ({len(engine.prefix_disk.pages)} pages, "
                  f"{len(engine.prefix_disk.stashes)} checkpoints, cap {engine.prefix_disk.disk.cap >> 30} GiB)", flush=True)
        if os.environ.get("EXL3_SERVE_RSS_LOG"):  # rssleak1: per-request host-RSS accounting, off by default
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from rss_probe import install
            install(engine)
            print(f"serve: rss probe -> {os.environ['EXL3_SERVE_RSS_LOG']}", flush=True)
    except BaseException as exc:                                  # noqa: BLE001
        startup_health.abort(f"{type(exc).__name__}: {exc}")
        raise
    app = create_app(engine, args.model_id, template)
    # Which run this is, in the log itself: ctx, sessions and the image-cache budget are what a later
    # "why was this request slow" question turns on, and they are not visible in the client's request.
    reqlog = app["reqlog"]
    ping = app["ping"]
    print(f"serve: {reqlog.describe()}", flush=True)
    print(f"serve: {ping.describe()}", flush=True)
    emb_cache = getattr(engine, "emb_cache", None)
    reqlog.log("serve_start", model=Path(args.model).name, model_id=args.model_id, pid=os.getpid(),
               ctx=args.max_ctx, cache_tokens=getattr(getattr(engine, "cache", None), "max_num_tokens", None),
               sessions=args.sessions, num_draft=args.num_draft, row_budget=getattr(engine, "row_budget", 0),
               max_history=args.max_history or max(args.num_draft, 1), vision=bool(engine.supports_vision),
               conv_gemm=os.environ.get("EXL3_CONV_GEMM"), dense_tune=os.environ.get("EXL3_DENSE_GEMM_TUNE"),
               moe_fused=os.environ.get("EXL3_MOE_FUSED", "0"),
               image_cache_mb=round(emb_cache.max_bytes / 2 ** 20, 1) if emb_cache is not None else None,
               image_cache_items=getattr(emb_cache, "max_items", None),
               image_cache_format=getattr(engine, "image_cache_format", "fp32"),
               image_cache_policy=getattr(getattr(engine, "emb_cache", None), "policy", None),
               image_cache_dir=str(engine.emb_disk.path) if getattr(engine, "emb_disk", None) else None,
               image_cache_dir_cap_mb=round(engine.emb_disk.max_bytes / 2 ** 20) if getattr(engine, "emb_disk", None) else None,
               image_cache_dir_lock_s=engine.emb_disk.lock_s if getattr(engine, "emb_disk", None) else None,
               image_cache_env=getattr(engine, "env_fingerprint", None),
               heartbeat_s=reqlog.heartbeat_s,
               sse_ping_s=ping.interval_s or None)
    sock = startup_health.finish()                                # the early listener's open socket: no refused connection
    if sock is not None:
        web.run_app(app, sock=sock)
    else:
        web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
