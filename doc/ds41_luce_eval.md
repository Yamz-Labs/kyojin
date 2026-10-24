# Scoring a DS4.1 pack on the Lucebox held-out tokens

`tools/ds41/luce_eval.py` scores a pack on the public Lucebox evaluation set, so our numbers sit in the same table as
their ROCMFP2S / MIX10 / FP23S rows.

## Data (about 17 MB)

From `huggingface.co/Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP-GGUF`, download only `eval/` and `results/`:

    DIR=~/ds41_work/luce_data   # DIR/eval/*, DIR/results/*
    for f in eval/heldout_8x1024.npy eval/teacher_8x1024.npz eval/teacher_kl.json results/*_kl.json; do ...; done

File formats:

- `eval/heldout_8x1024.npy`: int32 (8, 1024), eight token sequences.
- `eval/teacher_8x1024.npz`: `idx` int32 (8184, 256) and `lp` float32 (8184, 256), the unquantized 1017 model's top-256 ids
  (best first) and their full-vocabulary natural-log probabilities (row sums 0.95-1.0, so not renormalized); `nll`
  (2.14727, mean next-token NLL) and `tokens` (8184).
- Flat row `i` is sequence `i // 1023`, position `i % 1023`; it predicts `heldout[seq, pos + 1]`. 8184 = 8 x 1023.
- `results/<file>_kl.json`: per file `kl_top256`, `top1_agree`, `nll`, `ppl`, `teacher_nll`, `tokens`, plus build metadata.

## Metric definitions

What is proven by the public files, and what is inferred. Their scoring code (`lucebox_training/quantization`) is not public.

Proven (`luce_eval.py check`, `test_luce_eval.py`):

- ppl = exp(nll), nll = mean over the 8184 positions of -ln q(next token) under the full softmax (all six result files
  satisfy `exp(nll) == ppl` and carry the teacher's 2.14727). The alignment above is right: the next token is in the teacher
  top-256 on 97.5 % of rows and is the teacher argmax on 53.7 %. The 201 rows outside top-256 explain the rest of the
  stored teacher nll: implied mean miss NLL 11.5 nats, above the 9.3 floor the top-256 tail allows.
- top-1 = candidate argmax equals teacher argmax (`idx[:, 0]`).
- The result files hold only scalars, so no candidate number can be rebuilt from them. "Reproducing" a row means: teacher
  ppl 8.5615 from the npz, the table in their card from their json, and the same scorer fed with the teacher itself giving
  KL 0 and top-1 100 % (tested).

Inferred (the only definition their stored data allows; flagged as `definition: luce-inferred-v1` in our output):

- `kl_top256` = mean over positions of sum over the teacher's top-256 ids of `p_i (ln p_i - ln q_i)`; p = teacher
  probability, q = candidate FULL-vocabulary softmax probability at the same id, no renormalization, no tail bucket.
  Direction KL(teacher || candidate). Their file stores exactly top-256 ids plus log-probs, with no tail term, which is what
  this formula needs.
- Two variants are reported next to it so a mismatch is visible: `kl_top256_renorm` (p and q both renormalized on the 256
  ids) and `kld_tail` (our gate's definition at K = 256: the top-256 sum plus the tail bucket
  `p_tail ln(p_tail / q_tail)`; a lower bound of the full-vocabulary KLD).
- Check once the first pack is scored: sanity-compare with their card (ROCMFP2S KL 0.265 at ppl 9.77). A candidate with
  ppl near 9.8 should land in the 0.2-0.3 range under the right definition; the renorm variant is smaller, the tail
  variant larger. If our number for a model they also scored (e.g. any GGUF they published) is available, match it
  to pick the definition.

## Run

The pack is scored teacher-forced through `/v1/ds41/score` (same route as the quality gate, `ds41_serving` loader).
GPU jobs for DS4.1 run on the second (iGPU) machine under its GPU lock, not on the main box.

    # server (as run_gate.sh does) then:
    python3 tools/ds41/luce_eval.py run --data $DIR --base-url http://127.0.0.1:8399 \
        --label OURS-DS41-1017 --bytes <pack bytes> --out ours_kl.json --cache luce_cache

    # or in process
    python3 tools/ds41/luce_eval.py run --data $DIR --model PACK_DIR --label OURS --out ours_kl.json

    python3 tools/ds41/luce_eval.py check --data $DIR                    # teacher-file consistency, no model
    python3 tools/ds41/luce_eval.py table --data $DIR --ours ours_kl.json

Per position the candidate ids are the teacher's 256 ids plus the next token (for ppl); one decode step per token,
8184 steps in total (about 6 minutes at 25 tok/s). `--cache` keeps each sequence, so a killed run resumes.
`--rows` / `--max-len` give a quick partial run; its json has `comparable: false` and the table marks it.

## The table

    | model | KL top-256 vs teacher | top-1 | ppl | size |
    |---|---|---|---|---|
    | teacher (1017, unquantized) | 0.000 | 100.0% | 8.56 | |
    | ROCMFP2S | 0.265 | 78.6% | 9.77 | 383 GB |
    | ROCMFP2S-MIX10 | 0.244 | 78.8% | 9.59 | 388 GB |
    | ROCMFP23S | 0.195 | 81.5% | 9.42 | 405 GB |
    | OURS-DS41-1017 | (measured) | | | |

The `-IM1017` rows are also read. Our json adds `kl_top256_renorm` and `kld_tail`; our gate (`tools/ds41qual`) stays the
internal check against the original model on agentic text. This one is the public, comparable number.

## Tests

CPU only, 5 s: metric closed forms, the public teacher file when `LUCE_DATA` (default `~/ds41_work/luce_data`, layout
`eval/` + `results/`) holds it, and an end-to-end run of two tiny random models against a synthetic teacher written in their
format (same model as teacher gives KL 0 / top-1 100 %; a second model matches an independent numpy recomputation).

    cd <engine tree with a built exllamav3 extension>
    PYTHONPATH=.:<this repo>/tools/ds41 python -m pytest --noconftest <this repo>/tools/ds41/test_luce_eval.py

The test appends this repo to `sys.path`, so a `PYTHONPATH` engine (with its compiled extension) takes precedence; running
from this tree needs the extension built here. The pytest conftest of the repo opens the GPU, hence `--noconftest`.
