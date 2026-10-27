"""The DS4.1 port behind the server: `load_model(path, ctx, **options)` for `serve.py --loader ds41_model:load_model`
(the default loader name), with DSpark speculative decoding behind `--num-draft`.

The model is `exllamav3.dsv41.model.DSV41Model` (plain torch, CPU or HIP) built from the official safetensors folder
(or a V4.1 GGUF); the draft/verify loop is `dspark.DSparkDecoder` over `exllamav3.dsv41.spec.PortSpec`:

  * `num_draft` 0 (default): one decode step per token, nothing drafted;
  * `num_draft` N (1..10; above the block size the drafter chains passes): the DSpark stages draft up to N tokens, one block forward verifies them, the
    rejected tail is rolled back from an undo log. Greedy output equals the plain loop (up to the numerical noise
    of a [k+1, d] GEMM vs k+1 GEMVs, which only matters at an exact near-tie); sampled output follows the plain
    loop's law (the sampler draws the target token of each position in order, a draft counts when it equals the draw).

Loader options (`--engine-opt KEY=VALUE`): num_draft (set by --num-draft), device ("cpu", "cuda:0"), dtype
("float32" default, "bfloat16", "float16"), engram_dir (folder with the MXFP4 `layers.N.engram.embed.{weight,scale}.bin`
rows when the shards do not hold them), cache_experts (true: keep built experts resident), threads (CPU torch threads).

A converted pack (the EXL3 layout of tools/ds41/stream_convert.py: `layers.N.safetensors`, `engram/`, `quantization_config`)
is detected by its files and loaded with `exllamav3.dsv41.pack.load` into the serving model; device and dtype then
default to the GPU and fp16 when a GPU is visible. Pack-only options: layers (first N trunk layers, smoke tests),
moe ("auto" | "loop"), prefill_chunk, prefix_cache (default true), spec_hook (below).

Speculative hook: with `num_draft` > 0 the engine asks `spec_hook(model, max_seq)` for an object with `.target` and
`.drafter` (the `dspark.Target` / `dspark.Drafter` protocols). `spec_hook` is a `module:function` string (default
`exllamav3.dsv41.spec:PortSpec`); a function that returns None means "no drafter": the engine then runs plain decode.
With `num_draft` 0 the hook is never called.

One request at a time (`sessions = 1`). Prompt cache (plain and drafted decode alike): `exllamav3.dsv41.prefix_cache` keeps
host-side snapshots of the state at chosen positions (system block, the generation prompt, the prompt end, long-prompt chunk
ends, the end of the reply). A request resumes from the longest snapshot whose tokens are a prefix of its prompt and prefills
only the rest (`cached_tokens` reports the reused part). Knobs: DS41_PFX_GIB (byte budget, default 4, 0 = off), DS41_PFX_MAX
(entries, 48), DS41_PFX_MIN_FREE_GIB (store nothing below this MemAvailable, 3), DS41_PFX_LOG=1 (one line per request).
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import sys
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for _p in (HERE.parent.parent,):                       # the repo root, when run as a script from elsewhere
    if str(_p) not in sys.path:
        sys.path.append(str(_p))

import dspark  # noqa: E402
from ds41_engine import HFTokenizer  # noqa: E402

EOS_TEXT = "<｜end▁of▁sentence｜>"
USER_TEXT = "<｜User｜>"
ASSISTANT_TEXT = "<｜Assistant｜>"


def _special_id(tokenizer, text: str) -> int | None:
    """Id of a special token, None when the tokenizer has none (test tokenizers): the prefix marks then fall back to the prompt end."""
    tk = getattr(tokenizer, "_tk", tokenizer)
    try:
        v = tk.token_to_id(text)
        return int(v) if v is not None else None
    except Exception:                                       # noqa: BLE001
        return None


def prefix_cache_store():
    from exllamav3.dsv41.prefix_cache import PrefixStore
    return PrefixStore.from_env()


# ---- sampling --------------------------------------------------------------------------------------------------


class TopPSampler:
    """temperature + nucleus sampling, same draw rule as the official `sample` (Gumbel-max over the kept
    probabilities: one exponential draw per token). temperature <= 0 is argmax. Satisfies dspark.Sampler."""

    def __init__(self, temperature: float, top_p: float = 1.0, generator: Any = None):
        self.temperature, self.top_p, self.generator = temperature, top_p, generator
        self.stochastic = temperature > 0

    def probs(self, logits_row: Any) -> Any:
        import torch
        p = torch.softmax(logits_row.reshape(-1).float() / max(self.temperature, 1e-5), dim=-1)
        if self.top_p < 1.0:
            sp, si = p.sort(descending=True)
            cum = sp.cumsum(0)
            drop = (cum - sp) >= self.top_p              # keep the smallest prefix whose mass reaches top_p
            sp = sp.masked_fill(drop, 0.0)
            p = torch.zeros_like(p).scatter_(0, si, sp)
            p = p / p.sum()
        return p

    def sample(self, logits_row: Any) -> int:
        import torch
        if not self.stochastic:
            return int(logits_row.argmax())
        p = self.probs(logits_row)
        noise = torch.empty_like(p).exponential_(1, generator=self.generator)
        return int(p.div_(noise).argmax())

    def uniform(self) -> float:
        import torch
        return float(torch.rand(1, generator=self.generator))

    def draw(self, probs: Any) -> int:
        import torch
        return int(torch.multinomial(probs, 1, generator=self.generator))


class _Detok:
    """Incremental detokenizer: text of the new tokens, nothing while the tail is an incomplete UTF-8 sequence."""

    def __init__(self, tk):
        self.tk, self.ids, self.prefix, self.read = tk, [], 0, 0

    def push(self, new_ids) -> str:
        self.ids.extend(new_ids)
        old = self.tk.decode(self.ids[self.prefix:self.read], skip_special_tokens=False)
        new = self.tk.decode(self.ids[self.prefix:], skip_special_tokens=False)
        if len(new) > len(old) and not new.endswith("\ufffd"):
            out = new[len(old):]
            self.prefix, self.read = self.read, len(self.ids)
            return out
        return ""

    def flush(self) -> str:
        old = self.tk.decode(self.ids[self.prefix:self.read], skip_special_tokens=False)
        new = self.tk.decode(self.ids[self.prefix:], skip_special_tokens=False)
        self.prefix = self.read = len(self.ids)
        return new[len(old):]


# ---- engine ----------------------------------------------------------------------------------------------------


MAX_DEPTH = 10                                          # draft depth cap (chained passes above the block size)


class PortEngine:
    """`ds41_engine.Engine` over the port model."""

    sessions = 1

    def __init__(self, model, tokenizer, *, ctx: int, num_draft: int = 0, eos_id: int | None = None, seed: int | None = None,
                 draft_auto: bool = False, draft_conf_threshold: float | None = None,
                 spec_hook: Any = None, prefix_cache: bool = True, vision_dir: str | None = None,
                 vision_cfg: Any = None, vision_device: str | None = None, vision_dtype: str = "bfloat16",
                 vision_cache: int = 8, ngram: str | None = None):
        import torch
        self.ngram = os.environ.get("DS41_NGRAM", "") if ngram is None else ngram      # prompt lookup merge: '' | strong | extend
        self.model, self.tk = model, tokenizer
        # image input: the tower is read from `vision_dir` on the first image (about 1 GiB), nothing is held before
        self.vision_dir, self.vision_cfg = vision_dir, vision_cfg if vision_dir else None
        self.vision_device = vision_device or str(model.device)
        self.vision_dtype = getattr(torch, vision_dtype)
        self._tower = None
        self._rows_cache: dict[str, Any] = {}              # image key -> span rows (model thread only)
        self._rows_max = max(1, int(vision_cache))
        cfg = model.cfg
        self.ctx = int(ctx)
        self.vocab_size = int(cfg.vocab_size)
        self.eos_id = eos_id
        self.num_draft = int(num_draft)
        self.draft_auto = bool(draft_auto)
        self.draft_conf = float(draft_conf_threshold) if draft_conf_threshold else None
        self.spec = None
        if self.num_draft:
            if model.mtp is None:
                raise ValueError("--num-draft needs the DSpark stages (mtp.*) in the model; this one has none")
            if self.num_draft > MAX_DEPTH:
                raise ValueError(f"--num-draft {self.num_draft} exceeds the maximum draft depth {MAX_DEPTH}")
            if self.num_draft + 1 > cfg.sliding_window:
                raise ValueError(f"--num-draft {self.num_draft}: a verify block must fit the sliding window ({cfg.sliding_window})")
            # verify writes up to num_draft positions past the last kept one; depth above the block size chains passes
            spec_len = self.ctx + self.num_draft + cfg.dspark_block_size + 2
            if spec_hook is None:
                from exllamav3.dsv41.spec import PortSpec
                self.spec = PortSpec(model, spec_len, max_depth=self.num_draft)
            else:
                self.spec = resolve_spec_hook(spec_hook)(model, spec_len)
            if self.spec is None:
                self.num_draft = 0                       # the hook has no drafter: plain decode
        self.prefix_cache = bool(prefix_cache)
        self.pfx = prefix_cache_store() if self.prefix_cache else None
        if self.pfx is not None and self.pfx.persist_dir:
            from exllamav3.dsv41.prefix_cache import fingerprint
            self.pfx.fingerprint = fingerprint(cfg, getattr(model, "pack_id", ""))
            self.pfx.load_dir()                          # system/tools snapshots of earlier runs: the first call resumes from them
        self._user_id, self._assistant_id = _special_id(tokenizer, USER_TEXT), _special_id(tokenizer, ASSISTANT_TEXT)
        if self.spec is not None and hasattr(self.spec, "attach_prefix"):
            self.spec.attach_prefix(self.pfx, self._user_id, self._assistant_id)
        self.max_seq = self.ctx + (self.num_draft + cfg.dspark_block_size + 2 if self.spec is not None else 1)
        self.last_stats: dict[str, Any] = {}
        self._gen = torch.Generator().manual_seed(seed) if seed is not None else None
        self._pool = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="ds41-model")

    # -- tokenization
    def count_tokens(self, text: str) -> int:
        return len(self.tk.encode(text))

    @property
    def supports_vision(self) -> bool:
        return self.vision_cfg is not None

    def prepare_images(self, records: list[dict]) -> list[Any]:
        """Image records -> PreparedImage list (decode, resize, patches, pixel hash). CPU only; safe to call from any
        thread. Raises vision.ImageError for an unreadable image."""
        from exllamav3.dsv41 import vision
        if not self.supports_vision:
            raise vision.ImageError("this model has no vision tower")
        return [vision.prepare_image(r, self.vision_cfg) for r in records]

    def prompt_ids(self, prompt: str, images: list[Any] | None = None) -> list[int]:
        """Token ids of a rendered prompt; each image placeholder is expanded to its span of image-token ids."""
        ids = self.tk.encode(prompt)
        if not images:
            return ids
        from exllamav3.dsv41 import vision
        return vision.expand_image_tokens(ids, images, self.vision_cfg.image_token_id)[0]

    def _spans(self, located: list[tuple[int, Any]]) -> list[Any]:
        """[(start, PreparedImage)] -> vision.ImageSpan list (runs the tower for images not seen recently)."""
        from exllamav3.dsv41 import vision
        if not located:
            return []
        if self._tower is None:
            self._tower = vision.VisionTower.load(self.vision_dir, self.vision_cfg, self.vision_device, self.vision_dtype)
        out = []
        for start, img in located:
            rows = self._rows_cache.pop(img.key, None)
            if rows is None:
                rows = self._tower.span_rows(img)
            self._rows_cache[img.key] = rows                # most recent last
            while len(self._rows_cache) > self._rows_max:
                self._rows_cache.pop(next(iter(self._rows_cache)))
            out.append(vision.ImageSpan(start, rows, img.key))
        return out

    def encode_ids(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self.tk.encode(text, add_bos=add_special_tokens)

    async def reset_prefix_cache(self) -> None:
        if self.pfx is not None:
            self.pfx.clear()

    def warmup(self, prompt_tokens: int = 4608, decode_tokens: int = 16) -> float:
        """One prefill of about `prompt_tokens` and a short decode before serving, so kernel JIT, allocator growth and
        graph capture do not land on the first request. Leaves no prefix cache behind. Returns seconds."""
        n = max(8, min(int(prompt_tokens), self.ctx // 2))
        lo = min(1000, self.vocab_size - 1)
        ids = [(lo + i * 7919) % self.vocab_size or 1 for i in range(n)]
        t0 = time.perf_counter()
        kept = list(self.pfx.entries) if self.pfx is not None else []

        async def go():
            async for _ in self.generate("", max_tokens=int(decode_tokens), temperature=0.0, top_p=1.0, stop=[],
                                         prompt_ids=ids, ignore_eos=True):
                pass
        asyncio.run(go())
        if self.pfx is not None:
            self.pfx.entries = [e for e in self.pfx.entries if e in kept]          # keep what load_dir restored
        return time.perf_counter() - t0

    # -- one request, blocking (runs in the model thread)
    # DS41_DRAFT_SWITCH=1: request-level draft choice. A prompt that carries the tool block keeps auto cap N (best on tool turns);
    # any other prompt uses the timed tuner + want-cap (best on code, chat, follow-up). Measured: briefs/ds41_draftdecide.md.
    TOOL_MARK = "You have access to a set of tools"

    def _switch_plain_ctx(self) -> bool:
        return os.environ.get("DS41_DRAFT_SWITCH", "0") == "1" and not getattr(self, "_req_tools", True)

    def _timed(self) -> bool:
        return os.environ.get("DS41_TUNER", "") == "timed" or self._switch_plain_ctx()

    def _want(self) -> bool:
        return os.environ.get("DS41_DRAFT_WANT", "0") == "1" or self._switch_plain_ctx()

    def _make_tuner(self):
        if not self.draft_auto:
            return None
        if self._timed():
            return dspark.TimedTuner(int(os.environ.get("DS41_TUNER_KMIN", "0")), self.num_draft, cost_ratio=float(os.environ.get("DS41_TUNER_COST", "0.3")))
        return dspark.DraftTuner(1, self.num_draft)

    def _run(self, ids: list[int], max_tokens: int, sampler: TopPSampler, eos_id: int, cancel, emit,
             images: list[Any] | None = None, cache_prompt: bool = True) -> dict:
        import torch
        from exllamav3.dsv41 import vision
        t0 = time.perf_counter()
        t_first = None
        n_new = 0
        eos = False
        stats: dict[str, Any] = {}
        located = vision.locate_spans(ids, images, self.vision_cfg.image_token_id) if images else []
        keys = vision.prompt_key(ids, located) if located else ids      # image positions carry the pixel hash
        store = self.pfx if cache_prompt else None
        produced: list[int] = []                                   # every token the decode loop handed out (EOS included)
        if self.spec is None:
            from exllamav3.dsv41 import prefix_cache
            model = self.model
            pf = prefix_cache.prefill(
                model, store, ids, keys, max_seq=self.max_seq, user_id=self._user_id, assistant_id=self._assistant_id,
                spans_fn=lambda cached: self._spans([(s, im) for s, im in located if s + im.n_tokens > cached]))
            st, logits, cached = pf.st, pf.logits, pf.cached
            pos = len(ids)
            stats["cached_tokens"] = cached
            stats["prefix_lcp"] = pf.lcp

            def rounds():
                nonlocal logits, pos
                tok = sampler.sample(logits[-1])
                produced.append(tok)
                yield [tok]
                while tok != eos_id:
                    logits, _ = model.forward(torch.tensor([[tok]]), pos, st)
                    pos += 1
                    tok = sampler.sample(logits[-1])
                    produced.append(tok)
                    yield [tok]
            stream = rounds()
        else:
            dec = dspark.DSparkDecoder(self.spec.target, self.spec.drafter, eos_id=eos_id, sampler=sampler,
                                       num_draft=self.num_draft, conf_threshold=self.draft_conf,
                                       tuner=self._make_tuner(),
                                       want_cap=self._want(), ngram=self.ngram, ngram_n=int(os.environ.get("DS41_NGRAM_N", "3")),
                                       ngram_min_match=int(os.environ.get("DS41_NGRAM_MIN", "6")))
            if located and not getattr(self.spec, "supports_spans", False):
                raise ValueError("this speculative hook cannot take image input")
            if hasattr(self.spec, "supports_spans"):
                self.spec.spans = self._spans(located) if located else None      # never a stale set from an earlier request
            if hasattr(self.spec, "attach_prefix"):
                self.spec.begin_request(keys, store)
            stream = dec.stream(ids, max_tokens)
        for chunk in stream:
            if t_first is None:
                t_first = time.perf_counter()
            if self.spec is not None:
                produced.extend(chunk)
            if cancel is not None and cancel.is_set():
                break
            take = chunk[: max_tokens - n_new]
            n_new += len(take)
            if eos_id in take:
                take = take[: take.index(eos_id)]
                eos = True
            if take:
                emit(take)
            if eos or n_new >= max_tokens:
                break
        stream.close()
        self._store_reply(keys, produced, st if self.spec is None else None, store)
        if self.spec is not None:
            stats["cached_tokens"] = self.spec.cached
            stats["prefix_lcp"] = self.spec.miss_lcp
        if os.environ.get("DS41_PFX_LOG") == "1":
            print(f"[prefix] prompt={len(ids)} cached={stats.get('cached_tokens', 0)} lcp_miss={stats.get('prefix_lcp', 0)} "
                  f"store={len(store) if store is not None else 0}x{(store.nbytes() >> 20) if store is not None else 0}MiB", file=sys.stderr, flush=True)
        t1 = time.perf_counter()
        t_first = t_first or t1
        stats.update({"cached_tokens": stats.get("cached_tokens", 0), "new_tokens": n_new, "prompt_tokens": len(ids),
                      "time_prefill": t_first - t0, "time_generate": t1 - t_first, "eos": eos,
                      "eos_reason": "eos" if eos else "max_new_tokens"})
        if self.spec is not None:
            stats.update(dec.stats.as_engine_stats())
        stats["images"] = len(located)
        return stats

    def _store_reply(self, keys, produced: list[int], st, store) -> None:
        """Snapshot the state after the reply: the positions the decode loop fed (every produced token but the last) follow the
        prompt, and an agent loop resends the reply verbatim, so the next prompt extends exactly this."""
        if store is None or not produced:
            return
        if self.spec is not None:
            self.spec.store_reply(produced)
            return
        from exllamav3.dsv41 import prefix_cache
        fed = produced[:-1]
        if fed:
            store.put(prefix_cache.snapshot(self.model.cfg, st, len(keys) + len(fed), list(keys) + fed, None, "reply"))

    # -- teacher-forced scoring (quality gate): log-probs of given continuation positions
    def _score(self, ids: list[int], cont: list[int], cands: list[list[int]]) -> list[dict]:
        import torch
        st = self.model.new_state(1, self.max_seq)
        logits, _ = self.model.forward(torch.tensor([ids]), 0, st)
        pos = len(ids)
        out = []
        for i, tok in enumerate(cont):
            lp = torch.log_softmax(logits[-1].float(), dim=-1)
            top = int(lp.argmax())
            out.append({"top1": top, "top1_logp": float(lp[top]), "cand_logps": [float(lp[j]) for j in cands[i]]})
            if i + 1 < len(cont):
                logits, _ = self.model.forward(torch.tensor([[tok]]), pos, st)
                pos += 1
        return out

    async def score(self, prompt: str, cont_ids: list[int], cand_ids: list[list[int]], prompt_ids: list[int] | None = None) -> list[dict]:
        """For position i: the pack's argmax id and log-prob, and its log-prob of each id in cand_ids[i], given the
        prompt and cont_ids[:i]. Prefill once, then one decode step per continuation token (same path as generate)."""
        ids = list(prompt_ids) if prompt_ids is not None else self.prompt_ids(prompt)
        if not ids or not cont_ids or len(cand_ids) != len(cont_ids):
            raise ValueError("score needs a prompt, a continuation and one candidate list per continuation token")
        if len(ids) + len(cont_ids) + 1 > self.ctx:
            raise ValueError(f"{len(ids)}+{len(cont_ids)} tokens do not fit the context of {self.ctx}")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._pool, self._score, ids, list(cont_ids), [list(c) for c in cand_ids])

    async def generate(self, prompt: str, *, max_tokens: int, temperature: float, top_p: float, stop: list[str],
                       cancel: asyncio.Event | None = None, stats: dict | None = None,
                       prompt_ids: list[int] | None = None, ignore_eos: bool = False, cache_prompt: bool = True,
                       images: list[Any] | None = None, **_: Any) -> AsyncIterator[str]:
        if images and not self.supports_vision:
            raise ValueError("this model has no vision tower")
        ids = list(prompt_ids) if prompt_ids is not None else self.prompt_ids(prompt, images)
        req_tools = (self.TOOL_MARK in prompt) if prompt else True      # unknown = keep the tool-safe default
        if not ids:
            raise ValueError("empty prompt")
        if len(ids) + 1 > self.ctx:
            raise ValueError(f"prompt of {len(ids)} tokens does not fit the context of {self.ctx}")
        max_tokens = max(1, min(int(max_tokens), self.ctx - len(ids) - 1 - self.num_draft))
        sampler = TopPSampler(float(temperature), float(top_p if top_p is not None else 1.0), self._gen)
        eos_id = -1 if (ignore_eos or self.eos_id is None) else self.eos_id
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        self.last_stats = {}

        def work():
            try:
                self._req_tools = req_tools                 # set on the model thread: requests run one at a time
                st = self._run(ids, max_tokens, sampler, eos_id, cancel,
                               lambda toks: loop.call_soon_threadsafe(q.put_nowait, toks), images, cache_prompt)
                loop.call_soon_threadsafe(q.put_nowait, st)
            except BaseException as exc:                  # noqa: BLE001  (handed to the awaiting side)
                loop.call_soon_threadsafe(q.put_nowait, exc)

        fut = loop.run_in_executor(self._pool, work)
        detok = _Detok(getattr(self.tk, "_tk", self.tk))
        done: dict[str, Any] | None = None
        try:
            while True:
                item = await q.get()
                if isinstance(item, BaseException):
                    raise item
                if isinstance(item, dict):
                    done = item
                    break
                text = detok.push(item)
                if text:
                    yield text
            tail = detok.flush()
            if tail:
                yield tail
        finally:
            if done is None and cancel is not None:
                cancel.set()                                # the consumer went away: stop the worker
            await asyncio.shield(fut)
        self.last_stats = done
        if stats is not None:
            stats.update(done)


# ---- loader ----------------------------------------------------------------------------------------------------


def _hf_eos(path: Path, tk) -> int | None:
    try:
        v = json.loads((path / "config.json").read_text()).get("eos_token_id")
        if isinstance(v, int):
            return v
    except Exception:                                       # noqa: BLE001
        pass
    return tk.token_to_id(EOS_TEXT)


class _TokenizerShim:
    """What `engram.build_compressed_token_map` reads: `.backend_tokenizer` and `len()`."""

    def __init__(self, tk):
        self.backend_tokenizer = tk
        self._n = tk.get_vocab_size(with_added_tokens=True)

    def __len__(self):
        return self._n


def is_pack(path: str) -> bool:
    """A converted pack (stream_convert layout) and not an official checkpoint folder."""
    p = Path(path).expanduser()
    if not p.is_dir() or list(p.glob("model-*.safetensors")):
        return False
    if (p / "layers.0.safetensors").exists():
        return True
    try:
        return (json.loads((p / "config.json").read_text()).get("quantization_config") or {}).get("quant_method") == "exl3"
    except Exception:                                       # noqa: BLE001
        return False


def resolve_spec_hook(spec_hook: Any):
    """`module:function` string, callable or None (default: the port's PortSpec) -> `f(model, max_seq)`."""
    if spec_hook is None:
        from exllamav3.dsv41.spec import PortSpec
        return PortSpec
    if callable(spec_hook):
        return spec_hook
    import importlib
    mod, _, fn = str(spec_hook).partition(":")
    return getattr(importlib.import_module(mod), fn or "make_spec")


def load_pack(path: str, *, device: str, dtype: str, num_draft: int = 0, layers: int | None = None, moe: str = "auto",
              prefill_chunk: int = 2048, engram_dir: str | None = None):
    """The serving model of a pack. The DSpark stages are loaded only when drafting is asked for."""
    import torch
    from exllamav3.dsv41.pack import load
    if engram_dir:
        os.environ["EXL3_DSV41_ENGRAM_DIR"] = str(Path(engram_dir).expanduser())
    return load(str(Path(path).expanduser()), device=device, dtype=getattr(torch, dtype), layers=layers,
                mtp=bool(num_draft), moe=moe, prefill_chunk=int(prefill_chunk),
                log=lambda m: print(f"serve: {m}", flush=True))


def build_model(path: str, *, device: str = "cpu", dtype: str = "float32", engram_dir: str | None = None,
                cache_experts: bool = True):
    """DSV41Model from a model folder (official safetensors shards, or one V4.1 GGUF file / a folder holding it)."""
    import torch
    from exllamav3.dsv41.config import DSV41Config
    from exllamav3.dsv41.engram import build_compressed_token_map, RawBinEngramStore
    from exllamav3.dsv41.experts import SourceExpertProvider
    from exllamav3.dsv41.model import DSV41Model
    from exllamav3.dsv41.safetensors_io import SafetensorsWeightSource
    p = Path(path).expanduser()
    dt = getattr(torch, dtype)
    gguf = [p] if p.is_file() and p.suffix == ".gguf" else sorted(p.glob("*.gguf")) if p.is_dir() else []
    if gguf and not list(p.glob("*.safetensors")):
        from exllamav3.dsv41.gguf import load_gguf_model
        model, _ = load_gguf_model(str(gguf[0]), dtype=dt, device=device, cache_experts=cache_experts)
        return model
    cfg = DSV41Config.from_file(str(p / "config.json"))
    src = SafetensorsWeightSource(str(p), cfg.num_hidden_layers, cfg.engram_layer_ids)
    counts = {l: cfg.moe_counts(l)[0] for l in range(cfg.num_hidden_layers + cfg.n_mtp_layers)}
    prov = SourceExpertProvider(src, counts, cache=cache_experts, dtype=dt, device=device)
    store, tmap = None, None
    if cfg.engram_layer_ids:
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(str(p / "tokenizer.json"))
        tmap, nv = build_compressed_token_map(_TokenizerShim(tk))
        if not cfg.engram_compressed_vocab_size:
            cfg.engram_compressed_vocab_size = nv
        store = (RawBinEngramStore.open(engram_dir, cfg.engram_layer_ids, cfg.engram_head_dim, "auto")
                 if engram_dir else src.engram_store())
    return DSV41Model(cfg, src, prov, store, token_map=tmap, dtype=dt, device=device)


def vision_kwargs(tkdir: Path, vision: Any = "auto", vision_device: str | None = None, vision_dtype: str = "bfloat16",
                  vision_cache: int = 8) -> dict[str, Any]:
    """PortEngine arguments for image input. `vision`: "auto" (on when the folder holds a complete tower), true (fail at
    start when it does not), false. Shared by every loader that builds a PortEngine from a pack."""
    off = str(vision).lower() in ("0", "false", "no", "off", "")
    vcfg = None
    if not off:
        from exllamav3.dsv41 import vision as _vis
        vcfg = _vis.vision_available(tkdir)
        if vcfg is None and str(vision).lower() in ("1", "true", "yes", "on"):
            raise RuntimeError(f"vision was asked for but {tkdir} has no complete vision tower (config + 266 tensors)")
        if vcfg is not None:
            print(f"serve: vision tower found ({vcfg.n_layers} layers, up to {vcfg.max_n_token} tokens per image)", flush=True)
    return {"vision_dir": str(tkdir) if vcfg is not None else None, "vision_cfg": vcfg, "vision_device": vision_device,
            "vision_dtype": vision_dtype, "vision_cache": int(vision_cache)}


def pack_identity(root, engram_dir) -> str:
    """Weights identity for the on-disk prompt cache: the pack folder (path, size and mtime of its config/index files) + Engram dir."""
    import hashlib
    parts = [str(Path(root).resolve()), str(engram_dir or "")]
    for n in ("config.json", "model.safetensors.index.json", "quantization_config.json", "tokenizer.json"):
        try:
            st = (Path(root) / n).stat()
            parts.append(f"{n}:{st.st_size}:{int(st.st_mtime)}")
        except OSError:
            pass
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]


def load_model(path: str, ctx: int, *, num_draft: int = 0, device: str | None = None, dtype: str | None = None,
               engram_dir: str | None = None, cache_experts: bool = True, threads: int | None = None,
               seed: int | None = None, layers: int | None = None, moe: str = "auto", prefill_chunk: int = 2048,
               prefix_cache: bool = True, spec_hook: Any = None, draft_auto: Any = False,
               draft_conf_threshold: float | None = None, vision: Any = "auto", vision_device: str | None = None,
               vision_dtype: str = "bfloat16", vision_cache: int = 8, **_: Any) -> PortEngine:
    """`--loader ds41_model:load_model`. Unknown options are ignored (the server passes only what the command line gave)."""
    import torch
    if threads:
        torch.set_num_threads(int(threads))
    root = Path(path).expanduser()
    pack = is_pack(str(root))
    gpu = torch.cuda.is_available()                           # also true on ROCm
    device = device or ("cuda:0" if pack and gpu else "cpu")
    dtype = dtype or ("float16" if pack and device.startswith("cuda") else "float32")
    if pack:
        model = load_pack(str(root), device=device, dtype=dtype, num_draft=int(num_draft or 0), layers=layers, moe=moe,
                          prefill_chunk=prefill_chunk, engram_dir=engram_dir)
    else:
        model = build_model(str(root), device=device, dtype=dtype, engram_dir=engram_dir, cache_experts=bool(cache_experts))
    tkdir = root if root.is_dir() else root.parent
    tk = HFTokenizer(tkdir / "tokenizer.json")
    model.pack_id = pack_identity(root, engram_dir)             # the persistent prompt cache is keyed on the weights it was computed with
    vkw = vision_kwargs(tkdir, vision, vision_device, vision_dtype, vision_cache)
    return PortEngine(model, tk, ctx=int(ctx), num_draft=int(num_draft or 0), eos_id=_hf_eos(tkdir, tk._tk), seed=seed,
                      draft_auto=str(draft_auto).lower() not in ("0", "false", "no", "off", ""),
                      draft_conf_threshold=draft_conf_threshold,
                      spec_hook=spec_hook, prefix_cache=bool(prefix_cache),
                      **vkw)
