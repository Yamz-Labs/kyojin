"""The engine contract of the DS4.1 server, a stub that implements it, and the loader hook.

`serve.py` knows nothing about weights. It asks `load_model(path, ctx, ...)` for an engine object and calls only the
members listed in `Engine`; the model class that runs DeepSeek-V4.1-Flash-1017 on the GPU provides such an object
(a module exposing `load_model`, selected with `--loader module:function`). `StubEngine` is the same contract over
canned replies, for tests and for trying the HTTP layer without a GPU (`--loader stub`).
"""
from __future__ import annotations

import asyncio
import importlib
import json
import re
from collections.abc import AsyncIterator, Callable, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

DEFAULT_LOADER = "ds41_model:load_model"

SPECIAL_TOKENS = (
    "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>", "<｜User｜>", "<｜Assistant｜>", "<｜System｜>",
    "<think>", "</think>", "｜DSML｜", "<｜latest_reminder｜>", "<｜deepseek_image｜>")


@runtime_checkable
class Engine(Protocol):
    """What serve.py calls. Members marked optional may be missing; the server then skips the feature.

    `generate` yields the *decoded text* of the reply, special tokens included: the server parses `</think>` and the
    `｜DSML｜` tool markup out of that text, so a decoder that drops special tokens breaks reasoning and tool calls.
    The end-of-sentence token should stop the job and not appear in the text (the parser also accepts it).
    """

    ctx: int                                   # context window in tokens (None / 0: unknown, no room check)
    sessions: int                              # requests that may run at once (1 = one at a time)
    vocab_size: int                            # token ids of a /v1/completions token prompt must be below this

    def count_tokens(self, text: str) -> int: ...           # tokens of a reply text (special tokens count as one)
    def prompt_ids(self, prompt: str) -> Sequence[int]: ...  # the ids of a rendered prompt: special tokens parsed, no extra BOS
    def encode_ids(self, text: str, add_special_tokens: bool = False) -> list[int]: ...   # POST /tokenize

    def generate(self, prompt: str, *, max_tokens: int, temperature: float, top_p: float, stop: list[str],
                 cancel: asyncio.Event | None = None, stats: dict | None = None, prompt_ids: list[int] | None = None,
                 ignore_eos: bool = False, cache_prompt: bool = True, **_: Any) -> AsyncIterator[str]:
        """Async generator of decoded text pieces.

        `stats` (when sessions > 1) or `self.last_stats` (sessions == 1) receives, at the end of the job:
        cached_tokens, new_tokens, prompt_tokens, time_prefill, time_generate (seconds) and optionally
        accepted_draft_tokens / rejected_draft_tokens, eos (bool). `prompt_ids` replaces the prompt text.
        A set `cancel` event ends the job at the next step."""
        ...

    # optional, image input: `supports_vision` (bool), `prepare_images(records) -> list` (decode + resize, CPU, raises
    # ValueError on a bad image), `prompt_ids(prompt, images)` (placeholders expanded to image spans) and a `generate`
    # that takes `images=` together with the expanded `prompt_ids`. Without `supports_vision` the server refuses images.
    # optional: `async reset_prefix_cache()` (cache_salt namespace change), `async settle_reply()`,
    # `greedy_generator` (an object with get_cache_stats() and num_draft_tokens for /metrics), `slot_store`.


class EngineUnavailable(RuntimeError):
    """The model class named by --loader cannot be imported."""


def load_model(path: str, ctx: int, *, loader: str | None = None, **options: Any):
    """An engine for the model folder `path` with a context of `ctx` tokens.

    `loader` is `module:function` (default DEFAULT_LOADER); the function is called as `function(path, ctx, **options)`.
    `stub` selects StubEngine.
    """
    if loader == "stub":
        return StubEngine.from_model_dir(path, ctx=ctx, **options)
    spec = loader or DEFAULT_LOADER
    module_name, _, function = spec.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise EngineUnavailable(
            f"cannot import the model loader {module_name!r} ({exc}); pass --loader module:function "
            "for the DS4.1 model class, or --loader stub to run the HTTP layer on canned replies") from exc
    return getattr(module, function or "load_model")(path, ctx, **options)


# ---- tokenizers for the stub -----------------------------------------------------------------------------


class ByteTokenizer:
    """No-file tokenizer: every special token is one id (256 + its index), anything else is one id per UTF-8 byte."""
    bos_id = 256

    def __init__(self) -> None:
        self._rx = re.compile("(" + "|".join(re.escape(t) for t in SPECIAL_TOKENS) + ")")

    def encode(self, text: str, add_bos: bool = False) -> list[int]:
        ids = [self.bos_id] if add_bos else []
        for piece in self._rx.split(text):
            if piece in SPECIAL_TOKENS:
                ids.append(256 + SPECIAL_TOKENS.index(piece))
            else:
                ids.extend(piece.encode("utf-8"))
        return ids


class HFTokenizer:
    """The model's own tokenizer.json through the `tokenizers` library (the same ids the engine feeds the model)."""

    def __init__(self, path: str | Path):
        from tokenizers import Tokenizer
        self._tk = Tokenizer.from_file(str(path))
        self.bos_id = self._tk.token_to_id("<｜begin▁of▁sentence｜>")

    def encode(self, text: str, add_bos: bool = False) -> list[int]:
        ids = self._tk.encode(text, add_special_tokens=False).ids
        return [self.bos_id] + ids if add_bos else ids


# ---- stub engine -----------------------------------------------------------------------------------------


class StubEngine:
    """The Engine contract over canned replies.

    `replies` is a string (every request gets it), a list (request n gets replies[n], the last one repeats) or a
    callable `prompt -> reply`. The reply is yielded in `chunk`-character pieces, so a tag is cut at any place.
    Prompts are recorded in `.prompts`, full call arguments in `.calls`. `cached_tokens` reports the common token
    prefix with the previous prompt, as a real prefix cache would.
    """

    def __init__(self, replies: str | Sequence[str] | Callable[[str], str] = "", *, tokenizer: Any = None,
                 ctx: int = 65536, sessions: int = 1, chunk: int = 3, vocab_size: int = 129280, vision: bool = False):
        self.replies = replies
        self.tokenizer = tokenizer or ByteTokenizer()
        self.ctx = ctx
        self.sessions = sessions
        self.chunk = chunk
        self.vocab_size = vocab_size
        self.vision_cfg = None
        if vision:                                      # real image preprocessing, no tower
            from exllamav3.dsv41.vision import VisionConfig
            self.vision_cfg = VisionConfig(image_token_id=self.tokenizer.encode("<｜deepseek_image｜>")[0])
        self.prompts: list[str] = []
        self.calls: list[dict[str, Any]] = []
        self.resets = 0
        self.last_stats: dict[str, Any] = {}
        self._last_ids: list[int] = []

    @classmethod
    def from_model_dir(cls, path: str, *, ctx: int = 65536, **options: Any) -> "StubEngine":
        tk = Path(path).expanduser() / "tokenizer.json"
        replies = options.pop("replies", "This is the stub engine.")
        tokenizer = None
        if tk.is_file():
            try:
                tokenizer = HFTokenizer(tk)
            except Exception:                                       # noqa: BLE001  (no tokenizers library: bytes)
                tokenizer = None
        return cls(replies, tokenizer=tokenizer, ctx=ctx,
                   **{k: v for k, v in options.items() if k in ("chunk", "sessions", "vision")})

    # -- tokenization --
    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    @property
    def supports_vision(self) -> bool:
        return self.vision_cfg is not None

    def prepare_images(self, records: list[dict]) -> list[Any]:
        from exllamav3.dsv41 import vision
        if self.vision_cfg is None:
            raise vision.ImageError("this model has no vision tower")
        return [vision.prepare_image(r, self.vision_cfg) for r in records]

    def prompt_ids(self, prompt: str, images: list[Any] | None = None) -> list[int]:
        ids = self.tokenizer.encode(prompt)
        if not images:
            return ids
        from exllamav3.dsv41 import vision
        return vision.expand_image_tokens(ids, images, self.vision_cfg.image_token_id)[0]

    def encode_ids(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self.tokenizer.encode(text, add_bos=add_special_tokens)

    async def score(self, prompt: str, cont_ids: list[int], cand_ids: list[list[int]], prompt_ids: list[int] | None = None) -> list[dict]:
        """Deterministic fake of PortEngine.score: the argmax is the continuation token, except every 10th position."""
        out = []
        for i, tok in enumerate(cont_ids):
            top = tok if i % 10 else tok + 1
            out.append({"top1": top, "top1_logp": -0.1,
                        "cand_logps": [-0.1 if c == top else -3.0 for c in cand_ids[i]]})
        return out

    # -- generation --
    def _reply(self, prompt: str) -> str:
        r = self.replies
        if callable(r):
            return r(prompt)
        if isinstance(r, str):
            return r
        return r[min(len(self.prompts) - 1, len(r) - 1)]

    async def generate(self, prompt: str, *, max_tokens: int, temperature: float, top_p: float, stop: list[str],
                       cancel: asyncio.Event | None = None, stats: dict | None = None,
                       prompt_ids: list[int] | None = None, ignore_eos: bool = False, cache_prompt: bool = True,
                       images: list[Any] | None = None, **_: Any) -> AsyncIterator[str]:
        ids = list(prompt_ids) if prompt_ids is not None else self.prompt_ids(prompt, images)
        self.prompts.append(prompt)
        self.calls.append({"prompt": prompt, "ids": ids, "max_tokens": max_tokens, "temperature": temperature,
                           "top_p": top_p, "stop": list(stop), "ignore_eos": ignore_eos, "cache_prompt": cache_prompt,
                           "images": list(images or [])})
        keys = ids
        if images:                                      # image positions carry the pixel hash, like PortEngine
            from exllamav3.dsv41 import vision
            keys = vision.prompt_key(ids, vision.locate_spans(ids, images, self.vision_cfg.image_token_id))
        cached = 0
        if cache_prompt:
            for a, b in zip(keys, self._last_ids):
                if a != b:
                    break
                cached += 1
        self._last_ids = keys
        reply = self._reply(prompt)
        out, used = [], 0
        for ch in reply:                                            # the token budget cuts the reply
            if used + self.count_tokens(ch) > max_tokens:
                break
            used += self.count_tokens(ch)
            out.append(ch)
        text = "".join(out)
        done = {"cached_tokens": cached, "new_tokens": used, "prompt_tokens": len(ids), "time_prefill": 0.5,
                "time_generate": 1.0, "accepted_draft_tokens": 6, "rejected_draft_tokens": 6,
                "eos": len(text) == len(reply), "eos_reason": "eos" if len(text) == len(reply) else "max_new_tokens"}
        self.last_stats = {}
        for i in range(0, len(text), self.chunk):
            if cancel is not None and cancel.is_set():
                return
            await asyncio.sleep(0)
            yield text[i:i + self.chunk]
        self.last_stats = done
        if stats is not None:
            stats.update(done)

    async def reset_prefix_cache(self) -> None:
        self.resets += 1
        self._last_ids = []


def stub_reply(content: str = "", *, reasoning: str | None = None, calls: Sequence[tuple[str, dict]] = ()) -> str:
    """A well-formed DS4.1 reply for the stub: [reasoning</think>]content[\\n\\n<｜DSML｜ calls>...]<EOS>.

    Not a second encoder: golden tests render history through ds41_chat, this one builds the model's side only.
    """
    text = (reasoning + "</think>" if reasoning is not None else "") + content
    if calls:
        invokes = []
        for name, args in calls:
            params = "\n".join(
                f'<｜DSML｜ parameter name="{k}" string="{"true" if isinstance(v, str) else "false"}">'
                f'{v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)}</｜DSML｜ parameter>'
                for k, v in args.items())
            invokes.append(f'<｜DSML｜ invoke name="{name}">\n{params}\n</｜DSML｜ invoke>')
        text += "\n\n<｜DSML｜ calls>\n" + "\n".join(invokes) + "\n</｜DSML｜ calls>"
    return text
