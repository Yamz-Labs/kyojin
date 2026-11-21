"""Pure-Python helpers of the Hy4 OpenAI-compatible server (no torch, no engine import: unit-testable on any machine).

  * request parsing for POST /v1/chat/completions and /v1/completions (RequestError = HTTP 400)
  * the chat template: the Jinja source stored in the GGUF (tokenizer.chat_template), rendered like the reference runtimes do
  * the tokenizer, rebuilt from the GGUF vocabulary + merges (special tokens spelled exactly as in the GGUF)
  * the output side: incremental detokenisation (UTF-8 safe), stop strings, EOS, reasoning / content split
"""
from __future__ import annotations

import json
import math
import os
import re
import time
import uuid
from typing import Any, Iterable

import jinja2
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

PRETOK_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hy4_pretokenizer.json")

# OpenAI-style effort names -> the values the template accepts (max / high / low / no_think)
EFFORT_MAP = {"max": "max", "xhigh": "max", "high": "high", "medium": "high", "low": "low", "minimal": "low",
              "no_think": "no_think", "none": "no_think", "off": "no_think"}
DEFAULT_EFFORT = "high"


class RequestError(ValueError):
    """The request is malformed or cannot be served: the server answers `status` with an OpenAI error body."""

    def __init__(self, message: str, status: int = 400, code: str = "invalid_request_error"):
        super().__init__(message)
        self.status, self.code = status, code


# ------------------------------------------------------------------------------------------------ chat template
def _tojson(value: Any, ensure_ascii: bool = True, indent: int | None = None, sort_keys: bool = False, separators=None) -> str:
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, sort_keys=sort_keys, separators=separators)


def _raise_exception(message: str):
    raise jinja2.exceptions.TemplateError(message)


class ChatTemplate:
    """The GGUF chat template. Same environment as the reference runtimes: sandboxed, trim_blocks + lstrip_blocks, a tojson that takes
    ensure_ascii, raise_exception, loop controls."""

    def __init__(self, source: str):
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
        env.filters["tojson"] = _tojson
        env.globals["raise_exception"] = _raise_exception
        env.globals["strftime_now"] = lambda fmt: time.strftime(fmt)
        self.source = source
        self.tpl = env.from_string(source)

    def render_chat(self, r: dict) -> str:
        """the prompt of a parsed chat request (parse_chat): its tools and effort; a prefill keeps the last assistant turn open"""
        return self.render(r["messages"], r["tools"], True, reasoning_effort=r["effort"], **({"raw_last_assistant": True} if r.get("prefill") else {}))

    def render(self, messages: list[dict], tools: list | None = None, add_generation_prompt: bool = True, **kwargs: Any) -> str:
        try:
            return self.tpl.render(messages=messages, tools=tools, add_generation_prompt=add_generation_prompt, **kwargs)
        except jinja2.exceptions.TemplateError as e:
            raise RequestError(f"chat template: {e}") from e
        except (TypeError, ValueError, AttributeError, KeyError, IndexError) as e:      # a message shape the template cannot read (e.g. arguments that are not an object)
            raise RequestError(f"chat template: {type(e).__name__}: {e}") from e


# ------------------------------------------------------------------------------------------------ tokenizer
def build_tokenizer(tokens: list[str], merges: list[str], token_types: list[int], pretok_path: str = PRETOK_JSON):
    """tokenizers.Tokenizer for the byte-level BPE of the GGUF (tokenizer.ggml.model = gpt2, pre = hy_v4). Normal tokens (type 1) and merges come from the
    GGUF; every other token (control / user-defined) is registered as a special token under its GGUF spelling, so the template's
    `<｜hy_start:...｜>` markers encode to ONE id. The pre-tokenizer regexes are not stored in a GGUF: they come from hy4_pretokenizer.json."""
    from tokenizers import Tokenizer
    with open(pretok_path, encoding="utf-8") as f:
        extra = json.load(f)
    vocab = {t: i for i, t in enumerate(tokens) if token_types[i] == 1}
    spec = [{"id": i, "content": t, "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True}
            for i, t in enumerate(tokens) if token_types[i] != 1]
    doc = {"version": "1.0", "truncation": None, "padding": None, "added_tokens": spec, "normalizer": extra["normalizer"],
           "pre_tokenizer": extra["pre_tokenizer"], "post_processor": extra["post_processor"], "decoder": extra["decoder"],
           "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None, "end_of_word_suffix": None,
                     "fuse_unk": False, "byte_fallback": False, "ignore_merges": False, "vocab": vocab,
                     "merges": [m if isinstance(m, str) else " ".join(m) for m in merges]}}
    return Tokenizer.from_str(json.dumps(doc))


class Vocab:
    """tokenizer + the ids / markers the server needs, all derived from the GGUF metadata dict."""

    def __init__(self, meta: dict, pretok_path: str = PRETOK_JSON):
        toks = list(meta["tokenizer.ggml.tokens"])
        self.tokens = toks
        self.tok = build_tokenizer(toks, list(meta["tokenizer.ggml.merges"]), list(meta["tokenizer.ggml.token_type"]), pretok_path)
        self.size = len(toks)
        self.eos_id = int(meta["tokenizer.ggml.eos_token_id"])
        self.eos_ids = {self.eos_id}
        self.template = ChatTemplate(meta["tokenizer.chat_template"])
        self.think_begin = next((t for t in toks if t.startswith("<think")), None)
        self.think_end = next((t for t in toks if t.startswith("</think")), None)
        self.tool_tags = ToolTags(tag_suffix(self.think_begin))

    def chat_prompt(self, r: dict) -> tuple[str, str]:
        """parsed chat request -> (prompt text, forced prefix). tool_choice required / a named function: the assistant turn is prefilled with the call opening
        <tool_calls><tool_call> (and the function name when it is named); the output stream is seeded with the same text so the parser sees the whole call."""
        text = self.template.render_chat(r)
        prefix = ""
        if r.get("force"):
            prefix = self.tool_tags.calls_open + self.tool_tags.call_open + (r["tool_name"] or "")
        return text + prefix, prefix

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=False)


# ------------------------------------------------------------------------------------------------ tool calls
def tag_suffix(think_begin: str | None) -> str:
    """The suffix every control tag of the chat template carries (Hy4: ":6124c78e"), read from the think-open token `<think{suffix}>`."""
    m = re.fullmatch(r"<think(.*)>", think_begin or "")
    return m.group(1) if m else ""


class ToolTags:
    """The tool-call tags the chat template writes and the model reads back (GLM shape, every tag carries the template suffix):

        <tool_calls:S><tool_call:S>NAME<arg_key:S>K</arg_key:S><arg_value:S>V</arg_value:S>...</tool_call:S>...</tool_calls:S>
    """

    def __init__(self, suffix: str = ""):
        def tag(name: str, close: bool = False) -> str:
            return f"<{'/' if close else ''}{name}{suffix}>"
        self.suffix = suffix
        self.calls_open, self.calls_close = tag("tool_calls"), tag("tool_calls", True)
        self.call_open, self.call_close = tag("tool_call"), tag("tool_call", True)
        self.args_re = re.compile(re.escape(tag("arg_key")) + r"(.*?)" + re.escape(tag("arg_key", True)) + r"\s*" + re.escape(tag("arg_value")) + r"(.*?)"
                                  + re.escape(tag("arg_value", True)), re.S)
        self.opens = (self.calls_open, self.call_open)


_TOOL_NAME_RE = re.compile(r"\s*([^\s<]+)")


def _reject_constant(name: str):
    raise ValueError(name)


def _loads(text: str) -> Any:
    """json.loads that refuses NaN / Infinity (not JSON: the arguments string must stay valid JSON)"""
    return json.loads(text, parse_constant=_reject_constant)


def tool_schemas(tools: list | None) -> dict[str, dict]:
    """declared tool name -> its `parameters.properties` (empty when the tool takes none)"""
    out: dict[str, dict] = {}
    for t in tools or []:
        fn = t.get("function", t)
        out[fn.get("name")] = (fn.get("parameters") or {}).get("properties") or {}
    return out


def coerce_arg(value: str, schema: dict | None) -> Any:
    """One argument written as text -> a typed value. The declared type of the parameter wins: a string parameter keeps its text exactly ("3" stays "3");
    without a declaration the value is JSON when it parses and text when not."""
    if not schema:
        try:
            return _loads(value.strip())
        except ValueError:
            return value.strip()
    kind = schema.get("type")
    kinds = [k for k in (kind if isinstance(kind, list) else [kind]) if isinstance(k, str)]
    text = value.strip()
    if "string" in kinds and not (text == "null" and "null" in kinds):
        return value
    if "boolean" in kinds and text.lower() in ("true", "false"):
        return text.lower() == "true"
    if "null" in kinds and text in ("null", "None"):
        return None
    for want, conv in (("integer", int), ("number", float)):
        if want in kinds:
            try:
                got = conv(text)
            except ValueError:
                continue
            if isinstance(got, float) and not math.isfinite(got):
                continue
            return got
    try:
        return _loads(value)
    except ValueError:
        return value


def parse_tool_call(body: str, tags: ToolTags, schemas: dict[str, dict]) -> dict | None:
    """The text between <tool_call> and </tool_call> -> {"name", "arguments" (a JSON object string)}, or None when it is not a call: no name, or a name the
    request did not declare (the request declared tools and this is not one of them)."""
    m = _TOOL_NAME_RE.match(body)
    if not m:
        return None
    name = m.group(1)
    if schemas and name not in schemas:
        return None
    props = schemas.get(name) or {}
    args = {}
    for k, v in tags.args_re.findall(body[m.end():]):
        k = k.strip()
        args[k] = coerce_arg(v, props.get(k))
    return {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}


def new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


# ------------------------------------------------------------------------------------------------ request parsing
def _as_int(body: dict, key: str, default: int | None, lo: int = 0) -> int | None:
    v = body.get(key, default)
    if v is None:
        return default
    if isinstance(v, bool) or not isinstance(v, (int, float)) or int(v) != v or int(v) < lo:
        raise RequestError(f"{key} must be an integer >= {lo}")
    return int(v)


def _stops(body: dict) -> list[str]:
    s = body.get("stop")
    if s is None:
        return []
    if isinstance(s, str):
        s = [s]
    if not isinstance(s, list) or len(s) > 8 or not all(isinstance(x, str) and x for x in s):
        raise RequestError("stop must be a non-empty string or a list of at most 8 non-empty strings")
    return list(s)


def _common(body: dict, default_max: int) -> dict:
    if not isinstance(body, dict):
        raise RequestError("the body must be a JSON object")
    if _as_int(body, "n", 1, 1) != 1:
        raise RequestError("n must be 1: this server returns one choice per request")
    mt = body.get("max_completion_tokens")          # the newer name wins; an explicit null falls back to max_tokens
    if mt is None:
        mt = body.get("max_tokens")
    # None = the request names no budget and the server has no cap (--default-max-tokens 0): the server then gives the whole room left in the context window
    max_tokens = (default_max if default_max > 0 else None) if mt is None else _as_int({"max_tokens": mt}, "max_tokens", default_max, 1)
    for k in ("temperature", "top_p", "presence_penalty", "frequency_penalty"):
        v = body.get(k)
        if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float))):
            raise RequestError(f"{k} must be a number")
    opts = body.get("stream_options")
    if opts is not None and not isinstance(opts, dict):
        raise RequestError("stream_options must be an object")
    return {"max_tokens": max_tokens, "stream": bool(body.get("stream", False)), "stop": _stops(body),
            "include_usage": bool((opts or {}).get("include_usage", False)),
            # the speculative decoder is greedy: sampling parameters are accepted and ignored; reported back in the reply
            "sampling_ignored": body.get("temperature") not in (None, 0, 0.0),
            "speculative": body.get("speculative", True) is not False}


def parse_effort(body: dict) -> str:
    kw = body.get("chat_template_kwargs") or {}
    if not isinstance(kw, dict):
        raise RequestError("chat_template_kwargs must be an object")
    v = kw.get("reasoning_effort", body.get("reasoning_effort"))
    if v is None:
        return DEFAULT_EFFORT
    if kw.get("enable_thinking") is False and "reasoning_effort" not in kw and body.get("reasoning_effort") is None:
        return "no_think"
    if not isinstance(v, str) or v not in EFFORT_MAP:
        raise RequestError(f"reasoning_effort must be one of {sorted(EFFORT_MAP)}")
    return EFFORT_MAP[v]


# Content part types a chat message may carry -> what the server does with them. One line adds a type.
#   text       the part's text is kept, in order (any other key of the part, such as cache_control, is ignored)
#   reasoning  prior reasoning of an assistant message: it becomes the message's reasoning_content (the template decides what it shows); in other roles it is dropped
#   drop       ignored
# Image / audio / file parts (MEDIA_PARTS) cannot be read by the model: each becomes one line of text (KYOJIN_IMAGE_PARTS=note, the default) or a 400 that names the part (reject). A type that is in neither table is a 400 too.
PART_ACTIONS = {"text": "text", "input_text": "text", "output_text": "text", "refusal": "text",
                "reasoning": "reasoning", "thinking": "reasoning", "reasoning_content": "reasoning", "redacted_thinking": "drop"}
MEDIA_PARTS = {"image_url": "image", "input_image": "image", "image": "image", "input_audio": "audio", "audio": "audio", "file": "file", "input_file": "file"}
PART_TEXT_KEYS = ("text", "refusal", "thinking", "reasoning", "reasoning_content", "content")


def _part_text(p: dict) -> str:
    """the text of a text / refusal / reasoning part: the first string among its text-like keys (null or missing = empty text)"""
    for k in PART_TEXT_KEYS:
        if isinstance(p.get(k), str):
            return p[k]
    return ""


MEDIA_NOTES = {"image": "[an image was attached; this model cannot see images]",
               "audio": "[audio was attached; this model cannot hear audio]",
               "file": "[a file was attached; this model cannot read files]"}
_WARNED_IMAGE_PARTS: set = set()


def image_parts_mode() -> str:
    """KYOJIN_IMAGE_PARTS: `note` (default) replaces each image / audio / file part by one line of text; `reject` is the strict 400. Any other value is `note`
    (one warning line per value). Read at each request."""
    v = os.environ.get("KYOJIN_IMAGE_PARTS", "").strip().lower()
    if v in ("", "note", "reject"):
        return v or "note"
    if v not in _WARNED_IMAGE_PARTS:
        _WARNED_IMAGE_PARTS.add(v)
        print(f"warning: KYOJIN_IMAGE_PARTS={v!r} is not note or reject: using note", flush=True)
    return "note"


def _media_part(kind: str) -> str | None:
    """the text a media part of this kind (image, audio, file) is replaced by, or None to refuse it (KYOJIN_IMAGE_PARTS=reject)"""
    return MEDIA_NOTES[kind] if image_parts_mode() == "note" else None


def _content(c: Any, where: str = "content") -> tuple[str, str]:
    """OpenAI content (a string, null, or a list of parts) -> (text, reasoning). `where` names the message in errors ("message 7").
    A part is a bare string, or a dict whose `type` is a PART_ACTIONS key (no type and a `text` key = text). Null parts are skipped. Null / empty text is empty text.
    A part that cannot be used is a RequestError naming the message and the part: "message 7, part 2: type 'video' is not supported"."""
    if c is None:
        return "", ""
    if isinstance(c, str):
        return c, ""
    if not isinstance(c, list):
        raise RequestError(f"{where}: content must be a string or a list of parts")
    text, reasoning = [], []
    for j, p in enumerate(c):
        if p is None:
            continue
        if isinstance(p, str):
            text.append(p)
            continue
        if not isinstance(p, dict):
            raise RequestError(f"{where}, part {j}: a part must be a string or an object")
        kind = p.get("type")
        if kind is None and "text" in p:
            kind = "text"
        action = PART_ACTIONS.get(kind) if isinstance(kind, str) else None
        if action == "text" and any(p.get(k) is not None and not isinstance(p[k], str) for k in PART_TEXT_KEYS[:2]):
            raise RequestError(f"{where}, part {j}: the text of a {kind!r} part must be a string")
        if action == "text":
            text.append(_part_text(p))
        elif action == "reasoning":
            reasoning.append(_part_text(p))
        elif action == "drop":
            continue
        elif isinstance(kind, str) and kind in MEDIA_PARTS and _media_part(MEDIA_PARTS[kind]) is not None:
            text.append(_media_part(MEDIA_PARTS[kind]))
        elif isinstance(kind, str) and kind in MEDIA_PARTS:
            raise RequestError(f"{where}, part {j}: type {kind!r} is not supported (this model reads text only: no {MEDIA_PARTS[kind]} parts)")
        else:
            raise RequestError(f"{where}, part {j}: type {kind!r} is not supported")
    return "".join(text), "".join(reasoning)


def _content_text(c: Any, where: str = "content") -> str:
    return _content(c, where)[0]


def _function_tool(t: Any, where: str) -> dict:
    """one tool of the request -> {"type": "function", "function": {name, description?, parameters?}}. Both spellings are read: the nested one
    ({"type": "function", "function": {...}}) and the flat one the template also takes ({"name": ..., "parameters": ...})."""
    if not isinstance(t, dict):
        raise RequestError(f"{where} must be an object")
    if t.get("type", "function") != "function":
        raise RequestError(f"{where}: only function tools are supported (type {t.get('type')!r})")
    fn = t["function"] if isinstance(t.get("function"), dict) else t
    name = fn.get("name")
    if not isinstance(name, str) or not re.fullmatch(r"[^\s<>]+", name):
        raise RequestError(f"{where}: the function needs a name (a string without spaces)")
    if fn.get("parameters") is not None and not isinstance(fn["parameters"], dict):
        raise RequestError(f"{where}: parameters must be a JSON schema object")
    return {"type": "function", "function": {k: v for k, v in fn.items() if k != "type" and v is not None}}


def check_response_format(body: dict) -> None:
    """`response_format`: text is the default. json_object and json_schema need constrained decoding, which this decoder does not have: a clean 400 beats a reply
    that silently is not JSON."""
    rf = body.get("response_format")
    if rf is None:
        return
    kind = rf.get("type") if isinstance(rf, dict) else None
    if kind in ("json_object", "json_schema"):
        raise RequestError(f"response_format {kind} is not supported yet (no constrained decoding): ask for JSON in the prompt instead")
    if kind != "text":
        raise RequestError("response_format.type must be text, json_object or json_schema")


def parse_tools(body: dict) -> dict:
    """`tools` (or the legacy `functions`) and `tool_choice` (or the legacy `function_call`) of a chat request ->
    {tools: the declared function tools the prompt carries (None = none), tool_choice: none | auto | required | function, tool_name, parse_tools, force}.
    The chat template has no switch for tool_choice: "none" leaves the tools out of the prompt and the output is plain text; a named function keeps only that
    tool in the prompt; "required" and a named function are FORCED: `force` is True and the assistant turn is prefilled with the call opening (Vocab.chat_prompt),
    so the model can only write the arguments (named) or the call (required)."""
    raw = body.get("tools")
    if raw is None or raw == []:
        raw = body.get("functions")
    if raw is not None and not isinstance(raw, list):
        raise RequestError("tools must be a list")
    tools = [_function_tool(t, f"tools[{i}]") for i, t in enumerate(raw or [])]
    choice = body.get("tool_choice")
    if choice is None:
        choice = body.get("function_call")
    name = None
    if choice is None:
        mode = "auto"
    elif isinstance(choice, str):
        if choice not in ("none", "auto", "required"):
            raise RequestError("tool_choice must be none, auto, required or a function")
        mode = choice
    elif isinstance(choice, dict):
        fn = choice.get("function") if isinstance(choice.get("function"), dict) else choice
        name = fn.get("name")
        if choice.get("type", "function") != "function" or not isinstance(name, str) or not name:
            raise RequestError("tool_choice must be none, auto, required or {\"type\": \"function\", \"function\": {\"name\": ...}}")
        mode = "function"
    else:
        raise RequestError("tool_choice must be a string or an object")
    if mode in ("required", "function") and not tools:
        raise RequestError(f"tool_choice {mode if mode == 'required' else repr(name)} needs tools")
    if mode == "function":
        tools = [t for t in tools if t["function"]["name"] == name]
        if not tools:
            raise RequestError(f"tool_choice names the function {name!r}, which is not in tools")
    if mode == "none":
        tools = []
    return {"tools": tools or None, "tool_choice": mode, "tool_name": name, "parse_tools": bool(tools),
            "force": mode in ("required", "function")}


def _call_arguments(args: Any, where: str) -> dict:
    """arguments of an assistant tool call: a JSON object string (OpenAI) or an object -> the object the template iterates"""
    if args is None or (isinstance(args, str) and not args.strip()):
        return {}
    if isinstance(args, str):
        try:
            args = _loads(args)
        except ValueError:
            raise RequestError(f"{where}: function.arguments is not valid JSON") from None
    if not isinstance(args, dict):
        raise RequestError(f"{where}: function.arguments must be a JSON object")
    return args


def _assistant_calls(m: dict, where: str) -> list[dict]:
    calls = m.get("tool_calls")
    if calls is None and isinstance(m.get("function_call"), dict):          # legacy single call
        calls = [{"function": m["function_call"]}]
    if calls is None:
        return []
    if not isinstance(calls, list):
        raise RequestError(f"{where}: tool_calls must be a list")
    out = []
    for j, c in enumerate(calls):
        w = f"{where}.tool_calls[{j}]"
        fn = c.get("function") if isinstance(c, dict) else None
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str) or not fn["name"]:
            raise RequestError(f"{w}: needs function.name")
        item = {"type": "function", "function": {"name": fn["name"], "arguments": _call_arguments(fn.get("arguments"), w)}}
        if isinstance(c.get("id"), str) and c["id"]:
            item["id"] = c["id"]
        out.append(item)
    return out


CHAT_ROLES = {"system": "system", "developer": "system", "user": "user", "assistant": "assistant", "tool": "tool", "function": "tool"}


def parse_chat(body: dict, default_max: int = 1024) -> dict:
    """-> {messages, effort, max_tokens, stream, stop, include_usage, speculative, tools, tool_choice, parse_tools, prefill, ...}. Raises RequestError.
    Messages: assistant messages may carry tool_calls (arguments as a JSON string or an object), `tool` messages carry the result of a call (tool_call_id).
    All system / developer messages (and the instruction line of tool_choice) are merged into ONE leading system message, in order.
    A last assistant message is a prefill: the model continues it (`prefill` True; the template's raw_last_assistant); an empty one is dropped."""
    req = _common(body, default_max)
    req.update(parse_tools(body))
    check_response_format(body)
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs:
        raise RequestError("messages must be a non-empty list")
    system, out = [], []
    for i, m in enumerate(msgs):
        if not isinstance(m, dict) or m.get("role") not in CHAT_ROLES:
            raise RequestError(f"messages[{i}]: role must be system, user, assistant or tool")
        role = CHAT_ROLES[m["role"]]          # developer = system
        content, part_reasoning = _content(m.get("content"), f"message {i}")
        item = {"role": role, "content": content}
        rc = m.get("reasoning_content", m.get("reasoning"))
        if not (isinstance(rc, str) and rc) and role == "assistant":
            rc = part_reasoning                     # reasoning parts of the content (an explicit reasoning_content field wins)
        if role == "system":
            if item["content"]:
                system.append(item["content"])
            continue
        if role == "assistant":
            if isinstance(rc, str) and rc:
                item["reasoning_content"] = rc
            calls = _assistant_calls(m, f"messages[{i}]")
            if calls:
                item["tool_calls"] = calls
        elif role == "tool":
            tcid = m.get("tool_call_id")
            if tcid is not None and not isinstance(tcid, str):
                raise RequestError(f"messages[{i}]: tool_call_id must be a string")
            if tcid:
                item["tool_call_id"] = tcid
        out.append(item)
    while out and out[-1]["role"] == "assistant" and not out[-1]["content"] and "tool_calls" not in out[-1]:
        out.pop()                             # an empty trailing assistant message is not a prefill
    if not out:
        raise RequestError("messages must contain a user message")
    prefill = out[-1]["role"] == "assistant"
    if prefill and "tool_calls" in out[-1]:
        raise RequestError("the last assistant message has tool_calls: add a tool message with each result")
    if prefill and len(out) < 2:
        raise RequestError("the last message must come from the user")
    if system:
        out.insert(0, {"role": "system", "content": "\n\n".join(system)})
    effort = parse_effort(body)
    if req["force"]:
        if prefill:
            raise RequestError("tool_choice required / a named function cannot be combined with a last assistant message")
        effort = "no_think"                   # the forced call replaces the reasoning: the turn opens with an empty think block
    req.update(messages=out, effort=effort, prefill=prefill)
    return req


def parse_completion(body: dict, vocab_size: int | None, default_max: int = 256) -> dict:
    """-> {prompt: str | None, prompt_ids: list[int] | None, ...}"""
    req = _common(body, default_max)
    p = body.get("prompt")
    if isinstance(p, list) and len(p) == 1 and isinstance(p[0], (str, list)):
        p = p[0]
    if isinstance(p, str):
        req.update(prompt=p, prompt_ids=None)
    elif isinstance(p, list) and p and all(isinstance(x, int) and not isinstance(x, bool) for x in p):
        for x in p:
            if x < 0 or (vocab_size is not None and x >= vocab_size):
                raise RequestError(f"prompt token id {x} is outside the vocabulary")
        req.update(prompt=None, prompt_ids=list(p))
    else:
        raise RequestError("prompt must be a string or a list of token ids (one prompt per request)")
    if req["prompt"] == "":
        raise RequestError("prompt is empty")
    return req


# ------------------------------------------------------------------------------------------------ output side
class IncrementalDecoder:
    """token ids -> text deltas that never end in half a UTF-8 character (the standard prefix / read offset scheme)."""

    def __init__(self, vocab: Vocab):
        self.v, self.ids, self.prefix, self.read = vocab, [], 0, 0

    def push(self, ids: Iterable[int]) -> str:
        self.ids.extend(ids)
        old = self.v.decode(self.ids[self.prefix:self.read])
        new = self.v.decode(self.ids[self.prefix:])
        if len(new) > len(old) and not new.endswith("�"):
            self.prefix, self.read = self.read, len(self.ids)
            return new[len(old):]
        return ""

    def flush(self) -> str:
        old = self.v.decode(self.ids[self.prefix:self.read])
        new = self.v.decode(self.ids[self.prefix:])
        self.prefix = self.read = len(self.ids)
        return new[len(old):] if len(new) > len(old) else ""


def _hold(text: str, markers: list[str]) -> int:
    """length of the longest suffix of `text` that is a proper prefix of one of the markers (must wait for more text)"""
    best = 0
    for m in markers:
        for n in range(min(len(m) - 1, len(text)), best, -1):
            if text.endswith(m[:n]):
                best = n
                break
    return best


class OutputStream:
    """Per request: token ids in, (channel, payload) deltas out, channel = "reasoning" | "content" (payload: text) | "tool_call" (payload: {"name", "arguments"},
    arguments = a JSON object string).
    in_think: the prompt ended inside an open think block, so the text starts as reasoning until `think_end`.
    Stops on an EOS id (dropped) or a stop string (cut; only searched in the content channel).
    parse_tools: a tool block (<tool_calls> ... </tool_calls>, or a bare <tool_call> ... </tool_call>) in the content is held back and read call by call, each call
    leaves as a "tool_call" delta the moment its </tool_call> arrives. `tools` (the declared tools) types the arguments and names the calls that count: a block
    that is not a valid call (cut off, no name, a function the request did not declare) stays plain content. A tool block is only looked for after the reasoning.
    prefix: the forced call opening the prompt ended with; it is parsed as the first text of the reply."""

    def __init__(self, vocab: Vocab, in_think: bool = False, stops: list[str] | None = None, tools: list | None = None, parse_tools: bool = False, prefix: str = ""):
        self.v = vocab
        self.dec = IncrementalDecoder(vocab)
        self.in_think = bool(in_think and vocab.think_end)
        self.stops = stops or []
        self.tags = vocab.tool_tags
        self.parse_tools = bool(parse_tools)
        self.schemas = tool_schemas(tools)
        self.tool_text: str | None = None      # the tool block being read (None: not inside one); for <tool_calls> the text after the open tag
        self.tool_wrapped = False
        self.tool_pos = 0
        self.n_calls = 0
        self.buf = prefix                      # text the prompt already ended with (a forced tool call opening): read as the start of the reply
        self.done = False
        self.finish_reason: str | None = None
        self.n_tokens = 0          # generated tokens counted (EOS and anything after it excluded)

    def _tool_open(self) -> tuple[int, str]:
        hits = [(i, t) for t in self.tags.opens for i in [self.buf.find(t)] if i >= 0]
        return min(hits) if hits else (-1, "")

    def _read_tool_block(self, out: list, final: bool) -> bool:
        """read the held tool block as far as it is complete; True when the block ended (the text after it is content again)"""
        tg = self.tags
        self.tool_text += self.buf
        self.buf = ""
        while True:
            t, pos = self.tool_text, self.tool_pos
            end = t.find(tg.calls_close, pos) if self.tool_wrapped else -1
            start = t.find(tg.call_open, pos)
            if end >= 0 and (start < 0 or end < start):
                if t[pos:end].strip():
                    out.append(("content", t[pos:end].strip()))
                self.buf, self.tool_text = t[end + len(tg.calls_close):], None
                return True
            stop = t.find(tg.call_close, start + len(tg.call_open)) if start >= 0 else -1
            if stop < 0:
                break
            if t[pos:start].strip():
                out.append(("content", t[pos:start].strip()))
            call = parse_tool_call(t[start + len(tg.call_open):stop], tg, self.schemas)
            if call:
                self.n_calls += 1
                out.append(("tool_call", call))
            else:
                out.append(("content", t[start:stop + len(tg.call_close)]))
            self.tool_pos = stop + len(tg.call_close)
            if not self.tool_wrapped:
                self.buf, self.tool_text = t[self.tool_pos:], None
                return True
        if final:
            if t[pos:].strip():
                out.append(("content", t[pos:].strip()))         # a block cut off by the end of the stream
            self.tool_text = None
        return False

    def _drain(self, final: bool) -> list[tuple[str, Any]]:
        out: list[tuple[str, Any]] = []
        while True:
            if self.in_think:
                i = self.buf.find(self.v.think_end)
                if i >= 0:
                    if i:
                        out.append(("reasoning", self.buf[:i]))
                    self.buf = self.buf[i + len(self.v.think_end):]
                    self.in_think = False
                    continue
                k = 0 if final else _hold(self.buf, [self.v.think_end])
                if len(self.buf) > k:
                    out.append(("reasoning", self.buf[:len(self.buf) - k]))
                    self.buf = self.buf[len(self.buf) - k:]
                break
            if self.tool_text is not None:
                if self._read_tool_block(out, final):
                    continue
                break
            cut = min((j for j in (self.buf.find(s) for s in self.stops) if j >= 0), default=-1)
            at, tag = self._tool_open() if self.parse_tools else (-1, "")
            if cut >= 0 and (at < 0 or cut < at):
                if cut:
                    out.append(("content", self.buf[:cut]))
                self.buf, self.done, self.finish_reason = "", True, "stop"
                break
            if at >= 0:
                if self.buf[:at].rstrip():
                    out.append(("content", self.buf[:at].rstrip()))
                self.tool_wrapped, self.tool_pos = tag == self.tags.calls_open, 0
                self.tool_text = self.buf[at + len(tag):] if self.tool_wrapped else self.buf[at:]
                self.buf = ""
                continue
            k = 0 if final else _hold(self.buf, self.stops + (list(self.tags.opens) if self.parse_tools else []))
            if self.parse_tools and not final:       # the whitespace before a tool block is dropped: wait for the next character
                start = len(self.buf) - k
                while start > 0 and self.buf[start - 1].isspace():
                    start -= 1
                k = len(self.buf) - start
            if len(self.buf) > k:
                out.append(("content", self.buf[:len(self.buf) - k]))
                self.buf = self.buf[len(self.buf) - k:]
            break
        return out

    def push(self, ids: list[int]) -> list[tuple[str, Any]]:
        """feed the tokens of one decode round; returns the deltas that are safe to send. `done` turns True on EOS or a stop string."""
        if self.done:
            return []
        take, eos = [], False
        for t in ids:
            if t in self.v.eos_ids:
                eos = True
                break
            take.append(t)
        self.n_tokens += len(take)
        self.buf += self.dec.push(take)
        if eos:
            self.buf += self.dec.flush()
        out = self._drain(final=eos)
        if eos:
            self.done, self.finish_reason = True, "stop"
        return out

    def finish(self) -> list[tuple[str, Any]]:
        """flush at the end of the stream when it ended on the length limit (nothing is left after EOS or a stop string)"""
        if self.done:
            return []
        self.buf += self.dec.flush()
        out = self._drain(final=True)
        self.buf = ""
        return out

    def final_reason(self) -> str:
        """finish_reason of the reply: tool_calls when the model ended its turn with calls, else stop / length"""
        reason = self.finish_reason or "length"
        return "tool_calls" if self.n_calls and reason == "stop" else reason


# ------------------------------------------------------------------------------------------------ response bodies
ABLIT_OFF = ("", "off", "0", "false", "no", "none")
ABLIT_ON = ("on", "1", "preset", "true", "yes")


def ablit_spec(environ, tree: str) -> str | None:
    """Decensor preset of the Hy4 server (opt-in, off by default). HY4_ABLIT_SPEC unset, empty, 0, off, false, no or none -> None (no edit is installed, the forward is
    the unmodified one); on, 1, preset, true or yes -> the preset bundled in the repository (None when its files are absent); any other value is the path of a spec."""
    v = environ.get("HY4_ABLIT_SPEC")
    if v is None:
        return None
    v = v.strip()
    if v.lower() in ABLIT_OFF:
        return None
    if v.lower() in ABLIT_ON:
        p = os.path.join(tree, "presets", "hy4-uncensor", "edit_spec.json")
        return p if os.path.isfile(p) and os.path.isfile(os.path.join(os.path.dirname(p), "edit_spec.safetensors")) else None
    return v


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:24]}"


def chat_chunk(rid: str, model: str, created: int, delta: dict, finish: str | None = None, **extra: Any) -> dict:
    return {"id": rid, "object": "chat.completion.chunk", "created": created, "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}], **extra}


def tool_call_item(call: dict, call_id: str) -> dict:
    """{"name", "arguments"} of the parser -> an OpenAI tool_calls entry"""
    return {"id": call_id, "type": "function", "function": {"name": call["name"], "arguments": call["arguments"]}}


def tool_call_deltas(index: int, call_id: str, call: dict) -> list[dict]:
    """the two streaming deltas of one call, as OpenAI sends them: id + name (empty arguments), then the arguments"""
    return [{"tool_calls": [{"index": index, "id": call_id, "type": "function", "function": {"name": call["name"], "arguments": ""}}]},
            {"tool_calls": [{"index": index, "function": {"arguments": call["arguments"]}}]}]


def chat_response(rid: str, model: str, created: int, content: str, reasoning: str, finish: str, usage: dict, tool_calls: list[dict] | None = None, **extra: Any) -> dict:
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning:
        msg["reasoning_content"] = reasoning
    if tool_calls:
        msg["tool_calls"] = tool_calls
        if not content.strip():
            msg["content"] = None
    return {"id": rid, "object": "chat.completion", "created": created, "model": model,
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}], "usage": usage, **extra}


def completion_chunk(rid: str, model: str, created: int, text: str, finish: str | None = None, **extra: Any) -> dict:
    return {"id": rid, "object": "text_completion", "created": created, "model": model,
            "choices": [{"index": 0, "text": text, "finish_reason": finish, "logprobs": None}], **extra}


def error_body(message: str, code: str = "invalid_request_error") -> dict:
    return {"error": {"message": message, "type": code, "code": code}}
