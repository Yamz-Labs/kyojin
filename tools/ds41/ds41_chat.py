"""DeepSeek-V4.1 chat rendering for the DS4.1 server (pure Python, no engine, no tokenizer).

This is a text-only port of the official reference `encoding/encoding.py` of
deepseek-ai/DeepSeek-V4.1-Flash-1017 (MIT License, DeepSeek-AI). The rendering functions keep the official
structure and wording so that a diff against the reference stays readable; the golden tests in
`test_ds41_chat.py` compare this module with a vendored copy of the reference, string by string and token id by
token id. What this port adds on top of the reference:

- an OpenAI request layer (`normalize_messages`): `developer` role, `content: null`, list-of-parts content,
  string tool-call arguments, top-level `tools`, `tool` messages;
- request options (`chat_options`): thinking on/off and the numeric reasoning effort, from `reasoning_effort`,
  `chat_template_kwargs` and a few common switches (documented in SERVE.md);
- errors as `ChatFormatError` (HTTP 400) instead of `assert` / `NotImplementedError`.

Images: `render_chat_vl` also returns the image records of the request, in prompt order, and the prompt carries one
`<｜deepseek_image｜>` per image, exactly where the official `encode_messages(..., return_multi_modal_data=True)` puts
it. The span expansion (start, rows, newlines, end) is the engine's job (exllamav3/dsv41/vision.py). `render_chat`
without a collector still refuses image parts.

What it leaves out: the quick-instruction `task` tokens, `latest_reminder` and `response_format`.
"""
from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from typing import Any

BOS = "<｜begin▁of▁sentence｜>"
EOS = "<｜end▁of▁sentence｜>"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
DSML = "｜DSML｜"
USER_SP = "<｜User｜>"
ASSISTANT_SP = "<｜Assistant｜>"
SYSTEM_SP = "<｜System｜>"
IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"
IMAGE_PART_TYPES = ("image_url", "image", "input_image")
_MARK = "\x00IMG"                                  # sentinel put where an image part stood, replaced after rendering
_MARK_RE = re.compile(r"\x00IMG(\d+)\x00")

# V4.1 DSML tag names carry a leading space (V4 had none).
TOOL_CALLS_BLOCK = " calls"
TOOL_CALL_TAG = " invoke"
TOOL_PARAM_TAG = " parameter"

TOOL_OUTPUT_TEMPLATE = "<tool_result>{content}</tool_result>"

REASONING_EFFORT_TEMPLATE = (
    "Reasoning Effort: {budget} (range 1-100, the higher the value, the more thorough the reasoning)\n\n")

# Official aliases (low / high / max) plus the OpenAI level names the reference does not define.
# `minimal`, `medium` and `xhigh` are this server's choice; `none` is not an effort, it selects chat mode.
EFFORT_LEVELS: dict[str, int] = {
    "minimal": 25, "low": 50, "medium": 62, "high": 75, "xhigh": 100, "max": 100}
DEFAULT_EFFORT = "high"

TOOLS_TEMPLATE = """## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<{dsml_token}{tc_block_name}>" block like the following:

<{dsml_token}{tc_block_name}>
<{dsml_token}{tool_call_tag_name} name="$TOOL_NAME">
<{dsml_token}{tool_parameter_tag_name} name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{dsml_token}{tool_parameter_tag_name}>
...
</{dsml_token}{tool_call_tag_name}>
<{dsml_token}{tool_call_tag_name} name="$TOOL_NAME2">
...
</{dsml_token}{tool_call_tag_name}>
</{dsml_token}{tc_block_name}>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by {thinking_start_token}), you MUST output your complete reasoning inside {thinking_start_token}...{thinking_end_token} BEFORE any tool calls or final response.

Otherwise, output directly after {thinking_end_token} with tool calls or final response.

### Available Tool Schemas

{tool_schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
"""


class ChatFormatError(ValueError):
    """The request cannot be rendered (the server answers 400)."""


def to_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps(value, ensure_ascii=True)


# ---- reasoning effort and request options ----------------------------------------------------------------


def resolve_effort(effort: Any) -> int:
    """Budget 1..100 from an int in range or one of EFFORT_LEVELS. None means the default level."""
    if effort is None:
        effort = DEFAULT_EFFORT
    if isinstance(effort, bool):
        raise ChatFormatError(f"invalid reasoning effort {effort!r}")
    if isinstance(effort, int):
        if 1 <= effort <= 100:
            return effort
        raise ChatFormatError(f"reasoning effort {effort} is outside 1..100")
    if isinstance(effort, str) and effort in EFFORT_LEVELS:
        return EFFORT_LEVELS[effort]
    raise ChatFormatError(
        f"invalid reasoning effort {effort!r}: use an integer 1..100 or one of {', '.join(EFFORT_LEVELS)} "
        "('none' turns thinking off)")


@dataclass(frozen=True)
class ChatOptions:
    thinking: bool = True
    effort: int = 75
    drop_thinking: bool = True


def _switch(source: dict[str, Any]) -> bool | None:
    """Thinking on/off from one dict (chat_template_kwargs or the request body), None when it says nothing."""
    for key in ("thinking", "enable_thinking"):
        value = source.get(key)
        if value is None:
            continue
        if isinstance(value, bool):
            return value
        if isinstance(value, dict) and value.get("type") in ("enabled", "disabled", "auto"):
            return value["type"] != "disabled"            # DeepSeek API style: {"type": "disabled"}
        raise ChatFormatError(f"{key} must be a boolean or {{\"type\": \"enabled\"|\"disabled\"}}")
    mode = source.get("thinking_mode")
    if mode is not None:
        if mode not in ("chat", "thinking"):
            raise ChatFormatError("thinking_mode must be 'chat' or 'thinking'")
        return mode == "thinking"
    return None


def _reasoning_object(body: dict[str, Any]) -> tuple[bool | None, Any]:
    """OpenRouter-style `reasoning: {"enabled": bool, "effort": level}` (what Hermes sends to a chat_completions
    endpoint). Returns (thinking switch, effort). A level this server does not know is ignored, never a 400: the
    client's vocabulary is wider than ours (`ultra`)."""
    r = body.get("reasoning")
    if r is None:
        return None, None
    if not isinstance(r, dict):
        raise ChatFormatError("reasoning must be an object")
    enabled, effort = r.get("enabled"), r.get("effort")
    if enabled is not None and not isinstance(enabled, bool):
        raise ChatFormatError("reasoning.enabled must be a boolean")
    if effort == "none":
        enabled = False
    if effort not in EFFORT_LEVELS and effort != "none" and not (isinstance(effort, int) and not isinstance(effort, bool) and 1 <= effort <= 100):
        effort = None
    return enabled, effort


def chat_options(body: dict[str, Any], default_thinking: bool = True,
                 default_effort: Any = DEFAULT_EFFORT) -> ChatOptions:
    """Rendering options of a request. Order: chat_template_kwargs, then the top-level fields, then the server defaults.

    Thinking: an explicit switch (`thinking`, `enable_thinking`, `thinking_mode`) wins; else `reasoning_effort: "none"`
    turns it off; else the server default. Effort: an int 1..100 or a level name; "none" and absent use the default.
    """
    ctk = body.get("chat_template_kwargs") or {}
    if not isinstance(ctk, dict):
        raise ChatFormatError("chat_template_kwargs must be an object")
    effort_raw = ctk.get("reasoning_effort")
    if effort_raw is None:
        effort_raw = body.get("reasoning_effort")
    r_switch, r_effort = _reasoning_object(body)
    if effort_raw is None:
        effort_raw = r_effort
    thinking = _switch(ctk)
    if thinking is None:
        thinking = _switch(body)
    if thinking is None:
        thinking = r_switch
    if thinking is None:
        thinking = False if effort_raw == "none" else default_thinking
    effort = resolve_effort(default_effort if effort_raw in (None, "none") else effort_raw)
    drop = ctk.get("drop_thinking")
    if drop is None:
        drop = body.get("drop_thinking")
    if drop is not None and not isinstance(drop, bool):
        raise ChatFormatError("drop_thinking must be a boolean")
    return ChatOptions(thinking=bool(thinking), effort=effort, drop_thinking=True if drop is None else drop)


# ---- OpenAI request -> reference message form ------------------------------------------------------------


def _image_record(part: dict[str, Any]) -> dict[str, Any]:
    """Content part -> {"url"} / {"data"} record (see exllamav3.dsv41.vision.image_record, same rules)."""
    t = part.get("type")
    src = part.get("source")
    if t == "image" and isinstance(src, dict) and src.get("data"):
        return {"data": src["data"]}
    if t == "image" and isinstance(src, dict):
        url = src.get("url")
    else:
        url = part.get("image_url", part.get("url"))
        if isinstance(url, dict):
            url = url.get("url")
    if t == "input_image" and not url and part.get("image_data"):
        return {"data": part["image_data"]}
    if not isinstance(url, str) or not url:
        raise ChatFormatError("image part has no usable source (expected image_url.url, a data: URL or base64 data)")
    return {"url": url}


def _text_of(content: Any, images: list[dict[str, Any]] | None = None) -> str:
    """Message content -> text. With `images` (a list) image parts become a sentinel and their record is appended;
    without it an image part is an error."""
    if content is None:
        return ""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") in ("text", "input_text", "output_text"):
                parts.append(part.get("text") or "")
            elif isinstance(part, dict) and part.get("type") in IMAGE_PART_TYPES:
                if images is None:
                    raise ChatFormatError("image input is not supported here (no vision tower in this engine)")
                images.append(_image_record(part))
                parts.append(f"{_MARK}{len(images) - 1}\x00")
            else:
                kind = part.get("type") if isinstance(part, dict) else type(part).__name__
                raise ChatFormatError(f"unsupported content part: {kind}")
        text = "\n\n".join(parts)
    else:
        raise ChatFormatError("content must be a string or a list of parts")
    if IMAGE_PLACEHOLDER in text:
        raise ChatFormatError(f"content contains the reserved token {IMAGE_PLACEHOLDER}")
    if images is not None and isinstance(content, str) and _MARK in content:
        raise ChatFormatError("content contains a reserved control sequence")
    return text


def _tool_schema(tool: Any) -> dict[str, Any]:
    if not isinstance(tool, dict):
        raise ChatFormatError("each tool must be an object")
    if "function" not in tool and "name" in tool:        # flat form {"type": "function", "name": ...}
        tool = {"type": tool.get("type", "function"), "function": {k: v for k, v in tool.items() if k != "type"}}
    if not isinstance(tool.get("function"), dict) or not tool["function"].get("name"):
        raise ChatFormatError("each tool needs a function with a name")
    return tool


def normalize_messages(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
                       images: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """OpenAI chat messages -> the reference's message dicts (tool results still separate `tool` messages).

    The reference renders `tools` only from a system message, so top-level tools go on the first message when it is
    a system one, and on an empty system message put in front otherwise.
    """
    if not isinstance(messages, list) or not messages:
        raise ChatFormatError("messages must be a non-empty array")
    out: list[dict[str, Any]] = []
    for m in messages:
        if not isinstance(m, dict):
            raise ChatFormatError("each message must be an object")
        role = m.get("role")
        if role == "developer":                      # OpenAI's newer name for the system role
            role = "system"
        if role not in ("system", "user", "assistant", "tool"):
            raise ChatFormatError(f"unsupported message role: {role!r}")
        msg: dict[str, Any] = {"role": role, "content": _text_of(m.get("content"), images)}
        if role == "assistant":
            rc = m.get("reasoning_content")
            if rc is None:
                rc = m.get("reasoning")
            if rc:
                msg["reasoning_content"] = _text_of(rc)
            if m.get("tool_calls"):
                calls = []
                for c in m["tool_calls"]:
                    fn = c.get("function") if isinstance(c, dict) else None
                    if not isinstance(fn, dict) or not fn.get("name"):
                        raise ChatFormatError("each tool call needs a function with a name")
                    call: dict[str, Any] = {"id": c.get("id") or "", "type": "function",
                                            "function": {"name": fn["name"], "arguments": fn.get("arguments", "{}")}}
                    for holder in (c, fn):
                        if holder.get("namespace") is not None:
                            call["namespace"] = holder["namespace"]
                    calls.append(call)
                msg["tool_calls"] = calls
        elif role == "tool":
            msg["tool_call_id"] = m.get("tool_call_id") or ""
        out.append(msg)
    if tools:
        if not isinstance(tools, list):
            raise ChatFormatError("tools must be an array")
        schema = [_tool_schema(t) for t in tools]
        if out[0]["role"] != "system":
            out.insert(0, {"role": "system", "content": ""})
        out[0]["tools"] = schema
    last = out[-1]["role"]
    if last == "assistant" or (last == "system" and len(out) == 1):
        raise ChatFormatError("the last message must come from the user or be a tool result")
    return out


# ---- reference rendering (text only) ---------------------------------------------------------------------


def _split_tool_name(name: str, namespace: str | None = None) -> tuple[str | None, str]:
    prefix, sep, bare = name.partition("::")
    if sep:
        if namespace not in (None, prefix):
            raise ChatFormatError(f"conflicting tool namespaces: {namespace} != {prefix}")
        namespace, name = prefix, bare
    if "::" in name or (namespace is not None and "::" in namespace):
        raise ChatFormatError(f"tool name and namespace must not contain '::': {name}")
    return namespace, name


def _tool_name_for_encoding(tool: dict[str, Any]) -> str:
    namespace = tool.get("namespace")
    if isinstance(namespace, dict):
        namespace = namespace["name"]
    namespace, name = _split_tool_name(tool["name"], namespace)
    return name if namespace is None else f"{namespace}::{name}"


def tools_from_openai_format(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    functions = []
    for tool in tools:
        function = dict(tool["function"])
        if tool.get("namespace") is not None:
            function["namespace"] = tool["namespace"]
        function["name"] = _tool_name_for_encoding(function)
        namespace = function.pop("namespace", None)
        if isinstance(namespace, dict) and namespace.get("description"):
            function["description"] = namespace["description"] + "\n" + (function.get("description") or "")
        functions.append(function)
    return functions


def tool_calls_from_openai_format(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calls = []
    for tool_call in tool_calls:
        function = tool_call["function"]
        namespace, name = _split_tool_name(
            function["name"], tool_call.get("namespace") or function.get("namespace"))
        call = {"name": name, "arguments": function["arguments"]}
        if namespace is not None:
            call["namespace"] = namespace
        calls.append(call)
    return calls


def render_tools(tools: list[dict[str, Any]]) -> str:
    return TOOLS_TEMPLATE.format(
        tool_schemas="\n".join(to_json(t) for t in tools), dsml_token=DSML, tc_block_name=TOOL_CALLS_BLOCK,
        tool_call_tag_name=TOOL_CALL_TAG, tool_parameter_tag_name=TOOL_PARAM_TAG,
        thinking_start_token=THINK_OPEN, thinking_end_token=THINK_CLOSE)


def encode_arguments_to_dsml(tool_call: dict[str, Any]) -> str:
    arguments = tool_call["arguments"]
    if not isinstance(arguments, dict):
        for _ in range(2):                            # a JSON string, also a double-encoded one
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except ValueError:
                    break
            else:
                break
        if not isinstance(arguments, dict):
            arguments = {"arguments": tool_call["arguments"]}
    return "\n".join(
        f'<{DSML}{TOOL_PARAM_TAG} name="{k}" string="{"true" if isinstance(v, str) else "false"}">'
        f'{v if isinstance(v, str) else to_json(v)}</{DSML}{TOOL_PARAM_TAG}>'
        for k, v in arguments.items())


def merge_tool_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tool results become <tool_result> blocks of the user message that follows the tool calls."""
    merged: list[dict[str, Any]] = []
    for msg in messages:
        msg = copy.deepcopy(msg)
        role = msg.get("role")
        if role == "tool":
            block = {"type": "tool_result", "tool_use_id": msg.get("tool_call_id", ""), "content": msg.get("content", "")}
            if merged and merged[-1].get("role") == "user" and "content_blocks" in merged[-1]:
                merged[-1]["content_blocks"].append(block)
            else:
                merged.append({"role": "user", "content_blocks": [block]})
        elif role == "user":
            blocks = [{"type": "text", "text": msg.get("content", "")}]
            if merged and merged[-1].get("role") == "user" and "content_blocks" in merged[-1]:
                merged[-1]["content_blocks"].extend(blocks)
            else:
                msg["content_blocks"] = blocks
                merged.append(msg)
        else:
            merged.append(msg)
    return merged


def sort_tool_results_by_call_order(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tool results of one user message follow the order of the tool calls of the assistant message before it."""
    order: dict[str, int] = {}
    for msg in messages:
        role = msg.get("role")
        if role == "assistant" and msg.get("tool_calls"):
            order = {}
            for idx, tc in enumerate(msg["tool_calls"]):
                if tc.get("id"):
                    order[tc["id"]] = idx
        elif role == "user" and msg.get("content_blocks"):
            tool_blocks = [b for b in msg["content_blocks"] if b.get("type") == "tool_result"]
            if len(tool_blocks) > 1 and order:
                ranked = iter(sorted(tool_blocks, key=lambda b: order.get(b.get("tool_use_id", ""), 0)))
                msg["content_blocks"] = [next(ranked) if b.get("type") == "tool_result" else b
                                         for b in msg["content_blocks"]]
    return messages


def find_last_user_index(messages: list[dict[str, Any]]) -> int:
    """Last user message; a mid-conversation system message counts, a user message with tool results only does not."""
    for idx in range(len(messages) - 1, -1, -1):
        role = messages[idx].get("role")
        if role == "user" or (role == "system" and idx > 0):
            blocks = messages[idx].get("content_blocks")
            if blocks and not any(b.get("type") == "text" for b in blocks):
                continue
            return idx
    return -1


def render_message(index: int, messages: list[dict[str, Any]], thinking: bool, drop_thinking: bool, effort: int) -> str:
    msg = messages[index]
    last_user_idx = find_last_user_index(messages)
    role = msg.get("role")
    content = msg.get("content")
    tools = tools_from_openai_format(msg["tools"]) if msg.get("tools") else None
    tool_calls = tool_calls_from_openai_format(msg["tool_calls"]) if msg.get("tool_calls") else None

    effort_prompt = REASONING_EFFORT_TEMPLATE.format(budget=effort) if index == 0 and thinking else ""
    prompt = SYSTEM_SP if index == 0 and (effort_prompt or role == "system") else ""
    prompt += effort_prompt

    if role == "system":
        if index > 0:
            prompt += SYSTEM_SP
        prompt += content or ""
        if tools:
            prompt += "\n\n" + render_tools(tools)
    elif role == "user":
        prompt += USER_SP
        blocks = msg.get("content_blocks")
        if blocks:
            parts = []
            for block in blocks:
                if block["type"] == "text":
                    parts.append(block.get("text", ""))
                else:
                    parts.append(TOOL_OUTPUT_TEMPLATE.format(content=block.get("content", "")))
            prompt += "\n\n".join(parts)
        else:
            prompt += content or ""
    elif role == "assistant":
        thinking_part = ""
        tc_content = ""
        if tool_calls:
            invokes = [
                f'<{DSML}{TOOL_CALL_TAG} name="{_tool_name_for_encoding(tc)}">\n{encode_arguments_to_dsml(tc)}\n'
                f'</{DSML}{TOOL_CALL_TAG}>' for tc in tool_calls]
            tc_content = "\n\n" + f"<{DSML}{TOOL_CALLS_BLOCK}>\n" + "\n".join(invokes) + f"\n</{DSML}{TOOL_CALLS_BLOCK}>"
        if thinking and (not drop_thinking or index > last_user_idx):
            thinking_part = (msg.get("reasoning_content") or "") + THINK_CLOSE
        prompt += thinking_part + (content or "") + tc_content + EOS
    else:
        raise ChatFormatError(f"unsupported message role: {role!r}")

    if index + 1 < len(messages) and messages[index + 1].get("role") != "assistant":
        return prompt
    if role == "user" or (role == "system" and index > 0):
        prompt += ASSISTANT_SP
        if thinking and (not drop_thinking or index >= last_user_idx):
            prompt += THINK_OPEN
        else:
            prompt += THINK_CLOSE
    return prompt


def render_chat(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, *,
                thinking: bool = True, reasoning_effort: Any = DEFAULT_EFFORT, drop_thinking: bool = True,
                _images: list[dict[str, Any]] | None = None) -> str:
    """The prompt string of an OpenAI chat request, generation header included (it ends in <think> or </think>).

    The string starts with BOS and is meant to be tokenized with special tokens enabled and no extra BOS.
    """
    effort = resolve_effort(reasoning_effort)
    msgs = normalize_messages(messages, tools, _images)
    msgs = sort_tool_results_by_call_order(merge_tool_messages(msgs))
    if any(m.get("tools") for m in msgs):
        drop_thinking = False                         # tool conversations keep every reasoning block
    if thinking and drop_thinking:
        last_user = find_last_user_index(msgs)
        msgs = [dict((k, v) for k, v in m.items() if k != "reasoning_content") if m["role"] == "assistant" and i < last_user
                else m for i, m in enumerate(msgs)]
    return BOS + "".join(render_message(i, msgs, thinking, drop_thinking, effort) for i in range(len(msgs)))


def render_chat_vl(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, **kw: Any
                   ) -> tuple[str, list[dict[str, Any]]]:
    """`render_chat` for requests that may carry images: (prompt, image records in prompt order).

    The prompt holds one IMAGE_PLACEHOLDER per image. The order is read from the rendered prompt, after tool results
    were merged into user turns and sorted by call order (like the reference, which collects images last), so it is
    the order the engine must expand the placeholders in."""
    records: list[dict[str, Any]] = []
    prompt = render_chat(messages, tools, _images=records, **kw)
    order = [int(i) for i in _MARK_RE.findall(prompt)]
    if sorted(order) != list(range(len(records))):          # an image the render dropped (never: all content is rendered)
        raise ChatFormatError("an image part could not be placed in the prompt")
    return _MARK_RE.sub(lambda _: IMAGE_PLACEHOLDER, prompt), [records[i] for i in order]
