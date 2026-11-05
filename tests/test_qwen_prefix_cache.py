"""The qwen chat template keeps the system-turn prefix when thinking is off and an effort is sent."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

QWEN = Path(__file__).resolve().parents[1] / "tools" / "qwen"
sys.path.insert(0, str(QWEN))
import serve  # noqa: E402

TEMPLATE = (QWEN / "chat_template.jinja").read_text()
NEW_CONDITION = "{%- if enable_thinking is undefined or enable_thinking is true or reasoning_effort is defined %}"
OLD_CONDITION = "{%- if enable_thinking is undefined or enable_thinking is true %}"
LEGACY_TEMPLATE = TEMPLATE.replace(NEW_CONDITION, OLD_CONDITION)
assert LEGACY_TEMPLATE != TEMPLATE

DEFAULTS = serve.load_defaults("/x")
TOOLS = [{"type": "function", "function": {"name": "list_files", "parameters": {
    "type": "object", "properties": {"dir": {"type": "string"}}, "required": ["dir"]}}}]
MESSAGES = [
    {"role": "system", "content": "You are a coding assistant."},
    {"role": "user", "content": "List the files in src."},
    {"role": "assistant", "content": "", "reasoning_content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "list_files", "arguments": '{"dir": "src"}'}}]},
    {"role": "tool", "tool_call_id": "c1", "content": "main.py"},
    {"role": "user", "content": "Thanks, now summarize."},
]
GENERATION_TAIL = "<|im_start|>assistant\n<think>\n"
THINKING_OFF_TAIL = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def render(template: str, **kwargs) -> str:
    return serve.render_prompt(template, serve.normalize_messages(MESSAGES), TOOLS, **kwargs)


def test_explicit_effort_keeps_the_system_prefix_with_thinking_off():
    on_kw = serve.resolve_template_kwargs({}, DEFAULTS)
    off_kw = serve.resolve_template_kwargs({"enable_thinking": False, "reasoning_effort": "medium"}, DEFAULTS)
    assert on_kw == {"reasoning_effort": "medium"}
    assert off_kw == {"enable_thinking": False, "reasoning_effort": "medium"}

    thinking_on, thinking_off = render(TEMPLATE, **on_kw), render(TEMPLATE, **off_kw)
    assert thinking_on.endswith(GENERATION_TAIL)
    assert thinking_off.endswith(THINKING_OFF_TAIL)
    assert thinking_on[: -len("<think>\n")] == thinking_off[: -len("<think>\n\n</think>\n\n")]


def test_thinking_off_without_a_requested_effort_is_unchanged():
    new_kw = serve.resolve_template_kwargs({"enable_thinking": False}, DEFAULTS)
    old_kw = {"enable_thinking": False, "reasoning_effort": DEFAULTS["reasoning_effort"]}
    assert new_kw == {"enable_thinking": False}
    assert render(TEMPLATE, **new_kw) == render(LEGACY_TEMPLATE, **old_kw)


@pytest.mark.parametrize("body", [
    {"enable_thinking": False},
    {"reasoning_effort": "none"},
    {"chat_template_kwargs": {"enable_thinking": False}},
])
def test_thinking_off_without_effort_renders_as_before(body):
    kw = serve.resolve_template_kwargs(body, DEFAULTS)
    assert "reasoning_effort" not in kw
    assert render(TEMPLATE, **kw) == render(LEGACY_TEMPLATE, **kw)


def test_medium_effort_guidance_is_opt_in(monkeypatch):
    monkeypatch.delenv("KYOJIN_EFFORT_GUIDANCE", raising=False)
    plain = render(TEMPLATE, **serve.resolve_template_kwargs({"reasoning_effort": "medium"}, DEFAULTS))
    assert "Reasoning effort is set to medium." in plain and "Match the depth" not in plain
    on = render(TEMPLATE, **serve.resolve_template_kwargs(
        {"reasoning_effort": "medium", "chat_template_kwargs": {"effort_guidance": True}}, DEFAULTS))
    assert "Reasoning effort is set to medium. Match the depth" in on and "think only about the next step" in on
