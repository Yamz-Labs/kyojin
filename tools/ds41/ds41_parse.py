"""DeepSeek-V4.1 output parsing: thinking, answer and DSML tool calls (pure Python, no engine).

`StreamParser` is fed the decoded text of the reply in chunks cut anywhere (inside a tag too) and returns events:

    ("reasoning", text)                        text of the thinking block
    ("content", text)                          text of the answer
    ("tool_start", index, call_id, name, ns)   a call opened; `name` is the bare name, `ns` its namespace or None
    ("tool_args", index, fragment)             next piece of the call's JSON arguments
    ("tool_end", index)                        the call is closed (its arguments are complete JSON)

The arguments pieces add up, byte for byte, to what the official `decode_dsml_to_arguments` returns:
`{"k": "v", "n": 5}`. A model reply is never an error here: output the official parser would reject (no
</think>, no EOS, a cut-off tool call, text where a tag belongs) is turned into the closest valid events, and
`parser.truncated` says a tool call was cut short.

The wire format (the official encoding writes it, `ds41_chat.render_message` re-creates it):

    <reasoning></think>answer\\n\\n<｜DSML｜ calls>
    <｜DSML｜ invoke name="f">
    <｜DSML｜ parameter name="p" string="true">text</｜DSML｜ parameter>
    <｜DSML｜ parameter name="n" string="false">5</｜DSML｜ parameter>
    </｜DSML｜ invoke>
    </｜DSML｜ calls>

In thinking mode the prompt ends in `<think>`, so the reply starts inside the reasoning block; in chat mode it ends
in `</think>` and the reply starts with the answer.
"""
from __future__ import annotations

import json
import re
import uuid
from typing import Any

DSML = "｜DSML｜"
THINK_CLOSE = "</think>"
EOS = "<｜end▁of▁sentence｜>"
CALLS_OPEN = f"<{DSML} calls>"
CALLS_CLOSE = f"</{DSML} calls>"
INVOKE_OPEN = f"<{DSML} invoke"
INVOKE_CLOSE = f"</{DSML} invoke>"
PARAM_OPEN = f"<{DSML} parameter"
PARAM_CLOSE = f"</{DSML} parameter>"
_INVOKE_RE = re.compile(re.escape(INVOKE_OPEN) + r'\s+name="([^"\n]*)"\s*>')
_PARAM_RE = re.compile(re.escape(PARAM_OPEN) + r'\s+name="([^"\n]*)"\s+string="(true|false)"\s*>')
_MAX_TAG = 2048                        # an opening tag longer than this is not one

Event = tuple


def _partial_suffix(buf: str, markers: tuple[str, ...]) -> int:
    """Length of the longest suffix of buf that is a proper prefix of some marker."""
    best = 0
    for marker in markers:
        for n in range(min(len(marker) - 1, len(buf)), best, -1):
            if buf.endswith(marker[:n]):
                best = n
                break
    return best


def _trailing_newlines(text: str, limit: int = 2) -> int:
    n = 0
    while n < limit and n < len(text) and text[-1 - n] == "\n":
        n += 1
    return n


class StreamParser:
    """Incremental parser. `feed(chunk)` and `finish()` return lists of events."""

    def __init__(self, thinking: bool = True, parse_tools: bool = True):
        self.mode = "reasoning" if thinking else "content"      # reasoning | content | calls | invoke | param | end
        self.parse_tools = parse_tools
        self.buf = ""
        self.truncated = False
        self.tool_count = 0
        self._strip_lead = False        # after a calls block: skip whitespace before the next text
        self._args_open = False         # the "{" of the current call's arguments was emitted
        self._param_first = True
        self._param_string = True
        self._param_raw: list[str] = []
        self._finished = False

    # ---- public ----------------------------------------------------------------------------------------

    def feed(self, chunk: str) -> list[Event]:
        if self._finished:
            raise RuntimeError("feed() after finish()")
        self.buf += chunk
        return self._run(final=False)

    def finish(self) -> list[Event]:
        if self._finished:
            return []
        events = self._run(final=True)
        self._finished = True
        return events

    # ---- machine ---------------------------------------------------------------------------------------

    def _run(self, final: bool) -> list[Event]:
        out: list[Event] = []
        while self.mode != "end":
            before = (self.mode, len(self.buf))
            step = {"reasoning": self._text, "content": self._text, "calls": self._calls,
                    "invoke": self._invoke, "param": self._param}[self.mode]
            step(out, final)
            if (self.mode, len(self.buf)) == before:
                break
        if final:
            self._close(out)
        return out

    def _text(self, out: list[Event], final: bool) -> None:
        reasoning = self.mode == "reasoning"
        kind = "reasoning" if reasoning else "content"
        if self._strip_lead:
            self.buf = self.buf.lstrip()
            if not self.buf:
                return
            self._strip_lead = False
        markers = ((THINK_CLOSE,) if reasoning else ()) + (EOS,) + ((CALLS_OPEN,) if self.parse_tools else ())
        hits = [(self.buf.find(m), m) for m in markers if self.buf.find(m) >= 0]
        if hits:
            pos, marker = min(hits)
            text = self.buf[:pos]
            self.buf = self.buf[pos + len(marker):]
            if marker == CALLS_OPEN:
                text = text[:len(text) - _trailing_newlines(text)]       # the "\n\n" the template puts before it
            self._emit(out, kind, text)
            if marker == THINK_CLOSE:
                self.mode = "content"
            elif marker == EOS:
                self.buf = ""
                self.mode = "end"
            else:
                self.mode = "calls"
            return
        if final:
            self._emit(out, kind, self.buf)
            self.buf = ""
            return
        held = _partial_suffix(self.buf, markers)
        if self.parse_tools:
            held += _trailing_newlines(self.buf[:len(self.buf) - held])  # "\n\n" before <｜DSML｜ calls> is not text
        safe = len(self.buf) - held
        self._emit(out, kind, self.buf[:safe])
        self.buf = self.buf[safe:]

    def _structure(self, options: list[tuple[str, re.Pattern | str, str]], final: bool):
        """Next tag in buf after whitespace: (kind, match-or-None) / "wait" / None when the text is no tag at all."""
        s = self.buf.lstrip()
        if not s:
            return "wait"
        for kind, pattern, _ in options:
            m = pattern.match(s) if isinstance(pattern, re.Pattern) else (s.startswith(pattern) and pattern)
            if m:
                self.buf = s[(m.end() if isinstance(m, re.Match) else len(m)):]
                return kind, m
        for _, _, literal in options:
            if literal.startswith(s) or (s.startswith(literal) and "\n" not in s and len(s) < _MAX_TAG):
                return "drop" if final else "wait"       # a tag that was cut off by the end of the reply
        return None

    def _calls(self, out: list[Event], final: bool) -> None:
        got = self._structure([("invoke", _INVOKE_RE, INVOKE_OPEN), ("close", CALLS_CLOSE, CALLS_CLOSE)], final)
        if got == "wait":
            return
        if got == "drop":
            self.truncated, self.buf, self.mode = True, "", "end"
            return
        if got is None:
            self._malformed(out)
            return
        kind, m = got
        if kind == "close":
            self.mode, self._strip_lead = "content", True
            return
        namespace, sep, bare = m.group(1).partition("::")
        name, ns = (bare, namespace) if sep else (m.group(1), None)
        out.append(("tool_start", self.tool_count, f"call_{uuid.uuid4().hex[:24]}", name, ns))
        self._args_open, self._param_first = False, True
        self.mode = "invoke"

    def _invoke(self, out: list[Event], final: bool) -> None:
        got = self._structure([("param", _PARAM_RE, PARAM_OPEN), ("close", INVOKE_CLOSE, INVOKE_CLOSE)], final)
        if got == "wait":
            return
        if got == "drop":
            self.buf = ""
            return                                      # _close() ends the call
        if got is None:
            self._end_call(out)
            self._malformed(out)
            return
        kind, m = got
        if kind == "close":
            self._end_call(out)
            self.mode = "calls"
            return
        key = json.dumps(m.group(1), ensure_ascii=False)
        self._param_string = m.group(2) == "true"
        head = ("{" if not self._args_open else ", ") + key + ": "
        self._args_open = True
        out.append(("tool_args", self.tool_count, head + ('"' if self._param_string else "")))
        self._param_raw = []
        self.mode = "param"

    def _param(self, out: list[Event], final: bool) -> None:
        pos = self.buf.find(PARAM_CLOSE)
        if pos >= 0:
            value, self.buf = self.buf[:pos], self.buf[pos + len(PARAM_CLOSE):]
            self._value(out, value)
            self._finish_param(out)
            self.mode = "invoke"
            return
        held = 0 if final else _partial_suffix(self.buf, (PARAM_CLOSE,))
        safe = len(self.buf) - held
        self._value(out, self.buf[:safe])
        self.buf = self.buf[safe:]

    def _value(self, out: list[Event], text: str) -> None:
        if not text:
            return
        if self._param_string:
            out.append(("tool_args", self.tool_count, json.dumps(text, ensure_ascii=False)[1:-1]))
        else:
            self._param_raw.append(text)               # JSON values are short: held until the closing tag

    def _finish_param(self, out: list[Event]) -> None:
        if self._param_string:
            out.append(("tool_args", self.tool_count, '"'))
            return
        raw = "".join(self._param_raw)
        try:
            json.loads(raw)
            fixed = raw
        except ValueError:                             # not JSON despite string="false": keep it as a string
            fixed = json.dumps(raw, ensure_ascii=False)
        out.append(("tool_args", self.tool_count, fixed))

    def _end_call(self, out: list[Event]) -> None:
        out.append(("tool_args", self.tool_count, "}" if self._args_open else "{}"))
        out.append(("tool_end", self.tool_count))
        self.tool_count += 1

    def _malformed(self, out: list[Event]) -> None:
        """Text where a tag belongs: give up on markup, the rest of the reply is plain answer text."""
        self.parse_tools = False
        self.mode = "content"
        self._strip_lead = False

    def _close(self, out: list[Event]) -> None:
        """End of the reply: cut-off markup is closed so every call the client saw ends in valid JSON."""
        if self.mode == "param":
            self.truncated = True
            self._finish_param(out)
            self.mode = "invoke"
        if self.mode == "invoke":
            self.truncated = True
            self._end_call(out)
        self.mode = "end"
        self.buf = ""

    def _emit(self, out: list[Event], kind: str, text: str) -> None:
        if text:
            out.append((kind, text))


def split_events(events: list[Event]) -> dict[str, Any]:
    """Events -> {"reasoning_content", "content", "tool_calls"} with OpenAI tool-call objects."""
    reasoning, content, calls = [], [], []
    for ev in events:
        if ev[0] == "reasoning":
            reasoning.append(ev[1])
        elif ev[0] == "content":
            content.append(ev[1])
        elif ev[0] == "tool_start":
            call = {"id": ev[2], "type": "function", "function": {"name": ev[3], "arguments": ""}}
            if ev[4] is not None:
                call["namespace"] = ev[4]
            calls.append(call)
        elif ev[0] == "tool_args":
            calls[ev[1]]["function"]["arguments"] += ev[2]
    return {"reasoning_content": "".join(reasoning), "content": "".join(content), "tool_calls": calls}


def parse_completion(text: str, thinking: bool = True, parse_tools: bool = True) -> dict[str, Any]:
    """The whole reply at once. Adds "truncated" (a tool call was cut short) to split_events()."""
    parser = StreamParser(thinking, parse_tools)
    events = parser.feed(text) + parser.finish()
    message = split_events(events)
    message["truncated"] = parser.truncated
    return message
