"""CPU tests of Hy4 tool calling (tools/hy4/serve_core.py + serve.py): the request side (tools, tool_choice, assistant tool_calls, tool messages), the prompt the chat
template makes from them, the parser that reads the model's tool calls back out of the generated text (streamed in pieces of any size), the OpenAI response
shapes (message.tool_calls, streamed delta.tool_calls, finish_reason), and a client that closes the stream. No GPU, no model: a fake engine replays a fixed text.
Run: python -m pytest tests/test_hy4_tools_cpu.py   (needs aiohttp + jinja2 + tokenizers)

The tag format (every tag carries the template suffix, here ":6124c78e") is the one of tests/data/hy4_chat_template.jinja:
  <tool_calls:S><tool_call:S>NAME<arg_key:S>K</arg_key:S><arg_value:S>V</arg_value:S>...</tool_call:S>...</tool_calls:S>
and a tool result is written back as  <｜hy_start:S｜>tool<｜hy_middle:S｜><tool_response:S>RESULT</tool_response:S><｜hy_end:S｜>."""
import argparse
import asyncio
import contextlib
import io
import json
import os
import random
import sys
import threading
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools", "hy4"))
sys.path.insert(0, HERE)
import serve as S  # noqa: E402
import serve_core as C  # noqa: E402
from test_hy4_serve_core import TK, tiny_meta  # noqa: E402

from aiohttp import ClientConnectionResetError, web  # noqa: E402
from aiohttp.test_utils import TestClient, TestServer  # noqa: E402

THINK, UNTHINK = f"<think{TK}>", f"</think{TK}>"
START, MIDDLE, END = f"<｜hy_start{TK}｜>", f"<｜hy_middle{TK}｜>", f"<｜hy_end{TK}｜>"
CALLS, ENDCALLS = f"<tool_calls{TK}>", f"</tool_calls{TK}>"
TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Current weather of a city.",
                                      "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}},
                                                     "required": ["city"]}}},
    {"type": "function", "function": {"name": "search", "description": "Web search.",
                                      "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"},
                                                                                       "other": {"type": "array"}, "safe": {"type": "boolean"}}, "required": ["query"]}}},
]


def tool_meta():
    """tiny_meta + the tool tags as special tokens, like the real vocabulary (each tag is ONE token)"""
    m = tiny_meta()
    tags = [f"<{c}{n}{TK}>" for n in ("tool_calls", "tool_call", "arg_key", "arg_value", "tool_response") for c in ("", "/")]
    m["tokenizer.ggml.tokens"] = m["tokenizer.ggml.tokens"] + tags
    m["tokenizer.ggml.token_type"] = m["tokenizer.ggml.token_type"] + [3] * len(tags)
    return m


def call(name, **args):
    """one call written the way the template writes it (strings raw, anything else as JSON)"""
    body = "".join(f"<arg_key{TK}>{k}</arg_key{TK}><arg_value{TK}>{v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)}</arg_value{TK}>" for k, v in args.items())
    return f"<tool_call{TK}>{name}{body}</tool_call{TK}>"


def run_stream(vocab, text, in_think=False, step=3, eos=True, tools=TOOLS, parse_tools=True, stops=None):
    """text -> tokens -> OutputStream in rounds of `step` tokens; -> (events, stream)"""
    ids = vocab.encode(text)
    o = C.OutputStream(vocab, in_think, stops, tools, parse_tools)
    ev = []
    for i in range(0, len(ids), step):
        ev += o.push(ids[i:i + step])
    if eos:
        ev += o.push([vocab.eos_id])
    ev += o.finish()
    return ev, o


def collect(ev):
    return {"reasoning": "".join(p for c, p in ev if c == "reasoning"), "content": "".join(p for c, p in ev if c == "content"),
            "calls": [(p["name"], json.loads(p["arguments"])) for c, p in ev if c == "tool_call"]}


class Parser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vocabs = {"special tags": C.Vocab(tool_meta()), "tags as bytes": C.Vocab(tiny_meta())}

    def each(self, text, **kw):
        """the same text through every vocabulary and several round sizes: the result must not depend on how the tokens arrive"""
        results = []
        for name, v in self.vocabs.items():
            for step in (1, 2, 3, 7, 1000):
                ev, o = run_stream(v, text, step=step, **kw)
                results.append((f"{name} step {step}", collect(ev), o))
        first = results[0][1]
        for label, got, _ in results:
            self.assertEqual(got, first, label)
        return first, results[0][2]

    def test_tag_suffix_comes_from_the_vocabulary(self):
        v = self.vocabs["special tags"]
        self.assertEqual(v.tool_tags.suffix, TK)
        self.assertEqual(v.tool_tags.call_open, f"<tool_call{TK}>")
        self.assertEqual(C.ToolTags("").calls_close, "</tool_calls>")

    def test_one_call_after_reasoning(self):
        got, o = self.each(f"I should look.{UNTHINK}{CALLS}{call('get_weather', city='Paris', unit='c')}{ENDCALLS}", in_think=True)
        self.assertEqual(got, {"reasoning": "I should look.", "content": "", "calls": [("get_weather", {"city": "Paris", "unit": "c"})]})
        self.assertEqual((o.finish_reason, o.n_calls, o.final_reason()), ("stop", 1, "tool_calls"))

    def test_content_then_parallel_calls(self):
        text = f"Checking.\n{CALLS}{call('get_weather', city='Paris')}{call('search', query='rain', limit=3)}{ENDCALLS}"
        got, _ = self.each(text)
        self.assertEqual(got["content"], "Checking.")
        self.assertEqual(got["calls"], [("get_weather", {"city": "Paris"}), ("search", {"query": "rain", "limit": 3})])

    def test_arguments_are_typed_by_the_declared_schema(self):
        text = f"{CALLS}{call('search', query='42', limit='42', other=[1, 2], safe=True, extra='7')}{ENDCALLS}"
        got, _ = self.each(text)
        self.assertEqual(got["calls"], [("search", {"query": "42", "limit": 42, "other": [1, 2], "safe": True, "extra": 7})])   # a string parameter keeps its text
        got, _ = self.each(text, tools=None)                                                                                      # nothing declared: JSON when it parses
        self.assertEqual(got["calls"][0][1]["query"], 42)

    def test_unicode_and_quotes_in_arguments(self):
        city = 'Zürich "centre" 東京'
        got, _ = self.each(f"{CALLS}{call('get_weather', city=city)}{ENDCALLS}")
        self.assertEqual(got["calls"], [("get_weather", {"city": city})])

    def test_call_without_the_wrapper_and_wrapper_never_closed(self):
        got, _ = self.each(f"ok{call('get_weather', city='Paris')}")
        self.assertEqual((got["content"], got["calls"]), ("ok", [("get_weather", {"city": "Paris"})]))
        got, o = self.each(f"{CALLS}{call('get_weather', city='Paris')}")
        self.assertEqual((got["content"], got["calls"], o.final_reason()), ("", [("get_weather", {"city": "Paris"})], "tool_calls"))

    def test_text_after_the_calls_is_content(self):
        got, _ = self.each(f"{CALLS}{call('get_weather', city='Paris')}{ENDCALLS}done")
        self.assertEqual((got["content"], len(got["calls"])), ("done", 1))

    def test_a_call_inside_the_reasoning_is_reasoning_text(self):
        c = call("get_weather", city="Paris")
        got, o = self.each(f"maybe {c} ?{UNTHINK}final", in_think=True)
        self.assertEqual((got["reasoning"], got["content"], got["calls"], o.final_reason()), (f"maybe {c} ?", "final", [], "stop"))

    def test_malformed_blocks_stay_plain_content(self):
        cut = f"<tool_call{TK}>get_weather<arg_key{TK}>city"
        got, o = self.each(f"before {CALLS}{cut}", eos=False)                         # cut off by the length limit
        self.assertEqual(got["calls"], [])
        self.assertEqual(got["content"], f"before{cut}")
        self.assertEqual(o.final_reason(), "length")
        got, _ = self.each(f"{CALLS}{call('delete_everything', path='/')}{ENDCALLS}")   # a function the request did not declare
        self.assertEqual(got["calls"], [])
        self.assertIn("delete_everything", got["content"])
        got, _ = self.each(f"{CALLS}<tool_call{TK}><arg_key{TK}>a</arg_key{TK}><arg_value{TK}>1</arg_value{TK}></tool_call{TK}>{ENDCALLS}")   # no name
        self.assertEqual(got["calls"], [])
        got, _ = self.each(f"{CALLS}stray words {call('get_weather', city='Paris')}{ENDCALLS}")
        self.assertEqual((got["content"], len(got["calls"])), ("stray words", 1))      # text inside the wrapper is kept as content, the call still counts

    def test_a_call_without_arguments(self):
        got, _ = self.each(f"{CALLS}<tool_call{TK}>search</tool_call{TK}>{ENDCALLS}")
        self.assertEqual(got["calls"], [("search", {})])

    def test_tool_choice_none_leaves_the_tags_as_text(self):
        c = call("get_weather", city="Paris")
        got, o = self.each(f"{CALLS}{c}{ENDCALLS}", parse_tools=False)
        self.assertEqual((got["content"], got["calls"], o.final_reason()), (f"{CALLS}{c}{ENDCALLS}", [], "stop"))

    def test_a_stop_string_before_the_call_wins_and_after_it_does_not_cut_the_call(self):
        got, o = self.each(f"one STOP two {CALLS}{call('get_weather', city='Paris')}{ENDCALLS}", stops=["STOP"])
        self.assertEqual((got["content"], got["calls"], o.finish_reason), ("one ", [], "stop"))
        got, _ = self.each(f"{CALLS}{call('get_weather', city='STOP')}{ENDCALLS}", stops=["STOP"])
        self.assertEqual(got["calls"], [("get_weather", {"city": "STOP"})])

    def test_whole_text_is_not_held_back_without_tools(self):
        v = self.vocabs["special tags"]
        ev, _ = run_stream(v, f"plain {CALLS}text\n", parse_tools=False, step=1)
        self.assertEqual(collect(ev)["content"], f"plain {CALLS}text\n")
        ev, _ = run_stream(v, "trailing space \n", parse_tools=True, step=1)          # released at the end of the stream
        self.assertEqual(collect(ev)["content"], "trailing space \n")

    def test_calls_leave_as_soon_as_their_close_tag_arrives(self):
        v = self.vocabs["special tags"]
        first, second = call("get_weather", city="Paris"), call("search", query="x")
        o = C.OutputStream(v, False, None, TOOLS, True)
        seen = o.push(v.encode(f"{CALLS}{first}{second[:20]}"))
        self.assertEqual([p["name"] for c, p in seen if c == "tool_call"], ["get_weather"])
        seen = o.push(v.encode(second[20:] + ENDCALLS))
        self.assertEqual([p["name"] for c, p in seen if c == "tool_call"], ["search"])

    def test_a_forced_call_is_parsed_from_the_prefix_and_the_generated_rest(self):
        v = self.vocabs["special tags"]
        prefix = f"{CALLS}<tool_call{TK}>get_weather"
        rest = f"<arg_key{TK}>city</arg_key{TK}><arg_value{TK}>Paris</arg_value{TK}></tool_call{TK}>{ENDCALLS}"
        for step in (1, 3, 1000):
            ids = v.encode(rest)
            o, ev = C.OutputStream(v, False, None, TOOLS, True, prefix=prefix), []
            for i in range(0, len(ids), step):
                ev += o.push(ids[i:i + step])
            ev += o.push([v.eos_id]) + o.finish()
            self.assertEqual(collect(ev), {"reasoning": "", "content": "", "calls": [("get_weather", {"city": "Paris"})]})
            self.assertEqual(o.final_reason(), "tool_calls")
        o = C.OutputStream(v, False, None, TOOLS, True, prefix=f"{CALLS}<tool_call{TK}>")        # required: the model writes the name too
        ev = o.push(v.encode(f"search<arg_key{TK}>query</arg_key{TK}><arg_value{TK}>rain</arg_value{TK}></tool_call{TK}>{ENDCALLS}") + [v.eos_id]) + o.finish()
        self.assertEqual(collect(ev)["calls"], [("search", {"query": "rain"})])
        o = C.OutputStream(v, False, None, TOOLS, True, prefix=prefix)                          # no token at all: nothing valid, nothing lost
        self.assertEqual(collect(o.finish())["calls"], [])

    def test_random_piece_sizes(self):
        text = f"plan{UNTHINK}Let me check.{CALLS}{call('get_weather', city='Paris')}{call('search', query='a b', limit=2)}{ENDCALLS}"
        rng = random.Random(7)
        want = None
        for v in self.vocabs.values():
            ids = v.encode(text)
            for _ in range(20):
                o, ev, i = C.OutputStream(v, True, None, TOOLS, True), [], 0
                while i < len(ids):
                    n = rng.randint(1, 9)
                    ev += o.push(ids[i:i + n])
                    i += n
                ev += o.push([v.eos_id]) + o.finish()
                got = collect(ev)
                want = want or got
                self.assertEqual(got, want)
        self.assertEqual(want["reasoning"], "plan")
        self.assertEqual(want["content"], "Let me check.")
        self.assertEqual([c[0] for c in want["calls"]], ["get_weather", "search"])


class Request(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v = C.Vocab(tool_meta())

    def parse(self, **body):
        return C.parse_chat({"messages": [{"role": "user", "content": "weather in Paris?"}], **body})

    def test_no_tools_by_default(self):
        r = self.parse()
        self.assertEqual((r["tools"], r["parse_tools"], r["tool_choice"]), (None, False, "auto"))
        r = self.parse(tool_choice="auto")        # a client may send it without tools
        self.assertFalse(r["parse_tools"])

    def test_tools_nested_flat_and_legacy_functions(self):
        want = [{"type": "function", "function": TOOLS[0]["function"]}]
        self.assertEqual(self.parse(tools=[TOOLS[0]])["tools"], want)
        flat = {"type": "function", **TOOLS[0]["function"]}
        self.assertEqual(self.parse(tools=[flat])["tools"], want)
        self.assertEqual(self.parse(functions=[TOOLS[0]["function"]])["tools"], want)
        self.assertEqual(self.parse(tools=[], functions=[TOOLS[0]["function"]])["tools"], want)
        self.assertEqual(self.parse(tools=[TOOLS[0]], strict_ignored=1)["parse_tools"], True)

    def test_tool_choice(self):
        self.assertEqual(self.parse(tools=TOOLS, tool_choice="required")["tool_choice"], "required")
        r = self.parse(tools=TOOLS, tool_choice={"type": "function", "function": {"name": "search"}})
        self.assertEqual((r["tool_choice"], r["tool_name"], [t["function"]["name"] for t in r["tools"]]), ("function", "search", ["search"]))
        r = self.parse(tools=TOOLS, function_call={"name": "search"})
        self.assertEqual(r["tool_name"], "search")
        r = self.parse(tools=TOOLS, tool_choice="none")
        self.assertEqual((r["tools"], r["parse_tools"]), (None, False))
        r = self.parse(tools=TOOLS, function_call="auto")
        self.assertEqual(r["tool_choice"], "auto")

    def test_forced_tool_choice_prefills_the_call(self):
        r = self.parse(tools=TOOLS, tool_choice="required")
        self.assertEqual((r["force"], r["effort"], r["tool_name"]), (True, "no_think", None))
        r = self.parse(tools=TOOLS, tool_choice={"type": "function", "function": {"name": "search"}}, reasoning_effort="max")
        self.assertEqual((r["force"], r["effort"], r["tool_name"]), (True, "no_think", "search"))
        self.assertEqual(r["messages"], [{"role": "user", "content": "weather in Paris?"}])      # the messages are untouched
        for choice in ("auto", "none"):
            r = self.parse(tools=TOOLS, tool_choice=choice)
            self.assertFalse(r["force"])
            self.assertEqual(r["effort"], "high")
        with self.assertRaises(C.RequestError):
            C.parse_chat({"messages": [*r["messages"], {"role": "assistant", "content": "Sure"}], "tools": TOOLS, "tool_choice": "required"})

    def test_chat_prompt_ends_with_the_forced_call_opening(self):
        r = self.parse(tools=TOOLS, tool_choice={"type": "function", "function": {"name": "get_weather"}})
        text, prefix = self.v.chat_prompt(r)
        self.assertEqual(prefix, f"{CALLS}<tool_call{TK}>get_weather")
        self.assertTrue(text.endswith(f"{START}assistant{MIDDLE}{THINK}{UNTHINK}{prefix}"))
        self.assertNotIn('"name": "search"', text)                    # only the named tool is declared
        text, prefix = self.v.chat_prompt(self.parse(tools=TOOLS, tool_choice="required"))
        self.assertEqual(prefix, f"{CALLS}<tool_call{TK}>")
        self.assertTrue(text.endswith(prefix))
        text, prefix = self.v.chat_prompt(self.parse(tools=TOOLS))
        self.assertEqual((prefix, text.endswith(THINK)), ("", True))

    def test_bad_tool_requests_are_a_400(self):
        bad = [dict(tools="x"), dict(tools=[1]), dict(tools=[{"type": "function"}]), dict(tools=[{"type": "function", "function": {"name": "a b"}}]),
               dict(tools=[{"type": "web_search"}]), dict(tools=[{"type": "function", "function": {"name": "f", "parameters": "x"}}]),
               dict(tools=TOOLS, tool_choice="sometimes"), dict(tools=TOOLS, tool_choice=5), dict(tool_choice="required"),
               dict(tools=TOOLS, tool_choice={"type": "function", "function": {"name": "nope"}}), dict(tools=TOOLS, tool_choice={"type": "function"})]
        for b in bad:
            with self.assertRaises(C.RequestError, msg=str(b)):
                self.parse(**b)

    def history(self, args, content=None):
        return [{"role": "user", "content": "weather in Paris?"},
                {"role": "assistant", "content": content, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": args}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": '{"temp": 18}'}]

    def test_assistant_tool_calls_and_tool_messages(self):
        for args in ('{"city": "Paris"}', {"city": "Paris"}):
            r = C.parse_chat({"messages": self.history(args), "tools": TOOLS})
            a, t = r["messages"][1], r["messages"][2]
            self.assertEqual(a["tool_calls"], [{"type": "function", "id": "call_1", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}])
            self.assertEqual((a["content"], t), ("", {"role": "tool", "content": '{"temp": 18}', "tool_call_id": "call_1"}))
        for empty in ("", None, "  "):
            r = C.parse_chat({"messages": self.history(empty)})
            self.assertEqual(r["messages"][1]["tool_calls"][0]["function"]["arguments"], {})

    def test_legacy_function_call_and_function_role(self):
        msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": None, "function_call": {"name": "get_weather", "arguments": '{"city": "Rome"}'}},
                {"role": "function", "name": "get_weather", "content": "sunny"}]
        r = C.parse_chat({"messages": msgs, "functions": [TOOLS[0]["function"]]})
        self.assertEqual(r["messages"][1]["tool_calls"][0]["function"], {"name": "get_weather", "arguments": {"city": "Rome"}})
        self.assertEqual((r["messages"][2]["role"], r["messages"][2]["content"]), ("tool", "sunny"))

    def test_bad_histories_are_a_400(self):
        for args in ("{not json", "[1, 2]", 5, '"text"'):
            with self.assertRaises(C.RequestError, msg=str(args)):
                C.parse_chat({"messages": self.history(args)})
        bad = [[{"role": "user", "content": "x"}, {"role": "assistant", "tool_calls": "x"}, {"role": "tool", "content": "r"}],
               [{"role": "user", "content": "x"}, {"role": "assistant", "tool_calls": [{"function": {}}]}, {"role": "tool", "content": "r"}],
               [{"role": "user", "content": "x"}, {"role": "assistant", "tool_calls": [1]}, {"role": "tool", "content": "r"}],
               [{"role": "user", "content": "x"}, {"role": "tool", "tool_call_id": 5, "content": "r"}]]
        for msgs in bad:
            with self.assertRaises(C.RequestError, msg=str(msgs)):
                C.parse_chat({"messages": msgs})

    def render(self, **body):
        r = C.parse_chat(body)
        return self.v.template.render(r["messages"], r["tools"], True, reasoning_effort=r["effort"])

    def test_the_prompt_carries_the_tools(self):
        text = self.render(messages=[{"role": "user", "content": "weather?"}], tools=TOOLS)
        self.assertIn("# Tools", text)
        self.assertIn("<tools>\n" + json.dumps(TOOLS[0]["function"], ensure_ascii=False), text)
        self.assertIn(json.dumps(TOOLS[1]["function"], ensure_ascii=False), text)
        self.assertIn(f"you should first print {CALLS}", text)
        self.assertTrue(text.endswith(f"{START}assistant{MIDDLE}{THINK}"))
        text = self.render(messages=[{"role": "system", "content": "Be brief."}, {"role": "user", "content": "weather?"}], tools=TOOLS)
        self.assertLess(text.index("# Tools"), text.index("Be brief."))
        self.assertNotIn("# Tools", self.render(messages=[{"role": "user", "content": "x"}], tools=TOOLS, tool_choice="none"))
        self.assertNotIn('"name": "search"', self.render(messages=[{"role": "user", "content": "x"}], tools=TOOLS, tool_choice={"function": {"name": "get_weather"}}))

    def test_multi_turn_history_is_rendered_back_in_the_template_format(self):
        msgs = self.history('{"city": "Paris", "days": 2}', content="Let me check.")
        text = self.render(messages=msgs, tools=TOOLS)
        want = (f"{START}assistant{MIDDLE}{THINK}{UNTHINK}Let me check.{CALLS}<tool_call{TK}>get_weather<arg_key{TK}>city</arg_key{TK}><arg_value{TK}>Paris</arg_value{TK}>"
                f"<arg_key{TK}>days</arg_key{TK}><arg_value{TK}>2</arg_value{TK}></tool_call{TK}>{ENDCALLS}{END}"
                f"{START}tool{MIDDLE}<tool_response{TK}>{{\"temp\": 18}}</tool_response{TK}>{END}{START}assistant{MIDDLE}{THINK}")
        self.assertIn(want, text)

    def test_parallel_results_follow_the_order_of_the_calls(self):
        calls = [{"id": f"c{i}", "type": "function", "function": {"name": "get_weather", "arguments": {"city": c}}} for i, c in enumerate(("Paris", "Rome"))]
        msgs = [{"role": "user", "content": "both"}, {"role": "assistant", "content": "", "tool_calls": calls},
                {"role": "tool", "tool_call_id": "c1", "content": "rome-result"}, {"role": "tool", "tool_call_id": "c0", "content": "paris-result"}]
        text = self.render(messages=msgs, tools=TOOLS)
        self.assertLess(text.index("paris-result"), text.index("rome-result"))
        self.assertEqual(text.count(f"{START}tool{MIDDLE}"), 1)

    def test_the_reasoning_of_a_call_turn_is_kept_when_tools_are_declared(self):
        msgs = self.history('{"city": "Paris"}')
        msgs[1]["reasoning_content"] = "I need the weather."
        self.assertIn(f"{THINK}I need the weather.{UNTHINK}{CALLS}", self.render(messages=msgs, tools=TOOLS))

    def test_a_template_that_cannot_read_a_message_is_a_400(self):
        with self.assertRaises(C.RequestError):
            self.v.template.render([{"role": "user", "content": "x"}, {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "f", "arguments": "text"}}]},
                                    {"role": "tool", "content": "r"}], None, True, reasoning_effort="high")

    def test_response_bodies(self):
        item = C.tool_call_item({"name": "get_weather", "arguments": '{"city": "Paris"}'}, "call_1")
        self.assertEqual(item, {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}})
        b = C.chat_response("id", "m", 1, "", "thinking", "tool_calls", {"total_tokens": 3}, [item])
        msg = b["choices"][0]["message"]
        self.assertEqual((msg["content"], msg["tool_calls"], msg["reasoning_content"], b["choices"][0]["finish_reason"]), (None, [item], "thinking", "tool_calls"))
        self.assertEqual(C.chat_response("id", "m", 1, "Checking.", "", "tool_calls", {}, [item])["choices"][0]["message"]["content"], "Checking.")
        self.assertNotIn("tool_calls", C.chat_response("id", "m", 1, "hi", "", "stop", {})["choices"][0]["message"])
        d = C.tool_call_deltas(1, "call_1", {"name": "f", "arguments": "{}"})
        self.assertEqual(d[0]["tool_calls"][0], {"index": 1, "id": "call_1", "type": "function", "function": {"name": "f", "arguments": ""}})
        self.assertEqual(d[1]["tool_calls"][0], {"index": 1, "function": {"arguments": "{}"}})


class FakeEngine:
    """replays `reply` (then EOS) 3 tokens per round; with `delay` each round takes that long and looks at `cancel` like the real engine does"""

    def __init__(self, vocab, reply, delay=0.0, eos=True):
        self.ids = vocab.encode(reply) + ([vocab.eos_id] if eos else [])
        self.delay, self.calls, self.cancelled, self.rounds = delay, [], threading.Event(), 0

    def generate(self, ids, max_new, on_tokens, speculative=True, cancel=lambda: False):
        self.calls.append((list(ids), max_new))
        n, stop = 0, False
        while n < min(max_new, len(self.ids)) and not stop:
            if cancel():
                self.cancelled.set()
                break
            if self.delay:
                time.sleep(self.delay)
            new = self.ids[n:n + 3][:max_new - n]
            n += len(new)
            self.rounds += 1
            stop = on_tokens(new)
        return {"prompt_tokens": len(ids), "prefill_s": 0.5, "prefill_trunk_s": 0.4, "prime_s": 0.1, "prime_mode": "fake", "decode_s": 0.25, "decode_tokens": n,
                "ttft_s": 0.5, "rounds": self.rounds, "accept_hist": [0, 0, 0, self.rounds], "tau": n / max(self.rounds, 1), "speculative": speculative}


class ProdServer(TestServer):
    """the test server cancels a handler when the client goes away (handler_cancellation=True); web.run_app, which runs the real server, does not"""

    async def _make_runner(self, **kwargs):
        if "handler_cancellation" in kwargs:
            kwargs["handler_cancellation"] = False
        return await super()._make_runner(**kwargs)


def make_state(reply, **kw):
    st = S.State(argparse.Namespace(model_id="Hy4-test", max_ctx=8192, num_draft=3, default_max_tokens=256))
    st.vocab = C.Vocab(tool_meta())
    st.engine = FakeEngine(st.vocab, reply, **kw)
    st.ready = True
    st.lock = asyncio.Lock()
    return st


class Http(unittest.IsolatedAsyncioTestCase):
    async def start(self, reply, server=ProdServer, **kw):
        self.st = make_state(reply, **kw)
        self.client = TestClient(server(S.build_app(self.st)))
        await self.client.start_server()
        self.addAsyncCleanup(self.client.close)
        return self.st

    def body(self, **kw):
        return {"messages": [{"role": "user", "content": "weather in Paris?"}], "tools": TOOLS, "max_tokens": 200, "reasoning_effort": "no_think", **kw}

    async def sse_chunks(self, **kw):
        r = await self.client.post("/v1/chat/completions", json=self.body(stream=True, **kw))
        self.assertEqual(r.status, 200)
        events = [e[6:] for e in (await r.read()).decode().split("\n\n") if e.startswith("data: ")]
        self.assertEqual(events[-1], "[DONE]")
        return [json.loads(e) for e in events[:-1]]

    async def test_non_stream_tool_call(self):
        await self.start(f"{CALLS}{call('get_weather', city='Paris', unit='c')}{ENDCALLS}")
        r = await self.client.post("/v1/chat/completions", json=self.body())
        j = await r.json()
        self.assertEqual(r.status, 200, j)
        ch = j["choices"][0]
        self.assertEqual(ch["finish_reason"], "tool_calls")
        self.assertIsNone(ch["message"]["content"])
        [tc] = ch["message"]["tool_calls"]
        self.assertTrue(tc["id"].startswith("call_"))
        self.assertEqual((tc["type"], tc["function"]["name"]), ("function", "get_weather"))
        self.assertEqual(json.loads(tc["function"]["arguments"]), {"city": "Paris", "unit": "c"})
        self.assertIsInstance(tc["function"]["arguments"], str)
        prompt = self.st.vocab.decode(self.st.engine.calls[0][0])
        self.assertIn("# Tools", prompt)
        self.assertIn('"name": "get_weather"', prompt)

    async def test_reasoning_stays_in_reasoning_content(self):
        await self.start(f"I need the weather.{UNTHINK}Checking.{CALLS}{call('get_weather', city='Paris')}{ENDCALLS}")
        r = await self.client.post("/v1/chat/completions", json=self.body(reasoning_effort="high"))
        m = (await r.json())["choices"][0]["message"]
        self.assertEqual((m["reasoning_content"], m["content"], len(m["tool_calls"])), ("I need the weather.", "Checking.", 1))

    async def test_parallel_calls_get_distinct_ids(self):
        await self.start(f"{CALLS}{call('get_weather', city='Paris')}{call('get_weather', city='Rome')}{ENDCALLS}")
        j = await (await self.client.post("/v1/chat/completions", json=self.body())).json()
        calls = j["choices"][0]["message"]["tool_calls"]
        self.assertEqual([json.loads(c["function"]["arguments"])["city"] for c in calls], ["Paris", "Rome"])
        self.assertEqual(len({c["id"] for c in calls}), 2)

    async def test_plain_answer_with_tools_declared(self):
        await self.start("It is sunny.")
        j = await (await self.client.post("/v1/chat/completions", json=self.body())).json()
        ch = j["choices"][0]
        self.assertEqual((ch["message"]["content"], ch["finish_reason"]), ("It is sunny.", "stop"))
        self.assertNotIn("tool_calls", ch["message"])

    async def test_stream_tool_call(self):
        await self.start(f"Checking.{CALLS}{call('get_weather', city='Paris')}{call('search', query='rain', limit=2)}{ENDCALLS}")
        chunks = await self.sse_chunks(stream_options={"include_usage": True})
        self.assertEqual(chunks[0]["choices"][0]["delta"]["role"], "assistant")
        calls, text, finish = {}, "", []
        for c in chunks:
            if not c["choices"]:
                continue
            d = c["choices"][0]["delta"]
            text += d.get("content") or ""
            for tc in d.get("tool_calls", []):
                slot = calls.setdefault(tc["index"], {"id": None, "name": "", "args": ""})
                slot["id"] = tc.get("id", slot["id"])
                slot["name"] += tc.get("function", {}).get("name", "")
                slot["args"] += tc.get("function", {}).get("arguments", "")
            if c["choices"][0]["finish_reason"]:
                finish.append(c["choices"][0]["finish_reason"])
        self.assertEqual(text, "Checking.")
        self.assertEqual(finish, ["tool_calls"])
        self.assertEqual(sorted(calls), [0, 1])
        self.assertEqual((calls[0]["name"], json.loads(calls[0]["args"])), ("get_weather", {"city": "Paris"}))
        self.assertEqual((calls[1]["name"], json.loads(calls[1]["args"])), ("search", {"query": "rain", "limit": 2}))
        self.assertTrue(calls[0]["id"].startswith("call_") and calls[0]["id"] != calls[1]["id"])
        first = [c["choices"][0]["delta"]["tool_calls"][0] for c in chunks if c["choices"] and c["choices"][0]["delta"].get("tool_calls")]
        self.assertEqual(first[0]["function"], {"name": "get_weather", "arguments": ""})         # id + name first, the arguments in the next chunk
        self.assertEqual(list(first[1]), ["index", "function"])
        self.assertIn("usage", chunks[-1])

    async def test_multi_turn_round_trip(self):
        await self.start("It is 18 degrees and sunny.")
        msgs = [{"role": "user", "content": "weather in Paris?"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "call_abc", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}]},
                {"role": "tool", "tool_call_id": "call_abc", "content": '{"temp": 18, "sky": "sunny"}'}]
        r = await self.client.post("/v1/chat/completions", json=self.body(messages=msgs))
        j = await r.json()
        self.assertEqual(r.status, 200, j)
        self.assertEqual((j["choices"][0]["message"]["content"], j["choices"][0]["finish_reason"]), ("It is 18 degrees and sunny.", "stop"))
        prompt = self.st.vocab.decode(self.st.engine.calls[0][0])
        self.assertIn(f"{CALLS}<tool_call{TK}>get_weather<arg_key{TK}>city</arg_key{TK}><arg_value{TK}>Paris</arg_value{TK}></tool_call{TK}>{ENDCALLS}", prompt)
        self.assertIn(f"<tool_response{TK}>{{\"temp\": 18, \"sky\": \"sunny\"}}</tool_response{TK}>", prompt)

    async def test_malformed_blocks_are_content_never_a_500(self):
        for reply in (f"{CALLS}<tool_call{TK}>get_weather<arg_key{TK}>city", f"{CALLS}{call('unknown_fn', a=1)}{ENDCALLS}", f"{CALLS}{ENDCALLS}{ENDCALLS}</tool_call{TK}>"):
            await self.start(reply)
            r = await self.client.post("/v1/chat/completions", json=self.body())
            j = await r.json()
            self.assertEqual(r.status, 200, (reply, j))
            self.assertNotIn("tool_calls", j["choices"][0]["message"])
            self.assertIn(j["choices"][0]["finish_reason"], ("stop", "length"))
            await self.client.close()

    async def test_tool_choice_none_is_plain_text(self):
        reply = f"{CALLS}{call('get_weather', city='Paris')}{ENDCALLS}"
        await self.start(reply)
        j = await (await self.client.post("/v1/chat/completions", json=self.body(tool_choice="none"))).json()
        ch = j["choices"][0]
        self.assertEqual((ch["message"]["content"], ch["finish_reason"]), (reply, "stop"))
        self.assertNotIn("# Tools", self.st.vocab.decode(self.st.engine.calls[0][0]))

    async def test_bad_tool_requests_are_a_clean_400(self):
        await self.start("x")
        for extra in ({"tool_choice": "required", "tools": None}, {"tools": [{"type": "function"}]}, {"tool_choice": {"type": "function", "function": {"name": "nope"}}}):
            body = self.body(**extra)
            body = {k: v for k, v in body.items() if v is not None}
            r = await self.client.post("/v1/chat/completions", json=body)
            j = await r.json()
            self.assertEqual(r.status, 400, (extra, j))
            self.assertEqual(j["error"]["type"], "invalid_request_error")

    async def test_named_tool_choice_forces_the_call(self):
        """the model is only asked for the arguments: the prompt already ends with the call opening and the function name"""
        rest = f"<arg_key{TK}>city</arg_key{TK}><arg_value{TK}>Paris</arg_value{TK}></tool_call{TK}>{ENDCALLS}"
        for stream in (False, True):
            st = await self.start(rest)
            body = self.body(tool_choice={"type": "function", "function": {"name": "get_weather"}}, stream=stream, reasoning_effort="high")
            if stream:
                chunks = await self.sse_chunks(tool_choice=body["tool_choice"], reasoning_effort="high")
                deltas = [c["choices"][0]["delta"] for c in chunks if c["choices"]]
                tcs = [tc for d in deltas for tc in d.get("tool_calls", [])]
                self.assertEqual(tcs[0]["function"]["name"], "get_weather")
                self.assertEqual(json.loads(tcs[1]["function"]["arguments"]), {"city": "Paris"})
                self.assertEqual([c["choices"][0]["finish_reason"] for c in chunks if c["choices"]][-1], "tool_calls")
            else:
                j = await (await self.client.post("/v1/chat/completions", json=body)).json()
                ch = j["choices"][0]
                self.assertEqual(ch["finish_reason"], "tool_calls")
                self.assertIsNone(ch["message"]["content"])
                [tc] = ch["message"]["tool_calls"]
                self.assertEqual((tc["function"]["name"], json.loads(tc["function"]["arguments"])), ("get_weather", {"city": "Paris"}))
            prompt = st.vocab.decode(st.engine.calls[0][0])
            self.assertTrue(prompt.endswith(f"{UNTHINK}{CALLS}<tool_call{TK}>get_weather"), prompt[-120:])
            await self.client.close()

    async def test_required_tool_choice_forces_a_call(self):
        await self.start(f"search<arg_key{TK}>query</arg_key{TK}><arg_value{TK}>rain</arg_value{TK}></tool_call{TK}>{ENDCALLS}")
        j = await (await self.client.post("/v1/chat/completions", json=self.body(tool_choice="required", reasoning_effort="high"))).json()
        ch = j["choices"][0]
        self.assertEqual(ch["finish_reason"], "tool_calls")
        self.assertEqual(ch["message"]["tool_calls"][0]["function"]["name"], "search")
        self.assertTrue(self.st.vocab.decode(self.st.engine.calls[0][0]).endswith(f"{CALLS}<tool_call{TK}>"))

    async def test_apply_template_with_tools(self):
        await self.start("x")
        j = await (await self.client.post("/apply-template", json=self.body())).json()
        self.assertIn("# Tools", j["prompt"])

    # -- an engine error
    async def test_engine_error_after_the_stream_began_is_an_error_event(self):
        st = await self.start("hello world")

        class Boom(FakeEngine):
            def generate(self, ids, max_new, on_tokens, speculative=True, cancel=lambda: False):
                on_tokens(self.ids[:3])
                raise RuntimeError("HIP error: device lost")

        st.engine = Boom(st.vocab, "hello world")
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            r = await asyncio.wait_for(self.client.post("/v1/chat/completions", json=self.body(stream=True, tools=None)), 10)
            raw = (await asyncio.wait_for(r.read(), 10)).decode()
        self.assertEqual(r.status, 200)
        events = [e[6:] for e in raw.split("\n\n") if e.startswith("data: ")]
        self.assertEqual(events[-1], "[DONE]")
        err = json.loads(events[-2])
        self.assertEqual((err["error"]["type"], "device lost" in err["error"]["message"]), ("server_error", True))
        self.assertIn("stream failed", out.getvalue())
        st.engine = FakeEngine(st.vocab, "ok")                      # the lock was released
        r = await self.client.post("/v1/chat/completions", json=self.body(tools=None))
        self.assertEqual(r.status, 200)

    async def test_engine_error_without_a_stream_is_a_500_body(self):
        st = await self.start("hello world")

        class Boom(FakeEngine):
            def generate(self, ids, max_new, on_tokens, speculative=True, cancel=lambda: False):
                raise RuntimeError("HIP error: device lost")

        st.engine = Boom(st.vocab, "x")
        with contextlib.redirect_stderr(io.StringIO()):
            r = await asyncio.wait_for(self.client.post("/v1/chat/completions", json=self.body()), 10)
        j = await r.json()
        self.assertEqual((r.status, j["error"]["type"]), (500, "server_error"))

    # -- a client that closes the stream
    async def test_a_failed_write_cancels_the_generation_quietly(self):
        """the traceback of the report: resp.write raises ClientConnectionResetError ("Cannot write to closing transport")"""
        st = await self.start("A" * 600, delay=0.01)
        real, n = web.StreamResponse.write, {"n": 0}

        async def flaky(self, data):
            n["n"] += 1
            if n["n"] >= 4:
                raise ClientConnectionResetError("Cannot write to closing transport")
            return await real(self, data)

        out = io.StringIO()
        web.StreamResponse.write = flaky
        try:
            with contextlib.redirect_stdout(out), self.assertNoLogs("aiohttp.server", level="ERROR"):
                with contextlib.suppress(Exception):
                    r = await self.client.post("/v1/chat/completions", json=self.body(stream=True))
                    await r.read()
                self.assertTrue(await asyncio.get_running_loop().run_in_executor(None, st.engine.cancelled.wait, 5))
        finally:
            web.StreamResponse.write = real
        self.assertEqual(sum("generation cancelled" in line for line in out.getvalue().splitlines()), 1, out.getvalue())
        self.assertNotIn("Traceback", out.getvalue())
        self.assertLess(st.engine.rounds, 200)                       # 600 tokens = 200 rounds: stopped early
        st.engine.delay = 0
        r = await self.client.post("/v1/chat/completions", json=self.body())          # the lock was released: the next request is served
        self.assertEqual(r.status, 200)

    async def test_a_closed_stream_cancels_the_engine(self):
        st = await self.start("B" * 900, delay=0.02)
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertNoLogs("aiohttp.server", level="ERROR"):
            r = await self.client.post("/v1/chat/completions", json=self.body(stream=True))
            await r.content.readline()
            await r.content.readline()
            r.close()                                                # the client goes away mid-stream
            self.assertTrue(await asyncio.get_running_loop().run_in_executor(None, st.engine.cancelled.wait, 5))
            await asyncio.sleep(0.2)
        self.assertLess(st.engine.rounds, 300)
        self.assertNotIn("Traceback", out.getvalue())
        st.engine.delay = 0
        r = await self.client.post("/v1/chat/completions", json=self.body())
        self.assertEqual(r.status, 200)

    async def closed_non_stream_request(self, server):
        st = await self.start("C" * 900, delay=0.02, server=server)
        task = asyncio.ensure_future(self.client.post("/v1/chat/completions", json=self.body()))
        await asyncio.sleep(0.3)
        task.cancel()                                                # the client gives up while the server is still generating
        with contextlib.suppress(asyncio.CancelledError):
            await task
        self.assertTrue(await asyncio.get_running_loop().run_in_executor(None, st.engine.cancelled.wait, 5))
        self.assertLess(st.engine.rounds, 300)
        st.engine.delay = 0
        r = await self.client.post("/v1/chat/completions", json=self.body())
        self.assertEqual(r.status, 200)

    async def test_a_closed_non_stream_request_cancels_the_engine(self):
        """as web.run_app serves it: nothing tells the handler the client left, the connection watcher does"""
        await self.closed_non_stream_request(ProdServer)

    async def test_a_cancelled_handler_cancels_the_engine(self):
        """a server that cancels the handler when the connection drops (handler_cancellation=True)"""
        await self.closed_non_stream_request(TestServer)


if __name__ == "__main__":
    unittest.main()
