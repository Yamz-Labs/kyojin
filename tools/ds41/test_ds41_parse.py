"""DS4.1 output parsing: thinking / answer split and DSML tool calls, whole and streamed. CPU only."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ds41_parse as P  # noqa: E402
import ds41_testkit as kit  # noqa: E402
from ds41_engine import stub_reply  # noqa: E402

EOS = P.EOS


def norm(message: dict) -> dict:
    """A parsed message without the random call ids and the truncation flag."""
    m = {k: v for k, v in message.items() if k != "truncated"}
    m["tool_calls"] = [{**{k: v for k, v in c.items() if k != "id"}} for c in m["tool_calls"]]
    return m


def run_stream(text: str, cuts: list[int], thinking=True, parse_tools=True):
    """Feed `text` cut at the given offsets; returns (events, parser)."""
    parser = P.StreamParser(thinking, parse_tools)
    events, last = [], 0
    for cut in cuts + [len(text)]:
        events += parser.feed(text[last:cut])
        last = cut
    events += parser.finish()
    return events, parser


def streamed(text: str, cuts: list[int], thinking=True, parse_tools=True) -> dict:
    events, parser = run_stream(text, cuts, thinking, parse_tools)
    out = P.split_events(events)
    return {**norm(out), "truncated": parser.truncated}


# (name, thinking, text, expected message)
CALL1 = ('<｜DSML｜ calls>\n<｜DSML｜ invoke name="lookup">\n<｜DSML｜ parameter name="query" string="true">cats</｜DSML｜ parameter>\n'
         '<｜DSML｜ parameter name="limit" string="false">3</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>')


def call(name, args, ns=None):
    c = {"type": "function", "function": {"name": name, "arguments": args}}
    if ns:
        c["namespace"] = ns
    return c


FIXTURES = [
    ("plain_thinking", True, "I think.</think>The answer." + EOS,
     {"reasoning_content": "I think.", "content": "The answer.", "tool_calls": []}),
    ("plain_chat", False, "Just an answer." + EOS, {"reasoning_content": "", "content": "Just an answer.", "tool_calls": []}),
    ("no_eos", True, "r</think>c", {"reasoning_content": "r", "content": "c", "tool_calls": []}),
    ("spaces_kept", True, "  reason  </think>  summary  ", {"reasoning_content": "  reason  ", "content": "  summary  ", "tool_calls": []}),
    ("empty_reasoning", True, "</think>hi", {"reasoning_content": "", "content": "hi", "tool_calls": []}),
    ("trailing_newlines_kept", True, "r\n\n</think>a\n\nb\n\n", {"reasoning_content": "r\n\n", "content": "a\n\nb\n\n", "tool_calls": []}),
    ("one_call", True, "need data</think>Checking.\n\n" + CALL1 + EOS,
     {"reasoning_content": "need data", "content": "Checking.",
      "tool_calls": [call("lookup", '{"query": "cats", "limit": 3}')]}),
    ("call_no_content", True, "r</think>\n\n" + CALL1 + EOS,
     {"reasoning_content": "r", "content": "", "tool_calls": [call("lookup", '{"query": "cats", "limit": 3}')]}),
    ("call_in_chat_mode", False, "Sure.\n\n" + CALL1 + EOS,
     {"reasoning_content": "", "content": "Sure.", "tool_calls": [call("lookup", '{"query": "cats", "limit": 3}')]}),
    ("two_calls_namespace", True, "r</think>ok" + stub_reply(calls=[("a::f", {"x": 1}), ("g", {"y": "z"})]) + EOS,
     {"reasoning_content": "r", "content": "ok", "tool_calls": [call("f", '{"x": 1}', "a"), call("g", '{"y": "z"}')]}),
    ("no_params", True, "r</think>" + stub_reply(calls=[("ping", {})]) + EOS,
     {"reasoning_content": "r", "content": "", "tool_calls": [call("ping", "{}")]}),
    ("string_escapes", True, "r</think>" + stub_reply(calls=[("w", {"t": 'a "q" \\ b\nc\t<d> ça 日本'})]) + EOS,
     {"reasoning_content": "r", "content": "", "tool_calls": [call("w", json.dumps({"t": 'a "q" \\ b\nc\t<d> ça 日本'}, ensure_ascii=False))]}),
    ("json_values", True, "r</think>" + stub_reply(calls=[("w", {"a": [1, {"b": None}], "c": {"d": True}, "e": 2.5, "f": "7"})]) + EOS,
     {"reasoning_content": "r", "content": "", "tool_calls": [call(
         "w", '{"a": [1, {"b": null}], "c": {"d": true}, "e": 2.5, "f": "7"}')]}),
    ("value_looks_like_markup", True, "r</think>" + stub_reply(calls=[("w", {"t": "x </｜DSML｜ invok y <｜DSML｜ calls> z"})]) + EOS,
     {"reasoning_content": "r", "content": "", "tool_calls": [call("w", json.dumps({"t": "x </｜DSML｜ invok y <｜DSML｜ calls> z"}, ensure_ascii=False))]}),
    ("empty_string_param", True, "r</think>" + stub_reply(calls=[("w", {"t": "", "u": "v"})]) + EOS,
     {"reasoning_content": "r", "content": "", "tool_calls": [call("w", '{"t": "", "u": "v"}')]}),
    ("think_word_in_answer", False, "the tag <b>think</b> and < think > are text" + EOS,
     {"reasoning_content": "", "content": "the tag <b>think</b> and < think > are text", "tool_calls": []}),
    ("text_after_calls", True, "r</think>a\n\n" + CALL1 + "\n" + EOS,
     {"reasoning_content": "r", "content": "a", "tool_calls": [call("lookup", '{"query": "cats", "limit": 3}')]}),
]
IDS = [f[0] for f in FIXTURES]


@pytest.mark.parametrize("name,thinking,text,want", FIXTURES, ids=IDS)
def test_whole_parse(name, thinking, text, want):
    got = P.parse_completion(text, thinking)
    assert norm(got) == want
    assert got["truncated"] is False


@pytest.mark.parametrize("name,thinking,text,want", FIXTURES, ids=IDS)
def test_every_two_chunk_split_gives_the_same_message(name, thinking, text, want):
    for cut in range(len(text) + 1):
        assert streamed(text, [cut], thinking) == {**want, "truncated": False}, (name, cut, text[max(0, cut - 20):cut + 20])


@pytest.mark.parametrize("name,thinking,text,want", FIXTURES, ids=IDS)
def test_every_three_chunk_split_gives_the_same_message(name, thinking, text, want):
    step = max(1, len(text) // 40)                       # all pairs of cuts on a coarse grid, plus all cuts near markup
    cuts = sorted(set(range(0, len(text) + 1, step)) | {i for i, ch in enumerate(text) if ch in "<>/｜\n\"="})
    for a in cuts[::2]:
        for b in cuts[1::3]:
            if a <= b:
                assert streamed(text, [a, b], thinking) == {**want, "truncated": False}, (name, a, b)


@pytest.mark.parametrize("size", [1, 2, 3, 5, 7])
@pytest.mark.parametrize("name,thinking,text,want", FIXTURES, ids=IDS)
def test_fixed_size_chunks(name, thinking, text, want, size):
    assert streamed(text, list(range(size, len(text), size)), thinking) == {**want, "truncated": False}


@pytest.mark.parametrize("name,thinking,text,want", FIXTURES, ids=IDS)
def test_random_chunking(name, thinking, text, want):
    rng = random.Random(7)
    for _ in range(25):
        cuts = sorted(rng.sample(range(len(text) + 1), min(len(text) + 1, rng.randint(1, 9))))
        assert streamed(text, cuts, thinking) == {**want, "truncated": False}, (name, cuts)


@pytest.mark.parametrize("name,thinking,text,want", [f for f in FIXTURES if f[0] not in ("no_eos", "trailing_newlines_kept",
                                                                                       "spaces_kept", "text_after_calls", "call_no_content",
                                                                                       "empty_reasoning", "plain_chat", "think_word_in_answer",
                                                                                       "call_in_chat_mode")],
                         ids=lambda x: x if isinstance(x, str) else "")
def test_agrees_with_the_official_parser(name, thinking, text, want):
    ref = kit.official().parse_message_from_completion_text(text, "thinking" if thinking else "chat")
    got = P.parse_completion(text, thinking)
    assert (got["content"], got["reasoning_content"]) == (ref["content"], ref["reasoning_content"])
    assert [(c["function"]["name"], c["function"]["arguments"], c.get("namespace")) for c in got["tool_calls"]] == \
        [(c["function"]["name"], c["function"]["arguments"], c.get("namespace")) for c in ref["tool_calls"]]


# ---- streaming behaviour ----------------------------------------------------------------------------


def test_text_streams_while_it_is_decoded():
    events, _ = run_stream("abcdefghij</think>klmnopqrst", list(range(1, 27)))
    kinds = [e[0] for e in events]
    assert kinds.count("reasoning") > 3 and kinds.count("content") > 3                # many small deltas, not one blob
    assert "".join(e[1] for e in events if e[0] == "reasoning") == "abcdefghij"


def test_only_a_possible_marker_prefix_is_held_back():
    p = P.StreamParser(thinking=False)
    assert p.feed("Hello wor") == [("content", "Hello wor")]
    assert p.feed("ld\n") == [("content", "ld")]                  # one "\n" might start the "\n\n<｜DSML｜ calls>" lead
    assert p.feed("\n") == []
    assert p.feed("<｜DS") == []
    assert p.feed("x") == [("content", "\n\n<｜DSx")]            # not a marker after all: released unchanged
    assert p.finish() == []


def test_think_close_is_held_until_complete():
    p = P.StreamParser(thinking=True)
    assert p.feed("why</thi") == [("reasoning", "why")]
    assert p.feed("nk>yes") == [("content", "yes")]


def test_tool_call_starts_before_its_arguments_end():
    text = "r</think>" + stub_reply(calls=[("lookup", {"query": "x" * 30, "n": 5})])
    parser = P.StreamParser()
    seen = []
    for i, ch in enumerate(text):
        for ev in parser.feed(ch):
            seen.append((i, ev))
    start = next(i for i, ev in seen if ev[0] == "tool_start")
    end = next(i for i, ev in seen if ev[0] == "tool_end")
    first_args = next(i for i, ev in seen if ev[0] == "tool_args")
    assert start <= first_args < end
    assert text.index('name="lookup">') + len('name="lookup">') - 1 <= start < text.index("query")
    value_pieces = [ev for _, ev in seen if ev[0] == "tool_args" and "x" in ev[2]]
    assert len(value_pieces) > 5                                   # the string value streams, not one piece at the end
    start_event = next(ev for _, ev in seen if ev[0] == "tool_start")
    assert start_event[1] == 0 and start_event[3] == "lookup" and start_event[2].startswith("call_")


def test_argument_pieces_add_up_to_the_official_json():
    args = {"query": "a\"b\nc", "limit": 3, "nested": {"k": [1, "2"]}, "flag": False}
    text = stub_reply(calls=[("lookup", args)])
    ref = kit.official().parse_message_from_completion_text("r</think>" + text + EOS, "thinking")
    events, _ = run_stream("r</think>" + text, list(range(1, len(text) + 9, 2)))
    joined = "".join(e[2] for e in events if e[0] == "tool_args")
    assert joined == ref["tool_calls"][0]["function"]["arguments"]
    assert json.loads(joined) == args


def test_indexes_and_ids_of_several_calls():
    text = "r</think>" + stub_reply(calls=[("a", {"x": 1}), ("b", {}), ("c", {"y": "z"})])
    events, parser = run_stream(text, list(range(0, len(text), 4)))
    starts = [e for e in events if e[0] == "tool_start"]
    assert [e[1] for e in starts] == [0, 1, 2] and [e[3] for e in starts] == ["a", "b", "c"]
    assert len({e[2] for e in starts}) == 3 and parser.tool_count == 3
    assert [e[1] for e in events if e[0] == "tool_end"] == [0, 1, 2]


# ---- malformed and cut-off replies: never an exception, always valid JSON ---------------------------


@pytest.mark.parametrize("name,thinking,text,want", [f for f in FIXTURES if "call" in f[0] or f[0] in ("json_values", "string_escapes")],
                         ids=lambda x: x if isinstance(x, str) else "")
def test_every_prefix_parses_to_valid_json_calls(name, thinking, text, want):
    for cut in range(len(text) + 1):
        prefix = text[:cut]
        whole = P.parse_completion(prefix, thinking)
        for c in whole["tool_calls"]:
            assert isinstance(json.loads(c["function"]["arguments"]), dict), (cut, prefix[-40:])
        # chunked feeding of the same prefix agrees with feeding it at once
        assert streamed(prefix, list(range(3, cut, 3)), thinking) == norm(whole) | {"truncated": whole["truncated"]}, cut


def test_cut_inside_a_call_flags_truncation_and_closes_the_json():
    text = "r</think>\n\n" + CALL1
    cut = text.index("</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"limit\"")
    got = P.parse_completion(text[:cut + len("</｜DSML｜ parameter>\n")], True)
    assert got["truncated"] is True
    assert got["tool_calls"][0]["function"]["arguments"] == '{"query": "cats"}'
    got = P.parse_completion(text[:text.index("cats") + 2], True)
    assert got["truncated"] is True and json.loads(got["tool_calls"][0]["function"]["arguments"]) == {"query": "ca"}
    assert P.parse_completion(text, True)["truncated"] is False


def test_cut_before_the_name_closes_drops_the_call():
    text = "r</think>ok\n\n" + CALL1
    got = P.parse_completion(text[:text.index('invoke name="lookup"') + 8], True)
    assert got["tool_calls"] == [] and got["content"] == "ok" and got["truncated"] is True


def test_reasoning_cut_short_is_all_reasoning():
    got = P.parse_completion("still thinking about", True)
    assert (got["reasoning_content"], got["content"], got["tool_calls"]) == ("still thinking about", "", [])


def test_tool_call_inside_unclosed_reasoning_is_recovered():
    got = P.parse_completion("hmm\n\n" + CALL1 + EOS, True)
    assert got["reasoning_content"] == "hmm" and len(got["tool_calls"]) == 1


def test_non_json_value_with_string_false_becomes_a_string():
    text = '\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="f">\n<｜DSML｜ parameter name="n" string="false">not json</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>'
    for cuts in ([], list(range(1, len(text), 2))):
        got = streamed(text, cuts, thinking=False)
        assert json.loads(got["tool_calls"][0]["function"]["arguments"]) == {"n": "not json"}


def test_text_where_a_tag_belongs_ends_the_markup():
    text = 'ok\n\n<｜DSML｜ calls>\nsurprise text instead of an invoke tag'
    got = P.parse_completion(text, False)
    assert got["tool_calls"] == [] and got["content"].startswith("ok") and "surprise text" in got["content"]
    text = ('ok\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="f">\n<｜DSML｜ parameter name="a" string="true">1</｜DSML｜ parameter>\n'
            'junk here\n')
    got = P.parse_completion(text, False)
    assert json.loads(got["tool_calls"][0]["function"]["arguments"]) == {"a": "1"} and "junk here" in got["content"]


def test_parse_tools_off_keeps_the_markup_as_text():
    text = "answer\n\n" + CALL1 + EOS
    got = P.parse_completion(text, False, parse_tools=False)
    assert got["tool_calls"] == [] and got["content"] == "answer\n\n" + CALL1
    assert streamed(text, list(range(0, len(text), 5)), False, parse_tools=False)["content"] == got["content"]


def test_eos_in_the_text_ends_the_reply():
    got = P.parse_completion("r</think>a" + EOS + "ignored", True)
    assert got["content"] == "a"


def test_random_garbage_never_raises():
    rng = random.Random(3)
    alphabet = ["<", ">", "/", "｜DSML｜", " calls", " invoke", " parameter", ' name="x"', ' string="true"', "\n", "\n\n", "</think>",
                "text", "{", "}", '"', EOS, "<think>"]
    for _ in range(400):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
        thinking = rng.random() < 0.5
        whole = P.parse_completion(text, thinking)
        for c in whole["tool_calls"]:
            json.loads(c["function"]["arguments"])
        cuts = sorted(rng.sample(range(len(text) + 1), min(len(text) + 1, rng.randint(1, 5))))
        assert streamed(text, cuts, thinking) == norm(whole) | {"truncated": whole["truncated"]}, text
