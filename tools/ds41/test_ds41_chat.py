"""DS4.1 prompt rendering: golden tests against the official encoding, request mapping, round trip. CPU only."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ds41_chat as chat  # noqa: E402
import ds41_engine  # noqa: E402
import ds41_parse  # noqa: E402
import ds41_testkit as kit  # noqa: E402

REF_TESTS = Path(__file__).resolve().parent / "testdata" / "reference" / "tests"


def tool(name="lookup", desc="Look up a value", props=None, **extra):
    t = {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props or {"query": {"type": "string"}, "limit": {"type": "integer"}}}}}
    t.update(extra)
    return t


def call(cid, name, args, **extra):
    c = {"id": cid, "type": "function", "function": {"name": name, "arguments": args}}
    c.update(extra)
    return c


U, A, S = (lambda c: {"role": "user", "content": c}), (lambda c, **k: {"role": "assistant", "content": c, **k}), \
    (lambda c: {"role": "system", "content": c})
T = lambda cid, c: {"role": "tool", "tool_call_id": cid, "content": c}  # noqa: E731

# (id, messages, tools). Each one is rendered by the reference and by the port in thinking and chat mode.
CONVERSATIONS: list[tuple[str, list, list | None]] = [
    ("user_only", [U("Hello")], None),
    ("system_user", [S("You are a terse assistant."), U("What is 2+2?")], None),
    ("multi_turn", [S("Be brief."), U("Hi"), A("Hello!"), U("Capital of France?")], None),
    ("multi_turn_reasoning", [U("Hi"), A("Hello!", reasoning_content="greeting, answer shortly"), U("And 3+3?"),
                              A("6", reasoning_content="3+3=6"), U("Thanks")], None),
    ("cjk_emoji", [S("你是一个助手。"), U("日本語で答えて 🙂 café naïve — “quotes”")], None),
    ("newlines_specials", [U("line1\n\nline2\n\n\n<b>&amp;</b> {braces} $x$ \\ backslash \"q\"")], None),
    ("one_tool_no_system", [U("Find cats")], [tool()]),
    ("tools_with_system", [S("Use tools when needed."), U("Find cats")], [tool(), tool("weather", "Get weather", {
        "city": {"type": "string"}, "days": {"type": "integer"}})]),
    ("tool_round_trip", [S("sys"), U("Find cats"),
                         A("", reasoning_content="I should search", tool_calls=[
                             call("c1", "lookup", '{"query": "cats", "limit": 3}')]),
                         T("c1", "cat1, cat2, cat3"), A("Found three.", reasoning_content="done"), U("Another?")],
     [tool()]),
    ("parallel_calls_reversed_results", [U("Weather in two cities"),
                                         A("", reasoning_content="two lookups", tool_calls=[
                                             call("a", "lookup", '{"query": "paris"}'),
                                             call("b", "lookup", '{"query": "rome"}')]),
                                         T("b", "rome: sun"), T("a", "paris: rain")], [tool()]),
    ("dict_arguments", [U("go"), A(None, tool_calls=[call("c1", "lookup", {"query": "dogs", "limit": 2})]),
                        T("c1", "ok")], [tool()]),
    ("nested_json_args", [U("go"), A("Searching", tool_calls=[call(
        "c1", "lookup", json.dumps({"query": "ça", "filters": {"tags": ["a", "b"], "n": 1.5}, "flag": True, "none": None},
                                    ensure_ascii=False))]), T("c1", '{"hits": []}')], [tool()]),
    ("no_args_call", [U("tick"), A("", tool_calls=[call("c1", "lookup", "{}")]), T("c1", "tock")], [tool()]),
    ("content_and_calls", [U("go"), A("Let me check.", tool_calls=[call("c1", "lookup", '{"query": "x"}')]),
                           T("c1", "found x"), A("It is x."), U("ok")], [tool()]),
    ("content_none_with_calls", [U("go"), A(None, tool_calls=[call("c1", "lookup", '{"query": "x"}')]), T("c1", "r")],
     [tool()]),
    ("tool_result_parts", [U("go"), A("", tool_calls=[call("c1", "lookup", '{"query": "x"}')]),
                           {"role": "tool", "tool_call_id": "c1", "content": [
                               {"type": "text", "text": "part one"}, {"type": "text", "text": "part two"}]}], [tool()]),
    ("user_parts", [{"role": "user", "content": [{"type": "text", "text": "alpha"}, {"type": "text", "text": "beta"}]}],
     None),
    ("developer_role", [{"role": "developer", "content": "Be formal."}, U("hey")], None),
    ("mid_system", [S("base"), U("one"), A("two"), S("new rule: shout"), U("three")], None),
    ("consecutive_users", [U("first"), U("second"), A("ack"), U("third")], None),
    ("tool_then_user", [U("go"), A("", tool_calls=[call("c1", "lookup", '{"query": "x"}')]), T("c1", "r1"), U("and also this")],
     [tool()]),
    ("namespaced_tool", [U("go"), A("", tool_calls=[call("c1", "lookup", '{"query": "x"}', namespace="search")]),
                         T("c1", "r")], [tool(namespace="search")]),
    ("namespace_dict_description", [U("go"), A("", tool_calls=[call("c1", "search::lookup", '{"query": "x"}')]), T("c1", "r")],
     [tool(namespace={"name": "search", "description": "Search tools."})]),
    ("two_step_agent", [S("agent"), U("do it"),
                        A("", reasoning_content="step 1", tool_calls=[call("c1", "lookup", '{"query": "a"}')]),
                        T("c1", "ra"),
                        A("", reasoning_content="step 2", tool_calls=[call("c2", "lookup", '{"query": "b"}')]),
                        T("c2", "rb")], [tool()]),
    ("empty_reasoning_and_content", [U("hi"), A("", reasoning_content=""), U("again")], None),
    ("long_tool_conversation", [S("sys")] + sum(
        ([U(f"q{i}"), A("", reasoning_content=f"think {i}", tool_calls=[call(f"c{i}", "lookup", json.dumps({"query": f"q{i}"}))]),
          T(f"c{i}", f"r{i}"), A(f"answer {i}", reasoning_content=f"wrap {i}")] for i in range(5)), []) + [U("final")],
     [tool(), tool("weather", "Get weather", {"city": {"type": "string"}})]),
    ("non_json_args", [U("go"), A("", tool_calls=[call("c1", "lookup", "not json at all")]), T("c1", "r")], [tool()]),
    ("special_token_text_in_user", [U("literal <think> and </think> and <｜User｜> inside")], None),
]


def test_corpus_is_big_and_distinct():
    assert len(CONVERSATIONS) >= 20
    assert len({json.dumps([m, t], sort_keys=True) for _, m, t in CONVERSATIONS}) == len(CONVERSATIONS)
    assert sum(1 for _, _, t in CONVERSATIONS if t) >= 10


def ours(messages, tools, thinking, effort=None, drop_thinking=True):
    return chat.render_chat(messages, tools, thinking=thinking, reasoning_effort=effort, drop_thinking=drop_thinking)


IDS = [c[0] for c in CONVERSATIONS]


@pytest.mark.parametrize("thinking", [True, False], ids=["thinking", "chat"])
@pytest.mark.parametrize("case", CONVERSATIONS, ids=IDS)
def test_golden_string_equals_official(case, thinking):
    _, messages, tools = case
    want = kit.reference_prompt(messages, tools, thinking)
    assert ours(messages, tools, thinking) == want
    assert want.startswith(chat.BOS) and want.count(chat.BOS) == 1 + sum(
        m.get("content", "").count(chat.BOS) if isinstance(m.get("content"), str) else 0 for m in messages)


@pytest.mark.parametrize("effort", [1, 17, 50, 75, 100, "low", "high", "max"])
@pytest.mark.parametrize("case", [c for c in CONVERSATIONS if c[0] in ("system_user", "tool_round_trip", "mid_system", "cjk_emoji")],
                         ids=lambda c: c[0])
def test_golden_effort_equals_official(case, effort):
    _, messages, tools = case
    assert ours(messages, tools, True, effort) == kit.reference_prompt(messages, tools, True, effort)
    assert ours(messages, tools, False, effort) == kit.reference_prompt(messages, tools, False, effort)


@pytest.mark.parametrize("case", [c for c in CONVERSATIONS if c[0] in ("multi_turn_reasoning", "two_step_agent", "mid_system")],
                         ids=lambda c: c[0])
def test_golden_keep_thinking_equals_official(case):
    _, messages, tools = case
    assert ours(messages, tools, True, drop_thinking=False) == kit.reference_prompt(messages, tools, True, drop_thinking=False)


def test_drop_thinking_changes_the_prompt_only_without_tools():
    msgs = CONVERSATIONS[3][1]
    assert ours(msgs, None, True) != ours(msgs, None, True, drop_thinking=False)
    assert "greeting, answer shortly" not in ours(msgs, None, True)
    assert "greeting, answer shortly" in ours(msgs, None, True, drop_thinking=False)
    _, msgs, tools = CONVERSATIONS[8]
    assert ours(msgs, tools, True) == ours(msgs, tools, True, drop_thinking=False)    # tools keep every reasoning block


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_official_fixture_outputs(n):
    """The five input/expected-output pairs shipped with the official encoding."""
    data = json.loads((REF_TESTS / f"test_input_{n}.json").read_text())
    expected = (REF_TESTS / f"test_output_{n}.txt").read_text()
    case = data if isinstance(data, dict) else {"messages": data}
    # the official fixtures use `latest_reminder` roles (n = 3, 4), a role this server does not take
    if any(m["role"] == "latest_reminder" for m in case["messages"]):
        pytest.skip("latest_reminder is not part of the OpenAI API")
    mode = case.get("thinking_mode") or "chat"          # the official golden test encodes in chat mode by default
    want = kit.official().encode_messages(kit.reference_messages(case["messages"], case.get("tools")), mode,
                                          reasoning_effort=case.get("reasoning_effort"))
    assert want == expected
    messages = case["messages"]
    if any(isinstance(m["content"], list) and any(p.get("type") == "image_url" for p in m["content"]) for m in messages):
        with pytest.raises(chat.ChatFormatError, match="image"):     # text-only server: refused, not silently dropped
            chat.render_chat(messages, case.get("tools"), thinking=mode == "thinking")
        return
    if messages[-1]["role"] == "assistant":             # this server renders up to the generation prompt
        got = chat.render_chat(messages[:-1], case.get("tools"), thinking=mode == "thinking",
                               reasoning_effort=case.get("reasoning_effort"))
        assert expected.startswith(got)
    else:
        assert chat.render_chat(messages, case.get("tools"), thinking=mode == "thinking",
                                reasoning_effort=case.get("reasoning_effort")) == expected


# ---- token ids -----------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def hf_tokenizer():
    path = kit.tokenizer_json()
    if path is None:
        pytest.skip("tokenizer.json not available (set DS41_REF_DIR, run tools/ds41/fetch_reference.py, or allow one download)")
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(path)), ds41_engine.HFTokenizer(path)


@pytest.mark.parametrize("thinking", [True, False], ids=["thinking", "chat"])
@pytest.mark.parametrize("case", CONVERSATIONS, ids=IDS)
def test_golden_token_ids_equal_official(case, thinking, hf_tokenizer):
    """Token ids of the server's path (StubEngine over the model's tokenizer) equal the ids of the official string."""
    raw, ours_tk = hf_tokenizer
    _, messages, tools = case
    want_ids = raw.encode(kit.reference_prompt(messages, tools, thinking), add_special_tokens=False).ids
    engine = ds41_engine.StubEngine("", tokenizer=ours_tk)
    prompt = ours(messages, tools, thinking)
    got_ids = list(engine.prompt_ids(prompt))
    assert got_ids == want_ids
    assert got_ids[0] == ours_tk.bos_id and got_ids.count(ours_tk.bos_id) == 1
    assert got_ids[-1] == raw.token_to_id("<think>" if thinking else "</think>")


def test_dsml_and_think_are_single_special_tokens(hf_tokenizer):
    raw, _ = hf_tokenizer
    for tok in ("<think>", "</think>", "｜DSML｜", "<｜User｜>", "<｜Assistant｜>", "<｜System｜>", chat.EOS):
        assert raw.token_to_id(tok) is not None, tok
        assert len(raw.encode(tok, add_special_tokens=False).ids) == 1, tok


# ---- request mapping -----------------------------------------------------------------------------------

OPTIONS = [
    ({}, chat.ChatOptions(True, 75, True)),
    ({"reasoning_effort": "low"}, chat.ChatOptions(True, 50, True)),
    ({"reasoning_effort": "minimal"}, chat.ChatOptions(True, 25, True)),
    ({"reasoning_effort": "medium"}, chat.ChatOptions(True, 62, True)),
    ({"reasoning_effort": "high"}, chat.ChatOptions(True, 75, True)),
    ({"reasoning_effort": "xhigh"}, chat.ChatOptions(True, 100, True)),
    ({"reasoning_effort": "max"}, chat.ChatOptions(True, 100, True)),
    ({"reasoning_effort": 33}, chat.ChatOptions(True, 33, True)),
    ({"reasoning_effort": "none"}, chat.ChatOptions(False, 75, True)),
    ({"reasoning_effort": None}, chat.ChatOptions(True, 75, True)),
    ({"thinking": False}, chat.ChatOptions(False, 75, True)),
    ({"thinking": {"type": "disabled"}}, chat.ChatOptions(False, 75, True)),
    ({"thinking": {"type": "enabled"}, "reasoning_effort": "low"}, chat.ChatOptions(True, 50, True)),
    ({"enable_thinking": False}, chat.ChatOptions(False, 75, True)),
    ({"thinking_mode": "chat"}, chat.ChatOptions(False, 75, True)),
    ({"chat_template_kwargs": {"thinking": False}}, chat.ChatOptions(False, 75, True)),
    ({"chat_template_kwargs": {"enable_thinking": True}, "reasoning_effort": "none"}, chat.ChatOptions(True, 75, True)),
    ({"chat_template_kwargs": {"thinking_mode": "thinking", "reasoning_effort": 90}}, chat.ChatOptions(True, 90, True)),
    ({"chat_template_kwargs": {"reasoning_effort": "max"}, "reasoning_effort": "low"}, chat.ChatOptions(True, 100, True)),
    ({"chat_template_kwargs": {"thinking": True}, "thinking": False}, chat.ChatOptions(True, 75, True)),
    ({"chat_template_kwargs": {"drop_thinking": False}}, chat.ChatOptions(True, 75, False)),
    ({"drop_thinking": False}, chat.ChatOptions(True, 75, False)),
    ({"chat_template_kwargs": {"reasoning_effort": "none"}, "reasoning_effort": "max"}, chat.ChatOptions(False, 75, True)),
]


@pytest.mark.parametrize("body,want", OPTIONS, ids=[json.dumps(b, ensure_ascii=False)[:70] for b, _ in OPTIONS])
def test_chat_options_mapping(body, want):
    assert chat.chat_options(body) == want


def test_chat_options_server_defaults():
    assert chat.chat_options({}, default_thinking=False) == chat.ChatOptions(False, 75, True)
    assert chat.chat_options({"thinking": True}, default_thinking=False) == chat.ChatOptions(True, 75, True)
    assert chat.chat_options({}, default_effort="max").effort == 100
    assert chat.chat_options({"reasoning_effort": "none"}, default_effort=40) == chat.ChatOptions(False, 40, True)


@pytest.mark.parametrize("body", [
    {"reasoning_effort": "turbo"}, {"reasoning_effort": 0}, {"reasoning_effort": 101}, {"reasoning_effort": True},
    {"reasoning_effort": 5.5}, {"thinking": "yes"}, {"thinking": {"type": "maybe"}}, {"thinking_mode": "fast"},
    {"chat_template_kwargs": [1]}, {"drop_thinking": "no"}], ids=lambda b: json.dumps(b))
def test_chat_options_reject_bad_values(body):
    with pytest.raises(chat.ChatFormatError):
        chat.chat_options(body)


def test_effort_levels_are_monotone_and_inside_the_range():
    order = ["minimal", "low", "medium", "high", "xhigh", "max"]
    budgets = [chat.resolve_effort(x) for x in order]
    assert budgets == sorted(budgets) and all(1 <= b <= 100 for b in budgets)
    assert (chat.resolve_effort("low"), chat.resolve_effort("high"), chat.resolve_effort("max")) == (50, 75, 100)   # official
    assert kit.official().REASONING_EFFORT_MAPPINGS == {"low": 50, "high": 75, "max": 100}


def test_effort_prefix_text_and_position():
    p = ours([S("sys"), U("a"), A("b"), U("c")], None, True, 80)
    assert p.startswith(f"{chat.BOS}<｜System｜>Reasoning Effort: 80 (range 1-100, the higher the value, the more thorough the reasoning)\n\nsys<｜User｜>a")
    assert p.count("Reasoning Effort") == 1                      # first message only
    assert "Reasoning Effort" not in ours([S("sys"), U("a")], None, False)
    assert ours([U("a")], None, True).startswith(f"{chat.BOS}<｜System｜>Reasoning Effort: 75 ")
    assert ours([U("a")], None, False).startswith(f"{chat.BOS}<｜User｜>a")


def test_generation_prompt_ends_open_or_closed():
    assert ours([U("a")], None, True).endswith("<｜Assistant｜><think>")
    assert ours([U("a")], None, False).endswith("<｜Assistant｜></think>")
    assert ours([U("a"), A("b"), U("c")], None, True, drop_thinking=False).endswith("<｜Assistant｜><think>")


# ---- the OpenAI layer ----------------------------------------------------------------------------------


def test_tools_without_system_get_an_empty_system_message():
    msgs = chat.normalize_messages([U("hi")], [tool()])
    assert [m["role"] for m in msgs] == ["system", "user"] and msgs[0]["content"] == "" and msgs[0]["tools"]
    prompt = ours([U("hi")], [tool()], False)
    assert prompt.startswith(f"{chat.BOS}<｜System｜>\n\n## Tools\n\n")
    assert '{"name": "lookup", "description": "Look up a value", "parameters"' in prompt
    msgs = chat.normalize_messages([S("s"), U("hi")], [tool()])
    assert [m["role"] for m in msgs] == ["system", "user"] and msgs[0]["content"] == "s"


def test_flat_tool_form_is_accepted():
    flat = {"type": "function", "name": "lookup", "description": "Look up a value", "parameters": tool()["function"]["parameters"]}
    assert ours([U("hi")], [flat], True) == ours([U("hi")], [tool()], True)


def test_input_messages_are_not_modified():
    msgs, tools = copy.deepcopy(CONVERSATIONS[8][1]), copy.deepcopy(CONVERSATIONS[8][2])
    snap = copy.deepcopy((msgs, tools))
    ours(msgs, tools, True)
    assert (msgs, tools) == snap


@pytest.mark.parametrize("messages,tools,fragment", [
    ([], None, "non-empty"),
    ([A("only assistant")], None, "last message"),
    ([U("a"), A("b")], None, "last message"),
    ([S("only system")], None, "last message"),
    ([{"role": "function", "content": "x"}], None, "role"),
    ([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]}], None, "image"),
    ([{"role": "user", "content": [{"type": "audio", "x": 1}]}], None, "unsupported content part"),
    ([{"role": "user", "content": 5}], None, "string or a list"),
    ([U("see <｜deepseek_image｜> here")], None, "reserved"),
    ([U("a")], [{"type": "function"}], "name"),
    ([U("a"), A("", tool_calls=[{"id": "x"}]), T("x", "r")], None, "function"),
    ([U("a"), A("", tool_calls=[call("x", "a::b::c", "{}")]), T("x", "r")], None, "::"),
], ids=lambda x: x if isinstance(x, str) else "")
def test_bad_requests_raise_chat_format_error(messages, tools, fragment):
    with pytest.raises(chat.ChatFormatError) as err:
        chat.render_chat(messages, tools)
    assert fragment in str(err.value)


def test_bad_effort_raises_before_rendering():
    with pytest.raises(chat.ChatFormatError):
        chat.render_chat([U("a")], None, reasoning_effort="turbo")


def test_tool_results_follow_call_order_not_arrival_order():
    p = ours(CONVERSATIONS[9][1], CONVERSATIONS[9][2], True)
    assert p.index("paris: rain") < p.index("rome: sun")
    assert "</tool_result>\n\n<tool_result>" in p


# ---- round trip ----------------------------------------------------------------------------------------

REPLIES = [
    ("plain", ds41_engine.stub_reply("The answer is 4.", reasoning="2+2 is 4")),
    ("one_call", ds41_engine.stub_reply("Checking.", reasoning="need data", calls=[("lookup", {"query": "cats", "limit": 3})])),
    ("two_calls", ds41_engine.stub_reply("", reasoning="two things", calls=[
        ("lookup", {"query": "a\nb \"q\" ça"}), ("weather", {"city": "Paris", "days": 2, "ok": True, "tags": ["x", "y"]})])),
    ("call_no_args", ds41_engine.stub_reply("", reasoning="r", calls=[("lookup", {})])),
]


@pytest.mark.parametrize("name,reply", REPLIES, ids=[r[0] for r in REPLIES])
def test_parse_then_render_reproduces_the_reply(name, reply):
    """A model reply, parsed into an assistant message and rendered back as history, gives the original text."""
    parsed = ds41_parse.parse_completion(reply, thinking=True)
    assistant = {"role": "assistant", "content": parsed["content"], "reasoning_content": parsed["reasoning_content"]}
    if parsed["tool_calls"]:
        assistant["tool_calls"] = parsed["tool_calls"]
    messages = [U("go"), assistant]
    if parsed["tool_calls"]:
        messages += [T(c["id"], "result") for c in parsed["tool_calls"]]
    else:
        messages += [U("next")]
    tools = [tool(), tool("weather", "Get weather", {"city": {"type": "string"}})]
    prompt = ours(messages, tools, True)
    head = "<｜Assistant｜><think>"
    seg = prompt[prompt.index(head) + len(head):]
    assert seg.startswith(reply + chat.EOS), (seg[:200], reply[:200])
    assert prompt == kit.reference_prompt(messages, tools, True)
    # and the official parser reads the reply the same way
    want = kit.official().parse_message_from_completion_text(reply + chat.EOS, "thinking")
    assert (want["content"], want["reasoning_content"]) == (parsed["content"], parsed["reasoning_content"])
    assert [(c["function"]["name"], c["function"]["arguments"]) for c in want["tool_calls"]] == \
        [(c["function"]["name"], c["function"]["arguments"]) for c in parsed["tool_calls"]]


def test_chat_options_reasoning_object():
    """Hermes sends reasoning={"enabled", "effort"}; chat_template_kwargs and the top-level fields win over it."""
    assert chat.chat_options({"reasoning": {"enabled": True, "effort": "medium"}}) == chat.ChatOptions(True, 62, True)
    assert chat.chat_options({"reasoning": {"enabled": False, "effort": "none"}}).thinking is False
    assert chat.chat_options({"reasoning": {"effort": "none"}}).thinking is False
    assert chat.chat_options({"reasoning": {"enabled": True, "effort": "ultra"}}) == chat.ChatOptions(True, 75, True)
    assert chat.chat_options({"reasoning": {"effort": "low"}, "reasoning_effort": "max"}).effort == 100
    assert chat.chat_options({"reasoning": {"effort": "low"}, "chat_template_kwargs": {"reasoning_effort": "max"}}).effort == 100
    assert chat.chat_options({"reasoning": {"enabled": False}, "thinking": True}).thinking is True
    with pytest.raises(chat.ChatFormatError):
        chat.chat_options({"reasoning": "high"})
