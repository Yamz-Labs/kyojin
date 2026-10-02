"""CPU-only tests for the MiMo tool-call path (parser + template round trip). No GPU, no model weights.

    python tools/mimo/test_toolcalls.py
The real chat template is used when the pack is present (MIMO_MODEL or ~/models/MiMo-V2.6-Flash-MOPD-EXL3-Yamz), else skipped.
"""
import json, os, sys, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import serve

PACK = Path(os.environ.get("MIMO_MODEL", os.path.expanduser("~/models/MiMo-V2.6-Flash-MOPD-EXL3-Yamz")))
TEMPLATE = PACK / "chat_template.jinja"

TOOLS = [{"type": "function", "function": {"name": "search", "parameters": {"type": "object", "properties": {
    "query": {"type": "string"}, "limit": {"type": "integer"}, "exact": {"type": "boolean"},
    "score": {"type": "number"}, "tags": {"type": "array"}, "opts": {"type": "object"},
    "zip": {"type": "string"}}}}}]


def call(name, body):
    return f"<tool_call><function={name}>{body}</function></tool_call>"


def args(msg, i=0):
    return json.loads(msg["tool_calls"][i]["function"]["arguments"])


class ToolCalls(unittest.TestCase):
    def test_xml_single_call_typed_by_schema(self):
        m = serve.parse_completion(call("search", "<parameter=query>hello world</parameter>"
                                        "<parameter=limit>5</parameter><parameter=exact>true</parameter>"
                                        "<parameter=score>0.5</parameter><parameter=zip>007</parameter>"
                                        '<parameter=tags>["a","b"]</parameter><parameter=opts>{"k": 1}</parameter>'), TOOLS)
        self.assertEqual(args(m), {"query": "hello world", "limit": 5, "exact": True, "score": 0.5,
                                   "zip": "007", "tags": ["a", "b"], "opts": {"k": 1}})
        self.assertEqual(m["finish_reason"], "tool_calls")

    def test_xml_without_schema_keeps_scalars_as_text(self):
        m = serve.parse_completion(call("unknown", "<parameter=n>5</parameter><parameter=s>x</parameter>"), TOOLS)
        self.assertEqual(args(m), {"n": "5", "s": "x"})

    def test_string_typed_param_never_json_decoded(self):
        m = serve.parse_completion(call("search", "<parameter=query>123</parameter>"), TOOLS)
        self.assertEqual(args(m), {"query": "123"})
        m = serve.parse_completion(call("search", '<parameter=query>{"a": 1}</parameter>'), TOOLS)
        self.assertEqual(args(m), {"query": '{"a": 1}'})

    def test_multiline_string_value(self):
        m = serve.parse_completion(call("search", "<parameter=query>\nline1\nline2\n</parameter>"), TOOLS)
        self.assertEqual(args(m)["query"], "line1\nline2")

    def test_json_body(self):
        m = serve.parse_completion(call("search", '{"query": "x", "limit": 3}'), TOOLS)
        self.assertEqual(args(m), {"query": "x", "limit": 3})

    def test_no_arguments(self):
        m = serve.parse_completion(call("ping", ""), TOOLS)
        self.assertEqual(m["tool_calls"][0]["function"]["arguments"], "{}")

    def test_multi_call_separate_blocks_with_text_and_think(self):
        text = ("<think>plan</think>Let me check.\n" + call("search", "<parameter=query>a</parameter>")
                + call("search", "<parameter=query>b</parameter><parameter=limit>2</parameter>"))
        m = serve.parse_completion(text, TOOLS)
        self.assertEqual(m["content"], "Let me check.\n")
        self.assertEqual(m["reasoning_content"], "plan")
        self.assertEqual([args(m, i) for i in (0, 1)], [{"query": "a"}, {"query": "b", "limit": 2}])
        self.assertEqual(len({c["id"] for c in m["tool_calls"]}), 2)

    def test_multi_function_in_one_block(self):
        text = ("<tool_call><function=search><parameter=query>a</parameter></function>"
                "<function=search><parameter=query>b</parameter></function></tool_call>")
        m = serve.parse_completion(text, TOOLS)
        self.assertEqual([args(m, 0), args(m, 1)], [{"query": "a"}, {"query": "b"}])

    def test_malformed_json_is_passed_through_not_executed_as_empty(self):
        bad = '{"query": "x", "limit": '
        m = serve.parse_completion(call("search", bad), TOOLS)
        raw = m["tool_calls"][0]["function"]["arguments"]
        self.assertEqual(raw, bad.strip())
        with self.assertRaises(json.JSONDecodeError):
            json.loads(raw)  # the client reports the error back to the model

    def test_json_non_object_is_passed_through(self):
        m = serve.parse_completion(call("search", "[1, 2]"), TOOLS)
        self.assertEqual(m["tool_calls"][0]["function"]["arguments"], "[1, 2]")

    def test_nameless_block_does_not_leak_or_duplicate_text(self):
        m = serve.parse_completion("before <tool_call>garbage</tool_call> after", TOOLS)
        self.assertNotIn("tool_calls", m)
        self.assertEqual(m["content"], "before  after")

    def test_truncated_call_is_dropped_from_content(self):
        m = serve.parse_completion("ok <tool_call><function=search><parameter=query>par", TOOLS)
        self.assertNotIn("tool_calls", m)
        self.assertEqual(m["content"], "ok ")

    def test_missing_function_close_inside_block(self):
        m = serve.parse_completion("<tool_call><function=search><parameter=query>a</parameter></tool_call>", TOOLS)
        self.assertEqual(args(m), {"query": "a"})

    @unittest.skipUnless(TEMPLATE.exists(), "pack chat_template.jinja not present")
    def test_roundtrip_through_real_template(self):
        original = {"query": "hello", "limit": 5, "exact": False, "score": 1.5, "tags": ["a", "b"],
                    "opts": {"k": [1, 2]}, "zip": "007"}
        history = [{"role": "user", "content": "go"},
                   {"role": "assistant", "content": "", "tool_calls": [
                       {"id": "c1", "type": "function",
                        "function": {"name": "search", "arguments": json.dumps(original)}}]},
                   {"role": "tool", "tool_call_id": "c1", "content": "result"}]
        prompt = serve.render_prompt(TEMPLATE.read_text(encoding="utf-8"), history, TOOLS)
        start = prompt.index("<tool_call>")
        end = prompt.index("</tool_call>") + len("</tool_call>")
        rendered = prompt[start:end]
        self.assertIn("<parameter=limit>5</parameter>", rendered)
        m = serve.parse_completion(rendered, TOOLS)
        self.assertEqual(args(m), original)
        # and a second pass: parsed call -> template -> parse is a fixed point
        history[1]["tool_calls"][0]["function"]["arguments"] = m["tool_calls"][0]["function"]["arguments"]
        p2 = serve.render_prompt(TEMPLATE.read_text(encoding="utf-8"), history, TOOLS)
        self.assertEqual(p2[p2.index("<tool_call>"):p2.index("</tool_call>")], rendered[:-len("</tool_call>")])

    @unittest.skipUnless(TEMPLATE.exists(), "pack chat_template.jinja not present")
    def test_malformed_arguments_in_history_do_not_crash_template(self):
        history = [{"role": "user", "content": "go"},
                   {"role": "assistant", "content": "", "tool_calls": [
                       {"id": "c1", "type": "function", "function": {"name": "search", "arguments": '{"query": '}}]}]
        prompt = serve.render_prompt(TEMPLATE.read_text(encoding="utf-8"), history, TOOLS)
        self.assertIn('<function=search>{"query":', prompt)


if __name__ == "__main__":
    unittest.main()
