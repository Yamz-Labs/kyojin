"""CPU tests of the Hy4 server helpers (tools/hy4/serve_core.py): request parsing, chat template, tokenizer special tokens, streaming output.
Run: python -m unittest tests/test_hy4_serve_core.py   (needs jinja2 + tokenizers; no GPU, no model)
Optional ground-truth test against a real GGUF: set HY4_TEST_GGUF=<first shard> and HY4_TEST_ORACLE=<dir with p1_short_en/index.json + p1_short_en.txt>."""
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools", "hy4"))
import serve_core as C  # noqa: E402

with open(os.path.join(HERE, "data", "hy4_chat_template.jinja"), encoding="utf-8") as _f:
    TEMPLATE = _f.read()
TK = ":6124c78e"


def _bytes_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def tiny_meta():
    """a byte-level vocabulary (no merges) + the Hy4 special tokens, enough to test the special-token handling and UTF-8 round trips"""
    b2u = _bytes_to_unicode()
    toks = [b2u[b] for b in range(256)]
    types = [1] * 256
    specials = [f"<｜hy_start{TK}｜>", f"<｜hy_middle{TK}｜>", f"<｜hy_end{TK}｜>", f"<think{TK}>", f"</think{TK}>", f"<｜reasoning_mode{TK}｜>"]
    toks += specials
    types += [3] * len(specials)
    return {"tokenizer.ggml.tokens": toks, "tokenizer.ggml.merges": [], "tokenizer.ggml.token_type": types, "tokenizer.ggml.eos_token_id": 256 + 2,
            "tokenizer.chat_template": TEMPLATE}


class Core(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v = C.Vocab(tiny_meta())
        cls.ids = {t: i for i, t in enumerate(cls.v.tokens)}

    # -- template
    def test_template_default_effort_ends_in_open_think(self):
        r = C.parse_chat({"messages": [{"role": "user", "content": "Bonjour"}]})
        text = self.v.template.render(r["messages"], None, True, reasoning_effort=r["effort"])
        self.assertEqual(r["effort"], "high")
        self.assertTrue(text.startswith(f"<｜hy_start{TK}｜>system"))
        self.assertIn(f"<｜hy_start{TK}｜>user<｜hy_middle{TK}｜>Bonjour<｜hy_end{TK}｜>", text)
        self.assertTrue(text.endswith(f"<｜hy_start{TK}｜>assistant<｜hy_middle{TK}｜><think{TK}>"))

    def test_template_no_think_closes_the_think_block(self):
        r = C.parse_chat({"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}], "reasoning_effort": "none"})
        text = self.v.template.render(r["messages"], None, True, reasoning_effort=r["effort"])
        self.assertEqual(r["effort"], "no_think")
        self.assertIn("Be brief.", text)
        self.assertTrue(text.endswith(f"<｜hy_middle{TK}｜><think{TK}></think{TK}>"))

    def test_template_multi_turn_keeps_history(self):
        msgs = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": "c"}]
        text = self.v.template.render(C.parse_chat({"messages": msgs, "reasoning_effort": "low"})["messages"], None, True, reasoning_effort="low")
        self.assertLess(text.index(">a<"), text.index("b"))
        self.assertLess(text.index("b"), text.rindex(">c<"))

    def test_template_error_is_a_request_error(self):
        with self.assertRaises(C.RequestError):
            self.v.template.render([{"role": "user", "content": "x"}], None, True, reasoning_effort="bogus")

    # -- parsing
    def test_effort_names(self):
        for given, want in (("medium", "high"), ("max", "max"), ("minimal", "low"), ("no_think", "no_think")):
            self.assertEqual(C.parse_effort({"reasoning_effort": given}), want)
        self.assertEqual(C.parse_effort({"chat_template_kwargs": {"reasoning_effort": "low"}}), "low")
        self.assertEqual(C.parse_effort({"chat_template_kwargs": {"enable_thinking": False}}), "high")   # no effort given: template default
        with self.assertRaises(C.RequestError):
            C.parse_effort({"reasoning_effort": "turbo"})

    def test_chat_request_errors(self):
        bad = [{}, {"messages": []}, {"messages": [{"role": "user", "content": "x"}], "n": 2},
               {"messages": [{"role": "assistant", "content": "x"}]}, {"messages": [{"role": "robot", "content": "x"}]},
               {"messages": [{"role": "user", "content": "x"}], "tools": [{"type": "function"}]},
               {"messages": [{"role": "user", "content": "x"}], "max_tokens": 0},
               {"messages": [{"role": "user", "content": "x"}], "stop": ["a"] * 9},
               {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]}]
        for b in bad:
            with self.assertRaises(C.RequestError, msg=str(b)):
                C.parse_chat(b)

    def test_chat_request_fields(self):
        r = C.parse_chat({"messages": [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}], "max_completion_tokens": 7,
                          "stream": True, "stop": "END", "stream_options": {"include_usage": True}, "temperature": 0.7, "speculative": False}, 99)
        self.assertEqual((r["max_tokens"], r["stream"], r["stop"], r["include_usage"], r["speculative"], r["sampling_ignored"]), (7, True, ["END"], True, False, True))
        self.assertEqual(r["messages"][0]["content"], "ab")
        self.assertEqual(C.parse_chat({"messages": [{"role": "user", "content": "x"}]}, 99)["max_tokens"], 99)

    def test_completion_prompts(self):
        self.assertEqual(C.parse_completion({"prompt": "hi"}, 100)["prompt"], "hi")
        self.assertEqual(C.parse_completion({"prompt": [1, 2, 3]}, 100)["prompt_ids"], [1, 2, 3])
        self.assertEqual(C.parse_completion({"prompt": [[1, 2]]}, 100)["prompt_ids"], [1, 2])
        self.assertEqual(C.parse_completion({"prompt": ["hi"]}, 100)["prompt"], "hi")
        for b in ({}, {"prompt": ""}, {"prompt": [1, 200]}, {"prompt": ["a", "b"]}, {"prompt": 5}):
            with self.assertRaises(C.RequestError, msg=str(b)):
                C.parse_completion(b, 100)

    # -- tokenizer
    def test_special_markers_are_single_ids(self):
        text = self.v.template.render([{"role": "user", "content": "héllo 漢字 😀"}], None, True, reasoning_effort="high")
        ids = self.v.encode(text)
        self.assertEqual(ids[0], self.ids[f"<｜hy_start{TK}｜>"])
        self.assertEqual(ids[-1], self.ids[f"<think{TK}>"])
        self.assertEqual(ids.count(self.ids[f"<｜hy_end{TK}｜>"]), 2)      # system turn + user turn
        self.assertEqual(self.v.decode(ids), text)

    def test_unicode_round_trip(self):
        for s in ("ça va ?", "漢字かな", "emoji 😀🙂 end", "tab\tnewline\n  spaces", "Straße"):
            self.assertEqual(self.v.decode(self.v.encode(s)), s)

    def test_think_markers_found_from_vocab(self):
        self.assertEqual((self.v.think_begin, self.v.think_end, self.v.eos_id), (f"<think{TK}>", f"</think{TK}>", self.ids[f"<｜hy_end{TK}｜>"]))

    # -- output side
    def test_incremental_decoder_never_splits_a_character(self):
        s = "a😀é漢b"
        ids = self.v.encode(s)
        d, parts = C.IncrementalDecoder(self.v), []
        for t in ids:
            parts.append(d.push([t]))
        parts.append(d.flush())
        self.assertNotIn("�", "".join(parts))
        self.assertTrue(all(p == "" or p.encode("utf-8") for p in parts))
        self.assertEqual("".join(parts), s)

    def _run(self, text, in_think, stops=(), step=1, eos=False):
        ids = self.v.encode(text) + ([self.v.eos_id] + self.v.encode("junk") if eos else [])
        o = C.OutputStream(self.v, in_think=in_think, stops=list(stops))
        got = []
        for i in range(0, len(ids), step):
            got += o.push(ids[i:i + step])
        got += o.finish()
        agg = {"reasoning": "", "content": ""}
        for ch, tx in got:
            agg[ch] += tx
        return agg, o

    def test_think_split_with_marker_across_rounds(self):
        txt = f"let me think é</think{TK}>The answer."
        for step in (1, 2, 3, 5, 100):
            agg, o = self._run(txt, True, step=step)
            self.assertEqual(agg, {"reasoning": "let me think é", "content": "The answer."}, step)
            self.assertIsNone(o.finish_reason)

    def test_no_think_is_all_content(self):
        agg, _ = self._run("plain answer", False)
        self.assertEqual(agg, {"reasoning": "", "content": "plain answer"})

    def test_eos_stops_and_drops_the_rest(self):
        agg, o = self._run("done.", False, step=2, eos=True)
        self.assertEqual(agg["content"], "done.")
        self.assertEqual(o.finish_reason, "stop")
        self.assertEqual(o.n_tokens, len(self.v.encode("done.")))

    def test_stop_string_cuts_and_holds_back_partial_matches(self):
        for step in (1, 3, 50):
            agg, o = self._run("one two STOP three", False, stops=["STOP"], step=step)
            self.assertEqual(agg["content"], "one two ", step)
            self.assertEqual(o.finish_reason, "stop")
        agg, o = self._run("ends with ST", False, stops=["STOP"], step=1)      # a held-back partial match is released at the end
        self.assertEqual(agg["content"], "ends with ST")
        self.assertIsNone(o.finish_reason)

    def test_response_bodies(self):
        b = C.chat_response("id", "m", 1, "hi", "why", "stop", {"total_tokens": 3})
        self.assertEqual(b["choices"][0]["message"], {"role": "assistant", "content": "hi", "reasoning_content": "why"})
        self.assertEqual(C.chat_chunk("id", "m", 1, {"content": "x"})["choices"][0]["delta"], {"content": "x"})
        self.assertEqual(C.error_body("bad")["error"]["message"], "bad")


@unittest.skipUnless(os.environ.get("HY4_TEST_GGUF") and os.environ.get("HY4_TEST_ORACLE"), "set HY4_TEST_GGUF and HY4_TEST_ORACLE for the ground-truth test")
class GroundTruth(unittest.TestCase):
    def test_prompt_ids_match_the_reference_runtime(self):
        tree = os.environ.get("HY4_TEST_TREE", os.path.dirname(HERE))
        import types
        for n, s in (("exllamav3", ""), ("exllamav3.loader", "loader")):
            m = types.ModuleType(n)
            m.__path__ = [os.path.join(tree, "exllamav3", s)]
            sys.modules[n] = m
        from exllamav3.loader.gguf_collection import GGUFCollection
        v = C.Vocab(GGUFCollection(os.environ["HY4_TEST_GGUF"]).meta)
        d = os.environ["HY4_TEST_ORACLE"]
        want = json.load(open(os.path.join(d, "p1_short_en", "index.json")))["prompt_ids"]
        with open(os.path.join(d, "..", "prompts", "p1_short_en.txt"), encoding="utf-8") as f:
            text = f.read()
        self.assertEqual(v.encode(text.rstrip("\n")), want)


if __name__ == "__main__":
    unittest.main()
