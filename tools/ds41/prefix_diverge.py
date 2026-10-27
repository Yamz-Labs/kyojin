"""Where do two consecutive chat prompts diverge, token by token, with the real chat template and tokenizer?

Simulates a Hermes-style session: system + tools (~6K tokens), user, assistant (reasoning + tool call), tool result, assistant, follow-ups.
The model reply of each turn is written in the template's own format, parsed the way serve.py parses it, handed back to the "client" as a
message and re-rendered in the next prompt. Per turn it prints: prompt tokens, the common prefix (LCP) with the previous prompt and with
the state a cache holds after decode (previous prompt + reply), and the tokens around the first difference.

  python prefix_diverge.py TOKENIZER.json [--variant NAME ...]
Variants: R = client returns reasoning_content, N = client drops it; tools / notools; think / nothink.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ds41_chat as chat  # noqa: E402
import ds41_parse as parse  # noqa: E402


def lcp(a, b) -> int:
    n = min(len(a), len(b))
    lo, hi = 0, n
    while lo < hi:                       # prefix equality is monotone
        mid = (lo + hi + 1) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


TOOLS = [{"type": "function", "function": {"name": f"tool_{i}", "description": "Does thing number %d. " % i * 6,
          "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "a path " * 10},
                                                          "limit": {"type": "integer"}}, "required": ["path"]}}} for i in range(24)]
SYSTEM = ("You are a careful coding agent. " + "Follow the project rules, keep answers short, use tools when needed. " * 40)


def build(tk, variant: str, turns: int = 8):
    tools_on = "notools" not in variant
    think = "nothink" not in variant
    keep_reasoning = "N" not in variant.split("-")
    tools = TOOLS if tools_on else None
    kw = dict(thinking=think, reasoning_effort="high", drop_thinking=True)
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "List the files in src and tell me what they do."}]
    rows, prev_ids, prev_stored = [], None, None
    for t in range(turns):
        prompt = chat.render_chat(msgs, tools, **kw)
        ids = tk.encode(prompt, add_special_tokens=False).ids
        row = {"turn": t, "prompt": len(ids)}
        if prev_ids is not None:
            row["lcp_prompt"] = lcp(prev_ids, ids)
            row["prev_prompt"] = len(prev_ids)
            row["lcp_stored"] = lcp(prev_stored, ids)
            row["prev_stored"] = len(prev_stored)
            k = row["lcp_stored"]
            row["diverge_tokens"] = {"old": tk.decode(prev_stored[max(0, k - 3):k + 6], skip_special_tokens=False),
                                     "new": tk.decode(ids[max(0, k - 3):k + 6], skip_special_tokens=False)}
        rows.append(row)
        # the model reply, in the template's own format (what the model was trained to emit after the generation prompt)
        want_tool = t % 2 == 0 and tools_on
        asst = {"role": "assistant", "content": "" if want_tool else f"Answer {t}: the files are fine.\n", "reasoning_content": f"I think about step {t}.\n" if think else None}
        if want_tool:
            asst["tool_calls"] = [{"id": f"c{t}", "type": "function", "function": {"name": "tool_1", "arguments": json.dumps({"path": f"src/f{t}.py", "limit": 20})}}]
        asst = {k: v for k, v in asst.items() if v is not None}
        if tools_on:
            full = chat.render_chat(msgs + [asst, {"role": "user", "content": "x"}], tools, **kw)
            if not full.startswith(prompt):
                raise SystemExit("the assistant render does not extend the prompt (template re-renders the history)")
            raw = full[len(prompt):full.rindex(chat.USER_SP)]
        else:                                                  # no tools: the reply is plain text (the prompt already ends in <think> or </think>)
            raw = ((asst.get("reasoning_content", "") + chat.THINK_CLOSE) if think else "") + asst["content"] + chat.EOS
        reply_ids = tk.encode(raw, add_special_tokens=False).ids
        prev_ids = ids
        prev_stored = ids + reply_ids[:-1]                    # the last (EOS) token is never fed to the model
        row["reply"] = len(reply_ids)
        # serve.py: parse the raw text into a message; the client stores it and sends it back
        thinking_prompt = prompt.endswith(chat.THINK_OPEN)
        p = parse.parse_completion(raw.removesuffix(chat.EOS), thinking=thinking_prompt)
        back = {"role": "assistant", "content": p["content"]}
        if p.get("reasoning_content") and keep_reasoning:
            back["reasoning_content"] = p["reasoning_content"]
        if p.get("tool_calls"):
            back["tool_calls"] = p["tool_calls"]
        msgs.append(back)
        if back.get("tool_calls"):
            for tc in back["tool_calls"]:
                msgs.append({"role": "tool", "tool_call_id": tc.get("id", "c"), "content": "file contents " * 200})
        else:
            msgs.append({"role": "user", "content": f"Thanks. Now do follow-up {t}."})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tokenizer")
    ap.add_argument("--variant", action="append")
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(a.tokenizer)
    variants = a.variant or ["tools-think-R", "tools-think-N", "notools-think-R", "notools-think-N", "notools-nothink-R"]
    for v in variants:
        print(f"== {v}")
        for r in build(tk, v):
            print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
