#!/usr/bin/env python3
"""HTTP acceptance check for tools/qwen/serve.py (run against a live server; no engine import).

usage: serve_check.py URL OUT_DIR [--steps functional,speed,warmcold] [--ntok 160] [--nper 4] [--passes 2]
Writes OUT_DIR/check.json and prints one line per check. Everything is greedy (temperature 0) unless noted.
"""
from __future__ import annotations

import argparse
import base64
import glob
import io
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "qwen")), sys.path.insert(0, str(ROOT / "tools" / "glm"))

ap = argparse.ArgumentParser()
ap.add_argument("url"), ap.add_argument("out")
ap.add_argument("--steps", default="functional,speed,warmcold")
ap.add_argument("--ntok", type=int, default=160), ap.add_argument("--nper", type=int, default=4)
ap.add_argument("--passes", type=int, default=2), ap.add_argument("--model", default="Qwen3.8-Flash-Yamz")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
R: dict = {}


def post(payload: dict, stream: bool = False, timeout: int = 900):
    req = urllib.request.Request(a.url + "/v1/chat/completions", json.dumps({"model": a.model} | payload).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    if not stream:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read()), time.perf_counter() - t0
    chunks, ttft = [], None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line.endswith("[DONE]"):
                continue
            ev = json.loads(line[6:])
            chunks.append(ev)
            d = ev["choices"][0]["delta"] if ev.get("choices") else {}
            if ttft is None and (d.get("content") or d.get("reasoning_content") or d.get("tool_calls")):
                ttft = time.perf_counter() - t0
    return chunks, ttft


def rec(name, ok, **kw):
    R.setdefault("checks", []).append({"name": name, "ok": bool(ok), **kw})
    print(("OK  " if ok else "BAD ") + name, json.dumps(kw, ensure_ascii=False)[:400], flush=True)
    json.dump(R, open(a.out + "/check.json", "w"), indent=1, ensure_ascii=False)


def msg_of(body):
    return body["choices"][0]["message"]


BAD = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|vision_start|>", "<|image_pad|>", "<tool_call>"]


def functional():
    chat = [("capital", "What is the capital of France? Answer in one sentence.", "paris"),
            ("fib", "Write a Python function that returns the n-th Fibonacci number.", "def "),
            ("vaccin", "Explique en deux phrases comment fonctionne un vaccin.", "immun"),
            ("tips", "List three tips for staying focused while working from home.", "1"),
            ("math", "What is 17 times 23? Give only the number.", "391")]
    for nm, p, kw in chat:
        b, dt = post({"messages": [{"role": "user", "content": p}], "enable_thinking": False, "max_tokens": 200,
                      "temperature": 0})
        m = msg_of(b)
        rec("chat_nothink_" + nm, kw in m["content"].lower() and "reasoning_content" not in m
            and not any(x in m["content"] for x in BAD) and b["choices"][0]["finish_reason"] in ("stop", "length"),
            text=m["content"][:160], finish=b["choices"][0]["finish_reason"], usage=b["usage"], secs=round(dt, 2))
    # thinking on: reasoning split
    b, dt = post({"messages": [{"role": "user", "content": "A farmer has 17 sheep and all but 9 run away. How many are left? Think briefly."}],
                  "reasoning_effort": "low", "max_tokens": 3000, "temperature": 0})
    m = msg_of(b)
    rec("chat_think_on", bool(m.get("reasoning_content")) and "9" in m["content"] and b["choices"][0]["finish_reason"] == "stop"
        and "</think>" not in m["content"], reasoning_head=m.get("reasoning_content", "")[:100], content=m["content"][:200],
        finish=b["choices"][0]["finish_reason"], tokens=b["usage"]["completion_tokens"])
    # thinking cut by max_tokens -> finish_reason length, content empty
    b, _ = post({"messages": [{"role": "user", "content": "Prove that the square root of 2 is irrational."}], "max_tokens": 24, "temperature": 0})
    rec("think_cut_is_length", b["choices"][0]["finish_reason"] == "length" and b["usage"]["completion_tokens"] == 24,
        finish=b["choices"][0]["finish_reason"], completion_tokens=b["usage"]["completion_tokens"],
        has_reasoning=bool(msg_of(b).get("reasoning_content")))
    # tool round trip
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city",
              "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}}, "required": ["city"]}}}]
    msgs = [{"role": "user", "content": "What is the weather in Paris right now? Use the tool."}]
    b, _ = post({"messages": msgs, "tools": tools, "enable_thinking": False, "max_tokens": 200, "temperature": 0})
    m = msg_of(b)
    calls = m.get("tool_calls") or []
    args_ok = bool(calls) and "paris" in json.loads(calls[0]["function"]["arguments"]).get("city", "").lower()
    rec("tool_call", args_ok and b["choices"][0]["finish_reason"] == "tool_calls", calls=calls[:1], content=m["content"][:80])
    if calls:
        msgs2 = msgs + [m, {"role": "tool", "tool_call_id": calls[0]["id"], "content": "Sunny, 21 degrees Celsius."}]
        b, _ = post({"messages": msgs2, "tools": tools, "enable_thinking": False, "max_tokens": 200, "temperature": 0})
        t = msg_of(b)["content"]
        rec("tool_round_trip_answer", "21" in t and b["choices"][0]["finish_reason"] == "stop", text=t[:200])
    # streamed request
    ch, ttft = post({"messages": [{"role": "user", "content": "Write two sentences about the sea."}], "stream": True,
                     "enable_thinking": False, "max_tokens": 120, "temperature": 0}, stream=True)
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in ch if c.get("choices"))
    last = ch[-1]
    b, _ = post({"messages": [{"role": "user", "content": "Write two sentences about the sea."}],
                 "enable_thinking": False, "max_tokens": 120, "temperature": 0})
    rec("stream", len(ch) > 5 and last["choices"][0]["finish_reason"] in ("stop", "length") and "usage" in last
        and text == msg_of(b)["content"], chunks=len(ch), ttft_s=round(ttft or 0, 3), usage=last.get("usage"),
        same_as_non_stream=text == msg_of(b)["content"])
    # image
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (448, 448), (20, 40, 200))
    ImageDraw.Draw(img).rectangle([120, 120, 330, 330], fill=(230, 20, 20))
    buf = io.BytesIO(); img.save(buf, "PNG")
    uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    b, _ = post({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": uri}},
                  {"type": "text", "text": "What color is the square in the middle of the picture, and what color is the background? Answer briefly."}]}],
                 "enable_thinking": False, "max_tokens": 80, "temperature": 0})
    t = msg_of(b)["content"].lower()
    rec("image", "red" in t and "blue" in t, text=t[:200], usage=b["usage"])
    # 3-turn conversation, streamed: re-prefilled tokens and TTFT per turn
    import random
    random.seed(3)
    pool = "".join(open(f).read() for f in sorted(glob.glob(str(ROOT / "exllamav3" / "generator" / "*.py")))[:6])
    system = "You are a code reviewer. Reference material:\n" + pool[:9000]
    convo = [{"role": "system", "content": system}]
    rows = []
    for q in ("Summarize what the Generator class does in two sentences.",
              "Which file holds the page table, and what is a page?", "Now give me one risk in the design."):
        convo.append({"role": "user", "content": q})
        ch, ttft = post({"messages": convo, "stream": True, "enable_thinking": False, "max_tokens": 100, "temperature": 0}, stream=True)
        usage = ch[-1].get("usage", {}); tm = ch[-1].get("timings", {})
        reply = "".join(c["choices"][0]["delta"].get("content", "") for c in ch if c.get("choices"))
        convo.append({"role": "assistant", "content": reply})
        rows.append({"prompt_tokens": usage.get("prompt_tokens"), "cached": usage["prompt_tokens_details"]["cached_tokens"],
                     "reprefilled": tm.get("prompt_n"), "ttft_s": round(ttft or 0, 3), "prefill_ms": round(tm.get("prompt_ms", 0))})
    rec("three_turn", rows[1]["reprefilled"] < rows[1]["prompt_tokens"] * 0.5 and rows[2]["reprefilled"] < rows[2]["prompt_tokens"] * 0.5,
        turns=rows)


def classes():
    from prompts10 import PROMPTS
    from spec_prompts import EXTRA
    out = []
    for k in ("chat", "prose", "code"):
        out += [(k, i, p) for i, p in enumerate(PROMPTS[k][:a.nper])]
    for k in ("multi", "copy"):
        out += [(k, i, p) for i, p in enumerate(EXTRA[k][:a.nper])]
    return out


def gen(prompt, spec, ntok=None, **kw):
    b, _ = post({"messages": [{"role": "user", "content": prompt}], "enable_thinking": False, "max_tokens": ntok or a.ntok,
                 "temperature": 0, "speculative": spec, "return_token_ids": True} | kw)
    return b


def speed():
    ps = classes()
    gen("Hello", True, 24); gen("Hello", False, 24)
    res = {"plain": {}, "spec": {}}
    for ps_i in range(a.passes):
        order = ("plain", "spec") if ps_i % 2 == 0 else ("spec", "plain")
        for arm in order:
            gen("Hello", arm == "spec", 16)  # settle after the generator rebuild
            for k, i, p in ps:
                b = gen(p, arm == "spec")
                res[arm].setdefault((k, i), []).append({"tps": b["timings"]["predicted_per_second"], "toks": b["token_ids"],
                                                        "n": b["usage"]["completion_tokens"], "acc": b["timings"].get("accepted_draft_tokens")})
    rows, div = {}, []
    for (k, i, p) in ps:
        ref = res["plain"][(k, i)][0]["toks"]
        allt = [r["toks"] for arm in res for r in res[arm][(k, i)]]
        ident = all(t == ref for t in allt)
        if not ident:
            for t in allt:
                if t != ref:
                    div.append({"cls": k, "idx": i, "first_div": next((j for j, (u, v) in enumerate(zip(ref, t)) if u != v), min(len(ref), len(t)))})
                    break
        rows.setdefault(k, []).append({"plain": statistics.mean(r["tps"] for r in res["plain"][(k, i)]),
                                       "spec": statistics.mean(r["tps"] for r in res["spec"][(k, i)]), "ident": ident})
    table = {}
    for k, v in rows.items():
        pm, sm = statistics.mean(r["plain"] for r in v), statistics.mean(r["spec"] for r in v)
        table[k] = {"n": len(v), "plain_tps": round(pm, 1), "spec_tps": round(sm, 1), "gain_pct": round(100 * (sm / pm - 1), 1),
                    "identical": f"{sum(r['ident'] for r in v)}/{len(v)}"}
    rec("speed_plain_vs_spec", not div, passes=a.passes, ntok=a.ntok, table=table, divergences=div)


def warmcold():
    pool = "".join(open(f).read() for f in sorted(glob.glob(str(ROOT / "exllamav3" / "**" / "*.py"), recursive=True))[:40])
    qa = [("Explain what the first function in the material does.", "Now rewrite it in a simpler way."),
          ("List the classes you see in the material.", "Which one is the most complex and why?"),
          ("What is a recurrent state in this code?", "How is it checkpointed?"),
          ("Describe the cache design in three sentences.", "What could go wrong with it?"),
          ("Name three configuration options you can find.", "Which of them affects memory?")]
    rows = []
    for n, (q1, q2) in enumerate(qa):
        system = f"You are a code assistant. Material {n}:\n" + pool[n * 11000:n * 11000 + 11000]
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": q1}]
        b = post({"messages": msgs, "enable_thinking": False, "max_tokens": 100, "temperature": 0})[0]
        msgs += [{"role": "assistant", "content": msg_of(b)["content"]}, {"role": "user", "content": q2}]
        warm = post({"messages": msgs, "enable_thinking": False, "max_tokens": 160, "temperature": 0, "return_token_ids": True})[0]
        # reference = the split cold run: fresh generator (cache_prompt false), turn 1 with max_tokens 1, then turn 2.
        # An unsplit cold run prefills in other chunk shapes and drifts by chunk-boundary numerics (not a cache defect).
        post({"messages": msgs[:2], "enable_thinking": False, "max_tokens": 1, "temperature": 0, "cache_prompt": False})
        ref = post({"messages": msgs, "enable_thinking": False, "max_tokens": 160, "temperature": 0, "return_token_ids": True})[0]
        w, c = warm["token_ids"], ref["token_ids"]
        fd = first_div(w, c)
        rows.append({"prompt_tokens": warm["usage"]["prompt_tokens"], "warm_cached": warm["usage"]["prompt_tokens_details"]["cached_tokens"],
                     "ref_cached": ref["usage"]["prompt_tokens_details"]["cached_tokens"], "n_tokens": len(w), "first_divergence": fd})
    rec("warm_vs_cold_greedy", all(r["first_divergence"] is None for r in rows), rows=rows)


def first_div(w, c):
    fd = next((j for j, (u, v) in enumerate(zip(w, c)) if u != v), None)
    return min(len(w), len(c)) if fd is None and len(w) != len(c) else fd


def warmcold2():
    """Isolate the warm-vs-cold divergence: (A) plain decode, warm vs cold; (B) speculative, cold vs cold (noise floor)."""
    pool = "".join(open(f).read() for f in sorted(glob.glob(str(ROOT / "exllamav3" / "**" / "*.py"), recursive=True))[:40])
    qa = [("Explain what the first function in the material does.", "Now rewrite it in a simpler way."),
          ("List the classes you see in the material.", "Which one is the most complex and why?"),
          ("What is a recurrent state in this code?", "How is it checkpointed?"),
          ("Describe the cache design in three sentences.", "What could go wrong with it?"),
          ("Name three configuration options you can find.", "Which of them affects memory?")]
    base = {"enable_thinking": False, "temperature": 0, "return_token_ids": True}
    convs = []
    for n, (q1, q2) in enumerate(qa):
        system = f"You are a code assistant. Material {n}:\n" + pool[n * 11000:n * 11000 + 11000]
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": q1}]
        b = post({"messages": msgs, "max_tokens": 100, "speculative": False} | base)[0]
        convs.append(msgs + [{"role": "assistant", "content": msg_of(b)["content"]}, {"role": "user", "content": q2}])
    plain = []
    for msgs in convs:
        post({"messages": msgs[:2], "max_tokens": 100, "speculative": False} | base)
        warm = post({"messages": msgs, "max_tokens": 160, "speculative": False} | base)[0]
        cold = post({"messages": msgs, "max_tokens": 160, "speculative": False, "cache_prompt": False} | base)[0]
        plain.append({"warm_cached": warm["usage"]["prompt_tokens_details"]["cached_tokens"], "first_divergence": first_div(warm["token_ids"], cold["token_ids"])})
    rec("plain_warm_vs_cold", all(r["first_divergence"] is None for r in plain), rows=plain)
    cc = []
    for msgs in convs:
        c1 = post({"messages": msgs, "max_tokens": 160, "cache_prompt": False} | base)[0]
        c2 = post({"messages": msgs, "max_tokens": 160, "cache_prompt": False} | base)[0]
        cc.append({"first_divergence": first_div(c1["token_ids"], c2["token_ids"])})
    rec("spec_cold_vs_cold", all(r["first_divergence"] is None for r in cc), rows=cc)


def robust():
    import concurrent.futures as cf
    import http.client
    import urllib.parse
    u = urllib.parse.urlparse(a.url)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=60)
    conn.request("POST", "/v1/chat/completions", json.dumps({"model": a.model, "stream": True, "max_tokens": 600, "enable_thinking": False,
                 "messages": [{"role": "user", "content": "Write a long story about a lighthouse."}]}), {"Content-Type": "application/json"})
    r = conn.getresponse()
    for _ in range(6):
        r.fp.readline()
    conn.close()  # client leaves mid-stream
    t0 = time.perf_counter()
    b, _ = post({"messages": [{"role": "user", "content": "Say hello."}], "enable_thinking": False, "max_tokens": 20, "temperature": 0})
    waited = time.perf_counter() - t0
    rec("client_disconnect_then_request", b["choices"][0]["message"]["content"] != "" and waited < 30, waited_s=round(waited, 1))
    with cf.ThreadPoolExecutor(3) as ex:
        futs = [ex.submit(post, {"messages": [{"role": "user", "content": f"Count to {k}."}], "enable_thinking": False,
                                 "max_tokens": 60, "temperature": 0}) for k in (3, 5, 7)]
        outs = [f.result()[0]["choices"][0]["message"]["content"] for f in futs]
    rec("three_concurrent", all(outs), outs=[o[:40] for o in outs])
    try:
        urllib.request.urlopen(urllib.request.Request(a.url + "/v1/chat/completions", b"{}", {"Content-Type": "application/json"}))
        rec("bad_request_400", False)
    except urllib.error.HTTPError as e:
        rec("bad_request_400", e.code == 400, body=e.read().decode()[:120])
    h = json.loads(urllib.request.urlopen(a.url + "/health").read())
    rec("health", h["status"] == "ok", health=h)


for step in a.steps.split(","):
    {"functional": functional, "speed": speed, "warmcold": warmcold, "warmcold2": warmcold2, "robust": robust}[step]()
print("CHECK_DONE", sum(1 for c in R.get("checks", []) if not c["ok"]), "bad", flush=True)
