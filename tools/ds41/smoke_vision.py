#!/usr/bin/env python
"""End-to-end image check against a running ds41 server (run after go-live, never against a stub for the content check).

  python tools/ds41/smoke_vision.py --url http://127.0.0.1:8000 [--model NAME] [--no-content]

Draws an image (a big red square with the word BANANA on a blue background, plus a small green circle), then:
  1. /health reports vision true;
  2. non-stream chat: the answer must name the color red (and ideally the word); prompt_tokens includes the image span;
  3. non-thinking mode: prompt 1 is a strict token prefix of prompt 2 (prompt 1 plus an assistant and a user turn); sending
     prompt 2 reports cached_tokens == prompt 1 tokens, image span included (a repeat of an identical prompt is NOT cached);
  4. the same extension with a different image of the same size (swapped colors): cache stops before the image, and the
     answer to the plain request names the new color;
  5. stream: the stream carries content, usage and [DONE];
  6. tool call + reasoning with an image in the conversation: the reply parses (tool_calls or content, reasoning_content kept);
  7. a bad image answers 400, a local path answers 400.
Exit code 0 only when every step passed. `--no-content` skips the answer checks (use with a stub).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import urllib.error
import urllib.request

from PIL import Image, ImageDraw


def make_image(bg, fg, word, size=(448, 448)) -> str:
    img = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(img)
    d.rectangle([size[0] // 6, size[1] // 6, size[0] * 5 // 6, size[1] * 5 // 6], fill=fg)
    d.ellipse([size[0] - 90, size[1] - 90, size[0] - 30, size[1] - 30], fill=(40, 200, 60))
    d.text((size[0] // 4, size[1] // 2), word, fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def post(url, body, timeout=900):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def msg(url, text="Describe this image: the main shape and its color, and any word you can read."):
    return [{"role": "user", "content": [{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": url}}]}]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--no-content", action="store_true")
    a = ap.parse_args()
    base = a.url.rstrip("/")
    results: list[tuple[str, bool, str]] = []

    def check(name, ok, info=""):
        results.append((name, bool(ok), info))
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {info}", flush=True)

    with urllib.request.urlopen(base + "/health", timeout=60) as r:
        health = json.loads(r.read())
    check("health vision true", health.get("vision") is True, str(health.get("vision")))
    model = a.model
    if not model:
        with urllib.request.urlopen(base + "/v1/models", timeout=60) as r:
            model = json.loads(r.read())["data"][0]["id"]

    red = make_image((20, 40, 160), (210, 30, 30), "BANANA")
    yellow = make_image((20, 40, 160), (235, 215, 40), "BANANA")
    body = lambda m, **kw: {"model": model, "messages": m, "max_tokens": a.max_tokens, "temperature": 0, **kw}   # noqa: E731

    def text_of(j):
        c = j["choices"][0]["message"]
        return ((c.get("content") or "") + " " + (c.get("reasoning_content") or "")).lower()

    st, raw = post(base + "/v1/chat/completions", body(msg(red)))
    j = json.loads(raw) if st == 200 else {}
    check("chat with image answers 200", st == 200, raw[:200] if st != 200 else "")
    if st != 200:
        return 1
    u1 = j["usage"]
    check("prompt_tokens holds the image span", u1["prompt_tokens"] > 150, str(u1["prompt_tokens"]))
    if not a.no_content:
        check("answer names red", "red" in text_of(j), text_of(j)[:160])
    # prefix cache: the server reuses a stored prompt only when the next prompt strictly extends it, so the checks run in
    # non-thinking mode (a thinking-mode history does not render as an extension of the previous generation prompt)
    nt = {"thinking": False}

    def tokens_of(messages):
        st_, raw_ = post(base + "/tokenize", {"model": model, "messages": messages, **nt})
        return json.loads(raw_)["tokens"] if st_ == 200 else None
    first = msg(red)
    st, raw = post(base + "/v1/chat/completions", body(first, max_tokens=24, **nt))
    ua = json.loads(raw)["usage"]
    more = lambda m: m + [{"role": "assistant", "content": "Noted."}, {"role": "user", "content": "Thanks."}]   # noqa: E731
    ta, tb = tokens_of(first), tokens_of(more(first))
    check("prompt 1 is a strict prefix of prompt 2 (same image)", ta is not None and tb is not None and len(tb) > len(ta)
          and tb[:len(ta)] == ta and len(ta) == ua["prompt_tokens"], f"{len(ta or [])} -> {len(tb or [])}")
    st, raw = post(base + "/v1/chat/completions", body(more(first), max_tokens=24, **nt))
    u2 = json.loads(raw)["usage"]
    check("extension with the same image reuses the whole first prompt, image included",
          u2["prompt_tokens_details"]["cached_tokens"] == ua["prompt_tokens"],
          f"cached {u2['prompt_tokens_details']['cached_tokens']} of {u2['prompt_tokens']}")
    post(base + "/v1/chat/completions", body(first, max_tokens=24, **nt))                    # store prompt 1 again
    st, raw = post(base + "/v1/chat/completions", body(more(msg(yellow)), max_tokens=24, **nt))
    u3 = json.loads(raw)["usage"]
    check("same ids, other pixels: only the text before the image is reused",
          u3["prompt_tokens"] == u2["prompt_tokens"] and u3["prompt_tokens_details"]["cached_tokens"] < ua["prompt_tokens"] - 150,
          f"cached {u3['prompt_tokens_details']['cached_tokens']} of {u3['prompt_tokens']}")
    st, raw = post(base + "/v1/chat/completions", body(msg(yellow)))
    j3 = json.loads(raw)
    if not a.no_content:
        check("answer names yellow", "yellow" in text_of(j3), text_of(j3)[:160])

    st, raw = post(base + "/v1/chat/completions", body(msg(red), stream=True, stream_options={"include_usage": True}))
    events = [ln[6:] for ln in raw.splitlines() if ln.startswith("data: ")]
    chunks = [json.loads(e) for e in events if e != "[DONE]"]
    got = "".join((c["choices"][0]["delta"].get("content") or "") + (c["choices"][0]["delta"].get("reasoning_content") or "")
                  for c in chunks if c.get("choices"))
    check("stream ends with [DONE] and has text and usage", st == 200 and events[-1] == "[DONE]" and got
          and any(c.get("usage") for c in chunks), f"{len(chunks)} chunks")

    tools = [{"type": "function", "function": {"name": "set_color", "description": "Record the main color of an image.",
                                               "parameters": {"type": "object", "properties": {"color": {"type": "string"}},
                                                              "required": ["color"]}}}]
    st, raw = post(base + "/v1/chat/completions", body(
        msg(red, "Look at the image and call set_color with the main shape's color."), tools=tools))
    jt = json.loads(raw) if st == 200 else {}
    m = (jt.get("choices") or [{}])[0].get("message", {})
    call = (m.get("tool_calls") or [None])[0]
    check("tool call with an image in the turn", st == 200 and (call is not None or a.no_content),
          json.dumps(call)[:160] if call else (m.get("content") or "")[:120])
    if call and not a.no_content:
        args = json.loads(call["function"]["arguments"])
        check("tool call argument is the color", "red" in str(args.get("color", "")).lower(), str(args))

    st, _ = post(base + "/v1/chat/completions", body(msg("data:image/png;base64,AAAA")))
    check("undecodable image answers 400", st == 400, str(st))
    st, _ = post(base + "/v1/chat/completions", body(msg("/etc/hostname")))
    check("local path answers 400", st == 400, str(st))
    bad = [n for n, ok, _ in results if not ok]
    print("ALL PASS" if not bad else f"FAILED: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
