"""Eval docs for the parity/KV stages, generic: domain texts cut from the calibration corpus + chat/tool docs rendered with the PACK's
own chat template (adapted from nanb mk_docs.py). Tokenization uses the pack's tokenizer.json (tokenizers package)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from load import CAL_DIR

DOMAINS = {"wiki": ("wiki.utf8", 1_900_000), "code": ("code.utf8", 700_000), "fr": ("multilingual.utf8", 28_000), "tech": ("technical.utf8", 240_000)}
CHAT_FR = [{"role": "user", "content": "Écris une fonction Python qui fusionne des intervalles qui se chevauchent, avec une courte explication en français."},
           {"role": "assistant", "content": "Trie les intervalles par début, puis parcours-les une fois en étendant le dernier intervalle fusionné.\n\n```python\ndef merge(iv):\n    out = []\n    for s, e in sorted(iv):\n        if out and s <= out[-1][1]:\n            out[-1][1] = max(out[-1][1], e)\n        else:\n            out.append([s, e])\n    return out\n```\n\nComplexité O(n log n) à cause du tri."}]
CHAT_EN = [{"role": "user", "content": "Explain in three sentences why the sky is blue, then give a one-line analogy."},
           {"role": "assistant", "content": "Sunlight contains all colors. Air molecules scatter short wavelengths far more than long ones (Rayleigh scattering), so blue light reaches your eye from every direction. It is like a sieve that lets red through and bounces blue around."}]
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Get weather for a city.", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["C", "F"]}}, "required": ["city"]}}}]
TOOL_MSGS = [{"role": "user", "content": "What's the weather in Paris and Tokyo, in Celsius?"},
             {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Paris", "unit": "C"}}}, {"function": {"name": "get_weather", "arguments": {"city": "Tokyo", "unit": "C"}}}]},
             {"role": "tool", "content": "{\"city\": \"Paris\", \"temp\": 21}"}, {"role": "tool", "content": "{\"city\": \"Tokyo\", \"temp\": 17}"},
             {"role": "assistant", "content": "Paris is at 21 °C and Tokyo at 17 °C."}]


def render(template: str, messages: list, tools=None) -> str:
    from jinja2 import Environment
    env = Environment(); env.policies["json.dumps_kwargs"] = {"ensure_ascii": False}
    env.filters["tojson"] = lambda v, ensure_ascii=False: json.dumps(v, ensure_ascii=ensure_ascii)

    def rx(m):
        raise Exception(m)
    env.globals["raise_exception"] = rx
    return env.from_string(template).render(messages=messages, tools=tools, add_generation_prompt=False, preserve_thinking=True)


def make_docs(pack: str, n_tokens: int = 2048) -> dict:
    """{'names': [...], 'ids': [[...]]}: chat_fr, chat_en, tool, wiki, code, fr, tech (the weakest domain in drill 0 was wiki)."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(pack) / "tokenizer.json"))
    enc = lambda s: tok.encode(s, add_special_tokens=False).ids
    tpl = (Path(pack) / "chat_template.jinja").read_text(encoding="utf8")
    docs = {"chat_fr": enc(render(tpl, CHAT_FR)), "chat_en": enc(render(tpl, CHAT_EN)), "tool": enc(render(tpl, TOOL_MSGS, TOOLS))}
    for name, (f, off) in DOMAINS.items():
        docs[name] = enc((CAL_DIR / f).read_text(encoding="utf8")[off:off + n_tokens * 8])[:n_tokens]
    return dict(names=list(docs), ids=list(docs.values()))


def make_long_doc(pack: str, n_tokens: int = 32768) -> dict:
    """One 32K-token doc for the KV stage (wiki + code + fr + tech slices, as nanb's kv_kl.py used a 32K context)."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(pack) / "tokenizer.json"))
    ids = []
    for f, off in [("wiki.utf8", 1_000_000), ("code.utf8", 900_000), ("multilingual.utf8", 150_000), ("technical.utf8", 200_000)]:
        ids += tok.encode((CAL_DIR / f).read_text(encoding="utf8")[off:off + n_tokens * 3], add_special_tokens=False).ids[: n_tokens // 4]
    return dict(names=["long"], ids=[ids[:n_tokens]])


if __name__ == "__main__":      # run with the backend python (torch.save): docs.py short|long PACK OUT.pt
    import torch
    kind, pack, out = sys.argv[1:4]
    d = make_docs(pack) if kind == "short" else make_long_doc(pack)
    torch.save(d, out)
    print({n: len(i) for n, i in zip(d["names"], d["ids"])})
