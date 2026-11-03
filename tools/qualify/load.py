"""Served-side measurements, stdlib only. Adapted from nanb bench.py/agent.py, rel174c3 drv.py/bench_n.py, gdrv.py (sha set) and
lane-kv-conformance.py (rule 1: tools rendered before the system text). Prompts are cut from a plain-text corpus; the chars/token
ratio is calibrated on the server's own usage.prompt_tokens, so no tokenizer import is needed."""
from __future__ import annotations

import hashlib
import json
import random
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

CAL_DIR = Path(__file__).resolve().parents[2] / "exllamav3/conversion/standard_cal_data"
CORPORA = ("c4.utf8", "wiki.utf8", "code.utf8", "technical.utf8", "multilingual.utf8")
FR = ["Explique en quatre ou cinq phrases pourquoi le ciel est bleu, puis donne une analogie simple.",
      "Écris un court poème sur la mer en six vers, puis explique le choix des rimes.",
      "Décris les étapes pour préparer une bonne pâte à crêpes, avec les quantités pour six personnes."]
NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}, "enable_thinking": False}


# Decode-speed prompt kinds of the draft A/B (nanbdf bench kinds): drafters gain on copy-heavy agentic output, little on free chat.
CODE_Q = "Write a Python module with an LRU cache class (get/put/evict, O(1)), type hints, docstrings and 5 pytest tests. Output only code."
CHAT_Q = ["Raconte-moi une histoire courte sur un phare et un renard, avec des dialogues.", "Quels sont les avantages et les inconvénients du télétravail ? Réponds en un paragraphe argumenté."]


def agentic_prompt(n_lines: int = 70) -> str:
    """Copy-heavy turn typical of tool use: re-emit a file with two renamed identifiers (most output tokens repeat the input)."""
    src = "\n".join(f"def handler_{i}(request, ctx):\n    value = ctx.get('key_{i}', {i})\n    return {{'id': {i}, 'ok': value is not None}}\n" for i in range(n_lines // 3))
    return f"Voici un fichier Python.\n\n```python\n{src}```\n\nRéécris ce fichier en entier, à l'identique, mais en renommant handler_3 en process_three et handler_7 en process_seven. Donne uniquement le code."


class Client:
    def __init__(self, port: int, model: str = "qual", corpus_dir: Path = CAL_DIR):
        self.url = f"http://127.0.0.1:{port}"
        self.model = model
        self.big = "\n\n".join((corpus_dir / n).read_text(encoding="utf8", errors="replace") for n in CORPORA if (corpus_dir / n).exists())
        if len(self.big) < 1_000_000:
            raise RuntimeError(f"no calibration corpus under {corpus_dir}")
        self.cpt = 3.2          # chars per token, recalibrated by calibrate()
        self.salt = 0

    def post(self, path: str, body: dict, timeout: int = 3000):
        return urllib.request.Request(self.url + path, json.dumps(body).encode(), {"Content-Type": "application/json", "Authorization": "Bearer x"})

    def stream(self, prompt: str, max_tokens: int, temperature: float = 0.0) -> dict:
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": temperature,
                "stream": True, "stream_options": {"include_usage": True}, **NOTHINK}
        t0 = time.time(); first = None; usage = None; txt = []; err = None
        try:
            for line in urllib.request.urlopen(self.post("/v1/chat/completions", body), timeout=3000):
                line = line.decode().strip()
                if not line.startswith("data:") or line.endswith("[DONE]"):
                    continue
                j = json.loads(line[5:])
                if j.get("error"):
                    err = str(j["error"])[:300]
                usage = j.get("usage") or usage
                for c in j.get("choices", []):
                    d = c.get("delta", {})
                    s = (d.get("content") or "") + (d.get("reasoning_content") or "")
                    if s:
                        first = first or time.time(); txt.append(s)
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code}: {e.read().decode(errors='replace')[:240]}"
        except Exception as e:
            err = repr(e)[:300]
        te = time.time(); ct = (usage or {}).get("completion_tokens", 0)
        return dict(ptok=(usage or {}).get("prompt_tokens"), ntok=ct, ttft=round((first or te) - t0, 3), total=round(te - t0, 2),
                    dec_tps=round((ct - 1) / (te - first), 2) if first and te > first and ct > 1 else None,
                    sha=hashlib.sha1("".join(txt).encode()).hexdigest()[:10], err=err, t0=t0, first=first, end=te)

    def kind_prompts(self) -> dict:
        return dict(code=[CODE_Q], chat=CHAT_Q, agentic=[agentic_prompt()])

    def prompt(self, tokens: int, slot: int) -> str:
        """~tokens of corpus text, distinct for every (slot, call): fresh offsets so no prefix cache can answer."""
        self.salt += 1
        T = self.big
        n = int(tokens * self.cpt)
        if n + 1000 > len(T):
            raise ValueError(f"prompt of {tokens} tokens needs {n} chars, corpus has {len(T)}")
        off = (30_000 + slot * 337_000 + self.salt * 91_003) % max(1, len(T) - n - 10)
        return f"[{slot}.{self.salt}] Voici un document.\n\n{T[off:off + n]}\n\nEn deux phrases, dis quels sujets ce document mélange."

    def calibrate(self, tokens: int = 2000) -> float:
        r = self.stream(self.prompt(tokens, 0), 1)
        if r["ptok"]:
            self.cpt = self.cpt * tokens / r["ptok"]
        return self.cpt

    def near_full(self, ctx: int, slot: int, fill: float = 0.97, reply: int = 32) -> dict:
        """One request whose prompt is `fill` of -c (rounded down so prompt + reply < ctx). Corpora differ in chars/token (code and CJK are denser), so a
        refused prompt (HTTP 400 context overflow) is shrunk and a short one is topped up, up to 4 times, from the server's own usage.prompt_tokens."""
        return self.fit(int(ctx * fill) - reply, slot, reply)

    def fit(self, want: int, slot: int, reply: int, topup: bool = True) -> dict:
        """Stream a prompt of `want` tokens: corpora differ in chars/token (46K tokens came back for a 35K ask), so on HTTP 400 the length is
        rescaled from the server's stated count and, with topup, a short prompt is lengthened (up to 4 tries)."""
        n = want
        for _ in range(4):
            r = self.stream(self.prompt(n, slot), reply)
            if r["err"] and "400" in r["err"] and n > 256:
                m = re.search(r"prompt is (\d+) tokens", r["err"])
                n = int(n * want / int(m.group(1)) * 0.99) if m else int(n * 0.9)      # the server states the real count: rescale at once
            elif topup and not r["err"] and r["ptok"] and r["ptok"] < 0.9 * want:
                n = int(n * want / r["ptok"])
            else:
                break
        return r

    def concurrent(self, n: int, tokens: int, max_tokens: int = 128) -> dict:
        res = [None] * n
        bar = threading.Barrier(n)

        def run(i):
            bar.wait(); res[i] = self.fit(tokens, i, max_tokens, topup=False)
        th = [threading.Thread(target=run, args=(i,)) for i in range(n)]
        [t.start() for t in th]; [t.join() for t in th]
        ok = [r for r in res if not r["err"] and r["dec_tps"]]
        span = (max(r["end"] for r in ok) - min(r["first"] for r in ok)) if ok else 0
        return dict(n=n, ptok=[r["ptok"] for r in res], ttft=[r["ttft"] for r in res], dec_tps=[r["dec_tps"] for r in res],
                    agg_dec_tps=round(sum(r["ntok"] - 1 for r in ok) / span, 1) if span > 0 else None, errors=[r["err"] for r in res if r["err"]])

    def soak(self, sessions: int, ctx: int, seconds: float, seed: int = 7, vram=None) -> dict:
        """Agent-shaped load against a running server: `sessions` workers loop for `seconds` over a mixed prompt-length distribution
        (70 % short 200-3K, 25 % medium up to half a share, 5 % a near-full share), short replies, so every size follows every other.
        Drill 0: the synthetic near-full probes passed, the real agent mix OOMed 1-3 h later (duel: Nanbeige -c 73,728 x3 -> 61,440 x2 -> 57,344)."""
        share = max(1024, ctx // sessions - 512)
        rng = random.Random(seed)
        lock = threading.Lock()
        res, curve = [], []
        end = time.time() + seconds

        def pick():
            with lock:
                u = rng.random()
                if u < 0.70:
                    return rng.randint(200, min(3000, share))
                if u < 0.95:
                    return rng.randint(min(3000, share), max(3000, share // 2))
                return int(share * 0.95)

        def worker(i):
            while time.time() < end:
                n = pick()
                r = self.fit(n, 10 + i, 48, topup=False)
                with lock:
                    res.append(dict(want=n, ptok=r["ptok"], err=r["err"], ttft=r["ttft"]))
                    if vram and len(res) % 5 == 0:
                        curve.append(vram())
        th = [threading.Thread(target=worker, args=(i,)) for i in range(sessions)]
        [t.start() for t in th]; [t.join() for t in th]
        errs = [r["err"] for r in res if r["err"]]
        return dict(requests=len(res), errors=errs[:3], n_errors=len(errs), ptok_max=max((r["ptok"] or 0) for r in res), vram_curve=curve, seconds=seconds)

    def sha_set(self) -> list:
        return [{k: r[k] for k in ("sha", "ntok", "dec_tps", "err")} for r in (self.stream(p, 256) for p in FR)]

    def conformance(self) -> dict:
        """lane-kv-conformance rule 1: the template renders TOOLS BEFORE the system text (else every new session re-pays the tool schemas)."""
        msgs = [{"role": "system", "content": "SYSMARK"}, {"role": "user", "content": "USERMARK"}]
        tools = [{"type": "function", "function": {"name": "TOOLMARK", "description": "d", "parameters": {"type": "object", "properties": {}}}}]
        try:
            r = json.load(urllib.request.urlopen(self.post("/apply-template", {"messages": msgs, "tools": tools}), timeout=60))
            p = r["prompt"]; a, b = p.find("TOOLMARK"), p.find("SYSMARK")
            return dict(ok=0 <= a < b, detail=f"tools@{a} sys@{b}")
        except Exception as e:
            return dict(ok=False, detail=repr(e)[:200])
