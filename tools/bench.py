#!/usr/bin/env python3
"""One-command benchmark against a running Kyojin server (OpenAI-style API).

    python tools/bench.py                      # http://localhost:8000/v1, model taken from /v1/models
    python tools/bench.py --base http://localhost:8000/v1 --reps 3

Standard library only. Method and decode prompts are the ones behind the numbers in README.md:

  prefill tok/s = prompt_tokens / wall time      (max_tokens=1, temperature 0, a unique prompt each run)
  decode  tok/s = (completion_tokens - 1) / (last streamed chunk - first streamed chunk)
                  128 new tokens, temperature 0, three different prompts each for prose, chat and code

The prefill prompt is about 3.5K tokens of English text taken from the Markdown files of this
repository (the published figures used a WikiText-2 slice; pass --corpus <file> to use any text
file). Each request starts with a unique tag so no prompt cache can answer it. The first request
after a build or a first launch is slow while the kernels tune: a short warm-up runs first, and
you should run the script twice on a fresh install.

Prints one Markdown block to paste into a benchmark report issue. With --json, a fenced JSON block with the
same measurements (and every run) follows it, so results can be collected into a table.
"""
import argparse
import json
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DECODE_PROMPTS = {
    "prose": [
        "Write a 300-word story about a lighthouse.",
        "Write a 300-word story about a lighthouse. Use a different keeper and a different coast.",
        "Tell a 300-word story set in a lighthouse during a winter storm, with a different ending.",
    ],
    "chat": [
        "Explain how a hash map works, with its time complexity, in about 300 words.",
        "Explain how a hash map works and where it degrades, in about 300 words.",
        "Explain hash maps to a backend engineer: buckets, collisions, resizing. About 300 words.",
    ],
    "code": [
        "Write a Python function that parses an ISO-8601 duration string into seconds, with tests.",
        "Write a Python function that parses an ISO-8601 duration into seconds, handling negative "
        "durations, with unit tests.",
        "Write a Python function parsing ISO-8601 durations (P[n]Y[n]M[n]DT[n]H[n]M[n]S) into a "
        "timedelta, with tests for every unit.",
    ],
}
PREFILL_TARGET_TOKENS = 3500


def post(base, body, timeout=3600):
    req = urllib.request.Request(base + "/chat/completions", json.dumps(body).encode(),
                                 {"content-type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def chat(prompt):
    return [{"role": "user", "content": prompt}]


def prefill(base, model, text):
    t0 = time.perf_counter()
    d = json.load(post(base, {"model": model, "messages": chat(text), "max_tokens": 1, "temperature": 0}))
    wall = time.perf_counter() - t0
    n = d["usage"]["prompt_tokens"]
    return n, n / wall


def decode(base, model, prompt, max_tokens):
    body = {"model": model, "messages": chat(prompt), "max_tokens": max_tokens, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True}}
    first = last = None
    n_chunks = 0
    usage = None
    for raw in post(base, body):
        line = raw.decode().strip()
        if not line.startswith("data: {"):
            continue
        d = json.loads(line[6:])
        usage = d.get("usage") or usage
        delta = (d.get("choices") or [{}])[0].get("delta") or {}
        if delta.get("content") or delta.get("reasoning_content"):
            now = time.perf_counter()
            first = first or now
            last = now
            n_chunks += 1
    toks = (usage or {}).get("completion_tokens") or n_chunks
    if not last or last <= first or toks < 2:
        return None
    return (toks - 1) / (last - first)


def load_words(corpus):
    files = [Path(corpus)] if corpus else sorted(ROOT.rglob("*.md"))
    words = []
    for f in files:
        if any(p in f.parts for p in (".venv", ".git", "node_modules")):
            continue
        words += f.read_text(encoding="utf-8", errors="ignore").split()
    return words


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def cpu_name():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def ram_gib():
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                return f"{int(line.split()[1]) / 1048576:.0f} GiB"
    except OSError:
        pass
    return "unknown"


def server_compiler(base):
    """(compiler line, warning or None) the server reports on /health (the server root, one level above /v1)."""
    root = re.sub(r"/v1$", "", base)
    try:
        h = json.load(urllib.request.urlopen(root + "/health", timeout=10))
    except Exception:
        return "unknown", None
    return h.get("hip_compiler") or "unknown (server does not report it)", h.get("hip_compiler_warning")


def rocm_version():
    for f in ("/opt/rocm/.info/version", "/opt/rocm/.info/version-dev"):
        if Path(f).is_file():
            return Path(f).read_text().strip()
    for cmd in ("rocm-sdk version", "hipconfig --version"):
        if shutil.which(cmd.split()[0]):
            out = sh(cmd)
            if out:
                return out.splitlines()[0]
    out = sh(f"{sys.executable} -c 'import torch; print(torch.version.hip or \"\")'")
    return f"HIP {out} (PyTorch)" if out else "unknown"


KFD_NODES = Path("/sys/class/kfd/kfd/topology/nodes")


def gfx_name(target_version):
    """KFD gfx_target_version -> LLVM target name: 110501 -> gfx1151, 90010 -> gfx90a (minor and stepping are hex digits)."""
    major, minor, stepping = target_version // 10000, target_version // 100 % 100, target_version % 100
    return f"gfx{major}{minor:x}{stepping:x}"


def kfd_gpu_target(nodes=KFD_NODES):
    """GPU target from the kernel's KFD topology, which needs no ROCm tools. Returns an empty string when no GPU node is found."""
    try:
        dirs = sorted((d for d in Path(nodes).iterdir() if d.name.isdecimal()), key=lambda d: int(d.name))
    except OSError:
        return ""
    for d in dirs:
        try:
            m = re.search(r"^gfx_target_version (\d+)$", (d / "properties").read_text(), re.M)
        except (OSError, UnicodeDecodeError):
            continue
        if m and int(m.group(1)):                       # CPU nodes report 0
            return gfx_name(int(m.group(1)))
    return ""


def gpu_name():
    out = sh("rocminfo 2>/dev/null | grep -m1 -o 'gfx[0-9a-z]*'")
    return out or kfd_gpu_target() or "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default="http://localhost:8000/v1", help="OpenAI-style base URL")
    ap.add_argument("--model", help="model id (default: first id from /models)")
    ap.add_argument("--reps", type=int, default=3, help="runs per measurement (default 3)")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--corpus", help="text file for the prefill prompt (default: Markdown files of this repository)")
    ap.add_argument("--json", action="store_true", help="also print the measurements as a fenced JSON block")
    a = ap.parse_args()
    base = a.base.rstrip("/")
    reps = max(1, min(a.reps, 3))

    model = a.model
    if not model:
        try:
            model = json.load(urllib.request.urlopen(base + "/models", timeout=30))["data"][0]["id"]
        except Exception as e:
            sys.exit(f"cannot reach {base}/models ({e}). Start the server first (see the README quickstart).")

    words = load_words(a.corpus)
    if len(words) < 8000:
        sys.exit("not enough text for the prefill prompt: pass --corpus <file> with at least 8000 words")
    rng = random.Random(1234)
    tag = f"{int(time.time())}"

    def slice_text(n_words, salt):
        s = rng.randrange(0, len(words) - n_words)
        return f"[{salt}] " + " ".join(words[s:s + n_words]) + "\nSummarize in one word."

    print(f"warm-up on {model} at {base} ...", file=sys.stderr)
    n, _ = prefill(base, model, slice_text(300, f"{tag}-warm"))
    decode(base, model, DECODE_PROMPTS["chat"][0], 32)
    ratio = n / 300.0                                   # tokens per word on this text
    n_words = max(500, min(len(words) - 1, int(PREFILL_TARGET_TOKENS / ratio)))

    pf = []
    for i in range(reps):
        print(f"prefill {i + 1}/{reps}", file=sys.stderr)
        pf.append(prefill(base, model, slice_text(n_words, f"{tag}-{i}")))
    dec = {}
    for i in range(reps):                                # categories interleaved
        for cat, prompts in DECODE_PROMPTS.items():
            print(f"decode {cat} {i + 1}/{reps}", file=sys.stderr)
            v = decode(base, model, prompts[i], a.max_tokens)
            if v is not None:
                dec.setdefault(cat, []).append(v)

    def med(xs):
        return round(statistics.median(xs), 1) if xs else None

    def fmt(x):
        return f"{x:.1f}" if x is not None else "n/a"

    ptok = int(statistics.median(n for n, _ in pf))
    comp = server_compiler(base)
    if comp[1]:
        print(comp[1], file=sys.stderr)
    info = {"model": model, "cpu": cpu_name(), "ram": ram_gib(), "gpu_target": gpu_name(), "kernel": platform.release(),
            "rocm": rocm_version(), "kernel_compiler": comp[0], "server": base, "date": time.strftime('%Y-%m-%d')}
    prefill_med = med([t for _, t in pf])
    lines = [
        "### Kyojin benchmark",
        "",
        f"- Model: `{info['model']}`",
        f"- CPU: {info['cpu']}",
        f"- RAM: {info['ram']}",
        f"- GPU target: {info['gpu_target']}",
        f"- Kernel: {info['kernel']}",
        f"- ROCm: {info['rocm']}",
        f"- Kernel compiler: {info['kernel_compiler']}",
        f"- Server: {info['server']}",
        f"- Date: {info['date']}",
        "",
        "| Measurement | Result |",
        "|---|---|",
        f"| Prefill, ~{ptok} tokens (median of {len(pf)}) | {fmt(prefill_med)} tok/s |",
    ]
    for cat in DECODE_PROMPTS:
        xs = dec.get(cat, [])
        lines.append(f"| Decode, {cat}, {a.max_tokens} tokens, temperature 0 (median of {len(xs)}) | {fmt(med(xs))} tok/s |")
    lines += ["", "Method: prefill = prompt tokens / wall time (1 output token); decode = (tokens - 1) / streaming time. "
                  "Same prompts as the numbers in the README."]
    if a.json:
        result = info | {
            "prefill": {"prompt_tokens": ptok, "max_tokens": 1, "temperature": 0, "tok_s": prefill_med,
                        "runs": [round(t, 1) for _, t in pf]},
            "decode": {cat: {"max_tokens": a.max_tokens, "temperature": 0, "tok_s": med(dec.get(cat, [])),
                             "runs": [round(v, 1) for v in dec.get(cat, [])]} for cat in DECODE_PROMPTS},
        }
        lines += ["", "```json", json.dumps(result, indent=2), "```"]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
