"""Real-use Hermes smoke. Primary: hermes_opt/eval (30 hermetic tasks, programmatic graders) through the real `hermes chat` CLI against the
served port (run_arm.sh ... --allow-local --jobs 1 --tasks a,b). Fallback when the suite is incomplete/absent: the nanb scripted tool loop
(read/write/shell in a temp sandbox, fix 3 bugs, answer in French), adapted from fleet/scratch_nanb/tools/agent.py."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

EVAL_DIRS = [Path(os.environ["QUALIFY_EVAL_DIR"])] if os.environ.get("QUALIFY_EVAL_DIR") else []
DEFAULT_TASKS = ""        # filled by pick_tasks from the suite's own task list


def find_eval() -> Path | None:
    """The suite needs lib/taskdefs.py, so check for it."""
    for d in EVAL_DIRS:
        if (d / "run_arm.sh").exists() and (d / "lib/taskdefs.py").exists():
            return d
    return None


def run_hermes_eval(eval_dir: Path, port: int, model: str, tasks: str, arm: str, timeout: int = 3600) -> dict:
    env = {**os.environ, "EVAL_API_KEY": "x"}
    cmd = ["bash", str(eval_dir / "run_arm.sh"), arm, "-", f"http://127.0.0.1:{port}/v1", model, "1", "--allow-local", "--jobs", "1"] + (["--tasks", tasks] if tasks else [])
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
    f = eval_dir / "runs" / arm / "rep1.json"
    if not f.exists():
        return dict(kind="hermes_eval", total=0, passed=0, error=(r.stdout + r.stderr)[-400:])
    data = json.loads(f.read_text())
    rows = data if isinstance(data, list) else data.get("results") or data.get("tasks") or []
    ok = [x for x in rows if isinstance(x, dict) and x.get("success")]
    return dict(kind="hermes_eval", total=len(rows), passed=len(ok), tasks=[{k: x.get(k) for k in ("id", "task", "score", "success", "tool_calls")} for x in rows])


_BUGGY = '''def parse_line(line):
    """'name,qty,price' -> (name, int qty, float price)"""
    name, qty, price = line.split(",")
    return name, int(qty), float(price)

def total(lines):
    """total value of all items, rounded to 2 decimals; blank lines and '#' comments are skipped"""
    s = 0
    for l in lines:
        n, q, p = parse_line(l)
        s += q * p
    return round(s, 2)

def low_stock(lines, threshold=5):
    """names of items whose qty is below threshold, sorted alphabetically"""
    out = []
    for l in lines:
        n, q, p = parse_line(l)
        if q <= threshold:
            out.append(n)
    return out
'''
_TESTS = '''import unittest
from inventory import total, low_stock
L = ["# stock", "apple, 10, 0.5", "", "pear,3,1.25", "fig,5,2.0", "  kiwi , 0 , 3.1"]
class T(unittest.TestCase):
    def test_total(self): self.assertEqual(total(L), 18.75)
    def test_low(self): self.assertEqual(low_stock(L), ["kiwi", "pear"])
    def test_low_thr(self): self.assertEqual(low_stock(L, 10), ["fig", "kiwi", "pear"])
if __name__ == "__main__": unittest.main()
'''
_TOOLS = [{"type": "function", "function": {"name": n, "description": d, "parameters": {"type": "object", "properties": pr, "required": list(pr)}}} for n, d, pr in [
    ("read_file", "Read a text file in the working directory.", {"path": {"type": "string"}}),
    ("write_file", "Overwrite a file in the working directory with the given content.", {"path": {"type": "string"}, "content": {"type": "string"}}),
    ("run_shell", "Run a shell command in the working directory and return its output.", {"command": {"type": "string"}})]]


def run_scripted_agent(port: int, model: str, think: bool = False, max_turns: int = 14) -> dict:
    sb = tempfile.mkdtemp(prefix="qual_agent_")
    Path(sb, "inventory.py").write_text(_BUGGY); Path(sb, "test_inventory.py").write_text(_TESTS)

    def call(name, a):
        try:
            if name == "read_file":
                return Path(sb, a["path"]).read_text()
            if name == "write_file":
                Path(sb, a["path"]).write_text(a["content"]); return "ok"
            if name == "run_shell":
                r = subprocess.run(a["command"], shell=True, cwd=sb, capture_output=True, text=True, timeout=60); return (r.stdout + r.stderr)[-4000:]
        except Exception as e:
            return f"error: {e}"
        return "unknown tool"
    msgs = [{"role": "system", "content": "You are a coding agent. Use the tools to work in the project directory. Be concise."},
            {"role": "user", "content": "Les tests de test_inventory.py échouent. Corrige inventory.py (pas les tests), lance les tests jusqu'à ce qu'ils passent, puis réponds-moi en français en 2 phrases : quels bugs et comment tu les as corrigés."}]
    t0 = time.time(); ncalls = 0; toks = 0; err = None; final = None
    try:
        for turn in range(max_turns):
            body = {"model": model, "messages": msgs, "tools": _TOOLS, "max_tokens": 16000, "temperature": 0.6, "top_p": 0.95,
                    **({} if think else {"chat_template_kwargs": {"enable_thinking": False}, "enable_thinking": False})}
            try:
                r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900))
            except Exception as e:
                err = repr(e)[:300]; break
            m = r["choices"][0]["message"]; toks += r.get("usage", {}).get("completion_tokens", 0)
            msgs.append({k: v for k, v in m.items() if k in ("role", "content", "tool_calls", "reasoning_content") and v is not None})
            tc = m.get("tool_calls")
            if not tc:
                final = m.get("content"); break
            for c in tc:
                ncalls += 1
                try:
                    args = json.loads(c["function"]["arguments"])
                except Exception:
                    args = {}
                msgs.append({"role": "tool", "tool_call_id": c["id"], "content": call(c["function"]["name"], args)})
        ok = subprocess.run([sys.executable, "-I", "-m", "unittest", "-q"], cwd=sb, capture_output=True, text=True).returncode == 0
    finally:
        shutil.rmtree(sb, ignore_errors=True)
    return dict(kind="scripted_agent", total=1, passed=int(ok and err is None), tool_calls=ncalls, completion_tokens=toks, wall_s=round(time.time() - t0), error=err, final=(final or "")[:300])
