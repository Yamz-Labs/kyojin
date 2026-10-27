"""Shared helpers of the DS4.1 tests (no test cases here): the official reference, the tokenizer lookup, serve.py."""
from __future__ import annotations

import copy
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ID = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def official():
    """The vendored official encoding.py (module name `dsv41_official`)."""
    if "dsv41_official" in sys.modules:
        return sys.modules["dsv41_official"]
    return load_module("dsv41_official", HERE / "testdata" / "reference" / "encoding.py")


def serve_module():
    """tools/ds41/serve.py under a private name: tools/glm/serve.py is also called `serve`."""
    if "ds41_serve" in sys.modules:
        return sys.modules["ds41_serve"]
    return load_module("ds41_serve", HERE / "serve.py")


def tokenizer_json() -> Path | None:
    """tokenizer.json of the model: $DS41_REF_DIR, then tools/ds41/_ref (fetch_reference.py), then the Hugging Face cache
    or one download of that single file. None when none of them works (the id tests then skip)."""
    for d in (os.environ.get("DS41_REF_DIR"), str(HERE / "_ref")):
        if d and (Path(d) / "tokenizer.json").is_file():
            return Path(d) / "tokenizer.json"
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None
    for local_only in (True, False):
        try:
            return Path(hf_hub_download(REPO_ID, "tokenizer.json", local_files_only=local_only))
        except Exception:                                           # noqa: BLE001
            continue
    return None


def reference_messages(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """What the reference wants for an OpenAI request: tools on a leading system message, `developer` spelled `system`."""
    msgs = copy.deepcopy(messages)
    for m in msgs:
        if m["role"] == "developer":
            m["role"] = "system"
    if tools:
        if msgs[0]["role"] != "system":
            msgs.insert(0, {"role": "system", "content": ""})
        msgs[0]["tools"] = copy.deepcopy(tools)
    return msgs


def reference_prompt(messages, tools, thinking: bool, effort=None, drop_thinking: bool = True) -> str:
    enc = official()
    return enc.encode_messages(reference_messages(messages, tools), "thinking" if thinking else "chat",
                               drop_thinking=drop_thinking, reasoning_effort=effort)
