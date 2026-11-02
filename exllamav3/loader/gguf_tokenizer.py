"""tokenizer.json + tokenizer_config.json from the tokenizer.ggml.* metadata of a GGUF (byte-level BPE: gpt2 model with
the qwen2 pre-tokenizer). Other tokenizer models / pre-tokenizers raise: nothing is guessed."""
from __future__ import annotations
import json
import os

from .gguf_collection import GGUFCollection

# ggml token types: 1 normal, 2 unknown, 3 control, 4 user-defined, 5 unused, 6 byte
_QWEN2_REGEX = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
_PRE = {"qwen2": _QWEN2_REGEX}


def write_tokenizer_files(g: GGUFCollection, outdir: str) -> None:
    m = g.meta
    model, pre = m.get("tokenizer.ggml.model"), m.get("tokenizer.ggml.pre")
    if model != "gpt2" or pre not in _PRE:
        raise ValueError(f"GGUF tokenizer model={model!r} pre={pre!r} is not mapped (have gpt2 + {sorted(_PRE)})")
    toks, types = m["tokenizer.ggml.tokens"], m["tokenizer.ggml.token_type"]
    merges = [tuple(x.split(" ", 1)) for x in m["tokenizer.ggml.merges"]]
    added = [{"id": i, "content": t, "single_word": False, "lstrip": False, "rstrip": False, "normalized": False,
              "special": ty == 3} for i, (t, ty) in enumerate(zip(toks, types)) if ty in (3, 4)]
    tj = {
        "version": "1.0", "truncation": None, "padding": None, "added_tokens": added, "normalizer": None,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": _PRE[pre]}, "behavior": "Isolated", "invert": False},
            {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False}]},
        "post_processor": None,
        "decoder": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False},
        "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": "",
                  "end_of_word_suffix": "", "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
                  "vocab": {t: i for i, t in enumerate(toks)}, "merges": [list(x) for x in merges]},
    }
    with open(os.path.join(outdir, "tokenizer.json"), "w", encoding="utf8") as f:
        json.dump(tj, f, ensure_ascii=False)

    def tok(key):
        i = m.get(key)
        return toks[i] if i is not None else None

    tc = {"tokenizer_class": "PreTrainedTokenizerFast", "model_max_length": 131072, "clean_up_tokenization_spaces": False,
          "add_bos_token": bool(m.get("tokenizer.ggml.add_bos_token", False)),
          "eos_token": tok("tokenizer.ggml.eos_token_id"), "pad_token": tok("tokenizer.ggml.padding_token_id"),
          "bos_token": tok("tokenizer.ggml.bos_token_id"), "unk_token": None}
    if "tokenizer.chat_template" in m:
        tc["chat_template"] = m["tokenizer.chat_template"]
    with open(os.path.join(outdir, "tokenizer_config.json"), "w", encoding="utf8") as f:
        json.dump(tc, f, ensure_ascii=False, indent=1)
