#!/usr/bin/env python3
"""Fetch the small DeepSeek-V4.1-Flash-1017 reference files (config, tokenizer, official encoding).

Never downloads weight shards. Usage: python3 tools/ds41/_dl_ref.py DEST_DIR
"""
import sys

from huggingface_hub import hf_hub_download, list_repo_files

REPO = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
SINGLE = ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json", "README.md")

files = [f for f in list_repo_files(REPO) if f in SINGLE or f.startswith("encoding/")]
print(files)
for f in files:
    hf_hub_download(REPO, f, local_dir=sys.argv[1])
