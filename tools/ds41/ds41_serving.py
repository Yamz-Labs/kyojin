"""The DS4.1 SERVING model (converted EXL3 pack) behind the server:

    python3 tools/ds41/serve.py --model <pack-folder> --loader ds41_serving:load_model -c 131072 [--num-draft 3]

`load_model(path, ctx, **options)` loads the pack with `exllamav3.dsv41.pack.load` (framework Config / modules, EXL3 linears,
mixed-K grouped MoE on the GPU, the ds41kern kernels for the small ops) and wraps the resulting `DSV41Serving` in the same
engine as the port loader (`ds41_model.PortEngine`: sampling, incremental detokenizer, cancel, DSpark behind `--num-draft`).
Prefill is chunked (`prefill_chunk` tokens per pass), so a 100k-token prompt needs the cache of ~0.4 GB, not a full-context pass.

Loader options (`--engine-opt KEY=VALUE`): num_draft (set by --num-draft), device ("cuda:0"), dtype ("float16" default,
"float32" for CPU checks), layers (first N trunk layers only: smoke), mtp (false: skip the dspark stages), moe ("auto" or
"loop"), prefill_chunk (2048), engram_dir (folder with `layers.N.engram.embed.{weight,scale}.bin` when the shards do not hold
the tables), threads (CPU torch threads), seed, draft_auto (auto draft length, set by --draft-auto),
draft_conf_threshold (stop drafting below this confidence probability, set by --draft-conf), vision (auto | true | false:
image input when the pack holds the vision tower), vision_device, vision_dtype (bfloat16), vision_cache (8).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for _p in (HERE.parent.parent,):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))

from ds41_engine import HFTokenizer  # noqa: E402
from ds41_model import PortEngine, _hf_eos, vision_kwargs  # noqa: E402


def build_model(path: str, *, device: str = "cuda:0", dtype: str = "float16", layers: int | None = None,
                mtp: bool = True, moe: str = "auto", prefill_chunk: int = 2048, engram_dir: str | None = None):
    import torch
    from exllamav3.dsv41.pack import load
    store = None
    if engram_dir:
        from exllamav3.dsv41.config import DSV41Config
        from exllamav3.dsv41.engram import RawBinEngramStore
        cfg = DSV41Config.from_file(str(Path(path).expanduser() / "config.json"))
        store = RawBinEngramStore.open(engram_dir, cfg.engram_layer_ids, cfg.engram_head_dim, "auto",
                                       allow_buffered=True)
    return load(str(Path(path).expanduser()), device=device, dtype=getattr(torch, dtype), layers=layers, mtp=mtp,
                moe=moe, prefill_chunk=int(prefill_chunk), engram_store=store)


def _flag(v) -> bool:
    return v if isinstance(v, bool) else str(v).lower() not in ("0", "false", "no", "off", "")


def load_model(path: str, ctx: int, *, num_draft: int = 0, device: str = "cuda:0", dtype: str = "float16",
               layers: int | None = None, mtp: Any = True, moe: str = "auto", prefill_chunk: int = 2048,
               engram_dir: str | None = None, threads: int | None = None, seed: int | None = None,
               draft_auto: Any = False, draft_conf_threshold: float | None = None, vision: Any = "auto", vision_device: str | None = None,
               vision_dtype: str = "bfloat16", vision_cache: int = 8, **_: Any) -> PortEngine:
    import torch
    if threads:
        torch.set_num_threads(int(threads))
    root = Path(path).expanduser()
    model = build_model(str(root), device=device, dtype=dtype, layers=int(layers) if layers else None,
                        mtp=_flag(mtp) or bool(num_draft), moe=moe, prefill_chunk=prefill_chunk, engram_dir=engram_dir)
    tk = HFTokenizer(root / "tokenizer.json")
    return PortEngine(model, tk, ctx=int(ctx), num_draft=int(num_draft or 0), eos_id=_hf_eos(root, tk._tk), seed=seed,
                      draft_auto=_flag(draft_auto), draft_conf_threshold=draft_conf_threshold,
                      **vision_kwargs(root, vision, vision_device, vision_dtype, vision_cache))
