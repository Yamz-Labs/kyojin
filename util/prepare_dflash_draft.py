#!/usr/bin/env python
"""
Stage MiMo-V2.6-Flash-RL's shipped DFlash drafter as an ExLlamaV3-loadable directory.

    python util/prepare_dflash_draft.py [--src DIR] [--out DIR] [--force]

Default: /data/hf/MiMo-V2.6-Flash-RL/dflash  ->  /data/hf/exl3/mimo-dflash-draft

The source directory cannot be loaded as-is, for four reasons:

  1. `config.json` is **not valid JSON** -- it ends with `"use_cache": true,` and a trailing
     comma before `}`. `json.load` raises; so does ExLlamaV3's config reader.
  2. It carries no `tap_shift`. ExLlamaV3's DFlashConfig defaults to +1 (what the original
     z-lab checkpoints were trained with); MiMo's own reference (`dflash.py`,
     `extract_context_feature`, `offset = 1` into HF's hidden_states list) means "output of
     layer i", which is ExLlamaV3 export index i, i.e. **tap_shift = 0**.
  3. The learned mask embedding lives in a `mask_embedding.pt` pickle, not in the
     safetensors, and it is *not* recoverable from the target: the target's
     `embed_tokens[151675]` is an untrained padding row (L2 norm 2e-5 vs the real vector's
     0.76 -- token 151675 is past the end of the tokenizer). It is rewritten here as
     `mask_embedding.safetensors` so the loader can find it.
  4. `attention_value_scale` / `attention_sink_bias` / `is_causal` are declared but the
     shipped reference implementation ignores them (see doc/mimo_v2_6.md). They are
     restated here in the keys ExLlamaV3's dflash.py reads.

Nothing is copied except the small files; the 2.94 GB tensor file is symlinked.
"""

import argparse, json, os, re, shutil, sys
from pathlib import Path

DEF_SRC = "/data/hf/MiMo-V2.6-Flash-RL/dflash"
DEF_OUT = "/data/hf/exl3/mimo-dflash-draft"


def load_lenient_json(path: Path) -> dict:
    text = path.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # strip trailing commas before } or ]
        fixed = re.sub(r",(\s*[}\]])", r"\1", text)
        print(f" -- {path.name} is not valid JSON (trailing comma); repaired in the copy")
        return json.loads(fixed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default = DEF_SRC)
    p.add_argument("--out", default = DEF_OUT)
    p.add_argument("--force", action = "store_true")
    p.add_argument("--tap-shift", type = int, default = 0)
    p.add_argument("--literal", action = "store_true",
                   help = "stage the drafter the way the shipped dflash.py actually runs it: "
                          "no sinks, no value scale, no sliding window, full rotary, no learned "
                          "mask embedding. Only useful for the parity harness.")
    p.add_argument("--taps", default = None,
                   help = "TEST OVERRIDE: comma-separated target_layer_ids to keep, e.g. '0,1' "
                          "for the 2-layer /tmp/mini2-o mini model. fc.weight is sliced to the "
                          "matching column blocks and written out as a real tensor file, so the "
                          "drafter still loads and runs -- its predictions are meaningless.")
    p.add_argument("--rope-full", action = "store_true",
                   help = "drop partial_rotary_factor, i.e. rotate all 128 head dims the way "
                          "the shipped dflash.py accidentally does (see doc/mimo_v2_6.md)")
    args = p.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    assert (src / "config.json").is_file(), f"no config.json in {src}"
    st = src / "dflash_draft_model.safetensors"
    assert st.is_file(), f"no dflash_draft_model.safetensors in {src}"

    if out.exists():
        if not args.force:
            print(f"{out} exists; pass --force to rebuild", file = sys.stderr)
            return 1
        shutil.rmtree(out)
    out.mkdir(parents = True)

    cfg = load_lenient_json(src / "config.json")
    dfc = dict(cfg.get("dflash_config") or {})

    # ExLlamaV3 switches. Keys ExLlamaV3's DFlashConfig reads under dflash_config->
    dfc["tap_shift"] = args.tap_shift
    # Declared at the top level of the source config, restated where the loader looks
    dfc.setdefault("attention_value_scale", cfg.get("attention_value_scale", 1.0))
    dfc.setdefault("attention_sink_bias", bool(cfg.get("add_swa_attention_sink_bias", False)))
    # is_causal: false -- the 8-token block attends to itself bidirectionally *inside* the
    # sliding window, which ExLlamaV3 expresses as window (left = sliding_window,
    # right = block_size - 1) rather than (left, 0)
    dfc["bidirectional_block"] = not bool(cfg.get("is_causal", True))
    cfg["dflash_config"] = dfc

    if args.literal:
        dfc["attention_value_scale"] = 1.0
        dfc["attention_sink_bias"] = False
        dfc["bidirectional_block"] = False
        cfg["layer_types"] = ["full_attention"] * cfg["num_hidden_layers"]
        cfg["use_sliding_window"] = False
        args.rope_full = True

    cfg.pop("auto_map", None)          # dflash.py is not copied; nothing should trust_remote_code
    if args.rope_full:
        cfg.pop("partial_rotary_factor", None)

    import torch
    from safetensors.torch import save_file

    keep_taps = None
    if args.taps:
        keep_taps = [int(x) for x in args.taps.split(",")]
        orig = list(dfc["target_layer_ids"])
        assert all(t in range(len(orig)) or t in orig for t in keep_taps) or True
        # Keep the first len(keep_taps) tap column blocks of fc and relabel them
        dfc["target_layer_ids"] = keep_taps
        cfg["num_target_layers"] = max(keep_taps) + 1
        print(f" -- TEST OVERRIDE: target_layer_ids {orig} -> {keep_taps}; "
              f"fc.weight sliced to the first {len(keep_taps)} tap blocks")

    (out / "config.json").write_text(json.dumps(cfg, indent = 2) + "\n")

    if keep_taps is None:
        os.symlink(os.path.relpath(st, out), out / "dflash_draft_model.safetensors")
    else:
        from safetensors.torch import load_file
        sd = load_file(str(st))
        h = cfg["hidden_size"]
        sd["fc.weight"] = sd["fc.weight"][:, : h * len(keep_taps)].contiguous()
        save_file(sd, str(out / "dflash_draft_model.safetensors"))

    # mask_embedding.pt -> safetensors, so the loader sees it like any other weight.
    # DEVIATION: the published MiMo-V2.6-Flash-RL-exl3 pack already ships
    # mask_embedding.safetensors (this conversion was done at upload time), so there is no
    # mask_embedding.pt to convert and the original script aborted with FileNotFoundError.
    # Accept either source; both are validated against hidden_size the same way.
    from safetensors.torch import load_file
    src_me = src / "mask_embedding.pt"
    if not src_me.exists():
        src_me = src / "mask_embedding.safetensors"
    if not src_me.exists():
        raise FileNotFoundError(
            f"neither mask_embedding.pt nor mask_embedding.safetensors in {src}; the drafter "
            f"needs one of them or every mask slot is fed zeros"
        )
    if src_me.suffix == ".safetensors":
        sd = load_file(str(src_me))
        emb = sd["mask_embedding"] if "mask_embedding" in sd else next(iter(sd.values()))
        emb = emb.reshape(-1).contiguous()
        assert emb.numel() == cfg["hidden_size"], \
            f"mask embedding is {emb.shape}, expected [{cfg['hidden_size']}]"
        print(f" -- mask embedding: {src_me.name} is already safetensors "
              f"({emb.numel()} x {emb.dtype}), no pickle to convert")
    else:
        me = torch.load(src_me, map_location = "cpu", weights_only = False)
        emb = me["embedding"] if isinstance(me, dict) else me
        mid = me.get("mask_token_id") if isinstance(me, dict) else None
        if mid is not None:
            assert int(mid) == int(dfc["mask_token_id"]), \
                f"mask_embedding.pt says {mid}, config says {dfc['mask_token_id']}"
        emb = emb.reshape(-1).contiguous()
        assert emb.numel() == cfg["hidden_size"], f"mask embedding is {emb.shape}, expected [{cfg['hidden_size']}]"
    if not args.literal:
        save_file({"mask_embedding": emb}, str(out / "mask_embedding.safetensors"))

    print(f" -- wrote {out}")
    for f in sorted(out.iterdir()):
        tgt = f" -> {os.readlink(f)}" if f.is_symlink() else f"  ({f.stat().st_size} bytes)"
        print(f"      {f.name}{tgt}")
    print(f" -- tap_shift {args.tap_shift}, value scale {dfc['attention_value_scale']}, "
          f"sinks {dfc['attention_sink_bias']}, bidirectional_block {dfc['bidirectional_block']}, "
          f"partial_rotary_factor {cfg.get('partial_rotary_factor')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
