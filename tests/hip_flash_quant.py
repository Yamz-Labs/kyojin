"""Expert quantization of an exl3 checkpoint, read without loading weights.

The gfx12 flash-model route tests target fixed exl3 trellis sizes (K3 routed
experts for the grouped/MoE routes, K5 shared experts for the K5 GEMV route).
quantization_config.json records the per-tensor quantization of the
checkpoint, so a checkpoint whose expert families use different trellis sizes
cannot exercise the native route and is skipped rather than failed.
"""
from __future__ import annotations

import json
from pathlib import Path


def exl3_expert_quant(model_dir: Path) -> tuple[str | None, dict[str, int | None]]:
    """(codebook, {family: bits_per_weight}) for the exl3 expert tensors.

    family is "routed" (.experts.) or "shared" (shared_expert). A family maps
    to its single recorded bit width, or None when no exl3 tensor of that
    family is present or the widths are not uniform. codebook is the
    checkpoint-wide exl3 codebook name. Both are None for a directory without
    a quantization_config.json.
    """
    quant = model_dir / "quantization_config.json"
    codebook = None
    bits: dict[str, set[int]] = {"routed": set(), "shared": set()}
    if quant.is_file():
        config = json.loads(quant.read_text())
        codebook = config.get("codebook")
        for name, entry in config.get("tensor_storage", {}).items():
            if not isinstance(entry, dict) or entry.get("quant_format") != "exl3":
                continue
            family = ("shared" if "shared_expert" in name
                      else "routed" if ".experts." in name else None)
            if family is not None and entry.get("bits_per_weight") is not None:
                bits[family].add(entry["bits_per_weight"])
    return codebook, {
        family: next(iter(widths)) if len(widths) == 1 else None
        for family, widths in bits.items()
    }
