"""Stdlib-only part of the V4.1 vision support: config, image geometry, tensor names and shapes of the tower.
Shared by vision.py (runtime) and tools/ds41/validate_pack.py (which must not import torch). See vision.py."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(4)
MAX_IMAGE_BYTES = 32 * 1024 * 1024
MAX_SOURCE_PIXELS = 100_000_000          # decoded size cap (decompression bombs)


@dataclass
class VisionConfig:
    n_layers: int = 32
    dim: int = 1024
    n_heads: int = 16
    inter_dim: int = 2816
    patch_size: int = 14
    rope_theta: float = 10000.0
    downsample_ratio: int = 3
    max_n_token: int = 1024
    min_pixels: int = 295936
    max_wh_ratio: int | None = None
    image_token_id: int = 129264
    hidden_size: int = 5120               # LLM width = aligner output
    rms_norm_eps: float = 1e-6

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @property
    def num_channels(self) -> int:
        return 3


_NESTED = {"num_hidden_layers": "n_layers", "hidden_size": "dim", "num_attention_heads": "n_heads",
           "intermediate_size": "inter_dim", "patch_size": "patch_size", "rope_theta": "rope_theta",
           "downsample_ratio": "downsample_ratio", "max_image_tokens": "max_n_token", "min_pixels": "min_pixels",
           "max_wh_ratio": "max_wh_ratio"}


def read_vision_config(cfg: dict) -> VisionConfig | None:
    """Vision settings of a config.json: the HF layout (`vision_config` object + `image_token_id`) or the flat
    `vision_*` keys of the official inference/config.json. None when the model has no tower."""
    v = VisionConfig(image_token_id=int(cfg.get("image_token_id", 129264)),
                     hidden_size=int((cfg.get("text_config") or {}).get("hidden_size", cfg.get("hidden_size", cfg.get("dim", 5120)))))
    vc = cfg.get("vision_config")
    if isinstance(vc, dict):
        for src, dst in _NESTED.items():
            if src in vc:
                setattr(v, dst, vc[src])
    elif "vision_n_layers" in cfg:
        flat = {"vision_n_layers": "n_layers", "vision_dim": "dim", "vision_n_heads": "n_heads",
                "vision_inter_dim": "inter_dim", "vision_patch_size": "patch_size", "vision_rope_theta": "rope_theta",
                "vision_downsample_ratio": "downsample_ratio", "vision_max_n_token": "max_n_token",
                "vision_min_pixels": "min_pixels", "vision_max_wh_ratio": "max_wh_ratio"}
        for src, dst in flat.items():
            if src in cfg:
                setattr(v, dst, cfg[src])
    else:
        return None
    if int(v.n_layers) <= 0:
        return None
    v.n_layers, v.dim, v.n_heads, v.inter_dim = int(v.n_layers), int(v.dim), int(v.n_heads), int(v.inter_dim)
    v.patch_size, v.downsample_ratio, v.max_n_token = int(v.patch_size), int(v.downsample_ratio), int(v.max_n_token)
    v.min_pixels = int(v.min_pixels)
    v.rope_theta = float(v.rope_theta)
    v.max_wh_ratio = None if v.max_wh_ratio is None else int(v.max_wh_ratio)
    return v


def read_vision_config_file(model_dir: str | Path) -> VisionConfig | None:
    p = Path(model_dir).expanduser() / "config.json"
    if not p.is_file():
        return None
    return read_vision_config(json.loads(p.read_text()))


# --------------------------------------------------------------------------------------------------- geometry
def num_image_tokens(n_llm_h: int, n_llm_w: int) -> int:
    return n_llm_h * (n_llm_w + 1) + 2


def llm_grid(best_height: int, best_width: int, patch_size: int, downsample_ratio: int):
    return math.ceil((best_height // patch_size) / downsample_ratio), math.ceil(
        (best_width // patch_size) / downsample_ratio)


def solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token):
    """Largest aspect-preserving pixel size whose token grid still fits in max_n_token."""
    r = height / width
    max_w_float = math.sqrt((max_n_token - 2) / r + 0.25) - 0.5
    max_h_float = max_w_float * r
    cell = patch_size * downsample_ratio
    if max_w_float < 1.0:                       # very tall: collapse to a single column
        return (max_n_token - 2) // 2 * cell, cell
    if max_h_float < 1.0:                       # very wide: collapse to a single row
        return cell, (max_n_token - 3) * cell
    beta = min(math.floor(max_w_float) * cell / width, math.floor(max_h_float) * cell / height)
    return math.floor(height * beta / patch_size) * patch_size, math.floor(width * beta / patch_size) * patch_size


def safe_resize(height, width, best_height, best_width, patch_size, downsample_ratio, max_n_token):
    n_llm_h, n_llm_w = llm_grid(best_height, best_width, patch_size, downsample_ratio)
    if num_image_tokens(n_llm_h, n_llm_w) > max_n_token:
        best_height, best_width = solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token)
        n_llm_h, n_llm_w = llm_grid(best_height, best_width, patch_size, downsample_ratio)
        assert num_image_tokens(n_llm_h, n_llm_w) <= max_n_token
    return n_llm_h, n_llm_w, best_height, best_width


def plan_image_grid(width: int, height: int, v: VisionConfig):
    """(n_llm_h, n_llm_w, pixel_height, pixel_width) for an image of this size; pure function of its arguments."""
    p = v.patch_size
    if v.max_wh_ratio is not None and width > height * v.max_wh_ratio:
        width = height * v.max_wh_ratio
    if 0 < width * height < v.min_pixels:
        ratio = (v.min_pixels / (width * height)) ** 0.5
        width = int(width * ratio)
        height = int(height * ratio)
    best_width = math.ceil(width / p) * p
    best_height = math.ceil(height / p) * p
    return safe_resize(height, width, best_height, best_width, p, v.downsample_ratio, v.max_n_token)




def expected_tower_shapes(v: VisionConfig) -> dict[str, tuple]:
    """name -> shape of every tower tensor (all BF16 in the checkpoint)."""
    d, h, r = v.dim, v.hidden_size, v.downsample_ratio
    out = {"vision.patch_embed.proj.weight": (d, 3 * v.patch_size ** 2), "vision.patch_embed.proj.bias": (d,),
           "vision.norm.weight": (d,),
           "aligner.w1.weight": (h, d * r * r), "aligner.w1.bias": (h,), "aligner.w2.weight": (h, h),
           "aligner.w2.bias": (h,), "image_start": (h,), "image_newline": (h,), "image_end": (h,)}
    for i in range(v.n_layers):
        b = f"vision.blocks.{i}"
        out.update({f"{b}.norm1.weight": (d,), f"{b}.norm2.weight": (d,), f"{b}.attn.wqkv.weight": (3 * d, d),
                    f"{b}.attn.wqkv.bias": (3 * d,), f"{b}.attn.wo.weight": (d, d), f"{b}.attn.wo.bias": (d,),
                    f"{b}.mlp.w1.weight": (2 * v.inter_dim, d), f"{b}.mlp.w2.weight": (d, v.inter_dim)})
    return out


def is_tower_tensor(name: str) -> bool:
    return name.startswith(("vision.", "aligner.")) or name in ("image_start", "image_newline", "image_end")


def tower_tensor_names(v: VisionConfig) -> list[str]:
    return list(expected_tower_shapes(v))
