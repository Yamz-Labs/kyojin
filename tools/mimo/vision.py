"""MiMo-V2.6 native image input: the vision tower, the image preprocessing and the embedding builder.

The tower (28 blocks, hidden 1280, 4 full-attention blocks and 24 sliding-window blocks with attention sinks, a
2x2 spatial merger onto the 4096-wide text embeddings) is re-implemented here from the reference code in the
pack (`modeling_mimo_v2.py`, class MiMoVisionTransformer). Its weights are not part of the EXL3 text pack: they sit
in `<model dir>/vision/vision.safetensors` (bf16, tensors `visual.*`, 1.46 GB, cut from the upstream
`model_pp0_ep0_shard0.safetensors`). The tower runs in plain torch, outside the quantised model.

Preprocessing follows the Qwen2-VL image processor that the sibling MiMo-V2.5 release publishes
(`preprocessor_config.json`: patch 16, temporal patch 2, merge 2, CLIP mean/std); the pack has no such file.

Memory: 1.46 GB of bf16 weights. Time: one forward over S = (H/16)*(W/16) patches; S is capped by MIMO_IMAGE_MAX_PIXELS.
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn.functional as F

VISION_FILE = os.path.join("vision", "vision.safetensors")
IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)
PATCH, TEMPORAL, MERGE = 16, 2, 2
FACTOR = PATCH * MERGE
# The pack config allows 8 388 608 pixels (32 768 patches); one tower pass at that size needs far more
# attention memory than is worth it on the box. 2 097 152 pixels = 8192 patches = 2048 text tokens.
DEFAULT_MAX_PIXELS = 2_097_152


def max_pixels(config_max: int | None = None) -> int:
    cap = int(os.environ.get("MIMO_IMAGE_MAX_PIXELS", DEFAULT_MAX_PIXELS))
    return min(cap, config_max) if config_max else cap


def smart_resize(height: int, width: int, min_pixels: int, max_px: int) -> tuple[int, int]:
    """(new height, new width): multiples of 32, pixel count in [min_pixels, max_px], aspect kept."""
    if height < FACTOR or width < FACTOR:
        # too small: scale up so that the short side reaches the factor
        s = FACTOR / min(height, width)
        height, width = math.ceil(height * s), math.ceil(width * s)
    if max(height, width) / min(height, width) > 200:
        raise ValueError("image aspect ratio must be smaller than 200")
    h = round(height / FACTOR) * FACTOR
    w = round(width / FACTOR) * FACTOR
    if h * w > max_px:
        beta = math.sqrt(height * width / max_px)
        h = max(FACTOR, math.floor(height / beta / FACTOR) * FACTOR)
        w = max(FACTOR, math.floor(width / beta / FACTOR) * FACTOR)
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h = math.ceil(height * beta / FACTOR) * FACTOR
        w = math.ceil(width * beta / FACTOR) * FACTOR
    return h, w


def image_grid(size: tuple[int, int], min_pixels: int = 8192, max_px: int | None = None):
    """(grid_h, grid_w, text tokens) for a PIL size (width, height), without touching the pixels."""
    h, w = smart_resize(size[1], size[0], min_pixels, max_px or max_pixels())
    gh, gw = h // PATCH, w // PATCH
    return gh, gw, (gh // MERGE) * (gw // MERGE)


def preprocess(image, min_pixels: int = 8192, max_px: int | None = None):
    """PIL image -> (patches [S, 1536] float32, (grid_t, grid_h, grid_w)). Patch order: merge-group major."""
    from PIL import Image
    if image.mode != "RGB":
        rgba = image.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, "WHITE")
        bg.paste(rgba, (0, 0), rgba)
        image = bg.convert("RGB")
    h, w = smart_resize(image.height, image.width, min_pixels, max_px or max_pixels())
    if (w, h) != image.size:
        image = image.resize((w, h), resample=Image.BICUBIC)
    a = np.asarray(image, dtype=np.float32) / 255.0
    a = (a - np.array(IMAGE_MEAN, np.float32)) / np.array(IMAGE_STD, np.float32)
    a = a.transpose(2, 0, 1)                                   # C H W
    a = np.tile(a[None], (TEMPORAL, 1, 1, 1))                  # T C H W (a still image is repeated)
    gh, gw = h // PATCH, w // PATCH
    a = a.reshape(1, TEMPORAL, 3, gh // MERGE, MERGE, PATCH, gw // MERGE, MERGE, PATCH)
    a = a.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)                 # t, h/m, w/m, m, m, C, T, p, p
    return torch.from_numpy(np.ascontiguousarray(a.reshape(gh * gw, 3 * TEMPORAL * PATCH * PATCH))), (1, gh, gw)


class VisionTower:
    """Plain-torch MiMo vision tower. Weights are loaded once, forward() maps patches to text-width embeddings."""

    def __init__(self, path: str, cfg: dict, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        from safetensors import safe_open
        self.device, self.dtype = device, dtype
        self.depth = cfg["depth"]
        self.dim = cfg["hidden_size"]
        self.heads = cfg["num_heads"]
        self.kv_heads = cfg.get("num_key_value_heads", self.heads)
        self.head_dim = cfg.get("qk_channels", 64)
        self.full = set(cfg.get("fullatt_block_indexes", []))
        self.types = cfg.get("vit_window_attn_types") or [-1] * self.depth
        self.window = cfg.get("visual_token_window_size", -1)
        self.sinks = bool(cfg.get("use_sink", False))
        self.eps = 1e-6
        self.min_pixels = 8192
        w = {}
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys():
                w[k] = f.get_tensor(k).to(device=device, dtype=dtype)
        self.w = w
        self.conv = w["visual.patch_embed.proj.weight"].reshape(self.dim, -1)   # kernel = stride: a linear map
        self.inv_freq = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim // 2, 2, dtype=torch.float) / (self.head_dim // 2)))
        self.nbytes = sum(t.numel() * t.element_size() for t in w.values())

    # -- helpers (mirror MiMoVisionTransformer) ---------------------------------------------------------------
    @staticmethod
    def _rot_half(x):
        h = x.shape[-1] // 2
        return torch.cat((-x[..., h:], x[..., :h]), dim=-1)

    def _pos_emb(self, gh: int, gw: int):
        m = MERGE
        hp = torch.arange(gh).unsqueeze(1).expand(-1, gw).reshape(gh // m, m, gw // m, m).permute(0, 2, 1, 3).flatten()
        wp = torch.arange(gw).unsqueeze(0).expand(gh, -1).reshape(gh // m, m, gw // m, m).permute(0, 2, 1, 3).flatten()
        pos = torch.stack([hp, wp], dim=-1)
        table = torch.outer(torch.arange(max(gh, gw), dtype=torch.float), self.inv_freq)
        r = table[pos].flatten(1)
        return torch.cat((r, r), dim=-1)

    @staticmethod
    def _apply_index(x, index):
        return x.unflatten(0, (-1, MERGE * MERGE))[index].flatten(0, 1)

    def _ln(self, x, w):                                    # RMSNorm
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * w

    def _attn(self, x, i, cos, sin, full_attn):
        W, S = self.w, x.shape[0]
        p = f"visual.blocks.{i}.attn."
        qkv = F.linear(x, W[p + "qkv.weight"], W[p + "qkv.bias"])
        qd, kd = self.heads * self.head_dim, self.kv_heads * self.head_dim
        q = qkv[:, :qd].view(S, self.heads, self.head_dim)
        k = qkv[:, qd:qd + kd].view(S, self.kv_heads, self.head_dim)
        v = qkv[:, qd + kd:].view(S, self.kv_heads, self.head_dim)
        c, s = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
        q = (q.float() * c + self._rot_half(q.float()) * s).to(x.dtype)
        k = (k.float() * c + self._rot_half(k.float()) * s).to(x.dtype)
        g = self.heads // self.kv_heads
        if g > 1:
            k, v = k.repeat_interleave(g, dim=1), v.repeat_interleave(g, dim=1)
        q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)           # H S D
        sink = W.get(p + "sinks") if (self.sinks and not full_attn) else None
        win = self.window if (not full_attn and self.window > 0) else 0
        scale = self.head_dim ** -0.5
        out = torch.empty_like(q)
        cols = torch.arange(S, device=x.device)
        blk = max(64, int(2**27 // max(S, 1) // self.heads))            # query rows per chunk (bounded scores buffer)
        for a in range(0, S, blk):
            b = min(S, a + blk)
            sc = torch.matmul(q[:, a:b].float(), k.float().transpose(1, 2)) * scale        # H R S
            if win:
                rows = torch.arange(a, b, device=x.device).unsqueeze(1)
                sc = sc.masked_fill(((rows - cols).abs() > win).unsqueeze(0), float("-inf"))
            if sink is not None:
                sc[:, :, 0] += sink.float().view(-1, 1)
            out[:, a:b] = torch.matmul(torch.softmax(sc, dim=-1).to(v.dtype), v)
        o = out.transpose(0, 1).reshape(S, -1)
        return F.linear(o, W[p + "proj.weight"], W[p + "proj.bias"])

    @torch.no_grad()
    def forward(self, patches: torch.Tensor, grid: tuple[int, int, int]) -> torch.Tensor:
        """patches [S, 1536] from preprocess(), grid (t, h, w) -> embeddings [S/4, 4096] in the original token order."""
        W = self.w
        t, gh, gw = grid
        assert t == 1, "still images only"
        x = F.linear(patches.to(self.device, self.dtype), self.conv)
        emb = self._pos_emb(gh, gw).to(self.device)
        row = (emb.cos(), emb.sin())
        idx = torch.arange((gh // MERGE) * (gw // MERGE)).reshape(gh // MERGE, gw // MERGE).t().reshape(-1).to(self.device)
        rev = torch.argsort(idx)
        ce = self._apply_index(emb, idx)
        col = (ce.cos(), ce.sin())
        for i in range(self.depth):
            ty = self.types[i]
            if ty == 1 and (i == 0 or self.types[i - 1] != 1):
                x = self._apply_index(x, idx)
            if i > 0 and ty != 1 and self.types[i - 1] == 1:
                x = self._apply_index(x, rev)
            cos, sin = col if ty == 1 else row
            p = f"visual.blocks.{i}."
            x = x + self._attn(self._ln(x, W[p + "norm1.weight"]), i, cos, sin, i in self.full)
            h = self._ln(x, W[p + "norm2.weight"])
            h = F.silu(F.linear(h, W[p + "mlp.gate_proj.weight"], W[p + "mlp.gate_proj.bias"])) * \
                F.linear(h, W[p + "mlp.up_proj.weight"], W[p + "mlp.up_proj.bias"])
            x = x + F.linear(h, W[p + "mlp.down_proj.weight"], W[p + "mlp.down_proj.bias"])
        # the last block (27) is a full-attention block in row order; the layout is already original here
        x = F.layer_norm(x, (self.dim,), W["visual.merger.ln_q.weight"], W.get("visual.merger.ln_q.bias"), 1e-6)
        x = x.reshape(-1, self.dim * MERGE * MERGE)
        x = F.gelu(F.linear(x, W["visual.merger.mlp.0.weight"], W.get("visual.merger.mlp.0.bias")))
        return F.linear(x, W["visual.merger.mlp.2.weight"], W.get("visual.merger.mlp.2.bias"))

    def get_image_embeddings(self, image, text_alias: str | None = None):
        """PIL image -> MMEmbedding whose token string is only the N placeholder rows (the chat template already
        writes <|vision_start|> and <|vision_end|> around the single <|image_pad|>)."""
        from exllamav3.tokenizer import MMEmbedding
        patches, grid = preprocess(image, self.min_pixels)
        emb = self.forward(patches, grid).to(torch.float16).cpu()
        n = emb.shape[0]
        mme = MMEmbedding(embeddings=emb, text_alias=text_alias,
                          token_string=torch.full((1, n), -1, dtype=torch.long))
        mme.metadata.update({"original_size": image.size, "preprocessed_size": (grid[2] * PATCH, grid[1] * PATCH),
                             "model_architecture": "MiMoV2ForCausalLM"})
        return mme


def load_tower(model_dir: str, cfg: dict, device: str = "cuda"):
    path = os.path.join(model_dir, VISION_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no vision weights at {path}")
    return VisionTower(path, cfg, device=device)
