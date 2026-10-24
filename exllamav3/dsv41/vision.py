"""DeepSeek-V4.1 image input: preprocessing, the vision tower + aligner, and the splice into the prefill.

Everything here follows the official 1017 inference code (`inference/image_processor.py`, `inference/vision.py`,
`Transformer.merge_image_embeddings`), not the generic V4 path in `architecture/deepseek_v4_vision.py` (that one pads
the block to the compressor phase; V4.1 does not).

An image becomes a `n_vit_h x n_vit_w` patch grid for the ViT and, after the 3x3 aligner downsample, a
`n_llm_h x n_llm_w` token grid. The LLM sees the span

    [IMAGE_START] + ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_END]

and every position of it carries `image_token_id` in `input_ids`. The IMAGE rows are aligner outputs in reading order,
the three delimiter rows are the learned vectors `image_start` / `image_newline` / `image_end`.

Inside a span: the MoE router selects experts with `ffn.gate.bias_vl` instead of `ffn.gate.bias`, and the span takes no
part in an engram n-gram (hash blocked, engram gate 0). Both are driven by `span_mask`.

The model sees only `ImageSpan(start, rows)`: absolute prompt position and the full `[T, hidden]` rows of the span.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import torch
import torch.nn.functional as F

# config, geometry and tensor names live in the stdlib-only vision_spec (validate_pack loads it without torch)
from .vision_spec import (IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END, MAX_IMAGE_BYTES, MAX_SOURCE_PIXELS,  # noqa: F401
                          VisionConfig, read_vision_config_file, tower_tensor_names, num_image_tokens, plan_image_grid)


def image_token_types(n_llm_h: int, n_llm_w: int) -> torch.Tensor:
    types = [IMAGE_START] + ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_END]
    return torch.tensor(types, dtype=torch.int64)


# --------------------------------------------------------------------------------------------------- loading
class ImageError(ValueError):
    """The image cannot be loaded (the server answers 400)."""


def image_record(part: dict) -> dict:
    """A request content part (OpenAI image_url / Responses input_image / Anthropic image) -> {"url"} or {"data"}."""
    t = part.get("type")
    if t == "image_url":
        u = part.get("image_url")
        url = u if isinstance(u, str) else (u or {}).get("url")
    elif t == "input_image":
        u = part.get("image_url")
        url = u if isinstance(u, str) else (u or {}).get("url")
        if not url and part.get("image_data"):
            return {"data": part["image_data"]}
    elif t == "image":
        src = part.get("source")
        if isinstance(src, dict):
            if src.get("data"):
                return {"data": src["data"]}
            url = src.get("url")
        else:
            url = part.get("url") or part.get("image_url")
            if isinstance(url, dict):
                url = url.get("url")
    else:
        url = None
    if not isinstance(url, str) or not url:
        raise ImageError("image part has no usable source")
    return {"url": url}


def load_image_bytes(record: dict, *, allow_http: bool = True, allow_files: bool = False) -> bytes:
    data = record.get("data")
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    try:
        if isinstance(data, str):
            return base64.b64decode(data, validate=False)
        url = record.get("url")
        if not isinstance(url, str) or not url:
            raise ImageError(f"cannot load an image from record keys {sorted(record)}")
        if url.startswith("data:"):
            header, _, payload = url.partition(",")
            if ";base64" not in header:
                raise ImageError(f"unsupported data URL encoding: {header[:40]}")
            return base64.b64decode(payload)
        if url.startswith(("http://", "https://")):
            if not allow_http:
                raise ImageError("http image URLs are disabled")
            with urlopen(url, timeout=30) as resp:                   # noqa: S310  (operator-controlled server)
                raw = resp.read(MAX_IMAGE_BYTES + 1)
            if len(raw) > MAX_IMAGE_BYTES:
                raise ImageError(f"image larger than {MAX_IMAGE_BYTES >> 20} MiB")
            return raw
        if allow_files:
            with open(url, "rb") as f:
                return f.read()
    except ImageError:
        raise
    except Exception as exc:                                         # noqa: BLE001
        raise ImageError(f"cannot load image: {exc}") from exc
    raise ImageError("image URL must be a data: URL or an http(s) URL")


@dataclass
class PreparedImage:
    patches: torch.Tensor                  # [n_vit_h * n_vit_w, 3 * p * p] bf16, CPU
    n_vit_h: int
    n_vit_w: int
    n_llm_h: int
    n_llm_w: int
    key: str = ""                          # sha256 of the preprocessed pixels + grid (cache identity)
    size: tuple = (0, 0)                   # original (width, height)

    @property
    def n_tokens(self) -> int:
        return num_image_tokens(self.n_llm_h, self.n_llm_w)

    @property
    def types(self) -> torch.Tensor:
        return image_token_types(self.n_llm_h, self.n_llm_w)


def preprocess(raw: bytes, v: VisionConfig) -> PreparedImage:
    """Image bytes -> ViT patches, exactly `image_processor.load_image` (bf16 normalized pixels, gray padding)."""
    from PIL import Image, ImageOps
    p = v.patch_size
    try:
        with Image.open(io.BytesIO(raw)) as src:
            if src.width * src.height > MAX_SOURCE_PIXELS:
                raise ImageError(f"image of {src.width}x{src.height} is too large")
            image = src.convert("RGB")
    except ImageError:
        raise
    except Exception as exc:                                         # noqa: BLE001
        raise ImageError(f"cannot decode image: {exc}") from exc
    size = image.size
    n_llm_h, n_llm_w, best_height, best_width = plan_image_grid(image.width, image.height, v)
    n_vit_h, n_vit_w = best_height // p, best_width // p
    if v.max_wh_ratio is not None and image.width >= v.max_wh_ratio * image.height:
        image = image.resize((best_width, best_height))
    else:
        image = ImageOps.pad(image, (best_width, best_height), color=(127, 127, 127))
    x = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255
    x = ((x - 0.5) / 0.5).to(torch.bfloat16)
    patches = x.reshape(3, n_vit_h, p, n_vit_w, p).permute(1, 3, 0, 2, 4).reshape(n_vit_h * n_vit_w, 3 * p * p).contiguous()
    h = hashlib.sha256()
    h.update(f"{n_vit_h}x{n_vit_w}>{n_llm_h}x{n_llm_w}".encode())
    h.update(patches.view(torch.int16).numpy().tobytes())
    return PreparedImage(patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w, h.hexdigest(), size)


def prepare_image(record: dict, v: VisionConfig, **load_opts) -> PreparedImage:
    return preprocess(load_image_bytes(record, **load_opts), v)


# --------------------------------------------------------------------------------------------------- tower
def vision_cos_sin(n_h: int, n_w: int, rope_dim: int, theta: float, device):
    """2D rope tables [n_h * n_w, 1, rope_dim]: the h frequencies first, then the w frequencies (raster order)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim))
    hpos = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq
    freqs = freqs.flatten(1)
    return freqs.cos().unsqueeze(1).to(device), freqs.sin().unsqueeze(1).to(device)


def _rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(x.dtype)


def _rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    xf = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    return (w * xf).to(x.dtype)


def tensor_files(model_dir: Path, names) -> dict[str, Path]:
    """name -> safetensors file holding it: through model.safetensors.index.json, else global.safetensors."""
    idx = model_dir / "model.safetensors.index.json"
    if idx.is_file():
        wm = json.loads(idx.read_text())["weight_map"]
        return {n: model_dir / wm[n] for n in names if n in wm}
    g = model_dir / "global.safetensors"
    if g.is_file():
        from safetensors import safe_open
        with safe_open(str(g), "pt") as f:
            have = set(f.keys())
        return {n: g for n in names if n in have}
    return {}


def vision_available(model_dir: str | Path) -> VisionConfig | None:
    """The config of the tower when the folder holds all its tensors, else None."""
    d = Path(model_dir).expanduser()
    v = read_vision_config_file(d)
    if v is None:
        return None
    names = tower_tensor_names(v)
    return v if len(tensor_files(d, names)) == len(names) else None


class VisionTower:
    """ViT (full bidirectional attention, 2D rope) + aligner (3x3 merge, GELU MLP) + the three learned span vectors.
    Plain torch, same math as the official `ViT` / `Aligner`; query rows are processed in blocks so the attention
    scores of a 9000-patch image stay small."""

    Q_BLOCK = 2048

    def __init__(self, cfg: VisionConfig, w: dict[str, torch.Tensor], device="cuda:0", dtype=torch.bfloat16):
        self.cfg, self.device, self.dtype = cfg, torch.device(device), dtype
        self.w = {k: (t.float() if k.endswith("norm.weight") or ".norm1." in k or ".norm2." in k else t.to(dtype)).to(self.device)
                  for k, t in w.items()}

    @classmethod
    def load(cls, model_dir: str | Path, cfg: VisionConfig | None = None, device="cuda:0", dtype=torch.bfloat16):
        from safetensors import safe_open
        d = Path(model_dir).expanduser()
        cfg = cfg or read_vision_config_file(d)
        if cfg is None:
            raise ImageError("the model has no vision tower")
        names = tower_tensor_names(cfg)
        files = tensor_files(d, names)
        missing = [n for n in names if n not in files]
        if missing:
            raise ImageError(f"{len(missing)} vision tensors missing from {d} (first: {missing[0]})")
        out: dict[str, torch.Tensor] = {}
        by_file: dict[Path, list[str]] = {}
        for n, fp in files.items():
            by_file.setdefault(fp, []).append(n)
        for fp, ns in by_file.items():
            with safe_open(str(fp), "pt") as f:
                for n in ns:
                    out[n] = f.get_tensor(n)
        return cls(cfg, out, device, dtype)

    # ---------------------------------------------------------------- forward
    def _lin(self, x, name, bias=True):
        return F.linear(x, self.w[f"{name}.weight"], self.w.get(f"{name}.bias") if bias else None)

    def _attn(self, x, b, cos, sin):
        c = self.cfg
        n = x.size(0)
        q, k, v = (t.view(n, c.n_heads, c.head_dim) for t in self._lin(x, f"{b}.attn.wqkv").chunk(3, dim=-1))
        q, k = _rotary(q, cos, sin), _rotary(k, cos, sin)
        q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        if n <= self.Q_BLOCK:
            o = F.scaled_dot_product_attention(q, k, v)
        else:
            o = torch.cat([F.scaled_dot_product_attention(q[:, a:a + self.Q_BLOCK], k, v)
                           for a in range(0, n, self.Q_BLOCK)], dim=1)
        return self._lin(o.transpose(0, 1).reshape(n, -1), f"{b}.attn.wo")

    def _mlp(self, x, b):
        gate, up = self._lin(x, f"{b}.mlp.w1", bias=False).chunk(2, dim=-1)
        return self._lin(F.silu(gate) * up, f"{b}.mlp.w2", bias=False)

    @torch.inference_mode()
    def encode_vit(self, patches: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        c = self.cfg
        x = F.linear(patches.to(self.device, self.dtype), self.w["vision.patch_embed.proj.weight"],
                     self.w["vision.patch_embed.proj.bias"])
        cos, sin = vision_cos_sin(n_h, n_w, c.head_dim // 2, c.rope_theta, self.device)
        for i in range(c.n_layers):
            b = f"vision.blocks.{i}"
            x = x + self._attn(_rms(x, self.w[f"{b}.norm1.weight"], c.rms_norm_eps), b, cos, sin)
            x = x + self._mlp(_rms(x, self.w[f"{b}.norm2.weight"], c.rms_norm_eps), b)
        return _rms(x, self.w["vision.norm.weight"], c.rms_norm_eps)

    @torch.inference_mode()
    def align(self, x: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        r = self.cfg.downsample_ratio
        x = x.view(n_h, n_w, -1).permute(2, 0, 1)
        x = F.pad(x, (0, -n_w % r, 0, -n_h % r))
        x = F.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)
        return F.linear(F.gelu(F.linear(x, self.w["aligner.w1.weight"], self.w["aligner.w1.bias"])),
                        self.w["aligner.w2.weight"], self.w["aligner.w2.bias"])

    @torch.inference_mode()
    def encode(self, img: PreparedImage) -> torch.Tensor:
        """Aligner rows [n_llm_h * n_llm_w, hidden] in reading order."""
        out = self.align(self.encode_vit(img.patches, img.n_vit_h, img.n_vit_w), img.n_vit_h, img.n_vit_w)
        assert out.size(0) == img.n_llm_h * img.n_llm_w, (out.shape, img.n_llm_h, img.n_llm_w)
        return out

    @torch.inference_mode()
    def span_rows(self, img: PreparedImage) -> torch.Tensor:
        """The whole span [n_tokens, hidden] (bf16/half on the tower device): delimiters + aligner rows."""
        types = img.types.to(self.device)
        rows = torch.empty(types.numel(), self.cfg.hidden_size, dtype=self.dtype, device=self.device)
        rows[types == IMAGE_START] = self.w["image_start"]
        rows[types == IMAGE_END] = self.w["image_end"]
        rows[types == IMAGE_NEW_LINE] = self.w["image_newline"]
        rows[types == IMAGE] = self.encode(img)
        return rows


# --------------------------------------------------------------------------------------------------- the splice
@dataclass
class ImageSpan:
    """One image in a prompt: absolute start position and the full span rows [T, hidden]. `key` identifies the pixels."""
    start: int
    rows: torch.Tensor
    key: str = ""

    @property
    def end(self) -> int:
        return self.start + self.rows.size(0)


def span_mask(spans, start: int, length: int, device=None):
    """bool [length]: True for positions of [start, start+length) inside an image span. None without spans (or without overlap)."""
    if not spans:
        return None
    m = None
    for s in spans:
        a, b = max(s.start, start), min(s.end, start + length)
        if a < b:
            if m is None:
                m = torch.zeros(length, dtype=torch.bool, device=device)
            m[a - start:b - start] = True
    return m


def splice(h: torch.Tensor, start: int, spans) -> torch.Tensor | None:
    """Overwrite, in place, the rows of h [1, L, D] (prompt positions start..start+L-1) that lie in an image span
    with the span rows. Returns the bool mask [L] of the overwritten rows (None when nothing overlaps)."""
    if not spans:
        return None
    assert h.size(0) == 1, "image spans need batch 1"
    L = h.size(1)
    m = None
    for s in spans:
        a, b = max(s.start, start), min(s.end, start + L)
        if a < b:
            h[0, a - start:b - start] = s.rows[a - s.start:b - s.start].to(device=h.device, dtype=h.dtype)
            if m is None:
                m = torch.zeros(L, dtype=torch.bool, device=h.device)
            m[a - start:b - start] = True
    return m


def locate_spans(ids, images, image_token_id: int) -> list[tuple[int, PreparedImage]]:
    """[(start, image)] of an already expanded prompt: each image's span is the next `n_tokens` ids from the next
    `image_token_id`. Raises ImageError when the ids do not hold exactly these spans."""
    out, pos = [], 0
    for img in images:
        try:
            start = ids.index(image_token_id, pos)
        except ValueError:
            raise ImageError("the prompt has fewer image spans than images") from None
        n = img.n_tokens
        if start + n > len(ids) or any(t != image_token_id for t in ids[start:start + n]):
            raise ImageError("an image span in the prompt is shorter than its image needs")
        out.append((start, img))
        pos = start + n
    if image_token_id in ids[pos:]:
        raise ImageError("the prompt has more image tokens than images")
    return out


def prompt_key(ids, located) -> list:
    """Prefix-cache identity of a prompt: the ids, with every image position replaced by a negative number derived
    from the image pixels and the offset in its span, so two images of the same size never match."""
    keys = list(ids)
    for start, img in located or ():
        base = int(img.key[:15], 16) << 24 if img.key else 0
        for j in range(img.n_tokens):
            keys[start + j] = -1 - (base + j)
    return keys


def expand_image_tokens(ids: list[int], images, image_token_id: int):
    """Replace each `image_token_id` in `ids` by its image's span of ids (all `image_token_id`).
    Returns (ids, [(start, PreparedImage)]). The placeholder count must equal the image count."""
    n = sum(1 for t in ids if t == image_token_id)
    if n != len(images):
        raise ImageError(f"the prompt has {n} image placeholders but {len(images)} images were sent")
    out, starts, it = [], [], iter(images)
    for t in ids:
        if t != image_token_id:
            out.append(t)
            continue
        img = next(it)
        starts.append((len(out), img))
        out.extend([image_token_id] * img.n_tokens)
    return out, starts
