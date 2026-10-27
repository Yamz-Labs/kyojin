"""Pack metadata: everything the later stages need, read from the pack directory (no GPU, no torch).

Prior work: drill 0 (nanb) needed a hand-written chat template and hand flags (--draft-policy off, --no-vision);
here they are derived from the pack. Per-arch knowledge is the FAMILIES table (which layers hold a KV cache, serve entry point).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

KV_SCALE_OVERHEAD_BITS = 0.5   # per element, quantized cache scales/compand (nanb Q4 57 KB/token, realA Qwen27 K4V4 18.6 KB/token)

# family -> (serve entry, ref adapter or None). KV layers come from config (see kv_layers).
FAMILIES = {
    "LlamaForCausalLM": dict(family="dense_llama", serve="tools/qwen/serve.py", ref="llama_dense", vision=False),
    "Qwen3_5ForConditionalGeneration": dict(family="hybrid_gdn", serve="tools/qwen/serve.py", ref=None, vision=True),
    "Qwen3_5ForCausalLM": dict(family="hybrid_gdn", serve="tools/qwen/serve.py", ref=None, vision=False),
}


def _text_cfg(cfg: dict) -> dict:
    return cfg.get("text_config") or cfg


def kv_layers(t: dict) -> int:
    """Number of layers that hold a growing KV cache (full attention only)."""
    n = t["num_hidden_layers"]
    lt = t.get("layer_types")
    if lt:
        return sum(1 for x in lt if x in ("full_attention", "attention"))
    iv = t.get("full_attention_interval")
    return n // iv if iv else n


def kv_bytes_per_token(meta: dict, k_bits: int, v_bits: int, kv_mode: str = "quant") -> float:
    elems_k = meta["kv_layers"] * meta["kv_heads"] * meta["head_dim"]
    ovh = 16 / meta["head_dim"] if kv_mode == "tq" else KV_SCALE_OVERHEAD_BITS     # tq: one fp16 norm per head vector
    return elems_k * ((k_bits + ovh) + (v_bits + ovh)) / 8


def weights_dir_bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in p.glob("*.safetensors"))


def read_pack(path: str | os.PathLike) -> dict:
    p = Path(path).expanduser()
    cfg = json.loads((p / "config.json").read_text())
    t = _text_cfg(cfg)
    arch = (cfg.get("architectures") or ["?"])[0]
    fam = FAMILIES.get(arch, dict(family="unknown", serve=None, ref=None, vision=False))
    heads = t["num_attention_heads"]
    qc = {}
    for name in ("quantization_config.json",):
        if (p / name).exists():
            qc = json.loads((p / name).read_text())
    qc = qc or cfg.get("quantization_config") or {}
    idx = p / "model.safetensors.index.json"
    meta = dict(
        name=p.name, path=str(p), arch=arch, family=fam["family"], serve_entry=fam["serve"], ref_adapter=fam["ref"],
        model_type=t.get("model_type"), layers=t["num_hidden_layers"], kv_layers=kv_layers(t),
        heads=heads, kv_heads=t.get("num_key_value_heads", heads), head_dim=t.get("head_dim") or t["hidden_size"] // heads,
        hidden=t["hidden_size"], max_pos=t.get("max_position_embeddings"), vocab=t.get("vocab_size"),
        quant=qc.get("quant_method"), bpw=qc.get("bits"), head_bits=qc.get("head_bits"),
        vision=bool(fam["vision"] and cfg.get("vision_config") and not cfg.get("language_model_only")),
        mtp_layers=int(t.get("mtp_num_hidden_layers") or t.get("num_nextn_predict_layers") or 0),
        weights_bytes=weights_dir_bytes(p), index_sha=hashlib.sha1(idx.read_bytes()).hexdigest()[:12] if idx.exists() else None,
        license=_license(p), template=str(p / "chat_template.jinja") if (p / "chat_template.jinja").exists() else None,
    )
    meta["problems"] = problems(meta)
    return meta


def _license(p: Path) -> str | None:
    for f in ("LICENSE", "LICENSE.md", "LICENSE.txt"):
        if (p / f).exists():
            return f
    rd = p / "README.md"
    if rd.exists():
        for line in rd.read_text(errors="replace").splitlines()[:20]:
            if line.lower().startswith("license:"):
                return line.split(":", 1)[1].strip()
    return None


def problems(meta: dict) -> list[str]:
    out = []
    if meta["quant"] != "exl3":
        out.append(f"quant_method={meta['quant']!r}: only EXL3 packs are served by tools/qwen/serve.py")
    if not meta["serve_entry"]:
        out.append(f"no serve entry for architecture {meta['arch']}: add it to pack.FAMILIES")
    if not meta["template"]:
        out.append("pack has no chat_template.jinja (the conversion must copy the working template into the pack)")
    return out


def draft_policy(meta: dict) -> str:
    """Vendor MTP head > off. tools/qwen/serve.py loads the MTP drafter for every policy but 'off' (and 'mix' adds the lossless n-gram
    lookup on top of it), so a pack without an MTP head must be served with 'off'. Drill 0 gap 8 stays open: n-gram lookup WITHOUT an
    MTP head needs a server flag (not in serve.py today)."""
    return "mix" if meta["mtp_layers"] > 0 else "off"


def serve_flags(meta: dict, *, port: int, ctx: int, sessions: int, k_bits: int, v_bits: int, model_id: str = "qual",
                draft: str | None = None, kv_mode: str = "quant", kvwin: str | None = None) -> list[str]:
    """argv of the server (after the python executable) for this pack. No hand flags: every value is derived."""
    f = [meta["serve_entry"], "--model", meta["path"], "--port", str(port), "--model-id", model_id, "-c", str(ctx),
         "--sessions", str(sessions), "--k-bits", str(k_bits), "--v-bits", str(v_bits),
         "--draft-policy", draft or draft_policy(meta)]
    if kv_mode != "quant":
        f += ["--kv-mode", kv_mode]
    if kvwin:
        f += ["--kvwin", kvwin]
    if not meta["vision"]:
        f.append("--no-vision")
    if meta["template"]:
        f += ["--chat-template", meta["template"]]
    return f


def verify_shards(path: str | os.PathLike) -> list[str]:
    """Problems found when comparing the shard files with model.safetensors.index.json (names + existence + non-empty)."""
    p = Path(path).expanduser()
    idx = p / "model.safetensors.index.json"
    if not idx.exists():
        return [f"{p}: no model.safetensors.index.json"]
    files = sorted(set(json.loads(idx.read_text())["weight_map"].values()))
    bad = []
    for f in files:
        fp = p / f
        if not fp.exists():
            bad.append(f"missing shard {f}")
        elif fp.stat().st_size == 0:
            bad.append(f"empty shard {f}")
    return bad


def disk_free_gb(path: str | os.PathLike) -> float:
    q = Path(path).expanduser()
    while not q.exists():
        q = q.parent
    return shutil.disk_usage(q).free / 1e9
