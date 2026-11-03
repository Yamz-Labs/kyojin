"""Draft-path discovery (drill-0 gap 6: the vendor DFlash2 drafter was found, downloaded, checked and wired by hand).
Pure helpers + an HF search; the A/B that decides the policy is stages.job_draft, the rule is rules.choose_draft.

A candidate is accepted only when its config matches the target STRUCTURALLY (DFlash architecture, same hidden size, tapped layers inside the
target's depth, same vocabulary when the drafter states one), whatever it is called."""
from __future__ import annotations

import json
import re
from pathlib import Path

DRAFTER_HOME = Path("~/models/drafters").expanduser()


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def stem(pack_name: str) -> str:
    """Model stem of a pack directory name: everything before the quantization suffix (nanbeige43-33b-exl3-4.0 -> nanbeige43-33b)."""
    return re.split(r"[-_.](?:exl\d|gptq|awq|gguf|mlx|\d+(?:\.\d+)?bpw)", pack_name, maxsplit=1, flags=re.I)[0]


def validate(d: str | Path, meta: dict) -> list[str]:
    """Problems that make the drafter unusable for this pack (empty list = compatible)."""
    d = Path(d).expanduser()
    try:
        cfg = json.loads((d / "config.json").read_text())
    except (OSError, ValueError):
        return ["no readable config.json"]
    bad = []
    if not any("dflash" in a.lower() for a in cfg.get("architectures") or []):
        bad.append(f"architecture {cfg.get('architectures')} is not a DFlash drafter")
    if cfg.get("hidden_size") != meta.get("hidden"):
        bad.append(f"hidden_size {cfg.get('hidden_size')} != target {meta.get('hidden')}")
    taps = (cfg.get("dflash_config") or {}).get("target_layer_ids") or []
    if not taps or max(taps) >= meta.get("layers", 0):
        bad.append(f"target_layer_ids {taps} outside the target's {meta.get('layers')} layers")
    if cfg.get("vocab_size") and meta.get("vocab") and cfg["vocab_size"] != meta["vocab"]:
        bad.append(f"vocab_size {cfg['vocab_size']} != target {meta['vocab']}")
    if not list(d.glob("*.safetensors")):
        bad.append("no safetensors weights")
    return bad


def find_local(meta: dict, explicit: str | None = None) -> tuple[str | None, list[str]]:
    """(dir, notes): --drafter DIR, <pack>/dflash, then every dir under ~/models/drafters that validates. A checkpoint that states its tap
    convention (dflash_config.tap_shift, the exllamav3 export) wins over one that does not. Notes list the rejected candidates."""
    notes, ok = [], []
    cands = [Path(explicit).expanduser()] if explicit else []
    cands.append(Path(meta["path"]) / "dflash")
    if DRAFTER_HOME.is_dir():
        cands += sorted(p for p in DRAFTER_HOME.iterdir() if p.is_dir())
    for p in cands:
        if not p.is_dir():
            continue
        bad = validate(p, meta)
        if bad:
            notes.append(f"{p}: {'; '.join(bad)}")
        else:
            ok.append(p)
    explicit_ok = [p for p in ok if explicit and p == Path(explicit).expanduser()]
    if explicit_ok:
        return str(explicit_ok[0]), notes
    ok.sort(key=lambda p: "tap_shift" not in ((json.loads((p / "config.json").read_text()).get("dflash_config")) or {}))
    return (str(ok[0]) if ok else None), notes


def hf_revision(repo: str) -> str:
    """The 'exllamav3' branch when the repo has one (it carries the export this engine reads), else main."""
    try:
        from huggingface_hub import HfApi
        if "exllamav3" in [b.name for b in HfApi().list_repo_refs(repo).branches]:
            return "exllamav3"
    except Exception:      # noqa: BLE001
        pass
    return "main"


def search_hf(meta: dict, limit: int = 300) -> list[str]:
    """HF repos named <model stem> + 'dflash' (ids only; nothing is downloaded). Network errors -> []."""
    try:
        from huggingface_hub import HfApi
        ids = [m.id for m in HfApi().list_models(search="dflash", limit=limit)]
    except Exception:      # noqa: BLE001
        return []
    s = norm(stem(meta["name"]))
    return [i for i in ids if s and s in norm(i.split("/", 1)[-1])]


def pick_hf(ids: list[str]) -> str | None:
    """Prefer the plainest name (shortest id, then alphabetical): 'X-DFlash2' over 'X-DFlash2-GGUF'; GGUF/MLX/quantized variants never."""
    ok = [i for i in ids if not re.search(r"gguf|mlx|nvfp4|fp8|awq|gptq|exl\d", i, re.I)]
    return sorted(ok, key=lambda i: (len(i), i))[0] if ok else None
