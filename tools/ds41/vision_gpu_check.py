#!/usr/bin/env python
"""Vision tower check on the real weights: exllamav3.dsv41.vision.VisionTower (bf16) against the vendored official ViT +
Aligner. Loads only the 266 tower tensors (about 1 GiB), never the language model.

  python tools/ds41/vision_gpu_check.py --fetch-from worker:~/ds41_work/recipe/conv/pack --weights DIR   # once, read-only ssh
  ~/ds41_work/bin/gpurun --who ds41vision --need 8 -- python tools/ds41/vision_gpu_check.py --weights DIR

`--weights DIR` holds `global.safetensors` (or the pack's index + shards) and `config.json`. `--fetch-from host:packdir`
streams just the tower tensors out of the pack's global.safetensors over ssh into DIR (the pack is only read).
Reference runs: official tower in fp32 (truth) and in bf16 (what the official release computes). Our tower must be no
further from the fp32 truth than the official bf16 run is (times --slack), for the ViT output and the aligner rows.
fp16 is measured too (information only: the default stays bf16).
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

REMOTE = r'''
import json, os, struct, sys
pack = os.path.expanduser(sys.argv[1])
f = open(pack + "/global.safetensors", "rb")
n = struct.unpack("<Q", f.read(8))[0]
hdr = json.loads(f.read(n))
meta = hdr.pop("__metadata__", None)
keep = {k: v for k, v in hdr.items() if k.startswith(("vision.", "aligner.")) or k in ("image_start", "image_newline", "image_end")}
pos, out = 0, {}
for k, v in keep.items():
    a, b = v["data_offsets"]
    out[k] = {"dtype": v["dtype"], "shape": v["shape"], "data_offsets": [pos, pos + b - a]}
    pos += b - a
h = json.dumps(out).encode()
h += b" " * (-len(h) % 8)
o = sys.stdout.buffer
o.write(struct.pack("<Q", len(h))); o.write(h)
base = 8 + n
for k, v in keep.items():
    a, b = v["data_offsets"]
    f.seek(base + a)
    left = b - a
    while left:
        chunk = f.read(min(left, 1 << 24))
        o.write(chunk); left -= len(chunk)
o.flush()
sys.stderr.write("tensors %d bytes %d\n" % (len(keep), pos))
'''


def fetch(spec: str, dest: Path) -> None:
    host, _, packdir = spec.partition(":")
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / "global.safetensors"
    cmd = ["ssh", host, "python3 -c " + shlex.quote(REMOTE) + " " + shlex.quote(packdir)]
    with open(out.with_suffix(".part"), "wb") as fh:
        r = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=fh, timeout=3600)
    if r.returncode:
        raise SystemExit(f"fetch failed (ssh exit {r.returncode})")
    out.with_suffix(".part").rename(out)
    cfg = subprocess.run(["ssh", host, "cat " + shlex.quote(packdir.replace("~/", "")) + "/config.json"], stdin=subprocess.DEVNULL,
                         capture_output=True, timeout=120, check=True).stdout
    (dest / "config.json").write_bytes(cfg)
    print(f"fetched {out.stat().st_size / 2**20:.0f} MiB into {dest}")


def test_image(w: int, h: int):
    """A photo-like image: smooth gradients, noise, shapes and text (deterministic)."""
    import numpy as np
    from PIL import Image, ImageDraw
    g = np.random.default_rng(w * 7 + h)
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([128 + 100 * np.sin(xx / 37.0), 128 + 100 * np.cos(yy / 29.0), 128 + 90 * np.sin((xx + yy) / 53.0)], -1)
    img = Image.fromarray(np.clip(base + g.normal(0, 8, base.shape), 0, 255).astype("uint8"))
    d = ImageDraw.Draw(img)
    d.rectangle([w // 8, h // 8, w // 2, h // 2], fill=(220, 30, 30), outline=(255, 255, 255), width=4)
    d.ellipse([w // 2, h // 3, w - w // 8, h - h // 8], fill=(30, 60, 220))
    d.text((w // 10, h - h // 4), "DeepSeek V4.1 vision check 0123456789", fill=(255, 255, 0))
    return img


def metrics(a, ref):
    import torch
    a, ref = a.float(), ref.float()
    err = a - ref
    return {"rel_rms": float(err.square().mean().sqrt() / ref.square().mean().sqrt().clamp_min(1e-9)),
            "max_abs": float(err.abs().max()), "cos": float(torch.nn.functional.cosine_similarity(a.flatten(), ref.flatten(), dim=0))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="folder with global.safetensors + config.json")
    ap.add_argument("--fetch-from", help="host:packdir to stream the tower tensors from (read-only)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sizes", default="640x480,1500x1100,200x1400", help="WxH list")
    ap.add_argument("--slack", type=float, default=1.25, help="allowed ratio ours_err / official_bf16_err")
    ap.add_argument("--json", help="write the numbers here")
    a = ap.parse_args()
    wdir = Path(a.weights).expanduser()
    if a.fetch_from:
        fetch(a.fetch_from, wdir)

    import io

    import torch
    from safetensors.torch import load_file
    import ds41_testkit as kit
    from exllamav3.dsv41 import vision as V
    from exllamav3.dsv41 import vision_spec as VS

    cfg = VS.read_vision_config_file(wdir)
    assert cfg is not None, f"{wdir}/config.json has no vision_config"
    raw = load_file(str(wdir / "global.safetensors"))
    names = VS.tower_tensor_names(cfg)
    missing = [n for n in names if n not in raw]
    assert not missing, f"missing tensors: {missing[:3]}"
    for n, shape in VS.expected_tower_shapes(cfg).items():
        assert tuple(raw[n].shape) == shape and raw[n].dtype == torch.bfloat16, (n, raw[n].shape, raw[n].dtype)
    print(f"tower: {len(names)} tensors, {cfg.n_layers} layers, dim {cfg.dim}, hidden {cfg.hidden_size}", flush=True)

    ref = HERE / "testdata" / "reference" / "inference"
    off_vis = kit.load_module("ds41_gpu_official_vision", ref / "vision.py")
    args = SimpleNamespace(vision_dim=cfg.dim, vision_n_heads=cfg.n_heads, vision_inter_dim=cfg.inter_dim,
                           vision_n_layers=cfg.n_layers, vision_patch_size=cfg.patch_size, vision_rope_theta=cfg.rope_theta,
                           vision_downsample_ratio=cfg.downsample_ratio, dim=cfg.hidden_size)
    dev = torch.device(a.device)
    _cs = off_vis.get_vision_cos_sin
    off_vis.get_vision_cos_sin = lambda *x: tuple(t.to(dev) for t in _cs(*x))      # the official runs with a CUDA default device

    def official(dtype):
        vit, ali = off_vis.ViT(args), off_vis.Aligner(args)
        vit.load_state_dict({k[7:]: t.float() for k, t in raw.items() if k.startswith("vision.")})
        ali.load_state_dict({k[8:]: t.float() for k, t in raw.items() if k.startswith("aligner.")})
        for m in (vit, ali):
            m.to(dev).eval()
            if dtype != torch.float32:
                m.to(dtype)
                for mod in m.modules():                       # the official keeps RMSNorm weights in fp32
                    if isinstance(mod, off_vis.RMSNorm):
                        mod.weight.data = mod.weight.data.float()
        return vit, ali

    towers = {"ours_bf16": V.VisionTower(cfg, raw, a.device, torch.bfloat16),
              "ours_fp16": V.VisionTower(cfg, raw, a.device, torch.float16)}
    vit32, ali32 = official(torch.float32)
    vit16, ali16 = official(torch.bfloat16)
    result, ok = {"sizes": {}}, True
    for sz in a.sizes.split(","):
        w, h = (int(x) for x in sz.split("x"))
        buf = io.BytesIO()
        test_image(w, h).save(buf, "PNG")
        img = V.preprocess(buf.getvalue(), cfg)
        p = img.patches.to(dev)
        with torch.no_grad():
            x32 = vit32(p.float(), img.n_vit_h, img.n_vit_w)
            r32 = ali32(x32, img.n_vit_h, img.n_vit_w)
            x16 = vit16(p, img.n_vit_h, img.n_vit_w)
            r16 = ali16(x16, img.n_vit_h, img.n_vit_w)
        row = {"grid": [img.n_vit_h, img.n_vit_w, img.n_llm_h, img.n_llm_w], "tokens": img.n_tokens,
               "official_bf16_vit": metrics(x16, x32), "official_bf16_rows": metrics(r16, r32)}
        for name, t in towers.items():
            vit_out = t.encode_vit(img.patches, img.n_vit_h, img.n_vit_w)
            rows = t.align(vit_out, img.n_vit_h, img.n_vit_w)
            row[f"{name}_vit"], row[f"{name}_rows"] = metrics(vit_out, x32), metrics(rows, r32)
        for k in ("vit", "rows"):
            bar = row[f"official_bf16_{k}"]["rel_rms"] * a.slack + 1e-6
            good = row[f"ours_bf16_{k}"]["rel_rms"] <= bar
            row[f"pass_{k}"] = good
            ok &= good
        # the whole span through the public path
        span = towers["ours_bf16"].span_rows(img)
        assert span.shape == (img.n_tokens, cfg.hidden_size) and torch.isfinite(span.float()).all()
        result["sizes"][sz] = row
        print(f"{sz}: tokens {img.n_tokens} | rows rel_rms official_bf16 {row['official_bf16_rows']['rel_rms']:.4f} "
              f"ours_bf16 {row['ours_bf16_rows']['rel_rms']:.4f} ours_fp16 {row['ours_fp16_rows']['rel_rms']:.4f} | "
              f"vit rel_rms official_bf16 {row['official_bf16_vit']['rel_rms']:.4f} ours_bf16 {row['ours_bf16_vit']['rel_rms']:.4f} "
              f"| cos {row['ours_bf16_rows']['cos']:.5f} | {'PASS' if row['pass_vit'] and row['pass_rows'] else 'FAIL'}", flush=True)
    result["pass"] = ok
    if a.json:
        Path(a.json).write_text(json.dumps(result, indent=1))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
