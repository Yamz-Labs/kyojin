"""Fetch only the MTP head (mtp.* tensors) of a HF Qwen3.5 / 3.6 MoE checkpoint by HTTP range reads and store them as one
safetensors file, to sit beside a GGUF body (serve.py --gguf-mtp DIR).

    python tools/qwen/gguf_mtp_fetch.py --repo Qwen/Qwen3.6-35B-A3B --out DIR [--size-only]
"""
import argparse, json, os, struct, time, urllib.error, urllib.request


def rng(url, a, b):
    req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
    for i in range(8):
        try:
            return urllib.request.urlopen(req).read()
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            time.sleep(15 * (i + 1))
    raise SystemExit("HTTP 429 persists")


def experts_fp8(t, block=128):
    """Fused bf16 MoE experts -> one fp8 e4m3 tensor + weight_scale_inv block grid per expert projection (the engine's FP8 path,
    1 byte per weight: the head's experts are 1.6 GB in bf16 and a 24 GB card running a GGUF body has no room for that)."""
    import torch
    out = {k: v for k, v in t.items() if not k.endswith(("experts.gate_up_proj", "experts.down_proj"))}
    def q(w):
        N, K = w.shape
        g = w.float().reshape(N // block, block, K // block, block)
        sc = (g.abs().amax(dim=(1, 3)) / 448.0).clamp(min=1e-12)
        return (g / sc[:, None, :, None]).clamp(-448, 448).reshape(N, K).to(torch.float8_e4m3fn), sc
    for k, v in t.items():
        if k.endswith("experts.gate_up_proj"):
            base, I = k[:-len("gate_up_proj")], v.shape[1] // 2
            dn = t[base + "down_proj"]
            for e in range(v.shape[0]):
                for name, w in (("gate_proj", v[e, :I]), ("up_proj", v[e, I:]), ("down_proj", dn[e])):
                    out[f"{base}{e}.{name}.weight"], out[f"{base}{e}.{name}.weight_scale_inv"] = q(w.contiguous())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3.6-35B-A3B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size-only", action="store_true")
    ap.add_argument("--experts-fp8", action="store_true", help="store the routed experts as fp8 e4m3 + block scales (about half the size)")
    ap.add_argument("--from-file", default="", help="convert an already fetched mtp safetensors instead of fetching")
    a = ap.parse_args()
    base = f"https://huggingface.co/{a.repo}/resolve/main/"
    idx = json.loads(urllib.request.urlopen(base + "model.safetensors.index.json").read())["weight_map"]
    keys = [k for k in idx if k.startswith("mtp.")]
    if a.from_file:
        from safetensors.torch import load_file, save_file
        os.makedirs(a.out, exist_ok=True)
        save_file(experts_fp8(load_file(a.from_file)), os.path.join(a.out, "mtp_fp8e.safetensors"), metadata={"format": "pt"})
        return
    hd, total = {}, 0
    for sh in sorted({idx[k] for k in keys}):
        n = struct.unpack("<Q", rng(base + sh, 0, 7))[0]
        hd[sh] = (8 + n, json.loads(rng(base + sh, 8, 8 + n - 1)))
        total += sum(hd[sh][1][k]["data_offsets"][1] - hd[sh][1][k]["data_offsets"][0] for k in keys if idx[k] == sh)
    print(f"{len(keys)} tensors, {total} bytes in {sorted(hd)}")
    if a.size_only:
        return
    import torch
    from safetensors.torch import save_file
    out = {}
    for sh, (off, h) in hd.items():
        for k in keys:
            if idx[k] != sh:
                continue
            m = h[k]; lo, hi = m["data_offsets"]; buf = bytearray(); p = lo
            while p < hi:
                q = min(hi, p + (256 << 20)) - 1
                buf += rng(base + sh, off + p, off + q); p = q + 1
            dt = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}[m["dtype"]]
            out[k] = torch.frombuffer(buf, dtype=dt).reshape(m["shape"]).clone()
    os.makedirs(a.out, exist_ok=True)
    if a.experts_fp8:
        save_file(experts_fp8(out), os.path.join(a.out, "mtp_fp8e.safetensors"), metadata={"format": "pt"})
        return
    save_file(out, os.path.join(a.out, "mtp_bf16.safetensors"), metadata={"format": "pt"})


if __name__ == "__main__":
    main()
