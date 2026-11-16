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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3.6-35B-A3B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size-only", action="store_true")
    a = ap.parse_args()
    base = f"https://huggingface.co/{a.repo}/resolve/main/"
    idx = json.loads(urllib.request.urlopen(base + "model.safetensors.index.json").read())["weight_map"]
    keys = [k for k in idx if k.startswith("mtp.")]
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
    save_file(out, os.path.join(a.out, "mtp_bf16.safetensors"), metadata={"format": "pt"})


if __name__ == "__main__":
    main()
