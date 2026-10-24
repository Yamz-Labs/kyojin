"""Fetch single tensors from the remote checkpoint with HTTP Range requests.

    python tools/ds41/fetch.py --map tensor_map.json --name layers.0.attn.wq_a.weight --out wq_a.bin [--sha256 HEX]

Large tensors are read in chunks (default 64 MiB). With a destination file the download is
resumable: bytes already in `<dest>.part` are kept and reading continues from there.
"""

import argparse
import hashlib
import os
import re
import sys
import threading

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

CHUNK = 64 << 20

TORCH_DTYPE = {
    "BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F64": torch.float64,
    "I8": torch.int8, "U8": torch.uint8, "I16": torch.int16, "I32": torch.int32, "I64": torch.int64,
    "BOOL": torch.bool,
    "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2, "F8_E8M0": torch.float8_e8m0fnu,
}


class TensorFetcher:
    def __init__(self, tmap: dict, chunk: int = CHUNK, cache: str | None = None):
        self.map = tmap
        self.cache = cache   # local per-layer cache dir shared with the capture process (see cache_path)
        self.chunk = chunk
        self.repo = tmap.get("repo", C.REPO)
        self.revision = tmap.get("revision", C.REVISION)

    def info(self, name: str) -> dict:
        return self.map["tensors"][name]

    def _url(self, info):
        return C.file_url(info["file"], self.repo, self.revision)

    def iter_chunks(self, name: str, start: int = 0):
        """Yield the tensor bytes as consecutive chunks, beginning at byte `start` of the tensor."""
        info = self.info(name)
        url = self._url(info)
        pos = start
        while pos < info["length"]:
            n = min(self.chunk, info["length"] - pos)
            a = info["offset"] + pos
            yield C.range_get(url, a, a + n - 1)
            pos += n

    def cache_path(self, name: str) -> str | None:
        if not self.cache:
            return None
        m = re.match(r"(layers|mtp)\.(\d+)\.", name)
        return os.path.join(self.cache, f"{m.group(1)}.{m.group(2)}" if m else "global", name)

    def fetch_bytes(self, name: str, sha256: str | None = None) -> bytearray:
        info = self.info(name)
        cp = self.cache_path(name)
        if cp is not None and os.path.exists(cp) and os.path.getsize(cp) == info["length"]:
            with open(cp, "rb") as f:
                buf = bytearray(f.read())
        else:
            buf = bytearray(info["length"])
            pos = 0
            for c in self.iter_chunks(name):
                buf[pos:pos + len(c)] = c
                pos += len(c)
            if cp is not None:
                os.makedirs(os.path.dirname(cp), exist_ok=True)
                tmp = f"{cp}.tmp.{os.getpid()}.{threading.get_ident()}"
                with open(tmp, "wb") as f:
                    f.write(buf)
                os.replace(tmp, cp)
        if sha256 is not None:
            got = hashlib.sha256(buf).hexdigest()
            if got != sha256.lower():
                raise ValueError(f"{name}: sha256 {got} != expected {sha256}")
        return buf

    def fetch(self, name: str, sha256: str | None = None) -> torch.Tensor:
        """Return the tensor with its stored dtype and shape (CPU)."""
        info = self.info(name)
        buf = self.fetch_bytes(name, sha256)
        t = torch.frombuffer(buf, dtype=torch.uint8)
        dt = TORCH_DTYPE[info["dtype"]]
        if dt != torch.uint8:
            t = t.view(dt)
        return t.reshape(info["shape"])

    def fetch_to_file(self, name: str, dest: str, sha256: str | None = None) -> str:
        """Resumable raw download of the tensor bytes to `dest` (via `dest.part`)."""
        info = self.info(name)
        part = dest + ".part"
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if have > info["length"]:
            os.remove(part)
            have = 0
        with open(part, "ab") as f:
            for c in self.iter_chunks(name, have):
                f.write(c)
                f.flush()
        if os.path.getsize(part) != info["length"]:
            raise IOError(f"{name}: incomplete download")
        if sha256 is not None:
            h = hashlib.sha256()
            with open(part, "rb") as f:
                for blk in iter(lambda: f.read(1 << 24), b""):
                    h.update(blk)
            if h.hexdigest() != sha256.lower():
                os.remove(part)
                raise ValueError(f"{name}: sha256 {h.hexdigest()} != expected {sha256}")
        os.replace(part, dest)
        return dest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sha256")
    a = ap.parse_args()
    import tensor_map
    f = TensorFetcher(tensor_map.load(a.map))
    f.fetch_to_file(a.name, a.out, a.sha256)
    h = hashlib.sha256(open(a.out, "rb").read()).hexdigest()
    print(f"{a.name} -> {a.out} ({f.info(a.name)['length']} bytes) sha256={h}")
