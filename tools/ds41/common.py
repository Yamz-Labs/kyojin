"""Shared helpers for the DeepSeek-V4.1-Flash streaming converter: HTTP range reads, dtype tables."""

import json
import os
import struct
import sys
import time
import urllib.error
import urllib.request

REPO = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"
EXPECTED_TENSORS = 96085
EXPECTED_SHARDS = 48
EXPECTED_TOTAL_SIZE = 510286023000

# safetensors dtype name -> bytes per element (F4 packs two values per byte)
DTYPE_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
    "U16": 2, "U32": 4, "U64": 8,
    "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1,
    "F4": 0.5,
}


def file_url(name: str, repo: str = REPO, revision: str = REVISION) -> str:
    base = os.environ.get("DS41_BASE_URL", "https://huggingface.co").rstrip("/")   # override for offline tests
    return f"{base}/{repo}/resolve/{revision}/{name}"


def numel(shape) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def expected_nbytes(dtype: str, shape) -> int:
    n = numel(shape)
    b = DTYPE_BYTES[dtype]
    if b == 0.5:
        assert n % 2 == 0
        return n // 2
    return int(n * b)


def _headers(extra=None):
    h = {"User-Agent": "ds41-stream-convert/1"}
    tok = os.environ.get("HF_TOKEN")
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    if extra:
        h.update(extra)
    return h


def range_get(url: str, start: int, end: int, retries: int = 8, timeout: float = 120.0,
              with_total: bool = False):
    """Read bytes [start, end] inclusive. Retries with backoff; verifies the returned length.
    With with_total=True returns (data, total_file_size)."""
    want = end - start + 1
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=_headers({"Range": f"bytes={start}-{end}"}))
            with urllib.request.urlopen(req, timeout=timeout) as r:
                if r.status != 206:
                    raise IOError(f"expected 206 for range request, got {r.status}")
                data = r.read()
                total = None
                cr = r.headers.get("Content-Range")
                if cr and "/" in cr:
                    tail = cr.rsplit("/", 1)[1]
                    total = int(tail) if tail.isdigit() else None
            if len(data) != want:
                raise IOError(f"short read: got {len(data)} of {want} bytes")
            return (data, total) if with_total else data
        except (urllib.error.URLError, IOError, TimeoutError, ConnectionError) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code in (401, 403, 404):
                raise
            last = e
            time.sleep(min(60.0, 1.5 * 2 ** attempt))
    raise IOError(f"range {start}-{end} of {url} failed after {retries} attempts: {last}")


def read_header(url: str):
    """Return (header_dict, header_len, file_size) of a remote safetensors file."""
    raw, total = range_get(url, 0, 7, with_total=True)
    n = struct.unpack("<Q", raw)[0]
    hdr = json.loads(range_get(url, 8, 8 + n - 1))
    return hdr, n, total


def fetch_json(name: str, repo: str = REPO, revision: str = REVISION) -> dict:
    req = urllib.request.Request(file_url(name, repo, revision), headers=_headers())
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(2 ** attempt)
    raise IOError(f"cannot fetch {name}")


def atomic_write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def _load_engine_module(name: str, rel: str):
    """Import a torch-free module of exllamav3/dsv41 by path (the converter tools must not need the engine installed)."""
    import importlib.util
    key = f"_ds41_{name}"
    if key in sys.modules:
        return sys.modules[key]
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "exllamav3", "dsv41", rel)
    spec = importlib.util.spec_from_file_location(key, os.path.abspath(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[key] = m
    spec.loader.exec_module(m)
    return m


def packspec():
    """The pack tensor-name / size / quantization_config spec (exllamav3/dsv41/packspec.py, stdlib only)."""
    return _load_engine_module("packspec", "packspec.py")


def dsv41_config():
    """exllamav3/dsv41/config.py (DSV41Config, stdlib only)."""
    return _load_engine_module("config", "config.py")
