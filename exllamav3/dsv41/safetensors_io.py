"""
Official DeepSeek-V4.1-Flash checkpoint (HF safetensors shards) without mmap: header parse + pread.

  SafetensorsIndex(dir_or_files)    tensor name -> (file, dtype, shape, absolute offset, nbytes)
  LazyTensorMap                     Mapping[name -> tensor] reading on demand (experts / dense stream in)
  SafetensorsWeightSource           MapWeightSource whose Engram tables are read row-wise with O_DIRECT
"""
from __future__ import annotations
import glob
import json
import os
import struct
from collections.abc import Mapping
import torch
from .weights import MapWeightSource
from .engram import DirectRowReader, FileEngramStore, make_reader

_DT = {
    "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16, "F64": torch.float64,
    "I8": torch.int8, "U8": torch.uint8, "I16": torch.int16, "I32": torch.int32, "I64": torch.int64,
    "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2, "F8_E8M0": torch.float8_e8m0fnu,
    "F4": torch.float4_e2m1fn_x2,
}


class SafetensorsIndex:
    def __init__(self, src):
        files = sorted(glob.glob(os.path.join(src, "*.safetensors"))) if isinstance(src, str) and os.path.isdir(src) \
            else ([src] if isinstance(src, str) else list(src))
        assert files, f"no safetensors in {src}"
        self.t = {}
        for f in files:
            fd = os.open(f, os.O_RDONLY)
            try:
                n = struct.unpack("<Q", os.pread(fd, 8, 0))[0]
                hdr = json.loads(os.pread(fd, n, 8))
            finally:
                os.close(fd)
            for k, v in hdr.items():
                if k == "__metadata__":
                    continue
                b, e = v["data_offsets"]
                self.t[k] = (f, v["dtype"], tuple(v["shape"]), 8 + n + b, e - b)

    def __contains__(self, k):
        return k in self.t

    def info(self, k):
        return self.t[k]

    def read(self, k) -> torch.Tensor:
        f, dt, shape, off, nb = self.t[k]
        fd = os.open(f, os.O_RDONLY)
        try:
            buf = bytearray(nb)
            got = os.preadv(fd, [buf], off)
            if got != nb:
                raise IOError(f"{k}: short read {got}/{nb}")
        finally:
            os.close(fd)
        t = torch.frombuffer(buf, dtype=torch.uint8) if nb else torch.empty(0, dtype=torch.uint8)
        d = _DT[dt]
        return t.view(d).reshape(shape) if d != torch.uint8 else t.reshape(shape)


class LazyTensorMap(Mapping):
    def __init__(self, index: SafetensorsIndex):
        self.ix = index

    def __contains__(self, k):
        return k in self.ix

    def __getitem__(self, k):
        if k not in self.ix:
            raise KeyError(k)
        return self.ix.read(k)

    def __iter__(self):
        return iter(self.ix.t)

    def __len__(self):
        return len(self.ix.t)


def engram_store_from_index(ix, layer_ids, allow_buffered: bool = False):
    """Row store over the Engram tensors of safetensors shards, read in place (no copy). The format follows the tensors:
    e4m3 weights [N, hd] + e8m0 scales [N, hd/32] -> "fp8"; U8 [N, hd/2] -> the MXFP4 transcode."""
    readers, rows, fmts = [], [], []
    for lid in layer_ids:
        k = f"layers.{lid}.engram.embed"
        kw = k + ".weight" if k + ".weight" in ix else "model." + k + ".weight"
        ks = kw[:-6] + "scale"
        fw, dw, sw, ow, nw = ix.info(kw)
        fs, ds, ss, os_, ns = ix.info(ks)
        assert sw[0] == ss[0] and sw[1] in (ss[1] * 32, ss[1] * 16), (sw, ss)
        f = "fp8" if sw[1] == ss[1] * 32 else "mxfp4"
        assert not fmts or fmts[0] == f
        fmts.append(f)
        hd = sw[1]
        readers.append(make_reader([(fw, ow, hd, hd), (fs, os_, ss[1], ss[1])], allow_buffered=allow_buffered))
        rows.append(sw[0])
    return FileEngramStore(readers, rows, fmts[0] if fmts else "fp8")


class SafetensorsWeightSource(MapWeightSource):
    def __init__(self, src, n_layers: int, engram_layer_ids=(), allow_buffered_engram: bool = False):
        self.ix = SafetensorsIndex(src)
        super().__init__(LazyTensorMap(self.ix), n_layers)
        self._ids, self._buf, self._store = list(engram_layer_ids), allow_buffered_engram, None

    def engram_store(self):
        if self._store is None:
            self._store = engram_store_from_index(self.ix, self._ids, self._buf)
        return self._store

    def close(self):
        if self._store is not None:
            self._store.close()
