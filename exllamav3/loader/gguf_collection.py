"""GGUF tensor collection: serve raw GGUF payloads to the
engine's module loaders with the same call shape as SafetensorsCollection.

Quantized tensors come back as raw uint8 bytes shaped [N, nb*block_bytes]
(row-major GGUF order: dims[0] = K tiled in 256-groups, dims[1] = N rows,
dims[2] = experts), ready for the ForeignLinear q8_1-activation dots.
F16/F32 tensors come back as half/float with engine [N, K] (transposed vs
the stored dims) or flat for vectors. Reads go through pread into numpy
and a single H2D copy per tensor (no conversion, no requant).

Only the methods the modules call are implemented: has_tensor,
get_tensor (optional/allow_bf16/float2half/no_defer/transpose kwargs),
get_tensor_sizes. Anything else raises.
"""
from __future__ import annotations

import os
import re
import struct

import numpy as np
import torch

GGML_BBS = {
    12: 144,  # Q4_K
    13: 176,  # Q5_K
    14: 210,  # Q6_K
    23: 136,  # IQ4_XS
}
# (elements, bytes) per block of the formats served by gap_formats / fq_gemv_q8_0 (32- and 64-element blocks)
GGML_BLK = {t: (256, bb) for t, bb in GGML_BBS.items()}
GGML_BLK.update({2: (32, 18), 6: (32, 22), 7: (32, 24), 8: (32, 34), 20: (32, 18), 18: (256, 98), 21: (256, 110),
                 39: (32, 17), 40: (64, 36), 11: (256, 110), 17: (256, 74), 22: (256, 82), 10: (256, 84), 16: (256, 66)})
GGML_QUANT = set(GGML_BLK.keys())

_META_SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def _pad_to(t: torch.Tensor, pad_to) -> torch.Tensor:
    """Zero-pad like Linear.pad_out (STC pads on load when pad_to is passed):
    2-D [in, out] -> pad_to (in, out); 1-D bias -> pad_to[0]."""
    if pad_to is None:
        return t
    if t.dim() == 2:
        ti, to = pad_to[0], pad_to[1]
        if t.shape[0] >= ti and t.shape[1] >= to:
            return t
        p = torch.zeros((max(t.shape[0], ti), max(t.shape[1], to)),
                        dtype=t.dtype, device=t.device)
        p[:t.shape[0], :t.shape[1]] = t
        return p.contiguous()
    to = pad_to[0] if isinstance(pad_to, tuple) else pad_to
    if t.shape[0] >= to:
        return t
    p = torch.zeros((to,), dtype=t.dtype, device=t.device)
    p[:t.shape[0]] = t
    return p.contiguous()


class GGUFCollection:

    def __init__(self, path: str):
        self.path = path
        # split packs: "<name>-00001-of-0000N.gguf" -> every file; tensor offsets carry the file index in bits 48+
        real = os.path.realpath(path)
        m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", real)
        paths = [real] if m is None else [real[:m.start()] + f"-{i:05d}-of-{m.group(2)}.gguf" for i in range(1, int(m.group(2)) + 1)]
        self.fds, self.bases, self.meta, self.infos = [], [], {}, {}
        for i, p in enumerate(paths):
            meta, infos, base = self._read_header(p)
            for k, v in meta.items():
                self.meta.setdefault(k, v)
            for name, (ty, dims, off) in infos.items():
                self.infos[name] = (ty, dims, off | (i << 48))
            self.fds.append(os.open(p, os.O_RDONLY))
            self.bases.append(base)
        self.fd, self.base = self.fds[0], self.bases[0]
        # SafetensorsCollection attrs poked by Config/model_init paths
        self.directory = os.path.dirname(self.path)
        self.tensor_file_map = {k: self.path for k in self.infos}
        self.file_headers = {}
        # Engine-key -> gguf-name translation (set by the arch config).
        # Values are gguf names, or (gguf name, "T") when the engine wants
        # the transpose of the stored orientation (e.g. the GDN conv).
        self.name_map: dict[str, str | tuple] = {}
        # In-memory overrides (e.g. dequantized embedding), served first.
        self.new_tensors: dict[str, torch.Tensor] | None = None

    @staticmethod
    def _read_header(p: str):
        with open(p, "rb") as f:
            assert f.read(4) == b"GGUF"
            f.read(4)
            nt = struct.unpack("<Q", f.read(8))[0]
            nm = struct.unpack("<Q", f.read(8))[0]

            def rs():
                n = struct.unpack("<Q", f.read(8))[0]
                return f.read(n).decode(errors="replace")

            _SC = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}

            def sv(t=None):
                if t is None:
                    t = struct.unpack("<I", f.read(4))[0]
                if t in _SC:
                    return struct.unpack(_SC[t], f.read(struct.calcsize(_SC[t])))[0]
                if t == 8:
                    return rs()
                if t == 9:
                    at = struct.unpack("<I", f.read(4))[0]
                    n = struct.unpack("<Q", f.read(8))[0]
                    return [sv(at) for _ in range(n)]
                raise ValueError(f"gguf metadata type {t}")

            # Metadata kept (config + tokenizer are derived from it, see loader/gguf_config.py)
            meta = {}
            for _ in range(nm):
                k = rs()
                meta[k] = sv()
            infos = {}
            for _ in range(nt):
                name = rs()
                nd = struct.unpack("<I", f.read(4))[0]
                dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
                ty = struct.unpack("<I", f.read(4))[0]
                off = struct.unpack("<Q", f.read(8))[0]
                infos[name] = (ty, tuple(dims), off)
            base = (f.tell() + 31) // 32 * 32
        return meta, infos, base

    def _resolve(self, key: str) -> str:
        mapped = self.name_map.get(key, key)
        return mapped[0] if isinstance(mapped, tuple) else mapped

    def has_tensor(self, key: str) -> bool:
        if self.new_tensors and key in self.new_tensors:
            return True
        return self._resolve(key) in self.infos

    # Deferred-load arena API (SafetensorsCollection): ForeignLinear and the
    # fp16 paths load eagerly, so these are no-ops. Present because
    # model_ls brackets every load with them.
    def begin_deferred_load(self, arena=True):
        return None

    def end_deferred_load(self):
        return None

    def abort_deferred_load(self):
        return None

    def drop_page_cache(self):
        return None

    def close(self):
        for fd in getattr(self, "fds", [self.fd]):
            try:
                os.close(fd)
            except OSError:
                pass

    def has_tensor_group(self, key, subkeys) -> bool:
        if isinstance(key, list):
            return all(self.has_tensor_group(k, subkeys) for k in key)
        for sk in subkeys:
            if isinstance(sk, str):
                if not self.has_tensor(f"{key}.{sk}"):
                    return False
            elif not any(self.has_tensor(f"{key}.{s}") for s in sk):
                return False
        return True

    def get_tensor_sizes(self, key: str):
        info = self.infos.get(self._resolve(key))
        if info is None:
            raise KeyError(key)
        return info[1]

    def _pread(self, off: int, n: int) -> np.ndarray:
        fi, off = off >> 48, off & ((1 << 48) - 1)
        fd, base = self.fds[fi], self.bases[fi]
        buf = bytearray(n)
        mv = memoryview(buf)
        got = 0
        while got < n:
            r = os.pread(fd, n - got, base + off + got)
            if not r:
                raise IOError(f"short read at {off + got}")
            mv[got:got + len(r)] = r
            got += len(r)
        return np.frombuffer(buf, dtype=np.uint8)

    def get_tensor(
        self,
        key: str,
        device: torch.device | None = None,
        optional: bool = False,
        allow_bf16: bool = False,
        float2half: bool = False,
        no_defer: bool = False,
        transpose: bool = False,
        pad_to: tuple | None = None,
        fidx=None,
    ) -> torch.Tensor | None:
        if self.new_tensors and key in self.new_tensors:
            t = self.new_tensors[key]
            t = t.to(device) if device is not None else t
            # Overrides are stored in engine orientation; ignore transpose.
            if float2half and t.dtype == torch.float32:
                t = t.half()
            return _pad_to(t, pad_to)
        # Expert-slice protocol: "<gguf3d>.e<idx>[.weight]" serves expert
        # idx's [N, nb*bb] rows out of a [K, N, E] tensor (contiguous expert).
        exp_idx = None
        body = key[:-7] if key.endswith(".weight") else key
        if key not in self.infos and ".e" in body:
            base_body, esuf = body.rsplit(".e", 1)
            if esuf.isdigit() and (base_body + ".weight") in self.infos:
                key, exp_idx = base_body + ".weight", int(esuf)
        mapped = self.name_map.get(key, key)
        want_T = False
        if isinstance(mapped, tuple):
            mapped, want_T = mapped[0], (mapped[1] == "T")
        info = self.infos.get(mapped)
        if info is None:
            if not optional:
                raise ValueError(f"Required tensor {key} not found in {self.path}")
            return None
        ty, dims, off = info
        dev = device if device is not None else torch.device("cpu")
        if ty in GGML_QUANT:
            qk, bb = GGML_BLK[ty]
            K, N = dims[0], dims[1]
            assert K % qk == 0
            nb = K // qk
            if exp_idx is not None:
                assert len(dims) == 3
                E = dims[2]
                assert 0 <= exp_idx < E
                raw = self._pread(off + exp_idx * N * nb * bb, N * nb * bb)
                raw = raw.reshape(N, nb * bb)
            else:
                extra = int(np.prod(dims[2:])) if len(dims) > 2 else 1
                raw = self._pread(off, extra * N * nb * bb).reshape(extra * N, nb * bb)
            t = torch.from_numpy(np.ascontiguousarray(raw))
            if transpose:
                raise ValueError("transpose not supported for quant payloads")
            return t.to(dev)
        if ty in (0, 1, 30):  # F32 / F16 / BF16 (BF16 widened to float32: bit shift, exact)
            dt = np.float32 if ty == 0 else np.float16
            es = 4 if ty == 0 else 2
            n = int(np.prod(dims))
            raw = self._pread(off, n * es)
            if ty == 30:
                a = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
            else:
                a = np.frombuffer(raw, dtype=dt)
            # GGUF dims are ggml order (ne[0] = row length, fastest): the TRUE
            # matrix orientation is [dims[1], dims[0}] (dims[1] rows of dims[0]).
            # (A scrambled (dims[0], dims[1]) view silently corrupts every 2-D
            # F32 consumer, e.g. the MoE router: flat-identical, corr ~0.)
            # The transpose flag converts to engine [in, out] (like the STC);
            # tensors mapped with the "T" mark are transposed (e.g. GDN conv
            # is served [C, K] as stored; the engine flats it directly).
            if len(dims) == 2:
                a = a.reshape(dims[1], dims[0])
                if transpose or want_T:
                    a = a.T
            else:
                a = a.reshape(*dims) if len(dims) > 1 else a
            if key.endswith("shared_expert_gate.weight") and a.ndim == 1:
                a = a.reshape(-1, 1)  # Linear(in=H, out=1) weight [H, 1]
            t = torch.from_numpy(np.ascontiguousarray(a))
            if float2half and t.dtype == torch.float32:
                t = t.half()
            elif allow_bf16 and t.dtype == torch.float32:
                t = t.bfloat16()
            return _pad_to(t.to(dev), pad_to)
        raise ValueError(f"unsupported gguf type {ty} for {key}")
