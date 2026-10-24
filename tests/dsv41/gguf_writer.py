"""Minimal GGUF v3 writer for tests (independent of gguf-py, which is used only as a decoding oracle)."""
import struct
import numpy as np
import torch

Q8_0, Q2_K, IQ2_XXS, F32, F16, I8 = 8, 10, 16, 0, 1, 24


def _s(x):
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def _kv(k, v):
    out = _s(k)
    if isinstance(v, str):
        return out + struct.pack("<I", 8) + _s(v)
    if isinstance(v, bool):
        return out + struct.pack("<IB", 7, v)
    if isinstance(v, int):
        return out + struct.pack("<II", 4, v)
    if isinstance(v, float):
        return out + struct.pack("<If", 6, v)
    if isinstance(v, np.ndarray):
        et = {np.dtype(np.uint32): 4, np.dtype(np.uint64): 10, np.dtype(np.int32): 5}[v.dtype]
        return out + struct.pack("<IIQ", 9, et, v.size) + v.tobytes()
    raise TypeError(type(v))


def quantize_q8_0(w: torch.Tensor) -> bytes:
    x = w.reshape(-1, 32).float()
    d = (x.abs().amax(-1, keepdim=True) / 127.0).to(torch.float16)
    q = torch.round(x / d.float().clamp_min(1e-30)).clamp(-127, 127).to(torch.int8)
    blk = np.zeros((x.shape[0], 34), dtype=np.uint8)
    blk[:, :2] = d.view(torch.uint8).numpy().reshape(-1, 2)
    blk[:, 2:] = q.view(torch.uint8).numpy()
    return blk.tobytes()


def random_blocks(gtype, n_blocks, rng):
    """Valid random blocks (valid fp16 scales), for formats we do not need an encoder for."""
    if gtype == Q2_K:
        b = rng.integers(0, 256, size=(n_blocks, 84), dtype=np.uint8)
        b[:, 80:82] = (rng.random((n_blocks, 1)).astype(np.float16) * 0.05 + 0.01).view(np.uint8).reshape(-1, 2)
        b[:, 82:84] = (rng.random((n_blocks, 1)).astype(np.float16) * 0.05 + 0.01).view(np.uint8).reshape(-1, 2)
        return b.tobytes()
    if gtype == IQ2_XXS:
        b = rng.integers(0, 256, size=(n_blocks, 66), dtype=np.uint8)
        b[:, 0:2] = (rng.random((n_blocks, 1)).astype(np.float16) * 0.02 + 0.005).view(np.uint8).reshape(-1, 2)
        return b.tobytes()
    raise ValueError(gtype)


class GGUFBuilder:
    def __init__(self, alignment=4096):
        self.kv = []
        self.tensors = []          # (name, dims, gtype, bytes)
        self.alignment = alignment

    def add_kv(self, k, v):
        self.kv.append((k, v))

    def add(self, name, shape_torch, gtype, data: bytes):
        self.tensors.append((name, tuple(reversed(shape_torch)), gtype, data))

    def write(self, path):
        al = self.alignment
        kv = list(self.kv) + [("general.alignment", al)]
        head = b"GGUF" + struct.pack("<IQQ", 3, len(self.tensors), len(kv))
        for k, v in kv:
            head += _kv(k, v)
        offs, cur = [], 0
        for name, dims, gt, data in self.tensors:
            offs.append(cur)
            cur += (len(data) + al - 1) // al * al
        info = b""
        for (name, dims, gt, data), o in zip(self.tensors, offs):
            info += _s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
            info += struct.pack("<IQ", gt, o)
        hdr = head + info
        pad = (-len(hdr)) % al
        with open(path, "wb") as f:
            f.write(hdr + b"\0" * pad)
            for (name, dims, gt, data), o in zip(self.tensors, offs):
                f.write(data)
                f.write(b"\0" * ((-len(data)) % al))
