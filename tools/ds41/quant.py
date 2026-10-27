"""Quantizer back-ends shared by the single-process converter and the split workers.

A quantizer is a callable `q(key, w_bf16[out, in], bits, hess=None) -> {suffix: cpu tensor}`.
`hess` is None (uncalibrated: meta Hessian) or a `HessianEntry` (see split.py) carrying the captured
Hessian of the linear's input. Entries are shared between linears that share an input (gate/up of
every expert of a layer), and the quantizer caches its derived state on the entry.

* Exl3Quantizer: exllamav3's `quantize_exl3` (ROCm on the box, CUDA on the 3090).
* StubQuantizer: deterministic, GPU-free stand-in for tests. Output is a pure function of
  (key, bits, weight bytes, Hessian bytes), so any worker produces the same bytes.
"""

import sys
import zlib

import torch

SAFE_DTYPE = {
    torch.bfloat16: "BF16", torch.float16: "F16", torch.float32: "F32", torch.int32: "I32",
    torch.int16: "I16", torch.int8: "I8", torch.uint8: "U8",
}


def quant_record(bits: int, q: dict) -> dict:
    """`stored` marker record of one quantized linear (same layout as process_unit writes)."""
    rec = {"bits": bits, "stored": {}}
    for suffix, t in q.items():
        rec["stored"][suffix] = {"dtype": SAFE_DTYPE[t.dtype], "shape": list(t.shape)}
        if suffix == "mul1":
            rec["mul1_multiplier"] = int(t.item()) & 0xFFFFFFFF
    return rec


class Exl3Quantizer:
    """Thin wrapper around exllamav3.modules.quant.exl3_lib.quantize.quantize_exl3."""

    def __init__(self, device: str, codebook: str = "mul1", out_scales: bool = True):
        from exllamav3.modules.quant.exl3_lib import quantize as qmod
        self._qmod = qmod
        self._quantize = qmod.quantize_exl3
        self.device = device
        self.codebook = codebook
        self.out_scales = out_scales
        import threading
        self._lock = threading.Lock()
        self._tl = threading.local()
        self.proxy = {}   # key -> proxy error of the last call (calibrated linears; 0.0 when uncalibrated)
        self.energy = {}  # key -> mean output energy tr(W H W^T)/count of the linear on its calibration inputs

    def thread_init(self):
        """Each compute thread runs on its own CUDA stream (exllamav3's EXL3_QUANT_THREADS mechanism); the
        result is bit-identical to a single thread."""
        stream = torch.cuda.Stream(torch.device(self.device))
        self._tl.stream = stream
        self._qmod.set_worker_stream(stream)

    def _h_data(self, key: str, in_f: int, hess):
        if hess is None:
            return {
                "H": torch.empty((in_f, in_f), device="meta"),  # meta Hessian = uncalibrated path
                "first_key": key, "count": 0, "finalized": False, "num_total": 0,
                "inf_nan": torch.zeros(2, dtype=torch.long, device=self.device),
                "device": self.device,
            }
        with self._lock:
            if hess.cache is None:
                hd = hess.load()
                raw_H = hd["H"].clone() if str(self.device) == "cpu" else hd["H"]   # finalize works in place on cpu
                cache = {
                    "H": hd["H"].to(self.device),
                    "first_key": hess.name, "count": hd["count"], "finalized": False,
                    "num_total": hd["num_total"],
                    "inf_nan": torch.tensor(hd["inf_nan"], dtype=torch.long, device=self.device),
                    "device": self.device,
                }
                # Draw the input sign vector with a seed that depends on the Hessian only, not on
                # whichever linear reaches it first: every worker then derives the same su, L, H for
                # the same entry, and all experts of a layer share su exactly as one process would.
                gen = torch.Generator(device=self.device)
                gen.manual_seed(zlib.crc32(hess.name.encode()))
                self._qmod.finalize_capture_H(cache, {"K": 2}, False, gen)
                if torch.cuda.is_available():
                    torch.cuda.current_stream(torch.device(self.device)).synchronize()   # visible to other streams
                hess.raw = (raw_H, hd["count"])   # un-normalized sum of x^T x, original basis (for energy())
                hess.cache = cache   # published only once finalized
        return hess.cache

    def _energy(self, key, weight_out_in, hess):
        """Mean output energy of the linear on the calibration inputs, E = tr(W H W^T) / count (original basis, no
        regularization). Together with the relative proxy error it gives the absolute output error of a candidate:
        err_abs = proxy * E. None for an uncalibrated linear."""
        if hess is None or getattr(hess, "raw", None) is None:
            return None
        ek = (key, tuple(weight_out_in.shape))
        done = hess.__dict__.setdefault("energy_cache", {})
        if ek not in done:
            H, count = hess.raw
            Hd = H.to(self.device)
            W = weight_out_in.to(self.device).float()
            done[ek] = float(((W @ Hd) * W).sum().item()) / max(1, count)
        return done[ek]

    def __call__(self, key: str, w_bf16: torch.Tensor, bits: int, hess=None) -> dict:
        """w_bf16: [out, in]. Returns {suffix: cpu tensor} for the EXL3 tensors of this linear.
        On a compute thread with its own stream (thread_init) every torch op of the call must run on that
        stream too, exactly as convert_model's quantize worker does, or the custom kernels race with the
        default-stream ops."""
        stream = getattr(self._tl, "stream", None)
        if stream is None:
            return self._call(key, w_bf16, bits, hess)
        with torch.cuda.stream(stream):
            out = self._call(key, w_bf16, bits, hess)
        stream.synchronize()
        return out

    def _call(self, key, w_bf16, bits, hess):
        out_f, in_f = w_bf16.shape
        H_data = self._h_data(key, in_f, hess)
        qargs = {"seed": zlib.crc32(key.encode()), "K": bits, "devices": [self.device],
                 "device_ratios": None, "apply_out_scales": self.out_scales}
        if self.codebook in ("mul1", "mcg"):
            qargs[self.codebook] = True
        weight = w_bf16.float().T.contiguous()
        _, proxy, out = self._quantize(weight, H_data, qargs, False)
        self.proxy[key] = float(proxy)
        try:
            en = self._energy(key, w_bf16, hess)
        except Exception as ex:   # statistics only: never fail a quantization for them
            en = None
            if not getattr(self, "_warned", False):
                self._warned = True
                print(f"WARNING energy computation failed ({type(ex).__name__}: {ex})", file=sys.stderr, flush=True)
        if en is not None:
            self.energy[key] = en
        return {k: v.cpu().contiguous() for k, v in out.items()}


class CpuExl3Quantizer:
    """Uncalibrated EXL3 quantizer on the CPU (exllamav3.modules.quant.exl3_cpu): the real tensor layout, no GPU.
    Used by the offline end-to-end tests and for small dry packs."""

    def __init__(self, out_scales: bool = True):
        self.out_scales = out_scales
        self.proxy = {}
        self.energy = {}

    def __call__(self, key: str, w_bf16: torch.Tensor, bits: int, hess=None) -> dict:
        from exllamav3.modules.quant import exl3_cpu
        out = exl3_cpu.quantize_linear(w_bf16, bits, seed=zlib.crc32(key.encode()), out_scales=self.out_scales)
        w = w_bf16.float()
        rec = exl3_cpu.reconstruct(out["trellis"], out["suh"], out["svh"]).float().t()      # [out, in]
        self.proxy[key] = float((rec - w).square().sum() / w.square().sum().clamp_min(1e-30))
        return out


class StubQuantizer:
    """Deterministic fake quantizer with the real output layout (trellis, suh, svh, mul1)."""

    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self.proxy = {}
        self.energy = {}

    def __call__(self, key: str, w_bf16: torch.Tensor, bits: int, hess=None) -> dict:
        import time
        if self.delay:
            time.sleep(self.delay)
        out_f, in_f = w_bf16.shape
        crc = zlib.crc32(key.encode())
        crc = zlib.crc32(w_bf16.contiguous().view(torch.int16).numpy().tobytes(), crc)
        crc = zlib.crc32(bytes([bits]), crc)
        if hess is not None:
            crc = zlib.crc32(hess.digest().to_bytes(4, "little"), crc)
        # deterministic fake metrics with the real scaling (error x 1/4.07 per bit, K=2 ~ 0.049) for the allocation tests
        u = (zlib.crc32(key.encode()) % 1000) / 1000.0
        if hess is not None:
            self.proxy[key] = 0.0489 * 4.07 ** (2 - bits) * (0.7 + 0.6 * u)
            self.energy[key] = 0.5 + 2.0 * ((zlib.crc32(key.encode(), 7) % 1000) / 1000.0)
        g = torch.Generator().manual_seed(crc)
        return {
            "trellis": torch.randint(-32768, 32768, (in_f // 16, out_f // 16, 16 * bits), generator=g,
                                     dtype=torch.int16),
            "suh": torch.randn(in_f, generator=g).to(torch.float16),
            "svh": torch.randn(out_f, generator=g).to(torch.float16),
            "mul1": torch.tensor([0x83DCD12D - (1 << 32)], dtype=torch.int32),
        }
