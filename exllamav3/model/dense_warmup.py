"""Load-time warm-up of the dense GEMM paths.

The first call of a dense product at a row count class not seen before in a process (more exactly: a key (n, k, rows rounded up to 256,
row pitch, out dtype) missing from the dense-GEMM tune cache, see hgemm.cu dtune) screens every rocBLAS solution on the live operands
and costs 0.15-4 s inside one request. This runs each dense Linear shape once per row class a request can produce, so the screening
happens at load. With a populated tune cache file the warm-up is a few seconds of ordinary GEMMs; on a cold cache it pays the screening.
Switch: EXL3_WARM_DENSE=0 turns it off. Results are unchanged: tuned winners are bit-exact with the untuned path.
"""
from __future__ import annotations
import os, time
import torch

SMALL_ROWS = (1, 2, 3, 4, 5, 6, 7, 8, 16, 64, 255)   # untuned paths (decode, verify, short prompts): first-use of hipBLASLt / kernels


def _dense_linears(model):
    from ..modules.linear import Linear
    seen = {}
    for m in model:
        if not isinstance(m, Linear) or ".experts." in m.key or m.out_features == 0 or m.out_features > 65536:   # lm_head: only the last rows reach it
            continue
        inner = type(m.inner).__name__
        seen.setdefault((m.in_features, m.out_features, inner, str(m.out_dtype)), m)
    return list(seen.values())


@torch.inference_mode()
def warm_dense_gemm(models, max_rows: int = 4096, log=print) -> dict:
    """models: iterable of Model (target, draft). max_rows: largest prefill chunk in tokens. Returns stats."""
    if os.environ.get("EXL3_WARM_DENSE", "1") == "0":
        return {"skipped": True}
    from ..modules.quant.exl3 import row_pad_pitch
    t0 = time.perf_counter()
    mem0 = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    calls = 0
    nclass = (max_rows + 255) // 256
    for model in models:
        if model is None:
            continue
        for lin in _dense_linears(model):
            k = lin.in_features
            # The MTP input projection runs per stream: tokens x hc_count rows (4 x 4096 = class 64).
            mult = 4 if lin.key.endswith("mtp.fc_hidden") else 1
            rows_list = list(SMALL_ROWS) + [256 * c for c in range(1, nclass * mult + 1)]
            for rows in rows_list:
                # unpadded input (contiguous), and the row-padded input a producer gives (only above PAD_MIN_ROWS, and only
                # for activations of at most one prefill chunk: the per-stream MTP rows are never producer-padded)
                pitches = [0]
                if rows <= max_rows:
                    p = row_pad_pitch(lin, k, rows, {})
                    if p:
                        pitches.append(p)
                for pitch in pitches:
                    buf = torch.randn((rows, pitch or k), device=lin.device, dtype=torch.half).mul_(0.1)
                    x = buf[:, :k] if pitch else buf
                    lin.forward(x, {})
                    calls += 1
                    del buf, x
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - mem0
    torch.cuda.empty_cache()
    st = {"seconds": time.perf_counter() - t0, "calls": calls, "peak_extra_mib": peak / 2**20}
    log(f"warm_dense_gemm: {calls} calls in {st['seconds']:.1f} s, transient peak {st['peak_extra_mib']:.0f} MiB")
    return st
