"""Grouped MoE prefill down decode for 32-element blocks (fq_moe_prefill_kq._dq_q32: Q8_0 / Q5_1) vs a numpy reference of
the ggml block layouts: random blocks, 3 experts, bit-equal expected. Runs under pytest or as a script; skips without a GPU."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ != "__main__":
    import pytest

    if not torch.cuda.is_available():
        pytest.skip("needs a GPU (CUDA or ROCm)", allow_module_level=True)
    parametrize = pytest.mark.parametrize
else:
    def parametrize(*_a, **_k):
        return lambda f: f

from exllamav3.modules import fq_moe_prefill_kq as M  # noqa: E402


def _ref(fmt, raw, N, K):
    bb = 34 if fmt == M.FMT_Q8_0 else 24
    out = np.zeros((N, K), np.float32)
    for n in range(N):
        for kb in range(K // 32):
            b = raw[n, kb * bb:(kb + 1) * bb]
            d = np.frombuffer(b[0:2].tobytes(), np.float16)[0].astype(np.float32)
            if fmt == M.FMT_Q8_0:
                out[n, kb * 32:(kb + 1) * 32] = b[2:34].view(np.int8).astype(np.float32) * d
            else:
                m = np.frombuffer(b[2:4].tobytes(), np.float16)[0].astype(np.float32)
                qh = int(np.frombuffer(b[4:8].tobytes(), np.uint32)[0])
                qs = b[8:24]
                for j in range(16):
                    out[n, kb * 32 + j] = ((qs[j] & 15) | (((qh >> j) & 1) << 4)) * d + m
                    out[n, kb * 32 + j + 16] = ((qs[j] >> 4) | (((qh >> (j + 16)) & 1) << 4)) * d + m
    return out.astype(np.float16)


@parametrize("fmt", [M.FMT_Q8_0, M.FMT_Q5_1])
def test_dq_q32(fmt):
    rng = np.random.default_rng(0)
    E, N, K = 3, 40, 128
    bb = 34 if fmt == M.FMT_Q8_0 else 24
    raws = []
    for _ in range(E):
        raw = rng.integers(0, 256, (N, (K // 32) * bb), dtype=np.uint8)
        for kb in range(K // 32):                                              # sane f16 scales (and Q5_1 mins)
            raw[:, kb * bb:kb * bb + 2] = np.frombuffer(np.float16(rng.uniform(0.001, 0.05, N)).tobytes(), np.uint8).reshape(N, 2)
            if fmt == M.FMT_Q5_1:
                raw[:, kb * bb + 2:kb * bb + 4] = np.frombuffer(np.float16(rng.uniform(-0.1, 0.1, N)).tobytes(), np.uint8).reshape(N, 2)
        raws.append(raw)
    ts = [torch.from_numpy(r).cuda() for r in raws]
    ptrs = torch.tensor([t.data_ptr() for t in ts], dtype=torch.int64, device="cuda")
    out = torch.empty((E, N, K), dtype=torch.half, device="cuda")
    M._dq_q32[(M.triton.cdiv(N, 32), K // 32, E)](ptrs, out, 0, N, K, FMT=fmt, BN=32, num_warps=4)
    torch.cuda.synchronize()
    for e in range(E):
        assert np.array_equal(out[e].cpu().numpy().view(np.uint16), _ref(fmt, raws[e], N, K).view(np.uint16)), f"fmt {fmt} expert {e}"


if __name__ == "__main__":
    for f in (M.FMT_Q8_0, M.FMT_Q5_1):
        test_dq_q32(f)
    print("PASS")
