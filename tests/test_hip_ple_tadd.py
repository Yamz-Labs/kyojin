"""PLE transposed add (exllamav3.modules.ple._ple_tadd, Triton): one tiled transpose-through-shared-memory pass that adds the
(bsz, ch, seq)-transposed conv output straight onto the fp32 gated accumulator, replacing a strided broadcast add (EXL3_PLE_CONTIG's
old fallback: one transpose-copy, then a plain vectorized add). Checked bit-identical to the torch reference (fp16 widened to fp32,
then added) at the real Qwen3.8-Flash-Next PLE shape (H=4, D=2560) and a couple of edge shapes (short / odd seq, 2 batches, small D).
Runs under pytest or as a plain script (needs Triton + a CUDA/HIP device; skips otherwise)."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3.modules import ple  # noqa: E402

CASES = [  # (bsz, seq, H, D)
    (1, 8192, 4, 2560),
    (1, 1109, 4, 2560),
    (1, 37, 4, 2560),
    (2, 300, 4, 64),
    (1, 1, 4, 2560),
]


def _avail():
    if not torch.cuda.is_available() or ple._ple_tadd_kernel is None:
        return False
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


def run_case(bsz, seq, H, D, seed):
    dev = torch.device("cuda")
    ch = H * D
    g = torch.Generator(device="cpu").manual_seed(seed)
    y = (torch.randn(bsz, ch, seq, generator=g) * 2).half().to(dev)
    co = torch.nn.functional.silu(y).transpose(1, 2)  # (bsz, seq, ch), transposed view, stride(1) != 1
    gated = (torch.randn(bsz, seq, H, D, generator=g)).to(dev)
    ref = gated + co.reshape(bsz, seq, H, D).float()
    g2 = gated.clone()
    done = ple._ple_tadd(g2, co)
    torch.cuda.synchronize()
    ok = done and bool((g2 == ref).all())
    print(f"ple_tadd bsz={bsz} seq={seq} H={H} D={D}: dispatched={done} bit_identical={bool((g2 == ref).all())} -> "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def run_all():
    if not _avail():
        print("SKIP: needs Triton and a CUDA/HIP device")
        return None
    fails = sum(not run_case(*c, seed=100 + i) for i, c in enumerate(CASES))
    print(f"{'PASS' if fails == 0 else 'FAIL'}: {len(CASES) - fails}/{len(CASES)} cases")
    return fails == 0


def test_hip_ple_tadd():
    import pytest
    res = run_all()
    if res is None:
        pytest.skip("Triton PLE transposed-add op not available")
    assert res


if __name__ == "__main__":
    sys.exit(0 if run_all() in (True, None) else 1)
