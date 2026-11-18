"""CPU checks for EXL3_LEGACY_PREFILL_DQ (foreign_quant/legacy_dq.py): exact dequant vs gap_formats + dispatch in ForeignLinear._dequant_rows.
No GPU, no kernel launch: the extension is replaced by stubs. Needs the repo importable (exllamav3 package, built extension present so the
import does not start a build); skipped otherwise. Run: CUDA_VISIBLE_DEVICES= python -m pytest tests/test_legacy_prefill_dq_cpu.py -q"""
import glob
import os
import sys
import types

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not glob.glob(os.path.join(ROOT, "exllamav3_ext*.so")):
    pytest.skip("extension not built in " + ROOT + " (importing exllamav3 would start the build)", allow_module_level=True)
sys.path.insert(0, ROOT)
import torch



def test_legacy_dequant_exact():
    import torch
    from exllamav3.foreign_quant.gap_formats import dequant
    from exllamav3.foreign_quant import legacy_dq
    rng = np.random.default_rng(4)
    for fmt in ("q4_0", "q5_0"):
        raw = rng.integers(0, 256, size=(24, 40 * legacy_dq.BB[fmt]), dtype=np.uint8)      # fully random bytes incl. NaN / inf fp16 scales
        ref = legacy_dq.ref_dequant_np(fmt, raw)
        eng = dequant(fmt, torch.from_numpy(raw)).numpy()
        assert ref.shape == eng.shape, (ref.shape, eng.shape)
        nan = np.isnan(ref)
        assert np.array_equal(nan, np.isnan(eng)), fmt
        assert np.array_equal(ref.view(np.uint16)[~nan], eng.view(np.uint16)[~nan]), f"{fmt}: numpy reference != gap_formats.dequant"
        print(f"legacy dequant {fmt}: numpy reference == gap_formats.dequant bit-exact on {ref.size} random elements ({int(nan.sum())} NaN)")
    # the Triton kernel itself through the CPU interpreter, when available
    if legacy_dq.HAVE_TRITON:
        os.environ["TRITON_INTERPRET"] = "1"
        try:
            import importlib
            importlib.reload(legacy_dq)
            for fmt in ("q4_0", "q5_0"):
                raw = rng.integers(0, 256, size=(6, 4 * legacy_dq.BB[fmt]), dtype=np.uint8)
                out = legacy_dq.dequant(fmt, torch.from_numpy(raw)).numpy()
                ref = legacy_dq.ref_dequant_np(fmt, raw); nan = np.isnan(ref)
                assert np.array_equal(ref.view(np.uint16)[~nan], out.view(np.uint16)[~nan]), fmt
                print(f"legacy dequant {fmt}: Triton kernel (interpreter, CPU) == numpy reference")
        except Exception as e:      # the interpreter is optional: the GPU cell compares the real kernel
            print("triton interpreter check skipped:", type(e).__name__, str(e)[:120])
        finally:
            os.environ.pop("TRITON_INTERPRET", None)


class _FakeCudaBias:
    """Stand-in for a CUDA half bias on a CPU box."""
    is_cuda = True
    dtype = torch.half
    def __init__(self, n): self.n = n
    def is_contiguous(self): return True
    def numel(self): return self.n


@pytest.fixture(params=["0", "1"])
def flag(request, monkeypatch):
    monkeypatch.setattr(torch.version, "cuda", "13.0", raising=False)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    return request.param == "1"

def test_dispatch(flag, monkeypatch):
    from exllamav3.modules import foreign_linear as fl
    from exllamav3.foreign_quant import legacy_dq
    monkeypatch.setenv("EXL3_LEGACY_PREFILL_DQ", "1" if flag else "0")
    called = []
    monkeypatch.setattr(legacy_dq, "HAVE_TRITON", True)
    monkeypatch.setattr(legacy_dq, "dequant", lambda fmt, raw: (called.append(fmt), torch.from_numpy(legacy_dq.ref_dequant_np(fmt, raw.numpy())))[1])
    gcalled = []
    monkeypatch.setattr(fl, "gap_dequant", lambda fmt, raw: (gcalled.append(fmt), torch.zeros(raw.shape[0], 64, dtype=torch.half))[1])
    class FP(torch.Tensor):
        is_cuda = property(lambda s: True)
    for fmt in ("q4_0", "q5_0", "q5_1"):
        mm = object.__new__(fl.ForeignLinear); mm.format = fmt; mm.in_features = 64; mm.out_features = 8; mm.key = "t"
        mm.packed = torch.zeros(8, 2 * legacy_dq.BB.get(fmt, 24), dtype=torch.uint8).as_subclass(FP)
        called.clear(); gcalled.clear()
        mm._dequant_rows(0, 8)
        on = flag and fmt in ("q4_0", "q5_0")
        assert called == ([fmt] if on else []) and gcalled == ([] if on else [fmt]), (fmt, called, gcalled)
