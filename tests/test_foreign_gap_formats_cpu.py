import os, sys
import numpy as np
import pytest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
gguf = pytest.importorskip("gguf")
from gguf import GGMLQuantizationType as T

import importlib.util
# Load the two modules by path: importing the exllamav3 package would build/load the CUDA extension.
_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exllamav3", "foreign_quant")
_pkg = importlib.util.spec_from_file_location("fq_gap", os.path.join(_dir, "__init__.py"), submodule_search_locations=[_dir])
_mod = importlib.util.module_from_spec(_pkg)
sys.modules["fq_gap"] = _mod
_pkg.loader.exec_module(_mod)
import fq_gap.gap_formats as gf

NAMES = {"q4_0": T.Q4_0, "q5_0": T.Q5_0, "iq4_nl": T.IQ4_NL, "iq3_xxs": T.IQ3_XXS, "iq3_s": T.IQ3_S,
         "mxfp4": T.MXFP4, "nvfp4_gguf": T.NVFP4}


def _random_rows(fmt, n_rows, nb, seed):
    qk, bb = gf.BLOCKS[fmt]
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 256, size=(n_rows, nb, bb), dtype=np.uint8)
    d = rng.uniform(0.001, 0.05, size=(n_rows, nb)).astype(np.float16)    # sane fp16 scale in the first 2 bytes
    if fmt not in ("mxfp4", "nvfp4_gguf"):
        raw[:, :, 0:2] = d.view(np.uint8).reshape(n_rows, nb, 2)
    elif fmt == "mxfp4":
        raw[:, :, 0] = rng.integers(120, 135, size=(n_rows, nb), dtype=np.uint8)
    else:
        raw[:, :, 0:4] = rng.integers(40, 90, size=(n_rows, nb, 4), dtype=np.uint8)
    return raw.reshape(n_rows, nb * bb)


@pytest.mark.parametrize("fmt", list(NAMES))
def test_matches_gguf_py(fmt):
    qk, bb = gf.BLOCKS[fmt]
    nb = 3 if qk == 256 else 8
    raw = _random_rows(fmt, 5, nb, seed=7)
    ref = gguf.quants.dequantize(raw, NAMES[fmt])                       # [5, nb*qk] float32
    out = gf.dequant(fmt, torch.from_numpy(raw)).float().numpy()
    assert out.shape == ref.shape
    np.testing.assert_allclose(out, ref, rtol=2e-3, atol=2e-3 * np.abs(ref).max())
