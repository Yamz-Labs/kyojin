import numpy as np

from exllamav3.foreign_quant.k_quants import dequant_q6_K_block, dequant_q6_K_bulk


def test_q6_K_bulk_matches_block():
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 256, size=(64, 210), dtype=np.uint8)
    raw[:, 208:210] = np.frombuffer(rng.uniform(-0.05, 0.05, 64).astype(np.float16).tobytes(), np.uint8).reshape(64, 2)
    ref = np.stack([dequant_q6_K_block(bytes(r)) for r in raw]).astype(np.float16)
    out = dequant_q6_K_bulk(raw)
    assert out.shape == (64, 256)
    np.testing.assert_array_equal(out, ref)
