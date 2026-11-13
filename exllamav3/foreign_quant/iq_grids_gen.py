# SPDX-License-Identifier: MIT
"""Decode grids of IQ1_S/IQ1_M/IQ2_XXS/IQ2_XS/IQ2_S (and a cross-check of IQ3_XXS/IQ3_S) built the way ggml's iq2xs_init_impl / iq3xs_init_impl
build `the_grid`: entry k is the seed word kgrid[k] read as 8 (2-bit) or 4 (3-bit) little-endian digits l, each turned into a byte value.
init_impl itself writes 2*l+1; the decode tables of ggml-common.h hold a per-type value of that digit, fixed below (checked against ggml's
own tables in tests/test_gguf_types.py and by the ggml build in docs/reports/cgguftypes.md). The seeds come from foreign_seeds (extracted by script)."""
import functools

import numpy as np

from . import iq_seeds as S

IQ2_LEVELS = (8, 25, 43)                      # digit 0,1,2 -> byte of iq2xxs_grid / iq2xs_grid / iq2s_grid (init_impl: 1,3,5)
IQ3XXS_LEVELS = (4, 12, 20, 28, 36, 44, 52, 62)    # digit 0..7 -> byte of iq3xxs_grid (init_impl: 1,3,..,15)


def _digits(seed, bits, n):
    k = np.asarray(seed, np.uint32)[:, None]
    return ((k >> (bits * np.arange(n, dtype=np.uint32))) & ((1 << bits) - 1)).astype(np.int64)


@functools.lru_cache(None)
def iq2_grid(kind):
    """kind in xxs/xs/s -> uint8 [N, 8] (float-ready magnitudes)."""
    seed = {"xxs": S.KGRID_2BIT_256, "xs": S.KGRID_2BIT_512, "s": S.KGRID_2BIT_1024}[kind]
    return np.array(IQ2_LEVELS + (0,), np.uint8)[_digits(seed, 2, 8)]


@functools.lru_cache(None)
def iq1_grid():
    """int8 [2048, 8] in {-1, 0, 1}: (2*l+1 - 3) / 2 = l - 1, as quantize_iq1_s reads it back."""
    return (_digits(S.KGRID_1BIT_2048, 2, 8) - 1).astype(np.int8)


@functools.lru_cache(None)
def iq3_grid(kind):
    """kind xxs/s -> uint8 [N, 4]."""
    if kind == "xxs":
        return np.array(IQ3XXS_LEVELS, np.uint8)[_digits(S.KGRID_256, 3, 4)]
    return (2 * _digits(S.KGRID_512, 3, 4) + 1).astype(np.uint8)
