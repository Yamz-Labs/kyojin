"""CPU test: moe_i8.experts_rows (rows grouped per expert) equals the per-row maths of the draft head expert kernels. Needs torch + triton importable."""
import os
import sys
import types
import unittest

import torch

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for n, s in (("exllamav3", ""), ("exllamav3.loader", "loader")):
    m = types.ModuleType(n)
    m.__path__ = [os.path.join(TREE, "exllamav3", s)]
    sys.modules[n] = m
from exllamav3.loader import moe_i8 as MI  # noqa: E402


class Stack:
    def __init__(self, E, N, K):
        self.E, self.N, self.K = E, N, K
        self.q = torch.randint(-127, 128, (E, N, K), dtype=torch.int8)
        self.s = torch.rand(E, N) * 0.02 / 127


class Rows(unittest.TestCase):
    def test_matches_per_row_reference(self):
        torch.manual_seed(0)
        E, N, K, T, S, lat = 8, 64, 64, 40, 3, 32
        g, u, d = Stack(E, N, K), Stack(E, N, K), Stack(E, lat, N)
        z = torch.randn(T, K)
        sel = torch.stack([torch.randperm(E)[:S] for _ in range(T)])
        w = torch.rand(T, S)
        got = MI.experts_rows(g, u, d, z, sel, w, 10.0)
        ref = torch.zeros(T, lat)
        for t in range(T):
            for j in range(S):
                e = int(sel[t, j])
                gg = ((g.q[e].float() @ z[t]) * g.s[e]).clamp(max=10.0)
                uu = ((u.q[e].float() @ z[t]) * u.s[e]).clamp(-10.0, 10.0)
                h = torch.nn.functional.silu(gg) * uu
                ref[t] += w[t, j] * ((d.q[e].float() @ h) * d.s[e])
        self.assertLess(float((got - ref).abs().max() / ref.abs().max()), 5e-3)


if __name__ == "__main__":
    unittest.main()
