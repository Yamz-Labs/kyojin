"""
Per-expert K inside one MoE layer (exllamav3/modules/moe_mixedk.py), CPU only.

The grouped HIP kernels cannot run here, so they are emulated at their CONTRACT (same entry points, same
pointer tables, same masking rules as exl3_gemv.cu / exl3_moe_prefill.cu):

  exl3_moe_gfx12_k3 (decode)         picks outside [0, experts) contribute exact zeros; the kernel is
                                     templated on one K, so a pointer table that holds experts of another K
                                     is a hard error here (the device would decode garbage); the routed sum
                                     runs in expert-sorted order.
  exl3_moe_prefill_wmma (prefill)    NO masked picks (the reduce indexes svh_table[e] for every pick), `order`
                                     must be the stable expert sort of the picks and `expert_count` its
                                     histogram; the sum runs in routing-slot order, shared output added last.

Experts are dense fp32 matrices (what dequantize gives); the pointer tables are the real MultiLinear tables
over fake EXL3 tensors, and the emulator resolves every table entry back to its expert, so a wrong slot,
group or table is caught. The reference is plain dequantize-then-matmul per token (no tables, no kernels).
Numerics of the real trellis decode / Hadamard / WMMA are NOT covered: that needs the device.
"""
import os, sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import torch.nn.functional as F

from exllamav3.modules.moe_mixedk import (
    MixedKStore, MixedKUnsupported, expert_bits_from_plan, parse_expert_bits, check_expert_bits, EXPERT_BITS_KEY,
    layer_is_mixed, expert_ks)
from exllamav3.modules.multilinear import MultiLinear
from exllamav3.modules.block_sparse_mlp import _hip_wmma_workspace, _scatter_expert_count

H, I, TOPK = 256, 128, 6
DEV = torch.device("cpu")


# --------------------------------------------------------------------------- fake pack

class Pack:
    """E experts of dense weights + the fake EXL3 Linear objects MultiLinear needs."""

    def __init__(self, ks, seed = 0, kd = None):
        g = torch.Generator().manual_seed(seed)
        self.E = len(ks)
        self.ks = list(ks)
        self.kd = list(kd) if kd is not None else list(ks)
        self.W = []          # per expert: (Wg [H, I], Wu [H, I], Wd [I, H]) fp32 (fp16-representable)
        self.gates, self.ups, self.downs = [], [], []
        self.reg = {}        # table pointer -> (kind, expert, role, K)
        for e in range(self.E):
            ws = tuple((torch.randn(a, b, generator = g) * 0.06).half().float() for a, b in ((H, I), (H, I), (I, H)))
            self.W.append(ws)
            for role, lst, (a, b), K, w in (("g", self.gates, (H, I), ks[e], ws[0]),
                                            ("u", self.ups, (H, I), ks[e], ws[1]),
                                            ("d", self.downs, (I, H), self.kd[e], ws[2])):
                tr = torch.zeros((a // 16, b // 16, max(1, int(round(16 * K)))), dtype = torch.int16)
                suh = torch.zeros(a, dtype = torch.half)
                svh = torch.zeros(b, dtype = torch.half)
                lin = SimpleNamespace(
                    quant_type = "exl3", softcap = None, post_scale = 1.0, in_features = a, out_features = b,
                    inner = SimpleNamespace(bias = None, K = float(K), trellis = tr, suh = suh, svh = svh,
                                            mcg = False, mul1 = True))
                lst.append(lin)
                self.reg[tr.data_ptr()] = ("trellis", e, role, float(K), w)
                self.reg[suh.data_ptr()] = ("suh", e, role, float(K), None)
                self.reg[svh.data_ptr()] = ("svh", e, role, float(K), None)

    def mlp_ref(self, x, e, limit):
        """dequantize-then-matmul for one token and one expert (fp16 intermediates as in the kernels)."""
        Wg, Wu, Wd = self.W[e]
        xf = x.float()
        gate = (xf @ Wg).half().float()
        up = (xf @ Wu).half().float()
        if limit > 0:
            act = F.silu(torch.clamp(gate, max = limit)) * torch.clamp(up, -limit, limit)
        else:
            act = F.silu(gate) * up
        return act.half().float() @ Wd

    def reference(self, y, sel, w, limit, sh_add = None):
        out = torch.zeros(y.shape[0], H, dtype = torch.float64)
        for r in range(y.shape[0]):
            for k in range(sel.shape[1]):
                out[r] += self.mlp_ref(y[r], int(sel[r, k]), limit).double() * float(w[r, k])
        if sh_add is not None:
            out += sh_add.double()
        return out


# --------------------------------------------------------------------------- kernel-contract emulator

class KernelEmu:
    def __init__(self, pack: Pack):
        self.pack = pack
        self.calls = []

    def _resolve(self, tables, expected_roles, K, what):
        """table entries -> expert ids, checking role, K and that suh/svh/trellis of a slot are one expert."""
        out = []
        tr, su, sv = tables
        for i in range(tr.numel()):
            t = self.pack.reg[int(tr[i])]
            s1 = self.pack.reg[int(su[i])]
            s2 = self.pack.reg[int(sv[i])]
            assert (t[1], t[2]) == (s1[1], s1[2]) == (s2[1], s2[2]), f"{what}: table slot {i} mixes experts"
            assert t[2] == expected_roles, f"{what}: slot {i} holds role {t[2]}"
            assert t[3] == K, f"{what}: slot {i} holds K={t[3]} but the launch is K={K}"
            out.append(t[1])
        return out

    def _row(self, x, e, limit):
        return self.pack.mlp_ref(x, e, limit)

    def exl3_moe_gfx12_k3(self, A, output, selected, weights, gt, gs, gv, ut, us, uv, dt, ds, dv,
                          gu_had, gu_out, down_had, down_out, K, dedup, act_limit):
        R, Hh = A.shape
        top_k = selected.shape[1]
        assert A.dtype == torch.half and A.is_contiguous() and 1 <= R <= 16 and Hh % 128 == 0
        assert top_k in (6, 8, 10)
        assert selected.dtype == torch.long and selected.is_contiguous() and selected.shape[0] == R
        assert weights.dtype == torch.half and weights.is_contiguous() and tuple(weights.shape) == (R, top_k)
        assert output.dtype == torch.float and output.is_contiguous() and tuple(output.shape) == (R, Hh)
        experts = gt.numel()
        for t in (gs, gv, ut, us, uv, dt, ds, dv):
            assert t.numel() == experts and t.dtype == torch.long
        assert gu_had.numel() == 2 * R * top_k * Hh and down_out.numel() == R * top_k * Hh
        g_e = self._resolve((gt, gs, gv), "g", K, "gate")
        u_e = self._resolve((ut, us, uv), "u", K, "up")
        d_e = self._resolve((dt, ds, dv), "d", K, "down")
        assert g_e == u_e == d_e
        self.calls.append(("decode", K, experts, int((selected >= 0).sum())))
        for r in range(R):
            rows = []
            for k in range(top_k):
                e = int(selected[r, k])
                rows.append(self._row(A[r], self.pack_expert(g_e, e), act_limit) if 0 <= e < experts
                            else torch.zeros(Hh))
            acc = torch.zeros(Hh)
            for k in sorted(range(top_k), key = lambda k: (int(selected[r, k]), k)):     # expert-sorted reduce
                acc = acc + rows[k] * float(weights[r, k])
            output[r] = acc

    @staticmethod
    def pack_expert(table_experts, slot):
        return table_experts[slot]

    def exl3_moe_prefill_wmma(self, A, output, selected, weights, order, expert_count, gt, gs, gv, ut, us, uv,
                              dt, ds, dv, K_gu, K_down, gu_had, gu_out, down_out, offsets, inverse, tiles,
                              tile_count, act_limit, mcg, addend):
        rows, top_k = selected.shape
        assert A.dtype == torch.half and A.dim() == 2 and A.stride(1) == 1 and A.shape[0] <= rows
        assert selected.dtype == torch.long and selected.is_contiguous()
        assert weights.dtype == torch.float and weights.is_contiguous() and weights.shape == selected.shape
        assert output.dtype == torch.float and output.is_contiguous() and tuple(output.shape) == (rows, A.shape[1])
        experts = gt.numel()
        for t in (gs, gv, ut, us, uv, dt, ds, dv):
            assert t.numel() == experts
        flat = selected.view(-1)
        assert int(flat.min()) >= 0 and int(flat.max()) < experts, \
            "prefill kernel: every pick must be a valid expert (svh_table[e] is read unconditionally)"
        assert torch.equal(order, flat.argsort(stable = True)), "order must be the stable expert sort"
        assert expert_count.numel() >= experts
        assert torch.equal(expert_count[:experts], torch.bincount(flat, minlength = experts))
        assignments = rows * top_k
        Hh, Ii = A.shape[1], gu_out.shape[1]
        assert gu_had.numel() >= 2 * assignments * Hh and gu_out.shape[0] >= 2 * assignments
        assert down_out.numel() >= assignments * Hh
        assert offsets.numel() >= experts + 1 and inverse.numel() >= assignments
        assert tiles.numel() >= assignments // 64 + experts
        g_e = self._resolve((gt, gs, gv), "g", K_gu, "gate")
        u_e = self._resolve((ut, us, uv), "u", K_gu, "up")
        d_e = self._resolve((dt, ds, dv), "d", K_down, "down")
        assert g_e == u_e == d_e
        self.calls.append(("prefill", K_gu, K_down, experts, rows))
        for r in range(rows):
            acc = torch.zeros(Hh)
            for k in range(top_k):                                                    # routing-slot order
                e = g_e[int(selected[r, k])]
                acc = acc + self._row(A[r], e, act_limit) * float(weights[r, k])
            if addend is not None and r < addend.shape[0]:
                acc = acc + addend[r].float()
            output[r] = acc


# --------------------------------------------------------------------------- helpers

LIMIT = 10.0


def hot_cold(E, n_hot, seed = 1):
    g = torch.Generator().manual_seed(seed)
    hot = set(torch.randperm(E, generator = g)[:n_hot].tolist())
    return [3.0 if e in hot else 2.0 for e in range(E)]


def routing(R, E, seed = 5):
    g = torch.Generator().manual_seed(seed)
    sel = torch.stack([torch.randperm(E, generator = g)[:TOPK] for _ in range(R)]).long()
    w = torch.rand((R, TOPK), generator = g).half()
    return sel.contiguous(), w.contiguous()


def acts(R, seed = 9):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(R, H, generator = g) * 1.5).half().contiguous()


def buffers(R_max = 16):
    n = R_max * TOPK
    return SimpleNamespace(
        gu_had = torch.empty(2 * n, H, dtype = torch.half), gu_out = torch.empty(2 * n, I, dtype = torch.half),
        down_had = torch.empty(n, I, dtype = torch.half), down_out = torch.empty(n, H, dtype = torch.float))


def outs(G):
    return [torch.empty(16, H, dtype = torch.float) for _ in range(G)]


def ws_fn(a, e):
    return _hip_wmma_workspace(DEV, a, H, I, e)


def close(a, b, tol = 2e-5):
    a, b = a.double(), b.double()
    return float((a - b).abs().max()) <= tol * max(1.0, float(b.abs().max()))


# --------------------------------------------------------------------------- storage

@pytest.mark.parametrize("E,n_hot", [(24, 8), (24, 1), (24, 23), (7, 3)])
def test_store_maps_and_tables(E, n_hot):
    ks = hot_cold(E, n_hot)
    pack = Pack(ks)
    st = MixedKStore(DEV, pack.gates, pack.ups, pack.downs)
    assert st.num_groups == len(set(ks))
    assert [g.k_gu for g in st.groups] == sorted(set(ks))
    assert layer_is_mixed(expert_ks(pack.gates, pack.ups, pack.downs)) == (len(set(ks)) > 1)
    seen = set()
    for g in st.groups:
        assert g.experts == sorted(g.experts)                                 # ascending inside a group
        for slot, e in enumerate(g.experts):
            assert ks[e] == g.k_gu
            assert int(st.kmap[e]) == g.index and int(st.slot[e]) == slot
            assert int(st.lut[g.index, e]) == slot
            assert int(g.gate.ptrs_trellis[slot]) == pack.gates[e].inner.trellis.data_ptr()
            assert int(g.up.ptrs_suh[slot]) == pack.ups[e].inner.suh.data_ptr()
            assert int(g.down.ptrs_svh[slot]) == pack.downs[e].inner.svh.data_ptr()
            seen.add(e)
    assert seen == set(range(E))
    # a pick of group g maps to -1 in every other group's lut row
    for e in range(E):
        assert [int(st.lut[g.index, e]) >= 0 for g in st.groups].count(True) == 1
    assert st.decode_ok and st.wmma_ok


def test_store_rejects_gate_up_mismatch():
    pack = Pack([2.0] * 6)
    pack.ups[3].inner.K = 3.0
    with pytest.raises(MixedKUnsupported):
        MixedKStore(DEV, pack.gates, pack.ups, pack.downs)


def test_multilinear_still_asserts_single_k():
    pack = Pack([2.0, 3.0, 2.0, 3.0])
    with pytest.raises(AssertionError):
        MultiLinear(DEV, pack.gates)


def test_down_k_differs_from_gate_up_k():
    # (K_gu, K_down) is the group key: prefill takes both, decode needs them equal
    pack = Pack([3.0] * 8 + [2.0] * 8, kd = [2.0] * 8 + [2.0] * 8)
    st = MixedKStore(DEV, pack.gates, pack.ups, pack.downs)
    assert [(g.k_gu, g.k_down) for g in st.groups] == [(2.0, 2.0), (3.0, 2.0)]
    assert st.wmma_ok and not st.decode_ok


# --------------------------------------------------------------------------- decode

@pytest.mark.parametrize("R", [1, 3, 8, 16])
@pytest.mark.parametrize("dedup", [False, True])
@pytest.mark.parametrize("n_hot", [6, 12])
def test_decode_mixed_matches_dequant_reference(R, dedup, n_hot):
    E = 24
    pack = Pack(hot_cold(E, n_hot, seed = R))
    st = MixedKStore(DEV, pack.gates, pack.ups, pack.downs)
    emu = KernelEmu(pack)
    y = acts(R)
    sel, w = routing(R, E, seed = R + n_hot)
    out = st.decode(emu, y, sel, w, buffers(), outs(st.num_groups), LIMIT, dedup)
    assert close(out, pack.reference(y, sel, w, LIMIT))
    # one launch per K group, each on its own K
    assert [c[1] for c in emu.calls] == [g.k_gu for g in st.groups]
    assert all(c[0] == "decode" for c in emu.calls)


def test_decode_token_with_no_pick_in_a_group():
    E = 24
    ks = hot_cold(E, 8)
    pack = Pack(ks)
    st = MixedKStore(DEV, pack.gates, pack.ups, pack.downs)
    emu = KernelEmu(pack)
    cold = [e for e in range(E) if ks[e] == 2.0][:TOPK]
    hot = [e for e in range(E) if ks[e] == 3.0][:TOPK]
    sel = torch.tensor([cold, hot], dtype = torch.long)
    w = torch.rand(2, TOPK).half()
    y = acts(2)
    out = st.decode(emu, y, sel, w, buffers(), outs(2), LIMIT, True)
    assert close(out, pack.reference(y, sel, w, LIMIT))


def test_decode_group_sum_is_fp32_partials_in_ascending_k():
    E = 24
    pack = Pack(hot_cold(E, 10))
    st = MixedKStore(DEV, pack.gates, pack.ups, pack.downs)
    emu = KernelEmu(pack)
    y, (sel, w) = acts(2), routing(2, E, seed = 77)
    o = outs(2)
    res = st.decode(emu, y, sel, w, buffers(), o, LIMIT, False)
    assert res.data_ptr() == o[0].data_ptr()
    # group 0 partial + group 1 partial, nothing else
    p0 = torch.zeros(2, H)
    p1 = torch.zeros(2, H)
    for r in range(2):
        for k in sorted(range(TOPK), key = lambda k: (int(sel[r, k]), k)):
            e = int(sel[r, k])
            row = pack.mlp_ref(y[r], e, LIMIT) * float(w[r, k])
            if st.kmap_host[e] == 0:
                p0[r] = p0[r] + row
            else:
                p1[r] = p1[r] + row
    assert torch.equal(res, p0 + p1)


@pytest.mark.parametrize("K", [2.0, 3.0])
@pytest.mark.parametrize("R", [1, 4, 16])
@pytest.mark.parametrize("dedup", [False, True])
def test_decode_single_k_is_bit_identical_to_legacy_call(K, R, dedup):
    E = 20
    pack = Pack([K] * E)
    emu = KernelEmu(pack)
    y = acts(R)
    sel, w = routing(R, E, seed = 3 * R)
    bufs = buffers()
    # legacy single-K call, exactly as block_sparse_mlp's grouped branch makes it
    mg, mu, md = (MultiLinear(DEV, l) for l in (pack.gates, pack.ups, pack.downs))
    legacy = torch.empty(16, H, dtype = torch.float)[:R]
    emu.exl3_moe_gfx12_k3(y, legacy, sel, w, mg.ptrs_trellis, mg.ptrs_suh, mg.ptrs_svh, mu.ptrs_trellis,
                          mu.ptrs_suh, mu.ptrs_svh, md.ptrs_trellis, md.ptrs_suh, md.ptrs_svh,
                          bufs.gu_had[:2 * R * TOPK], bufs.gu_out[:2 * R * TOPK], bufs.down_had[:R * TOPK],
                          bufs.down_out[:R * TOPK], float(mu.K), dedup, LIMIT)
    st = MixedKStore(DEV, pack.gates, pack.ups, pack.downs)
    assert st.num_groups == 1 and st.groups[0].identity
    new = st.decode(emu, y, sel, w, bufs, outs(1), LIMIT, dedup)
    assert torch.equal(new, legacy)
    assert len(emu.calls) == 2                      # one launch set, not more


def test_emulator_has_teeth_on_wrong_k_table():
    pack = Pack([2.0, 3.0] * 8)
    emu = KernelEmu(pack)
    mixed = SimpleNamespace(gu_had = None)
    sel, w = routing(1, 16)
    bufs = buffers()
    # a single-K MultiLinear built over a table that contains both Ks is impossible; build the tables by hand
    mg = MultiLinear(DEV, [pack.gates[e] for e in range(0, 16, 2)])        # K = 2 only
    md = MultiLinear(DEV, [pack.downs[e] for e in range(0, 16, 2)])
    mu = MultiLinear(DEV, [pack.ups[e] for e in range(0, 16, 2)])
    with pytest.raises(AssertionError, match="K="):
        emu.exl3_moe_gfx12_k3(acts(1), torch.empty(1, H), sel, w, mg.ptrs_trellis, mg.ptrs_suh, mg.ptrs_svh,
                              mu.ptrs_trellis, mu.ptrs_suh, mu.ptrs_svh, md.ptrs_trellis, md.ptrs_suh, md.ptrs_svh,
                              bufs.gu_had[:12], bufs.gu_out[:12], bufs.down_had[:6], bufs.down_out[:6], 3.0, False, LIMIT)


# --------------------------------------------------------------------------- prefill

@pytest.mark.parametrize("T,chunk", [(32, 0), (70, 0), (70, 24), (200, 64)])
@pytest.mark.parametrize("n_hot", [6, 12])
def test_prefill_mixed_matches_dequant_reference(T, chunk, n_hot):
    E = 24
    pack = Pack(hot_cold(E, n_hot, seed = T))
    st = MixedKStore(DEV, pack.gates, pack.ups, pack.downs)
    emu = KernelEmu(pack)
    y = acts(T)
    sel, w = routing(T, E, seed = T + chunk)
    out = st.prefill(emu, y, sel, w.float().contiguous(), ws_fn, _scatter_expert_count, LIMIT, H, I, chunk = chunk)
    assert close(out, pack.reference(y, sel, w.float(), LIMIT))
    assert all(c[0] == "prefill" for c in emu.calls)
    ngroups_calls = len(emu.calls) / max(1, -(-T // chunk)) if chunk else len(emu.calls)
    assert ngroups_calls <= st.num_groups


def test_prefill_with_shared_output_folded_in():
    E = 24
    pack = Pack(hot_cold(E, 9))
    st = MixedKStore(DEV, pack.gates, pack.ups, pack.downs)
    emu = KernelEmu(pack)
    T = 40
    y, (sel, w) = acts(T), routing(T, E, seed = 4)
    sh = (torch.randn(T, H) * 0.3).half().contiguous()
    out = st.prefill(emu, y, sel, w.float().contiguous(), ws_fn, _scatter_expert_count, LIMIT, H, I, sh_add = sh)
    assert close(out, pack.reference(y, sel, w.float(), LIMIT, sh))


def test_prefill_group_without_picks_is_skipped():
    E = 24
    ks = hot_cold(E, 8)
    pack = Pack(ks)
    st = MixedKStore(DEV, pack.gates, pack.ups, pack.downs)
    emu = KernelEmu(pack)
    cold = [e for e in range(E) if ks[e] == 2.0]
    g = torch.Generator().manual_seed(2)
    sel = torch.stack([torch.tensor(cold)[torch.randperm(len(cold), generator = g)[:TOPK]] for _ in range(33)]).contiguous()
    w = torch.rand(33, TOPK)
    y = acts(33)
    out = st.prefill(emu, y, sel, w, ws_fn, _scatter_expert_count, LIMIT, H, I)
    assert close(out, pack.reference(y, sel, w, LIMIT))
    assert [c[1] for c in emu.calls] == [2.0]           # the hot group got no pick and no launch


def test_prefill_different_down_k():
    pack = Pack([3.0] * 10 + [2.0] * 10, kd = [2.0] * 10 + [2.0] * 10)
    st = MixedKStore(DEV, pack.gates, pack.ups, pack.downs)
    emu = KernelEmu(pack)
    T = 36
    y, (sel, w) = acts(T), routing(T, 20, seed = 8)
    out = st.prefill(emu, y, sel, w.float().contiguous(), ws_fn, _scatter_expert_count, LIMIT, H, I)
    assert close(out, pack.reference(y, sel, w.float(), LIMIT))
    assert sorted((c[1], c[2]) for c in emu.calls) == [(2.0, 2.0), (3.0, 2.0)]


@pytest.mark.parametrize("K", [2.0, 3.0])
@pytest.mark.parametrize("T", [32, 100])
def test_prefill_single_k_is_bit_identical_to_legacy_call(K, T):
    E = 20
    pack = Pack([K] * E)
    emu = KernelEmu(pack)
    y, (sel, w) = acts(T), routing(T, E, seed = T)
    rw = w.float().contiguous()
    # legacy call, as in block_sparse_mlp's grouped WMMA branch
    mg, mu, md = (MultiLinear(DEV, l) for l in (pack.gates, pack.ups, pack.downs))
    flat = sel.view(-1)
    order = flat.argsort(stable = True)
    count = _scatter_expert_count(flat, E + 1)
    assignments = T * TOPK
    ws = _hip_wmma_workspace(DEV, assignments, H, I, E)
    gu_had = ws["gu_had"][:2 * assignments * H].view(2 * assignments, H)
    down_out = ws["gu_had"][:2 * assignments * H].view(torch.float).view(assignments, H)
    legacy = torch.empty((T, H), dtype = torch.float)
    emu.exl3_moe_prefill_wmma(
        y, legacy, sel, rw, order, count,
        mg.ptrs_trellis, mg.ptrs_suh, mg.ptrs_svh, mu.ptrs_trellis, mu.ptrs_suh, mu.ptrs_svh,
        md.ptrs_trellis, md.ptrs_suh, md.ptrs_svh, float(mg.K), float(md.K), gu_had,
        ws["gu_out"][:2 * assignments * I].view(2 * assignments, I), down_out,
        ws["offsets"], ws["inverse"][:assignments], ws["tiles"], ws["tile_count"], LIMIT, False, None)
    st = MixedKStore(DEV, pack.gates, pack.ups, pack.downs)
    new = st.prefill(emu, y, sel, rw, ws_fn, _scatter_expert_count, LIMIT, H, I)
    assert torch.equal(new, legacy)


def test_prefill_emulator_has_teeth_on_masked_pick():
    pack = Pack([2.0] * 12)
    emu = KernelEmu(pack)
    g = MultiLinear(DEV, pack.gates[:4]), MultiLinear(DEV, pack.ups[:4]), MultiLinear(DEV, pack.downs[:4])
    sel = torch.tensor([[0, 1, 2, 3, 4, -1]] * 32)
    flat = sel.view(-1)
    ws = ws_fn(32 * 6, 4)
    with pytest.raises(AssertionError, match="valid expert"):
        emu.exl3_moe_prefill_wmma(
            acts(32), torch.empty(32, H), sel, torch.ones(32, 6), flat.argsort(stable = True),
            torch.zeros(5, dtype = torch.long),
            g[0].ptrs_trellis, g[0].ptrs_suh, g[0].ptrs_svh, g[1].ptrs_trellis, g[1].ptrs_suh, g[1].ptrs_svh,
            g[2].ptrs_trellis, g[2].ptrs_suh, g[2].ptrs_svh, 2.0, 2.0,
            ws["gu_had"][:2 * 192 * H].view(384, H), ws["gu_out"][:2 * 192 * I].view(384, I),
            ws["gu_had"][:2 * 192 * H].view(torch.float).view(192, H),
            ws["offsets"], ws["inverse"][:192], ws["tiles"], ws["tile_count"], LIMIT, False, None)


# --------------------------------------------------------------------------- pack config

def _plan(ks, layer = 3, extra = ()):
    ents = []
    for e, k in enumerate(ks):
        for w in ("w1", "w2", "w3"):
            ents.append({"action": "quant", "kind": "routed", "key": f"layers.{layer}.ffn.experts.{e}.{w}", "bits": k})
    ents += list(extra)
    return ents


def test_expert_bits_from_plan_roundtrip():
    ks = [3, 2, 2, 3, 2, 2, 2, 3]
    other = [{"action": "quant", "kind": "attn", "key": "layers.3.attn.wq_a", "bits": 5},
             {"action": "quant", "kind": "mtp", "key": "mtp.0.ffn.experts.0.w1", "bits": 4},
             {"action": "copy", "kind": "copy", "name": "x"}]
    cfg = {EXPERT_BITS_KEY: expert_bits_from_plan(_plan(ks) + other + _plan(ks[::-1], layer = 4))}
    assert cfg[EXPERT_BITS_KEY] == {"layers.3.ffn": ks, "layers.4.ffn": ks[::-1]}
    assert parse_expert_bits(cfg, "layers.3.ffn", 8) == ks
    assert parse_expert_bits(cfg, "layers.9.ffn", 8) is None
    assert parse_expert_bits(None, "layers.3.ffn", 8) is None
    with pytest.raises(ValueError):
        parse_expert_bits(cfg, "layers.3.ffn", 9)


def test_expert_bits_from_plan_rejects_split_projections():
    ents = _plan([2, 2, 2])
    ents[4]["bits"] = 3          # expert 1, w2
    with pytest.raises(ValueError, match="same bits"):
        expert_bits_from_plan(ents)


def test_check_expert_bits_against_loaded_tensors():
    ks = hot_cold(12, 4)
    pack = Pack(ks)
    st = MixedKStore(DEV, pack.gates, pack.ups, pack.downs)
    check_expert_bits([int(k) for k in ks], st)
    bad = [int(k) for k in ks]
    bad[5] = 4
    with pytest.raises(ValueError, match="disagrees"):
        check_expert_bits(bad, st)
