"""
K=1 (1-bit EXL3 trellis) and K=0 (pruned expert) for the DS4.1 expert kernels, CPU only.

1. Trellis windows: the lane code of the VALU decode matvec (exl3_moe_valu.cuh mv_tile + dq8_regs_1bit) and the generic
   dq8_aligned_1bit used by the WMMA decode body and the prefill GEMMs are re-implemented word for word and compared with
   the torch reference decoder (ref/frac_reconstruct.py: tail-biting ring, MSB-first words) on random tiles. K=2 and K=3 run
   through the same harness so the emulator is anchored on formats that are already in production.
2. Dequantized equality: windows -> mul1 codebook values equal ref decode_tile_row for the whole tile.
3. Mixed K=0/1/2/3 inside one layer through MixedKStore (decode and prefill) against dequantize-then-matmul.
4. Source guards: every dispatch carries K=1 and unsupported bitrates still fail loudly.
"""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import types
from unittest import mock

import pytest
import torch

# CPU only: never JIT-build the native extension (it needs the build lock and a GPU toolchain)
if os.environ.get("EXL3_REAL_EXT") != "1" and "exllamav3.ext" not in sys.modules:
    _stub = types.ModuleType("exllamav3.ext")
    _stub.exllamav3_ext = mock.MagicMock()
    _stub.__getattr__ = lambda name: mock.MagicMock()
    sys.modules["exllamav3.ext"] = _stub

from ref import frac_reconstruct as FR
from exllamav3.modules.moe_mixedk import (MixedKStore, MixedKUnsupported, PrunedLinear, expert_ks, layer_is_mixed,
                                          pruned_experts, check_expert_bits)
from exllamav3.modules.block_sparse_mlp import _scatter_expert_count
from test_moe_mixedk_cpu import (Pack, KernelEmu, DEV, H, I, TOPK, LIMIT, routing, acts, buffers, outs, ws_fn, close)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QD = os.path.join(ROOT, "exllamav3", "exllamav3_ext", "quant")
M32 = 0xFFFFFFFF


def rand_tiles(K, n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 1 << 16, (n, 16 * K), generator = g, dtype = torch.int64).to(torch.int32)


def words_of(tile_u16):
    return [int(tile_u16[2 * i]) + (int(tile_u16[2 * i + 1]) << 16) for i in range(len(tile_u16) // 2)]


def fshift(b, a, s):                      # engine fshift: (a:b) >> s, low 32 bits
    return ((((a << 32) | b) >> s) & M32)


def valu_windows(w, K):
    """windows the VALU lanes (g8, t) decode: dict ring position -> 16-bit window (mv_tile BITS == K)."""
    tw = 8 * K
    out = {}
    for g8 in range(8):
        own = [w[(K * g8 + i) % tw] for i in range(K)]
        wrap = w[(K * g8 - 1) % tw]
        wa = [wrap] + own
        for t in range(4):
            if K == 1:
                a, b = wa[0], wa[1]
                bb = fshift(b, a, (~(t << 3)) & 24)
                ws = [(bb >> (7 - j)) & 0xffff for j in range(8)]
            elif K == 2:
                a, b = wa[t >> 1], wa[1 + (t >> 1)]
                bb = fshift(b, a, ((~(t << 3)) & 8) << 1)
                ws = [(bb >> (2 * (7 - j))) & 0xffff for j in range(8)]
            else:
                ia = 0 if t == 0 else (1 if t == 1 else 2)
                s2 = 8 if t == 0 else (16 if t == 1 else (24 if t == 2 else 0))
                a, b = wa[ia], wa[ia + 1]
                w7 = fshift(b, a, s2)
                w3 = fshift(b, a, s2 + 12)
                ws = [w3 >> 9, w3 >> 6, w3 >> 3, w3, w7 >> 9, w7 >> 6, w7 >> 3, w7]
                ws = [x & 0xffff for x in ws]
            for j in range(8):
                out[8 * (4 * g8 + t) + j] = ws[j]
    return out


def generic_windows(w, K):
    """dq8_aligned_{1bit,2bits} (exl3_dq.cuh) over the whole tile: lane = group of 8 weights."""
    tw = 8 * K
    out = {}
    for lane in range(32):
        t_off = lane << 3
        if K == 1:
            i1 = t_off >> 5; i0 = (i1 + 7) & 7
            bb = fshift(w[i1], w[i0], (~t_off) & 24)
            ws = [(bb >> (7 - j)) & 0xffff for j in range(8)]
        else:
            i1 = t_off >> 4; i0 = (i1 + 15) & 15
            bb = fshift(w[i1], w[i0], ((~t_off) & 8) << 1)
            ws = [(bb >> (2 * (7 - j))) & 0xffff for j in range(8)]
        for j in range(8):
            out[t_off + j] = ws[j]
    return out


@pytest.mark.parametrize("K", [1, 2, 3])
def test_valu_lane_windows_match_reference(K):
    tiles = rand_tiles(K, 64, seed = 100 + K)
    ref = FR.windows_of_row(tiles, float(K))
    for n in range(tiles.shape[0]):
        got = valu_windows(words_of(tiles[n]), K)
        assert [got[p] for p in range(256)] == ref[n].tolist(), f"K={K} tile {n}"


@pytest.mark.parametrize("K", [1, 2])
def test_generic_dq8_windows_match_reference(K):
    tiles = rand_tiles(K, 64, seed = 200 + K)
    ref = FR.windows_of_row(tiles, float(K))
    for n in range(tiles.shape[0]):
        got = generic_windows(words_of(tiles[n]), K)
        assert [got[p] for p in range(256)] == ref[n].tolist(), f"K={K} tile {n}"


def test_valu_k1_addressing_stays_in_tile():
    # K=1: tile = 8 words, lane g8 owns word g8, wrap word = (g8 - 1) mod 8 (mv::own / mv::wrap with bits = 1)
    for g8 in range(8):
        own, wrap = 1 * g8, (((1 * g8 - 1) % 8) + 8) % 8
        assert 0 <= own < 8 and 0 <= wrap < 8 and wrap == (g8 + 7) % 8


def test_k1_dequantized_tile_equals_reference():
    tiles = rand_tiles(1, 8, seed = 7)
    ref = FR.decode_tile_row(tiles, 1.0, FR.CB_MUL1)                    # (16, 128) fp16
    vals = []
    for n in range(tiles.shape[0]):
        win = valu_windows(words_of(tiles[n]), 1)
        v = FR.decode_windows(torch.tensor([win[p] for p in range(256)]))
        vals.append(v[FR.PERM])
    got = torch.stack(vals).view(8, 16, 16).permute(1, 0, 2).reshape(16, 128)
    assert torch.equal(got.view(torch.int16), ref.view(torch.int16))


def test_k1_gemv_matches_dequant_reference():
    # y = x @ W for a K=1 matrix, W from the kernel-lane windows, against W from the reference decoder
    kt, nt = 4, 6
    g = torch.Generator().manual_seed(3)
    tr = torch.randint(0, 1 << 16, (kt, nt, 16), generator = g, dtype = torch.int64).to(torch.int32)
    Wref = FR.decode_block(tr.to(torch.int16), 1.0, FR.CB_MUL1)
    W = torch.empty_like(Wref)
    for k in range(kt):
        vs = []
        for n in range(nt):
            win = valu_windows(words_of(tr[k, n]), 1)
            vs.append(FR.decode_windows(torch.tensor([win[p] for p in range(256)]))[FR.PERM])
        W[16 * k:16 * k + 16] = torch.stack(vs).view(nt, 16, 16).permute(1, 0, 2).reshape(16, nt * 16)
    x = torch.randn(3, 16 * kt, generator = g).half().float()
    assert torch.equal(x @ W.float(), x @ Wref.float())


# --------------------------------------------------------------------------- mixed K0/K1/K2/K3 in one layer

class PackK0(Pack):
    def __init__(self, ks, seed = 0):
        super().__init__([k if k else 2.0 for k in ks], seed = seed)
        self.zero = {e for e, k in enumerate(ks) if not k}
        for e in self.zero:
            self.gates[e], self.ups[e], self.downs[e] = PrunedLinear(f"e{e}.w1"), PrunedLinear(f"e{e}.w3"), PrunedLinear(f"e{e}.w2")

    def mlp_ref(self, x, e, limit):
        return torch.zeros(H) if e in self.zero else super().mlp_ref(x, e, limit)


def ks_mix(E, seed = 1):
    g = torch.Generator().manual_seed(seed)
    return [float(v) for v in torch.tensor([0, 1, 1, 1, 2, 3])[torch.randint(0, 6, (E,), generator = g)]]


def test_expert_ks_and_layer_flags():
    pack = PackK0([0.0, 1.0, 2.0, 3.0])
    assert expert_ks(pack.gates, pack.ups, pack.downs)[0] == (0.0, 0.0, 0.0)
    assert layer_is_mixed(expert_ks(pack.gates, pack.ups, pack.downs))
    only = PackK0([0.0, 2.0, 2.0])
    assert layer_is_mixed(expert_ks(only.gates, only.ups, only.downs))      # K0 + one K still takes the group path


def test_store_skips_pruned_experts():
    pack = PackK0([0.0, 1.0, 1.0, 2.0, 0.0, 3.0])
    st = MixedKStore(DEV, pack.gates, pack.ups, pack.downs)
    assert st.pruned == [0, 4] and st.num_groups == 3
    assert st.kmap_host.tolist() == [-1, 0, 0, 1, -1, 2] and st.slot_host.tolist() == [-1, 0, 1, 0, -1, 0]
    assert all(int(st.lut[g.index][[0, 4]].max()) == -1 for g in st.groups)
    assert sum(g.n for g in st.groups) == 4


def test_all_pruned_layer_is_rejected():
    pack = PackK0([0.0, 0.0, 0.0])
    with pytest.raises(MixedKUnsupported):
        MixedKStore(DEV, pack.gates, pack.ups, pack.downs)


@pytest.mark.parametrize("R", [1, 4, 16])
@pytest.mark.parametrize("dedup", [False, True])
def test_decode_mixed_k0123(R, dedup):
    E = 24
    pack = PackK0(ks_mix(E, seed = R))
    st = MixedKStore(DEV, pack.gates, pack.ups, pack.downs)
    emu = KernelEmu(pack)
    y = acts(R)
    sel, w = routing(R, E, seed = R + 40)
    out = st.decode(emu, y, sel, w, buffers(), outs(st.num_groups), LIMIT, dedup)
    assert close(out, pack.reference(y, sel, w, LIMIT))
    assert sorted(c[1] for c in emu.calls) == sorted(g.k_gu for g in st.groups)
    assert 1.0 in [c[1] for c in emu.calls]


def test_decode_pick_on_pruned_expert_contributes_zero():
    E = 12
    pack = PackK0([0.0] * 3 + [1.0] * 9)
    st = MixedKStore(DEV, pack.gates, pack.ups, pack.downs)
    y = acts(2)
    sel = torch.tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 6, 7, 8]]).long().contiguous()
    w = torch.rand(2, TOPK).half().contiguous()
    out = st.decode(KernelEmu(pack), y, sel, w, buffers(), outs(st.num_groups), LIMIT, False)
    assert close(out, pack.reference(y, sel, w, LIMIT))


@pytest.mark.parametrize("T,chunk", [(32, 0), (70, 24), (200, 64)])
def test_prefill_mixed_k0123(T, chunk):
    E = 24
    pack = PackK0(ks_mix(E, seed = T))
    st = MixedKStore(DEV, pack.gates, pack.ups, pack.downs)
    emu = KernelEmu(pack)
    y = acts(T)
    sel, w = routing(T, E, seed = T + 3)
    out = st.prefill(emu, y, sel, w.float().contiguous(), ws_fn, _scatter_expert_count, LIMIT, H, I, chunk = chunk)
    assert close(out, pack.reference(y, sel, w.float(), LIMIT))
    assert all(c[0] == "prefill" for c in emu.calls)


def test_prefill_all_picks_pruned():
    E = 8
    pack = PackK0([0.0] * 6 + [1.0, 2.0])
    st = MixedKStore(DEV, pack.gates, pack.ups, pack.downs)
    y = acts(5)
    sel = torch.arange(6).repeat(5, 1).long().contiguous()
    w = torch.rand(5, TOPK).float().contiguous()
    emu = KernelEmu(pack)
    out = st.prefill(emu, y, sel, w, ws_fn, _scatter_expert_count, LIMIT, H, I)
    assert float(out.abs().max()) == 0.0 and emu.calls == []


def test_pruned_experts_from_pack_config():
    q = {"expert_bits": {"layers.3.ffn": [0, 1, 2, 0, 3]}}
    assert pruned_experts(q, "model.layers.3.ffn", 5) == {0, 3}
    assert pruned_experts(q, "layers.4.ffn", 5) == set()
    assert pruned_experts(None, "layers.3.ffn", 5) == set()
    pack = PackK0([0.0, 1.0, 2.0, 0.0, 3.0])
    check_expert_bits([0, 1, 2, 0, 3], expert_ks(pack.gates, pack.ups, pack.downs))
    with pytest.raises(ValueError):
        check_expert_bits([0, 1, 2, 1, 3], expert_ks(pack.gates, pack.ups, pack.downs))


# --------------------------------------------------------------------------- source guards

def _read(n):
    return open(os.path.join(QD, n)).read()


def test_dispatch_covers_k1():
    t = _read("exl3_gemv.cu")
    m = re.search(r"static void launch_moe_grouped_k.*?\n\}\n", t, re.S).group(0)
    assert "case 2: launch_moe_grouped<1, false" in m and "default: TORCH_CHECK(false" in m
    assert "k2 == 2 || k2 == 4" in t and "(!HALF && (K == 1 || K == 2" in t
    p = _read("exl3_moe_prefill.cu")
    assert p.count("case 2: MPW2_K(1, false)") == 1 and p.count("case 2: MPW_K(1, false)") == 1
    v = _read("exl3_moe_valu.cuh")
    assert "BITS == 1" in v and "dq8_regs_1bit<2>" in v
    assert "dq8_regs_1bit" in _read("exl3_gemv_kernel.cuh") and "bits == 1 || bits == 5" in _read("exl3_gemv_kernel.cuh")


def test_python_gates_carry_k1():
    b = open(os.path.join(ROOT, "exllamav3", "modules", "block_sparse_mlp.py")).read()
    assert b.count("multi.K in (1, 2, 2.5, 3, 4, 5, 6)") == 2
    assert "DECODE_K = (1, 2" in open(os.path.join(ROOT, "exllamav3", "modules", "moe_mixedk.py")).read()


# --------------------------------------------------------------------------- pack config round trip with pruned experts

def test_expert_bits_from_plan_marks_pruned_with_zero():
    from exllamav3.modules.moe_mixedk import expert_bits_from_plan, parse_expert_bits
    entries = []
    for e, (act, b) in enumerate([("quant", 1), ("pruned", 0), ("quant", 2), ("quant", 3)]):
        for w in ("w1", "w2", "w3"):
            entries.append({"action": act, "kind": "routed", "key": f"layers.5.ffn.experts.{e}.{w}", "bits": b})
    row = expert_bits_from_plan(entries)["layers.5.ffn"]
    assert row == [1, 0, 2, 3]
    q = {"expert_bits": {"layers.5.ffn": row}}
    assert parse_expert_bits(q, "model.layers.5.ffn", 4) == row          # prefixed module key, same row as pruned_experts
    assert pruned_experts(q, "model.layers.5.ffn", 4) == {1}
    assert parse_expert_bits(q, "model.layers.6.ffn", 4) is None
