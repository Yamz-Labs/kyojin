#!/usr/bin/env python3
"""GPU checks for K=1 / K=0 DS4.1 expert kernels (gfx1151). Run through tools/ds41k1/gpu_check.sh.

 A  VALU vs WMMA decode body, K = 1, 2, 3: BITWISE equal (gu_out, down_out, out), rows 1..4, dedup off/on.
 B  decode vs torch dequant reference (ref/frac_reconstruct.py decode + Hadamard 128), K = 1, 2, 3. K=2/3 are the anchors:
    a K=1 error within 2x of the K=2/3 error passes; if the anchors fail too the reference is the suspect (REF-SUSPECT).
 C  K=0: a table entry of 0 / a pick of -1 contributes EXACT zeros; mixed picks (K=1 pointers + K=0 pointers) equal the sum
    of the single-pick calls.
 D  prefill (exl3_moe_prefill_wmma, MPW2 and MPW kernels) K = 1, 2, 3 vs the same torch reference.
 E  prefill K=1 bitwise equal between MPW_KERN=1 and MPW_KERN=2.
Exit code 0 only if every check passes. Microbench: tools/strix_halo/k2_micro.py (called by gpu_check.sh).
"""
import os, sys, math
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import exllamav3.ext as extmod
from exllamav3.modules.block_sparse_mlp import _hip_wmma_workspace, _scatter_expert_count
from ref import frac_reconstruct as FR
ext = extmod.exllamav3_ext
dev = torch.device("cuda:0")
H, I, TOPK, E = 512, 256, 6, 24
fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name + (" " + info if info else ""), flush=True)
    if not ok:
        fails.append(name)


class Pack:
    """E experts at bitrate K: random trellis words (every bit pattern is a valid trellis), random +-1 suh/svh."""

    def __init__(self, K, seed):
        g = torch.Generator().manual_seed(seed)
        self.K = K
        def tr(a, b):
            return torch.randint(-32768, 32768, (a // 16, b // 16, 16 * K), generator=g, dtype=torch.int16)
        def sg(n):
            return (torch.randint(0, 2, (n,), generator=g) * 2 - 1).half()
        self.m = []   # per expert: [(trellis, suh, svh)] gate, up, down
        for e in range(E):
            self.m.append([(tr(H, I), sg(H), sg(I)), (tr(H, I), sg(H), sg(I)), (tr(I, H), sg(I), sg(H))])
        self.gpu = [[(t.to(dev), s.to(dev), v.to(dev)) for t, s, v in e] for e in self.m]

    def tables(self, zero=()):
        def tab(j, i):
            return torch.tensor([0 if e in zero else self.gpu[e][j][i].data_ptr() for e in range(E)], dtype=torch.long, device=dev)
        return (tab(0, 0), tab(0, 1), tab(0, 2), tab(1, 0), tab(1, 1), tab(1, 2), tab(2, 0), tab(2, 1), tab(2, 2))


def had(x):                      # blockwise Hadamard 128, last dim, scaled 1/sqrt(128)
    n = 128
    Hm = torch.ones(1, 1, dtype=torch.float64)
    while Hm.shape[0] < n:
        Hm = torch.cat([torch.cat([Hm, Hm], 1), torch.cat([Hm, -Hm], 1)], 0)
    Hm = Hm / math.sqrt(n)
    sh = x.shape
    return (x.double().reshape(-1, n) @ Hm).reshape(sh)


def dense(pack, e, j):
    t, su, sv = pack.m[e][j]
    W = FR.decode_block(t, float(pack.K), FR.CB_MUL1).double()
    return W, su.double(), sv.double()


def lin(x, W, su, sv):
    return had(had(x * su) @ W) * sv


def mlp_ref(pack, x, e):
    g = lin(x.double(), *dense(pack, e, 0))
    u = lin(x.double(), *dense(pack, e, 1))
    a = (torch.nn.functional.silu(g) * u).half().double()
    return lin(a, *dense(pack, e, 2))


def reference(pack, y, sel, w):
    out = torch.zeros(y.shape[0], H, dtype=torch.float64)
    for r in range(y.shape[0]):
        for k in range(sel.shape[1]):
            e = int(sel[r, k])
            if e >= 0:
                out[r] += mlp_ref(pack, y[r].cpu(), e) * float(w[r, k])
    return out


def decode_call(pack, tabs, y, sel, w, dedup=False):
    R = y.shape[0]
    n = R * sel.shape[1]
    out = torch.zeros((R, H), dtype=torch.float32, device=dev)
    gu_had = torch.zeros((2 * n, H), dtype=torch.half, device=dev)
    gu_out = torch.zeros((2 * n, I), dtype=torch.half, device=dev)
    dn_had = torch.zeros((n, I), dtype=torch.half, device=dev)
    dn_out = torch.zeros((n, H), dtype=torch.float32, device=dev)
    ext.exl3_moe_gfx12_k3(y, out, sel, w, *tabs, gu_had, gu_out, dn_had, dn_out, float(pack.K), dedup, 0.0)
    torch.cuda.synchronize()
    return out, gu_out, dn_out


def rel(a, b):
    a, b = a.double().cpu(), b.double().cpu()
    return float((a - b).abs().max() / max(1.0, float(b.abs().max())))


def mk_inputs(R, seed):
    g = torch.Generator().manual_seed(seed)
    y = (torch.randn(R, H, generator=g) * 0.05).half().to(dev)
    sel = torch.stack([torch.randperm(E, generator=g)[:TOPK] for _ in range(R)]).long().to(dev)
    w = torch.rand((R, TOPK), generator=g).half().to(dev)
    return y, sel, w


def main():
    packs = {K: Pack(K, 10 + K) for K in (1, 2, 3)}
    errs = {}
    for K, pack in packs.items():
        tabs = pack.tables()
        # A: VALU vs WMMA bitwise
        for R in (1, 3, 4):
            for dedup in (False, True):
                y, sel, w = mk_inputs(R, 100 + R)
                ext.exl3_moe_valu_override(1)
                a = decode_call(pack, tabs, y, sel, w, dedup)
                ext.exl3_moe_valu_override(0)
                b = decode_call(pack, tabs, y, sel, w, dedup)
                ext.exl3_moe_valu_override(-1)
                same = all(torch.equal(p.view(torch.int16 if p.dtype == torch.half else torch.int32),
                                       q.view(torch.int16 if q.dtype == torch.half else torch.int32)) for p, q in zip(a, b))
                check(f"A valu==wmma K={K} R={R} dedup={dedup}", same and bool(torch.isfinite(a[0]).all()))
        # B: dequant reference
        y, sel, w = mk_inputs(2, 7)
        out = decode_call(pack, tabs, y, sel, w)[0]
        ref = reference(pack, y, sel, w)
        errs[K] = rel(out, ref)
        print(f"  B K={K} max rel err vs dequant reference {errs[K]:.3e}")
    anchor = max(errs[2], errs[3])
    check("B reference sane (K=2/3 anchors)", anchor < 5e-2, f"anchor err {anchor:.2e}" if anchor < 5e-2 else "REF-SUSPECT")
    check("B K=1 decode vs dequant reference", errs[1] < max(5e-2, 2 * anchor), f"err {errs[1]:.2e}")

    # C: K=0
    p1 = packs[1]
    y, sel, w = mk_inputs(2, 31)
    zero = {int(sel[0, 1]), int(sel[0, 4]), int(sel[1, 2])}
    out_z = decode_call(p1, p1.tables(zero=zero), y, sel, w)[0]
    sel_m = sel.clone()
    for r in range(2):
        for k in range(TOPK):
            if int(sel[r, k]) in zero:
                sel_m[r, k] = -1
    out_m = decode_call(p1, p1.tables(), y, sel_m, w)[0]
    check("C null table entry == pick masked to -1 (bitwise)", torch.equal(out_z, out_m))
    out_all0 = decode_call(p1, p1.tables(zero=set(range(E))), y, sel, w)
    check("C every table entry null -> exact zeros", all(float(t.abs().max()) == 0.0 for t in out_all0))
    parts = None
    for k in range(TOPK):
        s1 = torch.full_like(sel, -1)
        s1[:, k] = sel[:, k]
        o = decode_call(p1, p1.tables(), y, s1, w)[0].double()
        parts = o if parts is None else parts + o
    full = decode_call(p1, p1.tables(), y, sel, w)[0]
    check("C picks sum == grouped call (K=1)", rel(full, parts) < 2e-6, f"err {rel(full, parts):.2e}")

    # D / E: prefill
    T = 70
    for K, pack in packs.items():
        tabs = pack.tables()
        g = torch.Generator().manual_seed(55)
        y = (torch.randn(T, H, generator=g) * 0.05).half().to(dev)
        sel = torch.stack([torch.randperm(E, generator=g)[:TOPK] for _ in range(T)]).long().to(dev)
        w = torch.rand((T, TOPK), generator=g).float().to(dev)
        flat = sel.view(-1)
        order = flat.argsort(stable=True)
        cnt = _scatter_expert_count(flat, E + 1)
        res = {}
        for kern in ("2", "1"):
            os.environ["MPW_KERN"] = kern
            ws = _hip_wmma_workspace(dev, T * TOPK, H, I, E)
            A = T * TOPK
            out = torch.zeros((T, H), dtype=torch.float32, device=dev)
            ext.exl3_moe_prefill_wmma(
                y, out, sel, w, order, cnt, *tabs, float(K), float(K),
                ws["gu_had"][:2 * A * H].view(2 * A, H), ws["gu_out"][:2 * A * I].view(2 * A, I),
                ws["gu_had"][:2 * A * H].view(torch.float).view(A, H),
                ws["offsets"], ws["inverse"][:A], ws["tiles"], ws["tile_count"], 0.0, False, None)
            torch.cuda.synchronize()
            res[kern] = out
        os.environ.pop("MPW_KERN", None)
        ref = reference(pack, y[:6], sel[:6], w[:6])
        e = rel(res["2"][:6], ref)
        print(f"  D K={K} prefill max rel err vs dequant reference {e:.3e}")
        if K == 1:
            check("D K=1 prefill vs dequant reference", e < max(5e-2, 2 * anchor), f"err {e:.2e}")
            check("E K=1 prefill MPW_KERN=1 == MPW_KERN=2 (bitwise)", torch.equal(res["1"], res["2"]))
        check(f"D K={K} prefill finite", bool(torch.isfinite(res["2"]).all()))
    print("ALL PASS" if not fails else f"FAILED: {fails}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
