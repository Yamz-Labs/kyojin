"""Grouped q8_1 decode MoE with IQ experts (fq_moe_kq_grouped_mixed_q8 FMT 5 IQ2_S gate/up + FMT 4 IQ3_XXS down, ggml vecdotq ports)
vs the torch reference (gap_formats.dequant, fp32 dots, torch-half SwiGLU like the per-expert path) on 8 real experts of an unsloth
UD-IQ3_XXS MoE GGUF (default: Qwen3.6-35B-A3B, first layer whose down experts are IQ3_XXS). q8_1 activations -> compared by relative
error (~0.5-1 % expected); a wrong port (layout, signs, scales, high index bits) gives errors of order 1.
Usage: python tests/test_moe_iq_q8.py [GGUF]"""
import os, struct, sys

import numpy as np
import torch

import exllamav3_ext as ext
from exllamav3.foreign_quant import gap_formats

GGUF = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MOE_IQ_GGUF", "model.gguf")
SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
BLK = {18: ("iq3_xxs", 98), 22: ("iq2_s", 82)}
FID = {"iq3_xxs": 4, "iq2_s": 5}


def index(path):
    f = open(path, "rb")
    u32 = lambda: struct.unpack("<I", f.read(4))[0]; u64 = lambda: struct.unpack("<Q", f.read(8))[0]
    def s(): return f.read(u64()).decode()
    def skip(t):
        if t == 8: f.seek(u64(), 1)
        elif t == 9:
            at, n = u32(), u64()
            if at in SZ: f.seek(SZ[at] * n, 1)
            else:
                for _ in range(n): skip(at)
        else: f.seek(SZ[t], 1)
    assert f.read(4) == b"GGUF"; u32(); nt = u64(); nkv = u64(); align = 32
    for _ in range(nkv):
        k = s(); t = u32()
        if k == "general.alignment": align = u32(); continue
        skip(t)
    infos = {}
    for _ in range(nt):
        n = s(); nd = u32(); dims = [u64() for _ in range(nd)]; t = u32(); off = u64(); infos[n] = (dims, t, off)
    return f, infos, (f.tell() + align - 1) // align * align


def experts(f, infos, base, name, ne):
    dims, t, off = infos[name]
    fmt, bb = BLK[t]
    Kin, Nout = dims[0], dims[1]
    per = Nout * (Kin // 256) * bb
    f.seek(base + off)
    raw = np.frombuffer(f.read(per * ne), dtype=np.uint8).reshape(ne, Nout, -1)
    return fmt, torch.from_numpy(raw.copy()).cuda(), Kin, Nout


def test_moe_iq_q8_matches_reference():
    f, infos, base = index(GGUF)
    layer = next(i for i in range(200) if infos.get(f"blk.{i}.ffn_down_exps.weight", (0, 0))[1] == 18
                 and infos[f"blk.{i}.ffn_gate_exps.weight"][1] == 22)
    ne = 8
    fg, wg, K, I = experts(f, infos, base, f"blk.{layer}.ffn_gate_exps.weight", ne)
    fu, wu, _, _ = experts(f, infos, base, f"blk.{layer}.ffn_up_exps.weight", ne)
    fd, wd, I2, K2 = experts(f, infos, base, f"blk.{layer}.ffn_down_exps.weight", ne)
    assert (fg, fu, fd) == ("iq2_s", "iq2_s", "iq3_xxs") and (I2, K2) == (I, K), (fg, fu, fd, K, I, I2, K2)
    dq = lambda fmt, w, n, k: torch.stack([gap_formats.dequant(fmt, w[e]).float().reshape(n, k) for e in range(ne)])
    Gd, Ud, Dd = dq(fg, wg, I, K), dq(fu, wu, I, K), dq(fd, wd, K, I)
    ptrs = lambda w: torch.tensor([w[e].data_ptr() for e in range(ne)], dtype=torch.int64, device="cuda")
    gp, upp, dp = ptrs(wg), ptrs(wu), ptrs(wd)
    gs = wg[0].stride(0); ds = wd[0].stride(0)
    g = torch.Generator(device="cpu").manual_seed(0)
    worst = 0.0
    for R in (1, 4, 8):
        topk = 4
        x = (torch.randn(R, K, generator=g) * 0.5).half().cuda()
        sel = torch.stack([torch.randperm(ne, generator=g)[:topk] for _ in range(R)]).cuda()
        wts = torch.rand(R, topk, generator=g).half().cuda()
        ref = torch.zeros(R, K, device="cuda")
        for r in range(R):
            for j in range(topk):
                e = int(sel[r, j])
                gg = (x[r].float() @ Gd[e].t()).half().float(); uu = (x[r].float() @ Ud[e].t()).half().float()
                act = (torch.sigmoid(gg) * gg * uu).half().float()
                ref[r] += float(wts[r, j]) * (act @ Dd[e].t())
        partial = torch.empty(R * topk, K, dtype=torch.float, device="cuda")
        out = torch.empty(R, K, dtype=torch.float, device="cuda")
        ext.fq_moe_kq_grouped_mixed_q8(x, sel.reshape(-1).contiguous(), wts.reshape(-1).contiguous(), gp, upp, dp,
                                       gs, gs, ds, FID[fg], FID[fd], partial, out)
        torch.cuda.synchronize()
        rel = float((out - ref).norm() / ref.norm())
        print(f"layer {layer} R={R} rel err {rel:.4f}")
        worst = max(worst, rel)
    assert worst < 0.03, f"relative error {worst:.4f} (q8_1 activations alone give ~0.005-0.01)"


if __name__ == "__main__":
    test_moe_iq_q8_matches_reference()
    print("PASS")
