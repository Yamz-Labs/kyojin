"""Grouped q8_1 decode MoE: v2 wide-load kernels (gate+up Q4_K, down Q4_K / Q5_K / Q6_K; BOLT_FQ_MOE_V2=1, default) vs the
v1 lane kernels (BOLT_FQ_MOE_V2=0) on real experts of a K-quant MoE GGUF (e.g. Qwen3.6-35B-A3B UD-Q4_K_XL: the first layer of
each (gate/up, down) format pair present):
outputs must be bit-identical (torch.equal), rows R = 1 / 4 / 8, 16-byte aligned expert rows and a misaligned copy (v1 fallback).
The flag is read once per process, so each arm runs in its own subprocess.
Usage: python tests/test_moe_v2.py [GGUF]"""
import os, struct, subprocess, sys, tempfile

import numpy as np
import torch

SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
BLK = {12: ("q4_k", 144, 0), 13: ("q5_k", 176, 1), 14: ("q6_k", 210, 2)}
GGUF = os.environ.get("MOE_V2_GGUF") or (sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "")


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


def experts(f, infos, base, name, ne, shift):
    dims, t, off = infos[name]
    fmt, bb, fid = BLK[t]
    Kin, Nout = dims[0], dims[1]
    per = Nout * (Kin // 256) * bb
    f.seek(base + off)
    raw = torch.from_numpy(np.frombuffer(f.read(per * ne), dtype=np.uint8).copy())
    buf = torch.zeros(per * ne + 64, dtype=torch.uint8, device="cuda")
    buf[shift:shift + per * ne] = raw.cuda()
    w = buf[shift:shift + per * ne].view(ne, Nout, -1)
    return fid, w, buf, Kin, Nout


def arm(out_path):
    import exllamav3_ext as ext
    f, infos, base = index(GGUF)
    pairs = {}
    for i in range(200):
        if f"blk.{i}.ffn_down_exps.weight" not in infos: continue
        key = (infos[f"blk.{i}.ffn_gate_exps.weight"][1], infos[f"blk.{i}.ffn_down_exps.weight"][1])
        if all(t in BLK for t in key) and infos[f"blk.{i}.ffn_up_exps.weight"][1] == key[0]: pairs.setdefault(key, i)
    ne, topk = 16, 8
    res = {}
    for key, layer in sorted(pairs.items()):
        for shift in (0, 4):
            fg, wg, bg, K, I = experts(f, infos, base, f"blk.{layer}.ffn_gate_exps.weight", ne, shift)
            fu, wu, bu, _, _ = experts(f, infos, base, f"blk.{layer}.ffn_up_exps.weight", ne, shift)
            fd, wd, bd, I2, K2 = experts(f, infos, base, f"blk.{layer}.ffn_down_exps.weight", ne, shift)
            assert (I2, K2) == (I, K)
            ptrs = lambda w: torch.tensor([w[e].data_ptr() for e in range(ne)], dtype=torch.int64, device="cuda")
            g = torch.Generator(device="cpu").manual_seed(1)
            for R in (1, 4, 8):
                x = (torch.randn(R, K, generator=g) * 0.5).half().cuda()
                sel = torch.stack([torch.randperm(ne, generator=g)[:topk] for _ in range(R)]).cuda()
                wts = torch.rand(R, topk, generator=g).half().cuda()
                partial = torch.empty(R * topk, K, dtype=torch.float, device="cuda")
                out = torch.empty(R, K, dtype=torch.float, device="cuda")
                ext.fq_moe_kq_grouped_mixed_q8(x, sel.reshape(-1).contiguous(), wts.reshape(-1).contiguous(), ptrs(wg), ptrs(wu),
                                               ptrs(wd), wg[0].stride(0), wu[0].stride(0), wd[0].stride(0), fg, fd, partial, out)
                torch.cuda.synchronize()
                res[f"L{layer}_ug{fg}_d{fd}_shift{shift}_R{R}"] = out.cpu()
    layer = sorted(pairs.items())
    torch.save({"layer": layer, "res": res}, out_path)


def test_moe_v2_bit_identical():
    assert GGUF, "set MOE_V2_GGUF or pass the GGUF path"
    d = tempfile.mkdtemp()
    outs = []
    for flag in ("0", "1"):
        p = os.path.join(d, f"arm{flag}.pt")
        env = dict(os.environ, BOLT_FQ_MOE_V2=flag, MOE_V2_GGUF=GGUF)
        subprocess.run([sys.executable, __file__, "--arm", p], env=env, check=True)
        outs.append(torch.load(p))
    a, b = outs[0]["res"], outs[1]["res"]
    for k in a:
        eq = torch.equal(a[k], b[k])
        print(f"{k}: equal={eq} max|d|={float((a[k] - b[k]).abs().max()):.3e}")
        assert eq, k


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--arm":
        arm(sys.argv[2])
    else:
        test_moe_v2_bit_identical()
        print("PASS")
