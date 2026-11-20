"""CPU tests for EXL3_FQ_MMVQ_LEGACY: (a) numpy ggml dequant == engine dequant for Q4_0/Q4_1/Q5_0/Q5_1 bytes,
(b) numpy emulation of the ported vec_dot (q8_1 activations, lane/iqs walk) ~= dequant(W) @ x, (c) forward() dispatch flag off/on.
(c) runs in a subprocess per flag value (the flag is read at import)."""
import os, sys, subprocess
import pytest
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _xl():
    """Import the exllamav3 sub-packages under the name `xl` WITHOUT running exllamav3/__init__ (it would build/load the CUDA
    extension): bare package shells, a stub `xl.ext` and a stub Module base class."""
    import types
    if "xl" in sys.modules:
        return
    root = os.path.join(ROOT, "exllamav3")
    for name, sub in (("xl", ""), ("xl.modules", "modules"), ("xl.util", "util")):
        m = types.ModuleType(name); m.__path__ = [os.path.join(root, sub)]; sys.modules[name] = m
    ext = types.ModuleType("xl.ext"); ext.exllamav3_ext = types.SimpleNamespace(); sys.modules["xl.ext"] = ext
    mod = types.ModuleType("xl.modules.module"); mod.Module = type("Module", (), {"__init__": lambda self, *a, **k: None})
    sys.modules["xl.modules.module"] = mod


BB = {"q4_0": 18, "q4_1": 20, "q5_0": 22, "q5_1": 24}

def rand_blocks(fmt, n, nb, rng):
    raw = rng.integers(0, 256, size=(n, nb, BB[fmt]), dtype=np.uint8)
    def f16(lo, hi, shape):
        return rng.uniform(lo, hi, shape).astype(np.float16).view(np.uint8).reshape(*shape, 2)
    raw[:, :, 0:2] = f16(0.01, 0.1, (n, nb))
    if fmt in ("q4_1", "q5_1"):
        raw[:, :, 2:4] = f16(-1, 1, (n, nb))
    return raw.reshape(n, nb * BB[fmt])

def f16at(b, o):
    return np.ascontiguousarray(b[..., o:o + 2]).view(np.float16)[..., 0].astype(np.float32)

def deq_np(fmt, raw):                      # ggml dequantize_row_q*: low nibbles = elements 0..15, high = 16..31
    n = raw.shape[0]; b = raw.reshape(n, -1, BB[fmt]); d = f16at(b, 0)[..., None]
    if fmt == "q4_0":
        qs = b[..., 2:18]; lo, hi = (qs & 15).astype(np.float32) - 8, (qs >> 4).astype(np.float32) - 8
        return (d * np.concatenate([lo, hi], -1)).reshape(n, -1)
    if fmt == "q4_1":
        m = f16at(b, 2)[..., None]; qs = b[..., 4:20]
        return (d * np.concatenate([(qs & 15), (qs >> 4)], -1).astype(np.float32) + m).reshape(n, -1)
    if fmt == "q5_0":
        qh = np.ascontiguousarray(b[..., 2:6]).view(np.uint32)[..., 0]; qs = b[..., 6:22]; j = np.arange(16)
        lo = (qs & 15) | (((qh[..., None] >> j) & 1) << 4); hi = (qs >> 4) | (((qh[..., None] >> (j + 16)) & 1) << 4)
        return (d * (np.concatenate([lo, hi], -1).astype(np.float32) - 16)).reshape(n, -1)
    m = f16at(b, 2)[..., None]; qh = np.ascontiguousarray(b[..., 4:8]).view(np.uint32)[..., 0]; qs = b[..., 8:24]; j = np.arange(16)
    lo = (qs & 15) | (((qh[..., None] >> j) & 1) << 4); hi = (qs >> 4) | (((qh[..., None] >> (j + 16)) & 1) << 4)
    return (d * np.concatenate([lo, hi], -1).astype(np.float32) + m).reshape(n, -1)

def q8_1(x):                               # block_q8_1: d = amax/127, q = round(x/d), s = d * sum(q)  (as fq_quant_q8_1_rows)
    xb = x.reshape(-1, 32).astype(np.float32); amax = np.abs(xb).max(1, keepdims=True)
    d = amax / 127.0; q = np.where(d > 0, np.rint(xb / np.where(d > 0, d, 1)), 0).astype(np.int32)
    d16 = d.astype(np.float16).astype(np.float32); s16 = (d * q.sum(1, keepdims=True)).astype(np.float16).astype(np.float32)
    return q, d16[:, 0], s16[:, 0]

def u32(a): return np.ascontiguousarray(a).view(np.uint32).astype(np.uint32)

def dp4a(a, b):                            # a: 4 unsigned bytes (packed), b: 4 signed int8 (packed) -> int
    a8 = np.frombuffer(np.uint32(a).tobytes(), np.uint8).astype(np.int32); b8 = np.frombuffer(np.uint32(b).tobytes(), np.int8).astype(np.int32)
    return int((a8 * b8).sum())

def emulate_vecdot(fmt, raw, x):           # port of the CUDA vec_dot_*_q8_1 with vdr=2, qi=4: 2 lanes (iqs 0, 2) per block, summed
    n = raw.shape[0]; K = x.shape[0]; nb = K // 32; qa, d8, s8 = q8_1(x)
    out = np.zeros(n, np.float32)
    for r in range(n):
        for kb in range(nb):
            blk = raw[r, kb * BB[fmt]:(kb + 1) * BB[fmt]]; u = [int(v) for v in u32(qa[kb].astype(np.int8).view(np.uint8))]
            for iqs in (0, 2):
                sumi = 0
                if fmt in ("q4_0", "q4_1"):
                    off = 2 if fmt == "q4_0" else 4; qs = u32(blk[off:off + 16])
                    for i in range(2):
                        v = int(qs[iqs + i]); sumi += dp4a(v & 0x0F0F0F0F, u[iqs + i]) + dp4a((v >> 4) & 0x0F0F0F0F, u[iqs + i + 4])
                else:
                    ho, qo = (2, 6) if fmt == "q5_0" else (4, 8); qh = int(u32(blk[ho:ho + 4])[0]); qs = u32(blk[qo:qo + 16])
                    for i in range(2):
                        vl = int(qs[iqs + i]); vh = qh >> (4 * (iqs + i)); M = 0xFFFFFFFF
                        vi0 = (vl & 0x0F0F0F0F) | ((vh << 4) & 0x10) | ((vh << 11) & 0x1000) | ((vh << 18) & 0x100000) | ((vh << 25) & 0x10000000)
                        vi1 = ((vl >> 4) & 0x0F0F0F0F) | ((vh >> 12) & 0x10) | ((vh >> 5) & 0x1000) | ((vh << 2) & 0x100000) | ((vh << 9) & 0x10000000)
                        sumi += dp4a(vi0 & M, u[iqs + i]) + dp4a(vi1 & M, u[iqs + i + 4])
                d, ds, ss = f16at(blk, 0), d8[kb], s8[kb]
                if fmt == "q4_0": out[r] += d * (sumi * ds - 4 * ss)
                elif fmt == "q5_0": out[r] += d * (sumi * ds - 8 * ss)
                else:
                    m = f16at(blk, 2); out[r] += sumi * d * ds + m * ss / 2
    return out

def test_numeric():
    import torch
    _xl()
    from xl.foreign_quant.gap_formats import dequant
    rng = np.random.default_rng(0)
    for fmt in BB:
        raw = rand_blocks(fmt, 6, 8, rng)               # 6 rows x K=256
        ref = deq_np(fmt, raw)
        # q4_1 is not an engine format yet (not in BLOCKS / _FORMATS): numpy reference only
        e1 = np.abs(ref - dequant(fmt, torch.from_numpy(raw)).float().numpy()).max() if fmt != "q4_1" else 0.0
        x = rng.standard_normal(256).astype(np.float32)
        y = emulate_vecdot(fmt, raw, x); yref = ref.astype(np.float64) @ x
        e2 = np.abs(y - yref).max() / np.abs(yref).max()
        print(f"{fmt}: numpy-vs-engine dequant max abs {e1:.2e} (fp16 rounding); vec_dot emulation vs W@x rel {e2:.2e}")
        assert e1 < 2e-3 * max(1, np.abs(ref).max()) and e2 < 2e-2, fmt

def child():
    import types, torch
    _xl()
    import xl.modules.foreign_linear as fl
    flag = fl._MMVQ_LEGACY; calls = []
    ext = types.SimpleNamespace(fq_q8_scratch=lambda dev, R, K: torch.zeros(R, K // 32 * 36, dtype=torch.uint8),
                                fq_quant_q8_1_rows=lambda x, qb: None)
    for f in ("q4_0", "q4_1", "q5_0", "q5_1", "iq4_nl", "q4_k"):
        setattr(ext, "fq_mmvq_" + f, (lambda f: lambda qb, p, out: calls.append(("mmvq", f)))(f))
    fl.ext = ext; torch.cuda.current_device = lambda: 0
    fl.triton_gap.gemv = lambda fmt, x, p: (calls.append(("triton", fmt)), torch.zeros(x.shape[0], p.shape[0], dtype=torch.half))[1]
    fl.triton_gap.supported = lambda *a, **k: True
    res = {}
    for fmt in ("q4_0", "q5_0", "q5_1"):   # q4_1 cannot be a ForeignLinear (not in _FORMATS)
        for R in (1, 2):
            m = object.__new__(fl.ForeignLinear); m.format = fmt; m.in_features = 64; m.out_features = 4; m.key = "t"
            m.packed = torch.zeros(4, 2 * BB[fmt], dtype=torch.uint8); m.scales = None; m.global_scale = None
            m.out_dtype = torch.half; m.post_scale = 1.0; m.softcap = 0.0; m.pre_scale = 1.0
            calls.clear()
            try: m.forward(torch.zeros(R, 64, dtype=torch.half), {})
            except Exception as e: calls.append(("exc", type(e).__name__ + str(e)[:80]))
            res[(fmt, R)] = calls[0] if calls else None
    print("flag", int(flag), res)
    for (fmt, R), c in res.items():
        want = ("mmvq", fmt) if (flag and R == 1) else ("triton", fmt)
        assert c == want, (fmt, R, c, want)
    print("dispatch OK flag", int(flag))

@pytest.mark.parametrize("flag", ["0", "1"])
def test_dispatch(flag):
    r = subprocess.run([sys.executable, __file__, "child"], env={**os.environ, "EXL3_FQ_MMVQ_LEGACY": flag},
                       capture_output=True, text=True, timeout=100)
    assert r.returncode == 0, r.stdout[-900:] + r.stderr[-1500:]


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "child":
    child()
