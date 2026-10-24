"""
Paged decode attention: pre-1.6 Kyojin kernels (triton_paged, ATTN16 forced off) vs the ported upstream exllamav3 1.6.0
(MIT) kernels (triton_paged_cu), CUDA only. fp16 and 6-bit quantized caches (built with ext.quant_cache_paged, as
cache/quant.py does), q_len 1 / 8, 24 q heads / 4 kv heads, head_dim 256, page size 256, contexts 2K / 20K / 64K,
no window and a 2048 left window (causal). Reports the max abs diff against an fp32 torch reference (dequantized
K/V for the quantized cache) for both paths and microseconds per call (CUDA events, 20 calls after 3 warm-ups).

  pytest -s tests/port/test_attn16.py        or        python tests/port/test_attn16.py
Needs a GPU (not run by the port; the GPU is shared).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import itertools
import torch
try:
    import pytest
except ImportError:   # the CUDA venv has no pytest: __main__ runs the same grid without it
    pytest = None

if pytest is not None:
    pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or bool(torch.version.hip), reason = "CUDA only")

from exllamav3.constants import PAGE_SIZE
from exllamav3.modules.attention_fn import triton_paged as tp

NQ, NKV, HD = 24, 4, 256
BITS = 6
dev = "cuda:0"
_ext = None


def ext():
    global _ext
    if _ext is None:
        from exllamav3.ext import exllamav3_ext as e
        _ext = e
    return _ext


def make_case(ctx, q_len, quant, seed = 0):
    """ctx = total kv length including the q_len query rows, which are already in the cache."""
    torch.manual_seed(seed)
    pages = -(-ctx // PAGE_SIZE)
    kc = torch.randn((pages, PAGE_SIZE, NKV, HD), dtype = torch.half, device = dev)
    vc = torch.randn_like(kc)
    bt = torch.randperm(pages, device = dev, dtype = torch.int32).view(1, pages)
    sl = torch.full((1,), ctx - q_len, dtype = torch.int32, device = dev)
    q = torch.randn((1, q_len, NQ, HD), dtype = torch.half, device = dev)
    if not quant:
        return dict(q = q, kc = kc, vc = vc, bt = bt, sl = sl, kref = kc, vref = vc, qc = None)
    e = ext()
    qk = torch.zeros((pages, PAGE_SIZE, NKV * HD // 32 * BITS), dtype = torch.int, device = dev)
    qv = torch.zeros_like(qk)
    sk = torch.zeros((pages, PAGE_SIZE, NKV * HD // 32), dtype = torch.half, device = dev)
    sv = torch.zeros_like(sk)
    # Logical token t of the sequence = flat row t of kc / vc, appended in chunks (the kernel's grid y = chunk length,
    # max 65535), as CacheLayer_quant.update_kv appends new tokens
    kf, vf = kc.view(1, pages * PAGE_SIZE, NKV, HD), vc.view(1, pages * PAGE_SIZE, NKV, HD)
    for s0 in range(0, ctx, 16384):
        n = min(16384, ctx - s0)
        start = torch.full((1,), s0, dtype = torch.int, device = dev)
        e.quant_cache_paged(kf[:, s0:s0 + n].contiguous(), qk, sk, vf[:, s0:s0 + n].contiguous(), qv, sv, start, bt,
                            PAGE_SIZE, n, 0.0, False)
    kd = torch.empty_like(kc); vd = torch.empty_like(vc)
    e.dequant_cache_paged(qk, sk, kd, qv, sv, vd, torch.full((1,), ctx, dtype = torch.int, device = dev), bt,
                          PAGE_SIZE, -1, 0.0)
    return dict(q = q, kc = qk, vc = qv, bt = bt, sl = sl, kref = kd, vref = vd, qc = (sk, sv, BITS, BITS))


def reference(c, ctx, q_len, window):
    q = c["q"]
    pages = c["bt"].shape[1]
    idx = c["bt"][0].long()
    K = c["kref"][idx].reshape(pages * PAGE_SIZE, NKV, HD)[:ctx]
    V = c["vref"][idx].reshape(pages * PAGE_SIZE, NKV, HD)[:ctx]
    lo = 0 if window is None else max(0, ctx - q_len - window)   # keys below carry zero weight
    K = K[lo:].float().permute(1, 0, 2)     # [kvh, T, d]
    V = V[lo:].float().permute(1, 0, 2)
    T = K.shape[1]
    g = NQ // NKV
    qf = q[0].float().view(q_len, NKV, g, HD).permute(1, 0, 2, 3)           # [kvh, q, g, d]
    s = torch.einsum("hqgd,htd->hqgt", qf, K) * HD ** -0.5
    qpos = (ctx - q_len + torch.arange(q_len, device = dev)).view(1, q_len, 1, 1)
    kpos = (lo + torch.arange(T, device = dev)).view(1, 1, 1, T)
    mask = kpos <= qpos
    if window is not None:
        mask &= kpos >= qpos - window
    s = s.masked_fill(~mask, -float("inf"))
    o = torch.einsum("hqgt,htd->hqgd", torch.softmax(s, -1), V)
    return o.permute(1, 0, 2, 3).reshape(1, q_len, NQ, HD)


def run_old(c, window, q_len):
    saved = tp.ATTN16
    tp.ATTN16 = False
    try:
        return tp.paged_attn_triton_decode(
            c["q"], None, None, c["kc"], c["vc"], c["bt"], c["sl"], causal = True, window_size = window,
            qc = c["qc"], pre_appended_len = q_len, n_kv_heads_override = NKV)
    finally:
        tp.ATTN16 = saved


def run_new(c, window, q_len):
    from exllamav3.modules.attention_fn.triton_paged_cu import paged_attn_triton_decode_cu
    return paged_attn_triton_decode_cu(
        c["q"], None, None, c["kc"], c["vc"], c["bt"], c["sl"], causal = True, window_size = window,
        qc = c["qc"], pre_appended_len = q_len, n_kv_heads_override = NKV)


def time_us(fn, n = 20, warm = 3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
    a.record()
    for _ in range(n):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1000 / n


GRID = dict(quant = [False, True], q_len = [1, 8], window = [None, 2048], ctx = [2048, 20480, 65536])


def _param(f):
    if pytest is None:
        return f
    for k in reversed(list(GRID)):
        f = pytest.mark.parametrize(k, GRID[k])(f)
    return f


@_param
def test_attn16(quant, q_len, window, ctx):
    c = make_case(ctx, q_len, quant, seed = ctx + q_len)
    ref = reference(c, ctx, q_len, window)
    o_old = run_old(c, window, q_len).float()
    o_new = run_new(c, window, q_len).float()
    if quant and not torch.isfinite(o_old).all():   # NaN triage (attn1: q6 64K NaN in BOTH paths): where does it start?
        print(f"\n[attn16] nan triage ctx={ctx} q_len={q_len} win={window}: kref nonfinite {(~torch.isfinite(c['kref'])).sum().item()}, "
              f"vref {(~torch.isfinite(c['vref'])).sum().item()}, sk {(~torch.isfinite(c['qc'][0])).sum().item()}, "
              f"sv {(~torch.isfinite(c['qc'][1])).sum().item()}, ref {(~torch.isfinite(ref)).sum().item()}, "
              f"old {(~torch.isfinite(o_old)).sum().item()}, new {(~torch.isfinite(o_new)).sum().item()}")
    e_old = (o_old - ref).abs().max().item()
    e_new = (o_new - ref).abs().max().item()
    t_old = time_us(lambda: run_old(c, window, q_len))
    t_new = time_us(lambda: run_new(c, window, q_len))
    print(f"\n[attn16] {'q6  ' if quant else 'fp16'} q_len={q_len} win={window} ctx={ctx:6d}: "
          f"maxabs old={e_old:.2e} new={e_new:.2e} old-vs-new={(o_old - o_new).abs().max().item():.2e} | "
          f"us/call old={t_old:7.1f} new={t_new:7.1f} ({t_old / t_new:.2f}x)")
    # fp16 outputs from fp16 inputs with unit-variance data: both paths sit at fp16 rounding noise
    assert e_new < 5e-3 and e_old < 5e-3
    assert e_new <= max(2 * e_old, 2e-3)


if __name__ == "__main__":
    if pytest is not None:
        sys.exit(pytest.main([__file__, "-s", "-q", "-p", "no:cacheprovider"]))
    fails = 0
    for vals in itertools.product(*GRID.values()):
        try:
            test_attn16(**dict(zip(GRID, vals)))
        except AssertionError as e:
            fails += 1; print("[attn16] FAIL", dict(zip(GRID, vals)), e)
    print(f"[attn16] {fails} failed of {len(list(itertools.product(*GRID.values())))}")
    sys.exit(1 if fails else 0)
