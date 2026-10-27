"""TurboQuant KV cache (cache/tq.py): codec reference, Triton writer / window dequant bit-exactness, and the dispatched attention
(rotation outside the kernels, codebook loaders, KVWIN ring, staged prefill) against a torch reference over the dequantized pool.

Tiny shapes on any GPU, or on CPU through the Triton interpreter: TQ_TEST_CPU=1 pytest tests/test_tq_cache.py"""
import os, sys, types, math, contextlib
CPU = os.environ.get("TQ_TEST_CPU") == "1"
if CPU:
    os.environ["TRITON_INTERPRET"] = "1"
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest

from exllamav3.cache import tq
from exllamav3.cache.kvwin import KVWinLayerState, KVWinCtx, KVWinCfg, parse_spec

DEV = "cpu" if CPU else "cuda:0"
PAGE = 256
needs_gpu = pytest.mark.skipif(not CPU and not torch.cuda.is_available(), reason = "needs a GPU (or TQ_TEST_CPU=1)")


@pytest.fixture(autouse = True, params = ["legacy", "hip", "cu"])
def impl(request, monkeypatch):
    """Decode path: the Triton split kernels (any device), qdec_hip (ROCm GPU only, the ROCm default) or qdec.cu (CUDA, the default there)"""
    from exllamav3.modules.attention_fn import triton_paged
    if request.param == "hip" and (CPU or not triton_paged._QDEC_HIP):
        pytest.skip("qdec_hip needs a ROCm GPU")
    if request.param == "cu" and (CPU or torch.version.hip):
        pytest.skip("qdec.cu needs a CUDA GPU")
    monkeypatch.setattr(triton_paged, "_QDEC_HIP", request.param == "hip")
    monkeypatch.setattr(triton_paged, "ATTN16", request.param == "cu")
    return request.param


@pytest.fixture(autouse = True)
def cpu_env(monkeypatch):
    if not CPU:
        return
    from exllamav3.modules.attention_fn import triton_paged, triton_paged_cu
    props = types.SimpleNamespace(multi_processor_count = 8, major = 8, minor = 6, shared_memory_per_block_optin = 99 * 1024)
    monkeypatch.setattr(torch.cuda, "device", lambda d: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda d: props)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d = None: (8, 6))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    for mod in (triton_paged, triton_paged_cu):
        monkeypatch.setattr(mod, "_check_tensor", lambda *a, **k: None)
        monkeypatch.setattr(mod, "_same_device", lambda *a, **k: True)


def test_codec_unbiased_norm():
    # norm correction: the decoded vector has exactly the input norm (up to the fp16 scale)
    for D in (128, 256):
        for b in (5, 6):
            st = tq.TQState.get(D, b, b, "cpu")
            x = torch.randn(512, D) * 3
            y = (x @ st.rot_t32)
            w, s = tq.encode_ref(y, st.cbk, st.midk, b)
            r = tq.decode_ref(w, s, st.cbk, b, D).float()
            assert ((r.norm(dim = -1) / y.norm(dim = -1)) - 1).abs().max() < 2e-3
            assert ((r @ st.rot32 - x).norm() / x.norm()) < {5: 0.055, 6: 0.03}[b]


def test_rotation_orthonormal():
    for D in (64, 128, 256):
        m = tq.rotation(D)
        assert torch.allclose(m @ m.t(), torch.eye(D), atol = 1e-5)


@needs_gpu
@pytest.mark.parametrize("D,bits", [(128, 5), (128, 6), (256, 5), (256, 6)])
def test_triton_write_and_window_dequant_exact(D, bits):
    from exllamav3.modules.attention_fn import tq_triton as T
    HKV = 2
    st = tq.TQState.get(D, bits, bits, DEV)
    bsz, length, pages = 2, 3, 6
    bt = torch.tensor([[3, 1, 5], [0, 4, 2]], dtype = torch.int32, device = DEV)
    cs = torch.tensor([PAGE - 1, 2 * PAGE - 2], dtype = torch.int32, device = DEV)      # straddle page boundaries
    x = torch.randn(bsz, length, HKV, D, device = DEV).half()
    words = torch.zeros(pages, PAGE, HKV * D // 32 * bits, dtype = torch.int32, device = DEV)
    sc = torch.zeros(pages, PAGE, HKV, dtype = torch.half, device = DEV)
    T._tq_write_kernel[(bsz * length, HKV)](x, words, sc, st.cbk, st.midk, bt, cs, length, bt.shape[1], bits, PAGE, HKV, D)
    tok = cs.long().unsqueeze(1) + torch.arange(length, device = DEV).unsqueeze(0)
    rows = (bt.long().gather(1, tok // PAGE) * PAGE + tok % PAGE).reshape(-1)
    wr, sr = tq.encode_ref(x.reshape(-1, D), st.cbk, st.midk, bits)
    assert torch.equal(words.view(-1, words.shape[-1])[rows], wr.view(rows.numel(), -1))
    assert torch.equal(sc.view(-1, HKV)[rows], sr.view(rows.numel(), -1))
    npps = bt.shape[1]
    out = torch.zeros(bsz * npps, PAGE, HKV, D, dtype = torch.half, device = DEV)
    T._tq_dequant_window_kernel[(bsz * npps * (PAGE // 16), HKV)](words, sc, st.cbk, out, bt, cs, length, npps, bits, PAGE, HKV, D, 16)
    ref = tq.decode_ref(wr, sr, st.cbk, bits, D)
    got = torch.stack([out[b * npps + int(tok[b, t]) // PAGE, int(tok[b, t]) % PAGE] for b in range(bsz) for t in range(length)]).reshape(ref.shape)
    assert torch.equal(got, ref)


class Rig:
    """TQ pool + KVWIN ring for one sequence, driven through attn_dispatch like the attention module does."""

    def __init__(self, bits, hd = 128, kvh = 2, qh = 8, pages = 4, S = 4, W = 32, seed = 0):
        from exllamav3.cache.tq import CacheLayer_tq
        self.bits, self.hd, self.kvh, self.qh, self.S, self.W = bits, hd, kvh, qh, S, W
        self.g = torch.Generator(device = DEV).manual_seed(seed)
        attn = types.SimpleNamespace(num_kv_heads = kvh, head_dim = hd, window_arg = lambda: -1, logit_softcapping = 0.0)
        self.cl = CacheLayer_tq(None, attn, 0, pages * PAGE, bits, bits)
        self.cl.alloc(torch.device(DEV))
        self.st = KVWinLayerState(attn, 2, 0, 0)
        self.st.cfg = KVWinCfg(S, W, W + 16, S + W + 16)
        self.st.cache_layer = self.cl
        self.st.alloc(torch.device(DEV))
        self.slots = torch.tensor([1], dtype = torch.int32, device = DEV)
        self.bt = torch.arange(pages, dtype = torch.int32, device = DEV).view(1, -1)
        self.len = 0
        self.K = torch.empty(0, kvh, hd, dtype = torch.half, device = DEV)
        self.V = torch.empty(0, kvh, hd, dtype = torch.half, device = DEV)

    def rnd(self, *shape):
        x = torch.randn(*shape, device = DEV, generator = self.g)
        return (x * (0.5 + torch.rand(*shape[:-1], 1, device = DEV, generator = self.g) * 2)).half()

    def step(self, n):
        from exllamav3.modules.attention_fn.dispatch import attn_dispatch
        q, k, v = self.rnd(1, n, self.qh, self.hd), self.rnd(1, n, self.kvh, self.hd), self.rnd(1, n, self.kvh, self.hd)
        sl = torch.tensor([self.len], dtype = torch.int32, device = DEV)
        o = attn_dispatch(q, k, v, cache = self.cl, causal = True, block_table = self.bt, cache_seqlens = sl,
                          max_kv_len = self.len, ov = KVWinCtx(self.st, self.slots))
        L = self.len
        self.K = torch.cat([self.K, k[0]]); self.V = torch.cat([self.V, v[0]]); self.len += n
        return q[0], o[0], L

    def ref(self, q, L):          # q [n, qh, hd] at positions L..L+n-1
        """fp32 reference: sinks + window rows exact, every other position from the dequantized (natural domain) pool."""
        st = self.cl.get_tq()
        k, v = self.cl.get_kv(None, None)
        kd = k.view(-1, self.kvh, self.hd)[: self.len].float(); vd = v.view(-1, self.kvh, self.hd)[: self.len].float()
        ke, ve = self.K.float(), self.V.float()
        out = torch.empty(q.shape, dtype = torch.float32, device = DEV)
        grp = self.qh // self.kvh
        for i in range(q.shape[0]):
            qa = L + i
            pos = torch.arange(qa + 1, device = DEV)
            # decode / verify (n <= 16): per-row window; prefill chunk: the past window [L-W, L) and every chunk row are exact
            exact = (pos < self.S) | ((pos > qa - self.W) if q.shape[0] <= 16 else (pos >= L - self.W))
            kk = torch.where(exact[:, None, None], ke[: qa + 1], kd[: qa + 1]).repeat_interleave(grp, dim = 1)
            vv = torch.where(exact[:, None, None], ve[: qa + 1], vd[: qa + 1]).repeat_interleave(grp, dim = 1)
            s = torch.einsum("hd,thd->ht", q[i].float(), kk) / math.sqrt(self.hd)
            out[i] = torch.einsum("ht,thd->hd", s.softmax(-1), vv)
        return out


@needs_gpu
@pytest.mark.parametrize("hd,bits", [(128, 5), (128, 6), (256, 5), (256, 6)])
def test_dispatch_matches_dequantized_reference(hd, bits):
    rig = Rig(bits, hd = hd)
    for n in (300, 1, 1, 5, 40, 1):         # staged prefill chunk, decode, verify, a second chunk
        q, o, L = rig.step(n)
        r = rig.ref(q, L)
        err = ((o.float() - r).norm() / r.norm()).item()
        assert err < 6e-3, (hd, bits, n, L, err)


@needs_gpu
@pytest.mark.parametrize("D", [128, 256])
def test_fused_rotation_matches_gemm(D):
    from exllamav3.modules.attention_fn.tq_triton import _tq_rotate_kernel
    st = tq.TQState.get(D, 6, 6, DEV)
    xs = [torch.randn(n, D, device = DEV).half() for n in (24, 4, 37)]
    ys = [torch.empty_like(x) for x in xs]
    nblk = sum(-(-x.shape[0] // 16) for x in xs)
    _tq_rotate_kernel[(nblk, D // 64)](xs[0], xs[1], xs[2], ys[0], ys[1], ys[2], st.rot_t, 24, 4, 37, D, 16, 64)
    for x, y in zip(xs, ys):
        ref = x.float() @ st.rot_t.float()
        assert ((y.float() - ref).norm() / ref.norm()) < 1e-3
