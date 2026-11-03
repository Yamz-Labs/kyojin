"""KVWIN (fp16 sink + recent-window ring next to a quantized KV pool): ring write / merge / rollback / restore checks.

Tiny shapes on any GPU (CUDA or ROCm); run with pytest or `python tests/test_kvwin.py`. The references are fp32 torch.
"""
import os, sys, types, math, contextlib
# KVWIN_TEST_CPU=1: run the Triton kernels in the Triton interpreter on CPU tensors (no GPU, slow); set before triton is imported
CPU = os.environ.get("KVWIN_TEST_CPU") == "1"
if CPU:
    os.environ["TRITON_INTERPRET"] = "1"
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest

pytestmark = pytest.mark.skipif(not CPU and not torch.cuda.is_available(), reason = "needs a GPU (or KVWIN_TEST_CPU=1)")

from hardware import require
require("ext")
from exllamav3.cache.quant import CacheLayer_quant
from exllamav3.cache import kvwin
from exllamav3.cache.kvwin import KVWinLayerState, KVWinCtx, KVWinCfg, ring_rows, ring_positions, ov_write_ref, parse_spec
from exllamav3.modules.attention_fn.kvwin_triton import ov_write
from exllamav3.modules.attention_fn.triton_paged import paged_attn_triton_decode, paged_attn_triton_prefill, _get_h32

DEV = "cpu" if CPU else "cuda:0"
PAGE = 256


@pytest.fixture(autouse = True)
def cpu_env(monkeypatch):
    """CPU/interpreter mode: the launch wrappers query the CUDA device (context, SM count, shared memory); stub those."""
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
    monkeypatch.setattr(triton_paged_cu, "pick_config", lambda device, name, key, candidates, probe: candidates[0])


@pytest.fixture(autouse = True, params = ["legacy", "cu", "hip"])
def impl(request, monkeypatch):
    """Decode wrappers: the pre-1.6 Triton kernels (ROCm default without qdec_hip), the 1.6 ones (CUDA default; run here on
    whatever GPU) and qdec_hip (ROCm default; skipped without a ROCm GPU)."""
    from exllamav3.modules.attention_fn import triton_paged
    if request.param == "hip" and (CPU or not triton_paged._QDEC_HIP):
        pytest.skip("qdec_hip needs a ROCm GPU")
    monkeypatch.setattr(triton_paged, "ATTN16", request.param == "cu")
    monkeypatch.setattr(triton_paged, "_QDEC_HIP", request.param == "hip")
    return request.param


class Rig:
    """A quantized pool + KVWIN ring for nseq sequences (slot i of sequence i unless slots given)."""

    def __init__(self, cfg, kvh = 2, qh = 8, hd = 128, kb = 4, vb = 4, pages = 8, nslots = 4, slots = (2, 0), seed = 0):
        self.cfg, self.kvh, self.qh, self.hd, self.kb, self.vb = cfg, kvh, qh, hd, kb, vb
        self.g = torch.Generator(device = DEV).manual_seed(seed)
        attn = types.SimpleNamespace(num_kv_heads = kvh, head_dim = hd, window_arg = lambda: -1, logit_softcapping = 0.0)
        self.cl = CacheLayer_quant(None, attn, 0, pages * PAGE, kb, vb)
        self.cl.alloc(torch.device(DEV))
        self.st = KVWinLayerState(attn, nslots, 0, 0)
        self.st.cfg = cfg
        self.st.cache_layer = self.cl
        self.st.alloc(torch.device(DEV))
        self.nseq = len(slots)
        self.slots = torch.tensor(slots, dtype = torch.int32, device = DEV)
        # non-trivial page order: sequence b owns pages perm[b*pp : (b+1)*pp]
        pp = pages // self.nseq
        perm = torch.randperm(pages, generator = torch.Generator().manual_seed(seed + 1))
        self.bt = perm[: pp * self.nseq].view(self.nseq, pp).to(torch.int32).to(DEV)
        self.len = [0] * self.nseq
        self.K = [torch.empty(0, kvh, hd, dtype = torch.half, device = DEV) for _ in slots]
        self.V = [torch.empty(0, kvh, hd, dtype = torch.half, device = DEV) for _ in slots]

    def rnd(self, *shape):
        # K/V with per-token scale variation so quantization error is visible
        x = torch.randn(*shape, device = DEV, generator = self.g)
        return (x * (0.5 + torch.rand(*shape[:-1], 1, device = DEV, generator = self.g) * 2)).half()

    def ctx(self, rows = None):
        rows = list(range(self.nseq)) if rows is None else rows
        return KVWinCtx(self.st, self.slots[rows].contiguous())

    def append(self, b, n, ring = True, k = None, v = None):
        """Quantize n new rows of sequence b into the pool and (ring) into the ring, like attn_dispatch does."""
        k = self.rnd(1, n, self.kvh, self.hd) if k is None else k
        v = self.rnd(1, n, self.kvh, self.hd) if v is None else v
        sl = torch.tensor([self.len[b]], dtype = torch.int32, device = DEV)
        bt = self.bt[b:b + 1].contiguous()
        self.cl.update_kv_direct(sl, bt, k, v, n)
        if ring:
            ov_write(self.ctx([b]), k, v, sl, n)
        self.K[b] = torch.cat([self.K[b][: self.len[b]], k[0]])
        self.V[b] = torch.cat([self.V[b][: self.len[b]], v[0]])
        self.len[b] += n
        return k, v

    def dequant(self, b):
        """fp32 dequantized K/V of sequence b, positions 0..len-1."""
        sl = torch.tensor([self.len[b]], dtype = torch.int32, device = DEV)
        k, v = self.cl.get_kv(sl, self.bt[b:b + 1].contiguous())
        pos = torch.arange(self.len[b], device = DEV)
        page = self.bt[b].long()[pos // PAGE]
        return k[page, pos % PAGE].float(), v[page, pos % PAGE].float()


def ref_attn(rig, b, q, L, mix = True):
    """q [q_len, qh, hd] at positions L..L+q_len-1 of sequence b (all of whose K/V are in the rig).
    mix: sinks + per-row window exact, the rest dequantized; else everything dequantized."""
    cfg = rig.cfg
    kd, vd = rig.dequant(b)
    ke, ve = rig.K[b].float(), rig.V[b].float()
    out = torch.empty(q.shape, dtype = torch.float32, device = DEV)
    group = rig.qh // rig.kvh
    for i in range(q.shape[0]):
        qa = L + i
        pos = torch.arange(qa + 1, device = DEV)
        exact = ((pos < cfg.S) | (pos > qa - cfg.W)) if mix else torch.zeros_like(pos, dtype = torch.bool)
        kk = torch.where(exact[:, None, None], ke[: qa + 1], kd[: qa + 1])
        vv = torch.where(exact[:, None, None], ve[: qa + 1], vd[: qa + 1])
        kk = kk.repeat_interleave(group, dim = 1)     # [T, qh, hd]
        vv = vv.repeat_interleave(group, dim = 1)
        s = torch.einsum("hd,thd->ht", q[i].float(), kk) / math.sqrt(rig.hd)
        out[i] = torch.einsum("ht,thd->hd", s.softmax(-1), vv)
    return out


def decode(rig, b, q, L, ring = True, **kw):
    q_len = q.shape[0]
    sl = torch.tensor([L], dtype = torch.int32, device = DEV)
    return paged_attn_triton_decode(
        q[None].contiguous(), None, None, rig.cl.qk, rig.cl.qv, rig.bt[b:b + 1].contiguous(), sl, causal = True,
        qc = (rig.cl.sk, rig.cl.sv, rig.kb, rig.vb), pre_appended_len = q_len, n_kv_heads_override = rig.kvh,
        ring = rig.ctx([b]) if ring else None, **kw)[0]


def close(a, b, rel = 6e-3):
    err = (a.float() - b.float()).abs().max().item()
    mx = b.float().abs().max().item()
    return err <= rel * max(mx, 1e-3), err, mx


CFGS = [KVWinCfg(4, 32, 32 + 16, 4 + 32 + 16), KVWinCfg(0, 64, 64 + 16, 64 + 16), KVWinCfg(4, 128, 128 + 16, 4 + 128 + 16)]


# -- ring row mapping / reference writer -------------------------------------------------------------------------------

def test_parse():
    assert parse_spec(None) is None and parse_spec("") is None and parse_spec("0") is None
    c = parse_spec("4:128")
    assert (c.S, c.W, c.Rc, c.R) == (4, 128, 144, 148)
    with pytest.raises(ValueError):
        parse_spec("12")


@pytest.mark.parametrize("cfg", CFGS)
def test_ring_write_matches_reference(cfg):
    """Chunks crossing a page boundary and several ring wraps, then decode and verify rows: ring == reference writer."""
    rig = Rig(cfg, pages = 8, slots = (3, 1))
    ref_k = torch.zeros(4, cfg.R, rig.kvh, rig.hd, dtype = torch.half, device = DEV)
    ref_v = torch.zeros_like(ref_k)
    for n in (200, 100, 257, 1, 1, 7, 16, 300, 1, 5):    # 889 rows: page edge at 256/512/768, wraps of 48..148
        for b in (0, 1):
            L = rig.len[b]
            if L + n > 8 // 2 * PAGE:
                continue
            k, v = rig.append(b, n)
            ov_write_ref(ref_k, ref_v, int(rig.slots[b]), k[0], v[0], L, cfg)
    for b in (0, 1):
        s = int(rig.slots[b])
        assert torch.equal(rig.st.ov_k[s], ref_k[s]) and torch.equal(rig.st.ov_v[s], ref_v[s]), f"slot {s}"
    # untouched slots stay zero
    assert not rig.st.ov_k[0].any() and not rig.st.ov_k[2].any()
    # positions held by the ring after the last write
    for b in (0, 1):
        p = ring_positions(rig.len[b], cfg)
        s = int(rig.slots[b])
        for r in range(cfg.R):
            if p[r] >= 0:
                assert torch.equal(rig.st.ov_k[s, r], rig.K[b][int(p[r])]), (b, r, int(p[r]))


# -- attention: LSE merge == attention over the dequantized + exact mix ----------------------------------------------------

@pytest.mark.parametrize("cfg", CFGS)
@pytest.mark.parametrize("bits", [(4, 4), (6, 6), (8, 8)])
@pytest.mark.parametrize("q_len", [1, 5, 16])
@pytest.mark.parametrize("hd", [128, 256])
def test_merge_equals_mix(cfg, bits, q_len, hd):
    if hd == 256 and CPU and (cfg is not CFGS[0] or bits != (4, 4)):
        pytest.skip("interpreter is slow: one hd 256 cell on CPU")
    for L0 in (3, 30, 150, 600):             # shorter than S+W, around the window, page crossing
        rig = Rig(cfg, hd = hd, kb = bits[0], vb = bits[1], pages = 8, slots = (2, 0), seed = L0)
        b = 0
        rig.append(b, L0)
        k, v = rig.append(b, q_len)            # the verify rows
        q = rig.rnd(q_len, rig.qh, rig.hd)
        o = decode(rig, b, q, L0)
        ok, err, mx = close(o, ref_attn(rig, b, q, L0, mix = True))
        assert ok, f"{cfg} {bits} q_len {q_len} L0 {L0}: err {err:.3e} vs max {mx:.3e}"


def test_tolerance_discriminates():
    """Negative control: the pool-only result (no ring) must NOT pass the mix check at 4 bit, or the check proves nothing."""
    cfg = CFGS[0]
    rig = Rig(cfg, kb = 4, vb = 4, seed = 21)
    rig.append(0, 300); rig.append(0, 1)
    q = rig.rnd(1, rig.qh, rig.hd)
    ok_on, e_on, mx = close(decode(rig, 0, q, 300), ref_attn(rig, 0, q, 300))
    ok_off, e_off, _ = close(decode(rig, 0, q, 300, ring = False), ref_attn(rig, 0, q, 300))
    assert ok_on and not ok_off, (e_on, e_off, mx)


def test_ring_beats_pool_only():
    """Sanity of the premise: against the exact attention the ring result is closer than the pool-only result."""
    cfg = CFGS[0]
    rig = Rig(cfg, kb = 4, vb = 4, seed = 11)
    rig.append(0, 400)
    rig.append(0, 1)
    # a query aligned with the newest keys, so the window carries real attention mass
    q = (rig.K[0][-3:].float().mean(0).repeat_interleave(rig.qh // rig.kvh, 0) * 1.5)[None].half()
    ke, ve = rig.K[0].float().repeat_interleave(rig.qh // rig.kvh, 1), rig.V[0].float().repeat_interleave(rig.qh // rig.kvh, 1)
    s = torch.einsum("hd,thd->ht", q[0].float(), ke) / math.sqrt(rig.hd)
    exact = torch.einsum("ht,thd->hd", s.softmax(-1), ve)
    e_ring = (decode(rig, 0, q, 400)[0].float() - exact).abs().mean().item()
    e_pool = (decode(rig, 0, q, 400, ring = False)[0].float() - exact).abs().mean().item()
    assert e_ring < e_pool, (e_ring, e_pool)


def test_batch_two_slots_and_lengths():
    """Two sequences with different lengths and permuted slots in one launch."""
    cfg = CFGS[0]
    rig = Rig(cfg, pages = 8, slots = (3, 1), seed = 5)
    rig.append(0, 500); rig.append(1, 90)
    for b in (0, 1):
        rig.append(b, 3)
    q = rig.rnd(2, 3, rig.qh, rig.hd)
    sl = torch.tensor([500, 90], dtype = torch.int32, device = DEV)
    o = paged_attn_triton_decode(
        q.contiguous(), None, None, rig.cl.qk, rig.cl.qv, rig.bt.contiguous(), sl, causal = True,
        qc = (rig.cl.sk, rig.cl.sv, rig.kb, rig.vb), pre_appended_len = 3, n_kv_heads_override = rig.kvh, ring = rig.ctx())
    for b, L in ((0, 500), (1, 90)):
        ok, err, mx = close(o[b], ref_attn(rig, b, q[b], L))
        assert ok, (b, err, mx)


@pytest.mark.parametrize("splits", [None, 1, 3])
def test_split_counts(splits):
    cfg = CFGS[0]
    rig = Rig(cfg, seed = 7)
    rig.append(0, 700); rig.append(0, 2)
    q = rig.rnd(2, rig.qh, rig.hd)
    o = decode(rig, 0, q, 700, num_splits = splits)
    ok, err, mx = close(o, ref_attn(rig, 0, q, 700))
    assert ok, (err, mx)


# -- speculative rollback ---------------------------------------------------------------------------------------------------

def test_rollback_of_draft_rows():
    """D draft rows are verified (written), only some accepted: the next block attends the right window."""
    cfg = CFGS[0]
    rig = Rig(cfg, seed = 3)
    rig.append(0, 300)
    L = 300
    D = 16
    rig.append(0, D)                              # verify block of 16 draft rows (positions 300..315)
    q = rig.rnd(D, rig.qh, rig.hd)
    ok, err, mx = close(decode(rig, 0, q, L), ref_attn(rig, 0, q, L))
    assert ok
    # reject all but 3: sequence continues at 303 with fresh rows (the stale rows 303..315 are overwritten as they land)
    rig.len[0] = L + 3
    rig.K[0] = rig.K[0][: L + 3]; rig.V[0] = rig.V[0][: L + 3]
    for step in (1, 4, 16):
        L2 = rig.len[0]
        rig.append(0, step)
        q = rig.rnd(step, rig.qh, rig.hd)
        ok, err, mx = close(decode(rig, 0, q, L2), ref_attn(rig, 0, q, L2))
        assert ok, (step, err, mx)


def test_rollback_across_ring_wrap():
    """Verify blocks of 16 at every offset around a ring wrap; reject everything each time (len stays)."""
    cfg = CFGS[0]
    rig = Rig(cfg, seed = 4)
    rig.append(0, 100)
    for rep in range(6):
        L = rig.len[0]
        rig.append(0, 16)
        rig.len[0] = L; rig.K[0] = rig.K[0][:L]; rig.V[0] = rig.V[0][:L]      # all rejected
        rig.append(0, 1 + rep)                                                  # real tokens land on the stale rows
        L2 = rig.len[0] - (1 + rep)
        q = rig.rnd(1 + rep, rig.qh, rig.hd)
        ok, err, mx = close(decode(rig, 0, q, L2), ref_attn(rig, 0, q, L2))
        assert ok, (rep, err, mx)


# -- stash / restore ----------------------------------------------------------------------------------------------------------

def test_stash_restore_bitwise():
    cfg = CFGS[0]
    rig = Rig(cfg, slots = (1, 2), seed = 9)
    rig.append(0, 450)
    stashed = rig.st.stash(1)
    pinned = rig.st.stash(1, pinned = True)
    torch.cuda.synchronize()
    assert torch.equal(stashed[0], pinned[0]) and torch.equal(stashed[1], pinned[1])
    rig.st.clear(3)
    rig.st.unstash(3, stashed)
    assert torch.equal(rig.st.ov_k[3], rig.st.ov_k[1]) and torch.equal(rig.st.ov_v[3], rig.st.ov_v[1])
    # disk round trip (prefix_disk uses torch.save of the stash dict)
    import io
    buf = io.BytesIO(); torch.save({"x": stashed}, buf); buf.seek(0)
    back = torch.load(buf)["x"]
    rig.st.clear(0)
    rig.st.unstash(0, back)
    assert torch.equal(rig.st.ov_k[0], rig.st.ov_k[1])
    # the restored slot attends identically to the original
    rig.slots[1] = 0
    q = rig.rnd(1, rig.qh, rig.hd)
    rig.append(0, 1)
    o_new = decode(rig, 0, q, 450)
    ok, err, mx = close(o_new, ref_attn(rig, 0, q, 450))
    assert ok, (err, mx)


def test_stash_async_matches_sync():
    """GDNState.stash(async_) needs every layer state to stash without a host sync: the ring's pinned D2H (pool reuse included)
    reads the same bytes as the blocking stash, and the recurrent cache's finisher turns it into pageable clones."""
    from exllamav3.cache.recurrent import RecurrentCache
    rig = Rig(CFGS[0], slots = (1, 2), seed = 11)
    rig.append(0, 450)
    sync = rig.st.stash(1)
    pool = {}
    a = rig.st.stash_async(1, pool)
    snap = rig.st.snapshot(1)
    rig.st.ov_k[1].zero_()                                   # the live ring moves on after the snapshot
    b = rig.st.stash_from_snapshot(snap, pool)
    ev = torch.cuda.Event(); ev.record(); ev.synchronize()
    for x in (a, b):
        assert all(t.is_pinned() for t in x)
        assert torch.equal(x[0], sync[0]) and torch.equal(x[1], sync[1])
    rc = RecurrentCache.__new__(RecurrentCache)
    rc._pin_pool = pool
    st = {"k": a, "_ev": ev}
    RecurrentCache._finish(rc, st)
    assert not st["k"][0].is_pinned() and torch.equal(st["k"][0], sync[0])
    assert len(pool[(tuple(sync[0].shape), sync[0].dtype)]) == 1       # the pinned buffers went back to the pool
    assert rig.st.stash_async(1) is not None and KVWinLayerState(rig.st.module, 2, 0, 0).stash_async(0) is None   # inert: nothing to copy


def test_ring_layers_do_not_change_path_eligibility(monkeypatch):
    """A declared ring puts attention modules into get_recurrent_layers(); the checkpoint-cadence checks (GDN no-tail
    mid-chunk checkpoints, big tail, wide prefill chunks) must see the GDN layers alone, as without the ring."""
    from exllamav3.model.model import Model
    from exllamav3.generator.job import Job
    from exllamav3.modules.gated_delta_net import GatedDeltaNet
    gdn = object.__new__(GatedDeltaNet)
    gdn.kda = False
    gdn.caps = {"recurrent_cache": True}
    a = _FakeAttn()
    plain = types.SimpleNamespace(caps = {"recurrent_cache": True}, layer_state_cls = object)
    m = type("M", (), {"recurrent_state_cls": object, "__iter__": lambda self: iter([gdn, a, plain]),
                       "_get_recurrent_layers": property(lambda self: Model._get_recurrent_layers.func(self))})()
    assert Model._get_recurrent_layers.func(m) == [gdn, a, plain]
    assert Model._get_checkpoint_layers.func(m) == [gdn, plain]
    mm = types.SimpleNamespace(loaded_tp = False, get_checkpoint_layers = lambda: [gdn])
    fake = types.SimpleNamespace(recurrent_state = object(), embeddings = None, generator = types.SimpleNamespace(model = mm))
    for k, v in (("EXL3_PF_NO_TAIL", "1"), ("EXL3_GDN_FUSE", "1")):
        monkeypatch.setenv(k, v)
    assert Job.gdn_no_tail_enabled(fake) is True


def test_cold_fill_from_pool():
    """A checkpoint without ring blob: fill rebuilds sinks + window from the dequantized pool (quant-level window)."""
    cfg = CFGS[0]
    rig = Rig(cfg, slots = (1, 2), seed = 10)
    rig.append(0, 420)
    rig.st.unstash_missing(2)
    assert rig.st.cold[2] and not rig.st.cold[1]
    rig.st.fill(2, rig.bt[0], 420)
    assert not rig.st.cold[2]
    kd, vd = rig.dequant(0)
    for p in list(range(cfg.S)) + list(range(420 - cfg.W, 420)):
        r = p if p < cfg.S else cfg.S + (p - cfg.S) % cfg.Rc
        assert torch.equal(rig.st.ov_k[2, r].float(), kd[p]) and torch.equal(rig.st.ov_v[2, r].float(), vd[p]), p
    # after the fill, new rows land normally and the window slides over filled + new rows
    rig.slots[0] = 2
    rig.append(0, 1)
    q = rig.rnd(1, rig.qh, rig.hd)
    o = decode(rig, 0, q, 420)
    # reference: window rows 420-W+1..419 are dequantized (filled), row 420 and the sinks exact
    kk, vv = rig.K[0].clone(), rig.V[0].clone()
    keep = torch.zeros(421, dtype = torch.bool, device = DEV); keep[:cfg.S] = True; keep[420] = True
    kk = torch.where(keep[:, None, None], kk.float(), torch.cat([kd, kd[-1:]])); vv = torch.where(keep[:, None, None], vv.float(), torch.cat([vd, vd[-1:]]))
    g = rig.qh // rig.kvh
    s = torch.einsum("hd,thd->ht", q[0].float(), kk.repeat_interleave(g, 1)) / math.sqrt(rig.hd)
    ref = torch.einsum("ht,thd->hd", s.softmax(-1), vv.repeat_interleave(g, 1))
    ok, err, mx = close(o[0], ref)
    assert ok, (err, mx)


# -- prefill staging patch ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("past,q_len", [(0, 300), (700, 300), (90, 40)])
def test_prefill_patch(past, q_len):
    cfg = CFGS[0]
    rig = Rig(cfg, hd = 128, pages = 8, slots = (2, 0), seed = 12)
    if past:
        rig.append(0, past)
    k = rig.rnd(1, q_len, rig.kvh, rig.hd); v = rig.rnd(1, q_len, rig.kvh, rig.hd)
    sl = torch.tensor([past], dtype = torch.int32, device = DEV)
    rig.cl.update_kv_direct(sl, rig.bt[0:1].contiguous(), k, v, q_len)
    ctx = rig.ctx([0])._replace(chunk_k = k, chunk_v = v)
    q = rig.rnd(q_len, rig.qh, rig.hd)
    o = paged_attn_triton_prefill(
        q[None].contiguous(), None, None, rig.cl.qk, rig.cl.qv, rig.bt[0:1].contiguous(), sl, causal = True,
        qc = (rig.cl.sk, rig.cl.sv, rig.kb, rig.vb), pre_appended_len = q_len, n_kv_heads_override = rig.kvh,
        max_kv_len = past, ring = ctx)[0]
    ov_write(rig.ctx([0]), k, v, sl, q_len)
    rig.K[0] = torch.cat([rig.K[0][:past], k[0]]); rig.V[0] = torch.cat([rig.V[0][:past], v[0]]); rig.len[0] = past + q_len
    # reference: chunk rows exact, past window / sinks exact, rest dequantized (a superset of the decode window)
    kd, vd = rig.dequant(0)
    ke, ve = rig.K[0].float(), rig.V[0].float()
    pos = torch.arange(past + q_len, device = DEV)
    exact = (pos < cfg.S) | (pos >= past - cfg.W)
    kk = torch.where(exact[:, None, None], ke, kd); vv = torch.where(exact[:, None, None], ve, vd)
    g = rig.qh // rig.kvh
    kk, vv = kk.repeat_interleave(g, 1), vv.repeat_interleave(g, 1)
    s = torch.einsum("qhd,thd->hqt", q.float(), kk) / math.sqrt(rig.hd)
    mask = pos[None, :] <= (past + torch.arange(q_len, device = DEV))[:, None]
    s = s.masked_fill(~mask[None], -float("inf"))
    ref = torch.einsum("hqt,thd->qhd", s.softmax(-1), vv)
    ok, err, mx = close(o, ref)
    assert ok, (err, mx)
    # and the ring now holds the end of the chunk
    pp = ring_positions(past + q_len, cfg)
    for r in range(cfg.R):
        if pp[r] >= 0:
            assert torch.equal(rig.st.ov_k[2, r], rig.K[0][int(pp[r])])


# -- Cache / GDNState integration (fake attention module, real Cache + GDNState) --------------------------------------------------

class _FakeAttn:
    layer_idx = 0
    num_kv_heads = 2
    head_dim = 128
    logit_softcapping = 0.0

    def __init__(self):
        self.caps = {"kv_cache": True, "recurrent_cache": True}
        self.layer_state_cls = KVWinLayerState
        self.cache_layers, self.recurrent_layers = [], []

    def window_arg(self):
        return -1


class _FakeModel:
    def __init__(self):
        from exllamav3.modules.gated_delta_net import GDNState
        self.config = types.SimpleNamespace(layer_map = None)
        self.recurrent_state_cls = GDNState
        self.loaded_tp = False
        self.cache_weakrefs = {}
        self.attn = _FakeAttn()

    def get_cache_layers(self): return [self.attn]
    def get_recurrent_layers(self): return [self.attn]
    def get_layer_instances(self, idx): return [(idx, 0)]


def _fake_cache(monkeypatch, spec = "4:32"):
    from exllamav3.cache import Cache
    monkeypatch.setenv("EXL3_KVWIN", spec)
    model = _FakeModel()
    cache = Cache(model, 8 * PAGE, layer_type = CacheLayer_quant, max_batch_size = 3, k_bits = 4, v_bits = 4)
    model.attn.cache_layers[0].alloc(torch.device(DEV))
    model.attn.recurrent_layers[0].alloc(torch.device(DEV))
    return cache, model, model.attn.cache_layers[0], model.attn.recurrent_layers[0]


def test_cache_binds_ring_to_recurrent_state_lifecycle(monkeypatch):
    cache, model, cl, st = _fake_cache(monkeypatch)
    assert st.active and cl.kvwin is st and st.cfg == parse_spec("4:32")
    state = cache.get_new_state()
    k = torch.randn(1, 300, 2, 128, device = DEV).half(); v = torch.randn_like(k)
    sl = torch.zeros(1, dtype = torch.int32, device = DEV)
    ov_write(KVWinCtx(st, torch.tensor([state.slot], dtype = torch.int32, device = DEV)), k, v, sl, 300)
    state.position = 300
    state.rewind(3)                                   # ring rewind is a no-op (positional rows); GDN-less cache here
    state.position = 300
    stashed = state.stash()
    assert stashed["checkpoint_size"] == st.get_checkpoint_size() > 0
    slot_a = state.slot
    state.free()
    # restore into a different slot: bitwise
    cache.free_list.rotate(1)
    state2 = cache.new_from_stashed(stashed, 300)
    assert state2.slot != slot_a
    assert torch.equal(st.ov_k[state2.slot], st.ov_k[slot_a]) and torch.equal(st.ov_v[state2.slot], st.ov_v[slot_a])
    assert not st.cold[state2.slot]
    # a checkpoint written with the switch off has no ring entry (or None): cold, then rebuilt from the pool
    for variant in ("missing", "none"):
        s2 = dict(stashed)
        if variant == "missing": del s2[(0, 0)]
        else: s2[(0, 0)] = None
        state2.free()
        state3 = cache.new_from_stashed(s2, 300)
        assert st.cold[state3.slot]
        params = {"recurrent_states": [state3], "recurrent_slots": torch.tensor([state3.slot], dtype = torch.int32),
                  "cache_seqlens": torch.tensor([300], dtype = torch.int32)}
        cl.update_kv_direct(sl, torch.tensor([[1, 4, 6, 0, 2, 3, 5, 7]], dtype = torch.int32, device = DEV)[:, :4].contiguous(), k, v, 300)
        from exllamav3.cache.kvwin import make_ctx, ensure_filled
        ctx = make_ctx(st, params, torch.device(DEV), 1)
        assert ctx is not None
        ensure_filled(ctx, params, torch.tensor([[1, 4, 6, 0]], dtype = torch.int32, device = DEV))
        assert not st.cold[state3.slot]
        assert st.ov_k[state3.slot].any()
        state2 = state3


def test_ring_inert_when_off_or_unsupported(monkeypatch):
    from exllamav3.cache import Cache
    monkeypatch.setenv("EXL3_KVWIN", "0")
    model = _FakeModel()
    cache = Cache(model, 8 * PAGE, layer_type = CacheLayer_quant, max_batch_size = 2, k_bits = 4, v_bits = 4)
    st = model.attn.recurrent_layers[0]
    st.alloc(torch.device(DEV))
    assert not st.active and st.ov_k is None and st.get_checkpoint_size() == 0 and st.storage_size() == 0
    state = cache.get_new_state()
    stashed = state.stash()
    assert stashed[(0, 0)] is None and stashed["checkpoint_size"] == 0
    # fp16 cache layer: inert even with the switch on
    from exllamav3.cache import CacheLayer_fp16
    monkeypatch.setenv("EXL3_KVWIN", "4:32")
    model2 = _FakeModel()
    Cache(model2, 8 * PAGE, layer_type = CacheLayer_fp16, max_batch_size = 2)
    assert model2.attn.recurrent_layers[0].cfg is None


@pytest.mark.parametrize("env, bits, expect", [
    (None, (4, 4), "4:128"), (None, (5, 5), "4:128"), (None, (6, 4), "4:128"), (None, (4, 8), "4:128"),
    (None, (6, 6), None), (None, (8, 8), None),
    ("0", (4, 4), None), ("off", (5, 5), None), ("", (4, 4), None),
    ("2:64", (6, 6), "2:64"), ("2:64", (4, 4), "2:64"),
])
def test_default_on_for_q5_and_below(monkeypatch, env, bits, expect):
    """Optimized Q4: unset switch = 4:128 when k or v bits <= 5, off (no ring, today's paths) at >= 6; explicit values win."""
    from exllamav3.cache import Cache, kvwin
    monkeypatch.setattr(kvwin, "default_platform", lambda: True)
    if env is None: monkeypatch.delenv("EXL3_KVWIN", raising = False)
    else: monkeypatch.setenv("EXL3_KVWIN", env)
    model = _FakeModel()
    Cache(model, 8 * PAGE, layer_type = CacheLayer_quant, max_batch_size = 2, k_bits = bits[0], v_bits = bits[1])
    st = model.attn.recurrent_layers[0]
    assert st.cfg == (parse_spec(expect) if expect else None)


@pytest.mark.parametrize("env, hip, expect", [(None, False, None), (None, True, "4:128"), ("4:128", False, "4:128"), ("0", True, None)])
def test_default_is_rocm_only_until_cuda_is_cleared(monkeypatch, env, hip, expect):
    """CUDA keeps the window off by default (DEFAULT_ON_CUDA = False); an explicit EXL3_KVWIN forces it on any backend."""
    from exllamav3.cache import Cache, kvwin
    monkeypatch.setattr(kvwin, "DEFAULT_ON_CUDA", False)
    monkeypatch.setattr(kvwin.torch.version, "hip", "7.0" if hip else None)
    if env is None: monkeypatch.delenv("EXL3_KVWIN", raising = False)
    else: monkeypatch.setenv("EXL3_KVWIN", env)
    model = _FakeModel()
    Cache(model, 8 * PAGE, layer_type = CacheLayer_quant, max_batch_size = 2, k_bits = 4, v_bits = 4)
    assert model.attn.recurrent_layers[0].cfg == (parse_spec(expect) if expect else None)
    assert kvwin.env_capable() is (hip if env is None else env not in ("0", "off", ""))


@pytest.mark.parametrize("env, capable", [(None, True), ("4:128", True), ("2:64", True), ("0", False), ("off", False), ("", False)])
def test_attention_declares_ring_state_unless_switched_off(monkeypatch, env, capable):
    """Attention modules are built before the cache (and its bits) exist: unless the switch is off they declare the ring state."""
    from exllamav3.cache import kvwin
    from exllamav3.cache.kvwin import env_capable
    monkeypatch.setattr(kvwin, "default_platform", lambda: True)
    if env is None: monkeypatch.delenv("EXL3_KVWIN", raising = False)
    else: monkeypatch.setenv("EXL3_KVWIN", env)
    assert env_capable() is capable


def test_ring_declines_qsa_caches(monkeypatch):
    """The sparse (QSA) decode path does not read the ring: its cache layers must not bind one."""
    from exllamav3.cache.qsa import CacheLayer_qsa_quant
    monkeypatch.setenv("EXL3_KVWIN", "4:32")
    attn = types.SimpleNamespace(num_kv_heads = 2, head_dim = 128, qsa_indexer = types.SimpleNamespace(head_dim = 128, compress_ratio = 4),
                                 window_arg = lambda: -1, logit_softcapping = 0.0)
    cl = CacheLayer_qsa_quant(None, attn, 0, 8 * PAGE, 4, 4)
    st = KVWinLayerState(attn, 2, 0, 0)
    st.bind(cl)
    assert st.cfg is None and getattr(cl, "kvwin", None) is None


def test_plain_attention_model_gets_a_ring_only_state():
    from exllamav3.model.model import Model
    from exllamav3.cache.kvwin import KVWinState
    a = _FakeAttn()
    plain = type("M", (), {"_recurrent_state_cls": None, "_get_recurrent_layers": [a]})()
    assert Model.kvwin_ring_only(plain)
    assert Model.recurrent_state_cls.fget(types.SimpleNamespace(_recurrent_state_cls = None, kvwin_ring_only = lambda: True)) is KVWinState
    none = type("M", (), {"_recurrent_state_cls": None, "_get_recurrent_layers": []})()
    assert not Model.kvwin_ring_only(none)
    hybrid = type("M", (), {"_recurrent_state_cls": object, "_get_recurrent_layers": [a]})()
    assert not Model.kvwin_ring_only(hybrid)


def test_ring_only_state_lifecycle_cold_resume_and_rewind(monkeypatch):
    from exllamav3.cache.kvwin import KVWinState, make_ctx, ensure_filled
    cache, model, cl, st = _fake_cache(monkeypatch)
    cache.recurrent_state_cls = KVWinState
    state = cache.get_new_state()
    assert isinstance(state, KVWinState) and not st.cold[state.slot]
    k = torch.randn(1, 300, 2, 128, device = DEV).half(); v = torch.randn_like(k)
    sl = torch.zeros(1, dtype = torch.int32, device = DEV)
    bt = torch.tensor([[1, 4, 6, 0, 2, 3, 5, 7]], dtype = torch.int32, device = DEV)
    cl.update_kv_direct(sl, bt[:, :5].contiguous(), k, v, 300)
    ov_write(KVWinCtx(st, torch.tensor([state.slot], dtype = torch.int32, device = DEV)), k, v, sl, 300)
    state.position = 300
    state.rewind(3)
    assert state.position == 297 and state.rollback_capacity() == kvwin.MAX_DRAFT
    state.free()
    # a job resumed from cached pages: cold ring, rebuilt from the pool at the first attention
    state2 = cache.new_from_stashed({"position": 256}, 256)
    assert state2.position == 256 and st.cold[state2.slot]
    params = {"recurrent_states": [state2], "recurrent_slots": torch.tensor([state2.slot], dtype = torch.int32),
              "cache_seqlens": torch.tensor([256], dtype = torch.int32)}
    ctx = make_ctx(st, params, torch.device(DEV), 1)
    ensure_filled(ctx, params, bt)
    assert not st.cold[state2.slot] and st.ov_k[state2.slot].any()
    # a new state clears its ring
    state2.free()
    state3 = cache.get_new_state()
    assert not st.ov_k[state3.slot].any() and not st.cold[state3.slot]


def test_attention_kvwin_ctx_entry(monkeypatch):
    """Attention.kvwin_ctx (the hook of decode_flash_attn / decode_flash_attn_shared) on a bound cache."""
    from exllamav3.modules.attn import Attention
    cache, model, cl, st = _fake_cache(monkeypatch)
    state = cache.get_new_state()
    me = types.SimpleNamespace(caps = {"recurrent_cache": True}, layer_idx = 0, device = torch.device(DEV))
    params = {"recurrent_states": [state], "recurrent_slots": torch.tensor([state.slot], dtype = torch.int32)}
    bt = torch.zeros((1, 4), dtype = torch.int32, device = DEV)
    ctx = Attention.kvwin_ctx(me, cache, params, 1, bt)
    assert ctx is not None and int(ctx.slots[0]) == state.slot
    assert Attention.kvwin_ctx(types.SimpleNamespace(caps = {}), cache, params, 1, bt) is None      # off: untouched
    assert Attention.kvwin_ctx(me, cache, {}, 1, bt) is None                                          # no job states
    assert Attention.kvwin_ctx(me, None, params, 1, bt) is None                                       # no Cache


def test_midchunk_checkpoint_keeps_working_with_a_ring():
    """EXL3_MIDCHUNK_CKPT stashes skip the ring (cold on restore) instead of being dropped."""
    from exllamav3.generator.job import Job
    ring = types.SimpleNamespace(kvwin = True)
    gdn = types.SimpleNamespace()
    puts = []
    page = types.SimpleNamespace(kv_position = PAGE, phash = b"h")
    fake = types.SimpleNamespace(
        sequences = [types.SimpleNamespace(allocated_pages = [page])],
        recurrent_state = types.SimpleNamespace(checkpoint_size = 7),
        generator = types.SimpleNamespace(
            recurrent_cache = types.SimpleNamespace(put = lambda key, st, stashed = None: puts.append((key, stashed)), metrics = {}),
            cache = types.SimpleNamespace(get_all_recurrent_layers = lambda: {(0, 0): ring, (1, 0): gdn})))
    t = (torch.zeros(2), torch.zeros(3))
    assert Job.stash_midchunk(fake, {PAGE: {(1, 0): t}}, 0) == 1
    (key, stashed), = puts
    assert key == b"h" and stashed[(0, 0)] is None and stashed[(1, 0)][0].shape == (2,) and stashed["position"] == PAGE
    assert Job.stash_midchunk(fake, {PAGE: {}}, 0) == 0                       # still skipped when a GDN layer is missing


def test_per_sequence_params_keep_their_ring_state():
    """The row-invariant multi-job step runs attention per sequence on split params: each keeps its own state / slot."""
    from exllamav3.modules.attn import split_params_by_sequence
    from exllamav3.cache.kvwin import make_ctx
    st = KVWinLayerState(_FakeAttn(), 4, 0, 0)
    st.cfg = parse_spec("4:32"); st.alloc(torch.device(DEV))
    rs = [types.SimpleNamespace(slot = 3), types.SimpleNamespace(slot = 1)]
    params = {"block_table": torch.zeros((2, 4), dtype = torch.int32), "cache_seqlens": torch.tensor([5, 9], dtype = torch.int32),
              "recurrent_states": rs, "recurrent_slots": torch.tensor([3, 1], dtype = torch.int32)}
    parts = split_params_by_sequence(params, 2)
    for b in range(2):
        ctx = make_ctx(st, parts[b], torch.device(DEV), 1)
        assert ctx is not None and int(ctx.slots[0]) == rs[b].slot
    with pytest.raises(RuntimeError):                 # a state / batch mismatch must not silently skip the ring
        make_ctx(st, params, torch.device(DEV), 1)
    assert make_ctx(st, {}, torch.device(DEV), 2) is None


def test_off_is_bitwise_today():
    """EXL3_KVWIN unset: attn_dispatch outputs (prefill / decode / verify rows, 4/6/8 bit) equal the tree before KVWIN bit for bit.
    KVWIN_BASE_TREE = checkout of the parent commit (default: the sibling worktree wt_empty27)."""
    import subprocess, tempfile
    here = os.path.dirname(os.path.abspath(__file__))
    new_tree = os.path.dirname(here)
    base = os.environ.get("KVWIN_BASE_TREE", os.path.join(new_tree, "..", "wt_empty27"))
    if not os.path.isdir(os.path.join(base, "exllamav3")):
        pytest.skip("no baseline tree (set KVWIN_BASE_TREE)")
    ext_dir = os.environ.get("KVWIN_EXT_DIR", "")
    outs = []
    with tempfile.TemporaryDirectory() as d:
        for name, tree in (("base", base), ("new", new_tree)):
            o = os.path.join(d, name + ".pt")
            env = {k: v for k, v in os.environ.items() if k != "EXL3_KVWIN"}
            subprocess.run([sys.executable, "-I", os.path.join(here, "kvwin_off_scenario.py"), tree, o] + ([ext_dir] if ext_dir else []),
                           check = True, env = env, stdout = subprocess.DEVNULL)
            outs.append(torch.load(o))
    for bits in outs[0]:
        for a, b in zip(outs[0][bits], outs[1][bits]):
            assert torch.equal(a, b), bits


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-x", "-q"] + sys.argv[1:]))


def test_whole_forward_graph_declines_only_for_a_cold_ring(monkeypatch):
    """A bound ring no longer blocks the whole-forward graph: only a step whose job has a cold ring (host-side fill) runs eager."""
    from exllamav3.model.cuda_graph import CudaGraphRunner
    cache, model, cl, st = _fake_cache(monkeypatch)
    state = cache.get_new_state()
    runner = object.__new__(CudaGraphRunner)
    runner.has_gdn, runner.has_conv, runner.has_mamba, runner.has_swa = True, False, False, False
    ids = torch.zeros((1, 1), dtype = torch.long)
    params = {"block_table": torch.zeros((1, 4), dtype = torch.int32), "cache_seqlens": torch.full((1,), 300, dtype = torch.int32),
              "cache": cache, "recurrent_states": [state], "recurrent_slots": torch.tensor([state.slot], dtype = torch.int32)}
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    assert kvwin.ANY_ACTIVE and not kvwin.any_cold(params)
    assert runner._decline(ids, params) is None
    st.cold[state.slot] = True
    assert kvwin.any_cold(params) and runner._decline(ids, params) == "kvwin_cold"
    st.cold[state.slot] = False
    assert runner._decline(ids, params) is None
    assert not kvwin.any_cold({}) and not kvwin.any_cold({"recurrent_states": [state]})
