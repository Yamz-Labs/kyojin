"""
DS4.1 SERVING model (exllamav3.dsv41.serve_model.DSV41Serving) on CPU, tiny random model with the real structure
(sliding-window ring that wraps, compressors at ratio 2 and 1, shared kv pools / indexers, two-level candidate selection,
MoE with a shared expert, Engram, dspark stages).

  * prefill + 16 decode steps with cache == the reference graph (DSV41Model), fp32 activations: bit for bit on the tiny model;
    fp16 activations: within fp16 tolerance, same argmax
  * chunked prefill (every chunk size, aligned or not, groups straddling chunk borders) == one-shot prefill, then decode
  * block verify / rollback / drafter feed / dspark greedy equality: the whole dswire suite (test_dsv41_spec.py) re-run on
    the serving model (only the test that expects a plain multi-token forward to be refused is replaced: serving accepts chunks)
  * paged cache: pools grow in pages, fork() is a deep copy, planner bytes
  * the pack loader: synthetic pack written in the converter layout -> framework Config / Model -> PackSource -> serving model
    equals the dense reference
  * MoE backends: LoopMoE == reference, GroupedMoE glue (decode / prefill / fallback dispatch, mixed-K bookkeeping) with a fake store
"""
import importlib.util, os, sys
import pytest
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "tools", "ds41"))
from hardware import require
require("ext")
import harness
from harness import FakeTok, make_state_dict, make_engram_tables, tiny_config
from exllamav3.dsv41.config import DSV41Config
from exllamav3.dsv41.engram import build_compressed_token_map, EngramLayout, MemoryEngramStore
from exllamav3.dsv41.weights import DictWeightSource
from exllamav3.dsv41.experts import SourceExpertProvider
from exllamav3.dsv41.model import DSV41Model
from exllamav3.dsv41.serve_model import DSV41Serving
from exllamav3.dsv41.serve_cache import DSV41Cache, PagedPool, cache_bytes

torch.set_num_threads(int(os.environ.get("DSV41_THREADS", "4")))


# ------------------------------------------------------------------------------------------------ fixtures
def make_world(seed=0, fp16_weights=True, **over):
    cfg = tiny_config(index_n_heads=16, dspark_block_size=5, dspark_noise_token_id=511, **over)
    tok = FakeTok(cfg.vocab_size)
    tmap, nv = build_compressed_token_map(tok)
    cfg.engram_compressed_vocab_size = nv
    lay = EngramLayout.build(cfg.engram_layer_ids, None, cfg.engram_max_ngram_size, cfg.engram_vocab_size,
                             cfg.engram_n_heads, cfg.engram_head_dim)
    cfg.engram_num_embeddings = list(lay.num_embeddings)
    tables = make_engram_tables(cfg, lay.num_embeddings, seed + 1)
    sd = make_state_dict(cfg, seed, fp8=False, fp4=False)
    if fp16_weights:                       # what an fp16 / EXL3 server holds: weights rounded to fp16 once
        sd = {k: (v.half().float() if (v.dtype in (torch.bfloat16, torch.float32) and v.dim() >= 2) else v)
              for k, v in sd.items()}
    return cfg, sd, tables, tmap


def build_ref(cfg, sd, tables, tmap):
    src = DictWeightSource(sd, cfg.num_hidden_layers)
    n = cfg.num_hidden_layers + cfg.n_mtp_layers
    prov = SourceExpertProvider(src, {l: cfg.moe_counts(l)[0] for l in range(n)})
    return DSV41Model(cfg, src, prov, MemoryEngramStore([t[0] for t in tables], [t[1] for t in tables]), token_map=tmap)


def build_serving(cfg, sd, tables, tmap, dtype=torch.float32, **kw):
    src = DictWeightSource(sd, cfg.num_hidden_layers)
    store = MemoryEngramStore([t[0] for t in tables], [t[1] for t in tables])
    return DSV41Serving(cfg, src, None, store, token_map=tmap, dtype=dtype, **kw)


@pytest.fixture(scope="module")
def world():
    cfg, sd, tables, tmap = make_world(2)
    return cfg, sd, tables, tmap, build_ref(cfg, sd, tables, tmap), build_serving(cfg, sd, tables, tmap)


def ids_of(cfg, n, seed=5):
    return torch.randint(0, cfg.vocab_size - 1, (1, n), generator=torch.Generator().manual_seed(seed))


# ------------------------------------------------------------------------------------------------ equality
def test_prefill_and_16_decode_steps_equal_the_reference(world):
    cfg, sd, tables, tmap, ref, srv = world
    L = 43                                                     # > 2 windows: the ring wraps, groups of both ratios close
    ids = ids_of(cfg, L + 16)
    a, b = ref.new_state(1, 128), srv.new_state(1, 128)
    lr, hr = ref.forward(ids[:, :L], 0, a)
    ls, hs = srv.forward(ids[:, :L], 0, b)
    assert lr.abs().max() > 1.0
    assert (lr - ls).abs().max().item() < 1e-5 and (hr - hs).abs().max().item() < 1e-5
    for i in range(16):
        lr, hr = ref.forward(ids[:, L + i:L + i + 1], L + i, a)
        ls, hs = srv.forward(ids[:, L + i:L + i + 1], L + i, b)
        assert (lr - ls).abs().max().item() < 1e-5, i
        assert (hr - hs).abs().max().item() < 1e-5, i
        assert int(lr.argmax()) == int(ls.argmax())


@pytest.mark.parametrize("chunk", [1, 3, 7, 8, 13, 16, 20, 43])
def test_chunked_prefill_equals_one_shot(world, chunk):
    cfg, sd, tables, tmap, ref, srv = world
    L = 43
    ids = ids_of(cfg, L + 6)
    a, b = ref.new_state(1, 128), srv.new_state(1, 128)
    lr, hr = ref.forward(ids[:, :L], 0, a)
    ls, hs = srv.prefill(ids[:, :L], b, 0, chunk=chunk, keep_hidden=L)
    # a last-bit GEMM-shape difference can flip an fp8 / fp4 grid cell of one row: bound relative to the scale
    assert (lr - ls).abs().max().item() < 1e-4 * lr.abs().max().item(), chunk
    assert (hr - hs).abs().max().item() < 3e-3 * hr.abs().max().item(), chunk
    for i in range(6):                                          # the state left behind decodes like the reference's
        lr, _ = ref.forward(ids[:, L + i:L + i + 1], L + i, a)
        ls, _ = srv.forward(ids[:, L + i:L + i + 1], L + i, b)
        assert (lr - ls).abs().max().item() < 1e-4 * lr.abs().max().item(), (chunk, i)
    assert srv.new_state(1, 128).length == 0 and b.length == L + 6


def test_forward_chunks_a_long_prompt_and_keeps_the_window_rows(world):
    cfg, sd, tables, tmap, ref, srv = world
    L = 43
    ids = ids_of(cfg, L)
    srv.prefill_chunk = 16
    try:
        lr, hr = ref.forward(ids, 0, ref.new_state(1, 128))
        ls, hs = srv.forward(ids, 0, srv.new_state(1, 128))      # 3 chunks
    finally:
        srv.prefill_chunk = 2048
    assert hs.shape[1] == cfg.sliding_window                     # only what the drafter's rings can use
    assert (lr - ls).abs().max().item() < 1e-4 * lr.abs().max().item()
    assert (hr[:, -hs.shape[1]:] - hs).abs().max().item() < 3e-3 * hr.abs().max().item()


def test_fp16_activations_within_tolerance(world):
    # A tiny random model amplifies fp16 noise (the reference graph itself run in fp16 is ~0.15 off its fp32 self),
    # so the bar is relative: the serving model in fp16 must be no worse than the reference graph in fp16.
    cfg, sd, tables, tmap, ref, _ = world
    src = DictWeightSource(sd, cfg.num_hidden_layers)
    n = cfg.num_hidden_layers + cfg.n_mtp_layers
    prov = SourceExpertProvider(src, {l: cfg.moe_counts(l)[0] for l in range(n)})
    r16 = DSV41Model(cfg, src, prov, MemoryEngramStore([t[0] for t in tables], [t[1] for t in tables]),
                     token_map=tmap, dtype=torch.float16)
    s16 = build_serving(cfg, sd, tables, tmap, dtype=torch.float16)
    L = 41
    ids = ids_of(cfg, L + 12, seed=8)
    a, b, c = ref.new_state(1, 128), s16.new_state(1, 128), r16.new_state(1, 128)
    lr, _ = ref.forward(ids[:, :L], 0, a)
    ls, _ = s16.prefill(ids[:, :L], b, 0, chunk=16)
    lh, _ = r16.forward(ids[:, :L], 0, c)
    rel = lambda x: (lr - x.float()).abs().max().item() / lr.abs().max().item()
    ws, wh = rel(ls), rel(lh)
    for i in range(12):
        lr, _ = ref.forward(ids[:, L + i:L + i + 1], L + i, a)
        ls, _ = s16.forward(ids[:, L + i:L + i + 1], L + i, b)
        lh, _ = r16.forward(ids[:, L + i:L + i + 1], L + i, c)
        ws, wh = max(ws, rel(ls)), max(wh, rel(lh))
        assert torch.isfinite(ls).all()
    print(f"fp16 vs fp32 reference: serving {ws:.3e}, reference graph in fp16 {wh:.3e}")
    assert ws < 2 * wh + 0.05
    assert b.win[0].dtype == torch.float16 and b.comp_kv[2].dtype == torch.float16


def test_generate_is_the_reference_greedy_loop(world):
    cfg, sd, tables, tmap, ref, srv = world
    prompt = ids_of(cfg, 21, seed=3)[0].tolist()
    st = ref.new_state(1, 128)
    logits, _ = ref.forward(torch.tensor([prompt]), 0, st)
    want = [int(logits[0].argmax())]
    while len(want) < 24:
        logits, _ = ref.forward(torch.tensor([[want[-1]]]), len(prompt) + len(want) - 1, st)
        want.append(int(logits[0].argmax()))
    assert srv.generate(prompt, 24, max_seq=128) == want


# ------------------------------------------------------------------------------------------------ cache
def test_cache_pages_grow_and_are_bounded(world):
    cfg, sd, tables, tmap, ref, srv = world
    st = DSV41Cache(srv, 1, 4096, page=16)
    assert st.comp_kv[2].shape[1] == 0 and st.nbytes() < 200_000
    srv.prefill(ids_of(cfg, 70), st, 0, chunk=32)
    assert st.comp_kv[2].shape[1] == 48 and st.comp_kv[4].shape[1] == 80       # ceil(70/2)=35 -> 48 rows, 70 -> 80 rows
    assert st.idx_k[2].shape[1] == 48 and set(st.comp_kv) == {2, 4}            # pools only for the kv source layers
    assert st.length == 70
    with pytest.raises(ValueError):
        st.ensure(5000)
    with pytest.raises(ValueError):
        srv._forward(ids_of(cfg, 8), 4090, st, None, "none")


def test_fork_is_a_deep_copy(world):
    cfg, sd, tables, tmap, ref, srv = world
    ids = ids_of(cfg, 40)
    st = srv.new_state(1, 128)
    srv.prefill(ids[:, :30], st, 0)
    twin = st.fork()
    la, _ = srv.forward(ids[:, 30:31], 30, st)
    lb, _ = srv.forward(ids[:, 30:31], 30, twin)
    assert torch.equal(la, lb)
    srv.forward(ids[:, 31:37], 31, st)                           # the original moves on, the fork is untouched
    lc, _ = srv.forward(ids[:, 31:32], 31, twin)
    ref_st = srv.new_state(1, 128)
    srv.prefill(ids[:, :31], ref_st, 0)
    ld, _ = srv.forward(ids[:, 31:32], 31, ref_st)
    assert (lc - ld).abs().max().item() < 1e-5


def test_cache_bytes_planner_matches_the_allocation(world):
    cfg, sd, tables, tmap, ref, srv = world
    st = DSV41Cache(srv, 1, 256, page=256)
    st.ensure(256)
    want = cache_bytes(cfg, 256, itemsize=4)
    assert abs(st.nbytes() - want) <= 0.02 * want


def test_cache_at_real_scale_is_small():
    cfg = DSV41Config()
    gib = cache_bytes(cfg, 131072) / 2 ** 30
    assert 0.3 < gib < 0.6, gib                                  # recurrent-free: ~0.4 GB at 128K, not tens of GB
    assert cache_bytes(cfg, 1024) < 40 << 20


def test_paged_pool_growth_keeps_rows():
    p = PagedPool(1, 4, torch.float32, "cpu", max_rows=100, page=8)
    p.ensure(5)[:, :5] = torch.arange(5.0).view(1, 5, 1)
    t = p.ensure(30)
    assert t.shape[1] == 32 and torch.equal(t[0, :5, 0], torch.arange(5.0)) and t[0, 5:].abs().sum() == 0
    assert p.ensure(1000).shape[1] == 100


# ------------------------------------------------------------------------------------------------ dswire suite on the serving model
def _load_spec_module():
    path = os.path.join(HERE, "test_dsv41_spec.py")
    spec = importlib.util.spec_from_file_location("dsv41_spec_on_serving", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def build_mine(cfg, sd, tables, tmap, dtype=torch.float32, device="cpu"):
        return build_serving(cfg, sd, tables, tmap, dtype=dtype, device=device)
    mod.build_mine = build_mine
    return mod


_spec = _load_spec_module()
for _name, _obj in vars(_spec).items():
    if _name.startswith("test_") and _name not in ("test_block_rejects_prefill_and_oversize",):
        globals()[_name + "_on_serving"] = _obj
globals()["tiny"] = _spec.tiny                                   # the module-scoped fixture the copied tests ask for


def test_block_verify_rejects_prefill_and_oversize_on_serving():
    cfg, mine = _spec.build(0)
    st = mine.new_state(1, 256)
    ids = torch.tensor([_spec.prompt_ids(cfg, 1, 40)])
    mine.forward(ids[:, :20], 0, st)
    with pytest.raises(ValueError):
        mine.forward_block(ids[:, :3], 0, st)
    with pytest.raises(ValueError):
        st.begin_block(20, cfg.sliding_window + 1)
    # unlike the reference graph, serving takes a multi-token chunk at any position (chunked prefill)
    lg, hid = mine.forward(ids[:, 20:23], 20, st)
    assert lg.shape == (1, cfg.vocab_size) and hid.shape[:2] == (1, 3)


# ------------------------------------------------------------------------------------------------ pack loader
@pytest.fixture(scope="module")
def pack(tmp_path_factory):
    from exllamav3.dsv41.synth_pack import write_pack, synth_config
    d = str(tmp_path_factory.mktemp("synth_pack"))
    cfg = synth_config()
    cfg, sd, tables, tmap = write_pack(d, cfg, seed=2)
    return d, cfg, sd, tables, tmap


def dense_reference_of_pack(cfg, sd, tables, tmap):
    sd2 = dict(sd)
    for l in range(cfg.num_hidden_layers + cfg.n_mtp_layers):
        p = f"layers.{l}" if l < cfg.num_hidden_layers else f"mtp.{l - cfg.num_hidden_layers}"
        sd2[f"{p}.attn.wo_a.weight"] = torch.cat([sd2.pop(f"{p}.attn.wo_a.slice.{g}.weight")
                                                  for g in range(cfg.o_groups)], 0)
    sd2 = {k: (v.float() if v.dtype == torch.float16 else v) for k, v in sd2.items()}
    src = DictWeightSource(sd2, cfg.num_hidden_layers)
    prov = SourceExpertProvider(src, {l: cfg.moe_counts(l)[0] for l in range(cfg.num_hidden_layers + cfg.n_mtp_layers)})
    store = MemoryEngramStore([t[0] for t in tables], [t[1] for t in tables], fmt="mxfp4")
    return DSV41Model(cfg, src, prov, store, token_map=tmap)


def test_pack_loader_equals_the_dense_reference(pack):
    from exllamav3.dsv41.pack import load
    d, cfg, sd, tables, tmap = pack
    m = load(d, device="cpu", dtype=torch.float32)
    assert m.mtp is not None and m.cfg.n_mtp_layers == 2 and m.cfg.num_hidden_layers == 8
    ref = dense_reference_of_pack(cfg, sd, tables, tmap)
    L = 30
    ids = ids_of(cfg, L + 8)
    a, b = ref.new_state(1, 64), m.new_state(1, 64)
    lr, hr = ref.forward(ids[:, :L], 0, a)
    ls, hs = m.forward(ids[:, :L], 0, b)
    assert (lr - ls).abs().max().item() < 1e-5 and (hr - hs).abs().max().item() < 1e-5
    for i in range(8):
        lr, _ = ref.forward(ids[:, L + i:L + i + 1], L + i, a)
        ls, _ = m.forward(ids[:, L + i:L + i + 1], L + i, b)
        assert (lr - ls).abs().max().item() < 1e-5 and int(lr.argmax()) == int(ls.argmax())
    # the dspark heads of the pack run too (block verify + drafter on the loaded stages)
    st = m.new_state(1, 64)
    _, hid = m.forward(ids[:, :L], 0, st)
    st.begin_block(L, 3)
    lg, _ = m.forward_block(ids[:, L:L + 3], L, st)
    out, logits, conf = m.mtp.forward_spec(ids[:, L + 2], hid[:, -1:], L + 2, st)
    assert lg.shape == (1, 3, cfg.vocab_size) and out.shape[1] == cfg.dspark_block_size + 1


def test_pack_loader_layer_subset_and_no_mtp(pack):
    from exllamav3.dsv41.pack import load
    d, cfg, sd, tables, tmap = pack
    m = load(d, device="cpu", dtype=torch.float32, layers=3, mtp=False)
    assert m.cfg.num_hidden_layers == 3 and m.mtp is None and set(m.lw) == {0, 1, 2}
    logits, hid = m.forward(ids_of(cfg, 12), 0, m.new_state(1, 32))
    assert logits.shape == (1, cfg.vocab_size) and hid is None and torch.isfinite(logits).all()


def test_pack_loader_fp16_activations(pack):
    from exllamav3.dsv41.pack import load
    d, cfg, sd, tables, tmap = pack
    m = load(d, device="cpu", dtype=torch.float16, mtp=False)
    ref = dense_reference_of_pack(cfg, sd, tables, tmap)
    ids = ids_of(cfg, 25)
    lr, _ = ref.forward(ids, 0, ref.new_state(1, 64))
    ls, _ = m.forward(ids, 0, m.new_state(1, 64))
    assert torch.isfinite(ls).all()
    assert (lr - ls).abs().max().item() / lr.abs().max().item() < 0.5      # tiny random model: fp16 noise is amplified


# ------------------------------------------------------------------------------------------------ MoE backends
class _FakeStore:
    """Stands in for moe_mixedk.MixedKStore: computes the routed sum from dense weights, records how it was called."""

    def __init__(self, ref, l):
        self.ref, self.l, self.calls = ref, l, []
        self.num_groups, self.decode_ok, self.wmma_ok = 2, True, True

    def _sum(self, y, sel, w):
        from exllamav3.dsv41.serve_moe import LoopMoE
        return LoopMoE(self.ref.provider, self.ref.cfg).run(self.l, y, w.float(), sel, self.ref.cfg.moe_counts(self.l)[0])

    def decode(self, ext, y, sel, w, buffers, outputs, act_limit, dedup):
        self.calls.append(("decode", y.shape[0], w.dtype, act_limit, dedup))
        out = outputs[0][:y.shape[0]]
        out.copy_(self._sum(y, sel, w))
        return out

    def prefill(self, ext, y, sel, rw, workspace, scatter_count, act_limit, hidden, interm, sh_add=None, chunk=0):
        self.calls.append(("prefill", y.shape[0], rw.dtype, act_limit, chunk))
        return self._sum(y, sel, rw)


def test_grouped_moe_dispatch_and_fallback(world, monkeypatch):
    from exllamav3.dsv41 import serve_moe
    cfg, sd, tables, tmap, ref, srv = world
    l = 3

    class G(serve_moe.GroupedMoE):                                       # bypass the GPU buffers / ext of the real constructor
        def __init__(self):
            self.dcfg, self.layer, self.fallback = cfg, l, srv.loop_moe
            self.store = _FakeStore(srv, l)
            self.device, self.H, self.I = torch.device('cpu'), cfg.hidden_size, 128
            self.max_rows, self.decode_ok, self.wmma_ok, self.prefill_min, self.align_chunk = 16, True, True, 32, 4096
            self.buffers = type("B", (), {})()
            self.outputs = [torch.zeros(16, cfg.hidden_size, dtype=torch.float32)]
            self.calls = {"decode": 0, "prefill": 0, "fallback": 0}
    g = G()
    import sys, types
    fake_ext = types.ModuleType("exllamav3.ext")
    fake_ext.exllamav3_ext = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "exllamav3.ext", fake_ext)
    import exllamav3.modules.block_sparse_mlp as bsm
    monkeypatch.setattr(bsm, "_hip_wmma_workspace", lambda *a: None, raising=False)
    monkeypatch.setattr(bsm, "_scatter_expert_count", lambda *a: None, raising=False)
    w = srv.lw[l]
    # 17-31 rows run the decode kernel in slices of max_rows (20 = 16 + 4): the loop fallback gave wrong logits there
    for rows, kind in ((1, "decode"), (5, "decode"), (20, "decode"), (40, "prefill")):
        x = torch.randn(rows, cfg.hidden_size)
        wts, idx = srv.router(x, w.gate_w, w.gate_b, w.topk)
        want = srv.loop_moe.run(l, x, wts, idx, w.n_exp)
        got = g.run(l, x, wts, idx, w.n_exp)
        assert torch.allclose(got, want, rtol=3e-3, atol=3e-3), (rows, kind)   # decode runs on fp16 router weights
        assert g.calls[kind] > 0, (rows, kind, g.calls)
    kinds = [c[0] for c in g.store.calls]
    assert kinds == ["decode", "decode", "decode", "decode", "prefill"]
    assert g.store.calls[0][2] == torch.float16 and g.store.calls[0][3] == cfg.swiglu_limit    # half weights, DS4.1 clamp
    assert g.store.calls[4][2] == torch.float32
    g.decode_ok = False                                                  # a layer the decode kernel cannot take: loop fallback
    x = torch.randn(20, cfg.hidden_size)
    wts, idx = srv.router(x, w.gate_w, w.gate_b, w.topk)
    assert torch.allclose(g.run(l, x, wts, idx, w.n_exp), srv.loop_moe.run(l, x, wts, idx, w.n_exp), rtol=3e-3, atol=3e-3)
    assert g.calls["fallback"] == 1


def test_moe_backend_is_used_per_layer(world):
    cfg, sd, tables, tmap, ref, srv = world

    class Spy:
        name = "spy"

        def __init__(self, inner):
            self.inner, self.n = inner, 0

        def run(self, l, x, wts, idx, n_exp):
            self.n += 1
            return self.inner.run(l, x, wts, idx, n_exp)
    spy = Spy(srv.loop_moe)
    srv.moe_backends[2] = spy
    try:
        srv.forward(ids_of(cfg, 10), 0, srv.new_state(1, 32))
    finally:
        srv.moe_backends.pop(2)
    assert spy.n == 1


def test_router_dispatch_matches_ops_route(world):
    from exllamav3.dsv41 import ops
    cfg, sd, tables, tmap, ref, srv = world
    w = srv.lw[2]
    x = torch.randn(7, cfg.hidden_size)
    a = srv.router(x, w.gate_w, w.gate_b, w.topk)
    b = ops.route(x, w.gate_w, w.gate_b, w.topk, cfg.scoring_func, cfg.norm_topk_prob, cfg.routed_scaling_factor, cfg.gate_temp)
    assert torch.equal(a[1], b[1]) and torch.equal(a[0], b[0])


def test_engram_row_blocks_equal_the_whole_chunk(world):
    """Big-chunk memory: the engram injection runs on row blocks (in place); the result must not change, and the chunked
    prefill keeps only the tail of the target-layer hidden (hid_rows)."""
    cfg, sd, tables, tmap, ref, srv = world
    ids = ids_of(cfg, 41)
    l0, h0 = srv.forward(ids, 0, srv.new_state(1, 128))
    old = srv.pf_eng_rows
    srv.pf_eng_rows = 8        # also the block size of the stream fill
    try:
        l1, h1 = srv.forward(ids, 0, srv.new_state(1, 128))
        l2, h2 = srv._forward(ids, 0, srv.new_state(1, 128), None, "last", hid_rows=5)
    finally:
        srv.pf_eng_rows = old
    tol = 1e-5 * l0.abs().max().item()      # CPU fp32 matmuls may differ in the last bit between row counts
    assert (l0 - l1).abs().max().item() <= tol
    assert h2.shape[1] == 5 and (h0[:, -5:] - h2).abs().max().item() <= 1e-5 * h0.abs().max().item() and (l0 - l2).abs().max().item() <= tol
    assert DSV41Serving._row_blocks(100, 32) == [(0, 25), (25, 50), (50, 75), (75, 100)]
    assert DSV41Serving._row_blocks(10, 32) == [(0, 10)]
