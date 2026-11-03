"""
Gate 2 (CPU): weight loaders on tiny synthetic files, in both layouts.
  official : HF safetensors shards (fp8 dense + fp4 experts + e8m0 scales, int8-packed experts), read with
             header+pread (no mmap), Engram rows with O_DIRECT
  GGUF     : antirez Q2 layout (IQ2_XXS gate/up, Q2_K down, Q8_0 dense, F16, F32, I8 Engram rows), decoders
             checked bitwise against gguf-py on random blocks, whole model end to end vs the official reference
"""
import fcntl, json, os, sys
import numpy as np
import pytest
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hardware import require
require("ext")
from harness import *
from gguf_writer import *
from test_dsv41_e2e import setup, build_mine, install_hooks, cmp_taps, TOL
from exllamav3.dsv41 import gguf as G
from exllamav3.dsv41.engram import (DirectRowReader, FileEngramStore, MemoryEngramStore, EngramLayout, ROW_BYTES,
                                    compute_multipliers)
from exllamav3.dsv41.safetensors_io import SafetensorsWeightSource, SafetensorsIndex
from exllamav3.dsv41.experts import SourceExpertProvider, ExpertProvider
from exllamav3.dsv41.model import DSV41Model

SCRATCH = os.environ.get("DSV41_SCRATCH", "./ds41_work/ds41port/scratch")
os.makedirs(SCRATCH, exist_ok=True)


@pytest.fixture
def work(request):
    d = os.path.join(SCRATCH, f"t_{request.node.name.replace('[', '_').replace(']', '')}")
    os.makedirs(d, exist_ok=True)
    yield d
    import shutil
    shutil.rmtree(d, ignore_errors=True)


def mapped_files():
    with open("/proc/self/maps") as f:
        return f.read()


# ----------------------------------------------------------------------------- official safetensors
def write_official(cfg, sd, tables, d, shards=3):
    from safetensors.torch import save_file
    sd = dict(sd)
    for i, lid in enumerate(cfg.engram_layer_ids):
        sd[f"layers.{lid}.engram.embed.weight"] = tables[i][0]
        sd[f"layers.{lid}.engram.embed.scale"] = tables[i][1]
    names = sorted(sd)
    wm = {}
    for s in range(shards):
        part = {n: sd[n].contiguous() for n in names[s::shards]}
        fn = f"model-{s + 1:05d}-of-{shards:05d}.safetensors"
        save_file(part, os.path.join(d, fn))
        wm.update({n: fn for n in part})
    json.dump({"metadata": {}, "weight_map": wm}, open(os.path.join(d, "model.safetensors.index.json"), "w"))
    return sd


@pytest.mark.parametrize("batch", [1, 2])
def test_official_safetensors_e2e(work, batch):
    cfg, tok, tmap, tables, sd = setup(True, 2)
    full = write_official(cfg, sd, tables, work)
    src = SafetensorsWeightSource(work, cfg.num_hidden_layers, cfg.engram_layer_ids)
    # every tensor reads back bit-identical, dtypes preserved (fp8 / e8m0 / int8-packed fp4 / bf16 / f32)
    for n, t in full.items():
        r = src.sd[n]
        assert r.dtype == t.dtype and r.shape == t.shape and torch.equal(r.view(torch.uint8), t.view(torch.uint8)), n
    assert os.path.basename(work) not in mapped_files(), "model files must not be mmapped"
    prov = SourceExpertProvider(src, {l: cfg.moe_counts(l)[0] for l in range(cfg.num_hidden_layers + cfg.n_mtp_layers)},
                                cache=False)
    store = src.engram_store()
    assert isinstance(store, FileEngramStore)
    for r in store.readers:                                     # O_DIRECT really on every fd (Direct/Parallel: .fd, Native: .fds)
        fds = [r.fd] if hasattr(r, "fd") else list(r.fds.values())
        assert r.direct and fds and all(fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_DIRECT for fd in fds)
    mine = DSV41Model(cfg, src, prov, store, token_map=tmap)
    ref, miss = build_reference(cfg, sd, tables, tok, 256, batch, True, True)
    st = mine.new_state(batch, 256)
    g = torch.Generator().manual_seed(21)
    ids = torch.randint(0, cfg.vocab_size, (batch, 43 + 8), generator=g)
    with torch.inference_mode():
        rids, rl, rm = ref(ids[:, :43])
        ml, mm = mine.forward(ids[:, :43], 0, st)
        assert torch.equal(rl, ml) and torch.equal(rm, mm)
        ref.forward_spec(rids, rm, 0); mine.mtp.forward_spec(rids, mm, 0, st)
        for i in range(43, 51):
            rids, rl, rm = ref(ids[:, i:i + 1], i)
            ml, mm = mine.forward(ids[:, i:i + 1], i, st)
            assert torch.equal(rl, ml) and torch.equal(rids, ml.argmax(-1))
            ro, mo = ref.forward_spec(rids, rm, i), mine.mtp.forward_spec(rids, mm, i, st)
            assert torch.equal(ro[0], mo[0]) and torch.equal(ro[1], mo[1])
    assert store.readers[0].reads > 0 and store.readers[0].bytes_read % 4096 == 0
    assert prov.loads == prov.lookups and prov.loads > 0        # streaming provider: nothing stays resident
    assert prov.resident_bytes(2) == 0
    src.close()


def test_engram_rows_direct_vs_memory(work):
    """O_DIRECT row reader: random rows, rows straddling sector boundaries, last row, split official segments."""
    n, hd = 5003, 256
    g = torch.Generator().manual_seed(3)
    vals = torch.randint(0, 256, (n, hd), generator=g, dtype=torch.uint8)
    scl = torch.randint(0, 256, (n, 8), generator=g, dtype=torch.uint8)
    path = os.path.join(work, "t.bin")
    with open(path, "wb") as f:
        f.write(b"\x07" * 1000)                                 # unaligned start of table
        o1 = f.tell(); f.write(vals.numpy().tobytes())
        o2 = f.tell(); f.write(scl.numpy().tobytes())
        o3 = f.tell(); f.write(torch.cat([vals, scl], 1).numpy().tobytes())      # interleaved GGUF style
    split = DirectRowReader([(path, o1, 256, 256), (path, o2, 8, 8)])
    inter = DirectRowReader([(path, o3, 264, 264)])
    rows = torch.cat([torch.randint(0, n, (400,), generator=g), torch.tensor([0, n - 1, 15, 16, 31, 15, 15])])
    want = torch.cat([vals, scl], 1)[rows]
    assert split.direct and fcntl.fcntl(split.fd, fcntl.F_GETFL) & os.O_DIRECT
    assert torch.equal(split.read_rows(rows.tolist()), want)
    assert torch.equal(inter.read_rows(rows.tolist()), want)
    assert split.bytes_read % 4096 == 0 and inter.bytes_read % 4096 == 0
    assert os.path.basename(work) not in mapped_files()
    mem = MemoryEngramStore([vals.view(torch.uint8)], [scl])
    assert torch.equal(mem.fetch(0, rows), want)
    # coalescing: neighbours in the same sector do not reread
    before = inter.reads
    inter.read_rows([100, 101, 102])
    assert inter.reads - before <= 2
    with pytest.raises(OSError):
        os.open("/proc/version", os.O_RDONLY | os.O_DIRECT)     # sanity: O_DIRECT is enforced by the fs layer


# ----------------------------------------------------------------------------- GGUF decoders
@pytest.fixture(scope="module")
def gg_oracle():
    pytest.importorskip("gguf")
    from gguf import quants, GGMLQuantizationType as T
    return quants, T


@pytest.mark.parametrize("gtype,bb,be", [(Q8_0, 34, 32), (Q2_K, 84, 256), (IQ2_XXS, 66, 256)])
def test_gguf_decoders_vs_gguf_py(gg_oracle, gtype, bb, be):
    quants, T = gg_oracle
    rng = np.random.default_rng(gtype)
    nb = 300
    if gtype == Q8_0:
        raw = quantize_q8_0(torch.randn(nb * 32) * 0.3)
    else:
        raw = random_blocks(gtype, nb, rng)
    arr = np.frombuffer(raw, dtype=np.uint8)
    orc = quants.dequantize(arr.reshape(nb, bb), T(gtype)).reshape(-1)
    mine = G.decode(torch.from_numpy(arr.copy()), gtype, (nb * be,)).numpy()
    assert np.array_equal(orc.view(np.uint32), mine.view(np.uint32)), np.abs(orc - mine).max()
    assert np.isfinite(mine).all()


def test_iq2_tables_vs_antirez_cuda_inc():
    inc = "./ds41_work/ds41port/ref/ds4src/ds4_iq2_tables_cuda.inc"
    if not os.path.exists(inc):
        pytest.skip("ds4 source not cloned")
    import re
    from exllamav3.dsv41 import iq2_tables
    txt = open(inc).read()
    ks = re.search(r"cuda_ksigns_iq2xs\[128\] = \{(.*?)\};", txt, re.S).group(1)
    gr = re.search(r"cuda_iq2xxs_grid\[256\] = \{(.*?)\};", txt, re.S).group(1)
    ks = [int(x) for x in re.findall(r"\d+", ks)]
    gr = [int(x, 16) for x in re.findall(r"0x[0-9a-fA-F]+", gr)]
    assert len(ks) == 128 and len(gr) == 256
    assert ks == iq2_tables.ksigns_table().tolist()
    grid = iq2_tables.grid_table().to(torch.int64)                      # [256,8] bytes of each u64, little endian
    mine = [sum(int(grid[i, j]) << (8 * j) for j in range(8)) for i in range(256)]
    assert mine == gr


# ----------------------------------------------------------------------------- real GGUF header
REAL_HEAD = "./ds41_work/ds41probe/q2head.bin"


@pytest.mark.skipif(not os.path.exists(REAL_HEAD), reason="no antirez GGUF header probe")
def test_real_q2_header_parity():
    g = G.GGUFFile(REAL_HEAD)
    import collections
    assert dict(collections.Counter(t.type_name for t in g.tensors.values())) == \
        {"F32": 493, "Q8_0": 321, "F16": 110, "IQ2_XXS": 80, "Q2_K": 40, "I8": 2}
    src = G.GGUFWeightSource(g, 40)
    cfg = src.config()
    em = src.engram_meta(cfg)
    ref_cfg = DSV41Config.from_file("./ds41_work/ds41port/ref/config.json")
    for k in ("hidden_size", "num_hidden_layers", "compress_ratios", "kv_source_layer_ids", "index_source_layer_ids",
              "candidate_source_layer_id", "engram_layer_ids", "hc_mult", "swiglu_limit", "routed_scaling_factor",
              "original_seq_len", "rope_factor", "n_routed_experts"):
        a, b = getattr(cfg, k), getattr(ref_cfg, k)
        assert list(a) == list(b) if isinstance(a, list) else a == b, k
    lay = EngramLayout.build(cfg.engram_layer_ids, None, cfg.engram_max_ngram_size, cfg.engram_vocab_size,
                             cfg.engram_n_heads, cfg.engram_head_dim)
    assert [[list(h) for h in l] for l in lay.primes] == [[list(h) for h in l] for l in em["primes"]]
    assert np.array_equal(compute_multipliers(cfg.engram_layer_ids, 4, em["compressed_vocab"]), em["multipliers"])
    assert list(lay.num_embeddings) == em["rows"] == list(cfg.engram_num_embeddings)
    assert int(em["token_map"].max()) + 1 == em["compressed_vocab"] == cfg.engram_compressed_vocab_size
    # routed expert tensors: shape / type / bytes per expert
    for l in (0, 39):
        for k, ty in (("gate", "IQ2_XXS"), ("up", "IQ2_XXS"), ("down", "Q2_K")):
            t = g.tensors[f"blk.{l}.ffn_{k}_exps.weight"]
            assert t.type_name == ty and t.dims[2] == 384
    e = sum(g.tensors[f"blk.0.ffn_{k}_exps.weight"].nbytes() for k in ("gate", "up", "down")) // 384
    assert e == 9_953_280                                        # (DESIGN.md had 9 792 000)
    et = g.tensors["blk.1.engram_embd.weight"]
    assert et.type_name == "I8" and et.dims == (264, 384006168)
    # real bytes: head of token_embd (F16 rows) decodes to sane embeddings
    x = np.frombuffer(os.pread(g.fd, 5120 * 2 * 64, g.data_start), dtype=np.float16).astype(np.float32)
    assert np.isfinite(x).all() and 0.0 < x.std() < 1.0


# ----------------------------------------------------------------------------- GGUF end to end
def tiny_gguf_cfg():
    cfg = tiny_config(hidden_size=256, moe_intermediate_size=256, num_nextn_predict_layers=0, dspark_block_size=0)
    return cfg


def build_gguf(cfg, sd, tables, tmap, lay, path, rng):
    """sd: plain official-layout state dict. Returns (expected plain fp32 state dict as the loader must see it)."""
    pytest.importorskip("gguf")
    from gguf import quants, GGMLQuantizationType as T
    gb = GGUFBuilder(alignment=4096)
    expect = {}

    def dq(raw, gt, n):
        nb = len(raw) // {Q8_0: 34, Q2_K: 84, IQ2_XXS: 66}[gt]
        return quants.dequantize(np.frombuffer(raw, np.uint8).reshape(nb, -1), T(gt)).reshape(-1)

    def put(gname, oname, w, gt):
        w = w.to(torch.float32)
        if gt == Q8_0:
            raw = quantize_q8_0(w)
            deq = torch.from_numpy(dq(raw, Q8_0, w.numel()).copy()).reshape(w.shape)
        elif gt == F16:
            raw = w.to(torch.float16).numpy().tobytes()
            deq = w.to(torch.float16).float()
        else:
            raw, deq = w.numpy().tobytes(), w
        gb.add(gname, tuple(w.shape), gt, raw)
        expect[oname] = deq

    dense_q8 = ["attn.wq_a", "attn.wq_b", "attn.wkv", "attn.wo_a", "attn.wo_b", "ffn.shared_experts.w1",
                "ffn.shared_experts.w2", "ffn.shared_experts.w3"]
    f16 = ["attn.compressor.wkv", "attn.compressor.wgate", "attn.indexer.wq_b", "attn.indexer.weights_proj",
           "attn.indexer.wk", "hc_attn_fn", "hc_ffn_fn", "engram.wkv"]
    inv = {v: k for k, v in G._LAYER_MAP.items()}
    for l in range(cfg.num_hidden_layers):
        p = f"layers.{l}"
        for o, gname in G._LAYER_MAP.items():
            key = f"{p}.{o}"
            cand = [key + ".weight", key]
            k = next((c for c in cand if c in sd), None)
            if k is None:
                continue
            gt = Q8_0 if o in dense_q8 else F16 if o in f16 else F32
            put(f"blk.{l}.{gname}", k, sd[k], gt)
        for e_key, (gname, gt) in {"w1": ("ffn_gate_exps.weight", IQ2_XXS), "w3": ("ffn_up_exps.weight", IQ2_XXS),
                                   "w2": ("ffn_down_exps.weight", Q2_K)}.items():
            out_f, in_f = sd[f"{p}.ffn.experts.0.{e_key}.weight"].shape
            be, bb = (256, 66) if gt == IQ2_XXS else (256, 84)
            raws, exps = [], []
            for e in range(cfg.n_routed_experts):
                raw = random_blocks(gt, out_f * in_f // be, rng)
                raws.append(raw)
                expect[f"{p}.ffn.experts.{e}.{e_key}.weight"] = torch.from_numpy(dq(raw, gt, out_f * in_f).copy()).reshape(out_f, in_f)
            gb.add(f"blk.{l}.{gname}", (cfg.n_routed_experts, out_f, in_f), gt, b"".join(raws))
    put("token_embd.weight", "embed.weight", sd["embed.weight"], F16)
    put("output_norm.weight", "norm.weight", sd["norm.weight"], F32)
    put("output.weight", "head.weight", sd["head.weight"], Q8_0)
    for i, lid in enumerate(cfg.engram_layer_ids):
        rows = torch.cat([tables[i][0].view(torch.uint8), tables[i][1].view(torch.uint8)], 1)
        gb.add(f"blk.{lid}.engram_embd.weight", tuple(rows.shape), I8, rows.numpy().tobytes())
    hf = {"text_config": cfg.to_hf_text_dict()}
    gb.add_kv("general.architecture", "deepseek41")
    gb.add_kv("deepseek41.config", json.dumps(hf))
    gb.add_kv("deepseek41.engram.encoding", "e4m3_e8m0_32_row264")
    gb.add_kv("deepseek41.engram.layer_ids", np.array(cfg.engram_layer_ids, dtype=np.uint32))
    gb.add_kv("deepseek41.engram.rows", np.array(lay.num_embeddings, dtype=np.uint32))
    gb.add_kv("deepseek41.engram.compressed_vocab_size", int(cfg.engram_compressed_vocab_size))
    gb.add_kv("deepseek41.engram.pad_id", int(cfg.engram_pad_token_id))
    gb.add_kv("deepseek41.engram.token_map", np.array(tmap, dtype=np.uint32))
    gb.add_kv("deepseek41.engram.primes", np.array(lay.primes, dtype=np.uint32).reshape(-1))
    gb.add_kv("deepseek41.engram.multipliers", compute_multipliers(cfg.engram_layer_ids, 4, cfg.engram_compressed_vocab_size).astype(np.uint64).reshape(-1))
    gb.write(path)
    return expect


def test_gguf_e2e_vs_official_reference(work):
    pytest.importorskip("gguf")
    cfg = tiny_gguf_cfg()
    tok = FakeTok(cfg.vocab_size)
    from exllamav3.dsv41.engram import build_compressed_token_map
    tmap, nv = build_compressed_token_map(tok)
    cfg.engram_compressed_vocab_size = nv
    lay = EngramLayout.build(cfg.engram_layer_ids, None, 4, cfg.engram_vocab_size, cfg.engram_n_heads, cfg.engram_head_dim)
    cfg.engram_num_embeddings = list(lay.num_embeddings)
    tables = make_engram_tables(cfg, lay.num_embeddings, 9)
    sd = make_state_dict(cfg, 4, fp8=False, fp4=False)
    path = os.path.join(work, "tiny.gguf")
    expect = build_gguf(cfg, sd, tables, tmap, lay, path, np.random.default_rng(0))
    # the official reference run on exactly the values the GGUF encodes (plain fp32 mode)
    ref_sd = dict(expect)
    ref, miss = build_reference(cfg, ref_sd, tables, tok, 256, 1, False, False)
    assert not miss, miss[:3]
    mine, src = G.load_gguf_model(path)
    assert mine.mtp is None and mine.cfg.n_mtp_layers == 0
    assert os.path.basename(work) not in mapped_files()
    st = mine.new_state(1, 256)
    ids = torch.randint(0, cfg.vocab_size, (1, 40), generator=torch.Generator().manual_seed(8))
    rtaps = install_hooks(ref, cfg)
    with torch.inference_mode():
        rids, rl, rm = ref(ids[:, :33])
        mt = {}
        ml, mm = mine.forward(ids[:, :33], 0, st, mt)
        cmp_taps(rtaps, mt, "gguf prefill", {})
        assert (rl - ml).abs().max() <= TOL and torch.equal(rids, ml.argmax(-1))
        for i in range(33, 40):
            rtaps.clear(); mt = {}
            rids, rl, rm = ref(ids[:, i:i + 1], i)
            ml, mm = mine.forward(ids[:, i:i + 1], i, st, mt)
            cmp_taps(rtaps, mt, f"gguf decode {i}", {})
            assert (rl - ml).abs().max() <= TOL and torch.equal(rids, ml.argmax(-1))
    assert mine.engram_store.readers[0].direct
    mine.engram_store.close()


# ----------------------------------------------------------------------------- provider API
class RecordingProvider(ExpertProvider):
    """Stands in for a tiered store: nothing is resident, every lookup rebuilds from the base provider."""

    def __init__(self, base):
        self.base, self.log = base, []

    def total_experts(self, layer):
        return self.base.total_experts(layer)

    def prefetch(self, layer, experts):
        self.log.append(("prefetch", layer, tuple(experts)))

    def lookup(self, layer, e):
        self.log.append(("lookup", layer, e))
        return self.base.lookup(layer, e)

    def release(self, h):
        self.log.append(("release", h.layer, h.expert))


def test_expert_provider_plug_in():
    cfg, tok, tmap, tables, sd = setup(True, 5)
    m1 = build_mine(cfg, sd, tables, tmap)
    m2 = build_mine(cfg, sd, tables, tmap)
    base = SourceExpertProvider(m2.src, {l: cfg.moe_counts(l)[0] for l in range(10)}, cache=False)
    rec = RecordingProvider(base)
    m2.provider = rec
    ids = torch.randint(0, cfg.vocab_size, (1, 21), generator=torch.Generator().manual_seed(2))
    a, _ = m1.forward(ids, 0, m1.new_state(1, 64))
    b, _ = m2.forward(ids, 0, m2.new_state(1, 64))
    assert torch.equal(a, b)
    # per MoE layer: one prefetch with the sorted unique experts, then lookup/release pairs in that order
    pf = [x for x in rec.log if x[0] == "prefetch"]
    assert len(pf) == cfg.num_hidden_layers and [p[1] for p in pf] == list(range(cfg.num_hidden_layers))
    i = 0
    for p in pf:
        j = rec.log.index(p, i)
        seq = [x for x in rec.log[j + 1: j + 1 + 2 * len(p[2])]]
        assert [x[2] for x in seq[0::2]] == list(p[2]) == sorted(p[2])
        assert [x[0] for x in seq] == ["lookup", "release"] * len(p[2])
        i = j + 1


# ----------------------------------------------------------------------------- real checkpoint names
REAL_INDEX = "./ds41_work/ds41port/ref/model.safetensors.index.json"
REAL_CFG = "./ds41_work/ds41port/ref/config.json"


@pytest.mark.skipif(not (os.path.exists(REAL_INDEX) and os.path.exists(REAL_CFG)), reason="no HF index fetched")
def test_graph_requests_exist_in_real_hf_index():
    """Instantiate the REAL-size graph against a recording source (no weights): every tensor the graph asks
    for must exist in the real model.safetensors.index.json, with the .scale companions where the official
    format has them, and the routed experts of all 40+3 layers must be addressable."""
    from exllamav3.dsv41.weights import WeightSource
    from exllamav3.dsv41.linear import DenseLin
    from exllamav3.dsv41.experts import ExpertHandle
    names = set(json.load(open(REAL_INDEX))["weight_map"])
    cfg = DSV41Config.from_file(REAL_CFG)
    asked = []

    class Rec(WeightSource):
        n_layers = cfg.num_hidden_layers

        def has(self, n):
            return True

        def tensor(self, n, dtype=torch.float32, device="cpu"):
            asked.append(("t", n))
            return torch.ones(2, 2)

        def linear(self, n, dtype=torch.float32, device="cpu"):
            asked.append(("l", n))
            return DenseLin(torch.ones(2, 2))

    class P(SourceExpertProvider):
        pass
    prov = SourceExpertProvider(Rec(), {l: cfg.moe_counts(l)[0] for l in range(43)})
    DSV41Model(cfg, Rec(), prov, None, token_map=[0] * cfg.vocab_size)
    quant_lin = {"wq_a", "wq_b", "wkv", "wo_b", "w1", "w2", "w3", "main_proj"}
    miss = []
    for kind, n in asked:
        if kind == "t":
            ok = n in names or (n + ".weight") in names
        else:
            leaf = n.rsplit(".", 1)[-1]
            ok = (n + ".weight") in names
            has_scale = (n + ".scale") in names
            if leaf in quant_lin or n.endswith("indexer.wq_b") or n.endswith("engram.wkv"):
                ok = ok and has_scale
        if not ok:
            miss.append((kind, n))
    assert not miss, miss[:10]
    assert len(asked) > 500
    for l in range(43):
        p = f"layers.{l}" if l < 40 else f"mtp.{l - 40}"
        ne = cfg.moe_counts(l)[0]
        for e in (0, ne - 1):
            for w in ("w1", "w2", "w3"):
                assert f"{p}.ffn.experts.{e}.{w}.weight" in names and f"{p}.ffn.experts.{e}.{w}.scale" in names
        assert f"{p}.ffn.experts.{ne}.w1.weight" not in names
    # engram tables are separate fp8 + e8m0 tensors of [rows, 256] / [rows, 8]
    for lid in cfg.engram_layer_ids:
        assert f"layers.{lid}.engram.embed.weight" in names and f"layers.{lid}.engram.embed.scale" in names
