"""Converter -> pack -> serving loader, end to end on the CPU with tiny random weights.

A tiny checkpoint in the OFFICIAL layout (fp8 / fp4 weights, MXFP4 Engram tables) is served over a local HTTP Range server;
the real converter (tools/ds41/stream_convert.py, plan + process_unit + finalize) quantizes it with the CPU EXL3 codec
(per-expert K including K=1 and pruned experts, MTP stages, Engram tables written as separate files); the serving loader
(`exllamav3.dsv41.pack.load`) reads the result. Logits are compared with
  * a tight reference: the same model on the decoded (dequantized) pack weights, and
  * a loose reference: the unquantized checkpoint.
"""
import http.server
import json
import os
import sys
import threading

import pytest
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "tools", "ds41"))

import tiny_ckpt  # noqa: E402
from exllamav3.dsv41 import packspec as PS  # noqa: E402
from exllamav3.dsv41.engram import MemoryEngramStore  # noqa: E402
from exllamav3.dsv41.experts import SourceExpertProvider  # noqa: E402
from exllamav3.dsv41.model import DSV41Model  # noqa: E402
from exllamav3.dsv41.weights import DictWeightSource  # noqa: E402
from exllamav3.modules.quant import exl3_cpu  # noqa: E402

torch.set_num_threads(int(os.environ.get("DSV41_THREADS", "4")))

REPO, REV = "tiny/ds41", "r1"
# per-expert K of two backbone layers: K=1 and pruned experts (0) mixed with 2..4; the other layers use the plan tiers
EXPERT_BITS = {"layers.2.ffn": [2, 1, 0, 3, 2, 4, 0, 2], "layers.5.ffn": [3, 3, 1, 2, 0, 2, 2, 4]}


class _Handler(http.server.BaseHTTPRequestHandler):
    files = {}

    def log_message(self, *a):
        pass

    def do_GET(self):
        data = self.files.get(self.path.rsplit("/", 1)[-1])
        if data is None:
            self.send_error(404)
            return
        rng = self.headers.get("Range")
        if rng:
            a, b = rng.split("=")[1].split("-")
            a, b = int(a), min(int(b), len(data) - 1)
            body = data[a:b + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {a}-{b}/{len(data)}")
        else:
            body = data
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _header(raw: bytes) -> dict:
    n = int.from_bytes(raw[:8], "little")
    return json.loads(raw[8:8 + n])


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    import stream_convert as S
    import tensor_map as TM
    src = str(tmp_path_factory.mktemp("src"))
    cfg, sd, tables, tmap = tiny_ckpt.write_tiny(src, seed=3, engram_fmt="mxfp4")
    files = {n: open(os.path.join(src, n), "rb").read() for n in os.listdir(src)}
    hdr = _header(files["model.safetensors"])
    files["model.safetensors.index.json"] = json.dumps(
        {"metadata": {"total_size": 0}, "weight_map": {k: "model.safetensors" for k in hdr if k != "__metadata__"}}).encode()
    _Handler.files = files
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    os.environ["DS41_BASE_URL"] = f"http://127.0.0.1:{srv.server_address[1]}"
    work = str(tmp_path_factory.mktemp("work"))
    try:
        tm = TM.build(os.path.join(work, "map.json"), REPO, REV, workers=1, log=lambda *_: None) \
            if False else None
        # the index is synthetic: skip the total_size check by scanning the shard directly
        tensors, info = TM.scan_shard("model.safetensors", REPO, REV)
        tm = {"repo": REPO, "revision": REV, "shards": {"model.safetensors": info}, "tensors": tensors}
        mp = os.path.join(work, "map.json")
        json.dump(tm, open(mp, "w"))
        eb = os.path.join(work, "expert_bits.json")
        json.dump(EXPERT_BITS, open(eb, "w"))
        out = os.path.join(work, "pack")
        S.main(["--map", mp, "--out", out, "--include", "global,mtp,engram", "--layers", "0-7", "--mtp-queue",
                "--quantizer", "cpu", "--o-groups", str(cfg.o_groups), "--expert-bits-file", eb, "--target", "2.3"])
    finally:
        srv.shutdown()
        os.environ.pop("DS41_BASE_URL", None)
    return out, cfg, sd, tables, tmap, tm


class NoAQ(DictWeightSource):
    """Reference source without the official fp8 activation fake-quant (EXL3 serves fp16 / fp32 activations)."""

    def linear(self, name, dtype=torch.float32, device="cpu", act_quant=False):
        return super().linear(name, dtype, device, act_quant=False)


def _ref(cfg, sd, tables, tmap):
    src = NoAQ(sd, cfg.num_hidden_layers)
    n = cfg.num_hidden_layers + cfg.n_mtp_layers
    prov = SourceExpertProvider(src, {l: cfg.moe_counts(l)[0] for l in range(n)})
    store = MemoryEngramStore([t[0] for t in tables], [t[1] for t in tables], fmt="mxfp4")
    return DSV41Model(cfg, src, prov, store, token_map=tmap)


def _pack_tensors(out):
    from safetensors.torch import load_file
    idx = json.load(open(os.path.join(out, "model.safetensors.index.json")))["weight_map"]
    t = {}
    for f in sorted(set(idx.values())):
        t.update(load_file(os.path.join(out, f)))
    return t, idx


def decoded_state_dict(out, cfg):
    """The pack as a dense float state dict: every EXL3 linear decoded, wo_a slices joined, pruned experts zero."""
    import re
    t, _ = _pack_tensors(out)
    sd, slices = {}, {}
    for k, v in t.items():
        if k.endswith((".suh", ".svh", ".mul1")):
            continue
        if k.endswith(".trellis"):
            key = k[:-8]
            w = exl3_cpu.reconstruct(v, t[key + ".suh"], t[key + ".svh"]).half().float().t().contiguous()   # [out, in]; the loader holds fp16
            m = re.match(r"^(.*\.attn\.wo_a)\.slice\.(\d+)$", key)
            if m:
                slices.setdefault(m.group(1), {})[int(m.group(2))] = w
            else:
                sd[key + ".weight"] = w
        else:
            sd[k] = v.float() if v.dtype in (torch.bfloat16, torch.float16) else v
    for base, parts in slices.items():
        sd[base + ".weight"] = torch.cat([parts[g] for g in sorted(parts)], 0)
    d, mi = cfg.hidden_size, cfg.moe_intermediate_size
    for l in range(cfg.num_hidden_layers):
        for e, k in enumerate(EXPERT_BITS.get(f"layers.{l}.ffn", [])):
            if k == 0:
                sd[f"layers.{l}.ffn.experts.{e}.w1.weight"] = torch.zeros(mi, d)
                sd[f"layers.{l}.ffn.experts.{e}.w3.weight"] = torch.zeros(mi, d)
                sd[f"layers.{l}.ffn.experts.{e}.w2.weight"] = torch.zeros(d, mi)
    return sd


@pytest.fixture(scope="module")
def models(converted):
    from exllamav3.dsv41.pack import load
    out, cfg, sd, tables, tmap, tm = converted
    served = load(out, device="cpu", dtype=torch.float32, moe="loop")
    tight = _ref(cfg, decoded_state_dict(out, cfg), tables, tmap)
    loose = _ref(cfg, sd, tables, tmap)
    return served, tight, loose


def _ids(cfg, n, seed=5):
    return torch.randint(0, cfg.vocab_size - 1, (1, n), generator=torch.Generator().manual_seed(seed))


def test_converter_output_validates(converted):
    import validate_pack
    out, cfg, *_ = converted
    rep = validate_pack.validate(out)
    assert rep.errors == [], rep.errors[:5]


def test_pack_carries_per_expert_k_and_pruned_experts(converted):
    out, cfg, *_ = converted
    t, idx = _pack_tensors(out)
    q = json.load(open(os.path.join(out, "config.json")))["quantization_config"]
    assert q["pack_format"] == PS.PACK_FORMAT
    assert all(q["expert_bits"][k] == v for k, v in EXPERT_BITS.items())
    assert q["pruned_experts"] == {"layers.2.ffn": [2, 6], "layers.5.ffn": [4]}
    for layer, ks in EXPERT_BITS.items():
        for e, k in enumerate(ks):
            for w in ("w1", "w2", "w3"):
                key = f"{layer}.experts.{e}.{w}"
                if k == 0:
                    assert key + ".trellis" not in t
                else:
                    assert t[key + ".trellis"].shape[-1] == 16 * k, key
        bias = t[f"{layer}.gate.bias"]
        pr = [e for e, k in enumerate(ks) if k == 0]
        assert (bias[pr] == -1e4).all() and int((bias == -1e4).sum()) == len(pr)


def test_engram_tables_are_separate_files_equal_to_the_source(converted):
    out, cfg, sd, tables, *_ = converted
    t, idx = _pack_tensors(out)
    assert not any(".engram.embed." in k for k in idx)
    for i, lid in enumerate(cfg.engram_layer_ids):
        for j, nm in enumerate(("weight", "scale")):
            f = os.path.join(out, "engram", f"layers.{lid}.engram.embed.{nm}.bin")
            assert open(f, "rb").read() == tables[i][j].contiguous().view(torch.uint8).numpy().tobytes()
    assert json.load(open(os.path.join(out, "config.json")))["quantization_config"]["engram"]["format"] == "mxfp4"


def test_served_logits_equal_the_decoded_reference(converted, models):
    out, cfg, *_ = converted
    served, tight, _ = models
    assert served.mtp is not None and served.cfg.n_mtp_layers == 2
    L = 30
    ids = _ids(cfg, L + 6)
    a, b = tight.new_state(1, 64), served.new_state(1, 64)
    lr, hr = tight.forward(ids[:, :L], 0, a)
    ls, hs = served.forward(ids[:, :L], 0, b)
    assert lr.abs().max() > 0.1
    assert (lr - ls).abs().max().item() < 2e-4 * lr.abs().max().item()
    assert (hr - hs).abs().max().item() < 2e-4 * hr.abs().max().item()
    for i in range(6):
        lr, _ = tight.forward(ids[:, L + i:L + i + 1], L + i, a)
        ls, _ = served.forward(ids[:, L + i:L + i + 1], L + i, b)
        assert (lr - ls).abs().max().item() < 2e-4 * lr.abs().max().item()
        assert int(lr.argmax()) == int(ls.argmax())


def test_served_pack_takes_image_spans_like_the_decoded_reference(converted, models, monkeypatch):
    """Random image rows spliced into the pack-loaded serving model and into the decoded reference give the same logits;
    the rows change the logits (they are not ignored), and `ffn.gate.bias_vl` is loaded from the pack."""
    from exllamav3.dsv41.vision import ImageSpan
    out, cfg, *_ = converted
    served, tight, _ = models
    ids = _ids(cfg, 40, seed=8)
    g = torch.Generator().manual_seed(2)
    spans = [ImageSpan(9, torch.randn(20, cfg.hidden_size, generator=g), "k")]
    with torch.inference_mode():
        lr, _ = tight.forward(ids, 0, tight.new_state(1, 64), spans=spans)
        ls, _ = served.forward(ids, 0, served.new_state(1, 64), spans=spans)
        plain, _ = served.forward(ids, 0, served.new_state(1, 64))
        # rows far from the embedding distribution sit near fp8 grid edges: the two graphs may flip a few cached values
        assert (lr - ls).abs().max().item() < 2e-2 * lr.abs().max().item()
        assert (ls - plain).abs().max().item() > 1e-2
        # chunked prefill is checked on the 16-index-head toy (test_dsv41_vision_model.py): with the 4 index heads of this
        # fixture the indexer's top-k has near ties that a different chunk shape breaks differently, with or without images
        # activation grids off: the two graphs agree tightly, so the gap above is grid noise, not routing or splice
        from exllamav3.dsv41 import quant
        monkeypatch.setattr(quant, "act_quant_fp8_", lambda x, *a, **k: x)
        monkeypatch.setattr(quant, "fp4_act_quant_", lambda x, *a, **k: x)
        lr0, _ = tight.forward(ids, 0, tight.new_state(1, 64), spans=spans)
        ls0, _ = served.forward(ids, 0, served.new_state(1, 64), spans=spans)
        assert (lr0 - ls0).abs().max().item() < 1e-3 * lr0.abs().max().item()
    loaded = 0
    for l in range(cfg.num_hidden_layers):                               # served and reference agree on having ffn.gate.bias_vl
        a, b = getattr(served.lw[l], "gate_b_vl", None), getattr(tight.lw[l], "gate_b_vl", None)
        assert (a is None) == (b is None)
        if a is not None:
            assert torch.equal(a.cpu().float(), b.cpu().float())
            loaded += 1
    assert loaded > 0                                                      # the pack carries ffn.gate.bias_vl and the loader reads it


def test_served_logits_track_the_unquantized_model(converted, models):
    """Quantization tolerance: K=1..4 on a random tiny model keeps the logits strongly correlated with the source."""
    out, cfg, *_ = converted
    served, _, loose = models
    ids = _ids(cfg, 30)
    lr, _ = loose.forward(ids, 0, loose.new_state(1, 64))
    ls, _ = served.forward(ids, 0, served.new_state(1, 64))
    cos = torch.nn.functional.cosine_similarity(lr.flatten().double(), ls.flatten().double(), dim=0).item()
    rel = ((lr - ls).norm() / lr.norm()).item()
    print("loose: cos", cos, "rel", rel)
    assert cos > 0.7 and rel < 0.8


# ------------------------------------------------------------------------------------------------ spec vs framework tree
def _walk(m):
    yield m
    for s in getattr(m, "modules", []) or []:
        yield from _walk(s)


def test_spec_matches_the_framework_module_tree(tmp_path):
    """Every Linear / raw tensor the framework builds is in the shared spec with the same shape, and vice versa."""
    from exllamav3.dsv41.synth_pack import write_pack, synth_config
    from exllamav3.model.config import Config
    from exllamav3.model.model import Model
    from exllamav3.modules import Linear
    from exllamav3.modules.dsv41 import DSV41Raw
    write_pack(str(tmp_path), synth_config(), seed=1)
    c = Config.from_directory(str(tmp_path))
    d = c.dsv41
    models = [Model.from_config(c)] + ([Model.from_config(c, component="mtp")] if c.dsv41_n_mtp else [])
    lins, raws = {}, set()
    for mo in models:
        for m in mo.modules:
            for x in _walk(m):
                if isinstance(x, Linear):
                    lins[x.key] = (x.in_features_unpadded, x.out_features_unpadded)
                if isinstance(x, DSV41Raw):
                    raws |= set(x.names) | set(x.optional)
    exp_l = {l.key: (l.in_f, l.out_f) for l in PS.linears(d)}
    exp_r = {r.name for r in PS.raws(d)}
    assert set(lins) == set(exp_l)
    assert [k for k in lins if lins[k] != exp_l[k]] == []
    assert raws == exp_r - {"embed.weight", "norm.weight"}          # those two belong to the Embedding / RMSNorm modules
    assert {"embed.weight", "norm.weight"} <= exp_r


# ------------------------------------------------------------------------------------------------ Engram wiring
@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    """The same synthetic pack twice: tables as files in <pack>/engram (pack layout) and inside shards (older layout)."""
    from exllamav3.dsv41.synth_pack import write_pack, synth_config
    a, b = str(tmp_path_factory.mktemp("eg_dir")), str(tmp_path_factory.mktemp("eg_shards"))
    cfg, sd, tables, tmap = write_pack(a, synth_config(), seed=4)
    write_pack(b, synth_config(), seed=4, engram="shards")
    return a, b, cfg, tables


def _logits(m, cfg, n=50, chunk=None):
    ids = _ids(cfg, n, seed=9)
    st = m.new_state(1, 96)
    if chunk:
        lg, _ = m.prefill(ids, st, 0, chunk=chunk)
        return lg
    lg, _ = m.forward(ids, 0, st)
    return lg


def test_pack_layout_has_engram_dir_and_loader_uses_it(synth):
    from exllamav3.dsv41.engram import PrefetchRowStore, RawBinEngramStore
    from exllamav3.dsv41.pack import load
    a, b, cfg, tables = synth
    assert sorted(os.listdir(os.path.join(a, "engram"))) == sorted(
        f"layers.{l}.engram.embed.{n}.bin" for l in cfg.engram_layer_ids for n in ("weight", "scale"))
    assert not os.path.isdir(os.path.join(b, "engram"))
    m = load(a, device="cpu", dtype=torch.float32, moe="loop")
    assert isinstance(m.engram_store, PrefetchRowStore) and isinstance(m.engram_store.store, RawBinEngramStore)
    mb = load(b, device="cpu", dtype=torch.float32, moe="loop")
    assert torch.equal(_logits(m, cfg), _logits(mb, cfg))             # dir layout == shard layout, bit for bit


@pytest.mark.parametrize("chunk", [None, 16])
def test_prefetch_changes_nothing_and_hits(synth, chunk):
    from exllamav3.dsv41.pack import load
    a, _, cfg, _ = synth
    on = load(a, device="cpu", dtype=torch.float32, moe="loop")
    off = load(a, device="cpu", dtype=torch.float32, moe="loop", engram_prefetch=False)
    assert not hasattr(off.engram_store, "hits")
    assert torch.equal(_logits(on, cfg, chunk=chunk), _logits(off, cfg, chunk=chunk))
    st = on.engram_store
    assert st.hits > 0 and st.misses == 0, (st.hits, st.misses)       # every row set was queued before it was needed


def test_prefetch_overlaps_reads_with_compute(synth):
    """A slow store (60 ms per read): the queued reads run while the layers in front of the Engram layer compute."""
    import time
    from exllamav3.dsv41.engram import EngramRowStore, PrefetchRowStore, RawBinEngramStore
    from exllamav3.dsv41.serve_model import DSV41Serving
    from exllamav3.dsv41.pack import load
    a, _, cfg, _ = synth

    class Slow(EngramRowStore):
        def __init__(self, inner):
            self.inner, self.fmt, self.calls = inner, inner.fmt, 0

        def num_rows(self, i):
            return self.inner.num_rows(i)

        def fetch(self, i, rows):
            self.calls += 1
            time.sleep(0.06)
            return self.inner.fetch(i, rows)

    base = load(a, device="cpu", dtype=torch.float32, moe="loop", engram_prefetch=False)
    inner = base.engram_store
    ids = _ids(cfg, 40, seed=2)

    def run(store):
        base.engram_store = store
        t = time.time()
        lg, _ = base.forward(ids, 0, base.new_state(1, 64))
        return lg, time.time() - t

    lg_sync, t_sync = run(Slow(inner))
    pre = PrefetchRowStore(Slow(inner))
    lg_pre, t_pre = run(pre)
    assert torch.equal(lg_sync, lg_pre)
    assert pre.hits > 0 and pre.misses == 0
    # reads run in the background from the start of the pass: no slower than the synchronous pass
    assert t_pre <= t_sync + 0.05, (t_pre, t_sync)


@pytest.mark.parametrize("kind", ["auto", "native", "parallel", "direct"])
def test_every_reader_returns_the_same_rows(synth, kind, monkeypatch):
    from exllamav3.dsv41.engram import RawBinEngramStore, MemoryEngramStore
    a, _, cfg, tables = synth
    monkeypatch.setenv("EXL3_DSV41_ENGRAM_READER", kind)
    st = RawBinEngramStore.open(os.path.join(a, "engram"), cfg.engram_layer_ids, cfg.engram_head_dim, "mxfp4",
                                allow_buffered=True)
    mem = MemoryEngramStore([t[0] for t in tables], [t[1] for t in tables], fmt="mxfp4")
    g = torch.Generator().manual_seed(1)
    for i in range(len(cfg.engram_layer_ids)):
        rows = torch.randint(0, st.num_rows(i), (200,), generator=g)
        assert torch.equal(st.fetch(i, rows), mem.fetch(i, rows)), (kind, i)


def test_reader_falls_back_when_the_native_build_is_missing(synth, monkeypatch):
    import exllamav3.modules.dsv41_fn.engram_io as io
    from exllamav3.dsv41.engram import RawBinEngramStore, DirectRowReader, MemoryEngramStore
    a, _, cfg, tables = synth

    def boom(*args, **kw):
        raise ImportError("no compiler")
    monkeypatch.setattr(io, "NativeRowReader", boom)
    monkeypatch.setattr(io, "ParallelRowReader", boom)
    monkeypatch.setenv("EXL3_DSV41_ENGRAM_READER", "auto")
    st = RawBinEngramStore.open(os.path.join(a, "engram"), cfg.engram_layer_ids, cfg.engram_head_dim, "mxfp4",
                                allow_buffered=True)
    assert all(isinstance(r, DirectRowReader) for r in st.readers)
    mem = MemoryEngramStore([t[0] for t in tables], [t[1] for t in tables], fmt="mxfp4")
    rows = torch.arange(0, 64)
    assert torch.equal(st.fetch(0, rows), mem.fetch(0, rows))


def test_odirect_on_a_real_disk(synth):
    """The tables sit on the disk the user's home lives on (not tmpfs): the readers use O_DIRECT, no buffered fallback."""
    import shutil
    import tempfile
    from exllamav3.dsv41.engram import RawBinEngramStore, MemoryEngramStore
    a, _, cfg, tables = synth
    base = os.path.expanduser("~/.cache")
    os.makedirs(base, exist_ok=True)
    d = tempfile.mkdtemp(dir=base, prefix="dsv41_engram_")
    try:
        shutil.copytree(os.path.join(a, "engram"), os.path.join(d, "engram"))
        try:
            st = RawBinEngramStore.open(os.path.join(d, "engram"), cfg.engram_layer_ids, cfg.engram_head_dim, "mxfp4")
        except OSError as ex:
            pytest.skip(f"no O_DIRECT here: {ex}")
        mem = MemoryEngramStore([t[0] for t in tables], [t[1] for t in tables], fmt="mxfp4")
        rows = torch.randint(0, st.num_rows(1), (100,), generator=torch.Generator().manual_seed(3))
        assert torch.equal(st.fetch(1, rows), mem.fetch(1, rows))
        assert all(getattr(r, "direct", True) for r in st.readers)
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------------------------------------ validate_pack
def _copy_pack(converted, tmp_path):
    import shutil
    out = converted[0]
    d = str(tmp_path / "pack")
    shutil.copytree(out, d)
    return d


def _edit_shard(pack, tensor, fn):
    """Apply fn(tensors) to the shard that holds `tensor` (a dict of that shard's tensors, mutated in place)."""
    from safetensors.torch import load_file, save_file
    idx = json.load(open(os.path.join(pack, "model.safetensors.index.json")))
    f = os.path.join(pack, idx["weight_map"][tensor])
    t = load_file(f)
    fn(t, idx)
    save_file(t, f, metadata={"format": "pt"})
    json.dump(idx, open(os.path.join(pack, "model.safetensors.index.json"), "w"))


def test_validate_missing_tensor(converted, tmp_path):
    import validate_pack
    d = _copy_pack(converted, tmp_path)
    key = "layers.3.attn.wq_a.suh"

    def drop(t, idx):
        del t[key]
        del idx["weight_map"][key]
    _edit_shard(d, key, drop)
    rep = validate_pack.validate(d)
    assert any(key in e for e in rep.errors), rep.errors[:5]


def test_validate_wrong_k(converted, tmp_path):
    import validate_pack
    d = _copy_pack(converted, tmp_path)
    p = os.path.join(d, "config.json")
    cfg = json.load(open(p))
    eb = cfg["quantization_config"]["expert_bits"]["layers.2.ffn"]
    eb[0] = 4 if eb[0] != 4 else 3
    json.dump(cfg, open(p, "w"))
    rep = validate_pack.validate(d)
    assert any("layers.2.ffn.experts.0" in e and "K" in e for e in rep.errors), rep.errors[:5]


def test_validate_unmasked_pruned_bias(converted, tmp_path):
    import validate_pack
    d = _copy_pack(converted, tmp_path)
    key = "layers.2.ffn.gate.bias"

    def unmask(t, idx):
        t[key] = t[key].clone()
        t[key][2] = 0.1
    _edit_shard(d, key, unmask)
    rep = validate_pack.validate(d)
    assert any("bias" in e and "layers.2" in e for e in rep.errors), rep.errors[:5]


def test_validate_missing_engram_file(converted, tmp_path):
    import validate_pack
    d = _copy_pack(converted, tmp_path)
    os.remove(os.path.join(d, "engram", "layers.5.engram.embed.scale.bin"))
    rep = validate_pack.validate(d)
    assert any("engram" in e and "layers.5" in e for e in rep.errors), rep.errors[:5]


def test_validate_truncated_engram_file(converted, tmp_path):
    import validate_pack
    d = _copy_pack(converted, tmp_path)
    f = os.path.join(d, "engram", "layers.1.engram.embed.weight.bin")
    os.truncate(f, os.path.getsize(f) - 64)
    rep = validate_pack.validate(d)
    assert any("engram" in e and "layers.1" in e for e in rep.errors), rep.errors[:5]


def test_validate_size_vs_plan(converted, tmp_path):
    import validate_pack
    d = _copy_pack(converted, tmp_path)
    p = os.path.join(d, "plan.json")
    plan = json.load(open(p))
    for e in plan["entries"]:
        if e["action"] == "quant" and e["kind"] == "attn":
            e["out_bytes"] += 1000
            break
    json.dump(plan, open(p, "w"))
    rep = validate_pack.validate(d)
    assert any("plan" in e.lower() or "bytes" in e for e in rep.errors), rep.errors[:5]


def test_validate_cli_exit_status(converted, tmp_path):
    import subprocess
    d = _copy_pack(converted, tmp_path)
    tool = os.path.join(ROOT, "tools", "ds41", "validate_pack.py")
    ok = subprocess.run([sys.executable, tool, d, "--json"], capture_output=True, text=True)
    assert ok.returncode == 0, ok.stdout[-500:] + ok.stderr[-500:]
    assert json.loads(ok.stdout)["errors"] == []
    os.remove(os.path.join(d, "engram", "layers.1.engram.embed.weight.bin"))
    bad = subprocess.run([sys.executable, tool, d], capture_output=True, text=True)
    assert bad.returncode == 1


# ------------------------------------------------------------------------------------------------ the codec
def test_codec_pack_unpack_roundtrip():
    import numpy as np
    rng = np.random.default_rng(0)
    for K in (1, 2, 3, 4, 8):
        words = rng.integers(-32768, 32768, (5, 16 * K), dtype=np.int16)
        st = exl3_cpu.unpack_states(words, K)
        assert st.shape == (5, 256) and np.array_equal(exl3_cpu.pack_states(st, K), words), K
        # tail-biting: every state is the previous one shifted by K bits plus the new K bits (cyclic)
        v = st.astype(np.uint32) & ((1 << K) - 1)
        assert np.array_equal(st[:, 1:], ((st[:, :-1].astype(np.uint32) << K) | v[:, 1:]) & 0xFFFF), K
        assert np.array_equal(st[:, 0], ((st[:, -1].astype(np.uint32) << K) | v[:, 0]) & 0xFFFF), K


def test_codec_quantization_error_by_k():
    g = torch.Generator().manual_seed(1)
    w = torch.randn(256, 128, generator=g)
    errs = {}
    for K in (1, 2, 3, 4):
        q = exl3_cpu.quantize_linear(w, K, seed=5)
        assert q["trellis"].shape == (128 // 16, 256 // 16, 16 * K) and q["trellis"].dtype == torch.int16
        assert q["mul1"].dtype == torch.int32 and q["mul1"].dim() == 0
        rec = exl3_cpu.reconstruct(q["trellis"], q["suh"], q["svh"]).t()
        errs[K] = ((rec - w).norm() / w.norm()).item()
    assert errs[1] < 0.65 and errs[2] < 0.33 and errs[3] < 0.2 and errs[4] < 0.12, errs
    assert errs[1] > errs[2] > errs[3] > errs[4]


def test_codec_cpp_viterbi_equals_numpy_reference():
    import numpy as np
    tiles = np.random.default_rng(2).standard_normal((4, 256)).astype(np.float32)
    for K in (2, 3):
        a = exl3_cpu.viterbi(tiles, K)
        b = exl3_cpu.viterbi(tiles, K, force_python=True)
        assert np.array_equal(a, b), K


def _add_tower(pack, drop=None, bad_shape=None):
    """Add a tiny vision tower (2 layers, dim 64) to a copied pack: config.json entry + one shard + index entries."""
    from safetensors.torch import save_file
    from exllamav3.dsv41 import vision_spec as VS
    cfgp = os.path.join(pack, "config.json")
    cfg = json.load(open(cfgp))
    hs = cfg["text_config"]["hidden_size"]
    cfg["image_token_id"] = 500
    cfg["vision_config"] = {"num_hidden_layers": 2, "hidden_size": 64, "num_attention_heads": 4, "intermediate_size": 96,
                            "patch_size": 14, "downsample_ratio": 3, "max_image_tokens": 36, "min_pixels": 2000}
    json.dump(cfg, open(cfgp, "w"))
    v = VS.read_vision_config(cfg)
    tens = {n: torch.zeros(s, dtype=torch.bfloat16) for n, s in VS.expected_tower_shapes(v).items()}
    if drop:
        del tens[drop]
    if bad_shape:
        tens[bad_shape] = torch.zeros(3, 3, dtype=torch.bfloat16)
    save_file(tens, os.path.join(pack, "global.safetensors"), metadata={"format": "pt"})
    ip = os.path.join(pack, "model.safetensors.index.json")
    idx = json.load(open(ip))
    idx["weight_map"].update({n: "global.safetensors" for n in tens})
    json.dump(idx, open(ip, "w"))
    return hs


def test_validate_vision_tower_complete_passes(converted, tmp_path):
    import validate_pack
    d = _copy_pack(converted, tmp_path)
    _add_tower(d)
    rep = validate_pack.validate(d)
    assert not [e for e in rep.errors if "vision" in e or "aligner" in e or "image_" in e], rep.errors[:5]
    from exllamav3.dsv41 import vision_spec as VS
    assert rep.stats.get("vision_tensors") == 2 * 8 + 10
