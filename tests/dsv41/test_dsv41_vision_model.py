"""Image input in the model, CPU fp32, against the OFFICIAL reference (vendored model.py with the vision tower):
  * plain graph (DSV41Model) with an image span == official Transformer.forward(images=, token_types=): logits and
    main_hidden of the prefill, then greedy decode steps
  * serving model (DSV41Serving) == plain graph, one pass and chunked prefill (a span crossing chunk edges)
  * negative controls: the same prompt without the span, and with the span but no VL routing / engram masking, differ
Image rows come from the repo tower (exllamav3.dsv41.vision) fed with the same random weights as the official ViT/Aligner.
"""
import dataclasses
import io
import os
import sys
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pathlib import Path
import harness
from harness import FakeTok, build_reference, make_engram_tables, make_state_dict, tiny_config
from exllamav3.dsv41 import vision as V
from exllamav3.dsv41 import vision_spec as VS
from exllamav3.dsv41.engram import EngramLayout, MemoryEngramStore, build_compressed_token_map
from exllamav3.dsv41.experts import SourceExpertProvider
from exllamav3.dsv41.model import DSV41Model
from exllamav3.dsv41.serve_model import DSV41Serving
from exllamav3.dsv41.weights import DictWeightSource

IMG_ID = 500
TOL = 1e-5


def tower_cfg(hidden):
    return VS.VisionConfig(n_layers=2, dim=64, n_heads=4, inter_dim=96, patch_size=14, downsample_ratio=3, max_n_token=36,
                           min_pixels=2000, hidden_size=hidden, image_token_id=IMG_ID)


def tower_weights(v, seed=11):
    g = torch.Generator().manual_seed(seed)
    out = {}
    for n, shape in VS.expected_tower_shapes(v).items():
        if n.endswith("norm.weight") or ".norm1." in n or ".norm2." in n:
            t = 1 + 0.1 * torch.randn(shape, generator=g)
        elif n.endswith(".bias") or n.startswith("image_"):
            t = 0.3 * torch.randn(shape, generator=g)
        else:
            t = torch.randn(shape, generator=g) * (shape[-1] ** -0.5)
        out[n] = t.to(torch.bfloat16).float()
    return out


def png(w, h, seed):
    g = torch.Generator().manual_seed(seed)
    buf = io.BytesIO()
    Image.fromarray((torch.rand(h, w, 3, generator=g) * 255).to(torch.uint8).numpy()).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def world():
    cfg = tiny_config(index_n_heads=16, dspark_block_size=5, dspark_noise_token_id=511)
    tok = FakeTok(cfg.vocab_size)
    tmap, nv = build_compressed_token_map(tok)
    cfg.engram_compressed_vocab_size = nv
    lay = EngramLayout.build(cfg.engram_layer_ids, None, cfg.engram_max_ngram_size, cfg.engram_vocab_size,
                             cfg.engram_n_heads, cfg.engram_head_dim)
    cfg.engram_num_embeddings = list(lay.num_embeddings)
    tables = make_engram_tables(cfg, lay.num_embeddings, 2)
    sd = make_state_dict(cfg, 1, fp8=False, fp4=False)
    v = tower_cfg(cfg.hidden_size)
    tw = tower_weights(v)
    sd_ref = dict(sd)
    sd_ref.update(tw)                                       # the official model names: vision.*, aligner.*, image_*
    # the official ModelArgs of the harness + the vision fields
    orig = harness.ref_args

    def args_with_vision(*a, **k):
        return dataclasses.replace(orig(*a, **k), vision_n_layers=2, vision_dim=64, vision_n_heads=4, vision_inter_dim=96,
                                   vision_patch_size=14, vision_downsample_ratio=3, vision_max_n_token=36,
                                   vision_min_pixels=2000, image_token_id=IMG_ID)
    harness.ref_args = args_with_vision
    try:
        ref, missing = build_reference(cfg, sd_ref, tables, tok, 256, 1, False, False)
    finally:
        harness.ref_args = orig
    assert not missing, missing
    src = DictWeightSource(sd, cfg.num_hidden_layers)
    n = cfg.num_hidden_layers + cfg.n_mtp_layers
    prov = SourceExpertProvider(src, {l: cfg.moe_counts(l)[0] for l in range(n)})
    store = MemoryEngramStore([t[0] for t in tables], [t[1] for t in tables])
    mine = DSV41Model(cfg, src, prov, store, token_map=tmap)
    srv = DSV41Serving(cfg, DictWeightSource(sd, cfg.num_hidden_layers), None, store, token_map=tmap, dtype=torch.float32)
    return cfg, v, tw, ref, mine, srv


def image_case(cfg, v, tw, head=12, tail=10, size=(200, 150), seed=3):
    """ids, official (images, token_types), repo spans for one image between `head` and `tail` text tokens."""
    import image_processor as IP                              # the official one, put on sys.path by load_reference()
    raw = png(*size, seed)
    args = SimpleNamespace(vision_patch_size=14, vision_downsample_ratio=3, vision_max_n_token=36, vision_min_pixels=2000,
                           vision_max_wh_ratio=None)
    patches, nvh, nvw, nlh, nlw = IP.load_image({"data": raw}, args)
    types = IP.image_token_types(nlh, nlw)
    g = torch.Generator().manual_seed(seed)
    txt = torch.randint(0, 400, (head + tail,), generator=g)
    ids = torch.cat([txt[:head], torch.full((types.numel(),), IMG_ID), txt[head:]]).unsqueeze(0)
    tt = torch.full(ids.shape, IP.TEXT)
    tt[0, head:head + types.numel()] = types
    images = [[IP.ImageInput(head, patches.float(), nvh, nvw, types)]]      # fp32 reference tower
    img = V.preprocess(raw, v)
    assert img.n_tokens == types.numel()
    tower = V.VisionTower(v, tw, "cpu", torch.float32)
    spans = [V.ImageSpan(head, tower.span_rows(img), img.key)]
    return ids, images, tt, spans


def test_plain_graph_with_an_image_equals_the_official_model(world):
    cfg, v, tw, ref, mine, srv = world
    ids, images, tt, spans = image_case(cfg, v, tw)
    S = ids.size(1)
    assert 25 < S < 60
    with torch.inference_mode():
        rids, rl, rm = ref(ids, 0, images=images, token_types=tt)
        st = mine.new_state(1, 256)
        ml, mm = mine.forward(ids, 0, st, spans=spans)
        assert (rl - ml).abs().max().item() <= TOL, (rl - ml).abs().max().item()
        assert (rm - mm).abs().max().item() <= TOL
        assert torch.equal(rids, ml.argmax(-1))
        # decode after the image: text only
        nxt = rids.view(1, 1)
        for i in range(6):
            rids, rl, rm = ref(nxt, S + i)
            ml, mm = mine.forward(nxt, S + i, st)
            assert (rl - ml).abs().max().item() <= TOL, i
            assert torch.equal(rids, ml.argmax(-1))
            nxt = rids.view(1, 1)


def test_the_span_matters_and_so_do_vl_routing_and_engram_masking(world):
    cfg, v, tw, ref, mine, srv = world
    ids, images, tt, spans = image_case(cfg, v, tw)
    with torch.inference_mode():
        good = mine.forward(ids, 0, mine.new_state(1, 256), spans=spans)[0]
        plain = mine.forward(ids, 0, mine.new_state(1, 256))[0]
        assert (good - plain).abs().max() > 1e-2                      # image rows are not the token embeddings
        base = ref(ids, 0, images=images, token_types=tt)[1]
        assert (base - good).abs().max().item() <= TOL
        # VL routing off (bias_vl := bias) on the reference: the logits must move
        saved = [l.ffn.gate.bias_vl.data.clone() for l in ref.layers]
        for l in ref.layers:
            l.ffn.gate.bias_vl.data = l.ffn.gate.bias.data.clone()
        no_vl = ref(ids, 0, images=images, token_types=tt)[1]
        for l, s_ in zip(ref.layers, saved):
            l.ffn.gate.bias_vl.data = s_
        assert (no_vl - base).abs().max() > 1e-4, "VL routing bias does not change the logits: the test cannot see it"
        # engram masking off (image positions hashed and gated like text) on the reference: the logits must move
        eh = ref.engram_hash
        eh.forward = lambda ids_, sp, m=None, o=eh.forward: o(ids_, sp, None)
        wrapped = []
        for l in ref.layers:
            if l.engram is not None:
                l.engram.forward = lambda h, hs, m=None, o=l.engram.forward: o(h, hs, None)
                wrapped.append(l.engram)
        no_eg = ref(ids, 0, images=images, token_types=tt)[1]
        del eh.forward
        for e in wrapped:
            del e.forward
        assert (no_eg - base).abs().max() > 1e-4, "engram masking does not change the logits: the test cannot see it"
        again = ref(ids, 0, images=images, token_types=tt)[1]
        assert torch.equal(again, base)


def test_serving_model_with_an_image_equals_the_plain_graph(world):
    cfg, v, tw, ref, mine, srv = world
    ids, images, tt, spans = image_case(cfg, v, tw)
    S = ids.size(1)
    with torch.inference_mode():
        a = mine.new_state(1, 256)
        la, ha = mine.forward(ids, 0, a, spans=spans)
        b = srv.new_state(1, 256)
        lb, hb = srv.forward(ids, 0, b, chunk=S, spans=spans)
        assert (la - lb).abs().max().item() <= 2e-4 * la.abs().max().item()
        assert torch.equal(la.argmax(-1), lb.argmax(-1))
        # decode on the filled cache: same greedy tokens as the plain graph
        ta = tb = la.argmax(-1)[-1:].view(1, 1)
        for i in range(5):
            xa, _ = mine.forward(ta, S + i, a)
            xb, _ = srv.forward(tb, S + i, b)
            assert (xa - xb).abs().max().item() <= 2e-4 * xa.abs().max().item(), i
            ta, tb = xa.argmax(-1)[-1:].view(1, 1), xb.argmax(-1)[-1:].view(1, 1)
            assert torch.equal(ta, tb), i


def test_chunked_prefill_splices_the_same_rows_at_the_same_positions(world):
    """Chunk edges inside the span (7, 16: spans start at 12 and end at 38). What reaches layer 0 is exactly the one-pass input;
    the logits agree to the noise of the simulated fp8 activation grids (a boundary flip of a power-of-two scale moves a few
    cached values; text-only chunking sits at 5e-6, a row near such a boundary moves the logits by up to a few percent of their range in this 64-wide toy)."""
    cfg, v, tw, ref, mine, srv = world
    ids, images, tt, spans = image_case(cfg, v, tw)
    S = ids.size(1)
    rec = []
    orig = srv._attention

    def hook(l, x, start, st, rt, taps):
        if l == 0:
            rec.append((start, x.clone()))
        return orig(l, x, start, st, rt, taps)
    srv._attention = hook
    try:
        with torch.inference_mode():
            one = srv.forward(ids, 0, srv.new_state(1, 256), chunk=S, spans=spans)[0]
            whole = rec.pop()[1]
            for chunk in (7, 16, 23):
                rec.clear()
                lg = srv.forward(ids, 0, srv.new_state(1, 256), chunk=chunk, spans=spans)[0]
                assert [s for s, _ in rec] == list(range(0, S, chunk))
                assert torch.equal(torch.cat([x for _, x in rec], 1), whole), chunk
                assert (lg - one).abs().max().item() <= 6e-2 * one.abs().max().item(), chunk
    finally:
        srv._attention = orig


def test_chunked_prefill_with_the_activation_grids_off_matches_one_pass_tightly(world, monkeypatch):
    """The few-percent chunk gap above is the fp8/fp4 activation grids amplifying ~1e-7 noise: with them off, every chunking
    (edges inside the span, on its first/last row, and a size that does not divide the prompt) agrees to the fp32 noise."""
    from exllamav3.dsv41 import quant
    cfg, v, tw, ref, mine, srv = world
    ids, images, tt, spans = image_case(cfg, v, tw)
    S = ids.size(1)
    monkeypatch.setattr(quant, "act_quant_fp8_", lambda x, *a, **k: x)
    monkeypatch.setattr(quant, "fp4_act_quant_", lambda x, *a, **k: x)
    with torch.inference_mode():
        one = srv.forward(ids, 0, srv.new_state(1, 256), chunk=S, spans=spans)[0]
        for chunk in (5, 7, 12, 16, 19, 23, 38):
            lg = srv.forward(ids, 0, srv.new_state(1, 256), chunk=chunk, spans=spans)[0]
            assert (lg - one).abs().max().item() <= 1e-4 * one.abs().max().item(), chunk


def test_serving_prefill_api_takes_spans(world):
    cfg, v, tw, ref, mine, srv = world
    ids, images, tt, spans = image_case(cfg, v, tw, size=(100, 60), seed=5)
    S = ids.size(1)
    with torch.inference_mode():
        one = srv.forward(ids, 0, srv.new_state(1, 256), spans=spans)[0]
        same = srv.prefill(ids, srv.new_state(1, 256), 0, chunk=S, spans=spans)[0]
        assert torch.equal(one, same)
        chunked = srv.prefill(ids, srv.new_state(1, 256), 0, chunk=9, spans=spans)[0]
        assert (one - chunked).abs().max().item() <= 6e-2 * one.abs().max().item()
        no_img = srv.prefill(ids, srv.new_state(1, 256), 0, chunk=9)[0]
        assert (chunked - no_img).abs().max() > 1e-2                      # the spans reached the chunked path


# ------------------------------------------------------------------------------- the production path: PortEngine on a serving model
def _engine(world, **kw):
    import asyncio
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "ds41"))
    import ds41_model as dm
    cfg, v, tw, ref, mine, srv = world

    class Tk:
        def encode(self, text, add_bos=False):
            return [1]

        def decode(self, ids, skip_special_tokens=False):
            return " ".join(f"t{i}" for i in ids)
    e = dm.PortEngine(srv, Tk(), ctx=200, vision_dir="unused", vision_cfg=v, vision_device="cpu", vision_dtype="float32", **kw)
    e._tower = V.VisionTower(v, tw, "cpu", torch.float32)
    return e, asyncio


def _ask(asyncio, e, ids, images, n=10):
    stats, out = {}, []

    async def go():
        async for p in e.generate("", max_tokens=n, temperature=0.0, top_p=1.0, stop=[], prompt_ids=ids, ignore_eos=True,
                                  stats=stats, images=images):
            out.append(p)
    asyncio.run(go())
    return "".join(out), stats


def _prepared(e, v, seed, size=(200, 150)):
    return e.prepare_images([{"data": png(*size, seed)}])[0]


def test_engine_on_a_serving_model_reuses_the_prefix_per_image(world):
    cfg, v, tw, ref, mine, srv = world
    e, asyncio = _engine(world)
    a, b = _prepared(e, v, 3), _prepared(e, v, 4)
    assert a.n_tokens == b.n_tokens and a.key != b.key
    head, tail = [11, 40, 7], [90, 23, 5]
    ids = head + [IMG_ID] * a.n_tokens + tail
    t1, s1 = _ask(asyncio, e, ids, [a])
    assert s1["cached_tokens"] == 0 and s1["images"] == 1
    t2, s2 = _ask(asyncio, e, ids + [4, 4], [a])                       # strict extension, same pixels
    assert s2["cached_tokens"] == len(ids)
    fresh, _ = _engine(world)
    t2f, s2f = _ask(asyncio, fresh, ids + [4, 4], [a])                 # no cache: same text
    assert s2f["cached_tokens"] == 0 and t2f == t2
    _ask(asyncio, e, ids, [a])
    t3, s3 = _ask(asyncio, e, ids + [4, 4], [b])                       # same ids, other pixels
    assert s3["cached_tokens"] <= len(head)
    t3f, _ = _ask(asyncio, _engine(world)[0], ids + [4, 4], [b])
    assert t3 == t3f


def test_engine_with_dspark_draft_equals_plain_greedy_on_an_image(world):
    cfg, v, tw, ref, mine, srv = world
    plain, asyncio = _engine(world)
    spec, _ = _engine(world, num_draft=3)
    img = _prepared(plain, v, 3)
    ids = [11, 40, 7] + [IMG_ID] * img.n_tokens + [90, 23, 5]
    tp, _ = _ask(asyncio, plain, ids, [img], n=24)
    ts, st = _ask(asyncio, spec, ids, [img], n=24)
    assert tp == ts and st["draft_steps"] > 0
