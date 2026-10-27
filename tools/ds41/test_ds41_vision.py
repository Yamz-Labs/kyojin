"""DS4.1 image input on CPU: config, preprocessing and prompt layout against the vendored official processor, the
tower against the official ViT/Aligner (tiny random weights), splice and cache identity, the server over the stub
engine, and a tiny checkpoint served end to end with an image."""
from __future__ import annotations

import asyncio
import base64
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "tests" / "dsv41"))
import ds41_chat as chat  # noqa: E402
import ds41_engine as eng  # noqa: E402
import ds41_testkit as kit  # noqa: E402
from exllamav3.dsv41 import vision as V  # noqa: E402
from exllamav3.dsv41 import vision_spec as VS  # noqa: E402

REF = HERE / "testdata" / "reference"
IMG_TOKEN = "<｜deepseek_image｜>"
serve = kit.serve_module()
OFF_IP = kit.load_module("ds41_official_image_processor", REF / "inference" / "image_processor.py")
OFF_VIS = kit.load_module("ds41_official_vision", REF / "inference" / "vision.py")

PACK_CFG = {"image_token_id": 129264, "text_config": {"hidden_size": 5120}, "vision_config": {
    "model_type": "deepseek_v41_vision", "num_hidden_layers": 32, "hidden_size": 1024, "num_attention_heads": 16,
    "intermediate_size": 2816, "patch_size": 14, "rope_theta": 10000, "downsample_ratio": 3, "max_image_tokens": 1024,
    "min_pixels": 295936, "max_wh_ratio": None}}


def png(w, h, seed=0) -> bytes:
    g = torch.Generator().manual_seed(seed)
    px = (torch.rand(h, w, 3, generator=g) * 255).to(torch.uint8).numpy()
    buf = io.BytesIO()
    Image.fromarray(px).save(buf, "PNG")
    return buf.getvalue()


def data_url(raw: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(raw).decode()


def official_args(v: VS.VisionConfig) -> SimpleNamespace:
    return SimpleNamespace(vision_patch_size=v.patch_size, vision_downsample_ratio=v.downsample_ratio,
                           vision_max_n_token=v.max_n_token, vision_min_pixels=v.min_pixels,
                           vision_max_wh_ratio=v.max_wh_ratio, vision_dim=v.dim, vision_n_heads=v.n_heads,
                           vision_inter_dim=v.inter_dim, vision_n_layers=v.n_layers, vision_rope_theta=v.rope_theta,
                           dim=v.hidden_size)


# ------------------------------------------------------------------------------------------------ config
def test_config_of_the_1017_pack():
    v = VS.read_vision_config(PACK_CFG)
    assert (v.n_layers, v.dim, v.n_heads, v.inter_dim, v.patch_size) == (32, 1024, 16, 2816, 14)
    assert (v.downsample_ratio, v.max_n_token, v.min_pixels, v.max_wh_ratio) == (3, 1024, 295936, None)
    assert v.image_token_id == 129264 and v.hidden_size == 5120 and v.head_dim == 64
    names = VS.tower_tensor_names(v)
    assert len(names) == 266 and len(set(names)) == 266
    assert VS.expected_tower_shapes(v)["aligner.w1.weight"] == (5120, 9216)


def test_config_flat_official_keys_and_no_tower():
    flat = {"dim": 4096, "vision_n_layers": 2, "vision_dim": 64, "vision_n_heads": 4, "vision_inter_dim": 96,
            "vision_max_n_token": 384, "vision_max_wh_ratio": 8, "image_token_id": 77}
    v = VS.read_vision_config(flat)
    assert (v.n_layers, v.dim, v.max_n_token, v.max_wh_ratio, v.image_token_id, v.hidden_size) == (2, 64, 384, 8, 77, 4096)
    assert VS.read_vision_config({"text_config": {}}) is None
    assert VS.read_vision_config({"vision_n_layers": 0}) is None


def test_validate_pack_knows_the_tower_names():
    assert VS.is_tower_tensor("vision.blocks.3.attn.wo.bias") and VS.is_tower_tensor("image_newline")
    assert not VS.is_tower_tensor("layers.3.attn.wo.bias") and not VS.is_tower_tensor("embed.weight")


# ------------------------------------------------------------------------------------------------ preprocessing
SIZES = [(16, 16), (100, 60), (640, 480), (1500, 1100), (200, 3000), (3000, 200), (277, 301), (1024, 1024), (1, 1)]


@pytest.mark.parametrize("w,h", SIZES)
def test_preprocess_matches_official(w, h):
    v = VS.read_vision_config(PACK_CFG)
    raw = png(w, h, seed=w + h)
    mine = V.prepare_image({"url": data_url(raw)}, v)
    patches, nvh, nvw, nlh, nlw = OFF_IP.load_image({"url": data_url(raw)}, official_args(v))
    assert (mine.n_vit_h, mine.n_vit_w, mine.n_llm_h, mine.n_llm_w) == (nvh, nvw, nlh, nlw)
    assert torch.equal(mine.patches, patches.flatten(1))
    types = OFF_IP.image_token_types(nlh, nlw)
    assert torch.equal(mine.types, types) and mine.n_tokens == types.numel()
    assert mine.n_llm_h * mine.n_llm_w <= v.max_n_token


def test_grids_with_vit_size_not_a_multiple_of_the_merge():
    v = VS.read_vision_config(PACK_CFG)
    odd = [(w, h) for w, h in SIZES if (VS.plan_image_grid(w, h, v)[2] // 14) % 3 or (VS.plan_image_grid(w, h, v)[3] // 14) % 3]
    assert odd, "no size exercises the padded merge"


def test_same_size_different_pixels_different_keys():
    v = VS.read_vision_config(PACK_CFG)
    a, b = (V.prepare_image({"url": data_url(png(300, 300, s))}, v) for s in (1, 2))
    assert a.n_tokens == b.n_tokens and a.key != b.key
    assert a.key == V.prepare_image({"url": data_url(png(300, 300, 1))}, v).key


def test_loading_rules():
    v = VS.read_vision_config(PACK_CFG)
    raw = png(40, 40)
    assert V.prepare_image({"data": base64.b64encode(raw).decode()}, v).n_tokens > 0
    with pytest.raises(V.ImageError):
        V.prepare_image({"url": "/etc/passwd"}, v)                       # local paths are refused
    with pytest.raises(V.ImageError):
        V.prepare_image({"url": "data:image/png;base64,AAAA"}, v)        # not an image
    with pytest.raises(V.ImageError):
        V.prepare_image({"url": "ftp://x/y.png"}, v)
    assert V.image_record({"type": "image_url", "image_url": {"url": "u"}}) == {"url": "u"}
    assert V.image_record({"type": "image", "source": {"type": "base64", "data": "QQ=="}}) == {"data": "QQ=="}
    with pytest.raises(V.ImageError):
        V.image_record({"type": "image_url", "image_url": {}})


# ------------------------------------------------------------------------------------------------ prompt layout
def test_official_image_fixture_renders_the_same_prompt():
    data = json.loads((REF / "tests" / "test_input_5.json").read_text())
    expected = (REF / "tests" / "test_output_5.txt").read_text()
    prompt, records = chat.render_chat_vl(data["messages"], data.get("tools"), thinking=data["thinking_mode"] == "thinking",
                                          reasoning_effort=data.get("reasoning_effort"))
    assert prompt == expected and prompt.count(IMG_TOKEN) == 2
    assert [r["url"] for r in records] == ["examples/images/carrots.jpeg", "examples/images/corn.jpeg"]


def test_text_only_render_refuses_images_without_collector():
    with pytest.raises(chat.ChatFormatError, match="image"):
        chat.render_chat([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}])


def test_image_order_follows_the_rendered_prompt_for_tool_results():
    i = lambda u: {"type": "image_url", "image_url": {"url": u}}          # noqa: E731
    msgs = [{"role": "user", "content": [{"type": "text", "text": "go"}, i("u0")]},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "f", "arguments": "{}"}},
                {"id": "b", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "b", "content": [i("ub")]},
            {"role": "tool", "tool_call_id": "a", "content": [i("ua")]}]
    tools = [{"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object", "properties": {}}}}]
    prompt, records = chat.render_chat_vl(msgs, tools)
    assert prompt.count(IMG_TOKEN) == 3
    assert [r["url"] for r in records] == ["u0", "ua", "ub"]              # call order, like the reference


def test_expand_and_locate_and_cache_key():
    v = VS.read_vision_config(PACK_CFG)
    a, b = (V.prepare_image({"url": data_url(png(300, 300, s))}, v) for s in (1, 2))
    tid = v.image_token_id
    ids, located = V.expand_image_tokens([5, tid, 6, tid, 7], [a, b], tid)
    assert located == [(1, a), (2 + a.n_tokens, b)] and len(ids) == 3 + a.n_tokens + b.n_tokens
    assert V.locate_spans(ids, [a, b], tid) == located
    with pytest.raises(V.ImageError):
        V.expand_image_tokens([5, tid], [a, b], tid)
    with pytest.raises(V.ImageError):
        V.locate_spans(ids[:-3], [a, b], tid)
    k1 = V.prompt_key(ids, located)
    k2 = V.prompt_key(ids, [(1, b), located[1]])
    assert k1[0] == 5 and all(x < 0 for x in k1[1:1 + a.n_tokens]) and k1 != k2 and len(set(k1[1:1 + a.n_tokens])) == a.n_tokens
    assert V.prompt_key(ids, None) == ids


# ------------------------------------------------------------------------------------------------ tower
def tiny_vcfg(**kw) -> VS.VisionConfig:
    d = dict(n_layers=2, dim=64, n_heads=4, inter_dim=96, patch_size=14, downsample_ratio=3, max_n_token=64,
             min_pixels=2000, hidden_size=48, image_token_id=500)
    d.update(kw)
    return VS.VisionConfig(**d)


def tiny_tower_weights(v: VS.VisionConfig, seed=0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    out = {}
    for n, shape in VS.expected_tower_shapes(v).items():
        if n.endswith("norm.weight") or ".norm1." in n or ".norm2." in n:
            t = 1 + 0.1 * torch.randn(shape, generator=g)
        elif n.endswith(".bias") or n.startswith("image_"):
            t = 0.1 * torch.randn(shape, generator=g)
        else:
            t = torch.randn(shape, generator=g) * (shape[-1] ** -0.5)
        out[n] = t.to(torch.bfloat16)
    return out


def official_tower(v, w):
    args = official_args(v)
    vit, ali = OFF_VIS.ViT(args), OFF_VIS.Aligner(args)
    vit.load_state_dict({k[len("vision."):]: t.float() for k, t in w.items() if k.startswith("vision.")})
    ali.load_state_dict({k[len("aligner."):]: t.float() for k, t in w.items() if k.startswith("aligner.")})
    return vit.eval(), ali.eval()


@pytest.mark.parametrize("w,h", [(100, 80), (290, 130), (60, 200)])
def test_tower_matches_official_in_fp32(w, h):
    v = tiny_vcfg()
    wt = tiny_tower_weights(v)
    tower = V.VisionTower(v, wt, "cpu", torch.float32)
    img = V.prepare_image({"url": data_url(png(w, h, 3))}, v)
    vit, ali = official_tower(v, wt)
    with torch.no_grad():
        x = vit(img.patches.float(), img.n_vit_h, img.n_vit_w)
        want = ali(x, img.n_vit_h, img.n_vit_w)
    got = tower.encode(img)
    assert got.shape == want.shape == (img.n_llm_h * img.n_llm_w, v.hidden_size)
    assert torch.allclose(tower.encode_vit(img.patches, img.n_vit_h, img.n_vit_w), x, atol=2e-4, rtol=1e-3)
    assert torch.allclose(got, want, atol=2e-4, rtol=1e-3)
    rows = tower.span_rows(img)
    t = img.types
    assert rows.shape == (img.n_tokens, v.hidden_size)
    assert torch.equal(rows[t == VS.IMAGE_START][0], wt["image_start"].float())
    assert torch.equal(rows[t == VS.IMAGE_END][0], wt["image_end"].float())
    assert torch.equal(rows[t == VS.IMAGE_NEW_LINE][0], wt["image_newline"].float())
    assert torch.equal(rows[t == VS.IMAGE], got)


def test_tower_query_blocks_equal_one_pass():
    v = tiny_vcfg()
    tower = V.VisionTower(v, tiny_tower_weights(v), "cpu", torch.float32)
    img = V.prepare_image({"url": data_url(png(200, 160, 5))}, v)
    a = tower.encode_vit(img.patches, img.n_vit_h, img.n_vit_w)
    tower.Q_BLOCK = 7
    assert torch.allclose(tower.encode_vit(img.patches, img.n_vit_h, img.n_vit_w), a, atol=1e-5)


def test_tower_loads_from_a_pack_folder(tmp_path):
    from safetensors.torch import save_file
    v = tiny_vcfg()
    wt = tiny_tower_weights(v)
    cfg = {"image_token_id": 500, "text_config": {"hidden_size": 48}, "vision_config": {
        "num_hidden_layers": 2, "hidden_size": 64, "num_attention_heads": 4, "intermediate_size": 96, "patch_size": 14,
        "downsample_ratio": 3, "max_image_tokens": 64, "min_pixels": 2000, "max_wh_ratio": None}}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    assert V.vision_available(tmp_path) is None
    save_file(wt, str(tmp_path / "global.safetensors"))
    got = V.vision_available(tmp_path)
    assert got is not None and got.n_layers == 2
    tower = V.VisionTower.load(tmp_path, device="cpu", dtype=torch.float32)
    assert set(tower.w) == set(wt)
    # the same tensors split over shards through the index file
    half = sorted(wt)[:100]
    save_file({k: wt[k] for k in half}, str(tmp_path / "a.safetensors"))
    save_file({k: wt[k] for k in wt if k not in half}, str(tmp_path / "b.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        k: "a.safetensors" if k in half else "b.safetensors" for k in wt}}))
    assert V.vision_available(tmp_path) is not None
    del wt["vision.norm.weight"]
    save_file({k: t for k, t in wt.items()}, str(tmp_path / "global.safetensors"))
    (tmp_path / "model.safetensors.index.json").unlink()
    assert V.vision_available(tmp_path) is None                     # one missing tensor: no half-working tower


# ------------------------------------------------------------------------------------------------ splice
def spans_of(starts_lens, hidden=8):
    return [V.ImageSpan(s, torch.arange(n * hidden, dtype=torch.float32).view(n, hidden) + 100 * i, f"k{i}")
            for i, (s, n) in enumerate(starts_lens)]


def test_span_mask_and_splice_over_chunks():
    spans = spans_of([(3, 5), (12, 4)])
    full = torch.zeros(1, 20, 8)
    m_all = V.splice(full, 0, spans)
    assert m_all.nonzero().flatten().tolist() == [3, 4, 5, 6, 7, 12, 13, 14, 15]
    assert torch.equal(full[0, 3:8], spans[0].rows) and torch.equal(full[0, 12:16], spans[1].rows)
    chunked = torch.zeros(1, 20, 8)
    for a in range(0, 20, 6):                                       # a span crosses the chunk edges
        h = torch.zeros(1, min(6, 20 - a), 8)
        m = V.splice(h, a, spans)
        chunked[:, a:a + h.size(1)] = h
        ref = V.span_mask(spans, a, h.size(1))
        assert (m is None) == (ref is None) and (m is None or torch.equal(m, ref))
    assert torch.equal(chunked, full)
    assert V.splice(torch.zeros(1, 2, 8), 0, spans) is None and V.span_mask(None, 0, 4) is None
    assert V.span_mask(spans, 8, 4) is None


# ------------------------------------------------------------------------------------------------ server, stub engine
def run(coro):
    return asyncio.run(coro)


def vl_message(url, text="what is this?"):
    return [{"role": "user", "content": [{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": url}}]}]


async def with_client(engine, fn):
    from aiohttp.test_utils import TestClient, TestServer
    client = TestClient(TestServer(serve.create_app(engine, "ds41-test", None)))
    await client.start_server()
    try:
        return await fn(client)
    finally:
        await client.close()


def test_stub_server_images_cache_and_health():
    a, b = data_url(png(300, 300, 1)), data_url(png(300, 300, 2))

    async def go(client):
        h = await (await client.get("/health")).json()
        assert h["vision"] is True
        out = []
        for url in (a, a, b):
            r = await client.post("/v1/chat/completions", json={"model": "ds41-test", "messages": vl_message(url), "max_tokens": 4})
            assert r.status == 200, await r.text()
            out.append(await r.json())
        n = V.prepare_image({"url": a}, VS.VisionConfig()).n_tokens
        total = out[0]["usage"]["prompt_tokens"]
        assert total > n and out[0]["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
        assert out[1]["usage"]["prompt_tokens_details"]["cached_tokens"] == total          # same image: all reused
        assert out[2]["usage"]["prompt_tokens"] == total
        assert out[2]["usage"]["prompt_tokens_details"]["cached_tokens"] <= total - n      # same size, other pixels: text only
        r = await client.post("/v1/chat/completions", json={"model": "ds41-test", "messages": vl_message(a), "max_tokens": 4, "stream": True})
        assert r.status == 200 and "data: [DONE]" in await r.text()
        r = await client.post("/v1/chat/completions", json={"model": "ds41-test", "messages": vl_message("/etc/passwd")})
        assert r.status == 400
        r = await client.post("/v1/chat/completions", json={"model": "ds41-test", "messages": vl_message("data:image/png;base64,AAAA")})
        assert r.status == 400

    run(with_client(eng.StubEngine("ok", vision=True), go))


def test_stub_server_without_vision_refuses_images():
    async def go(client):
        h = await (await client.get("/health")).json()
        assert h["vision"] is False
        r = await client.post("/v1/chat/completions", json={"model": "ds41-test", "messages": vl_message(data_url(png(40, 40)))})
        assert r.status == 400

    run(with_client(eng.StubEngine("ok"), go))


# ------------------------------------------------------------------------------------------------ tiny model, end to end
@pytest.fixture(scope="module")
def vl_ckpt(tmp_path_factory):
    from safetensors.torch import save_file
    from tiny_ckpt import fw_config, write_tiny
    d = tmp_path_factory.mktemp("ds41_tiny_vl")
    write_tiny(str(d), seed=4, cfg=fw_config(index_n_heads=16))
    v = tiny_vcfg(hidden_size=128, image_token_id=500, max_n_token=36)
    cfg = json.loads((d / "config.json").read_text())
    cfg["image_token_id"] = 500
    cfg["vision_config"] = {"num_hidden_layers": 2, "hidden_size": 64, "num_attention_heads": 4, "intermediate_size": 96,
                            "patch_size": 14, "downsample_ratio": 3, "max_image_tokens": 36, "min_pixels": 2000,
                            "max_wh_ratio": None}
    (d / "config.json").write_text(json.dumps(cfg))
    save_file(tiny_tower_weights(v, 9), str(d / "vision.safetensors"))
    (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        **{n: "vision.safetensors" for n in VS.tower_tensor_names(v)}}}))
    return str(d)


async def _collect(engine, ids, images, n=8):
    stats, pieces = {}, []
    async for p in engine.generate("", max_tokens=n, temperature=0.0, top_p=1.0, stop=[], prompt_ids=ids, ignore_eos=True,
                                   stats=stats, images=images):
        pieces.append(p)
    return "".join(pieces), stats


def test_tiny_model_image_changes_output_and_cache_is_per_image(vl_ckpt):
    import ds41_model as dm
    engine = eng.load_model(vl_ckpt, 512, loader="ds41_model:load_model", vision="true", vision_device="cpu",
                            vision_dtype="float32")
    assert isinstance(engine, dm.PortEngine) and engine.supports_vision
    imgs = engine.prepare_images([{"url": data_url(png(60, 60, s))} for s in (1, 2)])
    assert imgs[0].n_tokens == imgs[1].n_tokens
    head, tail = [11, 40, 7], [90, 23, 5]

    def ids_for(img):
        return head + [500] * img.n_tokens + tail

    t1, s1 = run(_collect(engine, ids_for(imgs[0]), [imgs[0]]))
    assert s1["images"] == 1 and s1["cached_tokens"] == 0
    n1 = len(ids_for(imgs[0]))
    _, s1b = run(_collect(engine, ids_for(imgs[0]) + [4, 4], [imgs[0]]))              # a longer prompt: the prefix is reused
    assert s1b["cached_tokens"] == n1 and s1b["images"] == 1                         # snapshots work on the plain model too
    t2, s2 = run(_collect(engine, ids_for(imgs[1]) + [4, 4], [imgs[1]]))
    assert s2["cached_tokens"] <= len(head)                                           # same ids, other pixels: no reuse
    # the text-only run of the same ids (image positions as plain token 500) is another computation
    plain = eng.load_model(vl_ckpt, 512, loader="ds41_model:load_model", vision="false")
    tp, _ = run(_collect(plain, ids_for(imgs[0]), None))
    assert not plain.supports_vision
    assert len({t1, t2, tp}) >= 2
    with pytest.raises(ValueError):
        run(_collect(plain, ids_for(imgs[0]), [imgs[0]]))
    # fresh engine, same image: same output as the cached run (the splice is deterministic)
    again = eng.load_model(vl_ckpt, 512, loader="ds41_model:load_model", vision="true", vision_device="cpu",
                           vision_dtype="float32")
    t1c, _ = run(_collect(again, ids_for(imgs[0]), [imgs[0]]))
    assert t1c == t1


def test_tiny_model_forward_with_spans_differs_and_is_deterministic(vl_ckpt):
    engine = eng.load_model(vl_ckpt, 512, loader="ds41_model:load_model", vision="true", vision_device="cpu",
                            vision_dtype="float32")
    img = engine.prepare_images([{"url": data_url(png(60, 60, 1))}])[0]
    ids = [11, 40, 7] + [500] * img.n_tokens + [90, 23, 5]
    spans = engine._spans([(3, img)])
    assert spans[0].start == 3 and spans[0].rows.shape == (img.n_tokens, 128)
    m = engine.model

    def logits(**kw):
        st = m.new_state(1, 512)
        out, _ = m.forward(torch.tensor([ids]), 0, st, **kw)
        return out[-1]

    plain, a, b = logits(), logits(spans=spans), logits(spans=spans)
    assert torch.equal(a, b) and not torch.allclose(a, plain)
    # rows that are the model's own embedding of the same tokens still differ: image rows route with the VL bias and
    # carry no engram, so the splice is more than an embedding swap
    emb = m.embed(torch.tensor([ids]))[0] if hasattr(m, "embed") else None
    if emb is not None and emb.shape[-1] == 128:
        own = [V.ImageSpan(3, emb[3:3 + img.n_tokens].float(), "own")]
        assert not torch.allclose(logits(spans=own), plain)


# ------------------------------------------------------------------------------------------------ token ids vs the official path
def test_prompt_ids_and_token_types_equal_the_official_prepare_vl_inputs(monkeypatch):
    """Official: encode_messages + prepare_vl_inputs. Ours: render_chat_vl + prepare_image + expand_image_tokens, with the
    1017 tokenizer. Two images (a tall one and a wide one) in one user turn, then a second user turn."""
    path = kit.tokenizer_json()
    if path is None:
        pytest.skip("tokenizer.json not available (set DS41_REF_DIR)")
    from tokenizers import Tokenizer
    raw = Tokenizer.from_file(str(path))
    v = VS.read_vision_config(PACK_CFG)
    monkeypatch.setitem(sys.modules, "encoding", kit.official())         # image_processor.py imports `encoding`

    class Tk:                                    # the surface prepare_vl_inputs reads
        unk_token_id = None

        def encode(self, text):
            return raw.encode(text, add_special_tokens=False).ids

        def convert_tokens_to_ids(self, tok):
            return raw.token_to_id(tok)
    blobs = [png(300, 900, 1), png(1300, 700, 2)]
    urls = [data_url(b) for b in blobs]
    messages = [
        {"role": "system", "content": "You describe pictures."},
        {"role": "user", "content": [{"type": "text", "text": "Compare these:"},
                                     {"type": "image_url", "image_url": {"url": urls[0]}},
                                     {"type": "text", "text": "and"},
                                     {"type": "image_url", "image_url": {"url": urls[1]}}]},
        {"role": "assistant", "content": "The first is tall, the second wide."},
        {"role": "user", "content": "Thanks."},
    ]
    off_args = official_args(VS.read_vision_config(PACK_CFG))
    off_args.image_token_id, off_args.vision_enabled = v.image_token_id, True
    for thinking in (False, True):
        off_prompt = kit.reference_prompt(messages, None, thinking)
        prompt, records = chat.render_chat_vl(messages, None, thinking=thinking, reasoning_effort=None)
        assert prompt == off_prompt, thinking
        imgs = [V.prepare_image(r, v) for r in records]
        ours, located = V.expand_image_tokens(raw.encode(prompt, add_special_tokens=False).ids, imgs, v.image_token_id)
        toks, types, inputs = OFF_IP.prepare_vl_inputs(off_prompt, [{"data": b} for b in blobs], Tk(), off_args)
        assert list(toks) == ours, thinking
        assert [i.start for i in inputs] == [s for s, _ in located]
        assert [int(i.types.numel()) for i in inputs] == [im.n_tokens for _, im in located]
        assert sum(1 for t in types if t != OFF_IP.TEXT) == sum(im.n_tokens for im in imgs)
