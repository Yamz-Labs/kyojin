"""The DS4.1 port behind the server: `ds41_model.load_model` on a tiny random official-layout checkpoint, with and
without `--num-draft`, through the engine contract and through the HTTP layer. CPU only (torch, tokenizers, safetensors)."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("safetensors")
pytest.importorskip("tokenizers")
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for p in (str(HERE), str(REPO), str(REPO / "tests" / "dsv41")):
    if p not in sys.path:
        sys.path.insert(0, p)
import ds41_engine as eng  # noqa: E402
import ds41_model as dm  # noqa: E402
import ds41_testkit as kit  # noqa: E402
from tiny_ckpt import fw_config, write_tiny  # noqa: E402

serve = kit.serve_module()
MODEL = "ds41-port-test"


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    d = tmp_path_factory.mktemp("ds41_tiny")
    # 16 index heads: with few, indexer scores tie at 0 and top-k breaks ties differently for different tensor shapes
    cfg = fw_config(index_n_heads=16, dspark_block_size=5, dspark_noise_token_id=511)
    write_tiny(str(d), seed=4, cfg=cfg)
    from tokenizers import Tokenizer, models, pre_tokenizers
    vocab = Tokenizer.from_file(str(d / "tokenizer.json")).get_vocab()
    tk = Tokenizer(models.WordLevel(vocab, unk_token="the"))     # an id for every word, including "<r12>" pieces
    tk.pre_tokenizer = pre_tokenizers.Whitespace()               # text prompts of several tokens
    tk.save(str(d / "tokenizer.json"))
    return str(d)


def run(coro):
    return asyncio.run(coro)


async def collect(engine, ids, n, **kw):
    stats, pieces = {}, []
    async for piece in engine.generate("", max_tokens=n, temperature=kw.pop("temperature", 0.0), top_p=1.0, stop=[],
                                       prompt_ids=ids, ignore_eos=True, stats=stats, **kw):
        pieces.append(piece)
    return "".join(pieces), stats


IDS = [[11, 40, 7, 90, 23, 5, 88, 130, 17, 9, 301, 12], [200, 3, 77, 410, 8, 8, 61, 140]]


def test_loader_hook_and_plain_generation(ckpt):
    engine = eng.load_model(ckpt, 256, loader="ds41_model:load_model")
    assert isinstance(engine, dm.PortEngine) and engine.num_draft == 0 and engine.sessions == 1
    assert engine.vocab_size == 512 and engine.ctx == 256
    text, st = run(collect(engine, IDS[0], 12))
    assert text and st["new_tokens"] == 12 and st["prompt_tokens"] == len(IDS[0]) and st["cached_tokens"] == 0
    assert st["eos_reason"] == "max_new_tokens" and "accepted_draft_tokens" not in st
    assert st["time_prefill"] > 0 and engine.last_stats == st


def test_num_draft_greedy_output_equals_plain_and_reports_draft_stats(ckpt):
    plain = eng.load_model(ckpt, 256, loader="ds41_model:load_model")
    spec = eng.load_model(ckpt, 256, loader="ds41_model:load_model", num_draft=5)
    assert spec.num_draft == 5 and spec.spec is not None
    for ids in IDS:
        a, sa = run(collect(plain, ids, 40))
        b, sb = run(collect(spec, ids, 40))
        assert a == b and sa["new_tokens"] == sb["new_tokens"] == 40
        assert sb["draft_steps"] > 0 and sb["accepted_draft_tokens"] + sb["rejected_draft_tokens"] > 0
        assert 0 <= sb["accepted_per_step"] <= 5
    # a smaller draft window is the same output again
    two = eng.load_model(ckpt, 256, loader="ds41_model:load_model", num_draft=2)
    c, sc = run(collect(two, IDS[1], 40))
    assert c == b and max(sc["draft_len_hist"]) <= 2


def test_sampled_generation_runs_with_and_without_drafting(ckpt):
    for nd in (0, 3):
        engine = eng.load_model(ckpt, 256, loader="ds41_model:load_model", num_draft=nd, seed=3)
        text, st = run(collect(engine, IDS[0], 20, temperature=0.8))
        assert st["new_tokens"] == 20 and text


def test_cancel_stops_the_job(ckpt):
    engine = eng.load_model(ckpt, 256, loader="ds41_model:load_model", num_draft=3)

    async def go():
        cancel, stats, n = asyncio.Event(), {}, 0
        async for _ in engine.generate("", max_tokens=200, temperature=0.0, top_p=1.0, stop=[], prompt_ids=IDS[0],
                                       ignore_eos=True, cancel=cancel, stats=stats):
            n += 1
            cancel.set()
        return stats
    st = run(go())
    assert st["new_tokens"] < 200


def test_budget_keeps_room_for_the_draft_window(ckpt):
    engine = eng.load_model(ckpt, 40, loader="ds41_model:load_model", num_draft=5)
    ids = IDS[0]
    _, st = run(collect(engine, ids, 1000))
    assert st["new_tokens"] <= 40 - len(ids) - 1 - 5
    with pytest.raises(ValueError):
        run(collect(engine, list(range(60)), 4))


def test_num_draft_is_refused_where_it_cannot_work(ckpt):
    with pytest.raises(ValueError):
        eng.load_model(ckpt, 256, loader="ds41_model:load_model", num_draft=11)     # the cap is 10 (chained passes above the block size of 5)
    engine = eng.load_model(ckpt, 256, loader="ds41_model:load_model")
    engine.model.mtp = None
    with pytest.raises(ValueError):
        dm.PortEngine(engine.model, engine.tk, ctx=256, num_draft=2)


def test_http_timings_carry_the_draft_fields(ckpt):
    from aiohttp.test_utils import TestClient, TestServer
    engine = eng.load_model(ckpt, 256, loader="ds41_model:load_model", num_draft=5)

    async def go():
        client = TestClient(TestServer(serve.create_app(engine, MODEL)))
        await client.start_server()
        try:
            resp = await client.post("/completion", json={"prompt": "the cat sat on the mat go run hello world ok",
                                                          "n_predict": 24, "temperature": 0})
            assert resp.status == 200, await resp.text()
            return await resp.json()
        finally:
            await client.close()
    body = run(go())
    t = body["timings"]
    assert t["predicted_n"] >= 4 and t["draft_steps"] > 0 and "accepted_per_step" in t
    assert t["accepted_draft_tokens"] + t["rejected_draft_tokens"] > 0


# ---- sampler ---------------------------------------------------------------------------------------------------


def test_top_p_keeps_the_smallest_prefix_that_reaches_the_mass():
    s = dm.TopPSampler(1.0, 0.6)
    p = s.probs(torch.log(torch.tensor([0.2, 0.5, 0.3])))
    assert torch.allclose(p, torch.tensor([0.0, 0.5 / 0.8, 0.3 / 0.8]))
    assert dm.TopPSampler(1.0, 1.0).probs(torch.log(torch.tensor([0.2, 0.5, 0.3]))).sum().item() == pytest.approx(1.0)
    g = torch.Generator().manual_seed(0)
    s = dm.TopPSampler(1.0, 0.6, g)
    draws = [s.sample(torch.log(torch.tensor([0.2, 0.5, 0.3]))) for _ in range(400)]
    assert 0 not in draws and 1 in draws and 2 in draws
    assert dm.TopPSampler(0.0).sample(torch.tensor([0.1, 3.0, 2.0])) == 1 and not dm.TopPSampler(0.0).stochastic


def test_detokenizer_holds_back_partial_utf8():
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders
    vocab = {"<0xE2>": 0, "<0x82>": 1, "<0xAC>": 2, "a": 3}
    tk = Tokenizer(models.BPE(vocab, [], byte_fallback=True))
    tk.decoder = decoders.ByteFallback()
    d = dm._Detok(tk)
    assert d.push([3]) == "a"
    assert d.push([0]) == "" and d.push([1]) == ""
    assert d.push([2]) == "€"
