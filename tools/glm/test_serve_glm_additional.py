#!/usr/bin/env python3
"""Local additions to tools/glm/test_serve.py: the request log and the image cache budget.

These tests cover work that is not upstream (yet), so they live in a file of their own instead of patching
upstream's test_serve.py: a merge of upstream's file never conflicts with them, and they can be folded back
into test_serve.py in one move if the features are ever proposed upstream. The fake engine below is a copy of
the one in test_serve.py, deliberately: this file then runs on its own and does not break when upstream edits
its fixtures.

    kyojin/.venv/bin/python -m unittest tools.glm.test_serve_glm_additional       # 11 tests, no GPU, no model
    kyojin/.venv/bin/python -m pytest -q tools/glm/test_serve_glm_additional.py
"""
from __future__ import annotations

import asyncio
import base64
import io
import os
import pathlib
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import serve  # noqa: E402

TEMPLATE = ("{% for m in messages %}<|im_start|>{{ m.role }}: {{ m.content }}<|im_end|>{% endfor %}"
            "{% if tools %}TOOLS={{ tools|length }}{% endif %}")
# The same template with the image placeholders the server matches against the request's image parts.
IMAGE_TEMPLATE = TEMPLATE.replace(
    "{{ m.content }}",
    "{% if m.content is string %}{{ m.content }}{% else %}{% for p in m.content %}"
    "{% if p.type == 'text' %}{{ p.text }}{% else %}" + serve.IMAGE_TRIPLE + "{% endif %}{% endfor %}{% endif %}")


class FakeGenerator:
    num_draft_tokens = 2  # /metrics derives the round count from the engine's window


class FakeEngine:
    def __init__(self, output: str):
        self.output = output
        self.prompts = []
        self.greedy_generator = FakeGenerator()

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def prompt_ids(self, prompt: str):
        """The ids the route counts tokens with; the real engine has an incremental encode cache here.

        This double counts words, so the ids are placeholders of that length: the route asks the result for
        `numel()` and puts the number in the log, which is the same value it got before."""
        import torch
        return torch.zeros(int(self.count_tokens(prompt)), dtype=torch.long)

    async def generate(self, prompt: str, **kwargs):
        self.prompts.append(prompt)
        self.last_stats = {"cached_tokens": 8, "new_tokens": 4, "prompt_tokens": 20,
                           "time_prefill": 0.5, "time_generate": 1.0,
                           "accepted_draft_tokens": 6, "rejected_draft_tokens": 6}
        for i in range(0, len(self.output), 3):  # small chunks cut tags in half
            yield self.output[i:i + 3]


def run(coro):
    return asyncio.run(coro)


def vision_engine():
    """A fake engine whose generate() runs the real _embed() on a real cache, plus a tower that counts encodes."""
    import itertools

    import torch
    from image_cache import ImageEmbeddingCache

    ids, calls = itertools.count(1_000_000_000, 100), []

    class Emb:
        def __init__(self):
            self.first_index = next(ids)
            self.text_alias = f"<$EMB_{self.first_index}$>"
            self.mm_length = 3
            self.embeddings = torch.zeros(3, 64, dtype=torch.half)
            self.deepstack_embeddings = None

    class Tower:
        def get_image_embeddings(self, *args, **kw):
            calls.append(1)
            return Emb()

    eng = object.__new__(serve.ResidentEngine)
    eng.emb_cache, eng.vision, eng.tokenizer = ImageEmbeddingCache(), Tower(), None
    return eng, calls


def run_log_file(directory) -> Path:
    """The file the last run wrote in a directory: `requests-<run>.log`, one file per run."""
    directory = Path(directory)
    if directory.suffix:                      # a file name handed in: its directory is what is used
        directory = directory.parent
    written = sorted(directory.glob("requests-*.log"))
    return written[-1] if written else directory / "requests-none.log"


def disk_engine(directory, budget_bytes: int = 768, mode: str = "fp32", cap_bytes: int = 0, on_disk: bool = True,
                lock_s: float = 0.0, policy: str = "tail", fingerprint: str = ""):
    """An engine whose _embed() runs over a real memory cache *and* a real second-level cache on disk.

    The memory cache runs the `tail` policy the disk layer is built for: keeping the head of a prompt is what
    lets a restored picture complete it. The server's own default is the shared cache's LRU (`head`).

    The fake pictures carry everything a restore needs (identity, token list, MRoPE metadata) so a test can ask
    whether the prompt would be reproduced: unlike the tower, the disk has to give back the same pseudo ids.
    """
    import itertools

    import torch
    from image_cache import ImageEmbeddingCache

    ids, calls = itertools.count(1_000_000_000, 100), []

    class Emb:
        def __init__(self):
            self.first_index = next(ids)
            self.mm_length = 3
            self.last_index = self.first_index + self.mm_length
            self.embeddings = torch.zeros(3, 64, dtype=torch.half)
            self.deepstack_embeddings = None
            self.grid_thw = (1, 1, 1)
            self.mrope_merge_size = 2
            self.metadata = {"original_size": (8, 8)}
            self.token_string = torch.tensor([[7] + list(range(self.first_index, self.last_index)) + [8]])
            self.token_list = self.token_string[0].tolist()
            self.text_alias = f"<$EMB_{self.first_index}$>"

    class Tower:
        def get_image_embeddings(self, *args, **kw):
            calls.append(1)
            return Emb()

    eng = object.__new__(serve.ResidentEngine)
    eng.emb_cache = ImageEmbeddingCache(max_bytes=budget_bytes, policy=policy)
    eng.vision, eng.tokenizer = Tower(), None
    eng.image_cache_format = mode
    eng.emb_disk = None
    if on_disk:
        from image_cache_disk import ImageEmbeddingDisk
        eng.emb_disk = ImageEmbeddingDisk(directory, max_bytes=cap_bytes, start_token=7, end_token=8,
                                          lock_s=lock_s, env_fingerprint=fingerprint)
    return eng, calls


def tiny_pictures(count: int):
    """Whatever `_embed` needs: a PIL image and its content key."""
    from PIL import Image
    return [(Image.new("RGB", (8, 8), (i * 30 % 255, 0, 0)), f"key{i}") for i in range(count)]


class ImageDiskCacheTests(unittest.TestCase):
    """--image-cache-dir: what leaves memory is written once, and comes back with its identity."""

    def test_the_disk_cache_is_a_flag_not_an_environment_variable(self):
        self.assertNotIn("EXL3_IMAGE_CACHE_DIR", os.environ)
        self.assertNotIn("EXL3_IMAGE_CACHE_MB", os.environ)

    def test_the_tenure_reaches_the_disk_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            eng, _ = disk_engine(tmp, budget_bytes=384, lock_s=42.0)
            self.assertEqual(eng.emb_disk.lock_s, 42.0)
            eng, _ = disk_engine(tmp, budget_bytes=384, lock_s=0.0)
            self.assertEqual(eng.emb_disk.lock_s, 0.0)

    def test_only_what_memory_gave_up_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            eng, calls = disk_engine(tmp, budget_bytes=768)        # room for two of the four pictures
            images = tiny_pictures(4)
            serve.ResidentEngine._embed(eng, images)
            self.assertEqual(len(calls), 4)                        # a cold pass encodes all four
            self.assertEqual(len(list(pathlib.Path(tmp).glob("*.pt"))), 2)   # ... and two were evicted
            self.assertEqual(eng.emb_disk.report()["stored"], 2)

            serve.ResidentEngine._embed(eng, images)               # same conversation again
            self.assertEqual(len(calls), 4)                        # two from memory, two from disk: no vision pass
            self.assertEqual(eng.emb_disk.report()["loaded"], 2)

    def test_a_restored_picture_keeps_the_ids_the_prompt_was_built_from(self):
        with tempfile.TemporaryDirectory() as tmp:
            eng, calls = disk_engine(tmp, budget_bytes=384)        # room for one
            images = tiny_pictures(2)
            first = serve.ResidentEngine._embed(eng, images)
            self.assertEqual(len(list(pathlib.Path(tmp).glob("*.pt"))), 1)
            self.assertEqual(first[1].embeddings.shape, (3, 64))   # the second picture went to disk

            again = serve.ResidentEngine._embed(eng, images)
            self.assertEqual(len(calls), 2)                        # no second vision pass
            self.assertEqual(again[1].first_index, first[1].first_index)
            self.assertEqual(again[1].token_list, first[1].token_list)
            self.assertEqual(again[1].text_alias, first[1].text_alias)
            self.assertEqual(again[1].grid_thw, first[1].grid_thw)

    def test_without_a_directory_nothing_is_written_and_the_tower_runs_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            eng, calls = disk_engine(tmp, budget_bytes=768, on_disk=False)
            images = tiny_pictures(4)
            serve.ResidentEngine._embed(eng, images)
            serve.ResidentEngine._embed(eng, images)
            self.assertEqual(list(pathlib.Path(tmp).rglob("*.pt")), [])
            self.assertEqual(len(calls), 6)                        # 4 cold, then the 2 that memory dropped

    def test_a_batch_that_does_not_fit_writes_nothing_at_all(self):
        # What matters for wear: a conversation that cannot be cached should cost *no* bytes at all, rather than
        # being written and deleted again on the next request.
        with tempfile.TemporaryDirectory() as tmp:
            eng, calls = disk_engine(tmp, budget_bytes=768, cap_bytes=1)     # room for nothing
            serve.ResidentEngine._embed(eng, tiny_pictures(4))
            report = eng.emb_disk.report()
            self.assertEqual(report["files"], 0)
            self.assertEqual(report["written_bytes"], 0)
            self.assertEqual(report["dropped"], 0)
            self.assertGreater(report["skipped"], 0)
            self.assertEqual(list(pathlib.Path(tmp).glob("*.pt")), [])

    def test_every_picture_is_written_once_when_it_fits(self):
        with tempfile.TemporaryDirectory() as tmp:
            eng, calls = disk_engine(tmp, budget_bytes=768, cap_bytes=1 << 20)
            serve.ResidentEngine._embed(eng, tiny_pictures(4))
            first = eng.emb_disk.report()
            self.assertEqual((first["stored"], first["dropped"], first["skipped"]), (2, 0, 0))
            serve.ResidentEngine._embed(eng, tiny_pictures(4))               # the same two are evicted again
            again = eng.emb_disk.report()
            self.assertEqual(again["stored"], 2)                            # already on disk: not rewritten
            self.assertEqual(again["written_bytes"], first["written_bytes"])
            self.assertEqual(again["dropped"], 0)

    def test_an_unusable_directory_does_not_stop_a_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            blocker = pathlib.Path(tmp) / "blocker"
            blocker.write_text("not a directory")
            eng, calls = disk_engine(str(blocker), budget_bytes=384)
            self.assertFalse(eng.emb_disk.enabled)
            out = serve.ResidentEngine._embed(eng, tiny_pictures(2))
            self.assertEqual(len(out), 2)
            self.assertEqual(len(calls), 2)

    def test_the_model_fingerprint_is_stable_and_setting_specific(self):
        """The name of every cached row comes from here, so it has to be stable and to notice a change.

        A fingerprint that moved on its own would make the disk tier unreachable and turn every restart into
        re-encoding; one that ignored a setting would hand back rows the new setting never produced. Built
        from a synthetic pack here, so the check does not need the 93 GB one to be on the machine."""
        import json
        import tempfile
        from serve import env_fingerprint_source, make_env_fingerprint

        class Pre:
            patch_size, merge_size, temporal_patch_size = 14, 2, 2
            min_image_tokens, max_image_tokens = 16, 8000
            spatial_merge_size, hidden_size = 2, 1024

        class Config:
            vision_config, vision_pp = Pre(), Pre()

        def pack(directory: pathlib.Path, name: str, tokens: int = 8000) -> str:
            """A pack is its directory name, its configuration and the size/date of the visual shards."""
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "model.safetensors.index.json").write_text(json.dumps(
                {"weight_map": {"model.visual.blocks.0.w": f"{name}.safetensors"}}))
            (directory / f"{name}.safetensors").write_bytes(b"x" * (1024 if tokens == 8000 else 2048))
            config = Config()
            config.vision_pp = type("Pre", (Pre,), {"max_image_tokens": tokens})()
            return make_env_fingerprint(str(directory), config)

        with tempfile.TemporaryDirectory() as tmp:
            base = pathlib.Path(tmp)
            first = pack(base / "pack-a", "shard")
            self.assertEqual(first, pack(base / "pack-a", "shard"))               # same pack, same name
            self.assertNotEqual(first, pack(base / "pack-b", "shard"))            # another pack directory
            self.assertNotEqual(first, pack(base / "pack-a", "shard", tokens=7999))   # another canvas budget

            source = env_fingerprint_source(str(base / "pack-a"), Config())
            for field in ("vision_pp.patch_size", "vision_pp.merge_size", "vision_pp.max_image_tokens"):
                self.assertIn(field, source)
            self.assertEqual(source["model"], "pack-a")
            self.assertTrue(any(k.startswith("shard.") for k in source))          # the weights are in it
            json.dumps(source, sort_keys=True)                                    # and it can be logged

    def test_the_defaults_match_upstream(self):
        # The memory budget is upstream's 1 GiB, and the disk cache is off unless a cap is given: no auto share
        # of RAM, and no way to ask for an unbounded directory.
        from image_cache import DEFAULT_MAX_ITEMS
        self.assertEqual(serve.IMAGE_CACHE_MB_DEFAULT, 1024)
        self.assertEqual(serve.make_image_cache(None).max_bytes, 1024 << 20)
        # The count cap defaults to the shared cache's, so the three servers behave the same out of the box.
        self.assertEqual(serve.IMAGE_CACHE_ITEMS_DEFAULT, DEFAULT_MAX_ITEMS)
        self.assertEqual(serve.make_image_cache(None).max_items, DEFAULT_MAX_ITEMS)
        self.assertEqual(serve.make_image_cache(None, 0).max_items, 0)          # 0 = no limit, when asked for
        self.assertEqual(serve.IMAGE_CACHE_DIR_MAX_GB, 0.0)
        self.assertTrue(serve.IMAGE_CACHE_DIR_DEFAULT.endswith(".cache/kyojin/image-embeddings"))
        self.assertEqual(serve.IMAGE_CACHE_DIR_LOCK_S, 300.0)

    def test_the_disk_rows_are_kept_per_environment(self):
        """A row is only reusable while the model and the preprocessing stay the same.

        The file name carries the fingerprint of what produced it, so a restart with another pack or other
        pixel settings cannot reach the old rows; they stay on disk until the cap takes them, the same
        treatment as any other unusable file."""
        with tempfile.TemporaryDirectory() as tmp:
            eng, _ = disk_engine(tmp, budget_bytes=384, cap_bytes=64 << 20, fingerprint="aaaa")   # room for one
            images = tiny_pictures(2)
            serve.ResidentEngine._embed(eng, images)
            names = sorted(p.name for p in pathlib.Path(tmp).glob("*.pt"))
            self.assertEqual(len(names), 1)
            self.assertTrue(names[0].startswith("aaaa."), names)         # the fingerprint names the file
            self.assertEqual(eng.emb_disk.report()["env"], "aaaa")

            other, calls = disk_engine(tmp, budget_bytes=384, cap_bytes=64 << 20, fingerprint="bbbb")
            images = tiny_pictures(2)
            serve.ResidentEngine._embed(other, images)
            self.assertEqual(len(calls), 2)                              # nothing of ours is reachable
            self.assertEqual(len(list(pathlib.Path(tmp).glob("*.pt"))), 2)

            again, calls = disk_engine(tmp, budget_bytes=384, cap_bytes=64 << 20, fingerprint="aaaa")  # a restart
            serve.ResidentEngine._embed(again, images)
            self.assertEqual(len(calls), 1)                              # the second picture came back
            self.assertEqual(len(list(pathlib.Path(tmp).glob("*.pt"))), 2)   # and nothing was written again

    def test_a_row_from_another_environment_is_refused_rather_than_injected(self):
        """The name keeps another environment out. The copy inside the file is the second line: a file that
        got there some other way (copied in, restored from a backup, a name collision) is refused and
        dropped, rather than having its rows injected into a prompt as if this environment had made them."""
        with tempfile.TemporaryDirectory() as tmp:
            eng, _ = disk_engine(tmp, budget_bytes=384, cap_bytes=64 << 20, fingerprint="aaaa")
            serve.ResidentEngine._embed(eng, tiny_pictures(2))
            path = next(pathlib.Path(tmp).glob("*.pt"))
            import torch
            blob = torch.load(path, weights_only=True)
            blob["env_fingerprint"] = "bbbb"                      # the contents say another environment
            torch.save(blob, path)

            disk = eng.emb_disk
            self.assertIsNone(disk.fetch(path.name.split(".")[1], "fp32"))   # refused, not injected
            self.assertEqual(disk.stats["errors"], 1)
            self.assertFalse(path.exists())                       # and dropped, so it is encoded fresh

    def test_the_fingerprint_is_not_mistaken_for_the_picture_key(self):
        """Every part of the disk cache that names a picture reads the key out of the file name - the
        eviction grouping, the tenure map, the set of pictures present. With the fingerprint in front, "the
        first part" is the fingerprint, which would make every file of one environment look like the same
        picture: the grouping would then free whatever it walks over instead of the pictures the cap is meant
        to give up, and the files of the environment being served would be sacrificed for the files of an
        old one. The disk layer's whole point is that a conversation keeps the head of its prompt, so the
        reading is pinned here."""
        import tempfile
        from image_cache_disk import content_key
        self.assertEqual(content_key(pathlib.Path("81a20c24930c549a.abcdef.q8.pt")), "abcdef")
        self.assertEqual(content_key(pathlib.Path("abcdef.fp32.pt")), "abcdef")          # before the prefix

        with tempfile.TemporaryDirectory() as tmp:
            # room for one picture in memory, so the other two are written to disk
            eng, _ = disk_engine(tmp, budget_bytes=384, cap_bytes=64 << 20, fingerprint="aaaa")
            serve.ResidentEngine._embed(eng, tiny_pictures(3))
            disk = eng.emb_disk
            written = sorted(p.name for p in pathlib.Path(tmp).glob("*.pt"))
            self.assertEqual(len(written), 2, written)
            # the keys of the files this cache holds: two pictures, not one fingerprint twice
            self.assertEqual(disk._present(), {"key1", "key2"})
            # ...and the same reading for a directory written before the fingerprint existed
            for entry in pathlib.Path(tmp).glob("aaaa.*.pt"):
                entry.rename(entry.with_name(entry.name.split(".", 1)[1]))
            self.assertEqual(disk._present(), {"key1", "key2"})

    def test_a_write_that_was_killed_leaves_a_tmp_the_next_start_removes(self):
        """One picture is written to `<name>.tmp` and moved onto its name, so a `.pt` is always complete.

        A process that dies mid-write leaves the `.tmp`, which nothing else looks at: not the cap, not the
        set of pictures present, not the eviction order. It is removed on the next start, but only once it
        is old enough to be certainly nobody's live write - another process could be writing it, and the
        servers cannot rule that out from here."""
        import tempfile
        import time
        from image_cache_disk import STALE_TMP_S, ImageEmbeddingDisk
        with tempfile.TemporaryDirectory() as tmp:
            stale = pathlib.Path(tmp) / "aaaa.deadbeef.q8.tmp"
            stale.write_bytes(b"x" * 4096)
            long_ago = time.time() - STALE_TMP_S - 60
            os.utime(stale, (long_ago, long_ago))
            fresh = pathlib.Path(tmp) / "aaaa.cafebabe.q8.tmp"
            fresh.write_bytes(b"y" * 2048)

            disk = ImageEmbeddingDisk(tmp, max_bytes=64 << 20, start_token=7, end_token=8,
                                      env_fingerprint="aaaa")
            self.assertFalse(stale.exists())                     # the leftover of a killed write
            self.assertTrue(fresh.exists())                      # young: possibly somebody's live write
            self.assertEqual(disk.report()["stale_tmp"], 1)
            self.assertEqual(disk.report()["files"], 0)          # a tmp was never one of the pictures

            os.utime(fresh, (long_ago, long_ago))
            again = ImageEmbeddingDisk(tmp, max_bytes=64 << 20, start_token=7, end_token=8,
                                       env_fingerprint="aaaa")
            self.assertFalse(fresh.exists())
            self.assertEqual(again.report()["stale_tmp"], 1)

    def test_the_disk_cache_needs_a_cap(self):
        # `--image-cache-dir-max-gb 0` (the default) is off; any positive value is a real cap, never "unlimited".
        self.assertIsNone(serve.make_image_disk(serve.IMAGE_CACHE_DIR_DEFAULT, 0, 300.0))
        self.assertIsNone(serve.make_image_disk(None, 0.0, 300.0))
        with tempfile.TemporaryDirectory() as tmp:
            disk = serve.make_image_disk(tmp, 0.5, 60.0, start_token=7, end_token=8)
            self.assertEqual(disk.max_bytes, int(0.5 * 2 ** 30))
            self.assertEqual(disk.lock_s, 60.0)

    def test_flags_reach_the_cache(self):
        cache = serve.make_image_cache(1024 << 20, image_cache_items=7, sessions=3)
        self.assertEqual(cache.max_bytes, 1024 << 20)
        self.assertEqual(cache.max_items, 7)
        self.assertEqual(cache.max_orders, 3)                # one prompt order per session
        self.assertEqual(serve.make_image_cache(512 << 20, sessions=1).max_orders, 2)
        self.assertEqual(serve.make_image_cache(0, 0).max_bytes, 0)              # 0 = no bound
        self.assertEqual(serve.make_image_cache(0, 0).max_items, 0)
        self.assertEqual(serve.make_image_cache().max_bytes, 1024 << 20)         # engine default: 1 GiB

    def test_the_cache_the_engine_would_build_reports_its_budget(self):
        cache = serve.make_image_cache(serve.IMAGE_CACHE_MB_DEFAULT << 20, 0)
        self.assertEqual(cache.max_bytes, 1024 << 20)
        self.assertEqual(cache.max_items, 0)

    def test_the_cli_format_list_covers_every_engine_mode(self):
        from exllamav3.tokenizer.mm_embedding import MM_QUANT_MODES
        self.assertEqual(set(serve.IMAGE_CACHE_FORMATS), {"fp32", "fp16"} | set(MM_QUANT_MODES))

    def test_embed_stores_the_rows_in_the_requested_format(self):
        # --image-cache-format: fp32 and fp16 leave a plain tensor, q8 keeps the rows 8 bit. Either way the
        # object, its pseudo token ids and its length are the ones the prompt cache is keyed on, and the cache
        # counts the bytes of the form it actually holds (3 rows x 64 half values = 384 B as they are).
        from PIL import Image
        from exllamav3.tokenizer.mm_embedding import QuantizedMMRows
        from image_cache import embedding_nbytes
        eng, _ = vision_engine()
        img = Image.new("RGB", (8, 8))
        counts, packed = {}, {}
        for mode in serve.IMAGE_CACHE_FORMATS:
            eng.image_cache_format = mode
            eng.emb_cache.clear()
            emb = serve.ResidentEngine._embed(eng, [(img, f"key-{mode}")], None)[0]
            self.assertEqual(emb.mm_length, 3)
            self.assertEqual(emb.text_alias, f"<$EMB_{emb.first_index}$>")    # ids and alias untouched
            self.assertEqual(int(emb.embeddings.numel()), 192)                # logical values, not packed bytes
            packed[mode] = isinstance(emb.embeddings, QuantizedMMRows)
            counts[mode] = embedding_nbytes(emb)
        self.assertEqual(packed, {"fp32": False, "fp16": False, "q8": True})
        self.assertEqual(counts["fp32"], 384)                                  # as the tower emitted them
        self.assertEqual(counts["fp16"], 384)                                  # two bytes a value
        self.assertEqual(counts["q8"], 192 + 3 * 2)                            # one byte each + one scale per row


class RequestLogTests(unittest.TestCase):
    """KYOJIN_REQLOG_DIR: off unless asked for; when on, one line per lifecycle event, numbers and names only."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.log_dir = Path(self.tmp.name)          # the switch names a directory
        self.log_path = self.log_dir / "requests.log"   # only used to assert that nothing appears
        for key in ("KYOJIN_REQLOG_DIR", "KYOJIN_REQLOG_HEARTBEAT_S"):
            os.environ.pop(key, None)
            self.addCleanup(os.environ.pop, key, None)

    def current_log(self) -> Path:
        """The file this run wrote: `<stem>-<run><suffix>` beside the base path, one file per run."""
        written = sorted(self.log_dir.glob("requests-*.log"))
        return written[-1] if written else self.log_path

    def lines(self) -> list[str]:
        path = self.current_log()
        return path.read_text(encoding="utf-8").splitlines() if path.exists() else []

    def events(self) -> list[tuple[str, dict]]:
        """(event name, fields) per line; the timestamp is field 0 and the event name field 1."""
        out = []
        for line in self.lines():
            parts = line.split(" ")
            fields = {}
            for item in parts[2:]:
                key, _, value = item.partition("=")
                fields[key] = value.strip('"')
            out.append((parts[1], fields))
        return out

    def names(self) -> list[str]:
        return [name for name, _ in self.events()]

    def test_off_by_default_and_a_request_writes_nothing(self):
        from serve_reqlog import ReqLog
        log = ReqLog.from_env()
        self.assertFalse(log.enabled)
        self.assertIsNone(log.path)
        self.assertIn("request log off", log.describe())
        log.log("req_start", req="r1")               # disabled: a no-op, never an error
        self.assertEqual(log.lines(), [])

        from aiohttp.test_utils import TestClient, TestServer

        async def check():
            app = serve.create_app(FakeEngine("hello"), "m", TEMPLATE)
            self.assertFalse(app["reqlog"].enabled)
            client = TestClient(TestServer(app))
            await client.start_server()
            response = await client.post("/v1/chat/completions", json={
                "model": "m", "messages": [{"role": "user", "content": "hi"}]})
            status = response.status
            await response.read()
            await client.close()
            return status

        self.assertEqual(run(check()), 200)
        self.assertFalse(self.log_path.exists())      # nothing is created, nothing is written
        self.assertFalse(list(Path(self.tmp.name).iterdir()))

    def test_lifecycle_lines_share_one_id_and_never_hold_request_text(self):
        os.environ["KYOJIN_REQLOG_DIR"] = str(self.log_dir)
        secret, reply = "PROMPT-SECRET-7f3a", "REPLY-SECRET-9b1c"
        engine = FakeEngine(reply)
        from aiohttp.test_utils import TestClient, TestServer

        async def check():
            app = serve.create_app(engine, "m", TEMPLATE)
            self.assertTrue(app["reqlog"].enabled)
            client = TestClient(TestServer(app))
            await client.start_server()
            response = await client.post("/v1/chat/completions", json={
                "model": "m", "messages": [{"role": "user", "content": secret}]})
            body = await response.json()
            await client.close()
            return body

        body = run(check())
        self.assertEqual(body["choices"][0]["message"]["content"], reply)
        events = dict(self.events())
        for want in ("req_start", "req_queued", "job_start", "job_first_token", "job_done", "req_done"):
            self.assertIn(want, events)
        ids = {fields["req"] for name, fields in self.events() if "req" in fields}
        self.assertEqual(len(ids), 1)                       # every line of the request carries the same id
        self.assertEqual(events["req_queued"]["queued_ahead"], "0")
        self.assertEqual(events["job_done"]["cache_n"], "8")   # the engine's own cached_tokens
        self.assertEqual(events["job_done"]["cache_pct"], "40")
        self.assertIn(secret, " ".join(engine.prompts))     # the prompt really did carry the secret
        blob = "\n".join(self.lines())
        self.assertNotIn(secret, blob)
        self.assertNotIn(reply, blob)

    def test_stream_headers_first_content_and_done(self):
        os.environ["KYOJIN_REQLOG_DIR"] = str(self.log_dir)
        from aiohttp.test_utils import TestClient, TestServer

        async def check():
            client = TestClient(TestServer(serve.create_app(FakeEngine("hello"), "m", TEMPLATE)))
            await client.start_server()
            response = await client.post("/v1/chat/completions", json={
                "model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
            await response.read()
            await client.close()

        run(check())
        names = self.names()
        for want in ("sse_headers", "sse_first_content", "sse_done", "req_done"):
            self.assertIn(want, names)
        self.assertLess(names.index("sse_headers"), names.index("sse_first_content"))
        self.assertLess(names.index("sse_first_content"), names.index("sse_done"))

    def test_refusals_are_logged_without_quoting_the_body(self):
        os.environ["KYOJIN_REQLOG_DIR"] = str(self.log_dir)
        from aiohttp.test_utils import TestClient, TestServer

        async def check():
            client = TestClient(TestServer(serve.create_app(FakeEngine("hello"), "m", TEMPLATE)))
            await client.start_server()
            bad = await client.post("/v1/chat/completions", json={"model": "m"})          # no messages: 400
            broken = await client.post("/v1/chat/completions", data=b'{"model": "m", ',
                                       headers={"Content-Type": "application/json"})       # malformed JSON: 400
            await client.close()
            return bad.status, broken.status

        self.assertEqual(run(check()), (400, 400))
        refused = [fields for name, fields in self.events() if name == "req_refused"]
        self.assertEqual(len(refused), 2)
        self.assertEqual({f["status"] for f in refused}, {"400"})
        reasons = " ".join(f.get("reason", "") for f in refused)
        self.assertIn("JSONDecodeError(line=1", reasons)     # reduced to line/column, not the body text
        self.assertNotIn("model", reasons)

    def test_image_request_logs_cache_hits_misses_and_budget(self):
        os.environ["KYOJIN_REQLOG_DIR"] = str(self.log_dir)
        from PIL import Image
        from aiohttp.test_utils import TestClient, TestServer
        eng, _ = vision_engine()

        class VisionEngine(FakeEngine):
            supports_vision = True

            def image_tokens(self, image):
                return 3

            async def generate(self, prompt, images=None, img_stats=None, **kw):
                if images:
                    serve.ResidentEngine._embed(eng, images, img_stats)
                async for piece in FakeEngine.generate(self, prompt, **kw):
                    yield piece

        def uri(color):
            buf = io.BytesIO()
            Image.new("RGB", (8, 8), color).save(buf, "PNG")
            return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

        body = {"model": "m", "messages": [{"role": "user", "content": [
            {"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": uri((255, 0, 0))}},
            {"type": "image_url", "image_url": {"url": uri((0, 0, 255))}}]}]}

        async def check():
            client = TestClient(TestServer(serve.create_app(VisionEngine("hello"), "m", IMAGE_TEMPLATE)))
            await client.start_server()
            for _ in range(2):                                     # second send: both pictures are held
                response = await client.post("/v1/chat/completions", json=body)
                await response.read()
            await client.close()

        run(check())
        cache = [fields for name, fields in self.events() if name == "img_cache"]
        self.assertEqual(len(cache), 2)
        self.assertEqual((cache[0]["imgs"], cache[0]["hits"], cache[0]["misses"]), ("2", "0", "2"))
        self.assertEqual(cache[0]["first_missing"], "0")           # the prefix can only match from the start
        self.assertEqual((cache[0]["entries"], cache[0]["evicted"]), ("2", "0"))
        self.assertEqual(cache[0]["budget_mb"], "1024")            # the shared cache's own default (1 GiB)
        self.assertEqual(cache[0]["item_cap"], "32")            # the shared cache's count default
        self.assertEqual((cache[1]["hits"], cache[1]["misses"]), ("2", "0"))
        self.assertNotIn("first_missing", cache[1])                # a full hit has no gap
        self.assertNotIn("data:image", "\n".join(self.lines()))     # the image URL is never written

    def test_heartbeat_while_a_job_runs(self):
        os.environ["KYOJIN_REQLOG_DIR"] = str(self.log_dir)
        os.environ["KYOJIN_REQLOG_HEARTBEAT_S"] = "0.05"
        from aiohttp.test_utils import TestClient, TestServer

        class Slow(FakeEngine):
            async def generate(self, prompt, **kwargs):
                await asyncio.sleep(0.3)
                async for piece in FakeEngine.generate(self, prompt, **kwargs):
                    yield piece

        async def check():
            client = TestClient(TestServer(serve.create_app(Slow("hello"), "m", TEMPLATE)))
            await client.start_server()
            response = await client.post("/v1/chat/completions", json={
                "model": "m", "messages": [{"role": "user", "content": "hi"}]})
            await response.read()
            await client.close()

        run(check())
        alive = [fields for name, fields in self.events() if name == "alive"]
        self.assertTrue(alive, "no heartbeat lines")
        self.assertTrue(all("elapsed_s" in f and "phase" in f for f in alive))
        self.assertTrue({f["phase"] for f in alive} <= {"parse", "queued", "prefill", "decode", "stream"})

    def test_the_switch_names_a_directory_and_a_file_name_means_its_directory(self):
        """`KYOJIN_REQLOG_DIR` is a directory: an operator who points it at `.../requests.log` means the
        directory that file lives in, and that file itself is never written to."""
        import tempfile
        from serve_reqlog import ReqLog, log_dir_from_env
        with tempfile.TemporaryDirectory() as tmp:
            logs = Path(tmp) / "logs"
            logs.mkdir()
            cases = {
                str(logs): logs,                                  # a directory
                str(logs / "requests.log"): logs,                 # a file name: its directory
                "1": Path("~/.cache/kyojin").expanduser(),        # on, default directory
                "0": None,                                        # off
            }
            for raw, expected in cases.items():
                os.environ["KYOJIN_REQLOG_DIR"] = raw
                self.assertEqual(log_dir_from_env(), expected, raw)

            os.environ["KYOJIN_REQLOG_DIR"] = str(logs / "requests.log")
            log = ReqLog.from_env(server="glm")
            try:
                self.assertTrue(str(log.path).startswith(str(logs) + os.sep))
                self.assertTrue(log.path.name.startswith("requests-"))
                self.assertTrue(log.path.name.endswith(".log"))
            finally:
                log.close()

    def test_v1_completions_is_logged_too(self):
        os.environ["KYOJIN_REQLOG_DIR"] = str(self.log_dir)
        from aiohttp.test_utils import TestClient, TestServer

        async def check():
            client = TestClient(TestServer(serve.create_app(FakeEngine("hello"), "m", TEMPLATE)))
            await client.start_server()
            response = await client.post("/v1/completions", json={"model": "m", "prompt": "hi", "max_tokens": 4})
            await response.read()
            await client.close()
            return response.status

        self.assertEqual(run(check()), 200)
        queued = [fields for name, fields in self.events() if name == "req_queued"]
        self.assertEqual([f["path"] for f in queued], ["/v1/completions"])
        self.assertIn("job_done", self.names())


class SSEKeepAliveTests(unittest.TestCase):
    """KYOJIN_SSE_PING_S: a stream that has nothing to send yet writes a ':' comment (off by default)."""

    def setUp(self):
        os.environ.pop("KYOJIN_SSE_PING_S", None)
        self.addCleanup(os.environ.pop, "KYOJIN_SSE_PING_S", None)

    def test_a_ping_reaches_the_client_before_the_first_token(self):
        os.environ["KYOJIN_SSE_PING_S"] = "0.2"
        from aiohttp.test_utils import TestClient, TestServer

        class Slow(FakeEngine):
            async def generate(self, prompt, **kwargs):
                await asyncio.sleep(0.9)          # a prefill the client has to sit through
                async for piece in FakeEngine.generate(self, prompt, **kwargs):
                    yield piece

        async def check():
            client = TestClient(TestServer(serve.create_app(Slow("hello"), "m", TEMPLATE)))
            await client.start_server()
            response = await client.post("/v1/chat/completions", json={
                "model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
            chunks = []
            while True:
                try:
                    data = await asyncio.wait_for(response.content.readany(), 5)
                except (asyncio.IncompleteReadError, asyncio.TimeoutError):
                    break
                if not data:
                    break
                chunks.append(data)
            await client.close()
            return b"".join(chunks)

        body = run(check())
        self.assertTrue([line for line in body.split(b"\n") if line.startswith(b":")], body[:400])
        self.assertLess(body.index(b": waiting"), body.index(b'"content":"hel'))   # before the first token
        self.assertTrue(body.rstrip().endswith(b"data: [DONE]"))          # the stream still ends the OpenAI way

    def test_a_ping_covers_a_tool_call_the_server_holds_back(self):
        """A reply that writes a file is withheld until the XML closes, so the client hears nothing for
        the whole generation: that silence was aborting real agent turns, and it has to be commented."""
        os.environ["KYOJIN_SSE_PING_S"] = "0.2"
        from aiohttp.test_utils import TestClient, TestServer

        # GLM's own tool-call shape (tools/glm/serve.py TOOL_OPEN): a big <arg_value> is what a
        # "write this file" call looks like, and the whole block is held until </tool_call>.
        tool_call = ("<tool_call>\nwrite_file<arg_key>path</arg_key><arg_value>/out/notes.md</arg_value>\n"
                     "<arg_key>content</arg_key><arg_value>"
                     + ("One transcribed line.\n" * 30) + "</arg_value>\n</tool_call>")

        class Writer(FakeEngine):
            async def generate(self, prompt, **kwargs):
                for i in range(0, len(tool_call), 3):        # ~ a decode step per chunk, like the engine
                    await asyncio.sleep(0.02)
                    yield tool_call[i:i + 3]

        async def check():
            client = TestClient(TestServer(serve.create_app(Writer(""), "m", TEMPLATE)))
            await client.start_server()
            response = await client.post("/v1/chat/completions", json={
                "model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
            chunks = []
            while True:
                try:
                    data = await asyncio.wait_for(response.content.readany(), 5)
                except (asyncio.IncompleteReadError, asyncio.TimeoutError):
                    break
                if not data:
                    break
                chunks.append(data)
            await client.close()
            return b"".join(chunks)

        body = run(check())
        self.assertGreaterEqual(body.count(b": waiting"), 2, body[:400])   # not only before the first byte
        # the call itself still arrives whole, once, and the stream still ends the OpenAI way
        self.assertIn(b'"tool_calls"', body)                    # the held call still arrives, whole
        self.assertLessEqual(body.count(b"tool_calls"), 2)      # once as a message, once in usage
        self.assertTrue(body.rstrip().endswith(b"data: [DONE]"))

    def test_the_log_file_and_its_directory_are_owner_only(self):
        """The log names every request's size, timing and outcome: owner-only, whatever the umask is.

        Checked under a umask of 0, which is where a plain `open("a")` would leave 0666 in a 0777
        directory; a file that already exists with a wider mode is tightened on the next start."""
        import stat as _stat
        import tempfile
        from serve_reqlog import ReqLog, secure_dir
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name) / "logs" / "requests.log"
        previous = os.umask(0)
        try:
            log = ReqLog(path=str(base), server="test")
            log.log("req_start", req="r1")
            written = log.path                                   # this run's own file, not the base path
            log.close()
            self.assertEqual(_stat.S_IMODE(base.parent.stat().st_mode), 0o700)
            self.assertEqual(_stat.S_IMODE(written.stat().st_mode), 0o600)

            os.chmod(written, 0o666)                             # as an earlier run would have left the file
            os.chmod(base.parent, 0o775)                         # ...and the directory the deployment owns
            again = ReqLog(path=str(base), server="test")        # the next run: another file of its own
            again.close()
        finally:
            os.umask(previous)
        self.assertEqual(_stat.S_IMODE(base.parent.stat().st_mode), 0o775)   # left alone
        self.assertEqual(_stat.S_IMODE(written.stat().st_mode), 0o600)       # this one is ours

    def test_every_directory_this_creates_is_owner_only(self):
        """Not just the leaf: `mkdir(parents=True, mode=...)` leaves the directories in between at 0777
        (the C implementation does not pass the mode on), which the umask clearing below would expose."""
        import stat as _stat
        import tempfile
        from serve_reqlog import secure_dir
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        leaf = Path(tmp.name) / "a" / "b" / "c"
        previous = os.umask(0)
        try:
            secure_dir(leaf)
        finally:
            os.umask(previous)
        for made in (leaf, leaf.parent, leaf.parent.parent):
            self.assertEqual(_stat.S_IMODE(made.stat().st_mode), 0o700, made)
        os.chmod(leaf.parent, 0o775)                             # a directory that is not ours to change
        secure_dir(leaf)
        self.assertEqual(_stat.S_IMODE(leaf.parent.stat().st_mode), 0o775)

    def test_the_shared_cache_folder_is_created_the_way_the_health_history_creates_it(self):
        """Two modules create the shared folder; whichever runs first, it must come out the same.

        `tools/startup_health.py` makes it with a plain `mkdir(parents=True, exist_ok=True)` under the
        process umask, so this server must not make it owner-only instead: that would leave a different
        directory depending on which of the two started first."""
        import stat as _stat
        import tempfile
        from serve_reqlog import secure_dir, shared_cache_dir
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["KYOJIN_HEALTH_DIR"] = str(Path(tmp) / "ours")
            self.addCleanup(os.environ.pop, "KYOJIN_HEALTH_DIR", None)
            shared = shared_cache_dir()
            self.assertEqual(shared, (Path(tmp) / "ours").expanduser())     # the same expression

            def modes(umask: int) -> tuple[int, int]:
                """(ours, theirs) for one umask, each in a fresh parent so both start from nothing."""
                mine = Path(tmp) / "run-ours"
                os.environ["KYOJIN_HEALTH_DIR"] = str(mine)
                previous = os.umask(umask)
                try:
                    secure_dir(shared_cache_dir())                          # ours first
                    (Path(tmp) / "run-theirs" / "startup-glm.json").parent.mkdir(parents=True, exist_ok=True)
                finally:
                    os.umask(previous)
                return (_stat.S_IMODE(mine.stat().st_mode),
                        _stat.S_IMODE((Path(tmp) / "run-theirs").stat().st_mode))

            for umask in (0o022, 0o002, 0o000):
                ours, theirs = modes(umask)
                self.assertEqual(ours, theirs, f"umask {umask:04o}: {ours:04o} != {theirs:04o}")
            self.assertEqual(modes(0o022)[0], 0o755)                        # and it really follows the umask

            # the folders above it are created the same way
            with tempfile.TemporaryDirectory() as deep:
                os.environ["KYOJIN_HEALTH_DIR"] = str(Path(deep) / ".cache" / "kyojin")
                previous = os.umask(0o022)
                try:
                    secure_dir(shared_cache_dir())
                finally:
                    os.umask(previous)
                self.assertEqual(_stat.S_IMODE((Path(deep) / ".cache").stat().st_mode), 0o755)
                self.assertEqual(_stat.S_IMODE((Path(deep) / ".cache" / "kyojin").stat().st_mode), 0o755)
                # and every level is traversable by others, which is the point of the exception
                for level in (Path(deep) / ".cache", Path(deep) / ".cache" / "kyojin"):
                    self.assertTrue(_stat.S_IMODE(level.stat().st_mode) & 0o001, level)

    def test_the_health_folder_switch_names_the_shared_folder(self):
        """`KYOJIN_HEALTH_DIR` is what makes a folder the shared one, whatever it is called."""
        import stat as _stat
        import tempfile
        from serve_reqlog import secure_dir, shared_cache_dir
        with tempfile.TemporaryDirectory() as tmp:
            custom = Path(tmp) / "shared-state"
            os.environ["KYOJIN_HEALTH_DIR"] = str(custom)
            self.addCleanup(os.environ.pop, "KYOJIN_HEALTH_DIR", None)
            self.assertEqual(shared_cache_dir(), custom)
            previous = os.umask(0o022)
            try:
                secure_dir(shared_cache_dir())
            finally:
                os.umask(previous)
            self.assertEqual(_stat.S_IMODE(custom.stat().st_mode), 0o755)     # shared: the umask decides
            previous = os.umask(0)
            try:
                secure_dir(custom / "image-embeddings")                      # ours: owner-only
            finally:
                os.umask(previous)
            self.assertEqual(_stat.S_IMODE((custom / "image-embeddings").stat().st_mode), 0o700)
            self.assertEqual(_stat.S_IMODE(custom.stat().st_mode), 0o755)    # and not touched again

    def test_a_directory_of_ours_inside_the_shared_folder_is_owner_only(self):
        """The exception is the shared folder itself, not everything under it."""
        import stat as _stat
        import tempfile
        from serve_reqlog import secure_dir, shared_cache_dir
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["KYOJIN_HEALTH_DIR"] = str(Path(tmp) / "kyojin")
            self.addCleanup(os.environ.pop, "KYOJIN_HEALTH_DIR", None)
            shared = shared_cache_dir()
            previous = os.umask(0o022)
            try:
                secure_dir(shared)                                      # created the upstream way
                mode_before = _stat.S_IMODE(shared.stat().st_mode)
                secure_dir(shared / "image-embeddings")                 # ...and then our own directory
            finally:
                os.umask(previous)
            self.assertEqual(_stat.S_IMODE((shared / "image-embeddings").stat().st_mode), 0o700)
            self.assertEqual(_stat.S_IMODE(shared.stat().st_mode), mode_before)   # not touched again

    @staticmethod
    def upstream_style_dir(path: Path) -> Path:
        """What `tools/startup_health.py` does before writing its history file."""
        (path / "startup-glm.json").parent.mkdir(parents=True, exist_ok=True)
        return path

    def test_every_run_writes_its_own_log_and_old_runs_are_kept_by_count(self):
        """One file per run, so a run's lines can be read without filtering the other runs out.

        The switch names a base path; a run writes `<stem>-<run><suffix>` beside it and the newest
        `KYOJIN_REQLOG_KEEP` runs stay (checked once, at start-up). Files this module did not write - an
        archive, or the plain file an earlier layout appended to - are never deleted."""
        import stat as _stat
        import tempfile
        from serve_reqlog import ReqLog
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "requests.log"
            for i in range(23):                                  # 23 runs already on disk
                (Path(tmp) / f"requests-20260101T{i:06d}.log").write_text("old\n")
            (base).write_text("the file an earlier layout appended to\n")
            (Path(tmp) / "other-20260101T000000.log").write_text("another program\n")

            os.environ["KYOJIN_REQLOG_KEEP"] = "20"
            self.addCleanup(os.environ.pop, "KYOJIN_REQLOG_KEEP", None)
            log = ReqLog(path=str(base), server="test")
            try:
                self.assertNotEqual(log.path, base)              # not the shared file any more
                self.assertTrue(log.path.name.startswith("requests-"))
                self.assertTrue(log.path.name.endswith(".log"))
                log.log("req_start", req="r1")
                line = log.path.read_text().splitlines()[0]
                self.assertIn(f"run={log.run}", line)            # every line names its run
                self.assertIn("keeping 20 runs", log.describe())
            finally:
                log.close()
            kept = sorted(p.name for p in Path(tmp).glob("requests-*.log"))
            self.assertEqual(len(kept), 20)                      # exactly the retention, this run included
            self.assertEqual(kept[-1], log.path.name)            # the newest is this run's
            self.assertIn("requests-20260101T000022.log", kept)  # the newest of the old ones stays...
            self.assertNotIn("requests-20260101T000003.log", kept)   # ...and the oldest go
            self.assertNotIn("requests-20260101T000000.log", kept)
            self.assertEqual(log.pruned, 4)                      # 23 old + this one - 20 kept
            self.assertTrue((Path(tmp) / "other-20260101T000000.log").exists())

    def test_the_log_says_whether_the_server_was_writing(self):
        """The accounting that separates "the client showed nothing" from "the server sent nothing".

        A streaming reply logs `out_events`/`out_chars` per heartbeat and one `sse_summary` at the end;
        a held-back tool call is the case where the numbers stay flat while the engine decodes.
        """
        import tempfile
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        os.environ["KYOJIN_REQLOG_DIR"] = tmp.name                 # the switch names a directory
        os.environ["KYOJIN_REQLOG_HEARTBEAT_S"] = "0.2"
        self.addCleanup(os.environ.pop, "KYOJIN_REQLOG_DIR", None)
        self.addCleanup(os.environ.pop, "KYOJIN_REQLOG_HEARTBEAT_S", None)
        from aiohttp.test_utils import TestClient, TestServer

        class Slow(FakeEngine):
            async def generate(self, prompt, **kwargs):
                await asyncio.sleep(1.2)                 # heartbeats land while nothing is sent
                async for piece in FakeEngine.generate(self, prompt, **kwargs):
                    yield piece

        async def check():
            client = TestClient(TestServer(serve.create_app(Slow("hello"), "m", TEMPLATE)))
            await client.start_server()
            response = await client.post("/v1/chat/completions", json={
                "model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
            body = await response.read()
            await client.close()
            return body

        run(check())
        events = []
        for line in run_log_file(os.environ["KYOJIN_REQLOG_DIR"]).read_text(encoding="utf-8").splitlines():
            parts = line.split(" ")
            fields = {}
            for item in parts[2:]:
                key, _, value = item.partition("=")
                fields[key] = value.strip('"')
            events.append((parts[1], fields))
        alive = [f for name, f in events if name == "alive"]
        self.assertTrue(alive)
        self.assertTrue(all({"out_events", "out_chars", "last_out_s"} <= set(f) for f in alive), alive[:3])
        # while the engine holds everything back, the counters stand still across heartbeats — which is
        # exactly the evidence the incident was missing (it cannot be reconstructed after the fact)
        stalled = [f for f in alive if f["phase"] in ("prefill", "decode")]
        self.assertGreaterEqual(len(stalled), 2, alive[:3])
        self.assertEqual({f["out_events"] for f in stalled}, {"1"}, stalled[:3])
        summary = [f for name, f in events if name == "sse_summary"]
        self.assertEqual(len(summary), 1, summary)
        self.assertGreaterEqual(int(summary[0]["out_events"]), 2)             # role chunk + content
        self.assertGreater(int(summary[0]["out_chars"]), 0)
        self.assertEqual(summary[0].get("errors", "0"), "0")

    def test_nothing_is_written_when_the_switch_is_off(self):
        from aiohttp.test_utils import TestClient, TestServer

        class Slow(FakeEngine):
            async def generate(self, prompt, **kwargs):
                await asyncio.sleep(0.5)
                async for piece in FakeEngine.generate(self, prompt, **kwargs):
                    yield piece

        async def check():
            client = TestClient(TestServer(serve.create_app(Slow("hello"), "m", TEMPLATE)))
            await client.start_server()
            response = await client.post("/v1/chat/completions", json={
                "model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
            body = await response.read()
            await client.close()
            return body

        body = run(check())
        self.assertEqual([line for line in body.split(b"\n") if line.startswith(b":")], [])
        self.assertIn(b'"content":"hel', body)


if __name__ == "__main__":
    unittest.main()
