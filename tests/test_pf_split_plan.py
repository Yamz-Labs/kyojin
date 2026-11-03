"""Job.split_plan / Job.exact_gate / Job._stash_exact: EXL3_PF_SPLIT exactness tags (CPU only)."""
import os, sys, types
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from hardware import require
require("ext")
from exllamav3.generator.job import Job
from exllamav3.generator.pagetable import PAGE_SIZE

ENVS = ("EXL3_PF_SPLIT", "EXL3_PF_SPLIT_MIN", "EXL3_PF_SPLIT_HEAD", "EXL3_PF_SPLIT_STEP", "EXL3_PF_SPLIT_Q",
        "EXL3_PREFILL_BIG_CHUNK", "EXL3_PREFILL_BIG_CHUNK_MAXPOS", "EXL3_PF_TAIL_MERGE")


def clean_env():
    for k in ENVS:
        os.environ.pop(k, None)
    os.environ["EXL3_PF_TAIL_MERGE"] = "1024"


def plan(total, split = "256", **env):
    clean_env()
    os.environ["EXL3_PF_SPLIT"] = split
    for k, v in env.items():
        os.environ[k] = str(v)
    me = types.SimpleNamespace(generator = types.SimpleNamespace(max_chunk_size = 4096))
    return Job.split_plan(me, total)


def test_off_and_short():
    assert plan(11010, split = "") is None
    assert plan(3000) is None                       # below EXL3_PF_SPLIT_MIN
    assert plan(4000) is None


def test_one_cut():
    # 11010 rows: 4096-row chunks while more than 4096 + 1024 remain -> merged last chunk starts at 8192
    assert plan(11010) == (8192, 10752)
    assert plan(28674) == (24576, 28416)
    # the cut depends on the prompt length only: 256 rows before the end, rounded down to a page
    assert plan(11008 + 255) == (8192, 10752)
    assert plan(11008 + 256) == (8192, 11008)


def test_two_cuts_and_spacing():
    assert plan(11010, split = "1536,256") == (8192, 9472, 10752)
    # cuts closer than EXL3_PF_SPLIT_STEP collapse into the first
    assert plan(11010, split = "512,256", EXL3_PF_SPLIT_STEP = 1024) == (8192, 10496)


def test_head_room():
    # a cut with fewer than EXL3_PF_SPLIT_HEAD rows in the first pass is dropped
    assert plan(8192 + 1100, split = "256") is None or plan(8192 + 1100)[1] >= 8192 + 1024


def test_kda_big_tail_plan():
    # KDA big-tail planner: 8192-row chunks, the last chunk is what remains (<= 8192)
    assert plan_mode(20000, "kda") == (16384, 19712)
    assert plan_mode(8192 + 100, "kda") is None          # 100 rows after the first 8192 chunk: no room for a first pass
    assert plan_mode(8000, "kda") == (0, 7680)
    assert plan_mode(3000, "kda") is None                 # below EXL3_PF_SPLIT_MIN
    assert plan_mode(20000, "kda", EXL3_PREFILL_BIG_CHUNK = 2048) is None   # big == chunk is the plain planner, not big-tail
    assert plan_mode(20000, "gdn", EXL3_PREFILL_BIG_CHUNK = 8192) is None   # GDN planner needs the plain chunking


def plan_mode(total, mode, split = "256", **env):
    clean_env()
    os.environ["EXL3_PF_SPLIT"] = split
    os.environ.setdefault("EXL3_PREFILL_BIG_CHUNK", "8192")
    for k, v in env.items():
        os.environ[k] = str(v)
    me = types.SimpleNamespace(generator = types.SimpleNamespace(max_chunk_size = 2048 if mode == "kda" else 4096))
    return Job.split_plan(me, total, mode)


def test_coarse_grid():
    assert plan(11010, EXL3_PF_SPLIT_Q = 1024) == (8192, 10240)


def test_adaptive_chunking_disabled():
    assert plan(11010, EXL3_PREFILL_BIG_CHUNK = 8192) is None


class Pg:
    def __init__(self, wgen): self.wgen = wgen


def gate_for(total, rc, pages, plan_, hashes):
    pt = types.SimpleNamespace(referenced_pages = {h: p for h, p in zip(hashes, pages)}, unreferenced_pages = {})
    seq = types.SimpleNamespace(sequence_ids = list(range(total + 1)), page_hashes = hashes)
    me = types.SimpleNamespace(generator = types.SimpleNamespace(recurrent_cache = rc), pagetable = pt, sequences = [seq],
                               split_mode = lambda alloc = False: "gdn", split_plan = lambda t, mode = None: plan_)
    return Job.exact_gate(me, seq)


def test_gate_tags():
    rc = types.SimpleNamespace(exact_meta = {}, metrics = {})
    hashes = [f"h{i}" for i in range(60)]
    pages = [Pg(i + 1) for i in range(60)]
    state, other = object(), object()
    p = (8192, 10752)
    pw = tuple(pg.wgen for pg in pages[8192 // PAGE_SIZE: 10752 // PAGE_SIZE])
    rc.exact_meta[hashes[10752 // PAGE_SIZE - 1]] = {"eid": id(state), "sig": p, "pw": pw}
    ph, pi = hashes[10752 // PAGE_SIZE - 1], 10752 // PAGE_SIZE - 1
    g = gate_for(11010, rc, pages, p, hashes)
    assert g(state, ph, pi) is True                                                 # tagged, same plan, pages untouched
    assert g.tagged == [pi]                                                         # allocate_pages prefers it over deeper states
    assert gate_for(11010, rc, pages, (8192, 10240), hashes)(state, ph, pi) is False   # other cut in this prompt's plan
    assert gate_for(11010, rc, pages, None, hashes)(state, ph, pi) is False        # this prompt has no plan
    assert gate_for(11010, rc, pages, p, hashes)(other, ph, pi) is False           # the state under the hash is not the tagged one
    pages[40].wgen = 999                                                            # a page under the cut was rewritten
    assert gate_for(11010, rc, pages, p, hashes)(state, ph, pi) is False
    # an untagged state exactly at a cut of this plan is refused, any other passes
    assert gate_for(11010, rc, pages, p, hashes)(other, hashes[41], 41) is False
    assert gate_for(11010, rc, pages, p, hashes)(other, hashes[20], 20) is True
    assert rc.metrics["exact_rejected"] >= 4


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
