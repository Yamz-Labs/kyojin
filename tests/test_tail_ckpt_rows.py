"""Job.tail_ckpt_rows: which chunk rows get a dense prompt-tail checkpoint (CPU only)."""
import os, sys, types
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from hardware import require
require("ext")
from exllamav3.generator.job import Job
from exllamav3.generator.pagetable import PAGE_SIZE


def rows(prompt_len, start, end, taken=(), dense="7"):
    os.environ["EXL3_TAIL_DENSE"] = dense
    seq = types.SimpleNamespace(sequence_ids = list(range(prompt_len + 1)))   # the last token is not prefilled
    return Job.tail_ckpt_rows(None, seq, start, end, list(taken))


def test_pages_below_last_page():
    # 97537-token prompt, last chunk [94208, 97537): last page 97536, dense rows are the 7 pages below it
    got = rows(97537, 94208, 97537)
    assert got == [97536 - k * PAGE_SIZE - 94208 for k in range(1, 8)]
    assert all(r % PAGE_SIZE == 0 and 0 < r < 97537 - 94208 for r in got)


def test_off_and_short_prompt():
    assert rows(97537, 94208, 97537, dense = "0") == []
    assert all(0 < r < 1000 for r in rows(1000, 0, 1000))


def test_taken_rows_and_slot_cap():
    last = 97536 - 94208
    got = rows(97537, 94208, 97537, taken = [last, last - 2 * PAGE_SIZE, None])
    assert last not in got and last - 2 * PAGE_SIZE not in got
    assert len(got) + 2 <= 8
    assert len(rows(97537, 94208, 97537, taken = [1, 2, 3, 4, 5, 6])) == 2


def sub(prompt_len, start, end, taken = (), n = "4"):
    os.environ["EXL3_TAIL_SUB"] = n
    me = types.SimpleNamespace(_tail_ok = lambda seq: True, generator = types.SimpleNamespace(tail_ckpt_min = 512))
    seq = types.SimpleNamespace(sequence_ids = list(range(prompt_len + 1)))
    return Job.sub_ckpt_rows(me, seq, start, end, list(taken))


def test_sub_rows_are_the_last_multiples_of_64():
    # last chunk [94208, 97537): 3329 rows -> rows 3328, 3264, 3200, 3136
    assert sub(97537, 94208, 97537) == [3328, 3264, 3200, 3136]
    assert sub(97537, 94208, 97537, n = "0") == []
    assert sub(97537, 94209, 97537) == []                  # unaligned chunk start
    assert sub(97537, 94208, 97000) == []                  # not the chunk that holds the prompt end
    assert sub(97537, 94208, 97537, taken = [3328]) == [3264, 3200, 3136, 3072]
    assert len(sub(97537, 94208, 97537, taken = [1, 2, 3, 4, 5, 6])) == 2


class _Page:
    def __init__(self, phash):
        self.phash, self.kv_position = phash, PAGE_SIZE


def torch_equal(a, b):
    import torch
    return torch.equal(a, b)


def _stash_env(sub_max, n_pages = 3):
    import torch
    ids = torch.arange(n_pages * PAGE_SIZE + 64)
    seq = types.SimpleNamespace(allocated_pages = [_Page(100 + i) for i in range(n_pages)],
                                sequence_ids = types.SimpleNamespace(torch_slice = lambda a, b: ids[a:b]))
    gen = types.SimpleNamespace(sub_ckpts = [], sub_ckpt_max = sub_max)
    return types.SimpleNamespace(generator = gen, mtp_defer = False), seq, ids


def _stash(me, seq, pos, salt = 0):
    import torch
    old = seq.sequence_ids
    seq.sequence_ids = types.SimpleNamespace(torch_slice = lambda a, b: old.torch_slice(a, b) + salt)
    try:
        return Job._stash_sub(me, seq, pos, {"position": pos})
    finally:
        seq.sequence_ids = old


def test_sub_stash_replaces_same_content_by_identity():
    # the entry to replace is not the first of the list: list.remove() would compare the dicts before it, tensors inside, and raise
    me, seq, _ = _stash_env(8)
    pos = PAGE_SIZE + 64
    assert _stash(me, seq, pos, salt = 0) == 1
    assert _stash(me, seq, pos, salt = 1) == 1
    first, second = me.generator.sub_ckpts
    assert _stash(me, seq, pos, salt = 1) == 1
    got = me.generator.sub_ckpts
    assert len(got) == 2 and got[0] is first and got[1] is not second and torch_equal(got[1]["ids"], second["ids"])


def test_sub_stash_cap_and_zero():
    me, seq, _ = _stash_env(2)
    for salt in range(5):
        _stash(me, seq, PAGE_SIZE + 64, salt = salt)
    assert len(me.generator.sub_ckpts) == 2
    me0, seq0, _ = _stash_env(0)
    assert _stash(me0, seq0, PAGE_SIZE + 64) == 0 and me0.generator.sub_ckpts == []   # EXL3_TAIL_SUB_MAX=0: IndexError before
    me0.generator.sub_ckpt_max = 0
    me0._tail_ok = lambda s: True
    me0.generator.tail_ckpt_min = 512
    os.environ["EXL3_TAIL_SUB"] = "4"
    assert Job.sub_ckpt_rows(me0, types.SimpleNamespace(sequence_ids = list(range(97538))), 94208, 97537, []) == []


def test_mode1_split_declines_page_spaced_rows():
    # EXL3_MIDCHUNK_CKPT=1 splits the layer at the rows and declines when a segment is 256 rows or less: rows one page apart
    # always do, which is why the job adds dense rows in mode 2 only
    from exllamav3.modules.gated_delta_net import GatedDeltaNet
    me = types.SimpleNamespace(kda = True)
    seg = lambda r: GatedDeltaNet._midchunk_segments(me, {"checkpoint_rows": r, "midchunk_mode": 1}, 1, 8192, True, False)
    assert seg([1024, 2048]) == [0, 1024, 2048, 8192]
    assert seg([1024, 1024 + PAGE_SIZE]) is None


def test_kda_fused_h_siblings_differ_only_in_slot_count():
    from exllamav3.vendor.fla.hip.kda_fused_h_hip import _siblings
    pf = "BRL=1 UNI=1 OPQ=15 -mllvm -disable-lsr"
    assert _siblings(pf + " CK4=1") == [pf, pf + " CK8=1"]
    assert _siblings(pf) == [pf + " CK4=1", pf + " CK8=1"]
    assert _siblings("") == ["CK4=1", "CK8=1"]
    assert _siblings("CK8=1") == ["", "CK4=1"]


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
