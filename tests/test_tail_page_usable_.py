"""
prefix_disk.tail_page_usable: an in-memory prompt-end checkpoint whose partial page was reused by another job is not resumable,
so the restore must take the deeper copy from the disk. Pure torch on CPU.
"""
import os, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from exllamav3.generator.prefix_disk import tail_page_usable

ids = torch.arange(100).view(1, -1)
def page(prev, n, seq):
    s = torch.zeros(1, 256, dtype=torch.long); s[:, :len(seq)] = torch.tensor(seq)
    return types.SimpleNamespace(prev_hash=prev, kv_position=n, sequence=s)

e = {"prev_hash": b"p", "ids": ids, "page": page(b"p", 100, list(range(100)))}
assert tail_page_usable(e, b"p", 100)
assert not tail_page_usable(dict(e, page=page(b"q", 100, list(range(100)))), b"p", 100)       # taken by a chain of another prompt
assert not tail_page_usable(dict(e, page=page(b"p", 0, [])), b"p", 100)                      # evicted and cleared
assert not tail_page_usable(dict(e, page=page(b"p", 100, list(range(99)) + [7])), b"p", 100)  # reused for other tokens
assert tail_page_usable({"prev_hash": b"p", "ids": ids, "page": None}, b"p", 100)            # no page recorded: unchanged behaviour
print("tail page usable ok")
