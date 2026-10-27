"""
SlotStore._restore_tail drops a tail entry that is already in memory. With list.remove() two entries of the same prev_hash and ids of
other lengths made the comparison of the dicts raise (RuntimeError: The size of tensor a (122) must match the size of tensor b (215)),
which ended the whole prefix disk restore of that request. The method is loaded by ast; its first statements run on fakes.
"""
import ast, os, sys, textwrap, types
import torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = open(os.path.join(ROOT, "exllamav3", "generator", "slot_store.py")).read()
fn = None
for n in ast.walk(ast.parse(src)):
    if isinstance(n, ast.FunctionDef) and n.name == "_restore_tail":
        fn = n
body = ast.get_source_segment(src, fn)
# keep the dedupe statement only: cut at the eviction order
head = body[:body.index("        if order is None:")]
ns = {"torch": torch}
exec(textwrap.dedent(head + "        return g.tail_ckpts\n"), ns)
f = ns["_restore_tail"]

def entry(ph, n):
    return {"prev_hash": ph, "ids": torch.arange(n).view(1, -1), "page": None}

g = types.SimpleNamespace(tail_ckpts=[entry(b"p", 122), entry(b"q", 7), entry(b"p", 215)], pagetable=None)
self = types.SimpleNamespace(generator=g)
out = f(self, {"ids": torch.arange(215).view(1, -1)}, b"p", None, None, None, None)
assert [(e["prev_hash"], e["ids"].shape[-1]) for e in out] == [(b"p", 122), (b"q", 7)], out   # only the identical entry is replaced
print("tail dedupe ok")
