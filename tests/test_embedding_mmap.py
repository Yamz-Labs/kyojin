import os, torch
from safetensors.torch import save_file
from exllamav3.loader.safetensors import SafetensorsCollection


def _anon_kib():
    for l in open("/proc/self/status"):
        if l.startswith("RssAnon:"):
            return int(l.split()[1])


def test_mmap_tensor_equals_loaded_and_costs_no_anonymous_memory(tmp_path):
    w = torch.randn(40000, 256).to(torch.bfloat16)   # 20 MB
    other = torch.arange(10, dtype = torch.float32)
    save_file({"a.weight": w, "b": other}, str(tmp_path / "m.safetensors"))
    stc = SafetensorsCollection(str(tmp_path))
    before = _anon_kib()
    t = stc.get_tensor_mmap("a.weight")
    ids = torch.tensor([0, 7, 39999, 123])
    assert t.dtype == torch.bfloat16 and t.shape == w.shape
    assert torch.equal(torch.nn.functional.embedding(ids, t), w[ids])
    assert _anon_kib() - before < 8000      # the 20 MB table is not copied into anonymous memory
    assert torch.equal(t, stc.get_tensor("a.weight", torch.device("cpu"), allow_bf16 = True))
    t2 = t.clone()                          # private mapping: writes never reach the file
    t[0] += 1
    assert torch.equal(stc.get_tensor_mmap("a.weight"), w)
    assert stc.get_tensor_mmap("b") is None        # fp32: not eligible, falls back to the normal load
    assert stc.get_tensor_mmap("missing") is None
    del t2
