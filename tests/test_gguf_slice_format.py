import importlib.util
import os

# Load the module by path: importing the exllamav3 package would build/load the CUDA extension (not needed here)
_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exllamav3", "loader", "gguf_collection.py")
_spec = importlib.util.spec_from_file_location("gguf_collection_under_test_fmt", _p)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_gguf_fmt = _mod.gguf_fmt


FM = {"blk.0.ffn_gate_exps.weight": "mxfp4", "blk.0.attn_q.weight": "q8_0"}


def test_plain_name():
    assert _gguf_fmt(FM, "blk.0.attn_q.weight") == "q8_0"


def test_expert_slice_takes_stack_format():
    assert _gguf_fmt(FM, "blk.0.ffn_gate_exps.e7.weight") == "mxfp4"
    assert _gguf_fmt(FM, "blk.0.ffn_gate_exps.e31.weight") == "mxfp4"


def test_unknown_and_empty():
    assert _gguf_fmt(FM, "blk.0.ffn_up_exps.e3.weight") is None
    assert _gguf_fmt(FM, "blk.0.ffn_gate_exps.ex.weight") is None
    assert _gguf_fmt(FM, None) is None
    assert _gguf_fmt(FM, "") is None
