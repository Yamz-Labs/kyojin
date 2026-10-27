"""Source-checkpoint format detection and the shared reader (header level; weights are only read by `read_linear`).

    python -m bigconv formats --src CKPT_DIR [--json]          (config.json + *.safetensors headers)
    python -m bigconv formats --map tensor_map.json --config config.json     (a remote model: no weights needed)

The reader itself is exllamav3/foreign_quant/vendor_input.py (numpy only, shared with the Mac converter and the
Linear loader); it is loaded by path so bigconv's plan side stays torch-free.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

_VI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "exllamav3", "foreign_quant", "vendor_input.py")
_vi = None


def vendor_input():
    global _vi
    if _vi is None:
        if "bigconv_vendor_input" in sys.modules:
            _vi = sys.modules["bigconv_vendor_input"]
        else:
            spec = importlib.util.spec_from_file_location("bigconv_vendor_input", os.path.normpath(_VI))
            _vi = importlib.util.module_from_spec(spec)
            sys.modules["bigconv_vendor_input"] = _vi
            spec.loader.exec_module(_vi)
    return _vi


def info_from_map(tmap: dict) -> dict:
    """tensor_map.json (bigconv.tensor_map) -> {name: (dtype tag, shape)}."""
    return {n: (t["dtype"], tuple(t["shape"])) for n, t in tmap["tensors"].items() if t.get("shape") is not None}


def detect(info: dict, config: dict | None = None) -> dict:
    return vendor_input().detect_checkpoint(info, config)


def describe(rep: dict) -> str:
    cnt = ", ".join(f"{k} x{v}" for k, v in sorted(rep["counts"].items()))
    ex = "; ".join(f"{k}: {v['prefix']} [{v['detail']}]" for k, v in rep["examples"].items() if not k.startswith("dense_"))
    qc = rep.get("quantization_config") or {}
    return f"{rep['primary']} ({cnt})" + (f"; config {json.dumps(qc)}" if qc else "") + (f"; e.g. {ex}" if ex else "")


def check_input_format(info: dict, config: dict | None = None):
    """preflight check: (name, status, message). FAIL only for integer payloads no reader understands."""
    rep = detect(info, config)
    msg = describe(rep)
    if rep["primary"] == "unsupported-integer-quant":
        return ("input-format", "FAIL", f"{msg}; {rep['unsupported_integer_linears']} integer linears without a known scale layout "
                                         f"(GPTQ / AWQ?), e.g. {rep['unsupported_examples']}")
    if rep["unsupported_integer_linears"]:
        return ("input-format", "warn", f"{msg}; {rep['unsupported_integer_linears']} integer linears left unread: {rep['unsupported_examples']}")
    if rep["primary"] == "unknown":
        return ("input-format", "warn", "no linear weights recognised in the headers")
    return ("input-format", "ok", msg)


def load_source(src=None, map_path=None, config_path=None):
    """-> (info, config). src = checkpoint dir; map_path = tensor_map.json."""
    vi = vendor_input()
    if src:
        ck = vi.Checkpoint(src)
        try:
            return ck.info(), ck.config
        finally:
            ck.close()
    info = info_from_map(json.load(open(map_path)))
    return info, (json.load(open(config_path)) if config_path else {})


def read_linear(src_dir_or_ck, prefix, block=None):
    """float32 [out, in] of the source linear `prefix` (any detected format); one linear at a time."""
    vi = vendor_input()
    if isinstance(src_dir_or_ck, str):
        ck = vi.Checkpoint(src_dir_or_ck)
        try:
            return vi.load_linear(ck, prefix, block)
        finally:
            ck.close()
    return vi.load_linear(src_dir_or_ck, prefix, block)


def cli(a):
    info, cfg = load_source(a.src, a.map, a.config)
    rep = detect(info, cfg)
    if a.json:
        print(json.dumps(rep, indent=1, default=str))
    else:
        name, status, msg = check_input_format(info, cfg)
        print(f"{status:5s} {name:12s} {msg}")
    raise SystemExit(1 if rep["primary"] == "unsupported-integer-quant" else 0)
