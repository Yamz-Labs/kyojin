"""CPU-only import helper: loads loader/gguf_collection.py, foreign_quant/gap_formats.py and loader/inkling2_forward.py under stub
packages (importing `exllamav3` for real builds the HIP/CUDA extension). Torch only; no GPU. install(root) returns the forward module."""
import importlib.util
import os
import sys
import types


def _pkg(name, path):
    m = types.ModuleType(name)
    m.__path__ = [path]
    sys.modules[name] = m
    return m


def _load(name, path, pkg):
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp)
    m.__package__ = pkg
    sys.modules[name] = m
    sp.loader.exec_module(m)
    return m


def install(root=None):
    root = root or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ex = os.path.join(root, "exllamav3")
    if "exllamav3.loader.inkling2_forward" in sys.modules:
        return sys.modules["exllamav3.loader.inkling2_forward"]
    if "exllamav3" not in sys.modules:
        _pkg("exllamav3", ex)
        _pkg("exllamav3.loader", os.path.join(ex, "loader"))
        _pkg("exllamav3.foreign_quant", os.path.join(ex, "foreign_quant"))
    _load("exllamav3.foreign_quant.gguf_tables", os.path.join(ex, "foreign_quant", "gguf_tables.py"), "exllamav3.foreign_quant")
    _load("exllamav3.foreign_quant.gap_formats", os.path.join(ex, "foreign_quant", "gap_formats.py"), "exllamav3.foreign_quant")
    _load("exllamav3.loader.gguf_collection", os.path.join(ex, "loader", "gguf_collection.py"), "exllamav3.loader")
    return _load("exllamav3.loader.inkling2_forward", os.path.join(ex, "loader", "inkling2_forward.py"), "exllamav3.loader")
