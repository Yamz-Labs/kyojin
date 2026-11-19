"""Small JIT-built extension beside the big exllamav3_ext (generic GGUF MoE decode kernels). get() returns the module or None.
Build: EXL3_GM_BUILD=1 python -c "from exllamav3 import gm_ext; gm_ext.get()" under tools/strix_halo/env.sh build. Loading a built one needs no compiler."""
import os, glob
_mod = None
_tried = False
_DIR = os.path.dirname(os.path.abspath(__file__))
_NAME = "exl3_gm_ext"


def get():
    global _mod, _tried
    if _tried:
        return _mod
    _tried = True
    if os.environ.get("EXL3_GM", "1") == "0":
        return None
    build = os.environ.get("EXL3_GM_BUILD_DIR", os.path.join(os.path.expanduser("~"), ".cache", "exl3_gm_ext"))
    so = glob.glob(os.path.join(build, _NAME + "*.so"))
    try:
        import torch
        if so and os.environ.get("EXL3_GM_BUILD", "0") != "1":
            import importlib.util
            spec = importlib.util.spec_from_file_location(_NAME, so[0])
            _mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_mod)
        elif os.environ.get("EXL3_GM_BUILD", "0") == "1":
            from torch.utils.cpp_extension import load
            os.makedirs(build, exist_ok=True)
            srcs = sorted(glob.glob(os.path.join(_DIR, "*.cu")) + glob.glob(os.path.join(_DIR, "*.cpp")))
            _mod = load(name=_NAME, sources=srcs, extra_include_paths=[_DIR],
                        extra_cflags=["-O3", "-DUSE_ROCM"], extra_cuda_cflags=["-O3", "-DUSE_ROCM"], build_directory=build, verbose=True)
    except Exception as e:  # missing build or toolchain: callers fall back to the stock path
        print(f"[gm_ext] unavailable: {e}")
        _mod = None
    return _mod
