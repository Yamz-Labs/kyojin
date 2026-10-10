import importlib.util
import os
import sys

from setuptools import setup

if torch := importlib.util.find_spec("torch") is not None:
    from torch.utils import cpp_extension
    from torch import version as torch_version

extension_name = "exllamav3_ext"
precompile = "EXLLAMA_NOCOMPILE" not in os.environ
verbose = "EXLLAMA_VERBOSE" in os.environ
ext_debug = "EXLLAMA_EXT_DEBUG" in os.environ

if precompile and not torch:
    print("Cannot precompile unless torch is installed.")
    print("To explicitly JIT install run EXLLAMA_NOCOMPILE= pip install <xyz>")

windows = os.name == "nt"

extra_cflags = []
extra_cuda_cflags = []

if torch and torch_version.hip:
    extra_cuda_cflags += ["-O3", "-DUSE_ROCM"]
    extra_cflags += ["-DUSE_ROCM"]
else:
    extra_cuda_cflags += [
        "-lineinfo", "-O3", "--use_fast_math",
        "-Xcudafe", "--diag_suppress=177",
        "-Xcudafe", "--diag_suppress=20012",
    ]

if windows:
    # NOMINMAX: windows.h otherwise defines min/max function-like macros that break every
    # std::min/std::max call site parsed after it (WIN32_LEAN_AND_MEAN does not suppress them).
    # Defined globally so it holds regardless of include order in any TU.
    # No -std flags here: torch's cpp_extension appends its own (unconditionally on the Windows
    # nvcc path), and a second -std argument is a fatal nvcc error, not an override.
    extra_cflags += ["/Ox", "/Zc:preprocessor", "/DWIN32_LEAN_AND_MEAN", "/DNOMINMAX"]
    extra_cuda_cflags += ["-DWIN32_LEAN_AND_MEAN", "-DNOMINMAX", "-Xcompiler=/Zc:preprocessor"]
    if ext_debug:
        extra_cflags += ["/Zi"]
        extra_cuda_cflags += []
else:
    extra_cflags += ["-Ofast"]
    extra_cuda_cflags += []
    if ext_debug:
        extra_cflags += ["-ftime-report", "-DTORCH_USE_CUDA_DSA"]
        extra_cuda_cflags += []

if cuda_host_cxx := os.environ.get("CUDAHOSTCXX"):
    extra_cuda_cflags += ["-ccbin", cuda_host_cxx]

if torch and torch_version.hip:
    extra_cuda_cflags += ["-DHIPBLAS_USE_HIP_HALF"]
    # Kernel A/B switches for the gfx11.5 GEMV work (see quant/exl3_gemv_kernel.cuh, codebook.cuh).
    # EXL3_HIP_DEFINES="EXL3_HIP_PF_AFTER_STAGE EXL3_MUL1_DECODE_DOT4" at build time.
    for d in os.environ.get("EXL3_HIP_DEFINES", "").split():
        extra_cuda_cflags += ["-D" + d]
    # ROCm 7.14 ships clang 22, which treats the deprecated `register` keyword as
    # a hard error in C++17 mode (older ROCm toolchains only warned).
    extra_cflags += ["-Wno-register"]
    extra_cuda_cflags += ["-Wno-register"]
    if dev_include := os.environ.get("EXL3_ROCM_DEV_INCLUDE"):
        extra_cflags += ["-I" + dev_include]
        extra_cuda_cflags += ["-I" + dev_include]

extra_compile_args = {
    "cxx": extra_cflags,
    "nvcc": extra_cuda_cflags,
}

library_dir = "exllamav3"
sources_dir = os.path.join(library_dir, extension_name)

# NOTE (nix/uv lock): setup.py self-imports exllamav3 for source discovery and
# arch detection. During isolated metadata builds (uv lock, PEP 517
# get_requires) the package isn't installed and torch is absent, and importing
# the top-level exllamav3 package raises (it requires torch). So load
# build_config.py directly by file path, bypassing exllamav3/__init__.py.
# The real build re-runs with torch present and behaves as before.
def _load_get_sources():
    try:
        _cfg_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            library_dir, extension_name, "build_config.py",
        )
        _spec = importlib.util.spec_from_file_location(
            "exllamav3_build_config", _cfg_path
        )
        if _spec is None or _spec.loader is None:
            return None
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        return _mod.get_sources
    except (ImportError, OSError):
        if torch:
            raise  # a real build must fail loudly, never produce an empty extension
        return None


_get_sources = _load_get_sources()

is_rocm = bool(torch and torch_version.hip)
if is_rocm:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from exllamav3.util.arch_list import maybe_set_arch_list_env
    maybe_set_arch_list_env()
if _get_sources is not None:
    sources = _get_sources(sources_dir, is_rocm, base_dir=os.path.dirname(__file__))
else:
    sources = []

setup_kwargs = (
    {
        "ext_modules": [
            cpp_extension.CUDAExtension(
                extension_name,
                sources,
                extra_compile_args=extra_compile_args,
                include_dirs=[sources_dir],
                libraries=["cublas"] if windows else [],
            )
        ],
        "cmdclass": {"build_ext": cpp_extension.BuildExtension},
    }
    if precompile and torch
    else {}
)

setup(
    verbose=verbose,
    **setup_kwargs,
)
