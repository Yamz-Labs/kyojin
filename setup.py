import importlib.util
import os
import shutil
import subprocess

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
    # EXL3_PORT_DEC=1 at build time compiles the HIP decode kernels (quant/exl3_dec.cu) for CUDA too
    # (see port.cuh); they are bound only when EXL3_PORT_DEC=1 at import as well.
    if os.environ.get("EXL3_PORT_DEC", "0") == "1":
        extra_cuda_cflags += ["-DEXL3_PORT_DEC"]
        extra_cflags += ["-DEXL3_PORT_DEC"]

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

def _host_cxx_has_format():
    # Probe whether the host C++ compiler ships a <format> header (GCC 13+).
    # Older toolchains have no such header, where -include would itself be a
    # fatal error, so the pre-include below only opts in when it exists.
    # Harmless on toolchains whose <format> predates the __noinline__
    # attribute use; required on libstdc++16+. Same compiler selection as
    # torch's cpp_extension (CXX or the default); on any detection failure
    # keep today's behavior (no flag).
    cxx = os.environ.get("CXX", "c++")
    if shutil.which(cxx) is None:
        cxx = "g++"
        if shutil.which(cxx) is None:
            return False
    try:
        probe = subprocess.run(
            [cxx, "-xc++", "-E", "-fsyntax-only", "-"],
            input="#include <format>\n",
            capture_output=True,
            text=True,
            timeout=60,
        )
        return probe.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


if torch and torch_version.hip and _host_cxx_has_format():
    # HIP's host_defines.h empties __host__/__device__/__noinline__ for host
    # compilation, which breaks libstdc++16's <format> ([[__gnu__::__noinline__]]
    # parses as an empty attribute name). Force <format> to parse first, before
    # any HIP header can poison the macro state (pragma once protects later
    # includes). Host-only: device TUs don't hit this path.
    extra_cflags += ["-include", "format"]

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
    # NOTE (same as build_config above): a normal import of
    # exllamav3.util.arch_list executes exllamav3/__init__.py, which needs
    # the full runtime (numpy, tokenizers, ...) absent from minimal build
    # envs, and along the way triggers ext.py's JIT load with torch's
    # default flags. arch_list itself needs only torch, guaranteed present
    # here, so load it by file path instead. No silent paths: anything
    # unexpected (missing file, detection failure) raises.
    _arch_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        library_dir, "util", "arch_list.py",
    )
    _arch_spec = importlib.util.spec_from_file_location(
        "exllamav3_arch_list", _arch_path
    )
    if _arch_spec is None or _arch_spec.loader is None:
        raise RuntimeError(f"Cannot load arch detection from {_arch_path}")
    _arch_mod = importlib.util.module_from_spec(_arch_spec)
    _arch_spec.loader.exec_module(_arch_mod)
    _arch_mod.maybe_set_arch_list_env()
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
