from __future__ import annotations
import importlib.machinery
import importlib.util
import torch
from torch.utils.cpp_extension import load
import os
import sys
from .util.arch_list import maybe_set_arch_list_env

extension_name = "exllamav3_ext"
verbose = False  # Print wall of text when compiling
ext_debug = False  # Compile with debug options

# Determine if we're on Windows

windows = (os.name == "nt")

# Determine if extension is already installed or needs to be built

def is_precompiled_extension_available():
    spec = importlib.util.find_spec(extension_name)
    if not spec or not spec.origin or not spec.loader:
        return False
    return any(
        spec.origin.endswith(suffix)
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
    )

if is_precompiled_extension_available():
    import exllamav3_ext
else:

    # Kludge to get compilation working on Windows

    if windows:

        def find_msvc():

            # Possible locations for MSVC, in order of preference

            program_files_x64 = os.environ["ProgramW6432"]
            program_files_x86 = os.environ["ProgramFiles(x86)"]

            msvc_dirs = \
            [
                a + "\\Microsoft Visual Studio\\" + b + "\\" + c + "\\VC\\Tools\\MSVC\\"
                for b in ["2022", "2019", "2017"]
                for a in [program_files_x64, program_files_x86]
                for c in ["BuildTools", "Community", "Professional", "Enterprise", "Preview"]
            ]

            for msvc_dir in msvc_dirs:
                if not os.path.exists(msvc_dir): continue

                # Prefer the latest version

                versions = sorted(os.listdir(msvc_dir), reverse = True)
                for version in versions:

                    compiler_dir = msvc_dir + version + "\\bin\\Hostx64\\x64"
                    if os.path.exists(compiler_dir) and os.path.exists(compiler_dir + "\\cl.exe"):
                        return compiler_dir

            # No path found

            return None

        import subprocess

        # Check if cl.exe is already in the path

        try:

            subprocess.check_output(["where", "/Q", "cl"])

        # If not, try to find an installation of Visual Studio and append the compiler dir to the path

        except subprocess.CalledProcessError as e:

            cl_path = find_msvc()
            if cl_path:
                if verbose:
                    print(" -- Injected compiler path:", cl_path)
                os.environ["path"] += ";" + cl_path
            else:
                print(" !! Unable to find cl.exe; compilation will probably fail", file = sys.stderr)

    # compiler flags

    library_dir = os.path.dirname(os.path.abspath(__file__))
    sources_dir = os.path.join(library_dir, extension_name)

    extra_cflags = []
    extra_cuda_cflags = []

    if torch.version.hip:
        extra_cuda_cflags += ["-Ofast", "-DUSE_ROCM", "-Wno-register"]
        extra_cflags += ["-DUSE_ROCM"]
    else:
        extra_cuda_cflags += [
            "-lineinfo", "-O3", "--use_fast_math",
            "-Xcudafe", "--diag_suppress=177",
            "-Xcudafe", "--diag_suppress=20012",
        ]

    if windows:
        # TODO: preprocessor and lean_and_mean flags are needed for Windows cu132 build, verify that they don't break
        #       older cu128 builds
        # NOMINMAX: windows.h otherwise defines min/max function-like macros that break every
        # std::min/std::max call site parsed after it (WIN32_LEAN_AND_MEAN does not suppress them).
        # Defined globally so it holds regardless of include order in any TU (mirrors setup.py).
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

    if not windows and (cuda_host_cxx := os.environ.get("CUDAHOSTCXX")):
        extra_cuda_cflags += ["-ccbin", cuda_host_cxx]

    if torch.version.hip:
        extra_cuda_cflags += ["-DHIPBLAS_USE_HIP_HALF"]

    if verbose:
        if torch.version.hip:
            extra_cuda_cflags += ["-verbose"]
        else:
            extra_cuda_cflags += ["--ptxas-options=-v"]

    # linker flags

    extra_ldflags = []

    if windows:
        extra_ldflags += ["cublas.lib"]
        if sys.base_prefix != sys.prefix:
            extra_ldflags += [f"/LIBPATH:{os.path.join(sys.base_prefix, 'libs')}"]

    # sources

    from .exllamav3_ext.build_config import get_sources as _get_sources
    is_rocm = bool(torch.version.hip)
    sources = _get_sources(sources_dir, is_rocm)

    # Load extension

    maybe_set_arch_list_env()
    exllamav3_ext = load(
        name = extension_name,
        sources = sources,
        extra_include_paths = [sources_dir],
        verbose = verbose,
        extra_ldflags = extra_ldflags,
        extra_cuda_cflags = extra_cuda_cflags,
        extra_cflags = extra_cflags
    )


# When a BC_* class is not compiled into the extension (e.g. on ROCm where the
# libtorch/ sources are excluded), make attribute access return a callable that
# yields None instead of raising AttributeError. This lets call sites write
# ``self.bc = ext.BC_Mamba2(...)`` unconditionally — they get None on platforms
# that lack the class, and the real object on platforms that have it.

if torch.version.hip:
    from .ext_fallbacks import _BCNone

    _bc_none = _BCNone()

    # BC_* constructors: return None when the class isn't compiled
    for _name in [
        'BC_Mamba2', 'BC_GatedDeltaNet', 'BC_GatedDeltaNetSplit',
        'BC_MLP', 'BC_GatedMLP', 'BC_BlockSparseMLP',
        'BC_Attention', 'BC_GatedRMSNorm',
        'BC_LinearEXL3', 'BC_LinearFP16',
        'BC_DSV4Compressor', 'BC_DSV4Attention', 'BC_DSV4BatchAttention',
        'BC_MLAttention', 'BC_SAM',
    ]:
        if not hasattr(exllamav3_ext, _name):
            setattr(exllamav3_ext, _name, _bc_none)

    # C++ functions from excluded source files: replace with PyTorch implementations
    from . import ext_fallbacks as _fb

    for _name in [
        'silu_mul', 'silu_oai_mul', 'gelu_mul', 'relu2_mul', 'relu_mul', 'xielu',
        'apply_logit_bitmask',
        'mul_sigmoid_', 'mul_sigmoid_broadcast_', 'mul_softplus_broadcast_',
        'add_sigmoid_gate', 'add_sigmoid_gate_proj', 'deinterleave_qg',
        'rms_norm', 'rms_norm_res_in', 'gated_rms_norm',
        'softcap',
        'quant_cache_cont', 'dequant_cache_cont',
        'quant_cache_paged', 'dequant_cache_paged', 'dequant_cache_paged_window',
        'count_inf_nan', 'dsa_topk',
    ]:
        if not hasattr(exllamav3_ext, _name):
            setattr(exllamav3_ext, _name, getattr(_fb, _name))

    # Native RMSNorm (exl3_dec.cu) for the common shapes; everything else keeps the torch fallback
    if hasattr(exllamav3_ext, "exl3_dec_rms_norm") and os.environ.get("EXL3_DEC_NORM", "1") != "0":
        _fb_rms_norm = _fb.rms_norm
        _fb_rms_norm_res_in = _fb.rms_norm_res_in
        _norm_dt = (torch.float16, torch.float32)

        def _w_ok(w, dim):
            return w is None or (w.dtype in (torch.float16, torch.bfloat16) and w.numel() == dim and w.is_contiguous())

        def _rms_norm_rocm(x, w, y, eps, constant_bias, constant_scale, span_heads, add_residual, w_groups = 1):
            if (not span_heads and w_groups == 1 and x.is_cuda and x.is_contiguous() and y.is_contiguous() and
                    x.shape == y.shape and x.dtype in _norm_dt and y.dtype in _norm_dt and
                    x.shape[-1] <= 8192 and _w_ok(w, x.shape[-1])):
                exllamav3_ext.exl3_dec_rms_norm(x, w, y, None, eps, constant_bias, constant_scale,
                                                1 if add_residual else 0)
                return
            _fb_rms_norm(x, w, y, eps, constant_bias, constant_scale, span_heads, add_residual, w_groups)

        def _rms_norm_res_in_rocm(x, w, y, r, eps, constant_bias, constant_scale):
            if (x.is_cuda and x.is_contiguous() and y.is_contiguous() and r.is_contiguous() and
                    x.shape == y.shape == r.shape and x.dtype in _norm_dt and r.dtype in _norm_dt and
                    y.dtype == torch.float16 and x.shape[-1] <= 8192 and _w_ok(w, x.shape[-1])):
                exllamav3_ext.exl3_dec_rms_norm(x, w, y, r, eps, constant_bias, constant_scale, 2)
                return
            _fb_rms_norm_res_in(x, w, y, r, eps, constant_bias, constant_scale)

        setattr(exllamav3_ext, "rms_norm", _rms_norm_rocm)
        setattr(exllamav3_ext, "rms_norm_res_in", _rms_norm_res_in_rocm)

    # Fused elementwise ops (fused_elt_rocm.cu): one launch instead of the fallback's torch chain
    if hasattr(exllamav3_ext, "fused_gated_rms_norm") and os.environ.get("EXL3_FUSE_GNORM", "1") != "0":
        _fb_gated_rms_norm = _fb.gated_rms_norm

        def _gated_rms_norm_rocm(x, w, y, g, eps, constant_bias, w_groups, gate_first, gate_act = 0, y_pitch = 0):
            if (x.is_cuda and x.dtype == torch.bfloat16 and y.dtype in (torch.float16, torch.float32) and
                    w.dtype in (torch.bfloat16, torch.float32) and
                    (g.dtype in (torch.bfloat16, torch.float32) or
                     (g.dtype == torch.float16 and ROCM_KNOBS["gnorm_f16g"])) and
                    x.is_contiguous() and (y.is_contiguous() or (y_pitch and y.dtype == torch.float16 and x.dim() >= 3)) and g.is_contiguous() and w.is_contiguous() and
                    x.shape == y.shape == g.shape and x.shape[-1] <= 512 and x.shape[-1] % 128 == 0 and w_groups >= 1 and
                    w.numel() == w_groups * x.shape[-1] and gate_act in (0, 1)):
                exllamav3_ext.fused_gated_rms_norm(x, w, y, g, eps, constant_bias, w_groups, gate_first, gate_act,
                                                   y_pitch, x.shape[-2] if y_pitch else 1)
                return
            assert not y_pitch, "padded gated-norm output needs the fused kernel"
            _fb_gated_rms_norm(x, w, y, g, eps, constant_bias, w_groups, gate_first, gate_act)

        def _gn_pad_ok(x, w, y_dtype, g, w_groups, gate_act = 0):
            """True when _gated_rms_norm_rocm can write a row-padded fp16 y: the fused kernel's own conditions."""
            return (x.is_cuda and x.dtype == torch.bfloat16 and x.dim() >= 3 and y_dtype == torch.float16 and
                    w.dtype in (torch.bfloat16, torch.float32) and
                    (g.dtype in (torch.bfloat16, torch.float32) or
                     (g.dtype == torch.float16 and ROCM_KNOBS["gnorm_f16g"])) and
                    x.is_contiguous() and g.is_contiguous() and w.is_contiguous() and x.shape == g.shape and
                    x.shape[-1] <= 512 and x.shape[-1] % 128 == 0 and w_groups >= 1 and
                    w.numel() == w_groups * x.shape[-1] and gate_act in (0, 1))
        _gated_rms_norm_rocm.pad_ok = _gn_pad_ok
        setattr(exllamav3_ext, "gated_rms_norm", _gated_rms_norm_rocm)

    # deinterleave_qg: hand HIP copy kernel (modules/ple_fn/ple_hip.hip), pure copy so bitwise equal; EXL3_DQ_HIP=1
    if True:  # flag read per call so one process can A/B it
        _fb_deinterleave_qg = _fb.deinterleave_qg

        def _deinterleave_qg_rocm(qg, q, g, head_dim):
            if (os.environ.get("EXL3_DQ_HIP", "0") == "1" and head_dim % 8 == 0 and qg.is_cuda and qg.dtype == torch.float16 and q.dtype == torch.float16
                    and g.dtype == torch.float16 and qg.is_contiguous() and q.is_contiguous() and g.is_contiguous()
                    and q.numel() == g.numel() and q.numel() * 2 == qg.numel() and q.numel() > 0
                    and qg.data_ptr() % 16 == 0 and q.data_ptr() % 16 == 0 and g.data_ptr() % 16 == 0
                    and qg.numel() % (2 * head_dim) == 0):
                try:
                    from .modules.ple_fn import ple_hip
                    ple_hip.deinterleave_qg(qg, q, g, head_dim)
                    return
                except Exception:
                    pass
            _fb_deinterleave_qg(qg, q, g, head_dim)

        setattr(exllamav3_ext, "deinterleave_qg", _deinterleave_qg_rocm)

    # r35: runtime-mutable knobs (in-process A/B)
    ROCM_KNOBS = {"silu_lim": os.environ.get("EXL3_FUSE_SILU_LIM", "1") != "0",
                  # fp16 gate (KDA z under EXL3_KDA_F16_GATES) into the fused gated norm; "0" = the torch
                  # fallback chain (r2 glm-next: cost 4K prefill -2.5 %, the F16_GATES "loss")
                  "gnorm_f16g": os.environ.get("EXL3_FUSE_GNORM_F16G", "1") != "0"}
    if hasattr(exllamav3_ext, "fused_silu_mul") and os.environ.get("EXL3_FUSE_SILU", "1") != "0":
        _fb_silu_mul = _fb.silu_mul

        def _silu_mul_rocm(x, y, z, act_limit = 0.0):
            if ((act_limit == 0.0 or ROCM_KNOBS["silu_lim"]) and x.is_cuda and
                    x.dtype in (torch.float16, torch.float32) and
                    y.dtype == x.dtype and z.dtype == torch.float16 and x.is_contiguous() and
                    y.is_contiguous() and z.is_contiguous() and x.shape == y.shape == z.shape):
                exllamav3_ext.fused_silu_mul(x, y, z, float(act_limit))
                return
            _fb_silu_mul(x, y, z, act_limit)

        setattr(exllamav3_ext, "silu_mul", _silu_mul_rocm)

    # Keep the public API stable while using the native QSA kernel (k <= 512, moeA5) only for its
    # validated gfx12 wave32 shape. Everything else retains the general PyTorch fallback.
    if hasattr(exllamav3_ext, "dsa_topk_gfx12"):
        def _dsa_topk_rocm(scores, indices, k, t_ptr = None, t_seq = 0):
            if (
                hasattr(exllamav3_ext, "dsa_topk_gfx12") and
                _fb.dsa_topk_gfx12_supported(scores, indices, k, t_ptr, t_seq)
            ):
                return exllamav3_ext.dsa_topk_gfx12(scores, indices, k)
            return _fb.dsa_topk(scores, indices, k, t_ptr, t_seq)

        setattr(exllamav3_ext, "dsa_topk", _dsa_topk_rocm)

    # Constants and functions guarded by fused_sampler_enable in generator/sampler/custom.py.
    # Disable the fused sampler path on ROCm by setting the flag and providing stub values.
    if not hasattr(exllamav3_ext, 'FUSED_SAMPLER_MAX_BLOCKS'):
        setattr(exllamav3_ext, 'FUSED_SAMPLER_MAX_BLOCKS', 0)
    if not hasattr(exllamav3_ext, 'FUSED_SAMPLER_HIST_STRIDE'):
        setattr(exllamav3_ext, 'FUSED_SAMPLER_HIST_STRIDE', 0)
    os.environ['EXL3_FUSED_SAMPLER'] = '0'
