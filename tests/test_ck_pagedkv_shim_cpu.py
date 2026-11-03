"""Compile the actual C ABI shim with inert headers; never load HIP or a .so.

Run directly with unittest to avoid framework-importing repository conftest files.
CK_SHIM_CPU_RECEIPTS optionally retains generated fixtures and subprocess receipts.
"""
import json
import os
from pathlib import Path
import resource
import shlex
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SHIM = ROOT / "tools/strix_halo/ck_pagedkv_shim.cpp"

# Only host-side declarations consumed by the real shim. index_t matches pinned
# CK's int32_t, including its batch/head stride fields. There is deliberately no
# generated support dispatcher and no implementation of the primary template.
HEADERS = {
    "hip/hip_runtime.h": "#pragma once\nstruct ihipStream_t;\nusing hipStream_t = ihipStream_t*;\n",
    "bias.hpp": "#pragma once\n",
    "mask.hpp": "#pragma once\nenum class mask_enum { mask_bottom_right = 2 };\n",
    "fmha_fwd.hpp": r"""#pragma once
#include <cstdint>
#include <hip/hip_runtime.h>
namespace ck_tile {
using index_t = std::int32_t;
struct gfx11_t {};
template <bool> struct SimplifiedGenericAttentionMask {};
enum class BlockFmhaPipelineEnum { QRKSVS };
enum class BlockAttentionBiasEnum { NO_BIAS };
struct stream_config {
    hipStream_t stream_id_;
    bool time_kernel_;
    int log_level_;
    int cold_niters_;
    int nrepeat_;
    bool is_gpu_timer_;
};
}
struct FmhaFwdFp16 {};
template <ck_tile::index_t, typename, bool, ck_tile::index_t,
          ck_tile::index_t, ck_tile::index_t, ck_tile::index_t,
          ck_tile::index_t, ck_tile::index_t, bool,
          ck_tile::BlockFmhaPipelineEnum, bool, typename,
          ck_tile::BlockAttentionBiasEnum, bool, bool, bool, bool,
          bool, bool, bool, bool, bool, int>
struct fmha_fwd_pagedkv_traits_ {};
struct fmha_fwd_pagedkv_args {
    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    void* o_ptr;
    void* block_table_ptr;
    ck_tile::index_t batch_stride_block_table;
    ck_tile::index_t page_block_size;
    const void* seqlen_k_ptr;
    ck_tile::index_t seqlen_q, seqlen_k, batch, max_seqlen_q;
    ck_tile::index_t hdim_q, hdim_v, nhead_q, nhead_k;
    float scale_s, scale_p, scale_o;
    ck_tile::index_t window_size_left, window_size_right, mask_type;
    ck_tile::index_t stride_q, stride_k, stride_v, stride_o;
    ck_tile::index_t nhead_stride_q, nhead_stride_k, nhead_stride_v, nhead_stride_o;
    ck_tile::index_t batch_stride_q, batch_stride_k, batch_stride_v, batch_stride_o;
};
template <typename Traits, typename Arch>
float fmha_fwd_pagedkv_(const ck_tile::stream_config&, fmha_fwd_pagedkv_args);
""",
}

DRIVER = r"""#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include "@SHIM@"
namespace {
int mode = 0;
int calls = 0;
fmha_fwd_pagedkv_args last_args{};
ck_tile::stream_config last_config{};
}
template <>
float fmha_fwd_pagedkv_<ck_trait, ck_tile::gfx11_t>(
    const ck_tile::stream_config& config, fmha_fwd_pagedkv_args args) {
    ++calls;
    last_args = args;
    last_config = config;
    switch (mode) {
    case 0: return 0.0f;
    case 1: return 0.25f;
    case 2: return -1.0f;
    case 3: return -2.0f;
    case 4: return std::numeric_limits<float>::quiet_NaN();
    case 5: return std::numeric_limits<float>::infinity();
    case 6: return -std::numeric_limits<float>::infinity();
    // CK's HIP_CHECK_ERROR throws std::runtime_error at the host boundary.
    case 7: throw std::runtime_error("inert HIP launch failure");
    case 8: throw 7;
    case 9: return -0.0f;
    case 10: return -0.5f;
    default: throw std::runtime_error("bad fixture mode");
    }
}
int main(int argc, char** argv) {
    if (argc != 9) return 64;
    mode = std::atoi(argv[1]);
    const int q_len = std::atoi(argv[2]);
    const int kv_len = std::atoi(argv[3]);
    const int pages = std::atoi(argv[4]);
    const int layout = std::atoi(argv[5]);
    const int stride = std::atoi(argv[6]);
    const int null_pointer = std::atoi(argv[7]);
    const int time_it = std::atoi(argv[8]);
    int32_t storage = 0;
    const void* q = null_pointer == 0 ? nullptr : &storage;
    const void* k = null_pointer == 1 ? nullptr : &storage;
    const void* v = null_pointer == 2 ? nullptr : &storage;
    void* out = null_pointer == 3 ? nullptr : &storage;
    const int32_t* table = null_pointer == 4 ? nullptr : &storage;
    const int32_t* lengths = null_pointer == 5 ? nullptr : &storage;
    const float result = ck_pagedkv_forward(
        q, k, v, out, table, lengths, q_len, kv_len, pages, 0.5f,
        layout, stride, nullptr, time_it);
    // Explicit output, not C assert: checks survive Python -O and C++ NDEBUG.
    std::cout << "{\"result\":\"" << result << "\",\"calls\":" << calls
              << ",\"time_kernel\":" << last_config.time_kernel_
              << ",\"null_stream\":" << (last_config.stream_id_ == nullptr)
              << ",\"log_level\":" << last_config.log_level_
              << ",\"cold_niters\":" << last_config.cold_niters_
              << ",\"nrepeat\":" << last_config.nrepeat_
              << ",\"gpu_timer\":" << last_config.is_gpu_timer_
              << ",\"pointers_forwarded\":"
              << (last_args.q_ptr == q && last_args.k_ptr == k &&
                  last_args.v_ptr == v && last_args.o_ptr == out &&
                  last_args.block_table_ptr == table && last_args.seqlen_k_ptr == lengths)
              << ",\"q_len\":" << last_args.seqlen_q
              << ",\"kv_len\":" << last_args.seqlen_k
              << ",\"pages\":" << last_args.batch_stride_block_table
              << ",\"page_size\":" << last_args.page_block_size
              << ",\"batch\":" << last_args.batch
              << ",\"max_seqlen_q\":" << last_args.max_seqlen_q
              << ",\"hdim_q\":" << last_args.hdim_q
              << ",\"hdim_v\":" << last_args.hdim_v
              << ",\"nhead_q\":" << last_args.nhead_q
              << ",\"nhead_k\":" << last_args.nhead_k
              << ",\"scale_s\":" << last_args.scale_s
              << ",\"scale_p\":" << last_args.scale_p
              << ",\"scale_o\":" << last_args.scale_o
              << ",\"window_left\":" << last_args.window_size_left
              << ",\"window_right\":" << last_args.window_size_right
              << ",\"mask_type\":" << last_args.mask_type
              << ",\"stride_q\":" << last_args.stride_q
              << ",\"head_stride_q\":" << last_args.nhead_stride_q
              << ",\"batch_stride_q\":" << last_args.batch_stride_q
              << ",\"stride_o\":" << last_args.stride_o
              << ",\"head_stride_o\":" << last_args.nhead_stride_o
              << ",\"batch_stride_o\":" << last_args.batch_stride_o
              << ",\"stride_k\":" << last_args.stride_k
              << ",\"head_stride_k\":" << last_args.nhead_stride_k
              << ",\"batch_stride_k\":" << last_args.batch_stride_k
              << ",\"stride_v\":" << last_args.stride_v
              << ",\"head_stride_v\":" << last_args.nhead_stride_v
              << ",\"batch_stride_v\":" << last_args.batch_stride_v
              << "}\n";
}
"""


def no_core_dump():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


class ShimBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shlex.split(os.environ.get("CXX", ""))
        if not compiler:
            found = shutil.which("g++") or shutil.which("clang++")
            if not found:
                raise unittest.SkipTest("CPU C++ compiler required")
            compiler = [found]
        receipt_root = os.environ.get("CK_SHIM_CPU_RECEIPTS")
        if receipt_root:
            cls.temporary = None
            cls.directory = Path(tempfile.mkdtemp(prefix="cpu-shim-", dir=receipt_root))
        else:
            cls.temporary = tempfile.TemporaryDirectory(prefix="cpu-shim-")
            cls.directory = Path(cls.temporary.name)
        cls.commands = []
        cls.binaries = {}
        include = cls.directory / "include"
        for relative, content in HEADERS.items():
            path = include / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        driver = cls.directory / "driver.cpp"
        driver.write_text(DRIVER.replace("@SHIM@", SHIM.as_posix()))
        cls.run_command(compiler + ["--version"])
        for optimization in ("-O0", "-O2"):
            binary = cls.directory / ("shim" + optimization)
            cls.run_command(compiler + [
                "-std=c++17", "-Wall", "-Wextra", "-Werror", optimization, "-Wno-cpp",
                "-isystem", str(include), str(driver), "-o", str(binary),
            ])
            cls.binaries[optimization] = binary

    @classmethod
    def tearDownClass(cls):
        if cls.temporary is not None:
            cls.temporary.cleanup()

    @classmethod
    def run_command(cls, argv):
        result = subprocess.run(argv, text=True, capture_output=True, preexec_fn=no_core_dump)
        cls.commands.append({
            "argv": [str(item) for item in argv], "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr,
        })
        (cls.directory / "commands.json").write_text(json.dumps(cls.commands, indent=2) + "\n")
        if result.returncode:
            raise AssertionError(
                f"command failed ({result.returncode}): {argv!r}\n{result.stdout}{result.stderr}"
            )
        return result.stdout

    def invoke(self, optimization, *, mode=0, q_len=64, kv_len=257,
               pages=2, layout=0, stride=192, null_pointer=-1, time_it=0):
        result = json.loads(self.run_command([
            str(self.binaries[optimization]), *map(str, (
                mode, q_len, kv_len, pages, layout, stride, null_pointer, time_it,
            )),
        ]))
        result["result"] = float(result["result"])
        return result

    def test_success_preserves_layout_arguments_and_timing(self):
        shapes = [(1, 1, 1), (64, 256, 1), (64, 257, 2), (64, 257, 3),
                  (174762, 174762, 683), (64, 2147483647, 8388608),
                  (64, 257, 2147483647)]
        for optimization in self.binaries:
            for q_len, kv_len, pages in shapes:
                for layout in (0, 1):
                    for mode, time_it, expected in ((0, 0, 1.0), (1, 0, 1.0),
                                                    (1, 1, 0.25), (9, 0, 1.0)):
                        with self.subTest(optimization=optimization, q_len=q_len,
                                          kv_len=kv_len, pages=pages, layout=layout,
                                          mode=mode, time_it=time_it):
                            result = self.invoke(
                                optimization, mode=mode, time_it=time_it, q_len=q_len,
                                kv_len=kv_len, pages=pages, layout=layout,
                            )
                            self.assertEqual(result["result"], expected)
                            self.assertEqual(result["calls"], 1)
                            expected_fields = {
                                "time_kernel": time_it, "null_stream": 1,
                                "log_level": 0, "cold_niters": 0, "nrepeat": 1,
                                "gpu_timer": 0, "pointers_forwarded": 1,
                                "q_len": q_len, "kv_len": kv_len, "pages": pages,
                                "page_size": 256, "batch": 1, "max_seqlen_q": q_len,
                                "hdim_q": 192, "hdim_v": 128,
                                "nhead_q": 64, "nhead_k": 4,
                                "scale_s": 0.5, "scale_p": 1.0, "scale_o": 1.0,
                                "window_left": -1, "window_right": 0, "mask_type": 2,
                                "stride_q": 192 if layout else 64 * 192,
                                "head_stride_q": q_len * 192 if layout else 192,
                                "batch_stride_q": q_len * 64 * 192,
                                "stride_o": 192 if layout else 64 * 192,
                                "head_stride_o": q_len * 192 if layout else 192,
                                "batch_stride_o": q_len * 64 * 192,
                                "stride_k": 192, "head_stride_k": 256 * 192,
                                "batch_stride_k": 4 * 256 * 192,
                                "stride_v": 192, "head_stride_v": 256 * 192,
                                "batch_stride_v": 4 * 256 * 192,
                            }
                            for field, value in expected_fields.items():
                                self.assertEqual(result[field], value, field)

    def test_cpp_exceptions_cannot_escape_c_abi(self):
        for optimization in self.binaries:
            for mode in (7, 8):
                for time_it in (0, 1):
                    with self.subTest(optimization=optimization, mode=mode, time_it=time_it):
                        result = self.invoke(optimization, mode=mode, time_it=time_it)
                        self.assertEqual(result["result"], -2.0)
                        self.assertEqual(result["calls"], 1)

    def test_invalid_elapsed_is_execution_error_not_rejection(self):
        cases = [(mode, time_it) for mode in (2, 3, 4, 5, 6, 10) for time_it in (0, 1)]
        cases += [(0, 1), (9, 1)]
        for optimization in self.binaries:
            for mode, time_it in cases:
                with self.subTest(optimization=optimization, mode=mode, time_it=time_it):
                    result = self.invoke(optimization, mode=mode, time_it=time_it)
                    self.assertEqual(result["result"], -2.0)
                    self.assertEqual(result["calls"], 1)

    def test_invalid_arguments_reject_before_specialization(self):
        cases = [{"null_pointer": index} for index in range(6)] + [
            {"q_len": 0}, {"q_len": -1}, {"q_len": -2147483648},
            {"kv_len": 63}, {"kv_len": 0}, {"kv_len": -2147483648},
            {"pages": 0}, {"pages": -1}, {"pages": -2147483648},
            {"pages": 1}, {"layout": -1}, {"layout": 2},
            {"stride": 128}, {"stride": 0}, {"stride": -2147483648},
            # Actual CK stride fields are int32: both layouts must reject a
            # batch stride that cannot be represented, before multiplication.
            {"q_len": 174763, "kv_len": 174763, "pages": 683},
            {"q_len": 174763, "kv_len": 174763, "pages": 683, "layout": 1},
            {"q_len": 2147483647, "kv_len": 2147483647, "pages": 8388608},
            {"kv_len": 2147483647, "pages": 8388607},
        ]
        for optimization in self.binaries:
            for case in cases:
                with self.subTest(optimization=optimization, case=case):
                    result = self.invoke(optimization, **case)
                    self.assertEqual(result["result"], -1.0)
                    self.assertEqual(result["calls"], 0)


if __name__ == "__main__":
    unittest.main()
