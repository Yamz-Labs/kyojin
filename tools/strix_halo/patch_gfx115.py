#!/usr/bin/env python3
"""Add gfx11.5 (RDNA3.5 / Strix Halo) WMMA support to the exllamav3 AMD fork.

Fragment layout ORACLE-VERIFIED on gfx1151 (ROCm 7.13) -- candidate layouts
scored against an fp32 CPU reference, exact match at maxerr=0.0:
    A: lane l, reg r 0..15 -> A[row = l&15][k = r]
    B: lane l, reg r 0..15 -> B[k = r][col = l&15]
    C: lane l, reg r 0..7  -> C[row = 2*r + (l>>4)][col = l&15]   INTERLEAVED

vs gfx12 (RDNA4): different builtin (_w32 vs _w32_gfx12), 16 vs 8 halves per
lane for A/B, and interleaved vs blocked C rows. Hence a separate path rather
than widening the gfx12 arch check.

Run from the repo root. Idempotent: re-running is a no-op.
"""
import re
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent
EXT = ROOT / "exllamav3" / "exllamav3_ext"

changed = []


def edit(path, old, new, desc, count=1):
    p = pathlib.Path(path)
    s = p.read_text()
    if new in s:
        print(f"  SKIP (already applied): {desc}")
        return False
    n = s.count(old)
    if n != count:
        print(f"  FAIL: {desc}\n        expected {count} occurrence(s) of anchor, found {n}")
        sys.exit(1)
    p.write_text(s.replace(old, new, count))
    print(f"  OK: {desc}")
    changed.append(str(p))
    return True


print("=== 1. hip_mma.cuh: gfx11.5 family macro + typedefs ===")

mma = EXT / "hip" / "hip_mma.cuh"

edit(mma,
"""// WMMA operand registers on gfx12: 8 fp16 per lane (A and B) -> one 128-bit vector reg.
// Use the HIP vector types so the compiler sees the required v8f16 register form.
#if defined(__gfx1200__) || defined(__gfx1201__)
using HipFp16x8 = __attribute__((__vector_size__(8 * sizeof(__fp16)))) __fp16;
using HipFp32x8 = __attribute__((__vector_size__(8 * sizeof(float)))) float;
#endif""",
"""// Arch family macros. gfx12 (RDNA4) and gfx11.5 (RDNA3.5) both have a wave32
// v_wmma_f32_16x16x16_f16, but with DIFFERENT builtins and DIFFERENT fragment
// layouts -- they are not interchangeable. See the layout comments below.
#if defined(__gfx1200__) || defined(__gfx1201__)
    #define EXL3_HIP_WMMA_GFX12 1
#endif
#if defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__)
    #define EXL3_HIP_WMMA_GFX115 1
#endif

// WMMA operand registers on gfx12: 8 fp16 per lane (A and B) -> one 128-bit vector reg.
// Use the HIP vector types so the compiler sees the required v8f16 register form.
#if defined(EXL3_HIP_WMMA_GFX12)
using HipFp16x8 = __attribute__((__vector_size__(8 * sizeof(__fp16)))) __fp16;
#endif
// gfx11.5 takes 16 fp16 per lane (the full k range) -> one 256-bit vector reg.
#if defined(EXL3_HIP_WMMA_GFX115)
using HipFp16x16 = __attribute__((__vector_size__(16 * sizeof(__fp16)))) __fp16;
#endif
#if defined(EXL3_HIP_WMMA_GFX12) || defined(EXL3_HIP_WMMA_GFX115)
using HipFp32x8 = __attribute__((__vector_size__(8 * sizeof(float)))) float;
#endif""",
"family macros + HipFp16x16 typedef")

# Swap the remaining raw gfx1200/gfx1201 guards in hip_mma.cuh to the macro so
# the file reads consistently (there are several).
s = mma.read_text()
raw = "#if defined(__gfx1200__) || defined(__gfx1201__)"
if raw in s:
    s = s.replace(raw, "#if defined(EXL3_HIP_WMMA_GFX12)")
    mma.write_text(s)
    print(f"  OK: normalized {raw!r} -> EXL3_HIP_WMMA_GFX12 in hip_mma.cuh")
    changed.append(str(mma))

# Append the gfx11.5 implementation just before the generic mma_ab_h_hip dispatcher.
anchor = """// mma m16n8k16 x2 (n=16)  ->  one v_wmma_f32_16x16x16_f16"""

gfx115_impl = r"""// =====================================================================================
// gfx11.5 (RDNA3.5, Strix Halo):  v_wmma_f32_16x16x16_f16   (wave32)
// =====================================================================================
// D[16x16 f32] = A[16x16 f16] * B[16x16 f16] + C[16x16 f32], one instruction per tile.
//
// FRAGMENT LAYOUT (ORACLE-VERIFIED on gfx1151, ROCm 7.13; candidate layouts scored
// against an fp32 CPU reference, exact match at maxerr = 0.0):
//     A: lane l, reg r 0..15  ->  A[row = l & 15][k = r]        (lane = one row, all 16 k)
//     B: lane l, reg r 0..15  ->  B[k = r][col = l & 15]        (lane = one column)
//     C: lane l, reg r 0..7   ->  C[row = 2*r + (l >> 4)][col = l & 15]
//
// THREE differences from gfx12 (RDNA4) -- this is why the gfx12 path cannot simply have
// its arch check widened:
//   1. the builtin is ..._f16_w32, NOT ..._f16_w32_gfx12 (the latter does not exist here)
//   2. A and B carry 16 halves per lane (the full k range), not 8 with k split across
//      the two half-warps
//   3. C rows INTERLEAVE as 2*reg + half, they do not block as (half)*8 + reg
//
// Like gfx12, operands are assembled through the warp-private LDS staging buffer rather
// than __shfl_sync (see the gfx12 note above on the shuffle -> WMMA hazard).
#if defined(EXL3_HIP_WMMA_GFX115)

// Build the gfx11.5 A operand (16 halves = full k) from the CUDA m16n8k16 A fragment.
// CUDA semantics: lane l holds row l/4 (a01/a23 elem 0) and row l/4+8 (elem 1),
// k = 2*(l&3){,+1} for a01 and +8{,+9} for a23.
__device__ __forceinline__ HipFp16x16 assemble_a_frag_gfx115(const FragB& a01, const FragB& a23)
{
    const int lane = (int)(threadIdx.x & 31);
    const int wid  = hip_mma_warp_id();
    __half2 (*stg)[4] = hip_mma_stg[wid];

    stg[lane][0] = a01[0];   // row lane/4,     k 2t, 2t+1
    stg[lane][1] = a01[1];   // row lane/4 + 8, k 2t, 2t+1
    stg[lane][2] = a23[0];   // row lane/4,     k 2t+8, 2t+9
    stg[lane][3] = a23[1];   // row lane/4 + 8, k 2t+8, 2t+9
    __syncwarp();

    const int R    = lane & 15;         // AMD A row this lane owns
    const int base = 4 * (R & 7);       // the four CUDA lanes holding that row
    const int s_lo = (R < 8) ? 0 : 1;   // slot for k 0..7
    const int s_hi = (R < 8) ? 2 : 3;   // slot for k 8..15

    HipFp16x16 a;
    uint32_t* aw = reinterpret_cast<uint32_t*>(&a);   // 16 halves = 8 dwords
    #pragma unroll
    for (int j = 0; j < 4; ++j)
    {
        const __half2 lo = stg[base + j][s_lo];       // k 2j, 2j+1
        const __half2 hi = stg[base + j][s_hi];       // k 8+2j, 9+2j
        aw[j]     = *reinterpret_cast<const uint32_t*>(&lo);
        aw[4 + j] = *reinterpret_cast<const uint32_t*>(&hi);
    }
    return a;
}

// Build the gfx11.5 B operand (16 halves = full k) from the decoded tile halves.
// CUDA semantics: lane l holds col g = l>>2 (b_hi) and g+8 (b_lo), same k pattern.
__device__ __forceinline__ HipFp16x16 assemble_b_frag_gfx115(const FragB& b_hi, const FragB& b_lo)
{
    const int lane = (int)(threadIdx.x & 31);
    const int wid  = hip_mma_warp_id();
    __half2 (*stg)[4] = hip_mma_stg[wid];

    __syncwarp();            // the A operand read the same buffer
    stg[lane][0] = b_hi[0];  // col g,     k 2t, 2t+1
    stg[lane][1] = b_hi[1];  // col g,     k 2t+8, 2t+9
    stg[lane][2] = b_lo[0];  // col g + 8, k 2t, 2t+1
    stg[lane][3] = b_lo[1];  // col g + 8, k 2t+8, 2t+9
    __syncwarp();

    const int C    = lane & 15;         // AMD B column this lane owns
    const int base = 4 * (C & 7);
    const int s_lo = (C < 8) ? 0 : 2;   // slot for k 0..7
    const int s_hi = (C < 8) ? 1 : 3;   // slot for k 8..15

    HipFp16x16 b;
    uint32_t* bw = reinterpret_cast<uint32_t*>(&b);
    #pragma unroll
    for (int j = 0; j < 4; ++j)
    {
        const __half2 lo = stg[base + j][s_lo];
        const __half2 hi = stg[base + j][s_hi];
        bw[j]     = *reinterpret_cast<const uint32_t*>(&lo);
        bw[4 + j] = *reinterpret_cast<const uint32_t*>(&hi);
    }
    return b;
}

template <typename FragC_t>
__device__ __forceinline__ void mma_ab_h_hip_gfx115_preassembled_a(
    const HipFp16x16& a,
    const FragB& b_hi,
    const FragB& b_lo,
    FragC_t& c)
{
    static_assert(sizeof(FragC_t) == sizeof(HipFp32x8),
                  "gfx11.5 path accumulates in 8 fp32 per lane (FragC8)");
    HipFp16x16 b = assemble_b_frag_gfx115(b_hi, b_lo);
    HipFp32x8 d = *reinterpret_cast<HipFp32x8*>(&c);
    d = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, d);
    *reinterpret_cast<HipFp32x8*>(&c) = d;
}

// C fragment <-> tile map for the kernel epilogue:
//   C(lane, reg) = C[row = 2*reg + (lane >> 4)][col = lane & 15]
#endif  // EXL3_HIP_WMMA_GFX115

// mma m16n8k16 x2 (n=16)  ->  one v_wmma_f32_16x16x16_f16"""

edit(mma, anchor, gfx115_impl, "gfx11.5 assemble + mma functions", count=1)

# Extend the generic dispatcher so non-preassembled callers work too.
edit(mma,
"""#if defined(EXL3_HIP_WMMA_GFX12)
    HipFp16x8 a = assemble_a_frag_gfx12(a01, a23);
    mma_ab_h_hip_gfx12_preassembled_a(a, b_hi, b_lo, c);
#elif defined(__gfx90a__) || defined(__gfx94__) || defined(__gfx950__)""",
"""#if defined(EXL3_HIP_WMMA_GFX12)
    HipFp16x8 a = assemble_a_frag_gfx12(a01, a23);
    mma_ab_h_hip_gfx12_preassembled_a(a, b_hi, b_lo, c);
#elif defined(EXL3_HIP_WMMA_GFX115)
    HipFp16x16 a = assemble_a_frag_gfx115(a01, a23);
    mma_ab_h_hip_gfx115_preassembled_a(a, b_hi, b_lo, c);
#elif defined(__gfx90a__) || defined(__gfx94__) || defined(__gfx950__)""",
"generic mma_ab_h_hip dispatch for gfx11.5")

print()
print("=== 2. exl3_gemv.cu: family-aware support predicate ===")

gemv = EXT / "quant" / "exl3_gemv.cu"

edit(gemv,
"""bool exl3_gemv_supported(int device)
{
#if defined(USE_ROCM)
    static std::map<int, bool> support_cache;
    static std::mutex support_cache_mtx;
    std::lock_guard<std::mutex> lock(support_cache_mtx);
    auto it = support_cache.find(device);
    if (it != support_cache.end()) return it->second;

    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, device) != hipSuccess) return false;
    const char* arch = prop.gcnArchName;
    bool supported = (!std::strncmp(arch, "gfx1200", 7) || !std::strncmp(arch, "gfx1201", 7))
        && (arch[7] == '\\0' || arch[7] == ':');
    support_cache[device] = supported;
    return supported;
#else
    (void) device;
    return true;
#endif
}""",
"""// Which WMMA family does this device have? The two are NOT interchangeable:
// different builtins, different A/B widths, different C row mapping.
//   0 = none, 1 = gfx12 (RDNA4), 2 = gfx11.5 (RDNA3.5 / Strix Halo)
int exl3_gemv_wmma_family(int device)
{
#if defined(USE_ROCM)
    static std::map<int, int> family_cache;
    static std::mutex family_cache_mtx;
    std::lock_guard<std::mutex> lock(family_cache_mtx);
    auto it = family_cache.find(device);
    if (it != family_cache.end()) return it->second;

    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, device) != hipSuccess) return 0;
    const char* arch = prop.gcnArchName;
    auto is = [&](const char* name) {
        const size_t n = std::strlen(name);
        return !std::strncmp(arch, name, n) && (arch[n] == '\\0' || arch[n] == ':');
    };
    int family = 0;
    if (is("gfx1200") || is("gfx1201")) family = 1;
    else if (is("gfx1150") || is("gfx1151") || is("gfx1152")) family = 2;
    family_cache[device] = family;
    return family;
#else
    (void) device;
    return 1;
#endif
}

// True when the EXL3 WMMA GEMV can run. NOTE: the gfx12-only MoE/routing entry
// points must check exl3_gemv_wmma_family(device) == 1 instead of this.
bool exl3_gemv_supported(int device)
{
#if defined(USE_ROCM)
    return exl3_gemv_wmma_family(device) != 0;
#else
    (void) device;
    return true;
#endif
}""",
"exl3_gemv_wmma_family() + family-based exl3_gemv_supported()")

# The two MoE entry points are gfx12-only kernels; pin them to family 1.
edit(gemv,
"""    TORCH_CHECK(exl3_gemv_supported(device), "exl3_moe_gfx12_k3 requires gfx1200/gfx1201");""",
"""    TORCH_CHECK(exl3_gemv_wmma_family(device) == 1,
                "exl3_moe_gfx12_k3 requires gfx1200/gfx1201");""",
"pin exl3_moe_gfx12_k3 to gfx12")

edit(gemv,
"""    TORCH_CHECK(exl3_gemv_supported(device),
                "exl3_moe_gfx12_k3_prefill requires gfx1200/gfx1201");""",
"""    TORCH_CHECK(exl3_gemv_wmma_family(device) == 1,
                "exl3_moe_gfx12_k3_prefill requires gfx1200/gfx1201");""",
"pin exl3_moe_gfx12_k3_prefill to gfx12")

# Make sure <cstring> is available for strlen (strncmp already used).
s = gemv.read_text()
if "#include <cstring>" not in s:
    s = s.replace("#include <map>", "#include <map>\n#include <cstring>", 1)
    gemv.write_text(s)
    print("  OK: added <cstring> include")
    changed.append(str(gemv))

print()
print("=== 3. exl3_gemv.cuh / bindings: declare the new symbol ===")

cuh = EXT / "quant" / "exl3_gemv.cuh"
edit(cuh,
"bool exl3_gemv_supported(int device);",
"bool exl3_gemv_supported(int device);\nint exl3_gemv_wmma_family(int device);",
"declare exl3_gemv_wmma_family in header")

bindings = EXT / "bindings.cpp"
s = bindings.read_text()
n = s.count('m.def("exl3_gemv_supported", &exl3_gemv_supported, "exl3_gemv_supported");')
if 'exl3_gemv_wmma_family' in s:
    print("  SKIP (already applied): bind exl3_gemv_wmma_family")
else:
    s = s.replace(
        'm.def("exl3_gemv_supported", &exl3_gemv_supported, "exl3_gemv_supported");',
        'm.def("exl3_gemv_supported", &exl3_gemv_supported, "exl3_gemv_supported");\n'
        '    m.def("exl3_gemv_wmma_family", &exl3_gemv_wmma_family, "exl3_gemv_wmma_family");')
    bindings.write_text(s)
    print(f"  OK: bound exl3_gemv_wmma_family ({n} site(s))")
    changed.append(str(bindings))

print()
print("=== 4. exl3_gemv_kernel.cuh: A-fragment + mma call + epilogue ===")

kern = EXT / "quant" / "exl3_gemv_kernel.cuh"

edit(kern,
"""#if defined(__gfx1200__) || defined(__gfx1201__)
            // Every adjacent N tile uses the same A rows for this K slice. Assemble the full
            // operand before the B tiles reuse the warp-private staging buffer.
            HipFp16x8 a_frag = assemble_a_frag_gfx12(a01, a23);
#endif""",
"""#if defined(EXL3_HIP_WMMA_GFX12)
            // Every adjacent N tile uses the same A rows for this K slice. Assemble the full
            // operand before the B tiles reuse the warp-private staging buffer.
            HipFp16x8 a_frag = assemble_a_frag_gfx12(a01, a23);
#elif defined(EXL3_HIP_WMMA_GFX115)
            HipFp16x16 a_frag = assemble_a_frag_gfx115(a01, a23);
#endif""",
"A fragment assembly for gfx11.5")

edit(kern,
"""#if defined(__gfx1200__) || defined(__gfx1201__)
                mma_ab_h_hip_gfx12_preassembled_a(a_frag, f0, f1, acc[t]);
#else
                mma_ab_h_hip(a01, a23, f0, f1, acc[t]);
#endif""",
"""#if defined(EXL3_HIP_WMMA_GFX12)
                mma_ab_h_hip_gfx12_preassembled_a(a_frag, f0, f1, acc[t]);
#elif defined(EXL3_HIP_WMMA_GFX115)
                mma_ab_h_hip_gfx115_preassembled_a(a_frag, f0, f1, acc[t]);
#else
                mma_ab_h_hip(a01, a23, f0, f1, acc[t]);
#endif""",
"mma call for gfx11.5")

# --- Epilogue: MMODE 2 (9..16 rows) ---
# gfx12: lane<16 -> rows 0..7 (reg = row); lane>=16 -> rows 8..15.
# gfx11.5: row = 2*reg + half, so rows 0..7 come from regs 0..3 across BOTH
# halves, and rows 8..15 from regs 4..7 across both halves.
edit(kern,
"""            if (lane < 16)
            {
                #pragma unroll
                for (int t = 0; t < WNT; ++t)
                    #pragma unroll
                    for (int r = 0; r < 8; ++r)
                        sh_red[warp][r][t * 16 + lane] = acc[t][r];
            }
            __syncthreads();

            const int rows_out = min(size_m, 8);""",
"""#if defined(EXL3_HIP_WMMA_GFX115)
            // gfx11.5 C rows interleave (row = 2*reg + half), so the low eight rows
            // live in regs 0..3 of BOTH half-warps rather than in lanes 0..15.
            {
                const int h = lane >> 4;
                const int c = lane & 15;
                #pragma unroll
                for (int t = 0; t < WNT; ++t)
                    #pragma unroll
                    for (int r = 0; r < 4; ++r)
                        sh_red[warp][2 * r + h][t * 16 + c] = acc[t][r];
            }
#else
            if (lane < 16)
            {
                #pragma unroll
                for (int t = 0; t < WNT; ++t)
                    #pragma unroll
                    for (int r = 0; r < 8; ++r)
                        sh_red[warp][r][t * 16 + lane] = acc[t][r];
            }
#endif
            __syncthreads();

            const int rows_out = min(size_m, 8);""",
"MMODE2 low-half epilogue for gfx11.5")

edit(kern,
"""                if (lane >= 16)
                {
                    #pragma unroll
                    for (int t = 0; t < WNT; ++t)
                        #pragma unroll
                        for (int r = 0; r < 8; ++r)
                            sh_red[warp][r][t * 16 + (lane & 15)] = acc[t][r];
                }
                __syncthreads();""",
"""#if defined(EXL3_HIP_WMMA_GFX115)
                // upper eight rows: regs 4..7, both half-warps (row = 2*(r-4) + half + 8)
                {
                    const int h = lane >> 4;
                    const int c = lane & 15;
                    #pragma unroll
                    for (int t = 0; t < WNT; ++t)
                        #pragma unroll
                        for (int r = 4; r < 8; ++r)
                            sh_red[warp][2 * (r - 4) + h][t * 16 + c] = acc[t][r];
                }
#else
                if (lane >= 16)
                {
                    #pragma unroll
                    for (int t = 0; t < WNT; ++t)
                        #pragma unroll
                        for (int r = 0; r < 8; ++r)
                            sh_red[warp][r][t * 16 + (lane & 15)] = acc[t][r];
                }
#endif
                __syncthreads();""",
"MMODE2 high-half epilogue for gfx11.5")

# --- Epilogue: MMODE 0/1 (1..8 rows) ---
# gfx12: all valid rows sit in lanes 0..15, reg = row.
# gfx11.5: row = 2*reg + half -> rows 0..7 span regs 0..3 in both halves.
edit(kern,
"""            if (lane < 16)
            {
                #pragma unroll
                for (int t = 0; t < WNT; ++t)
                    #pragma unroll
                    for (int r = 0; r < min(ROWS, 8); ++r)
                        sh_red[warp][r][t * 16 + lane] = acc[t][r];
            }
            __syncthreads();""",
"""#if defined(EXL3_HIP_WMMA_GFX115)
            // gfx11.5: row = 2*reg + (lane>>4). For m <= 8 the rows are spread over
            // regs 0..3 of both half-warps, not regs 0..7 of the first one.
            {
                const int h = lane >> 4;
                const int c = lane & 15;
                #pragma unroll
                for (int t = 0; t < WNT; ++t)
                    #pragma unroll
                    for (int r = 0; r < 4; ++r)
                    {
                        const int row = 2 * r + h;
                        if (row < min(ROWS, 8))
                            sh_red[warp][row][t * 16 + c] = acc[t][r];
                    }
            }
#else
            if (lane < 16)
            {
                #pragma unroll
                for (int t = 0; t < WNT; ++t)
                    #pragma unroll
                    for (int r = 0; r < min(ROWS, 8); ++r)
                        sh_red[warp][r][t * 16 + lane] = acc[t][r];
            }
#endif
            __syncthreads();""",
"MMODE0/1 epilogue for gfx11.5")

print()
print("=== 5. Python-side gate: exl3.py ===")

py = ROOT / "exllamav3" / "modules" / "quant" / "exl3.py"
s = py.read_text()
if "_hip_gemv_support_cache" in s and "exl3_gemv_supported" in s:
    print("  OK: exl3.py already delegates to ext.exl3_gemv_supported() (now family-aware)")
else:
    print("  NOTE: exl3.py gate shape unexpected; inspect manually")

print()
print("=== SUMMARY ===")
for c in sorted(set(changed)):
    print("  modified:", c)
if not changed:
    print("  no changes (all already applied)")
