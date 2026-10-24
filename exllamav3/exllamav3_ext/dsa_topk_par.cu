// QSA dsa_topk, parallel form: one row of scores spread over nb blocks instead of one block.
//
// WHY (flat1, Step 3): ext.dsa_topk on ROCm is dsa_topk_gfx12, ONE 1024-thread block per row, and
// it sweeps the row four times (coarse histogram, fine histogram, count, write). At decode batch 1
// on a wide row (459K tokens -> 115K pools) that is 2.63 ms/token over 12 layers for a row that is
// 229 KB: 220x over its DRAM read floor, because a single block uses 1 of 40 CUs. The indexer that
// produces the row is already at its floor, so the selection is the only depth cost left.
//
// WHAT: the same two-pass radix select, cut into four kernels so no grid-wide barrier is needed.
//   1 hi    grid (nb, R): per-block 256-bin histogram of key >> 8 (LDS) merged into a per-row
//          256-bin global histogram with atomics; the last block of the row (atomic ticket) runs
//          find_descending_bucket(H, k) -> high bucket, count above.
//   2 lo    grid (nb, R): per-block 256-bin histogram of key & 0xff over the high bucket only;
//          the last block runs find_descending_bucket(low, k - count above) -> threshold, ties.
//   3 count grid (nb, R): per-block counts of keys above the threshold and of threshold ties;
//          the last block turns them into the per-block output offsets and the tie offsets (one
//          thread walks the nb <= 256 blocks) and fills the -1 padding of the output.
//   4 write grid (nb, R): each block re-reads its slice and writes its selected columns ascending,
//          at the offset the count pass computed. Ties inside a block are admitted in ascending
//          index order, exactly as the single-block kernel admits them (its per-thread ranges are
//          contiguous and ascending, so its tie prefix over ranges is a tie prefix over indices).
//
// EXACTNESS: identical to dsa_topk_gfx12 bit for bit, in the output buffer, not just as a set:
//   - the same key (topk_key) and the same validity test (key > 0x03ff, i.e. -inf and negative NaNs
//     excluded; positive NaNs rank above +inf; +0 above -0);
//   - the same histogram, the same descending cumulative search and the same
//     threshold / ties_needed formulas, so the same threshold key and the same number of admitted
//     ties;
//   - every key above the threshold is selected, plus the first ties_needed threshold keys in
//     ascending column order -- the block partition is ascending and disjoint, so prefixing ties
//     over blocks is prefixing them over columns;
//   - the selected columns are written ascending (block offsets ascending, ascending inside a
//     block), then -1 from the total to out_w.
// Fewer than k valid keys is the same case in both: the descending search finds no bucket, the
// threshold stays KEY_NEG_INF and every valid key is selected.
//
// BOUNDS (checked on the host before any launch):
//   R <= R_MAX (16), 1 <= k <= 512 <= out_w, out_w <= 512, width >= 1, stride >= width,
//   stride % 128 == 0, stride <= INT_MAX, width <= INT_MAX, nb in [1, NB_MAX].
//   1. Row base: scores + (int64) row * stride, row < R_MAX and stride <= INT_MAX, so the product
//      is computed in 64 bit and never wraps; every column read is < width <= stride.
//   2. Chunk bounds: lo = width * blk / nb and hi = width * (blk + 1) / nb in int64, so
//      0 <= lo <= hi <= width; both sweep loops (i < hi, i < width) read only [lo, hi).
//   3. Output writes: the count pass emits sel_off[blk] (exclusive prefix of the per-block selected
//      counts) and sel_off[blk + 1] - sel_off[blk] == the block's own count, so the write pass
//      fills exactly [sel_off[blk], sel_off[blk] + count) and the padding loop fills
//      [total, out_w); total <= k <= out_w, so no write leaves the row of `indices`.
//   4. Per-thread sub-ranges inside a slice: rb = lo + span * tid / THREADS,
//      re = lo + span * (tid + 1) / THREADS in int64, so they partition [lo, hi) exactly (each
//      column counted once, ties prefixed in ascending order).
//   5. LDS: 2 x 256 ints of histogram staging + 2 x WARPS ints of scan scratch <= 2.1 KB/block.
//   6. Workspace: WS_ROW = 1545 ints per row slot, WS_INTS = WS_ROW * R_MAX ints, allocated once
//      per device and zeroed. Every scratch index is inside its row slot, every row slot inside the
//      allocation; the four counter arrays are indexed by blk < nb <= NB_MAX.
//   7. Grid: (nb, R) with nb <= 256 and R <= 16, both far below the 65535 grid.y limit.
//   8. Ticket: exactly nb blocks per row each add 1, so exactly one block sees nb - 1 and merges;
//      that block resets the ticket to 0 for the next call, so no call needs a fresh memset.
//
// The workspace is a per-device static, shared by every call on that device and stream-ordered
// like dsa_topk_ws (allocated during warmup, never grown, so a captured graph cannot dangle).

#if defined(USE_ROCM)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "dsa_topk_par.cuh"
#include "util.cuh"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>

namespace {

constexpr int WAVE_SIZE = 32;
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / WAVE_SIZE;
constexpr int K_MAX = 512;
constexpr int R_MAX = 16;
constexpr int NB_MAX = 256;
constexpr int NB_MIN = 1;
constexpr uint16_t KEY_NEG_INF = 0x03ff;

// Columns per block when nb is chosen from the width (nb = min(NB_MAX, ceil(width / CHUNK))).
constexpr int CHUNK = 1024;

// Workspace layout, int32, per row slot
constexpr int OFF_HI = 0;                        // 256 coarse bins
constexpr int OFF_LO = OFF_HI + 256;             // 256 fine bins
constexpr int OFF_ST = OFF_LO + 256;             // 8 ints of select state
constexpr int OFF_CNT_HI = OFF_ST + 8;           // NB_MAX per-block counts above the threshold
constexpr int OFF_CNT_TIE = OFF_CNT_HI + NB_MAX; // NB_MAX per-block counts of threshold ties
constexpr int OFF_SEL_OFF = OFF_CNT_TIE + NB_MAX;// NB_MAX exclusive output offset per block
constexpr int OFF_TIE_OFF = OFF_SEL_OFF + NB_MAX;// NB_MAX exclusive tie offset per block
constexpr int OFF_TICKET = OFF_TIE_OFF + NB_MAX; // 1
constexpr int WS_ROW = OFF_TICKET + 1;           // 1545
constexpr int WS_INTS = WS_ROW * R_MAX;

constexpr int ST_HI_BUCKET = 0;
constexpr int ST_COUNT_ABOVE = 1;
constexpr int ST_THRESHOLD = 2;
constexpr int ST_TIES = 3;

__device__ __forceinline__ uint16_t topk_key(uint16_t bits)
{
    return bits & 0x8000 ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
}

// Identical to dsa_topk_gfx12's find_descending_bucket: the highest bucket whose running count from
// the top reaches target, with the count above it; (-1, total) when the row holds fewer than target
// valid keys.
__device__ __forceinline__ void find_descending_bucket(const int* hist, int target, int* result)
{
    int total = 0;
    for (int bucket = 255; bucket >= 0; --bucket)
    {
        if (total + hist[bucket] >= target)
        {
            result[0] = bucket;
            result[1] = total;
            return;
        }
        total += hist[bucket];
    }
    result[0] = -1;
    result[1] = total;
}

__device__ __forceinline__ void chunk_range(int width, int blk, int nb, int& lo, int& hi)
{
    lo = static_cast<int>(static_cast<int64_t>(width) * blk / nb);
    hi = static_cast<int>(static_cast<int64_t>(width) * (blk + 1) / nb);
}

// dsa_topk_gfx12's block_exclusive_scan, over one value (the write pass needs two).
__device__ __forceinline__ int block_exclusive_scan(
    int value, int lane, int warp, int* warp_sums, int* warp_offsets)
{
    int inclusive = value;
    #pragma unroll
    for (int offset = 1; offset < WAVE_SIZE; offset <<= 1)
    {
        const int other = __shfl_up_sync(0xffffffffu, inclusive, offset);
        if (lane >= offset) inclusive += other;
    }
    if (lane == WAVE_SIZE - 1) warp_sums[warp] = inclusive;
    __syncthreads();

    if (warp == 0)
    {
        int warp_inclusive = warp_sums[lane];
        #pragma unroll
        for (int offset = 1; offset < WAVE_SIZE; offset <<= 1)
        {
            const int other = __shfl_up_sync(0xffffffffu, warp_inclusive, offset);
            if (lane >= offset) warp_inclusive += other;
        }
        warp_offsets[lane] = warp_inclusive - warp_sums[lane];
    }
    __syncthreads();
    return warp_offsets[warp] + inclusive - value;
}

// Ticket protocol: every block of the row adds 1, the last one merges the row's histogram and
// leaves the ticket at 0 for the next call (same idiom as hc_mix_norm_1pass_kernel).
__device__ __forceinline__ bool take_ticket(int* ticket, int nb)
{
    __shared__ bool last_s;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last_s = atomicAdd(ticket, 1) == nb - 1;
    __syncthreads();
    if (!last_s) return false;
    __threadfence();
    if (threadIdx.x == 0) *ticket = 0;
    return true;
}

__global__ __launch_bounds__(THREADS)
void dsa_topk_par_hi_kernel
(
    const half* __restrict__ scores,
    int* __restrict__ ws,                 // (R, WS_ROW) int32, zeroed
    int width,
    int stride,
    int K,
    int nb
)
{
    const int row = blockIdx.y;
    const int blk = blockIdx.x;
    const int tid = threadIdx.x;
    int* w = ws + static_cast<int64_t>(row) * WS_ROW;
    __shared__ int hist[256];

    for (int i = tid; i < 256; i += THREADS) hist[i] = 0;
    __syncthreads();

    const half* row_scores = scores + static_cast<int64_t>(row) * stride;
    int lo, hi;
    chunk_range(width, blk, nb, lo, hi);
    for (int i = lo + tid; i < hi; i += THREADS)
    {
        const uint16_t key = topk_key(__half_as_ushort(row_scores[i]));
        if (key > KEY_NEG_INF) atomicAdd(&hist[key >> 8], 1);
    }
    __syncthreads();

    int* hist_hi = w + OFF_HI;
    for (int i = tid; i < 256; i += THREADS)
        if (hist[i]) atomicAdd(&hist_hi[i], hist[i]);

    if (!take_ticket(w + OFF_TICKET, nb)) return;

    __shared__ int bins[256];
    if (tid < 256) bins[tid] = hist_hi[tid];
    __syncthreads();
    if (tid == 0)
    {
        int search[2];
        find_descending_bucket(bins, K, search);
        w[OFF_ST + ST_HI_BUCKET] = search[0];
        w[OFF_ST + ST_COUNT_ABOVE] = search[0] >= 0 ? search[1] : 0;
        if (search[0] < 0)
        {
            // fewer valid keys than k: every valid key is selected, no ties
            w[OFF_ST + ST_THRESHOLD] = KEY_NEG_INF;
            w[OFF_ST + ST_TIES] = 0;
        }
    }
    __syncthreads();
    // this row's coarse bins are zero again for the next call (read above is complete)
    for (int i = tid; i < 256; i += THREADS) hist_hi[i] = 0;
}

__global__ __launch_bounds__(THREADS)
void dsa_topk_par_lo_kernel
(
    const half* __restrict__ scores,
    int* __restrict__ ws,
    int width,
    int stride,
    int K,
    int nb
)
{
    const int row = blockIdx.y;
    const int blk = blockIdx.x;
    const int tid = threadIdx.x;
    int* w = ws + static_cast<int64_t>(row) * WS_ROW;
    // written by the previous kernel: a kernel boundary is a device-scope fence, so a plain load
    // cannot serve a stale line
    const int hi_bucket = w[OFF_ST + ST_HI_BUCKET];
    if (hi_bucket < 0) return;

    __shared__ int hist[256];
    for (int i = tid; i < 256; i += THREADS) hist[i] = 0;
    __syncthreads();

    const half* row_scores = scores + static_cast<int64_t>(row) * stride;
    int lo, hi;
    chunk_range(width, blk, nb, lo, hi);
    for (int i = lo + tid; i < hi; i += THREADS)
    {
        const uint16_t key = topk_key(__half_as_ushort(row_scores[i]));
        if (key > KEY_NEG_INF && (key >> 8) == hi_bucket) atomicAdd(&hist[key & 0xff], 1);
    }
    __syncthreads();

    int* hist_lo = w + OFF_LO;
    for (int i = tid; i < 256; i += THREADS)
        if (hist[i]) atomicAdd(&hist_lo[i], hist[i]);

    if (!take_ticket(w + OFF_TICKET, nb)) return;

    __shared__ int bins[256];
    const int count_above = w[OFF_ST + ST_COUNT_ABOVE];
    if (tid < 256) bins[tid] = hist_lo[tid];
    __syncthreads();
    if (tid == 0)
    {
        int search[2];
        find_descending_bucket(bins, K - count_above, search);
        w[OFF_ST + ST_THRESHOLD] = (hi_bucket << 8) | search[0];
        w[OFF_ST + ST_TIES] = K - count_above - search[1];
    }
    __syncthreads();
    for (int i = tid; i < 256; i += THREADS) hist_lo[i] = 0;
}

__global__ __launch_bounds__(THREADS)
void dsa_topk_par_count_kernel
(
    const half* __restrict__ scores,
    int* __restrict__ ws,
    int* __restrict__ out,                // (R, out_w) int32
    int width,
    int stride,
    int out_w,
    int nb
)
{
    const int row = blockIdx.y;
    const int blk = blockIdx.x;
    const int tid = threadIdx.x;
    int* w = ws + static_cast<int64_t>(row) * WS_ROW;
    const int threshold = w[OFF_ST + ST_THRESHOLD];
    const uint16_t thr = static_cast<uint16_t>(threshold);
    const int ties_needed = w[OFF_ST + ST_TIES];

    const half* row_scores = scores + static_cast<int64_t>(row) * stride;
    int lo, hi;
    chunk_range(width, blk, nb, lo, hi);
    int hi_count = 0, tie_count = 0;
    for (int i = lo + tid; i < hi; i += THREADS)
    {
        const uint16_t key = topk_key(__half_as_ushort(row_scores[i]));
        if (key > thr && key > KEY_NEG_INF) ++hi_count;
        else if (key == thr && key > KEY_NEG_INF) ++tie_count;
    }

    const int lane = tid % WAVE_SIZE;
    const int warp = tid / WAVE_SIZE;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        hi_count += __shfl_down_sync(0xffffffffu, hi_count, offset);
        tie_count += __shfl_down_sync(0xffffffffu, tie_count, offset);
    }
    __shared__ int red[2 * WARPS];
    __shared__ int s_total;
    if (lane == 0) { red[warp] = hi_count; red[WARPS + warp] = tie_count; }
    __syncthreads();
    if (tid == 0)
    {
        int h = 0, t = 0;
        for (int w_ = 0; w_ < WARPS; ++w_) { h += red[w_]; t += red[WARPS + w_]; }
        w[OFF_CNT_HI + blk] = h;
        w[OFF_CNT_TIE + blk] = t;
    }

    if (!take_ticket(w + OFF_TICKET, nb)) return;

    // Exclusive prefixes over the blocks: ties by tie count, output slots by selected count
    // (high + the ties this block may admit). nb <= NB_MAX (256), so a single thread walks the
    // blocks; a warp-shuffle walk would need a carry across 32-wide tiles (nb > 32 otherwise
    // restarts the offsets, and a prefix over high counts alone forgets earlier ties, which can
    // drive block 0's output offset negative and out of bounds).
    if (tid == 0)
    {
        int tie_run = 0, sel_run = 0;
        for (int b = 0; b < nb; ++b)
        {
            const int c_hi = w[OFF_CNT_HI + b];
            const int c_tie = w[OFF_CNT_TIE + b];
            w[OFF_TIE_OFF + b] = tie_run;
            tie_run += c_tie;
            const int admitted = max(0, min(c_tie, ties_needed - w[OFF_TIE_OFF + b]));
            w[OFF_SEL_OFF + b] = sel_run;
            sel_run += c_hi + admitted;
        }
        s_total = sel_run;
    }
    __syncthreads();

    int* row_out = out + static_cast<int64_t>(row) * out_w;
    for (int p = s_total + tid; p < out_w; p += THREADS) row_out[p] = -1;
}

__global__ __launch_bounds__(THREADS)
void dsa_topk_par_write_kernel
(
    const half* __restrict__ scores,
    int* __restrict__ ws,
    int* __restrict__ out,                // (R, out_w) int32
    int width,
    int stride,
    int out_w,
    int nb
)
{
    const int row = blockIdx.y;
    const int blk = blockIdx.x;
    const int tid = threadIdx.x;
    int* w = ws + static_cast<int64_t>(row) * WS_ROW;
    const uint16_t thr = static_cast<uint16_t>(w[OFF_ST + ST_THRESHOLD]);
    const int ties_needed = w[OFF_ST + ST_TIES];
    const int local_ties_block = max(0, min(w[OFF_CNT_TIE + blk], ties_needed - w[OFF_TIE_OFF + blk]));

    const half* row_scores = scores + static_cast<int64_t>(row) * stride;
    int lo, hi;
    chunk_range(width, blk, nb, lo, hi);
    const int span = hi - lo;
    const int range_begin = lo + static_cast<int>(static_cast<int64_t>(span) * tid / THREADS);
    const int range_end = lo + static_cast<int>(static_cast<int64_t>(span) * (tid + 1) / THREADS);

    int high_count = 0, tie_count = 0;
    for (int i = range_begin; i < range_end; ++i)
    {
        const uint16_t key = topk_key(__half_as_ushort(row_scores[i]));
        if (key > thr && key > KEY_NEG_INF) ++high_count;
        else if (key == thr && key > KEY_NEG_INF) ++tie_count;
    }

    // Each thread owns one contiguous range, so prefixing its local output count produces ascending
    // global indices. A separate tie prefix lets only the earliest threshold ties survive.
    const int lane = tid % WAVE_SIZE;
    const int warp = tid / WAVE_SIZE;
    // WAVE_SIZE entries, not WARPS: the scan has warp 0 read warp_sums[lane] for all 32 lanes
    __shared__ int sums_t[WAVE_SIZE], offs_t[WAVE_SIZE], sums_s[WAVE_SIZE], offs_s[WAVE_SIZE];
    const int tie_offset = block_exclusive_scan(tie_count, lane, warp, sums_t, offs_t);
    const int local_ties = max(0, min(tie_count, local_ties_block - tie_offset));
    int output = block_exclusive_scan(high_count + local_ties, lane, warp, sums_s, offs_s) +
                 w[OFF_SEL_OFF + blk];

    int* row_out = out + static_cast<int64_t>(row) * out_w;
    int seen_ties = 0;
    for (int i = range_begin; i < range_end; ++i)
    {
        const uint16_t key = topk_key(__half_as_ushort(row_scores[i]));
        if (key > thr && key > KEY_NEG_INF) row_out[output++] = i;
        else if (key == thr && key > KEY_NEG_INF)
        {
            if (seen_ties < local_ties) row_out[output++] = i;
            ++seen_ties;
        }
    }
}

// Fixed-size per-device workspace, zeroed once: the kernels leave it zeroed for the next call.
static at::Tensor& dsa_topk_par_ws(int device)
{
    static std::mutex mtx;
    static std::map<int, at::Tensor> ws;
    std::lock_guard<std::mutex> lock(mtx);
    auto it = ws.find(device);
    if (it == ws.end())
    {
        at::Tensor t = at::zeros
        (
            {(int64_t) WS_INTS},
            at::TensorOptions().dtype(at::kInt).device(at::Device(at::kCUDA, device))
        );
        it = ws.emplace(device, std::move(t)).first;
    }
    return it->second;
}

} // namespace

void dsa_topk_par(const at::Tensor& scores, at::Tensor& indices, int64_t k, int64_t nb_in)
{
    TORCH_CHECK(scores.is_cuda() && indices.is_cuda(), "dsa_topk_par requires device tensors");
    const at::cuda::OptionalCUDAGuard device_guard(scores.device());
    const int device = scores.get_device();
    TORCH_CHECK(indices.device() == scores.device(), "dsa_topk_par tensors must share a device");
    TORCH_CHECK(scores.dtype() == at::kHalf && indices.dtype() == at::kInt,
                "dsa_topk_par requires fp16 scores and int32 indices");
    TORCH_CHECK(scores.dim() == 2 && indices.dim() == 2 && scores.size(0) == indices.size(0),
                "dsa_topk_par requires matching 2D tensors");
    TORCH_CHECK(scores.size(1) <= INT_MAX && scores.stride(0) <= INT_MAX,
                "dsa_topk_par requires score width and row stride <= INT_MAX");
    TORCH_CHECK(k >= 1 && k <= K_MAX, "dsa_topk_par requires 1 <= k <= 512");
    TORCH_CHECK(scores.size(1) >= 1 && scores.stride(1) == 1 && scores.stride(0) >= scores.size(1) &&
                scores.stride(0) % 128 == 0,
                "dsa_topk_par requires QSA scores with T >= 1 and a 128-aligned row stride");
    TORCH_CHECK(indices.size(1) >= k && indices.size(1) <= K_MAX && indices.is_contiguous(),
                "dsa_topk_par requires contiguous int32 [R, W] output with k <= W <= 512");
    TORCH_CHECK(scores.size(0) <= R_MAX, "dsa_topk_par requires at most 16 rows");
    if (scores.size(0) == 0) return;

    const int rows = scores.size(0);
    const int width = scores.size(1);
    const int stride = scores.stride(0);
    const int out_w = indices.size(1);
    int nb = nb_in > 0 ? static_cast<int>(nb_in)
                       : std::min(NB_MAX, std::max(NB_MIN, (width + CHUNK - 1) / CHUNK));
    TORCH_CHECK(nb >= NB_MIN && nb <= NB_MAX, "dsa_topk_par requires 1 <= nb <= 256");

    int* ws = dsa_topk_par_ws(device).data_ptr<int>();
    int* out = indices.data_ptr<int>();
    const half* sc = reinterpret_cast<const half*>(scores.data_ptr());
    hipStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
    const dim3 grid(nb, rows);
    dsa_topk_par_hi_kernel<<<grid, THREADS, 0, stream>>>(sc, ws, width, stride, (int) k, nb);
    cuda_check(hipPeekAtLastError());
    dsa_topk_par_lo_kernel<<<grid, THREADS, 0, stream>>>(sc, ws, width, stride, (int) k, nb);
    cuda_check(hipPeekAtLastError());
    dsa_topk_par_count_kernel<<<grid, THREADS, 0, stream>>>(sc, ws, out, width, stride, out_w, nb);
    cuda_check(hipPeekAtLastError());
    dsa_topk_par_write_kernel<<<grid, THREADS, 0, stream>>>(sc, ws, out, width, stride, out_w, nb);
    cuda_check(hipPeekAtLastError());
}

#endif // USE_ROCM