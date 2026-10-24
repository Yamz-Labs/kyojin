"""
Multi-row router top-k (EXL3_ROUTER_TOPK, default 1, read per call). On ROCm the ext has no multi-row
routing_std, so prefill routing ran torch.sort(stable = True) over all experts per row plus ~5 staging copies
(~2.6 ms per 16384-row layer). This selects the same top-k directly: per row, k rounds of
(max, lowest index among the maxima), i.e. the order of a stable descending sort, NaN first as in torch.sort.
Values are copied, not computed, so the softmax that follows sees the same inputs: bit-identical routing.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _topk_kernel(x_ptr, v_ptr, i_ptr, E, XS, K: tl.constexpr, BE: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    o = tl.arange(0, BE)
    m = o < E
    x = tl.load(x_ptr + r * XS + o, mask = m, other = 0.0).to(tl.float32)
    avail = m
    for j in tl.static_range(K):
        nan = avail & (x != x)
        hv = tl.max(tl.where(avail, tl.where(nan, float("inf"), x), float("-inf")), axis = 0)
        any_nan = tl.max(nan.to(tl.int32), axis = 0) > 0
        cand = tl.where(any_nan, nan, avail & (x == hv))
        idx = tl.min(tl.where(cand, o, BE), axis = 0)
        # x[idx] == hv (a signed zero may come back as the other zero: same softmax)
        tl.store(v_ptr + r * K + j, tl.where(any_nan, float("nan"), hv))
        tl.store(i_ptr + r * K + j, idx.to(tl.int64))
        avail = avail & (o != idx)


def router_topk(logits: torch.Tensor, k: int):
    """logits (R, E) fp16/bf16/fp32, unit column stride -> (values (R, k) fp32, indices (R, k) int64),
    equal to torch.sort(logits.float(), -1, descending = True, stable = True)[:, :k]."""
    R, E = logits.shape
    assert logits.stride(1) == 1 and 1 <= k <= E
    v = torch.empty((R, k), dtype = torch.float, device = logits.device)
    i = torch.empty((R, k), dtype = torch.long, device = logits.device)
    if R:
        _topk_kernel[(R,)](logits, v, i, E, logits.stride(0), k, triton.next_power_of_2(E), num_warps = 4)
    return v, i
