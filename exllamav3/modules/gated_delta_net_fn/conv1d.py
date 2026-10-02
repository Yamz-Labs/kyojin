import os
import torch
import torch.nn.functional as F
from ...util.tensor import get_for_device, buffered_arange, host_cuts
from ...ext import exllamav3_ext as ext

# Above this length the triton kernel splits into separate output/state kernels and its launch
# overhead is amortized anyway; the CUDA kernel keeps the conv window in registers per thread
# so it only makes sense for short sequences (decode and SD verification steps)
MAX_CUDA_SEQLEN = 32
MAX_CUDA_K = 16

import triton
import triton.language as tl


@triton.jit
def _causal_conv1d_update_slotted_kernel(
    x,
    conv_state,
    slots,
    weight,
    bias,
    out,
    dim: tl.constexpr,
    seq_len,             # runtime: chunk length varies per job; only masks and address math
    sxb, sxd, sxs,       # x strides (batch, channel, time): x may be a transposed fp16 view
    state_size: tl.constexpr,
    conv_kernel_size: tl.constexpr,
    history: tl.constexpr,
    has_bias: tl.constexpr,
    transpose_output: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_STATE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_s = tl.arange(0, BLOCK_S)
    offs_k = tl.arange(0, BLOCK_K)
    offs_state = tl.arange(0, BLOCK_STATE)

    slot = tl.load(slots + pid_b)
    mask_d = offs_d < dim
    mask_s = offs_s < seq_len
    mask_state = offs_state < state_size

    acc = tl.zeros((BLOCK_D, BLOCK_S), dtype = tl.float32)

    for k in range(BLOCK_K):
        if k < conv_kernel_size:
            src_t = offs_s + k + 1
            from_x = src_t >= conv_kernel_size
            x_t = src_t - conv_kernel_size

            state_vals = tl.load(
                conv_state + (slot * dim + offs_d[:, None]) * state_size + src_t[None, :],
                mask = mask_d[:, None] & mask_s[None, :] & (src_t[None, :] < conv_kernel_size),
                other = 0.0,
            )
            x_vals = tl.load(
                x + pid_b * sxb + offs_d[:, None] * sxd + x_t[None, :] * sxs,
                mask = mask_d[:, None] & mask_s[None, :] & from_x[None, :] & (x_t[None, :] >= 0),
                other = 0.0,
            )
            x_vals = x_vals.to(conv_state.dtype.element_ty)
            vals = tl.where(from_x[None, :], x_vals, state_vals)
            w = tl.load(weight + offs_d * conv_kernel_size + k, mask = mask_d, other = 0.0)
            acc += vals * w[:, None]

    if has_bias:
        b = tl.load(bias + offs_d, mask = mask_d, other = 0.0)
        acc += b[:, None]

    acc = acc * tl.sigmoid(acc)
    if transpose_output:
        tl.store(
            out + (pid_b * seq_len + offs_s[None, :]) * dim + offs_d[:, None],
            acc,
            mask = mask_d[:, None] & mask_s[None, :],
        )
    else:
        tl.store(
            out + (pid_b * dim + offs_d[:, None]) * seq_len + offs_s[None, :],
            acc,
            mask = mask_d[:, None] & mask_s[None, :],
        )

    history_write_size = tl.where(state_size < conv_kernel_size + seq_len, state_size, conv_kernel_size + seq_len)
    state_write_size = tl.where(history, history_write_size, conv_kernel_size)
    dst_start = tl.where(history, state_size - state_write_size, 0)
    history_src_start = tl.where(conv_kernel_size + seq_len > state_size, conv_kernel_size + seq_len - state_size, 0)
    no_history_start = seq_len
    src_t = tl.where(history, history_src_start + offs_state - dst_start, no_history_start + offs_state)
    valid_state = (offs_state >= dst_start) & (offs_state < dst_start + state_write_size)
    from_x = src_t >= conv_kernel_size
    x_t = src_t - conv_kernel_size
    state_vals = tl.load(
        conv_state + (slot * dim + offs_d[:, None]) * state_size + src_t[None, :],
        mask = mask_d[:, None] & valid_state[None, :] & (src_t[None, :] >= 0) & (src_t[None, :] < conv_kernel_size),
        other = 0.0,
    )
    x_vals = tl.load(
        x + pid_b * sxb + offs_d[:, None] * sxd + x_t[None, :] * sxs,
        mask = mask_d[:, None] & valid_state[None, :] & from_x[None, :] & (x_t[None, :] >= 0),
        other = 0.0,
    )
    x_vals = x_vals.to(conv_state.dtype.element_ty)
    new_state = tl.where(from_x[None, :], x_vals, state_vals)
    tl.store(
        conv_state + (slot * dim + offs_d[:, None]) * state_size + offs_state[None, :],
        new_state,
        mask = mask_d[:, None] & mask_state[None, :] & valid_state[None, :],
    )


@triton.jit
def _causal_conv1d_update_slotted_output_kernel(
    x,
    conv_state,
    slots,
    weight,
    bias,
    out,
    dim: tl.constexpr,
    seq_len,             # runtime: chunk length varies per job; only masks and address math
    sxb, sxd, sxs,       # x strides (batch, channel, time): x may be a transposed fp16 view
    state_size: tl.constexpr,
    conv_kernel_size: tl.constexpr,
    has_bias: tl.constexpr,
    transpose_output: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PLANE: tl.constexpr = 0,  # EXL3_PF_GLUE: > 0 = planar (dim // PLANE, bsz, seq, PLANE) output
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_s = tl.program_id(2)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)

    slot = tl.load(slots + pid_b)
    mask_d = offs_d < dim
    mask_s = offs_s < seq_len

    acc = tl.zeros((BLOCK_D, BLOCK_S), dtype = tl.float32)

    for k in range(BLOCK_K):
        if k < conv_kernel_size:
            src_t = offs_s + k + 1
            from_x = src_t >= conv_kernel_size
            x_t = src_t - conv_kernel_size

            state_vals = tl.load(
                conv_state + (slot * dim + offs_d[:, None]) * state_size + src_t[None, :],
                mask = mask_d[:, None] & mask_s[None, :] & (src_t[None, :] < conv_kernel_size),
                other = 0.0,
            )
            x_vals = tl.load(
                x + pid_b * sxb + offs_d[:, None] * sxd + x_t[None, :] * sxs,
                mask = mask_d[:, None] & mask_s[None, :] & from_x[None, :] & (x_t[None, :] >= 0),
                other = 0.0,
            )
            x_vals = x_vals.to(conv_state.dtype.element_ty)
            vals = tl.where(from_x[None, :], x_vals, state_vals)
            w = tl.load(weight + offs_d * conv_kernel_size + k, mask = mask_d, other = 0.0)
            acc += vals * w[:, None]

    if has_bias:
        b = tl.load(bias + offs_d, mask = mask_d, other = 0.0)
        acc += b[:, None]

    acc = acc * tl.sigmoid(acc)
    if PLANE > 0:
        # q | k | v planes, each (bsz, seq, PLANE) contiguous: the split needs no copies downstream.
        # Same values as the transposed store, only the address differs (bit-exact)
        plane = offs_d // PLANE
        tl.store(
            out + plane[:, None] * (tl.num_programs(0) * seq_len * PLANE)
            + (pid_b * seq_len + offs_s[None, :]) * PLANE + (offs_d - plane * PLANE)[:, None],
            acc,
            mask = mask_d[:, None] & mask_s[None, :],
        )
    elif transpose_output:
        tl.store(
            out + (pid_b * seq_len + offs_s[None, :]) * dim + offs_d[:, None],
            acc,
            mask = mask_d[:, None] & mask_s[None, :],
        )
    else:
        tl.store(
            out + (pid_b * dim + offs_d[:, None]) * seq_len + offs_s[None, :],
            acc,
            mask = mask_d[:, None] & mask_s[None, :],
        )


@triton.jit
def _causal_conv1d_update_slotted_state_kernel(
    x,
    conv_state,
    slots,
    dim: tl.constexpr,
    seq_len,             # runtime: chunk length varies per job; only masks and address math
    sxb, sxd, sxs,       # x strides (batch, channel, time): x may be a transposed fp16 view
    state_size: tl.constexpr,
    conv_kernel_size: tl.constexpr,
    history: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_STATE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_state = tl.arange(0, BLOCK_STATE)

    slot = tl.load(slots + pid_b)
    mask_d = offs_d < dim
    mask_state = offs_state < state_size

    history_write_size = tl.where(state_size < conv_kernel_size + seq_len, state_size, conv_kernel_size + seq_len)
    state_write_size = tl.where(history, history_write_size, conv_kernel_size)
    dst_start = tl.where(history, state_size - state_write_size, 0)
    history_src_start = tl.where(conv_kernel_size + seq_len > state_size, conv_kernel_size + seq_len - state_size, 0)
    no_history_start = seq_len
    src_t = tl.where(history, history_src_start + offs_state - dst_start, no_history_start + offs_state)
    valid_state = (offs_state >= dst_start) & (offs_state < dst_start + state_write_size)
    from_x = src_t >= conv_kernel_size
    x_t = src_t - conv_kernel_size
    state_vals = tl.load(
        conv_state + (slot * dim + offs_d[:, None]) * state_size + src_t[None, :],
        mask = mask_d[:, None] & valid_state[None, :] & (src_t[None, :] >= 0) & (src_t[None, :] < conv_kernel_size),
        other = 0.0,
    )
    x_vals = tl.load(
        x + pid_b * sxb + offs_d[:, None] * sxd + x_t[None, :] * sxs,
        mask = mask_d[:, None] & valid_state[None, :] & from_x[None, :] & (x_t[None, :] >= 0),
        other = 0.0,
    )
    x_vals = x_vals.to(conv_state.dtype.element_ty)
    new_state = tl.where(from_x[None, :], x_vals, state_vals)
    tl.store(
        conv_state + (slot * dim + offs_d[:, None]) * state_size + offs_state[None, :],
        new_state,
        mask = mask_d[:, None] & mask_state[None, :] & valid_state[None, :],
    )


CONV_WIDE_TILE = (1024, 4, 8)  # (BLOCK_D, BLOCK_S, num_warps), gfx1151 sweep


def causal_conv1d_update_slotted_triton(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    slots: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    transpose_output: bool = False,
    history: bool = False,
    plane: int = 0,
) -> torch.Tensor:
    if not x.is_cuda:
        raise RuntimeError("causal_conv1d_update_slotted_triton requires CUDA tensors")
    if not conv_state.is_contiguous() or not weight.is_contiguous():
        raise RuntimeError("causal_conv1d_update_slotted_triton requires contiguous conv_state and weight")
    if conv_state.device != x.device:
        raise RuntimeError(f"conv_state is on {conv_state.device}, expected {x.device}")
    if slots.device != x.device:
        raise RuntimeError(f"slots is on {slots.device}, expected {x.device}")
    if weight.device != x.device:
        raise RuntimeError(f"weight is on {weight.device}, expected {x.device}")
    if bias is not None and bias.device != x.device:
        raise RuntimeError(f"bias is on {bias.device}, expected {x.device}")

    bsz, dim, seq_len = x.shape
    sxb, sxd, sxs = x.stride()
    state_size = conv_state.shape[-1]
    conv_kernel_size = weight.shape[-1]
    if slots.shape != (bsz,):
        raise ValueError("slots must be [bsz]")
    if conv_state.dim() != 3 or conv_state.size(1) != dim:
        raise ValueError("conv_state must be [num_slots, dim, state_size]")
    if weight.shape[0] != dim:
        raise ValueError("weight must be [dim, conv_kernel_size]")
    if bias is not None and bias.shape != (dim,):
        raise ValueError("bias must be [dim]")
    if conv_kernel_size > 16:
        raise ValueError("causal_conv1d_update_slotted_triton supports conv_kernel_size <= 16")
    if state_size < conv_kernel_size:
        raise ValueError("conv_state must have at least conv_kernel_size entries")
    out_shape = (bsz, seq_len, dim) if transpose_output else (bsz, dim, seq_len)
    # planar output only on the long-sequence output kernel, with whole planes per channel tile
    if plane and not (transpose_output and seq_len > 256 and dim % plane == 0):
        plane = 0
    if plane:
        out_shape = (dim // plane, bsz, seq_len, plane)
    # x is cast to the state dtype in-kernel (fp16 views skip a separate transpose-cast pass)
    out = torch.empty(out_shape, dtype = conv_state.dtype, device = x.device)
    block_d = 32
    block_k = triton.next_power_of_2(conv_kernel_size)
    block_state = triton.next_power_of_2(state_size)

    with torch.cuda.device(x.device):
        if seq_len <= 256:
            block_s = triton.next_power_of_2(seq_len)
            grid = (bsz, triton.cdiv(dim, block_d))
            _causal_conv1d_update_slotted_kernel[grid](
                x,
                conv_state,
                slots,
                weight,
                bias if bias is not None else weight,
                out,
                dim,
                seq_len,
                sxb, sxd, sxs,
                state_size,
                conv_kernel_size,
                history,
                bias is not None,
                transpose_output,
                BLOCK_D = block_d,
                BLOCK_S = block_s,
                BLOCK_K = block_k,
                BLOCK_STATE = block_state,
                num_warps = 4,
            )
        else:
            # EXL3_KDA_CONV_WIDE: the prefill x is the fp16 (bsz, dim, seq) view of a (bsz, seq, dim) projection
            # (channel stride 1) and the output is transposed, so a wide channel tile coalesces both the
            # loads and the stores. Same per-element tap order: bit-exact (scratch/dg/s7_conv_mb*.py)
            block_s, nw_out = 256, 4
            if sxd == 1 and transpose_output and os.environ.get("EXL3_KDA_CONV_WIDE", "1") == "1":
                block_d, block_s, nw_out = CONV_WIDE_TILE
            if plane and plane % block_d != 0:
                raise ValueError(f"planar conv output needs plane % BLOCK_D == 0 ({plane}, {block_d})")
            output_grid = (bsz, triton.cdiv(dim, block_d), triton.cdiv(seq_len, block_s))
            _causal_conv1d_update_slotted_output_kernel[output_grid](
                x,
                conv_state,
                slots,
                weight,
                bias if bias is not None else weight,
                out,
                dim,
                seq_len,
                sxb, sxd, sxs,
                state_size,
                conv_kernel_size,
                bias is not None,
                transpose_output,
                BLOCK_D = block_d,
                BLOCK_S = block_s,
                BLOCK_K = block_k,
                PLANE = plane,
                num_warps = nw_out,
            )
            block_d = 32
            state_grid = (bsz, triton.cdiv(dim, block_d))
            _causal_conv1d_update_slotted_state_kernel[state_grid](
                x,
                conv_state,
                slots,
                dim,
                seq_len,
                sxb, sxd, sxs,
                state_size,
                conv_kernel_size,
                history,
                BLOCK_D = block_d,
                BLOCK_STATE = block_state,
                num_warps = 4,
            )
    return out


def causal_conv1d_update_function_torch(
    x,
    conv_state,
    weight,
    bias = None,
    history: bool = False,
):
    bsz, dim, seq_len = x.shape
    state_size = conv_state.shape[-1]
    conv_kernel_size = weight.shape[-1]

    y = torch.cat([conv_state[:, :, :conv_kernel_size], x], dim = -1).to(weight.dtype)
    if history:
        write_size = min(state_size, y.shape[-1])
        conv_state[:, :, -write_size:].copy_(y[:, :, -write_size:])
    else:
        conv_state[:, :, :conv_kernel_size].copy_(y[:, :, -conv_kernel_size:])
    y = F.conv1d(y, weight.unsqueeze(1), bias, padding = 0, groups = dim)
    y = F.silu(y[:, :, -seq_len:])
    y = y.to(x.dtype)
    return y


def causal_conv1d_update(
    mixed_qkv: torch.Tensor,
    conv_state: torch.Tensor,
    recurrent_slots: torch.Tensor,
    conv1d_weight: torch.Tensor,
    conv1d_bias: torch.Tensor,
    history: bool = False,
    params: dict = None,
    plane: int = 0,
):
    bsz, dim, seqlen = mixed_qkv.shape

    if params is None:
        params = {}

    if conv_state is None:
        assert not history
        conv_state = torch.zeros((bsz, dim, conv1d_weight.shape[-1]), dtype = torch.bfloat16, device = mixed_qkv.device)
        dummy_slots = True
    else:
        dummy_slots = False

    # A non-bf16 or strided mixed_qkv (the fp16 (bsz, dim, seq) transposed view of the projection) is
    # read and cast in-kernel by the Triton path; the short-sequence CUDA path needs contiguous bf16
    if mixed_qkv.is_cuda and seqlen <= MAX_CUDA_SEQLEN and \
            (mixed_qkv.dtype != torch.bfloat16 or not mixed_qkv.is_contiguous()):
        if host_cuts():
            # One kernel instead of two. The decode input is the fp16 (bsz, f, seq) transposed
            # view of the qkv projection, so .to(bf16) alone keeps the strided (non-contiguous)
            # output and the following .contiguous() is a second full pass over the same bytes.
            # memory_format=contiguous_format makes the single copy write contiguous bf16
            # directly: same source elements, same RNE rounding, one kernel instead of two
            # (2 launches and 2 passes per KDA layer per round on the served GLM-5.3 path).
            mixed_qkv = mixed_qkv.to(dtype = torch.bfloat16, memory_format = torch.contiguous_format)
        else:
            mixed_qkv = mixed_qkv.to(torch.bfloat16).contiguous()

    if (
        mixed_qkv.is_cuda and
        seqlen <= MAX_CUDA_SEQLEN and
        conv1d_weight.shape[-1] <= MAX_CUDA_K and
        mixed_qkv.dtype == torch.bfloat16 and
        conv_state.dtype == torch.bfloat16 and
        conv1d_weight.dtype == torch.bfloat16 and
        (conv1d_bias is None or conv1d_bias.dtype == torch.bfloat16)
    ):
        out = torch.empty((bsz, seqlen, dim), dtype = torch.bfloat16, device = mixed_qkv.device)
        ext.cuda_causal_conv1d_update(
            mixed_qkv,
            conv_state,
            None if dummy_slots else recurrent_slots,
            conv1d_weight,
            conv1d_bias,
            out,
            True,
            history,
        )
        return out

    if dummy_slots:
        recurrent_slots = buffered_arange(bsz, mixed_qkv.device)
    mixed_qkv = causal_conv1d_update_slotted_triton(
        mixed_qkv,
        conv_state,
        recurrent_slots,
        conv1d_weight,
        conv1d_bias,
        transpose_output = True,
        history = history,
        plane = plane,
    )

    return mixed_qkv
