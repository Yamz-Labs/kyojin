#!/usr/bin/env python3
"""Small-scratch gfx1151 correctness checks for the EXL3 tile quantizer."""

import argparse
import os
import numpy as np
import torch

from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.quant.exl3_lib.quantize import tensor_core_perm


THRESHOLD = {2: 0.10, 3: 0.10, 4: 0.10, 5: 0.10, 6: 0.07}


def cpu_viterbi(tile, lut, K):
    """CPU reference for the kernel's two-pass tail-biting trellis search."""
    L = len(tile)
    edges = 65536 >> K
    kr = 16 - K
    branches = np.arange(1 << K, dtype=np.uint32)[:, None]
    out_edges = np.arange(edges, dtype=np.uint32)[None, :]
    states = (branches << kr) | out_edges
    in_edges = states >> K
    decoded = lut[states]
    samples = np.asarray(tile, dtype=np.float16)
    history = np.empty((L, edges), dtype=np.uint16)

    def forward(order, start_edge=None):
        costs = None
        for step, pos in enumerate(order):
            diff = (decoded - samples[pos]).astype(np.float16)
            squared = (diff.astype(np.float32) * diff.astype(np.float32)).astype(np.float16)
            metric = squared.astype(np.float32)
            if costs is not None:
                metric += costs[in_edges].astype(np.float32)
            if step == 0 and start_edge is not None:
                metric[in_edges != start_edge] = np.inf
            metric = metric.astype(np.float16)
            best = np.argmin(metric, axis=0)
            costs = metric[best, np.arange(edges)].copy()
            history[pos] = in_edges[best, np.arange(edges)]
        return costs

    # Match the kernel's rolled first pass, then its rank-stable minimum-cost edge.
    costs = forward(list(range(L // 2, L)) + list(range(L // 2)))
    if costs is None:
        raise AssertionError("reference pass did not produce trellis costs")
    edge_ids = np.arange(edges, dtype=np.uint32)
    v = edge_ids & 1023
    rev5 = lambda x: (((x & 1) << 4) | ((x & 2) << 2) | (x & 4) |
                      ((x & 8) >> 2) | ((x & 16) >> 4))
    ranks = (rev5(v >> 5) << 10) | (rev5(v & 31) << 5) | (edge_ids >> 10)
    cost_bits = costs.view(np.uint16)
    end_edge = int(np.lexsort((ranks, cost_bits))[0])

    edge = end_edge
    for i in range(L - 1, -1, -1):
        pos = (i + L // 2) % L
        edge = int(history[pos, edge])
        if pos == 0:
            break
    start_edge = edge

    forward(range(L), start_edge)
    result = np.empty(L, dtype=np.uint16)
    edge = start_edge
    for pos in range(L - 1, -1, -1):
        prev = int(history[pos, edge])
        result[pos] = ((prev << K) | edge) & 0xffff
        edge = prev
    return result


def quantize_batch(tiles, K, codebook):
    mcg, mul1 = codebook
    edges = 65536 >> K
    x = tiles.contiguous().to(device="cuda", dtype=torch.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.ndim != 2 or x.shape[1] not in (256, 160):
        raise ValueError("tiles must have shape (N, 256) or (N, 160)")
    out = torch.empty_like(x)
    indices = torch.empty_like(x, dtype=torch.int16)
    device_index = x.device.index if x.device.index is not None else torch.cuda.current_device()
    optimized, wave = getattr(ext, "quantize_tiles_scratch")(device_index, K, mcg, mul1, x.shape[1])
    if optimized:
        history_cols = edges // (8 if K == 1 else 1)
        if torch.version.hip and K <= 2:
            costs = torch.empty((wave, 2, edges), device="cuda", dtype=torch.float16)
        else:
            costs = torch.empty((1,), device="cuda", dtype=torch.float16)
        history = torch.empty((wave, x.shape[1], history_cols), device="cuda", dtype=torch.uint8)
    else:
        costs = torch.empty((x.shape[0], 2, edges), device="cuda", dtype=torch.float16)
        history = torch.empty((x.shape[0], x.shape[1], edges), device="cuda", dtype=torch.int16)
    getattr(ext, "quantize_tiles")(x, out, indices, costs, history, K, mcg, mul1)
    decoded = torch.empty_like(out)
    getattr(ext, "decode")(indices, decoded, mcg, mul1)
    torch.cuda.synchronize()
    if not torch.equal(out, decoded):
        raise AssertionError(f"K={K}, cb={codebook}: encoder output != decoder output")
    idx = indices.cpu().numpy().astype(np.uint16, copy=False)
    mask = (1 << (16 - K)) - 1
    if not np.array_equal(idx[:, 0] >> K, idx[:, -1] & mask):
        raise AssertionError(f"K={K}, cb={codebook}: invalid tail-biting boundary")
    return x, out, indices


def quantize(tile, K, codebook):
    x, out, indices = quantize_batch(tile.reshape(1, -1), K, codebook)
    return x, out, indices.cpu().numpy().astype(np.uint16, copy=False)[0]


def check_pack_reconstruct(tiles, K, codebook, permutation):
    _, decoded_tiles, indices = quantize_batch(tiles[:, permutation], K, codebook)
    packed = torch.empty((1, indices.shape[0], 256 * K // 16), device="cuda", dtype=torch.int16)
    encoded = indices.contiguous().reshape(1, indices.shape[0], 256)
    getattr(ext, "pack_trellis")(packed, encoded, K)
    unpacked = torch.empty_like(encoded)
    getattr(ext, "unpack_trellis")(unpacked, packed, K)
    if not torch.equal(unpacked, encoded):
        raise AssertionError(f"K={K}, cb={codebook}: pack/unpack changed trellis codes")

    reconstructed = torch.empty((16, indices.shape[0] * 16), device="cuda", dtype=torch.float16)
    getattr(ext, "reconstruct")(reconstructed, packed, K, *codebook)
    row_major_tiles = torch.empty_like(decoded_tiles, dtype=torch.float16)
    row_major_tiles[:, permutation] = decoded_tiles.half()
    expected = row_major_tiles.reshape(indices.shape[0], 16, 16).permute(1, 0, 2).reshape_as(reconstructed)
    torch.cuda.synchronize()
    if not torch.equal(reconstructed, expected):
        max_diff = (reconstructed.float() - expected.float()).abs().max().item()
        raise AssertionError(f"K={K}, cb={codebook}: reconstruct differs (max_abs={max_diff})")
    print(f"pack/unpack/reconstruct K={K} codebook={codebook}: exact")


def check_cpu_reference(tile, K, gpu_indices, lut):
    ref = cpu_viterbi(tile.detach().cpu().numpy().reshape(-1), lut, K)
    if not np.array_equal(ref, gpu_indices):
        mismatches = int(np.count_nonzero(ref != gpu_indices))
        raise AssertionError(f"CPU/GPU K={K} trellis indices differ at {mismatches}/{len(ref)} positions")


def check_optimized_parity(tiles, K, codebook, lut=None):
    previous = os.environ.get("EXL3_QT_OPTIMIZED")
    try:
        os.environ["EXL3_QT_OPTIMIZED"] = "0"
        _, out_base, idx_base = quantize_batch(tiles, K, codebook)
        os.environ["EXL3_QT_OPTIMIZED"] = "1"
        x_opt, out_opt, idx_opt = quantize_batch(tiles, K, codebook)
    finally:
        if previous is None:
            os.environ.pop("EXL3_QT_OPTIMIZED", None)
        else:
            os.environ["EXL3_QT_OPTIMIZED"] = previous
    if not torch.equal(idx_base, idx_opt):
        mismatches = int((idx_base != idx_opt).sum().item())
        raise AssertionError(f"optimized/unoptimized K={K}, cb={codebook}: {mismatches} indices differ")
    if not torch.equal(out_base, out_opt):
        raise AssertionError(f"optimized/unoptimized K={K}, cb={codebook}: decoded outputs differ")
    if lut is not None and K == 3 and codebook == (False, False):
        idx_cpu = idx_opt.cpu().numpy().astype(np.uint16, copy=False)[0]
        check_cpu_reference(x_opt[0], K, idx_cpu, lut)
    print(f"optimized/unoptimized K={K} codebook={codebook}: bit-identical indices")



def check_kernel_mode_parity(tiles, K, codebook, mode, lut=None):
    previous_mode = os.environ.get("EXL3_QT_KERNEL")
    previous_optimized = os.environ.get("EXL3_QT_OPTIMIZED")
    try:
        os.environ.pop("EXL3_QT_OPTIMIZED", None)
        os.environ["EXL3_QT_KERNEL"] = "base"
        _, out_base, idx_base = quantize_batch(tiles, K, codebook)
        os.environ["EXL3_QT_KERNEL"] = mode
        x_mode, out_mode, idx_mode = quantize_batch(tiles, K, codebook)
    finally:
        if previous_mode is None:
            os.environ.pop("EXL3_QT_KERNEL", None)
        else:
            os.environ["EXL3_QT_KERNEL"] = previous_mode
        if previous_optimized is None:
            os.environ.pop("EXL3_QT_OPTIMIZED", None)
        else:
            os.environ["EXL3_QT_OPTIMIZED"] = previous_optimized
    if not torch.equal(idx_base, idx_mode):
        mismatches = int((idx_base != idx_mode).sum().item())
        raise AssertionError(f"kernel mode {mode} K={K}, cb={codebook}: {mismatches} indices differ")
    if not torch.equal(out_base, out_mode):
        raise AssertionError(f"kernel mode {mode} K={K}, cb={codebook}: decoded outputs differ")
    if lut is not None and K == 3 and codebook == (False, False):
        idx_cpu = idx_mode.cpu().numpy().astype(np.uint16, copy=False)[0]
        check_cpu_reference(x_mode[0], K, idx_cpu, lut)
    print(f"kernel mode base/{mode} K={K} codebook={codebook}: bit-identical indices")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-file", help="Optional local safetensors shard for a real-weight check")
    parser.add_argument("--tensor-key", help="Tensor name to read from --weights-file")
    args = parser.parse_args()
    if bool(args.weights_file) != bool(args.tensor_key):
        parser.error("--weights-file and --tensor-key must be provided together")

    if not torch.cuda.is_available():
        raise RuntimeError("HIP GPU unavailable")
    props = torch.cuda.get_device_properties(0)
    if props.gcnArchName != "gfx1151":
        raise RuntimeError(f"expected gfx1151, got {props.gcnArchName}")
    print(f"device={props.name} arch={props.gcnArchName}")

    codes = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16).reshape(1, -1)
    lut_tensor = torch.empty((1, 65536), device="cuda", dtype=torch.float32)
    getattr(ext, "decode")(codes, lut_tensor, False, False)
    torch.cuda.synchronize()
    lut = lut_tensor.cpu().numpy().reshape(-1).astype(np.float16)

    for K in range(2, 7):
        scale = 1.24371088
        tile = torch.randn((256,), generator=torch.Generator().manual_seed(20260923)) * scale
        x, out, indices = quantize(tile, K, (False, False))
        mse = torch.mean((x / scale - out / scale) ** 2).item()
        if mse >= THRESHOLD[K]:
            raise AssertionError(f"Gaussian K={K}: MSE {mse:.6g} exceeds {THRESHOLD[K]}")
        if K == 3:
            check_cpu_reference(x, K, indices, lut)
        print(f"gaussian K={K}: mse={mse:.8g} cpu_reference={'exact' if K == 3 else 'n/a'}")

    for K, codebook in ((3, (True, False)), (4, (False, True))):
        scale = 1.24371088 if not codebook[1] else 1.0
        tile = torch.randn((256,), generator=torch.Generator().manual_seed(9000 + K)) * scale
        x, out, _ = quantize(tile, K, codebook)
        mse = torch.mean((x / scale - out / scale) ** 2).item()
        print(f"gaussian K={K} codebook={codebook}: mse={mse:.8g}")

    for K in range(1, 9):
        scale = 1.24371088
        tiles = torch.randn((2, 256), generator=torch.Generator().manual_seed(71000 + K)) * scale
        check_optimized_parity(tiles, K, (False, False), lut if K == 3 else None)
    for K, codebook, scale in ((3, (True, False), 1.24371088), (4, (False, True), 1.0)):
        tiles = torch.randn((2, 256), generator=torch.Generator().manual_seed(72000 + K)) * scale
        check_optimized_parity(tiles, K, codebook)
    rdna_cases = ((2, (False, False), 1.24371088), (3, (False, False), 1.24371088),
                  (4, (False, False), 1.24371088), (3, (True, False), 1.24371088),
                  (4, (False, True), 1.0))
    for K, codebook, scale in rdna_cases:
        tiles = torch.randn((2, 256), generator=torch.Generator().manual_seed(73000 + K)) * scale
        check_kernel_mode_parity(tiles, K, codebook, "rdna", lut if K == 3 and not any(codebook) else None)

    raw_tiles = (torch.randn((8, 256), generator=torch.Generator().manual_seed(30003)) * 1.24371088).to("cuda")
    check_pack_reconstruct(raw_tiles, 3, (False, False), tensor_core_perm("cuda"))

    if args.weights_file:
        from safetensors import safe_open
        with safe_open(args.weights_file, framework="pt", device="cpu") as sf:
            if args.tensor_key not in sf.keys():
                raise KeyError(f"{args.tensor_key!r} not found in {args.weights_file}")
            weights = sf.get_tensor(args.tensor_key).reshape(-1)[:256].float()
        if weights.numel() != 256 or not torch.isfinite(weights).all():
            raise ValueError("selected real-weight tensor must contain at least 256 finite values")
        rms = weights.square().mean().sqrt()
        if rms == 0:
            raise ValueError("selected real-weight tile has zero RMS")
        tile = weights * (1.24371088 / rms)
        for K in (3, 4):
            check_optimized_parity(tile.reshape(1, -1), K, (False, False), lut if K == 3 else None)
            check_kernel_mode_parity(tile.reshape(1, -1), K, (False, False), "rdna", lut if K == 3 else None)
            x, out, indices = quantize(tile, K, (False, False))
            mse = torch.mean((x / 1.24371088 - out / 1.24371088) ** 2).item()
            if K == 3:
                check_cpu_reference(x, K, indices, lut)
            print(f"real_weight {args.tensor_key} K={K}: normalized_mse={mse:.8g}")

    print("PASS")


if __name__ == "__main__":
    main()
