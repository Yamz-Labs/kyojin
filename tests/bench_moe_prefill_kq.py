"""Prefill MoE (N = 512..8192 rows) on real unsloth UD experts: Triton grouped K-quant GEMM vs the per-expert dequant + matmul
path (MCPM_FQ_PF_GROUPED_KQ=0): time per layer and relative difference.

    FMOE_GGUF=<file.gguf> python tests/bench_moe_prefill_kq.py [layer]
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch

ENGINE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ENGINE)
from exllamav3.loader.gguf_collection import GGUFCollection  # noqa: E402
from exllamav3.modules.foreign_linear import ForeignLinear  # noqa: E402
from exllamav3.modules.block_sparse_mlp import BlockSparseMLP  # noqa: E402
from exllamav3.foreign_quant import k_quants  # noqa: E402

LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 0
FMT = {12: ("q4_k", 144, k_quants.dequant_q4_K_block), 13: ("q5_k", 176, k_quants.dequant_q5_K_block),
       14: ("q6_k", 210, k_quants.dequant_q6_K_block)}


def deq(packed: np.ndarray, rows: int, cols: int, bb: int, fn) -> np.ndarray:
    nb = cols // 256
    out = np.empty((rows, cols), np.float32)
    for r in range(rows):
        for b in range(nb):
            out[r, b * 256:(b + 1) * 256] = fn(packed[r, b * bb:(b + 1) * bb].tobytes())
    return out


def main() -> int:
    dev = torch.device("cuda")
    g = GGUFCollection(os.environ["FMOE_GGUF"])
    pre = f"blk.{LAYER}"
    H, I = g.infos[f"{pre}.ffn_gate_exps.weight"][1][0], g.infos[f"{pre}.ffn_gate_exps.weight"][1][1]
    E = g.infos[f"{pre}.ffn_gate_exps.weight"][1][2]
    tg, tu, td = (g.infos[f"{pre}.ffn_{n}_exps.weight"][0] for n in ("gate", "up", "down"))
    print(f"layer {LAYER}: H={H} I={I} E={E} gate/up type {tg}/{tu} down type {td}")
    fg, bbg, fng = FMT[tg]
    fd, bbd, fnd = FMT[td]
    assert FMT[tu][0] == fg

    def mk(name, fin, fout, fmt, out_dtype=None):
        m = ForeignLinear(None, name, fin, fout, fmt, out_dtype=out_dtype)
        m.packed = g.get_tensor(f"{pre}.ffn_{name.split('.')[0]}_exps.e{int(name.split('e')[-1])}.weight", device=dev)
        m.device = dev
        return m

    gates = [mk(f"gate.e{e}", H, I, fg) for e in range(E)]
    ups = [mk(f"up.e{e}", H, I, fg) for e in range(E)]
    downs = [mk(f"down.e{e}", I, H, fd, torch.float) for e in range(E)]
    print("packed shapes", tuple(gates[0].packed.shape), tuple(downs[0].packed.shape), "resident MiB",
          sum(m.packed.numel() for m in gates + ups + downs) >> 20)

    moe = BlockSparseMLP(None, "moe", hidden_size=H, intermediate_size=I, num_experts=E, num_experts_per_tok=8,
                         activation_fn="silu", gates=gates, ups=ups, downs=downs)
    for m in gates + ups + downs:
        moe.register_submodule(m)
    moe.device = dev

    torch.manual_seed(3)
    for N in (256, 1024, 4096, 8192):
        x = (torch.randn(N, H, device=dev) * 0.5).half()
        sel = torch.stack([torch.randperm(E, device=dev)[:8] for _ in range(N)])
        wts = torch.softmax(torch.randn(N, 8, device=dev), -1).half()
        res = {}
        for mode in ("0", "1"):
            os.environ["MCPM_FQ_PF_GROUPED_KQ"] = mode
            def run():
                return moe.forward(x, {"dec_routed": (sel.clone(), wts)})
            y = run().float().clone()
            torch.cuda.synchronize()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(5):
                run()
            b.record()
            torch.cuda.synchronize()
            res[mode] = (a.elapsed_time(b) / 5, y)
        if os.environ.get("PROF") and N == 4096:
            from torch.profiler import profile, ProfilerActivity
            os.environ["MCPM_FQ_PF_GROUPED_KQ"] = "1"
            with profile(activities=[ProfilerActivity.CUDA]) as p:
                run(); torch.cuda.synchronize()
            for e in sorted(p.key_averages(), key=lambda e: -e.device_time_total)[:6]:
                print(f"   {e.device_time_total / 1e3:8.1f} ms  n={e.count:4d}  {e.key[:70]}")
        rel = ((res["1"][1] - res["0"][1]).norm() / res["0"][1].norm()).item()
        print(f"N={N:5d}: per-expert {res['0'][0]:8.1f} ms  grouped {res['1'][0]:8.1f} ms  ({res['0'][0] / res['1'][0]:.1f}x)  rel_diff={rel:.1e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
