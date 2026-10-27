"""
Per-expert bitrate inside one MoE layer (hot-3 / cold-2 routed experts, DeepSeek-V4.1 384 x top-6).

The grouped HIP kernels are templated on ONE trellis bitrate K per launch, and MultiLinear asserts one K
per table. A layer whose experts differ in K therefore keeps one MultiLinear triple per distinct K and a
per-expert (group, slot) map:

    group g      = the experts sharing (K_gate_up, K_down), ascending by expert id inside the group
    slot[e]      = position of expert e inside its group's pointer tables (the group's "stack")
    lut[g, e]    = slot[e] if kmap[e] == g else -1           (int64, device)

Storage is the experts' own tensors (nothing is copied or repacked); what is contiguous per K is the pointer
tables (9 int64 tables per group, built once at load). Everything per call is device-side tensor ops, none
per token and none per expert.

Decode (rows <= 16, exl3_moe_gfx12_k3): the kernel already zero-fills picks outside [0, experts) (all
launch/had/reduce kernels test `expert >= 0 && expert < experts`), so group g runs on `lut[g][selected]`
with every foreign pick turned into -1. One launch set per K group; no host sync. The group outputs are
fp32 partial sums (each in the kernel's expert-sorted order, foreign picks contribute exact +0.0) and are
added in ascending (K_gu, K_down) order. A layer with one group is the legacy call, argument for argument.

Prefill (exl3_moe_prefill_wmma): that kernel has no masked picks (its reduce reads svh_table[e] for every
pick), so a K group is compacted instead: the group's picks become top_k = 1 pseudo rows (token rows
gathered once), one launch per group, and the per-pick results are summed per token in routing-slot order
(the order of the kernel's own reduce) into one fp32 buffer. Cost: one host read per call (the group sizes),
one [rows * top_k, hidden] fp32 buffer, and the group launches run on n_g picks instead of on all of them,
so the GEMM work is the same as a single-K layer's.
"""
from __future__ import annotations
import os
import types
try:
    from .dsv41_fn import ds41_pf as _pf
except Exception:     # no HIP stack (CPU checks)
    _pf = None
import torch
from . import Linear
from .multilinear import MultiLinear

DECODE_K = (1, 2, 2.5, 3, 4, 5, 6)
WMMA_K = (1, 2, 2.5, 3, 4, 5, 6)

# quantization_config field written by the converter (see doc/ds41-mixedk.md)
EXPERT_BITS_KEY = "expert_bits"


class MixedKUnsupported(ValueError):
    pass


class PrunedLinear:
    """Stand-in for a projection of a pruned (dead) expert, K = 0: no tensors, never loaded, never launched.
    The router bias already masks the expert; if a pick still lands on it, every path contributes exactly zero
    (grouped decode: lut = -1 zero-fills the pick; prefill: the pick is dropped; dense: the expert is skipped)."""
    quant_type = "exl3"
    pruned = True

    def __init__(self, key: str = "", in_features: int = 0, out_features: int = 0):
        self.key = key
        self.in_features = in_features
        self.out_features = self.out_features_unpadded = out_features
        self.trim_padded_out = False
        self.inner = types.SimpleNamespace(K = 0.0, bias = None, dec_ok_moe = True)

    def optimizer_targets(self): return []
    def storage_size(self): return 0
    def recons_size(self): return 0
    def unload(self): pass


def is_pruned(lin) -> bool:
    return getattr(lin, "pruned", False)


def pruned_experts(qcfg: dict | None, layer_key: str, n_experts: int) -> set:
    """Experts whose pack-config bits are 0. layer_key may carry a prefix ("model.layers.3.ffn" matches "layers.3.ffn")."""
    table = (qcfg or {}).get(EXPERT_BITS_KEY)
    if not table:
        return set()
    row = parse_expert_bits(qcfg, layer_key, n_experts)
    return set() if row is None else {e for e, b in enumerate(row) if b == 0}


def expert_ks(gates: list, ups: list, downs: list) -> list[tuple[float, float, float]]:
    """(K_gate, K_up, K_down) of every expert; all projections must be EXL3."""
    if not all(l.quant_type == "exl3" for l in gates + ups + downs):
        raise MixedKUnsupported("per-expert K needs EXL3 experts only")
    out = []
    for g, u, d in zip(gates, ups, downs):
        if is_pruned(g) or is_pruned(u) or is_pruned(d):
            if not (is_pruned(g) and is_pruned(u) and is_pruned(d)):
                raise MixedKUnsupported("an expert is pruned in some projections only")
            out.append((0.0, 0.0, 0.0))
        else:
            out.append((float(g.inner.K), float(u.inner.K), float(d.inner.K)))
    return out


def layer_is_mixed(ks: list) -> bool:
    return len(set(ks)) > 1 or any(k[0] == 0 for k in ks)   # any K0 expert needs the group path


class KGroup:
    """Experts of one layer that share (K_gate_up, K_down)."""

    def __init__(self, device, index: int, experts: list[int], gates: list, ups: list, downs: list):
        self.index = index
        self.experts = experts
        self.gate = MultiLinear(device, [gates[e] for e in experts], allow_bias = True)
        self.up = MultiLinear(device, [ups[e] for e in experts], allow_bias = True)
        self.down = MultiLinear(device, [downs[e] for e in experts], allow_bias = True)
        self.k_gu = self.gate.K
        self.k_down = self.down.K
        if self.gate.K != self.up.K:
            raise MixedKUnsupported(f"experts {experts[:4]}...: gate K {self.gate.K} != up K {self.up.K}")
        self.mul1 = self.up.mul1
        self.mcg = self.up.mcg
        self.n = len(experts)
        self.identity = False      # set by the store: group covers 0..E-1 in order

    @property
    def decode_ok(self) -> bool:
        return (self.k_gu == self.k_down and self.k_gu in DECODE_K and
                all(m.mul1 and not m.mcg for m in (self.gate, self.up, self.down)))

    @property
    def wmma_ok(self) -> bool:
        cbs = {(m.mul1, m.mcg) for m in (self.gate, self.up, self.down)}
        return (self.k_gu in WMMA_K and self.k_down in WMMA_K and len(cbs) == 1 and
                next(iter(cbs))[0] != next(iter(cbs))[1])

    def tables(self):
        return (self.gate.ptrs_trellis, self.gate.ptrs_suh, self.gate.ptrs_svh,
                self.up.ptrs_trellis, self.up.ptrs_suh, self.up.ptrs_svh,
                self.down.ptrs_trellis, self.down.ptrs_suh, self.down.ptrs_svh)

    def unload(self):
        for m in (self.gate, self.up, self.down):
            m.unload()


class MixedKStore:
    def __init__(self, device, gates: list[Linear], ups: list[Linear], downs: list[Linear]):
        assert len(gates) == len(ups) == len(downs) and len(gates) > 0
        self.device = device
        self.num_experts = E = len(gates)
        ks = expert_ks(gates, ups, downs)
        for e, (kg, ku, kd) in enumerate(ks):
            if kg != ku:
                raise MixedKUnsupported(f"expert {e}: gate K {kg} != up K {ku} (they share one grouped launch)")
        self.expert_k = [(kg, kd) for kg, _, kd in ks]
        self.pruned = [e for e, k in enumerate(self.expert_k) if k == (0.0, 0.0)]
        keys = sorted({k for k in self.expert_k if k != (0.0, 0.0)})
        if not keys:
            raise MixedKUnsupported("every expert of the layer is pruned")
        self.keys = keys
        by_key = {k: [] for k in keys}
        for e, k in enumerate(self.expert_k):
            if k != (0.0, 0.0):
                by_key[k].append(e)
        self.groups = [KGroup(device, i, by_key[k], gates, ups, downs) for i, k in enumerate(keys)]
        kmap = torch.full((E,), -1, dtype = torch.long)      # -1 = pruned (K0)
        slot = torch.full((E,), -1, dtype = torch.long)
        lut = torch.full((len(keys), E), -1, dtype = torch.long)
        for g in self.groups:
            idx = torch.tensor(g.experts, dtype = torch.long)
            kmap[idx] = g.index
            slot[idx] = torch.arange(g.n)
            lut[g.index, idx] = torch.arange(g.n)
            g.identity = (g.n == E)
        self.kmap = kmap.to(device)
        self.slot = slot.to(device)
        self.lut = lut.to(device)
        self.kmap_host = kmap
        self.slot_host = slot

    # ---------------------------------------------------------------- introspection

    @property
    def num_groups(self) -> int:
        return len(self.groups)

    @property
    def decode_ok(self) -> bool:
        return all(g.decode_ok for g in self.groups)

    @property
    def wmma_ok(self) -> bool:
        return all(g.wmma_ok for g in self.groups)

    def k_of(self, e: int):
        return self.expert_k[e]

    def summary(self) -> dict:
        return {"groups": [{"k_gu": g.k_gu, "k_down": g.k_down, "experts": g.n} for g in self.groups]}

    def unload(self):
        for g in self.groups:
            g.unload()
        self.groups = []

    # ---------------------------------------------------------------- decode

    def group_selected(self, g: KGroup, selected: torch.Tensor) -> torch.Tensor:
        """Group-local slot of every pick, -1 for picks of other groups. One gather, device only."""
        if g.identity:
            return selected
        return self.lut[g.index].index_select(0, selected.reshape(-1)).view(selected.shape)

    def decode(self, ext, y, selected, weights, buffers, outputs, act_limit: float, dedup: bool):
        """Grouped decode for rows <= 16. `outputs[i]` is a contiguous fp32 [>= rows, hidden] buffer for group i
        (outputs[0] is the layer's usual output buffer); returns the summed fp32 [rows, hidden] view."""
        rows = y.shape[0]
        assignments = rows * selected.shape[1]
        acc = None
        for g in self.groups:
            out = outputs[g.index][:rows]
            ext.exl3_moe_gfx12_k3(
                y, out, self.group_selected(g, selected), weights,
                *g.tables(),
                buffers.gu_had[:2 * assignments], buffers.gu_out[:2 * assignments],
                buffers.down_had[:assignments], buffers.down_out[:assignments],
                float(g.k_gu), dedup, float(act_limit),
            )
            if acc is None:
                acc = out
            else:
                acc.add_(out)
        return acc

    # ---------------------------------------------------------------- prefill

    def prefill(self, ext, y, selected, rw, workspace, scatter_count, act_limit: float, hidden: int, interm: int,
                sh_add = None, chunk: int = 0):
        """Grouped WMMA prefill. y fp16 [T, H]; selected int64 [T, top_k]; rw fp32 [T, top_k]; `workspace(assignments,
        experts)` returns the dict of block_sparse_mlp._hip_wmma_workspace; `scatter_count(flat, bins)` is the
        sync-free histogram. sh_add: fp16 [T, H] shared-expert output folded into the sum (or None).
        Returns fp32 [T, H]."""
        T, top_k = selected.shape
        step = chunk if chunk and chunk > 0 else max(T, 1)
        single = len(self.groups) == 1 and self.groups[0].identity
        if (chunk and T > chunk and (single or self.unified_ok(ext)) and sh_add is None
                and os.environ.get("EXL3_MIXEDK_ALIGNED", "1") != "0"):
            # EXL3_MIXEDK_LAUNCH_ROWS (default 8192): rows per aligned launch, rounded down to a multiple of chunk;
            # bounds the workspace (2 * rows * top_k * (H + I) fp16) while keeping the sub-chunk segmentation
            L = max(chunk, int(os.environ.get("EXL3_MIXEDK_LAUNCH_ROWS", "8192")) // chunk * chunk)
            if T <= L:
                return self._prefill_unified_aligned(ext, y, selected, rw, workspace, scatter_count, act_limit, hidden,
                                                     interm, chunk)
            out = torch.empty((T, hidden), dtype = torch.float, device = y.device)
            for t0 in range(0, T, L):
                t1 = min(T, t0 + L)
                out[t0:t1] = self._prefill_unified_aligned(ext, y[t0:t1], selected[t0:t1].contiguous(),
                                                           rw[t0:t1].contiguous(), workspace, scatter_count,
                                                           act_limit, hidden, interm, chunk)
            return out
        out = torch.empty((T, hidden), dtype = torch.float, device = y.device)
        for t0 in range(0, T, step):
            t1 = min(T, t0 + step)
            self._prefill_chunk(ext, y[t0:t1], selected[t0:t1].contiguous(), rw[t0:t1].contiguous(), out[t0:t1],
                                workspace, scatter_count, act_limit, hidden, interm,
                                None if sh_add is None else sh_add[t0:t1])
        return out

    def _prefill_chunk(self, ext, y, sel, rw, out, workspace, scatter_count, act_limit, hidden, interm, sh_add):
        T, top_k = sel.shape
        flat = sel.view(-1)
        dev = y.device

        # single identity group: the legacy call, argument for argument
        if len(self.groups) == 1 and self.groups[0].identity:
            g = self.groups[0]
            order = flat.argsort(stable = True)
            expert_count = scatter_count(flat, g.n + 1)
            ws = workspace(T * top_k, g.n)
            self._launch(ext, g, y, out, sel, rw, order, expert_count, ws, T * top_k, hidden, interm,
                         act_limit, sh_add)
            return

        if self.unified_ok(ext):
            self._prefill_unified(ext, y, sel, rw, out, workspace, scatter_count, act_limit, hidden, interm, sh_add)
            return

        G = len(self.groups)
        gid = self.kmap.index_select(0, flat)
        if self.pruned:
            gid = torch.where(gid < 0, torch.full_like(gid, G), gid)   # bucket G = picks on pruned experts
        # the one host read of the mixed path: how many picks each group got
        sizes = torch.bincount(gid, minlength = G + 1).tolist()
        perm = gid.argsort(stable = True)                 # picks grouped by K, flat (token, slot) order inside
        n_valid = sum(sizes[:G])
        res_all = torch.empty((n_valid, hidden), dtype = torch.float, device = dev)
        off = 0
        for g in self.groups:
            n = sizes[g.index]
            if n == 0:
                continue
            p = perm[off:off + n]
            tok = p // top_k
            a_g = y.index_select(0, tok)
            sel_g = self.lut[g.index].index_select(0, flat.index_select(0, p)).view(n, 1)
            w_g = rw.view(-1).index_select(0, p).view(n, 1).contiguous()
            order = sel_g.view(-1).argsort(stable = True)
            expert_count = scatter_count(sel_g.view(-1), g.n + 1)
            ws = workspace(n, g.n)
            self._launch(ext, g, a_g, res_all[off:off + n], sel_g, w_g, order, expert_count, ws, n, hidden, interm, act_limit, None)
            off += n
        # slot-order reduction of each token's picks (the kernel's own reduce order); pruned picks (bucket G) add nothing
        inv = torch.full((T * top_k,), -1, dtype = torch.int32, device = dev)
        inv[perm[:n_valid]] = torch.arange(n_valid, dtype = torch.int32, device = dev)
        if _pf is not None and res_all.is_cuda and hidden % 4 == 0:
            _pf.gather_sum(res_all, inv, out, T, top_k, hidden)
        else:
            contrib = torch.zeros((T * top_k, hidden), dtype = torch.float, device = dev)
            ok = inv >= 0
            contrib[ok.nonzero().squeeze(1)] = res_all[inv[ok].long()]
            c = contrib.view(T, top_k, hidden)
            out.copy_(c[:, 0])
            for k in range(1, top_k):
                out.add_(c[:, k])
        if sh_add is not None:
            out.add_(sh_add)

    # ---- one pipeline for a mixed-K layer without pruned experts (EXL3_MIXEDK_UNIFIED=0 turns it off)

    def unified_ok(self, ext) -> bool:
        return (not self.pruned and len(self.groups) > 1 and hasattr(ext, "exl3_moe_prefill_wmma_mixed") and
                os.environ.get("EXL3_MIXEDK_UNIFIED", "1") != "0" and
                len({(g.mcg, g.mul1) for g in self.groups}) >= 1 and all(g.wmma_ok for g in self.groups))

    def _unified_tables(self):
        u = None if getattr(self, "volatile_tables", False) else getattr(self, "_utab", None)
        if u is None:
            base, b = [], 0
            for g in self.groups:
                base.append(b); b += g.n
            base_t = torch.tensor(base, dtype = torch.long, device = self.device)
            gidx = base_t.index_select(0, self.kmap) + self.slot
            tabs = [torch.cat([g.tables()[j] for g in self.groups]).index_select(0, gidx).contiguous() for j in range(9)]
            u = (tabs, self.kmap.to(torch.int32).contiguous())
            if not getattr(self, "volatile_tables", False):
                self._utab = u      # tier-bound tables change with every expert load: rebuilt per call
        return u

    def _prefill_unified(self, ext, y, sel, rw, out, workspace, scatter_count, act_limit, hidden, interm, sh_add):
        T, top_k = sel.shape
        E, G = self.num_experts, len(self.groups)
        flat = sel.view(-1)
        A = T * top_k
        tabs, egroup = self._unified_tables()
        order = flat.argsort(stable = True)
        expert_count = scatter_count(flat, E + 1)
        ws = workspace(A, E)
        cache = getattr(self, "_utiles", None)
        need = A // 64 + E + 2
        if cache is None or cache[0].shape[1] < need:
            cache = self._utiles = (torch.empty((G, need), dtype = torch.int32, device = y.device),
                                    torch.empty((G,), dtype = torch.int32, device = y.device))
        ext.exl3_moe_prefill_wmma_mixed(
            y, out, sel, rw, order, expert_count, *tabs, egroup,
            [float(g.k_gu) for g in self.groups], [float(g.k_down) for g in self.groups],
            [1 if g.mcg else 0 for g in self.groups], [g.n for g in self.groups],
            ws["gu_had"][:2 * A * hidden].view(2 * A, hidden),
            ws["gu_out"][:2 * A * interm].view(2 * A, interm),
            ws["gu_had"][:2 * A * hidden].view(torch.float).view(A, hidden),
            ws["offsets"], ws["inverse"][:A], cache[0], cache[1], float(act_limit), sh_add)

    def _prefill_unified_aligned(self, ext, y, sel, rw, workspace, scatter_count, act_limit, hidden, interm, chunk):
        """A chunk of more than `chunk` rows as ONE unified launch whose per-expert row lists are laid out exactly like
        the chunk-by-chunk path (block_sparse_mlp._aligned_expert_layout: every (expert, sub-chunk) segment starts on an
        even row, padded with dummy rows that read as zeros), so the output is bit-identical to the sub-chunk loop but
        each expert tile is decoded once for all sub-chunks. Returns fp32 [T, H]."""
        from .block_sparse_mlp import _aligned_expert_layout
        T, top_k = sel.shape
        E, G = self.num_experts, len(self.groups)
        order, flat, dummy = _aligned_expert_layout(sel.view(-1), T, top_k, E, chunk)
        sel_k = flat.view(T + dummy, top_k)
        rw_k = torch.cat((rw, torch.zeros((dummy, top_k), dtype = rw.dtype, device = rw.device)))
        A = (T + dummy) * top_k
        if G == 1 and self.groups[0].identity:       # one K group: the legacy launch with the aligned layout
            g = self.groups[0]
            y_k = torch.cat((y, torch.zeros((dummy, y.shape[1]), dtype = y.dtype, device = y.device)))
            out = torch.empty((T + dummy, hidden), dtype = torch.float, device = y.device)
            self._launch(ext, g, y_k, out, sel_k, rw_k, order, scatter_count(flat, g.n + 1), workspace(A, g.n), A,
                         hidden, interm, act_limit, None)
            return out[:T]
        tabs, egroup = self._unified_tables()
        expert_count = scatter_count(flat, E + 1)
        ws = workspace(A, E)
        cache = getattr(self, "_utiles", None)
        need = A // 64 + E + 2
        if cache is None or cache[0].shape[1] < need:
            cache = self._utiles = (torch.empty((G, need), dtype = torch.int32, device = y.device),
                                    torch.empty((G,), dtype = torch.int32, device = y.device))
        out = torch.empty((T + dummy, hidden), dtype = torch.float, device = y.device)
        ext.exl3_moe_prefill_wmma_mixed(
            y, out, sel_k, rw_k, order, expert_count, *tabs, egroup,
            [float(g.k_gu) for g in self.groups], [float(g.k_down) for g in self.groups],
            [1 if g.mcg else 0 for g in self.groups], [g.n for g in self.groups],
            ws["gu_had"][:2 * A * hidden].view(2 * A, hidden),
            ws["gu_out"][:2 * A * interm].view(2 * A, interm),
            ws["gu_had"][:2 * A * hidden].view(torch.float).view(A, hidden),
            ws["offsets"], ws["inverse"][:A], cache[0], cache[1], float(act_limit), None)
        return out[:T]

    def _launch(self, ext, g, a, out, sel, w, order, expert_count, ws, assignments, hidden, interm, act_limit, sh_add):
        ext.exl3_moe_prefill_wmma(
            a, out, sel, w, order, expert_count,
            *g.tables(),
            float(g.k_gu), float(g.k_down),
            ws["gu_had"][:2 * assignments * hidden].view(2 * assignments, hidden),
            ws["gu_out"][:2 * assignments * interm].view(2 * assignments, interm),
            ws["gu_had"][:2 * assignments * hidden].view(torch.float).view(assignments, hidden),
            ws["offsets"], ws["inverse"][:assignments], ws["tiles"], ws["tile_count"],
            float(act_limit), bool(g.mcg), sh_add,
        )


# -------------------------------------------------------------------- pack config (quantization_config)

def parse_expert_bits(qcfg: dict | None, layer_key: str, n_experts: int):
    """quantization_config["expert_bits"][layer_key] -> list[int|float] of n_experts, or None when absent.
    layer_key is the module path of the MoE block as written in the tensor names ("layers.3.ffn")."""
    if not qcfg:
        return None
    table = qcfg.get(EXPERT_BITS_KEY)
    if not table:
        return None
    if layer_key not in table:     # tensor names carry no model prefix ("layers.3.ffn"), module keys may ("model.layers.3.ffn")
        hits = [k for k in table if layer_key.endswith("." + k)]
        if not hits:
            return None
        layer_key = hits[0]
    bits = table[layer_key]
    if not isinstance(bits, list) or len(bits) != n_experts or any(not isinstance(b, (int, float)) for b in bits):
        raise ValueError(f"{EXPERT_BITS_KEY}[{layer_key!r}]: expected {n_experts} numbers")
    return bits


def check_expert_bits(expected: list, store_or_ks) -> None:
    """Raise when the tensors loaded for a layer disagree with the pack config's per-expert bits."""
    if isinstance(store_or_ks, MixedKStore):
        got = [k[0] for k in store_or_ks.expert_k]
    else:
        got = [k[0] for k in store_or_ks]
    bad = [e for e, (a, b) in enumerate(zip(expected, got)) if float(a) != float(b)]
    if bad:
        raise ValueError(f"{EXPERT_BITS_KEY} disagrees with the loaded tensors for experts {bad[:8]} "
                         f"(config {[expected[e] for e in bad[:4]]}, tensors {[got[e] for e in bad[:4]]})")


def expert_bits_from_plan(entries: list[dict]) -> dict[str, list[int]]:
    """Converter side: plan entries (tools/ds41/plan.py) -> {"layers.L.ffn": [bits per expert]}. All three
    projections of an expert must carry the same bits (the grouped kernels share one K per launch)."""
    import re
    rx = re.compile(r"^(layers\.\d+\.ffn)\.experts\.(\d+)\.(w[123])$")
    seen: dict[str, dict[int, dict[str, int]]] = {}
    for e in entries:
        if e.get("action") not in ("quant", "pruned") or e.get("kind") != "routed":   # pruned entries carry bits 0
            continue
        m = rx.match(e["key"])
        if not m:
            continue
        seen.setdefault(m.group(1), {}).setdefault(int(m.group(2)), {})[m.group(3)] = int(e["bits"])
    out = {}
    for layer, per in seen.items():
        n = max(per) + 1
        if sorted(per) != list(range(n)):
            raise ValueError(f"{layer}: experts are not 0..{n - 1}")
        row = []
        for e in range(n):
            b = per[e]
            if len(b) != 3 or len(set(b.values())) != 1:
                raise ValueError(f"{layer} expert {e}: projections {b} must all have the same bits")
            row.append(next(iter(b.values())))
        out[layer] = row
    return out
