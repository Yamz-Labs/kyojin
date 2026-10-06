"""CPU test of block_sparse_mlp._aligned_expert_layout (EXL3_MOE_ALIGN_CHUNK): layout properties + mutants.
Run: nice -n 19 taskset -c 0-7 python tools/qwen/test_moe_align.py [--mutants]"""
import inspect, os, sys, textwrap, unittest
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import torch
from exllamav3.modules import block_sparse_mlp as bsm

CASES = [(9000, 10, 512, 4096, "u"), (16384, 10, 512, 4096, "u"), (16384, 10, 512, 4096, "d0.2"), (4097, 10, 512, 4096, "u"),
         (4096 * 3 + 1, 10, 512, 4096, "d0.5"), (8192, 10, 64, 1024, "u"), (50, 3, 8, 16, "u"), (33, 10, 512, 16, "d0.1"),
         (20000, 10, 512, 4096, "one"), (8, 3, 10, 4, "allodd")]

def routing(T, k, E, kind, seed):
    g = torch.Generator().manual_seed(seed)
    if kind == "allodd":   # every (expert, chunk) segment has an odd length: the pad capacity is exactly used
        return torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 9, 9]] * (T // 4))
    if kind == "one":   # every token picks experts 0..k-1 except a few
        sel = torch.arange(k).repeat(T, 1)
        sel[::7] = torch.arange(E - k, E)
        return sel.long()
    prob = torch.ones(E) if kind == "u" else torch.distributions.Dirichlet(torch.full((E,), float(kind[1:]))).sample()
    x = torch.rand((T, E), generator=g).clamp_min(1e-9)
    return torch.topk(torch.log(prob) + (-torch.log(-torch.log(x))), k, dim=-1).indices.contiguous()

def check(fn, T, k, E, chunk, kind, seed=0):
    sel = routing(T, k, E, kind, seed); flat = sel.view(-1); a0 = T * k
    order, sel_flat, D = fn(flat, T, k, E, chunk)
    total = a0 + D * k
    assert order.shape == (total,) and sel_flat.shape == (total,), "shapes"
    assert torch.equal(order.sort().values, torch.arange(total)), "order is not a permutation of all slots"
    assert torch.equal(sel_flat[:a0], flat), "real slots keep their expert"
    es = sel_flat[order]
    assert bool((es[1:] >= es[:-1]).all()), "order not grouped by expert"
    nc = (T + chunk - 1) // chunk
    cid = torch.arange(a0) // (k * chunk)
    for e in range(E):
        lst = order[es == e].tolist()
        runs = []   # (start row in the expert list, chunk, slots, followed by a pad)
        pos = 0
        for s in lst:
            if s < a0:
                c = int(cid[s])
                if runs and runs[-1][1] == c and not runs[-1][3] and runs[-1][0] + len(runs[-1][2]) == pos:
                    runs[-1][2].append(s)
                else:
                    runs.append([pos, c, [s], False])
            elif runs and runs[-1][0] + len(runs[-1][2]) == pos and not runs[-1][3]:
                runs[-1][3] = True
            pos += 1
        got = {}
        for st, c, run, padded in runs:
            assert c not in got, f"expert {e} chunk {c} split in two runs"
            got[c] = run
            assert run == sorted(run), "slot order inside a segment"
            assert st % 2 == 0, f"segment of expert {e} chunk {c} starts at odd row {st}"
            if not (e == E - 1 and (st, c, run, padded) is not None and runs[-1][1] == c):   # the last expert's final run may be followed by the extra dummy slots
                assert padded == (len(run) % 2 == 1), f"pad rule expert {e} chunk {c}"
        want = {}
        for s in (flat == e).nonzero().flatten().tolist(): want.setdefault(int(cid[s]), []).append(s)
        assert got == want, f"segments of expert {e} differ"
    # all dummy slots are distinct slots of dummy tokens
    assert int((order >= a0).sum()) == D * k

class T(unittest.TestCase):
    def test_all(self):
        for i, (Tn, k, E, ch, kind) in enumerate(CASES):
            if Tn * k > 200000: continue
            with self.subTest(case=(Tn, k, E, ch, kind)): check(bsm._aligned_expert_layout, Tn, k, E, ch, kind, i)

MUTANTS = [
    ("no even padding", "odd = kc & 1", "odd = kc & 0"),
    ("pad always", "odd = kc & 1", "odd = (kc >= 0).long()"),
    ("chunk width", "torch.arange(a0, device = dev) // (top_k * chunk)", "torch.arange(a0, device = dev) // (top_k * chunk * 2)"),
    ("pad after not before shift", "pos = torch.arange(a0, device = dev) + padbefore[key[srt]]", "pos = torch.arange(a0, device = dev)"),
    ("pad position off by one", "pad_pos = torch.cumsum(kc, 0) + padbefore", "pad_pos = torch.cumsum(kc, 0) + padbefore + 1"),
    ("pad expert wrong", "torch.arange(nseg, device = dev) // nc\n", "torch.arange(nseg, device = dev) % num_experts\n"),
    ("pad slot id wrong", "pad_slot = a0 + padbefore", "pad_slot = a0 + padbefore * 0"),
    ("unstable sort order", "srt = key.argsort(stable = True)", "srt = key.argsort(stable = True, descending = True)"),
    ("dummy token count", "dummy_tokens = (nseg + top_k - 1) // top_k", "dummy_tokens = nseg // top_k"),
    ("extras expert", "torch.full((total + 1,), num_experts - 1,", "torch.full((total + 1,), 0,"),
]

def run_mutants():
    src = textwrap.dedent(inspect.getsource(bsm._aligned_expert_layout)); bad = 0
    for name, a, b in MUTANTS:
        assert a in src, f"mutant anchor missing: {name}"
        ns = {"torch": torch}; exec(src.replace(a, b, 1), ns); fn = ns["_aligned_expert_layout"]
        failed = False
        for i, (Tn, k, E, ch, kind) in enumerate(CASES):
            if Tn * k > 200000: continue
            try: check(fn, Tn, k, E, ch, kind, i)
            except Exception as ex:
                failed = True; break
        print(f"mutant {name:30s}: {'KILLED' if failed else 'SURVIVED'}"); bad += (not failed)
    print("survivors:", bad); return bad

if __name__ == "__main__":
    if "--mutants" in sys.argv: sys.exit(1 if run_mutants() else 0)
    unittest.main()
