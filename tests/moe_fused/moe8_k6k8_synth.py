"""Check K6/K8 wide launches on synthetic weights. Requires a ROCm GPU.

SHAPE=small or real; GS=0 or 1; SEED sets the random seed.
This check also measures wide and split launch times.
"""
import os, sys, torch
TREE = os.environ.get('EXL3_TREE', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))); sys.path.insert(0, TREE)
import exllamav3
from exllamav3.ext import exllamav3_ext as ext
assert exllamav3.__file__.startswith(TREE)
RBSB = {'k6s8': (6, 8), 'k24': (2, 4), 'k3s4': (3, 4), 'k3s5': (3, 5), 'k4s4': (4, 4), 'k4s5': (4, 5)}[os.environ.get('SYNTH', 'k6s8')]
KIND = os.environ.get('SHAPE', 'small')
D, H, LR, NEXP, TOPK, INTER = (256, 4, 32, 64, 4, 128) if KIND == 'small' else (2560, 4, 320, 512, 10, 640)
RB, SB = RBSB
SHAPE = (D, H, LR, NEXP, TOPK, INTER, RB, SB)
assert ext.exl3_moe_fused_supported(*SHAPE), SHAPE
VAR = 0x161000
SEED = int(os.environ.get('SEED', '1234'))
GS = os.environ.get('GS', '1') == '1'
MR = LR + H; D4 = D // 4
dev = 'cuda'
g = torch.Generator(device='cpu'); g.manual_seed(SEED)
def ri(n): return torch.randint(-128, 128, (n,), dtype=torch.int8, generator=g).to(dev)
fn = ri(MR * H * D); upt = ri(H * D4 * LR * 4)
fn_scale = (torch.rand(MR * (H*D//128 if GS else 1), generator=g) * 0.02 + 0.001).to(dev); up_scale = (torch.rand(H * D * ((LR+63)//64 if GS else 1), generator=g) * 0.02 + 0.001).to(dev)
w_h = (torch.randn(H * D, generator=g) * 0.5 + 1).half().to(dev)
router = (torch.randn(NEXP * D, generator=g) * 0.1).half().to(dev); sgate = (torch.randn(D, generator=g) * 0.1).half().to(dev)
def mat(k, n, bits): return torch.randint(-32768, 32768, (k // 16, n // 16, 16 * bits), dtype=torch.int16, generator=g).to(dev)
def vec(n): return (torch.randn(n, generator=g) * 0.3 + 1).half().to(dev)
units = []
for u in range(NEXP + 1):
    b = SB if u == NEXP else RB
    units.append((mat(D, INTER, b), mat(D, INTER, b), mat(INTER, D, b), [vec(D), vec(INTER), vec(D), vec(INTER), vec(INTER), vec(D)]))
wt, keep = [], []
for gt, ut, dt, _ in units: wt += [gt.data_ptr(), ut.data_ptr(), dt.data_ptr()]
svt = [v.data_ptr() for *_, sv in units for v in sv]
wt = torch.tensor(wt, dtype=torch.int64, device=dev); svt = torch.tensor(svt, dtype=torch.int64, device=dev)
ro = [fn, upt, fn_scale, up_scale, w_h, router, sgate, wt, svt] + [m for u in units for m in u[:3]] + [v for u in units for v in u[3]]
ro_ref = [t.clone() for t in ro]
off = ext.exl3_moe_fused_ws_offsets(*SHAPE); TAIL = 1 << 20
ws = torch.full((off[-1] + TAIL,), 0xAB, dtype=torch.uint8, device=dev); ws[:256] = 0
print('shape', SHAPE, KIND, 'ws bytes', off[-1], flush=True)
PAD = 4096
def launch(x, variant=VAR):
    """x (R, H, D) fp32 -> output copy; checks canaries and ctl."""
    R = x.shape[0]
    buf = torch.full((R * H * D + 2 * PAD,), 12345.0, dtype=torch.float, device=dev)
    xv = buf[PAD: PAD + R * H * D].view(1, R, H, D); xv.copy_(x.view(1, R, H, D))
    torch.ops.mf9.half(xv, fn, fn_scale, upt, up_scale, w_h, router, sgate, wt, svt, ws, 1e-6, 1, 0, variant, *SHAPE)
    torch.cuda.synchronize()
    assert bool((buf[:PAD] == 12345.0).all()) and bool((buf[PAD + R * H * D:] == 12345.0).all()), 'x canary overwritten'
    assert bool((ws[off[-1]:] == 0xAB).all()), 'workspace tail canary overwritten'
    c = ws[:12].view(torch.int32).tolist(); assert c == [0, 0, 0], ('ctl after launch (bar, done, err)', c)
    return xv.view(R, H, D).clone()
def split(x):
    R = x.shape[0]
    return torch.cat([launch(x[lo:lo + 4]) for lo in range(0, R, 4)])
def ngroups_max(x):   # largest number of rows sharing one expert (from the single-row selections) is not exported; report via seldbg of the wide launch instead
    return None
torch.manual_seed(SEED)
base = torch.randn(8, H, D, device=dev)
cases = {
    'random': lambda R: base[:R].clone(),
    'dup_all': lambda R: base[:1].repeat(R, 1, 1).contiguous(),                       # every group has R rows
    'dup_pairs': lambda R: torch.stack([base[i // 2] for i in range(R)]).contiguous(),
    'dup_5_of_R': lambda R: torch.stack([base[0] if i < 5 else base[i] for i in range(R)]).contiguous(),
}
nchecks = 0
for name, mk in cases.items():
    for R in range(5, 9):
        x = mk(R)
        w = launch(x)
        assert bool(torch.isfinite(w).all()), f'non-finite {name} R={R}'
        assert not torch.equal(w, x), 'output unchanged'
        s = split(x)
        assert torch.equal(w, s), f'{name} R={R}: wide != 4+{R-4} split ({(w - s).abs().max().item()})'
        singles = torch.cat([launch(x[i:i + 1]) for i in range(R)])
        assert torch.equal(w, singles), f'{name} R={R}: wide != single rows'
        # a prefix of the rows in a smaller wide launch / 4-row launch gives the same rows (row independence)
        if R > 5: assert torch.equal(launch(x[:R - 1])[:R - 1], w[:R - 1]), f'{name} R={R}: R-1 rows differ'
        assert torch.equal(launch(x[:4]), w[:4]), f'{name} R={R}: first 4 rows differ from the 4-row launch'
        nchecks += 1
    print(f'{name}: R=5..8 wide == 4-row split == single rows: IDENTICAL', flush=True)
for k in range(5): assert torch.equal(launch(base[:7]), launch(base[:7])), 'repeat differs'
print('repeats identical', flush=True)
xn = torch.full((8, H, D), float('nan'), device=dev); launch(xn); launch(xn[:5])
print('all-NaN rows: launched, canaries intact, ctl zero', flush=True)
# 1..4 rows still go through the unchanged kernels and match the base path
for R in (1, 2, 3, 4):
    a = launch(base[:R]); b = torch.cat([launch(base[i:i + 1]) for i in range(R)])
    assert torch.equal(a, b), 'small-row regression'
print('SYNTH8 OK', SHAPE, KIND, 'cases', nchecks, flush=True)

# Exercise the actual Python dispatcher, including two batches of four rows.
from types import SimpleNamespace
from exllamav3.modules import moe_fused as mf
obj = object.__new__(mf.FusedMoEHalf)
obj.native, obj.RB, obj.SB, obj.shape = True, RB, SB, SHAPE
obj.gr = SimpleNamespace(q8_set=lambda: (fn, fn_scale, upt, up_scale), w_h=w_h, rms_eps=1e-6)
obj.router, obj.sgate, obj.wt, obj.svt, obj.ws = router, sgate, wt, svt, ws
mf.R8['on'] = True
mf.MF9.update(variant=0x160000, timing=0, tk2=True)
for R in (5, 6, 7, 8):
    x = base[:R].view(2, 4, H, D) if R == 8 else base[:R].view(1, R, H, D)
    os.environ['EXL3_MOE_R8_K6S8'] = '0'
    assert not obj._wide_ok()
    a = obj(x.clone()).clone()
    os.environ['EXL3_MOE_R8_K6S8'] = '1'
    assert obj._wide_ok()
    before = mf.R8['calls']
    b = obj(x.clone())
    torch.cuda.synchronize()
    assert mf.R8['calls'] == before + 1 and torch.equal(a, b), 'dispatch parity failed'
for R in (5, 8):
    a = launch(base[:R], 0x160000)
    b = torch.cat([launch(base[i:min(i+4, R)], 0x160000) for i in range(0, R, 4)])
    assert torch.equal(a, b), 'plain top-k wide/split mismatch'
print('PASS: Python dispatch, batch shape, and both top-k modes', flush=True)

# Reset identical inputs BEFORE each timed interval; do not repeatedly feed
# the output back as the next input. Alternate arm order to reduce drift.
import statistics, json
for label in ('random', 'dup_all'):
    for R in (5, 6, 7, 8):
        x = cases[label](R).view(1, R, H, D)
        work = x.clone()
        def raw(z):
            torch.ops.mf9.half(z, fn, fn_scale, upt, up_scale, w_h, router,
                              sgate, wt, svt, ws, 1e-6, 1, 0, VAR, *SHAPE)
        def measure(wide):
            work.copy_(x)
            torch.cuda.synchronize()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            if wide:
                raw(work)
            else:
                raw(work[:, :4]); raw(work[:, 4:])
            b.record(); b.synchronize()
            return a.elapsed_time(b)
        for _ in range(3):
            measure(False); measure(True)
        samples = {False: [], True: []}
        for repeat in range(16):
            for wide in ((False, True) if repeat % 2 == 0 else (True, False)):
                samples[wide].append(measure(wide))
        old, new = (statistics.median(samples[k]) for k in (False, True))
        print('BENCH:', json.dumps(dict(shape=KIND, gs=GS, seed=SEED, case=label,
              rows=R, split_ms=old, wide_ms=new, speedup=old/new)), flush=True)
for a, b in zip(ro, ro_ref):
    assert torch.equal(a, b), 'read-only input modified'
assert bool((ws[off[-1]:] == 0xAB).all()), 'workspace overrun'
print('PASS: wide K6/K8 synthetic parity, guards, repeatability, dispatch and timing', flush=True)
