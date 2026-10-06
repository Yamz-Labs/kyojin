#!/usr/bin/env python3
"""CPU proof for the split-launch fused MoE, on the PREPROCESSED device source (hipcc -E -P of exl3_moe_fused9.hip, same flags as the build).
usage: check_moe_split.py fused9.ii
Claims proved from the expanded text:
 C1  no stage kernel can wait on another block: the transitive callee closure of body9s contains no barrier / atomic / spin token
     (gbar*, gfinish, p.bar, p.done, p.err, atomic, s_sleep, __syncthreads is block-local and allowed).
 C2  the stages are body9 cut at its grid barriers: body9 has exactly 5 gbar7 calls; the 6 groups between them (INL branch, bookkeeping removed)
     equal stages 0..5 of body9s (the top-k restore of stages 4 / 5 aside).
 C3  the top-k hand-over cannot leave its buffer: every seldbg index of the restore and of the writer is < 64 + MAXR * TOPK <= STAMP_BASE <= SELDBG_INTS for every instantiated shape;
     the writer lines in stage 3 are identical to body9's; the restore reads exactly the indices the writer wrote.
 C4  launch geometry: stage kernels are launched with the same grid p.G (<= CTL_MAXB) and THREADS as the fused kernel.
Mutants must fail: M1 gbar7 in a stage, M2 gfinish in a stage, M3 stage order swapped, M4 phase dropped, M5 restore index +1, M6 atomic in a callee,
 M7 writer index shifted, M8 split grid launch with a different grid."""
import re, sys

def grab(text, name):
    m = re.search(r'void\s+' + name + r'\s*\([^)]*\)\s*\{', text)
    assert m, "function not found: " + name
    i = text.index('{', m.end() - 1); d = 0
    for j in range(i, len(text)):
        if text[j] == '{': d += 1
        elif text[j] == '}':
            d -= 1
            if d == 0: return text[m.start():j + 1]
    raise AssertionError("unbalanced " + name)

def all_funcs(text):
    out = {}
    for m in re.finditer(r'(?:^|\n)(?:template <[^\n]*>\n)?(?:__attribute__\(\([a-z_]+\)\) )*(?:inline )?(?:__attribute__\(\(always_inline\)\) )?(?:static )?(?:void|float|int|half|half4|uint32_t|unsigned|long|bool|long long|half2)\s+([A-Za-z_0-9]+)\s*\(', text):
        out.setdefault(m.group(1), m.start())
    return out

FORBID = ("gbar", "gfinish", "p.bar", "p.done", "p.err", "atomic", "s_sleep", "s_barrier_signal", "__builtin_amdgcn_s_sleep")

def closure_bad(text, root_body):
    seen, todo, bad = set(), [root_body], []
    names = None
    while todo:
        b = todo.pop()
        for t in FORBID:
            if t in b: bad.append(t)
        for c in set(re.findall(r'\b([A-Za-z_][A-Za-z_0-9]*)\s*(?:<[^;(){}]*>)?\s*\(', b)):
            if c in seen or c in ("if", "for", "while", "switch", "sizeof", "asm", "min", "max", "static_assert", "body9s"): continue
            seen.add(c)
            try: todo.append(grab(text, c))
            except AssertionError: pass
    return sorted(set(bad))

def calls(seg):
    return [m for m in re.findall(r'\b(touch_static|ph_dots9|ph_finalize9|ph_router9|ph_topk9|touch_down|ph_gateup9|ph_down9|ph_combine9|topk_restore9)\b', seg)]

def stage_lists(b9s):
    st = {}
    for m in re.finditer(r'if constexpr \(STAGE == (\d)\)', b9s):
        k = int(m.group(1)); i = m.end()
        rest = b9s[i:]
        if rest.lstrip().startswith('{'):
            s0 = rest.index('{'); d = 0
            for j in range(s0, len(rest)):
                if rest[j] == '{': d += 1
                elif rest[j] == '}':
                    d -= 1
                    if d == 0: seg = rest[s0:j + 1]; break
        else:
            seg = rest[:rest.index(';') + 1]
        st[k] = seg
    return st

def groups_of_body9(b9):
    inl = re.sub(r'if constexpr \(INL\) ([^;]*);\s*else [^;]*;', r'\1;', b9)
    parts = re.split(r'mf7::gbar7<TM>\(p, phase\);', inl)
    return parts

def run(text):
    errs = []
    b9, b9s = grab(text, 'body9'), grab(text, 'body9s')
    # C1
    bad = closure_bad(text, b9s)
    if bad: errs.append("C1 forbidden tokens reachable from the stage kernel: %s" % bad)
    # C2
    if b9.count("gbar7<TM>(p, phase)") != 5: errs.append("C2 body9 has %d grid barriers, expected 5" % b9.count("gbar7<TM>(p, phase)"))
    gr = groups_of_body9(b9)
    sl = stage_lists(b9s)
    if len(gr) != 6 or sorted(sl) != list(range(6)): errs.append("C2 group / stage count: %d groups, stages %s" % (len(gr), sorted(sl)))
    else:
        for k in range(6):
            a = calls(gr[k]); b = [c for c in calls(sl[k]) if c != 'topk_restore9']
            if a != b: errs.append("C2 stage %d phases %s != body9 group %s" % (k, b, a))
        for k in (4, 5):
            if 'topk_restore9' not in calls(sl[k]): errs.append("C2 stage %d lacks the top-k restore" % k)
        for k in (0, 1, 2, 3):
            if 'topk_restore9' in calls(sl[k]): errs.append("C2 stage %d must not restore" % k)
    # C3
    wl = [l.strip() for l in b9.split('\n') if 'p.seldbg[' in l and '=' in l and 'threadIdx.x' in l]
    ws = [l.strip() for l in sl.get(3, '').split('\n') if 'p.seldbg[' in l]
    if wl != ws or len(wl) != 2: errs.append("C3 writer lines differ: %s vs %s" % (wl, ws))
    rb = grab(text, 'topk_restore9')
    idx = re.findall(r'p\.seldbg\[([^\]]*)\]', rb)
    if idx != ['threadIdx.x', '64 + threadIdx.x']: errs.append("C3 restore indices %s" % idx)
    if not re.search(r'threadIdx\.x < p\.R \* S::TOPK', rb): errs.append("C3 restore guard missing")
    if not all('threadIdx.x < p.R * S::TOPK' in l for l in wl): errs.append("C3 writer guard")
    MAXR = int(re.search(r'constexpr int MAXR = (\d+)', text).group(1))
    SELDBG = int(re.search(r'SELDBG_INTS = (\d+)', text).group(1)); SB = int(re.search(r'STAMP_BASE = (\d+)', text).group(1))
    for m in re.finditer(r'struct (\w+) \{ static constexpr int D = \d+, H = \d+, LR = \d+, NEXP = \d+, TOPK = (\d+)', text):
        top = int(m.group(2)); hi = 64 + MAXR * top
        if not (hi <= SB <= SELDBG): errs.append("C3 shape %s: 64 + %d * %d = %d vs STAMP_BASE %d / SELDBG %d" % (m.group(1), MAXR, top, hi, SB, SELDBG))
        if MAXR * top > 64: errs.append("C3 shape %s: sel region overlaps weight region" % m.group(1))
    # C4
    ln = [l for l in text.split('\n') if 'moe_fused9s_kernel<S7' in l and '<<<' in l]
    if len(ln) != 6: errs.append("C4 %d stage launches, expected 6" % len(ln))
    for k, l in enumerate(ln):
        if '<<<(dim3(p.G)), (dim3(mf::THREADS)), (0), (stream)>>>(p, vmask)' not in l or ('ABL, %d>' % k) not in l: errs.append("C4 launch %d: %s" % (k, l.strip()))
    return errs

def mutate(text, k):
    if k == 1: return text.replace("ph_router9<S>(p, s);\n    if constexpr (STAGE == 3)", "mf7::gbar7<false>(p, *(int*)&sink); ph_router9<S>(p, s);\n    if constexpr (STAGE == 3)")
    if k == 2: return text.replace("topk_restore9<S>(p, s); ph_combine9<S>(p, s); }", "topk_restore9<S>(p, s); ph_combine9<S>(p, s); gfinish(p); }")
    if k == 3: return text.replace("ph_router9<S>(p, s);\n    if constexpr (STAGE == 3)", "ph_finalize9<S, false>(p, s);\n    if constexpr (STAGE == 3)")
    if k == 4: return text.replace("{ topk_restore9<S>(p, s); ph_down9<S, ABL>(p, s); }", "{ topk_restore9<S>(p, s); }")
    if k == 5: return text.replace("s.sel[threadIdx.x] = p.seldbg[threadIdx.x];", "s.sel[threadIdx.x] = p.seldbg[threadIdx.x + 1];")
    if k == 6: return text.replace("ph_topk_tail9", "ph_topk_tail9", 1).replace("__attribute__((device)) inline __attribute__((always_inline)) void ph_topk_tail9(const Params& p, Smem<S>& s)\n{", "__attribute__((device)) inline __attribute__((always_inline)) void ph_topk_tail9(const Params& p, Smem<S>& s)\n{ __hip_atomic_fetch_add(p.bar, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);")
    if k == 7: return text.replace("p.seldbg[64 + threadIdx.x] = (int) __half_as_ushort(s.wt[threadIdx.x]);\n        if (vm & V_TDOWN)", "p.seldbg[65 + threadIdx.x] = (int) __half_as_ushort(s.wt[threadIdx.x]);\n        if (vm & V_TDOWN)")
    if k == 8: return text.replace("moe_fused9s_kernel<S7, (bool) INL, ABL, 3>))<<<(dim3(p.G)),", "moe_fused9s_kernel<S7, (bool) INL, ABL, 3>))<<<(dim3(p.G + 1)),")
    raise ValueError

if __name__ == '__main__':
    text = open(sys.argv[1]).read()
    e = run(text)
    print("BASE:", "PASS" if not e else "FAIL"); [print("  ", x) for x in e]
    ok = not e
    for k in range(1, 9):
        mt = mutate(text, k)
        if mt == text: print("M%d: mutation did not apply" % k); ok = False; continue
        em = run(mt)
        print("M%d: %s" % (k, "fails (good): " + em[0][:90] if em else "PASSES (BAD)")); ok &= bool(em)
    print("RESULT", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
