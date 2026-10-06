"""CPU bounds proof for gdn_fused_h.hip (run BEFORE any GPU run of a kernel revision).
Parses the kernel source itself: the IX_* / TCL macros, the guard of every global store, and every LDS access expression
(ld16 calls, uint32 / uint16 stores into sW sX sY sA sG sGE sRS). Replays all of them over the real shapes
(Qwen H 16, HV 48, K = V = 128, BT 64; T = 1..4097 incl. non-multiples of 64; B 1 and 2) and asserts every element
index (plus access width and 16 B alignment of vector accesses) stays inside the tensor / LDS buffer.
Usage: check_gdn_bounds.py <file.hip> [--mutate N] [--defs "ABL=0 ..."]  (mutations 1..9 must FAIL, 16..18 with --defs "OST=1 CKPT=1" (checkpoint store index / column / batch item), 10..15 with --defs "OST=1": they model a dropped clamp, a dropped
store guard, an LDS row-stride overflow, a bad state-store index and a bad LDS q write).
Exit 0 = proof clean, 1 = violation found."""
import itertools, re, sys
import numpy as np

src = open(sys.argv[1]).read()
mut = int(sys.argv[sys.argv.index("--mutate") + 1]) if "--mutate" in sys.argv else 0
if mut == 1:
    src = src.replace("min((t_), (T_) - 1)", "(t_)")                       # drop the row clamp
elif mut == 2:
    src = src.replace("if (t < T) o[IX_V(bos + t, i_hv, col)]", "if (t < T + 64) o[IX_V(bos + t, i_hv, col)]")
elif mut == 3:
    src = src.replace("sA + ar * LDT + ac", "sA + ar * (LDT + 8) + ac")
elif mut == 4:
    src = src.replace("ht[IX_S(i_nh, 16 * kt + 2 * r + half, col)]", "ht[IX_S(i_nh, 16 * kt + 2 * r + half + 8, col)]")
elif mut == 5:
    src = src.replace("sX + (16 * tg + s) * LDR + 2 * cp)", "sX + (16 * tg + s + 4) * LDR + 2 * cp)")
elif mut == 6:
    src = src.replace("#define IX_VR(tc, hv, c) ((((int64_t) (tc)) * PV)", "#define IX_VR(tc, hv, c) ((((int64_t) (tc)) * (PV + 8))")     # wrong v pitch
elif mut == 7:
    src = src.replace("uint32_t vv = *(const uint32_t*) (v + IX_VR(tc, i_hv, 2 * cp));", "uint32_t vv = *(const uint32_t*) (v + IX_VR(t, i_hv, 2 * cp));")   # unclamped row
elif mut == 8:
    src = src.replace("#define TCL(bos_, t_, T_) ((bos_) + min((t_), (T_) - 1))", "#define TCL(bos_, t_, T_) ((bos_) + min((t_), (T_) - 1) + t)")   # macro argument capturing the outer loop variable t
elif mut == 9:
    src = src.replace("#define IX_G(tc, hv) (((int64_t) (tc)) * HV + (hv))", "#define IX_G(tc, hv) (((int64_t) (tc)) * HV + (hv) + t0)")       # macro body capturing an outer variable
elif mut == 10:
    src = src.replace("if (t < T) *(uint4*) (o + IX_V(bos + t, i_hv, cc)) = x;", "if (t < T + 64) *(uint4*) (o + IX_V(bos + t, i_hv, cc)) = x;")     # OST: dropped row guard of the vector store
elif mut == 11:
    src = src.replace("const uint4 x = *(const uint4*) (sW + rr * LDR + cc);", "const uint4 x = *(const uint4*) (sW + (rr + 8) * LDR + cc);")       # OST: LDS read past the tile
elif mut == 12:
    src = src.replace("cc = 8 * (tid & 15)", "cc = 8 * (tid & 31)")                                                  # OST: columns past the head
elif mut == 13:
    src = src.replace("        __syncthreads();  // every wave is done reading sW (w) before the o tile is staged into it\n", "")      # OST: staging races with the vn chain reads of sW
elif mut == 14:
    src = src.replace("        SYNCA(64);  // k^T visible\n", "")                                                         # OST: copy-out races with the staging writes
elif mut == 15:
    src = src.replace("sW[(16 * a + 2 * r + half) * LDR + col] = (uint16_t) f2bf(res);", "sW[(16 * a + 2 * r + half) * LDR + col + 80] = (uint16_t) f2bf(res);")   # OST: staging write past the tile
elif mut == 16:
    src = src.replace("hc[IX_S(i_nh, 16 * kt + 2 * r + half, col)]", "hc[IX_S(i_nh, 16 * kt + 2 * r + half + 8, col)]")      # CKPT: bad state-row index of the mid-chunk checkpoint store
elif mut == 17:
    src = src.replace("hc2[IX_S(i_nh, 16 * kt + 2 * r + half, col)]", "hc2[IX_S(i_nh, 16 * kt + 2 * r + half, col + 64)]")     # CKPT: bad column of the second checkpoint store
elif mut == 18:
    src = src.replace("hc[IX_S(i_nh, 16 * kt + 2 * r + half, col)]", "hc[IX_S(i_nh + HV, 16 * kt + 2 * r + half, col)]")      # CKPT: checkpoint written one batch item too far
if mut:
    print(f"[mutation {mut}] expecting a violation")

fails = []
def fail(msg):
    if len(fails) < 12: fails.append(msg)

# ---- constants and macros from the source
consts = {}
for m in re.finditer(r"^#define (BT|KD|VD|LDR|LDT)\s+(\d+)", src, re.M): consts[m.group(1)] = int(m.group(2))
for k in ("BT", "KD", "VD", "LDR", "LDT"): assert k in consts, k
BT, KD, VD, LDR, LDT = (consts[k] for k in ("BT", "KD", "VD", "LDR", "LDT"))
macros = {}
for m in re.finditer(r"^#define ((?:IX_\w+|TCL))\(([^)]*)\)\s+(.*)$", src, re.M):
    name, args, body = m.group(1), [a.strip() for a in m.group(2).split(",")], re.sub(r"//.*$", "", m.group(3)).strip()
    macros[name] = (args, body)
def c2py(e):
    e = e.replace("(int64_t)", "")
    e = re.sub(r"\bmin\(", "np.minimum(", e)
    return e
def mcall(name, *vals, env):
    args, body = macros[name]
    e = c2py(body)
    local = dict(zip(args, vals))
    return eval(e, {"np": np, "BT": BT, "KD": KD, "VD": VD}, {**env, **local})

# ---- global accesses over the real shapes
def global_proof(B, T, H, HV, PV):
    NT = (T + BT - 1) // BT
    ext = {"qk": B * T * H * KD, "v": B * T * HV * VD, "vr": (B * T - 1) * PV + HV * VD, "g": B * T * HV, "A": B * T * HV * BT, "S": B * HV * KD * VD}
    # guard of the o store, taken from the source
    gm = re.search(r"if \(([^)]*)\) o\[IX_V\(bos \+ t, i_hv, col\)\]", src)
    assert gm, "o store guard not found"
    ostore_guard = c2py(gm.group(1))
    tid = np.arange(256); cp = tid & 63; tg = tid >> 6; ar = tid >> 2; ac = 16 * (tid & 3)
    for i_n in range(B):
        bos = i_n * T
        for i_hv in sorted({0, 1, HV // H - 1, HV // H, HV // 2, HV - 2, HV - 1}):   # extremes of every index term (all affine in i_hv)
            i_h = i_hv // (HV // H)
            env = {"H": H, "HV": HV, "T": T, "bos": bos, "PV": PV}
            i_nh = i_n * HV + i_hv
            def chk(name, idx, width, tens, tag):
                idx = np.asarray(idx, dtype=np.int64)
                if idx.size and (idx.min() < 0 or (idx + width).max() > ext[tens]):
                    fail(f"{tag}: B{B} T{T} hv{i_hv}: {name} idx [{idx.min()}, {(idx + width).max()}) outside 0..{ext[tens]}")
            for i_t in range(NT):
                t0 = i_t * BT
                last = min(t0 + BT, T) - 1
                chk("g_last", mcall("IX_G", bos + last, i_hv, env=env), 1, "g", "gl")
                tcA = mcall("TCL", bos, t0 + ar, T, env=env)
                chk("A", mcall("IX_A", tcA, i_hv, ac, env=env), 16, "A", "A")
                if (mcall("IX_A", tcA, i_hv, ac, env=env) % 16).any(): fail("A vector alignment")
                for s2, e in itertools.product(range(8), range(2)):
                    t = t0 + 16 * tg + 2 * s2 + e
                    tc = mcall("TCL", bos, t, T, env=env)
                    chk("k/q", mcall("IX_QK", tc, i_h, 2 * cp, env=env), 2, "qk", "kq")
                    chk("v", mcall("IX_VR", tc, i_hv, 2 * cp, env=env), 2, "vr", "v")
                    chk("g/beta", mcall("IX_G", tc, i_hv, env=env), 1, "g", "gb")
                t = t0 + np.arange(64)
                tcg = mcall("TCL", bos, t, T, env=env)
                chk("sG load", mcall("IX_G", tcg[t < T], i_hv, env=env), 1, "g", "sG")
                # o store: rows t0 + 16a + 2r + half, cols 0..127, guarded by the source's guard
                a, r, half, col = np.meshgrid(np.arange(4), np.arange(8), np.arange(2), np.arange(128), indexing="ij")
                t = t0 + 16 * a + 2 * r + half
                ok = eval(ostore_guard, {"np": np}, {"t": t, "T": T})
                idx = mcall("IX_V", bos + t[ok], i_hv, col[ok], env=env)
                chk("o store", idx, 1, "v", "o")
            kt, r, half, col = np.meshgrid(np.arange(8), np.arange(8), np.arange(2), np.arange(128), indexing="ij")
            idx = mcall("IX_S", i_nh, 16 * kt + 2 * r + half, col, env=env)
            chk("h0/ht", idx, 1, "S", "state")
    # the state-store expression text must be the one proven above (mutation 4 changes it)
    for m in re.finditer(r"(?:ht|h0|hc2?)\[IX_S\(i_nh, ([^,]*), col\)\]", src):
        kt, r, half = np.meshgrid(np.arange(8), np.arange(8), np.arange(2), indexing="ij")
        row = eval(c2py(m.group(1)), {"np": np}, {"kt": kt, "r": r, "half": half})
        if row.min() < 0 or row.max() >= KD: fail(f"state row expr {m.group(1)} range {row.min()}..{row.max()}")

# ---- OST (o tile staged through sW, 16 B vector copy-out): bounds of the vector store and of the LDS read, and the barrier order
OST_RAW = re.search(r"#if OST\n\s*// o tile.*?\n\s*#pragma unroll\n\s*for \(int i = 0; i < 4; \+\+i\)\n\s*\{\n\s*const int (rr = .*?), (cc = .*?), t = t0 \+ rr;\n\s*const uint4 x = \*\(const uint4\*\) \(sW \+ (.*?)\);\n\s*if \((.*?)\) \*\(uint4\*\) \(o \+ (IX_V\(.*?\))\) = x;", src, re.S)
assert OST_RAW, "OST copy-out block not found"
def ost_global_proof(B, T, H, HV, PV):
    NT = (T + BT - 1) // BT
    tid, i = np.meshgrid(np.arange(256), np.arange(4), indexing="ij")
    rr = eval(c2py(OST_RAW.group(1).split("=", 1)[1]), {"np": np}, {"tid": tid, "i": i})
    cc = eval(c2py(OST_RAW.group(2).split("=", 1)[1]), {"np": np}, {"tid": tid, "i": i})
    for ldx in (eval(c2py(OST_RAW.group(3)), {"np": np}, {"rr": rr, "cc": cc, "LDR": LDR}),):
        if ldx.min() < 0 or ldx.max() + 8 > BT * LDR: fail(f"OST LDS sW read range {ldx.min()}..{ldx.max() + 8} > {BT * LDR}")
        if (ldx % 8).any(): fail("OST LDS sW read alignment")
    ext = B * T * HV * VD
    for i_n in range(B):
        bos = i_n * T
        for i_hv in sorted({0, 1, HV // H - 1, HV // H, HV // 2, HV - 2, HV - 1}):
            env = {"H": H, "HV": HV, "T": T, "bos": bos, "PV": PV}
            for i_t in range(NT):
                t = i_t * BT + rr
                ok = eval(c2py(OST_RAW.group(4)), {"np": np}, {"t": t, "T": T})
                idx = np.asarray(mcall("IX_V", bos + t[ok], i_hv, cc[ok], env=env), dtype=np.int64)
                if idx.size and (idx.min() < 0 or idx.max() + 8 > ext): fail(f"OST o vector store: B{B} T{T} hv{i_hv}: [{idx.min()}, {idx.max() + 8}) outside 0..{ext}")
                if (idx % 8).any(): fail("OST o vector store alignment")
                # every in-range (t, col) element must be written exactly once (coverage)
                if i_hv == 0:
                    cover = np.zeros((min(BT, T - i_t * BT), VD), dtype=np.int32)
                    for tt_, c_ in zip((rr[ok] - 0).ravel(), cc[ok].ravel()):
                        if tt_ < cover.shape[0]: cover[tt_, c_:c_ + 8] += 1
                        else: fail(f"OST store of row {tt_} past the chunk (T {T})")
                    if not (cover == 1).all(): fail(f"OST coverage of the o tile: B{B} T{T} chunk {i_t}")
def ost_order_check():
    i_chain = src.rfind("ld16(sW, 16 * a + ll, LDR, 16 * kt)")
    i_bar1 = src.find("__syncthreads();  // every wave is done reading sW")
    i_stage = src.find("sW[(16 * a + 2 * r + half) * LDR + col] = (uint16_t) f2bf(res);")
    i_bar2 = src.find("SYNCA(64);  // k^T visible")
    i_copy = src.find("const uint4 x = *(const uint4*) (sW + ")
    if not (0 <= i_chain < i_bar1 < i_stage < i_bar2 < i_copy): fail(f"OST barrier order broken: chain {i_chain} bar1 {i_bar1} stage {i_stage} bar2 {i_bar2} copy {i_copy}")

# ---- LDS accesses from the source text
sizes = {"sW": BT * LDR, "sX": KD * LDT, "sY": KD * LDT, "sA": BT * LDT, "sG": BT, "sGE": BT, "sRS": BT}
VARS = {"tid": range(256), "cp": range(64), "tg": range(4), "s2": range(8), "s": range(16), "c": range(2), "ar": range(64),
        "ac": (0, 16, 32, 48), "a": range(4), "j": range(4), "r": range(8), "half": range(2), "col": range(128),
        "ll": range(16), "kt": range(8), "tix": range(10), "wv": range(8), "e": range(2), "tt": range(64)}
DERIVED = {"ri": "16 * a + 2 * r + half", "cj": "16 * j + ll", "wv": None}
def lds_check(buf, expr, width, align, where):
    e = expr
    for d, de in DERIVED.items():
        if de: e = re.sub(rf"\b{d}\b", f"({de})", e)
    names = sorted(set(re.findall(r"[A-Za-z_]\w*", e)) & set(VARS) - {"LDR", "LDT"})
    # Aqk tile decode: a, j come from tix; use the full a, j grid (superset)
    grids = [range(64) if (n == "tid" and buf in ("sG", "sGE", "sRS")) else VARS[n] for n in names]
    if "tid" in names and buf in ("sG", "sGE", "sRS"):
        assert "if (tid < BT)" in src, "sG writes must stay under tid < BT"
    envc = {"LDR": LDR, "LDT": LDT, "BT": BT, "KD": KD}
    arrs = np.meshgrid(*[np.array(list(g)) for g in grids], indexing="ij") if grids else []
    val = eval(c2py(e), {"np": np}, {**envc, **dict(zip(names, arrs))})
    val = np.asarray(val)
    if val.min() < 0 or val.max() + width > sizes[buf]:
        fail(f"LDS {buf} [{where}] {expr}: range {val.min()}..{val.max() + width} > {sizes[buf]}")
    if align > 1 and (val % align).any():
        fail(f"LDS {buf} [{where}] {expr}: alignment {align}")
n_lds = 0
for m in re.finditer(r"ld16\((s[WXYA]),\s*([^,]+),\s*(LDR|LDT),\s*([^)]+)\)", src):
    buf, row, ld, colx = m.groups()
    lds_check(buf, f"({row}) * {ld} + ({colx})", 16, 8, "ld16"); n_lds += 1
for m in re.finditer(r"\*\(uint32_t\*\) \((s[WXYA]) \+ (.*?)\) = ", src):
    lds_check(m.group(1), m.group(2), 2, 2, "st32"); n_lds += 1
for m in re.finditer(r"(?<![\w])(s[WXYA]|sG|sGE|sRS)\[([^\]]+)\]\s*=", src):
    lds_check(m.group(1), m.group(2), 1, 1, "st16"); n_lds += 1
for m in re.finditer(r"uint4\* d = \(uint4\*\) \((s[WXYA]) \+ (.*?)\);", src):
    lds_check(m.group(1), m.group(2), 16, 8, "st128"); n_lds += 1
for m in re.finditer(r"(?<![\w=])(sG|sGE|sRS)\[([^\]]+)\](?!\s*=)", src):
    if m.group(2).strip() == "BT": continue   # the declaration
    lds_check(m.group(1), m.group(2), 1, 1, "ld32"); n_lds += 1

# ---- layer 2 (rule 14): the same global accesses evaluated from the PREPROCESSED source (cpp: macros expanded, -D flags applied), so a macro
# argument / body that captures an outer variable is seen exactly as the compiler sees it. Statement forms are matched in the expanded text.
import subprocess, tempfile, os
defs = sys.argv[sys.argv.index("--defs") + 1].split() if "--defs" in sys.argv else []
def expanded_text():
    raw = "\n".join(l for l in src.split("\n") if not l.startswith("#include"))
    with tempfile.NamedTemporaryFile("w", suffix=".cpp", delete=False) as f: f.write(raw); fn = f.name
    out = subprocess.run(["cpp", "-P", "-undef", "-nostdinc", "-x", "c++"] + [f"-D{d}" for d in defs] + [fn], capture_output=True, text=True, check=True).stdout
    os.unlink(fn)
    return out
EXP = expanded_text()
def site(pattern, count=None):
    ms = re.findall(pattern, EXP, re.M)
    if not ms or (count is not None and len(ms) != count):
        fail(f"expanded-source site not found / wrong count ({len(ms)}): {pattern}"); return []
    return ms
def balanced_arg(text_after):
    d = 1; i = 0
    while i < len(text_after) and d:
        d += text_after[i] == "("; d -= text_after[i] == ")"; i += 1
    return text_after[:i - 1]
def pyx(e):
    e = e.replace("(int64_t)", "").replace("(int)", "")
    return re.sub(r"\bmin\(", "np.minimum(", e)
S_GL = site(r"const float gl = g\[(.+)\];", 1)
S_A = site(r"ld16g\(A \+ (.+), t0 \+ ar < T\)", 1)
S_TC = site(r"const int64_t tc = (.+);", 1)
S_K = site(r"uint32_t kk = \*\(const uint32_t\*\) \(k \+ (.+)\);$", 1)
S_Q = site(r"uint32_t qq = \*\(const uint32_t\*\) \(q \+ (.+)\);$", 1)
S_V = site(r"uint32_t vv = \*\(const uint32_t\*\) \(v \+ (.+)\);$", 1)
S_G1 = site(r"float gv = g\[(.+)\];$", 1)
S_B1 = site(r"uint32_t bb = beta\[(.+)\];$", 1)
S_SG = site(r"const float gv = \(t < T\) \? g\[(.+)\] : 0\.0f;", 1)
OSTMODE = "OST=1" in defs
if OSTMODE:
    S_O = []
    S_OV = site(r"if \((.+)\) \*\(uint4\*\) \(o \+ (.+)\) = x;", 1)
    S_RC = site(r"const int rr = (.+), cc = (.+), t = (.+);", 1)
    S_LR = site(r"const uint4 x = \*\(const uint4\*\) \(sW \+ (.+)\);", 1)
else:
    S_O = site(r"if \((.+)\) o\[(.+)\] = \(uint16_t\) f2bf\(res\);", 1)
S_H0 = site(r"h0 \? h0\[(.+)\] : 0\.0f", 1)
S_HT = site(r"ht\[(.+)\] = h\[kt\]\[r\];", 1)
S_HC = site(r"hc2?\[(.+)\] = h\[kt\]\[r\];", 2) if "CKPT=1" in defs else []
def expanded_proof(B, T, H, HV, PV):
    NT = (T + 63) // 64
    ext = {"qk": B * T * H * KD, "vr": (B * T - 1) * PV + HV * VD, "v": B * T * HV * VD, "g": B * T * HV, "A": B * T * HV * BT, "S": B * HV * KD * VD}
    tid = np.arange(256); cp = tid & 63; tg = tid >> 6; ar = tid >> 2; ac = 16 * (tid & 3)
    def chk(name, idx, width, tens, tag, B_, T_, hv):
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size and (idx.min() < 0 or (idx + width).max() > ext[tens]):
            fail(f"[expanded] {tag}: B{B_} T{T_} hv{hv}: {name} idx [{idx.min()}, {(idx + width).max()}) outside 0..{ext[tens]}")
    for i_n in range(B):
        bos = i_n * T
        for i_hv in sorted({0, HV // H - 1, HV // H, HV // 2, HV - 1}):
            i_h = i_hv // (HV // H); i_nh = i_n * HV + i_hv
            base = {"np": np, "bos": bos, "T": T, "H": H, "HV": HV, "PV": PV, "i_h": i_h, "i_hv": i_hv, "i_nh": i_nh}
            for i_t in range(NT):
                t0 = i_t * 64; last = min(t0 + 64, T) - 1
                env = {**base, "t0": t0, "last": last, "ar": ar, "ac": ac}
                chk("g_last", eval(pyx(S_GL[0]), env), 1, "g", "gl", B, T, i_hv)
                ia = eval(pyx(S_A[0]), env); chk("A", ia, 16, "A", "A", B, T, i_hv)
                if (np.asarray(ia) % 16).any(): fail("[expanded] A alignment")
                s2, e = np.meshgrid(np.arange(8), np.arange(2), indexing="ij")
                tt_ = tg[:, None, None] * 16 + 2 * s2[None] + e[None]
                env2 = {**base, "t0": t0, "cp": cp[:, None, None], "tg": tg[:, None, None], "t": t0 + tt_, "tt": tt_}
                tc = eval(pyx(S_TC[0]), env2); env2["tc"] = tc
                for nm, ex, tens, w in (("k", S_K, "qk", 2), ("q", S_Q, "qk", 2), ("v", S_V, "vr", 2), ("g", S_G1, "g", 1), ("beta", S_B1, "g", 1)):
                    chk(nm, eval(pyx(ex[0]), env2), w, tens, nm, B, T, i_hv)
                tq = np.arange(64); env3 = {**base, "t0": t0, "tid": tq, "t": t0 + tq}
                ok = env3["t"] < T
                chk("sG load", np.asarray(eval(pyx(S_SG[0]), env3))[ok], 1, "g", "sG", B, T, i_hv)
                a, r, half, col = np.meshgrid(np.arange(4), np.arange(8), np.arange(2), np.arange(128), indexing="ij")
                env4 = {**base, "t0": t0, "a": a, "r": r, "half": half, "col": col, "t": t0 + 16 * a + 2 * r + half}
                if OSTMODE:
                    tid_, i_ = np.meshgrid(np.arange(256), np.arange(4), indexing="ij")
                    env6 = {**base, "t0": t0, "tid": tid_, "i": i_}
                    env6["rr"] = eval(pyx(S_RC[0][0]), env6); env6["cc"] = eval(pyx(S_RC[0][1]), env6); env6["t"] = eval(pyx(S_RC[0][2]), env6)
                    okv = np.broadcast_to(eval(pyx(S_OV[0][0]), env6), tid_.shape)
                    idx = np.asarray(eval(pyx(S_OV[0][1]), env6)); lds = np.asarray(eval(pyx(S_LR[0]), {**env6, "LDR": LDR}))
                    chk("o vector store", idx[okv], 8, "v", "o", B, T, i_hv)
                    if (idx[okv] % 8).any(): fail("[expanded] o vector store alignment")
                    if lds.min() < 0 or lds.max() + 8 > BT * LDR or (lds % 8).any(): fail(f"[expanded] OST LDS read {lds.min()}..{lds.max() + 8}")
                else:
                    okm = eval(pyx(S_O[0][0]), env4)
                    idx = np.asarray(eval(pyx(S_O[0][1]), env4))
                    chk("o store", idx[np.broadcast_to(okm, idx.shape)], 1, "v", "o", B, T, i_hv)
            kt, r, half, col = np.meshgrid(np.arange(8), np.arange(8), np.arange(2), np.arange(128), indexing="ij")
            env5 = {**base, "kt": kt, "r": r, "half": half, "col": col}
            for ex in (S_H0, S_HT): chk("state", eval(pyx(ex[0]), env5), 1, "S", "state", B, T, i_hv)
            for ex in S_HC: chk("ckpt state", eval(pyx(ex), env5), 1, "S", "ckpt", B, T, i_hv)

ost_order_check()
shapes = [(1, T, 16, 48) for T in list(range(1, 130)) + [255, 511, 1000, 1999, 2047, 2048, 2049, 4095, 4096, 4097]] + [(2, 100, 16, 48), (2, 2048, 16, 48), (1, 4096, 16, 64)]
if "--quick" in sys.argv: shapes = [(1, T, 16, 48) for T in (1, 63, 64, 65, 1999, 2048)] + [(2, 100, 16, 48)]
PVS = lambda HV_: (HV_ * VD, 2 * 16 * KD + HV_ * VD)    # contiguous v, and the conv-output slice pitch (q | k | v)
n_exp = 0
for (B, T, H, HV) in shapes:
    for PV in PVS(HV):
        global_proof(B, T, H, HV, PV)
        ost_global_proof(B, T, H, HV, PV)
    if T in (1, 2, 17, 63, 64, 65, 100, 129, 1999, 2048, 4097) or B > 1 or HV != 48:
        for PV in PVS(HV):
            try: expanded_proof(B, T, H, HV, PV); n_exp += 1
            except NameError as ex: fail(f"[expanded] expression uses a name that is not in scope at its site ({ex}): macro capture")
print(f"expanded-source layer: {n_exp} (shape, pitch) replays, {len(S_GL + S_A + S_TC + S_K + S_Q + S_V + S_G1 + S_B1 + S_SG + S_O + S_H0 + S_HT + S_HC)} sites")
print(f"checked {len(shapes)} shapes, {len(macros)} macros, {n_lds} LDS access expressions")
if fails:
    print("VIOLATIONS:"); [print("  ", f) for f in fails]; sys.exit(1)
print("bounds proof CLEAN")
