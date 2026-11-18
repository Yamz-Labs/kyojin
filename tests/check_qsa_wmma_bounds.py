"""CPU bounds check for qsa_prefill_wmma.hip (no GPU). usage: python tests/check_qsa_wmma_bounds.py [file.hip]
 1. static: every global dereference of q / kc / vc / bt / idx / o goes through an IX_* macro, the o store is row-guarded,
    K/V token and page are clamped (TOKC / PGC), q and idx rows are clamped with min(.., R - 1).
 2. replay: macros are taken from the source text; every global access of the kernel (idx int4 loads, q fragments, K/V gathers
    through the block table, o stores) is replayed over adversarial data (indices past the context up to 2^31-1, huge / negative
    block-table entries, R not a multiple of 4) and compared with the exact element range of its buffer.
 3. LDS: static offsets of the aliased region are checked against its size."""
import os, re, sys
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exllamav3", "modules", "attention_fn", "qsa_prefill_wmma.hip")
src = open(path).read()
def die(m): print("FAIL:", m); sys.exit(1)

macros = {m.group(1): (m.group(2).replace(" ", "").split(","), m.group(3).strip())
          for m in re.finditer(r"^#define (\w+)\(([^)]*)\)\s+(.*?)(?://.*)?$", src, re.M)}
consts = {m.group(1): int(m.group(2)) for m in re.finditer(r"^#define (\w+)\s+(\d+)\b", src, re.M)}
consts["NGRP"] = consts["KPAD"] // 4
def mk(name):
    args, body = macros[name]
    body = body.replace("(int64_t)", "")
    return eval("lambda " + ",".join(args) + ": " + body, dict(consts, min=min, max=max))
IX_Q, IX_O, IX_IDX, IX_KV, PGC, TOKC = [mk(n) for n in ("IX_Q", "IX_O", "IX_IDX", "IX_KV", "PGC", "TOKC")]
HD, GRP, PAGE, KPAD, NR, NGRP, MAXU = [consts[k] for k in ("HD", "GRP", "PAGE", "KPAD", "NR", "NGRP", "MAXU")]

# ---- part 1: static
body = src[src.index('extern "C" __global__'):]
for d in re.findall(r"[^;\n]*\b(?:q|kc|vc|bt|idx|o)\s*(?:\+|\[)[^;\n]*", body):
    if "IX_" not in d and "bt[PGC(" not in d: die("raw global access: " + d.strip())
for ln in body.splitlines():
    if re.search(r"\(o \+", ln) and "row < R" not in ln: die("unguarded store: " + ln.strip())
for pat in ("TOKC(pg", "PGC(t0", "min(row0 + rho, R - 1)", "min(row0 + rhoq, R - 1)"):
    if pat not in body: die("missing clamp: " + pat)
print("static ok")

# ---- part 3: LDS
BIG = int(re.search(r"sBig\[(\d+)\]", body).group(1))
LDQ, LDT, SORTN = consts["LDQ"], consts["LDT"], consts["SORTN"]
offs = {"sKV": 8448, "sV": 8448 + 0, "sP": 20736, "sSp": 23040}
assert int(re.search(r"sBig \+ (\d+)\)", body).group(1)) == 8448
lds = [("sort", 4 * SORTN), ("K tile", 2 * (15 * LDQ + 15 * 16 + 8 + 8)), ("V^T", 8448 + 2 * (255 * LDT + 15 + 1)),
       ("P", 20736 + 2 * (47 * LDT + 15 + 1)), ("partial scores", 23040 + 4 * ((2 * 32 + 31) * 8 + 8)),
       ("out staging", 2 * (23 * LDQ + 16 * 15 + 2 * 7 + 1 + 1)), ("out read", 2 * (23 * LDQ + 31 * 8 + 8))]
for n, e in lds:
    if e > BIG: die(f"LDS overflow {n}: {e} > {BIG}")
if consts["NGRP"] * NR > MAXU: die("union exceeds MAXU")
print("lds ok (aliased region %d B, max use %d B)" % (BIG, max(e for _, e in lds)))

# ---- part 2: replay
rng = np.random.default_rng(11)
viol = []
def chk(kind, lo, n_el, size, what):
    if lo < 0 or lo + n_el > size: viol.append((kind, what, lo, n_el, size))

def scenario(ctx, R, q_stride, o_stride, tok_stride, adversarial):
    npages = ctx // PAGE; ntok = ctx
    q_size = (R - 1) * q_stride + 24 * HD; o_size = (R - 1) * o_stride + 24 * HD
    kv_size = (ntok - 1) * tok_stride + 2 * HD; idx_size = R * KPAD
    if adversarial:
        bt = rng.integers(-2**31, 2**31 - 1, npages); bt[::3] = rng.integers(0, npages, len(bt[::3]))
    else:
        bt = rng.permutation(npages)
    blocks = sorted({0, R // 2, max(0, ((R - 1) // NR)) , (R - 1) // NR})
    for rb in blocks:
        row0 = rb * NR
        nU = 0; ublk = []
        for g in range(2):
            for rho in range(NR):                       # phase A: idx loads
                row = min(row0 + rho, R - 1)
                n = int(rng.integers(1, KPAD + 1))
                ids = np.sort(rng.integers(0, ctx if not adversarial else 2**31 - 1, n))
                if adversarial: ids[-1] = 2**31 - 1
                lst = np.full(KPAD, -1, np.int64); lst[:n] = ids
                for j in range(NGRP):
                    chk("idx", IX_IDX(row, 4 * j), 4, idx_size, (row, j))
                    f = int(lst[4 * j]) if lst[4 * j] >= 0 else -1
                    if f >= 0:
                        key = ((f >> 2) << 6) & 0xFFFFFFFF      # uint32 wrap as in the kernel
                        ublk.append(key >> 6)
            for w in range(6):                           # phase B: q fragments
                ctq, dhq = w % 3, w // 3
                for ln in (0, 15, 16, 31):
                    colq = 16 * ctq + (ln & 15); rhoq = colq // GRP; hq = colq - rhoq * GRP
                    rowq = min(row0 + rhoq, R - 1)
                    for ks in range(8): chk("q", IX_Q(rowq, g * GRP + hq, 16 * (dhq * 8 + ks), q_stride), 16, q_size, (rowq, hq))
            ublk = sorted(set(ublk))[:MAXU - 8] or [0]
            ublk_p = ublk + [0] * 8                      # padding entries written by the kernel
            ntile = (len(ublk) + 3) >> 2
            for i in range(ntile):                       # phase C: K / V gathers
                for kk in range(16):
                    blk = ublk_p[4 * i + (kk >> 2)]; t0 = blk * 4
                    pg = int(bt[PGC(t0 // PAGE, npages)])
                    tk = TOKC(pg * PAGE + (t0 % PAGE) + (kk & 3), ntok)
                    for dch in (0, 15):
                        chk("kv", IX_KV(tk, g, dch * 16, tok_stride), 16, kv_size, (tk, g))
            for c in range(24 * 32 * 2):                 # phase D: stores
                ph = c // (24 * 32); cc = c % (24 * 32); cl = cc >> 5; ch = cc & 31
                col = cl + 24 * ph; rho = col // GRP; h = col - rho * GRP; row = row0 + rho
                if row < R: chk("o", IX_O(row, g * GRP + h, ch * 8, o_stride), 8, o_size, (row, h))
        if len(ublk) > MAXU - 8: viol.append(("union", "too many", len(ublk), 0, MAXU))

for ctx in (16384, 131072, 163840):
    for R in (1, 2, 3, 5, 2048, 8190, 8192):
        for adv in (False, True):
            scenario(ctx, R, 24 * HD, 24 * HD, 2 * HD, adv)
    scenario(ctx, 4097, 3 * 24 * HD, 24 * HD, 2 * HD + 64, True)       # padded strides
if viol:
    print("FAIL: %d violations, first: %s" % (len(viol), viol[:3])); sys.exit(1)
print("replay ok")
print("OK")
