#!/usr/bin/env python3
# roundfloor1: per-kernel-family "measured vs byte floor" for ONE served MTP round (R=2, n1f2).
#
#   usage: round_floor.py <run.rocprof dir> <round_decomp.json> [--bw 215] [--json out.json]
#
# Two independent halves, joined only at the end:
#   MEASURED  from the rocprofv3 kernel trace, clipped into the marker-delimited round windows:
#             union of kernel intervals per family, plus the host idle (round wall - busy).
#   BYTES     from the EXL3 tensor inventory dumped by round_decomp.py (element_size*numel of
#             every weight tensor, i.e. what the kernels actually stream) times a stated
#             per-round read multiplicity, and the distinct routed experts the union touches.
#
# The byte floor is bytes / BW. It is a LOWER BOUND on a perfect-bandwidth machine, not a target:
# a family whose data is L2-resident (norms, mHC) has a ~0 floor by construction and its whole
# measured cost is gap. Read that way, the gap column ranks where bytes are NOT the limit.
import csv, glob, json, os, re, sys, collections, bisect, argparse

ap = argparse.ArgumentParser()
ap.add_argument("dir"); ap.add_argument("json")
ap.add_argument("--bw", type=float, default=215.0, help="GB/s used for the byte floor")
ap.add_argument("--bw2", type=float, default=230.0, help="second BW for the sensitivity column")
ap.add_argument("--last", type=int, default=0, help="use only the last N rounds (steady state)")
ap.add_argument("--out", default=None)
ap.add_argument("--arm", default=None, help="verifyrow1: only rounds inside this arm_<i>_R<r> span")
A = ap.parse_args()

# ---------------------------------------------------------------- families
# Base name = Kernel_Name up to the argument list, "void " stripped. The strip matters: rocprofv3
# records "void had_hf_r_128_kernel<..>(..)", so an anchored ^hc_ / ^Cijk_ never fires and the
# mHC kernels silently fall into the catch-all (that is where D1-1's "dense GEMM 26.9 ms" hid
# 9.0 ms of KDA qkv gemv).
FAMS = [
    ("MoE expert gemv",  r"moe_gu|moe_down|moe_combine|mpw_gemm|mpw2_gemm|moe_union|fused_silu_mul|"
                         r"router_kernel|gatherTopK|sigmoid_kernel|topk_"),
    ("KDA",              r"kda_|conv1d_update|short_conv|gated_delta|local_cumsum|l2norm|"
                         r"causal_conv1d|gdn"),
    ("DSA/MLA attn",     r"^_mla_|^_dsa_|_mla_|_dsa_|dsa_topk|dsa_index|radixSort|sbtopk|"
                         r"index_kpool|absorb|unfold|plane_update|kv_update"),
    ("norms/mHC",        r"^hc_|rms_norm|layer_norm|softmax|norm"),
    ("dense/shared gemv", r"had_hf_r|had_ff_r|exl3_gemv_kernel|exl3dec::gemv_kernel|gemv_kernel|"
                          r"skinny|exl3dec_gemv"),
]
FAM_RE = [(n, re.compile(p)) for n, p in FAMS]
CATCH = "norms/mHC"          # first-match-wins fallthrough: elementwise, copies, gather
def fam(name):
    base = name.split("(")[0].replace("void ", "").strip()
    for n, r in FAM_RE:
        if r.search(base): return n
    return CATCH

# ---------------------------------------------------------------- bytes: key -> family
# Weight bytes are attributed to the family that OWNS the module, but the trace can only tell
# gemv kernels apart by kernel name, not by layer: the KDA and MLA projection gemvs are the same
# exl3_gemv_kernel instantiation as the shared-expert one. So the main table keeps every
# non-KDA/MLA-specific gemv in "dense/shared gemv" (bytes included) and prints the projection
# split separately, from the grid sizes.
KDA_ATTN = r"self_attn\.(qkv_proj|o_proj|z_proj|qkvz_proj|ba_proj|b_proj|a_proj|f_a_proj|f_b_proj|g_a_proj|g_b_proj|conv1d|o_norm|A_log|dt_bias|short_conv)"
MLA_ATTN = r"self_attn\.(q_a_proj|q_b_proj|q_proj|kv_a_proj_with_mqa|kv_b_proj|o_proj|indexer|q_a_layernorm|kv_a_layernorm)"
DENSE_MLP = r"\.mlp\.(gate_proj|up_proj|down_proj)$"
SHARED = r"\.mlp\.shared_experts"
ROUTER = r"\.mlp\.gate$|routing_gate|\.gate\.e_score"
def layer_of(key):
    m = re.search(r"layers\.(\d+)\.", key)
    return int(m.group(1)) if m else None

# Which attention a layer has, read off its sibling keys. Decided BEFORE the per-key match:
# `o_proj` exists in both KDA and MLA, and whichever regex is tried first would claim the
# other's 33.5 MB per layer (402 MB/round across 12 MLA layers -- enough to push the KDA
# row's floor above its own measurement).
def attn_types(inv):
    t = {}
    for k in inv:
        lay = layer_of(k)
        if lay is None or ".self_attn." not in k: continue
        if "kv_b_proj" in k or "kv_a_proj_with_mqa" in k: t[lay] = "mla"
        elif "qkv_proj" in k or "conv1d" in k or "f_a_proj" in k: t.setdefault(lay, "kda")
    return t
ATTN = {}

def key_class(key, lay):
    """(family, role) for one inventory key. lay = layer index or None (non-layer key)."""
    if ".mlp.shared_experts" in key: return "dense/shared gemv", "shared_expert"
    if re.search(DENSE_MLP, key): return "dense/shared gemv", "dense_mlp"
    if re.search(ROUTER, key): return "MoE expert gemv", "router"
    if ".self_attn." in key:
        # o_proj deliberately stays in dense/shared gemv: KDA's and MLA's o_proj are the same
        # shape class, so their launches sit in one unsplittable 140-launch gemv signature. The
        # bytes follow the time there rather than the layer, which keeps the two halves of the
        # table consistent signature by signature.
        if key.endswith("o_proj"):
            return "dense/shared gemv", "o_proj"
        at = ATTN.get(lay) or ATTN.get(("mtp", lay))
        if at == "mla": return ("DSA/MLA attn", "mla_proj")
        if at == "kda": return ("KDA", "kda_proj")
    if re.search(r"layernorm|\.norm$|o_norm|hc_", key) or ".norm" in key: return "norms/mHC", "norm"
    if "embed_tokens" in key: return None, "embed"
    if "lm_head" in key: return "dense/shared gemv", "lm_head"
    return None, "other"

# ---------------------------------------------------------------- main
def main():
    J = json.load(open(A.json))
    kfs = glob.glob(f"{A.dir}/**/*kernel_trace.csv", recursive=True)
    MARKRE = re.compile(r"FillFunctor<c10::complex")
    kern = []; marks = []
    for f in kfs:
        for r in csv.DictReader(open(f)):
            s = int(r["Start_Timestamp"]); e = int(r["End_Timestamp"])
            n = r["Kernel_Name"]; g = int(r.get("Grid_Size_X") or 0)
            (marks if MARKRE.search(n) else kern).append((s, e, n, g))
    kern.sort(); marks.sort()
    KS = [k[0] for k in kern]
    def in_round(rs, re_):
        # kernels overlapping [rs, re_) via bisect (the trace holds all arms)
        i = max(0, bisect.bisect_left(KS, rs) - 64)
        j = bisect.bisect_left(KS, re_)
        return [k for k in kern[i:j] if k[1] > rs]
    def sub(sp, rs, re_):
        return [(a, b) for a, b in sp if b > rs and a < re_]
    GRID = 256
    EV = J["events"]; want = [x[0] for x in EV]
    raw = [m[3] for m in marks]
    # greedy alignment -- the trace also holds unrelated complex64 fills (grid capped
    # at 96*256 for big buffers, e.g. during the per-arm prefills), so skip any mark that is not
    # the next expected id.
    mk = []; j = 0
    for m in marks:
        if j < len(EV) and m[3] // GRID == want[j]:
            mk.append((m[0], m[1], want[j])); j += 1
    ok = j == len(EV)
    print(f"marker id sequence match: {ok} ({len(EV)} events, {len(marks) - j} unrelated fills skipped)")
    if not ok: print("  per-phase split NOT trustworthy"); return 1
    spans = []; st = []
    for i, ev in enumerate(EV):
        if ev[2] == "in": st.append((ev[1], i))
        else:
            if not st: break
            lab, j = st.pop(); spans.append((lab, mk[j][0], mk[i][1]))
    rounds = [(s, e) for lab, s, e in spans if lab == "round"]
    if A.arm:
        arm = next((s, e) for lab, s, e in spans if lab == A.arm)
        rounds = [(s, e) for s, e in rounds if arm[0] <= s and e <= arm[1]]
        aj = next(a for a in J["arms"] if a["label"] == A.arm)
        J.update({k: aj[k] for k in ("uniq_experts", "round_ms", "accept", "tok_per_round")})
    if A.last and len(rounds) > A.last:
        # the last N rounds only: the trace covers the whole 128-token job, and the first
        # rounds still carry warm-up (autotune caches filling, first-touch page faults)
        rounds = rounds[-A.last:]
    nr = len(rounds)
    if not nr: print("no round spans"); return 1
    mtp_spans = [(s, e) for lab, s, e in spans if lab == "mtp_gen"]
    ver_spans = [(s, e) for lab, s, e in spans if lab.startswith("verify_fwd")]

    # ---- measured: union of kernel intervals per family inside the round windows ----
    T = collections.defaultdict(float)   # ms per round
    C = collections.defaultdict(int)     # launches per round
    RAW = collections.defaultdict(float) # raw sum (not union), for the overlap check
    grids = collections.defaultdict(lambda: collections.Counter())   # family -> grid -> count
    gt = collections.defaultdict(lambda: collections.Counter())      # family -> sig -> ns
    for rs, re_ in rounds:
        msp, vsp = sub(mtp_spans, rs, re_), sub(ver_spans, rs, re_)
        for s, e, n, g in in_round(rs, re_):
            if not (e > rs and s < re_): continue
            dt = min(e, re_) - max(s, rs)
            in_mtp = any(a < s < b for a, b in msp)
            in_ver = any(a < s < b for a, b in vsp)
            if in_mtp: f = "MTP draft"
            elif in_ver: f = fam(n)
            else: f = "sampling/accept"          # sample_accept minus verify minus draft
            T[f] += dt / 1e6; C[f] += 1; RAW[f] += dt / 1e6
            sig = (n.split("(")[0].replace("void ", "")[:120], g)
            grids[f][sig] += 1; gt[f][sig] += dt
    wall = sum(e - s for s, e in rounds) / 1e6 / nr
    busy = sum(T.values()) / nr
    idle = wall - busy
    T["host idle"] += idle * nr

    # ---- gemv launches -> which layer type? ------------------------------------------------
    # The trace cannot name a layer, but a decode gemv runs once per layer, so the per-round
    # LAUNCH COUNT identifies the layer type: 34 = the 34 KDA layers, 12 = the 11 DSA/MLA layers
    # plus the MTP layer's, 3 = the dense-MLP layers. Signatures whose count matches nothing in
    # that set stay in dense/shared gemv (shared experts, lm_head, mixed). Printed for audit.
    inv0, inv_mtp0 = J.get("inv", {}), J.get("inv_mtp", {})
    n_kda = sum(1 for k in inv0 if k.endswith(".self_attn.qkv_proj")) or 34
    n_mla = (sum(1 for k in list(inv0) + list(inv_mtp0)
                 if re.search(MLA_ATTN, k) and "kv_b_proj" in k)) or 12
    n_dmlp = sum(1 for k in inv0 if re.search(DENSE_MLP, k) and "down_proj" in k) or 3
    LAYERS = (("KDA", n_kda), ("DSA/MLA attn", n_mla), ("dense/shared gemv", n_dmlp))
    GEMV = re.compile(r"had_hf_r|had_ff_r|exl3_gemv_kernel|exl3dec::gemv_kernel|gemv_kernel|skinny")
    tsig = collections.Counter(); csig = collections.Counter()
    for rs, re_ in rounds:
        msp, vsp = sub(mtp_spans, rs, re_), sub(ver_spans, rs, re_)
        for s, e, nm, g in in_round(rs, re_):
            if not (e > rs and s < re_): continue
            if any(a < s < b for a, b in msp): continue
            if not any(a < s < b for a, b in vsp): continue
            b = nm.split("(")[0].replace("void ", "")
            if not GEMV.search(b): continue
            tsig[(b[:120], g)] += min(e, re_) - max(s, rs); csig[(b[:120], g)] += 1
    moved = []; moved_us = 0.0
    for sig, t in tsig.items():
        c = csig[sig] / nr
        tgt = next((lab for lab, nl in LAYERS if abs(c - nl) <= 1.5), None)
        if not tgt or tgt == "dense/shared gemv": continue
        T[tgt] += t / 1e6; T["dense/shared gemv"] -= t / 1e6
        C[tgt] += csig[sig]; C["dense/shared gemv"] -= csig[sig]
        moved.append((sig[0], sig[1], round(c, 2), tgt, round(t / 1e6 / nr, 3)))
        moved_us += t / 1e6 / nr
    busy = sum(v for k, v in T.items() if k != "host idle") / nr
    idle = wall - busy
    T["host idle"] = idle * nr

    # ---- bytes: inventory x read multiplicity ----
    inv = J.get("inv", {}); inv_mtp = J.get("inv_mtp", {})
    ATTN.update(attn_types(inv)); ATTN.update({("mtp", k): v for k, v in [] })
    # the MTP head is its own layer 45; tag it so its MLA is not read as KDA
    for k in inv_mtp:
        if ".self_attn." in k and ("kv_b_proj" in k or "kv_a_proj_with_mqa" in k):
            ATTN[("mtp", layer_of(k))] = "mla"
        elif ".self_attn." in k and ("qkv_proj" in k or "conv1d" in k):
            ATTN.setdefault(("mtp", layer_of(k)), "kda")
    slot = J.get("moe_expert_slot_b", {})
    uniq = J.get("uniq_experts", {})
    ucalls = J.get("union_calls", {})
    # distinct routed experts per union call, trunk verify vs MTP draft, by rows. The phase
    # label seen inside the draft forward is draft_fwd_R*, not mtp_gen (the inner wrap wins).
    def umean(pred):
        v = [u["mean"] * u["n"] for k, u in uniq.items() if pred(k)]
        n = [u["n"] for k, u in uniq.items() if pred(k)]
        return (sum(v) / sum(n) if sum(n) else 0.0, sum(n))
    TRUNK = lambda k: k.startswith("verify_fwd") or k.startswith("round") or k.startswith("catchup")
    DRAFT = lambda k: k.startswith("draft_") or k.startswith("mtp_gen")
    u_trunk, n_trunk = umean(TRUNK)
    u_draft, n_draft = umean(DRAFT)
    rows = [k for k in inv if k.endswith(".mlp") and k in slot]
    moe_fwd = len(rows)                       # MoE forward passes per verify (42 trunk layers)
    moe_fwd_d = len([k for k in slot if k.startswith("mtp:")])
    B = collections.Counter()
    for k, e in inv.items():
        f, role = key_class(k, layer_of(k))
        if f is None: continue
        B[(f, role)] += e["b"]
    # MoE layer keys: each verify MoE layer runs once; the MTP MoE layer runs once per draft
    moe_keys = [k for k in inv if k.endswith(".mlp") and k in slot]
    slot_t = [v for k, v in slot.items() if not k.startswith("mtp:")]
    slot_d = [v for k, v in slot.items() if k.startswith("mtp:")]
    expert_slot = (sum(slot_t) / len(slot_t)) if slot_t else 0.0
    expert_slot_d = (sum(slot_d) / len(slot_d)) if slot_d else expert_slot
    # --- multiplicity per round: one verify forward + one draft forward -------------------
    # Every trunk weight is read once (all 45 layers run in the single R=2 verify forward);
    # the MTP layer's weights once (one draft forward); lm_head twice (verify logits +
    # draft logits). The routed experts are read once per DISTINCT pick, not rows x topk.
    BY = collections.Counter(); WHY = collections.defaultdict(list)
    for k, e in inv.items():
        f, role = key_class(k, layer_of(k))
        if f is None: continue
        BY[f] += e["b"]
        WHY[f].append((role, e["b"], k))
    head_b = sum(e["b"] for k, e in inv.items() if key_class(k, layer_of(k))[1] == "lm_head")
    for k, e in inv_mtp.items():
        f, role = key_class(k, layer_of(k))
        if f is None: continue
        if role == "lm_head":
            BY["MTP draft"] += e["b"]; WHY["MTP draft"].append(("lm_head_draft", e["b"], k))
            continue
        BY["MTP draft"] += e["b"]; WHY["MTP draft"].append((role, e["b"], k))
    if head_b:
        # The MTP head is SHARED with the trunk (glm5_next_mtp.py: self.target_lm_head), so it is
        # not a second inventory entry -- but it is read twice per round, once per logits call.
        # The trace shows it: one gemv at grid_x=77568 inside verify_fwd and the same grid inside
        # mtp_gen (see the per-family signature lists). Count the second read with the MTP row.
        BY["MTP draft"] += head_b
        WHY["MTP draft"].append(("lm_head_shared", head_b, "lm_head re-read for draft logits"))
    routed_trunk = expert_slot * u_trunk * moe_fwd
    routed_draft = expert_slot_d * u_draft * moe_fwd_d
    if routed_trunk:
        BY["MoE expert gemv"] += routed_trunk
        WHY["MoE expert gemv"].append((f"routed_x{u_trunk:.1f}x{moe_fwd}", routed_trunk, "-"))
    if routed_draft:
        BY["MTP draft"] += routed_draft
        WHY["MTP draft"].append((f"routed_x{u_draft:.1f}", routed_draft, "-"))
    st_head = J.get("kda_state_b")
    state_b = 0
    if st_head and n_kda:
        rws_v = max((int(k.split("|R")[1]) for k in uniq if k.split("|R")[1].isdigit()), default=2)
        b = st_head * n_kda * (1 + rws_v)
        BY["KDA"] += b; state_b = b
        WHY["KDA"].append((f"state_x{n_kda}x{1 + rws_v}", b, "gdn.cu:722-736"))
    kv = J.get("mla_latent_b", {})
    if kv:
        rws = max((int(k.split("|R")[1]) for k in uniq if k.split("|R")[1].isdigit()), default=2)
        nl = kv.get("n_layers", 11)
        b = (kv.get("absorb", 0) + kv.get("indexer", 0)) * nl * rws
        BY["DSA/MLA attn"] += b
        WHY["DSA/MLA attn"].append((f"cache_x{nl}x{rws}", b, "absorb+indexer"))

    out_rows = []
    names = [n for n, _ in FAMS] + ["MTP draft", "sampling/accept", "host idle"]
    tot_gap = 0.0; floors = {}
    for f in names:
        b = BY[f]
        for bw in (A.bw, A.bw2):
            floors.setdefault(f, {})[bw] = b / (bw * 1e9) * 1e6
        tot_gap += T[f] / nr * 1e3 - b / (A.bw * 1e9) * 1e6
    for f in sorted(names, key=lambda f: -(T[f] / nr * 1e3 - BY[f] / (A.bw * 1e9) * 1e6)):
        m = T[f] / nr * 1e3; b = BY[f]
        fl = b / (A.bw * 1e9) * 1e6; g = m - fl
        # the KDA recurrent state is 4 MiB per layer per slot (136 MiB over 34 layers), which
        # does not come from DRAM every round at 34 layers x (1 read + R writes): the measured
        # row is FASTER than the modelled floor, so the state term is reported separately.
        flx = (b - (state_b if f == "KDA" else 0)) / (A.bw * 1e9) * 1e6
        out_rows.append(dict(family=f, calls=round(C[f] / nr, 1), us=round(m, 1),
                             bytes=b, floor_us=round(fl, 1), gap_us=round(g, 1),
                             floor_excl_cache=round(flx, 1),
                             gap_excl_cache=round(m - flx, 1),
                             achieved_gbs=round(b / (m * 1e-6) / 1e9, 1) if m > 0 and b else 0.0,
                             pct=round(100 * g / tot_gap, 1) if tot_gap else 0.0,
                             floor230=round(b / (A.bw2 * 1e9) * 1e6, 1)))
    # ---- report ----
    print(f"\n== {nr} rounds | round wall {wall:.2f} ms | GPU busy {busy:.2f} | idle {idle:.2f} | "
          f"unprofiled host round {J['round_ms']:.2f} ms (accept {J['accept']:.3f})")
    print(f"== distinct routed experts/union: trunk verify {u_trunk:.2f} (n={n_trunk}), "
          f"MTP draft {u_draft:.2f} (n={n_draft}); expert slot {expert_slot/1e6:.2f} MB "
          f"(mtp {expert_slot_d/1e6:.2f} MB); MoE forwards {moe_fwd} trunk + {moe_fwd_d} mtp")
    if slot:
        allb = sum(e["b"] for e in inv.values()) + sum(e["b"] for e in inv_mtp.values())
        print(f"== inventory: {allb/1e9:.1f} GB of weights "
              f"({sum(e['b'] for e in inv.values())/1e9:.1f} GB trunk + "
              f"{sum(e['b'] for e in inv_mtp.values())/1e9:.1f} GB mtp, excluding routed experts); "
              f"routed experts {sum(slot_t)*288/1e9:.1f} GB trunk + {sum(slot_d)*288/1e9:.1f} GB mtp")
    print(f"\n| family | calls/round | measured us/round | bytes read/round | "
          f"byte floor us @{A.bw:.0f} GB/s | gap us | % of gap | achieved GB/s | "
          f"floor us @{A.bw2:.0f} GB/s |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in out_rows:
        print(f"| {r['family']} | {r['calls']:.0f} | {r['us']:.1f} | {r['bytes']/1e6:.1f} MB | "
              f"{r['floor_us']:.1f} | {r['gap_us']:.1f} | {r['pct']:.1f} | "
              f"{r['achieved_gbs'] if r['achieved_gbs'] else '-'} | {r['floor230']:.1f} |")
    print(f"| **sum** | {sum(r['calls'] for r in out_rows):.0f} | "
          f"{sum(r['us'] for r in out_rows):.1f} | {sum(r['bytes'] for r in out_rows)/1e6:.1f} MB | "
          f"{sum(r['floor_us'] for r in out_rows):.1f} | {sum(r['gap_us'] for r in out_rows):.1f} | 100 | "
          f"- | {sum(r['floor230'] for r in out_rows):.1f} |")
    kr = next((r for r in out_rows if r["family"] == "KDA"), None)
    if kr and state_b:
        print(f"\nKDA floor is an OVER-estimate: the recurrent state term is {state_b/1e6:.0f} MB "
              f"(4 MiB x {n_kda} layers x (1 read + R writes)) and the row measures FASTER than "
              f"its own floor, so the state cannot be coming from DRAM at that multiplicity. "
              f"Excluding it: floor {kr['floor_excl_cache']:.1f} us, gap {kr['gap_excl_cache']:.1f} us "
              f"({kr['achieved_gbs']:.0f} GB/s on the weights).")
    print(f"\nsum of floors {sum(r['floor_us'] for r in out_rows)/1e3:.2f} ms vs round wall "
          f"{wall:.2f} ms -> the round could be {wall - sum(r['floor_us'] for r in out_rows)/1e3:.2f} ms "
          f"shorter if every family ran at {A.bw:.0f} GB/s")
    print(f"raw kernel sum {sum(RAW.values())/nr:.2f} vs union busy {busy:.2f} ms/round -> "
          f"{100*(sum(RAW.values())/nr - busy)/busy:.2f} % concurrent overlap")
    print("\n-- per-family kernel signatures: calls/round, ms/round, grid_x, name --")
    for f in names:
        if not grids[f]: continue
        print(f"  {f}  ({T[f]/nr*1e3:.0f} us/round, {C[f]/nr:.0f} launches):")
        for sig, c in grids[f].most_common(24):
            print(f"    {c/nr:8.2f}/rd {gt[f][sig]/1e6/nr:8.3f} ms  grid_x={sig[1]:<8d} {sig[0]}")
    if moved:
        print(f"\n-- gemv signatures re-attributed by launch count "
              f"(KDA={n_kda}, MLA={n_mla}, dense-MLP={n_dmlp}), {moved_us:.2f} ms/round --")
        for n, g, c, tgt, t in sorted(moved, key=lambda x: -x[4]):
            print(f"    {c:6.2f}/round {t:8.3f} ms  grid_x={g:<8d} -> {tgt:16s} {n}")
    gemv_ms = sum(v for k, v in T.items() if k in ("dense/shared gemv", "KDA", "DSA/MLA attn")) / nr
    gemv_n = sum(C[k] for k in ("dense/shared gemv", "KDA", "DSA/MLA attn")) / nr
    print(f"\n-- every gemv signature, time and calls (the gemv class is {gemv_ms:.2f} ms/round "
          f"over {gemv_n:.0f} launches) --")
    for (n, g), t in sorted(tsig.items(), key=lambda x: -x[1]):
        print(f"    {csig[(n,g)]/nr:6.2f}/round {t/1e6/nr:8.3f} ms  grid_x={g:<8d} {n}")
    print("\n-- byte model detail (role, bytes, key) --")
    for f in names:
        agg = collections.Counter()
        for role, b, k in WHY[f]: agg[role] += b
        if agg:
            print(f"  {f} ({sum(agg.values())/1e6:.1f} MB): " +
                  ", ".join(f"{r} {v/1e6:.1f}" for r, v in agg.most_common()))
    if A.out:
        json.dump({"rows": out_rows, "wall_ms": wall, "busy_ms": busy, "idle_ms": idle,
                   "uniq_trunk": u_trunk, "uniq_draft": u_draft, "expert_slot_b": expert_slot,
                   "moved_ms": moved_us, "n_kda": n_kda, "n_mla": n_mla, "n_dmlp": n_dmlp,
                   "bw": A.bw, "bw2": A.bw2, "rounds": nr, "arm": A.arm,
                   "sigs": [dict(family=f, name=sig[0], grid=sig[1], calls=c / nr,
                                 ms=gt[f][sig] / 1e6 / nr) for f in names for sig, c in grids[f].items()]}, open(A.out, "w"), indent=1)
    return 0

sys.exit(main())
