# D1-1 rounddecomp analyzer: slice a rocprofv3 trace of a served MTP round (ndt 1, R=2 verify)
# into phases (from the marker kernels written by round_decomp.py), split kernel time per family,
# and attribute every millisecond of GPU idle inside the round.
#
#   usage: round_decomp_table.py <run.rocprof dir> <round_decomp.json>
#
# Markers: round_decomp.py enqueues one marker kernel per phase edge, in a fixed host order, so
# the Nth marker on the GPU timeline is the Nth marker it enqueued. The marker is a complex64
# zero-fill -- a signature no EXL3 kernel uses -- sized so Grid_Size_X == 256 * marker_id, and
# every id is checked against the trace before any phase number is trusted.
import csv, glob, json, os, re, sys, collections, bisect

PEAK, BW = 47e12, 256e9
FAMILIES = [  # first match wins, same list as tools/glm/prof_families.py (decode subset)
    ("MoE routed",      r"^mpw_gemm|mpw2_gemm|moe_gu|moe_down"),
    ("MoE aux/router",  r"^mpw_|router_kernel|gatherTopK|sigmoid_kernel"),
    ("dense GEMM",      r"^Cijk_|skinny|gemv_kernel|had_hf_r|had_ff_r"),
    ("dense gemv exl3", r"^exl3_|gemv"),
    ("KDA",             r"kda|gla_|gated_delta|causal_conv1d|local_cumsum|l2norm"),
    ("MLA/DSA attn",    r"^_mla_|^_dsa_attn"),
    ("DSA indexer",     r"^_dsa_index|radixSort|rocprim|sbtopk|topk"),
    ("mHC mix/apply",   r"^hc_"),
    ("norms/elementwise", r"."),
]
FAM_RE = [(n, re.compile(p)) for n, p in FAMILIES]

def _base(name):
    """rocprofv3 records e.g. "void hc_mix_norm_1pass_kernel<4, 24, __half>(...)" or
    "exl3dec::router_kernel<false>": drop "void " and leading namespaces so ^-anchors fire."""
    name = name.strip()
    if name.startswith("void "): name = name[5:]
    head = name.split("<", 1)[0].split("(", 1)[0]
    if "::" in head: name = name[len(head.rsplit("::", 1)[0]) + 2:]
    return name

def fam(name):
    name = _base(name)
    for n, r in FAM_RE:
        if r.search(name): return n
    return "other"

def main():
    d, jf = sys.argv[1], sys.argv[2]
    J = json.load(open(jf))
    kf = glob.glob(f"{d}/**/*kernel_trace.csv", recursive=True)
    cf = glob.glob(f"{d}/**/*csv", recursive=True)
    mcf = [f for f in cf if "memory_copy" in f]
    apf = [f for f in cf if "api" in os.path.basename(f).lower()]
    ops = []
    for f in kf:
        for r in csv.DictReader(open(f)):
            ops.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"],
                        int(r.get("Grid_Size_X") or 0)))
    ops.sort()
    MARKRE = re.compile(r"FillFunctor<c10::complex")
    marks = [(s, e, int(g)) for s, e, n, g in ops if MARKRE.search(n)]
    kern = [(s, e, n) for s, e, n, g in ops if not MARKRE.search(n)]
    API = []
    for f in apf:
        for r in csv.DictReader(open(f)):
            nm = r.get("Function") or r.get("Name") or ""
            if nm.startswith(("hipLaunchKernel", "hipModuleLaunchKernel", "hipMemcpy",
                              "hipStreamSynchronize", "hipDeviceSynchronize",
                              "hipEventSynchronize")):
                API.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), nm))
    API.sort()
    LCH = [a[0] for a in API if "LaunchKernel" in a[2]]
    print(f"trace {d}: {len(kern)} kernels, {len(marks)} markers, memcpy "
          f"{[os.path.basename(f) for f in mcf]}, host api {len(API)} calls "
          f"({len(LCH)} hipLaunchKernel)" if apf else
          f"trace {d}: {len(kern)} kernels, {len(marks)} markers, memcpy "
          f"{[os.path.basename(f) for f in mcf]}, NO host api trace")

    # grid = 256 * marker_id (1024-id complex64 fill, 256 threads/block, 4-wide vectorization)
    GRID = 256
    raw = [g for _, _, g in marks]
    EV = J["events"]                      # [(marker_id, label, "in"/"out")] in enqueue order
    n_ev = len(EV)
    want = [e[0] for e in EV]
    # the buffer's own init fill (id NID) is traced too; drop leading trace markers until aligned
    off = next((i for i, g in enumerate(raw) if g // GRID == want[0]), 0) if want else 0
    got = [g // GRID for g in raw[off:off + n_ev]]
    ok = got == want
    print(f"markers: trace {len(raw)} (grid=256*id), json {n_ev} events, leading init fills skipped {off}")
    print(f"marker id sequence match: {ok}")
    if not ok:
        bad = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), None)
        print(f"  MISMATCH at index {bad}: trace {got[max(0,bad-3):bad+3]} vs json {want[max(0,bad-3):bad+3]}")
        print("  per-phase split NOT trustworthy; only the round totals below are")
    else:
        print("  marker ids verified against trace grid sizes -> phase mapping exact")
    mk = [(s, e, g) for s, e, g in marks][off:off + n_ev]
    marks = [(s, e, g // GRID) for s, e, g in mk]
    marks_all = marks                      # un-realigned copy, for the marker-overhead self-check

    # phase spans: markers nest (round > mtp_gen > ...), so pair in/out with a stack.
    spans = []          # (label, t_start, t_end)  in GPU ns
    st = []
    for i, ev in enumerate(EV):
        if ev[2] == "in":
            st.append((ev[1], i))
        else:
            if not st:
                print(f"  unmatched 'out' at event {i} ({ev}) -- stopping span list")
                break
            lab, j = st.pop()
            spans.append((lab, marks[j][0], marks[i][1]))   # entry marker start .. exit marker end
    if st:
        print(f"  {len(st)} phases never closed (e.g. {st[0]}) -- their spans dropped")
    spans.sort(key=lambda x: x[1])   # chronological: a parent follows its children
    # round windows
    rounds = [(s, e) for lab, s, e in spans if lab == "round"]
    nr = len(rounds)
    if not nr:
        print("no round spans found"); return
    def clip(s, e):
        """kernels overlapping [s,e), clipped to it, in start order"""
        return sorted((max(a, s), min(b, e), n) for a, b, n in kern if b > s and a < e)
    def union(inside):
        """total covered time of a sorted, clipped kernel list"""
        busy, cs, ce = 0.0, None, None
        for ks, ke, _ in inside:
            if cs is None: cs, ce = ks, ke
            elif ks <= ce: ce = max(ce, ke)
            else: busy += ce - cs; cs, ce = ks, ke
        return busy + (ce - cs if cs is not None else 0.0)

    INS = {sp: clip(sp[1], sp[2]) for sp in spans}   # spans are tuples, hashable
    def stalls(phase_spans):
        """(gap, t_idle_start, t_idle_end, prev_family, next_family, phase) for each stall."""
        for sp in phase_spans:
            for a, b in zip(INS[sp], INS[sp][1:]):
                if b[0] > a[1]:
                    yield (b[0] - a[1], a[1], b[0], fam(a[2]), fam(b[2]), sp[0])
    def launched_early(t0, t1):
        """True if the host had already issued `t1`'s kernel when the GPU idled at t0."""
        i = bisect.bisect_left(LCH, t1)
        return i > 0 and LCH[i - 1] <= t0
    BUSY = collections.defaultdict(float)   # union busy ms per phase
    IDLE = collections.defaultdict(float)   # idle ms per phase (wall - union of kernel time inside)
    GAPAFTER = collections.defaultdict(float)
    for sp in spans:
        lab, s, e = sp
        inside = INS[sp]
        b = union(inside)
        BUSY[lab] += b / 1e6
        IDLE[lab] += ((e - s) - b) / 1e6
        for a, nxt in zip(inside, inside[1:]):
            if nxt[0] > a[1]: GAPAFTER[f"{lab}:{fam(a[2])}"] += (nxt[0] - a[1]) / 1e6

    # D2H copies inside the round windows (from the memory-copy trace)
    d2h = collections.Counter()
    d2h_bytes = 0
    for f in mcf:
        for r in csv.DictReader(open(f)):
            if "DEVICE_TO_HOST" not in (r.get("Direction") or ""): continue
            s = int(r["Start_Timestamp"])
            if any(rs <= s <= re_ for rs, re_ in rounds):
                d2h["copies"] += 1
                d2h_bytes += int(r["End_Timestamp"]) - s

    W = sum(e - s for s, e in rounds) / nr / 1e6      # ms per round
    # families: counted inside the TOP-LEVEL round spans only (child phases are nested inside it)
    top = [sp for sp in spans if sp[0] == "round"]
    tot_fam = collections.defaultdict(float)
    tot_n = collections.defaultdict(int)
    for lab, s, e in top:
        for ks, ke, n in kern:
            if ke > s and ks < e:
                tot_fam[fam(n)] += (min(ke, e) - max(ks, s)) / 1e6
                tot_n[fam(n)] += 1
    tot_idle = sum(IDLE[lab] for lab, _, _ in top) / nr
    tot_wall = sum((e - s) for _, s, e in top) / 1e6 / nr
    tot_busy = sum(BUSY[lab] for lab, _, _ in top) / nr

    print(f"\n== {nr} rounds, round wall {W:.2f} ms/round (marker-to-marker)")
    print(f"host (unprofiled arm, same command): round_ms {J['round_ms']:.2f}, accept {J['accept']:.3f}, "
          f"rounds {J['rounds']}, tok/round {J['tok_per_round']:.3f}")
    print("\n-- GPU busy inside the ROUND window, by family (ms/round) --")
    for k, v in sorted(tot_fam.items(), key=lambda x: -x[1]):
        print(f"{v/nr:8.3f} ms/round  {100*v/tot_wall/nr:5.1f}%  {tot_n[k]/nr:7.1f} launches/round  {k}")
    print(f"{tot_busy/nr:8.3f} ms/round GPU busy (union of kernel intervals inside round)")
    print(f"{tot_idle/nr:8.3f} ms/round GPU idle inside the round window")
    print(f"{tot_wall:8.3f} ms/round round wall")
    print("\n-- phases (child phases are NESTED in the parent, their wall overlaps) --")
    for lab in sorted(BUSY, key=lambda l: -sum((e - s) for l2, s, e in spans if l2 == l)):
        w = sum((e - s) for l2, s, e in spans if l2 == lab) / 1e6 / nr
        print(f"{lab:20s} wall {w:7.3f}  busy {BUSY[lab]/nr:7.3f}  idle {IDLE[lab]/nr:7.3f}  "
              f"host {J['host_ms_per_round'].get(lab, float('nan')):7.3f}  "
              f"calls {J['host_calls_per_round'].get(lab, float('nan')):5.2f}")
    print("\n-- idle inside the round window, by preceding kernel family (ms/round) --")
    for k, v in sorted(GAPAFTER.items(), key=lambda x: -x[1]):
        if not k.startswith("round:"): continue
        print(f"{v/nr:8.3f}  {k[6:]}")

    # individual gaps: where exactly does the idle sit, and how big is each stall
    top_spans = [sp for sp in spans if sp[0] == "round"]
    gaps = sorted(stalls(top_spans), reverse=True)
    gb = [g for g, *_ in gaps]
    print(f"\n-- the {len(gb)} GPU stalls inside round windows: {len(gb)/nr:.0f} per round --")
    if gb:
        q = lambda p: gb[min(len(gb) - 1, int(len(gb) * p))] / 1e6
        print(f"   stall size ms: median {q(.5):.3f}  p90 {q(.9):.3f}  p99 {q(.99):.3f}  max {gb[0]/1e6:.3f}")
        print(f"   stalls > 100 us: {sum(1 for g in gb if g > 1e5)/nr:.1f} per round, "
              f"carrying {sum(g for g in gb if g > 1e5)/nr/1e6:.3f} ms/round of the idle")
        print(f"   stalls <= 100 us: {sum(1 for g in gb if g <= 1e5)/nr:.1f} per round, "
              f"{sum(g for g in gb if g <= 1e5)/nr/1e6:.3f} ms/round")
        print("   ten largest stalls (ms, prev family -> next family):")
        seen = set()
        for g, _t, _u, fa, fb, _ph in gaps[:400]:
            key = (round(g / 1e3), fa, fb)
            if key in seen: continue
            seen.add(key)
            print(f"     {g/1e6:7.3f}  {fa:22s} -> {fb}")
            if len(seen) >= 10: break

    # ---- idle attribution: was the next kernel already enqueued when the GPU went idle? ----
    # A stall [t_end(prev), t_start(next)] is HOST-BOUND (dispatch starvation) if the host had
    # already issued the launch for `next` before the GPU reached t_end(prev) -- i.e. the queue
    # was non-empty and the gap is GPU-side (tail/dependency), not a submit-latency artifact.
    # It is HOST-SUBMIT if the launch for `next` happens after t_end(prev).
    if LCH and gaps:
        host_bound = sum(g for g, t0, t1, *_ in gaps if launched_early(t0, t1))
        submit_bound = sum(gb) - host_bound
        hb_n = sum(1 for g, t0, t1, *_ in gaps if launched_early(t0, t1))
        sb_n = len(gb) - hb_n
        tot_id = host_bound + submit_bound
        print(f"\n-- idle attribution over the {tot_id/nr/1e6:.2f} ms/round of stalls in round windows "
              f"({len(gb)/nr:.0f} stalls/round) --")
        print(f"   HOST-SUBMIT (the launch for the next kernel was issued after the GPU went idle,")
        print(f"     i.e. the GPU starved waiting for the host): {submit_bound/nr/1e6:6.3f} ms/round "
              f"in {sb_n/nr:6.1f} stalls")
        print(f"   GPU-SIDE   (the next kernel was already queued: tail/dependency/occupancy): "
              f"{host_bound/nr/1e6:6.3f} ms/round in {hb_n/nr:6.1f} stalls")
        print(f"   host API time inside round windows: "
              f"{sum(min(a[1],e)-max(a[0],s) for a in API for _,s,e in top if a[1]>s and a[0]<e)/nr/1e6:6.3f}"
              f" ms/round (of {len(LCH)} launches total)")
        print("\n-- host-attributable idle PER PHASE (ms/round) --")
        print("   NOTE: phases NEST (round > sample_accept > verify_fwd_R2, round > mtp_gen > draft_*),")
        print("   so the rows below OVERLAP and must not be summed. Only the 'round' row is the")
        print("   whole-round figure; it is the number to compare against the gate.")
        ph_hb = collections.defaultdict(float); ph_sb = collections.defaultdict(float)
        ph_hn = collections.defaultdict(int); ph_sn = collections.defaultdict(int)
        for g, t0, t1, _fa, _fb, lab in stalls(spans):
            if launched_early(t0, t1):
                ph_hb[lab] += g; ph_hn[lab] += 1
            else:
                ph_sb[lab] += g; ph_sn[lab] += 1
        print(f"   {'phase':20s} {'host-submit':>12s} {'n':>7s} {'gpu-side':>10s} {'n':>7s} {'total':>9s}")
        for lab in sorted(set(ph_hb) | set(ph_sb), key=lambda l: -(ph_hb[l] + ph_sb[l])):
            print(f"   {lab:20s} {ph_sb[lab]/nr/1e6:12.3f} {ph_sn[lab]/nr:7.1f} "
                  f"{ph_hb[lab]/nr/1e6:10.3f} {ph_hn[lab]/nr:7.1f} "
                  f"{(ph_sb[lab]+ph_hb[lab])/nr/1e6:9.3f}")
        hs_all = ph_sb.get("round", 0.0) / nr / 1e6
        gs_all = ph_hb.get("round", 0.0) / nr / 1e6
        # self-check: the instrumentation is not allowed to manufacture the idle it reports.
        # 43 marker kernels/round are real GPU work; the stall immediately before a marker is
        # partly the marker's own launch latency, so it is instrument overhead, not round idle.
        m_in = [(max(a, s), min(b, e)) for lab, s, e in spans for a, b, g in marks_all if b > s and a < e]
        m_starts = [a for a, b in m_in]
        m_before = m_n = 0
        for sp in top_spans:
            ins = INS[sp]
            for x, y in zip(ins, ins[1:]):
                if any(x[1] <= m < y[0] for m in m_starts):
                    m_before += y[0] - x[1]; m_n += 1
        m_busy = sum(b - a for a, b in m_in) / nr / 1e6
        m_stall = m_before / nr / 1e6
        print(f"   INSTRUMENTATION SELF-CHECK: {len(m_in)/nr:.1f} marker kernels/round, "
              f"{m_busy:.4f} ms/round of marker GPU time,")
        print(f"     {m_n/nr:.1f} stalls/round totalling {m_stall:.3f} ms/round end at a marker "
              f"-> that is marker launch latency, not round idle.")
        print(f"     Idle corrected for instrumentation: {hs_all+gs_all:.3f} - {m_busy:.3f} - {m_stall:.3f} "
              f"= {hs_all+gs_all-m_busy-m_stall:.3f} ms/round; host-submit {hs_all:.3f} ms/round.")
        print(f"   WHOLE ROUND: host-submit {hs_all:.3f} + gpu-side {gs_all:.3f} = "
              f"{hs_all+gs_all:.3f} ms/round of idle")
        print(f"   Host-idle gate (>= 3.0 ms/round of idle attributed to host sync or submit): "
              f"{'PASS' if hs_all >= 3.0 else 'FAIL'} at {hs_all:.3f} ms/round "
              f"(instrumentation-corrected: {hs_all-m_stall:.3f})")
    print(f"\n-- D2H copies inside round windows: {d2h['copies']/nr:.2f} per round "
          f"({d2h_bytes/nr/1e3:.1f} kB/round)")
    d2h_all = sum(1 for f in mcf for r in csv.DictReader(open(f))
                  if "DEVICE_TO_HOST" in (r.get("Direction") or ""))
    print(f"   NOTE: the memory-copy trace holds {d2h_all} D2H copies in the whole process and "
          f"{d2h['copies']} of them fall inside the decode rounds;\n"
          f"   the per-round .cpu() transfers in the decode loop are NOT emitted as MEMORY_COPY rows\n"
          f"   by rocprofv3 7.2.4, so the D2H sync cost below is HOST-SIDE (wall - GPU), not traced.")
    print("-- host d2h sync points per round (python-level counters) --")
    for k, v in sorted(J["syncs_per_round"].items(), key=lambda x: -x[1]):
        print(f"{v:8.3f}  {k}")
    print("\n-- host-lag check: host wall minus GPU wall per phase (ms/round) --")
    print("   (host > GPU wall = the host is blocked waiting for the GPU: a pipeline stall, not GPU idle)")
    for lab in sorted(BUSY, key=lambda l: -sum((e - s) for l2, s, e in spans if l2 == l)):
        w = sum((e - s) for l2, s, e in spans if l2 == lab) / 1e6 / nr
        h = J["host_ms_per_round"].get(lab)
        if h is None: continue
        print(f"{lab:20s} host {h:7.3f}  gpu {w:7.3f}  host-gpu {h-w:+7.3f}")
    SU = (tot_busy + tot_idle) / nr
    print(f"\nSUM CHECK: union busy {tot_busy/nr:.2f} + idle {tot_idle/nr:.2f} = {SU:.2f} "
          f"vs round wall {tot_wall:.2f} ms -> {100*abs(SU-tot_wall)/tot_wall:.2f} %")
    print("   (this check is definitional: idle is computed as wall - union-busy, so it is 0 % by "
          "construction.\n    The load-bearing checks are: family sum vs union busy, the round wall vs "
          "the unprofiled host round, and the marker-id match above.)")
    print(f"   family sum {sum(tot_fam.values())/nr:.2f} vs union busy {tot_busy/nr:.2f} ms/round "
          f"-> {100*(sum(tot_fam.values())-tot_busy)/tot_busy:.2f} % (concurrent-stream overlap)")
    print(f"round wall (marker-to-marker) {W:.2f} vs unprofiled host round {J['round_ms']:.2f} ms "
          f"-> profiler factor {W/J['round_ms']:.3f}")

if __name__ == "__main__":
    main()
