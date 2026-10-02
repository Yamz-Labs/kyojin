"""stall1 external 10 Hz sampler: GPU clocks/power/temp/busy + memory/vmstat + target-process CPU.

Usage: python tools/glm/stall_sampler.py <pid | pgrep pattern> <out.jsonl> [hz=10]
One JSON line per sample: t (epoch), sclk, mclk, fclk, socclk, W, C, busy, free/avail MB, rss MB,
utime/stime (ms, cumulative), and cumulative vmstat counters (diff offline). Exits when the process is gone.
"""
import glob, json, os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu_metrics_v3 as GMV

PAT, OUT = sys.argv[1], sys.argv[2]
HZ = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
DEV = glob.glob("/sys/class/drm/card*/device/gpu_busy_percent")[0].replace("gpu_busy_percent", "")
HW = glob.glob(DEV + "hwmon/hwmon*/")[0]
VMK = ("compact_stall", "compact_migrate_scanned", "compact_free_scanned", "pgmigrate_success", "pgsteal_kswapd",
       "pgsteal_direct", "pgscan_kswapd", "allocstall_normal", "allocstall_movable", "pswpin", "pswpout", "pgmajfault",
       "thp_fault_alloc", "thp_collapse_alloc", "workingset_refault_anon", "workingset_refault_file", "nr_free_pages")
TCK = os.sysconf("SC_CLK_TCK")

def rd(p):
    try:
        with open(p) as f: return f.read()
    except Exception: return ""
def cur(f):
    for l in rd(DEV + f).splitlines():
        if "*" in l: return int(l.split(":")[1].strip().rstrip("*").strip().lower().replace("mhz", ""))
    return None

pid = int(PAT) if PAT.isdigit() else None
for _ in range(0 if pid else 3600):
    r = subprocess.run(["pgrep", "-f", PAT], capture_output=True, text=True).stdout.split()
    r = [p for p in r if int(p) != os.getpid()]
    if r: pid = int(r[0]); break
    time.sleep(1)
if pid is None: sys.exit("no process")
with open(OUT, "a") as out:
    while os.path.exists(f"/proc/{pid}"):
        d = {"t": round(time.time(), 3), "sclk": cur("pp_dpm_sclk"), "mclk": cur("pp_dpm_mclk"), "fclk": cur("pp_dpm_fclk"),
             "socclk": cur("pp_dpm_socclk"), "W": int(rd(HW + "power1_average").strip() or rd(HW + "power1_input").strip() or 0) / 1e6,
             "C": int(rd(HW + "temp1_input").strip() or 0) / 1000, "busy": int(rd(DEV + "gpu_busy_percent").strip() or -1)}
        for l in rd("/proc/meminfo").splitlines():
            if l.startswith(("MemFree:", "MemAvailable:")): d[l.split(":")[0]] = int(l.split()[1]) // 1024
        st = rd(f"/proc/{pid}/stat").rsplit(")", 1)[-1].split()
        if len(st) > 13: d["ut"] = int(st[11]) * 1000 // TCK; d["st"] = int(st[12]) * 1000 // TCK
        for l in rd(f"/proc/{pid}/status").splitlines():
            if l.startswith("VmRSS:"): d["rss"] = int(l.split()[1]) // 1024
        for l in rd("/proc/vmstat").splitlines():
            k, v = l.split()
            if k in VMK: d[k] = int(v)
        for f in glob.glob(f"/sys/class/kfd/kfd/proc/{pid}/stats_*/evicted_ms"): d["kfd_ev"] = int(rd(f) or 0)
        try:
            g, _, _ = GMV.read(DEV + "gpu_metrics")
            d.update({"g_" + k.replace("throttle_residency_", "thr_").replace("average_", "").replace("temperature_", "T_"): v for k, v in g.items()})
        except Exception:
            pass
        out.write(json.dumps(d) + "\n"); out.flush()
        time.sleep(1.0 / HZ)
