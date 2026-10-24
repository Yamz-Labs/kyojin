"""Layer-wise streaming converter: remote FP4/FP8 checkpoint -> EXL3.

    python tools/ds41/stream_convert.py --map tensor_map.json --out OUT --target 2.3 \
        [--freq freq.json] [--layers 0-39] [--include global,mtp,engram] [--device cuda:0] [--dry-run]

Nothing is downloaded as a whole shard. Per unit (global, one decoder layer, one MTP layer, one
engram table) the converter reads each tensor with Range requests, dequantizes it to bf16 and hands
every linear to exllamav3's `quantize_exl3` (no calibration Hessian: the uncalibrated path, as used
for modules without calibration data). Each unit becomes one output shard `<unit>.safetensors`,
published by atomic rename together with a `<unit>.done.json` marker; a rerun skips finished units.
When every unit is done, `model.safetensors.index.json`, `config.json` (with `quantization_config`)
and `quantization_config.json` are written.

Peak RAM (host), per unit: the unit's quantized output is held until its shard is written, at most
about 5.2 GB (a routed layer with every expert at 3 bits), plus about 3 GB of working tensors
(largest: the output head in fp32, 2.6 GB) plus `--prefetch` in-flight fetches (a few MB each).
Plan on 10 GB. Engram tables never enter RAM: they are copied byte for byte in 64 MiB chunks
straight into the output file. GPU memory: the quantizer working set of one linear.

--dry-run does everything except the quantization (and the engram byte copy): it fetches and
dequantizes each tensor, then records the bf16 size, the planned output size and the plan in
`<out>/dry_run/`. It needs no GPU and no exllamav3 import.
"""

import argparse
import gc
import json
import os
import resource
import struct
import sys
import time
import zlib
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import dequant as D  # noqa: E402
import fetch as F  # noqa: E402
import plan as P  # noqa: E402

PS = P.PS      # tensor names / qcfg fields shared with the serving loader and validate_pack.py
from quant import SAFE_DTYPE, CpuExl3Quantizer, Exl3Quantizer, StubQuantizer, quant_record  # noqa: E402,F401


# ----------------------------------------------------------------------------------------------
# Output side
# ----------------------------------------------------------------------------------------------

def write_streamed(path: str, items: list) -> None:
    """Write a safetensors file from [(name, dtype_str, shape, nbytes, chunk_iterable)] without
    holding the data in memory. The header is built from the declared sizes."""
    header, pos = OrderedDict(), 0
    for name, dt, shape, nbytes, _ in items:
        header[name] = {"dtype": dt, "shape": list(shape), "data_offsets": [pos, pos + nbytes]}
        pos += nbytes
    header["__metadata__"] = {"format": "pt"}
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for name, _, _, nbytes, chunks in items:
            written = 0
            for c in chunks:
                f.write(c)
                written += len(c)
            if written != nbytes:
                raise IOError(f"{name}: wrote {written} of {nbytes} bytes")


def save_tensors(path: str, tensors: dict) -> None:
    from safetensors.torch import save_file
    save_file(tensors, path, metadata={"format": "pt"})


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


# ----------------------------------------------------------------------------------------------
# Unit processing
# ----------------------------------------------------------------------------------------------

def group_jobs(entries: list) -> list:
    """Group plan entries of a unit by source weight so a shared tensor is fetched once."""
    jobs = OrderedDict()
    for e in entries:
        if e.get("action") == "pruned":       # no tensors: the expert is left out of the pack
            continue
        k = e.get("weight") or e["name"]
        jobs.setdefault(k, []).append(e)
    return list(jobs.values())


def fetch_job(fetcher: F.TensorFetcher, job: list):
    e = job[0]
    if e["action"] == "quant":
        w = fetcher.fetch(e["weight"])
        s = fetcher.fetch(e["scale"]) if e.get("scale") else None
        return w, s
    return fetcher.fetch(e["name"]), None


def slice_rows(w, s, rows):
    if rows is None:
        return w, s
    a, b = rows
    bo = w.shape[0] // s.shape[0] if s is not None else 1
    return w[a:b], (s[a // bo:b // bo] if s is not None else None)


def process_unit(unit, entries, fetcher, out_dir, dry, quantizer, prefetch, log, merge=None):
    t0 = time.time()
    tensors = {}
    stored = {}   # module key -> record for quantization_config
    lin_tab = {}      # key -> allocation-table row (proxy / energy / bytes per candidate K)
    cands = {}        # K -> {tensor name: tensor} of alternative-K candidates (kept outside the shard)
    cand_stored = {}  # K -> {key: stored record}
    stats = {"bf16_bytes": 0, "planned_bytes": 0, "fetched_bytes": 0, "tensors": 0}
    jobs = group_jobs(entries)
    final = os.path.join(out_dir, f"{unit}.safetensors")

    engram = None
    if jobs and jobs[0][0]["action"] == "copy_raw":
        # Engram tables: raw bytes, streamed to plain files in <out>/engram/ (read by O_DIRECT row gathers at serving
        # time; they are not safetensors shards and not in the index)
        files = {}
        for e in sorted(entries, key=lambda e: e["name"]):
            info = fetcher.info(e["name"])
            stats["planned_bytes"] += info["length"]
            stats["tensors"] += 1
            if not dry:
                fname = f"{e['name']}.bin"
                dst = os.path.join(out_dir, PS.ENGRAM_DIR, fname)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with open(dst + ".tmp", "wb") as f:
                    for chunk in fetcher.iter_chunks(e["name"]):
                        f.write(chunk)
                os.replace(dst + ".tmp", dst)
                files[e["name"]] = {"file": f"{PS.ENGRAM_DIR}/{fname}", "size": os.path.getsize(dst),
                                    "dtype": info["dtype"], "shape": info["shape"]}
                stored[e["name"]] = {"dtype": info["dtype"], "shape": info["shape"]}
        if not dry:
            stats["fetched_bytes"] = stats["planned_bytes"]
            wn = [n for n in files if n.endswith(".weight")][0]
            sn = wn[:-6] + "scale"
            (rows, wcols), scols = files[wn]["shape"], files[sn]["shape"][1]
            fmt = "fp8" if wcols == scols * 32 else "mxfp4"
            engram = {"layer": int(unit.split(".")[1]), "rows": rows, "format": fmt, "head_dim": scols * 32,
                      "files": files}
    else:
        with ThreadPoolExecutor(max_workers=max(1, prefetch)) as ex:
            futs = [ex.submit(fetch_job, fetcher, j) for j in jobs[:prefetch]]
            nxt = len(futs)
            for i, job in enumerate(jobs):
                w, s = futs[i].result()
                futs[i] = None
                if nxt < len(jobs):
                    futs.append(ex.submit(fetch_job, fetcher, jobs[nxt]))
                    nxt += 1
                else:
                    futs.append(None)
                e0 = job[0]
                stats["fetched_bytes"] += w.numel() * w.element_size() + (s.numel() if s is not None else 0)
                for e in job:
                    stats["tensors"] += 1
                    if e["action"] == "copy":
                        if e.get("mask_experts"):       # pruned routed experts: the router must never select them
                            w = w.clone()
                            w[list(e["mask_experts"])] = -1e4
                        stats["bf16_bytes"] += w.numel() * w.element_size()
                        stats["planned_bytes"] += e["out_bytes"]
                        if not dry:
                            tensors[e["name"]] = w
                            stored[e["name"]] = {"dtype": fetcher.info(e["name"])["dtype"], "shape": list(w.shape)}
                        continue
                    ws, ss = slice_rows(w, s, e["rows"])
                    bf = D.dequant(ws, ss)
                    assert tuple(bf.shape) == (e["out"], e["in"]), (e["key"], bf.shape, e["out"], e["in"])
                    stats["bf16_bytes"] += bf.numel() * 2
                    stats["planned_bytes"] += e["out_bytes"]
                    if not dry:
                        q = quantizer(e["key"], bf, e["bits"])
                        for suffix, t in q.items():
                            tensors[f"{e['key']}.{suffix}"] = t
                        stored[e["key"]] = quant_record(e["bits"], q)
                        base = getattr(quantizer, "base", quantizer)
                        pe = getattr(base, "proxy", {}).pop(e["key"], None)
                        en = getattr(base, "energy", {}).pop(e["key"], None)
                        row = {"kind": e["kind"], "in": e["in"], "out": e["out"], "bits": e["bits"],
                               "k": {str(e["bits"]): {"proxy": pe, "bytes": e["out_bytes"], "where": "shard"}}}
                        for k_alt in e.get("alt_bits", []):
                            qa = quantizer(e["key"], bf, k_alt)
                            for suffix, t in qa.items():
                                cands.setdefault(k_alt, {})[f"{e['key']}.{suffix}"] = t
                            cand_stored.setdefault(k_alt, {})[e["key"]] = quant_record(k_alt, qa)
                            pa = getattr(base, "proxy", {}).pop(e["key"], None)
                            en = getattr(base, "energy", {}).pop(e["key"], en)
                            row["k"][str(k_alt)] = {"proxy": pa, "bytes": P.exl3_bytes(e["in"], e["out"], k_alt),
                                                    "where": "cand"}
                        row["E"] = en
                        lin_tab[e["key"]] = row
                    del bf, ws, ss
                del w, s
        if not dry:
            if merge is not None:   # split mode: add the expert shards computed elsewhere
                merge(tensors, stored, stats)
            if cands or lin_tab:
                import split as SPL
                for k_alt, tt in sorted(cands.items()):
                    d = os.path.join(out_dir, "cands", unit)
                    os.makedirs(d, exist_ok=True)
                    save_tensors(os.path.join(d, f"lin.k{k_alt}.safetensors.tmp"), tt)
                    os.replace(os.path.join(d, f"lin.k{k_alt}.safetensors.tmp"),
                               os.path.join(d, f"lin.k{k_alt}.safetensors"))
                    C.atomic_write_json(os.path.join(d, f"lin.k{k_alt}.json"), {"bits": k_alt, "stored": cand_stored[k_alt]})
                try:
                    SPL.alloc_update(out_dir, unit, {"linears": lin_tab})
                except Exception as ex:   # allocation statistics must never fail a unit
                    log(f"{unit}: WARNING allocation table not written: {ex}")
            save_tensors(final + ".tmp", tensors)
    if not dry and engram is not None:
        primary = engram["files"][f"layers.{engram['layer']}.engram.embed.weight"]
        marker = {"unit": unit, "file": primary["file"], "size": primary["size"], "engram": engram, "stored": stored,
                  **stats}
        C.atomic_write_json(os.path.join(out_dir, f"{unit}.done.json"), marker)
    elif not dry:
        os.replace(final + ".tmp", final)
        marker = {"unit": unit, "file": f"{unit}.safetensors", "size": os.path.getsize(final), "stored": stored,
                  **stats}
        C.atomic_write_json(os.path.join(out_dir, f"{unit}.done.json"), marker)
    stats.update({"seconds": round(time.time() - t0, 1), "peak_rss_gb": round(peak_rss_gb(), 2)})
    del tensors
    free_memory()
    log(f"{unit}: {stats}")
    return stats


def expert_idx(key: str) -> int:
    return int(key.split(".")[4])


def unit_done(unit, out_dir, dry) -> bool:
    if dry:
        return os.path.exists(os.path.join(out_dir, "dry_run", f"{unit}.json"))
    mp = os.path.join(out_dir, f"{unit}.done.json")
    if not os.path.exists(mp):
        return False
    m = json.load(open(mp))
    fp = os.path.join(out_dir, m["file"])
    if not (os.path.exists(fp) and os.path.getsize(fp) == m["size"]):
        return False
    return all(os.path.exists(os.path.join(out_dir, f["file"])) and os.path.getsize(os.path.join(out_dir, f["file"])) == f["size"]
               for f in (m.get("engram") or {}).get("files", {}).values())


def select_units(all_units, layers, include):
    sel = []
    lo, hi = (None, None)
    if layers:
        a, _, b = layers.partition("-")
        lo, hi = int(a), int(b or a)
    inc = set(include.split(",")) if include else set()
    for u in all_units:
        if u.endswith(".engram"):
            ok = "engram" in inc or (layers is None and not include)
        elif u == "global":
            ok = "global" in inc or (layers is None and not include)
        elif u.startswith("mtp."):
            ok = "mtp" in inc or (layers is None and not include)
        else:
            n = int(u.split(".")[1])
            ok = (lo is None or lo <= n <= hi) if (layers or not include) else False
        if ok:
            sel.append(u)
    return sel


def finalize(plan, out_dir, tmap, log, src_dir=None, extra_qcfg=None):
    """Index, config.json and quantization_config.json from the unit markers. With `src_dir` (re-assembly of a pack from
    its candidates) the base config and the tokenizer files come from that directory instead of the hub."""
    weight_map, storage = {}, {}
    total = 0
    engram_units = []
    for u in P.units(plan):
        m = json.load(open(os.path.join(out_dir, f"{u}.done.json")))
        if m.get("engram"):
            engram_units.append(m["engram"])        # plain files in <out>/engram/: not shards, not in the index
            continue
        total += m["size"]
        for key, rec in m["stored"].items():
            if "bits" in rec:
                for suffix in rec["stored"]:
                    weight_map[f"{key}.{suffix}"] = m["file"]
                storage[key] = {"quant_format": "exl3", "bits_per_weight": rec["bits"],
                                "stored_tensors": {f"{key}.{s}": v for s, v in rec["stored"].items()},
                                **({"mul1_multiplier": rec["mul1_multiplier"]} if "mul1_multiplier" in rec else {})}
            else:
                weight_map[key] = m["file"]
                storage[key] = {"stored_tensors": {key: rec}}
    C.atomic_write_json(os.path.join(out_dir, "model.safetensors.index.json"),
                        {"metadata": {"total_size": total}, "weight_map": weight_map})
    if src_dir:
        cfg = json.load(open(os.path.join(src_dir, "config.json")))
        orig = (cfg.get("quantization_config") or {}).get("original_quantization_config")
    else:
        cfg = C.fetch_json("config.json", tmap.get("repo", C.REPO), tmap.get("revision", C.REVISION))
        orig = cfg.get("quantization_config")
    plan["summary"] = P.summarize(plan["entries"])      # entries may have been re-decided at queue time (ramfit)
    s = plan["summary"]
    bits = plan["bits"]
    eb = P.expert_bits_from_entries(plan["entries"], plan.get("n_experts"))    # K per expert; 0 = pruned expert
    eg = None
    if engram_units:
        engram_units.sort(key=lambda g: g["layer"])
        fmts = {g["format"] for g in engram_units}
        assert len(fmts) == 1, f"mixed engram table formats {fmts}"
        eg = {"dir": PS.ENGRAM_DIR, "format": fmts.pop(), "layers": [g["layer"] for g in engram_units],
              "rows": [g["rows"] for g in engram_units], "head_dim": engram_units[0]["head_dim"]}
    qcfg = PS.build_qcfg(bits=round(s["routed_avg_bpw"], 4), head_bits=bits["head"], mtp_bits=bits["mtp"],
                         expert_bits=eb, engram=eg,
                         plan={"target": plan["target"], "bits": bits, "expert_frequency": plan["freq"]}, original=orig)
    qcfg.update(extra_qcfg or {})
    cfg["quantization_config"] = qcfg
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=4)
    with open(os.path.join(out_dir, "quantization_config.json"), "w") as f:
        json.dump({**qcfg, "tensor_storage": storage}, f, indent=4)
    for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        if src_dir:
            if os.path.exists(os.path.join(src_dir, name)) and \
                    os.path.abspath(src_dir) != os.path.abspath(out_dir):
                import shutil
                shutil.copyfile(os.path.join(src_dir, name), os.path.join(out_dir, name))
            continue
        try:
            data = C.range_get(C.file_url(name, tmap.get("repo", C.REPO), tmap.get("revision", C.REVISION)), 0, 0,
                               with_total=True)[1]
            with open(os.path.join(out_dir, name), "wb") as f:
                f.write(C.range_get(C.file_url(name, tmap.get("repo", C.REPO), tmap.get("revision", C.REVISION)),
                                    0, data - 1))
        except Exception as e:  # optional files
            log(f"skip {name}: {e}")
    log(f"finalized: {len(weight_map)} tensors, {total / 1e9:.1f} GB")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", default="single", choices=["single", "coordinator", "worker"],
                    help="single: one process does everything (default); coordinator: owns --out and the expert "
                         "queue, quantizes the non-expert linears itself; worker: pull experts over stdin/stdout")
    ap.add_argument("--map")
    ap.add_argument("--out")
    ap.add_argument("--target", type=float, default=2.3)
    ap.add_argument("--freq")
    ap.add_argument("--plan", help="use a saved plan.json instead of building one")
    ap.add_argument("--layers", help="decoder layers only, e.g. 0-1")
    ap.add_argument("--include", help="extra units: global,mtp,engram")
    ap.add_argument("--o-groups", type=int, default=P.O_GROUPS, help="attention output groups (wo_a is sliced per group)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--codebook", default="mul1", choices=["mul1", "mcg"])
    ap.add_argument("--prefetch", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--layer-bits", help="per-layer K of the routed experts, e.g. '0-9:3,10-29:2,30-39:3' "
                    "(replaces the hot/cold tiers)")
    ap.add_argument("--mtp-queue", action="store_true", help="MTP routed experts go through the expert queue")
    ap.add_argument("--no-engram", action="store_true",
                    help="leave the engram tables out of the plan (copied to the pack separately)")
    ap.add_argument("--alt-attn", help="extra candidate K for attention/indexer linears (quantized from the same "
                    "Hessian, stored under cands/, not in the shard), e.g. '3,5'")
    ap.add_argument("--alt-shared", help="extra candidate K for shared-expert linears, e.g. '3,5'")
    ap.add_argument("--max-experts", type=int, help="testing aid: keep only routed experts below this index")
    ap.add_argument("--role-bits", help="attention K per role, e.g. 'q_b=3,wo_a=3,wo_b=3,q_a=5,kv=5' (default: --attn-bits)")
    ap.add_argument("--expert-bits-file", help="JSON {'layers.L.ffn': [K per expert, 0 = pruned]}: fixed per-expert K "
                    "(K=1 allowed; 0 = pruned), overrides --layer-bits")
    for k, v in P.DEFAULT_BITS.items():
        ap.add_argument(f"--{k.replace('_', '-')}-bits", type=int, default=v)
    sp = ap.add_argument_group("split roles (see DESIGN_SPLIT.md)")
    sp.add_argument("--quantizer", default="exl3", choices=["exl3", "stub", "cpu"], help="stub = fake GPU-free test quantizer, cpu = real EXL3 codec on the CPU (slow)")
    sp.add_argument("--stub-delay", type=float, default=0.0, help="seconds per linear in the stub quantizer")
    sp.add_argument("--hessians", default="none", choices=["none", "spool", "synthetic"],
                    help="none: uncalibrated; spool: read the capture's Hessians from --spool; "
                         "synthetic: deterministic random SPD Hessians (tests, smoke)")
    sp.add_argument("--spool", help="Hessian spool directory (default OUT/spool)")
    sp.add_argument("--worker", action="append",
                    help="NAME=SHELL_COMMAND of a worker (e.g. 'box=python tools/ds41/stream_convert.py --role "
                         "worker --device cuda:0'); prefix the name with 'late:' to feed it only once the layer's "
                         "capture is done. Repeatable")
    sp.add_argument("--worker-ssh", action="append", help="HOST[:CUDA_INDEX]: worker over ssh, code shipped on stdin")
    sp.add_argument("--ssh-python", default="~/lab/exl3-strix/venv-cuda/bin/python")
    sp.add_argument("--ssh-pythonpath",
                    default="~/ds41_work/port1/release/kyojin-cuda-0d725cb-q35")
    sp.add_argument("--ssh-env", default="", help="extra 'VAR=value ...' for the remote worker, e.g. EXL3_QUANT_PROFILE=1")
    sp.add_argument("--ssh-lock", default="~/strix-bench/gpu3090.lock")
    sp.add_argument("--ssh-min-mem-mib", type=int, default=12288)
    sp.add_argument("--ssh-restarts", type=int, default=100000)
    sp.add_argument("--ssh-retry-s", type=float, default=300.0)
    sp.add_argument("--local-device", help="coordinator's own quantizer device (default --device)")
    sp.add_argument("--chunk", type=int, default=8, help="experts per job (smaller near the end of a layer)")
    sp.add_argument("--depth", type=int, default=2, help="jobs in flight per worker")
    sp.add_argument("--worker-threads", type=int, default=2,
                    help="compute threads (CUDA streams) per worker; output is bit-identical to 1")
    sp.add_argument("--reissue-after", type=float, default=240.0,
                    help="seconds before a straggling last job is also given to an idle worker")
    sp.add_argument("--worker-restarts", type=int, default=3)
    sp.add_argument("--retry-s", type=float, default=30.0)
    sp.add_argument("--min-mem-gib", type=float, default=6.0, help="worker waits while MemAvailable is below this")
    sp.add_argument("--hess-timeout", type=float, default=7200.0)
    sp.add_argument("--alt-experts", help="routed-expert candidate K set, e.g. '2,3': experts are also quantized at the "
                    "K of this set that differs from their plan K, if selected by --alt-up-frac / --alt-down-frac")
    sp.add_argument("--alt-up-frac", type=float, default=0.0625,
                    help="per layer, the hottest fraction of experts (calibration routing count) that gets the "
                         "higher-K candidates")
    sp.add_argument("--alt-down-frac", type=float, default=0.25,
                    help="per layer, the coldest fraction of experts that gets the lower-K candidates")
    sp.add_argument("--ramfit", help="ramfit.json (tools/ds41/ramfit.py plan): choose K in {1,2,3,..} (0 = prune) of "
                    "every routed expert AT QUEUE TIME from the layer's routing statistics, filling the plan's per-layer "
                    "byte budget. Replaces --layer-bits / --alt-experts for the backbone experts")
    sp.add_argument("--ramfit-smooth", type=float, default=0.02,
                    help="routing-mass prior as a fraction of the layer mean (an unseen expert is not a dead one)")
    sp.add_argument("--ramfit-prune-tau", type=float, default=None,
                    help="with prune enabled in the plan: only experts routed to by < tau of the calibration tokens "
                         "(default: the plan's prune_tau, 1e-4)")
    sp.add_argument("--stats-timeout", type=float, default=3600.0,
                    help="seconds to wait for the capture's routing statistics of a layer before running without "
                         "alternative candidates (a capture that already emits Hessians without them is detected earlier)")
    sp.add_argument("--synthetic-slices", type=int, default=4)
    sp.add_argument("--synthetic-gap", type=float, default=0.0)
    sp.add_argument("--experts-only", action="store_true",
                    help="coordinator: only run the expert queue (no assembly, no GPU on this side); for benchmarks")
    sp.add_argument("--keep-spool", action="store_true", help="keep each expert's Hessians after its result is stored")
    sp.add_argument("--cache", help="local per-layer source cache shared with capture.py (workers read it first)")
    sp.add_argument("--keep-lin-hess", action="store_true",
                    help="keep the attention/shared Hessians of every layer under OUT/hess_keep/")
    sp.add_argument("--keep-experts", action="store_true", help="keep experts/<unit>/ after a layer is assembled")
    a = ap.parse_args(argv)
    if a.role == "worker":
        import split
        return split.worker_main()
    if not a.map or not a.out:
        ap.error("--map and --out are required")

    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
    tmap = json.load(open(a.map))
    os.makedirs(a.out, exist_ok=True)
    if a.plan:
        plan = json.load(open(a.plan))
    else:
        plan = P.build_plan(tmap, a.target, P.load_freq(a.freq) if a.freq else None,
                            {k: getattr(a, f"{k}_bits") for k in P.DEFAULT_BITS},
                            P.parse_layer_bits(a.layer_bits) if a.layer_bits else None, a.no_engram, a.mtp_queue,
                            {"attn": P.parse_ks(a.alt_attn), "shared": P.parse_ks(a.alt_shared)},
                            P.parse_role_bits(a.role_bits),
                            P.load_expert_bits(a.expert_bits_file) if a.expert_bits_file else None,
                            o_groups=a.o_groups)
    if a.max_experts is not None:
        plan["entries"] = [e for e in plan["entries"]
                           if e["kind"] != "routed" or int(e["key"].split(".")[4]) < a.max_experts]
        plan["summary"] = P.summarize(plan["entries"])
    C.atomic_write_json(os.path.join(a.out, "plan.json"), plan)
    all_units = P.units(plan)
    sel = select_units(all_units, a.layers, a.include)
    log(f"{len(sel)}/{len(all_units)} units selected, plan {plan['summary']['total_gb']:.1f} GB, "
        f"routed avg {plan['summary']['routed_avg_bpw']:.3f} bpw, dry_run={a.dry_run}")

    by_unit = {}
    for e in plan["entries"]:
        by_unit.setdefault(e["unit"], []).append(e)
    if a.role == "coordinator":
        import split
        if a.dry_run:
            ap.error("--dry-run is a single-role option")
        try:
            split.coordinate(a, plan, tmap, sel, by_unit, sys.modules[__name__], log)
        finally:
            if getattr(a, "ramfit", None):     # K / pruned set decided per layer at queue time: keep plan.json in step
                plan["summary"] = P.summarize(plan["entries"])
                C.atomic_write_json(os.path.join(a.out, "plan.json"), plan)
        if a.experts_only:
            log("experts-only: layer not assembled")
        elif all(unit_done(u, a.out, False) for u in all_units):
            finalize(plan, a.out, tmap, log)
        else:
            log("not all units done; finalize skipped")
        return

    fetcher = F.TensorFetcher(tmap)
    quantizer = None if a.dry_run else (StubQuantizer(a.stub_delay) if a.quantizer == "stub"
                                        else CpuExl3Quantizer() if a.quantizer == "cpu"
                                        else Exl3Quantizer(a.device, a.codebook))
    if a.dry_run:
        os.makedirs(os.path.join(a.out, "dry_run"), exist_ok=True)

    for u in sel:
        if unit_done(u, a.out, a.dry_run):
            log(f"{u}: done, skipping")
            continue
        st = process_unit(u, by_unit[u], fetcher, a.out, a.dry_run, quantizer, a.prefetch, log)
        if a.dry_run:
            C.atomic_write_json(os.path.join(a.out, "dry_run", f"{u}.json"), st)

    if a.dry_run:
        rep = {u: json.load(open(os.path.join(a.out, "dry_run", f"{u}.json")))
               for u in all_units if os.path.exists(os.path.join(a.out, "dry_run", f"{u}.json"))}
        C.atomic_write_json(os.path.join(a.out, "dry_run_report.json"),
                            {"units": rep,
                             "bf16_bytes": sum(r["bf16_bytes"] for r in rep.values()),
                             "planned_bytes": sum(r["planned_bytes"] for r in rep.values())})
        log("dry run report written")
    elif all(unit_done(u, a.out, False) for u in all_units):
        finalize(plan, a.out, tmap, log)
    else:
        log("not all units done; finalize skipped")


if __name__ == "__main__":
    main()
