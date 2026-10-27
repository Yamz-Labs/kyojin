"""Build the final EXL3 pack from the conversion output and the optimizer's choice (no re-quantization).

    python assemble_pack.py --src OUT --alloc OUT/alloc.json --dst PACK [--link]

OUT holds the conversion's layer shards (every module at its plan K) plus `cands/` (the alternative-K candidates of the
experts and of the attention / shared linears). For every module whose chosen K differs from the shard's, the tensors are
swapped for the candidate's; everything else is copied (or hard-linked with --link) untouched. The index, config.json
(`quantization_config.expert_bits`, `bits`) and quantization_config.json are rewritten for the result.

Runs on any machine that has OUT (CPU only, one layer shard in RAM at a time, ~6 GB).
"""

import argparse
import json
import os
import re
import shutil
import sys
import time
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
import plan as P  # noqa: E402


def load_blob_tensors(path: str, meta_path: str) -> dict:
    from safetensors.torch import load
    blob = open(path, "rb").read()
    meta = json.load(open(meta_path))
    if len(blob) != meta["size"] or zlib.crc32(blob) != meta["crc32"]:
        raise IOError(f"{path}: size/CRC mismatch with its marker")
    return load(blob), meta


def read_header(path: str) -> dict:
    import struct
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def blob_of(tensors: dict) -> bytes:
    from safetensors.torch import save
    return save(tensors, metadata={"format": "pt"})


def write_cand(path: str, meta_path: str, tensors: dict, meta: dict):
    blob = blob_of(tensors)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "wb") as f:
        f.write(blob)
    os.replace(path + ".tmp", path)
    C.atomic_write_json(meta_path, {**meta, "size": len(blob), "crc32": zlib.crc32(blob)})


def expected_trellis_shape(in_f, out_f, k):
    return [in_f // 16, out_f // 16, 16 * k]


def assemble(src: str, alloc: dict, dst: str, link: bool = False, log=print):
    """Build `dst` from `src` + `alloc`. dst == src: swap in place (the displaced tensors go to cands/, so the operation is
    reversible and the disk footprint does not grow: one layer shard of scratch space). The current K of every module
    is read from the shard's own safetensors header, never from a marker, so an interrupted run can simply be repeated."""
    from safetensors.torch import load_file, save_file
    inplace = os.path.abspath(src) == os.path.abspath(dst)
    plan = json.load(open(os.path.join(src, "plan.json")))
    expert_bits = alloc.get("expert_bits", {})
    module_bits = alloc.get("module_bits", {})
    os.makedirs(dst, exist_ok=True)
    by_key = {e["key"]: e for e in plan["entries"] if e["action"] == "quant"}
    changed_total = {"experts": 0, "modules": 0}
    final_bits = {}
    for unit in P.units(plan):
        marker = json.load(open(os.path.join(src, f"{unit}.done.json")))
        if marker.get("engram"):          # Engram tables: plain files in <dir>/engram/, carried over untouched
            if not inplace:
                for f in marker["engram"]["files"].values():
                    sf, df = os.path.join(src, f["file"]), os.path.join(dst, f["file"])
                    os.makedirs(os.path.dirname(df), exist_ok=True)
                    if not (os.path.exists(df) and os.path.getsize(df) == f["size"]):
                        if os.path.exists(df):
                            os.remove(df)
                        try:
                            if not link:
                                raise OSError
                            os.link(sf, df)
                        except OSError:
                            shutil.copyfile(sf, df)
                shutil.copyfile(os.path.join(src, f"{unit}.done.json"), os.path.join(dst, f"{unit}.done.json"))
            continue
        shard = os.path.join(src, marker["file"])
        hdr = read_header(shard)
        cur = lambda key: hdr[key + ".trellis"]["shape"][-1] // 16
        want_e = expert_bits.get(f"{unit}.ffn")
        swap_e = {}
        if want_e is not None:
            for e, k in enumerate(want_e):
                key = f"{unit}.ffn.experts.{e}.w1"
                if k is not None and key + ".trellis" in hdr and cur(key) != k:
                    swap_e[e] = k
        swap_m = {key: k for key, k in module_bits.items()
                  if key.startswith(unit + ".") and key + ".trellis" in hdr and cur(key) != k}
        dst_shard = os.path.join(dst, marker["file"])
        if not swap_e and not swap_m:
            if not inplace:
                if not (os.path.exists(dst_shard) and os.path.getsize(dst_shard) == marker["size"]):
                    if os.path.exists(dst_shard):
                        os.remove(dst_shard)
                    try:
                        if not link:
                            raise OSError
                        os.link(shard, dst_shard)
                    except OSError:
                        shutil.copyfile(shard, dst_shard)
                shutil.copyfile(os.path.join(src, f"{unit}.done.json"), os.path.join(dst, f"{unit}.done.json"))
            continue
        t0 = time.time()
        tensors = load_file(shard)
        stored = dict(marker["stored"])
        consumed, displaced = [], []
        cdir = os.path.join(src, "cands", unit)

        def swap_in(prefixes, k, cand_f, cand_m, kind):
            """Replace the tensors of `prefixes` (module keys) by the K=k candidate; returns the displaced tensors."""
            if not os.path.exists(cand_f):
                raise FileNotFoundError(f"{unit} {kind}: no K={k} candidate ({cand_f})")
            if kind == "expert":
                new, meta = load_blob_tensors(cand_f, cand_m)
            else:
                new, meta = load_file(cand_f), json.load(open(cand_m))
            out = {}
            for key in prefixes:
                old_k = cur(key)
                for name in [n for n in tensors if n.startswith(key + ".")]:
                    out[name] = tensors.pop(name)
                for name, t in new.items():
                    if name.startswith(key + "."):
                        tensors[name] = t
                ent = by_key[key]
                assert list(tensors[key + ".trellis"].shape) == expected_trellis_shape(ent["in"], ent["out"], k), (key, k)
                out_stored = stored[key]
                stored[key] = meta["stored"][key]
            return out, old_k, out_stored

        for e, k in sorted(swap_e.items()):
            keys = [f"{unit}.ffn.experts.{e}.{w}" for w in ("w1", "w2", "w3")]
            cf = os.path.join(cdir, f"e{e:03d}.k{k}.safetensors")
            cm = os.path.join(cdir, f"e{e:03d}.k{k}.json")
            old_k = cur(keys[0])
            out, _, _ = swap_in(keys, k, cf, cm, "expert")
            if inplace:
                st_old = {key: json.loads(json.dumps(marker["stored"][key])) for key in keys}
                displaced.append((os.path.join(cdir, f"e{e:03d}.k{old_k}.safetensors"),
                                  os.path.join(cdir, f"e{e:03d}.k{old_k}.json"), out,
                                  {"bits": old_k, "stored": st_old, "e": e, "displaced": True}))
                consumed.append((cf, cm))
            changed_total["experts"] += 1
        if swap_m:
            by_k = {}
            for key, k in swap_m.items():
                by_k.setdefault(k, []).append(key)
            for k, keys in sorted(by_k.items()):
                cf = os.path.join(cdir, f"lin.k{k}.safetensors")
                cm = os.path.join(cdir, f"lin.k{k}.json")
                old = {key: cur(key) for key in keys}
                out, _, _ = swap_in(sorted(keys), k, cf, cm, "module")
                changed_total["modules"] += len(keys)
                if inplace:   # displaced linears of one old K are appended to that K's linear candidate file
                    for old_k in sorted(set(old.values())):
                        ks_ = [key for key in keys if old[key] == old_k]
                        pf, pm = (os.path.join(cdir, f"lin.k{old_k}.safetensors"),
                                  os.path.join(cdir, f"lin.k{old_k}.json"))
                        prev = load_file(pf) if os.path.exists(pf) else {}
                        pmeta = json.load(open(pm)) if os.path.exists(pm) else {"bits": old_k, "stored": {}}
                        for key in ks_:
                            for name in [n for n in out if n.startswith(key + ".")]:
                                prev[name] = out[name]
                            pmeta["stored"][key] = marker["stored"][key]
                        displaced.append((pf, pm, prev, pmeta))
        new_shard = dst_shard + ".new"
        save_file(tensors, new_shard, metadata={"format": "pt"})
        m2 = {**marker, "size": os.path.getsize(new_shard), "stored": stored}
        if inplace:
            for pf, pm, tt, meta in displaced:               # 1. displaced variants (nothing is destroyed yet)
                write_cand(pf, pm, tt, meta)
            C.atomic_write_json(os.path.join(dst, f"{unit}.done.json.new"), m2)
            os.replace(new_shard, dst_shard)                  # 2. shard + marker swap back to back
            os.replace(os.path.join(dst, f"{unit}.done.json.new"), os.path.join(dst, f"{unit}.done.json"))
            for cf, cm in consumed:                           # 3. candidates now living in the shard
                for f in (cf, cm):
                    if os.path.exists(f):
                        os.remove(f)
        else:
            if os.path.exists(dst_shard):
                os.remove(dst_shard)           # may be a hard link to the source shard: never write through it
            os.replace(new_shard, dst_shard)
            C.atomic_write_json(os.path.join(dst, f"{unit}.done.json"), m2)
        log(f"{unit}: {len(swap_e)} experts, {len(swap_m)} modules swapped ({time.time() - t0:.0f}s)")
        del tensors, displaced
    # plan of the result: K of every module as now stored
    for e in plan["entries"]:
        if e["action"] != "quant":
            continue
        k = None
        m = P.RE_EXPERT.match(e.get("weight", "")) if e["kind"] == "routed" else None
        if m and f"{m.group(1)}.ffn" in expert_bits:
            k = expert_bits[f"{m.group(1)}.ffn"][int(m.group(2))]
        elif e["key"] in module_bits:
            k = module_bits[e["key"]]
        if k is not None and k != e["bits"]:
            e["bits"], e["out_bytes"] = k, P.exl3_bytes(e["in"], e["out"], k)
    plan["summary"] = P.summarize(plan["entries"])
    plan["alloc"] = {"from": os.path.abspath(src), "summary": alloc.get("summary"), "swapped": changed_total}
    C.atomic_write_json(os.path.join(dst, "plan.json"), plan)
    import stream_convert as S
    S.finalize(plan, dst, {}, log, src_dir=src, extra_qcfg={"allocation": {"optimizer": "optimize_alloc.py",
                                                                      "params": alloc.get("params")}})
    log(f"assembled {dst}: {changed_total}")
    return plan


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--alloc", required=True)
    ap.add_argument("--dst", required=True, help="output pack dir; equal to --src: swap in place (reversible, cands/ "
                    "receives the displaced variants, no extra disk)")
    ap.add_argument("--link", action="store_true", help="hard-link unchanged shards (same filesystem)")
    a = ap.parse_args(argv)
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
    assemble(a.src, json.load(open(a.alloc)), a.dst, a.link, log)


if __name__ == "__main__":
    main()
