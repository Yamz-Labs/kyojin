#!/usr/bin/env python3
"""Check a converted DS4.1 pack against what the serving model expects, before the first (expensive) load.

    python validate_pack.py PACK [--plan PACK/plan.json] [--engram-dir DIR] [--no-mtp] [--json]

Checks (headers only, nothing is dequantized; CPU, a few seconds on a real pack):
  * config.json: text_config parses, quantization_config (quant_method, codebook, pack_format)
  * index <-> shard headers: every indexed tensor is in its shard and vice versa; shard sizes; total_size
  * every tensor the serving model builds (`exllamav3/dsv41/packspec.py`, the module the converter shares):
    present, dtype, shape; EXL3 linears have trellis / suh / svh / mul1 with consistent dims and K in 1..8
  * per-expert K (quantization_config.expert_bits) against the trellis of each expert; pruned experts (K = 0) have no
    tensors and the router bias of their layer is masked (<= -1e3) at exactly their ids
  * sizes against the plan (plan.json): bytes of every planned linear / tensor
  * vision tower (when config.json has vision_config): all 266 vision.* / aligner.* / image_* tensors with the shapes
    the runtime reads, and ffn.gate.bias_vl in every layer (masked at the pruned ids like ffn.gate.bias)
  * Engram tables: directory, one weight + one scale file per engram layer, row counts, file sizes
Exit status 1 when any error was found. Warnings do not fail.
"""
import argparse
import json
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

PS = C.packspec()
VS = C._load_engine_module("vision_spec", "vision_spec.py")
DT_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "U16": 2, "U32": 4,
            "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1, "BOOL": 1}
FLOATS = {"F32", "F16", "BF16", "F64", "F8_E4M3", "F8_E5M2"}


class Report:
    def __init__(self):
        self.errors, self.warnings, self.stats = [], [], {}

    def err(self, msg):
        self.errors.append(msg)

    def warn(self, msg):
        self.warnings.append(msg)

    @property
    def ok(self):
        return not self.errors

    def as_dict(self):
        return {"ok": self.ok, "errors": self.errors, "warnings": self.warnings, "stats": self.stats}


def _numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def _tensor_bytes(h):
    return h["data_offsets"][1] - h["data_offsets"][0]


def _read_floats(path, data_start, h):
    """Values of a small float tensor (router bias): list of floats."""
    import numpy as np
    with open(path, "rb") as f:
        f.seek(data_start + h["data_offsets"][0])
        raw = f.read(_tensor_bytes(h))
    dt = h["dtype"]
    if dt == "F32":
        return np.frombuffer(raw, dtype="<f4").astype("float64")
    if dt == "F16":
        return np.frombuffer(raw, dtype="<f2").astype("float64")
    if dt == "BF16":
        u = np.frombuffer(raw, dtype="<u2").astype("<u4") << 16
        return u.view("<f4").astype("float64")
    raise ValueError(dt)


def load_config(pack, rep):
    path = os.path.join(pack, "config.json")
    if not os.path.exists(path):
        rep.err("config.json missing")
        return None, {}
    hf = json.load(open(path))
    try:
        cfg = C.dsv41_config().DSV41Config.from_hf_dict(hf)
        cfg.validate()
    except Exception as ex:
        rep.err(f"config.json: text_config does not parse as a DS4.1 config ({type(ex).__name__}: {ex})")
        return None, hf.get("quantization_config") or {}
    return cfg, hf.get("quantization_config") or {}


def validate(pack, plan_path=None, engram_dir=None, mtp=True):
    rep = Report()
    cfg, qcfg = load_config(pack, rep)
    if cfg is None:
        return rep
    if qcfg.get("quant_method") != "exl3":
        rep.warn(f"quantization_config.quant_method = {qcfg.get('quant_method')!r} (expected 'exl3')")
    if qcfg.get("codebook", "mul1") != "mul1":
        rep.err(f"codebook {qcfg.get('codebook')!r}: the DS4.1 serving path decodes mul1 only")
    if qcfg.get("pack_format", PS.PACK_FORMAT) != PS.PACK_FORMAT:
        rep.err(f"pack_format {qcfg.get('pack_format')} != {PS.PACK_FORMAT}")
    synthetic = bool(qcfg.get("synthetic"))
    exl3 = not synthetic
    has_mtp = bool(mtp and cfg.n_mtp_layers)
    rep.stats.update({"layers": cfg.num_hidden_layers, "mtp_layers": cfg.n_mtp_layers if has_mtp else 0,
                      "format": "plain fp16 (synthetic)" if synthetic else "exl3"})

    # ---- index and shard headers
    ipath = os.path.join(pack, "model.safetensors.index.json")
    if not os.path.exists(ipath):
        rep.err("model.safetensors.index.json missing")
        return rep
    index = json.load(open(ipath))
    wmap = index.get("weight_map", {})
    heads, starts, shard_bytes = {}, {}, 0
    for shard in sorted(set(wmap.values())):
        sp = os.path.join(pack, shard)
        if not os.path.exists(sp):
            rep.err(f"shard {shard} listed in the index is missing")
            continue
        try:
            h = PS.read_header(sp)
            ds = PS.header_data_start(sp)
        except Exception as ex:
            rep.err(f"{shard}: unreadable safetensors header ({ex})")
            continue
        size = os.path.getsize(sp)
        end = max((t["data_offsets"][1] for t in h.values()), default=0)
        if ds + end != size:
            rep.err(f"{shard}: file size {size} != header end {ds + end}")
        shard_bytes += size
        for n, t in h.items():
            if _tensor_bytes(t) != _numel(t["shape"]) * DT_BYTES.get(t["dtype"], 1):
                rep.err(f"{n}: {_tensor_bytes(t)} bytes for shape {t['shape']} {t['dtype']}")
            heads[n] = (shard, t)
        starts[shard] = (sp, ds)
    for n, f in wmap.items():
        if n in heads and heads[n][0] != f:
            rep.err(f"{n}: index says {f}, found in {heads[n][0]}")
        elif n not in heads and f in starts:
            rep.err(f"{n}: in the index but not in {f}")
    for n, (f, _) in heads.items():
        if n not in wmap:
            rep.err(f"{n}: in {f} but not in the index")
    want = index.get("metadata", {}).get("total_size")
    if want is not None and want != shard_bytes:
        rep.warn(f"index total_size {want} != sum of shard sizes {shard_bytes}")
    rep.stats["tensors"] = len(heads)
    rep.stats["shard_gb"] = round(shard_bytes / 1e9, 4)

    # ---- plan / per-module K
    plan = None
    ppath = plan_path or os.path.join(pack, "plan.json")
    if os.path.exists(ppath):
        plan = json.load(open(ppath))
    module_bits = {}
    if plan is not None:
        for e in plan["entries"]:
            if e["action"] == "quant" and e.get("kind") != "routed":
                module_bits[e["key"]] = e["bits"]
    else:
        qpath = os.path.join(pack, "quantization_config.json")
        if os.path.exists(qpath):
            for k, v in (json.load(open(qpath)).get("tensor_storage") or {}).items():
                if "bits_per_weight" in v:
                    module_bits[k] = v["bits_per_weight"]
    try:
        expert_bits = {k: PS.expert_bits_of(qcfg, k, cfg.moe_counts(int(k.split(".")[1]))[0])
                       for k in (qcfg.get("expert_bits") or {})}
    except ValueError as ex:
        rep.err(str(ex))
        expert_bits = {}
    for k in qcfg.get("pruned_experts") or {}:
        want = [i for i, b in enumerate(expert_bits.get(k) or []) if b == 0]
        if want != list(qcfg["pruned_experts"][k]):
            rep.err(f"pruned_experts[{k!r}] does not match expert_bits")

    # ---- every tensor the serving model builds
    exp = PS.expected_tensors(cfg, exl3=exl3, expert_bits=expert_bits, module_bits=module_bits, mtp=has_mtp)
    # The converter copies the Engram projection raw (F8_E4M3 weight + F8_E8M0 32x32 scale grid, never quantized): accept
    # that form in place of the EXL3 tensors (the serving Linear dequantizes it at load).
    for lid in cfg.engram_layer_ids:
        base = f"layers.{lid}.engram.wkv"
        if f"{base}.weight" in heads:
            for n in [n for n in exp if n.startswith(base + ".")]:
                del exp[n]
            o_f = cfg.hidden_size * (cfg.hc_mult + 1)
            i_f = (cfg.engram_max_ngram_size - 1) * cfg.engram_n_heads * cfg.engram_head_dim
            exp[f"{base}.weight"] = PS.Expected(f"{base}.weight", "F8_E4M3", (o_f, i_f), "raw")
            exp[f"{base}.scale"] = PS.Expected(f"{base}.scale", "F8_E8M0", (o_f // 32, i_f // 32), "raw")
    missing = [n for n, e in exp.items() if n not in heads and not e.optional]
    for n in missing[:40]:
        rep.err(f"missing tensor {n}")
    if len(missing) > 40:
        rep.err(f"... {len(missing) - 40} more missing tensors")
    extra = [n for n in heads if n not in exp and not PS.is_engram_table(n)]
    # the vision tower (vision.*, aligner.*, image_* markers) is copied raw by the converter and not built by the text
    # model: checked below against the shapes the vision runtime (exllamav3/dsv41/vision.py) reads
    extra = [n for n in extra if not VS.is_tower_tensor(n)]
    vcfg = VS.read_vision_config(json.load(open(os.path.join(pack, "config.json"))))
    if vcfg is None:
        if any(VS.is_tower_tensor(n) for n in heads):
            rep.warn("the pack holds vision tensors but config.json has no vision_config: image input stays off")
    else:
        want_v = VS.expected_tower_shapes(vcfg)
        miss_v = [n for n in want_v if n not in heads]
        for n in miss_v[:10]:
            rep.err(f"missing vision tensor {n}")
        if len(miss_v) > 10:
            rep.err(f"... {len(miss_v) - 10} more missing vision tensors")
        for n, shape in want_v.items():
            if n in heads:
                h = heads[n][1]
                if tuple(h["shape"]) != shape:
                    rep.err(f"{n}: shape {h['shape']} != {list(shape)}")
                if h["dtype"] not in FLOATS:
                    rep.err(f"{n}: dtype {h['dtype']} is not a float type")
        for n in heads:
            if VS.is_tower_tensor(n) and n not in want_v:
                rep.err(f"unexpected vision tensor {n}")
        rep.stats["vision_tensors"] = len(want_v) - len(miss_v)
        # the VL router bias of every layer that has one is masked at the pruned ids too (image rows route with it)
        vlmiss = [l for l in range(cfg.num_hidden_layers) if f"layers.{l}.ffn.gate.bias" in heads
                  and f"layers.{l}.ffn.gate.bias_vl" not in heads]
        if vlmiss:
            rep.err(f"vision config present but layers {vlmiss[:8]} have no ffn.gate.bias_vl")
    if not has_mtp:
        extra = [n for n in extra if not n.startswith("mtp.")]
    for n in extra[:40]:
        rep.err(f"unexpected tensor {n}")
    if len(extra) > 40:
        rep.err(f"... {len(extra) - 40} more unexpected tensors")
    lins = {l.key: l for l in PS.linears(cfg, mtp=has_mtp)}
    ks = {}
    for n, e in exp.items():
        if n not in heads:
            continue
        shard, h = heads[n]
        if PS.unit_of(n) != shard[:-len(".safetensors")]:
            rep.err(f"{n}: stored in {shard}, expected {PS.shard_file(PS.unit_of(n))}")
        if e.dtype == "*":
            if h["dtype"] not in FLOATS:
                rep.err(f"{n}: dtype {h['dtype']} is not a float type")
        elif h["dtype"] != e.dtype:
            rep.err(f"{n}: dtype {h['dtype']} != {e.dtype}")
        if e.shape is not None and tuple(h["shape"]) != tuple(e.shape):
            rep.err(f"{n}: shape {h['shape']} != {list(e.shape)}")
        if e.kind == "exl3" and n.endswith(".trellis"):
            lin = lins[e.key]
            sh = h["shape"]
            if len(sh) != 3 or sh[2] % 16 or not (PS.K_MIN <= sh[2] // 16 <= PS.K_MAX) or \
                    sh[:2] != [lin.in_pad // 16, lin.out_pad // 16]:
                rep.err(f"{n}: trellis shape {sh} does not fit a {lin.in_pad}x{lin.out_pad} linear with K in 1..8")
            else:
                ks[e.key] = sh[2] // 16
    # K against the declared per-expert K
    for key, k in ks.items():
        lin = lins[key]
        if lin.expert is not None:
            want = PS.lin_k(lin, expert_bits)
            if want is not None and want != k:
                rep.err(f"{key}: trellis K = {k}, expert_bits says {want}")
    # pruned experts: no tensors, router bias masked at exactly those ids
    for lk, bits in expert_bits.items():
        if not bits or 0 not in bits:
            continue
        layer = lk[:-len(".ffn")]
        pruned = [i for i, b in enumerate(bits) if b == 0]
        for i in pruned:
            for w in ("w1", "w2", "w3"):
                if any(n.startswith(f"{layer}.ffn.experts.{i}.{w}.") for n in heads):
                    rep.err(f"{layer}: pruned expert {i} still has {w} tensors")
        for bn in (f"{layer}.ffn.gate.bias", f"{layer}.ffn.gate.bias_vl"):
            if bn in heads:
                shard, h = heads[bn]
                vals = _read_floats(*starts[shard], h)
                masked = [i for i, v in enumerate(vals) if v <= PS.MASK_THRESHOLD]
                if masked != pruned:
                    rep.err(f"{bn}: masked ids {masked[:8]}... != pruned experts {pruned[:8]}...")
    # sizes vs plan
    if plan is not None:
        bad = 0
        for e in plan["entries"]:
            if e["action"] == "pruned":
                continue
            if e["action"] == "copy_raw":
                continue                         # engram tables: checked against the engram files below
            if e["action"] == "quant":
                got = sum(_tensor_bytes(heads[f"{e['key']}.{s}"][1]) for s in ("trellis", "suh", "svh", "mul1")
                          if f"{e['key']}.{s}" in heads)
                if not got and f"{e['key']}.slice.0.trellis" in heads:
                    # one planned linear stored as per-group slices (split_mtp_wo_a.py): trellis + svh split, suh/mul1 shared
                    sl = [n for n in heads if n.startswith(e["key"] + ".slice.")]
                    got = sum(_tensor_bytes(heads[n][1]) for n in sl if n.endswith((".trellis", ".svh")))
                    got += sum(_tensor_bytes(heads[f"{e['key']}.slice.0.{s}"][1]) for s in ("suh", "mul1"))
                name = e["key"]
            else:
                got = _tensor_bytes(heads[e["name"]][1]) if e["name"] in heads else 0
                name = e["name"]
            if got != e["out_bytes"] and (name in heads or e["action"] == "quant"):
                bad += 1
                if bad <= 20:
                    rep.err(f"{name}: {got} bytes in the pack, plan says {e['out_bytes']}")
        if bad > 20:
            rep.err(f"... {bad - 20} more size mismatches against the plan")
        rep.stats["plan_total_gb"] = round(plan["summary"]["total_bytes"] / 1e9, 4)
    elif exl3:
        rep.warn("no plan.json: sizes were not checked against the plan")

    # ---- Engram tables
    if cfg.engram_layer_ids:
        validate_engram(pack, cfg, qcfg, heads, starts, engram_dir, rep)
    return rep


def validate_engram(pack, cfg, qcfg, heads, starts, override, rep):
    egc = PS.engram_cfg(qcfg)
    edir = PS.resolve_engram_dir(pack, qcfg, override)
    rows_cfg = list(cfg.engram_num_embeddings or [])
    hd = cfg.engram_head_dim
    fmt = egc.get("format", PS.ENGRAM_FMT)
    wb, sb = PS.engram_row_bytes(hd, fmt)
    in_shards = [lid for lid in cfg.engram_layer_ids if f"layers.{lid}.engram.embed.weight" in heads]
    if edir is None and not in_shards:
        rep.err("no Engram tables: no engram directory (<pack>/engram, quantization_config.engram.dir, "
                "$EXL3_DSV41_ENGRAM_DIR) and none in the shards")
        return
    if len(rows_cfg) != len(cfg.engram_layer_ids):
        rep.warn("text_config.engram_num_embeddings missing: row counts are not checked against the config")
    rep.stats["engram_dir"] = edir
    import glob
    official = {}                                  # tensor name -> (dtype, shape) of the official FP8 shards in the directory
    for fn in sorted(glob.glob(os.path.join(edir, "*.safetensors"))) if edir else []:
        with open(fn, "rb") as f:
            hdr = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
        official.update({k.replace("model.", "", 1) if k.startswith("model.layers.") else k: (v["dtype"], v["shape"])
                         for k, v in hdr.items() if k != "__metadata__"})
    for i, lid in enumerate(cfg.engram_layer_ids):
        want_rows = rows_cfg[i] if i < len(rows_cfg) else None
        wf, sf = PS.engram_files(lid)
        ow, os_ = official.get(f"layers.{lid}.engram.embed.weight"), official.get(f"layers.{lid}.engram.embed.scale")
        if ow is not None:
            if os_ is None:
                rep.err(f"engram layer {lid}: scale tensor missing in the official shards")
            elif ow[0] != "F8_E4M3" or ow[1] != [os_[1][0], hd] or os_[1][1] != hd // 32:
                rep.err(f"engram layer {lid}: official tensors {ow} / {os_} are not e4m3 [rows, {hd}] + e8m0 [rows, {hd // 32}]")
            elif want_rows is not None and ow[1][0] != want_rows:
                rep.err(f"engram layer {lid}: {ow[1][0]} rows in the official shards, config says {want_rows}")
        elif edir and os.path.exists(os.path.join(edir, wf)):
            ws, ss = os.path.getsize(os.path.join(edir, wf)), (os.path.getsize(os.path.join(edir, sf))
                                                               if os.path.exists(os.path.join(edir, sf)) else None)
            if ss is None:
                rep.err(f"engram layer {lid}: {sf} missing")
                continue
            rows = ss // sb
            if ss % sb or ws != rows * wb:
                rep.err(f"engram layer {lid}: {wf} is {ws} bytes, {sf} {ss}: not {wb} + {sb} bytes per row")
            if want_rows is not None and rows != want_rows:
                rep.err(f"engram layer {lid}: {rows} rows in the files, config says {want_rows}")
            if egc.get("rows") and i < len(egc["rows"]) and egc["rows"][i] != rows:
                rep.err(f"engram layer {lid}: {rows} rows in the files, quantization_config.engram says {egc['rows'][i]}")
        elif lid in in_shards:
            rep.warn(f"engram layer {lid}: tables are inside the shards (older layout)")
            h = heads[f"layers.{lid}.engram.embed.weight"][1]
            hs = heads.get(f"layers.{lid}.engram.embed.scale")
            if hs is None:
                rep.err(f"engram layer {lid}: scale tensor missing")
            elif want_rows is not None and h["shape"][0] != want_rows:
                rep.err(f"engram layer {lid}: {h['shape'][0]} rows in the shard, config says {want_rows}")
        else:
            rep.err(f"engram layer {lid}: {wf} not found in {edir}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pack")
    ap.add_argument("--plan", help="plan.json (default PACK/plan.json when present)")
    ap.add_argument("--engram-dir")
    ap.add_argument("--no-mtp", action="store_true", help="ignore the dspark stages")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    rep = validate(a.pack, a.plan, a.engram_dir, mtp=not a.no_mtp)
    if a.json:
        print(json.dumps(rep.as_dict(), indent=2))
    else:
        for e in rep.errors:
            print(f"ERROR   {e}")
        for w in rep.warnings:
            print(f"warning {w}")
        print(f"{'OK' if rep.ok else 'FAILED'}: {len(rep.errors)} errors, {len(rep.warnings)} warnings; "
              + ", ".join(f"{k}={v}" for k, v in rep.stats.items()))
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
