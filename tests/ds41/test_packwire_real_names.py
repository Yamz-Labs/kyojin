"""The converter's plan on the REAL V4.1-Flash tensor names yields exactly the tensors the serving loader asks for.

Tensor headers of the real checkpoint (tests/dsv41/fetch_real_headers.py) -> plan.build_plan with ramfit-style per-expert K
(K = 0 and K = 1 included) and the MTP stages -> names of the pack (EXL3 suffixes per quantized key, copied tensors) ->
diff against `packspec.expected_tensors` of the real config. Skipped when the headers were not fetched.
"""
import glob
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools", "ds41"))
sys.path.insert(0, ROOT)

import common as C  # noqa: E402
import plan as P  # noqa: E402

PS = C.packspec()
DSV41Config = C.dsv41_config().DSV41Config      # loaded by path: no CUDA extension needed
HEADERS = os.environ.get("DSV41_REAL_DIR", "./ds41_work/ds41port/scratch/real") + "/headers"
pytestmark = pytest.mark.skipif(not os.path.isdir(HEADERS), reason="real headers not fetched")


def real_map():
    T = {}
    for f in glob.glob(HEADERS + "/*.json"):
        for k, v in json.load(open(f)).items():
            if k != "__metadata__":
                a, b = v["data_offsets"]
                T[k] = {"dtype": v["dtype"], "shape": v["shape"], "length": b - a, "file": os.path.basename(f)}
    return {"tensors": T}


def pack_names(plan):
    names = {}
    for e in plan["entries"]:
        if e["action"] == "quant":
            for sfx in ("trellis", "suh", "svh", "mul1"):
                names[f"{e['key']}.{sfx}"] = e
        elif e["action"] == "copy":
            names[e["name"]] = e
    return names


def test_plan_on_real_names_matches_the_spec():
    tm = real_map()
    cfg = DSV41Config.from_file(os.path.join(ROOT, "tests", "dsv41", "data", "real_hf_config.json"))
    n = P.infer_n_experts(tm["tensors"])
    assert n == cfg.n_routed_experts
    # ramfit-style: layers 3 and 10 mix K = 0, 1, 2, 3; the others use the plan tiers
    ks = [(e * 7) % 4 for e in range(n)]
    eb = {"layers.3.ffn": ks, "layers.10.ffn": list(reversed(ks))}
    plan = P.build_plan(tm, 2.0, mtp_queue=True, skip_engram=True, expert_bits=eb, o_groups=cfg.o_groups, n_experts=n)
    got = pack_names(plan)
    want = PS.expected_tensors(cfg, exl3=True, expert_bits=eb, mtp=True)
    miss = sorted(set(want) - set(got))
    extra = sorted(set(got) - set(want))
    # tensors the pack legitimately omits from the plan: none; and the plan must not carry what the loader never reads
    # (vision / unused heads are allowed as copies only when the spec lists them as optional raws)
    assert not [m for m in miss if not want[m].optional], miss[:10]
    # the text-only loader never reads the vision tower / aligner (copied bf16 by the plan, harmless)
    assert not [x for x in extra if not x.startswith(("vision.", "aligner.", "image_"))], extra[:10]
    # K per expert in the plan equals the spec's expectation, pruned experts have no tensors
    assert P.expert_bits_from_entries(plan["entries"], n)["layers.3.ffn"] == ks
    for e, k in enumerate(ks):
        assert (f"layers.3.ffn.experts.{e}.w1.trellis" in got) == (k != 0)
    # trellis shapes from the plan's K
    for name, ent in got.items():
        if name.endswith(".trellis") and want[name].shape is not None:       # K known to the spec: expert_bits
            assert want[name].shape == (ent["in"] // 16, ent["out"] // 16, 16 * ent["bits"]), name
    for ent in plan["entries"]:
        if ent["action"] == "quant":
            assert ent["out_bytes"] == PS.exl3_bytes(ent["in"], ent["out"], ent["bits"]), ent["key"]
    # pruned experts are masked on the router bias entries
    bias = [e for e in plan["entries"] if e.get("name") == "layers.3.ffn.gate.bias"][0]
    assert bias["mask_experts"] == [e for e, k in enumerate(ks) if k == 0]
