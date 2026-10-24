"""Check numerical error, buffer bounds, dispatch, and speed on a ROCm GPU."""

from pathlib import Path
import gc, importlib.util, json, os, statistics, sys
import argparse
parser = argparse.ArgumentParser(description="Check dense K8 projections with real weights.")
parser.add_argument("--model", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--bits", type=int, default=8,
                    help="K of the projections to check. 8 checks the five K8 shapes. Another value is a baseline run of the "
                         "existing direct path on dense projections of that K, picked with --match.")
parser.add_argument("--match", default="",
                    help="with --bits other than 8: substring of the module keys to check, for example indexer.index_qk_proj")
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
out = Path(args.output)
out.mkdir(parents=True, exist_ok=False)
sys.path.insert(0, str(root))
spec = importlib.util.find_spec("exllamav3_ext")
assert spec and Path(spec.origin).resolve().parent == root, "Wrong extension"
print("Extension:", spec.origin, flush=True)
os.environ["EXL3_DENSE_K8"] = "1"
from tools.qwen.serve import SERVE_ENV
for key, value in SERVE_ENV:
    os.environ.setdefault(key, value)
import torch
from exllamav3 import Config, Model
from exllamav3.modules.linear import Linear
from exllamav3.modules.quant import exl3 as q
from exllamav3.ext import exllamav3_ext as ext
torch.set_grad_enabled(False)
torch.set_num_threads(4)
assert torch.version.hip
assert os.environ.get("EXL3_VERIFY_GEMV_R") == "1"
assert os.environ.get("EXL3_VERIFY_ROW_SPLIT") == "2"
dev = torch.device("cuda:0")
print("GPU:", torch.cuda.get_device_name(dev), flush=True)
cfg = Config.from_directory(args.model)
models = [Model.from_config(cfg), Model.from_config(cfg, component="mtp")]
if args.bits == 8:
    groups = {shape: [] for shape in q._DENSE_K8_SHAPES}
else:
    assert args.match, "--bits other than 8 needs --match"
    groups = {}
for model in models:
    for module in model:
        if isinstance(module, Linear) and (".self_attn." in module.key or ".linear_attn." in module.key):
            shape = (module.in_features, module.out_features)
            if args.bits == 8:
                if shape in groups:
                    groups[shape].append(module)
            elif args.match in module.key:
                groups.setdefault(shape, []).append(module)
assert groups and all(groups.values()), "Not all projection shapes were found"
selected = []
for shape in sorted(groups):
    members = groups[shape]
    selected.extend([members[i] for i in sorted({0, len(members)-1})])

# Guard the direct kernel's scratch and counter buffers.
scratch = torch.full((q.DEC_SCRATCH_FLOATS+256,), 12345., device=dev)
counters = torch.full((4096+256,), 12345, dtype=torch.int32, device=dev)
scratch[128:-128].zero_(); counters[128:-128].zero_()
q._dec_workspaces[str(dev)] = (scratch[128:-128], counters[128:-128])
def guards():
    assert bool((scratch[:128] == 12345).all()) and bool((scratch[-128:] == 12345).all()), "Scratch overrun"
    assert bool((counters[:128] == 12345).all()) and bool((counters[-128:] == 12345).all()), "Counter overrun"
    assert int(counters[128:-128].abs().sum()) == 0, "Counters not reset"

records = []
flush = torch.zeros(64 << 20, device=dev)
for module in selected:
    print("Loading:", module.key, flush=True)
    module.load(dev)
    try:
        inner = module.inner
        if getattr(inner, "K", None) != args.bits:   # only reachable for --bits other than 8 (fp16 or other-K projections)
            assert args.bits != 8, f"{module.key} is not K8"
            print("Skipping (not K%d):" % args.bits, module.key, flush=True)
            continue
        assert inner.mul1 and not inner.mcg
        assert inner.dec_ok and inner.bc is None
        if args.bits == 8:
            os.environ["EXL3_DENSE_K8"] = "0"
            assert not q.dec_supported(inner), "Opt-out guard failed"
            os.environ["EXL3_DENSE_K8"] = "1"
        calls = [0, 0]
        originals = [ext.exl3_dec_gemv, ext.exl3_dec_gemv_r]
        def spy(index):
            def invoke(*a, **kw):
                calls[index] += 1
                return originals[index](*a, **kw)
            return invoke
        ext.exl3_dec_gemv, ext.exl3_dec_gemv_r = spy(0), spy(1)
        quality = []
        try:
            for seed in (1234, 5678):
                gen = torch.Generator(device=dev).manual_seed(seed)
                base = torch.randn((9, inner.in_features), generator=gen, device=dev)
                if seed == 1234:
                    base = base.half()
                else:
                    base *= 8
                    base[1:8:2] = base[0:8:2].clone()
                base = base.half()  # Both paths require FP16 input; output dtype varies.
                for rows in range(1, 9):
                    x = base[:rows].contiguous()
                    saved = x.clone()
                    for dtype in (torch.float16, torch.float32):
                        before = sum(calls)
                        ref = inner.forward(x, {"reconstruct": True}, dtype).clone()
                        assert sum(calls) == before, "Reference used candidate kernel"
                        before = calls.copy()
                        got = inner.forward(x, {"dflash_verify": True}, dtype).clone()
                        assert calls[rows > 1] > before[rows > 1], "Candidate dispatch missing"
                        repeat = inner.forward(x, {"dflash_verify": True}, dtype).clone()
                        torch.cuda.synchronize()
                        assert bool(torch.isfinite(ref).all()) and bool(torch.isfinite(got).all())
                        assert torch.equal(got, repeat), "Candidate not repeatable"
                        assert torch.equal(x, saved), "Input modified"
                        guards()
                        error = got.double() - ref.double()
                        quality.append({"rows": rows, "seed": seed, "dtype": str(dtype),
                            "exact": torch.equal(got, ref),
                            "max_abs": error.abs().max().item(),
                            "rel_l2": (error.norm()/ref.double().norm().clamp_min(1e-30)).item()})
            # Larger batches must still use the existing path. For K8 that path is reconstruct + hgemm, so it matches the
            # reference bit for bit. For other K the existing path can be the HIP GEMV, so only the dispatch is checked.
            before = sum(calls)
            ref9 = inner.forward(base, {"reconstruct": True}).clone()
            got9 = inner.forward(base, {"dflash_verify": True}).clone()
            torch.cuda.synchronize()
            assert sum(calls) == before, "Nine-row fallback used the direct kernel"
            assert args.bits != 8 or torch.equal(ref9, got9), "Nine-row fallback changed"
            guards()
        finally:
            ext.exl3_dec_gemv, ext.exl3_dec_gemv_r = originals
        worst = max(quality, key=lambda r: r["rel_l2"])
        print("CHECK:", json.dumps({"key": module.key, "comparisons": len(quality),
            "exact": sum(r["exact"] for r in quality), "worst_relative_error": worst}), flush=True)
        record = {"key": module.key, "quality": quality, "bench": []}
        records.append(record)
        (out/"results.json").write_text(json.dumps(records, indent=2))

        # Alternate order; flush GPU cache outside each timed measurement.
        for rows in (1, 4, 8):
            x = base[:rows].contiguous()
            def calculate(candidate):
                return inner.forward(x, {"dflash_verify": True} if candidate else {"reconstruct": True}, torch.float32)
            for _ in range(3):
                calculate(False); calculate(True)
            torch.cuda.synchronize()
            samples = {False: [], True: []}
            for trial in range(7):
                for candidate in ((False, True) if trial % 2 == 0 else (True, False)):
                    flush.add_(1)
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    result = calculate(candidate)
                    end.record(); end.synchronize()
                    samples[candidate].append(start.elapsed_time(end))
            old, new = (statistics.median(samples[k]) for k in (False, True))
            timing = {"rows": rows, "existing_ms": old, "candidate_ms": new, "speedup": old/new}
            record["bench"].append(timing)
            print("BENCH:", json.dumps(timing), flush=True)
        guards()
        (out/"results.json").write_text(json.dumps(records, indent=2))
    finally:
        module.unload()
        gc.collect()
        torch.cuda.empty_cache()
print("Structural checks passed. Review numerical differences and timings before a full-model A/B.", flush=True)
