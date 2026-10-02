#!/usr/bin/env python
"""Ad-hoc, device-free verification of the GLM dec2 wiring. NOT a test suite and NOT suite-green.

This checks the invariants the pytest suite cannot see: that every K the Python gates admit has a
kernel instantiation behind it, that CB 1 really is the mcg codebook in all three dispatches, that
the four touched entry points still accept their pre-change positional arity (the trailing-default
claim), and that the decode probe's own bindings got the same defaults.

What it does NOT cover, and must not be read as covering: fused-vs-reference numeric agreement, the
discriminating-power checks, determinism, and the MLA projection numerics. Those are in
tests/test_glm_dec2.py and need a device.

    PYTHONPATH=<repo> python tests/check_glm_dec2_wiring.py     # exit 0 = all checks pass
"""
import os
import re
import subprocess
import sys

ROOT = os.environ.get("EXL3_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXT = os.path.join(ROOT, "exllamav3", "exllamav3_ext")
fails = []


def check(name, ok, detail = ""):
    print(("PASS  " if ok else "FAIL  ") + name + (("   [" + str(detail) + "]") if detail else ""))
    if not ok:
        fails.append(name)

def read(p):
    with open(p, encoding = "utf8") as f:
        return f.read()


def k2_of(ks):
    return {int(round(k * 2)) for k in ks}


mpw = read(os.path.join(EXT, "quant", "exl3_moe_prefill.cu"))
gemv = read(os.path.join(EXT, "quant", "exl3_dec.cu"))
cbk = read(os.path.join(EXT, "quant", "codebook.cuh"))
bsm = read(os.path.join(ROOT, "exllamav3", "modules", "block_sparse_mlp.py"))
mla = read(os.path.join(ROOT, "exllamav3", "modules", "mla_attn.py"))
qexl3 = read(os.path.join(ROOT, "exllamav3", "modules", "quant", "exl3.py"))
probe = read(os.path.join(ROOT, "tools", "decode", "devext", "bind.cpp"))

print("== 1. dispatch tables vs the Python gates ==")

# mpw_gemm's K switch, sliced out of the file so a stray case elsewhere cannot pass this
mpw_cases = {int(c) for c in re.findall(r"case (\d+):", mpw.split("void launch_gemm_cb")[1]
                                       .split("// CB 1 = mcg")[0])}
# 4,5,6,8 = K 2,2.5,3,4; 7 = K 3.5 (the HALF variant), which no Python gate reaches
check("mpw_gemm_cb dispatch carries K*2 {4,5,6,7,8}", mpw_cases == {4, 5, 6, 7, 8}, sorted(mpw_cases))

m = re.search(r"multi\.K in \(([\d.,\s]+)\)", bsm)
wmma_k2 = k2_of([float(x) for x in m.group(1).split(",")])
check("wmma gate's K set is covered by the dispatch", wmma_k2 <= mpw_cases,
      f"gate {sorted(wmma_k2)}")

gemv_cases = {int(c) for c in re.findall(r"case (\d+):", gemv.split("inline void launch_gemv_cb")[1]
                                         .split("inline void launch_gemv(")[0])}
check("dense gemv dispatch is exactly K*2 {4,5,6,8,10,12}", gemv_cases == {4, 5, 6, 8, 10, 12},
      sorted(gemv_cases))
# _DEC_KB2 is already in K*2 units -- no k2_of here
prod_k2 = {int(x) for x in re.search(r"_DEC_KB2 = \(([^)]+)\)", qexl3).group(1).split(",")}
mla_k2 = {int(x) for x in re.search(r"_DEC_KB2 = \(([^)]+)\)", mla).group(1).split(",")}
check("quant/exl3.py's _DEC_KB2 == the dispatch", prod_k2 == gemv_cases, sorted(prod_k2))
check("mla_attn's _DEC_KB2 == quant/exl3.py's (the wiring accepts no K the kernel lacks)",
      mla_k2 == prod_k2, sorted(mla_k2))

print("== 2. codebook convention (1 = mcg, 2 = mul1), both kernels ==")

check("codebook.cuh: decode_3inst_2's cb 1 is mcg (0xCBAC1FED then decode_mcg_product_2)",
      bool(re.search(r"x0 \*= 0xCBAC1FEDu;\s*x1 \*= 0xCBAC1FEDu;\s*return decode_mcg_product_2",
                     cbk)))

for name, src, fn in (("mpw launch_gemm", mpw, "void launch_gemm\n"), ("dense launch_gemv", gemv,
                                                                      "inline void launch_gemv(")):
    body = src.split(fn)[1].split("\n}")[0] if fn in src else ""
    check(f"{name}: mcg -> CB 1, else CB 2",
          "<OUT_FP32, 1>" in body and "<OUT_FP32, 2>" in body if "OUT_FP32" in body
          else "cb<1>" in body and "cb<2>" in body, body.strip().splitlines()[0][:70])

check("mpw_gemm_kernel takes CB and forwards it to dq_dispatch",
      "int CB = 2>" in mpw.split("void mpw_gemm_kernel")[0].split("template <")[-1]
      or "int CB = 2" in mpw and "dq_dispatch<bits, CB, HALF>" in mpw)
check("dense gemv_kernel takes CB and forwards it to lane_gemv/block_reduce",
      "int CB = 2>" in gemv.split("void gemv_kernel")[0].split("template <int KB2")[-1]
      and "lane_gemv<KB2, CB>" in gemv and "block_reduce<CB>" in gemv)

print("== 3. the probe extension's bindings got the same defaults ==")

for fn, n in (("gemv", 8), ("gemv_multi", 8), ("gemv_strided", 10)):
    blk = re.search(r'm\.def\("' + fn + r'"(.*?)\);', probe, re.S).group(1)
    args = re.findall(r'py::arg\("([^"]+)"\)\s*(=\s*[^,)]+)?', blk)
    check(f"devext/bind.cpp {fn}: {n} required args + a defaulted mcg",
          len(args) == n + 1 and args[-1][0] == "mcg" and args[-1][1].strip() == "= false",
          f"{len(args)} args, last {args[-1]}")

print("== 4. the built extension: signatures, defaults, positional compatibility ==")

import torch  # noqa: E402
from exllamav3.ext import exllamav3_ext as ext  # noqa: E402

docs = {n: (getattr(ext, n).__doc__ or "") for n in
        ("exl3_dec_gemv", "exl3_dec_gemv_multi", "exl3_dec_gemv_strided", "exl3_moe_prefill_wmma")}
check("exl3_dec_gemv exposes mcg: bool = False", "mcg: bool = False" in docs["exl3_dec_gemv"])
check("exl3_dec_gemv_multi exposes mcg: bool = False", "mcg: bool = False" in docs["exl3_dec_gemv_multi"])
check("exl3_dec_gemv_strided exposes mcg: bool = False",
      "mcg: bool = False" in docs["exl3_dec_gemv_strided"])
check("exl3_moe_prefill_wmma exposes act_limit + mcg defaults",
      re.search(r"act_limit: [^,]+ = 0\.0", docs["exl3_moe_prefill_wmma"]) is not None and
      "mcg: bool = False" in docs["exl3_moe_prefill_wmma"])

# A wrong arity is a TypeError from pybind *before* any kernel is reached; a right one gets past
# the binding and fails later (CPU tensors here). So: accepted arity => no TypeError. The negative
# cases prove the probe can tell the difference at all.
def binds(fn, args):
    try:
        fn(*args)
    except TypeError:
        return False
    except Exception:
        return True
    return True


def tensors(device = "cpu"):
    t64 = torch.zeros(1, dtype = torch.long, device = device)
    th = torch.zeros(1, dtype = torch.half, device = device)
    tf = torch.zeros(1, dtype = torch.float, device = device)
    ti = torch.zeros(1, dtype = torch.int32, device = device)
    return t64, th, tf, ti


t64, th, tf, ti = tensors()
gemv_old = (th, th, th, th, th, th, ti, 3.0)
check("exl3_dec_gemv still accepts its pre-change 8 positional args",
      binds(ext.exl3_dec_gemv, gemv_old))
check("exl3_dec_gemv rejects a short call (the probe can tell)", not binds(ext.exl3_dec_gemv,
                                                                         gemv_old[:-1]))
check("exl3_dec_gemv_multi still accepts its pre-change 8 positional args",
      binds(ext.exl3_dec_gemv_multi, (th, [th], [th], [th], [th], th, ti, [3.0])))
check("exl3_dec_gemv_strided still accepts its pre-change 10 positional args",
      binds(ext.exl3_dec_gemv_strided, (th, th, th, th, th, th, ti, 3.0, 512, 0)))
mpw_old = (th, tf, t64, tf, t64, t64) + (t64,) * 18
check("exl3_moe_prefill_wmma still accepts its pre-change 24 positional args",
      binds(ext.exl3_moe_prefill_wmma, mpw_old),
      "24 args = the pre-change signature")
check("exl3_moe_prefill_wmma rejects 23 args (the probe can tell)",
      not binds(ext.exl3_moe_prefill_wmma, mpw_old[:-1]))

print("== 5. the real suite is well-formed ==")

# pytest is not installed in the venv, so collection goes through the runner
r = subprocess.run([sys.executable, os.path.join(ROOT, "tests", "run_glm_dec2.py"),
                    "--collect-only", "-q"],
                   capture_output = True, text = True,
                   env = dict(os.environ, PYTHONPATH = ROOT, EXL3_REPO = ROOT))
out = (r.stdout or "") + (r.stderr or "")
# pytest --collect-only -q prints "<file>: <n>" for the file, or "<n> tests collected"
n_tests = (re.search(r"(\d+) tests? collected", out) or
           re.search(r"tests/test_glm_dec2\.py: (\d+)", out))
check("tests/test_glm_dec2.py collects with no import error",
      r.returncode == 0 and n_tests is not None and int(n_tests.group(1)) == 45,
      f"rc={r.returncode}, {n_tests.group(1) if n_tests else 'no count'} tests")

print()
print(f"{len(fails)} failed" + (": " + ", ".join(fails) if fails else ""))
print("NOT covered here (needs the device): fused-vs-reference numeric agreement, the")
print("discriminating-power checks, determinism, and the MLA projection numerics.")
sys.exit(1 if fails else 0)
