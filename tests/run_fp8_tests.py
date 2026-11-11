"""Run tests/test_linear_fp8.py without pytest (the engine venv has none): every parametrized case, one line per failure."""
import itertools, sys, traceback, types
sys.modules.setdefault("pytest", types.SimpleNamespace(
    mark=types.SimpleNamespace(skipif=lambda *a, **k: (lambda f: f), parametrize=lambda *a, **k: (lambda f: f)),
    skip=lambda msg: (_ for _ in ()).throw(RuntimeError("SKIP " + msg))))
sys.path.insert(0, "tests")
import test_linear_fp8 as T

ok = bad = skip = 0
for grid, (N, K), R in itertools.product([False, True], [(256, 512), (300, 768), (5120, 17408), (1000, 1000)], [1, 3, 16, 40]):
    try:
        T.test_linear_fp8_matches_dequant_matmul(grid, N, K, R); ok += 1
    except RuntimeError as e:
        if str(e).startswith("SKIP"):
            skip += 1
        else:
            bad += 1; print("FAIL", grid, N, K, R, repr(e)[:200])
    except Exception as e:
        bad += 1; print("FAIL", grid, N, K, R, repr(e)[:200])
try:
    T.test_e4m3_decode_is_exact_on_every_code(); ok += 1
except Exception:
    bad += 1; print("FAIL e4m3", traceback.format_exc()[-400:])
print(f"{ok} passed, {bad} failed, {skip} skipped")
