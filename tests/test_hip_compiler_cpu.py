"""CPU tests of the shared JIT compiler resolver (exllamav3/util/hip_compiler.py) and of its use by the HIP loaders.
Fake hipcc scripts on a temp tree; no GPU, no ROCm, no built extension (the package __init__ files are bypassed
with empty namespace stubs while the modules are imported, then removed)."""
import importlib, os, stat, sys, types
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
_PKGS = ["exllamav3", "exllamav3.util", "exllamav3.vendor", "exllamav3.vendor.fla", "exllamav3.vendor.fla.hip",
         "exllamav3.modules", "exllamav3.modules.ple_fn", "gr"]
_added = []
for _n in _PKGS:
    if _n not in sys.modules:
        _m = types.ModuleType(_n); _m.__path__ = [str(ROOT.joinpath(*_n.split(".")))]
        sys.modules[_n] = _m; _added.append(_n)
try:
    hip_compiler = importlib.import_module("exllamav3.util.hip_compiler")
    LOADERS = {n: importlib.import_module(n) for n in (
        "gr.gr_mix_hip", "exllamav3.vendor.fla.hip.gdn_fused_h_hip", "exllamav3.vendor.fla.hip.kda_fused_h_hip",
        "exllamav3.modules.ple_fn.ple_hip")}
finally:
    for _n in _added:
        sys.modules.pop(_n, None)
_REAL_SDK_FROM_PACKAGE = hip_compiler._sdk_from_package


def fake_hipcc(path: Path, version: str, log: Path | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#!/bin/bash\n"
        f'if [ "$1" = "--version" ]; then echo "{version}"; exit 0; fi\n'
        + (f'echo "$0" >> {log}\n' if log else "")
        + 'while [ $# -gt 0 ]; do if [ "$1" = "-o" ]; then echo built > "$2"; fi; shift; done\n')
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


@pytest.fixture
def env(tmp_path, monkeypatch):
    for k in ("EXL3_HIPCC", "EXL3_ROCM_SDK"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PATH", str(tmp_path / "emptybin"))
    monkeypatch.setattr(hip_compiler, "_sdk_from_package", lambda: None)
    hip_compiler.reset()
    yield tmp_path
    hip_compiler.reset()


def test_order_env_hipcc_first(env, monkeypatch):
    a = fake_hipcc(env / "a" / "hipcc", "clang A")
    sdk = env / "sdk"; fake_hipcc(sdk / "bin" / "hipcc", "clang SDK")
    pkg = env / "pkg"; fake_hipcc(pkg / "bin" / "hipcc", "clang PKG")
    fake_hipcc(env / "pathbin" / "hipcc", "clang PATH")
    monkeypatch.setenv("PATH", str(env / "pathbin"))
    monkeypatch.setattr(hip_compiler, "_sdk_from_package", lambda: str(pkg))
    monkeypatch.setenv("EXL3_HIPCC", str(a)); monkeypatch.setenv("EXL3_ROCM_SDK", str(sdk))
    assert hip_compiler.resolve()["path"] == str(a)
    hip_compiler.reset(); monkeypatch.delenv("EXL3_HIPCC")
    assert hip_compiler.resolve()["path"] == str(sdk / "bin" / "hipcc")
    hip_compiler.reset(); monkeypatch.delenv("EXL3_ROCM_SDK")
    r = hip_compiler.resolve()
    assert r["path"] == str(pkg / "bin" / "hipcc") and r["source"] == "rocm-sdk package" and r["sdk_ok"]
    hip_compiler.reset(); monkeypatch.setattr(hip_compiler, "_sdk_from_package", lambda: None)
    r = hip_compiler.resolve()
    assert r["path"] == str(env / "pathbin" / "hipcc") and r["source"] == "PATH" and not r["sdk_ok"]


def test_package_sdk_found_without_env(env, monkeypatch):
    # a fake site-packages with _rocm_sdk_devel next to the interpreter's purelib
    sp = env / "site-packages"; fake_hipcc(sp / "_rocm_sdk_devel" / "bin" / "hipcc", "clang SDK")
    monkeypatch.setattr(hip_compiler, "_sdk_from_package", _REAL_SDK_FROM_PACKAGE)
    import sysconfig
    monkeypatch.setattr(sysconfig, "get_paths", lambda *a, **k: {"purelib": str(sp), "platlib": str(sp)})
    monkeypatch.setattr(sys, "path", [str(sp)] + sys.path)
    hip_compiler.reset()
    if True:
        assert hip_compiler.resolve()["path"] == str(sp / "_rocm_sdk_devel" / "bin" / "hipcc")
        assert hip_compiler.warning() is None


def test_none_found_is_safe(env, monkeypatch):
    monkeypatch.setattr(os.path, "isfile", lambda p: False)
    assert hip_compiler.resolve()["path"] is None
    assert "no hipcc found" in hip_compiler.warning()
    with pytest.raises(RuntimeError, match="hipcc not found"):
        hip_compiler.hipcc()


def test_key_depends_on_version(env, monkeypatch):
    a = fake_hipcc(env / "a" / "hipcc", "clang version 22")
    b = fake_hipcc(env / "b" / "hipcc", "clang version 23")
    monkeypatch.setenv("EXL3_HIPCC", str(a)); ka = hip_compiler.key()
    hip_compiler.reset(); monkeypatch.setenv("EXL3_HIPCC", str(b)); kb = hip_compiler.key()
    assert ka != kb and len(ka) == len(kb) == 8
    assert "clang version 23" in hip_compiler.describe() and str(b) in hip_compiler.describe()


def test_version_recorded_once(env, monkeypatch):
    log = env / "calls.log"
    a = fake_hipcc(env / "a" / "hipcc", "clang X")
    a.write_text(a.read_text().replace('echo "clang X"', f'echo called >> {log}; echo "clang X"'))
    monkeypatch.setenv("EXL3_HIPCC", str(a))
    for _ in range(3):
        hip_compiler.key(); hip_compiler.describe(); hip_compiler.warning()
    assert log.read_text().count("called") == 1


def test_warning_text(env, monkeypatch):
    sdk = env / "sdk"
    fake_hipcc(sdk / "bin" / "hipcc", "clang 23")
    sysc = fake_hipcc(env / "opt" / "bin" / "hipcc", "clang 22")
    monkeypatch.setenv("EXL3_ROCM_SDK", str(sdk))
    assert hip_compiler.warning() is None
    hip_compiler.reset(); monkeypatch.delenv("EXL3_ROCM_SDK"); monkeypatch.setenv("EXL3_HIPCC", str(sysc))
    w = hip_compiler.warning()
    assert w.count("\n") == 0 and "export EXL3_ROCM_SDK=$(rocm-sdk path --root)" in w and "about 20 % slower" in w
    lines = []
    hip_compiler.report(lines.append)
    assert len(lines) == 2 and "HIP kernel compiler" in lines[0] and "WARNING" in lines[1]


@pytest.mark.parametrize("modname,call", [
    ("gr.gr_mix_hip", lambda m: m.compile_hsaco()),
    ("exllamav3.vendor.fla.hip.gdn_fused_h_hip", lambda m: m._compile("", "gfx1151")),
    ("exllamav3.vendor.fla.hip.kda_fused_h_hip", lambda m: m._compile("", "gfx1151")),
    ("exllamav3.modules.ple_fn.ple_hip", lambda m: m.compile_hsaco()),
])
def test_cache_key_includes_compiler(env, monkeypatch, modname, call):
    m = LOADERS[modname]
    log = env / "calls.log"
    a = fake_hipcc(env / "a" / "hipcc", "clang version 22", log)
    b = fake_hipcc(env / "b" / "hipcc", "clang version 23", log)
    monkeypatch.setenv("EXL3_HIPCC", str(a))
    pa = call(m)
    assert os.path.isfile(pa) and log.read_text().count("\n") == 1
    call(m)                                           # same compiler: cache hit, no second build
    assert log.read_text().count("\n") == 1
    # an "old" cache file (name without the compiler hash) is never reused: it is not the file the loader asks for
    hip_compiler.reset(); monkeypatch.setenv("EXL3_HIPCC", str(b))
    pb = call(m)
    assert pb != pa and log.read_text().count("\n") == 2 and open(pa).read() == "built\n"
    assert os.path.isfile(pa), "old files are not deleted"


def test_old_cache_name_not_reused(env, monkeypatch):
    import hashlib
    m = LOADERS["gr.gr_mix_hip"]
    log = env / "calls.log"
    a = fake_hipcc(env / "a" / "hipcc", "clang version 22", log)
    monkeypatch.setenv("EXL3_HIPCC", str(a))
    # recompute the pre-fix name (no compiler hash) and plant a stale slow binary there
    src = open(m._SRC, "rb").read()
    flags = ["--genco", "--offload-arch=gfx1151", "-O3", "-ffp-contract=off"]
    gcc = os.environ.get("EXL3_GCC_INSTALL_DIR", "/usr/lib/gcc/x86_64-linux-gnu/13")
    if os.path.isdir(gcc): flags.append(f"--gcc-install-dir={gcc}")
    old = Path(os.path.expanduser("~/.cache/exllamav3")) / f"gr_mix_gfx1151_{hashlib.sha1(src + ' '.join(flags).encode()).hexdigest()[:16]}.hsaco"
    old.parent.mkdir(parents=True); old.write_text("slow")
    got = m.compile_hsaco()
    assert Path(got) != old and log.exists() and old.read_text() == "slow"


def test_servers_wired():
    for f in ("qwen", "glm", "mimo"):
        s = (ROOT / "tools" / f / "serve.py").read_text()
        assert "startup_health.report_compiler()" in s and "startup_health.compiler_health()" in s, f
    assert "hip_compiler.report" in (ROOT / "tools/strix_halo/env.sh").read_text()
    assert "Kernel compiler" in (ROOT / "tools/bench.py").read_text()


def test_compiler_health_fields(env, monkeypatch):
    sys.path.insert(0, str(ROOT / "tools"))
    import startup_health
    for n in ("exllamav3", "exllamav3.util"):
        m = types.ModuleType(n); m.__path__ = [str(ROOT.joinpath(*n.split(".")))]
        monkeypatch.setitem(sys.modules, n, m)
    monkeypatch.setitem(sys.modules, "exllamav3.util.hip_compiler", hip_compiler)
    monkeypatch.setattr(sys.modules["exllamav3.util"], "hip_compiler", hip_compiler, raising=False)
    sysc = fake_hipcc(env / "opt" / "bin" / "hipcc", "clang 22")
    monkeypatch.setenv("EXL3_HIPCC", str(sysc))
    h = startup_health.compiler_health()
    assert "clang 22" in h["hip_compiler"] and "20 %" in h["hip_compiler_warning"]


def test_every_jit_site_uses_the_resolver():
    sites = ["gr/gr_mix_hip.py", "exllamav3/vendor/fla/hip/gdn_fused_h_hip.py", "exllamav3/vendor/fla/hip/kda_fused_h_hip.py",
             "exllamav3/modules/ple_fn/ple_hip.py", "exllamav3/modules/attention_fn/qsa_prefill_hip.py"]
    for f in sites:
        s = (ROOT / f).read_text()
        assert "hip_compiler.hipcc()" in s and "hip_compiler.key()" in s and "EXL3_ROCM_SDK" not in s, f
    import subprocess
    out = subprocess.run(["grep", "-rl", "--include=*.py", "-e", "--genco", str(ROOT / "exllamav3"), str(ROOT / "gr")],
                         capture_output=True, text=True).stdout.split()
    assert sorted(Path(o).relative_to(ROOT).as_posix() for o in out) == sorted(sites)
