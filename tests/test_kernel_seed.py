"""Kernel seed: portable HIP cache keys, install() and pack() (CPU only; needs a Triton wheel with the AMD backend)."""
import json, os, subprocess, sys, tempfile
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from hardware import require
require("ext")
from exllamav3.util import kernel_seed

REPO = str(Path(__file__).resolve().parents[1])

CHILD = r'''
import hashlib, json, sys
sys.path.insert(0, sys.argv[1])
import triton, triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource, make_backend
from triton.compiler.compiler import get_cache_key
sys.path.insert(0, sys.argv[2])
from exllamav3.util import kernel_seed

@triton.jit
def k(x_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(x_ptr + i, tl.load(x_ptr + i, mask=i < n) + 1, mask=i < n)

def key():
    backend = make_backend(GPUTarget("hip", "gfx1151", 32))
    src = ASTSource(k, {"x_ptr": "*fp32", "n": "i32"}, {"BLOCK": 128})
    return hashlib.sha256(get_cache_key(src, backend, backend.parse_options({"num_warps": 4}), {}).encode()).hexdigest()

raw = key()
ok = kernel_seed.portable_keys("gfx1151")
print(json.dumps({"raw": raw, "ok": ok, "portable": key()}))
'''


def _skip(msg):
    try:
        import pytest
        pytest.skip(msg)
    except ImportError:
        print("skip:", msg)


def _have_amd_triton() -> bool:
    try:
        from triton.backends.amd import compiler  # noqa: F401
        return True
    except Exception:
        return False


def _farm(src: Path, dst: Path):
    """dst/triton: symlinks to the real package, except the directories on the way to backends/amd/compiler.py, which are real
    (the file is copied): the same Triton under another install prefix."""
    def rec(s: Path, d: Path, rel=()):
        d.mkdir(parents=True, exist_ok=True)
        for e in s.iterdir():
            r = rel + (e.name,)
            if r == ("backends", "amd", "compiler.py"):
                (d / e.name).write_bytes(e.read_bytes())
            elif r in (("backends",), ("backends", "amd")):
                rec(e, d / e.name, r)
            else:
                (d / e.name).symlink_to(e)
    rec(src, dst / "triton")


def test_cache_key_is_the_same_under_another_install_prefix():
    if not _have_amd_triton():
        return _skip("no Triton with the AMD backend")
    import triton
    real = Path(triton.__file__).parent
    outs = []
    with tempfile.TemporaryDirectory() as td:
        sites = [None, Path(td) / "one" / "site", Path(td) / "second" / "other" / "site"]
        for site in sites[1:]:
            site.mkdir(parents=True)
            _farm(real, site)
        child = Path(td) / "child.py"
        child.write_text(CHILD)
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        for site in sites:
            r = subprocess.run([sys.executable, str(child), str(site or "/nonexistent"), REPO], capture_output=True, text=True, env=env)
            assert r.returncode == 0, r.stderr[-2000:]
            outs.append(json.loads(r.stdout.strip().splitlines()[-1]))
    assert all(o["ok"] for o in outs)
    assert len({o["raw"] for o in outs}) == 3          # Triton's own key depends on the prefix ...
    assert len({o["portable"] for o in outs}) == 1     # ... the seed's does not


def test_scrub_json_and_local_path_check():
    meta = json.dumps({"extern_libs": [["ocml", "/opt/x/.venv/lib/python3.12/site-packages/triton/backends/amd/lib/ocml.bc"],
                                       ["ockl", "/home/u/v/triton/backends/amd/lib/ockl.bc"]], "name": "k"}).encode()
    out = kernel_seed._scrub_json(meta)
    assert json.loads(out)["extern_libs"] == [["ocml", kernel_seed.LIB_PLACEHOLDER + "/ocml.bc"], ["ockl", kernel_seed.LIB_PLACEHOLDER + "/ockl.bc"]]
    assert not kernel_seed._LOCAL_PATH.search(out)
    assert kernel_seed._LOCAL_PATH.search(b"x /home/someone/repo/file.py y")
    assert kernel_seed._LOCAL_PATH.search(b"C:\\Users\\me")
    assert not kernel_seed._LOCAL_PATH.search(b"amdgcn-amd-amdhsa--gfx1151 exllamav3/modules/attention.py")


def test_pack_refuses_a_member_with_a_local_path():
    with tempfile.TemporaryDirectory() as td:
        cache, kdir = Path(td) / "cache", Path(td) / "cache" / "KEY"
        kdir.mkdir(parents=True)
        (kdir / "k.autotune.json").write_text(json.dumps({"key": [1], "path": "/home/someone/x"}))
        try:
            kernel_seed.pack(cache, Path(td) / "out.tar.xz")
        except RuntimeError as exc:
            assert "local path" in str(exc)
        else:
            raise AssertionError("pack() accepted a local path")


def test_pack_merges_caches_first_wins():
    import tarfile
    with tempfile.TemporaryDirectory() as td:
        a, b = Path(td) / "a", Path(td) / "b"
        for root, keys in ((a, ("K1", "K2")), (b, ("K2", "K3"))):
            for k in keys:
                (root / k).mkdir(parents=True)
                (root / k / "f.autotune.json").write_text(json.dumps({"from": root.name}))
        out = Path(td) / "o.tar.xz"
        assert kernel_seed.pack([a, b], out) == 3
        with tarfile.open(out) as tf:
            got = {m.name: json.load(tf.extractfile(m))["from"] for m in tf.getmembers()}
        assert got == {"K1/f.autotune.json": "a", "K2/f.autotune.json": "a", "K3/f.autotune.json": "b"}


def test_install_other_architecture_and_off_switch_do_nothing():
    with tempfile.TemporaryDirectory() as td:
        for arch in ("gfx1100", "sm_89"):
            assert kernel_seed.install(arch, cache = Path(td) / "c") == 0
        os.environ["EXL3_KERNEL_SEED"] = "0"
        try:
            assert kernel_seed.install("gfx1151", cache = Path(td) / "c") == 0
        finally:
            del os.environ["EXL3_KERNEL_SEED"]
        assert not (Path(td) / "c").exists()


def test_shipped_seed_has_no_local_path_and_installs_with_this_prefix():
    if not _have_amd_triton():
        return _skip("no Triton with the AMD backend")
    seed = kernel_seed.find_seed("gfx1151")
    if seed is None:
        return _skip("no seed for this Triton version")
    import tarfile
    libdir = kernel_seed._amd_libdir()
    with tarfile.open(seed, "r:xz") as tf:
        members = [m for m in tf.getmembers() if m.isfile()]
        assert members
        for m in members:
            data = tf.extractfile(m).read()
            assert not kernel_seed._LOCAL_PATH.search(data), m.name
        keys = {m.name.split("/")[0] for m in members}
    with tempfile.TemporaryDirectory() as td:
        cache = Path(td) / "cache"
        assert kernel_seed.install("gfx1151", cache = cache, log = lambda s: None) == len(keys)
        assert kernel_seed.install("gfx1151", cache = cache, log = lambda s: None) == 0     # never overwrites
        for j in cache.glob("*/*.json"):
            if j.name.startswith("__grp__") or j.name.endswith(".autotune.json"):
                continue
            text = j.read_text()
            assert kernel_seed.LIB_PLACEHOLDER not in text
            for lib in json.loads(text)["extern_libs"]:
                assert lib[1].startswith(libdir)
        assert kernel_seed.portable_keys("gfx1151")


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
