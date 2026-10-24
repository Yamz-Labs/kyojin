"""Pre-compiled Triton kernels shipped with the code (exllamav3/kernel_seed/triton-<version>-<arch>.tar.xz).

A fresh container has an empty Triton cache, and every kernel the servers use (and every constexpr / divisibility variant a request
reaches) is compiled on first use: 25 s during start-up plus 30 s inside the first requests of a GLM start. The compiled objects
are plain files keyed by a hash of the kernel source, its arguments and the Triton version, so they can be shipped.

Portable keys. Triton 3.7 puts the absolute path of its own bundled device libraries (backends/amd/lib/ocml.bc, ockl.bc) into the
HIP compile options and hashes it, so the same kernel gets a different cache key under every install prefix: a seed built in one
virtual environment would never hit in another (the container installs under its own prefix). portable_keys() replaces that
prefix by a placeholder inside HIPOptions.hash(); only the key changes, not what is compiled. The seed is built and used with it.

install() unpacks the members whose key directory is missing from the live cache (never overwrites) and writes the group file Triton
reads, with paths of the live cache. Only the code object and its metadata are shipped; Triton does not need the IR files to load a kernel.
The autotune results (<kernel>.autotune.json) are shipped too: without them the first call of an autotuned kernel times every
configuration (several seconds for the GLM linear-attention kernels) even when all configurations are compiled.
A seed built for another Triton version or GPU architecture is not used, and on another architecture (or without a GPU) install()
does nothing. EXL3_KERNEL_SEED=0 turns it off; EXL3_KERNEL_SEED=build uses the portable keys but installs nothing (the mode in which a
seed is built from an empty cache).
pack() builds a seed from a populated cache (tools/kernel_seed/build.py): it strips the debug sections of the code objects (they carry
the build machine's source paths), replaces the library paths in the metadata by a placeholder, and refuses a seed that still holds a
local path. install() writes the live paths back.
"""
from __future__ import annotations
import hashlib, io, json, os, re, shutil, subprocess, sys, tarfile, tempfile, time
from pathlib import Path

SEED_DIR = Path(__file__).resolve().parent.parent / "kernel_seed"
LIB_PLACEHOLDER = "@TRITON_AMD_LIB@"   # stands for <triton>/backends/amd/lib in the cache key and in the shipped metadata
_LOCAL_PATH = re.compile(rb"/(?:home|Users|root|tmp|mnt|var|opt|usr|srv)/|[A-Za-z]:\\")   # a seed must hold none of these


def _cache_dir() -> Path:
    """The directory Triton itself reads (honours TRITON_CACHE_DIR, TRITON_HOME and HOME)."""
    try:
        from triton import knobs
        if knobs.cache.dir:
            return Path(knobs.cache.dir)
    except Exception:   # noqa: BLE001
        pass
    d = os.environ.get("TRITON_CACHE_DIR")
    return Path(d) if d else Path(os.environ.get("HOME", "/tmp")) / ".triton" / "cache"


def seed_name(triton_version: str, arch: str) -> str:
    return f"triton-{triton_version}-{arch}.tar.xz"


def find_seed(arch: str = "gfx1151") -> Path | None:
    try:
        import triton
    except Exception:   # noqa: BLE001
        return None
    p = SEED_DIR / seed_name(triton.__version__, arch)
    return p if p.is_file() else None


def device_arch() -> str | None:
    """gcnArchName of the current GPU without the feature flags ("gfx1151"), None without a HIP GPU."""
    try:
        import torch
        if not (torch.version.hip and torch.cuda.is_available()):
            return None
        name = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName
        return name.split(":")[0] or None
    except Exception:   # noqa: BLE001
        return None


def _amd_libdir() -> str | None:
    try:
        from triton.backends.amd import compiler as amd
        return str(Path(amd.__file__).parent / "lib")
    except Exception:   # noqa: BLE001
        return None


def portable_keys(arch: str = "gfx1151") -> bool:
    """Make the cache key of a HIP kernel independent of the Triton install prefix. Returns True when the keys are portable.

    The original HIPOptions.hash() is sha256 of "name-value" for every option joined with "_"; the options hold the absolute paths of
    ocml.bc / ockl.bc under the Triton package. This hash is the same text with that package lib directory replaced by LIB_PLACEHOLDER.
    It is installed only after it is checked to reproduce the original on a real options object (with the directory left in), so a
    Triton whose options or hashing differ keeps its own hash and the seed is simply not used."""
    try:
        from triton.backends.amd import compiler as amd
    except Exception:   # noqa: BLE001
        return False
    cls, libdir = amd.HIPOptions, _amd_libdir()
    if getattr(cls.hash, "_exl3_portable", False):
        return True
    if not libdir:
        return False

    def text(self) -> str:
        return "_".join(f"{name}-{val}" for name, val in self.__dict__.items())

    try:
        probe = cls(arch = arch)
        if libdir not in text(probe) or hashlib.sha256(text(probe).encode("utf-8")).hexdigest() != cls.hash(probe):
            return False
    except Exception:   # noqa: BLE001
        return False

    def portable_hash(self):
        return hashlib.sha256(text(self).replace(libdir, LIB_PLACEHOLDER).encode("utf-8")).hexdigest()

    portable_hash._exl3_portable = True
    cls.hash = portable_hash
    return True


def install(arch: str | None = None, cache: Path | None = None, log=None) -> int:
    """Returns the number of kernels added to the cache. Does nothing without a HIP GPU, on an architecture without a seed,
    with EXL3_KERNEL_SEED=0, or when there is no seed for this Triton version."""
    mode = os.environ.get("EXL3_KERNEL_SEED", "1")
    if mode == "0":
        return 0
    arch = arch or device_arch()
    if not arch or not arch.startswith("gfx"):
        return 0
    seed = find_seed(arch)
    if mode != "build" and seed is None:
        return 0
    if not portable_keys(arch):
        print("kernel seed: not used (this Triton's HIP cache key could not be made portable)", file=sys.stderr, flush=True)
        return 0
    if mode == "build":
        return 0
    libdir = _amd_libdir() or ""
    cache = Path(cache) if cache else _cache_dir()
    t0 = time.perf_counter()
    added = 0
    try:
        cache.mkdir(parents=True, exist_ok=True)
        with tarfile.open(seed, "r:xz") as tf:
            by_key: dict[str, list[tarfile.TarInfo]] = {}
            for m in tf.getmembers():
                if m.isfile() and m.name.count("/") == 1 and ".." not in m.name:
                    by_key.setdefault(m.name.split("/")[0], []).append(m)
            for key, members in by_key.items():
                kdir = cache / key
                if kdir.exists():
                    continue
                tmp = cache / f".seed-{os.getpid()}-{key}"
                tmp.mkdir(parents=True, exist_ok=True)
                names = []
                for m in members:
                    data = tf.extractfile(m).read()
                    if m.name.endswith(".json"):   # the library paths of this install, as Triton would have written them
                        data = data.replace(LIB_PLACEHOLDER.encode(), libdir.encode())
                    (tmp / Path(m.name).name).write_bytes(data)
                    names.append(Path(m.name).name)
                meta = [n for n in names if n.endswith(".json") and not n.endswith(".autotune.json")]
                for n in meta:   # the group file Triton reads: kernel files with the live paths
                    group = {f: str(kdir / f) for f in names if f.startswith(n[:-5] + ".")}
                    (tmp / f"__grp__{n}").write_text(json.dumps({"child_paths": group}))
                try:
                    os.rename(tmp, kdir)
                    added += 1
                except OSError:   # another process installed it first
                    for f in tmp.iterdir():
                        f.unlink()
                    tmp.rmdir()
    except Exception as exc:   # noqa: BLE001  a seed is an optimisation: never keep the server from starting
        print(f"kernel seed: not used ({type(exc).__name__}: {exc})", file=sys.stderr, flush=True)
        return added
    (log or (lambda s: print(s, file=sys.stderr, flush=True)))(
        f"kernel seed: {added} Triton kernels from {seed.name} added to {cache} in {time.perf_counter() - t0:.1f} s")
    return added


def _llvm_objcopy() -> str:
    """llvm-objcopy of a ROCm toolchain (GNU objcopy does not know the AMDGPU ELF machine)."""
    for cand in (os.environ.get("LLVM_OBJCOPY"), shutil.which("llvm-objcopy")):
        if cand and os.path.exists(cand):
            return cand
    roots = [os.environ.get("ROCM_PATH"), os.environ.get("EXL3_ROCM_SDK"), "/opt/rocm"]
    try:
        import _rocm_sdk_devel
        roots.append(os.path.dirname(_rocm_sdk_devel.__file__))
    except Exception:   # noqa: BLE001
        pass
    for root in filter(None, roots):
        for rel in ("lib/llvm/bin/llvm-objcopy", "llvm/bin/llvm-objcopy"):
            if os.path.exists(os.path.join(root, rel)):
                return os.path.join(root, rel)
    raise RuntimeError("llvm-objcopy not found: set LLVM_OBJCOPY (it strips the debug sections of the code objects)")


def _scrub_hsaco(data: bytes, objcopy: str) -> bytes:
    """The code object without its debug sections: they hold the source path of the machine that built it."""
    with tempfile.TemporaryDirectory() as td:
        src, dst = Path(td) / "k.hsaco", Path(td) / "k.stripped.hsaco"
        src.write_bytes(data)
        subprocess.run([objcopy, "--strip-debug", str(src), str(dst)], check = True, capture_output = True)
        return dst.read_bytes()


def _scrub_json(data: bytes) -> bytes:
    """Metadata with the library paths replaced by the placeholder (install() writes the live ones back)."""
    return re.sub(rb'"[^"]*/backends/amd/lib/(ocml|ockl)\.bc"', b'"' + LIB_PLACEHOLDER.encode() + rb'/\1.bc"', data)


def pack(cache, out: Path) -> int:
    """Write the seed tar from one or several populated Triton caches (a key present in more than one: the first wins): per key
    directory the code object (debug sections stripped) and its metadata (library paths replaced). Raises when a member still
    holds a local path."""
    caches = [Path(c) for c in ([cache] if isinstance(cache, (str, Path)) else cache)]
    out = Path(out)
    objcopy = None
    n = 0
    kdirs: dict = {}
    for c in caches:
        for p in c.iterdir():
            if p.is_dir() and not p.name.startswith("."):
                kdirs.setdefault(p.name, p)
    with tarfile.open(out, "w:xz", preset=9) as tf:
        for kdir in (kdirs[k] for k in sorted(kdirs)):
            files = sorted(f for f in kdir.iterdir() if f.is_file() and (f.suffix in (".hsaco", ".json")) and not f.name.startswith("__grp__"))
            if not any(f.suffix == ".hsaco" or f.name.endswith(".autotune.json") for f in files):
                continue
            for f in files:
                data = f.read_bytes()
                if f.suffix == ".hsaco":
                    objcopy = objcopy or _llvm_objcopy()
                    data = _scrub_hsaco(data, objcopy)
                elif not f.name.endswith(".autotune.json"):
                    data = _scrub_json(data)
                bad = _LOCAL_PATH.search(data)
                if bad:
                    raise RuntimeError(f"{kdir.name}/{f.name}: still holds a local path ({data[bad.start():bad.start() + 60]!r})")
                info = tarfile.TarInfo(f"{kdir.name}/{f.name}")
                info.size, info.mtime, info.mode = len(data), 0, 0o644
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                tf.addfile(info, io.BytesIO(data))
            n += 1
    return n
