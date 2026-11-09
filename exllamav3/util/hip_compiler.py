"""One resolver for the hipcc that builds every JIT HIP kernel (qsa_pf, gr_mix, gdn_fused_h, kda_fused_h, ple).

Order: $EXL3_HIPCC, $EXL3_ROCM_SDK/bin/hipcc, the rocm-sdk pip package of the running interpreter (found without any
environment variable), hipcc on PATH, /opt/rocm/bin/hipcc. The SDK compiler (clang 23) builds prefill kernels about
20 % faster than the system ROCm 7.2.4 compiler (clang 22), so a user who forgot the export must not silently get
the slow build. The path and `--version` string are recorded once per process; `key()` is a short hash of the
version string that the kernel cache names include, so a binary built by another compiler is never reused.
No torch import here.
"""
import glob, hashlib, os, platform, re, shutil, subprocess, sys, sysconfig

FIX_LINE = "export EXL3_ROCM_SDK=$(rocm-sdk path --root)"
SLOW_COST = "prefill can be about 20 % slower"
_info = {}
_gcc = {}


def _sdk_from_package():
    """Root of the rocm-sdk devel tree installed next to the running interpreter, or None. Cheap, never raises."""
    try:
        roots = {sysconfig.get_paths().get(k) for k in ("purelib", "platlib")}
        roots |= {p for p in sys.path if p.endswith("site-packages")}
        for r in sorted(x for x in roots if x):
            d = os.path.join(r, "_rocm_sdk_devel")
            if os.path.isfile(os.path.join(d, "bin", "hipcc")):
                return d
        tool = os.path.join(os.path.dirname(sys.executable), "rocm-sdk")
        if os.path.isfile(tool):
            out = subprocess.run([tool, "path", "--root"], capture_output=True, text=True, timeout=20).stdout.strip()
            if out and os.path.isfile(os.path.join(out, "bin", "hipcc")):
                return out
    except Exception:  # noqa: BLE001
        pass
    return None


def _inside(path, root):
    if not path or not root:
        return False
    p, r = os.path.realpath(path), os.path.realpath(root)
    return p == r or p.startswith(r + os.sep)


def _version(path):
    try:
        r = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=30)
        return " ".join((r.stdout or r.stderr).split())
    except Exception as e:  # noqa: BLE001
        return f"unknown ({type(e).__name__})"


def resolve():
    """dict(path, version, key, sdk, sdk_ok, source), or path None when no hipcc exists. Cached per process."""
    if _info:
        return _info
    env_sdk = os.environ.get("EXL3_ROCM_SDK")
    found_sdk = _sdk_from_package()
    cands = [("EXL3_HIPCC", os.environ.get("EXL3_HIPCC")),
             ("EXL3_ROCM_SDK", os.path.join(env_sdk, "bin", "hipcc") if env_sdk else None),
             ("rocm-sdk package", os.path.join(found_sdk, "bin", "hipcc") if found_sdk else None),
             ("PATH", shutil.which("hipcc")), ("/opt/rocm", "/opt/rocm/bin/hipcc")]
    path, source = None, None
    for src, p in cands:
        if p and os.path.isfile(p):
            path, source = p, src
            break
    sdk = env_sdk if env_sdk and os.path.isdir(env_sdk) else found_sdk
    ver = _version(path) if path else ""
    _info.update(path=path, source=source, version=ver, sdk=sdk,
                 sdk_ok=bool(path) and (_inside(path, found_sdk) or _inside(path, env_sdk)),
                 key=hashlib.sha1(ver.encode()).hexdigest()[:8] if path else "nocc")
    return _info


def hipcc():
    p = resolve()["path"]
    if not p:
        raise RuntimeError("hipcc not found (set EXL3_HIPCC, or: " + FIX_LINE + ")")
    return p


def key():
    """Short hash of the compiler version string, for kernel cache keys."""
    return resolve()["key"]


def _gcc_version_key(path):
    return tuple(int(x) for x in re.findall(r"\d+", os.path.basename(path)))


def gcc_install_dir():
    """GCC installation directory for clang's --gcc-install-dir, or None when there is none.

    EXL3_GCC_INSTALL_DIR when it names a directory, else the GCC on PATH, else the newest GCC tree
    installed without the driver (slim images). The scan prefers a host-triplet tree so a cross
    compiler under /usr/lib/gcc/<triple>/<version> cannot win on version alone. Cached per process:
    the flag is part of every kernel cache tag, and the kernel loaders call this once per arch."""
    if "dir" in _gcc:
        return _gcc["dir"]
    env = os.environ.get("EXL3_GCC_INSTALL_DIR")
    if env:
        if os.path.isdir(env):
            _gcc["dir"] = env
            return env
        if not _gcc.get("warned"):
            _gcc["warned"] = True
            print(f" -- EXL3_GCC_INSTALL_DIR={env!r} is not a directory; probing gcc", flush=True)
    try:
        out = subprocess.run(["gcc", "-print-libgcc-file-name"], capture_output=True, text=True, timeout=10).stdout.strip()
        d = os.path.dirname(out) if out else ""
        if d and os.path.isdir(d):
            _gcc["dir"] = d
            return d
    except Exception:  # noqa: BLE001
        pass
    cands = []
    for pattern in ("/usr/lib/gcc/*/*", "/usr/lib64/gcc/*/*", "/usr/lib/gcc-cross/*/*"):
        cands += [d for d in glob.glob(pattern) if os.path.isfile(os.path.join(d, "libgcc.a"))]
    host = platform.machine()
    host_cands = [d for d in cands if host and host in os.path.dirname(d)]
    found = max(host_cands or cands, key=_gcc_version_key) if cands else None
    _gcc["dir"] = found
    return found


def gcc_flags():
    """--gcc-install-dir flag for the JIT compilers, or [] when no GCC installation is found."""
    d = gcc_install_dir()
    return [f"--gcc-install-dir={d}"] if d else []


def describe():
    i = resolve()
    if not i["path"]:
        return "HIP kernel compiler: none found"
    return f"HIP kernel compiler: {i['path']} (from {i['source']}): {i['version'][:160]}"


def warning():
    """One line when the kernels are not built by the rocm-sdk compiler the install guide names, else None."""
    i = resolve()
    if i["sdk_ok"]:
        return None
    what = "no hipcc found" if not i["path"] else f"kernels are built with {i['path']}, not the rocm-sdk compiler"
    return f"WARNING: {what}; {SLOW_COST}. Fix: {FIX_LINE} (then restart)"


def report(log=print):
    """Start-up lines for the servers and bench: the compiler, plus the warning when it is the slow one."""
    log("[kyojin] " + describe())
    w = warning()
    if w:
        log("[kyojin] " + w)


def reset():
    _info.clear()
    _gcc.clear()
