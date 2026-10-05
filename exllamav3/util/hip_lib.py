# Find torch's bundled HIP runtime. Stable wheels ship libamdhip64.so; nightly ROCm wheels ship only versioned
# names (libamdhip64.so.7, libamdhip64.so.7.1.2 ...). Plain name first, then the versioned ones, highest first.

import ctypes, glob, os, re, sys

_NAME = "libamdhip64.so"
_logged = set()


def _version_key(path: str):
    return tuple(int(x) for x in re.findall(r"\d+", os.path.basename(path)[len(_NAME):]))


def find_hip_runtime(lib_dir: str | None = None) -> str | None:
    """Path of libamdhip64 inside lib_dir (default: <torch>/lib), or None when there is none."""
    if lib_dir is None:
        import torch
        lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
    plain = os.path.join(lib_dir, _NAME)
    if os.path.isfile(plain):
        return plain
    versioned = [p for p in glob.glob(os.path.join(lib_dir, _NAME + ".*")) if os.path.isfile(p)
                 and re.fullmatch(r"(\.\d+)+", os.path.basename(p)[len(_NAME):])]
    if not versioned:
        return None
    best = max(versioned, key = _version_key)
    if best not in _logged:
        _logged.add(best)
        print(f"hip_lib: {_NAME} not found in {lib_dir}, using {os.path.basename(best)}", file = sys.stderr, flush = True)
    return best


def load_hip_runtime(lib_dir: str | None = None) -> ctypes.CDLL:
    path = find_hip_runtime(lib_dir)
    if path is None:
        raise OSError(f"no {_NAME} or {_NAME}.<version> in {lib_dir or 'the torch lib directory'}")
    return ctypes.CDLL(path)
