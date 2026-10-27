#!/usr/bin/env python3
"""Pack a populated Triton cache into the shipped kernel seed.

  1. Start the server with an EMPTY cache and the portable cache keys, without installing the old seed:
         EXL3_KERNEL_SEED=build TRITON_CACHE_DIR=/scratch/tcache  (and HOME on a scratch dir) ...
  2. python tools/kernel_seed/sweep.py --url http://127.0.0.1:8731 --model <id>
  3. python tools/kernel_seed/build.py /scratch/tcache [/scratch/tcache2 ...]
Run steps 1-2 once per server family (same or separate cache dirs; build.py merges several) to get one seed for qwen + glm. Writes
exllamav3/kernel_seed/triton-<triton version>-<arch>.tar.xz (replace, do not merge: rebuild from the full set).
The cache keys of a seed built without EXL3_KERNEL_SEED=build hold the install prefix of the build machine: they would not hit
anywhere else (see exllamav3/util/kernel_seed.py). pack() strips the debug sections of the code objects with llvm-objcopy (ROCm
toolchain; LLVM_OBJCOPY overrides the lookup), writes a placeholder for the library paths in the metadata, and fails when a member
still holds a local path.
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from exllamav3.util import kernel_seed   # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cache", nargs="+", help="populated Triton cache directories (one per server family; a key in several: first wins)")
    ap.add_argument("--arch", default=None, help="default: the architecture of the GPU in this machine, else gfx1151")
    ap.add_argument("--triton-version", default=None)
    a = ap.parse_args()
    if a.arch is None:
        a.arch = kernel_seed.device_arch() or "gfx1151"
    if a.triton_version is None:
        import triton
        a.triton_version = triton.__version__
    out = kernel_seed.SEED_DIR / kernel_seed.seed_name(a.triton_version, a.arch)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = kernel_seed.pack([Path(c) for c in a.cache], out)
    print(f"{n} kernels -> {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
