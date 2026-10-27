"""Build + load the CPU emulator of dsv41_hip.hip and plug it into dsv41_hip._launch (tests only)."""
import ctypes, os, pathlib, subprocess

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def build():
    out = ROOT / "scratch" / "build"
    out.mkdir(parents=True, exist_ok=True)
    so = out / "dsv41_emu.so"
    deps = [HERE / "emu.cpp", HERE / "hip" / "hip_runtime.h", ROOT / "exllamav3/modules/dsv41_fn/dsv41_hip.hip",
            ROOT / "exllamav3/modules/dsv41_fn/dsv41_num.h"]
    if not so.exists() or so.stat().st_mtime < max(p.stat().st_mtime for p in deps):
        subprocess.run(["g++", "-std=gnu++20", "-O1", "-ffp-contract=off", "-fPIC", "-shared", "-pthread", "-w",
                        f"-I{HERE}", "-o", str(so), str(HERE / "emu.cpp")], check=True)
    return ctypes.CDLL(str(so))


def install(dsv41_hip):
    lib = build()

    def emu(name, grid, block, params):
        f = getattr(lib, "emu_" + name)
        f(int(grid[0]), int(grid[1]), int(grid[2]), int(block), ctypes.cast(params, ctypes.c_void_p))

    dsv41_hip._state["emu"] = emu
