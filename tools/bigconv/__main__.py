"""bigconv CLI:  python -m bigconv {plan,sheet} ...   (run with PYTHONPATH=tools)"""
import argparse
import sys

from . import adapters  # noqa: F401  registers the adapters
from .arch import ADAPTERS


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bigconv")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="bits-per-weight plan from a tensor map")
    p.add_argument("--arch", required=True, choices=sorted(ADAPTERS))
    p.add_argument("--config", help="model config.json (default: the adapter's bundled one)")
    p.add_argument("--map", required=True)
    p.add_argument("--target", type=float, default=2.3)
    p.add_argument("--freq")
    p.add_argument("--out")
    p.add_argument("--layer-bits")
    p.add_argument("--no-engram", action="store_true")
    p.add_argument("--mtp-queue", action="store_true")
    p.add_argument("--alt-attn")
    p.add_argument("--alt-shared")
    p.add_argument("--role-bits")
    p.add_argument("--expert-bits-file")
    p.add_argument("--alloc")
    for k in ("attn", "shared", "mtp", "head", "hot", "cold", "mtp_exp"):
        d = {"attn": 5, "shared": 5, "mtp": 4, "head": 6, "hot": 3, "cold": 2, "mtp_exp": None}[k]
        p.add_argument(f"--{k.replace('_', '-')}-bits", type=int, default=d)

    s = sub.add_parser("sheet", help="resource sheet: pack size, RAM/SSD split, capture RAM/disk, time")
    s.add_argument("--arch", required=True, choices=sorted(ADAPTERS))
    s.add_argument("--index", required=True, help="model.safetensors.index.json (names + total_size)")
    s.add_argument("--config", required=True)
    s.add_argument("--map", help="tensor_map.json with shapes (exact); default: shapes derived from the config")
    s.add_argument("--calib-tokens", type=int, required=True)
    s.add_argument("--bpw", type=float, required=True, help="routed-expert average bits per weight")
    s.add_argument("--ram-gib", type=float, default=100.0, help="RAM budget for experts at serve time (SSD tier above)")
    s.add_argument("--row-len", type=int, default=2048)
    s.add_argument("--json", action="store_true")
    s.add_argument("--kron-ho", action="store_true", help="model the kron output-side Hessians in the spool (upper bound)")
    s.add_argument("--memmap", action="store_true", help="capture with the memmap design (default shows both)")

    f = sub.add_parser("preflight", help="pre-flight checks (sysctl, disk, spool budget, trim, calib, GPU wrapper exemption)")
    f.add_argument("--root", required=True)
    f.add_argument("--spool-gb", type=float, required=True, help="spool per layer (from the sheet)")
    f.add_argument("--capture-min-per-layer", type=float)
    f.add_argument("--rows", type=int)
    f.add_argument("--row-len", type=int)
    f.add_argument("--gpurun", help="path of the GPU wrapper script (idle-kill exemption check)")
    f.add_argument("--who")
    f.add_argument("--floor-gb", type=float)
    f.add_argument("--no-qlc", action="store_true")
    f.add_argument("--scratch", help="capture memmap scratch dir (must be tmpfs)")
    f.add_argument("--memmap-gb", type=float, default=0.0, help="sheet: capture disk (row state + X + Y)")

    f.add_argument("--src", help="source checkpoint dir: report the detected input format (BF16 / FP8 / MXFP4 / NVFP4)")
    f.add_argument("--map", help="tensor_map.json instead of --src (remote model)")
    f.add_argument("--config", help="config.json next to --map")

    o = sub.add_parser("formats", help="detect the source checkpoint format from headers only")
    o.add_argument("--src", help="checkpoint dir (config.json + *.safetensors)")
    o.add_argument("--map", help="tensor_map.json instead of --src")
    o.add_argument("--config", help="config.json next to --map")
    o.add_argument("--json", action="store_true")

    k = sub.add_parser("check", help="release-day arch check: new config.json + index.json against an adapter")
    k.add_argument("--arch", required=True, choices=sorted(ADAPTERS))
    k.add_argument("--config", required=True)
    k.add_argument("--index", required=True)
    k.add_argument("--json", action="store_true")

    a = ap.parse_args(argv)
    if a.cmd == "preflight":
        from . import preflight
        preflight.cli(a)
    elif a.cmd == "formats":
        from . import formats
        formats.cli(a)
    elif a.cmd == "check":
        from . import archcheck
        archcheck.cli(a)
    elif a.cmd == "plan":
        from . import plan
        plan.cli(a)
    else:
        from . import sheet
        sheet.cli(a)


if __name__ == "__main__":
    sys.exit(main())
