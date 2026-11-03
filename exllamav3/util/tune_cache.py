"""Per-pack tuning cache for the EXL3 GEMM autotuner (day-0 autotune, qualify --stage autotune).

The cooperative GEMM autotuner (exllamav3_ext/quant/coop_autotune.cu) times its candidate kernels on the first call of every
(row class, K, N, bits, dtype, device, cc, SMs, codebook) and appends the winner to a disk cache (EXLLAMAV3_TUNE_CACHE, default
~/.cache/exllamav3/autotune/coop_autotune_v1.bin). A model with new shapes pays that timing inside its first requests / graph warm-up.
The qualify autotune stage runs the tuner for every shape of a pack ahead of time and ships the winners next to the pack:

    <pack>/tuning/<signature>.bin      same record format as the live cache (32-byte records, header EX3ATUNE)
    <pack>/tuning/<signature>.json     manifest: device, sweep, timings, engine version
    <pack>/tuning/<signature>.dense.txt  ROCm only: dense-GEMM rocBLAS tune lines (EXL3_DENSE_GEMM_TUNE_FILE format)

install_pack(model_dir) merges the file whose signature matches this device into the live cache before the first GEMM
(records already present win; nothing is overwritten). A record is validated against the live candidates by the C++ loader
(tag, block_dim, SM count, concurrency), and the key is salted with COOP_AUTOTUNE_VERSION, so a stale or foreign file only misses.
EXL3_PACK_TUNE=0 turns the merge off (cold-cache A/B).
"""
from __future__ import annotations
import json, os, re, struct
from pathlib import Path

MAGIC = b"EX3ATUNE"
FORMAT = 1
REC = struct.Struct("<Q4i2I")          # hash, tag, block_dim, num_sms, concurrency, reserved x2 (DiskCacheRecordV1)
HDR = struct.Struct("<8sII")           # magic, format, record_size
TUNING_DIR = "tuning"


def live_cache_path() -> Path:
    """Same resolution as disk_cache_path() in coop_autotune.cu (POSIX)."""
    o = os.environ.get("EXLLAMAV3_TUNE_CACHE")
    if o:
        p = Path(o)
        return p / "coop_autotune_v1.bin" if p.is_dir() else p
    base = os.environ.get("XDG_CACHE_HOME") or str(Path(os.environ.get("HOME", "/tmp")) / ".cache")
    return Path(base) / "exllamav3" / "autotune" / "coop_autotune_v1.bin"


def read_records(path) -> dict:
    """hash -> record bytes. Tolerates an embedded header (two first writers) and a torn tail, like the C++ reader."""
    p = Path(path)
    if not p.is_file():
        return {}
    b = p.read_bytes()
    if len(b) < HDR.size:
        return {}
    magic, fmt, rsz = HDR.unpack_from(b, 0)
    if magic != MAGIC or fmt != FORMAT or rsz < REC.size:
        return {}
    out, off = {}, HDR.size
    while off + rsz <= len(b):
        chunk = b[off:off + rsz]
        if chunk[:8] == MAGIC:                     # embedded header: skip it, keep reading records after it
            off += HDR.size
            continue
        out[struct.unpack_from("<Q", chunk)[0]] = chunk[:REC.size]
        off += rsz
    return out


def write_records(path, recs: dict) -> None:
    """Atomic rewrite (tmp + rename), header first."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    tmp.write_bytes(HDR.pack(MAGIC, FORMAT, REC.size) + b"".join(recs.values()))
    os.replace(tmp, p)


def decode(rec: bytes) -> dict:
    h, tag, bd, sms, conc, _, _ = REC.unpack(rec)
    return dict(hash=f"{h:016x}", tag=tag, block_dim=bd, num_sms=sms, concurrency=conc)


def signature(device: int | None = None) -> dict:
    """What a tuned record depends on besides its (salted) key: the device model, compute capability and SM/CU count."""
    import torch
    d = torch.cuda.current_device() if device is None else device
    pr = torch.cuda.get_device_properties(d)
    arch = getattr(pr, "gcnArchName", None) if torch.version.hip else None      # CUDA builds expose the attribute too (device name)
    cc = arch.split(":")[0] if arch else f"sm{pr.major}{pr.minor}"
    slug = re.sub(r"[^a-z0-9]+", "-", pr.name.lower()).strip("-")
    return dict(name=pr.name, cc=cc, sms=pr.multi_processor_count, device_index=d,
                backend="rocm" if arch else "cuda", file=f"{cc}-{pr.multi_processor_count}sm-{slug}")


def pack_files(model_dir, sig: dict | None = None) -> list[Path]:
    d = Path(model_dir) / TUNING_DIR
    if not d.is_dir():
        return []
    sig = sig or signature()
    p = d / f"{sig['file']}.bin"
    return [p] if p.is_file() else []


def merge_into(dst, srcs) -> int:
    """Add the records of srcs missing from dst (never overwrites). Returns the number added."""
    have = read_records(dst)
    new = {}
    for s in srcs:
        for k, v in read_records(s).items():
            if k not in have and k not in new:
                new[k] = v
    if not new:
        return 0
    if have:                                       # append, like the C++ writer: concurrent servers keep working
        with open(dst, "ab") as f:
            f.write(b"".join(new.values()))
    else:
        write_records(dst, new)
    return len(new)


def dense_live_path() -> Path:
    """ROCm dense-GEMM tune file (hgemm.cu dtune): EXL3_DENSE_GEMM_TUNE_FILE or ~/.cache/exllamav3/dense_gemm_tune.txt."""
    o = os.environ.get("EXL3_DENSE_GEMM_TUNE_FILE")
    return Path(o) if o else Path(os.environ.get("HOME", "/tmp")) / ".cache/exllamav3/dense_gemm_tune.txt"


def merge_dense_text(dst, src) -> int:
    """Append the lines of src (7 fields: key = first 6) whose key dst lacks. Same rule as dense_warmup.seed_dense_tune."""
    key = lambda ln: tuple(ln.split()[:6])
    dst, src = Path(dst), Path(src)
    have = {key(l) for l in dst.read_text().splitlines() if len(l.split()) == 7} if dst.exists() else set()
    new = [l for l in src.read_text().splitlines() if len(l.split()) == 7 and key(l) not in have]
    if new:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "a") as f:
            f.write("\n".join(new) + "\n")
    return len(new)


def install_pack(model_dir, log=print) -> int:
    """Call before the first GEMM of the process (the C++ cache is read once, lazily). Returns records added to the live cache."""
    if os.environ.get("EXL3_PACK_TUNE", "1") == "0":
        return 0
    try:
        sig = signature()
        n = 0
        files = pack_files(model_dir, sig)
        if files:
            n = merge_into(live_cache_path(), files)
            log(f"tune cache: {n} records from {files[0].name} merged into {live_cache_path()}")
        dense = Path(model_dir) / TUNING_DIR / f"{sig['file']}.dense.txt"      # ROCm rocBLAS solution per dense shape class
        if dense.is_file():
            m = merge_dense_text(dense_live_path(), dense)
            log(f"tune cache: {m} dense-GEMM lines from {dense.name} merged into {dense_live_path()}")
            n += m
        return n
    except Exception as e:   # noqa: BLE001  a bad tuning file must never stop a serve start
        log(f"tune cache: pack cache ignored ({e})")
        return 0
