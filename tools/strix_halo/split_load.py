#!/usr/bin/env python3
"""Phase 3 loader gate: load the GLM pack across iGPU + eGPU and print the table.

No forward pass is run.  Usage (from the repo root, after `source tools/strix_halo/env.sh`):

    python tools/strix_halo/split_load.py \
        -m /data/hf/hub/models--NicSir--GLM-5.3-Flash-EXL3-2.72bpw/snapshots/db672fc5e61f1d61bdf6d641fb63538b80bb7613 \
        --front-layers 4 --endcaps on -c 65536
"""
from __future__ import annotations

import argparse
import os
import resource
import time


def mem_available_gib() -> float:
    try:
        for line in open("/proc/meminfo", encoding="ascii"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 ** 2
    except OSError:
        pass
    return 0.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--model", required=True)
    p.add_argument("--front-layers", type=int, default=4)
    p.add_argument("--front-device", default="auto")
    p.add_argument("--endcaps", choices=("on", "off"), default="on")
    p.add_argument("--front-reserve-gb", type=float, default=1.2)
    p.add_argument("-c", "--cache-size", type=int, default=65536)
    p.add_argument("--num-draft", type=int, default=2)
    p.add_argument("--no-vision", action="store_true")
    p.add_argument("--baseline", action="store_true", help="load everything on the main device, for the host-RAM comparison")
    args = p.parse_args()

    import torch
    from exllamav3 import Cache, CacheLayer_quant, Config, Model
    from exllamav3.device_map import LayerDeviceMap, print_split_table, resolve_devices

    torch.set_grad_enabled(False)
    endcaps = args.endcaps == "on"
    host_before = mem_available_gib()

    config = Config.from_directory(args.model)
    model = Model.from_config(config)

    # Same cache format as tools/glm/serve.py defaults: TurboQuant 6-bit latent.
    qkw = dict(layer_type=CacheLayer_quant, k_bits=6, v_bits=6)
    from exllamav3.cache.tq import CacheLayer_tq, supported as tq_supported
    if tq_supported(512, 6):
        qkw = dict(layer_type=CacheLayer_tq, k_bits=6, v_bits=6)
    max_history = max(args.num_draft, 1)
    cache = Cache(model, max_num_tokens=args.cache_size + 4096,
                  max_history=max_history, max_batch_size=1, **qkw)

    if args.baseline:
        t0 = time.perf_counter()
        model.load(device="cuda:0", progressbar=True)
        draft = Model.from_config(config, component="mtp") if args.num_draft > 0 else None
        if draft is not None:
            draft.load(device="cuda:0", progressbar=True)
        vision = None
        if not args.no_vision:
            try:
                vision = Model.from_config(config, component="vision")
                vision.load(device="cuda:0", progressbar=True)
            except Exception as exc:                    # noqa: BLE001
                print(f"[baseline] vision not loaded: {type(exc).__name__}: {exc}", flush=True)
        print(f"[baseline] load {time.perf_counter() - t0:.1f} s; "
              f"main allocated {torch.cuda.memory_allocated(torch.device('cuda:0')) / 2**30:.2f} GiB; "
              f"peak {torch.cuda.max_memory_allocated(torch.device('cuda:0')) / 2**30:.2f} GiB", flush=True)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
        print(f"[baseline] host RSS peak {peak:.2f} GiB; "
              f"MemAvailable {host_before:.1f} -> {mem_available_gib():.1f} GiB", flush=True)
        return

    devices = resolve_devices(args.front_device)
    text_map = LayerDeviceMap.from_model(model, args.front_layers, endcaps, devices,
                                         component="text", front_reserve_gb=args.front_reserve_gb)
    if not text_map.enabled:
        raise SystemExit("split map is not enabled (no front device)")
    print(f"[split] {text_map.describe()}", flush=True)
    print(f"[split] main {devices.main} ({devices.main_arch}), "
          f"front {devices.front} ({devices.front_arch})", flush=True)

    t0 = time.perf_counter()
    model.load(device_map=text_map, progressbar=True, max_chunk_size=4096)
    t_model = time.perf_counter() - t0

    draft = draft_cache = None
    if args.num_draft > 0:
        draft = Model.from_config(config, component="mtp")
        draft_cache = Cache(draft, max_num_tokens=args.cache_size + 4096,
                            max_history=max_history, max_batch_size=1, **qkw)
        draft_map = LayerDeviceMap.from_model(draft, 0, endcaps, devices,
                                              component="mtp", front_reserve_gb=args.front_reserve_gb)
        draft_map.reserve_bytes.clear()                 # text_map owns the budget
        t1 = time.perf_counter()
        draft.load(device_map=draft_map, progressbar=True, max_chunk_size=4096)
        print(f"[split] drafter loaded in {time.perf_counter() - t1:.1f} s", flush=True)

    vision = None
    if not args.no_vision:
        try:
            vision = Model.from_config(config, component="vision")
            vdev = devices.front if endcaps else devices.main
            t1 = time.perf_counter()
            vision.load(device=vdev, progressbar=True)
            print(f"[split] vision loaded in {time.perf_counter() - t1:.1f} s on {vdev}", flush=True)
        except Exception as exc:                        # noqa: BLE001
            print(f"[split] vision not loaded: {type(exc).__name__}: {exc}", flush=True)
            vision = None

    extra = []
    if draft is not None:
        extra.append(("mtp", draft, draft_cache))
    if vision is not None:
        extra.append(("vision", vision, None))
    print_split_table(model, cache, text_map, extra=extra, ctx=args.cache_size)
    text_map.release_budget()

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
    print(f"[split] target load {t_model:.1f} s; host RSS peak {peak:.2f} GiB; "
          f"MemAvailable {host_before:.1f} -> {mem_available_gib():.1f} GiB", flush=True)


if __name__ == "__main__":
    main()
