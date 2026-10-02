#!/usr/bin/env python
"""HIP-graph capture of the batch-1 decode step (torch.cuda.CUDAGraph on ROCm / gfx1151).

Why this is capturable at all: the generator builds the decode step's params from *persistent*
pinned staging buffers (`Generator._staging`: batch_ids, cache_seqlens, positions, block_index,
generator.py:931-942) that are rewritten in place every step, and the K/V cache tensors never
move. So every batch-1 step issues the same kernel sequence against the same addresses, and the
only thing that changes between steps is the *content* of the staging buffers -- which the
captured H2D copies re-read on replay.

Design (mirrors exllamav3/modules/block_graph.py, which does the same thing one block at a time
but only for GatedDeltaNet blocks and therefore declines every MiMo block):

    warmup  -> N real calls, outputs used
    capture -> records without executing (torch.cuda.graph), then one replay commits that step
    replay  -> graph.replay() for every following batch-1 call

Anything that is not a single-row decode call (prefill, verification windows) passes straight
through to the original forward, and any exception during capture disables the graph for the rest
of the run instead of poisoning it.

Enable with --graph 1 in tools/decode/dev.py.
"""
from __future__ import annotations

import torch


class StepGraph:

    def __init__(self, model, warmups: int = 3, verbose: bool = False, bisect = None):
        self.model = model
        self.orig = model.forward
        self.warmups_left = max(1, warmups)
        self.verbose = verbose
        self.bisect = bisect          # list of fwd_modules prefixes to try, smallest first
        self.graph = None
        self.out = None
        self.state = "warmup"
        self.installed = False
        self.last_module = None
        self._traced = []
        self.stats = {
            "passthrough": 0, "warmups": 0, "captures": 0, "replays": 0,
            "capture_error": None, "decline_after": 0,
        }

    # -- install / uninstall ---------------------------------------------------

    def install(self):
        if not self.installed:
            self.model.forward = self
            self.installed = True
            # Per-module tracer. The Python side still executes while the stream is capturing, so
            # the last module entered before a capture error names the offending op.
            self.last_module = None
            self._traced = []
            for entry in getattr(self.model, "fwd_modules", []):
                module, instance, idx = entry[0], entry[1], entry[2]
                orig = module.forward
                self._traced.append((module, orig))
                module.forward = self._tracer(module, orig, idx)

    def _tracer(self, module, orig, idx):
        def f(*a, **k):
            self.last_module = f"{idx}:{type(module).__name__}"
            return orig(*a, **k)
        return f

    def uninstall(self):
        if self.installed:
            for module, orig in self._traced:
                module.forward = orig
            self._traced = []
            self.model.forward = self.orig
            self.installed = False

    def summary(self) -> str:
        s = self.stats
        txt = (f"graph {self.state}: warmups {s['warmups']} captures {s['captures']} "
               f"replays {s['replays']} passthrough {s['passthrough']}")
        if s["capture_error"]:
            txt += f" CAPTURE FAILED: {s['capture_error']}"
        return txt

    # -- forward replacement ---------------------------------------------------

    def __call__(self, input_ids, params = None):
        params = {} if params is None else params

        # Only the single-row decode path is graphed. Prefill chunks and draft verification
        # windows have a different launch sequence, and a job that switches between them would
        # replay a stale graph.
        rows = int(input_ids.shape[0]) * (int(input_ids.shape[1]) if input_ids.dim() > 1 else 1)
        if rows != 1 or self.state == "off":
            self.stats["passthrough"] += 1
            return self.orig(input_ids, params)

        if self.state == "warmup":
            self.stats["warmups"] += 1
            out = self.orig(input_ids, params)
            self.warmups_left -= 1
            if self.warmups_left <= 0:
                self.state = "ready"
            return out

        if self.state == "ready":
            if self.bisect:
                self._run_bisect(input_ids, params)
                self.bisect = None
                self.state = "off"      # bisect only; the plain path resumes below
                return self.orig(input_ids, params)
            return self._capture(input_ids, params)

        # state == "replay"
        try:
            self.graph.replay()
        except Exception as e:
            self.stats["decline_after"] += 1
            self.state = "off"
            self.stats["capture_error"] = f"replay: {type(e).__name__}: {e}"
            return self.orig(input_ids, params)
        self.stats["replays"] += 1
        return self.out

    # -- bisect ----------------------------------------------------------------

    def _call_truncated(self, input_ids, params, scope):
        """One real call over only the first `scope` modules of fwd_modules."""
        saved = self.model.fwd_modules
        self.model.fwd_modules = saved[:scope]
        try:
            return self.orig(input_ids, params)
        finally:
            self.model.fwd_modules = saved

    def _run_bisect(self, input_ids, params):
        """Capture+replay the step over growing module prefixes, smallest first, printing each
        result before the next attempt. The first prefix whose replay faults names the module
        that carries the non-capturable op (a fault poisons the device, so the run ends there
        and the last printed line is the answer)."""
        import traceback
        for scope in self.bisect:
            mods = self.model.fwd_modules
            tail = mods[min(scope, len(mods)) - 1]
            print(f"[bisect] scope {scope}/{len(mods)} (last module {tail[2]}:{type(tail[0]).__name__})",
                  flush = True)
            try:
                for _ in range(2):
                    self._call_truncated(input_ids, params, scope)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, capture_error_mode = "thread_local"):
                    self._call_truncated(input_ids, params, scope)
                torch.cuda.synchronize()
                g.replay()
                torch.cuda.synchronize()
                print(f"[bisect] scope {scope}: capture+replay OK", flush = True)
            except Exception as e:
                print(f"[bisect] scope {scope}: FAILED {type(e).__name__}: "
                      f"{str(e).splitlines()[0]}", flush = True)
                print(traceback.format_exc(), flush = True)
                raise

    # -- capture ---------------------------------------------------------------

    def _capture(self, input_ids, params):
        self.state = "capturing"
        try:
            torch.cuda.synchronize()
            # torch.cuda.graph requires the ops to have run once on a side stream first
            # (allocator + kernel module warmup); a duplicate call at the same cache position
            # rewrites the same K/V slots with the same values, so it is harmless.
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                self.orig(input_ids, params)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            side = None

            g = torch.cuda.CUDAGraph()
            # thread_local, not the default global: with global, one non-capturable op inside the
            # step invalidates every stream on the device and the whole run dies with an
            # asynchronous "illegal memory access" at the next sync, which says nothing about the
            # real cause. thread_local raises at the offending op and leaves the device usable,
            # so a decline costs one step instead of the run.
            with torch.cuda.graph(g, capture_error_mode = "thread_local"):
                out = self.orig(input_ids, params)
            torch.cuda.synchronize()
            self.graph = g
            self.out = out
            self.state = "replay"
            self.stats["captures"] += 1
            # The capture itself did not execute: replay once to commit this step for real.
            g.replay()
            torch.cuda.synchronize()
            self.stats["replays"] += 1
            return self.out
        except Exception as e:
            import traceback
            self.graph = None
            self.out = None
            self.state = "off"
            self.stats["capture_error"] = f"{type(e).__name__}: {e} (last module {self.last_module})"
            print(f"[step_graph] capture declined at module {self.last_module}:\n{traceback.format_exc()}", flush = True)
            try:
                torch.cuda.synchronize()
            except Exception as e2:
                print(f"[step_graph] device still poisoned after decline: {e2}", flush = True)
                raise
            return self.orig(input_ids, params)
