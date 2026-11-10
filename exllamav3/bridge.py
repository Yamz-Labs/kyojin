from __future__ import annotations

"""Async host-staged bridge for the contiguous layer split.

The iGPU (gfx1151) and the eGPU (gfx1201) have no working, fast peer path: a direct
device-to-device copy on this link class is both slow (under 1 GB/s observed on the
9070 side) and asymmetric.  Every segment crossing therefore goes through pinned host
memory:

    src tensor --D2H--> pinned slot --H2D--> dst tensor

Both legs are enqueued asynchronously on the two devices' current compute streams, so
the host thread never waits for the copy.  An event carries the dependency from the
D2H leg to the H2D leg; a second event per ring slot lets the next hop reuse a pinned
slot without overwriting data that the destination has not read yet.  The destination
tensor is a fresh allocation, so the caller may keep using it after the slot cycles.

``hop`` is the generic hidden-state crossing and ``hop_chunk`` the prefill-sized
crossing.  Both share the same double-buffered channels (one channel per device
pair, one slot ring per dtype).

Set ``KYOJIN_BRIDGE=0`` to fall back to ``util.device_copy.to_device``; the split then
still runs, but every crossing is a blocking or direct copy.
"""

import os
import torch


_ROUND = 2   # pinned slots per (device pair, dtype): reuse is stream-ordered, never host-blocking


class _Slot:
    __slots__ = ("buf", "cap", "filled", "read", "read_valid")

    def __init__(self):
        self.buf = None
        self.cap = 0
        self.filled = None
        self.read = None
        self.read_valid = False


class _Channel:
    __slots__ = ("src", "dst", "slots", "next")

    def __init__(self, src: torch.device, dst: torch.device):
        self.src = src
        self.dst = dst
        self.slots: dict[torch.dtype, list[_Slot]] = {}
        self.next = 0


class Bridge:

    def __init__(self, main: torch.device, front: torch.device):
        self.main = torch.device(main)
        self.front = torch.device(front)
        self.channels: dict[tuple[int, int], _Channel] = {}
        self.stats = {"hops": 0, "bytes": 0, "chunk_hops": 0, "chunk_bytes": 0, "reallocs": 0}

    # -- configuration -----------------------------------------------------

    @property
    def pair(self) -> tuple[torch.device, torch.device]:
        return self.main, self.front

    def describe(self) -> str:
        return (f"host-staged async bridge {self.main} <-> {self.front} "
                f"(pinned slots x{_ROUND})")

    def summary(self) -> str:
        return (f"[split] bridge: {self.stats['hops']} hops, "
                f"{self.stats['bytes'] / 2**20:.1f} MiB total "
                f"({self.stats['chunk_hops']} chunk hops, "
                f"{self.stats['chunk_bytes'] / 2**20:.1f} MiB)")

    # -- public API --------------------------------------------------------

    def hop(self, t: torch.Tensor, dst: torch.device) -> torch.Tensor:
        """Move ``t`` to ``dst`` through pinned host staging, asynchronously."""
        return self._hop(t, dst, chunk = False)

    def hop_chunk(self, t: torch.Tensor, dst: torch.device) -> torch.Tensor:
        """Prefill-sized crossing. Double-buffered like every hop; reported separately."""
        return self._hop(t, dst, chunk = True)

    # -- implementation ----------------------------------------------------

    def _channel(self, src: torch.device, dst: torch.device) -> _Channel:
        key = (src.index or 0, dst.index or 0)
        ch = self.channels.get(key)
        if ch is None:
            ch = self.channels[key] = _Channel(src, dst)
        return ch

    def _slot(self, ch: _Channel, dtype: torch.dtype) -> _Slot:
        slots = ch.slots.get(dtype)
        if slots is None:
            slots = ch.slots[dtype] = [_Slot() for _ in range(_ROUND)]
        slot = slots[ch.next % _ROUND]
        ch.next = (ch.next + 1) % _ROUND
        return slot

    def _drain(self, ch: _Channel) -> None:
        """A slot is about to be reallocated: let both devices finish using it first.
        Only happens while the pinned pool grows, i.e. once on the first prefill chunk."""
        torch.cuda.synchronize(ch.src)
        torch.cuda.synchronize(ch.dst)

    def _grow(self, ch: _Channel, slot: _Slot, dtype: torch.dtype, numel: int) -> None:
        self._drain(ch)
        slot.buf = torch.empty(numel, dtype = dtype, pin_memory = True)
        slot.cap = numel
        slot.read_valid = False
        slot.filled = torch.cuda.Event()
        slot.read = torch.cuda.Event()
        self.stats["reallocs"] += 1

    def _hop(self, t: torch.Tensor, dst: torch.device, chunk: bool) -> torch.Tensor:
        dst = torch.device(dst)
        if t.device == dst:
            return t
        if t.device.type != "cuda" or dst.type != "cuda":
            # CPU <-> device moves already have their own pinned/async paths
            return t.to(dst)
        if not t.is_contiguous():
            t = t.contiguous()

        src = t.device
        ch = self._channel(src, dst)
        slot = self._slot(ch, t.dtype)
        numel = t.numel()
        if numel > slot.cap:
            self._grow(ch, slot, t.dtype, numel)

        src_stream = torch.cuda.current_stream(src)
        dst_stream = torch.cuda.current_stream(dst)

        # Reusing a host slot: the source must not overwrite it until the destination
        # finished reading it.  This is a cross-device stream wait, not a host wait.
        if slot.read_valid:
            src_stream.wait_event(slot.read)

        hb = slot.buf[:numel].view(t.shape)

        with torch.cuda.stream(src_stream):
            hb.copy_(t, non_blocking = True)
            slot.filled.record(src_stream)

        with torch.cuda.stream(dst_stream):
            dst_stream.wait_event(slot.filled)
            out = hb.to(dst, non_blocking = True)
            slot.read.record(dst_stream)
        slot.read_valid = True

        nbytes = numel * t.element_size()
        if chunk:
            self.stats["chunk_hops"] += 1
            self.stats["chunk_bytes"] += nbytes
        self.stats["hops"] += 1
        self.stats["bytes"] += nbytes
        return out


def bridge_enabled() -> bool:
    return os.environ.get("KYOJIN_BRIDGE", "1") != "0"


def install_bridge(config, main: torch.device, front: torch.device | None) -> Bridge | None:
    """Attach the bridge to a config and mark the model as split.

    The config is shared by the target, the drafter and the vision tower, so one call
    covers every component.  ``split_active`` also defuses the graph paths: a whole-
    forward or per-block graph must not span the bridge (Phase 6 owns per-segment
    capture).
    """
    split_active = front is not None and torch.device(main) != torch.device(front)
    config.split_active = split_active
    if not split_active:
        return getattr(config, "device_bridge", None)
    # Kernel-level fallbacks that must apply only in split mode (the gfx12 KDA
    # determinant fallback) read this; split off stays on the main code path.
    os.environ["KYOJIN_SPLIT_ACTIVE"] = "1"
    existing = getattr(config, "device_bridge", None)
    if not bridge_enabled():
        return existing
    if existing is not None and existing.pair == (torch.device(main), torch.device(front)):
        return existing
    bridge = Bridge(main, front)
    config.device_bridge = bridge
    return bridge
