"""
llama.cpp-compatible slot persistence for the paged EXL3 generator.

The swap orchestrator keeps lane caches warm on disk through the llama.cpp slots API (GET /slots, POST
/slots/{id}?action=save|restore|erase). The generator has no slots: every sequence shares one PageTable, and a
cached prefix is a chain of hashed pages (plus, for recurrent models, a recurrent-state checkpoint anchored on a
page hash). This module maps the API onto that model:

- save writes the page chain of the last prompt that is still resident (token ids, page hashes, the page
  tensors of the main and draft caches) and, for recurrent models, the deepest matching checkpoint. Pages past
  that checkpoint are dropped, since allocate_pages cannot resume past it anyway.
- restore injects the saved pages into the PageTable as unreferenced, hashed pages and puts the checkpoint back
  in the recurrent cache, so the next request with the same prefix is a normal prompt-cache hit.
- erase only clears the slot view (prompt, token count). The PageTable is shared by all conversations, so
  flushing it would destroy the in-memory reuse the next request relies on.

The caller must hold the generation lock: the generator must be idle for the duration of a save or restore.
"""
from __future__ import annotations

import os
import time

import torch

from .pagetable import PAGE_SIZE

FORMAT_VERSION = 1


class SlotStore:

    def __init__(self, generator, caches: list, slot_dir: str, model_id: str):
        self.generator = generator
        self.caches = caches
        self.slot_dir = os.path.expanduser(slot_dir) if slot_dir else None
        self.model_id = model_id
        self.prompt = ""
        self.ids = None
        self.busy = False
        self.id_task = -1

    # Slot view

    def note_prompt(self, prompt: str, ids: torch.Tensor):
        self.prompt = prompt
        self.ids = ids.view(-1).to("cpu", torch.long)
        self.id_task += 1

    def slots(self) -> list[dict]:
        n = 0 if self.ids is None else int(self.ids.numel())
        return [{"id": 0, "id_task": self.id_task, "n_ctx": self.caches[0].max_num_tokens,
                 "is_processing": self.busy, "n_prompt_tokens": n, "prompt": self.prompt}]

    def erase(self) -> dict:
        n = 0 if self.ids is None else int(self.ids.numel())
        self.prompt, self.ids = "", None
        return {"id_slot": 0, "n_erased": n}

    # Persistence

    def _path(self, filename: str) -> str:
        if not self.slot_dir:
            raise ValueError("slot save path not set (--slot-save-path)")
        if not filename or os.path.basename(filename) != filename or filename.startswith("."):
            raise ValueError(f"invalid filename: {filename!r}")
        return os.path.join(self.slot_dir, filename)

    def _tensors(self) -> list[torch.Tensor]:
        tensors = []
        for c in self.caches:
            if c.model.loaded_tp:
                raise RuntimeError("slot save/restore does not support tensor-parallel caches")
            tensors += c.get_all_tensors()
        return tensors

    def _resident_chain(self) -> tuple[list, list]:
        """Hashes and live pages of the full pages of the last prompt still resident in the PageTable."""
        pt = self.generator.pagetable
        from .pagetable import tensor_hash_checksum
        hashes, pages, prev = [], [], None
        ids = self.ids.view(1, -1)
        for pi in range(ids.shape[1] // PAGE_SIZE):
            h = tensor_hash_checksum(ids[:, pi * PAGE_SIZE:(pi + 1) * PAGE_SIZE], prev)
            page = pt.referenced_pages.get(h) or pt.unreferenced_pages.get(h)
            if page is None or page.kv_position != PAGE_SIZE:
                break
            hashes.append(h)
            pages.append(page)
            prev = h
        return hashes, pages

    @torch.inference_mode()
    def save(self, filename: str) -> dict:
        t0 = time.perf_counter()
        path = self._path(filename)
        if self.ids is None:
            raise ValueError("slot is empty")
        hashes, pages = self._resident_chain()
        stash = None
        rc = self.generator.recurrent_cache
        if rc is not None:
            depth = 0
            for pi in range(len(hashes) - 1, -1, -1):
                s = rc.get(hashes[pi])
                if s is not None and s["position"] == (pi + 1) * PAGE_SIZE:
                    depth, stash = pi + 1, s
                    break
            if stash is not None and "tp_handle" in stash:
                raise RuntimeError("tensor-parallel recurrent checkpoints cannot be saved")
            hashes, pages = hashes[:depth], pages[:depth]
        n_tokens = len(pages) * PAGE_SIZE
        data = {
            "version": FORMAT_VERSION,
            "model_id": self.model_id,
            "prompt": self.prompt,
            "ids": self.ids[:n_tokens].clone(),
            "hashes": hashes,
            "sequences": [p.sequence.cpu().clone() for p in pages],
            "stash": stash,
            "tensors": [],
        }
        if pages:
            torch.cuda.synchronize()
            for t in self._tensors():
                idx = torch.tensor([p.page_index for p in pages], dtype=torch.long, device=t.device)
                data["tensors"].append(t.index_select(0, idx).cpu())
        tmp = path + ".tmp"
        torch.save(data, tmp)
        os.replace(tmp, path)
        return {"id_slot": 0, "filename": filename, "n_saved": n_tokens,
                "n_written": os.path.getsize(path),
                "timings": {"save_ms": (time.perf_counter() - t0) * 1000}}

    @torch.inference_mode()
    def restore(self, filename: str) -> dict:
        t0 = time.perf_counter()
        data = torch.load(self._path(filename), map_location="cpu", weights_only=False)
        if data.get("version") != FORMAT_VERSION or data.get("model_id") != self.model_id:
            raise ValueError(f"slot file is for {data.get('model_id')!r} v{data.get('version')}, "
                             f"not {self.model_id!r} v{FORMAT_VERSION}")
        tensors = self._tensors()
        if len(data["tensors"]) not in (0, len(tensors)) or any(
                s.shape[1:] != t.shape[1:] or s.dtype != t.dtype for s, t in zip(data["tensors"], tensors)):
            raise ValueError("slot file cache layout does not match the loaded cache")
        pt = self.generator.pagetable
        hashes = data["hashes"]
        protected = set(hashes)
        order = None
        n_copied = 0
        prev = None
        for pi, h in enumerate(hashes):
            live = pt.referenced_pages.get(h) or pt.unreferenced_pages.get(h)
            if live is not None and live.kv_position == PAGE_SIZE:
                prev = h
                continue
            if live is not None:
                raise RuntimeError("partially written page holds a saved hash")
            if order is None:
                order = pt.build_eviction_order(protected)
            while order[0].ref_count:
                order.popleft()
            page = order.popleft()
            pt.evict(page, protected)
            pt.access_serial += 1
            page.add_ref_clear(pt.access_serial, h)
            page.prev_hash = prev
            page.sequence.copy_(data["sequences"][pi])
            page.kv_position = PAGE_SIZE
            for s, t in zip(data["tensors"], tensors):
                t[page.page_index].copy_(s[pi], non_blocking=False)
            page.sub_ref()
            n_copied += 1
            prev = h
        rc = self.generator.recurrent_cache
        if data["stash"] is not None and rc is not None and hashes:
            key = hashes[-1]
            if key not in rc:
                rc[key] = data["stash"]
            rc.move_to_end(key)
            rc.update_total_size()
        torch.cuda.synchronize()
        n = int(data["ids"].numel())
        self.prompt, self.ids = data["prompt"], data["ids"]
        return {"id_slot": 0, "filename": filename, "n_restored": n, "n_read": n_copied * PAGE_SIZE,
                "timings": {"restore_ms": (time.perf_counter() - t0) * 1000}}
