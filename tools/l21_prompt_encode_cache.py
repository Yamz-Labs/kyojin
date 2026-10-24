# Incremental prompt encoding for chat servers (model-agnostic, Python only, portable HIP / CUDA / Metal).
# Clients (Hermes, OpenAI-style) re-send the whole conversation every turn; a full encode costs ~1.3 ms of CPU per 1K tokens
# (90 ms at 66K, 166 ms at 128K, Qwen3.8 tokenizer). PromptEncodeCache keeps the last n (prompt, ids) pairs; a prompt that shares a text
# prefix with one of them encodes only from the last turn-start special token (cut: one string or a list, e.g. '<|im_start|>' for
# ChatML, ['<|system|>', '<|user|>', '<|assistant|>', '<|observation|>'] for GLM) inside the shared part. The tokenizer splits its input
# at special tokens and encodes the pieces independently, so the ids equal the full encode's. A self-test at construction (split ==
# full on a rendered 2-turn conversation, or on a synthetic text without render) disables the cache where that fails; check=True also
# compares every request with the full encode (debug).
#
#   pec = PromptEncodeCache(tokenizer, n=8, cut="<|im_start|>", render=lambda msgs: render_prompt(template, msgs, None))
#   ids, hit = pec.encode(prompt)          # ids: 1-D LongTensor (CPU), same as tokenizer.encode(prompt, add_bos=False, encode_special_tokens=True)
#
# Soft cuts: a long text that shares a prefix but has no turn-start token inside it (same document, another question in the same
# user message) is also cut after a newline that is followed by a non-space character. That position is a pre-tokenizer boundary
# (the newline run ends there whatever follows), so the ids equal the full encode's; the cache needs the backend's token offsets
# for it (HF tokenizers, no extra specials) and a construction self-test on punctuation / digits / indentation / unicode text turns
# soft cuts off where the equality fails.
#
# Not for prompts with image embeddings (encode those in full). Thread-safe (one lock).
import re, threading
import numpy as np
import torch


class PromptEncodeCache:

    def __init__(self, tokenizer, n: int = 8, cut="<|im_start|>", render=None, check: bool = False, log=print):
        cuts = [cut] if isinstance(cut, str) else list(cut)
        self.tokenizer, self.n, self.check, self.log = tokenizer, n, check, log
        self.e = []   # [text, ids, marker char starts, marker token positions, soft cuts, token starts, token ends], most recent first
        self.soft, self.n_soft_hits = False, 0
        self.lock = threading.Lock()
        mids = [self.full(c) for c in cuts]
        if not cuts or any(m.numel() != 1 for m in mids):
            log(f"prompt encode cache off: {cuts!r} not one token each"); self.n = 0; return
        self.mids = torch.tensor([int(m[0]) for m in mids])
        self.soft_re = re.compile(r"(?<=\n)(?=\S)")
        self.cut_re = re.compile("|".join(re.escape(c) for c in sorted(cuts, key=len, reverse=True)))
        msgs = [{"role": "system", "content": "S x"}, {"role": "user", "content": "Hi there, how are you?"},
                {"role": "assistant", "content": "Fine. And you?"}, {"role": "user", "content": "Tell me about pages."}]
        try:
            if render is not None: p1, p2 = render(msgs[:2]), render(msgs)
            else:
                p1 = "Sys x " + cuts[-1] + " Hi there, how are you?\n" + cuts[0] + " ok"
                p2 = p1[:-3] + " Fine. And you?\n" + cuts[-1] + " Tell me about pages.\n" + cuts[0]
            self.encode(p1); ids, hit = self.encode(p2)
            ok = hit and torch.equal(ids, self.full(p2))
        except Exception as ex:
            ok = False; log(f"prompt encode cache self-test error: {ex}")
        if not ok:
            log("prompt encode cache off: self-test failed"); self.n = 0
        self.e = []
        self.soft = False
        hf = getattr(tokenizer, "tokenizer", None)
        if self.n and hf is not None and hasattr(hf, "encode") and not getattr(tokenizer, "unspecial_piece_to_id", None) \
                and not getattr(tokenizer, "missing_special_piece_to_id", None):
            self.hf, self.soft = hf, True
            try:
                doc = ("Intro line.\n\nSecond paragraph, with numbers 12345 and 'quotes' (and) {braces}:\n    def f(x):\n"
                       "        return x + 1  # c\n\n\nThird  para,  odd   spacing...\n\n  indented\n\ttab\n\nnaïve café 日本語 \U0001F600\n\n")
                for tail in ("Question one: why?", "Question two: how 42?", " x"):
                    self.encode(doc + doc.upper() + tail)
                    p2 = doc + doc.upper() + tail.replace("one", "three").replace("two", "four") + "\n" + cuts[0]
                    ids, hit = self.encode(p2)
                    if not (hit and torch.equal(ids, self.full(p2))):
                        self.soft = False; break
                self.soft = self.soft and self.n_soft_hits > 0
            except Exception as ex:
                self.soft = False; log(f"prompt encode cache soft cuts error: {ex}")
            if not self.soft:
                log("prompt encode cache: soft cuts off")
            self.e = []

    def full(self, p: str) -> torch.Tensor:
        ids = self.tokenizer.encode(p, add_bos=False, encode_special_tokens=True)
        ids = ids[0] if isinstance(ids, tuple) else ids
        return ids.reshape(-1).cpu()

    def full_off(self, p: str):
        """ids, token start offsets, token end offsets (numpy) of p, one backend pass (same ids as full())."""
        self.hf.encode_special_tokens = False
        enc = self.hf.encode(p, add_special_tokens = False)
        off = np.asarray(enc.offsets, dtype = np.int64).reshape(-1, 2)
        return torch.tensor(enc.ids, dtype = torch.long), off[:, 0], off[:, 1]

    def encode(self, p: str):
        if not self.n:
            return self.full(p), False
        with self.lock:
            best = None   # (entry index, char cut, token position, extends-last-item, is soft)
            for i, (t, ids, cs, ts, sc, st, en) in enumerate(self.e):
                lo, hi = -1, len(cs) - 1   # last marker whose start lies inside the shared text: binary search over the markers
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if cs[mid] <= len(p) and p[:cs[mid]] == t[:cs[mid]]: lo = mid
                    else: hi = mid - 1
                if lo >= 0 and cs[lo] > 0 and (best is None or cs[lo] > best[1]):
                    best = (i, cs[lo], ts[lo], lo == len(cs) - 1, False)   # last item: p extends this entry (same conversation)
                if self.soft and sc:
                    # soft cut c: t[c - 1] is a newline, t[c] is not whitespace; p must share t[:c + 1] (so p[c] is that character too)
                    lo, hi = -1, len(sc) - 1
                    while lo < hi:
                        mid = (lo + hi + 1) // 2
                        if sc[mid] + 1 <= len(p) and p[:sc[mid] + 1] == t[:sc[mid] + 1]: lo = mid
                        else: hi = mid - 1
                    while lo >= 0 and (best is None or sc[lo] > best[1]):
                        k = int(np.searchsorted(st, sc[lo]))
                        if 0 < k < len(st) and st[k] == sc[lo] and en[k - 1] == sc[lo]:
                            best = (i, sc[lo], k, False, True)
                            break
                        lo -= 1
            if best is not None:
                i, c, tpos, _, soft = best
                if self.soft:
                    sids, sst, sen = self.full_off(p[c:])
                    e = self.e[i]
                    ids = torch.cat([e[1][:tpos], sids])
                    off = (np.concatenate([e[5][:tpos], sst + c]), np.concatenate([e[6][:tpos], sen + c]))
                else:
                    ids = torch.cat([self.e[i][1][:tpos], self.full(p[c:])]); off = None
                self.n_soft_hits += 1 if soft else 0
                if self.check and not torch.equal(ids, self.full(p)):
                    self.log("prompt encode cache MISMATCH, full encode used"); ids, best, off = self.full(p), None, None
            elif self.soft:
                ids, st, en = self.full_off(p); off = (st, en)
            else:
                ids = self.full(p); off = None
            self._add(p, ids, best[0] if best is not None and best[3] else None, off)
            return ids, best is not None

    def _add(self, p, ids, drop=None, off=None):
        cs = [m.start() for m in self.cut_re.finditer(p)]
        ts = torch.isin(ids, self.mids).nonzero().reshape(-1).tolist()
        if len(cs) == len(ts):
            if drop is not None: self.e.pop(drop)
            sc = [m.start() for m in self.soft_re.finditer(p)] if (self.soft and off is not None) else []
            st, en = off if off is not None else (None, None)
            self.e.insert(0, [p, ids, cs, ts, sc, st, en]); del self.e[self.n:]
