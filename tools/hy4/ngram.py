"""Prompt-lookup n-gram proposer for the Hy4 spec loop. Index of every n-gram (n = nmin) of the sequence seen so far (prompt + emitted tokens); a draft of
k MTP tokens is extended with the tokens that followed the latest earlier occurrence of the sequence's current suffix. Pure host code (dict of tuples)."""


class NGram:
    def __init__(self, nmin=4, ext=4):
        self.nmin, self.ext = nmin, ext
        self.seq: list[int] = []
        self.idx: dict[tuple, int] = {}      # n-gram -> index of its LAST token (latest occurrence)

    def reset(self, ids):
        self.seq, self.idx = [], {}
        self.add(ids)

    def add(self, toks):
        n, s, ix = self.nmin, self.seq, self.idx
        for t in toks:
            s.append(int(t))
            if len(s) >= n:
                ix[tuple(s[-n:])] = len(s) - 1

    def propose(self, d, ext=None):
        """d: the MTP drafts (host ints) that follow self.seq. Returns (extra tokens, match length) with extra = [] when there is no usable match."""
        n, s = self.nmin, self.seq
        ext = self.ext if ext is None else ext
        if ext <= 0:
            return [], 0
        ss = s + list(d)
        if len(ss) < n:
            return [], 0
        end = len(ss) - 1
        # the draft region ends at `end`; the index holds only positions < len(s), so a hit is always an earlier occurrence
        i = self.idx.get(tuple(ss[-n:]))
        if i is None or i >= end:
            return [], 0
        m = n                                    # extend the match backwards
        while i - m >= 0 and end - m >= 0 and ss[i - m] == ss[end - m] and m < 32:
            m += 1
        cont = ss[i + 1:i + 1 + ext]
        return cont, m
