"""zero-GPU check: NGram.propose finds an earlier occurrence of the sequence's own suffix when there are no MTP drafts (the long-match arm HY4_NGRAM_LONG)"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools", "hy4"))
from ngram import NGram


def test_own_suffix_hits_earlier_occurrence_without_drafts():
    ng = NGram(4, 4)
    ng.reset([1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4])
    cont, m = ng.propose([], 4)
    assert cont == [5, 6, 7, 8] and m >= 4


def test_no_earlier_occurrence_no_hit():
    ng = NGram(4, 4)
    ng.reset([1, 2, 3, 4, 5, 6, 7, 8])
    assert ng.propose([], 4) == ([], 0)


def test_with_drafts_unchanged():
    ng = NGram(4, 4)
    ng.reset([1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2])
    assert ng.propose([3, 4], 4)[0] == [5, 6, 7, 8]
    ng.add([3, 4])
    ng.add([5])
    assert ng.propose([], 3)[0] == [6, 7, 8]
