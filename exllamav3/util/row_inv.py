"""Row invariance of multi-row verify forwards (MTP / DFlash).

ROW_INV["on"] (default on, EXL3_ROW_INV=0 turns it off, mutable at run time for one-load A/B):
every row of a verify forward, for any number of jobs decoded together, gets the same bits as the
same row would get when its job is decoded alone. Callers that bake the choice into a captured graph
must purge the graphs when the flag flips (block_graph.purge()).
"""
import os

ROW_INV = {"on": os.environ.get("EXL3_ROW_INV", "1") not in ("", "0")}
# Rows a verify forward may have before the per-row exact paths give way to the batched ones.
ROW_INV_MAX_ROWS = 8
