import os
from unittest import mock
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools", "ds41"))
from ds41_model import PortEngine
import dspark

def eng(tools):
    e = object.__new__(PortEngine); e.draft_auto = True; e.num_draft = 4; e._req_tools = tools; return e

def test_off_by_default():
    with mock.patch.dict(os.environ, {}, clear=True):
        assert not eng(False)._want() and type(eng(False)._make_tuner()) is dspark.DraftTuner

def test_switch_no_tools_uses_timed_and_want():
    with mock.patch.dict(os.environ, {"DS41_DRAFT_SWITCH": "1"}, clear=True):
        assert eng(False)._want() and isinstance(eng(False)._make_tuner(), dspark.TimedTuner)

def test_switch_tools_keeps_auto():
    with mock.patch.dict(os.environ, {"DS41_DRAFT_SWITCH": "1"}, clear=True):
        assert not eng(True)._want() and type(eng(True)._make_tuner()) is dspark.DraftTuner
