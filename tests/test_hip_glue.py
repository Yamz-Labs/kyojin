"""KYOJIN_HIP_GLUE: the glue knob is registered, off by default, and its row gate switches the
decode site fusion on without touching the prefill range. CPU-only: no extension, no GPU.

The lever it gates (hyperconnections.HyperConnection.pf_can_defer -> _pf_apply_mix_norm,
ext.hc_pf_apply_mix_norm) is exercised on the device by tools/depth/glueab.py; these tests pin
the gate and the default so a later edit cannot turn the decode fusion on by accident.
"""
import inspect
import os

import pytest

from exllamav3.util import knobs


@pytest.fixture(autouse = True)
def clean_env():
    saved = {n: os.environ.get(n) for n in knobs.KNOBS}
    os.environ.pop("KYOJIN_HIP_GLUE", None)
    yield
    for n, v in saved.items():
        if v is None:
            os.environ.pop(n, None)
        else:
            os.environ[n] = v
    knobs.invalidate()


def _glue():
    from exllamav3.modules import hip_glue
    return hip_glue


def test_knob_registered_and_off():
    assert "KYOJIN_HIP_GLUE" in knobs.KNOBS
    assert knobs.KNOBS["KYOJIN_HIP_GLUE"] == "0"
    assert _glue().glue_on() is False


def test_knob_reads_the_environment_per_call():
    g = _glue()
    knobs.set_knobs({"KYOJIN_HIP_GLUE": "1"})
    assert g.glue_on() is True
    assert g.hc_fuse_rows_ok(1) is True
    knobs.set_knobs({"KYOJIN_HIP_GLUE": "0"})
    assert g.glue_on() is False
    assert g.hc_fuse_rows_ok(1) is False


def test_prefill_range_is_the_same_on_both_sides():
    g = _glue()
    for rows in (257, 300, 4096, g.HC_FUSE_MAX_ROWS):
        assert g.hc_fuse_rows_ok(rows, glue = False) is True
        assert g.hc_fuse_rows_ok(rows, glue = True) is True


def test_decode_range_only_with_glue():
    g = _glue()
    for rows in (1, 2, 5, 256):
        assert g.hc_fuse_rows_ok(rows, glue = False) is False
        assert g.hc_fuse_rows_ok(rows, glue = True) is True


def test_zero_and_oversized_rows_never_fuse():
    g = _glue()
    for glue in (False, True):
        assert g.hc_fuse_rows_ok(0, glue = glue) is False
        assert g.hc_fuse_rows_ok(g.HC_FUSE_MAX_ROWS + 1, glue = glue) is False


def test_the_site_pair_goes_through_the_gate():
    """pf_can_defer and _pf_apply_mix_norm must decide through hc_fuse_rows_ok: a copy of the old
    `b * s > 256` literal would leave the knob inert on decode."""
    from exllamav3.modules import hyperconnections as hc
    assert "hc_fuse_rows_ok" in inspect.getsource(hc.HyperConnection.pf_can_defer)
    assert "hc_fuse_rows_ok" in inspect.getsource(hc.HyperConnection._pf_apply_mix_norm)
    assert "b * s > 256" not in inspect.getsource(hc.HyperConnection.pf_can_defer)
