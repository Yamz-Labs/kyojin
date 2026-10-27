"""CPU gate: which decode steps take the R-row exact paths (params["dflash_verify"]) without drafting."""
from exllamav3.generator.row_groups import PLAIN_ROWS, plain_rows_verify
from exllamav3.util.row_inv import ROW_INV, ROW_INV_MAX_ROWS


def test_off_by_default_state_is_respected():
    old = PLAIN_ROWS["on"]
    try:
        PLAIN_ROWS["on"] = False
        assert not any(plain_rows_verify(r, False) for r in range(0, ROW_INV_MAX_ROWS + 4))
    finally:
        PLAIN_ROWS["on"] = old


def test_on_covers_rows_two_to_max_only():
    old, old_inv = PLAIN_ROWS["on"], ROW_INV["on"]
    try:
        PLAIN_ROWS["on"] = True
        ROW_INV["on"] = True
        got = [r for r in range(0, ROW_INV_MAX_ROWS + 4) if plain_rows_verify(r, False)]
        assert got == list(range(2, ROW_INV_MAX_ROWS + 1))
        assert not plain_rows_verify(4, True)            # a drafted step is already marked by its caller
        ROW_INV["on"] = False
        assert not plain_rows_verify(4, False)           # the exact paths need row invariance on
    finally:
        PLAIN_ROWS["on"], ROW_INV["on"] = old, old_inv
