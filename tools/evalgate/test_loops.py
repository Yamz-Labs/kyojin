import random

from tools.evalgate.loops import degenerate


def _varied(n, seed=0):
    rnd = random.Random(seed)
    word = lambda: "".join(rnd.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rnd.randint(3, 9)))
    return "\n".join(f"Line {i}: " + " ".join(word() for _ in range(10)) for i in range(n))


def test_block_repeated_beyond_tail_window_fires():
    block = "\n".join(f"Step {i}: check the collision between the player and the wall segment number {i}" for i in range(30))
    text = "Intro paragraph.\n" + (block + "\n") * 4
    assert degenerate(text) is not None


def test_varied_thinking_does_not_fire():
    assert degenerate(_varied(400)) is None


def test_three_code_redrafts_inside_varied_thinking_do_not_fire():
    code = "\n".join(["const o=AC.createOscillator(),g=AC.createGain();", "ctx.setTransform(dpr,0,0,dpr,0,0); // reset the canvas"])
    text = _varied(60, 1) + "\n" + code + "\n" + _varied(60, 2) + "\n" + code + "\n" + _varied(60, 3) + "\n" + code + "\n" + _varied(60, 4)
    assert degenerate(text) is None


def test_short_text_does_not_fire():
    assert degenerate("ok\n" * 50) is None
