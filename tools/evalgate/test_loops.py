from tools.evalgate.loops import long_period_line


def test_block_repeated_beyond_window():
    block = "\n".join(f"Step {i}: check the collision between the player and the wall segment number {i}" for i in range(30))
    text = "Intro paragraph.\n" + (block + "\n") * 3
    assert long_period_line(text) == len("Intro paragraph.\n") + len(block + "\n") * 2 + len(block.split("\n")[0]) + 1


def test_distinct_and_short_lines_do_not_fire():
    text = "\n".join(f"Distinct reasoning line number {i} about the game loop and its scoring rules" for i in range(50))
    text += "\n" + "ok\n" * 10
    assert long_period_line(text) is None


def test_two_occurrences_do_not_fire():
    line = "This exact sentence has more than forty characters in it."
    assert long_period_line(f"{line}\nother\n{line}\n") is None
