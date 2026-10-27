import os, re, ast

# The placement rule is a pure function; load it from source so the test needs neither a GPU nor the compiled extension
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src = open(os.path.join(_ROOT, "exllamav3/modules/ngram_embedding.py")).read()
_tree = ast.parse(_src)
_ns = {}
for _n in _tree.body:
    if (isinstance(_n, ast.FunctionDef) and _n.name == "decide_ngram_stream") or \
       (isinstance(_n, ast.Assign) and getattr(_n.targets[0], "id", "") == "RAM_MARGIN_BYTES"):
        exec(compile(ast.Module([_n], []), "ngram_embedding", "exec"), _ns)
decide_ngram_stream = _ns["decide_ngram_stream"]

GiB = 1 << 30


def test_unset_is_disk_even_with_lots_of_ram():
    assert decide_ngram_stream(None, 76 * GiB, 107 * GiB)[0] is True
    assert decide_ngram_stream(None, 76 * GiB, None)[0] is True


def test_explicit_disk():
    assert decide_ngram_stream(True, 10 * GiB, 500 * GiB)[0] is True


def test_explicit_ram_honored_with_warning_when_tight():
    stream, why = decide_ngram_stream(False, 76 * GiB, 80 * GiB)
    assert stream is False and "WARNING" in why
    stream, why = decide_ngram_stream(False, 76 * GiB, 200 * GiB)
    assert stream is False and "WARNING" not in why


def test_config_unset_stays_none():
    # unset must reach the placement rule as None (which decides disk), never as an RAM request
    src = open(os.path.join(_ROOT, "exllamav3/model/config.py")).read()
    assert re.search(r"ngram_stream_from_disk = None if _ns is None", src)
