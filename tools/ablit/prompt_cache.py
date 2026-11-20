"""prompt_cache.py -- where the ablit tools find their public prompt sets.

The four jsonl files are not in the repository. They are fetched by build_prompts.py (sources and licenses: prompts/SOURCES.md)
into the user cache dir the first time a tool needs them: $XDG_CACHE_HOME/kyojin/ablit_prompts (default ~/.cache/kyojin/ablit_prompts).
"""
import os
import sys

NAMES = ("harmful_collect.jsonl", "harmful_heldout.jsonl", "harmless_collect.jsonl", "harmless_heldout.jsonl")


def cache_dir():
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "kyojin", "ablit_prompts")


def _complete(d):
    return all(os.path.isfile(os.path.join(d, n)) for n in NAMES)


def ensure(legacy=None):
    """directory that holds the four files: `legacy` if it already has them (an older scratch copy), else the cache dir (fetched when missing)"""
    if legacy and _complete(legacy):
        return legacy
    d = cache_dir()
    if not _complete(d):
        import build_prompts
        print(f"prompt sets not found: fetching them into {d}", file=sys.stderr, flush=True)
        tmp = d + f".part{os.getpid()}"
        build_prompts.build(tmp)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        if os.path.isdir(d):
            import shutil
            shutil.rmtree(d)
        os.replace(tmp, d)
    return d


def default_file(name, legacy=None):
    """full path of one of NAMES (fetching the sets first when they are missing)"""
    assert name in NAMES, name
    return os.path.join(ensure(legacy), name)
