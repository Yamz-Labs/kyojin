import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ENG = os.environ.get("BIGCONV_ENGINE", "")      # directory that contains exllamav3/dsv41 (the pinned DS4.1 engine)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "tools"))
if ENG:                                          # first, so it shadows any full exllamav3 checkout (needs a built ext)
    for m in [m for m in sys.modules if m == "exllamav3" or m.startswith("exllamav3.")]:
        del sys.modules[m]
    sys.path.insert(0, ENG)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
