"""run tests/test_hy4_mu_slots.py on the GPU (compiled Triton kernels): small, no model. pytest comes from the system site-packages through a symlink dir."""
import os, sys
sys.path.insert(0, "/tmp/hy4mu_pt")
os.environ["MU_DEV"] = "cuda"
import pytest
sys.exit(pytest.main(["-x", "-q", os.path.join(os.getcwd(), "tests", "test_hy4_mu_slots.py")]))
