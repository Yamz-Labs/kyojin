# Development: tests and CUDA build

## Tests
```bash
pip install pytest
for t in tests/test_ablit_runtime_cpu.py tests/test_uncensor_bundled_cpu.py tools/glm/test_serve.py tools/mimo/test_serve.py tools/mimo/test_toolcalls.py; do PYTHONPATH=. pytest -q $t; done
```
Run each file separately because two files share a name (`test_serve.py`). These tests need no GPU and no built extension. GPU tests need a built extension and a free GPU.

## Build on CUDA
The CUDA build is the upstream one and is unchanged. Install a CUDA 12.4 or newer build of PyTorch, then `pip install -r requirements.txt && pip install .`. `README.upstream.md` has the full upstream guide (wheels, PyPI, uv, Windows, architecture list, conversion tool, examples). `README.strix-halo.md` has the kernel notes and benchmark harnesses.
