"""Decensor edit hook of the Hy4 forward (loader/ablit_hy4.py + the two call sites in hy4_forward.layer + the fused resid kernel).
CPU, torch only (a second run with TRITON_INTERPRET=1 HY4_HC_FUSED=1 also runs the Triton kernels in the interpreter).
  - formula vs a float64 reference; w = 0 returns the input bit for bit (also -0.0)
  - spec round trip (edit_spec.json + edit_spec.safetensors, global and per-layer r); weight-side levers rejected
  - truncated depth (embedding + layers 0..2): a zero-weight spec on every site gives the SAME residual stream as no spec (torch.equal),
    a non-zero spec changes it, clearing the state brings the original back"""
import json, os, sys, tempfile
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
import test_gguf_hy4_hc as T                      # stub packages + registry
from test_gguf_hy4_hc import gg
from exllamav3.loader import hy4_forward as HF

AB = HF._AB
needs_gguf = pytest.mark.skipif(not os.path.exists(T.GGUF1), reason="Hy4 GGUF absent")


def _unit(n, seed):
    r = torch.randn(n, generator=torch.Generator().manual_seed(seed))
    return r / r.norm()


def test_formula_and_zero_weight_identity():
    g = torch.Generator().manual_seed(1)
    y = torch.randn(5, 64, generator=g)
    y[0, 3] = -0.0
    y[1] = 0.0
    r = _unit(64, 2)
    st = AB.EditState(2, 64, "cpu")
    st.set([0.0, 0.7], [0.0, 1.3], [0.0, 0.2], [0.0, -0.4], r.numpy())
    for L, s in ((0, 0), (0, 1)):                                    # w = 0: the same bits
        o = st.edit(y, L, s)
        assert torch.equal(o, y) and torch.equal(o.view(torch.int32), y.view(torch.int32))
    for s, (w, mu) in ((0, (0.7, 0.2)), (1, (1.3, -0.4))):
        o = st.edit(y, 1, s)
        y64, r64 = y.double(), r.double()
        ref = y64 - w * ((y64 @ r64) - mu)[:, None] * r64[None]
        assert (o.double() - ref).abs().max() < 1e-5
        assert (o @ r - (1 - w) * (y @ r) - w * mu).abs().max() < 1e-4        # the r-component moves by w toward mu


def _write(d, n, D, r, wa, wm, per_layer=False, extra=None):
    from safetensors.numpy import save_file
    spec = {"hidden": D, "n_layers": n, "attn_w": list(wa), "mlp_w": list(wm), "attn_mu": [0.1] * n, "mlp_mu": [-0.2] * n}
    if per_layer:
        spec["r_per_layer"] = True
    ten = {"r": np.ascontiguousarray(r, dtype=np.float32)}
    ten.update(extra or {})
    json.dump(spec, open(os.path.join(d, "edit_spec.json"), "w"))
    save_file(ten, os.path.join(d, "edit_spec.safetensors"))
    return os.path.join(d, "edit_spec.json")


def test_spec_roundtrip_and_rejects_levers():
    D, n = 32, 4
    with tempfile.TemporaryDirectory() as d:
        r = np.stack([_unit(D, 10 + L).numpy() for L in range(n)])
        p = _write(d, n, D, r, [0.5, 0, 0, 1.0], [0, 0.25, 0, 0], per_layer=True)
        AB.uninstall()
        spec = AB.load_spec(p, "cpu", n_layers=n, D=D)
        st = AB.ST
        assert st.sites == [[True, False], [False, True], [False, False], [True, False]]
        assert torch.equal(st.R, torch.from_numpy(r)) and float(st.W[3, 0]) == 1.0 and abs(float(st.MU[1, 1]) + 0.2) < 1e-7
        r1 = r[0]
        p = _write(d, n, D, r1, [0.5] * n, [0.5] * n)                 # global r
        AB.uninstall(); AB.load_spec(p, "cpu", n_layers=n, D=D)
        assert torch.equal(AB.ST.R[2], torch.from_numpy(r1))
        p = _write(d, n, D, r1, [0.5] * n, [0.5] * n, extra={"gain.attn.0": np.ones(D, np.float32)})
        with pytest.raises(ValueError):
            AB.load_spec(p, "cpu", n_layers=n, D=D)
    AB.uninstall()


def _stream(M, ids):
    M.reset_for_test() if hasattr(M, "reset_for_test") else None
    M.step(ids)
    return M._x.clone()


@needs_gguf
def test_zero_weight_spec_bit_identical_truncated():
    g, _ = gg()
    ids = [7, 1234, 99, 5, 4021, 77, 318, 12]
    AB.uninstall()
    M = HF.Hy4Forward(g, n_layers=3, cache_weights=True)
    x0 = _stream(M, ids)
    D = M.D
    st = AB.install(3, D, "cpu")
    st.pin_all_sites()
    r = _unit(D, 3).numpy()
    st.set([0.0] * 3, [0.0] * 3, [0.3] * 3, [-0.7] * 3, r)           # every site hooked, every weight zero (mu set: must not matter)
    M2 = HF.Hy4Forward(g, n_layers=3, cache_weights=True)
    x1 = _stream(M2, ids)
    assert torch.equal(x0, x1) and torch.equal(x0.view(torch.int32), x1.view(torch.int32))
    st.set([1.5] * 3, [1.5] * 3, [0.0] * 3, [0.0] * 3, r)
    M3 = HF.Hy4Forward(g, n_layers=3, cache_weights=True)
    x2 = _stream(M3, ids)
    assert not torch.equal(x0, x2) and torch.isfinite(x2).all()
    st.clear()
    st.sites = [[False, False]] * 3
    AB.uninstall()
    M4 = HF.Hy4Forward(g, n_layers=3, cache_weights=True)
    assert torch.equal(x0, _stream(M4, ids))


def test_only_the_residual_direction_moves():
    """hook on: y' - y lies along r (per-layer r), the orthogonal part of y is untouched"""
    D, n = 48, 3
    g = torch.Generator().manual_seed(7)
    y = torch.randn(6, D, generator=g)
    r = torch.stack([_unit(D, 20 + L) for L in range(n)])
    st = AB.EditState(n, D, "cpu")
    st.set([0.4, 0.0, 1.0], [0.0, 0.9, 0.0], [0.1] * n, [0.0] * n, r.numpy())
    for L, s in ((0, 0), (1, 1), (2, 0)):
        o = st.edit(y, L, s)
        d = o - y
        assert d.abs().max() > 1e-3
        par = (d @ r[L])[:, None] * r[L][None]
        assert (d - par).abs().max() < 1e-5                          # the change is parallel to r
    assert torch.equal(st.edit(y, 1, 0), y)                           # w = 0 site: unchanged bits


def test_loader_rejects_malformed_specs():
    D, n = 16, 3
    r = _unit(D, 1).numpy()
    with tempfile.TemporaryDirectory() as d:
        good = dict(wa=[0.5] * n, wm=[0.5] * n)
        p = _write(d, n, D, r, **good)
        AB.uninstall(); AB.load_spec(p, "cpu", n_layers=n, D=D)       # sanity: the good one loads
        for bad_kw in (dict(wa=[0.5] * (n - 1), wm=[0.5] * n),        # wrong length
                       dict(wa=[0.5, float("nan"), 0.5], wm=[0.5] * n),   # not finite
                       dict(wa=[0.5] * n, wm=[0.5] * n, extra={"suhs.mlp.0": np.ones(D, np.float32)})):
            p = _write(d, n, D, r, **bad_kw)
            AB.uninstall()
            with pytest.raises(ValueError):
                AB.load_spec(p, "cpu", n_layers=n, D=D)
        p = _write(d, n, D, r, **good)
        with pytest.raises(ValueError):                               # hidden size differs from the model
            AB.load_spec(p, "cpu", n_layers=n, D=D + 1)
        p = _write(d, n, D, np.zeros((n, D + 2), np.float32), per_layer=True, **good)   # direction of the wrong width
        with pytest.raises(ValueError):
            AB.load_spec(p, "cpu", n_layers=n, D=D)
        spec = json.load(open(p)); del spec["mlp_w"]; json.dump(spec, open(p, "w"))        # missing field
        with pytest.raises(ValueError):
            AB.load_spec(p, "cpu", n_layers=n, D=D)
        from safetensors.numpy import save_file
        p = _write(d, n, D, r, **good)
        save_file({"q": r}, os.path.join(d, "edit_spec.safetensors"))                      # no tensor r
        with pytest.raises(ValueError):
            AB.load_spec(p, "cpu", n_layers=n, D=D)
    AB.uninstall()


def test_fused_resid_edit_matches_unfused_and_off_is_identical():
    """the fused resid kernel (Triton, interpreter on CPU): edit=None == no hook, edit on == torch reference"""
    triton = pytest.importorskip("triton")
    if os.environ.get("TRITON_INTERPRET", "0") != "1" and not torch.cuda.is_available():
        pytest.skip("run with TRITON_INTERPRET=1 (CPU) or on a GPU")
    from exllamav3.loader import hc_fused
    T_, H, D = 3, 4, 32
    g = torch.Generator().manual_seed(11)
    s = torch.randn(T_, H, D, generator=g)
    f = torch.randn(T_, D, generator=g)
    post = torch.rand(T_, H, generator=g)
    wp = torch.rand(D, generator=g) + 0.5
    r = _unit(D, 4)
    base = hc_fused.resid(s, f, post, wp, 1e-6)
    st = AB.EditState(1, D, "cpu")
    st.set([0.0], [0.0], [0.0], [0.0], r.numpy())
    assert torch.equal(hc_fused.resid(s, f, post, wp, 1e-6, edit=st.args(0, 1)), base)      # w = 0: same bits
    st.set([0.0], [0.8], [0.0], [0.3], r.numpy())
    got = hc_fused.resid(s, f, post, wp, 1e-6, edit=st.args(0, 1))
    fn = f * torch.rsqrt((f * f).mean(-1, keepdim=True) + 1e-6) * wp
    fe = AB.edit_torch(fn, r, torch.tensor(0.8), torch.tensor(0.3))
    ref = s + post[:, :, None] * fe[:, None, :]
    assert (got - ref).abs().max() < 1e-4
    AB.uninstall()


if __name__ == "__main__":
    sys.exit(pytest.main(["-q", "-s", __file__]))
