"""
Runtime refusal-direction projection hook (no weight edit, no re-quantisation):  EXL3_ABLIT_RUNTIME=<edit_spec.json>
Also applied automatically when the model directory contains uncensor_spec.json (+ uncensor_direction.st, or
uncensor_spec.safetensors). EXL3_ABLIT_RUNTIME=off (or the serve flag --no-uncensor) disables it; an explicit path wins.

Editing every writer as W' = (I - w r r^T) W is linear in the writer output, so it equals projecting
each block's attention output and MLP (routed + shared) output before the residual / hc apply:

    y <- y - w_L * r (r . y)         w = spec attn_w[L] / mlp_w[L]; layers >= spec n_layers (MTP) untouched

Spec: JSON with n_layers, hidden, attn_w[L], mlp_w[L] and a direction file (safetensors, tensor "r"). r and the per-layer weights are loaded once per device in
TransformerBlock.load (static tensors, python-float weights baked into graph captures).
"""
import json, os, re, sys
import torch

_cache = {}
_logged = set()
_LAYER_RX = re.compile(r"\.layers\.(\d+)$")

BUNDLED_NAME = "uncensor_spec.json"        # dropped in a model directory: applied automatically
BUNDLED_DIRECTION = "uncensor_direction.st"  # safetensors layout, but not *.safetensors so the weight loader never indexes it
_OFF = ("off", "0", "false", "no", "none")
_REQUIRED = ("hidden", "n_layers", "attn_w", "mlp_w")


def _env() -> str:
    return os.environ.get("EXL3_ABLIT_RUNTIME", "").strip()


def resolve(model_dir = None):
    """Return (spec_path, source) or (None, reason).
    Precedence: EXL3_ABLIT_RUNTIME=off -> disabled; EXL3_ABLIT_RUNTIME=<path> -> that file; else <model_dir>/uncensor_spec.json if present."""
    env = _env()
    if env.lower() in _OFF and env:
        return None, "off"
    if env:
        return env, "env"
    if model_dir:
        p = os.path.join(model_dir, BUNDLED_NAME)
        if os.path.isfile(p):
            return p, "bundled"
    return None, "none"


def enabled(model_dir = None) -> bool:
    return resolve(model_dir)[0] is not None


def _log_once(key, msg):
    if key not in _logged:
        _logged.add(key)
        print(msg, file = sys.stderr, flush = True)


def _load(path: str, source: str = "env"):
    if path not in _cache:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"ablit runtime: spec file not found: {path} (source: {source})")
        try:
            spec = json.load(open(path))
        except ValueError as e:
            raise ValueError(f"ablit runtime: spec {path} is not valid JSON: {e}") from None
        missing = [k for k in _REQUIRED if not isinstance(spec, dict) or k not in spec]
        if missing:
            raise ValueError(f"ablit runtime: spec {path} is missing field(s) {missing}")
        d = os.path.dirname(os.path.abspath(path))
        r_file = spec.get("r_file")
        if r_file:
            r_file = r_file if os.path.isabs(r_file) else os.path.join(d, r_file)
        else:
            cands = [os.path.splitext(path)[0] + ".safetensors"]
            if source == "bundled":
                cands.insert(0, os.path.join(d, BUNDLED_DIRECTION))
            r_file = next((c for c in cands if os.path.isfile(c)), cands[-1])
        if not os.path.isfile(r_file):
            raise FileNotFoundError(f"ablit runtime: direction file not found: {r_file} (spec {path})")
        from safetensors import safe_open
        try:
            with safe_open(r_file, "pt") as f:
                r = f.get_tensor("r").double()
        except Exception as e:
            raise ValueError(f"ablit runtime: cannot read direction 'r' from {r_file}: {e}") from None
        if r.dim() != 1 or r.shape[0] != spec["hidden"]:
            raise ValueError(f"ablit runtime: direction shape {tuple(r.shape)} != hidden {spec['hidden']}")
        if not (len(spec["attn_w"]) >= spec["n_layers"] and len(spec["mlp_w"]) >= spec["n_layers"]):
            raise ValueError(f"ablit runtime: spec {path}: attn_w/mlp_w shorter than n_layers {spec['n_layers']}")
        _cache[path] = (spec, (r / r.norm()).float(), {})
        _log_once(("on", path), f" -- ablit runtime: spec {path} active "
                                f"({'bundled in the model directory' if source == 'bundled' else 'from EXL3_ABLIT_RUNTIME'})")
    return _cache[path]


def prepare(block, device):
    """Set block.ablit = (w_attn, w_mlp, r_fp32, r_fp16) or None."""
    block.ablit = None
    cfg = getattr(block, "config", None)
    path, source = resolve(getattr(cfg, "directory", None))
    if path is None and source == "off" and getattr(cfg, "directory", None) \
            and os.path.isfile(os.path.join(cfg.directory, BUNDLED_NAME)):
        _log_once(("off", cfg.directory), f" -- ablit runtime: OFF (EXL3_ABLIT_RUNTIME=off); bundled {BUNDLED_NAME} in the model directory ignored")
    m = _LAYER_RX.search(block.key or "")
    if not path or m is None:
        return
    if (block.key or "").split(".")[0] == "mtp" or ".mtp." in block.key:
        return  # MTP drafter block (key mtp.layers.N reuses trunk numbers): never edited
    spec, r, per_dev = _load(path, source)
    L = int(m.group(1))  # from the key: MTP blocks reuse layer_idx 0..n but keep their checkpoint layer number
    if L >= spec["n_layers"]:
        return
    wa = float(spec["attn_w"][L]) if block.attn is not None else 0.0
    wm = float(spec["mlp_w"][L]) if block.mlp is not None else 0.0
    if abs(wa) <= 1e-9: wa = 0.0
    if abs(wm) <= 1e-9: wm = 0.0
    if wa == 0.0 and wm == 0.0:
        return
    if os.environ.get("EXL3_ABLIT_COST_ONLY", "0") != "0":
        wa, wm = wa * 1e-30, wm * 1e-30  # bench aid: full kernel cost, output unchanged (tokens identical to hook off)
    key = str(device)
    if key not in per_dev:
        r32 = r.to(device)
        per_dev[key] = (r32, r32.half())
        warmup(device, r32.shape[0])
    block.ablit = (wa, wm) + per_dev[key]


try:
    import triton
    import triton.language as tl

    @triton.jit
    def _project_kernel(y_ptr, r_ptr, w, H: tl.constexpr, BLOCK: tl.constexpr):
        # BLOCK = next power of two >= H (tl.arange needs one); lanes >= H are masked (hidden 2560 for Qwen)
        row = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, BLOCK)
        m = offs < H
        y = tl.load(y_ptr + row * H + offs, mask = m, other = 0.0)
        r = tl.load(r_ptr + offs, mask = m, other = 0.0)
        d = tl.sum(y.to(tl.float32) * r, axis = 0)
        tl.store(y_ptr + row * H + offs, (y.to(tl.float32) - (w * d) * r).to(y.dtype), mask = m)
except ImportError:
    triton = None

def _block(h: int) -> int:
    return 1 << (h - 1).bit_length()


_USE_TRITON = triton is not None and os.environ.get("EXL3_ABLIT_TORCH", "0") == "0"


def warmup(device, hidden = None):
    """Compile the fused kernel for both dtypes now, so no JIT happens inside a graph capture."""
    global _USE_TRITON
    hidden = hidden or HIDDEN_WARM
    if _USE_TRITON:
        try:
            for dt in (torch.float32, torch.float16):
                y = torch.zeros(1, hidden, dtype = dt, device = device)
                _project_kernel[(1,)](y, torch.zeros(hidden, dtype = torch.float32, device = device), 0.0, H = hidden, BLOCK = _block(hidden))
        except Exception as e:  # compile failure: the torch path is equivalent (mv + addr_), just slower
            _USE_TRITON = False
            print(f" -- ablit runtime: triton projection kernel failed to compile for hidden {hidden} ({e!r}); using the torch path", file = sys.stderr, flush = True)


HIDDEN_WARM = 4096


ROW_MODE = None  # measurement aid, off by default: None = every row; ("from", k) = rows >= k only; "last" = in a multi-row forward only the last row


def project(y: torch.Tensor, w: float, r32: torch.Tensor, r16: torch.Tensor) -> torch.Tensor:
    """In place: y -= w * r (r . y) over the last (hidden) dim. Returns y."""
    if w == 0.0:
        return y
    h = r32.shape[0]
    y2 = y.view(-1, h)
    if ROW_MODE is not None and y2.shape[0] > 1:
        y2 = y2[ROW_MODE[1]:] if isinstance(ROW_MODE, tuple) else y2[-1:]
        if y2.shape[0] == 0:
            return y
    if _USE_TRITON and y2.is_contiguous():
        _project_kernel[(y2.shape[0],)](y2, r32, w, H = h, BLOCK = _block(h))
    elif y2.dtype == torch.float32:
        d = torch.mv(y2, r32)
        y2.addr_(d, r32, alpha = -w)
    else:
        d = torch.mv(y2.float(), r32).to(y2.dtype)
        y2.addr_(d, r16, alpha = -w)
    return y
