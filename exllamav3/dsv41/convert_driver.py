"""
Calibrated conversion driver for DeepSeek-V4.1-Flash: the loop of `exllamav3.conversion.convert_model.main`
(load module -> capture Hessians over calibration rows -> quantize its linears -> reload quantized -> advance the
rows) specialised for the DS4.1 modules, with a pluggable quantizer and an MTP pass.

    from exllamav3.dsv41.convert_driver import run_conversion, StubQuantizer, Exl3Quantizer
    out = run_conversion(model_dir, rows, Exl3Quantizer(["cuda:0"]), bits = 2.3, device = "cuda:0")

* `rows`: list of [1, L] token tensors (calibration rows); L may differ per row.
* `quantizer(linear, H_data | None, K, seed) -> proxy_err`: replaces `linear.inner`. `StubQuantizer` (CPU, per-row
  uniform K-bit rounding, reports the Hessian-weighted proxy error) exercises the whole path without the EXL3 kernels;
  `Exl3Quantizer` calls `Linear.convert_exl3` (the real quantizer, needs the ROCm/CUDA extension).
* trunk modules are processed in order (embed, hc_expand, layers.0..n-1, norm, head); per-row cross-layer state
  lives in `quant_preserve["dsv41"]` (see calib.py). The MTP stages are calibrated afterwards from the target
  taps the trunk left in that state (DSparkCalib), unless `mtp = False`.
* output: dict module key -> {tensor name: tensor} (also written to `out_dir/<module>.safetensors` when given).
  The engram tables (`layers.N.engram.embed.*`) and the vision tower are not module tensors; the converter
  front-end copies them.
"""
from __future__ import annotations
import json
import os
import time
import torch

from ..model.config import Config
from ..model.model import Model
from ..modules import Linear
from ..modules.dsv41 import ModLin, DSV41Block
from ..modules.quant.fp16 import LinearFP16
from ..conversion.allocation import create_q_strategy
from . import ops
from .calib import DSparkCalib


# ------------------------------------------------------------------------------------------------ quantizers
class StubQuantizer:
    """CPU stand-in: per-output-row symmetric uniform rounding to K bits. Replaces `linear.inner` with a
    LinearFP16 of the rounded weight. Returns the proxy error tr(E H E^T) / tr(W H W^T) when a Hessian was
    captured (H counted and normalized like the real quantizer's finalize), else the plain relative rmse."""

    reload_quantized = False

    def __call__(self, lin: Linear, H_data, K: int, seed: int) -> float:
        inner = lin.inner
        assert isinstance(inner, LinearFP16)
        w = inner.weight.float()                                  # [in, out]
        qmax = 2 ** (K - 1) - 1
        scale = w.abs().amax(dim = 0, keepdim = True).clamp_min(1e-12) / max(qmax, 1)
        wq = (w / scale).round().clamp(-qmax - 1, max(qmax, 1)) * scale
        e = (w - wq)
        inner.weight = wq.to(inner.weight.dtype)
        if H_data is not None and H_data["count"] > 0:
            H = H_data["H"].float().to(w.device) / H_data["count"]
            num = torch.einsum("io,ij,jo->", e, H, e)
            den = torch.einsum("io,ij,jo->", w, H, w).clamp_min(1e-30)
            return float(num / den)
        return float(e.norm() / w.norm().clamp_min(1e-30))


class NoopQuantizer:
    """Leaves every linear unquantized (Hessian capture only; the rows advance through the reference weights)."""

    reload_quantized = False

    def __call__(self, lin, H_data, K, seed):
        return 0.0


class Exl3Quantizer:
    """The real EXL3 quantizer (Linear.convert_exl3 -> quantize_exl3). Linears swap their fp16 weight to host RAM
    first, like the main conversion loop."""

    reload_quantized = True

    def __init__(self, devices: list, codebook: str = "mul1", apply_out_scales = True, use_hessian: bool = True):
        self.devices = devices
        self.codebook = codebook
        self.apply_out_scales = apply_out_scales
        self.use_hessian = use_hessian          # False: uncalibrated quantization (A/B reference)

    def __call__(self, lin: Linear, H_data, K: int, seed: int) -> float:
        if lin.inner.swap_device is None and hasattr(lin.inner, "swap_cpu"):
            lin.inner.swap_cpu()
        qa = {"seed": seed, "K": K, "devices": self.devices, "device_ratios": None,
              "apply_out_scales": self.apply_out_scales}
        if self.codebook in ("mcg", "mul1"):
            qa[self.codebook] = True
        hd = H_data if (H_data is not None and self.use_hessian) else lin.init_H_data(False)
        return float(lin.convert_exl3(hd, quant_args = qa, progress_str = None))


# ------------------------------------------------------------------------------------------------ driver
def _rel_err(x, ref):
    x, ref = x.float().flatten(), ref.float().flatten().to(x.device)
    return float((x - ref).norm() / ref.norm().clamp_min(1e-30))


def _swap_H_to_host(capture_H: dict, device):
    if torch.device(device).type == "cpu":
        return
    for v in capture_H.values():
        v["H_swap_device"] = v["H"].device
        v["H"] = v["H"].cpu()


def _linears(module):
    return [m for m in module if isinstance(m, Linear) and m.qmap and m.device is not None]


def run_conversion(
    model_dir: str,
    rows: list,
    quantizer,
    bits: float = 2.0,
    head_bits: int = 6,
    mtp_bits: int = 3,
    hq: bool = False,
    strategy: dict | None = None,
    device = "cpu",
    moe_mode: str | None = None,
    mtp: bool = True,
    mtp_anchors: int = 8,
    num_ref_states: int = 2,
    max_blocks: int | None = None,
    out_dir: str | None = None,
    log = print,
    config_kwargs: dict | None = None,
    on_capture = None,
):
    """Returns {"tensors": {module: {name: tensor}}, "report": [per module dict], "strategy": {...}}.
    on_capture(module_key, capture_H): called after the Hessians of a module are captured (before quantization)."""
    torch.set_grad_enabled(False)
    device = torch.device(device)
    kw = dict(config_kwargs or {})
    if moe_mode:
        kw["dsv41_moe_mode"] = moe_mode
    config = Config.from_directory(model_dir, **kw)
    model = Model.from_config(config)
    mtp_model = Model.from_config(config, component = "mtp") if "mtp" in config.model_classes else None
    if strategy is None:
        strategy, final_bpw = create_q_strategy(model, mtp_model, config, bits, head_bits, mtp_bits, hq)
        log(f" -- strategy: {len(strategy)} linears, bpw (excluding head) {final_bpw:.3f}")

    n_rows = len(rows)
    for r in rows:
        assert r.dim() == 2 and r.shape[0] == 1 and 0 <= int(r.min()) and int(r.max()) < config.dsv41.vocab_size, \
            "calibration rows must be [1, L] token ids inside the vocabulary"
    if mtp and mtp_model is not None:
        assert 0 <= config.dsv41.dspark_noise_token_id < config.dsv41.vocab_size, "dspark noise token outside vocab"
    state = [r.clone() for r in rows]
    preserves = [{} for _ in range(n_rows)]
    out, report = {}, []
    all_experts = model.calibration_all_experts
    t_start = time.time()

    def run(module, i, capture, activate_all):
        params = {"attn_mode": "flash_attn_nc", "input_ids": rows[i], "quant_preserve": preserves[i]}
        if capture is not None:
            params["capture"] = capture
        if activate_all:
            params["activate_all_experts"] = True
        return module.forward(module.prepare_for_device(state[i], params), params)

    for idx, module in enumerate(model.modules):
        if max_blocks is not None and module.key.startswith("layers.") and int(module.key.split(".")[1]) >= max_blocks:
            break
        t0 = time.time()
        module.load(device)
        qmaps = module.get_qmaps()
        capture_H, refs = {}, {}
        if qmaps:
            for i in range(n_rows):
                run(module, i, capture_H, all_experts)
                if i < num_ref_states:
                    refs[i] = run(module, i, None, False).cpu()
            if on_capture is not None:
                on_capture(module.key, capture_H)
            _swap_H_to_host(capture_H, device)
        linears = _linears(module)
        proxies = []
        for lin in linears:
            assert lin.key in strategy, f"no quantization strategy for {lin.key}"
            K = strategy[lin.key]
            if K == 16:
                continue
            H = capture_H.get(lin.qmap) if qmaps else None
            proxies.append(quantizer(lin, H, K, idx))
        q_tensors = {}
        for m in module:
            q_tensors.update(m.get_tensors())
        out[module.key] = {k: v.detach().cpu().clone() for k, v in q_tensors.items()}
        if getattr(quantizer, "reload_quantized", False) and linears:
            config.stc.set_new_tensors(q_tensors)
            module.load(device, source = q_tensors)
            config.stc.set_new_tensors(None)
        err = 0.0
        n_err = 0
        for i in range(n_rows):
            rs = run(module, i, None, False)
            state[i] = rs.cpu()
            if i in refs:
                err += _rel_err(state[i], refs[i])
                n_err += 1
        rec = {
            "module": module.key, "linears": len(linears), "qmaps": len(qmaps), "state_rel_err": err / max(n_err, 1),
            "proxy_mean": sum(proxies) / len(proxies) if proxies else None, "seconds": time.time() - t0,
            "hessian_counts": {k: v["count"] for k, v in capture_H.items()},
        }
        report.append(rec)
        log(f" -- {module.key:12s} lin {rec['linears']:4d}  state err {rec['state_rel_err']:.4f}  "
            f"proxy {rec['proxy_mean'] if rec['proxy_mean'] is None else round(rec['proxy_mean'], 4)}  "
            f"{rec['seconds']:.1f}s")
        module.unload()
        config.stc.close()
        if out_dir:
            _save(out_dir, module.key, out[module.key])

    if mtp and mtp_model is not None and max_blocks is None:
        _mtp_pass(config, model, mtp_model, rows, preserves, quantizer, strategy, device, mtp_anchors, out, report,
                  out_dir, log, on_capture)
    log(f" -- done, {time.time() - t_start:.1f}s")
    if out_dir:
        with open(os.path.join(out_dir, "driver_report.json"), "w") as f:
            json.dump({"report": report, "strategy": strategy}, f, indent = 1)
    return {"tensors": out, "report": report, "strategy": strategy}


def _save(out_dir, key, tensors):
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok = True)
    save_file({k: v.contiguous() for k, v in tensors.items()}, os.path.join(out_dir, f"{key}.safetensors"),
              metadata = {"format": "pt"})


# ------------------------------------------------------------------------------------------------ MTP
def _mtp_pass(config, model, mtp_model, rows, preserves, quantizer, strategy, device, n_anchors, out, report,
              out_dir, log, on_capture=None):
    """Calibrate the dspark stages: draft blocks anchored at row positions (see DSparkCalib), stage by stage.
    The target taps come from the quantized trunk's final pass over each row."""
    d = config.dsv41
    n = d.num_hidden_layers
    stages = [m for m in mtp_model.modules if isinstance(m, DSV41Block)]
    extra = mtp_model.modules[-1]
    n_rows = len(rows)
    emb_w = config.stc.get_tensor("embed.weight", device, allow_bf16 = True).float()
    extra.load(device)
    main_proj = [m for m in extra if isinstance(m, Linear)][0]
    main_norm = extra.t["mtp.0.main_norm.weight"].float()
    anchors = [_anchors(r.shape[1], n_anchors) for r in rows]

    # ---- main_proj: input = target taps of every position
    t0 = time.time()
    cap = {}
    for i in range(n_rows):
        ho = preserves[i]["dsv41"]
        mh = torch.cat([ho["main"][l].float() for l in d.dspark_target_layer_ids], dim = -1).to(device)
        ModLin(main_proj, {"capture": cap})(mh)
    if on_capture is not None:
        on_capture(extra.key, cap)
    _swap_H_to_host(cap, device)
    proxy = quantizer(main_proj, cap.get(main_proj.qmap), strategy[main_proj.key], 1000) \
        if strategy[main_proj.key] != 16 else 0.0
    q = {}
    for m in extra:
        q.update(m.get_tensors())
    out[extra.key] = {k: v.detach().cpu().clone() for k, v in q.items()}
    if getattr(quantizer, "reload_quantized", False):
        config.stc.set_new_tensors(q)
        main_proj.load(device, source = q)
        config.stc.set_new_tensors(None)
    report.append({"module": extra.key, "linears": 1, "proxy_mean": proxy, "seconds": time.time() - t0,
                   "hessian_counts": {k: v["count"] for k, v in cap.items()}, "qmaps": 1, "state_rel_err": 0.0})
    log(f" -- {extra.key:12s} lin    1  proxy {round(proxy, 4)}  main_proj rows {cap[main_proj.qmap]['count']}")

    # per-row main_x (with the quantized main_proj) and draft state
    mx = []
    for i in range(n_rows):
        ho = preserves[i]["dsv41"]
        mh = torch.cat([ho["main"][l].float() for l in d.dspark_target_layer_ids], dim = -1).to(device)
        mx.append(ops.rmsnorm(ModLin(main_proj, {})(mh), main_norm, d.rms_norm_eps).cpu())
    hstate = [None] * n_rows
    for j, block in enumerate(stages):
        l = n + j
        t0 = time.time()
        block.load(device)
        capture_H, refs = {}, {}
        stage_state = {}

        def stage_run(i, params):
            g = block.bind(params)
            dc = DSparkCalib(g, emb_w, None, None)
            ids = rows[i].to(device)
            ancs = anchors[i]
            if not ancs:
                block.unbind()
                return None
            mkv = dc.stage_mkv(l, mx[i].to(device))
            if hstate[i] is None:
                h, pm = dc.draft_input(ids, ancs)
            else:
                h, pm = hstate[i][0].to(device), hstate[i][1].to(device)
            res = dc.run_stage(l, h, pm, mkv, ancs)
            block.unbind()
            return res

        for i in range(n_rows):
            stage_run(i, {"capture": capture_H, "activate_all_experts": True})
            if i < 2:
                res = stage_run(i, {})
                if res is not None:
                    refs[i] = res[0].cpu()
        if on_capture is not None:
            on_capture(block.key, capture_H)
        _swap_H_to_host(capture_H, device)
        linears = _linears(block)
        proxies = [quantizer(lin, capture_H.get(lin.qmap), strategy[lin.key], 2000 + j) for lin in linears
                   if strategy[lin.key] != 16]
        q = {}
        for m in block:
            q.update(m.get_tensors())
        out[block.key] = {k: v.detach().cpu().clone() for k, v in q.items()}
        if getattr(quantizer, "reload_quantized", False):
            config.stc.set_new_tensors(q)
            block.load(device, source = q)
            config.stc.set_new_tensors(None)
        err, ne = 0.0, 0
        for i in range(n_rows):
            res = stage_run(i, {})
            if res is None:
                continue
            hstate[i] = (res[0].cpu(), res[1].cpu())
            if i in refs:
                err += _rel_err(res[0], refs[i])
                ne += 1
        rec = {"module": block.key, "linears": len(linears), "qmaps": len(capture_H), "state_rel_err": err / max(ne, 1),
               "proxy_mean": sum(proxies) / len(proxies) if proxies else None, "seconds": time.time() - t0,
               "hessian_counts": {k: v["count"] for k, v in capture_H.items()}}
        report.append(rec)
        log(f" -- {block.key:12s} lin {rec['linears']:4d}  state err {rec['state_rel_err']:.4f}  "
            f"proxy {rec['proxy_mean'] if rec['proxy_mean'] is None else round(rec['proxy_mean'], 4)}  "
            f"{rec['seconds']:.1f}s")
        block.unload()
        config.stc.close()
        if out_dir:
            _save(out_dir, block.key, out[block.key])
    if out_dir:
        _save(out_dir, extra.key, out[extra.key])
    extra.unload()


def _anchors(L: int, n: int) -> list:
    lo, hi = 1, L - 2
    if hi < lo:
        return []
    n = min(n, hi - lo + 1)
    return sorted({lo + int(round(i * (hi - lo) / max(n - 1, 1))) for i in range(n)})
