from __future__ import annotations
from typing_extensions import override
import torch
from ..util.tensor import to2
from ..model.config import Config
from . import Module, RMSNorm, LayerNorm, Attention, GatedDeltaNet, GatedMLP, MLP, BlockSparseMLP, Linear
from .hyperconnections import HyperConnection
from . import ablit_runtime
from ..util import profile_opt

class TransformerBlock(Module):

    def __init__(
        self,
        config: Config | None,
        key: str,
        layer_idx: int | None = None,
        attn_norm: RMSNorm | LayerNorm | None = None,
        attn: Attention | GatedDeltaNet | None = None,
        attn_post_norm: RMSNorm | LayerNorm | None = None,
        mlp_norm: RMSNorm | LayerNorm | None = None,
        mlp: MLP | GatedMLP | BlockSparseMLP | None = None,
        mlp_post_norm: RMSNorm | LayerNorm | None = None,
        attn_hc: HyperConnection | None = None,
        mlp_hc: HyperConnection | None = None,
        key_layer_scalar: str | None = None,
        key_attn_resid_scalar: str | None = None,
        key_mlp_resid_scalar: str | None = None,
        qmap: str | None = None,
        qbits_key: str = "bits",
        out_dtype: torch.dtype = None
    ):
        super().__init__(config, key, None)

        self.layer_idx = layer_idx
        self.attn_norm = attn_norm
        self.attn = attn
        self.attn_post_norm = attn_post_norm
        self.mlp_norm = mlp_norm
        self.mlp = mlp
        self.mlp_post_norm = mlp_post_norm
        self.attn_hc = attn_hc
        self.mlp_hc = mlp_hc
        self.qbits_key = qbits_key
        self.out_dtype = out_dtype

        self.key_layer_scalar = key_layer_scalar
        self.key_attn_resid_scalar = key_attn_resid_scalar
        self.key_mlp_resid_scalar = key_mlp_resid_scalar
        self.layer_scalar_t = None
        self.layer_scalar_f = None
        self.attn_resid_scalar = None
        self.mlp_resid_scalar = None
        self.ablit = None  # EXL3_ABLIT_RUNTIME: (w_attn, w_mlp, r_fp32, r_fp16), set in load()

        # Hyperconnection sites (mHC): the block's residual is (bsz, seq, hc_mult, hidden)
        # fp32 streams, mixed at each sublayer site instead of the plain residual add
        if attn_hc is not None or mlp_hc is not None:
            assert attn_hc is not None and mlp_hc is not None, \
                "hyperconnections require both attn_hc and mlp_hc"
            assert all(v is None for v in (
                attn_post_norm, mlp_post_norm,
                key_layer_scalar, key_attn_resid_scalar, key_mlp_resid_scalar,
            )), \
                "hyperconnections cannot combine with residual scalars/post-norms"

        self.register_submodule(self.attn_hc)
        self.register_submodule(self.attn_norm)
        self.register_submodule(self.attn)
        self.register_submodule(self.attn_post_norm)
        self.register_submodule(self.mlp_hc)
        self.register_submodule(self.mlp_norm)
        self.register_submodule(self.mlp)
        self.register_submodule(self.mlp_post_norm)

        self.num_slices = mlp.num_slices if mlp else 1


    @override
    def optimizer_targets(self):
        a = self.attn.optimizer_targets() if self.attn else []
        m = self.mlp.optimizer_targets() if self.mlp else []
        return [a, m]

    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        ablit_runtime.prepare(self, device)
        if self.key_layer_scalar:
            self.layer_scalar_t = self.config.stc.get_tensor(
                self.key + "." + self.key_layer_scalar,
                None,
                allow_bf16 = True,
                no_defer = True,
            )
            assert self.layer_scalar_t.numel() == 1
            self.layer_scalar_f = self.layer_scalar_t.float().item()

        # TODO: Residual scalar tensors could be baked into preceding modules for models that use them
        #       (currently only Step3.7 vision tower)
        if self.key_attn_resid_scalar:
            self.attn_resid_scalar = self.config.stc.get_tensor(
                self.key + "." + self.key_attn_resid_scalar,
                device,
                allow_bf16 = True,
                no_defer = True,
            )
        if self.key_mlp_resid_scalar:
            self.mlp_resid_scalar = self.config.stc.get_tensor(
                self.key + "." + self.key_mlp_resid_scalar,
                device,
                allow_bf16 = True,
                no_defer = True,
            )

    def unload(self):
        super().unload()
        self.layer_scalar_t = None
        self.attn_resid_scalar = None
        self.mlp_resid_scalar = None

    def get_tensors(self):
        t = {}
        if self.key_layer_scalar is not None:
            t[self.key + "." + self.key_layer_scalar] = self.layer_scalar_t.data.contiguous()
        if self.key_attn_resid_scalar is not None:
            t[self.key + "." + self.key_attn_resid_scalar] = self.attn_resid_scalar.data.contiguous()
        if self.key_mlp_resid_scalar is not None:
            t[self.key + "." + self.key_mlp_resid_scalar] = self.mlp_resid_scalar.data.contiguous()
        return t

    def weights_numel(self):
        return (
            super().weights_numel() +
            (1 if self.key_layer_scalar is not None else 0) +
            (self.attn_resid_scalar.numel() if self.attn_resid_scalar is not None else 0) +
            (self.mlp_resid_scalar.numel() if self.mlp_resid_scalar is not None else 0)
        )

    def _forward_mlp(self, x: torch.Tensor, y_resid: torch.Tensor | None, params: dict,
                     hc_pending = None, hc_defer: bool = False) -> torch.Tensor:
        """MLP half of forward (norm, MLP, residual/hc apply). `y_resid` is a pending attention
        output whose residual add is folded into the MLP input norm. Returns the new residual."""
        if self.mlp:
            if self.mlp_hc:
                if hc_pending is not None and not self.mlp_norm:
                    HyperConnection.flush_pending(x, hc_pending)
                    hc_pending = None
                fused = self.mlp_hc.mix_norm(x, params, self.mlp_norm, hc_pending) if self.mlp_norm else None
                if fused is not None:
                    hc_post, hc_comb, y = fused
                else:
                    hc_post, hc_comb, y = self.mlp_hc.mix(x, params)
                    y = y.half()
                    if self.mlp_norm:
                        y = self.mlp_norm.forward(y, params, out_dtype = torch.half)
            else:
                params["residual"] = x
                if y_resid is not None:
                    # Decode (batch-1 or DFlash verify, R<=MOE_R_MAX): the MLP may fuse this
                    # pre-norm into its router launch(es) -- dec_norm_route_r loops the real
                    # batch-1 kernel per row for R>1, so it's bit-exact by construction (see its
                    # docstring); it also covers R==1 (self.mlp.dec_norm_route's own path).
                    fused_r = getattr(self.mlp, "dec_norm_route_r", None)
                    y = None
                    if fused_r is not None:
                        try:
                            xa_r = y_resid.view(-1, self.mlp.hidden_size)
                            r_r = x.view(-1, self.mlp.hidden_size)
                        except RuntimeError:
                            xa_r = r_r = None
                        if xa_r is not None:
                            y = fused_r(self.mlp_norm, xa_r, r_r, params)
                            if y is not None:
                                y = y.view(y_resid.shape)
                    if y is None:
                        y = self.mlp_norm.forward(y_resid, params, out_dtype = torch.half, residual_in = x)
                elif self.mlp_norm:
                    y = self.mlp_norm.forward(x, params, out_dtype = torch.half)
                else:
                    y = x.half()
                # Attention modules that return flattened (rows, dim) output (MLA decode) hand a
                # 2D y_resid to the fused residual norm; the MLP (shared experts) expects x's shape
                if y.shape != x.shape and y.numel() == x.numel():
                    y = y.view(x.shape)
            # Plain residual add after the MLP: offer the residual so a fused MLP kernel can add
            # into it directly (it then sets mlp_residual_done and y is not added again)
            plain_add = self.mlp_resid_scalar is None and not self.mlp_hc and not self.mlp_post_norm \
                and not self.ablit
            if plain_add:
                params["mlp_residual_out"] = x
            y = self.mlp.forward(y, params)
            if plain_add:
                params.pop("mlp_residual_out", None)
            if self.ablit:
                ablit_runtime.project(y, self.ablit[1], *self.ablit[2:])
            if self.mlp_resid_scalar is not None:
                y *= self.mlp_resid_scalar
            if self.mlp_hc:
                # EXL3_PF_HC_FUSE: defer into the next block's attn mix_norm (see pf_can_defer)
                if hc_defer and HyperConnection.pf_can_defer(x, y, hc_post, hc_comb, params):
                    params["hc_pending"] = (x, y, hc_post, hc_comb)
                else:
                    x = self.mlp_hc.apply_(x, y, hc_post, hc_comb, params)
            elif self.mlp_post_norm:
                self.mlp_post_norm.forward(y, params, residual = x)
            elif params.pop("mlp_residual_done", False):
                pass
            else:
                x += y
        return x


    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        # Opt-in per-block HIP graph capture (EXL3_BLOCK_GRAPH, ROCm decode only). Declines
        # return None and fall through to the eager path below.
        # EXL3_PF_HC_FUSE: a residual apply deferred by the previous block (prefill only, so
        # never combined with the decode block graph)
        hc_pending = params.pop("hc_pending", None)
        if hc_pending is None:
            from .block_graph import maybe_graph_forward
            y_graph = maybe_graph_forward(self, x, params)
            if y_graph is not None:
                return y_graph

        export_state = params.get("export_state_layers")
        export_state = export_state and self.layer_idx in export_state and params.get("layer_instance", 0) == 0
        if hc_pending is not None and not (self.attn and self.attn_hc and self.attn_norm):
            HyperConnection.flush_pending(x, hc_pending)
            hc_pending = None
        hc_defer = not export_state and self.layer_scalar_f is None

        y_resid = None  # pending attn output whose residual add is folded into the MLP input norm
        mlp_pending = None

        if self.attn:
            if self.attn_hc:
                fused = self.attn_hc.mix_norm(x, params, self.attn_norm, hc_pending) if self.attn_norm else None
                if fused is not None:
                    hc_post, hc_comb, y = fused
                else:
                    hc_post, hc_comb, y = self.attn_hc.mix(x, params)
                    y = y.half()
                    if self.attn_norm:
                        y = self.attn_norm.forward(y, params, out_dtype = torch.half)
            elif self.attn_norm:
                y = self.attn_norm.forward(x, params, out_dtype = torch.half)
            else:
                y = x.half()
            y = self.attn.forward(y, params)
            if y is None:
                # EXL3_PF_SKIP: kv_only -- attention wrote its K/V (and indexer) rows and the
                # caller discards this block's output, so nothing below has an input to read
                return None
            if params.get("prefill") and not export_state:
                return x
            if self.ablit:
                ablit_runtime.project(y, self.ablit[0], *self.ablit[2:])
            if self.attn_resid_scalar is not None:
                y *= self.attn_resid_scalar
            if self.attn_hc:
                if self.mlp and self.mlp_hc and self.mlp_norm \
                        and HyperConnection.pf_can_defer(x, y, hc_post, hc_comb, params):
                    mlp_pending = (x, y, hc_post, hc_comb)
                else:
                    x = self.attn_hc.apply_(x, y, hc_post, hc_comb, params)
            elif self.attn_post_norm:
                self.attn_post_norm.forward(y, params, residual = x)
            elif self.mlp is not None and self.mlp_norm is not None and self.mlp_norm.can_fuse_residual(x, y):
                y_resid = y
            else:
                x += y

        x = self._forward_mlp(x, y_resid, params, mlp_pending, hc_defer)

        if export_state:
            s = params.get("export_states")
            if not s:
                s = params["export_states"] = []
            # With hyperconnections the residual is a stream stack; export the stream mean as
            # the collapsed hidden state (streams start as broadcast copies of the embedding)
            x_ = x.mean(dim = 2) if self.attn_hc else x
            if x_.dtype == torch.half:
                s.append(x_.clamp_(-65504.0, 65504.0))
            else:
                x_ = x_.half()
                x_.clamp_(-65504.0, 65504.0)
                s.append(x_)

        if self.layer_scalar_f is not None:
            x *= self.layer_scalar_f

        return to2(x, out_dtype, self.out_dtype)


    def get_name(self):
        name = super().get_name()
        if not self.attn and not self.mlp:
            name += " (no-op)"
        return name


    def tp_export(self, plan, producer):
        assert self.device is not None, "Cannot export module for TP before loading."

        def _export(child):
            nonlocal producer
            return child.tp_export(plan, producer) if child is not None else None

        return {
            "cls": TransformerBlock,
            "kwargs": {
                "key": self.key,
                "layer_idx": self.layer_idx,
                "out_dtype": self.out_dtype,
                "key_layer_scalar": self.key_layer_scalar,
                "key_attn_resid_scalar": self.key_attn_resid_scalar,
                "key_mlp_resid_scalar": self.key_mlp_resid_scalar,
            },
            **{name: _export(getattr(self, name, None)) for name in (
                "attn_hc",
                "attn_norm",
                "attn",
                "attn_post_norm",
                "mlp_hc",
                "mlp_norm",
                "mlp",
                "mlp_post_norm",
            )},
            # Per-layer scalars load from the tensor collection, which TP children don't have
            "layer_scalar_f": self.layer_scalar_f,
            "attn_resid_scalar": producer.send(self.attn_resid_scalar),
            "mlp_resid_scalar": producer.send(self.mlp_resid_scalar),
            "device": self.device,
        }


    @staticmethod
    def tp_import(local_context, exported, plan):
        consumer = local_context["consumer"]
        device = local_context["device"]

        def _import(name):
            nonlocal exported, plan
            return exported[name]["cls"].tp_import(local_context, exported[name], plan) \
                if exported.get(name) else None

        module = TransformerBlock(
            config = None,
            **exported["kwargs"],
            attn_hc = _import("attn_hc"),
            attn_norm = _import("attn_norm"),
            attn = _import("attn"),
            attn_post_norm = _import("attn_post_norm"),
            mlp_hc = _import("mlp_hc"),
            mlp_norm = _import("mlp_norm"),
            mlp = _import("mlp"),
            mlp_post_norm = _import("mlp_post_norm"),
        )

        module.layer_scalar_f = exported.get("layer_scalar_f")
        module.attn_resid_scalar = consumer.recv(exported.get("attn_resid_scalar"), cuda = True)
        module.mlp_resid_scalar = consumer.recv(exported.get("mlp_resid_scalar"), cuda = True)
        module.device = device
        return module


class ParallelDecoderBlock(Module):

    def __init__(
        self,
        config: Config | None,
        key: str,
        layer_idx: int | None = None,
        input_norm: RMSNorm | LayerNorm | None = None,
        attn: Attention | None = None,
        mlp: MLP | GatedMLP | None = None,
        qmap: str | None = None,
        qbits_key: str = "bits",
        out_dtype: torch.dtype = None
    ):
        super().__init__(config, key, None)

        self.layer_idx = layer_idx
        self.input_norm = input_norm
        self.attn = attn
        self.mlp = mlp
        self.qbits_key = qbits_key
        self.out_dtype = out_dtype

        self.register_submodule(self.input_norm)
        self.register_submodule(self.attn)
        self.register_submodule(self.mlp)

        self.num_slices = mlp.num_slices if mlp else 1

        self.tp_reduce = False


    @override
    def optimizer_targets(self):
        a = self.attn.optimizer_targets() if self.attn else []
        m = self.mlp.optimizer_targets() if self.mlp else []
        return [a, m]


    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:

        y = self.input_norm.forward(x, params, out_dtype = torch.half)
        y1 = self.attn.forward(y, params)
        if not params.get("prefill"):
            y2 = self.mlp.forward(y, params)
            y1 += y2

            if self.tp_reduce:
                params["backend"].all_reduce(y1)

            x += y1

        return to2(x, out_dtype, self.out_dtype)


    def get_name(self):
        name = super().get_name()
        if not self.attn and not self.mlp:
            name += " (no-op)"
        return name


    def tp_export(self, plan, producer):
        assert self.device is not None, "Cannot export module for TP before loading."

        def _export(child):
            nonlocal producer
            return child.tp_export(plan, producer) if child is not None else None

        return {
            "cls": ParallelDecoderBlock,
            "kwargs": {
                "key": self.key,
                "layer_idx": self.layer_idx,
                "out_dtype": self.out_dtype,
            },
            **{name: _export(getattr(self, name, None)) for name in (
                "input_norm",
                "attn",
                "mlp",
            )},
            "device": self.device,
        }


    @staticmethod
    def tp_import(local_context, exported, plan):
        device = local_context["device"]

        def _import(name, **kwargs):
            nonlocal exported, plan
            return exported[name]["cls"].tp_import(local_context, exported[name], plan, **kwargs) \
                if exported.get(name) else None

        module = ParallelDecoderBlock(
            config = None,
            **exported["kwargs"],
            input_norm = _import("input_norm"),
            attn = _import("attn", skip_reduction = True),
            mlp = _import("mlp", skip_reduction = True),
        )
        module.device = device

        # Use single reduction for sum of mlp and attn
        module.tp_reduce = True
        return module
