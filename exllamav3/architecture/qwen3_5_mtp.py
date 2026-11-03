from __future__ import annotations
from typing_extensions import override
import os
import torch
import weakref

from ..cache import Cache
from ..ext import exllamav3_ext as ext
from ..model.config import Config, no_default
from ..model.model import Model
from ..util.rope import RopeStyle
from ..modules import RMSNorm, Embedding, TransformerBlock, Attention, GatedMLP, Linear, BlockSparseMLP
from ..modules.arch_specific.qwen3_5_mtp import Qwen3_5MTPInputLayer
from ..modules.attn import prepare_for_attn
from ..util.tensor import get_for_device

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .qwen3_5 import Qwen3_5Config, Qwen3_5MoeConfig


def exl3_block_subhead(inner, blocks):
    """The EXL3 linear `inner` restricted to the given 128-row output blocks (sorted LongTensor): output column j of the
    result is column blocks[j // 128] * 128 + j % 128 of `inner`, bit for bit."""
    from ..modules.quant.exl3 import LinearEXL3
    dev = inner.trellis.device
    rows = (blocks.to(dev)[:, None] * 128 + torch.arange(128, device = dev)).reshape(-1)
    tiles = (blocks.to(dev)[:, None] * 8 + torch.arange(8, device = dev)).reshape(-1)
    return LinearEXL3(None, inner.in_features, rows.numel(), suh = inner.suh, svh = inner.svh[rows].contiguous(),
                      trellis = inner.trellis[:, tiles].contiguous(), mcg = inner.mcg_tensor,
                      mul1 = inner.mul1_tensor, out_dtype = inner.out_dtype)


class Qwen3_5MTPModel(Model):

    def __init__(
        self,
        config: Qwen3_5Config | Qwen3_5MoeConfig,
        use_moe: bool = False,
        **kwargs
    ):
        super().__init__(config, **kwargs)
        self.use_moe = use_moe

        # Module list: optional embed, then pre_fc norms + fc, then num_mtp_layers * TransformerBlock, then norm
        self.input_layer = Qwen3_5MTPInputLayer(
            config = config,
            key = "mtp",
            key_pre_fc_norm_hidden = "mtp.pre_fc_norm_hidden",
            key_pre_fc_norm_embedding = "mtp.pre_fc_norm_embedding",
            key_fc = "mtp.fc",
            hidden_size = config.hidden_size,
            rms_norm_eps = config.rms_norm_eps,
            native_draft_len = 1,
            out_dtype = torch.float,
            qbits_key = "mtp_bits",
        )

        self.modules = [self.input_layer]

        self.first_block_idx = len(self.modules)
        self.attn_modules = []

        for idx in range(config.mtp_num_hidden_layers):
            attn = Attention(
                config = config,
                key = f"mtp.layers.{idx}.self_attn",
                layer_idx = idx,
                hidden_size = config.hidden_size,
                head_dim = config.head_dim,
                num_q_heads = config.num_q_heads,
                num_kv_heads = config.num_kv_heads,
                rope_settings = config.rope_settings,
                sm_scale = None,
                key_q = "q_proj",
                key_k = "k_proj",
                key_v = "v_proj",
                key_o = "o_proj",
                qmap = "block.attn",
                q_norm = RMSNorm(
                    config = config,
                    key = f"mtp.layers.{idx}.self_attn.q_norm",
                    rms_norm_eps = config.rms_norm_eps,
                    constant_bias = 1.0,
                ),
                k_norm = RMSNorm(
                    config = config,
                    key = f"mtp.layers.{idx}.self_attn.k_norm",
                    rms_norm_eps = config.rms_norm_eps,
                    constant_bias = 1.0,
                ),
                out_dtype = torch.float,
                interleaved_gate = True,
                qbits_key = "mtp_bits",
            )
            self.attn_modules.append(attn)

            self.modules.append(
                TransformerBlock(
                    config = config,
                    key = f"mtp.layers.{idx}",
                    layer_idx = idx,
                    attn_norm = RMSNorm(
                        config = config,
                        key = f"mtp.layers.{idx}.input_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                        constant_bias = 1.0,
                    ),
                    attn = attn,
                    mlp_norm = RMSNorm(
                        config = config,
                        key = f"mtp.layers.{idx}.post_attention_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                        constant_bias = 1.0,
                    ),
                    mlp = (
                        BlockSparseMLP(
                            config = config,
                            key = f"mtp.layers.{idx}.mlp",
                            hidden_size = config.hidden_size,
                            intermediate_size = config.moe_intermediate_size,
                            num_experts = config.num_experts,
                            num_experts_per_tok = config.num_experts_per_tok,
                            key_up = "experts.{expert_idx}.up_proj",
                            key_gate = "experts.{expert_idx}.gate_proj",
                            key_down = "experts.{expert_idx}.down_proj",
                            key_gate_up_split = "experts.gate_up_proj",
                            key_down_split = "experts.down_proj",
                            key_routing_gate = "gate",
                            key_shared_gate = "shared_expert_gate",
                            transpose_fused_weights = False,
                            qmap = "block.mlp",
                            interm_dtype = torch.half,
                            out_dtype = torch.float,
                            qbits_key = "mtp_bits",
                            shared_experts = GatedMLP(
                                config = config,
                                key = f"mtp.layers.{idx}.mlp.shared_expert",
                                hidden_size = config.hidden_size,
                                intermediate_size = config.shared_expert_intermediate_size,
                                key_up = "up_proj",
                                key_gate = "gate_proj",
                                key_down = "down_proj",
                                qmap = "block.mlp",
                                interm_dtype = torch.half,
                                out_dtype = torch.float,
                                qbits_key = "mtp_bits",
                                select_hq_bits = 2,
                            )
                        ) if use_moe else
                        GatedMLP(
                            config = config,
                            key = f"mtp.layers.{idx}.mlp",
                            hidden_size = config.hidden_size,
                            intermediate_size = config.intermediate_size,
                            key_up = "up_proj",
                            key_gate = "gate_proj",
                            key_down = "down_proj",
                            qmap = "block.mlp",
                            interm_dtype = torch.half,
                            out_dtype = torch.float,
                            qbits_key = "mtp_bits",
                        )
                    ),
                )
            )

        self.last_kv_module_idx = len(self.modules) - 1

        # Final norm
        self.final_norm = RMSNorm(
            config = config,
            key = "mtp.norm",
            rms_norm_eps = config.rms_norm_eps,
            out_dtype = torch.half,
            constant_bias = 1.0,
        )
        self.modules.append(self.final_norm)

        self.caps.update({
            "supports_tp": False,
            "attach_target": True,
            "mtp_draft": True,
            "default_draft_size": 4,  # best measured performance
            "autosplit_load_fwd": False,
        })

        # Cross-references populated by attach_to()
        self.target_embed = None
        self.target_lm_head = None
        self.attached_model = None


    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        # MTP doesn't take input_ids through Embedding here — embedding is handled in step()
        # But prepare_for_attn still wires up flash-attn params
        return prepare_for_attn(input_ids, params)


    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        raise NotImplementedError("MTP draft model does not have its own chat template")


    def attach_to(self, target):
        """
        Bind to target model: borrow embed_tokens / lm_head and tell target to export hidden state.

        Qwen3.5/3.6 MTP consumes the trunk's post-final-norm hidden state. This differs from
        DeepSeek/GLM-style MTP heads, which consume a pre-norm residual stream.
        """
        self.input_layer.attached_model = weakref.ref(target)
        self.attached_model = weakref.ref(target)

        # Find the target's embedding (first module of class Embedding)
        target_embed = None
        for m in target.modules:
            if isinstance(m, Embedding):
                target_embed = m
                break
        assert target_embed is not None, "Could not locate target's Embedding module"
        self.target_embed = weakref.ref(target_embed)

        # lm_head is the last module
        assert isinstance(target.modules[-1], Linear), "Expected Linear lm_head as last target module"
        self.target_lm_head = weakref.ref(target.modules[-1])

        target_norm = target.modules[target.logit_layer_idx - 1]
        assert isinstance(target_norm, RMSNorm), "Expected target final RMSNorm immediately before lm_head"
        self.draft_verifier_params.update({
            "export_state_norm_keys": {target_norm.key},
        })


    def default_load_shape_dtype(self, chunk_size):
        return (1, 1), torch.long


    def default_load_params(self, max_chunk_size):
        return {}


    @staticmethod
    def _head_plain(lm):
        inner = getattr(lm, "inner", None)
        return inner is not None and type(inner).__name__ == "LinearEXL3" and isinstance(inner.K, int) \
            and inner.bias is None and lm.pre_scale == 1.0 and lm.post_scale == 1.0 and lm.softcap == 0.0

    def _draft_subhead(self, lm, blocks = None):
        """The draft head restricted to whole 128-row blocks of the vocabulary (the head's output Hadamard works on 128-row
        blocks, so any set of whole blocks is a self-contained EXL3 linear with the same trellis tiles). blocks = None: the
        first EXL3_DRAFT_VOCAB rows. Else a sorted LongTensor of block numbers. None when off or when the head is not a plain
        integer-K EXL3 linear. Every draft is verified by the target, so a draft that misses a rare id only costs an accept,
        never changes the text."""
        n = int(os.environ.get("EXL3_DRAFT_VOCAB", "0") or 0) // 128 * 128
        if n <= 0:
            return None
        if not self._head_plain(lm):
            return None
        inner = lm.inner
        from ..modules.quant.exl3 import LinearEXL3
        if blocks is None:
            cache = self.__dict__.setdefault("_dsub", {})
            if n in cache:
                return cache[n]
            sub = None
            if n < lm.out_features:
                sub = LinearEXL3(None, inner.in_features, n, suh = inner.suh, svh = inner.svh[:n].contiguous(),
                                 trellis = inner.trellis[:, :n // 16].contiguous(), mcg = inner.mcg_tensor,
                                 mul1 = inner.mul1_tensor, out_dtype = inner.out_dtype)
            cache[n] = sub
            return sub
        return exl3_block_subhead(inner, blocks)

    def lazy_vram_bytes(self):
        """Device memory the adaptive draft-head subset takes at its first build (about 2048 draft rows into the first request),
        at most EXL3_DRAFT_VOCAB rows of the head. 0 when off, not plain, or already built. The start-up VRAM guard counts it:
        measured 3090 Qwen3.8-27B (98304 rows): 150 MiB that appeared during the first 32K request and tipped the second
        distinct 32K prompt into CUDA out of memory, 48 MiB short."""
        if int(os.environ.get("EXL3_DRAFT_VOCAB", "0") or 0) < 128 or os.environ.get("EXL3_DRAFT_ADAPT", "1") == "0":
            return 0
        model = self.attached_model()
        lm = model.modules[model.logit_layer_idx]
        if not self._head_plain(lm) or (self.__dict__.get("_dad") or {}).get("sub") is not None:
            return 0
        inner = lm.inner
        n = min(int(os.environ.get("EXL3_DRAFT_VOCAB", "0")) // 128 * 128, lm.out_features)
        return inner.trellis.numel() * inner.trellis.element_size() * n // lm.out_features + n * inner.svh.element_size()

    def _adapt_state(self, lm):
        """Adaptive draft-head subset (EXL3_DRAFT_ADAPT, default on when EXL3_DRAFT_VOCAB is set; 0 = the fixed prefix).
        Counts the head's own argmax ids per 128-id block (every step while the full head runs, the probe steps while the
        subset runs), then keeps the most frequent blocks that cover EXL3_DRAFT_COVER (default 0.99) of the counts, at most
        EXL3_DRAFT_VOCAB rows. Keyed on the model's own token statistics, no language or model list."""
        st = self.__dict__.get("_dad")
        if st is None:
            nb = lm.out_features // 128
            lim = getattr(self, "vocab_limit", None) or self.attached_model().config.vocab_size
            st = self.__dict__["_dad"] = {
                "counts": torch.zeros(nb, dtype = torch.float32, device = lm.inner.trellis.device),
                "seen": 0, "built_at": 0, "numeric": 0.0, "member": None, "probes": 0, "misses": 0.0, "sub": None, "gmap": None, "ok_blocks": lim // 128,
                "calls": 0, "ema": 0.0, "on": False, "nbuilt": 0, "vs": self.attached_model().config.vocab_size,
                "warm": int(os.environ.get("EXL3_DRAFT_WARM", "2048") or 2048),
                "maxmiss": float(os.environ.get("EXL3_DRAFT_MAXMISS", "0.05") or 0.05),
                "log": bool(os.environ.get("EXL3_DRAFT_ADAPT_LOG")),
                "every": int(os.environ.get("EXL3_DRAFT_PROBE", "16") or 16),     # the probe runs the full head too: 1.0 ms at 4 rows
            }
            path = os.environ.get("EXL3_DRAFT_COUNTS")
            if path and os.path.exists(path):
                try:
                    c = torch.load(path, map_location = st["counts"].device)
                    if c.shape == st["counts"].shape:
                        prior = float(os.environ.get("EXL3_DRAFT_PRIOR", "20000") or 20000)    # the file weighs this many rows: the live ids take over
                        st["counts"] += c.float() * (prior / max(float(c.sum().item()), 1.0))
                        st["seen"] = int(os.environ.get("EXL3_DRAFT_WARM", "2048") or 2048)     # warm start: the first step builds
                except Exception as e:
                    print(f"[draft-head] counts file ignored: {e}", flush = True)
        return st

    def _adapt_build(self, lm, st):
        """(Re)build the subset from the counts. Blocks holding ids above the real vocabulary are left out (a draft there
        is rejected for sure); the counts decay by half so a change of language takes over."""
        cap = int(os.environ.get("EXL3_DRAFT_VOCAB", "0") or 0) // 128
        cover = float(os.environ.get("EXL3_DRAFT_COVER", "0.99") or 0.99)
        c = st["counts"][:st["ok_blocks"]]
        tot = float(c.sum().item())
        order = torch.argsort(c, descending = True, stable = True)
        cum = torch.cumsum(c[order], 0) / max(tot, 1.0)
        k = int(torch.searchsorted(cum, torch.tensor(cover, device = cum.device)).item()) + 1
        k = max(int(os.environ.get("EXL3_DRAFT_MINROWS", "49152") or 0) // 128, k)       # a short calibration under-counts the tail
        k = max(2, min(k + (k & 1), cap, order.numel()))       # an even number of blocks: the 1-2 row int8 GEMV needs n % 256 == 0
        blocks = torch.sort(order[:k]).values
        st["sub"] = st["gmap"] = None
        sub = self._draft_subhead(lm, blocks)
        if sub is None:
            return False
        st["sub"] = sub
        st["member"] = torch.zeros(st["counts"].numel(), dtype = torch.bool, device = blocks.device)
        st["member"][blocks] = True
        st["gmap"] = (blocks[:, None] * 128 + torch.arange(128, device = blocks.device)).reshape(-1)
        st["built_at"] = st["seen"]
        st["nbuilt"] += 1
        path = os.environ.get("EXL3_DRAFT_COUNTS")
        if path:
            try:
                torch.save(st["counts"].cpu(), path + ".tmp"); os.replace(path + ".tmp", path)
            except OSError:
                pass
        st["counts"] *= 0.5
        st["ema"] = 0.0
        st["on"] = True
        if st["log"]:
            print(f"[draft-head] subset #{st['nbuilt']}: {k * 128} rows, covers {float(cum[k - 1]):.4f} of {int(tot)} counts "
                  f"(seen {st['seen']})", flush = True)
        return True

    def sample_from_state(
        self,
        state: torch.Tensor,
        params: dict
    ) -> torch.Tensor:
        if not self.attached_model().loaded_tp:
            ll = self.attached_model().logit_layer_idx
            lm = self.attached_model().modules[ll]
            logits = lm.prepare_for_device(state, params)
            # EXL3_DRAFT_VOCAB=N: draft steps read only part of the head. Every EXL3_DRAFT_PROBE-th step (16) runs both heads and compares the argmax
            # per row: a language whose tokens sit outside the subset switches it off until the probes agree again.
            # Adaptive (default): the subset is the most frequent 128-id blocks of the head's own argmax ids, built after
            # EXL3_DRAFT_WARM rows and rebuilt when the probes disagree. Fixed (EXL3_DRAFT_ADAPT=0): the first N rows, 1-row steps only.
            sub = None
            adapt = None
            mode = self.__dict__.get("_dmode")
            if mode is None:        # read once: this runs on every draft step
                mode = self._dmode = 0
                if int(os.environ.get("EXL3_DRAFT_VOCAB", "0") or 0) >= 128:
                    mode = self._dmode = (2 if os.environ.get("EXL3_DRAFT_ADAPT", "1") != "0" else 1) if self._head_plain(lm) else 0
            if mode and not params.get("export_draft_logits"):
                if mode == 2:
                    adapt = self._adapt_state(lm)
                    sub = adapt["sub"]
                elif state.shape[0] * state.shape[1] == 1:
                    sub = self._draft_subhead(lm)
            vs = adapt["vs"] if adapt is not None else self.attached_model().config.vocab_size
            if adapt is not None:
                rows = state.shape[0] * state.shape[1]
                adapt["calls"] += 1
                probe = adapt["calls"] % adapt["every"] == 0
                if sub is None or not adapt["on"] and not probe:
                    # cold (no subset yet) or off: the full head, counting its argmax
                    fl = lm.forward(logits, params)[..., :vs].float()
                    fconf, fids = torch.max(fl, dim = -1)
                    adapt["counts"].index_add_(0, (fids.reshape(-1) // 128), torch.ones(rows, device = fids.device))
                    adapt["seen"] += rows
                    if adapt["seen"] - adapt["built_at"] >= adapt["warm"]:
                        sub = None      # a rebuild must not hold the old subset while the new one is made (another 150 MiB at 98304 rows)
                        self._adapt_build(lm, adapt)
                    if params.get("export_draft_conf"):
                        params["draft_conf"] = fconf
                    params["draft_p"] = (1.0 / torch.exp(fl - fconf.unsqueeze(-1)).sum(-1)).reshape(-1)
                    return fids
                sl = sub.forward(logits, params).float()
                conf, li = torch.max(sl, dim = -1)
                ids = adapt["gmap"][li]
                pp = (1.0 / torch.exp(sl - conf.unsqueeze(-1)).sum(-1)).reshape(-1)    # max probability over the subset
                if probe:
                    fl = lm.forward(logits, params)[..., :vs].float()
                    fconf, fids = torch.max(fl, dim = -1)
                    # per row, not "any row". A row whose full argmax sits inside the subset but differs (fp16 tie broken the other way
                    # by the other GEMV shape) costs nothing: only an argmax outside the subset can never be drafted.
                    out = ~adapt["member"][fids // 128]
                    miss, num = (float(v) for v in torch.stack((out.float().mean(), (fids != ids).float().mean())).tolist())
                    adapt["numeric"] += num - miss
                    adapt["ema"] += (miss - adapt["ema"]) / 32       # a row that misses costs ~one accepted token in 1.5: the break-even rate is a few %
                    adapt["counts"].index_add_(0, (fids.reshape(-1) // 128), torch.ones(rows, device = fids.device))
                    adapt["seen"] += rows
                    adapt["probes"] += 1
                    adapt["misses"] += miss
                    if adapt["log"] and adapt["probes"] % 25 == 0:
                        print(f"[draft-head] {adapt['probes']} probes, row miss rate {adapt['misses'] / adapt['probes']:.4f} (+{adapt['numeric'] / adapt['probes']:.4f} fp16 ties), "
                              f"on={adapt['on']}, subset {adapt['sub'].out_features} rows", flush = True)
                    adapt["on"] = adapt["ema"] < adapt["maxmiss"]
                    conf, ids = fconf, fids
                    pp = (1.0 / torch.exp(fl - fconf.unsqueeze(-1)).sum(-1)).reshape(-1)
                params["draft_p"] = pp
                if params.get("export_draft_conf"):
                    params["draft_conf"] = conf
                return ids
            if sub is not None:
                st = self.__dict__.setdefault("_dsub_st", {"calls": 0, "ema": 0.0, "on": True})
                st["calls"] += 1
                probe = st["calls"] % 8 == 0
                if st["on"] or probe:
                    sl = sub.forward(logits, params)[..., :min(sub.out_features, vs)].float()
                    conf, ids = torch.max(sl, dim = -1)
                    pp = (1.0 / torch.exp(sl - conf.unsqueeze(-1)).sum(-1)).reshape(-1)    # max probability over the subset
                    if probe:
                        fl = lm.forward(logits, params)[..., :vs].float()
                        fconf, fids = torch.max(fl, dim = -1)
                        miss = float((fids != ids).any().item())
                        st["ema"] = 0.9 * st["ema"] + 0.1 * miss
                        st["on"] = st["ema"] < (0.12 if st["on"] else 0.04)     # hysteresis: ~1 % misses on English / code, ~20 % on French
                        conf, ids = fconf, fids
                        pp = (1.0 / torch.exp(fl - fconf.unsqueeze(-1)).sum(-1)).reshape(-1)
                    params["draft_p"] = pp
                    if params.get("export_draft_conf"):
                        params["draft_conf"] = conf
                    return ids
            logits = lm.forward(logits, params)
            lim = getattr(self, "vocab_limit", None)   # set by the Generator: the tokenizer's piece count when the head is padded
            if lim is not None:
                logits = logits[..., :lim]   # the padding columns have no piece: a draft there is rejected for sure
            if params.get("export_draft_conf"):
                # Per-position confidence for the generator's draft truncation: the argmax logit
                # value, over the unpadded vocabulary
                logits = logits[..., :self.attached_model().config.vocab_size]
                conf, ids = torch.max(logits, dim = -1)
                params["draft_conf"] = conf
                return ids
            return torch.argmax(logits, dim = -1)
        else:
            state = self.attached_model().tp_producer.send(state)
            argmax = self.attached_model().tp_dispatch_lm_head_argmax((state, {}))
            return argmax


class Qwen3_5MoeMTPModel(Qwen3_5MTPModel):

    def __init__(
        self,
        config: Qwen3_5Config | Qwen3_5MoeConfig,
        **kwargs
    ):
        super().__init__(config, use_moe = True, **kwargs)

