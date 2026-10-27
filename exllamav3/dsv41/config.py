"""DeepSeek-V4.1 configuration (HF `text_config` of model_type deepseek_v41)."""
from __future__ import annotations
import json
from dataclasses import dataclass, field, fields, asdict


@dataclass
class DSV41Config:
    # --- trunk
    vocab_size: int = 129280
    hidden_size: int = 5120
    moe_intermediate_size: int = 2304
    num_hidden_layers: int = 40
    num_attention_heads: int = 64
    head_dim: int = 512
    qk_rope_head_dim: int = 64
    q_lora_rank: int = 1280
    o_lora_rank: int = 1024
    o_groups: int = 8
    rms_norm_eps: float = 1e-20
    swiglu_limit: float = 10.0
    # --- rope (sliding layers: plain rope_theta, no yarn; compressed layers: compress_rope_theta + yarn)
    rope_theta: float = 10000.0
    compress_rope_theta: float = 160000.0
    rope_factor: float = 16.0
    beta_fast: int = 32
    beta_slow: int = 1
    original_seq_len: int = 65536
    # --- MoE
    n_routed_experts: int = 384
    n_shared_experts: int = 1
    num_experts_per_tok: int = 6
    scoring_func: str = "sqrtsoftplus"
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.5
    gate_temp: float = 1.0
    # --- attention
    sliding_window: int = 128
    compress_ratios: list = field(default_factory=lambda: [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0])
    kv_source_layer_ids: list = field(default_factory=lambda: [2, 8, 14, 20])
    index_source_layer_ids: list = field(default_factory=lambda: [2, 8, 14, 20, 24, 28, 32, 36])
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 512
    candidate_source_layer_id: int = 20      # < 0 disables the two-level selection
    candidate_topk_blocks: int = 2048
    candidate_block_size: int = 8
    # --- hyper-connections
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    # --- engram
    engram_layer_ids: list = field(default_factory=lambda: [1, 14])
    engram_num_embeddings: list = field(default_factory=lambda: [384006168, 384016682])
    engram_max_ngram_size: int = 4
    engram_vocab_size: int = 16000000
    engram_n_heads: int = 8
    engram_head_dim: int = 256
    engram_pad_token_id: int = 2
    engram_compressed_vocab_size: int = 99092
    # --- MTP / dspark
    num_nextn_predict_layers: int = 3
    dspark_block_size: int = 5
    dspark_noise_token_id: int = 128799
    dspark_target_layer_ids: list = field(default_factory=lambda: [37, 38, 39])
    dspark_markov_rank: int = 256
    dspark_n_routed_experts: int = 128
    dspark_num_experts_per_tok: int = 3
    # --- quant layout of the *official* checkpoint
    fp8_block_size: int = 32
    fp4_block_size: int = 32

    # ------------------------------------------------------------------ derived
    @property
    def n_mtp_layers(self) -> int:
        return self.num_nextn_predict_layers if self.dspark_block_size else 0

    def ratio(self, layer: int) -> int:
        return int(self.compress_ratios[layer])

    def is_kv_source(self, layer: int) -> bool:
        return layer < self.num_hidden_layers and layer in self.kv_source_layer_ids

    def is_index_source(self, layer: int) -> bool:
        return layer < self.num_hidden_layers and layer in self.index_source_layer_ids

    def kv_source_of(self, layer: int):
        """Layer whose compressed-KV pool `layer` reads (None for ratio-0 layers)."""
        if self.ratio(layer) == 0:
            return None
        c = [s for s in self.kv_source_layer_ids if s <= layer]
        return max(c) if c else None

    def index_source_of(self, layer: int):
        if self.ratio(layer) == 0:
            return None
        c = [s for s in self.index_source_layer_ids if s <= layer]
        return max(c) if c else None

    def moe_counts(self, layer: int):
        if layer < self.num_hidden_layers:
            return self.n_routed_experts, self.num_experts_per_tok
        return (self.dspark_n_routed_experts or self.n_routed_experts,
                self.dspark_num_experts_per_tok or self.num_experts_per_tok)

    def validate(self):
        n = self.num_hidden_layers
        assert len(self.compress_ratios) >= n
        for l in range(n):
            if self.ratio(l):
                ks, is_ = self.kv_source_of(l), self.index_source_of(l)
                assert ks is not None and self.ratio(ks) == self.ratio(l), (l, ks, "kv source ratio mismatch")
                assert is_ is not None
                assert is_ >= ks, (l, "index source must come after the kv source it indexes")
                owner = [s for s in self.kv_source_layer_ids if s <= is_]
                assert owner, "index source without an index-key owner"
        for s in self.kv_source_layer_ids:
            assert self.is_index_source(s), "every kv source owns an indexer (wk/k_norm)"
        assert self.head_dim > self.qk_rope_head_dim
        assert self.num_attention_heads * self.head_dim % self.o_groups == 0
        return self

    # ------------------------------------------------------------------ io
    @classmethod
    def from_hf_dict(cls, d: dict) -> "DSV41Config":
        t = d.get("text_config", d)
        kw = {}
        names = {f.name for f in fields(cls)}
        for k, v in t.items():
            if k in names:
                kw[k] = v
        rs = t.get("rope_scaling")
        if rs:
            kw["rope_factor"] = rs.get("factor", 16)
            kw["beta_fast"] = rs.get("beta_fast", 32)
            kw["beta_slow"] = rs.get("beta_slow", 1)
            kw["original_seq_len"] = rs.get("original_max_position_embeddings", 0)
        if "n_routed_experts" not in kw and "num_experts" in t:
            kw["n_routed_experts"] = t["num_experts"]
        return cls(**kw).validate()

    @classmethod
    def from_file(cls, path: str) -> "DSV41Config":
        with open(path) as f:
            return cls.from_hf_dict(json.load(f))

    def to_hf_text_dict(self) -> dict:
        d = asdict(self)
        d["rope_scaling"] = {"rope_type": "yarn", "factor": self.rope_factor, "beta_fast": self.beta_fast,
                             "beta_slow": self.beta_slow,
                             "original_max_position_embeddings": self.original_seq_len}
        return d


def tiny_config(**over) -> DSV41Config:
    """Same structure as V4.1-Flash, small dims. 8 trunk layers + 2 MTP stages.
    ratios 0,0,2,2,1,1,1,0 (+ mtp 0,0); kv sources (2,4) with readers 3 and 5,6; index sources (2,4,5), the
    candidate source is 4 and layer 5 consumes its blocks; engram on layers 1 and 5."""
    d = dict(
        vocab_size=512, hidden_size=64, moe_intermediate_size=64, num_hidden_layers=8,
        num_attention_heads=4, head_dim=64, qk_rope_head_dim=16, q_lora_rank=64, o_lora_rank=64, o_groups=2,
        rms_norm_eps=1e-20, swiglu_limit=10.0, rope_theta=10000.0, compress_rope_theta=40000.0,
        rope_factor=16.0, beta_fast=32, beta_slow=1, original_seq_len=64,
        n_routed_experts=8, n_shared_experts=1, num_experts_per_tok=2, routed_scaling_factor=1.5,
        sliding_window=8, compress_ratios=[0, 0, 2, 2, 1, 1, 1, 0, 0, 0],
        kv_source_layer_ids=[2, 4], index_source_layer_ids=[2, 4, 5],
        index_n_heads=4, index_head_dim=32, index_topk=4,
        candidate_source_layer_id=4, candidate_topk_blocks=3, candidate_block_size=2,
        hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6,
        engram_layer_ids=[1, 5], engram_num_embeddings=[0, 0], engram_max_ngram_size=4,
        engram_vocab_size=500, engram_n_heads=2, engram_head_dim=32, engram_pad_token_id=2,
        engram_compressed_vocab_size=0,   # filled in by the tokenizer-derived map
        num_nextn_predict_layers=2, dspark_block_size=3, dspark_noise_token_id=500,
        dspark_target_layer_ids=[5, 6, 7], dspark_markov_rank=32, dspark_n_routed_experts=4,
        dspark_num_experts_per_tok=2,
    )
    d.update(over)
    return DSV41Config(**d)
