"""CPU-only tests for the Nemotron-H Puzzle config adapter.

Loads ``exllamav3/architecture/nemotronh_puzzle.py`` directly by file path so
the test runs without torch (the adapter module is stdlib-only by design).
Validates against the real Bolt checkpoint config when available
(``$BOLT_CONFIG_JSON``) plus a hermetic synthetic fixture that always runs.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import pytest

ADAPTER_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "exllamav3", "architecture", "nemotronh_puzzle.py"))
REAL_CONFIG = os.environ.get(
    "BOLT_CONFIG_JSON", "")


def _load_adapter():
    name = "nemotronh_puzzle_adapter_under_test"
    spec = importlib.util.spec_from_file_location(name, ADAPTER_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses need the module registered pre-exec
    spec.loader.exec_module(mod)
    return mod


ap = _load_adapter()


def _needs_real_config():
    return pytest.mark.skipif(
        not os.path.isfile(REAL_CONFIG),
        reason=f"Bolt config not found: {REAL_CONFIG}")


def _real_llm():
    with open(REAL_CONFIG, encoding="utf8") as f:
        return json.load(f)["llm_config"]


# --------------------------------------------------------------------------
# Real Bolt config
# --------------------------------------------------------------------------

@_needs_real_config()
def test_bolt_block_layout():
    spec = ap.load_puzzle_spec(REAL_CONFIG)
    assert spec.from_omni_wrapper is True
    assert spec.num_hidden_layers == 88
    assert len(spec.blocks) == 88
    assert spec.block_counts == {"mamba": 40, "moe": 40, "attention": 8}
    assert spec.attention_indices == (7, 16, 25, 36, 47, 58, 69, 78)


@_needs_real_config()
def test_bolt_per_layer_experts_match_block_configs():
    llm = _real_llm()
    spec = ap.load_puzzle_spec(REAL_CONFIG)
    assert len(llm["block_configs"]) == len(llm["layers_block_type"]) == 88
    for i, (node, bcfg, btype) in enumerate(
            zip(spec.blocks, llm["block_configs"], llm["layers_block_type"])):
        assert node.index == i
        assert node.block_type == btype == bcfg["block_type"]
        if btype == "moe":
            assert node.n_routed_experts == bcfg["n_routed_experts"] > 0
            assert node.moe_intermediate_size == bcfg["moe_intermediate_size"] > 0
            assert node.num_experts_per_tok == bcfg["num_experts_per_tok"] == 9
        else:
            assert node.n_routed_experts == 0
            assert node.moe_intermediate_size == 0
    # Heterogeneous Puzzle MoE: 12 distinct (n_experts, intermediate) specs.
    assert len(spec.distinct_moe_specs()) == 12
    assert spec.block_at(1).n_routed_experts == 240
    assert spec.block_at(1).moe_intermediate_size == 2048


@_needs_real_config()
def test_bolt_globals():
    spec = ap.load_puzzle_spec(REAL_CONFIG)
    assert spec.hidden_size == 4096
    assert spec.vocab_size == 131072
    assert spec.moe_latent_size == 1024
    assert spec.moe_shared_expert_intermediate_size == 5376
    assert spec.num_attention_heads == 32
    assert spec.num_key_value_heads == 2
    assert spec.head_dim == 128
    assert spec.mamba_num_heads == 128
    assert spec.mamba_head_dim == 64
    assert spec.ssm_state_size == 128


@_needs_real_config()
def test_bolt_graph_tensor_keys():
    spec = ap.load_puzzle_spec(REAL_CONFIG)
    graph = ap.build_puzzle_graph(spec)
    assert graph.num_layers == 88
    assert graph.find("backbone.layers.7").block_type == "attention"

    mamba_keys = set(ap.iter_block_tensor_keys(graph.find("backbone.layers.0")))
    assert mamba_keys == {
        "backbone.layers.0.norm.weight",
        "backbone.layers.0.mixer.in_proj.weight",
        "backbone.layers.0.mixer.out_proj.weight",
        "backbone.layers.0.mixer.conv1d.weight",
        "backbone.layers.0.mixer.conv1d.bias",
        "backbone.layers.0.mixer.A_log",
        "backbone.layers.0.mixer.D",
        "backbone.layers.0.mixer.dt_bias",
        "backbone.layers.0.mixer.norm.weight",
    }
    attn_keys = set(ap.iter_block_tensor_keys(graph.find("backbone.layers.7")))
    assert attn_keys == {
        "backbone.layers.7.norm.weight",
        "backbone.layers.7.mixer.q_proj.weight",
        "backbone.layers.7.mixer.k_proj.weight",
        "backbone.layers.7.mixer.v_proj.weight",
        "backbone.layers.7.mixer.o_proj.weight",
    }
    moe_keys = set(ap.iter_block_tensor_keys(graph.find("backbone.layers.1")))
    assert len(moe_keys) == 1 + 6 + 2 * 240  # norm + shared/router + experts
    assert "backbone.layers.1.mixer.gate.weight" in moe_keys
    assert "backbone.layers.1.mixer.gate.e_score_correction_bias" in moe_keys
    assert "backbone.layers.1.mixer.fc1_latent_proj.weight" in moe_keys
    assert "backbone.layers.1.mixer.shared_experts.down_proj.weight" in moe_keys
    assert "backbone.layers.1.mixer.experts.0.up_proj.weight" in moe_keys
    assert "backbone.layers.1.mixer.experts.239.down_proj.weight" in moe_keys


def test_key_mapping():
    assert ap.strip_llm_prefix("language_model.backbone.layers.0.mixer.D") == \
        "backbone.layers.0.mixer.D"
    assert ap.strip_llm_prefix("language_model.lm_head.weight") == "lm_head.weight"
    assert ap.map_checkpoint_key("language_model.lm_head.weight") == "lm_head.weight"
    assert ap.map_checkpoint_key("vision_model.radio_model.blocks.0.attn.qkv") is None
    assert ap.map_checkpoint_key("mlp1.1.weight") is None
    assert ap.map_checkpoint_key("backbone.layers.0.mixer.D") == \
        "backbone.layers.0.mixer.D"  # bare puzzle checkpoint
    with pytest.raises(ValueError):
        ap.strip_llm_prefix("vision_model.x")


def test_strip_prefix_from_maps():
    file_map = {
        "language_model.lm_head.weight": "shard1",
        "language_model.backbone.layers.0.mixer.D": "shard2",
        "vision_model.blocks.0.weight": "shard9",
        "mlp1.1.weight": "shard9",
    }
    headers = {
        "shard1": {"language_model.lm_head.weight": {"shape": [1]},
                    "_header_offset": 100},
        "shard2": {"language_model.backbone.layers.0.mixer.D": {"shape": [2]},
                    "__metadata__": {}},
        "shard9": {"vision_model.blocks.0.weight": {"shape": [3]},
                    "mlp1.1.weight": {"shape": [4]},
                    "_header_offset": 200},
    }
    new_map, new_headers = ap.strip_llm_prefix_from_maps(file_map, headers)
    assert new_map == {
        "lm_head.weight": "shard1",
        "backbone.layers.0.mixer.D": "shard2",
    }
    assert new_headers["shard1"] == {
        "lm_head.weight": {"shape": [1]}, "_header_offset": 100}
    assert new_headers["shard2"] == {
        "backbone.layers.0.mixer.D": {"shape": [2]}}
    assert new_headers["shard9"] == {"_header_offset": 200}


# --------------------------------------------------------------------------
# Hermetic synthetic fixture (always runs)
# --------------------------------------------------------------------------

def _synthetic_llm(n_layers=6):
    pattern = ["mamba", "moe", "mamba", "attention", "moe", "mamba"]
    block_configs = []
    for t in pattern[:n_layers]:
        if t == "moe":
            block_configs.append({"block_type": t, "moe_intermediate_size": 1536,
                                  "n_routed_experts": 240, "num_experts_per_tok": 9})
        else:
            block_configs.append({"block_type": t})
    return {
        "model_type": "nemotron_h_puzzle",
        "architectures": ["NemotronHPuzzleForCausalLM"],
        "hidden_size": 4096, "vocab_size": 131072,
        "num_attention_heads": 32, "num_key_value_heads": 2,
        "layers_block_type": pattern[:n_layers],
        "block_configs": block_configs,
        "moe_latent_size": 1024, "moe_shared_expert_intermediate_size": 5376,
    }


def test_synthetic_bare_and_wrapped(tmp_path):
    llm = _synthetic_llm()
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps(llm), encoding="utf8")
    spec = ap.load_puzzle_spec(str(bare))
    assert spec.from_omni_wrapper is False
    assert spec.num_hidden_layers == 6
    assert spec.block_counts == {"mamba": 3, "moe": 2, "attention": 1}
    assert spec.attention_indices == (3,)
    assert spec.block_at(1).n_routed_experts == 240

    wrapped = {"architectures": ["NemotronH_Omni_Reasoning_V3"],
               "model_type": "nemotron_h_omni", "llm_config": llm}
    wrap = tmp_path / "wrap.json"
    wrap.write_text(json.dumps(wrapped), encoding="utf8")
    wspec = ap.load_puzzle_spec(str(wrap))
    assert wspec.from_omni_wrapper is True
    assert wspec.block_counts == spec.block_counts
    graph = ap.build_puzzle_graph(wspec)
    assert graph.num_layers == 6
    assert "puzzle" in ap.summarize(wspec).lower() or "Puzzle" in ap.summarize(wspec)


def test_rejects_bad_configs():
    good = _synthetic_llm()
    bad_len = dict(good, block_configs=good["block_configs"][:-1])
    with pytest.raises(ValueError):
        ap.parse_puzzle_spec(bad_len)
    bad_type = json.loads(json.dumps(good))
    bad_type["block_configs"][0] = {"block_type": "dense"}
    with pytest.raises(ValueError):
        ap.parse_puzzle_spec(bad_type)
    bad_model = dict(good, model_type="llama")
    with pytest.raises(ValueError):
        ap.parse_puzzle_spec(bad_model)
    with pytest.raises(ValueError):
        ap.load_llm_config_dict({"architectures": ["Nope"], "model_type": "nope"})
