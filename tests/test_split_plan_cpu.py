"""CPU contract tests for the universal split core.

The placement rules, endcap allowlist, auto-N arithmetic and the layout-neutral
front reserve are pure Python/torch-device logic: they run without a GPU or the
compiled extension.  The integration gates (loads, determinism, needle) stay in
the strix_halo harnesses.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope = "module")
def split_modules():
    added = []
    if "exllamav3" not in sys.modules:
        pkg = types.ModuleType("exllamav3")
        pkg.__path__ = [str(ROOT / "exllamav3")]
        sys.modules["exllamav3"] = pkg
        added.append("exllamav3")
    if "exllamav3.util" not in sys.modules:
        util = types.ModuleType("exllamav3.util")
        util.__path__ = [str(ROOT / "exllamav3" / "util")]
        sys.modules["exllamav3.util"] = util
        added.append("exllamav3.util")
    if "exllamav3.util.arch_list" not in sys.modules:
        _load_module("exllamav3.util.arch_list", ROOT / "exllamav3" / "util" / "arch_list.py")
        added.append("exllamav3.util.arch_list")
    dm = _load_module("exllamav3.device_map", ROOT / "exllamav3" / "device_map.py")
    bridge = _load_module("exllamav3.bridge", ROOT / "exllamav3" / "bridge.py")
    split = _load_module("exllamav3.serve_split", ROOT / "exllamav3" / "serve_split.py")
    yield dm, bridge, split
    for name in ("exllamav3.serve_split", "exllamav3.bridge", "exllamav3.device_map"):
        sys.modules.pop(name, None)
    for name in added:
        sys.modules.pop(name, None)


class FakeModule:
    def __init__(self, key, layer_idx = None, caps = None, numel = 0):
        self.key = key
        self.layer_idx = layer_idx
        self.caps = caps or {}
        self._numel = numel

    def weights_numel(self):
        return self._numel


class FakeModel:
    def __init__(self, modules):
        self.modules = modules


class FakeSTC:
    def __init__(self, sizes):
        self.sizes = sizes

    def get_tensor_sizes(self, key):
        return [self.sizes.get(key, 0)]


class FakeConfig:
    def __init__(self, sizes):
        self.stc = FakeSTC(sizes)


class FakeCacheLayer:
    def __init__(self, size):
        self.size = size
        self.device = None
        self.attention = SimpleNamespace(device = None)

    def storage_size(self):
        return self.size


class FakeRecurrent:
    def __init__(self, size, device):
        self.size = size
        self.device = device

    def storage_size(self):
        return self.size


class FakeCache:
    def __init__(self, per_layer = 0):
        self.layers = {}
        self.recurrent_layers = {}
        if per_layer:
            for i in range(16):
                self.layers[(i, 0)] = FakeCacheLayer(per_layer)


class FakeProps:
    def __init__(self, total):
        self.total_memory = total


def _devices(dm):
    return dm.Devices(
        main = torch.device("cuda", 0),
        front = torch.device("cuda", 1),
        main_arch = "gfx1151",
        front_arch = "gfx1201",
    )


def _infos(dm, *pairs):
    """Synthetic device inventory: (arch, optional name/uuid) per device index."""
    out = []
    for i, item in enumerate(pairs):
        if isinstance(item, str):
            arch, name, uuid = item, f"device-{i}", ""
        else:
            arch = item[0]
            name = item[1] if len(item) > 1 else f"device-{i}"
            uuid = item[2] if len(item) > 2 else ""
        out.append(dm.DeviceInfo(index = i, name = name, arch = arch, uuid = uuid))
    return out


@pytest.fixture
def clean_split_env(monkeypatch):
    for key in ("KYOJIN_FRONT_DEVICE", "KYOJIN_MAIN_DEVICE", "KYOJIN_FRONT_LAYERS", "KYOJIN_ENDCAPS"):
        monkeypatch.delenv(key, raising = False)


def _glm_model():
    embedding = FakeModule("model.language_model.embed_tokens", caps = {"prefer_cpu": True})
    expand = FakeModule("hc_expand")
    layers = [FakeModule(f"model.language_model.layers.{i}", layer_idx = i) for i in range(4)]
    norm = FakeModule("model.language_model.norm")
    head = FakeModule("lm_head", caps = {"logits_output": True})
    ngram = FakeModule("ngram", caps = {})
    return FakeModel([embedding, expand] + layers + [norm, head, ngram])


def test_component_policy_defaults(split_modules):
    dm, _bridge, _split = split_modules
    text = dm.component_policy("text")
    mtp = dm.component_policy("mtp")
    aux = dm.component_policy("ngram")
    assert text.kind == "text" and text.front_eligible and text.endcaps_front
    assert "RMSNorm" in text.endcap_types and "lm_head" in text.endcap_key_suffixes
    assert mtp.kind == "mtp" and mtp.whole_component and mtp.front_eligible
    assert aux.kind == "aux" and not aux.front_eligible


def test_text_placement_uses_endcap_allowlist_and_safe_unknown_default(split_modules):
    dm, _bridge, _split = split_modules
    model = _glm_model()
    cmap = dm.ComponentMap.from_model(model, 2, True, _devices(dm), component = "text")
    devices = {m.key: cmap.device_for(m) for m in model.modules}

    assert devices["model.language_model.embed_tokens"] == torch.device("cpu")
    assert devices["hc_expand"] == torch.device("cuda", 1)
    assert devices["model.language_model.norm"] == torch.device("cuda", 1)
    assert devices["lm_head"] == torch.device("cuda", 1)
    assert devices["ngram"] == torch.device("cuda", 0)          # unknown non-layer: main, warning default
    for i in range(4):
        assert devices[f"model.language_model.layers.{i}"] == torch.device("cuda", 1 if i < 2 else 0)


def test_endcaps_off_follows_neighbours_only_for_allowlisted_modules(split_modules):
    dm, _bridge, _split = split_modules
    model = _glm_model()
    cmap = dm.ComponentMap.from_model(model, 2, False, _devices(dm), component = "text")
    devices = {m.key: cmap.device_for(m) for m in model.modules}

    assert devices["hc_expand"] == torch.device("cuda", 1)      # prologue follows layer 0
    assert devices["model.language_model.norm"] == torch.device("cuda", 0)   # epilogue: last layer is main
    assert devices["lm_head"] == torch.device("cuda", 0)
    assert devices["ngram"] == torch.device("cuda", 0)

    all_front = dm.ComponentMap.from_model(model, 4, False, _devices(dm), component = "text")
    devices = {m.key: all_front.device_for(m) for m in model.modules}
    assert devices["model.language_model.norm"] == torch.device("cuda", 1)
    assert devices["lm_head"] == torch.device("cuda", 1)


def test_layer_device_map_alias_keeps_harness_api(split_modules):
    dm, _bridge, _split = split_modules
    model = _glm_model()
    cmap = dm.LayerDeviceMap.from_model(model, 2, True, _devices(dm), component = "text")
    assert isinstance(cmap, dm.ComponentMap)
    assert cmap.front_layers == 2 and cmap.enabled


def test_whole_component_places_every_module_front_only_with_endcaps(split_modules):
    dm, _bridge, _split = split_modules
    mtp = FakeModel([
        FakeModule("mtp.pre", caps = {}),
        FakeModule("mtp.block.0", layer_idx = 0),
        FakeModule("mtp.block.1", layer_idx = 1),
        FakeModule("mtp.head", caps = {}),
    ])
    on = dm.ComponentMap.from_model(mtp, 0, True, _devices(dm), component = "mtp")
    off = dm.ComponentMap.from_model(mtp, 0, False, _devices(dm), component = "mtp")
    assert all(on.device_for(m) == torch.device("cuda", 1) for m in mtp.modules)
    assert all(off.device_for(m) == torch.device("cuda", 0) for m in mtp.modules)


def test_module_bytes_uses_compile_or_alt_key_when_prefix_is_empty(split_modules):
    dm, _bridge, _split = split_modules
    stc = FakeSTC({"embed_tokens": 456})
    head = FakeModule("hc_head")
    head.get_compile_sizes = lambda s: [123]
    assert dm._module_bytes(head, stc) == 123
    tied = FakeModule("lm_head")
    tied.alt_key = "embed_tokens"
    assert dm._module_bytes(tied, stc) == 456


def test_policy_can_disable_endcap_front(split_modules):
    dm, _bridge, _split = split_modules
    model = _glm_model()
    policy = dm.component_policy("text", endcaps_front = False)
    cmap = dm.ComponentMap.from_model(model, 2, True, _devices(dm), component = "text", policy = policy)
    devices = {m.key: cmap.device_for(m) for m in model.modules}
    assert devices["hc_expand"] == torch.device("cuda", 0)
    assert devices["model.language_model.norm"] == torch.device("cuda", 0)
    assert devices["model.language_model.layers.0"] == torch.device("cuda", 1)


def test_recurrent_placement_audit_reads_state_device(split_modules):
    dm, _bridge, _split = split_modules
    model = FakeModel([FakeModule("layer.0", layer_idx = 0)])
    cache = FakeCache()
    cache.recurrent_layers[(0, 0)] = FakeRecurrent(1024, torch.device("cuda", 0))
    plan = dm.SplitPlan(_devices(dm), endcaps = True, front_layers = 1,
                        logger = lambda *_a: None)
    plan.add("text", model, cache = cache)
    plan.build()
    assert "text" in plan.audit_caches()

    cache.recurrent_layers[(0, 0)] = FakeRecurrent(1024, torch.device("cuda", 1))
    assert plan.audit_caches() == {}


def test_auto_n_is_capped_for_long_context(split_modules, monkeypatch):
    dm, _bridge, _split = split_modules
    gib = dm.GIB
    model = FakeModel([FakeModule(f"layer.{i}", layer_idx = i) for i in range(4)])
    cfg = FakeConfig({f"layer.{i}": gib for i in range(4)})
    monkeypatch.setattr(dm.torch.cuda, "get_device_properties", lambda dev: FakeProps(16 * gib))
    plan = dm.SplitPlan(_devices(dm), endcaps = True, front_layers = "auto",
                        front_reserve_gb = 1.0, auto_cap = 2, logger = lambda *_a: None)
    plan.add("text", model, config = cfg)
    plan.build()
    assert plan.front_layers == 2


def test_auto_n_uses_actual_bytes_cache_and_reserve(split_modules, monkeypatch):
    dm, _bridge, _split = split_modules
    gib = dm.GIB
    sizes = {f"layer.{i}": gib for i in range(4)}
    sizes["final_norm"] = gib // 10
    model = FakeModel([
        FakeModule(f"layer.{i}", layer_idx = i) for i in range(4)
    ] + [FakeModule("final_norm")])
    cfg = FakeConfig(sizes)
    cache = FakeCache(per_layer = gib // 5)                    # 0.2 GiB per layer, only 16 keys exist
    monkeypatch.setattr(dm.torch.cuda, "get_device_properties", lambda dev: FakeProps(6 * gib))

    plan = dm.SplitPlan(_devices(dm), endcaps = True, front_layers = "auto",
                        front_reserve_gb = 1.0, logger = lambda *_a: None)
    plan.add("text", model, config = cfg, cache = cache)
    plan.build()

    # budget 6 - 1 reserve - 0.25 safety = 4.75 GiB; each layer is 1.0 weights +
    # 0.2 cache, endcap 0.1. N=3 -> 3.7, N=4 -> 4.9.
    assert plan.front_layers == 3
    assert plan.map("text").front_layers == 3

    plan_off = dm.SplitPlan(_devices(dm), endcaps = False, front_layers = "auto",
                            front_reserve_gb = 1.0, logger = lambda *_a: None)
    plan_off.add("text", model, config = cfg, cache = cache)
    plan_off.build()
    assert plan_off.front_layers == 3


def test_front_reserve_is_a_slab_not_an_allocator_cap(split_modules, monkeypatch):
    dm, _bridge, _split = split_modules
    model = _glm_model()
    cmap = dm.ComponentMap.from_model(model, 2, True, _devices(dm), component = "text",
                                      front_reserve_gb = 0.5)
    calls = []

    def fake_empty(size, dtype = None, device = None):
        calls.append((size, dtype, device))
        return object()

    monkeypatch.setattr(dm.torch, "empty", fake_empty)
    monkeypatch.setattr(dm.torch.cuda, "set_per_process_memory_fraction",
                        lambda *_a, **_k: pytest.fail("the split must not cap the allocator"))
    cmap.engage_budget()
    assert len(calls) == 1
    assert calls[0][0] == int(0.5 * dm.GIB)
    assert calls[0][2] == torch.device("cuda", 1)
    cmap.release_front_budget()
    assert cmap._reserve_slabs == {}


def test_bridge_installs_per_config_and_is_reused(split_modules, monkeypatch):
    dm, bridge_mod, _split = split_modules
    monkeypatch.setenv("KYOJIN_BRIDGE", "1")
    monkeypatch.setenv("KYOJIN_SPLIT_ACTIVE", "0")
    main, front = torch.device("cuda", 0), torch.device("cuda", 1)
    cfg = SimpleNamespace()
    draft_cfg = SimpleNamespace()

    b1 = bridge_mod.install_bridge(cfg, main, front)
    b2 = bridge_mod.install_bridge(cfg, main, front)
    b3 = bridge_mod.install_bridge(draft_cfg, main, front)

    assert cfg.split_active is True and draft_cfg.split_active is True
    assert b1 is b2
    assert b1 is not b3
    assert b1.pair == (main, front)
    assert os.environ["KYOJIN_SPLIT_ACTIVE"] == "1"

    single = SimpleNamespace()
    assert bridge_mod.install_bridge(single, main, None) is None
    assert single.split_active is False


def test_named_preset_is_applied_by_the_plan(split_modules, monkeypatch):
    dm, _bridge, split = split_modules
    monkeypatch.setenv("EXL3_TEST_PRESET", "0")
    old = split.SPLIT_PRESETS.get("glm", {})
    split.SPLIT_PRESETS["glm"] = {"EXL3_TEST_PRESET": "1"}
    try:
        plan = dm.SplitPlan(_devices(dm), endcaps = True, front_layers = 1,
                            preset = "glm", logger = lambda *_a: None)
        plan.add("text", _glm_model())
        plan.build()
        assert os.environ["EXL3_TEST_PRESET"] == "1"
    finally:
        split.SPLIT_PRESETS["glm"] = old


@pytest.mark.parametrize("archs", [
    ("gfx1100", "gfx1101"),                 # RDNA3
    ("gfx1150", "gfx1151"),                 # RDNA3.5
    ("gfx1200", "gfx1201"),                 # RDNA4
    ("gfx1151", "gfx1100"),                 # mixed, reversed
    ("gfx1201", "gfx1201"),                 # same arch
    ("gfx1100", "gfx1100"),
    ("", "sm_90"),                          # unknown / CUDA-style names: selection only
])
def test_auto_selection_is_positional_and_arch_independent(split_modules, clean_split_env, archs):
    dm, _bridge, _split = split_modules
    devices = dm.resolve_devices(devices = _infos(dm, *archs))
    assert devices.main == torch.device("cuda", 0)
    assert devices.front == torch.device("cuda", 1)
    assert devices.main_arch == archs[0] and devices.front_arch == archs[1]
    assert [d.index for d in devices.inventory] == [0, 1]


def test_explicit_selectors_support_both_positions(split_modules, clean_split_env):
    dm, _bridge, _split = split_modules
    infos = _infos(dm, "gfx1151", "gfx1201")

    mixed = dm.resolve_devices("cuda:1", "cuda:0", devices = infos)
    assert (mixed.main_arch, mixed.front_arch) == ("gfx1151", "gfx1201")

    reversed_pair = dm.resolve_devices("gfx1151", "gfx1201", devices = infos)
    assert reversed_pair.main == torch.device("cuda", 1)
    assert reversed_pair.front == torch.device("cuda", 0)
    assert (reversed_pair.main_arch, reversed_pair.front_arch) == ("gfx1201", "gfx1151")

    by_name_uuid = dm.resolve_devices("device-1", "device-0", devices = infos)
    assert by_name_uuid.main == torch.device("cuda", 0)
    assert by_name_uuid.front == torch.device("cuda", 1)


def test_explicit_front_with_auto_main_uses_the_other_device(split_modules, clean_split_env):
    dm, _bridge, _split = split_modules
    infos = _infos(dm, "gfx1151", "gfx1201")

    by_index = dm.resolve_devices("cuda:0", devices = infos)
    assert by_index.main == torch.device("cuda", 1)
    assert by_index.front == torch.device("cuda", 0)

    by_arch = dm.resolve_devices("gfx1151", devices = infos)
    assert by_arch.main == torch.device("cuda", 1)
    assert by_arch.front == torch.device("cuda", 0)

    many = _infos(dm, "gfx1100", "gfx1151", "gfx1201")
    with pytest.raises(RuntimeError, match = "set KYOJIN_MAIN_DEVICE"):
        dm.resolve_devices("cuda:0", devices = many)


def test_uuid_and_name_substring_selectors(split_modules, clean_split_env):
    dm, _bridge, _split = split_modules
    infos = _infos(
        dm,
        ("gfx1201", "AMD Radeon RX 9070 XT", "aa-bb-cc"),
        ("gfx1151", "AMD Radeon 8060S", "dd-ee-ff"),
    )
    dev = dm.resolve_devices("aa-bb-cc", "8060s", devices = infos)
    assert dev.main == torch.device("cuda", 1)
    assert dev.front == torch.device("cuda", 0)


def test_main_device_env_is_honoured(split_modules, clean_split_env, monkeypatch):
    dm, _bridge, _split = split_modules
    monkeypatch.setenv("KYOJIN_MAIN_DEVICE", "cuda:1")
    dev = dm.resolve_devices(devices = _infos(dm, "gfx1151", "gfx1201"))
    assert dev.main == torch.device("cuda", 1)
    assert dev.front == torch.device("cuda", 0)


def test_one_device_fails_clearly(split_modules, clean_split_env):
    dm, _bridge, _split = split_modules
    with pytest.raises(RuntimeError, match = "two devices"):
        dm.resolve_devices(devices = _infos(dm, "gfx1151"))


def test_three_devices_require_an_explicit_front(split_modules, clean_split_env):
    dm, _bridge, _split = split_modules
    infos = _infos(dm, "gfx1100", "gfx1151", "gfx1201")
    with pytest.raises(RuntimeError, match = "3 devices.*--front-device"):
        dm.resolve_devices(devices = infos)
    dev = dm.resolve_devices("cuda:2", "cuda:1", devices = infos)
    assert dev.main == torch.device("cuda", 1)
    assert dev.front == torch.device("cuda", 2)


def test_same_device_and_bad_selectors_fail_clearly(split_modules, clean_split_env):
    dm, _bridge, _split = split_modules
    infos = _infos(dm, "gfx1151", "gfx1201")
    with pytest.raises(RuntimeError, match = "same device"):
        dm.resolve_devices("cuda:0", "cuda:0", devices = infos)
    with pytest.raises(RuntimeError, match = "no such device"):
        dm.resolve_devices("cuda:9", devices = infos)
    with pytest.raises(RuntimeError, match = "did not match any device"):
        dm.resolve_devices("gfx9999", devices = infos)


def test_settings_parse_main_device_and_keep_split_off_off(split_modules, clean_split_env, monkeypatch):
    dm, _bridge, split = split_modules
    monkeypatch.setenv("KYOJIN_MAIN_DEVICE", "cuda:1")
    args = SimpleNamespace(front_layers = None, front_device = None, main_device = None,
                           endcaps = None, split_preset = None, split_graphs = None,
                           front_reserve_gb = None)
    settings = split.settings_from_namespace(args)
    assert settings.main_device == "cuda:1"
    assert settings.enabled is False

    args.main_device = "gfx1100"
    args.front_device = "gfx1151"
    args.front_layers = "2"
    settings = split.settings_from_namespace(args)
    assert (settings.main_device, settings.front_device) == ("gfx1100", "gfx1151")
    assert settings.enabled is True


def test_split_off_does_not_touch_device_selection(split_modules, monkeypatch):
    dm, _bridge, split = split_modules

    def boom(*_a, **_k):
        pytest.fail("split off must not enumerate or resolve devices")

    monkeypatch.setattr(dm, "resolve_devices", boom)
    controller = split.SplitController(split.SplitSettings(front_layers = 0,
                                                           main_device = "cuda:9",
                                                           front_device = "cuda:9"),
                                       logger = lambda *_a: None)
    assert controller.devices is None
    assert controller.plan is None


def test_split_core_and_help_name_no_hardware(split_modules):
    for rel in ("exllamav3/device_map.py", "exllamav3/serve_split.py"):
        text = (ROOT / rel).read_text()
        for name in ("gfx1151", "gfx1201", "gfx1100", "9070", "8060S", "Strix", "eGPU", "iGPU", "USB4"):
            assert name not in text, f"{rel} still names {name}"


def _load_split_eval():
    path = ROOT / "tools" / "strix_halo" / "split_eval.py"
    spec = importlib.util.spec_from_file_location("split_eval_harness", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_harness_graph_env_matches_shared_list(split_modules):
    _dm, _bridge, split = split_modules
    mod = _load_split_eval()
    assert set(mod.GRAPH_ENV_OFF) == set(split.GRAPH_ENV_OFF)


def test_harness_default_front_layers_depends_on_context(monkeypatch):
    mod = _load_split_eval()
    for ctx, expected in ((131072, 5), (65536, 6)):
        monkeypatch.setattr(sys, "argv", ["split_eval.py", "--max-ctx", str(ctx)])
        assert mod.parse_args().front_layers == expected


def test_harness_front_layers_are_explicit_flags(monkeypatch):
    mod = _load_split_eval()
    monkeypatch.setattr(sys, "argv", ["split_eval.py", "--max-ctx", "131072", "--front-layers", "7"])
    assert mod.parse_args().front_layers == 7
    monkeypatch.setattr(sys, "argv", ["split_eval.py", "--max-ctx", "131072",
                                      "--long-ctx-front-layers", "3"])
    args = mod.parse_args()
    assert args.front_layers == 3 and args.front_layers_defaulted is True


def test_adapter_unsafe_env_is_applied_and_logged(split_modules, monkeypatch):
    dm, _bridge, _split = split_modules
    monkeypatch.setenv("EXL3_TEST_UNSAFE", "0")
    policy = dm.component_policy("text", unsafe_env = (("EXL3_TEST_UNSAFE", "1"),))
    plan = dm.SplitPlan(_devices(dm), endcaps = True, front_layers = 1, logger = lambda *_a: None)
    plan.add("text", _glm_model(), policy = policy)
    plan.build()
    assert os.environ["EXL3_TEST_UNSAFE"] == "1"


def test_split_settings_off_is_a_noop(split_modules, monkeypatch):
    _dm, _bridge, split = split_modules
    monkeypatch.setenv("EXL3_BLOCK_GRAPH", "1")
    for _k in ("EXL3_PF_HC_FUSE", "EXL3_PF_GR_FUSE", "EXL3_GR_PF_FUSE"):
        monkeypatch.setenv(_k, "1")
    off = split.SplitSettings(front_layers = 0)
    assert off.enabled is False
    assert split.apply_split_env(off) == []
    assert os.environ["EXL3_BLOCK_GRAPH"] == "1"

    on = split.SplitSettings(front_layers = "auto")
    assert on.enabled is True and on.auto is True
    split.apply_split_env(on)
    assert os.environ["EXL3_BLOCK_GRAPH"] == "0"
    # Deferred residual fusions cannot cross the host-staged bridge.
    assert os.environ["EXL3_PF_HC_FUSE"] == "0"
    assert os.environ["EXL3_PF_GR_FUSE"] == "0"
    assert os.environ["EXL3_GR_PF_FUSE"] == "0"
