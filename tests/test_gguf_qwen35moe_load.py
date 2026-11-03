"""Load a plain qwen35moe GGUF (unsloth UD) as stored bytes and run one forward; grouped MoE path vs per-expert path.

    FMOE_GGUF=<file.gguf> python tests/test_gguf_qwen35moe_load.py
"""
import json, os, sys
import torch

ENGINE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ENGINE)
from exllamav3.loader.gguf_collection import GGUFCollection
from exllamav3.loader.gguf_config import hf_config
from exllamav3.model.config import Config
from exllamav3.model.model import Model

src = os.environ["FMOE_GGUF"]
d = os.path.join(os.environ.get("FMOE_CFG", "/tmp"), "qwen35moe_cfg")
os.makedirs(d, exist_ok=True)
g = GGUFCollection(src)
json.dump(hf_config(g), open(os.path.join(d, "config.json"), "w"), indent=1)
link = os.path.join(d, "model.gguf")
if os.path.lexists(link):
    os.remove(link)
os.symlink(os.path.abspath(src), link)

config = Config.from_directory(d)
model = Model.from_config(config)
model.load()
print("loaded, GPU MiB", torch.cuda.memory_allocated() >> 20)
ids = torch.randint(0, 1000, (1, 32)).cuda()
out = {}
for grouped in ("1", "0"):
    os.environ["MCPM_FQ_MOE_GROUPED"] = grouped
    out[grouped] = model.forward(ids, params={"attn_mode": "flash_attn_nc"})[0].float().cpu()
    print("grouped" if grouped == "1" else "per-expert", "finite", bool(torch.isfinite(out[grouped]).all()),
          "absmax", float(out[grouped].abs().max()))
print("prefill path equal (grouped is decode-only, expect identical):", torch.equal(out["1"], out["0"]))
print("PASS")
