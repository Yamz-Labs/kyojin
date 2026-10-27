"""arch.py -- per-architecture constants for the ablit tools (one table, no model imports).

Every tool takes --arch {auto,glm,mimo_v2}. "auto" reads <model>/config.json
(model_type / architectures) and falls back to "glm" so the GLM path is unchanged.

Fields
  n_layers   decoder blocks the hook edits (the MTP / draft blocks are NOT counted and
             must stay untouched: they carry layer numbers >= n_layers)
  hidden     residual width
  readout    "hc_mean": block output is [1, seq, hc, hidden], average the hc streams (mHC)
             "plain":   block output is [1, seq, hidden]
  chat       chat wrapper, "{p}" is the user prompt
  pca_layer / proj_f   GLM-tuned defaults (ablit4/5) for the projected-direction arm and
             the subspace arm; other archs get them scaled by relative depth
  writers    text recorded in edit_spec.json ("apply" field)
"""
import json
import os

DS41_THINKING = True      # server default (ds41_chat.ChatOptions)
DS41_EFFORT = "high"
GLM_LAST = 44  # GLM has 45 blocks: layer-index defaults below are tuned on 0..44

ARCHS = {
    "glm": {
        "n_layers": 45,
        "hidden": 4096,
        "readout": "hc_mean",
        "chat": "[gMASK]<sop><|user|>\n{p}<|assistant|>\n",
        "pca_layer": 29,
        "proj_f": 29.344821947743544,
        "writers": ("every layer attn o_proj + dense down_proj (L0-2) + shared+routed "
                    "expert down_proj (L3-44); MTP/embed/router/gate-up/norms/"
                    "lm_head untouched."),
    },
    "mimo_v2": {
        "n_layers": 48,
        "hidden": 4096,
        "readout": "plain",
        # default template (thinking on): generation prompt ends right after the assistant tag
        "chat": "<|im_start|>user\n{p}<|im_end|><|im_start|>assistant\n",
        "pca_layer": None,  # derived: round(29 * 47 / 44)
        "proj_f": None,     # derived: 29.34 * 47 / 44
        "writers": ("every layer attn o_proj + dense down_proj (L0) + fused routed "
                    "expert down_proj (L1-47, no shared experts); MTP layers/embed/router/"
                    "gate-up/norms/lm_head untouched."),
    },
    "deepseek_v41": {
        "n_layers": 40,
        "hidden": 5120,
        # mHC: block output is [1, seq, hc=4, hidden]; the direction is read on the mean over the 4 streams
        # (same readout as the dspark target taps, h.mean(dim=2), and as GLM)
        "readout": "hc_mean",
        # rendered by tools/ds41/ds41_chat.render_chat (BOS + fullwidth <｜User｜> ... <｜Assistant｜><think>), the
        # server default (thinking on, effort high). "{p}" form kept for documentation only.
        "chat": "<｜begin▁of▁sentence｜><｜User｜>{p}<｜Assistant｜><think>",
        "renderer": "ds41_chat",
        "pca_layer": None,  # derived: round(29 * 39 / 44)
        "proj_f": None,     # derived: 29.34 * 39 / 44
        "writers": ("every layer attn wo_b + shared expert w2 + routed expert w2 (all 40 layers; the hook "
                    "projects the sublayer output before the mHC post/comb mix); MTP/dspark stages, embed, "
                    "Engram injection, router, gate/up, norms, hc params and head untouched."),
    },
}


def detect(model_dir):
    """Name of the architecture of an HF/EXL3 model dir (fallback: glm)."""
    try:
        c = json.load(open(os.path.join(os.path.expanduser(model_dir), "config.json")))
    except Exception:
        return "glm"
    s = (str(c.get("model_type", "")) + " " + " ".join(c.get("architectures", []))).lower()
    if "deepseek_v41" in s or "deepseekv41" in s:
        return "deepseek_v41"
    return "mimo_v2" if "mimo" in s else "glm"


def get(name="auto", model_dir=None):
    """Return a copy of the arch dict with derived fields filled and 'name' set."""
    if name in (None, "auto"):
        name = detect(model_dir) if model_dir else "glm"
    a = dict(ARCHS[name])
    a["name"] = name
    scale = (a["n_layers"] - 1) / GLM_LAST
    if a["pca_layer"] is None:
        a["pca_layer"] = int(round(ARCHS["glm"]["pca_layer"] * scale))
    if a["proj_f"] is None:
        a["proj_f"] = ARCHS["glm"]["proj_f"] * scale
    a["scale"] = scale
    return a


def chat_wrap(prompt, arch=None):
    a = arch or ARCHS["glm"]
    if a.get("renderer") == "ds41_chat":
        import sys
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ds41")
        if d not in sys.path:
            sys.path.insert(0, d)
        import ds41_chat
        return ds41_chat.render_chat([{"role": "user", "content": prompt}], thinking = DS41_THINKING,
                                     reasoning_effort = DS41_EFFORT)
    return a["chat"].replace("{p}", prompt)
