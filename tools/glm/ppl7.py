# glm-mtp step 7: PPL 10x4096 (glm_base.eval_ppl protocol) with the current worktree env. Prints RESULT + DONE.
import json, os, sys, time, torch
H = os.path.expanduser("~"); HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else sys.argv[1]
sys.argv = ["glm_base", "fast", "-o", OUT + ".glm_base.json", "-m", f"{H}/models/glm53-exl3-td205",
            "--corpus", f"{H}/bench/ppl/wiki.test.raw"]
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE))); sys.path.insert(0, HERE)
import glm_base as G
torch.set_grad_enabled(False)
from exllamav3 import Config, Model, Tokenizer
t0 = time.time()
config = Config.from_directory(G.args.model)
model = Model.from_config(config); tok = Tokenizer.from_config(config)
model.load(device="cuda:0", progressbar=False)
print(f"loaded {time.time() - t0:.0f}s", flush=True)
r = G.eval_ppl(model, tok, 10, 4096, "step7")
r["env"] = {k: v for k, v in os.environ.items() if k.startswith("EXL3_")}
json.dump(r, open(OUT, "w"), indent=1)
print("RESULT", json.dumps({"ppl": round(r["ppl"], 6), "seconds": round(r["seconds"], 1)}), flush=True)
print("DONE errors 0", flush=True)
