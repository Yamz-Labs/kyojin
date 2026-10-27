"""draft_counts.py MODEL_DIR OUT.pt FILE... : per-128-id-block token counts of a calibration corpus (any mix of languages / code),
the seed of the adaptive draft-head subset (EXL3_DRAFT_COUNTS=OUT.pt, see exllamav3/architecture/qwen3_5_mtp.py). Tokenizer only, no GPU."""
import sys, os, json
import torch
from tokenizers import Tokenizer

def main():
    model, out, files = sys.argv[1], sys.argv[2], sys.argv[3:]
    cfg = json.load(open(os.path.join(model, "config.json")))
    vocab = (cfg.get("text_config") or cfg)["vocab_size"]
    tk = Tokenizer.from_file(os.path.join(model, "tokenizer.json"))
    counts = torch.zeros((vocab + 127) // 128, dtype = torch.float32)
    n = 0
    for f in files:
        try:
            ids = tk.encode(open(f, errors = "strict").read()[:200000]).ids
        except (UnicodeDecodeError, OSError):
            continue
        counts += torch.bincount(torch.tensor(ids, dtype = torch.long) // 128, minlength = counts.numel())[:counts.numel()].float()
        n += len(ids)
    torch.save(counts, out)
    print(f"{n} tokens, {int((counts > 0).sum())} of {counts.numel()} blocks seen -> {out}")

main()
