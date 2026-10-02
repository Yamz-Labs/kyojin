# P0 coherent-document control: is the long-context NLL loss a property of the shuffled depth_curve doc?
#
# One load. Doc = frozen token ids of ONE coherent book (tools/glm/data/book_ids.pt, made by
# scratch/p0doc/freeze.py: Dumas, Les Trois Mousquetaires, Gutenberg body, add_bos=False).
#   1) one prefill 0..E_max with logits, NLL of every 2048 chunk (full context, same as curve.json)
#   2) for each end E: the same targets (chunks p0 >= E-4096) scored with only 4K context
#      (fresh state, window ids[E-8192:E] at positions 0..8191; depth_curve ctrl scoring verbatim)
# Usage: p0_coherent.py <model_dir> <out.json> [ids.pt]
import json, os, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
ENDS = [int(x) for x in os.environ.get("P0_ENDS", "16384,32768,65536,81920,98304,114688,131072").split(",")]
model_dir, out = sys.argv[1], sys.argv[2]
ids_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "data", "book_ids.pt")
RES = {"mode": "p0_coherent", "ids": ids_path, "ends": ENDS, "env": {k: v for k, v in os.environ.items() if k.startswith("EXL3_")}}


def save():
    with open(out, "w") as f:
        json.dump(RES, f, indent=1)


def avail_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1048576


@torch.inference_mode()
def main():
    from exllamav3 import Config, Model, Cache, Tokenizer
    PAGE, C = 256, 2048
    Emax = max(ENDS)
    ids = torch.load(ids_path)
    assert ids.shape[-1] >= Emax + 1, ids.shape
    import hashlib
    RES["sha1_first_Emax1"] = hashlib.sha1(ids[0, : Emax + 1].numpy().tobytes()).hexdigest()
    cap = ((Emax + 64 + PAGE - 1) // PAGE + 1) * PAGE
    t0 = time.time()
    config = Config.from_directory(model_dir)
    model = Model.from_config(config)
    tok = Tokenizer.from_config(config)
    cache = Cache(model, max_num_tokens=cap, max_batch_size=1)
    model.load(device="cuda:0", progressbar=False)
    V = tok.actual_vocab_size
    RES["load_s"] = round(time.time() - t0, 1)
    print(f"loaded {RES['load_s']}s sha {RES['sha1_first_Emax1']}", flush=True)
    bt = torch.arange(cap // PAGE, dtype=torch.int32).unsqueeze(0)
    st = {"s": cache.get_new_state()}

    def fwd(x, pos):
        T = x.shape[-1]
        p = {"attn_mode": "flash_attn", "block_table": bt[:, : (pos + T + PAGE - 1) // PAGE], "cache": cache,
             "cache_seqlens": torch.tensor([pos], dtype=torch.int32), "recurrent_states": [st["s"]]}
        return model.forward(x, p)

    def nll(lg, p0, T):
        tgt = ids[0, p0 + 1 : p0 + T + 1].to(lg.device)
        lp = torch.log_softmax(lg[0, :, :V].float(), dim=-1)
        return -lp.gather(-1, tgt.unsqueeze(-1)).sum().item()

    # 1) full-context prefill with per-chunk NLL
    a = time.time()
    chunk = []
    for p0 in range(0, Emax, C):
        lg = fwd(ids[:, p0 : p0 + C], p0)
        chunk.append(nll(lg, p0, C) / C); del lg
        if (p0 + C) % 16384 == 0:
            RES["chunk_nll"] = [round(x, 4) for x in chunk]
            save()
            print(f"[pre] {p0 + C} nll16k {sum(chunk[-8:]) / 8:.4f} avail {avail_gb():.1f}", flush=True)
            if avail_gb() < 12:
                raise RuntimeError("MemAvailable < 12 GiB")
    RES["prefill_s"] = round(time.time() - a, 1)
    RES["chunk_nll"] = [round(x, 4) for x in chunk]
    save()

    # 2) 4K-context control per end
    RES["table"] = []
    a = time.time()
    for E in ENDS:
        st["s"].free(); st["s"] = cache.get_new_state()
        tot, n = 0.0, 0
        for p0 in range(E - 8192, E, C):
            lg = fwd(ids[:, p0 : p0 + C], p0 - (E - 8192))
            if p0 >= E - 4096:
                tot += nll(lg, p0, C); n += C
            del lg
        full = sum(chunk[(E - 4096) // C : E // C]) / 2
        row = {"end": E, "nll_full": round(full, 5), "nll_ctx4k": round(tot / n, 5)}
        RES["table"].append(row)
        print(f"[ctrl] {json.dumps(row)}", flush=True)
        save()
    RES["ctrl_s"] = round(time.time() - a, 1)
    save()
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
