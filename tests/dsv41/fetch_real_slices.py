"""
Fetch a few MB of REAL V4.1 weights by HTTP range (no full download) into DSV41_REAL_DIR so test_dsv41_real_slices
can cross-check the official (fp8/fp4/e8m0) and the antirez GGUF (Q8_0/IQ2_XXS/Q2_K/F16/I8) decoders on real bytes.
    python tests/dsv41/fetch_real_slices.py
"""
import json, os, struct, sys, urllib.request

REAL = os.environ.get("DSV41_REAL_DIR", "./ds41_work/ds41port/scratch/real")
HF = "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/main/"
GG = "https://huggingface.co/antirez/deepseek-v4.1-flash-gguf/resolve/main/DeepSeek-V4.1-Flash-Q2.gguf"
INDEX = "./ds41_work/ds41port/ref/model.safetensors.index.json"
HEAD = "./ds41_work/ds41probe/q2head.bin"


def rng(url, a, n):
    req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{a + n - 1}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = r.read()
    assert len(d) == n, (url, a, n, len(d))
    return d


_hdr = {}


def st_info(shard):
    if shard not in _hdr:
        n = struct.unpack("<Q", rng(HF + shard, 0, 8))[0]
        _hdr[shard] = (n, json.loads(rng(HF + shard, 8, n)))
    return _hdr[shard]


def st_tensor(name, rows=None, row_bytes=None):
    wm = json.load(open(INDEX))["weight_map"]
    shard = wm[name]
    n, h = st_info(shard)
    t = h[name]
    b, e = t["data_offsets"]
    base = 8 + n + b
    if rows is None:
        return t, rng(HF + shard, base, e - b)
    return t, {r: rng(HF + shard, base + r * row_bytes, row_bytes) for r in rows}


def main():
    os.makedirs(REAL, exist_ok=True)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from exllamav3.dsv41.gguf import GGUFFile
    g = GGUFFile(HEAD)
    # 0. same file as the header probe?
    first = rng(GG, 0, 4096)
    assert first == open(HEAD, "rb").read(4096), "remote GGUF differs from the local header probe"
    out = {}

    def gget(name, start=0, n=None):
        t = g.tensors[name]
        n = t.nbytes() - start if n is None else n
        return rng(GG, g.data_start + t.offset + start, n)

    # 1. Q8_0 attn_kv (layer 0) vs official wkv
    t, w = st_tensor("layers.0.attn.wkv.weight"); _, s = st_tensor("layers.0.attn.wkv.scale")
    open(f"{REAL}/off_wkv.bin", "wb").write(w); open(f"{REAL}/off_wkv_s.bin", "wb").write(s)
    open(f"{REAL}/gg_wkv.bin", "wb").write(gget("blk.0.attn_kv.weight"))
    # 2. experts: gate (w1) and down (w2) of expert 3, layer 0
    for w_, gname, per_name in (("w1", "ffn_gate_exps", None), ("w2", "ffn_down_exps", None)):
        _, a = st_tensor(f"layers.0.ffn.experts.3.{w_}.weight"); _, sc = st_tensor(f"layers.0.ffn.experts.3.{w_}.scale")
        open(f"{REAL}/off_e3_{w_}.bin", "wb").write(a); open(f"{REAL}/off_e3_{w_}_s.bin", "wb").write(sc)
        t_ = g.tensors[f"blk.0.{gname}.weight"]
        per = t_.dims[1] * t_.row_bytes()
        open(f"{REAL}/gg_e3_{w_}.bin", "wb").write(gget(f"blk.0.{gname}.weight", 3 * per, per))
    # 3. engram rows (layer 1): official values+scales vs GGUF interleaved rows
    rows = [0, 1, 2, 3, 4095, 123456789, 384006167]
    t, vals = st_tensor("layers.1.engram.embed.weight", rows, 256)
    t2, scl = st_tensor("layers.1.engram.embed.scale", rows, 8)
    gt = g.tensors["blk.1.engram_embd.weight"]
    assert gt.dims == (264, 384006168) and t["shape"][0] == 384006168
    gg_rows = {r: rng(GG, g.data_start + gt.offset + r * 264, 264) for r in rows}
    json.dump({"rows": rows, "off": {r: (vals[r] + scl[r]).hex() for r in rows}, "gg": {r: gg_rows[r].hex() for r in rows}},
              open(f"{REAL}/engram_rows.json", "w"))
    print("fetched:", {f: os.path.getsize(f"{REAL}/{f}") for f in sorted(os.listdir(REAL))})


if __name__ == "__main__":
    main()
