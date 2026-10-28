"""Per-layer MLA attention outputs of one short prefill, fp16 vs tq cache (separate processes: a module cannot mix cache kinds).
  tq_layer_dbg.py PACK DOCS.pt OUT.pt MODE(fp16|tq6) [N_TOKENS] [CHUNK]"""
import sys
import torch


def main(pack, docs, out, mode, n=256, chunk=0):
    from exllamav3 import Cache, CacheLayer_quant, Config, Model
    from exllamav3.cache.tq import CacheLayer_tq
    from exllamav3.modules.mla_attn import MLAttention
    ids = torch.load(docs)["ids"][0][:n]
    m = Model.from_config(Config.from_directory(pack))
    kw = dict(max_num_tokens=(n + 255) // 256 * 256 + 256, max_batch_size=2, max_history=3)
    cache = Cache(m, **kw) if mode == "fp16" else Cache(m, layer_type=CacheLayer_tq, k_bits=int(mode[2:]), v_bits=int(mode[2:]), **kw)
    m.load(progressbar=False)
    rec = []
    orig = MLAttention.forward
    saved = {}
    from exllamav3.cache.tq import TQState
    ro = TQState.rotate
    cnt = [0]

    def rot(self, *xs):
        y = ro(self, *xs)
        if cnt[0] < 3 and xs[0].shape[-1] == 512:
            cnt[0] += 1
            x0 = xs[0].float()
            print("rotate in finite", bool(torch.isfinite(x0).all()), "max", x0.abs().max().item(), "out finite", bool(torch.isfinite(y.float()).all()), "rot_t", self.rot_t.dtype, bool(torch.isfinite(self.rot_t.float()).all()), flush=True)
        return y
    TQState.rotate = rot
    if mode != "fp16":
        from exllamav3.cache.mla import CacheLayer_MLA_tq
        uo = CacheLayer_MLA_tq.update_kv_direct

        def ukd(self, cache_seqlens, block_table, k, v, length):
            saved.setdefault(id(self), []).append((int(cache_seqlens[0]), k.detach().clone()))
            return uo(self, cache_seqlens, block_table, k, v, length)
        CacheLayer_MLA_tq.update_kv_direct = ukd

    def fwd(self, x, params, out_dtype=None):
        o = orig(self, x, params, out_dtype) if out_dtype is not None else orig(self, x, params)
        rec.append((self.key, o.detach().float().cpu()))
        return o
    MLAttention.forward = fwd
    x = torch.tensor([ids])
    bt = torch.arange((n + 256) // 256, dtype=torch.int32)[None]
    chunk = chunk or n
    with torch.inference_mode():
        for a in range(0, n, chunk):
            b = min(a + chunk, n)
            m.prefill(x[:, a:b], {"attn_mode": "flash_attn", "cache": cache, "block_table": bt[:, : (b + 255) // 256],
                                  "cache_seqlens": torch.tensor([a], dtype=torch.int32)})
    if saved:
        from exllamav3.modules.attention_fn.tq_triton import tq_dequant_rows
        for lid, lst in list(saved.items())[:3]:
            layer = [l for l in cache.layers.values() if id(l) == lid][0]
            pg = torch.arange(len(lst) and 8, dtype=torch.int64, device=layer.qk.device)
            G = 512 // 32
            rows = sum(k.shape[1] for _, k in lst)
            n_pg = (rows + 255) // 256
            pc = layer.qk.reshape(-1, 256, G * layer.bits)[:n_pg].reshape(-1, G * layer.bits)
            ps = layer.sk.reshape(-1, 256)[:n_pg].reshape(-1)
            o = torch.empty((pc.shape[0], 512), dtype=torch.half, device=pc.device)
            tq_dequant_rows(pc, ps, layer.state.cbk, o, layer.bits, 512)
            ref = torch.cat([k.reshape(-1, 512) for _, k in lst]).float()
            d = o[: ref.shape[0]].float()
            bad = ~torch.isfinite(ref).all(-1); print("nan rows", int(bad.sum()), "of", ref.shape[0], "first", int(bad.nonzero()[0]) if bad.any() else -1, "chunks", [(a, tuple(k.shape)) for a, k in lst][:3]); ref = ref[~bad]; d = d[~bad]
            print("roundtrip", lid, "rel", ((d - ref).norm() / ref.norm()).item(), "|ref|", ref.norm(dim=-1).mean().item(), "|dq|", d.norm(dim=-1).mean().item(), flush=True)
    torch.save(rec, out)
    print("saved", len(rec), flush=True)


def cmp(a, b):
    ra, rb = torch.load(a), torch.load(b)
    for (k, x), (_, y) in zip(ra, rb):
        e = ((x - y).abs().max() / y.abs().max().clamp_min(1e-6)).item()
        print(k, f"rel {e:.3e}", f"|ref| {y.abs().max().item():.3g}")


if __name__ == "__main__":
    if sys.argv[1] == "cmp":
        cmp(*sys.argv[2:4])
    else:
        a = sys.argv
        main(a[1], a[2], a[3], a[4], int(a[5]) if len(a) > 5 else 256, int(a[6]) if len(a) > 6 else 0)
