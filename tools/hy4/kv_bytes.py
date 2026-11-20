"""Hy4 persistent KV bytes per token, analytic from the model shapes and the allocation sites (the measured side: tools/hy4/kv_bytes_probe.py, runner job 50_k3_e_kvbytes).
Shapes (GGUF): 12 trunk full-attention (DSA) layers + the MTP draft layer, 30 window-512 layers, latent 512 + rope 64 per token, indexer: one 128-d key per 4 tokens (pool) and a
(value | gate) row of 2 x 128 per token (vgb). Sites: step_graph.GraphDecoder.prepare (kv / pool / vgb), mtp_head.Hy4MTP (kbuf), step_nosync.State (tok, ar), rope tables."""
FULL, MTP, SWA, R, P, KP, DIX = 12, 1, 30, 512, 64, 4, 128
RING_ROWS, VG_RING_ROWS = 1024, 4096
ROPE_TABLES_B = 256 + 2 * 512        # assumption, checked by the measured residual: indexer cos / sin (rope 64, shared) + two attn_eager tables [n, 2, 32, 2] fp32


def per_token(fp8=False, fp8_mtp=False, swa_ring=False, pool_bytes=4):
    """-> {component: (bytes per token, constant bytes)} of the persistent buffers at full capacity"""
    lat32, lat8 = (R + P) * 4, R + 4 + 2 * P            # fp32 latent + rope, fp8 e4m3 + fp32 scale + bf16 rope
    d = {}
    d["trunk latent K (12 full layers)"] = (FULL * (lat8 if fp8 else lat32), 0)
    d["MTP layer K"] = (MTP * (lat8 if (fp8 and fp8_mtp) else lat32), 0)
    d["window layers K (30)"] = (0, SWA * RING_ROWS * lat32) if swa_ring else (SWA * lat32, 0)
    d["indexer pool (13 layers)"] = ((FULL + MTP) * (DIX // KP) * pool_bytes, 0)
    d["indexer value|gate rows (13)"] = (0, (FULL + MTP) * VG_RING_ROWS * 2 * DIX * 4) if swa_ring else ((FULL + MTP) * 2 * DIX * 4, 0)
    d["token history + arange"] = (16, 0)
    d["rope tables"] = (ROPE_TABLES_B, 0)
    return d


def total(d):
    return sum(v[0] for v in d.values()), sum(v[1] for v in d.values())


if __name__ == "__main__":
    for name, kw in (("1.9.1 default (fp32)", {}), ("fp8 trunk only", dict(fp8=True)), ("HY4_KV_COMPACT=1", dict(fp8=True, fp8_mtp=True, swa_ring=True)),
                     ("COMPACT + fp16 pool", dict(fp8=True, fp8_mtp=True, swa_ring=True, pool_bytes=2))):
        d = per_token(**kw)
        t, c = total(d)
        print(f"== {name}: {t} B / token ({t / 1024:.1f} KB) + constant {c / 2**20:.0f} MiB")
        for k, (b, cc) in d.items():
            print(f"   {k:34s} {b:7d} B/token  const {cc / 2**20:7.1f} MiB")
