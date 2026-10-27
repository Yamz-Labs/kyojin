"""Native Triton paged cache writer (any K/V width 2..8) vs the PyTorch fallback: the packed words and fp16 scales must be bit-identical.
python tests/test_qwrite.py"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch
from exllamav3 import ext_fallbacks as fb

dev = torch.device("cuda"); torch.manual_seed(0)
PAGE = 256; bad = 0
for kvh, hd in ((4, 128), (2, 256)):
    dim = kvh * hd; G = dim // 32
    for kb, vb in ((8, 8), (6, 6), (4, 4), (5, 3), (7, 2), (2, 8), (3, 3), (4, 8), (2, 2)):
        for bsz, seq_len, contig in ((1, 1, True), (2, 1, True), (1, 37, True), (2, 300, True), (1, 40, False)):
            pages = bsz * (-(-(seq_len + 80) // PAGE)) + 2
            seqlens = torch.tensor([17 * (i + 1) for i in range(bsz)], dtype=torch.int32, device=dev)
            ppseq = -(-(int(seqlens.max()) + seq_len) // PAGE) + 1
            bt = torch.randperm(pages, device=dev)[: bsz * ppseq].to(torch.int32).view(bsz, ppseq).contiguous()
            n_in = bsz * seq_len * dim if contig else pages * PAGE * dim
            k_in = (torch.randn(n_in, device=dev) * torch.exp(torch.randn(n_in, device=dev))).clamp(-60000, 60000).to(torch.float16)
            v_in = (torch.randn(n_in, device=dev) * 2).to(torch.float16)
            shp = (bsz, seq_len, kvh, hd) if contig else (pages, PAGE, kvh, hd)
            k_in = k_in.view(shp); v_in = v_in.view(shp)
            outs = []
            for native in (True, False):
                fb._QW_OFF = not native
                ko = torch.zeros(pages, PAGE, G * kb, dtype=torch.int32, device=dev); vo = torch.zeros(pages, PAGE, G * vb, dtype=torch.int32, device=dev)
                ks = torch.zeros(pages, PAGE, G, dtype=torch.float16, device=dev); vs = torch.zeros_like(ks)
                fb.quant_cache_paged(k_in, ko, ks, v_in, vo, vs, seqlens, bt, PAGE, seq_len, 0.0, contig)
                outs.append((ko, vo, ks, vs))
            ok = all(torch.equal(a, b) for a, b in zip(*outs))
            if not ok:
                bad += 1
                nd = [int((a != b).sum()) for a, b in zip(*outs)]
                print(f"MISMATCH kvh{kvh} hd{hd} k{kb}v{vb} bsz{bsz} seq{seq_len} contig{contig}: differing elements k,v,ks,vs = {nd}")
print("writer: all identical" if not bad else f"writer: {bad} mismatching configs")
# speed: decode-sized (1 row) and prefill-sized (4096 rows) append, native vs fallback
for kb, vb in ((6, 6), (4, 4)):
    kvh, hd = 4, 128; dim = kvh * hd; G = dim // 32
    for seq_len in (1, 4096):
        pages = -(-(seq_len + 300) // PAGE) + 1
        bt = torch.arange(pages, device=dev, dtype=torch.int32).view(1, -1); seqlens = torch.tensor([100], dtype=torch.int32, device=dev)
        k_in = torch.randn(1, seq_len, kvh, hd, device=dev).to(torch.float16); v_in = torch.randn(1, seq_len, kvh, hd, device=dev).to(torch.float16)
        for native in (True, False):
            fb._QW_OFF = not native
            ko = torch.zeros(pages, PAGE, G * kb, dtype=torch.int32, device=dev); vo = torch.zeros(pages, PAGE, G * vb, dtype=torch.int32, device=dev)
            ks = torch.zeros(pages, PAGE, G, dtype=torch.float16, device=dev); vs = torch.zeros_like(ks)
            for _ in range(3): fb.quant_cache_paged(k_in, ko, ks, v_in, vo, vs, seqlens, bt, PAGE, seq_len, 0.0, True)
            torch.cuda.synchronize(); t = time.perf_counter(); n = 20 if seq_len == 1 else 3
            for _ in range(n): fb.quant_cache_paged(k_in, ko, ks, v_in, vo, vs, seqlens, bt, PAGE, seq_len, 0.0, True)
            torch.cuda.synchronize(); print(f"k{kb}v{vb} rows {seq_len:5d} {'native  ' if native else 'fallback'}: {(time.perf_counter() - t) / n * 1e6:10.1f} us/call")
