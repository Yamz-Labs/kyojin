# Decode-step byte budget of DS4.1-Flash on the box (pack: routed 2.5 bpw, attn/shared/engram/head 4.5, hc/router/compressor fp32 raw)
BW = 190e9   # practical GB/s (k2kern measured 177 on the VALU GEMV, dedup 164; peak 200-256)
M = 1e6
H, I, E, TK = 5120, 2304, 384, 6
attn_p = H*1280 + 1280*64*512 + H*512 + 8*(4096*1024) + 8192*H           # wq_a wq_b wkv wo_a wo_b
sh_p   = 3*H*I
rt_p   = TK*3*H*I
b = lambda p, bpw: p*bpw/8
rows = []
def add(name, byt, n=1): rows.append((name, byt*n))
add("attn proj (wq_a,wq_b,wkv,wo_a,wo_b) x40", b(attn_p, 4.5), 40)
add("shared expert x40", b(sh_p, 4.5), 40)
add("routed top-6 x40 (2.5 bpw)", b(rt_p, 2.5), 40)
add("router gate fp32 x40", E*H*4, 40)
add("hc fn fp32 (2 sites) x40", 2*24*4*H*4, 40)
add("compressor wkv+wgate fp32 (layers 2,8,14: both; 20: wkv)", 3*2*H*512*4 + H*512*4)
add("indexer wq_b (8 layers, 4.5bpw)", b(1280*4096, 4.5), 8)
add("indexer weights_proj+wk (8 layers, fp16)", (H*32 + 512*128)*2, 8)
add("engram wkv 6144->25600 (2 layers, 4.5bpw)", b(6144*25600, 4.5), 2)
add("engram rows (2 layers x 24 rows x 136 B)", 2*24*136)
add("LM head 129280x5120 (4.5bpw)", b(129280*H, 4.5))
tot = sum(v for _, v in rows)
print(f"{'component':62s} {'MB':>8s} {'ms@190':>7s} {'%':>5s}")
for n, v in rows: print(f"{n:62s} {v/M:8.1f} {v/BW*1e3:7.2f} {100*v/tot:5.1f}")
print(f"{'weights total':62s} {tot/M:8.1f} {tot/BW*1e3:7.2f}")
# KV-side traffic per decode step
def kv(ctx):
    win = 128*512*2                                  # fp16 ring per layer, 40 layers
    gather = 640*512*2                               # window(128)+top512 entries fp16 (pool stored fake-quant fp16)
    idx = 0
    for l in (2, 8, 14): idx += (ctx//2)*128*2       # ratio-2 index keys
    for l in (20, 24, 28, 32, 36): idx += ctx*128*2  # ratio-1 index keys (full scan; candidates mask applied after)
    idx_cand = idx - 4*ctx*128*2 + 4*min(ctx, 16384)*128*2   # layers 24..36 scoring only the 2048x8 candidate rows
    return 40*gather + 40*win, idx, idx_cand
for ctx in (8192, 131072):
    g, i, ic = kv(ctx)
    print(f"ctx {ctx:6d}: window+gather {g/M:6.1f} MB ({g/BW*1e3:.2f} ms)  indexer key scan {i/M:6.1f} MB ({i/BW*1e3:.2f} ms)  candidate-restricted {ic/M:6.1f} MB ({ic/BW*1e3:.2f} ms)")
print("decode ceiling w/o spec, 8K :", 1/((tot+kv(8192)[0]+kv(8192)[1])/BW), "tok/s")
print("decode ceiling w/o spec,128K:", 1/((tot+kv(131072)[0]+kv(131072)[1])/BW), "tok/s")
# prefill FLOPs per token
act = (attn_p + sh_p + rt_p)*40 + 2*6144*25600 + 129280*H*0
print("active matmul params/token (no head):", (attn_p+sh_p+rt_p)*40/1e9, "B ->", 2*(attn_p+sh_p+rt_p)*40/1e9, "GFLOP/token")
for L in (8192, 131072):
    # indexer scoring FLOPs for a full prefill of L tokens: per index layer sum_t (t/r) * 32*128*2
    f = 0
    for l in (2, 8, 14): f += (L*L/2/2)*32*128*2
    for l in (20, 24, 28, 32, 36): f += (L*L/2)*32*128*2
    fc = f - 4*(L*L/2)*32*128*2 + 4*L*min(L, 16384)*32*128*2*0.5 if L > 16384 else f
    main = 2*(attn_p+sh_p+rt_p)*40*L
    print(f"prefill L={L}: main matmul {main/1e12:7.1f} TFLOP, indexer scoring {f/1e12:7.1f} TFLOP (candidate-restricted {fc/1e12:7.1f})")
