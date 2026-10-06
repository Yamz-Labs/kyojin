"""CPU bounds proof of the cache WRITER (quant_cache_paged_kernel<8,8>, q_cache.cu launch + q_cache_kernels.cuh quant_block_x4) at the Qwen shape.
The address model is derived from the kernel text (read line by line, see notes): launch geometry + per-warp spans + the staging buffer.
Every store / load of every (z, y, warp, lane) is checked against the declared tensor sizes. --mutate N breaks one rule; must FAIL."""
import sys, itertools
mut = int(sys.argv[sys.argv.index("--mutate") + 1]) if "--mutate" in sys.argv else 0
KVH, HD, BITS, PAGE, NPAGES = 2, 256, 8, 256, 1024
dim = KVH * HD; gpt = dim // 32                         # groups_per_token = 16
chunks = -(-gpt // 4); tb_per_token = -(-chunks // 8); tb_usage = -(-chunks // tb_per_token)
assert (gpt, chunks, tb_per_token, tb_usage) == (16, 4, 1, 4)
n_tok = NPAGES * PAGE
W = n_tok * gpt * BITS; S = n_tok * gpt                # words, scales
viol = []
def chk(what, lo, hi, size):
    if not (0 <= lo and hi <= size): viol.append(f"{what}: [{lo},{hi}) outside {size}")
import random; rnd = random.Random(1)
def run(bsz, seq_len, seqlens, bt, in_contig):
    in_rows = bsz * seq_len if in_contig else n_tok
    for z in range(bsz):
        for y in range(seq_len):
            token_idx = y + seqlens[z]; page_idx = token_idx // PAGE
            assert page_idx < len(bt[z]), "block table too short"
            token_pos = (bt[z][page_idx] + (1 if mut == 3 else 0)) * PAGE + token_idx % PAGE
            in_pos = z * seq_len + y if in_contig else token_pos
            for x in range(tb_per_token):
                for warp in range(tb_usage):
                    g0 = (x * tb_usage + warp) * 4
                    if g0 >= gpt: continue
                    active = min(4, gpt - g0) + 0
                    base = token_pos * gpt + g0 + (4 if mut == 2 else 0); in_base = in_pos * gpt + g0
                    # staging: lanes < 4*bits clear sh_pack[lane]; atomicOr index = sg*bits + word_base + (off>>5), sg<active, word_base 0, off = sl*4*8
                    for sg in range(min(active, 4)):
                        for sl in range(8):
                            idx = sg * BITS + ((sl * 32) >> 5); assert 0 <= idx < 32, "sh_pack overflow"
                    # global writes: out[lane] for lane < 4*bits and lane/bits < active -> words base*bits + lane
                    for lane in range(32):
                        if lane < 4 * BITS and (lane // BITS) < active: chk("k/v words", base * BITS + lane, base * BITS + lane + 1, W)
                    for lane in range(4):
                        if lane < active: chk("k/v scales", base + lane, base + lane + 1, S)
                    # loads: half2 reads in[lane*2], in[lane*2+1] for the 4 groups (128 values) when sg < active
                    for lane in range(32):
                        if (lane >> 3) < active:
                            chk("input rows", in_base * 32 + lane * 4, in_base * 32 + lane * 4 + 4, in_rows * gpt * 32)
cases = [(1, 1, [255], [[1023]], True), (1, 1, [255], [[1023]], False), (1, 1, [0], [[1023, 0, 255, 7] + [1] * 4], True), (1, 4, [1020], [[1023, 0, 255, 7] + [1] * 4], True),
         (1, 2048, [0], [list(range(1015, 1023)) + [0, 1, 2]], False), (1, 2048, [1500], [[1023 - i for i in range(16)]], False),
         (2, 3, [255, 700], [[5, 6, 7, 8], [1023, 1022, 1021, 1020]], True)]
for c in cases: run(*c)
# int overflow: widest index products
big = (n_tok * gpt * BITS, (n_tok - 1) * gpt * BITS * 32 // 32)
assert n_tok * gpt * BITS < 2**31, "int overflow in base * bits"
print("MUTANT", mut, "VIOLATIONS" if viol else "CLEAN", viol[:3]); sys.exit(1 if viol else 0)
