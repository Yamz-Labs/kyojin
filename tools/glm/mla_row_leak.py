# Does row 0 of a q_len=2 MLA decode depend on row 1 (its query or its appended key)?
import sys, torch
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from exllamav3.modules.attention_fn.mla_triton import mla_attn_triton_decode as dec
torch.manual_seed(0)
H, Dc, Dr, ps = 64, 512, 64, 256
for L in (100, 1000, 2087):
    pages = (L + 2 + ps - 1) // ps + 1
    ckv = torch.randn(pages, ps, 1, Dc, device="cuda", dtype=torch.half) * 0.5
    kpe = torch.randn(pages, ps, 1, Dr, device="cuda", dtype=torch.half) * 0.5
    bt = torch.arange(pages, device="cuda", dtype=torch.int32).view(1, -1)
    sl = torch.tensor([L], device="cuda", dtype=torch.int32)
    ql = torch.randn(H, 2, Dc, device="cuda", dtype=torch.half); qp = torch.randn(H, 2, Dr, device="cuda", dtype=torch.half)
    def run(ql, qp, ckv, kpe, **kw):
        return dec(ql.contiguous(), qp.contiguous(), ckv, kpe, bt, sl, bsz=1, q_len=2, causal=True,
                   softmax_scale=0.07, pre_appended_len=2, **kw).float()
    for sp in (None, 1):
        kw = {} if sp is None else {"num_splits": sp}
        o = run(ql, qp, ckv, kpe, **kw)
        ql2 = ql.clone(); ql2[:, 1] = torch.randn(H, Dc, device="cuda", dtype=torch.half)
        o_q = run(ql2, qp, ckv, kpe, **kw)
        # row 1's own key sits at position L + 1 (page L+1 // ps)
        ckv2, kpe2 = ckv.clone(), kpe.clone(); p, s = divmod(L + 1, ps)
        ckv2[p, s] *= -4; kpe2[p, s] *= -4
        o_k = run(ql, qp, ckv2, kpe2, **kw)
        # sanity: row 0's own key (L) must matter
        ckv3 = ckv.clone(); p0, s0 = divmod(L, ps); ckv3[p0, s0] *= -4
        o_s = run(ql, qp, ckv3, kpe, **kw)
        print(f"L={L} splits={sp} row0 dq1={float((o[:,0]-o_q[:,0]).abs().max()):.3g} dk1={float((o[:,0]-o_k[:,0]).abs().max()):.3g}"
              f" | row1 dk1={float((o[:,1]-o_k[:,1]).abs().max()):.3g} row0 dk0={float((o[:,0]-o_s[:,0]).abs().max()):.3g}", flush=True)
