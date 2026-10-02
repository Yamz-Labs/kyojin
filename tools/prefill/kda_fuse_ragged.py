# Step 12: HIP fused_h (EXL3_KDA_FUSE=3) vs Triton fused_h (2) on ragged T (partial last 64-block), whole chunk_kda.
import os, sys, torch
sys.path.insert(0, os.getcwd())
torch.set_grad_enabled(False)
import exllamav3.vendor.fla as F
d = torch.load("scratch/step10/cap/t4096.pt")
d = {k: (v.to("cuda:0") if torch.is_tensor(v) else v) for k, v in d.items()}
for T in (513, 777, 1500, 3001, 4095):
    def ck(a):
        os.environ["EXL3_KDA_FUSE"] = a
        s0 = d["initial_state"].clone() if d["initial_state"] is not None else None
        return F.chunk_kda(d["q"][:, :T].contiguous(), d["k"][:, :T].contiguous(), d["v"][:, :T].contiguous(),
                           g=d["g"][:, :T].contiguous(), beta=d["beta"][:, :T].contiguous(), initial_state=s0,
                           output_final_state=True, use_qk_l2norm_in_kernel=bool(d["use_qk_l2norm_in_kernel"]))
    o0, s0 = ck("2"); o2, s2 = ck("3"); assert not F._HIP_FAIL, F._HIP_FAIL
    print(f"T={T} o_bitwise={torch.equal(o0, o2)} s_bitwise={torch.equal(s0, s2)} "
          f"o_mad={(o0.float() - o2.float()).abs().max().item():.3e}", flush=True)


def ck2(a, q, k, v, g, beta, s0, ofs):
    os.environ["EXL3_KDA_FUSE"] = a
    return F.chunk_kda(q, k, v, g=g, beta=beta, initial_state=None if s0 is None else s0.clone(), output_final_state=ofs,
                       use_qk_l2norm_in_kernel=bool(d["use_qk_l2norm_in_kernel"]))


# B=2 (two different slices), no initial state, no final state
st = lambda n, a, b: torch.cat([d[n][:, a:a + 1000], d[n][:, b:b + 1000]], 0).contiguous()
q2, k2, v2, g2, b2 = (st(n, 0, 2048) for n in ("q", "k", "v", "g", "beta"))
h2 = None if d["initial_state"] is None else torch.cat([d["initial_state"], 0.5 * d["initial_state"]], 0).contiguous()
for name, s0_, ofs in (("B2 h0", h2, True), ("B2 no-h0", None, True), ("B2 no-ht", h2, False)):
    oa, sa = ck2("2", q2, k2, v2, g2, b2, s0_, ofs); ob, sb = ck2("3", q2, k2, v2, g2, b2, s0_, ofs)
    assert not F._HIP_FAIL, F._HIP_FAIL
    print(f"T={name} o_bitwise={torch.equal(oa, ob)} s_bitwise={sa is None and sb is None or torch.equal(sa, sb)}", flush=True)
