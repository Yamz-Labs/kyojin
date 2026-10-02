# Step 11: pinned intra (EXL3_KDA_INTRA=2) vs fla autotune (0) on ragged T (partial last 64-block), whole chunk_kda.
import os, sys, torch
sys.path.insert(0, os.getcwd())
torch.set_grad_enabled(False)
import exllamav3.vendor.fla as F
d = torch.load("scratch/step10/cap/t4096.pt")
d = {k: (v.to("cuda:0") if torch.is_tensor(v) else v) for k, v in d.items()}
for T in (513, 777, 1500, 3001, 4095):
    def ck(a):
        os.environ["EXL3_KDA_INTRA"] = a
        s0 = d["initial_state"].clone() if d["initial_state"] is not None else None
        return F.chunk_kda(d["q"][:, :T].contiguous(), d["k"][:, :T].contiguous(), d["v"][:, :T].contiguous(),
                           g=d["g"][:, :T].contiguous(), beta=d["beta"][:, :T].contiguous(), initial_state=s0,
                           output_final_state=True, use_qk_l2norm_in_kernel=bool(d["use_qk_l2norm_in_kernel"]))
    o0, s0 = ck("0"); o2, s2 = ck("2")
    print(f"T={T} o_bitwise={torch.equal(o0, o2)} s_bitwise={torch.equal(s0, s2)} "
          f"o_mad={(o0.float() - o2.float()).abs().max().item():.3e}", flush=True)
