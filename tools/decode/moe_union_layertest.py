# Layer-level proof for exl3_dec_moe_union (device table, production workspace) vs R independent batch-1 routed forwards (same routing forced).
# usage: EXL3_REPO=. MODEL=<pack> python moe_union_layertest.py LAYERS|all [N]   env RS=2,3,..8 NO_FUSE=1
import os, sys, torch, json, time
sys.path.insert(0, os.environ["EXL3_REPO"])
from exllamav3 import Config, Model
import exllamav3.modules.block_sparse_mlp as bsm
from exllamav3.modules.quant.exl3 import dec_workspace
ext = bsm.ext; DEV = "cuda:0"; torch.set_grad_enabled(False)
cfg = Config.from_directory(os.environ["MODEL"]); model = Model.from_config(cfg)
RS = [int(v) for v in os.environ.get("RS", "1,2,3,4,5,6,7,8").split(",")]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 8
scratch, counters = dec_workspace(torch.device(DEV))
os.environ["EXL3_DEC_MOE_UNION_DEV"] = "1"; os.environ["EXL3_MOE_UNION_V2"] = "0"
def make_x(R, H, alpha, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    base = torch.randn(H, device=DEV, generator=g); x = base[None] + alpha * torch.randn(R, H, device=DEV, generator=g)
    x = x.view(R, H); x[:, :8] *= 40.0
    return (x / x.pow(2).mean(-1, keepdim=True).sqrt()).half().view(1, R, H)
def capture(mlp, x, R):
    seen = {}
    def spy_u(y, out, sel, wts, *a, **k): seen["sel"] = sel.clone(); seen["wts"] = wts.clone(); return real_u(y, out, sel, wts, *a, **k)
    def spy_1(y, out, sel, wts, *a, **k): seen["sel"] = sel.clone(); seen["wts"] = wts.clone(); return real_1(y, out, sel, wts, *a, **k)
    real_u, real_1 = ext.exl3_dec_moe_union, ext.exl3_dec_moe
    ext.exl3_dec_moe_union, ext.exl3_dec_moe = spy_u, spy_1
    try: mlp.forward(x, {"attn_mode": "flash_attn", "dflash_verify": True})
    finally: ext.exl3_dec_moe_union, ext.exl3_dec_moe = real_u, real_1
    return seen["sel"].view(R, -1).contiguous(), seen["wts"].view(R, -1).contiguous()
def ref_rows(mlp, x, sel, wts, R):
    return torch.cat([mlp.forward(x[:, i:i+1], {"attn_mode": "flash_attn", "dec_routed": (sel[i:i+1].contiguous(), wts[i:i+1].contiguous())}).float().clone() for i in range(R)], dim=1)
def union(mlp, x, sel, wts, R, acc_init=None):
    H = mlp.expert_size; numex = mlp.num_experts_per_tok
    out = mlp.dec_moe_union_out[:R] if acc_init is None else acc_init.clone().view(R, H)
    ext.exl3_dec_moe_union(x.view(R, H).contiguous(), out, sel, wts,
        mlp.multi_gate.ptrs_trellis, mlp.multi_gate.ptrs_suh, mlp.multi_gate.ptrs_svh,
        mlp.multi_up.ptrs_trellis, mlp.multi_up.ptrs_suh, mlp.multi_up.ptrs_svh,
        mlp.multi_down.ptrs_trellis, mlp.multi_down.ptrs_suh, mlp.multi_down.ptrs_svh,
        mlp.dec_moe_union_act[:R * numex], mlp.dec_moe_union_down_part[:R * numex], scratch, counters,
        mlp.intermediate_size_padded, mlp.multi_up.K, mlp.multi_down.K, acc_init is not None, mlp.act_limit, mlp.dec_moe_mcg, True)
    return out.view(1, R, H).float().clone()
mods = [i for i, m in enumerate(model.modules) if hasattr(m, "mlp") and hasattr(m.mlp, "multi_up")]
layers = [i - 1 for i in mods] if sys.argv[1] == "all" else [int(v) for v in sys.argv[1].split(",")]
classes = {}; t0 = time.time()
for L in layers:
    mlp = model.modules[L + 1].mlp
    mlp.load(torch.device(DEV)); H = mlp.expert_size
    cls = f"K{mlp.multi_gate.K}/{mlp.multi_up.K}/{mlp.multi_down.K} H{H} I{mlp.intermediate_size_padded} E{mlp.num_experts} top{mlp.num_experts_per_tok}"
    c = classes.setdefault(cls, {"layers": [], "max_abs_diff": 0.0, "max_rel": 0.0, "not_exact": 0, "nondet": 0, "leaks": 0, "calls": 0, "worst_fused_rel": 0.0, "by_R": {}})
    c["layers"].append(L); counters.zero_()
    for R in RS:
        if R == 1: continue
        for alpha, seed in ((0.6, 100 + R), (3.0, 200 + R), (0.2, 300 + R)):
            x = make_x(R, H, alpha, seed); sel, wts = capture(mlp, x, R)
            ref = ref_rows(mlp, x, sel, wts, R); sc = ref.abs().max().item()
            o0 = union(mlp, x, sel, wts, R)
            res0 = torch.randn(R, H, device=DEV, dtype=torch.float) * 3
            for it in range(N):
                o = union(mlp, x, sel, wts, R)
                d = (o - ref).abs().max().item(); c["calls"] += 1
                c["max_abs_diff"] = max(c["max_abs_diff"], d); c["max_rel"] = max(c["max_rel"], d / sc)
                b = c["by_R"].setdefault(R, 0.0); c["by_R"][R] = max(b, d / sc)
                if d != 0.0: c["not_exact"] += 1
                if not torch.equal(o, o0): c["nondet"] += 1
                if int((counters != 0).sum().item()): c["leaks"] += 1; counters.zero_()
                if os.environ.get("NO_FUSE") is None:
                    of = union(mlp, x, sel, wts, R, acc_init=res0)
                    exp = res0.view(1, R, H) + ref; fr = (of - exp).abs().max().item() / max(exp.abs().max().item(), 1e-9)
                    c["worst_fused_rel"] = max(c["worst_fused_rel"], fr)
                    if int((counters != 0).sum().item()): c["leaks"] += 1; counters.zero_()
    print(f"layer {L:2d} {cls}: max_rel={c['max_rel']:.1e} not_exact={c['not_exact']} nondet={c['nondet']} leaks={c['leaks']} fused_rel={c['worst_fused_rel']:.1e} [{time.time()-t0:.0f}s]", flush=True)
    mlp.unload(); torch.cuda.empty_cache()
print("CLASSES", json.dumps(classes))
