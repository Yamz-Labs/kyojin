"""offline exec of Engine._mla2_precompile (extracted by ast) on fake objects: CPU, TRITON_INTERPRET=1"""
import ast, os, sys, types, importlib.util, torch
os.environ["HY4_MLA_ROWS2"] = "1"; os.environ["HY4_MLA_ROWS2_PRECOMPILE"] = "1"; os.environ["TRITON_INTERPRET"] = "1"
spec = importlib.util.spec_from_file_location("ref_fused", "exllamav3/loader/ref_fused.py"); rf = importlib.util.module_from_spec(spec); spec.loader.exec_module(rf)
src = open("tools/hy4/engine.py").read()
fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "_mla2_precompile")
ns = {}; exec(compile(ast.Module([fn], []), "engine.py", "exec"), ns)
mla = lambda w, ring: types.SimpleNamespace(H=16, R=64, rope_dim=16, window=w, kv_ring=ring)
layers = [types.SimpleNamespace(mla=mla(0, 0)), types.SimpleNamespace(mla=mla(128, 0)), types.SimpleNamespace(mla=mla(128, 192))]
gb = types.ModuleType("graph_bucket"); gb.buckets = lambda cap, base=512, *a: list(range(base, cap + 1, base))
pkg = types.ModuleType("exllamav3.loader"); pkg.graph_bucket = gb
sys.modules["exllamav3"] = types.ModuleType("exllamav3"); sys.modules["exllamav3.loader"] = pkg; sys.modules["exllamav3.loader.graph_bucket"] = gb
self = types.SimpleNamespace(hf=types.SimpleNamespace(_rf=rf), M=types.SimpleNamespace(n_layers=3, _layer=lambda i: layers[i]), MT=types.SimpleNamespace(mla=mla(0, 0)),
                             MG=types.SimpleNamespace(mt=types.SimpleNamespace(cap=1024), BUCKET=512), GR=types.SimpleNamespace(cap=1024), SG=types.SimpleNamespace(TM=2), dev=torch.device("cpu"))
ns["_mla2_precompile"](self)
rf.R2_PRECOMPILE = False
n = []; rf.mla2_precompile = lambda *a, **k: n.append(1); ns["_mla2_precompile"](self); assert not n    # flag off: no-op
print("ENGINE_HOOK_OK")
