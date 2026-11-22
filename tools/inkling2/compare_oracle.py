"""Inkling2Forward (torch, CPU) vs the llama.cpp CPU oracle dumps (llama-inkling-dump): l_out per application, result_norm, result_output.
usage: compare_oracle.py GGUF DUMPDIR "1,5,9,100;300,4000" [--tol 2e-3]
Prints one line per application (max abs / max|ref|, cosine) and a verdict line; exit 1 when any bar fails."""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cpu_env


def load(dump, name, call):
    ix = {}
    for l in open(os.path.join(dump, "index.txt")):
        f, *ne = l.rstrip("\n").rsplit(" ", 4)
        ix[f] = list(map(int, ne))
    fn = f"c{call}_{name}.bin"
    ne = ix[fn]
    a = np.fromfile(os.path.join(dump, fn), dtype=np.float32)
    return a.reshape(ne[1], ne[0]) if ne[1] > 1 else a.reshape(ne[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf"); ap.add_argument("dump"); ap.add_argument("ids")
    ap.add_argument("--tol", type=float, default=2e-3, help="bar on max abs diff / max |ref| per tensor")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=256)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    F = cpu_env.install()
    from exllamav3.loader.gguf_collection import GGUFCollection
    rec = {}
    call = [0]
    def hook(name, il, t):
        if name in ("l_out", "result_norm", "result_output"):
            rec[(call[0], name, il)] = t.detach().float().cpu().numpy()
    m = F.Inkling2Forward(GGUFCollection(a.gguf), hook=hook, chunk=a.chunk)
    bad, worst = 0, 0.0
    for c, grp in enumerate(a.ids.split(";")):
        call[0] = c
        ids = [int(x) for x in grp.split(",")]
        lg = m.step(ids)
        for il in list(range(m.n_app)) + [-1]:
            names = ["l_out"] if il >= 0 else ["result_norm", "result_output"]
            for nm in names:
                ref = load(a.dump, f"{nm}-{il}" if nm == "l_out" else nm, c)
                got = rec[(c, nm, il)]
                if nm == "result_output":
                    got = got[-1:] if got.ndim == 2 else got
                    ref = ref.reshape(-1)[:got.shape[-1]]
                    got = got.reshape(-1)
                if nm == "result_norm":
                    got = got[-1].reshape(-1)
                    ref = ref.reshape(-1)
                if nm == "l_out":
                    ref = ref.reshape(len(ids), -1)
                err = float(np.abs(got - ref).max() / (np.abs(ref).max() + 1e-12))
                cos = float((got * ref).sum() / (np.linalg.norm(got) * np.linalg.norm(ref) + 1e-12))
                worst = max(worst, err)
                flag = "" if err <= a.tol else "  <-- FAIL"
                bad += err > a.tol
                print(f"call {c} {nm}{'-' + str(il) if il >= 0 else ''}: maxrel {err:.2e} cos {cos:.6f}{flag}", flush=True)
                if nm == "result_output":
                    print(f"call {c} argmax ours {int(np.argmax(got))} ref {int(np.argmax(ref))}  top5 overlap {len(set(np.argsort(-got)[:5]) & set(np.argsort(-ref)[:5]))}/5", flush=True)
    print(f"VERDICT worst maxrel {worst:.2e} bar {a.tol:.0e} fails {bad}: {'PASS' if bad == 0 else 'FAIL'}")
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
