"""block_ldl with concurrent same-device worker threads (EXL3_QUANT_THREADS > 1) must give the same factor as a single
worker: no spurious Cholesky failure, no silently wrong factor. Needs a GPU (take gpu-lease on shared boxes).
Run: PYTHONPATH=$PWD python tests/test_block_ldl_threads.py"""
import threading, torch
from exllamav3.modules.quant.exl3_lib import quantize as q

N, B = 1024, 16
QA = {"sigma_reg": 0.025}


def make_h(seed):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(N, 4 * N, generator = g) * torch.logspace(0, -2, N).unsqueeze(1)
    H = X @ X.T / X.shape[1]
    H.diagonal().add_(0.025 * H.diagonal().mean())
    return H


def check_single_worker_bit_identical():
    H = make_h(7).cuda()
    assert torch.equal(q._cholesky_checked(H), torch.linalg.cholesky(H))
    A = torch.randn(N // B, B, B, device = "cuda") + 4 * torch.eye(B, device = "cuda")
    assert torch.equal(q._inv_checked(A), torch.linalg.inv(A))


def check_threads(nt = 3, reps = 40):
    Hs = [make_h(s) for s in range(4)]
    ref = [q.block_ldl(H.cuda().clone(), B, QA, False)[0].cpu() for H in Hs]  # single worker, GPU path
    worst = [0.0] * nt

    def work(i):
        s = torch.cuda.Stream()
        q.set_worker_stream(s)
        x = torch.randn(2048, 2048, device = "cuda")
        try:
            with torch.cuda.stream(s):
                for r in range(reps):
                    x @ x  # unrelated concurrent GPU work, what used to corrupt the solver
                    L, _ = q.block_ldl(Hs[r % 4].cuda().clone(), B, QA, False)
                    d = ((L.cpu() - ref[r % 4]).abs().max() / ref[r % 4].abs().max()).item()
                    worst[i] = max(worst[i], d)
        finally:
            q.set_worker_stream(None)

    ts = [threading.Thread(target = work, args = (i,)) for i in range(nt)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert max(worst) < 1e-4, worst


def check_indefinite_still_raises():
    H = make_h(3).cuda()
    H.diagonal()[0] = -1.0
    for ws in (None, torch.cuda.Stream()):
        q.set_worker_stream(ws)
        try:
            q._cholesky_checked(H)
        except torch._C._LinAlgError:
            continue
        finally:
            q.set_worker_stream(None)
        raise AssertionError("indefinite matrix must still raise _LinAlgError")


if __name__ == "__main__":
    check_single_worker_bit_identical()
    check_indefinite_still_raises()
    check_threads()
    print("OK")
