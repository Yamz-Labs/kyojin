"""luce_eval: metric closed forms, the public Lucebox teacher file (when downloaded), and an end-to-end run of two tiny
random DS4.1 models (CPU) against a synthetic teacher written in the Lucebox file format."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
for p in (str(REPO), str(REPO / "tests" / "dsv41")):      # appended: a PYTHONPATH engine (with a built extension) wins
    if p not in sys.path:
        sys.path.append(p)
import luce_eval as le  # noqa: E402

DATA = Path(os.environ.get("LUCE_DATA", "~/ds41_work/luce_data")).expanduser()


def test_metrics_closed_form():
    p = np.array([[0.5, 0.25, 0.125]])                       # teacher top-3, tail mass 0.125
    q = np.array([[0.25, 0.25, 0.25]])
    idx = np.array([[7, 3, 5]])
    m = le.compute_metrics(idx, np.log(p), np.array([7]), np.log(q), np.array([np.log(0.25)]), np.array([3]))
    kl = 0.5 * np.log(2) + 0 + 0.125 * np.log(0.5)
    assert m["kl_top256"] == pytest.approx(kl)
    assert m["top1_agree"] == 0.0 and m["ppl"] == pytest.approx(4.0)
    pr, qr = p / p.sum(), q / q.sum()
    assert m["kl_top256_renorm"] == pytest.approx(float((pr * np.log(pr / qr)).sum()))
    assert m["kld_tail"] == pytest.approx(kl + 0.125 * np.log(0.125 / 0.25))


def test_identical_distribution_is_zero():
    lp = np.log(np.array([[0.6, 0.3, 0.05]]))
    m = le.compute_metrics(np.array([[1, 2, 3]]), lp, np.array([1]), lp, np.array([lp[0, 0]]), np.array([1]))
    assert m["kl_top256"] == 0 and m["kl_top256_renorm"] == 0 and m["kld_tail"] == pytest.approx(0, abs=1e-12) and m["top1_agree"] == 1


@pytest.mark.skipif(not (DATA / "eval" / "teacher_8x1024.npz").exists(), reason="public Lucebox eval files not downloaded")
def test_public_teacher_file_matches_its_reported_numbers():
    tokens, idx, lp, nll = le.load_teacher(str(DATA))
    assert tokens.shape == (8, 1024) and idx.shape == (8184, 256) and (np.diff(lp, axis=1) <= 0).all()
    c = le.teacher_consistency(idx, lp, le.next_tokens(tokens), nll)
    t = json.load(open(DATA / "eval" / "teacher_kl.json"))
    assert c["ppl_from_nll"] == pytest.approx(t["ppl"]) and nll == pytest.approx(t["nll"]) and t["tokens"] == 8184
    assert 0.97 < c["next_token_in_topk"] < 0.98 and c["teacher_argmax_is_next"] > 0.5
    assert c["implied_mean_miss_nll"] >= c["miss_nll_lower_bound"]        # the stored nll covers the misses consistently
    for f in (DATA / "results").glob("*_kl.json"):
        d = json.load(open(f))
        assert np.exp(d["nll"]) == pytest.approx(d["ppl"]) and d["teacher_nll"] == pytest.approx(nll)
    tab = le.table(str(DATA), [])
    assert "ROCMFP2S-MIX10 | 0.244 | 78.8% | 9.59" in tab and "| 0.265 | 78.6% | 9.77" in tab


# ---- end to end on tiny models -------------------------------------------------------------------------------------

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
pytest.importorskip("tokenizers")
import ds41_engine as eng  # noqa: E402
from tiny_ckpt import fw_config, write_tiny  # noqa: E402


def _ckpt(root, seed):
    d = root / f"tiny{seed}"
    d.mkdir()
    write_tiny(str(d), seed=seed, cfg=fw_config(index_n_heads=16, dspark_block_size=5, dspark_noise_token_id=511))
    from tokenizers import Tokenizer, models, pre_tokenizers
    vocab = Tokenizer.from_file(str(d / "tokenizer.json")).get_vocab()
    tk = Tokenizer(models.WordLevel(vocab, unk_token="the"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.save(str(d / "tokenizer.json"))
    return str(d)


class Backend(le.EngineBackend):
    def __init__(self, path):
        self.engine = eng.load_model(path, 256, loader="ds41_model:load_model")


def test_end_to_end_tiny(tmp_path):
    teacher_pack, student_pack = _ckpt(tmp_path, 4), _ckpt(tmp_path, 5)
    rng = np.random.default_rng(0)
    R, L, K = 2, 24, 64
    tokens = rng.integers(0, 512, (R, L)).astype(np.int32)
    ref, stu = Backend(teacher_pack), Backend(student_pack)
    # synthetic teacher in the Lucebox format: full-vocabulary scores of the teacher model, top-K ids (best first)
    idx = np.zeros((R * (L - 1), K), np.int32); lp = np.zeros((R * (L - 1), K), np.float32); nll = []
    for s in range(R):
        full = [list(range(512))] * (L - 1)
        pos = ref.score(tokens[s, :1].tolist(), tokens[s, 1:].tolist(), full)
        for p, d in enumerate(pos):
            v = np.array(d["cand_logps"])
            top = np.argsort(-v, kind="stable")[:K]
            idx[s * (L - 1) + p], lp[s * (L - 1) + p] = top, v[top]
            nll.append(-v[tokens[s, p + 1]])
    ev = tmp_path / "data" / "eval"
    ev.mkdir(parents=True)
    np.save(ev / "heldout_8x1024.npy", tokens)
    np.savez(ev / "teacher_8x1024.npz", idx=idx, lp=lp, nll=np.float64(np.mean(nll)), tokens=np.int64(len(nll)))
    data = str(tmp_path / "data")

    same = le.evaluate(ref, data, "same", log=lambda *_: None)
    assert same["kl_top256"] == pytest.approx(0, abs=1e-5) and same["top1_agree"] == 1.0
    assert same["ppl"] == pytest.approx(np.exp(np.mean(nll)), rel=1e-5) and same["comparable"]

    other = le.evaluate(stu, data, "other", log=lambda *_: None, cache_dir=str(tmp_path / "cache"))
    assert other["kl_top256"] > 1e-4 and other["tokens"] == R * (L - 1)
    # independent recomputation: full student distribution at every position
    tgt = le.next_tokens(tokens.astype(np.int64))
    kl, t1, nn = [], [], []
    for s in range(R):
        pos = stu.score(tokens[s, :1].tolist(), tokens[s, 1:].tolist(), [list(range(512))] * (L - 1))
        for p, d in enumerate(pos):
            q = np.array(d["cand_logps"]); i = s * (L - 1) + p
            pt = np.exp(lp[i].astype(np.float64))
            kl.append((pt * (lp[i].astype(np.float64) - q[idx[i]])).sum())
            t1.append(int(q.argmax()) == idx[i, 0]); nn.append(-q[tgt[i]])
    assert other["kl_top256"] == pytest.approx(np.mean(kl), rel=1e-5)
    assert other["top1_agree"] == pytest.approx(np.mean(t1)) and other["nll"] == pytest.approx(np.mean(nn), rel=1e-5)
    again = le.evaluate(stu, data, "other", log=lambda *_: None, cache_dir=str(tmp_path / "cache"))   # served from cache
    assert again["kl_top256"] == other["kl_top256"]
    part = le.evaluate(stu, data, "part", rows=[1], max_len=5, log=lambda *_: None)
    assert part["comparable"] is False and part["tokens"] == 5
