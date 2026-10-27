"""CPU tests of tools/qualify (planner, rules, search, pack metadata, report). Run: cd tools/qualify && python3 -m unittest discover -s tests"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import pack as packmod  # noqa: E402
import plan as planmod  # noqa: E402
import report  # noqa: E402
import rules  # noqa: E402
import search  # noqa: E402
import stages  # noqa: E402


def fake_pack(d: Path, arch="LlamaForCausalLM", mtp=0, template=True, quant="exl3", extra=None):
    cfg = {"architectures": [arch], "model_type": "llama", "num_hidden_layers": 4, "num_attention_heads": 8, "num_key_value_heads": 2,
           "head_dim": 16, "hidden_size": 128, "max_position_embeddings": 65536, "vocab_size": 100, **(extra or {})}
    if mtp:
        cfg["mtp_num_hidden_layers"] = mtp
    (d / "config.json").write_text(json.dumps(cfg))
    (d / "quantization_config.json").write_text(json.dumps({"quant_method": quant, "bits": 4.0, "head_bits": 6}))
    (d / "model-00001-of-00001.safetensors").write_bytes(b"x" * 100)
    (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "model-00001-of-00001.safetensors"}}))
    if template:
        (d / "chat_template.jinja").write_text("{{ messages }}")
    return d


def args(**kw):
    a = dict(backend="cuda", sessions=[1, 4], ref=None, kv_kl_max=0.02, kv_bits=[8, 6, 4], smoke_tasks="", ctx_cap=None, out="/tmp/x", seq=None)
    a.update(kw)
    return a


class PackTests(unittest.TestCase):
    def test_dense_llama_flags_come_from_the_pack(self):
        with tempfile.TemporaryDirectory() as t:
            m = packmod.read_pack(fake_pack(Path(t)))
            self.assertEqual((m["family"], m["kv_layers"], m["kv_heads"], m["head_dim"]), ("dense_llama", 4, 2, 16))
            f = packmod.serve_flags(m, port=1, ctx=4096, sessions=2, k_bits=6, v_bits=6)
            self.assertIn("--no-vision", f)
            self.assertEqual(f[f.index("--draft-policy") + 1], "off")      # no MTP head
            self.assertEqual(f[f.index("--chat-template") + 1], str(Path(t) / "chat_template.jinja"))
            self.assertEqual(m["problems"], [])

    def test_mtp_head_selects_mix(self):
        with tempfile.TemporaryDirectory() as t:
            m = packmod.read_pack(fake_pack(Path(t), mtp=1))
            self.assertEqual(packmod.draft_policy(m), "mix")

    def test_missing_template_and_wrong_quant_are_reported(self):
        with tempfile.TemporaryDirectory() as t:
            m = packmod.read_pack(fake_pack(Path(t), template=False, quant="gptq"))
            self.assertEqual(len(m["problems"]), 2)

    def test_hybrid_gdn_counts_only_full_attention_layers(self):
        t = {"num_hidden_layers": 64, "full_attention_interval": 4}
        self.assertEqual(packmod.kv_layers(t), 16)
        self.assertEqual(packmod.kv_layers({"num_hidden_layers": 4, "layer_types": ["linear_attention", "full_attention"] * 2}), 2)

    def test_kv_bytes_per_token_matches_drill0_measurements(self):
        # nanb: Q4 ~57 KB/token (52 layers x 8 x 128); realA: Qwen27 K4V4 18.6 KB/token (16 layers x 4 x 256)
        nanb = dict(kv_layers=52, kv_heads=8, head_dim=128)
        q27 = dict(kv_layers=16, kv_heads=4, head_dim=256)
        self.assertAlmostEqual(packmod.kv_bytes_per_token(nanb, 4, 4) / 1024, 57, delta=4)
        self.assertAlmostEqual(packmod.kv_bytes_per_token(q27, 4, 4) / 1024, 18.6, delta=1.5)

    def test_verify_shards(self):
        with tempfile.TemporaryDirectory() as t:
            d = fake_pack(Path(t))
            self.assertEqual(packmod.verify_shards(d), [])
            (d / "model-00001-of-00001.safetensors").unlink()
            self.assertEqual(packmod.verify_shards(d), ["missing shard model-00001-of-00001.safetensors"])


class PlanTests(unittest.TestCase):
    def meta(self, **kw):
        with tempfile.TemporaryDirectory() as t:
            m = packmod.read_pack(fake_pack(Path(t), **kw))
        return m

    def names(self, plan, status):
        return [s.name for s in plan if s.status == status]

    def test_full_plan_without_ref_skips_parity_and_kv_with_reason(self):
        p = planmod.build_plan(self.meta(), args(), {})
        self.assertEqual([s.name for s in p], planmod.ORDER)
        self.assertEqual(self.names(p, "skip"), ["parity", "kv"])
        self.assertIn("k6v6", next(s for s in p if s.name == "kv").note)

    def test_ref_enables_parity_for_a_family_with_an_adapter(self):
        p = planmod.build_plan(self.meta(), args(ref="/r"), {})
        self.assertEqual(self.names(p, "skip"), [])

    def test_family_without_adapter_is_unmeasured_not_fatal(self):
        p = planmod.build_plan(self.meta(arch="Qwen3_5ForCausalLM", extra={"full_attention_interval": 4}), args(ref="/r"), {})
        self.assertIn("no reference adapter", next(s for s in p if s.name == "parity").note)

    def test_resume_skips_done_stage_with_same_key_only(self):
        m = self.meta()
        key = planmod.fingerprint("maxctx", m, args())
        st = {"maxctx": dict(status="done", key=key, wall_s=12.0, data={})}
        self.assertEqual(self.names(planmod.build_plan(m, args(), st), "done"), ["maxctx"])
        self.assertEqual(self.names(planmod.build_plan(m, args(sessions=[1]), st), "done"), [])      # changed input -> stale
        km = dict(status="done", key=planmod.fingerprint("meta", m, args()), wall_s=0, data={})
        self.assertEqual(self.names(planmod.build_plan(m, args(sessions=[1]), {"meta": km}), "done"), ["meta"])   # unrelated arg keeps meta
        self.assertEqual(self.names(planmod.build_plan(m, args(), st, redo=["maxctx"]), "done"), [])

    def test_stage_selector_blocks_missing_dependency(self):
        p = planmod.build_plan(self.meta(), args(), {}, only=["serve"])
        self.assertEqual(self.names(p, "blocked"), ["serve"])
        p = planmod.build_plan(self.meta(), args(), {}, only=["serve"], force_deps=True)
        self.assertEqual(self.names(p, "blocked"), [])
        self.assertIn("maxctx", self.names(p, "todo"))

    def test_stage_selector_runs_cpu_stages_alone(self):
        p = planmod.build_plan(self.meta(), args(), {}, only=["fetch", "meta"])
        self.assertEqual(self.names(p, "todo"), ["fetch", "meta"])
        self.assertEqual(planmod.estimate(p)["gpu_s"], 0)

    def test_state_roundtrip(self):
        with tempfile.TemporaryDirectory() as t:
            planmod.save_stage(Path(t), "meta", "k", "done", 1.23, {"a": 1})
            self.assertEqual(planmod.load_state(Path(t))["meta"]["data"], {"a": 1})


class RuleTests(unittest.TestCase):
    def test_kv_choice_follows_drill0_numbers(self):
        self.assertEqual(rules.choose_kv({6: 0.014, 4: 0.123}), (6, 6))          # Nanbeige: k6v6 vs k8v8 KL .014 -> Q6
        self.assertEqual(rules.choose_kv({6: 0.05}), (8, 8))                     # k6v6 too lossy -> k8v8
        self.assertEqual(rules.choose_kv({6: 0.05, 7: 0.01}), (7, 7))
        self.assertEqual(rules.choose_kv({4: 0.001, 3: 0.002}), (8, 8))          # never below Q6 even if Q4 looks fine
        self.assertEqual(rules.choose_kv({}), (8, 8))

    def good(self):
        return dict(parity=dict(kl_mean=0.086, by_domain={"wiki": dict(top1=0.79)}), kv=dict(chosen=[6, 6]), maxctx={"1": 77824, "4": 40960},
                    serve=dict(sha_self_equal=True, oom=False, conformance=dict(ok=True)), smoke=dict(passed=4, total=4))

    def test_verdict(self):
        self.assertTrue(rules.verdict(self.good())["pass_"])
        q = self.good(); q["parity"]["by_domain"]["wiki"]["top1"] = 0.5
        self.assertIn("wiki", rules.verdict(q)["reasons"][0])
        q = self.good(); q["serve"]["sha_self_equal"] = False
        self.assertFalse(rules.verdict(q)["pass_"])
        q = self.good(); q["smoke"] = dict(passed=1, total=4)
        self.assertFalse(rules.verdict(q)["pass_"])

    def test_unmeasured_is_never_a_pass(self):
        q = self.good(); q["parity"] = None
        v = rules.verdict(q)
        self.assertFalse(v["pass_"]); self.assertEqual(v["unmeasured"], ["parity"])


class SearchTests(unittest.TestCase):
    def run_search(self, limit, lo=8192, hi=131072):
        calls = []

        def probe(c):
            calls.append(c)
            return c <= limit, "x"
        return search.bisect_max(lo, hi, probe), calls

    def test_finds_grid_value_below_limit(self):
        r, calls = self.run_search(77_900)
        self.assertEqual(r["best"], 77_824)
        self.assertLessEqual(len(calls), 8)
        self.assertTrue(all(c % search.STEP == 0 for c in calls))

    def test_upper_bound_when_everything_fits(self):
        r, calls = self.run_search(10**9)
        self.assertEqual((r["best"], calls), (131072, [131072]))

    def test_nothing_fits(self):
        r, _ = self.run_search(1000)
        self.assertIsNone(r["best"])

    def test_upper_bound_from_memory(self):
        ub = search.upper_bound(free_bytes=6.0e9, bytes_per_token=57_000, headroom_bytes=2.5 * 2**30, cap=524288)
        self.assertEqual(ub % search.STEP, 0)
        self.assertEqual(search.upper_bound(1e12, 1000, 0, 65536), 65536)


class ParityTests(unittest.TestCase):
    def test_sqnr_parse_and_summary(self):
        txt = " -- Quantized: model.layers.0                  bpw:  4.00  rfn: 0.025822  cos: 0.000324  sqnr: 34.195930  [109.56 s]  rss: 19.00 GB\n" \
              " -- Quantized: model.layers.1.mlp.down_proj    bpw:  4.00  proxy_err: 0.001\n" \
              " -- Quantized: model.layers.1                  bpw:  4.00  rfn: 0.02  cos: 0.0003  sqnr: 38.5  [1 s]\n"
        per = stages.parse_sqnr(txt)
        self.assertEqual(per, {0: 34.19593, 1: 38.5})
        self.assertEqual(stages.sqnr_summary(per)["worst_layer"], 0)
        self.assertFalse(stages.sqnr_summary({})["measured"])

    def test_aggregate_is_token_weighted(self):
        r = stages.aggregate_parity({"a": dict(kl=0.1, top1=0.9, dnll=0.0, tokens=100), "b": dict(kl=0.3, top1=0.7, dnll=0.1, tokens=300)})
        self.assertAlmostEqual(r["kl_mean"], 0.25)


class ReportTests(unittest.TestCase):
    def test_assemble_and_render(self):
        with tempfile.TemporaryDirectory() as t:
            m = packmod.read_pack(fake_pack(Path(t)))
        m["draft_policy"] = "off"
        done = lambda d: dict(status="done", wall_s=1.0, data=d)
        stg = {"maxctx": done({"1": 77824, "4": 40960, "tried_s1": []}),
               "kv": done(dict(chosen=[6, 6], kl_max=0.02, rule="r", table={"8": dict(kl_ref=0.04, top1_ref=0.9), "6": dict(kl_ref=0.05, top1_ref=0.89, kl_vs_k8=0.014)})),
               "serve": done(dict(cmd=["py", "serve.py"], sha_self_equal=True, oom=False, conformance=dict(ok=True), decode_tps_median=36.7, prefill_8k_tps=1380.0,
                                  curve=[dict(prompt=2048, ttft=1.5, prefill_tps=1365.0, dec_tps=37.0, err=None)], agg=[dict(n=1, agg_dec_tps=36.0, dec_tps=[36.0], ttft=[5.8])],
                                  sha=[dict(sha="abc")])),
               "smoke": done(dict(kind="scripted_agent", total=2, passed=2))}
        j = report.assemble(m, args(), stg, HERE)
        self.assertEqual(j["serve"]["ctx"], {"1": 77824, "4": 40960})
        self.assertEqual(j["serve"]["kv"], [6, 6])
        self.assertEqual(j["verdict"]["unmeasured"], ["parity"])
        md = report.render_md(j)
        self.assertIn("max -c", md); self.assertIn("36.7", md)
        json.dumps(j, default=str)


class CliTests(unittest.TestCase):
    def test_dry_run_touches_nothing_and_prints_plan(self):
        import io, contextlib, qualify
        with tempfile.TemporaryDirectory() as t:
            pk = fake_pack(Path(t) / "p") if (Path(t) / "p").mkdir() is None else None
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = qualify.main(["--backend", "cuda", "--pack", str(pk), "--dry-run", "--out", str(Path(t) / "o")])
            self.assertEqual(rc, 0)
            self.assertIn("serve command", buf.getvalue())
            self.assertEqual([p.name for p in (Path(t) / "o").iterdir()], [])



class KvModeRule(unittest.TestCase):
    BPV = {"k6v6": 6.5, "kw6": 6.5, "tq6": 6.0625, "tq5": 5.0625}
    TQ = ("k6v6", "kw6", "tq6", "tq5")

    def test_choose_kv_mode(self):
        bpv = self.BPV
        # tq opt-in (default): tq6 / tq5 never win, kw6 does when it beats k6v6 by the margin
        self.assertEqual(rules.choose_kv_mode({"k6v6": 0.0536, "kw6": 0.0216, "tq6": 0.0319, "tq5": 0.0544}, bpv), "kw6")      # Nanbeige 32K p1
        self.assertEqual(rules.choose_kv_mode({"k6v6": 0.010, "kw6": 0.0095, "tq6": 0.004, "tq5": 0.009}, bpv), "k6v6")       # kw6 inside the noise
        self.assertEqual(rules.choose_kv_mode({"k6v6": 0.002}, {"k6v6": 6.5}), "k6v6")                                         # nothing else measured
        # with tq allowed (QUALIFY_KV_TQ=1): the old rule, fewest bytes among the qualifying formats
        kw = dict(allowed=self.TQ)
        self.assertEqual(rules.choose_kv_mode({"k6v6": 0.010, "kw6": 0.005, "tq6": 0.004, "tq5": 0.009}, bpv, **kw), "tq5")
        self.assertEqual(rules.choose_kv_mode({"k6v6": 0.010, "kw6": 0.005, "tq6": 0.008, "tq5": 0.020}, bpv, **kw), "tq6")
        self.assertEqual(rules.choose_kv_mode({"k6v6": 0.005, "kw6": 0.006, "tq6": 0.0125, "tq5": 0.0048}, bpv, **kw), "k6v6")
        self.assertEqual(rules.choose_kv_mode({"k6v6": 0.002, "kw6": 0.0021, "tq6": 0.004, "tq5": 0.010}, bpv, **kw), "k6v6")

    def test_job_kvmode_glue(self):
        """kv stage -> job_kvmode -> kv_cache.json: the winner is written as the lane preset, serve flags follow it"""
        import contextlib
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            out = t / "out"; out.mkdir(); (t / "pack").mkdir()
            raw = {"tq6": dict(kl_vs_k8=0.004, top1_vs_k8=0.99, bits_per_val=6.0625), "tq5": dict(kl_vs_k8=0.020, top1_vs_k8=0.97, bits_per_val=5.0625),
                   "kw6": dict(kl_vs_k8=0.006, top1_vs_k8=0.99, bits_per_val=6.5), "k6v6": dict(kl_vs_k8=0.010, top1_vs_k8=0.98, bits_per_val=6.5)}
            def py(script, *a, **kw):
                self.assertEqual((script, a[0]), ("gpu_eval.py", "kvmode"))
                (out / "kvmode_raw.json").write_text(json.dumps({k: v for k, v in raw.items()}))
            c = SimpleNamespace(out=out, args=dict(kvmode_steps=64), meta=dict(path=str(t / "pack"), index_sha="x", serve_entry="tools/qwen/serve.py", vision=False,
                                template=None, mtp_layers=0), py=py, locks=contextlib.nullcontext, repo=t, data={})
            stages._git = lambda repo: "deadbeef"
            # default: tq stays opt-in, kw6 wins (6 <= 0.9 x 10)
            kv = stages.job_kvmode(c, out / "docs.pt")
            self.assertEqual((kv["mode"], kv["chosen"], kv["kvwin"]), ("quant", [6, 6], "4:128"))
            stages.write_kv_choice(c, kv)
            j = json.loads((t / "pack" / "kv_cache.json").read_text())
            self.assertEqual((j["kv_mode"], j["k_bits"], j["v_bits"], j["kvwin"]), ("quant", 6, 6, "4:128"))
            c.data = dict(kv=kv)
            self.assertEqual((stages.chosen_kv_mode(c), stages.chosen_kvwin(c)), ("quant", "4:128"))
            f = packmod.serve_flags(c.meta, port=1, ctx=4096, sessions=1, k_bits=6, v_bits=6, kvwin=stages.chosen_kvwin(c))
            self.assertEqual(f[f.index("--kvwin") + 1], "4:128")
            # QUALIFY_KV_TQ=1: tq6 wins the same table (fewer bytes than kw6; tq5 fails the margin)
            os.environ["QUALIFY_KV_TQ"] = "1"
            try:
                kv = stages.job_kvmode(c, out / "docs.pt")
            finally:
                del os.environ["QUALIFY_KV_TQ"]
            self.assertEqual((kv["mode"], kv["chosen"]), ("tq", [6, 6]))
            stages.write_kv_choice(c, kv)
            self.assertEqual(json.loads((t / "pack" / "kv_cache.json").read_text())["kv_mode"], "tq")
            c.data = dict(kv=kv)
            self.assertEqual(stages.chosen_kv_mode(c), "tq")

    def test_serve_flags_kv_mode(self):
        meta = dict(serve_entry="tools/qwen/serve.py", path="/p", vision=False, template=None, mtp_layers=0, kv_layers=16, kv_heads=4, head_dim=256)
        f = packmod.serve_flags(meta, port=1, ctx=4096, sessions=1, k_bits=5, v_bits=5, kv_mode="tq")
        self.assertEqual(f[f.index("--kv-mode") + 1], "tq")
        self.assertNotIn("--kv-mode", packmod.serve_flags(meta, port=1, ctx=4096, sessions=1, k_bits=6, v_bits=6))
        self.assertLess(packmod.kv_bytes_per_token(meta, 5, 5, "tq"), packmod.kv_bytes_per_token(meta, 6, 6))


if __name__ == "__main__":
    unittest.main()
