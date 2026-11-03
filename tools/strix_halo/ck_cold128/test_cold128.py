"""Focused stdlib checks; no Kyojin/framework/device imports."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parent


def load():
    spec = importlib.util.spec_from_file_location('cold128_public', ROOT / 'cold128.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PublicPlan(unittest.TestCase):
    def test_parameterized_inert_plan_preserves_measured_settings(self):
        module = load()
        args = module.parser().parse_args([
            '--source', '/example/kyojin', '--native', '/example/native/ext.so',
            '--ck', '/example/ck/lib.so', '--model', '/example/model',
            '--python', '/example/venv/bin/python', '--prompts', '/example/prompts',
            '--out', '/example/results'])
        plan = module.make_plan(args)
        self.assertEqual(plan['status'], 'PLANNED_NOT_EXECUTED')
        self.assertEqual(plan['cache_bits'], [16, 16])
        self.assertEqual(plan['capacity'], 163840)
        self.assertEqual(plan['chunk_size'], 4096)
        self.assertEqual(plan['prompt_tokens'], 131079)
        self.assertFalse(plan['drafter'])
        self.assertEqual(len(plan['brackets']), 4)
        for bracket in plan['brackets']:
            self.assertEqual([leg['arm'] for leg in bracket], ['control', 'candidate', 'control2'])
            self.assertEqual(len({leg['prompt'] for leg in bracket}), 1)
        self.assertEqual(plan['source'], '/example/kyojin')
        self.assertEqual(plan['control_backend'], 'Stock Triton; not historical dense CK')


class PromptPreparation(unittest.TestCase):
    def test_reexec_preserves_normalized_paths_after_home_changes(self):
        for spelling in ('model-tilde', 'all-tilde', 'absolute'):
            with self.subTest(spelling=spelling), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                owned = Path(directory)
                home = owned / 'caller-home'
                paths = dict(source=home / 'source', native=home / 'native/ext.so',
                             ck=home / 'ck/lib.so', model=home / 'model',
                             python=home / 'venv/bin/python', prompts=home / 'prompts',
                             out=home / 'results')
                # Only file-presence checks use these inert inputs; none is imported.
                for path in (paths['native'], paths['ck'], paths['model'] / 'config.json',
                             paths['source'] / 'tools/mimo/serve.py',
                             paths['source'] / 'tools/mimo/chat_template.jinja'):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text('inert fixture, not an engine or model\n')
                paths['python'].parent.mkdir(parents=True)
                paths['python'].symlink_to(sys.executable)
                # Honor the real script-directory/output-disjointness guard. Load
                # identical production bytes from a disjoint, owned fixture dir.
                script = owned / 'helper/cold128.py'
                script.parent.mkdir()
                script.write_bytes((ROOT / 'cold128.py').read_bytes())
                spec = importlib.util.spec_from_file_location('cold128_prep_fixture', script)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                cli = ['--make-prompts', '--port', '23456', '--seed', '987654']
                for key, path in paths.items():
                    value = str(path)
                    if spelling == 'all-tilde' or (spelling == 'model-tilde' and key == 'model'):
                        value = '~/' + str(path.relative_to(home))
                    cli += ['--' + key, value]
                # Only execve and the tokenizer entry are inert. Both passes use
                # the actual main/runtime/normalized/parser and real filesystem.
                with mock.patch.object(module, 'make_prompts', return_value=0) as prepared, \
                     mock.patch.object(os, 'execve', side_effect=SystemExit('inert execve')) as reexec:
                    with mock.patch.dict(os.environ, {'HOME': str(home)}, clear=True), \
                         mock.patch.object(sys, 'argv', [str(script), *cli]):
                        with self.assertRaisesRegex(SystemExit, 'inert execve'):
                            module.main()
                    prepared.assert_not_called()
                    reexec.assert_called_once()
                    executable, argv, environment = reexec.call_args.args
                    self.assertEqual(executable, str(paths['python']))
                    self.assertEqual(argv[:5], [str(paths['python']), '-I', '-B', '-u', str(script)])
                    self.assertNotEqual(environment['HOME'], str(home))
                    self.assertTrue(Path(environment['HOME']).is_dir())
                    self.assertEqual(environment['PUBLIC_COLD128_PROMPT_ENV'], '1')
                    with mock.patch.dict(os.environ, environment, clear=True), \
                         mock.patch.object(sys, 'argv', argv[4:]):
                        try:
                            result = module.main()
                        except ValueError as exc:
                            self.fail('prompt re-exec changed an input after HOME switched: ' + str(exc))
                    self.assertEqual(result, 0)
                    reexec.assert_called_once()
                    prepared.assert_called_once()
                    replayed = prepared.call_args.args[0]
                    for key, path in paths.items():
                        self.assertEqual(getattr(replayed, key), path)
                    self.assertTrue(replayed.python.is_symlink())
                    self.assertNotEqual(replayed.python, replayed.python.resolve())
                    self.assertTrue(replayed.make_prompts)
                    self.assertFalse(replayed.execute)
                    self.assertIsNone(replayed.worker)
                    self.assertEqual(replayed.port, 23456)
                    self.assertEqual(replayed.seed, 987654)


class PromptFitting(unittest.TestCase):
    def test_exact_rendered_length_and_distinct_prefix(self):
        module = load()
        # A synthetic character tokenizer makes the expected length independent.
        # Real prompt creation later uses the actual engine tokenizer and template.
        for index in range(5):
            text = module.fit_prompt(lambda value: len(value),
                                     lambda value: 'USER:' + value + ':ASSISTANT',
                                     'case-' + str(index), 1234 + index, target=2000)
            self.assertEqual(len('USER:' + text + ':ASSISTANT'), 2000)
            self.assertIn('case-' + str(index), text[:80])
            self.assertTrue(text.endswith('Summarize the experimental record.'))


class NativeResponse(unittest.TestCase):
    def test_native_seconds_and_reasoning_fallback(self):
        module = load()
        row = dict(prompt_sha256='rendered', error=None,
                   native=dict(prompt_tokens=131079, cached_tokens=0, new_tokens=64,
                               accepted_draft_tokens=0, rejected_draft_tokens=0,
                               time_prefill=200.0))
        response = dict(timings=dict(cache_n=0, prompt_n=131079, predicted_n=64,
                                    prompt_ms=200000.0, prompt_per_second=655.395),
                        choices=[dict(message=dict(content=' \n', reasoning_content='valid reasoning'))])
        self.assertEqual(module.checked_request(row, response, 'rendered')['prefill_s'], 200.0)
        self.assertEqual(module.reply_source(response), 'reasoning_content')
        response['choices'][0]['message']['content'] = 'actual answer'
        self.assertEqual(module.reply_source(response), 'content')
        response['timings']['prompt_ms'] = 200.0
        with self.assertRaisesRegex(ValueError, 'API/native timing mismatch'):
            module.checked_request(row, response, 'rendered')


class OwnedTermination(unittest.TestCase):
    def test_sigterm_reaps_exact_detached_child(self):
        # Exercise real subprocesses, not a mocked terminate() callback. The
        # child/marker and output are confined to this test's owned scratch.
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            owned = Path(directory)
            probe = owned / 'probe.py'
            probe.write_text(
                'import importlib.util, pathlib, subprocess, sys, time\n'
                's=importlib.util.spec_from_file_location("p",sys.argv[1])\n'
                'm=importlib.util.module_from_spec(s);s.loader.exec_module(m)\n'
                'child=subprocess.Popen([sys.executable,"-I","-S","-c",'
                '"import time; time.sleep(60)"],start_new_session=True)\n'
                'try:\n'
                ' with m.parent_signals():\n'
                '  pathlib.Path(sys.argv[2]).write_text(str(child.pid))\n'
                '  while True:time.sleep(0.05)\n'
                'finally:\n'
                ' m.stop_owned(child)\n')
            log_path = owned / 'log.txt'
            with log_path.open('wb') as log:
                parent = subprocess.Popen([sys.executable, '-I', '-S', '-B', str(probe),
                                           str(ROOT / 'cold128.py'), str(owned / 'pid')],
                                          stdout=log, stderr=subprocess.STDOUT)
                child_pid = None
                try:
                    for _ in range(200):
                        if (owned / 'pid').exists():
                            child_pid = int((owned / 'pid').read_text())
                            break
                        if parent.poll() is not None:
                            break
                        time.sleep(0.01)
                    self.assertIsNotNone(child_pid, log_path.read_text())
                    parent.terminate()
                    self.assertNotEqual(parent.wait(timeout=10), 0)
                    self.assertFalse(Path('/proc', str(child_pid)).exists(), log_path.read_text())
                finally:
                    if parent.poll() is None:
                        parent.kill()
                    parent.wait()
                    # Test cleanup only, with this exact test-created child ID.
                    if child_pid is not None and Path('/proc', str(child_pid)).exists():
                        os = __import__('os')
                        os.kill(child_pid, 15)


class TreatmentEnvironment(unittest.TestCase):
    def test_worker_treatment_is_import_time_and_cache_private(self):
        module = load()
        a = module.parser().parse_args([
            '--source', '/example/kyojin', '--native', '/example/native/ext.so',
            '--ck', '/example/ck/lib.so', '--model', '/example/model',
            '--python', '/example/venv/bin/python', '--prompts', '/example/prompts',
            '--out', '/example/results'])
        c = module.arm_env(a, 'control', Path('/example/c'))
        t = module.arm_env(a, 'candidate', Path('/example/t'))
        self.assertNotIn('EXL3_CK_PAGEDKV', c)
        self.assertEqual(t['EXL3_CK_PAGEDKV'], 'sparse')
        self.assertEqual(t['EXL3_CK_SPARSE_RETAIN'], '0.3')
        self.assertEqual(t['EXL3_PREFILL_CHUNK'], '4096')
        self.assertNotEqual(c['TRITON_CACHE_DIR'], t['TRITON_CACHE_DIR'])
        self.assertEqual(t['PYTORCH_ROCM_ARCH'], 'gfx1151')
        self.assertNotIn('PYTHONPATH', t)
        self.assertAlmostEqual(module.summarize_bracket([200.0, 100.0, 202.0])['gain_percent'],
                               101.0, places=12)


class RuntimeReceipt(unittest.TestCase):
    def test_real_owned_child_and_two_http_fixture_receipts(self):
        # Full run_leg producer/consumer seam with an actual tiny subprocess;
        # only the model and HTTP response producer are inert fixtures.
        module = load()
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            owned = Path(directory)
            prompt = owned / 'prompt.txt'
            prompt.write_text('synthetic test fixture, never a model result')
            args = module.parser().parse_args([
                '--source', str(owned / 'source'), '--native', str(owned / 'native/ext.so'),
                '--ck', str(owned / 'ck/lib.so'), '--model', str(owned / 'model'),
                '--python', sys.executable, '--prompts', str(owned / 'prompts'),
                '--out', str(owned / 'out')])
            args.out.mkdir()
            leg = dict(tag='fixture-control', arm='control', pair=0, prompt=str(prompt))
            script = owned / 'worker.py'
            script.write_text(
                'import json,os,pathlib,signal,sys,time\n'
                'out=pathlib.Path(sys.argv[1]); prompt=sys.argv[2]\n'
                'ready=dict(pid=os.getpid(),rendered_prompt_hashes={prompt:"rendered"})\n'
                '(out/"ready.json").write_text(json.dumps(ready))\n'
                'signal.signal(signal.SIGTERM,lambda *_:sys.exit(0))\n'
                'while True:time.sleep(.01)\n')
            module.worker_argv = lambda a, index: [sys.executable, '-I', '-S', '-B',
                str(script), str(args.out / leg['tag']), str(prompt)]
            module.arm_env = lambda a, arm, out: dict(PATH='/usr/bin:/bin', LC_ALL='C')
            module.lane_state = lambda: dict(compute_owners=[], busy={'inert_gpu':'0'})
            module.policy = lambda: {'inert_policy':'fixed'}
            module.port_open = lambda a: (args.out / leg['tag'] / 'ready.json').exists()
            module.request = lambda a, path: b'{"generator":true}'
            module.make_plan = lambda a: {'warmup': str(prompt)}
            count = [0]
            def produced(a, child, body, tag):
                count[0] += 1
                row = dict(prompt_sha256='rendered', error=None, counters_before={}, counters_after={},
                           completion='synthetic fixture only',
                           native=dict(prompt_tokens=131079, cached_tokens=0, new_tokens=64,
                                       accepted_draft_tokens=0, rejected_draft_tokens=0,
                                       time_prefill=200.0, time_generate=2.0))
                with (args.out / leg['tag'] / 'requests.jsonl').open('a') as stream:
                    stream.write(__import__('json').dumps(row) + '\n')
                response = dict(timings=dict(cache_n=0, prompt_n=131079, predicted_n=64,
                                            prompt_ms=200000.0, prompt_per_second=655.395),
                                choices=[dict(message=dict(content='', reasoning_content='fixture'))])
                return __import__('json').dumps(response).encode()
            module.monitored_request = produced
            # The current process's read-only maps are not a model map. Select a
            # nonexistent CK pathname so the stock-control mapping assertion holds.
            record = None
            original_stop = module.stop_owned
            def stop(child):
                code = original_stop(child)
                (args.out / leg['tag'] / 'ready.json').unlink()
                return code
            module.stop_owned = stop
            with module.parent_signals():
                record = module.run_leg(args, 0, leg, {'inert_policy':'fixed'})
            self.assertEqual(count[0], 2)
            self.assertEqual(record['server_returncode'], 0)
            self.assertTrue(record['port_closed'])
            self.assertNotIn('error', record)
            self.assertFalse(Path('/proc', str(record['pid'])).exists())
            self.assertEqual(record['measured']['native_score']['prefill_s'], 200.0)
            self.assertTrue((args.out / leg['tag'] / 'result.json').is_file())
            self.assertTrue((args.out / leg['tag'] / 'measured-response.json').is_file())


if __name__ == '__main__':
    unittest.main(verbosity=2)
