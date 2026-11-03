#!/usr/bin/env python3
"""Derived public cold128K recipe, not the literal measured private runner.

Default: print a plan; no framework imports. --make-prompts uses the selected
checkout's tokenizer/template but never loads model weights. --execute runs four
fresh-server control/candidate/control2 brackets on previously generated prompts.
Only the selected --python interpreter may perform either explicit operation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


CAPACITY = 163840
CHUNK = 4096
TOKENS = 131079
BUDGET = 64


def need(ok, message):
    if not ok:
        raise ValueError(message)


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--make-prompts', action='store_true')
    modes.add_argument('--execute', action='store_true')
    for key in ('source', 'native', 'ck', 'model', 'python', 'prompts', 'out'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--port', type=int, default=18748)
    p.add_argument('--seed', type=int, default=1282026)
    p.add_argument('--worker', type=int, choices=range(12), help=argparse.SUPPRESS)
    return p


def make_plan(a):
    need(1024 <= a.port <= 65535, 'invalid port')
    def path(name):
        return str(getattr(a, name).expanduser().absolute())
    brackets = []
    for pair in range(4):
        brackets.append([dict(arm=arm, pair=pair,
                              prompt=str(Path(path('prompts')) / f'measured-{pair}.txt'),
                              tag=f'pair{pair}-{arm}')
                         for arm in ('control', 'candidate', 'control2')])
    return dict(status='PLANNED_NOT_EXECUTED', cache_bits=[16, 16], capacity=CAPACITY,
                chunk_size=CHUNK, prompt_tokens=TOKENS, generated_token_budget=BUDGET,
                retention=0.3, drafter=False, reasoning_effort='none',
                thinking='Template none omits medium effort instruction; not thinking-disabled',
                source=path('source'), native=path('native'), ck=path('ck'), model=path('model'),
                python=path('python'), prompts=path('prompts'), output=path('out'), seed=a.seed, port=a.port,
                warmup=str(Path(path('prompts')) / 'warmup.txt'), brackets=brackets,
                control_backend='Stock Triton; not historical dense CK',
                primary_metric='Native time_prefill seconds; not HTTP TTFT or pure GPU time',
                warmup_policy='Full-depth unscored warmup on each fresh server; measured KV cold',
                statistics='base=(control_s+control2_s)/2; gain=100*(base/candidate_s-1); '
                           'spread=abs(100*(control_s/control2_s-1)); statistics.median',
                gate='Four valid brackets, four wins, median gain>2 and greater than median spread',
                derived_helper=True, model_tested=False, quality_adoption=False,
                scope='Public synthetic workload, not byte-identical historical prompts or a '
                      'transfer of historical FullK rates; no broad quality/default approval')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def fit_prompt(count, render, label, seed, target=TOKENS):
    """Create only synthetic public text; count the actual rendered chat prompt."""
    import random
    rng = random.Random(seed)
    words = ('archive amber birch cedar delta elm field granite harbor iris '
             'junction kernel lake meadow north oak pebble quartz river stone '
             'timber valley willow xenon yellow zinc').split()
    head = 'Public synthetic benchmark ' + label + '.\n'
    # Different first pages, not just a different suffix after a shared cache key.
    head += (label + ' experiment record ') * 40 + '\n'
    tail = '\nSummarize the experimental record.'
    def total(body):
        return count(render(head + body + tail))
    need(total('') < target, 'target too short for the public prefix')
    body = ' '.join(rng.choice(words) for _ in range(target))
    while total(body) < target:
        body += ' ' + ' '.join(rng.choice(words) for _ in range(target // 4 + 1))
    # Most tokenizers have a near-monotonic length in this ASCII corpus. Binary
    # trim is a starting point only; exactness is checked, never inferred.
    lo, hi = 0, len(body)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if total(body[:mid]) <= target:
            lo = mid
        else:
            hi = mid - 1
    body = body[:lo]
    for _ in range(32):
        n = total(body)
        if n == target:
            return head + body + tail
        if n > target:
            body = body[:-1]
        else:
            # Separated single ASCII digits offer one-token padding in MiMo;
            # the exact tokenizer count remains authoritative at every step.
            body += ' 0'
    raise ValueError('cannot fit exact rendered length; no prompt was published')


def reply_source(response):
    message = response['choices'][0]['message']
    source = next((name for name in ('content', 'reasoning_content')
                   if isinstance(message.get(name), str) and message[name].strip()), None)
    need(source is not None, 'empty completed response')
    return source


def checked_request(row, response, expected_prompt_sha):
    import math
    need(row['prompt_sha256'] == expected_prompt_sha and not row['error'], 'request receipt mismatch')
    n = row['native']
    need(n['prompt_tokens'] == TOKENS and n['cached_tokens'] == 0 and
         type(n['new_tokens']) in (int, float) and 0 < n['new_tokens'] <= BUDGET and
         n['accepted_draft_tokens'] == n['rejected_draft_tokens'] == 0,
         'cache/token/speculation contract failure')
    seconds = n['time_prefill']
    need(type(seconds) in (int, float) and math.isfinite(seconds) and seconds > 0,
         'invalid native prefill seconds')
    t = response['timings']
    need(t['cache_n'] == 0 and t['prompt_n'] == TOKENS and
         t['predicted_n'] == n['new_tokens'] and t['prompt_ms'] == seconds * 1000 and
         math.isclose(t['prompt_per_second'], TOKENS / seconds, rel_tol=1e-12),
         'API/native timing mismatch')
    return dict(prefill_s=seconds, prefill_tps=TOKENS / seconds,
                generated_tokens=n['new_tokens'], reply_source=reply_source(response))


def parent_signals():
    """Let SIGTERM unwind Python finally blocks; SIGKILL cannot be handled."""
    import contextlib
    import signal
    @contextlib.contextmanager
    def installed():
        def terminate(signum, frame):
            raise KeyboardInterrupt('parent received signal ' + str(signum))
        previous = signal.signal(signal.SIGTERM, terminate)
        try:
            yield
        finally:
            signal.signal(signal.SIGTERM, previous)
    return installed()


def stop_owned(child):
    """Terminate and reap only this Popen child, without a forced-kill timeout."""
    import signal
    import subprocess
    previous = {number: signal.signal(number, signal.SIG_IGN)
                for number in (signal.SIGTERM, signal.SIGINT)}
    try:
        if child.poll() is None:
            child.terminate()
        while True:
            try:
                return child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                print('WAIT_OWNED_TEARDOWN', child.pid, flush=True)
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def arm_env(a, arm, out):
    venv = a.python.absolute().parent.parent
    sdk = venv / 'lib/python3.12/site-packages/_rocm_sdk_devel'
    core = venv / 'lib/python3.12/site-packages/_rocm_sdk_core/lib'
    private = Path(out) / 'private'
    env = dict(HOME=str(private / 'home'), PATH='/usr/bin:/bin', LC_ALL='C',
               PYTHONDONTWRITEBYTECODE='1', PYTORCH_ROCM_ARCH='gfx1151',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               EXL3_ROCM_SDK=str(sdk), EXL3_HSA_LIB=str(core / 'libhsa-runtime64.so.1'),
               LD_PRELOAD=str(core / 'libhsa-runtime64.so.1'),
               LD_LIBRARY_PATH=f'{core}:{sdk / "lib"}', EXL3_EXPANDABLE_SEGMENTS='0',
               EXL3_CK_PAGEDKV_SO=str(a.ck), EXL3_CK_SPARSE_RETAIN='0.3',
               EXL3_PREFILL_CHUNK='4096', EXL3_MOE_CFG='2',
               EXL3_HIP_PREFILL_MIN_ROWS='2', EXL3_PF_MSPLIT='2048', EXL3_MPW2X='2',
               EXL3_LOAD_DEVICE='cuda:0', EXL3_HOST_LEAN='1', EXL3_HOST_CUTS='1',
               EXL3_SPEC_GATE='0', KYOJIN_REQLOG_DIR='off', KYOJIN_SSE_PING_S='0')
    for key, leaf in (('TRITON_CACHE_DIR', 'triton'), ('TORCH_EXTENSIONS_DIR', 'torch'),
                      ('XDG_CACHE_HOME', 'cache'), ('TMPDIR', 'tmp'), ('HIP_CACHE_DIR', 'hip'),
                      ('MIOPEN_USER_DB_PATH', 'miopen'), ('KYOJIN_HEALTH_DIR', 'health')):
        env[key] = str(private / leaf)
    env['EXL3_DENSE_GEMM_TUNE_FILE'] = str(private / 'dense-gemm-tune.txt')
    if arm == 'candidate':
        env['EXL3_CK_PAGEDKV'] = 'sparse'
    return env


def summarize_bracket(times):
    import math
    need(len(times) == 3 and all(type(x) in (int, float) and math.isfinite(x) and x > 0
                                for x in times), 'invalid bracket times')
    control, candidate, control2 = times
    base = (control + control2) / 2
    return dict(control_s=control, candidate_s=candidate, control2_s=control2,
                base_s=base, gain_percent=100 * (base / candidate - 1),
                spread_percent=abs(100 * (control / control2 - 1)), win=candidate < base)


def normalized(a):
    for name in ('source', 'native', 'ck', 'model', 'prompts', 'out'):
        setattr(a, name, getattr(a, name).expanduser().resolve())
    # Do not resolve the venv executable symlink to the system interpreter.
    a.python = a.python.expanduser().absolute()
    protected = (a.source, a.native.parent, a.ck.parent, a.model,
                 a.python.parent.parent, Path(__file__).resolve().parent)
    for output in (a.prompts, a.out):
        for path in protected:
            need(not output.is_relative_to(path) and not path.is_relative_to(output),
                 'writable output overlaps an input: ' + str(path))
    need(not a.out.is_relative_to(a.prompts) and not a.prompts.is_relative_to(a.out),
         'prompts and runtime output must be disjoint')
    for path in (a.python, a.native, a.ck, a.source / 'tools/mimo/serve.py',
                 a.source / 'tools/mimo/chat_template.jinja', a.model / 'config.json'):
        need(path.is_file(), 'missing input: ' + str(path))
    return a


def source_pins(a):
    selected = {Path(__file__).resolve(), a.native, a.ck}
    for directory in ('exllamav3', 'tools/mimo', 'tools/strix_halo'):
        selected.update((a.source / directory).rglob('*.py'))
    selected.add(a.source / 'tools/mimo/chat_template.jinja')
    selected.add(a.prompts / 'manifest.json')
    selected.update(a.prompts / name for name in ['warmup.txt'] +
                    [f'measured-{index}.txt' for index in range(4)])
    for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json',
                 'generation_config.json', 'model.safetensors.index.json'):
        path = a.model / name
        if path.is_file():
            selected.add(path)
    need(len(selected) > 20, 'source inventory unexpectedly small')
    return {str(path): sha(path) for path in sorted(selected)}


def weight_identity(a):
    result = {}
    for path in sorted(a.model.glob('*.safetensors')):
        st = path.stat()
        result[str(path)] = [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns]
    need(result, 'model has no weight shards')
    return result


def bind_runtime(a):
    admission = json.loads((a.out / 'admission.json').read_bytes())
    need(admission['plan'] == make_plan(a), 'parent/worker plan differs')
    for name, digest in admission['pins'].items():
        need(sha(name) == digest, 'input changed: ' + name)
    need(weight_identity(a) == admission['weights_stat'], 'weight identity changed')
    return admission


def load_modules(a):
    import importlib
    import importlib.util
    need(sys.version_info[:2] == (3, 12), 'select the built Python3.12 venv')
    need(Path(sys.prefix).absolute() == a.python.parent.parent,
         'wrong selected interpreter environment')
    sys.path[:0] = [str(a.source), str(a.native.parent),
                   str(a.source / 'tools/mimo'), str(a.source / 'tools')]
    need(Path(importlib.util.find_spec('exllamav3').origin).resolve() ==
         a.source / 'exllamav3/__init__.py', 'wrong package discovery')
    need(Path(importlib.util.find_spec('exllamav3_ext').origin).resolve() == a.native,
         'wrong native discovery')
    import torch
    import torch.utils.cpp_extension as cpp
    def no_build(*args, **kwargs):
        raise RuntimeError('official native extension rebuild forbidden')
    cpp.load = no_build
    cpp.load_inline = no_build
    native = importlib.import_module('exllamav3_ext')
    from exllamav3.ext import exllamav3_ext as bound
    need(bound is native and Path(native.__file__).resolve() == a.native,
         'wrong loaded native extension')
    serve = importlib.import_module('serve')
    need(Path(serve.__file__).resolve() == a.source / 'tools/mimo/serve.py',
         'wrong serving module')
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    return torch, serve


def rendered(a, serve, text):
    return serve.render_prompt((a.source / 'tools/mimo/chat_template.jinja').read_text(),
                               [dict(role='user', content=text)], None,
                               reasoning_effort='none')


def make_prompts(a):
    _, serve = load_modules(a)
    from exllamav3 import Config, Tokenizer
    tokenizer = Tokenizer.from_config(Config.from_directory(str(a.model)))
    count = lambda text: tokenizer.encode(text, encode_special_tokens=True).numel()
    render = lambda text: rendered(a, serve, text)
    need(not a.prompts.exists(), 'preserve existing prompt directory')
    a.prompts.mkdir(parents=True)
    rows, first_pages = [], set()
    for index, name in enumerate(['warmup'] + [f'measured-{i}' for i in range(4)]):
        text = fit_prompt(count, render, name, a.seed + index)
        encoded = tokenizer.encode(render(text), encode_special_tokens=True).flatten()
        need(encoded.numel() == TOKENS, 'rendered prompt is not exactly131079tokens')
        first = tuple(encoded[:256].tolist())
        need(first not in first_pages, 'public prompts share a first cache page')
        first_pages.add(first)
        path = a.prompts / (name + '.txt')
        with path.open('x') as stream:
            stream.write(text)
        rows.append(dict(name=path.name, sha256=sha(path), prompt_tokens=TOKENS,
                         rendered_sha256=hashlib.sha256(render(text).encode()).hexdigest(),
                         first_page_sha256=hashlib.sha256(json.dumps(first).encode()).hexdigest()))
    save(a.prompts / 'manifest.json', dict(schema='public-cold128-prompts-v1', seed=a.seed,
         synthetic=True, scope='New public workload, not retained historical prompt bytes',
         template_sha256=sha(a.source / 'tools/mimo/chat_template.jinja'),
         tokenizer_sha256=sha(a.model / 'tokenizer.json'), rows=rows))
    print('PUBLIC_COLD128_PROMPTS_CREATED', json.dumps(rows), flush=True)
    return 0


def check_prompts(a):
    manifest = json.loads((a.prompts / 'manifest.json').read_bytes())
    need(manifest['schema'] == 'public-cold128-prompts-v1' and manifest['synthetic'] is True,
         'wrong public prompt manifest')
    need(manifest['template_sha256'] == sha(a.source / 'tools/mimo/chat_template.jinja') and
         manifest['tokenizer_sha256'] == sha(a.model / 'tokenizer.json'),
         'prompt template/tokenizer differs')
    need({row['name'] for row in manifest['rows']} ==
         {'warmup.txt', *(f'measured-{i}.txt' for i in range(4))}, 'wrong prompt population')
    for row in manifest['rows']:
        need(row['prompt_tokens'] == TOKENS and sha(a.prompts / row['name']) == row['sha256'],
             'public prompt changed')
    return manifest


def checked_sparse(row):
    name = 'ck_pagedkv_sparse_engine'
    before, after = row['counters_before'][name], row['counters_after'][name]
    delta = {key: after.get(key, 0) - before.get(key, 0)
             for key in ('accepted', 'declined', 'pages_kept', 'pages_total')}
    need(delta['accepted'] > 0 and 0 < delta['pages_kept'] < delta['pages_total'],
         'measured sparse page dropping not engaged')
    return delta


def run_worker(a):
    import copy
    import importlib
    bind_runtime(a)
    leg = [leg for group in make_plan(a)['brackets'] for leg in group][a.worker]
    out = a.out / leg['tag']
    expected = arm_env(a, leg['arm'], out)
    need(all(os.environ.get(key) == value for key, value in expected.items()),
         'worker environment mismatch')
    need((os.environ.get('EXL3_CK_PAGEDKV') == 'sparse') if leg['arm'] == 'candidate'
         else 'EXL3_CK_PAGEDKV' not in os.environ, 'wrong import-time treatment')
    torch, serve = load_modules(a)
    dispatch = importlib.import_module('exllamav3.modules.attention_fn.dispatch')
    active = [fn.__name__ for fn in dispatch.attn_fns]
    need((active[0] == 'fn_ck_paged_attn_sparse_engine') if leg['arm'] == 'candidate'
         else all(not name.startswith('fn_ck_') for name in active), 'FP16 dispatcher mismatch')
    def counters():
        return {name: copy.deepcopy(module._stats) for name in
                ('ck_pagedkv', 'ck_pagedkv_sparse', 'ck_pagedkv_sparse_engine')
                if (module := sys.modules.get('exllamav3.modules.attention_fn.' + name)) is not None}
    request_log = (out / 'requests.jsonl').open('x', buffering=1)
    class ObservedEngine(serve.ResidentEngine):
        async def generate(self, prompt, **kwargs):
            before, pieces, error = counters(), [], None
            try:
                async for piece in super().generate(prompt, **kwargs):
                    pieces.append(piece)
                    yield piece
            except BaseException as exc:
                error = type(exc).__name__ + ': ' + str(exc)
                raise
            finally:
                st = self.last_stats
                row = dict(prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                           native={key: st.get(key, 0) for key in ('prompt_tokens', 'cached_tokens',
                                   'new_tokens', 'time_prefill', 'time_generate',
                                   'accepted_draft_tokens', 'rejected_draft_tokens')},
                           completion=st.get('full_completion') or ''.join(pieces),
                           counters_before=before, counters_after=counters(), error=error)
                request_log.write(json.dumps(row, allow_nan=False) + '\n')
    serve.SERVE_DEFAULTS.update(serve.pack_sampling(str(a.model)))
    engine = ObservedEngine(str(a.model), drafter_path=None, ctx=CAPACITY, vision=False)
    from exllamav3 import CacheLayer_fp16
    need(engine.cache.initialized and len(engine.cache.layers) == 9 and
         all(type(layer) is CacheLayer_fp16 for layer in engine.cache.layers.values()),
         'not the expected MiMo FP16 cache')
    need(engine.generator.max_chunk_size == CHUNK and engine.generator.draft_model is None,
         'wrong chunk/drafter')
    app = serve.create_app(engine, 'MiMo-2.6-EXL3',
                           (a.source / 'tools/mimo/chat_template.jinja').read_text())
    async def cleanup(_):
        engine.generator.clear_queue()
        engine.generator.filter_pool.shutdown(wait=True)
        engine.model.unload()
        request_log.close()
    app.on_cleanup.append(cleanup)
    prompts = {path: rendered(a, serve, Path(path).read_text())
               for path in (make_plan(a)['warmup'], leg['prompt'])}
    counts = {path: engine.count_tokens(text) for path, text in prompts.items()}
    need(all(n == TOKENS for n in counts.values()), 'rendered prompt token count mismatch')
    hashes = {path: hashlib.sha256(text.encode()).hexdigest() for path, text in prompts.items()}
    first_pages = {tuple(engine.tokenizer.encode(text, encode_special_tokens=True).flatten()[:256].tolist())
                   for text in prompts.values()}
    need(len(set(hashes.values())) == len(first_pages) == 2, 'warmup shares prompt KV prefix')
    save(out / 'ready.json', dict(pid=os.getpid(), arm=leg['arm'], cache_bits=[16, 16],
                                 capacity=engine.cache.max_num_tokens, chunk=engine.generator.max_chunk_size,
                                 prompt_counts=counts, rendered_prompt_hashes=hashes,
                                 fp16_dispatch=active, native_sha256=sha(a.native), ck_sha256=sha(a.ck)))
    serve.web.run_app(app, host='127.0.0.1', port=a.port)
    return 0


def lane_state():
    owners = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            descriptors = list((proc / 'fd').iterdir())
        except OSError:
            continue
        for fd in descriptors:
            try:
                target = os.readlink(fd)
            except OSError:
                continue
            if target == '/dev/kfd' or target.startswith('/dev/accel/accel'):
                owners.append(dict(pid=int(proc.name), device=target))
                break
    busy = {str(path): path.read_text().strip()
            for path in Path('/sys/class/drm').glob('card[0-9]*/device/gpu_busy_percent')}
    return dict(compute_owners=owners, busy=busy)


def policy():
    result = {}
    for pattern in ('card[0-9]*/device/pp_dpm_fclk',
                    'card[0-9]*/device/power_dpm_force_performance_level'):
        for path in Path('/sys/class/drm').glob(pattern):
            text = path.read_text().strip()
            result[str(path)] = [line for line in text.splitlines() if '*' in line] if 'fclk' in pattern else text
    for name in ('control', 'runtime_status'):
        path = Path('/sys/class/accel/accel0/device/power') / name
        if path.exists():
            result[str(path)] = path.read_text().strip()
    return result


def port_open(a):
    import socket
    with socket.socket() as sock:
        return sock.connect_ex(('127.0.0.1', a.port)) == 0


def request(a, path, body=None):
    import urllib.request
    raw = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f'http://127.0.0.1:{a.port}' + path, data=raw,
                                 headers={'Content-Type': 'application/json'})
    # No overall inference timeout; parent ownership and child liveness supervise it.
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req) as response:
        return response.read()


def monitored_request(a, child, body, tag):
    import queue
    import threading
    import time
    result = queue.Queue()
    def work():
        try:
            result.put((True, request(a, '/v1/chat/completions', body)))
        except BaseException as exc:
            result.put((False, exc))
    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    started = time.monotonic()
    while True:
        try:
            ok, value = result.get(timeout=30)
            if not ok:
                raise value
            return value
        except queue.Empty:
            print('COLD128_PROGRESS', tag, round(time.monotonic() - started, 1), flush=True)
            need(child.poll() is None, 'server exited during request')


def worker_argv(a, index):
    argv = [str(a.python), '-I', '-B', '-u', str(Path(__file__).resolve()), '--execute']
    for key in ('source', 'native', 'ck', 'model', 'python', 'prompts', 'out', 'port', 'seed'):
        argv += ['--' + key, str(getattr(a, key))]
    return argv + ['--worker', str(index)]


def run_leg(a, index, leg, fixed):
    import subprocess
    import time
    out = a.out / leg['tag']
    out.mkdir()
    for leaf in ('home', 'triton', 'torch', 'cache', 'tmp', 'hip', 'miopen', 'health'):
        (out / 'private' / leaf).mkdir(parents=True)
    state = lane_state()
    need(not state['compute_owners'] and state['busy'] and
         all(value == '0' for value in state['busy'].values()) and not port_open(a),
         'device lane occupied')
    need(policy() == fixed, 'policy drift')
    record = dict(leg=leg, argv=worker_argv(a, index), environment=arm_env(a, leg['arm'], out),
                  before=state, policy_before=fixed)
    child = None
    try:
        with (out / 'server.log').open('xb') as log:
            child = subprocess.Popen(record['argv'], cwd=out / 'private', env=record['environment'],
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            record['pid'] = child.pid
            record['start_ticks'] = Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
            save(out / 'launch.json', record)
            heartbeat = 0.0
            while True:
                need(child.poll() is None, 'server exited before ready: ' + str(out / 'server.log'))
                if (out / 'ready.json').is_file() and port_open(a):
                    ready = json.loads((out / 'ready.json').read_bytes())
                    need(ready['pid'] == child.pid, 'wrong ready PID')
                    if json.loads(request(a, '/health')).get('generator'):
                        break
                if time.monotonic() - heartbeat > 30:
                    print('WAIT_READY', leg['tag'], child.pid, flush=True)
                    heartbeat = time.monotonic()
                time.sleep(1)
            record['effective_environment'] = dict(item.split('=', 1) for item in
                Path(f'/proc/{child.pid}/environ').read_bytes().decode().strip('\0').split('\0'))
            need(all(record['effective_environment'].get(key) == value
                     for key, value in record['environment'].items()), 'child environment differs')
            for request_index, (kind, prompt_path) in enumerate(
                    (('warmup', make_plan(a)['warmup']), ('measured', leg['prompt']))):
                body = dict(model='MiMo-2.6-EXL3', messages=[dict(role='user', content=Path(prompt_path).read_text())],
                            reasoning_effort='none', max_tokens=BUDGET, temperature=0.0, top_p=1.0, stream=False)
                print('COLD128_BEGIN', leg['tag'], kind, flush=True)
                started = time.perf_counter()
                raw = monitored_request(a, child, body, leg['tag'] + ':' + kind)
                elapsed = time.perf_counter() - started
                with (out / (kind + '-response.json')).open('xb') as stream:
                    stream.write(raw)
                response = json.loads(raw)
                rows = [json.loads(line) for line in (out / 'requests.jsonl').read_text().splitlines()]
                need(len(rows) == request_index + 1, 'missing/extra native request')
                rendered_sha = ready['rendered_prompt_hashes'][str(prompt_path)]
                score = checked_request(rows[-1], response, rendered_sha)
                delta = checked_sparse(rows[-1]) if leg['arm'] == 'candidate' else None
                if leg['arm'] != 'candidate':
                    need(not rows[-1]['counters_before'] and not rows[-1]['counters_after'],
                         'stock control unexpectedly loaded CK')
                record[kind] = dict(prompt=str(prompt_path), prompt_sha256=sha(prompt_path),
                                    rendered_sha256=rendered_sha, native_score=score,
                                    response_sha256=hashlib.sha256(raw).hexdigest(), sparse_delta=delta,
                                    timings=response['timings'], client_wall_s=elapsed)
                save(out / (kind + '-summary.json'), record[kind])
                print('COLD128_END', leg['tag'], kind, score['prefill_tps'], flush=True)
            maps = Path(f'/proc/{child.pid}/maps').read_text()
            with (out / 'after-request-maps.txt').open('x') as stream:
                stream.write(maps)
            record['ck_mapped_after_request'] = str(a.ck) in maps
            need(record['ck_mapped_after_request'] == (leg['arm'] == 'candidate'), 'wrong CK mapping')
    except BaseException as exc:
        record['error'] = type(exc).__name__ + ': ' + str(exc)
    finally:
        if child is not None:
            record['server_returncode'] = stop_owned(child)
        record['port_closed'] = not port_open(a)
        state = lane_state()
        while not state['compute_owners'] and state['busy'] and any(value != '0' for value in state['busy'].values()):
            print('WAIT_GPU_IDLE', leg['tag'], state['busy'], flush=True)
            time.sleep(2)
            state = lane_state()
        record['after'], record['policy_after'] = state, policy()
        save(out / 'result.json', record)
    need(not record.get('error') and record.get('server_returncode') == 0 and record['port_closed']
         and not state['compute_owners'] and record['policy_after'] == fixed,
         'leg instrument/cleanup failure; inspect ' + str(out / 'result.json'))
    return record


def run_parent(a):
    import statistics
    check_prompts(a)
    need(not a.out.exists(), 'preserve existing output root')
    os.umask(0o077)
    a.out.mkdir(parents=True)
    fixed = policy()
    admission = dict(plan=make_plan(a), pins=source_pins(a), weights_stat=weight_identity(a), policy=fixed,
                     meminfo=Path('/proc/meminfo').read_text(), vmstat=Path('/proc/vmstat').read_text())
    save(a.out / 'admission.json', admission)
    records, brackets = [], []
    with parent_signals():
        for index, leg in enumerate(leg for group in make_plan(a)['brackets'] for leg in group):
            bind_runtime(a)
            records.append(run_leg(a, index, leg, fixed))
            if len(records) % 3 == 0:
                group = records[-3:]
                need(len({row['measured']['rendered_sha256'] for row in group}) == 1,
                     'bracket prompt hashes differ')
                score = summarize_bracket([row['measured']['native_score']['prefill_s'] for row in group])
                score['pair'] = leg['pair']
                brackets.append(score)
                save(a.out / f'bracket{leg["pair"]}.json', score)
                print('COLD128_BRACKET', json.dumps(score), flush=True)
    bind_runtime(a)
    gain = statistics.median(row['gain_percent'] for row in brackets)
    spread = statistics.median(row['spread_percent'] for row in brackets)
    wins = sum(row['win'] for row in brackets)
    result = dict(status='PUBLIC_COLD128_COMPLETE', pairs_complete='4 of 4', brackets=brackets,
                  records=records, wins=wins, median_gain_percent=gain, median_spread_percent=spread,
                  screen_gate=wins == 4 and gain > 2 and gain > spread,
                  candidate_tps_from_median_duration=TOKENS / statistics.median(
                      row['measured']['native_score']['prefill_s'] for row in records if row['leg']['arm'] == 'candidate'),
                  control_tps_from_median_duration=TOKENS / statistics.median(
                      row['measured']['native_score']['prefill_s'] for row in records if row['leg']['arm'] != 'candidate'),
                  quality_adoption=False, default_enablement=False, plan=make_plan(a),
                  vmstat_after=Path('/proc/vmstat').read_text())
    save(a.out / 'result.json', result)
    print('PUBLIC_COLD128_COMPLETE', json.dumps({key: result[key] for key in
          ('pairs_complete', 'wins', 'median_gain_percent', 'median_spread_percent', 'screen_gate')}), flush=True)
    return 0


def runtime(a):
    a = normalized(a)
    if a.make_prompts:
        # Re-exec with the selected venv and clean import-time environment.
        if os.environ.get('PUBLIC_COLD128_PROMPT_ENV') != '1':
            need(not a.prompts.exists(), 'preserve existing prompt directory')
            prep = a.prompts.parent / ('.' + a.prompts.name + '-prep')
            for protected in (a.source, a.native.parent, a.ck.parent, a.model,
                              a.python.parent.parent, a.out, Path(__file__).resolve().parent):
                need(not prep.is_relative_to(protected) and not protected.is_relative_to(prep),
                     'prompt preparation overlaps an input/output')
            prep.mkdir(parents=True, exist_ok=False)
            env = arm_env(a, 'control', prep)
            for key in ('HOME', 'TRITON_CACHE_DIR', 'TORCH_EXTENSIONS_DIR', 'XDG_CACHE_HOME',
                        'TMPDIR', 'HIP_CACHE_DIR', 'MIOPEN_USER_DB_PATH', 'KYOJIN_HEALTH_DIR'):
                Path(env[key]).mkdir(parents=True, exist_ok=True)
            env['PUBLIC_COLD128_PROMPT_ENV'] = '1'
            # Carry paths normalized under the caller's HOME into the private HOME.
            argv = [str(a.python), '-I', '-B', '-u', str(Path(__file__).resolve()), '--make-prompts']
            for key in ('source', 'native', 'ck', 'model', 'python', 'prompts', 'out', 'port', 'seed'):
                argv += ['--' + key, str(getattr(a, key))]
            os.execve(str(a.python), argv, env)
        return make_prompts(a)
    return run_worker(a) if a.worker is not None else run_parent(a)


def main():
    a = parser().parse_args()
    plan = make_plan(a)
    if not a.execute and not a.make_prompts:
        print(json.dumps(plan, indent=2))
        return 0
    return runtime(a)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print('COLD128_INSTRUMENT_FAILED', type(exc).__name__, str(exc), file=sys.stderr, flush=True)
        sys.exit(6)
