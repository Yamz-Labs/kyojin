#!/usr/bin/env python3
"""Standard-library-only orchestration for the pinned optional CK backend."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile

CK_COMMIT = '6f44fdf90f69adee742ec3660b95d32963b9bd2d'
SOURCE_SCHEMA = 'ck-pagedkv-owned-source-v1'
INSTANCE_NAME = (
    'fmha_fwd_pagedkv_d192_fp16_batch_b64x64x32x128x32x192_r4x1x1_r4x1x1_'
    'w16x16x16_w16x16x16_qr_pagedkv_vr_pssk_nlogits_nbias_mask_nlse_nskip_nsquant_pagedkv_nsink_gfx11.cpp'
)


def git(repo, *args, check=True):
    # Repository redirects/config injection must not escape -C or modify a
    # caller-selected index/worktree. The donor is always a read-only input.
    env = {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}
    env.update(GIT_OPTIONAL_LOCKS='0', GIT_CONFIG_NOSYSTEM='1',
               GIT_CONFIG_GLOBAL='/dev/null', GIT_NO_REPLACE_OBJECTS='1',
               GIT_NO_LAZY_FETCH='1', GIT_ALLOW_PROTOCOL='file', LC_ALL='C')
    result = subprocess.run(['git', '-c', 'core.hooksPath=/dev/null', '-C', str(repo), *args],
                            env=env, capture_output=True, text=True)
    if check and result.returncode:
        raise BuildError('git ' + ' '.join(args) + ': ' + result.stderr.strip())
    return result


def apply_patch(stage, patch):
    """Only an owned stage may be passed here. Never patch the donor worktree."""
    # Classify the whole tracked delta BEFORE making a change. A forward
    # apply check alone can succeed even with unrelated dirty source edits.
    actual = git(stage, 'diff', '--no-ext-diff', '--no-color', '--full-index', 'HEAD', '--').stdout
    # Bare-empty unified context keeps the patch file whitespace-clean. Git
    # emits a space marker there; normalize only that empty context spelling.
    actual = re.sub(r'(?m)^ $', '', actual)
    expected = re.sub(r'(?m)^ $', '', patch.read_text())
    if actual not in ('', expected):
        raise BuildError('Owned CK stage has changes beyond the exact pinned patch')
    if git(stage, 'ls-files', '--others', '--exclude-standard').stdout:
        raise BuildError('Owned CK stage has untracked source files')
    if actual == '' and git(stage, 'apply', '--check', str(patch), check=False).returncode == 0:
        git(stage, 'apply', str(patch))
        return 'applied'
    if actual == expected and git(stage, 'apply', '--reverse', '--check', str(patch), check=False).returncode == 0:
        return 'already-applied'
    raise BuildError('CK stage is incompatible or partially patched; refusing to change it')


def prepare_source(donor, stage, pin, patch):
    """Use only the donor commit object. Ignore its HEAD and dirty/untracked bytes.

    The real CK checkout happens only during a future real build. Unit tests use
    two-file local repositories; no real CK archive/checkout is a preparation test.
    """
    if Path(stage).is_symlink():
        raise BuildError('Refusing a symlink as the owned CK stage')
    donor, stage, patch = Path(donor).resolve(), Path(stage).resolve(), Path(patch).resolve()
    if donor == stage or donor in stage.parents or stage in donor.parents:
        raise BuildError('Source stage and donor must be disjoint')
    if not re.fullmatch(r'[0-9a-f]{40}', pin):
        raise BuildError('CK pin must be a full 40-hex commit')
    if git(donor, 'rev-parse', '--verify', pin + '^{commit}').stdout.strip() != pin:
        raise BuildError('Local donor does not contain the exact CK commit')
    expected = {'schema': SOURCE_SCHEMA, 'commit': pin,
                'patch_sha256': hashlib.sha256(patch.read_bytes()).hexdigest()}
    marker = stage / '.git/ck-pagedkv-source.json'
    if stage.exists():
        if not marker.is_file() or json.loads(marker.read_text()) != expected:
            raise BuildError('Refusing existing unowned or differently pinned CK stage')
        if git(stage, 'rev-parse', 'HEAD').stdout.strip() != pin:
            raise BuildError('Owned CK stage HEAD changed')
        return apply_patch(stage, patch)
    stage.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix='.ck-source-', dir=stage.parent))
    try:
        # --shared writes only an alternates pointer in our clone. No fetch,
        # checkout, index refresh, config edit or patch is performed in donor.
        git(stage.parent, 'clone', '--shared', '--no-checkout', '--no-tags', '--', str(donor), str(temp))
        git(temp, '-c', 'advice.detachedHead=false', 'checkout', '--detach', pin)
        state = apply_patch(temp, patch)
        (temp / '.git/ck-pagedkv-source.json').write_text(json.dumps(expected, indent=2) + '\n')
        temp.rename(stage)
        return state
    finally:
        if temp.exists():
            shutil.rmtree(temp)


def compile_command(entry, source, output, compiler):
    """Retain the native selected-instance recipe, changing only its I/O paths.

    Dependency paths are separate for shim and instance; warning, target,
    language, include, optimization and ABI flags are not stripped.
    """
    cwd = Path(entry['directory']).resolve()
    if not cwd.is_dir():
        raise BuildError('Compile command directory is absent: ' + str(cwd))
    argv = list(entry.get('arguments') or shlex.split(entry['command']))
    if not argv or Path(argv[0]).resolve() != Path(compiler).resolve():
        raise BuildError('Compile database did not select the explicit ROCm compiler')
    if any(x in argv for x in ('-fno-pic', '-fno-PIC', '-fpie', '-fPIE')):
        raise BuildError('Compile recipe has conflicting PIC/PIE flags')
    original = Path(entry['file'])
    original = (cwd / original).resolve() if not original.is_absolute() else original.resolve()
    result = [argv[0]]
    count_source = count_output = 0
    i = 1
    while i < len(argv):
        item = argv[i]
        if item in ('-o', '-MF', '-MT', '-MQ'):
            if i + 1 == len(argv):
                raise BuildError('Truncated compile option: ' + item)
            replacement = str(output) + '.d' if item == '-MF' else str(output)
            result.extend([item, replacement])
            count_output += item == '-o'
            i += 2
            continue
        if not item.startswith('-') and (cwd / item).resolve() == original:
            result.append(str(source))
            count_source += 1
        else:
            result.append(item)
        i += 1
    if count_source != 1 or count_output != 1 or '-c' not in argv:
        raise BuildError('Expected exactly one compile input/output and -c')
    if '-fPIC' not in result:
        result.append('-fPIC')
    return result, cwd


def build_backend(donor, cache, sdk, shim, patch, *, pin=CK_COMMIT, tool_commands=None):
    """Build only the exact instance and shim. tool_commands is a unit-test seam.

    The public CLI never supplies tool_commands or a replacement pin. Tests pass
    inert stdlib recorder commands and a two-file local Git fixture explicitly.
    """
    if Path(cache).is_symlink():
        raise BuildError('Refusing a symlink as the owned build cache')
    donor, cache, sdk, shim, patch = [Path(p).resolve() for p in (donor, cache, sdk, shim, patch)]
    if donor == cache or donor in cache.parents or cache in donor.parents:
        raise BuildError('Cache and user-supplied source must be disjoint')
    clang, clang_c = sdk / 'lib/llvm/bin/clang++', sdk / 'lib/llvm/bin/clang'
    for path in (clang, clang_c):
        if not path.is_file() or not os.access(path, os.X_OK):
            raise BuildError('Missing explicit ROCm compiler: ' + str(path))
    for path in (shim, patch):
        if not path.is_file():
            raise BuildError('Missing builder input: ' + str(path))
    identity = {'schema': 'ck-pagedkv-build-v1', 'commit': pin, 'sdk': str(sdk),
                'patch_sha256': hashlib.sha256(patch.read_bytes()).hexdigest(),
                'shim_sha256': hashlib.sha256(shim.read_bytes()).hexdigest(),
                'builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'inert_test_tools': tool_commands is not None}
    owner = cache / 'owner.json'
    # A matching marker is not permission to follow new administrative
    # symlinks into a user tree on a retry. This is not an adversarial sandbox.
    for relative in ('owner.json', 'builder.lock', 'source', 'source/.git',
                     'source/.git/ck-pagedkv-source.json', 'build', 'receipt.json',
                     'receipt.json.tmp', 'libck_pagedkv.so', 'libck_pagedkv.so.tmp'):
        if (cache / relative).is_symlink():
            raise BuildError('Refusing symlink in owned cache: ' + relative)
    if cache.exists():
        if not owner.is_file() or json.loads(owner.read_text()) != identity:
            raise BuildError('Refusing existing unowned or incompatible cache; choose a fresh EXL3_CK_CACHE')
    else:
        cache.mkdir(parents=True)
        owner.write_text(json.dumps(identity, indent=2) + '\n')
    with (cache / 'builder.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise BuildError('Another builder owns this cache') from error
        stage = cache / 'source'
        patch_state = prepare_source(donor, stage, pin, patch)
        build = cache / 'build'
        build.mkdir(exist_ok=True)
        attempt = Path(tempfile.mkdtemp(prefix='attempt-', dir=cache))
        commands = []
        log = attempt / 'commands.json'
        # Never consult ambient loader overrides in compiler subprocesses.
        env = dict(os.environ, LC_ALL='C', PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(attempt))
        for key in ('LD_PRELOAD', 'LD_LIBRARY_PATH', 'PYTHONPATH', 'PYTHONHOME', 'HSA_OVERRIDE_GFX_VERSION'):
            env.pop(key, None)
        def run(argv, cwd):
            argv = [str(x) for x in argv]
            row = {'argv': argv, 'cwd': str(cwd)}
            commands.append(row)
            result = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True)
            row.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
            log.write_text(json.dumps(commands, indent=2) + '\n')
            if result.returncode:
                raise BuildError('Command failed (' + str(result.returncode) + '); see ' + str(log) + '\n' + result.stderr)
            return result.stdout
        tools = {'cmake': ['cmake'], 'codegen': ['/usr/bin/python3', '-I', '-S'], 'nm': ['nm']}
        if tool_commands is not None:
            if set(tool_commands) != set(tools):
                raise BuildError('Tests must explicitly replace every configure/codegen/nm command')
            tools = tool_commands
        run(tools['cmake'] + [
            '-S', str(stage), '-B', str(build), '-G', 'Unix Makefiles',
            '-DCMAKE_BUILD_TYPE=Release', '-DGPU_TARGETS=gfx1151', '-DAMDGPU_TARGETS=gfx1151',
            '-DGPU_ARCHS=', '-DCOMPILER_HAS_TARGET_ID_gfx1151=ON',
            '-DBUILD_TESTING=OFF', '-DBUILD_DEV=ON', '-DBUILD_CK_EXAMPLES=ON',
            '-DBUILD_CK_TILE_FMHA_TESTS=ON', '-DBUILD_CK_DEVICE_INSTANCES=OFF',
            '-DBUILD_CK_PROFILER=OFF', '-DBUILD_CK_TUTORIALS=OFF',
            '-DFMHA_FWD_ENABLE_APIS=pagedkv_prefill', '-DCMAKE_POSITION_INDEPENDENT_CODE=ON',
            '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON', '-DPython3_EXECUTABLE=/usr/bin/python3',
            '-DCMAKE_PREFIX_PATH=' + str(sdk) + ';/usr', '-DROCM_DIR=' + str(sdk / 'share/rocm/cmake'),
            '-Dhip_DIR=' + str(sdk / 'lib/cmake/hip'), '-Dhip-lang_DIR=' + str(sdk / 'lib/cmake/hip-lang'),
            '-DCMAKE_C_COMPILER=' + str(clang_c), '-DCMAKE_CXX_COMPILER=' + str(clang),
            '-DCMAKE_HIP_COMPILER=' + str(clang)], cache)
        database = json.loads((build / 'compile_commands.json').read_text())
        matches = [entry for entry in database if Path(entry['file']).name == INSTANCE_NAME]
        if len(matches) != 1:
            raise BuildError('Expected exactly one selected instance compile command; found ' + str(len(matches)))
        entry = matches[0]
        directory = Path(entry['directory']).resolve()
        source = (directory / entry['file']).resolve()
        if build not in source.parents or build not in directory.parents and directory != build:
            raise BuildError('Selected compile command escaped the owned build directory')
        # CK CMake only lists blobs at configure; the custom generation command
        # is not run by reading compile_commands.json. Explicitly generate just
        # this instance (and the API text, which is NOT compiled or linked).
        codegen = stage / 'example/ck_tile/01_fmha/generate.py'
        # -I removes script-directory imports. -S is enough for pinned CK's local
        # codegen imports; no framework/package site initialization is needed.
        if tool_commands is None:
            tools['codegen'] = ['/usr/bin/python3', '-S']
        # A prior successful attempt must not hide missing/failed fresh output.
        source.unlink(missing_ok=True)
        run(tools['codegen'] + [str(codegen), '--targets', 'gfx1151', '--api', 'pagedkv_prefill',
                               '--optdim', '192', '--mask', 'simplified', '--receipt', '0',
                               '--filter', INSTANCE_NAME.removesuffix('_gfx11.cpp'),
                               '--output_dir', str(source.parent)], build)
        if not source.is_file():
            raise BuildError('Selected instance was not generated: ' + str(source))
        instance_obj, shim_obj = attempt / 'instance.o', attempt / 'shim.o'
        for input_source, output in ((source, instance_obj), (shim, shim_obj)):
            argv, cwd = compile_command(entry, input_source, output, clang)
            run(argv, cwd)
            if not output.is_file():
                raise BuildError('Compiler returned success without an object: ' + str(output))
        symbol = undefined_specialization(run(tools['nm'] + ['-P', '--no-demangle', '--undefined-only', str(shim_obj)], attempt))
        defined = nm_symbols(run(tools['nm'] + ['-P', '--no-demangle', '--defined-only', str(instance_obj)], attempt))
        if sum(name == symbol and kind == 'T' for name, kind in defined) != 1:
            raise BuildError('Selected instance does not define the exact mangled shim specialization')
        library = attempt / 'libck_pagedkv.so'
        # The HIP link driver supplies the HIP runtime, unlike bare host linking.
        # -z defs catches unresolved host references now rather than at dlopen.
        run([str(clang), '-shared', '-fPIC', '--hip-link', '--offload-arch=gfx1151',
             '--rocm-path=' + str(sdk), '-Wl,-z,defs',
             '-Wl,-rpath,' + str(sdk / 'lib'), '-o', str(library), str(shim_obj), str(instance_obj)], attempt)
        exported = nm_symbols(run(tools['nm'] + ['-P', '--no-demangle', '-D', '--defined-only', str(library)], attempt))
        for expected in ('ck_pagedkv_forward', symbol):
            if sum(name == expected and kind == 'T' for name, kind in exported) != 1:
                raise BuildError('Missing exact shared-library export: ' + expected)
        unresolved = nm_symbols(run(tools['nm'] + ['-P', '--no-demangle', '-D', '--undefined-only', str(library)], attempt))
        if any('fmha_fwd_pagedkv_' in name for name, kind in unresolved):
            raise BuildError('Shared library still has an unresolved paged-KV specialization')
        if not library.is_file():
            raise BuildError('Linker returned success without a library')
        destination = cache / 'libck_pagedkv.so'
        receipt_path = cache / 'receipt.json'
        previous_library = attempt / 'previous-libck_pagedkv.so'
        had_previous = destination.exists()
        if had_previous != receipt_path.exists():
            raise BuildError('Incomplete published pair; recover ' + str(destination) + ' and ' + str(receipt_path))
        temporary = cache / 'libck_pagedkv.so.tmp'
        shutil.copy2(library, temporary)
        receipt = dict(identity, donor=str(donor), patch_state=patch_state,
                       specialization=symbol, instance=str(source), commands=str(log),
                       output=str(destination), output_sha256=hashlib.sha256(temporary.read_bytes()).hexdigest(),
                       native_load_performed=False, numerical_validation_performed=False)
        (attempt / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        temporary_receipt = cache / 'receipt.json.tmp'
        temporary_receipt.write_text(json.dumps(receipt, indent=2) + '\n')
        if had_previous:
            shutil.copy2(destination, previous_library)
        # Recover ordinary exceptions only, not crashes or concurrent readers.
        # Receipt replacement is last: on failure its prior bytes stay in place.
        library_published = False
        try:
            temporary.replace(destination)
            library_published = True
            temporary_receipt.replace(receipt_path)
        except Exception as error:
            if library_published:
                try:
                    if had_previous:
                        previous_library.replace(destination)
                    else:
                        destination.unlink()
                except Exception as recovery_error:
                    raise BuildError('Publication failed: ' + str(error) + '; recovery failed: ' + str(recovery_error)
                                     + '; previous library backup (if any): ' + str(previous_library)
                                     + '; preserved receipt: ' + str(receipt_path)) from error
            raise
        return destination


class BuildError(RuntimeError):
    """A prerequisite, source state, command, or symbol contract failed."""


def nm_symbols(text):
    """Parse GNU/LLVM nm -P --no-demangle (name, type, optional value/size)."""
    rows = []
    for line in text.splitlines():
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 2 or not re.fullmatch(r'[A-Za-z?]', fields[1]):
            raise BuildError('Unexpected nm -P row: ' + line)
        rows.append((fields[0], fields[1]))
    return rows


def undefined_specialization(text):
    names = [name for name, kind in nm_symbols(text)
             if kind == 'U' and 'fmha_fwd_pagedkv_' in name]
    if len(names) != 1:
        raise BuildError('Expected one undefined paged-KV specialization, found ' + str(names))
    return names[0]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Build the pinned gfx1151 d192 CK backend in a separate owned cache; no implicit network.')
    parser.add_argument('--source', default=os.environ.get('EXL3_CK_SOURCE'),
                        help='Read-only local Git donor containing the exact pin (EXL3_CK_SOURCE)')
    parser.add_argument('--sdk', default=os.environ.get('EXL3_ROCM_SDK'),
                        help='Already-expanded ROCm SDK directory (EXL3_ROCM_SDK)')
    default_cache = Path(os.environ.get('XDG_CACHE_HOME', str(Path.home() / '.cache'))) / 'kyojin/ck-pagedkv'
    parser.add_argument('--cache', default=os.environ.get('EXL3_CK_CACHE', str(default_cache)),
                        help='Disjoint owned cache/output directory (EXL3_CK_CACHE)')
    args = parser.parse_args(argv)
    if not args.source:
        parser.error('Set EXL3_CK_SOURCE or --source to a local Git donor containing ' + CK_COMMIT)
    if not args.sdk:
        parser.error('Set EXL3_ROCM_SDK or --sdk to an already-expanded SDK')
    for command in ('git', 'cmake', 'nm'):
        if shutil.which(command) is None:
            parser.error('Required command is absent: ' + command)
    here = Path(__file__).resolve().parent
    try:
        library = build_backend(Path(args.source), Path(args.cache), Path(args.sdk),
                                here / 'ck_pagedkv_shim.cpp', here / 'patches/ck-pagedkv-d192.patch')
    except (BuildError, OSError, ValueError, KeyError) as error:
        parser.exit(1, 'CK backend build stopped: ' + str(error) + '\n')
    print('Built ' + str(library) + '; native loading and numerics were not performed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
