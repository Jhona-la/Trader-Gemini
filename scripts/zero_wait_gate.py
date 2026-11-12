"""Additive, input-bound zero-wait publication evidence; never launches Cargo.

The shared capture API observes files twice, without a hardware/compile lock.
It detects observed drift, not an adversarial change-and-restore (ABA).
"""
import datetime as dt
import base64
from contextlib import contextmanager
import ctypes
from ctypes import wintypes as w
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile
import tomllib
import types
import uuid

ENVELOPE = 'trader-gemini-prepush-zero-wait-v1'
SUBJECT = 'trader-gemini-zero-wait-subject-v1'
PRODUCER = 'trader-gemini-zero-wait-producer-v1'
RECEIPT = 'trader-gemini-compiler-bus-receipt-v1'
COMMANDS = {
    'check': ('check', '--workspace', '--all-targets', '--locked', '--message-format=json'),
    'tests': ('test', '-p', 'execution-engine', '-p', 'feature-engine', '--lib', '--tests', '--locked', '--message-format=json'),
    'daz-tests': ('test', '-p', 'os-guardian', '--lib', '--test', 'f8_001_silicon_physics_contract', '--locked', '--message-format=json'),
    'daz-clippy': ('clippy', '-p', 'os-guardian', '--lib', '--locked', '--message-format=json', '--', '-D', 'warnings'),
    'clippy': ('clippy', '--workspace', '--lib', '--locked', '--message-format=json', '--', '-D', 'warnings'),
    'fmt': ('format-workspace', '--all-local-targets', '--check'),
    'storage-tests': ('test', '-p', 'storage-engine', '--lib', '--tests', '--locked', '--message-format=json'),
    'omni-tests': ('test', '-p', 'omniscient-registry', '--lib', '--tests', '--locked', '--message-format=json'),
    'semantic-tests': ('test', '--no-fail-fast', '-p', 'audit-engine', '-p', 'data-ingest', '-p', 'data-pipeline',
                       '-p', 'flight-recorder', '-p', 'god-engine-core', '-p', 'omniscient-registry',
                       '-p', 'quantum-arena', '-p', 'storage-engine', '-p', 'strategy-core',
                       '-p', 'metacortex-engine', '-p', 'signal-engine', '-p', 'telemetry-server',
                       '-p', 'backtest-engine', '-p', 'dark-alpha-engine', '-p', 'evolution-engine',
                       '-p', 'risk-engine', '-p', 'trader-gemini-v5',
                       '--lib', '--tests', '--locked', '--message-format=json'),
}
MANDATORY = ('check', 'tests', 'daz-tests', 'daz-clippy', 'clippy', 'fmt')
TOOLS = {'cargo', 'rustc', 'rustdoc', 'linker', 'git', 'python', 'runner',
         'cargo_fmt', 'cargo_clippy', 'rustfmt', 'clippy_driver'}
ISOLATION_FILES = {'runner.py', 'fixture_manifest.json', 'README.md'}
FORBIDDEN = ('.agent_tx/**', 'target/**', 'logs/**', '.env', '.env.*')
GOLDEN = {
    'crates/execution-engine/tests/simulator_reduce_only_accounting_contract.rs': (
        'partial_short_close_preserves_residual_and_accounts_only_the_fill',
        'oversize_reduce_only_matches_exact_position_size_including_impact_and_fee',
        'reduce_only_rejects_the_same_side_for_both_position_directions',
        'invalid_close_quantities_preserve_capital_and_position',
        'close_without_position_is_not_a_successful_fill'),
    'crates/feature-engine/tests/multifractal_invalid_params_contract.rs': (
        'validation_rejects_finite_weights_whose_sum_overflows',
        'invalid_params_use_whole_default_fallback_without_changing_later_state',
        'large_valid_weights_preserve_the_same_convex_combination',
        'validated_lower_bounds_are_used_at_the_exact_boundary'),
}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(',', ':')).encode()


def hash_value(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def read_json(raw):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, 'Duplicate JSON field')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs)


def regular(path, *, single_link=True):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path, 'Noncanonical absolute path')
    for item in (path, *path.parents):
        info = item.lstat()
        require(not item.is_symlink() and not getattr(info, 'st_file_attributes', 0) & 0x400, 'Reparse path')
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and (not single_link or info.st_nlink == 1), 'Nonregular/linked file')
    with path.open('rb') as stream:
        opened = os.fstat(stream.fileno())
        raw = stream.read()
        finished = os.fstat(stream.fileno())
    require((info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) ==
            (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) ==
            (finished.st_dev, finished.st_ino, finished.st_size, finished.st_mtime_ns), 'File changed during read')
    return raw


def relative(name):
    require(isinstance(name, str) and name and '\\' not in name and ':' not in name and
            all(ord(c) >= 32 and ord(c) != 127 for c in name), 'Invalid relative path')
    require(not name.startswith('/') and all(p not in ('', '.', '..', '.git') for p in name.split('/')), 'Unsafe relative path')
    return name


def artifact(root, reference):
    require(isinstance(reference, dict) and set(reference) == {'path', 'sha256'} and hash_value(reference['sha256']), 'Closed artifact reference required')
    path = root / relative(reference['path'])
    raw = regular(path)
    require(digest(raw) == reference['sha256'], 'Artifact changed: ' + reference['path'])
    return raw


def registration(root, reference):
    value = read_json(artifact(root, reference))
    keys = {'schema', 'status', 'reviewer', 'review_reference', 'helpers', 'tools', 'isolation',
            'golden_inventory', 'python_suites', 'commands', 'forbidden_paths', 'classifications'}
    require(set(value) == keys and value['schema'] == PRODUCER and value['status'] == 'REVIEWED'
            and value['reviewer'] == 'ROOT' and isinstance(value['review_reference'], str)
            and value['review_reference'].strip(), 'Explicit reviewed zero-wait producer required')
    require(isinstance(value['helpers'], dict) and {'scripts/pre_push_gate.py', 'scripts/zero_wait_gate.py', 'scripts/format_workspace.py'} <= set(value['helpers']), 'Verifier helpers are not pinned')
    for name, wanted in value['helpers'].items():
        require(hash_value(wanted) and digest(regular(root / relative(name))) == wanted, 'Producer helper changed')
    require(value['helpers']['scripts/zero_wait_gate.py'] == digest(Path(__file__).read_bytes()), 'Unknown zero-wait verifier')
    tools = value['tools']
    require(isinstance(tools, dict) and set(tools) == TOOLS, 'Exact eleven tools/plugins/runner required')
    for item in tools.values():
        require(isinstance(item, dict) and set(item) == {'path', 'sha256'} and hash_value(item['sha256'])
                and digest(regular(Path(item['path']), single_link=False)) == item['sha256'], 'Tool changed')
    native_bin = Path(tools['cargo']['path']).parent
    for key in ('rustc', 'rustdoc', 'cargo_fmt', 'cargo_clippy', 'rustfmt', 'clippy_driver'):
        require(Path(tools[key]['path']).parent == native_bin, 'Mixed/proxy Rust toolchain')
    isolation = value['isolation']
    require(set(isolation) == {'directory', 'files'} and set(isolation['files']) == ISOLATION_FILES, 'Closed isolation transport required')
    directory = Path(isolation['directory'])
    require(Path(tools['runner']['path']) == directory / 'runner.py', 'Runner is outside registered isolation')
    for name, wanted in isolation['files'].items():
        require(hash_value(wanted) and digest(regular(directory / name)) == wanted, 'Isolation helper changed')
    require(isinstance(value['commands'], dict) and set(MANDATORY) <= set(value['commands']) <= set(COMMANDS)
            and all(value['commands'][p] == list(COMMANDS[p]) for p in value['commands']), 'Unknown command profile')
    require(isinstance(value['forbidden_paths'], list) and all(isinstance(s, str) and s for s in value['forbidden_paths']), 'Explicit forbidden paths required')
    return value


def execution_env(root, tools):
    cpus = getattr(os, 'process_cpu_count', os.cpu_count)() or os.cpu_count()
    require(type(cpus) is int and cpus > 0 and set(tools) == TOOLS, 'Available CPUs/tools unavailable')
    python, runner = tools['python']['path'], tools['runner']['path']
    require(not any(c.isspace() for c in python + runner), 'Cargo runner transport needs unambiguous no-space paths')
    result = {name: None for name in ('RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS', 'RUSTDOCFLAGS',
              'RUSTC_WRAPPER', 'RUSTC_WORKSPACE_WRAPPER', 'RUSTUP_TOOLCHAIN', 'RUST_TEST_THREADS',
              'MAKEFLAGS', 'CARGO_MAKEFLAGS', 'NUM_JOBS', 'CLIPPY_ARGS', 'CLIPPY_CONF_DIR',
              'CLIPPY_DRIVER_PATH', 'RUSTC_REAL')}
    for name in os.environ:
        if name.upper().startswith(('CARGO_', 'RUST_', 'RUSTUP_', 'CLIPPY_')):
            result[name] = None
    result.update(CARGO_BUILD_JOBS=str(cpus), CARGO_TARGET_DIR=str(root / 'target'),
                  RUSTC=tools['rustc']['path'], RUSTDOC=tools['rustdoc']['path'],
                  CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER=tools['linker']['path'],
                  CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUNNER=python + ' -I -B ' + runner,
                  CARGO_UNSTABLE_CHECKSUM_FRESHNESS='true',
                  GIT_CEILING_DIRECTORIES=str(root / '.agent_tx/staging'))
    result['CARGO_HOME'] = str(Path(os.environ.get('CARGO_HOME', str(Path.home() / '.cargo'))).resolve())
    # Native Cargo plugins win over rustup proxies; no shell-dependent PATH tail.
    git_directory = Path(tools['git']['path']).parent
    paths = [Path(tools['cargo']['path']).parent, Path(python).parent,
             git_directory, git_directory.parent / 'usr/bin']
    if os.name == 'nt':
        system = Path(os.environ['SystemRoot'])
        paths += [system / 'System32', system, system / 'System32/Wbem']
    else:
        paths += [Path(p) for p in os.defpath.split(os.pathsep)]
    result['PATH'] = os.pathsep.join(dict.fromkeys(str(p) for p in paths))
    result['RUSTFMT'] = tools['rustfmt']['path']
    toolchain = Path(tools['rustc']['path']).parent.parent
    require(toolchain.parent.name == 'toolchains', 'Native rustup toolchain layout required')
    result['RUSTUP_TOOLCHAIN'] = toolchain.name
    result['RUSTUP_HOME'] = str(toolchain.parent.parent)
    return result


def git(root, index, tools, *arguments, data=None, allow=(0,)):
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith('GIT_')}
    env.update(GIT_INDEX_FILE=str(index), GIT_OPTIONAL_LOCKS='0', GIT_NO_LAZY_FETCH='1', GIT_TERMINAL_PROMPT='0')
    result = subprocess.run([tools['git']['path'], '--no-optional-locks', '-C', str(root), *arguments],
                            input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    require(result.returncode in allow, 'Git query failed: ' + arguments[0])
    return result.stdout


def tree_oid(files):
    """Git tree hashing without write-tree or modification of any index/object."""
    root = {}
    for name, row in files.items():
        parts = name.split('/'); node = root
        for part in parts[:-1]:
            require(part not in node or isinstance(node[part], dict), 'File/directory alias')
            node = node.setdefault(part, {})
        require(parts[-1] not in node, 'Duplicate tree name')
        node[parts[-1]] = (row['mode'], row['blob'])
    def encode(node):
        rows = []
        for name, value in node.items():
            directory = isinstance(value, dict)
            mode, oid = ('40000', encode(value)) if directory else value
            rows.append((name.encode() + (b'/' if directory else b''), mode.encode() + b' ' + name.encode() + b'\0' + bytes.fromhex(oid)))
        raw = b''.join(v for _, v in sorted(rows))
        return hashlib.sha1(b'tree ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
    return encode(root)


def tree_map(query, oid):
    result = {}
    for row in query('ls-tree', '-r', '-z', oid).split(b'\0'):
        if not row: continue
        header, name = row.split(b'\t', 1); mode, kind, blob = header.decode().split()
        name = relative(os.fsdecode(name))
        require(kind == 'blob' and mode in ('100644', '100755') and name not in result, 'Nonregular publication tree')
        result[name] = {'mode':mode, 'blob':blob}
    return dict(sorted(result.items()))


def content_settings(root, query):
    content = query('config', '--null', '--get-regexp', r'^(core\.(autocrlf|eol|attributesfile|hookspath)|filter\.)', allow=(0, 1))
    result = {'content_settings_sha256':digest(content)}
    paths = [Path(os.fsdecode(query('rev-parse', '--path-format=absolute', '--git-path', 'info/attributes')).strip())]
    configured = query('config', '--path', '--get', 'core.attributesfile', allow=(0, 1)).strip()
    if configured:
        path = Path(os.fsdecode(configured)); paths.append(path if path.is_absolute() else root / path)
    else:
        paths.append(Path(os.environ.get('XDG_CONFIG_HOME', str(Path.home() / '.config'))) / 'git/attributes')
    for path in paths:
        result['attributes:' + str(path)] = digest(regular(path)) if path.exists() else None
    return result


def projection(subject):
    keys = ('schema', 'root', 'branch', 'candidate_tree', 'files', 'raw', 'configs', 'git_settings', 'tools', 'producer', 'profile')
    return {**{k: subject[k] for k in keys}, 'private_index': subject['private_index']['path']}


def capture_subject(root, private_index, producer_reference, *, tools, generation):
    root = Path(root).resolve()
    review = registration(root, producer_reference)
    require(tools == review['tools'] and isinstance(generation, str) and generation, 'Capture producer/tools/generation differ')
    index = Path(private_index)
    if not index.is_absolute(): index = root / index
    require(index.parent == root / '.git' and re.fullmatch('index_[A-Za-z0-9_-]+', index.name), 'Explicit primary private index required')
    original_index = regular(index)
    source = regular(root / 'scripts/pre_push_gate.py')
    legacy = types.ModuleType('zero_wait_legacy_definitions'); legacy.__file__ = str(root / 'scripts/pre_push_gate.py')
    exec(compile(source, legacy.__file__, 'exec'), legacy.__dict__)
    query = lambda *a, **kw: git(root, index, tools, *a, **kw)
    require(query('branch', '--show-current').strip() == b'main' and Path(os.fsdecode(query('rev-parse', '--show-toplevel')).strip()).resolve() == root
            and Path(os.fsdecode(query('rev-parse', '--path-format=absolute', '--git-common-dir')).strip()).resolve() == root / '.git', 'Primary main required')
    def mapping():
        result = {}
        for line in query('ls-files', '--stage', '-z').split(b'\0'):
            if not line: continue
            header, name = line.split(b'\t', 1); mode, blob, slot = header.decode().split(); name = relative(os.fsdecode(name))
            require(slot == '0' and mode in ('100644', '100755') and re.fullmatch('[0-9a-f]{40}', blob) and name not in result, 'Unmerged/nonregular private index')
            result[name] = {'mode': mode, 'blob': blob}
        require(len(result) == len({n.casefold() for n in result}), 'Case-colliding index')
        flags = query('-c', 'core.fsmonitor=false', 'ls-files', '--debug', '-z')
        values = re.findall(rb'flags: ([0-9a-fA-F]+)', flags)
        require(len(values) == len(result) and all(int(v, 16) == 0 for v in values), 'Private index flags must all be zero')
        return dict(sorted(result.items()))
    files = mapping()
    raw = {n: digest(regular(root / n)) for n in files}
    def configuration_map():
        result = legacy.configurations(root)
        for parent in sorted({root, *root.parents, Path.home()}, key=str):
            for n in ('.rustfmt.toml', 'rustfmt.toml'):
                p = parent / n
                result[str(p)] = digest(regular(p)) if p.exists() else None
        # Bind the exact canonical Cargo home exported to the child, including
        # both present and absent config probes. Never inherit an uncaptured path.
        for n in ('config', 'config.toml'):
            p = Path(execution_env(root, tools)['CARGO_HOME']) / n
            result[str(p)] = digest(regular(p)) if p.exists() else None
        for name, wanted in result.items():
            require(wanted is None or digest(regular(Path(name))) == wanted, 'Configuration changed')
        return result
    configs, settings = configuration_map(), content_settings(root, query)
    attributes = query('check-attr', '-z', 'filter', '--stdin', data=b'\0'.join(os.fsencode(n) for n in files) + b'\0').split(b'\0')
    require(len(attributes) == 3 * len(files) + 1 and all(v in (b'unset', b'unspecified') for v in attributes[2::3]), 'Custom/unknown clean filters refused')
    mapped = query('hash-object', '--stdin-paths', data=''.join(json.dumps(n, ensure_ascii=False) + '\n' for n in files).encode()).splitlines()
    require(len(mapped) == len(files) and all(blob.decode() == files[n]['blob'] for n, blob in zip(files, mapped)), 'RAW does not map to candidate private index')
    untracked = [relative(os.fsdecode(n)) for n in query('ls-files', '--others', '--exclude-standard', '-z').split(b'\0') if n]
    require(not any(Path(n).suffix in ('.rs', '.toml', '.lock') or n.startswith('.cargo/') for n in untracked), 'Untracked build input')
    require(files == mapping() and raw == {n: digest(regular(root / n)) for n in files}
            and settings == content_settings(root, query) and configs == configuration_map(), 'Observed input drift during capture')
    registration(root, producer_reference)
    value = {'schema': SUBJECT, 'state': 'CAPTURED_NOT_VALIDATED', 'root': str(root), 'branch': 'main',
             'baseline_head': query('rev-parse', 'HEAD').strip().decode(), 'candidate_tree': tree_oid(files),
             'private_index': {'path': index.relative_to(root).as_posix(), 'before_raw_sha256': digest(original_index),
                               'after_raw_sha256': digest(regular(index))},
             'files': files, 'raw': raw, 'configs': configs, 'git_settings': settings, 'tools': tools,
             'producer': producer_reference, 'profile': execution_env(root, tools), 'generation': generation}
    value['inputs_digest'] = digest(canonical(projection(value)))
    return value


def runner_receipts(root, producer_reference):
    root = Path(root).resolve(); review = registration(root, producer_reference)
    directory = Path(review['isolation']['directory']) / 'runs'
    result = []
    if directory.exists():
        for path in sorted(directory.glob('*/runner-receipt.json')):
            raw = regular(path); result.append({'path': str(path), 'sha256': digest(raw)})
    return result


class ReadPin:
    """Read-only Win32 handle: protect candidate copies, never Root sources."""
    def __init__(self, path, wanted, *, single_link=True):
        require(os.name == 'nt', 'Publication candidate protection requires native Windows')
        self.path = Path(path); self.single_link = single_link
        require(digest(regular(self.path, single_link=single_link)) == wanted, 'Pin source changed')
        self.api = ctypes.WinDLL('kernel32', use_last_error=True)
        self.api.CreateFileW.argtypes = [w.LPCWSTR, w.DWORD, w.DWORD, ctypes.c_void_p, w.DWORD, w.DWORD, w.HANDLE]
        self.api.CreateFileW.restype = w.HANDLE
        self.api.CloseHandle.argtypes = [w.HANDLE]; self.api.CloseHandle.restype = w.BOOL
        self.handle = self.api.CreateFileW(str(self.path), 0x80000000, 1, None, 3, 0x80, None)
        require(self.handle not in (None, ctypes.c_void_p(-1).value), 'Cannot protect candidate input: ' + str(self.path))
        try:
            require(digest(regular(self.path, single_link=single_link)) == wanted, 'Input changed while acquiring READ pin')
        except BaseException:
            self.close(); raise
        self.sha256 = wanted

    def close(self):
        if self.handle is not None:
            handle = self.handle; self.handle = None
            require(self.api.CloseHandle(handle), 'READ pin release uncertain')


def candidate_command(phase, slot, configs):
    require(phase in COMMANDS, 'Unknown publication phase')
    require(phase != 'fmt', 'Formatting requires the actual Python helper argv')
    base = COMMANDS[phase]
    extra = ['--manifest-path', str(slot / 'Cargo.toml')]
    for config in configs: extra += ['--config', config]
    return (base[0], *extra, *base[1:])


def candidate_argv(phase, slot, configs, tools):
    if phase != 'fmt':
        return [tools['cargo']['path'], *candidate_command(phase, slot, configs)]
    result = [tools['python']['path'], '-I', '-B', str(slot / 'scripts/format_workspace.py'),
              '--slot', str(slot), '--cargo', tools['cargo']['path'], '--cargo-sha256', tools['cargo']['sha256'],
              '--rustfmt', tools['rustfmt']['path'], '--rustfmt-sha256', tools['rustfmt']['sha256'],
              '--config-path', str(slot / '.zero-wait-rustfmt.toml')]
    for config in configs: result += ['--cargo-config', config]
    return result


@contextmanager
def prepare_candidate(root, private_index, producer_reference, *, tools, generation, subject):
    """One persistent source slot. A single service updates it only between passes.

    External Cargo configs/include graphs are deliberately unsupported in v1.
    They are rejected, not silently omitted or claimed immutable.
    """
    root = Path(root).resolve(); checked_subject(root, subject)
    require(capture_subject(root, private_index, producer_reference, tools=tools, generation=generation)['inputs_digest'] == subject['inputs_digest'], 'STALE before candidate copy')
    slot = root / '.agent_tx/staging/compiler-candidate'
    slot.mkdir(parents=True, exist_ok=True)
    require(slot.resolve() == slot and not slot.is_symlink() and not getattr(slot.lstat(), 'st_file_attributes', 0) & 0x400, 'Invalid candidate directory')
    for parent in slot.parents:
        require(not parent.is_symlink() and not getattr(parent.lstat(), 'st_file_attributes', 0) & 0x400, 'Candidate ancestor reparse')
    marker = slot / '.candidate.json'; previous = {}
    if marker.exists():
        previous = read_json(regular(marker))
        require(set(previous) == {'schema', 'root', 'files', 'candidate_id'} and previous['schema'] == 'trader-gemini-compiler-candidate-v1'
                and previous['root'] == str(root), 'Foreign candidate slot')
        previous = previous['files']
    else:
        require(not any(slot.iterdir()), 'Unowned candidate content')
    for name, wanted in previous.items():
        if name not in subject['raw']:
            path = slot / relative(name); require(digest(regular(path)) == wanted, 'Previous candidate changed')
            path.unlink()
    copied = {}
    for name, wanted in subject['raw'].items():
        name = relative(name)
        require(not name.startswith(('target/', 'logs/', '.agent_tx/')), 'Non-source path cannot enter candidate')
        data = regular(root / name); require(digest(data) == wanted, 'STALE source while copying candidate')
        path = slot / name; path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or regular(path) != data:
            temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
            with temporary.open('xb') as stream:
                stream.write(data); stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, path)
        require(digest(regular(path)) == wanted and path.stat().st_nlink == 1, 'Candidate copy differs/uses hardlinks')
        copied[name] = wanted
    configs = []
    for name, wanted in subject['configs'].items():
        if wanted is None: continue
        source = Path(name)
        require(source.parent == root / '.cargo' and source.relative_to(root).as_posix() in copied,
                'External Cargo config unsupported; explicit reviewed closure required')
        parsed = tomllib.loads(regular(source).decode())
        require(set(parsed) <= {'build', 'unstable'} and set(parsed.get('build', {})) <= {'rustflags'}
                and set(parsed.get('unstable', {})) <= {'checksum-freshness'}, 'Cargo config graph/path overrides need explicit closure')
        configs.append(str(slot / source.relative_to(root)))
    require('.zero-wait-rustfmt.toml' not in copied, 'Reserved compiler config path')
    format_path = slot / '.zero-wait-rustfmt.toml'
    if format_path.exists(): require(regular(format_path) == b'', 'Foreign formatter config')
    else:
        with format_path.open('xb') as stream:
            stream.flush(); os.fsync(stream.fileno())
    # Reject local manifest inputs which escape the copied project.
    def paths(value, parent):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == 'path' and isinstance(item, str):
                    require((parent / item).resolve().is_relative_to(slot), 'External local dependency/path is not captured')
                paths(item, parent)
        elif isinstance(value, list):
            for item in value: paths(item, parent)
    for name in copied:
        if name == 'Cargo.toml' or name.endswith('/Cargo.toml'):
            paths(tomllib.loads(regular(slot / name).decode()), (slot / name).parent)
    cwd = Path(tempfile.gettempdir()).resolve() / ('tg-compiler-cwd-' + digest(str(root).encode()))
    cwd.mkdir(exist_ok=True)
    require(cwd.resolve() == cwd and not any(cwd.iterdir()), 'External Cargo cwd must stay empty')
    for parent in (cwd, *cwd.parents):
        for config in ('config', 'config.toml'):
            require(not (parent / '.cargo' / config).exists(), 'External cwd Cargo configuration is not captured')
    if (slot / '.cargo/config').exists():
        configs = [str(slot / '.cargo/config')]
    state = {'schema': 'trader-gemini-compiler-candidate-v1', 'root': str(root), 'files': copied, 'candidate_id': subject['inputs_digest']}
    temp_marker = slot / ('.candidate.' + uuid.uuid4().hex + '.tmp')
    with temp_marker.open('xb') as stream:
        stream.write(canonical(state) + b'\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp_marker, marker)
    actual_files = set()
    for directory, directories, names in os.walk(slot, followlinks=False):
        for child in directories:
            path = Path(directory) / child; info = path.lstat()
            require(not path.is_symlink() and not getattr(info, 'st_file_attributes', 0) & 0x400, 'Candidate directory reparse')
        for name in names:
            path = Path(directory) / name; regular(path)
            actual_files.add(path.relative_to(slot).as_posix())
    require(actual_files == set(copied) | {'.candidate.json', '.zero-wait-rustfmt.toml'}, 'Unowned candidate input')
    require(capture_subject(root, private_index, producer_reference, tools=tools, generation=generation)['inputs_digest'] == subject['inputs_digest'], 'STALE after candidate copy')
    pins = []
    class Candidate:
        def command(self, phase): return candidate_command(phase, slot, configs)
        def argv(self, phase): return candidate_argv(phase, slot, configs, tools)
        def proof(self):
            require(pins and all(digest(regular(pin.path, single_link=pin.single_link)) == pin.sha256 for pin in pins), 'Protected candidate changed')
            return {'schema': 'trader-gemini-protected-compiler-candidate-v1', 'root': str(root), 'slot': str(slot),
                    'cwd': str(cwd), 'candidate_id': subject['inputs_digest'], 'tree': subject['candidate_tree'],
                    'files': copied, 'configs': configs, 'pin_count': len(pins), 'protection_active': True,
                    'format_config': {'path':str(format_path), 'sha256':digest(b'')},
                    'tools': tools, 'config_policy': 'ROOT_BUILD_UNSTABLE_ONLY_EXTERNAL_CONFIGS_ABSENT',
                    'limits': 'Protects copied source and declared tools/isolation. Not a universal claim about external compiler caches or host inputs.'}
    candidate = Candidate(); candidate.cwd = cwd
    try:
        for name, wanted in copied.items(): pins.append(ReadPin(slot / name, wanted))
        pins.append(ReadPin(format_path, digest(b'')))
        for item in tools.values(): pins.append(ReadPin(Path(item['path']), item['sha256'], single_link=False))
        review = registration(root, producer_reference)
        for name, wanted in review['isolation']['files'].items():
            path = Path(review['isolation']['directory']) / name
            if not any(pin.path == path for pin in pins): pins.append(ReadPin(path, wanted))
        yield candidate
    finally:
        errors = []
        for pin in reversed(pins):
            try: pin.close()
            except Exception as error: errors.append(str(error))
        require(not errors, 'Candidate READ pin cleanup uncertain: ' + '; '.join(errors))


def checked_subject(root, subject):
    keys = {'schema', 'state', 'root', 'branch', 'baseline_head', 'candidate_tree', 'private_index',
            'files', 'raw', 'configs', 'git_settings', 'tools', 'producer', 'profile', 'generation', 'inputs_digest'}
    require(isinstance(subject, dict) and set(subject) == keys and subject['schema'] == SUBJECT
            and subject['state'] == 'CAPTURED_NOT_VALIDATED' and subject['root'] == str(root)
            and subject['branch'] == 'main' and hash_value(subject['inputs_digest'])
            and subject['inputs_digest'] == digest(canonical(projection(subject))), 'Unknown/changed zero-wait subject')
    require(set(subject['private_index']) == {'path', 'before_raw_sha256', 'after_raw_sha256'}
            and all(hash_value(subject['private_index'][n]) for n in ('before_raw_sha256', 'after_raw_sha256')),
            'Private index observations missing')
    require(subject['candidate_tree'] == tree_oid(subject['files']) and set(subject['raw']) == set(subject['files'])
            and all(hash_value(v) for v in subject['raw'].values()), 'Subject tree/RAW differs')
    return subject


def validate_runner(root, references, review, executables):
    require(isinstance(references, list) and references, 'Actual isolated test executions required')
    found = set(); directory = Path(review['isolation']['directory'])
    for reference in references:
        require(set(reference) == {'path', 'sha256'} and hash_value(reference['sha256']), 'Closed runner reference required')
        path = Path(reference['path'])
        require(path.name == 'runner-receipt.json' and path.parent.parent == directory / 'runs', 'Runner receipt outside registered runs')
        raw = regular(path); require(digest(raw) == reference['sha256'], 'Runner receipt changed')
        row = read_json(raw); sandbox = Path(row['sandbox']); executable = Path(row['executable'])
        require(set(row) == {'executable', 'arguments', 'original_cwd', 'sandbox', 'roots', 'fixture_manifest_sha256',
                'runner_sha256', 'fixture_files', 'removed_credential_or_promotion_names', 'exit_code'}
                and row['exit_code'] == 0 and type(row['exit_code']) is int and row['arguments'] == []
                and str(executable) in executables and Path(row['original_cwd']) == executables[str(executable)]
                and sandbox == path.parent
                and executable.is_absolute() and executable.resolve() == executable and executable.is_relative_to(root / 'target'),
                'Isolated test execution differs')
        require(row['runner_sha256'] == review['tools']['runner']['sha256']
                and row['fixture_manifest_sha256'] == review['isolation']['files']['fixture_manifest.json']
                and row['roots'] == {'TGM_DATA_DIR': str(sandbox / 'data'), 'TGM_CONFIG_DIR': str(sandbox / 'config_dir'),
                                     'TG_CONFIG_ROOT': str(sandbox / 'config_dir')}, 'Runner roots/fixture identity differs')
        fixture = read_json(regular(directory / 'fixture_manifest.json'))
        require(row['fixture_files'] == {r['path']: r['sha256'] for r in fixture['files']}, 'Runner fixture set differs')
        require(str(executable) not in found, 'Duplicate test executable receipt'); found.add(str(executable))
    require(found == set(executables), 'Cargo executable inventory differs from isolated executions')


def validate_candidate(root, subject, candidate, released, cleanup_error):
    require(isinstance(candidate, dict) and set(candidate) == {'schema', 'root', 'slot', 'cwd', 'candidate_id',
            'tree', 'files', 'configs', 'format_config', 'pin_count', 'protection_active', 'tools', 'config_policy', 'limits'}
            and candidate['schema'] == 'trader-gemini-protected-compiler-candidate-v1' and candidate['root'] == str(root)
            and candidate['slot'] == str(root / '.agent_tx/staging/compiler-candidate')
            and candidate['candidate_id'] == subject['inputs_digest'] and candidate['tree'] == subject['candidate_tree']
            and candidate['files'] == subject['raw'] and candidate['tools'] == subject['tools']
            and candidate['format_config'] == {'path':str(root / '.agent_tx/staging/compiler-candidate/.zero-wait-rustfmt.toml'), 'sha256':digest(b'')}
            and candidate['protection_active'] is True and type(candidate['pin_count']) is int
            and candidate['pin_count'] >= len(subject['raw']) + len(TOOLS)
            and released is True and cleanup_error is None,
            'Publication requires a protected immutable candidate and certain cleanup')
    expected_configs = [str(Path(candidate['slot']) / Path(p).relative_to(root))
                        for p, sha in subject['configs'].items() if sha is not None]
    if str(Path(candidate['slot']) / '.cargo/config') in expected_configs:
        expected_configs = [str(Path(candidate['slot']) / '.cargo/config')]
    require(Path(candidate['cwd']) == Path(tempfile.gettempdir()).resolve() / ('tg-compiler-cwd-' + digest(str(root).encode()))
            and not Path(candidate['cwd']).is_relative_to(root) and candidate['configs'] == expected_configs
            and candidate['config_policy'] == 'ROOT_BUILD_UNSTABLE_ONLY_EXTERNAL_CONFIGS_ABSENT',
            'Candidate execution/config paths differ')
    return candidate


def validate_format(log, subject, candidate, review):
    """Independent closure/argv checker. Never imports the formatter helper.

    Metadata bytes and child results are evidence from the pinned helper. Target
    closure and bounded argv are recomputed here, not accepted from its summary.
    """
    report = read_json(log)
    keys = {'schema', 'slot', 'cwd', 'config', 'tools', 'cargo_configs', 'metadata',
            'inventory', 'batches', 'raw_before', 'raw_after', 'status', 'exit_code', 'error'}
    require(isinstance(report, dict) and set(report) == keys and report['schema'] == 'trader-gemini-workspace-format-v1'
            and report['status'] == 'PASSED' and type(report['exit_code']) is int and report['exit_code'] == 0
            and report['error'] is None, 'Incomplete/failed workspace formatter')
    slot = Path(candidate['slot']); config = str(slot / '.zero-wait-rustfmt.toml')
    require(report['slot'] == str(slot) and report['cwd'] == candidate['cwd'] and report['config'] == config
            and report['cargo_configs'] == candidate['configs']
            and report['tools'] == {key: subject['tools'][key] for key in ('cargo', 'rustfmt')}, 'Formatter profile differs')
    helper = 'scripts/format_workspace.py'
    require(hash_value(review['helpers'].get(helper)) and review['helpers'][helper] == subject['raw'].get(helper), 'Formatter helper is not protected/pinned')
    marker = {'schema': 'trader-gemini-compiler-candidate-v1', 'root': candidate['root'],
              'files': subject['raw'], 'candidate_id': subject['inputs_digest']}
    expected_raw = {**subject['raw'], '.zero-wait-rustfmt.toml': digest(b''),
                    '.candidate.json': digest(canonical(marker) + b'\n')}
    require(report['raw_before'] == expected_raw == report['raw_after'], 'Formatter RAW set differs/changed')

    def input_path(value):
        require(isinstance(value, str), 'Metadata path must be text')
        path = Path(value)
        require(path.is_absolute(), 'Formatter path must be absolute')
        # Cargo preserves legal ../ aliases in explicit target paths. Normalize
        # lexically; validation must not reread a later live Slot to resolve it.
        path = Path(os.path.normpath(value))
        require(path.is_absolute() and path.is_relative_to(slot), 'Formatter path escapes slot')
        name = path.relative_to(slot).as_posix()
        relative(name)
        require(name in subject['raw'], 'Formatter input is not protected source')
        return str(path)

    def command_units(argv):
        require(isinstance(argv, list) and argv and all(isinstance(s, str) for s in argv), 'Invalid formatter child argv')
        return len(subprocess.list2cmdline(argv).encode('utf-16-le')) // 2 + 1

    def child(record, argv):
        require(isinstance(record, dict) and set(record) == {'argv', 'cwd', 'exit_code', 'error',
                'stdout_b64', 'stderr_b64', 'stdout_sha256', 'stderr_sha256'}
                and record['argv'] == argv and record['cwd'] == candidate['cwd']
                and command_units(argv) <= 32767 and type(record['exit_code']) is int
                and record['exit_code'] == 0 and record['error'] is None, 'Formatter child argv/status differs')
        outputs = []
        for key in ('stdout', 'stderr'):
            require(isinstance(record[key + '_b64'], str), 'Invalid child output transport')
            raw = base64.b64decode(record[key + '_b64'], validate=True)
            require(digest(raw) == record[key + '_sha256'], 'Formatter child output changed')
            outputs.append(raw)
        return outputs[0]

    require(isinstance(report['metadata'], list) and isinstance(report['batches'], list), 'Missing actual formatter child records')
    pending = {input_path(str(slot / 'Cargo.toml'))}; manifests = set(); targets = set(); position = 0
    while pending:
        manifest = min(pending); pending.remove(manifest)
        if manifest in manifests: continue
        require(position < len(report['metadata']), 'Local dependency metadata omitted')
        argv = [subject['tools']['cargo']['path'], 'metadata', '--locked', '--format-version', '1', '--no-deps', '--manifest-path', manifest]
        for cargo_config in candidate['configs']: argv += ['--config', cargo_config]
        metadata = read_json(child(report['metadata'][position], argv)); position += 1
        require(metadata.get('version') == 1 and isinstance(metadata.get('packages'), list) and metadata['packages'], 'Unknown local metadata')
        returned = set()
        for package in metadata['packages']:
            require(package['source'] is None, 'Nonlocal no-deps package')
            returned.add(input_path(package['manifest_path']))
            require(isinstance(package['targets'], list) and isinstance(package['dependencies'], list), 'Metadata target/dependency lists missing')
            for target in package['targets']:
                path = input_path(target['src_path']); edition = target['edition']
                require(edition in ('2015', '2018', '2021', '2024'), 'Unknown target edition')
                targets.add((path, edition))
            for dependency in package['dependencies']:
                if dependency.get('path') is not None:
                    pending.add(input_path(str(Path(dependency['path']) / 'Cargo.toml')))
        workspace_manifest = input_path(str(Path(metadata['workspace_root']) / 'Cargo.toml'))
        require(manifest in returned or manifest == workspace_manifest, 'Requested local manifest absent')
        manifests.update(returned | {manifest}); pending.difference_update(manifests)
    require(position == len(report['metadata']), 'Unexpected metadata invocation')
    inventory = [{'path': path, 'edition': edition} for path, edition in sorted(targets)]
    require(inventory and report['inventory'] == inventory, 'Formatter target inventory incomplete/different')
    commands = []
    for edition in sorted({edition for path, edition in targets}):
        prefix = [subject['tools']['rustfmt']['path'], '--edition', edition, '--config-path', config, '--check']
        batch = []
        for path in sorted(path for path, value in targets if value == edition):
            require(command_units([*prefix, path]) <= 32767, 'Single formatter target exceeds Windows limit')
            if batch and command_units([*prefix, *batch, path]) > 32767:
                commands.append([*prefix, *batch]); batch = []
            batch.append(path)
        if batch: commands.append([*prefix, *batch])
    require(len(report['batches']) == len(commands), 'Formatter target batch omitted/added')
    for record, argv in zip(report['batches'], commands): child(record, argv)
    return report


def validate_run(root, subject, phase, reference, review):
    row = read_json(artifact(root, reference))
    keys = {'schema', 'generation', 'request_ids', 'kind', 'phase', 'feedback_only', 'state', 'argv', 'cwd',
            'env_profile', 'available_cpus', 'before', 'after', 'before_inputs_digest', 'after_inputs_digest',
            'log_ref', 'exit_code', 'error', 'diagnostics_by_file', 'runner_receipts', 'observed_utc',
            'producer_reference', 'writer', 'candidate', 'candidate_released', 'candidate_cleanup_error', 'test_counts'}
    require(set(row) == keys and row['schema'] == RECEIPT and row['kind'] == 'publication-phase'
            and row['feedback_only'] is False and row['state'] == 'PASSED' and row['phase'] == phase
            and type(row['exit_code']) is int and row['exit_code'] == 0 and row['error'] is None,
            'Failed/stale/diagnostic/legacy receipt is not publication evidence')
    candidate = validate_candidate(root,subject,row['candidate'],row['candidate_released'],row['candidate_cleanup_error'])
    require(row['producer_reference'] == subject['producer'] and row['cwd'] == candidate['cwd']
            and row['argv'] == candidate_argv(phase, Path(candidate['slot']), candidate['configs'], subject['tools'])
            and row['env_profile'] == subject['profile'] and type(row['available_cpus']) is int
            and str(row['available_cpus']) == row['env_profile']['CARGO_BUILD_JOBS'], 'Receipt command/profile differs')
    writer = row['writer']; require(set(writer) == {'path', 'sha256'}
            and review['helpers'].get(writer['path']) == writer['sha256'], 'Unregistered receipt writer')
    artifact(root, writer)
    require(isinstance(row['request_ids'], list) and row['request_ids'] and len(set(row['request_ids'])) == len(row['request_ids'])
            and all(isinstance(s, str) and s for s in row['request_ids']), 'No exact requests')
    for key in ('before', 'after'):
        snapshot = checked_subject(root, read_json(artifact(root, row[key])))
        require(snapshot['generation'] == row['generation'] and snapshot['inputs_digest'] == subject['inputs_digest']
                and row[key + '_inputs_digest'] == subject['inputs_digest'], 'Run captured another input/generation')
    log = artifact(root, row['log_ref']).decode('utf-8', 'replace')
    if phase == 'fmt': validate_format(log, subject, candidate, review)
    require(log.strip() and not re.search(r'(?m)^test .* \.\.\. FAILED', log), 'Failed test output')
    summaries = re.findall(r'(?m)^test result: (\w+)\. ([0-9]+) passed; ([0-9]+) failed; ([0-9]+) ignored; ([0-9]+) measured; ([0-9]+) filtered out', log)
    counts = ({'suites': len(summaries), **{name: sum(int(row[i + 1]) for row in summaries)
              for i, name in enumerate(('passed', 'failed', 'ignored', 'measured', 'filtered_out'))}} if summaries else {})
    require(isinstance(row['test_counts'], dict) and all(type(v) is int and v >= 0 for v in row['test_counts'].values())
            and row['test_counts'] == counts and all(v[0] == 'ok' and int(v[2]) == 0 for v in summaries), 'Actual test counts/status differ')
    executables = {}
    for line in log.splitlines():
        try: message = json.loads(line)
        except ValueError: continue
        if isinstance(message, dict) and message.get('reason') == 'compiler-artifact' and message.get('profile', {}).get('test') is True:
            executable = message.get('executable')
            if executable:
                manifest = Path(message.get('manifest_path', ''))
                require(manifest.is_absolute() and manifest.is_relative_to(Path(candidate['slot']))
                        and manifest.relative_to(Path(candidate['slot'])).as_posix() in subject['raw']
                        and manifest.name == 'Cargo.toml', 'Cargo artifact manifest is not protected candidate input')
                name = str(Path(executable)); require(name not in executables or executables[name] == manifest.parent, 'Ambiguous test executable package')
                executables[name] = manifest.parent
        if isinstance(message, dict) and message.get('reason') == 'compiler-message':
            require(message.get('message', {}).get('level') != 'error', 'Compiler error in zero-exit receipt log')
    if phase in ('tests', 'daz-tests', 'storage-tests', 'omni-tests', 'semantic-tests'):
        require(executables and re.search(r'test result: ok\. [0-9]+ passed; 0 failed;', log), 'No terminal actual test executions')
        validate_runner(root, row['runner_receipts'], review, executables)
    else: require(row['runner_receipts'] == [], 'Unexpected test runner evidence')
    return row, log


def validate_python(root, subject, references, review):
    suites = review['python_suites']
    require(isinstance(suites, dict) and set(suites) == {'legacy', 'zero_wait', 'bus'}
            and isinstance(references, dict) and set(references) == set(suites), 'Legacy30/new gate/bus contracts all required')
    for name, reference in references.items():
        row = read_json(artifact(root, reference)); spec = suites[name]
        require(set(spec) == {'command', 'tests_run'} and type(spec['tests_run']) is int and spec['tests_run'] > 0
                and (name != 'legacy' or spec['tests_run'] == 30), 'Invalid Python suite registration')
        require(set(row) == {'schema', 'command', 'inputs_digest', 'before_inputs_digest', 'after_inputs_digest',
                    'exit_code', 'tests_run', 'failures', 'errors', 'skipped', 'log_ref',
                    'candidate', 'candidate_released', 'candidate_cleanup_error'}
                and row['schema'] == 'trader-gemini-zero-wait-python-v1' and row['command'] == spec['command']
                and all(row[k] == subject['inputs_digest'] for k in ('inputs_digest', 'before_inputs_digest', 'after_inputs_digest'))
                and all(type(row[k]) is int and row[k] == 0 for k in ('exit_code', 'failures', 'errors', 'skipped'))
                and type(row['tests_run']) is int and row['tests_run'] == spec['tests_run'], 'Python contracts did not pass for this input')
        candidate = validate_candidate(root,subject,row['candidate'],row['candidate_released'],row['candidate_cleanup_error'])
        command = row['command']; slot = Path(candidate['slot'])
        require(isinstance(command,list) and len(command) >= 4 and command[:3] == [subject['tools']['python']['path'], '-I', '-B']
                and isinstance(command[3],str) and Path(command[3]).is_absolute() and Path(command[3]).is_relative_to(slot), 'Python must execute the protected candidate copy')
        source = Path(command[3]).relative_to(slot).as_posix()
        require(source in review['helpers'] and review['helpers'][source] == subject['raw'].get(source), 'Python suite source not pinned')
        log = artifact(root, row['log_ref']).decode('utf-8')
        require(re.search(r'(?m)^Ran ' + str(spec['tests_run']) + r' tests in [0-9.]+s\r?$', log)
                and log.rstrip().endswith('\nOK'), 'Nonterminal Python contract output')


def validate_golden(root, subject, reference, review, tests_reference, tests_row, tests_log):
    value = read_json(artifact(root, reference)); inventory = read_json(artifact(root, review['golden_inventory']))
    require(inventory['schema'] == 'codex-required-pr65-golden9-read-only-v1' and inventory['expected_test_count'] == 9
            and inventory['require_no_ignored_or_filtered_cases'] is True, 'Unknown Golden inventory')
    names = []
    for suite in inventory['suites']:
        require(suite['source'] in GOLDEN and suite['test_names'] == list(GOLDEN[suite['source']])
                and suite['sha256'] == subject['raw'].get(suite['source']), 'Golden source/name inventory changed')
        names.extend(suite['test_names'])
    require(len(names) == len(set(names)) == 9 and set(value) == {'schema', 'inputs_digest', 'tests_receipt', 'tests_log', 'inventory', 'tests', 'passed'}
            and value['schema'] == 'trader-gemini-zero-wait-golden9-v1' and value['inputs_digest'] == subject['inputs_digest']
            and value['tests_receipt'] == tests_reference and value['tests_log'] == tests_row['log_ref']
            and value['inventory'] == review['golden_inventory'] and value['tests'] == names and value['passed'] is True, 'Golden artifact is not this actual tests phase')
    for name in names:
        require(len(re.findall(r'(?m)^test (?:[A-Za-z0-9_:]+::)?' + re.escape(name) + r' \.\.\. ok\r?$', tests_log)) == 1,
                'Golden test missing/ignored/duplicate: ' + name)


def classifications(root, subject, review, required_rust, successful_phases, query):
    value = read_json(artifact(root, review['classifications']))
    require(set(value) == {'schema', 'status', 'reviewer', 'review_reference', 'baseline_commit', 'changes'}
            and value['schema'] == 'trader-gemini-zero-wait-rust-review-v1' and value['status'] == 'REVIEWED'
            and value['reviewer'] == 'ROOT' and isinstance(value['review_reference'], str) and value['review_reference'].strip(),
            'Explicit ROOT per-Rust review required; no crate exemptions')
    baseline = value['baseline_commit']; require(re.fullmatch('[0-9a-f]{40}', baseline) is not None, 'Unknown review baseline')
    before = {}
    for row in query('ls-tree', '-r', '-z', baseline).split(b'\0'):
        if row:
            header, name = row.split(b'\t', 1); mode, kind, blob = header.decode().split()
            require(kind == 'blob', 'Nonregular baseline'); before[os.fsdecode(name)] = {'mode': mode, 'blob': blob}
    changed = {n for n in set(before) | set(subject['files']) if before.get(n) != subject['files'].get(n) and n.endswith('.rs')}
    require(required_rust <= changed, 'Review baseline does not cover published Rust delta')
    records = value['changes']; require(isinstance(records, list) and {r['path'] for r in records} == changed
            and len(records) == len(changed), 'Missing/duplicate Rust classification')
    for row in records:
        require(set(row) == {'path', 'before', 'after', 'kind', 'review_reference', 'tests', 'provenance'}
                and row['kind'] in {'SEMANTIC', 'SEMANTIC_THEN_FORMAT', 'SEMANTIC_LINT_REVIEWED', 'TEST_ONLY', 'COMMENT_ONLY', 'PINNED_FORMAT_ONLY'}
                and isinstance(row['review_reference'], str) and row['review_reference'].strip()
                and isinstance(row['tests'], list) and row['tests'] and set(row['tests']) <= successful_phases,
                'Unreviewed/untested Rust classification')
        name = relative(row['path']); expected_after = subject['files'].get(name)
        require(row['before'] == before.get(name, 'ABSENT')
                and row['after'] == ({**expected_after, 'sha256': subject['raw'][name]} if expected_after else 'ABSENT'), 'Rust before/after tuple differs')
        if row['kind'] == 'PINNED_FORMAT_ONLY':
            provenance = read_json(artifact(root, row['provenance']))
            matches = [v for v in provenance.get('rows', []) if v.get('row', {}).get('path') == name]
            require(len(matches) == 1, 'No exact historical formatter row')
            old = matches[0]; fmt = old['row']
            require(fmt['mode'] == expected_after['mode'] and fmt['baseline_blob'] == row['before']['blob']
                    and fmt['after_blob'] == expected_after['blob'] and fmt['after_sha256'] == subject['raw'][name]
                    and old['format_exit_code'] == old['check_exit_code'] == 0, 'Historical format provenance differs; never reuse its subject/result')
            if fmt['god_exception'] is not None:
                exception = read_json(artifact(root, fmt['god_exception']))
                require(exception.get('schema') != 'trader-gemini-omni-two-lints-v1', 'Omni semantic lint is not pure format')
        else:
            require(row['provenance'] is None or isinstance(row['provenance'], dict), 'Invalid provenance reference')
    return baseline


def valid_main_update(row):
    return (len(row) == 4 and row[2] == 'refs/heads/main'
            and (row[0] == 'refs/heads/main' or row[0] == row[1])
            and all(re.fullmatch('[0-9a-f]{40}', value) and value != '0' * 40
                    for value in (row[1], row[3])))


def verify(root, attestation, stdin, remote_name, legacy):
    root = Path(root).resolve()
    require(remote_name == 'origin' and isinstance(attestation, dict) and set(attestation) ==
            {'schema', 'subject', 'runs', 'python_contracts', 'golden9'} and attestation['schema'] == ENVELOPE,
            'Unknown zero-wait envelope')
    subject = checked_subject(root, attestation['subject']); review = registration(root, subject['producer'])
    index = root / subject['private_index']['path']; tools = subject['tools']
    query = lambda *a, **kw: git(root, index, tools, *a, **kw)
    current = capture_subject(root, index, subject['producer'], tools=tools, generation=subject['generation'])
    require(current['inputs_digest'] == subject['inputs_digest'], 'STALE current source/private index')
    require(query('rev-parse', '--show-object-format').strip() == b'sha1', 'Only measured SHA1 repository profile supported')
    updates = []; rust = set(); published = set()
    for line in stdin.splitlines():
        row = line.split()
        require(valid_main_update(row), 'Malformed/non-main publication update')
        require(query('rev-parse', row[1]+'^{commit}').strip().decode() == row[1], 'Publication OID is not a commit')
        updates.append(row)
    require(len(updates) == 1, 'One explicit main publication required')
    forbidden = tuple(s.casefold() for s in (*FORBIDDEN, *review['forbidden_paths']))
    for _, local, _, remote in updates:
        if set(local) == {'0'}: continue
        require(local == query('rev-parse', 'HEAD').strip().decode(), 'Published OID is not current HEAD')
        require(query('rev-parse', local + '^{tree}').strip().decode() == subject['candidate_tree']
                and tree_map(query, local) == subject['files'], 'Publication differs from validated candidate tree')
        require(set(remote) != {'0'}, 'Zero-wait profile requires an existing remote baseline')
        query('merge-base', '--is-ancestor', remote, local)
        revisions = [local, '^' + remote]
        for commit in query('rev-list', *revisions).splitlines():
            for name in tree_map(query, commit.decode()):
                require(not legacy['any_glob'](name.casefold(), forbidden), 'Forbidden path in publication history')
        delta = {relative(os.fsdecode(n)) for n in query('diff-tree','--no-commit-id','--name-only','-r','-z',remote,local).split(b'\0') if n}
        published |= delta; rust |= {n for n in delta if n.endswith('.rs')}
    require(published, 'No actual publication delta')
    runs = attestation['runs']; require(isinstance(runs, dict) and set(MANDATORY) <= set(runs) <= set(review['commands']), 'Six current phases required')
    validated = {p: validate_run(root, subject, p, r, review) for p, r in runs.items()}
    baseline = classifications(root, subject, review, rust, set(validated), query)
    for _, local, _, remote in updates:
        if set(local) != {'0'}: query('merge-base', '--is-ancestor', baseline, remote)
    validate_python(root, subject, attestation['python_contracts'], review)
    validate_golden(root, subject, attestation['golden9'], review, runs['tests'], *validated['tests'])
    final = capture_subject(root, index, subject['producer'], tools=tools, generation=subject['generation'])
    require(final['inputs_digest'] == subject['inputs_digest'], 'STALE inputs during publication verification')
    return 'LOCAL_ZERO_WAIT_VALIDATED_CI_PENDING'
