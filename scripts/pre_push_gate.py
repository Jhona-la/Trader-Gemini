"""Verify local input-bound evidence before Git publication; never runs Cargo.

`capture` writes an UNVALIDATED subject before commit. A separately reviewed
adapter must assemble `trader-gemini-prepush-v1` from real guard receipts.
Native-six receipts have a separate registered schema; no conversion to PASS.
CI is a subsequent coordinator operation, not a future condition of this hook.
"""
import argparse
import datetime as dt
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
import tomllib


COMMANDS = {
    'fmt': ['fmt', '--all', '--', '--check'],
    'clippy': ['clippy', '--lib', '--locked', '--jobs', '1', '--', '-D', 'warnings'],
    'check': ['check', '--workspace', '--all-targets', '--locked', '--jobs', '1'],
    'tests': ['test', '-p', 'execution-engine', '-p', 'feature-engine', '--lib',
              '--tests', '--locked', '--jobs', '1', '--', '--test-threads=1'],
    'daz-clippy': ['clippy', '-p', 'os-guardian', '--lib', '--locked', '--jobs', '1',
                   '--', '-D', 'warnings'],
    'daz-tests': ['test', '-p', 'os-guardian', '--lib', '--test',
                  'f8_001_silicon_physics_contract', '--locked', '--jobs', '1',
                  '--', '--test-threads=1'],
}
BASE_PHASES = frozenset({'fmt', 'clippy', 'check', 'tests'})
PYTHON_COMMAND = ['python', '-I', '-B', 'scripts/tests/test_pre_push_gate.py',
                  'PrePushGateTests', '-v']
PYTHON_PATHS = frozenset({'.githooks/pre-push', 'scripts/pre_push_gate.py',
                          'scripts/tests/test_pre_push_gate.py'})
CONTROL_PATHS = ('scripts/cargo_semaforo.sh',)
NATIVE_SCHEMA = 'trader-gemini-controller-six-v1'
NATIVE_ATTESTATION = 'trader-gemini-prepush-native-six-v1'
INPUT53_SCHEMA = 'trader-gemini-controller-input-exclusion-v1'
INPUT53_ATTESTATION = 'trader-gemini-prepush-input-exclusion-v1'
NATIVE_SCHEMAS = {NATIVE_SCHEMA, INPUT53_SCHEMA}
FORMAT_SCHEMA = 'trader-gemini-format-witness-v1'
FORMAT_REVIEW = 'trader-gemini-format-review-v1'
NATIVE_PHASES = ('check', 'tests', 'daz-tests', 'daz-clippy', 'clippy', 'fmt')
BUNDLE_FILES = frozenset({'controller.py', 'contracts.py', 'sequence.sh', 'run.ps1'})
ISOLATION_FILES = frozenset({'runner.py', 'fixture_manifest.json', 'README.md'})
LEASE_FILES = ('dueno', 'creado', 'identidad')
NON_BUILD_PATHS = {'graphify-out/cache/last_query_stamp', 'docs/audit/V33_INVENTARIO_2026-10-24.tsv'}
NATIVE_PYTHON_COMMAND = [*PYTHON_COMMAND[:-2], 'PrePushGateTests', 'NativePrePushGateTests', '-v']
NATIVE_CONTRACT_COUNT = 20
INPUT53_PYTHON_COMMAND = [*NATIVE_PYTHON_COMMAND[:-1], 'Input53FormatTests', 'ReadonlyWorktreeTests', '-v']
INPUT53_CONTRACT_COUNT = 30
READONLY_WORKTREE_PATHS = frozenset({'docs/PLAN_MAESTRO_FASES_2026-10-22.md', 'docs/audit/V33_INVENTARIO_2026-10-24.tsv'})
MINIMUM_FORBIDDEN = ('.agent_tx/**', 'target/**', 'logs/**', '.env', '.env.local')


class GateError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise GateError(message)


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def digest_json(value):
    return digest_bytes(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                  separators=(',', ':')).encode('utf-8'))


def sha(path):
    return digest_bytes(Path(path).read_bytes())


def git(root, *args, data=None):
    result = subprocess.run(['git', '--no-optional-locks', '-C', str(root), *args],
                            input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            env={**os.environ, 'GIT_OPTIONAL_LOCKS': '0'})
    require(result.returncode == 0, 'Git query failed: ' + args[0])
    return result.stdout


def text_git(root, *args):
    return git(root, *args).decode('utf-8', 'surrogateescape').strip()


def safe_file(root, name):
    require(isinstance(name, str) and name and '\\' not in name and ':' not in name,
            'Evidence/input path must be repository-relative')
    require(all(ord(char) >= 32 and ord(char) != 127 and not 0xD800 <= ord(char) <= 0xDFFF
                for char in name), 'Control/invalid Unicode path rejected')
    require(not name.startswith('/') and all(p not in ('', '.', '..', '.git')
                                            for p in name.split('/')), 'Unsafe path')
    path = root / name
    require(path.resolve().is_relative_to(root.resolve()), 'Path escapes repository')
    for part in (path, *path.parents):
        if part == root.parent:
            break
        require(part.exists(), 'Missing evidence/input')
        metadata = part.lstat()
        require(not part.is_symlink() and not getattr(metadata, 'st_file_attributes', 0) & 0x400,
                'Symlink/reparse evidence/input rejected')
    require(stat.S_ISREG(path.lstat().st_mode), 'Nonregular evidence/input')
    return path


def load_json(path):
    def pairs(values):
        result = {}
        for key, value in values:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=pairs)
    except (ValueError, OSError) as error:
        raise GateError('Unreadable JSON evidence') from error


def policy_checked(policy):
    require(isinstance(policy, dict) and policy.get('version') == 1, 'Unknown policy')
    require(type(policy.get('max_blob_bytes')) is int and policy['max_blob_bytes'] > 0,
            'Explicit positive blob budget required')
    for name in ('forbidden_paths', 'docs_allowlist', 'ignored_untracked'):
        require(isinstance(policy.get(name), list) and all(isinstance(v, str) and v
                for v in policy[name]), 'Missing explicit path policy: ' + name)
    producer = policy.get('reviewed_producer')
    require(isinstance(producer, dict), 'Explicit reviewed-producer registration required')
    native = producer.get('schema') in NATIVE_SCHEMAS
    expected = {'guard', 'job', 'review_reference'}
    if native:
        expected |= {'schema', 'bundle', 'bundle_files', 'reviewed_inputs', 'isolation'}
    require(set(producer) == expected and isinstance(producer['review_reference'], str)
            and producer['review_reference'].strip(), 'Unknown/unreviewed producer schema')
    for kind in (('guard', 'job', 'bundle', 'reviewed_inputs') if native else ('guard', 'job')):
        entry = producer[kind]
        require(isinstance(entry, dict) and set(entry) == {'path', 'sha256'}
                and isinstance(entry['path'], str) and entry['path'].startswith('.agent_tx/')
                and valid_sha(entry['sha256']), 'Invalid reviewed-producer registration')
    require(producer['guard']['path'] != producer['job']['path'], 'Different producer paths required')
    if native:
        files, isolation = producer['bundle_files'], producer['isolation']
        require(isinstance(files, dict) and set(files) == BUNDLE_FILES and all(valid_sha(v) for v in files.values()),
                'All four runtime bundle members need explicit reviewed hashes')
        require(isinstance(isolation, dict) and set(isolation) == {'directory', 'files'}
                and isinstance(isolation['directory'], str) and Path(isolation['directory']).is_absolute()
                and not any(ch.isspace() for ch in isolation['directory'])
                and isinstance(isolation['files'], dict) and set(isolation['files']) == ISOLATION_FILES
                and all(valid_sha(v) for v in isolation['files'].values()), 'Closed isolation transport required')

    readonly = readonly_policy_rows(policy)
    require(not readonly or producer.get('schema') == INPUT53_SCHEMA,
            'Readonly worktree rows require the physical InputExclusion producer')
    return policy



def valid_sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None



def readonly_policy_rows(policy):
    rows = policy.get('readonly_worktree', [])
    require(isinstance(rows, list) and len(rows) <= 2, 'Closed readonly_worktree list required')
    result = {}
    for row in rows:
        require(isinstance(row, dict) and set(row) == {'path', 'raw_sha256', 'index', 'review_reference'},
                'Unknown readonly worktree row')
        name, entry = row['path'], row['index']
        require(isinstance(name, str) and name in READONLY_WORKTREE_PATHS and name not in result,
                'Only the two literal readonly document paths are admissible')
        require(valid_sha(row['raw_sha256']) and isinstance(entry, dict) and set(entry) == {'mode', 'blob'}
                and entry['mode'] in ('100644', '100755') and isinstance(entry['blob'], str)
                and re.fullmatch('(?:[0-9a-f]{40}|[0-9a-f]{64})', entry['blob']) is not None
                and isinstance(row['review_reference'], str) and row['review_reference'].strip(),
                'Root-reviewed RAW/index/reference pins required for readonly worktree row')
        result[name] = row
    return result


def readonly_worktree_checked(root, policy, files, raw):
    rows = readonly_policy_rows(policy)
    if not rows:
        return rows
    current_head = text_git(root, 'rev-parse', 'HEAD')
    published = tree_map(root, current_head)
    for name, row in rows.items():
        # This is an admission of physical RAW mismatch, never a staging/commit exemption.
        # Both tracked RAW and immutable published/index entries remain fully in the subject.
        safe_file(root, name)
        require(raw.get(name) == row['raw_sha256'] and files.get(name) == row['index']
                and published.get(name) == row['index'],
                'Readonly worktree RAW/index/HEAD pin changed or foreign content was staged')
    return rows


def non_build(name):
    return name.endswith('.md') or name in NON_BUILD_PATHS


def external_file(directory, name):
    directory = Path(directory)
    require(directory.is_absolute(), 'Absolute isolation transport required')
    path = directory / name
    require(path.resolve() == path and path.is_file(), 'Aliased/missing isolation artifact')
    for part in (path, *path.parents):
        meta = part.lstat()
        require(not part.is_symlink() and not getattr(meta, 'st_file_attributes', 0) & 0x400,
                'Reparse isolation artifact')
    return path


def native_registration(root, producer):
    # ROOT explicitly pins this exact reviewed bundle and closed-input JSON.
    controls = {}
    for kind in ('guard', 'job', 'bundle', 'reviewed_inputs'):
        ref = producer[kind]; path = safe_file(root, ref['path'])
        require(sha(path) == ref['sha256'], 'Registered helper/JSON bytes changed: ' + kind)
        controls[ref['path']] = ref['sha256']
    bundle = load_json(root / producer['bundle']['path'])
    require(set(bundle) == {'schema', 'status', 'files', 'base_guard_sha256', 'job_sha256', 'identity_contract_sha256'}
            and type(bundle['schema']) is int and bundle['schema'] == 1 and bundle['status'] == 'REVIEWED'
            and bundle['files'] == producer['bundle_files']
            and bundle['base_guard_sha256'] == producer['guard']['sha256']
            and bundle['job_sha256'] == producer['job']['sha256']
            and bundle['identity_contract_sha256'] == '99435aefc199cb6b8d16499e1fef0679de5fe108b19d9682b319db217e57f562',
            'DRAFT/unknown/inconsistent controller bundle')
    folder = Path(producer['bundle']['path']).parent
    for name, digest in producer['bundle_files'].items():
        relative = (folder / name).as_posix()
        require(sha(safe_file(root, relative)) == digest, 'Changed runtime controller member')
        controls[relative] = digest
    review = load_json(root / producer['reviewed_inputs']['path'])
    require(set(review) == {'schema', 'reviewed_wip', 'pinned_inputs', 'cleanup_helpers'}
            and type(review['schema']) is int and review['schema'] == 1, 'Unknown closed input schema')
    maps, aliases = {}, {}
    for section in ('reviewed_wip', 'pinned_inputs'):
        rows = review[section]
        require(isinstance(rows, list) and len(rows) <= 2048, 'Invalid reviewed input rows')
        values = {}
        for row in rows:
            require(isinstance(row, dict) and set(row) == {'path', 'sha256'} and valid_sha(row['sha256']),
                    'Unknown reviewed input entry')
            name = row['path']; safe_file(root, name)
            require(name not in values and (name.casefold() not in aliases or aliases[name.casefold()] == name),
                    'Duplicate/aliased closed input')
            require(sha(root / name) == row['sha256'], 'Reviewed source changed')
            aliases[name.casefold()] = name; values[name] = row['sha256']
        maps[section] = values
    auxiliary = review['cleanup_helpers']
    require(isinstance(auxiliary, list) and len(auxiliary) in (0, 1, 2)
            and all(isinstance(name, str) for name in auxiliary)
            and len({name.casefold() for name in auxiliary}) == len(auxiliary)
            and 'scripts/cargo_semaforo.sh' not in auxiliary
            and set(maps['pinned_inputs']) == {'scripts/cargo_semaforo.sh', *auxiliary},
            'Semaphore and ALL runtime auxiliaries must be pinned; no extra helper')
    for name in set(maps['reviewed_wip']) & set(maps['pinned_inputs']):
        require(maps['reviewed_wip'][name] == maps['pinned_inputs'][name], 'Conflicting reviewed source')
    controls.update(maps['pinned_inputs'])
    isolation = producer['isolation']
    for name, digest in isolation['files'].items():
        path = external_file(isolation['directory'], name)
        require(sha(path) == digest, 'Isolation runtime changed')
        controls['isolation:' + str(path)] = digest
    fixture = load_json(external_file(isolation['directory'], 'fixture_manifest.json'))
    require(isinstance(fixture.get('workspace'), str) and Path(fixture['workspace']).is_absolute()
            and Path(fixture['workspace']).resolve() == root, 'Isolation fixtures need an explicit matching absolute root')
    return controls, maps


def any_glob(name, patterns):
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


def index_map(root):
    stage = git(root, 'ls-files', '--stage', '-z')
    result = {}
    for row in stage.split(b'\0'):
        if not row:
            continue
        header, raw_name = row.split(b'\t', 1)
        mode, oid, slot = header.decode('ascii').split()
        name = os.fsdecode(raw_name).replace(os.sep, '/')
        require(slot == '0' and name not in result, 'Unmerged/duplicate index')
        require(mode in ('100644', '100755'), 'Nonregular Git mode')
        result[name] = {'mode': mode, 'blob': oid}
    for row in git(root, 'ls-files', '-v', '-z').split(b'\0'):
        if row:
            require(row.startswith(b'H '), 'Hidden index flags')
    folded = [name.casefold() for name in result]
    require(len(folded) == len(set(folded)), 'Case-colliding inputs')
    return result


def configurations(root):
    result = {}
    for parent in sorted({root, *root.parents, Path.home()}, key=str):
        for filename in ('config', 'config.toml'):
            path = parent / '.cargo' / filename
            require(not path.is_symlink(), 'Symlink Cargo configuration')
            result[str(path)] = sha(path) if path.is_file() else None
    return result


def content_settings(root):
    content = subprocess.run(['git', '-C', str(root), 'config', '--null', '--get-regexp',
                              r'^(core\.(autocrlf|eol|attributesfile|hookspath)|filter\.)'],
                             capture_output=True)
    require(content.returncode in (0, 1), 'Unknown Git content settings')
    settings = {'content_settings_sha256': digest_bytes(content.stdout)}
    attribute_files = [Path(text_git(root, 'rev-parse', '--path-format=absolute',
                                    '--git-path', 'info/attributes'))]
    configured = subprocess.run(['git', '-C', str(root), 'config', '--path', '--get',
                                'core.attributesfile'], capture_output=True)
    if configured.returncode == 0:
        path = Path(os.fsdecode(configured.stdout).strip())
        attribute_files.append(path if path.is_absolute() else root / path)
    else:
        require(configured.returncode == 1, 'Unknown attributes file')
        attribute_files.append(Path(os.environ.get('XDG_CONFIG_HOME', str(Path.home() / '.config')))
                               / 'git/attributes')
    for path in attribute_files:
        require(not path.is_symlink(), 'Symlink external attributes')
        settings['attributes:' + str(path)] = sha(path) if path.is_file() else None
    return settings


def capture_subject(root, policy):
    root = Path(root).resolve()
    policy_checked(policy)
    require(text_git(root, 'branch', '--show-current') == 'main', 'Only primary main is admitted')
    require(Path(text_git(root, 'rev-parse', '--show-toplevel')).resolve() == root,
            'Wrong repository root')
    common = Path(text_git(root, 'rev-parse', '--path-format=absolute', '--git-common-dir'))
    require(common.resolve().parent == root, 'Only primary checkout is admitted')
    files = index_map(root)
    settings = content_settings(root)
    configs = configurations(root)
    raw = {name: sha(safe_file(root, name)) for name in files}
    readonly = readonly_worktree_checked(root, policy, files, raw)
    # Git hashes physical paths with its real clean filters/CRLF in one process.
    # C-quoted path lines preserve spaces, Unicode, quotes and shell metacharacters.
    path_lines = ''.join(json.dumps(name, ensure_ascii=False) + '\n' for name in files)
    blobs = git(root, 'hash-object', '--stdin-paths', data=path_lines.encode('utf-8')).splitlines()
    require(len(blobs) == len(files), 'Incomplete Git content mapping')
    for (name, entry), blob in zip(files.items(), blobs):
        require(blob.decode('ascii') == entry['blob'] or name in readonly,
                'Working input differs from candidate index without an exact readonly row: ' + name)
    require(raw == {name: sha(safe_file(root, name)) for name in files}
            and files == index_map(root) and settings == content_settings(root)
            and configs == configurations(root), 'Inputs changed during capture')
    for row in git(root, 'ls-files', '--others', '--exclude-standard', '-z').split(b'\0'):
        if row:
            name = os.fsdecode(row).replace(os.sep, '/')
            require(Path(name).suffix not in ('.rs', '.toml', '.lock')
                    and not name.startswith('.cargo/'), 'Untracked build input: ' + name)
            require(any_glob(name, policy['ignored_untracked']), 'Unreviewed untracked input: ' + name)
    registered = policy['reviewed_producer']
    pins = {registered[kind]['path']: registered[kind]['sha256'] for kind in ('guard', 'job')}
    if registered.get('schema') in NATIVE_SCHEMAS:
        controls, _ = native_registration(root, registered)
    else:
        controls = {name: sha(safe_file(root, name)) for name in (*CONTROL_PATHS, *pins)}
        require(all(controls[name] == expected for name, expected in pins.items()),
                'Missing/changed registered private producer; bootstrap review required')
    extra,_,_=format_controls(root,policy)
    controls.update(extra)
    if policy.get('semaphore_contracts') is not None:
        for ref in policy['semaphore_contracts'].values():
            registered_artifact(root,ref);controls[ref['path']]=ref['sha256']
    subject = {'schema': 1, 'state': 'CAPTURED_NOT_VALIDATED', 'root': str(root),
               'baseline_head': text_git(root, 'rev-parse', 'HEAD'), 'branch': 'main',
               'files': files, 'raw': raw, 'configs': configs,
               'controls': controls, 'git_settings': settings,
               'policy': policy}
    subject['inputs_digest'] = digest_json(input_projection(subject))
    return subject


def input_projection(subject):
    keys = ('root', 'branch', 'files', 'raw', 'configs', 'controls', 'git_settings', 'policy')
    return {key: subject[key] for key in keys}


def check_subject(root, subject):
    require(subject.get('schema') == 1 and subject.get('state') == 'CAPTURED_NOT_VALIDATED',
            'Unknown subject schema/state')
    require(subject.get('inputs_digest') == digest_json(input_projection(subject)),
            'Subject digest changed')
    current = capture_subject(root, subject['policy'])
    require(current['inputs_digest'] == subject['inputs_digest'], 'Inputs changed after validation')
    return current


def evidence_file(root, reference):
    require(isinstance(reference, dict) and re.fullmatch('[0-9a-f]{64}',
            reference.get('sha256', '')), 'Missing evidence digest')
    name = reference.get('path', '')
    require(name.startswith('logs/'), 'Evidence must be an own ignored log artifact')
    path = safe_file(root, name)
    require(sha(path) == reference['sha256'], 'Evidence bytes changed')
    return path


def validate_guard_run(root, subject, phase, references):
    require(phase in COMMANDS and set(references) == {'receipt', 'plan', 'worker', 'log'},
            'Incomplete guard witness')
    paths = {kind: evidence_file(root, ref) for kind, ref in references.items()}
    require(len({path.parent for path in paths.values()}) == 1, 'Mixed run directories')
    receipt, plan, worker = (load_json(paths[kind]) for kind in ('receipt', 'plan', 'worker'))
    require(plan.get('phase') == phase and plan.get('command') == COMMANDS[phase], 'Wrong fixed command')
    require(Path(plan.get('root', '')).resolve() == root, 'Foreign run root')
    token = plan.get('token', '')
    require(re.fullmatch('Codex-Relevo-PR65-[0-9a-f]{32}', token) is not None
            and paths['plan'].parent.name == token, 'Foreign/unknown run owner')
    producer = subject['policy']['reviewed_producer']
    guard_sha, job_sha = (producer[kind]['sha256'] for kind in ('guard', 'job'))
    require(plan.get('wrapper_sha256') == guard_sha and plan.get('job_sha256') == job_sha,
            'Unreviewed guard/Job producer')
    require(plan.get('semaphore_sha256') == subject['controls']['scripts/cargo_semaforo.sh'],
            'Different semaphore')
    require(Path(receipt.get('plan', '')).resolve() == paths['plan'], 'Receipt points elsewhere')
    require(receipt.get('worker') == worker and receipt.get('cargo_log_sha256') == sha(paths['log']),
            'Worker/log not linked to receipt')
    for value in (receipt, worker):
        try:
            start = dt.datetime.fromisoformat(value['started_utc'])
            finish = dt.datetime.fromisoformat(value['completed_utc'])
            require(start.tzinfo is not None and finish.tzinfo is not None and finish >= start,
                    'Nonterminal/invalid run interval')
        except (KeyError, ValueError, TypeError) as error:
            raise GateError('Nonterminal/invalid run interval') from error
    require(receipt.get('pass') is True and all(type(receipt.get(k)) is int and receipt[k] == 0
            for k in ('exit_code', 'bash_exit_code')), 'Guard phase did not pass')
    require(receipt.get('semaphore_released') is True and receipt.get('root_lock_released') is True,
            'Own locks not released')
    require(worker.get('cargo_launched') is True and worker.get('drain_safe') is True
            and worker.get('build_inputs_unchanged') is True
            and type(worker.get('cargo_exit_code')) is int and worker['cargo_exit_code'] == 0
            and type(worker.get('exit_code')) is int and worker['exit_code'] == 0,
            'No completed native phase')
    job = worker.get('job', {})
    require(all(type(job.get(key)) is int and job[key] == 0
                for key in ('exit_code', 'active_processes')), 'Job not drained')
    for snapshot in (plan.get('before'), worker.get('after'), receipt.get('after')):
        require(isinstance(snapshot, dict) and snapshot.get('tracked') == subject['raw']
                and snapshot.get('configs') == subject['configs']
                and snapshot.get('branch') == 'main'
                and snapshot.get('wrapper_sha256') == guard_sha
                and snapshot.get('job_sha256') == job_sha
                and not snapshot.get('untracked_build_inputs'), 'Run used different inputs')
    require(receipt.get('build_inputs_unchanged') is True, 'Run input identity absent')





FORMATTER_SHA = '25a61fa2f18b3b5f51b39a4e1cc2a4463aabe64daf31b6f58961b760e7f90497'
GOD_TUPLE = ('src/bin/god_engine.rs', '818850a49653f345a485d8de7b6aac1c2a9054e3723c5ef455f83efbe22f51a5',
             '91ea88063bf11aab2db329f4b441d4a122ebcc3115291b635f9ea7ddb75423d3',
             'dd3b770eb5b71d4285bf7a6f33c7d189d9078ac39e216b2eb7346a499c3add68')

OMNI_TUPLE = ('crates/omniscient-registry/src/lib.rs', 'd5768621d0f1cb838110ea2e9a135f84c2a71c870745dae42716e4a1caeeca5e', '52d499e36c42dc3b558fe69729d87bc44360e4d8bc63a2b3c7e69680d354d0e9', 'd242aeb6a9b4cb0a36c0fbf887d06b4c076d6c118210270b1f4d5bba68b487e4')
OMNI_EDITS = [{'offset': 7039, 'before_hex': '20202020202020202020202069662066616c7365207b7d0a', 'after_hex': '', 'context_sha256': '128428c136a6b64d01ff1f4e74237e3cceaf4ac05e7d422b645b96b01af844a1'}, {'offset': 17986, 'before_hex': '7374643a3a696f3a3a4572726f723a3a6e6577287374643a3a696f3a3a4572726f724b696e643a3a4f746865722c20652e746f5f737472696e67282929', 'after_hex': '7374643a3a696f3a3a4572726f723a3a6f7468657228652e746f5f737472696e67282929', 'context_sha256': 'eb2afaf27a7c2041b29c2b0616010fe4135342bc2c9f213e797a68fa820f5799'}]
SEM_PATHS = frozenset({'scripts/cargo_semaforo.sh','scripts/semaforo_lock_cleanup.py','scripts/tests/test_cargo_semaforo.py'})


def registered_artifact(root, reference):
    require(isinstance(reference,dict) and set(reference)=={'path','sha256'} and valid_sha(reference['sha256'])
            and isinstance(reference['path'],str) and reference['path'].startswith(('.agent_tx/','logs/')),
            'Explicit private/log artifact registration required')
    path=safe_file(root,reference['path']);require(sha(path)==reference['sha256'],'Registered artifact changed')
    return path


def custom_filters_absent(root,names):
    data=b'\0'.join(os.fsencode(name) for name in names)+b'\0'
    values=git(root,'check-attr','-z','filter','--stdin',data=data).split(b'\0')
    require(all(value in (b'unspecified',b'unset') for value in values[2::3]),'Unapproved custom Git content filter')


def format_config(root,name):
    parents=[]
    for parent in [(root/name).parent,*(root/name).parent.parents,Path.home()]:
        if parent not in parents:parents.append(parent)
    observed={};chosen=None
    for key in ('APPDATA','XDG_CONFIG_HOME'):
        if os.environ.get(key):
            parent=Path(os.environ[key])/'rustfmt'
            if parent not in parents:parents.append(parent)
    for parent in parents:
        present=[]
        for filename in ('rustfmt.toml','.rustfmt.toml'):
            path=parent/filename
            try:path.lstat()
            except FileNotFoundError:observed[str(path)]=None;continue
            external_file(parent,filename);observed[str(path)]=sha(path);present.append(path)
        require(len(present)<=1,'Ambiguous formatter configurations')
        if chosen is None and present:chosen=present[0]
    return observed, ({'path':str(chosen),'sha256':sha(chosen)} if chosen else None)


def edition_for(root,name):
    selected=next((parent/'Cargo.toml' for parent in [(root/name).parent,*(root/name).parent.parents]
                   if parent.is_relative_to(root) and (parent/'Cargo.toml').is_file()),None)
    require(selected is not None,'No Cargo manifest for formatter edition')
    safe_file(root,selected.relative_to(root).as_posix())
    value=tomllib.loads(selected.read_text(encoding='utf-8'));edition=value.get('package',{}).get('edition')
    pins={selected.relative_to(root).as_posix():sha(selected)}
    if isinstance(edition,dict) and edition=={'workspace':True}:
        safe_file(root,'Cargo.toml')
        workspace=tomllib.loads((root/'Cargo.toml').read_text(encoding='utf-8'))
        edition=workspace.get('workspace',{}).get('package',{}).get('edition');pins['Cargo.toml']=sha(root/'Cargo.toml')
    require(edition in {'2015','2018','2021','2024'},'Explicit package/workspace edition required')
    return edition,pins


def before_bytes(root,row,baseline):
    path=registered_artifact(root,row['before_ref']);raw=path.read_bytes()
    entry=tree_map(root,baseline).get(row['path'])
    require(entry=={'mode':row['mode'],'blob':row['baseline_blob']},'Before baseline blob/mode differs')
    payload=git(root,'cat-file','blob',row['baseline_blob'])
    require(row['before_mapping'] in {'blob-payload','git-checkout'},'Unknown before-byte mapping')
    if row['before_mapping']=='blob-payload':require(raw==payload,'Raw before is not the actual baseline blob payload')
    else:
        checkout=git(root,'cat-file','--filters','--path='+row['path'],baseline+':'+row['path'])
        require(raw==checkout,'Explicit checkout reproduction differs from raw before')
    return raw,digest_bytes(payload)


def omni_effective_before(root,row,exception,before,edition):
    fields={'schema','status','reviewer','review_reference','path','before_sha256','effective_sha256','after_sha256','edition','edits','patch_ref','classification'}
    require(isinstance(exception,dict) and set(exception)==fields and exception['schema']=='trader-gemini-omni-two-lints-v1'
            and exception['status']=='REVIEWED' and exception['reviewer']=='ROOT'
            and isinstance(exception['review_reference'],str) and exception['review_reference'].strip()
            and exception['classification']=='SEMANTIC_LINT_CLEANUP_THEN_PINNED_FORMATTER'
            and (exception['path'],exception['before_sha256'],exception['effective_sha256'],exception['after_sha256'])==OMNI_TUPLE
            and row['path']==OMNI_TUPLE[0] and row['after_sha256']==OMNI_TUPLE[3]
            and edition==exception['edition']=='2021' and digest_bytes(before)==OMNI_TUPLE[1],
            'Only the exact ROOT-reviewed Omni two-lint tuple is admitted')
    edits=exception['edits'];require(isinstance(edits,list) and edits==OMNI_EDITS,'Omni exact two edits differ')
    for edit in edits:
        require(set(edit)=={'offset','before_hex','after_hex','context_sha256'} and type(edit['offset']) is int,'Unknown Omni edit/offset')
        offset=edit['offset'];old=bytes.fromhex(edit['before_hex'])
        require(0<offset<len(before) and before[offset:offset+len(old)]==old
                and digest_bytes(before[max(0,offset-80):offset+len(old)+80])==edit['context_sha256'],'Omni exact offset/context differs')
    effective=before
    for edit in reversed(edits):
        offset=edit['offset'];old=bytes.fromhex(edit['before_hex']);effective=effective[:offset]+bytes.fromhex(edit['after_hex'])+effective[offset+len(old):]
    require(digest_bytes(effective)==OMNI_TUPLE[2],'Omni operation changed more than the exact two edits')
    registered_artifact(root,exception['patch_ref'])
    return effective


def reproduce_format(root,row,formatter,baseline):
    fields={'path','mode','baseline_blob','before_ref','before_mapping','after_sha256','after_blob','edition','cargo_manifests','config_ref','god_exception'}
    require(isinstance(row,dict) and set(row)==fields and row['mode'] in ('100644','100755')
            and valid_sha(row['after_sha256']), 'Unknown closed formatter row')
    safe_file(root,row['path']);custom_filters_absent(root,[row['path']])
    edition,manifests=edition_for(root,row['path']);configs,config=format_config(root,row['path'])
    require(row['edition']==edition and row['cargo_manifests']==manifests and row['config_ref']==config,
            'Formatter edition/manifest/config differs from reviewed row')
    before,blob_sha=before_bytes(root,row,baseline);effective=before
    if row['god_exception'] is not None:
        exception_path=registered_artifact(root,row['god_exception']);exception=load_json(exception_path)
        if exception.get('schema')=='trader-gemini-omni-two-lints-v1':
            effective=omni_effective_before(root,row,exception,before,edition)
        else:
            require(set(exception)=={'schema','status','reviewer','review_reference','path','before_sha256','effective_sha256',
                    'after_sha256','edition','offset','context_sha256','patch_ref'}
                    and exception['schema']=='trader-gemini-god-eight-spaces-v1' and exception['status']=='REVIEWED'
                    and exception['reviewer']=='ROOT' and exception['review_reference'].strip()
                    and (exception['path'],exception['before_sha256'],exception['effective_sha256'],exception['after_sha256'])==GOD_TUPLE
                    and row['path']==GOD_TUPLE[0] and row['after_sha256']==GOD_TUPLE[3]
                    and edition==exception['edition']=='2021' and digest_bytes(before)==GOD_TUPLE[1],
                    'Only the exact ROOT-reviewed God eight-space tuple is admitted')
            offset=exception['offset'];require(type(offset) is int and 0<offset<len(before)-8,'Invalid God patch offset')
            require(before[offset-1:offset+9]==b'\n        \n'
                    and digest_bytes(before[max(0,offset-80):offset+88])==exception['context_sha256'],'God exact context differs')
            effective=before[:offset]+before[offset+8:]
            require(digest_bytes(effective)==GOD_TUPLE[2],'God operation changed more than the exact eight spaces')
            registered_artifact(root,exception['patch_ref'])
    with tempfile.TemporaryDirectory(prefix='tg-format-proof-') as temporary:
        directory=Path(temporary);source=directory/row['path'];source.parent.mkdir(parents=True)
        source.write_bytes(effective);config_path=directory/'explicit-rustfmt.toml'
        config_path.write_bytes(Path(config['path']).read_bytes() if config else b'')
        argv=[str(formatter),'--edition',edition,'--config','skip_children=true','--config-path',str(config_path),str(source)]
        env={key:value for key,value in os.environ.items() if not key.startswith(('RUSTFMT','RUSTC','CARGO_'))}
        started=dt.datetime.now(dt.timezone.utc).isoformat()
        result=subprocess.run(argv,cwd=directory,env=env,capture_output=True,timeout=90)
        after=source.read_bytes()
        require(result.returncode==0,'Native rustfmt reproduction failed')
        require(digest_bytes(after)==row['after_sha256'] and after==safe_file(root,row['path']).read_bytes(),
                'After contains a change not reproduced by the pinned native formatter')
        check_argv=argv[:-1]+['--check',str(source)]
        check=subprocess.run(check_argv,cwd=directory,env=env,capture_output=True,timeout=90)
        require(check.returncode==0,'Native rustfmt check did not pass on reproduced bytes')
        return {'row':row,'baseline_blob_payload_sha256':blob_sha,'effective_before_sha256':digest_bytes(effective),
                'config_inputs':configs,'format_argv':argv,'format_exit_code':result.returncode,
                'format_stdout_hex':result.stdout.hex(),'format_stderr_hex':result.stderr.hex(),
                'format_output_sha256':digest_bytes(result.stdout+b'\0'+result.stderr),
                'check_argv':check_argv,'check_exit_code':check.returncode,
                'check_stdout_hex':check.stdout.hex(),'check_stderr_hex':check.stderr.hex(),
                'check_output_sha256':digest_bytes(check.stdout+b'\0'+check.stderr),
                'started_utc':started,'completed_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
                'after_sha256':digest_bytes(after)}


def build_format_witness(root,spec_reference,output):
    spec_path=registered_artifact(root,spec_reference);spec=load_json(spec_path)
    require(set(spec)=={'schema','baseline_commit','formatter','rows','review_reference'} and spec['schema']=='trader-gemini-format-spec-v1'
            and isinstance(spec['review_reference'],str) and spec['review_reference'].strip()
            and isinstance(spec['rows'],list) and spec['rows'],'Closed ROOT formatter recipe required')
    formatter_ref=spec['formatter']
    require(isinstance(formatter_ref,dict) and set(formatter_ref)=={'path','sha256'}
            and formatter_ref['sha256']==FORMATTER_SHA,'Unreviewed native formatter')
    formatter=external_file(Path(formatter_ref['path']).parent,Path(formatter_ref['path']).name)
    require(formatter.name=='rustfmt.exe' and sha(formatter)==FORMATTER_SHA,'Native formatter changed')
    baseline=spec['baseline_commit'];require(text_git(root,'rev-parse',baseline+'^{commit}')==baseline,'Unknown baseline commit')
    rows=spec['rows'];names=[row['path'] for row in rows]
    require(len(names)==len({name.casefold() for name in names}),'Duplicate/aliased formatter rows')
    custom_filters_absent(root,names)
    output=Path(output).absolute()
    require(output.resolve()==output and (output.is_relative_to(root/'.agent_tx') or output.is_relative_to(root/'logs'))
            and not output.exists(),'New own private output path required')
    for parent in output.parents:
        require(not parent.is_symlink() and not getattr(parent.lstat(),'st_file_attributes',0)&0x400,'Output reparse ancestor')
    stage=index_map(root);settings=content_settings(root);cargo_configs=configurations(root);head=text_git(root,'rev-parse','HEAD')
    for row in rows:
        require(stage.get(row['path'])=={'mode':row['mode'],'blob':row['after_blob']},'Reviewed after is not staged')
        mapped=git(root,'hash-object','--path='+row['path'],'--stdin',data=safe_file(root,row['path']).read_bytes()).decode().strip()
        require(mapped==row['after_blob'],'Physical after does not map to staged Git blob')
    # Producer is the exact current verifier source; ROOT must separately review the result.
    producer_path=Path(__file__).resolve();require(producer_path.is_relative_to(root),'Witness producer outside source root')
    results=[reproduce_format(root,row,formatter,baseline) for row in rows]
    require(stage==index_map(root) and settings==content_settings(root) and cargo_configs==configurations(root)
            and head==text_git(root,'rev-parse','HEAD') and sha(spec_path)==spec_reference['sha256']
            and sha(formatter)==FORMATTER_SHA and all(sha(safe_file(root,row['path']))==row['after_sha256'] for row in rows),
            'Staged/configuration inputs changed during formatter reproduction')
    value={'schema':FORMAT_SCHEMA,'state':'REPRODUCED_NOT_ROOT_APPROVED','root':str(root),
           'producer':{'path':producer_path.relative_to(root).as_posix(),'sha256':sha(producer_path)},
           'recipe':spec_reference,'baseline_commit':baseline,'captured_head':head,
           'formatter':formatter_ref,'index_files_digest':digest_json(stage),'git_settings':settings,
           'cargo_configs':cargo_configs,'rows':results,'review_reference':spec['review_reference']}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8',newline='\n') as stream:json.dump(value,stream,indent=2);stream.write('\n')
    return value


def format_controls(root,policy):
    registration=policy.get('format_coverage')
    if registration is None:return {},None,None
    require(isinstance(registration,dict) and set(registration)=={'witness','root_review'},'Closed format witness registration required')
    witness_path=registered_artifact(root,registration['witness']);review_path=registered_artifact(root,registration['root_review'])
    witness,review=load_json(witness_path),load_json(review_path)
    require(isinstance(review,dict) and set(review)=={'schema','status','reviewer','review_reference','witness','producer','rows_digest','baseline_commit'}
            and review['schema']==FORMAT_REVIEW and review['status']=='REVIEWED' and review['reviewer']=='ROOT'
            and review['review_reference'].strip() and review['witness']==registration['witness']
            and witness.get('schema')==FORMAT_SCHEMA and witness.get('state')=='REPRODUCED_NOT_ROOT_APPROVED'
            and Path(witness.get('root','')).resolve()==root and review['producer']==witness.get('producer')
            and review['rows_digest']==digest_json(witness.get('rows')) and review['baseline_commit']==witness.get('baseline_commit'),
            'No explicit ROOT-reviewed closed format witness')
    producer_ref=witness['producer'];require(isinstance(producer_ref,dict) and set(producer_ref)=={'path','sha256'} and valid_sha(producer_ref['sha256']), 'Explicit witness builder registration required')
    producer=safe_file(root,producer_ref['path'])
    require(sha(producer)==producer_ref['sha256']==sha(Path(__file__).resolve()), 'Unknown/changed witness builder')
    recipe=registered_artifact(root,witness['recipe'])
    formatter=witness['formatter'];path=external_file(Path(formatter['path']).parent,Path(formatter['path']).name)
    require(formatter['sha256']==FORMATTER_SHA and sha(path)==FORMATTER_SHA,'Witness formatter changed')
    controls={ref['path']:ref['sha256'] for ref in (registration['witness'],registration['root_review'],witness['producer'],witness['recipe'])}
    controls['format-formatter:'+str(path)]=FORMATTER_SHA
    return controls,witness,review


def validate_format_coverage(root,subject,unsupported,updates):
    controls,witness,_=format_controls(root,subject['policy'])
    require(witness is not None,'Module coverage not defined for changed Rust paths without explicit format witness')
    require(subject['policy']['reviewed_producer'].get('schema')==INPUT53_SCHEMA,'Format coverage needs reviewed InputExclusion producer')
    require(all(subject['controls'].get(name)==digest for name,digest in controls.items()),'Format controls changed')
    require(witness['captured_head']==subject['baseline_head'] and witness['index_files_digest']==digest_json(subject['files'])
            and witness['git_settings']==subject['git_settings'] and witness['cargo_configs']==subject['configs'], 'Format witness used another composition/config')
    rows=witness['rows'];names=[entry['row']['path'] for entry in rows]
    require(len(names)==len(set(names)) and set(names)==unsupported,'Closed format witness does not enumerate exact unsupported paths')
    baseline=witness['baseline_commit'];formatter=Path(witness['formatter']['path'])
    for _,local,_,remote in updates:
        if set(local)=={'0'}:continue
        require(set(remote)!={'0'},'Format witness cannot cover a first publication')
        ancestry=subprocess.run(['git','-C',str(root),'merge-base','--is-ancestor',baseline,remote],capture_output=True)
        require(ancestry.returncode==0,'Remote does not contain the formatter functional baseline')
    for record in rows:
        row=record['row'];require(not row['path'].startswith(('crates/os-guardian/','crates/execution-engine/','crates/feature-engine/')),
                'Semantic module paths are not format exemptions')
        require(subject['raw'].get(row['path'])==row['after_sha256'] and subject['files'].get(row['path'])=={'mode':row['mode'],'blob':row['after_blob']},
                'Witness after differs from native raw/index/commit composition')
        require(record['format_exit_code']==record['check_exit_code']==0 and record['after_sha256']==row['after_sha256'],'Nonterminal format reproduction')
        for kind in ('format','check'):
            require(record[kind+'_output_sha256']==digest_bytes(bytes.fromhex(record[kind+'_stdout_hex'])+b'\0'+bytes.fromhex(record[kind+'_stderr_hex'])), 'Formatter output/log hash changed')
        require(record['config_inputs']==format_config(root,row['path'])[0],'Formatter config footprint changed')
        # Reproduce again from baseline in fresh Temp; never execute an argv from evidence.
        fresh=reproduce_format(root,row,formatter,baseline)
        require(fresh['after_sha256']==record['after_sha256'] and fresh['effective_before_sha256']==record['effective_before_sha256']
                and fresh['baseline_blob_payload_sha256']==record['baseline_blob_payload_sha256'], 'Formatter witness did not reproduce')


def validate_semaphore_contracts(root,subject):
    registration=subject['policy'].get('semaphore_contracts')
    require(isinstance(registration,dict) and set(registration)=={'receipt','log'},'Explicit semaphore46 witness registration required')
    receipt_path=registered_artifact(root,registration['receipt']);log_path=registered_artifact(root,registration['log'])
    value=load_json(receipt_path);interval(value)
    expected={name:subject['raw'].get(name) for name in SEM_PATHS}
    require(all(valid_sha(digest) for digest in expected.values()) and value.get('kind')=='REAL_HERMETIC_TEMP_CONTRACT_EXECUTION'
            and value.get('exit_code')==0 and type(value.get('exit_code')) is int and value.get('passed') is True
            and value.get('tests_run')==46 and type(value.get('tests_run')) is int
            and value.get('source_inputs_unchanged') is True and value.get('source_hashes_before')==value.get('source_hashes_after')==expected
            and value.get('log_sha256')==sha(log_path) and value.get('cargo_executed') is False
            and value.get('rustc_executed') is False and value.get('live_locks_touched') is False,'Semaphore46 scope/source/result mismatch')
    command=value.get('command',[])
    require(isinstance(command,list) and len(command)==5 and all(isinstance(part,str) for part in command)
            and Path(command[0]).resolve()==Path('C:/Python314/python.exe').resolve()
            and command[1:3]==['-I','-B'] and command[4]=='-v','Unknown semaphore46 command')
    test_path=Path(command[3])
    require(test_path.is_absolute() and test_path.resolve()==test_path and test_path.resolve().is_relative_to(root),
            'Semaphore46 producer outside canonical source root')
    external_file(test_path.parent,test_path.name)
    require(sha(test_path)==expected['scripts/tests/test_cargo_semaforo.py'],'Unreviewed semaphore46 test producer')
    scripts=test_path.parent.parent
    for relative in ('cargo_semaforo.sh','semaforo_lock_cleanup.py'):
        require(sha(external_file(scripts,relative))==expected['scripts/'+relative], 'Semaphore46 runtime source group changed')
    text=log_path.read_text(encoding='utf-8')
    require(re.search(r'(?m)^Ran 46 tests in [0-9.]+s\r?$',text) and text.rstrip().endswith('\nOK'), 'Semaphore46 log missing terminal0skip result')


def file_id_checked(value):
    require(isinstance(value, list) and len(value) == 2
            and type(value[0]) is int and 0 <= value[0] <= 0xffffffff
            and type(value[1]) is int and 0 <= value[1] <= 0xffffffffffffffff, 'Invalid physical file identity')


def validate_input_exclusion(root, plan, worker, evidence, plan_path, producer, sem_released_utc):
    require(isinstance(evidence, dict) and set(evidence) == {'started_utc','state','before','release','remaining','completed_utc'}
            and evidence == worker.get('input_exclusion') and evidence['state'] == 'released'
            and worker.get('input_exclusion_released') is True, 'Physical input exclusion was not released/certified')
    start,end=interval(evidence);worker_start,worker_end=interval(worker)
    sem_end=dt.datetime.fromisoformat(sem_released_utc)
    require(sem_end.tzinfo is not None and worker_start <= start <= end <= worker_end <= sem_end,
            'Physical exclusion interval outside own worker/semaphore release')
    before, remaining = evidence['before'], evidence['remaining']
    fields = {'schema','owner_token','owner_instance','job_launched','no_launch_certain','incomplete_sentinel_path',
              'sentinels','pins','directory_pins','release'}
    identity = worker['worker_identity']
    owner = {'pid':identity['pid'],'creation_filetime_100ns':identity['creation_filetime_100ns'],
             'image':str(Path(identity['executable']).resolve())}
    for record in (before,remaining):
        require(isinstance(record,dict) and set(record) == fields and type(record['schema']) is int and record['schema'] == 1
                and record['owner_token'] == plan['token'] and record['owner_instance'] == owner
                and record['incomplete_sentinel_path'] is None, 'Unknown/incomplete physical exclusion owner')
    require(before['job_launched'] is False and before['no_launch_certain'] is True and before['release'] is None
            and remaining['job_launched'] is True and remaining['no_launch_certain'] is False
            and remaining['sentinels'] == {}, 'Wrong exclusion lifecycle')
    paths = [str(root / '.git/index.lock'), str(root / '.git/refs/heads/main.lock')]
    require(set(before['sentinels']) == set(paths), 'Physical sentinel inventory differs')
    for name in paths:
        entry=before['sentinels'][name]
        require(set(entry) == {'path','hex','sha256','file_id'} and entry['path'] == name, 'Unknown sentinel evidence')
        file_id_checked(entry['file_id'])
        require(isinstance(entry['hex'],str) and re.fullmatch('[0-9a-f]+',entry['hex']), 'Noncanonical sentinel bytes')
        raw=bytes.fromhex(entry['hex'])
        canonical={'schema':1,'kind':'controller-input-exclusion','owner_token':plan['token'],'owner_instance':owner,
                   'path':name,'file_id':entry['file_id']}
        require(raw == (json.dumps(canonical,sort_keys=True,separators=(',',':'))+'\n').encode()
                and entry['sha256'] == digest_bytes(raw), 'Sentinel bytes/owner/hash/identity mismatch')
    frame=plan['before']; expected={}
    def add(path,digest):
        name=str(Path(path).absolute())
        require(name not in expected or expected[name] == digest,'Conflicting physical input')
        expected[name]=digest
    for group in ('tracked','untracked_build_inputs','reviewed_source_hashes'):
        for name,digest in frame[group].items():add(root/name,digest)
    for group in ('configs','isolation'):
        for name,digest in frame[group].items():
            if digest is not None:add(name,digest)
    parent=(root/producer['bundle']['path']).parent
    for name,digest in producer['bundle_files'].items():add(parent/name,digest)
    for kind in ('guard','job','bundle','reviewed_inputs'):
        add(root/producer[kind]['path'],producer[kind]['sha256'])
    add(plan_path,sha(plan_path))
    add(root/'.git/HEAD',digest_bytes(b'ref: refs/heads/main\n'))
    add(root/'.git/refs/heads/main',digest_bytes((frame['head']+'\n').encode()))
    add(root/'.git/config',sha(external_file(root/'.git','config')))
    # A physical index hash is opaque historical evidence, not the logical entries digest.
    index_name=str(root/'.git/index')
    require(index_name in before['pins'],'Missing physical index pin')
    add(index_name,before['pins'][index_name].get('sha256'))
    require(set(before['pins']) == set(expected) and before['pins'] == remaining['pins'], 'Physical pin inventory changed')
    folded=[name.casefold() for name in before['pins']]
    require(len(folded) == len(set(folded)), 'Aliased physical input inventory')
    for name,row in before['pins'].items():
        require(set(row)=={'sha256','file_id'} and valid_sha(row['sha256']) and row['sha256']==expected[name], 'Physical pin SHA differs')
        require(Path(name).is_absolute() and str(Path(name)) == name, 'Noncanonical physical input path')
        file_id_checked(row['file_id'])
    directories={str(parent) for name in expected for parent in Path(name).parents if parent != Path(parent.anchor)}
    require(set(before['directory_pins']) == directories and before['directory_pins'] == remaining['directory_pins'], 'Directory pin inventory changed')
    for value in before['directory_pins'].values():file_id_checked(value)
    release={'mode':'observed-job-zero','job_observation':worker['job'],'removed_own_paths':paths}
    require(evidence['release'] == remaining['release'] == release, 'No own Job0 physical sentinel release')
    require(worker['job']['active_processes']==0 and worker['job']['limit_commit_bytes']==16*1024**3,
            'Unknown own Job cannot certify physical release')


def interval(value):
    try:
        started, ended = (dt.datetime.fromisoformat(value[key]) for key in ('started_utc', 'completed_utc'))
        require(started.tzinfo is not None and ended.tzinfo is not None and ended >= started, 'Nonterminal native interval')
        return started, ended
    except (KeyError, ValueError, TypeError) as error:
        raise GateError('Nonterminal native interval') from error


def native_frame(root, subject, frame, helpers, maps):
    producer = subject['policy']['reviewed_producer']
    stage = git(root, 'ls-files', '--stage', '-z')
    build_stage = b'\0'.join(row for row in stage.split(b'\0') if row and not non_build(os.fsdecode(row.split(b'\t', 1)[1])))
    expected_isolation = {str(Path(producer['isolation']['directory']) / name): digest
                          for name, digest in producer['isolation']['files'].items()}
    expected_reviewed = {**maps['reviewed_wip'], **maps['pinned_inputs']}
    require(isinstance(frame, dict) and frame.get('tracked') == subject['raw']
            and frame.get('configs') == subject['configs'] and frame.get('branch') == 'main'
            and frame.get('head') == subject['baseline_head']
            and frame.get('tree') == text_git(root, 'rev-parse', subject['baseline_head'] + '^{tree}')
            and frame.get('index_entries_sha256') == digest_bytes(stage)
            and frame.get('index_build_entries_sha256') == digest_bytes(build_stage)
            and frame.get('wrapper_sha256') == producer['guard']['sha256']
            and frame.get('job_sha256') == producer['job']['sha256']
            and frame.get('isolation') == expected_isolation
            and frame.get('untracked_build_inputs') == {}
            and frame.get('single_turn_helpers') == helpers
            and frame.get('reviewed_source_hashes') == expected_reviewed,
            'Native snapshot differs from exact staged/captured composition')
    require(all(subject['raw'].get(name) == digest for name, digest in expected_reviewed.items()),
            'Reviewed runtime/source input is not in the published composition')


def native_lease(value, token, bash):
    require(isinstance(value, dict) and set(value) == {'directory_id', 'owner_instance', 'files'}
            and isinstance(value['files'], dict) and set(value['files']) == set(LEASE_FILES),
            'Native lease must bind exactly three entries')
    for file_id in [value['directory_id'], *[item.get('file_id') for item in value['files'].values()]]:
        file_id_checked(file_id)
    raw = {}
    for name, item in value['files'].items():
        require(set(item) == {'hex', 'sha256', 'file_id'} and isinstance(item['hex'], str), 'Invalid native lease entry')
        try: raw[name] = bytes.fromhex(item['hex'])
        except ValueError as error: raise GateError('Invalid native lease bytes') from error
        require(item['sha256'] == digest_bytes(raw[name]), 'Lease byte/hash mismatch')
    owner = re.escape(token.encode()) + rb' pid=[1-9][0-9]* t=[0-9]+ cargo ' + re.escape(' '.join(COMMANDS['check']).encode()) + rb'\n'
    require(re.fullmatch(owner, raw['dueno']) is not None and re.fullmatch(rb'[0-9]+\n', raw['creado']) is not None,
            'Lease owner/command mismatch')
    lines = raw['identidad'].split(b'\n')
    require(len(lines) == 3 and not lines[2] and lines[0] + b'\n' == raw['dueno'], 'Lease identity not bound to owner')
    match = re.fullmatch(rb'v1 windows ([1-9][0-9]*) ([1-9][0-9]*)', lines[1])
    require(match is not None, 'Legacy/unknown lease identity')
    pid, start = map(int, match.groups())
    require(pid <= 0xffffffff and start <= 0xffffffffffffffff and value['owner_instance'] ==
            {'pid': pid, 'creation_filetime_100ns': start, 'image': str(Path(bash).resolve())},
            'Native Bash instance mismatch')
    return value


def validate_native_runner(root, producer, name, phase, begin, used):
    records = phase.get('test_runner_receipts')
    require(phase.get('isolation_runtime_verified') is True and isinstance(records, list) and len(records) >= 2,
            'Missing native per-EXE isolation evidence')
    directory = Path(producer['isolation']['directory'])
    required = ('execution_engine-', 'feature_engine-') if name == 'tests' else ('os_guardian-', 'f8_001_silicon_physics_contract-')
    fixture = load_json(external_file(directory, 'fixture_manifest.json'))
    fixture_hashes = {row['path']: row['sha256'] for row in fixture['files']}
    require(len(fixture_hashes) == len(fixture['files']) and all(valid_sha(row['sha256'])
            and digest_bytes(row['content_utf8'].encode('utf-8')) == row['sha256'] for row in fixture['files']),
            'Invalid pinned fixture content')
    stems = []
    for record in records:
        require(isinstance(record, dict) and set(record) == {'path', 'sha256', 'receipt'} and valid_sha(record['sha256']),
                'Unknown runner witness')
        path = Path(record['path'])
        require(path.is_absolute() and path.parent.parent == directory / 'runs'
                and path.name == 'runner-receipt.json' and str(path) not in used
                and str(path) not in begin['runner_receipts_before'], 'Old/reused/foreign runner witness')
        external_file(path.parent, path.name)
        require(sha(path) == record['sha256'], 'Changed runner witness bytes')
        evidence = load_json(path)
        require(evidence == record['receipt'] and type(evidence.get('exit_code')) is int and evidence['exit_code'] == 0
                and evidence.get('runner_sha256') == producer['isolation']['files']['runner.py']
                and evidence.get('fixture_manifest_sha256') == producer['isolation']['files']['fixture_manifest.json']
                and evidence.get('arguments') == ['--test-threads=1'] and evidence.get('fixture_files') == fixture_hashes
                and Path(evidence.get('sandbox', '')).resolve() == path.parent,
                'Failed/wrong-schema isolation receipt')
        exe = Path(evidence.get('executable', ''))
        require(exe.is_absolute() and exe.is_relative_to(root / 'target') and exe.suffix.lower() == '.exe', 'Foreign test executable')
        external_file(exe.parent, exe.name)
        roots = evidence.get('roots', {})
        expected = {'TGM_DATA_DIR': path.parent / 'data', 'TGM_CONFIG_DIR': path.parent / 'config_dir', 'TG_CONFIG_ROOT': path.parent / 'config_dir'}
        require(set(roots) == set(expected) and all(Path(roots[key]).resolve() == target for key, target in expected.items()),
                'Test CWD/environment roots not isolated')
        require(Path(evidence.get('original_cwd', '')).resolve().is_relative_to(root), 'Foreign Cargo package CWD')
        stems.append(exe.stem); used.add(str(path))
    require(all(any(stem.startswith(prefix) for stem in stems) for prefix in required), 'Missing required suite EXE witness')


def validate_native_turn(root, subject, references):
    producer = subject['policy']['reviewed_producer']
    require(producer.get('schema') in NATIVE_SCHEMAS, 'Native receipt requires explicit native producer registration')
    input53 = producer.get('schema') == INPUT53_SCHEMA
    expected_refs = {'receipt', 'plan', 'worker', 'log', 'release', 'lease'} | ({'input_exclusion'} if input53 else set())
    require(isinstance(references, dict) and set(references) == expected_refs, 'Incomplete native turn references')
    paths = {kind: evidence_file(root, ref) for kind, ref in references.items()}
    require(len({path.parent for path in paths.values()}) == 1, 'Mixed native turn directories')
    directory = paths['plan'].parent
    for kind, filename in {'receipt':'receipt.json', 'plan':'plan.json', 'worker':'worker.json',
                           'log':'turn.log', 'release':'release.json', 'lease':'lease.json'}.items():
        require(paths[kind].name == filename, 'Unknown native artifact name')
    if input53:
        require(paths['input_exclusion'].name == 'input-exclusion.json', 'Wrong physical exclusion evidence')
    receipt, plan, worker, release, lease = (load_json(paths[k]) for k in ('receipt','plan','worker','release','lease'))
    token = plan.get('token', '')
    require(re.fullmatch('Codex-ReviewedController-[0-9a-f]{32}', token) is not None
            and directory == root / 'logs/codex-reviewed-controller-20261024' / token
            and isinstance(plan.get('root'), str) and Path(plan['root']).is_absolute()
            and Path(plan['root']).resolve() == root, 'Wrong native root/owner namespace')
    require(isinstance(plan.get('bash'), str) and Path(plan['bash']).is_absolute()
            and Path(plan['bash']).resolve() == Path('C:/Program Files/Git/usr/bin/bash.exe').resolve(),
            'Unregistered native Bash executable')
    require(plan.get('phases') == list(NATIVE_PHASES)
            and plan.get('commands') == {name: COMMANDS[name] for name in NATIVE_PHASES}, 'Unknown/incomplete native phase sequence')
    controls, maps = native_registration(root, producer)
    require(all(subject['controls'].get(k) == v for k,v in controls.items()), 'Registered helper composition changed')
    helpers = {**producer['bundle_files'], 'bundle_sha256':producer['bundle']['sha256'],
               'review_path':str((root / producer['reviewed_inputs']['path']).resolve()),
               'review_sha256':producer['reviewed_inputs']['sha256'], 'guard_sha256':producer['guard']['sha256'],
               'job_sha256':producer['job']['sha256'], 'pinned_inputs':maps['pinned_inputs']}
    require(plan.get('approval') == {key:helpers[key] for key in ('bundle_sha256','review_path','review_sha256')}
            and plan.get('helpers') == helpers and plan.get('reviewed_source_wip') == maps['reviewed_wip']
            and plan.get('semaphore_sha256') == controls['scripts/cargo_semaforo.sh'], 'Unreviewed native plan inputs')
    environment = plan.get('environment', {})
    require(isinstance(environment, dict) and isinstance(plan.get('cargo'), str), 'Missing fixed native environment')
    cargo = Path(plan['cargo']); toolchain = cargo.parent
    require(cargo.is_absolute() and cargo.name == 'cargo.exe', 'Native Cargo path absent')
    exact_env = {'CARGO_BUILD_JOBS':'1', 'CARGO_UNSTABLE_CHECKSUM_FRESHNESS':'true',
        'SEMAFORO_AGENTE':token, 'SEMAFORO_RAM_MIN_GB':'0.0', 'SEMAFORO_BARRIDO':'0', 'SEMAFORO_ESPERA_MAX_MIN':'0',
        'TG_ST_CLEANUP_PROFILE':'helpers' if len(maps['pinned_inputs']) > 1 else 'inline',
        'CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUNNER':Path('C:/Python314/python.exe').as_posix() + ' -I -B ' +
            (Path(producer['isolation']['directory']) / 'runner.py').as_posix()}
    path_env = {'CARGO_TARGET_DIR':root / 'target', 'RUSTC':toolchain / 'rustc.exe',
        'RUSTDOC':toolchain / 'rustdoc.exe', 'RUSTFMT':toolchain / 'rustfmt.exe',
        'CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER':toolchain.parent / 'lib/rustlib/x86_64-pc-windows-msvc/bin/rust-lld.exe',
        'TG_ST_PYTHON':Path('C:/Python314/python.exe'), 'TG_ST_CARGO':cargo, 'TG_ST_DIR':directory,
        'TG_ST_EXECUTOR':(root / producer['bundle']['path']).parent / 'controller.py'}
    require(all(environment.get(key) == value for key,value in exact_env.items()) and
        all(isinstance(environment.get(key), str) and Path(environment[key]).resolve() == value.resolve() for key,value in path_env.items()),
        'Native target/jobs/runner/profile/toolchain environment differs')
    baseline = tree_map(root, subject['baseline_head'])
    changed = {name for name in set(baseline) | set(subject['files']) if baseline.get(name) != subject['files'].get(name) and not non_build(name)}
    require(changed == set(maps['reviewed_wip']), 'Closed WIP does not describe this exact baseline/candidate pair')
    require(Path(receipt.get('plan', '')).resolve() == paths['plan'] and receipt.get('plan_sha256') == sha(paths['plan'])
            and receipt.get('worker') == worker and receipt.get('turn_log_sha256') == sha(paths['log']), 'Unlinked native aggregate/worker/log')
    require(receipt.get('pass') is True and all(type(receipt.get(key)) is int and receipt[key] == 0 for key in ('exit_code','bash_exit_code'))
            and receipt.get('semaphore_released') is True and receipt.get('root_lock_released') is True
            and receipt.get('build_inputs_unchanged') is True, 'Native aggregate did not pass/release')
    require(worker.get('job_launched') is True and worker.get('drain_safe') is True
            and type(worker.get('exit_code')) is int and worker['exit_code'] == 0
            and worker.get('build_inputs_unchanged') is True and 'error' not in worker and 'job_failure' not in worker,
            'Native worker failed or uncertain')
    job = worker.get('job', {})
    require(set(job) == {'exit_code','process_id','limit_commit_bytes','peak_commit_bytes','active_processes'}
            and all(type(v) is int for v in job.values()) and job['exit_code'] == job['active_processes'] == 0
            and job['process_id'] > 0 and job['limit_commit_bytes'] == 16 * 1024**3
            and 0 <= job['peak_commit_bytes'] <= job['limit_commit_bytes'], 'Native own Job not certified drained within bound')
    identity, root_owner = worker.get('worker_identity', {}), plan.get('root_owner', {})
    require(type(identity.get('pid')) is int and identity['pid'] > 0
            and type(identity.get('creation_filetime_100ns')) is int and identity['creation_filetime_100ns'] > 0
            and identity.get('owner_token') == root_owner.get('token') == token
            and type(root_owner.get('pid')) is int and root_owner['pid'] > 0
            and Path(identity.get('executable', '')).resolve() == Path('C:/Python314/python.exe').resolve()
            and isinstance(identity.get('argv'), list) and len(identity['argv']) == 3
            and Path(identity['argv'][0]).resolve() == (root / producer['bundle']['path']).parent / 'controller.py'
            and identity['argv'][1] == '--worker' and Path(identity['argv'][2]).resolve() == paths['plan'],
            'Worker/root ownership identity absent')
    marker = load_json(safe_file(root, (directory / 'worker-started.json').relative_to(root).as_posix()))
    require(marker.get('pid') == identity['pid'], 'Worker marker not linked')
    native_lease(lease, token, plan.get('bash', ''))
    require(worker.get('lease_before') == worker.get('lease_after') == worker.get('semaphore_owner') == lease
            and worker.get('lease_record_sha256') == sha(paths['lease'])
            and release.get('owner') == lease and release.get('lease_record_sha256') == sha(paths['lease']), 'Changed/unlinked native lease')
    evidence = release.get('evidence', {})
    require(evidence.get('before') == lease and evidence.get('job_observation') == job
            and evidence.get('removed_own_entries') == list(LEASE_FILES) and evidence.get('directory_removed') is True,
            'No native three-entry own release after Job0')
    if input53:
        require(receipt.get('publication_inputs_unchanged') is True and worker.get('publication_inputs_unchanged') is True, 'Full publication frame not certified')
        validate_input_exclusion(root, plan, worker, load_json(paths['input_exclusion']), paths['plan'], producer, release['released_utc'])
    for frame in (plan.get('before'), worker.get('after'), receipt.get('after')):
        native_frame(root, subject, frame, helpers, maps)
    for observation in (plan.get('initial_admission'), worker.get('admission')):
        require(isinstance(observation, dict) and observation.get('native') == []
                and type(observation.get('available_mib')) is int and observation['available_mib'] >= 0,
                'Missing real admission observation/census')
    started, ended = interval(receipt); worker_start, worker_end = interval(worker)
    require(started <= worker_start <= worker_end <= ended, 'Native worker outside aggregate interval')
    release_time = dt.datetime.fromisoformat(release.get('released_utc', ''))
    require(release_time.tzinfo is not None and worker_end <= release_time <= ended, 'Release preceded native worker completion')
    phases = worker.get('phases', {})
    require(isinstance(phases, dict) and set(phases) == set(NATIVE_PHASES)
            and worker.get('native_exit_codes') == {name:'0' for name in NATIVE_PHASES}, 'Missing six terminal native phases')
    previous, used = worker_start, set()
    for name in NATIVE_PHASES:
        phase_dir = directory / 'phases' / name
        phase_path = safe_file(root, (phase_dir / 'receipt.json').relative_to(root).as_posix())
        phase = load_json(phase_path)
        require(phase == phases[name] and phase.get('command') == COMMANDS[name]
                and phase.get('pass') is True and phase.get('build_inputs_unchanged') is True
                and all(type(phase.get(k)) is int and phase[k] == 0 for k in ('cargo_exit_code','exit_code')),
                'Failed/mismatched native phase')
        if input53:
            require(phase.get('publication_inputs_unchanged') is True, 'Phase publication frame changed')
        native_frame(root, subject, phase.get('before'), helpers, maps)
        native_frame(root, subject, phase.get('after'), helpers, maps)
        log = safe_file(root, (phase_dir / 'cargo.log').relative_to(root).as_posix())
        code = safe_file(root, (phase_dir / 'native-exit-code').relative_to(root).as_posix())
        begin = load_json(safe_file(root, (phase_dir / 'before.json').relative_to(root).as_posix()))
        require(phase.get('cargo_log_sha256') == sha(log) and code.read_bytes() == b'0\n'
                and phase.get('lease_after') == lease and begin.get('lease_before') == lease
                and begin.get('before') == phase['before'] and begin.get('started_utc') == phase.get('started_utc')
                and isinstance(begin.get('runner_receipts_before'), list), 'Changed phase log/begin/code/lease')
        observation = begin.get('admission', {})
        require(observation.get('native') == [] and type(observation.get('available_mib')) is int
                and observation['available_mib'] >= 0, 'Missing phase census')
        phase_start, phase_end = interval(phase)
        require(previous <= phase_start <= phase_end <= worker_end, 'Wrong native phase chronology')
        previous = phase_end
        if name in {'tests','daz-tests'}:
            validate_native_runner(root, producer, name, phase, begin, used)
    # Settings/attribute hashing and Git's real mapping are checked by capture_subject.
    # Unlike legacy compatibility, native publication refuses custom clean filters.
    attributes = git(root, 'check-attr', '-z', 'filter', '--stdin', data=b'\0'.join(os.fsencode(n) for n in subject['files']) + b'\0').split(b'\0')
    require(all(value in (b'unspecified', b'unset') for value in attributes[2::3]), 'Native publication with custom clean filters is not approved')


def parse_updates(root, lines):
    length = 64 if text_git(root, 'rev-parse', '--show-object-format') == 'sha256' else 40
    updates = []
    for line in lines.splitlines():
        fields = line.split()
        require(len(fields) == 4, 'Malformed/truncated pre-push input')
        local_ref, local_oid, remote_ref, remote_oid = fields
        require(all(re.fullmatch('[0-9a-f]{' + str(length) + '}', oid)
                    for oid in (local_oid, remote_oid)), 'Invalid object id')
        for ref in (remote_ref,):
            require(subprocess.run(['git', 'check-ref-format', ref], capture_output=True).returncode == 0,
                    'Invalid ref')
        if local_oid != '0' * length:
            oid_source = re.fullmatch('[0-9a-f]{' + str(length) + '}', local_ref) is not None
            require((local_ref == 'refs/heads/main' or (oid_source and local_ref == local_oid))
                    and remote_ref == 'refs/heads/main', 'Only main publication is admitted')
            require(text_git(root, 'rev-parse', local_oid + '^{commit}') == local_oid,
                    'Local OID is not a commit')
        else:
            require(local_ref == '(delete)' and remote_oid != '0' * length,
                    'Malformed deletion input')
        updates.append(fields)
    require(bool(updates), 'No update evidence provided')
    require(len({row[2] for row in updates}) == len(updates), 'Duplicate remote ref')
    return updates


def tree_map(root, oid):
    result = {}
    for row in git(root, 'ls-tree', '-r', '-z', oid).split(b'\0'):
        if row:
            header, raw_name = row.split(b'\t', 1)
            mode, kind, blob = header.decode('ascii').split()
            require(kind == 'blob' and mode in ('100644', '100755'), 'Nonregular publication object')
            result[os.fsdecode(raw_name)] = {'mode': mode, 'blob': blob}
    return result


def changed_paths(root, local_oid, remote_oid):
    if set(remote_oid) == {'0'}:
        return set(tree_map(root, local_oid))
    text_git(root, 'rev-parse', remote_oid + '^{commit}')
    return {os.fsdecode(name) for name in git(root, 'diff', '--name-only', '-z', remote_oid,
                                             local_oid).split(b'\0') if name}


def check_history(root, local_oid, remote_oid, policy):
    revisions = [local_oid]
    if set(remote_oid) != {'0'}:
        revisions.append('^' + remote_oid)
    forbidden = tuple(pattern.casefold() for pattern in (*MINIMUM_FORBIDDEN, *policy['forbidden_paths']))
    # Examine paths in every published commit too: an old blob can acquire a
    # forbidden alias without being a newly reachable object in rev-list.
    for commit in git(root, 'rev-list', *revisions).splitlines():
        for name in tree_map(root, commit.decode('ascii')):
            require(not any_glob(name.casefold(), forbidden), 'Forbidden path in publication history')
    arguments = ['rev-list', '--objects', local_oid]
    if set(remote_oid) != {'0'}:
        arguments += ['^' + remote_oid]
    for row in git(root, *arguments).splitlines():
        oid, separator, raw_name = row.partition(b' ')
        if not separator or text_git(root, 'cat-file', '-t', oid.decode()) != 'blob':
            continue
        size = int(text_git(root, 'cat-file', '-s', oid.decode()))
        require(size <= policy['max_blob_bytes'], 'Blob exceeds explicit publication budget')



def input53_python_command(root):
    return ['C:/Python314/python.exe','-I','-B',str(Path(root).resolve()/'scripts/tests/test_pre_push_gate.py'),
            'PrePushGateTests','NativePrePushGateTests','Input53FormatTests','ReadonlyWorktreeTests','-v']


def validate_python_contracts(root, subject, references, native=False):
    require(isinstance(references, dict) and set(references) == {'receipt', 'log'},
            'Python gate contracts need a reviewed-adapter receipt and log')
    receipt_path = evidence_file(root, references['receipt'])
    log_path = evidence_file(root, references['log'])
    receipt = load_json(receipt_path)
    input53 = subject['policy']['reviewed_producer'].get('schema') == INPUT53_SCHEMA
    expected_count = INPUT53_CONTRACT_COUNT if input53 else (NATIVE_CONTRACT_COUNT if native else 12)
    expected_command = input53_python_command(root) if input53 else (NATIVE_PYTHON_COMMAND if native else PYTHON_COMMAND)
    require(receipt.get('schema') == 'trader-gemini-prepush-python-v1'
            and Path(receipt.get('root', '')).resolve() == root
            and receipt.get('command') == expected_command, 'Wrong Python contract producer/command')
    require(all(receipt.get(key) == subject['inputs_digest']
                for key in ('inputs_digest', 'before_inputs_digest', 'after_inputs_digest')),
            'Python contracts used different inputs')
    require(type(receipt.get('exit_code')) is int and receipt['exit_code'] == 0
            and type(receipt.get('tests_run')) is int and receipt['tests_run'] == expected_count
            and all(type(receipt.get(key)) is int and receipt[key] == 0
                    for key in ('skipped', 'failures', 'errors'))
            and receipt.get('log_sha256') == sha(log_path), 'Python contract run did not pass')
    log = log_path.read_text(encoding='utf-8')
    require(re.search(r'(?m)^Ran ' + str(expected_count) + r' tests in [0-9.]+s\r?$', log) is not None
            and log.rstrip().endswith('\nOK'), 'Nonterminal/skipped Python contract log')


def verify(root, attestation, stdin, remote_name):
    root = Path(root).resolve()
    if isinstance(attestation, dict) and attestation.get('schema') == 'trader-gemini-prepush-zero-wait-v1':
        import types
        path = safe_file(root, 'scripts/zero_wait_gate.py')
        module = types.ModuleType('reviewed_zero_wait_gate'); module.__file__ = str(path)
        try:
            exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
            return module.verify(root, attestation, stdin, remote_name, globals())
        except RuntimeError as error:
            raise GateError(str(error)) from error
    require(remote_name == 'origin', 'Only configured origin is admitted')
    updates = parse_updates(root, stdin)
    if all(set(update[1]) == {'0'} for update in updates):
        return 'DELETION_NO_RUST_CERTIFICATION'
    require(isinstance(attestation, dict) and attestation.get('schema') in {'trader-gemini-prepush-v1', NATIVE_ATTESTATION, INPUT53_ATTESTATION},
            'Missing registered reviewed-producer attestation')
    native = attestation['schema'] in {NATIVE_ATTESTATION, INPUT53_ATTESTATION}
    subject = attestation.get('subject', {})
    check_subject(root, subject)
    phases = set(BASE_PHASES)
    changes = set()
    for _, local_oid, _, remote_oid in updates:
        if set(local_oid) == {'0'}:
            continue
        require(local_oid == text_git(root, 'rev-parse', 'HEAD'), 'Update OID is not current prepared HEAD')
        require(tree_map(root, local_oid) == subject['files'], 'Published tree differs from tested index')
        if native and set(remote_oid) != {'0'}:
            ancestry = subprocess.run(['git', '-C', str(root), 'merge-base', '--is-ancestor', remote_oid, local_oid], capture_output=True)
            require(ancestry.returncode == 0, 'Native publication must preserve remote ancestry')
        check_history(root, local_oid, remote_oid, subject['policy'])
        changes |= changed_paths(root, local_oid, remote_oid)
    # Docs exemptions are deliberately not enabled: .md can be a compiled input.
    unsupported = {name for name in changes if name.endswith('.rs') and not name.startswith(
        ('crates/os-guardian/', 'crates/execution-engine/', 'crates/feature-engine/'))}
    if unsupported:
        validate_format_coverage(root,subject,unsupported,updates)
    if any(name.startswith('crates/os-guardian/') and name.endswith('.rs') for name in changes):
        phases |= {'daz-clippy', 'daz-tests'}
    if native:
        require((attestation['schema']==INPUT53_ATTESTATION) == (subject['policy']['reviewed_producer'].get('schema')==INPUT53_SCHEMA), 'Envelope/producer schema mismatch')
        validate_native_turn(root, subject, attestation.get('native_turn'))
        if changes & SEM_PATHS:
            validate_semaphore_contracts(root,subject)
        if changes & PYTHON_PATHS:
            validate_python_contracts(root, subject, attestation.get('python_contracts'), native=True)
        return 'LOCAL_VALIDATED_CI_PENDING'
    require(subject['policy']['reviewed_producer'].get('schema') not in NATIVE_SCHEMAS, 'Native producer cannot use legacy normalized receipts')
    runs = attestation.get('runs', {})
    require(isinstance(runs, dict) and phases <= set(runs) and set(runs) <= set(COMMANDS),
            'Required local coverage absent')
    for phase in sorted(phases):
        validate_guard_run(root, subject, phase, runs[phase])
    if changes & PYTHON_PATHS:
        validate_python_contracts(root, subject, attestation.get('python_contracts'))
    return 'LOCAL_VALIDATED_CI_PENDING'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    capture = sub.add_parser('capture')
    capture.add_argument('--policy', required=True)
    capture.add_argument('--output', required=True)
    builder=sub.add_parser('format-build')
    builder.add_argument('--spec',required=True)
    builder.add_argument('--spec-sha256',required=True)
    builder.add_argument('--output',required=True)
    hook = sub.add_parser('verify')
    hook.add_argument('remote_name')
    hook.add_argument('remote_location')  # Never print a possibly authenticated URL.
    hook.add_argument('--receipt', default='logs/prepush/attestation.json')
    args = parser.parse_args(argv)
    try:
        root = Path(text_git(Path.cwd(), 'rev-parse', '--show-toplevel')).resolve()
        if args.mode == 'format-build':
            reference={'path':args.spec,'sha256':args.spec_sha256}
            build_format_witness(root,reference,args.output)
            print('[pre-push] REPRODUCED_NOT_ROOT_APPROVED; no publication PASS emitted')
        elif args.mode == 'capture':
            output = Path(args.output).resolve()
            require(output.is_relative_to(root / 'logs'), 'Subject output must be under ignored logs/')
            require(not output.exists(), 'Do not overwrite evidence')
            subject = capture_subject(root, load_json(args.policy))
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open('x', encoding='utf-8') as stream:
                json.dump(subject, stream, sort_keys=True, indent=2)
            print('[pre-push] CAPTURED_NOT_VALIDATED; no command or PASS generated')
        else:
            stdin = sys.stdin.read()
            parsed = parse_updates(root, stdin)
            deletion_only = all(set(update[1]) == {'0'} for update in parsed)
            attestation = None if deletion_only else load_json(safe_file(root, args.receipt))
            print('[pre-push] ' + verify(root, attestation, stdin, args.remote_name))
        return 0
    except (GateError, OSError, ValueError, KeyError, TypeError) as error:
        print('[pre-push] refused: ' + str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
