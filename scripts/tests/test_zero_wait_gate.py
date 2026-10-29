"""Protocol contracts using real temporary Git/Windows IO; no Cargo/market mocks.

Synthetic review/tool metadata is confined to Temp. These tests never emit a
project publication envelope or claim a Cargo/Native result.
"""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
NATIVE_RUSTFMT = Path(os.environ.get('RUSTFMT', 'C:/Users/jhona/.rustup/toolchains/nightly-2026-06-30-x86_64-pc-windows-msvc/bin/rustfmt.exe')).resolve()


class ZeroWaitGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='zero-wait-real-')
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve(); self.root = self.base / 'repo with spaces'; self.root.mkdir()
        home = self.base / 'home'; home.mkdir(); config = home / 'gitconfig'; config.write_bytes(b'')
        env = {k: v for k, v in os.environ.items() if not k.upper().startswith(('GIT_', 'CARGO_', 'RUST'))}
        env.update(GIT_CONFIG_GLOBAL=str(config), GIT_CONFIG_NOSYSTEM='1', GIT_OPTIONAL_LOCKS='0',
                   HOME=str(home), USERPROFILE=str(home), GIT_TERMINAL_PROMPT='0')
        context = patch.dict(os.environ, env, clear=True); context.start(); self.addCleanup(context.stop)
        self.git_path = str(Path(shutil.which('git')).resolve())
        self.git('init', '-q', '-b', 'main'); self.git('config', 'user.email', 'fixture@example.invalid')
        self.git('config', 'user.name', 'ZERO WAIT SYNTHETIC CONTRACT')
        self.git('config', 'core.autocrlf', 'false'); self.git('config', 'core.fsmonitor', 'false')
        self.put('.gitignore', '.agent_tx/\ntarget/\nlogs/\n')
        self.put('Cargo.toml', '[workspace]\nmembers=[]\n')
        self.put('.cargo/config.toml', '[build]\nrustflags=[]\n[unstable]\nchecksum-freshness=true\n')
        self.put('.gitattributes', '*.rs text eol=lf\n')
        self.put('src/lib.rs', 'pub fn old() {}\n')
        for name in ('pre_push_gate.py', 'zero_wait_gate.py', 'format_workspace.py'):
            self.put('scripts/' + name, (SCRIPTS / name).read_bytes())
        self.put('scripts/compiler_bus.py', '# SYNTHETIC SOURCE METADATA ONLY\n')
        self.git('add', '.'); self.git('commit', '-qm', 'synthetic baseline')
        self.index = self.root / '.git/index_contract'
        self.index.write_bytes((self.root / '.git/index').read_bytes())
        runtime = self.base / 'rustup/toolchains/toolchain'; binary = runtime / 'bin'; binary.mkdir(parents=True)
        self.tools = {}
        for key, name in {'cargo':'cargo.exe', 'rustc':'rustc.exe', 'rustdoc':'rustdoc.exe',
                          'cargo_fmt':'cargo-fmt.exe', 'cargo_clippy':'cargo-clippy.exe',
                          'rustfmt':'rustfmt.exe', 'clippy_driver':'clippy-driver.exe'}.items():
            p = binary / name; p.write_text('SYNTHETIC METADATA; NEVER EXECUTED ' + name)
            self.tools[key] = self.tool(p)
        linker = runtime / 'rust-lld.exe'; linker.write_bytes(b'SYNTHETIC LINKER METADATA ONLY')
        isolation = self.base / 'isolation'; isolation.mkdir()
        for name, content in {'runner.py':'# SYNTHETIC RUNNER METADATA ONLY\n',
                              'fixture_manifest.json':json.dumps({'schema':1, 'workspace':str(self.root), 'files':[]}),
                              'README.md':'SYNTHETIC protocol fixture, no Rust executed\n'}.items():
            (isolation / name).write_text(content)
        self.tools.update(linker=self.tool(linker), git=self.tool(Path(self.git_path)),
                          python=self.tool(Path(sys.executable).resolve()), runner=self.tool(isolation / 'runner.py'))
        self.put('.agent_tx/fixture/classification.json', '{}')
        self.put('.agent_tx/fixture/golden.json', '{}')
        self.review = {'schema':'trader-gemini-zero-wait-producer-v1', 'status':'REVIEWED', 'reviewer':'ROOT',
            'review_reference':'SYNTHETIC TEMP PROTOCOL FIXTURE; NOT PROJECT ROOT AUTHORITY OR CARGO EVIDENCE',
            'helpers':{n:self.hash((self.root / n).read_bytes()) for n in ('scripts/pre_push_gate.py','scripts/zero_wait_gate.py','scripts/compiler_bus.py','scripts/format_workspace.py')},
            'tools':self.tools, 'isolation':{'directory':str(isolation), 'files':{n:self.hash((isolation/n).read_bytes()) for n in ('runner.py','fixture_manifest.json','README.md')}},
            'golden_inventory':self.ref('.agent_tx/fixture/golden.json'), 'python_suites':{},
            'commands':{p:list(v) for p,v in self.module().COMMANDS.items()}, 'forbidden_paths':[],
            'classifications':self.ref('.agent_tx/fixture/classification.json')}
        self.put('.agent_tx/fixture/review.json', json.dumps(self.review)); self.producer = self.ref('.agent_tx/fixture/review.json')
        self.gate = self.module()

    @staticmethod
    def hash(raw): return hashlib.sha256(raw).hexdigest()

    def tool(self, path): return {'path':str(path), 'sha256':self.hash(path.read_bytes())}

    def put(self, name, data):
        p = self.root / name; p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data if isinstance(data, bytes) else data.encode()); return p

    def ref(self, name): return {'path':name, 'sha256':self.hash((self.root/name).read_bytes())}

    def module(self):
        p = SCRIPTS / 'zero_wait_gate.py'; spec = importlib.util.spec_from_file_location('synthetic_zero_wait', p)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module

    def git(self, *args, private=False):
        env = dict(os.environ)
        if private: env['GIT_INDEX_FILE'] = str(self.index)
        else: env.pop('GIT_INDEX_FILE', None)
        r = subprocess.run([self.git_path, '--no-optional-locks', '-C', str(self.root), *args], env=env, capture_output=True)
        self.assertEqual(r.returncode, 0, r.stderr.decode('utf-8','replace')); return r.stdout

    def subject(self):
        return self.gate.capture_subject(self.root, self.index, self.producer, tools=self.tools, generation='synthetic-case')

    def test_oid_source_is_bound_to_main_and_other_refs_stay_rejected(self):
        oid = self.git('rev-parse', 'HEAD').strip().decode()
        self.assertTrue(self.gate.valid_main_update([oid, oid, 'refs/heads/main', oid]))
        self.assertTrue(self.gate.valid_main_update(['refs/heads/main', oid, 'refs/heads/main', oid]))
        self.assertFalse(self.gate.valid_main_update(['refs/heads/other', oid, 'refs/heads/main', oid]))
        self.assertFalse(self.gate.valid_main_update([oid, oid, 'refs/heads/other', oid]))

    def test_private_index_is_explicit_and_tree_hash_matches_real_git(self):
        shared = (self.root/'.git/index').read_bytes()
        self.put('src/lib.rs', 'pub fn candidate() {}\n'); self.git('add', 'src/lib.rs', private=True)
        subject = self.subject()
        self.assertEqual(subject['candidate_tree'], self.git('write-tree', private=True).strip().decode())
        self.assertEqual((self.root/'.git/index').read_bytes(), shared)
        with self.assertRaisesRegex(RuntimeError, 'private index'):
            self.gate.capture_subject(self.root, self.root/'.git/index', self.producer, tools=self.tools, generation='bad')

    def test_real_crlf_clean_mapping_and_unstaged_drift(self):
        self.put('src/lib.rs', b'pub fn crlf() {}\r\n'); self.git('add', 'src/lib.rs', private=True)
        value = self.subject(); self.assertEqual(value['raw']['src/lib.rs'], self.hash((self.root/'src/lib.rs').read_bytes()))
        self.assertNotEqual(value['raw']['src/lib.rs'], value['files']['src/lib.rs']['blob'])
        self.put('src/lib.rs', 'pub fn drift() {}\n')
        with self.assertRaisesRegex(RuntimeError, 'RAW does not map'):
            self.subject()

    def test_assume_skip_and_intent_flags_remain_rejected(self):
        for flag, undo in [('--assume-unchanged','--no-assume-unchanged'), ('--skip-worktree','--no-skip-worktree')]:
            self.git('update-index', flag, 'src/lib.rs', private=True)
            with self.assertRaisesRegex(RuntimeError, 'flags'): self.subject()
            self.git('update-index', undo, 'src/lib.rs', private=True)
        self.put('new.rs', 'pub fn fresh() {}\n'); self.git('add', '-N', 'new.rs', private=True)
        with self.assertRaisesRegex(RuntimeError, 'flags'): self.subject()

    def test_tool_substitution_and_untracked_build_source_rejected(self):
        Path(self.tools['rustfmt']['path']).write_bytes(b'CHANGED')
        with self.assertRaisesRegex(RuntimeError, 'Tool changed'): self.subject()
        Path(self.tools['rustfmt']['path']).write_text('SYNTHETIC METADATA; NEVER EXECUTED rustfmt.exe')
        self.put('untracked.rs', 'pub fn hidden() {}\n')
        with self.assertRaisesRegex(RuntimeError, 'Untracked build input'): self.subject()

    def test_profile_uses_available_cpu_and_pins_external_cwd_plugins(self):
        env = self.gate.execution_env(self.root, self.tools)
        cpus = getattr(os, 'process_cpu_count', os.cpu_count)() or os.cpu_count()
        self.assertEqual(env['CARGO_BUILD_JOBS'], str(cpus)); self.assertIsNone(env['RUST_TEST_THREADS'])
        self.assertEqual(env['RUSTUP_TOOLCHAIN'], 'toolchain'); self.assertEqual(env['RUSTFMT'], self.tools['rustfmt']['path'])
        self.assertEqual(env['PATH'].split(os.pathsep)[0], str(Path(self.tools['cargo']['path']).parent))
        path_entries = env['PATH'].split(os.pathsep)
        python_directory = str(Path(self.tools['python']['path']).parent)
        self.assertIn(python_directory, path_entries)
        self.assertEqual(path_entries.count(python_directory), 1)
        self.assertLessEqual(path_entries.index(python_directory),
                             path_entries.index(str(Path(self.tools['git']['path']).parent)))
        self.assertTrue(all('--jobs' not in v and '--test-threads=1' not in v for v in self.gate.COMMANDS.values()))
        semantic = self.gate.COMMANDS['semantic-tests']
        self.assertEqual(semantic, ('test','--no-fail-fast','-p','audit-engine','-p','data-ingest','-p','data-pipeline',
            '-p','flight-recorder','-p','god-engine-core','-p','omniscient-registry','-p','quantum-arena',
            '-p','storage-engine','-p','strategy-core','-p','metacortex-engine','-p','signal-engine',
            '-p','telemetry-server','-p','backtest-engine','-p','dark-alpha-engine',
            '-p','evolution-engine','-p','risk-engine','-p','trader-gemini-v5',
            '--lib','--tests','--locked','--message-format=json'))
        self.assertNotIn('semantic-tests', self.gate.MANDATORY)

    def test_hostile_environment_is_removed_from_real_python_child(self):
        bound_home = self.base/'bound-cargo-home'
        hostile = {'CARGO_BUILD_RUSTFLAGS':'HOSTILE_BUILD_FLAGS',
                   'CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUSTFLAGS':'HOSTILE_TARGET_FLAGS',
                   'CARGO_UNREVIEWED_FUTURE_OVERRIDE':'HOSTILE_CARGO',
                   'CLIPPY_ARGS':'-A warnings','CLIPPY_CONF_DIR':'HOSTILE_CONF',
                   'CLIPPY_UNREVIEWED_FUTURE_OVERRIDE':'HOSTILE_CLIPPY',
                   'RUSTUP_HOME':'HOSTILE_RUSTUP','RUSTUP_TOOLCHAIN':'HOSTILE_TOOLCHAIN',
                   'CARGO_HOME':str(bound_home),'CARGO_BUILD_JOBS':'1','CARGO_TARGET_DIR':'HOSTILE_TARGET'}
        with patch.dict(os.environ,hostile):
            profile = self.gate.execution_env(self.root,self.tools)
            env = dict(os.environ)
            for key,value in profile.items():
                if value is None: env.pop(key,None)
                else: env[key] = value
            args = [sys.executable,'-I','-B','-c',
                    'import json,os,sys;print(json.dumps({k:os.environ.get(k) for k in json.loads(sys.argv[1])}))',
                    json.dumps(list(hostile))]
            child = subprocess.run(args,env=env,capture_output=True)
            self.assertEqual(child.returncode,0,child.stderr)
            observed = json.loads(child.stdout)
            for key in ('CARGO_BUILD_RUSTFLAGS','CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUSTFLAGS',
                        'CARGO_UNREVIEWED_FUTURE_OVERRIDE','CLIPPY_ARGS','CLIPPY_CONF_DIR','CLIPPY_UNREVIEWED_FUTURE_OVERRIDE'):
                self.assertIsNone(profile[key]);self.assertIsNone(observed[key])
            self.assertEqual(observed['RUSTUP_HOME'],str(self.base/'rustup'))
            self.assertEqual(observed['RUSTUP_TOOLCHAIN'],'toolchain')
            self.assertEqual(observed['CARGO_BUILD_JOBS'],str(getattr(os,'process_cpu_count',os.cpu_count)() or os.cpu_count()))
            self.assertEqual(observed['CARGO_TARGET_DIR'],str(self.root/'target'))
            subject = self.subject()
            self.assertEqual(subject['profile']['CARGO_HOME'],str(bound_home))
            self.assertEqual(observed['CARGO_HOME'],str(bound_home))
            for config in ('config','config.toml'):
                self.assertIn(str(bound_home/config),subject['configs'])
                self.assertIsNone(subject['configs'][str(bound_home/config)])

    @unittest.skipUnless(os.name == 'nt', 'Real Win32 sharing contract')
    def test_protected_slot_denies_writer_but_root_stays_editable(self):
        subject = self.subject()
        with self.gate.prepare_candidate(self.root, self.index, self.producer, tools=self.tools, generation='synthetic-case', subject=subject) as candidate:
            proof = candidate.proof(); slot = Path(proof['slot']); cwd = candidate.cwd
            self.addCleanup(lambda: cwd.rmdir() if cwd.exists() and not any(cwd.iterdir()) else None)
            self.assertTrue(proof['protection_active']); self.assertNotEqual((self.root/'src/lib.rs').stat().st_ino, (slot/'src/lib.rs').stat().st_ino)
            code = 'from pathlib import Path\nimport sys\ntry: Path(sys.argv[1]).write_text("denied")\nexcept PermissionError: sys.exit(0)\nsys.exit(17)'
            child = subprocess.run([sys.executable, '-I', '-B', '-c', code, str(slot/'src/lib.rs')], capture_output=True)
            self.assertEqual(child.returncode, 0, child.stderr)
            self.put('src/lib.rs', 'pub fn root_is_free() {}\n')
            self.assertEqual((slot/'src/lib.rs').read_text(), 'pub fn old() {}\n')
            self.assertIn('--manifest-path', candidate.command('check'))
            self.assertNotEqual(candidate.cwd, self.root)
        (slot/'src/lib.rs').write_text('released')

    @unittest.skipUnless(os.name == 'nt', 'Real Win32 sharing contract')
    def test_identical_candidate_preserves_mtime_between_passes(self):
        subject = self.subject(); first = None
        for _ in range(2):
            with self.gate.prepare_candidate(self.root, self.index, self.producer, tools=self.tools, generation='synthetic-case', subject=subject) as candidate:
                p = Path(candidate.proof()['slot'])/'src/lib.rs'; value = (p.stat().st_mtime_ns, p.read_bytes())
                if first is None: first = value
                else: self.assertEqual(value, first)
                self.addCleanup(lambda cwd=candidate.cwd: cwd.rmdir() if cwd.exists() and not any(cwd.iterdir()) else None)

    def test_failed_stale_diagnostic_and_legacy_receipts_reject(self):
        subject = self.subject()
        for fields in [{'schema':'trader-gemini-prepush-input-exclusion-v1'},
                       {'schema':self.gate.RECEIPT,'state':'STALE'}, {'schema':self.gate.RECEIPT,'state':'FAILED'},
                       {'schema':self.gate.RECEIPT,'kind':'diagnostic','feedback_only':True}]:
            self.put('.agent_tx/fixture/receipt.json', json.dumps(fields))
            with self.assertRaisesRegex(RuntimeError, 'publication evidence'):
                self.gate.validate_run(self.root, subject, 'check', self.ref('.agent_tx/fixture/receipt.json'), self.review)

    def test_real_git_delta_needs_exact_per_path_review(self):
        baseline = self.git('rev-parse', 'HEAD').strip().decode()
        self.put('src/lib.rs', 'pub fn reviewed() {}\n'); self.git('add', 'src/lib.rs', private=True)
        subject = self.subject()
        before = {'mode':'100644','blob':self.git('rev-parse', baseline + ':src/lib.rs').strip().decode()}
        record = {'path':'src/lib.rs','before':before,
                  'after':{**subject['files']['src/lib.rs'],'sha256':subject['raw']['src/lib.rs']},
                  'kind':'SEMANTIC_LINT_REVIEWED','review_reference':'SYNTHETIC TEMP ONLY',
                  'tests':['check'],'provenance':None}
        bundle = {'schema':'trader-gemini-zero-wait-rust-review-v1','status':'REVIEWED','reviewer':'ROOT',
                  'review_reference':'SYNTHETIC TEMP ONLY','baseline_commit':baseline,'changes':[record]}
        def validate(value):
            self.put('.agent_tx/fixture/classification.json', json.dumps(value))
            review = {**self.review,'classifications':self.ref('.agent_tx/fixture/classification.json')}
            return self.gate.classifications(self.root,subject,review,{'src/lib.rs'},{'check'},
                lambda *a: self.gate.git(self.root,self.index,self.tools,*a))
        self.assertEqual(validate(bundle), baseline)
        missing = copy.deepcopy(bundle); missing['changes'] = []
        with self.assertRaisesRegex(RuntimeError,'Missing/duplicate'): validate(missing)
        other = copy.deepcopy(bundle); other['changes'][0]['before']['blob'] = '0'*40
        with self.assertRaisesRegex(RuntimeError,'tuple differs'): validate(other)
        generic = copy.deepcopy(bundle); generic['changes'][0]['kind'] = 'CRATE_EXEMPT'
        with self.assertRaisesRegex(RuntimeError,'classification'): validate(generic)

    def test_golden_parser_rejects_ignored_missing_and_duplicate(self):
        # Transcript fixtures exercise validation only; no actual Rust result is emitted.
        raw = {}; suites = []
        for path,names in self.gate.GOLDEN.items():
            payload = ('// SYNTHETIC TRANSCRIPT SOURCE ' + path).encode()
            raw[path] = self.hash(payload)
            suites.append({'source':path,'sha256':raw[path],'test_names':list(names)})
        names = [n for row in suites for n in row['test_names']]
        inventory = {'schema':'codex-required-pr65-golden9-read-only-v1','expected_test_count':9,
                     'require_no_ignored_or_filtered_cases':True,'suites':suites}
        self.put('.agent_tx/fixture/golden-inventory.json',json.dumps(inventory))
        review = {**self.review,'golden_inventory':self.ref('.agent_tx/fixture/golden-inventory.json')}
        subject = {'raw':raw,'inputs_digest':'a'*64}
        tests_ref = {'path':'.agent_tx/fixture/synthetic-tests.json','sha256':'b'*64}
        tests_row = {'log_ref':{'path':'.agent_tx/fixture/synthetic-tests.log','sha256':'c'*64}}
        golden = {'schema':'trader-gemini-zero-wait-golden9-v1','inputs_digest':'a'*64,
                  'tests_receipt':tests_ref,'tests_log':tests_row['log_ref'],
                  'inventory':review['golden_inventory'],'tests':names,'passed':True}
        self.put('.agent_tx/fixture/golden-result.json',json.dumps(golden)); reference = self.ref('.agent_tx/fixture/golden-result.json')
        log = ''.join('test module::'+name+' ... ok\n' for name in names)
        self.gate.validate_golden(self.root,subject,reference,review,tests_ref,tests_row,log)
        for bad in (log.replace(names[0]+' ... ok',names[0]+' ... ignored'),
                    '\n'.join(log.splitlines()[1:]),log+log.splitlines()[0]+'\n'):
            with self.assertRaisesRegex(RuntimeError,'missing/ignored/duplicate'):
                self.gate.validate_golden(self.root,subject,reference,review,tests_ref,tests_row,bad)

    @unittest.skipUnless(os.name == 'nt', 'Real Win32 sharing contract')
    def test_unknown_slot_injection_and_root_drift_are_not_admitted(self):
        subject = self.subject()
        with self.gate.prepare_candidate(self.root,self.index,self.producer,tools=self.tools,generation='synthetic-case',subject=subject) as candidate:
            slot = Path(candidate.proof()['slot']); cwd = candidate.cwd
            self.addCleanup(lambda: cwd.rmdir() if cwd.exists() and not any(cwd.iterdir()) else None)
        (slot/'injected.rs').write_text('// foreign slot input')
        with self.assertRaisesRegex(RuntimeError,'Unowned candidate input'):
            with self.gate.prepare_candidate(self.root,self.index,self.producer,tools=self.tools,generation='synthetic-case',subject=subject):
                self.fail('Unknown slot file admitted')

    @unittest.skipUnless(os.name == 'nt', 'Real Win32 sharing contract')
    def test_receipt_counts_keep_ordinary_ignored_and_reject_failures(self):
        # Deliberately SYNTHETIC parser transcript; never project Cargo evidence.
        subject = self.subject()
        with self.gate.prepare_candidate(self.root,self.index,self.producer,tools=self.tools,generation='synthetic-case',subject=subject) as candidate:
            proof = candidate.proof(); command = [self.tools['cargo']['path'],*candidate.command('check')]
            self.addCleanup(lambda cwd=candidate.cwd: cwd.rmdir() if cwd.exists() and not any(cwd.iterdir()) else None)
        self.put('.agent_tx/fixture/capture.json',json.dumps(subject))
        self.put('.agent_tx/fixture/transcript.log','SYNTHETIC PARSER FIXTURE; NO CARGO EXECUTED\ntest optional ... ignored\ntest result: ok. 2 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.01s\n')
        counts = dict(suites=1,passed=2,failed=0,ignored=1,measured=0,filtered_out=0)
        row = dict(schema=self.gate.RECEIPT,generation='synthetic-case',request_ids=['synthetic-parser-fixture'],
                   kind='publication-phase',phase='check',feedback_only=False,state='PASSED',argv=command,cwd=proof['cwd'],
                   env_profile=subject['profile'],available_cpus=int(subject['profile']['CARGO_BUILD_JOBS']),
                   before=self.ref('.agent_tx/fixture/capture.json'),after=self.ref('.agent_tx/fixture/capture.json'),
                   before_inputs_digest=subject['inputs_digest'],after_inputs_digest=subject['inputs_digest'],
                   log_ref=self.ref('.agent_tx/fixture/transcript.log'),exit_code=0,error=None,diagnostics_by_file=[],
                   runner_receipts=[],test_counts=counts,observed_utc='SYNTHETIC_TEMP_ONLY',producer_reference=self.producer,
                   writer={'path':'scripts/compiler_bus.py','sha256':self.review['helpers']['scripts/compiler_bus.py']},
                   candidate=proof,candidate_released=True,candidate_cleanup_error=None)
        def validate(value):
            self.put('.agent_tx/fixture/parser-receipt.json',json.dumps(value))
            return self.gate.validate_run(self.root,subject,'check',self.ref('.agent_tx/fixture/parser-receipt.json'),self.review)
        self.assertEqual(validate(row)[0]['test_counts']['ignored'],1)
        bad = copy.deepcopy(row); bad['test_counts']['ignored'] = 0
        with self.assertRaisesRegex(RuntimeError,'counts/status'): validate(bad)
        bad = copy.deepcopy(row); bad['candidate_released'] = False
        with self.assertRaisesRegex(RuntimeError,'certain cleanup'): validate(bad)
        self.put('.agent_tx/fixture/transcript.log','test regression ... FAILED\n')
        bad = copy.deepcopy(row); bad['log_ref'] = self.ref('.agent_tx/fixture/transcript.log')
        with self.assertRaisesRegex(RuntimeError,'Failed test output'): validate(bad)

    @unittest.skipUnless(os.name == 'nt', 'Real Win32 sharing contract')
    def test_workspace_formatter_real_module_failure_and_omitted_target_rejected(self):
        # Real metadata/rustfmt in owned Temp only; no Root Native receipt.
        slot = self.base / 'format slot'; slot.mkdir(); cwd = self.base / 'format cwd'; cwd.mkdir()
        def put(name, raw):
            path = slot / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw.encode())
        put('Cargo.toml', '[workspace]\nmembers=["pkg2021", "pkg2024"]\nresolver="2"\n')
        put('pkg2021/Cargo.toml', '[package]\nname="format-temp-contract-2021"\nversion="0.1.0"\nedition="2021"\nbuild=false\n[lib]\npath="../src/lib.source"\n[[example]]\nname="inactive"\npath="../examples/inactive.source"\nrequired-features=["inactive"]\n[features]\ninactive=[]\n')
        put('pkg2024/Cargo.toml', '[package]\nname="format-temp-contract-2024"\nversion="0.1.0"\nedition="2024"\nbuild=false\n[lib]\npath="../src/lib.source"\n')
        put('src/lib.source', 'mod child;\n')
        put('src/child.rs', 'pub fn child() {}\n')
        put('examples/inactive.source', 'fn main() {}\n')
        put('scripts/format_workspace.py', (SCRIPTS / 'format_workspace.py').read_text())
        raw = {path.relative_to(slot).as_posix(): self.hash(path.read_bytes()) for path in slot.rglob('*') if path.is_file()}
        native = {'cargo': self.tool(NATIVE_RUSTFMT.with_name('cargo.exe')), 'rustfmt': self.tool(NATIVE_RUSTFMT),
                  'python': self.tool(Path(sys.executable).resolve())}
        self.assertEqual(native['rustfmt']['sha256'], '25a61fa2f18b3b5f51b39a4e1cc2a4463aabe64daf31b6f58961b760e7f90497')
        subject = {'raw': raw, 'tools': native, 'inputs_digest': 'f' * 64}
        candidate = {'root': str(self.root), 'slot': str(slot), 'cwd': str(cwd), 'configs': []}
        review = {'helpers': {'scripts/format_workspace.py': raw['scripts/format_workspace.py']}}
        marker = {'schema': 'trader-gemini-compiler-candidate-v1', 'root': str(self.root),
                  'files': raw, 'candidate_id': subject['inputs_digest']}
        (slot / '.candidate.json').write_bytes(self.gate.canonical(marker) + b'\n')
        put('.zero-wait-rustfmt.toml', '')
        argv = self.gate.candidate_argv('fmt', slot, [], native)
        self.assertEqual(argv[:4], [sys.executable, '-I', '-B', str(slot / 'scripts/format_workspace.py')])
        env = dict(os.environ); env['RUSTC'] = str(NATIVE_RUSTFMT.with_name('rustc.exe'))
        env['CARGO_TARGET_DIR'] = str(self.base / 'format target')
        process = subprocess.run(argv, cwd=cwd, env=env, capture_output=True)
        self.assertEqual(process.returncode, 0, process.stdout.decode() + process.stderr.decode())
        report = self.gate.validate_format(process.stdout, subject, candidate, review)
        self.assertEqual({Path(row['path']).suffix for row in report['inventory']}, {'.source'})
        self.assertEqual(len(report['inventory']), 3)  # Two editions of one source plus inactive example.
        shared = str(slot / 'src/lib.source')
        self.assertEqual([row for row in report['inventory'] if row['path'] == shared],
                         [{'path': shared, 'edition': '2021'}, {'path': shared, 'edition': '2024'}])
        self.assertEqual([(row['argv'][2], shared in row['argv'][6:]) for row in report['batches']],
                         [('2021', True), ('2024', True)])
        for mutated in ('inventory', 'batches'):
            bad = copy.deepcopy(report); bad[mutated].pop()
            with self.assertRaisesRegex(RuntimeError, 'inventory|batch'):
                self.gate.validate_format(json.dumps(bad), subject, candidate, review)
        # Default rustfmt traversal must inspect an unlisted child module.
        put('src/child.rs', 'pub fn child( ) { }\n')
        failed = subprocess.run(argv, cwd=cwd, env=env, capture_output=True)
        self.assertNotEqual(failed.returncode, 0)
        failed_report = json.loads(failed.stdout)
        self.assertTrue(any(row['exit_code'] != 0 for row in failed_report['batches']))
        # A forged successful summary cannot cover an actual failing child.
        failed_report.update(status='PASSED', exit_code=0, error=None)
        # Rebind only the deliberately changed fixture bytes, not its child exit.
        changed = self.hash((slot / 'src/child.rs').read_bytes())
        subject['raw']['src/child.rs'] = changed
        marker['files'] = subject['raw']; (slot / '.candidate.json').write_bytes(self.gate.canonical(marker) + b'\n')
        expected = {**subject['raw'], '.zero-wait-rustfmt.toml': self.hash(b''),
                    '.candidate.json': self.hash(self.gate.canonical(marker) + b'\n')}
        failed_report.update(raw_before=expected, raw_after=expected)
        with self.assertRaisesRegex(RuntimeError, 'child argv/status'):
            self.gate.validate_format(json.dumps(failed_report), subject, candidate, review)

    def test_workspace_formatter_batches_use_actual_windows_utf16_bound(self):
        spec = importlib.util.spec_from_file_location('temp_format_planner', SCRIPTS / 'format_workspace.py')
        helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
        inventory = [{'path': 'C:\\long path\\' + '\U0001f680' * 35 + str(i) + '.source', 'edition': '2021'} for i in range(700)]
        commands = helper.planned_batches('C:\\native tools\\rustfmt.exe', 'C:\\slot\\empty.toml', inventory)
        self.assertGreater(len(commands), 1)
        self.assertEqual([path for command in commands for path in command[6:]], sorted(row['path'] for row in inventory))
        for command in commands:
            actual = len(subprocess.list2cmdline(command).encode('utf-16-le')) // 2 + 1
            self.assertLessEqual(actual, 32767)
        too_long = [{'path': 'C:\\' + '\U0001f680' * 17000, 'edition': '2021'}]
        with self.assertRaisesRegex(RuntimeError, 'Single target'):
            helper.planned_batches('rustfmt', 'empty.toml', too_long)

    def test_python_evidence_requires_slot_and_certain_release(self):
        # Synthetic protocol transcript counts are not project suite results.
        subject = self.subject()
        with self.gate.prepare_candidate(self.root,self.index,self.producer,tools=self.tools,generation='synthetic-case',subject=subject) as candidate:
            proof = candidate.proof(); cwd = candidate.cwd
            self.addCleanup(lambda: cwd.rmdir() if cwd.exists() and not any(cwd.iterdir()) else None)
        specs = {}; refs = {}; rows = {}
        for name,count in (('legacy',30),('zero_wait',1),('bus',1)):
            command = [self.tools['python']['path'],'-I','-B',str(Path(proof['slot'])/'scripts/compiler_bus.py')]
            specs[name] = {'command':command,'tests_run':count}
            self.put('.agent_tx/fixture/python-'+name+'.log','SYNTHETIC PARSER TRANSCRIPT ONLY\nRan '+str(count)+' tests in 0.001s\n\nOK\n')
            rows[name] = dict(schema='trader-gemini-zero-wait-python-v1',command=command,
                inputs_digest=subject['inputs_digest'],before_inputs_digest=subject['inputs_digest'],after_inputs_digest=subject['inputs_digest'],
                exit_code=0,tests_run=count,failures=0,errors=0,skipped=0,log_ref=self.ref('.agent_tx/fixture/python-'+name+'.log'),
                candidate=proof,candidate_released=True,candidate_cleanup_error=None)
        def validate(values,registered):
            for name,value in values.items():
                path='.agent_tx/fixture/python-'+name+'.json';self.put(path,json.dumps(value));refs[name]=self.ref(path)
            return self.gate.validate_python(self.root,subject,refs,{**self.review,'python_suites':registered})
        validate(rows,specs)
        bad = copy.deepcopy(rows); bad['bus']['candidate_released'] = False
        with self.assertRaisesRegex(RuntimeError,'certain cleanup'): validate(bad,specs)
        bad = copy.deepcopy(rows); registered=copy.deepcopy(specs)
        bad['bus']['command'][3] = str(self.root/'scripts/compiler_bus.py');registered['bus']['command']=bad['bus']['command']
        with self.assertRaisesRegex(RuntimeError,'protected candidate copy'): validate(bad,registered)

    def test_purge_compares_against_fresh_origin_main(self):
        source = (SCRIPTS / 'zero_wait_bus.ps1').read_text(encoding='utf-8-sig')
        purge = source.split('"PurgeLegacyBranches" {', 1)[1].split('"StageFiles" {', 1)[0]
        self.assertIn('git ls-remote --heads origin refs/heads/main', source)
        self.assertIn('git rev-parse refs/remotes/origin/main', source)
        self.assertIn('$freshOriginMain = Get-FreshOriginMain', purge)
        self.assertEqual(purge.count('Test-BranchPurgeAdmission -Branch'), 2)
        self.assertIn('git merge-base --is-ancestor $Tip $FreshOriginMain', source)
        self.assertIn('git rev-list --count "${Tip}..${FreshOriginMain}"', source)
        self.assertNotIn('git merge-base --is-ancestor $br main', source)
        self.assertNotIn('git merge-base --is-ancestor "origin/$rbr" main', source)

    def test_purge_preserves_unmerged_leased_and_worktree_branches(self):
        source = (SCRIPTS / 'zero_wait_bus.ps1').read_text(encoding='utf-8-sig')
        worktree_guard = source.split('function Test-WorktreeReservation', 1)[1].split('function ', 1)[0]
        lease_guard = source.split('function Test-LeaseOrReservation', 1)[1].split('function ', 1)[0]
        self.assertIn('branch refs/heads/', worktree_guard)
        self.assertIn('HEAD ', worktree_guard)
        self.assertIn('AGENT_SYNC.md', lease_guard)
        self.assertIn('Get-ChildItem', lease_guard)
        self.assertIn('-Recurse', lease_guard)
        self.assertIn('(?i)(lease|reserva|reservation)', lease_guard)
        self.assertIn('function Test-MergedBranchEvidence', source)
        self.assertIn('gh pr list --head $Branch --state all', source)
        self.assertIn('headRefOid', source)
        self.assertIn('mergeCommit', source)
        self.assertIn('MERGED', source)
        self.assertIn('git show -s --format=%P $mergeSha', source)

    def test_a4_manifest_separates_original_ten_from_revised_seven(self):
        original = {
            'crates/execution-engine/src/entry_dispatch.rs',
            'crates/execution-engine/src/execution_evidence.rs',
            'crates/execution-engine/src/executor.rs',
            'crates/execution-engine/tests/entry_route_contract.rs',
            'crates/execution-engine/tests/f325_min_notional_ioc_open_diagnostics.rs',
            'crates/quantum-arena/src/symbol_registry.rs',
            'crates/quantum-arena/src/symbols.rs',
            'src/bin/god_engine.rs',
            'src/host_oms.rs',
            'tests/host_oms_contract.rs',
        }
        revised_expected = {
            'crates/execution-engine/src/entry_dispatch.rs',
            'crates/execution-engine/tests/entry_route_contract.rs',
            'crates/execution-engine/tests/f325_min_notional_ioc_open_diagnostics.rs',
            'crates/quantum-arena/src/symbol_registry.rs',
            'src/bin/god_engine.rs',
            'src/host_oms.rs',
            'tests/host_oms_contract.rs',
        }
        sync = (SCRIPTS.parent / 'AGENT_SYNC.md').read_text(encoding='utf-8-sig')
        section = sync.split('### Manifiesto A4 — corte F0', 1)[1].split('\n## ', 1)[0]
        rows = {}
        for line in section.splitlines():
            if not line.startswith('| `'):
                continue
            cells = [cell.strip() for cell in line.strip('|').split('|')]
            path = cells[0].strip('`')
            if path in original:
                rows[path] = cells
        self.assertEqual(set(rows), original)
        revised = set()
        for path, cells in rows.items():
            self.assertEqual(len(cells), 6)
            self.assertIn('AGENT-01', cells[1])
            self.assertTrue(cells[2])
            base = cells[3].strip('`').partition(':')
            candidate = cells[4].strip('`').partition(':')
            self.assertEqual(base[0], 'root')
            self.assertEqual(len(base[2]), 64)
            int(base[2], 16)
            self.assertEqual(len(candidate[2]), 64)
            int(candidate[2], 16)
            if candidate[0] == 'staging':
                revised.add(path)
                self.assertNotEqual(base[2], candidate[2])
            else:
                self.assertEqual(candidate[0], 'root')
                self.assertEqual(base[2], candidate[2])
        self.assertEqual(revised, revised_expected)
        for relative in ('docs/PLAN_MAESTRO_FASES_2026-10-22.md', 'docs/BINANCE_API_MAP.md'):
            document = (SCRIPTS.parent / relative).read_text(encoding='utf-8-sig').lower()
            self.assertIn('comportamiento instalado', document)
            self.assertIn('candidato revisado no instalado', document)


if __name__ == '__main__':
    unittest.main(verbosity=2)
