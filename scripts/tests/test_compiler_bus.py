"""Hermetic tests: Python compiler sentinel, real Git/rustfmt in owned Temp only."""
import hashlib
import ctypes
from ctypes import wintypes
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest

HERE = Path(__file__).resolve().parent
MODULES = HERE if (HERE/'compiler_bus.py').is_file() else HERE.parent


def module(name):
    spec = importlib.util.spec_from_file_location(name, MODULES/(name+'.py'))
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


bus, staging = module('compiler_bus'), module('private_staging')
TEMPS = []


class TempGate:
    """Test-only adapter; never represented as the production Gate or Native."""
    def __init__(self, root, mode='ok'):
        self.root = root; self.fail_capture = False
        self.candidate_active = False
        self.COMMANDS = {'check':('-I','-B',str(root/'sentinel.py'),mode)}

    def execution_env(self, root, tools):
        return {'CARGO_BUILD_JOBS':str(os.process_cpu_count() or os.cpu_count()),
                'CARGO_TARGET_DIR':str(root/'target'), 'RUST_TEST_THREADS':None,
                'MAKEFLAGS':None, 'CARGO_MAKEFLAGS':None, 'NUM_JOBS':None}

    def capture_subject(self, root, index, producer, *, tools, generation):
        if self.fail_capture:
            raise ValueError('Own fixture capture uncertainty')
        raw = {'source.rs':bus.sha((root/'source.rs').read_bytes())}
        return {'inputs_digest':bus.sha(bus.canonical(raw)), 'raw':raw,
                'generation':generation, 'test_only_no_authority':True,
                'candidate_active_during_observation':self.candidate_active}

    def runner_receipts(self, root, producer):
        return []

    def prepare_candidate(self, root, index, producer, *, tools, generation, subject):
        return TempCandidate(self,root,subject,tools)


class TempCandidate:
    """Only a fixture for the bus context API; no production materializer claim."""
    def __init__(self, gate, root, subject, tools):
        self.gate,self.root,self.subject = gate,root,subject
        self.tools = tools
        self.slot = root/'.agent_tx/staging/compiler-candidate'
        self.cwd = Path(tempfile.mkdtemp(prefix='tg-compiler-cwd-')).resolve()
        TEMPS.append(str(self.cwd)); self.handle = None

    def __enter__(self):
        self.slot.mkdir(parents=True,exist_ok=True); (self.slot/'source.rs').write_bytes((self.root/'source.rs').read_bytes())
        if os.name == 'nt':
            self.api = ctypes.WinDLL('kernel32',use_last_error=True)
            self.api.CreateFileW.argtypes = [wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
            self.api.CreateFileW.restype = wintypes.HANDLE
            self.api.CloseHandle.argtypes = [wintypes.HANDLE]; self.api.CloseHandle.restype = wintypes.BOOL
            self.handle = self.api.CreateFileW(str(self.slot/'source.rs'),0x80000000,1,None,3,0,None)
            assert self.handle != ctypes.c_void_p(-1).value
        self.gate.candidate_active = True; return self

    def command(self,phase):
        return (*self.gate.COMMANDS[phase],str(self.root),str(self.slot))

    def argv(self,phase):
        return [self.tools['cargo']['path'], *self.command(phase)]

    def proof(self):
        assert self.gate.candidate_active
        return {'fixture_only':True,'candidate_id':self.subject['inputs_digest'],
                'slot_source_sha256':bus.sha((self.slot/'source.rs').read_bytes())}

    def __exit__(self,*errors):
        if self.handle is not None:assert self.api.CloseHandle(self.handle)
        self.gate.candidate_active = False


class BusTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='tg-compiler-bus-')).resolve()
        assert self.root.parent == Path(tempfile.gettempdir()).resolve()
        TEMPS.append(str(self.root))
        (self.root/'source.rs').write_bytes(b'fn main() {}\n')
        (self.root/'sentinel.py').write_text('''import json,os,pathlib,sys,time
r=pathlib.Path(sys.argv[2]) if len(sys.argv)>2 else pathlib.Path.cwd(); mode=sys.argv[1]
call={'argv':sys.argv,'jobs':os.environ.get('CARGO_BUILD_JOBS'),'target':os.environ.get('CARGO_TARGET_DIR'),'threads':os.environ.get('RUST_TEST_THREADS'),'cwd':str(pathlib.Path.cwd())}
if len(sys.argv)>3:
 slot=pathlib.Path(sys.argv[3]);call['slot_read']=(slot/'source.rs').read_text()
 try:(slot/'source.rs').write_text('unexpected write');call['slot_writer_denied']=False
 except PermissionError:call['slot_writer_denied']=True
with (r/'calls.jsonl').open('a') as f:f.write(json.dumps(call)+'\\n')
if mode=='block':
 (r/'started').write_text('started')
 while not (r/'release').exists():time.sleep(.01)
if mode=='mutate':(r/'source.rs').write_text('fn changed() {}\\n')
if mode=='counts':
 print('test result: ok. 2 passed; 0 failed; 25 ignored; 0 measured; 3 filtered out; finished in 0.01s')
 print('test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s')
print('42')
print(json.dumps({'reason':'compiler-message','message':{'level':'warning','message':'sentinel','spans':[{'file_name':'source.rs'}]}}))
if mode=='unsafe-span':print(json.dumps({'reason':'compiler-message','message':{'spans':[{'file_name':'../../escape.rs'}]}}))
if mode=='absolute-slot-span':
 span=slot/'crates/example/src/lib.rs';span.parent.mkdir(parents=True,exist_ok=True);span.write_bytes((r/'source.rs').read_bytes())
 print(json.dumps({'reason':'compiler-message','message':{'level':'warning','message':'absolute-slot-sentinel','spans':[{'file_name':str(span)}]}}))
sys.exit(7 if mode=='fail' else 0)
''', encoding='utf-8')

    def worker(self, mode='ok'):
        gate = TempGate(self.root, mode)
        return bus.CompilerBus(self.root,self.root/'.git/index_temp',{'path':'test-only','sha256':'0'*64},
                               {'cargo':{'path':sys.executable}},gate)

    def test_grouped_pass_and_diagnostic_kind(self):
        bus.enqueue(self.root,'one','check',['source.rs'])
        bus.enqueue(self.root,'two','check',['source.rs'])
        receipts = self.worker().drain_once()
        self.assertEqual(len(receipts),1); receipt = receipts[0]
        self.assertEqual(len(receipt['request_ids']),2); self.assertEqual(receipt['state'],'PASSED')
        self.assertTrue(receipt['feedback_only']); self.assertEqual(receipt['exit_code'],0)
        self.assertEqual(receipt['producer_reference'],{'path':'test-only','sha256':'0'*64})
        self.assertEqual(receipt['writer']['sha256'],hashlib.sha256((MODULES/'compiler_bus.py').read_bytes()).hexdigest())
        calls = (self.root/'calls.jsonl').read_text().splitlines(); self.assertEqual(len(calls),1)
        profile = json.loads(calls[0]); self.assertEqual(profile['jobs'],str(os.process_cpu_count() or os.cpu_count()))
        self.assertEqual(Path(profile['target']),self.root/'target'); self.assertIsNone(profile['threads'])
        diagnostic = json.loads((self.root/'.agent_tx/diagnostics/source.rs.json').read_bytes())
        self.assertEqual(diagnostic['generation'],receipt['generation']); self.assertEqual(len(diagnostic['messages']),1)

    def test_enqueue_returns_while_own_pass_is_blocked(self):
        worker = self.worker('block'); bus.enqueue(self.root,'one','check',['source.rs'])
        result = []; thread = threading.Thread(target=lambda:result.extend(worker.drain_once()))
        thread.start(); deadline = time.monotonic()+10
        while not (self.root/'started').exists() and time.monotonic()<deadline:
            time.sleep(.01)
        self.assertTrue((self.root/'started').exists())
        second = bus.enqueue(self.root,'two','check',['source.rs'])
        self.assertTrue(thread.is_alive()); self.assertFalse((self.root/'release').exists())
        self.assertTrue((worker.pending/(second['id']+'.json')).exists())
        (self.root/'release').write_text('own release'); thread.join(10)
        self.assertFalse(thread.is_alive()); self.assertEqual(result[0]['state'],'PASSED')

    def test_stale_source_requeues_without_pass_transfer(self):
        bus.enqueue(self.root,'one','check',['source.rs'],kind='publication-phase')
        worker = self.worker('mutate'); first = worker.drain_once()[0]
        self.assertEqual(first['exit_code'],0); self.assertEqual(first['state'],'STALE')
        self.assertFalse(first['feedback_only'])
        queued = list((worker.directory/'deferred').glob('*.json')); self.assertEqual(len(queued),1)
        self.assertEqual(json.loads(queued[0].read_bytes())['prior_generation'],first['generation'])
        self.assertEqual(worker.drain_once(),[])  # No self-triggered repeated compilation.
        self.assertEqual(bus.inputs_changed(self.root),1)
        second = worker.drain_once()[0]
        self.assertEqual(second['state'],'PASSED'); self.assertNotEqual(first['generation'],second['generation'])

    @unittest.skipUnless(os.name=='nt','Windows immutable-copy read pin contract')
    def test_publication_candidate_is_pinned_while_root_changes_and_after_capture(self):
        bus.enqueue(self.root,'one','check',['source.rs'],kind='publication-phase')
        receipt = self.worker('mutate').drain_once()[0]
        self.assertEqual(receipt['state'],'STALE'); self.assertEqual(receipt['exit_code'],0)
        self.assertTrue(receipt['candidate_released']); self.assertIsNone(receipt['candidate_cleanup_error'])
        self.assertEqual(receipt['candidate']['slot_source_sha256'],bus.sha(b'fn main() {}\n'))
        self.assertNotEqual(Path(receipt['cwd']),self.root)
        call = json.loads((self.root/'calls.jsonl').read_text().splitlines()[0])
        self.assertEqual(call['slot_read'],'fn main() {}\n'); self.assertTrue(call['slot_writer_denied'])
        after = json.loads((self.root/receipt['after']['path']).read_bytes())
        self.assertTrue(after['candidate_active_during_observation'])
        self.assertEqual((self.root/'source.rs').read_text(),'fn changed() {}\n')

    def test_publication_executes_actual_candidate_argv_without_cargo_rewrap(self):
        worker = self.worker()
        # A real Python sentinel is the candidate's declared executable. The
        # intentionally nonexistent Cargo path proves no second wrapper runs.
        worker.tools['cargo']['path'] = str(self.root / 'cargo-must-not-execute.exe')
        expected = [sys.executable, '-I', '-B', str(self.root/'sentinel.py'), 'ok', str(self.root),
                    str(self.root/'.agent_tx/staging/compiler-candidate')]
        original = worker.gate.prepare_candidate
        def prepare(*args, **kwargs):
            candidate = original(*args, **kwargs)
            candidate.argv = lambda phase: expected
            return candidate
        worker.gate.prepare_candidate = prepare
        bus.enqueue(self.root,'actual-wrapper','check',['source.rs'],kind='publication-phase')
        receipt = worker.drain_once()[0]
        self.assertEqual(receipt['argv'], expected)
        self.assertEqual((receipt['state'], receipt['exit_code']), ('PASSED', 0))
        self.assertTrue(receipt['candidate_released'])
        calls = [json.loads(line) for line in (self.root/'calls.jsonl').read_text().splitlines()]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]['argv'], expected[3:])

    def test_real_child_failure_and_untrusted_diagnostic_path(self):
        bus.enqueue(self.root,'one','check',['source.rs']); failed = self.worker('fail').drain_once()[0]
        self.assertEqual((failed['state'],failed['exit_code']),('FAILED',7))
        bus.enqueue(self.root,'one','check',['source.rs']); receipt = self.worker('unsafe-span').drain_once()[0]
        self.assertIn('_unmapped',receipt['diagnostics_by_file'])
        self.assertFalse((self.root.parent/'escape.rs.json').exists())

    def test_absolute_slot_span_maps_to_original_source_diagnostic(self):
        bus.enqueue(self.root,'one','check',['source.rs'],kind='publication-phase')
        receipt = self.worker('absolute-slot-span').drain_once()[0]
        self.assertEqual(receipt['state'],'PASSED')  # Test-only Python adapter, not a Cargo result.
        name = 'crates/example/src/lib.rs'
        slot_source = self.root/'.agent_tx/staging/compiler-candidate'/name
        self.assertTrue(slot_source.is_file())
        self.assertEqual(slot_source.read_bytes(),(self.root/'source.rs').read_bytes())
        self.assertIn(name,receipt['diagnostics_by_file'])
        self.assertNotIn('.agent_tx/staging/compiler-candidate/'+name,receipt['diagnostics_by_file'])
        diagnostic = json.loads((self.root/'.agent_tx/diagnostics'/(name+'.json')).read_bytes())
        self.assertEqual(diagnostic['messages'][0]['message'],'absolute-slot-sentinel')
        self.assertEqual(Path(diagnostic['messages'][0]['spans'][0]['file_name']),slot_source)
        self.assertFalse((self.root/'.agent_tx/diagnostics/.agent_tx').exists())
        self.assertTrue(receipt['candidate_released'])
        self.assertIsNone(receipt['candidate_cleanup_error'])

    def test_unknown_capture_is_stale_and_never_launches(self):
        worker = self.worker(); worker.gate.fail_capture = True
        bus.enqueue(self.root,'one','check',['source.rs']); receipt = worker.drain_once()[0]
        self.assertEqual(receipt['state'],'STALE'); self.assertIsNone(receipt['exit_code'])
        self.assertFalse((self.root/'calls.jsonl').exists())

    def test_profile_exception_terminalizes_claim_and_defers_without_launch(self):
        worker = self.worker()
        def broken(*args):raise ValueError('own profile fixture failure')
        worker.gate.execution_env = broken
        bus.enqueue(self.root,'one','check',['source.rs']); receipt = worker.drain_once()[0]
        self.assertEqual(receipt['state'],'STALE'); self.assertIn('profile fixture',receipt['error'])
        self.assertEqual(list((worker.directory/'processing').glob('*.json')),[])
        self.assertEqual(len(list((worker.directory/'deferred').glob('*.json'))),1)
        self.assertFalse((self.root/'calls.jsonl').exists())

    def test_malformed_request_is_preserved_without_aborting_valid_batch(self):
        worker = self.worker(); (worker.pending/'000-invalid.json').write_bytes(b'{truncated')
        bus.enqueue(self.root,'one','check',['source.rs']); receipts = worker.drain_once()
        self.assertEqual(len(receipts),2)
        rejected = next(r for r in receipts if r['schema']=='trader-gemini-compiler-bus-rejection-v1')
        accepted = next(r for r in receipts if r['schema']=='trader-gemini-compiler-bus-receipt-v1')
        self.assertEqual(rejected['state'],'FAILED'); self.assertFalse(rejected['compiler_started'])
        self.assertEqual((self.root/rejected['original_request']['path']).read_bytes(),b'{truncated')
        self.assertEqual(accepted['state'],'PASSED'); self.assertEqual(list((worker.directory/'processing').glob('*.json')),[])

    def test_foreign_processing_collision_preserves_both_and_valid_request_runs(self):
        worker = self.worker()
        collided = bus.enqueue(self.root,'one','check',['source.rs'])
        path = worker.pending/(collided['id']+'.json'); before = path.read_bytes()
        processing = worker.directory/'processing'; processing.mkdir()
        foreign = processing/path.name; foreign.write_bytes(b'foreign original claim')
        bus.enqueue(self.root,'two','check',['source.rs']); receipts = worker.drain_once()
        rejected = next(r for r in receipts if r['schema']=='trader-gemini-compiler-bus-rejection-v1')
        accepted = next(r for r in receipts if r['schema']=='trader-gemini-compiler-bus-receipt-v1')
        self.assertTrue(rejected['preserved_in_place']); self.assertFalse(rejected['owned_claim'])
        self.assertIsNone(rejected['original_request']); self.assertEqual(accepted['state'],'PASSED')
        self.assertEqual(path.read_bytes(),before); self.assertEqual(foreign.read_bytes(),b'foreign original claim')

    @unittest.skipUnless(os.name=='nt','NTFS directory junction namespace contract')
    def test_reparse_entry_is_not_followed_or_moved_and_valid_request_runs(self):
        worker = self.worker(); target = self.root/'junction-data'; target.mkdir()
        sentinel = target/'sentinel'; sentinel.write_bytes(b'untouched target')
        entry = worker.pending/'000-reparse.json'
        result = subprocess.run(['cmd.exe','/d','/c','mklink','/J',str(entry),str(target)],capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        identity = entry.lstat().st_ino
        bus.enqueue(self.root,'valid','check',['source.rs']); receipts = worker.drain_once()
        rejected = next(r for r in receipts if r['schema']=='trader-gemini-compiler-bus-rejection-v1')
        accepted = next(r for r in receipts if r['schema']=='trader-gemini-compiler-bus-receipt-v1')
        self.assertTrue(rejected['preserved_in_place']); self.assertFalse(rejected['owned_claim'])
        self.assertIsNone(rejected['original_request']); self.assertEqual(accepted['state'],'PASSED')
        self.assertEqual(entry.lstat().st_ino,identity); self.assertEqual(sentinel.read_bytes(),b'untouched target')

    def test_test_counts_preserve_legitimate_global_ignored(self):
        bus.enqueue(self.root,'one','check',['source.rs']); receipt = self.worker('counts').drain_once()[0]
        self.assertEqual(receipt['test_counts'],{'suites':2,'passed':3,'failed':0,'ignored':25,'measured':0,'filtered_out':3})

    def test_client_cli_is_operational_and_bad_gate_sha_never_imports(self):
        result = subprocess.run([sys.executable,'-I','-B',str(MODULES/'compiler_bus.py'),'enqueue','--repo',str(self.root),
                                 '--agent','cli','--phase','check','--file','source.rs'],capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr); self.assertEqual(json.loads(result.stdout)['agent'],'cli')
        gate = self.root/'unreviewed.py'; gate.write_text("raise RuntimeError('must not import')\n")
        result = subprocess.run([sys.executable,'-I','-B',str(MODULES/'compiler_bus.py'),'once','--repo',str(self.root),
            '--private-index',str(self.root/'.git/index_cli'),'--producer-path','none','--producer-sha256','0'*64,
            '--tools-json',str(self.root/'absent.json'),'--gate',str(gate),'--gate-sha256','0'*64],capture_output=True)
        self.assertEqual(result.returncode,75); self.assertIn(b'Gate SHA changed',result.stderr)
        self.assertFalse((self.root/'calls.jsonl').exists())

    def test_promoted_layout_resolves_modules_and_cli_without_source_outputs(self):
        layout = self.root/'promoted'; scripts = layout/'scripts'; tests = scripts/'tests'
        tests.mkdir(parents=True)
        for name in ('compiler_bus.py','private_staging.py'):
            (scripts/name).write_bytes((MODULES/name).read_bytes())
        test_path = tests/'test_compiler_bus.py'; test_path.write_bytes(Path(__file__).read_bytes())
        originals = {p:p.read_bytes() for p in (scripts/'compiler_bus.py',scripts/'private_staging.py',test_path)}
        output = self.root/'explicit-test-report'; output.mkdir()
        result = subprocess.run([sys.executable,'-I','-B',str(test_path),
            'BusTests.test_client_cli_is_operational_and_bad_gate_sha_never_imports'],
            env={**os.environ,'TG_BUS_TEST_OUTPUT':str(output)},capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertTrue((output/'TEMP_INVENTORY.json').is_file())
        self.assertFalse((tests/'TEMP_INVENTORY.json').exists()); self.assertFalse((scripts/'TEMP_INVENTORY.json').exists())
        self.assertEqual({p:p.read_bytes() for p in originals},originals)

    def test_persistent_capture_error_waits_for_an_explicit_new_revision(self):
        worker = self.worker(); worker.gate.fail_capture = True
        bus.enqueue(self.root,'one','check',['source.rs']); worker.drain_once()
        self.assertEqual(worker.drain_once(),[])
        self.assertEqual(len(list((worker.directory/'deferred').glob('*.json'))),1)
        bus.enqueue(self.root,'two','check',['source.rs'])
        receipt = worker.drain_once()[0]
        self.assertEqual(len(receipt['request_ids']),2); self.assertEqual(receipt['state'],'STALE')
        self.assertEqual(worker.drain_once(),[]); self.assertFalse((self.root/'calls.jsonl').exists())

    @unittest.skipUnless(os.name=='nt','Windows kernel directory event contract')
    def test_real_windows_directory_notification_and_own_stop(self):
        pending = self.root/'pending'; pending.mkdir(); watcher = bus.DirectoryWake(pending)
        try:
            (pending/'request').write_text('own notification')
            self.assertTrue(watcher.wait(10000)); watcher.stop(); self.assertFalse(watcher.wait(10000))
        finally:
            watcher.close()

    def test_syntax_buffer_rejects_truncation_without_overwriting_valid(self):
        binary = Path(os.environ.get('RUSTFMT') or 'C:/Users/jhona/.rustup/toolchains/nightly-2026-06-30-x86_64-pc-windows-msvc/bin/rustfmt.exe')
        formatter = {'path':str(binary),'sha256':'25a61fa2f18b3b5f51b39a4e1cc2a4463aabe64daf31b6f58961b760e7f90497'}
        good = staging.stage_rust(self.root,'agent','source.rs',b'fn main() {}\n',formatter)
        self.assertEqual(good['exit_code'],0); target = self.root/good['candidate_path']; original = target.read_bytes()
        bad = staging.stage_rust(self.root,'agent','source.rs',b'fn main() {',formatter)
        self.assertNotEqual(bad['exit_code'],0); self.assertIsNone(bad['candidate_path'])
        self.assertEqual(target.read_bytes(),original); self.assertEqual((self.root/bad['pending_path']).read_bytes(),b'fn main() {')
        self.assertEqual((self.root/'source.rs').read_bytes(),b'fn main() {}\n')

    def test_private_index_clean_filter_and_merge_tree_preserve_primary(self):
        git = Path(shutil.which('git')).resolve()
        def command(*args, data=None, index=None):
            env = {k:v for k,v in os.environ.items() if not k.startswith('GIT_')}
            if index:env.update(staging.git_environment(self.root,index))
            r = subprocess.run([str(git),'--no-optional-locks','-C',str(self.root),'-c','user.name=Temp','-c','user.email=temp@example.invalid',*args],input=data,capture_output=True,env=env)
            self.assertEqual(r.returncode,0,r.stderr.decode('utf-8','replace')); return r.stdout.strip()
        command('init','-q'); command('config','core.autocrlf','true')
        (self.root/'other.rs').write_bytes(b'fn original() {}\n')
        command('add','source.rs','other.rs'); command('commit','-qm','own Temp baseline')
        head = command('rev-parse','HEAD').decode(); original_index = (self.root/'.git/index').read_bytes()
        index = staging.create_index(self.root,'agent',head,git)
        content = b'fn changed() {}\r\n'; candidate = self.root/'.agent_tx/staging/agent/source.rs'; candidate.parent.mkdir(parents=True); candidate.write_bytes(content)
        record = {'path':'source.rs','exit_code':0,'semantic_or_typecheck':False,'candidate_path':candidate.relative_to(self.root).as_posix(),'candidate_sha256':staging.sha(content)}
        oid = staging.index_staged(self.root,index,record,git)
        self.assertEqual(command('cat-file','blob',oid),b'fn changed() {}')
        tree = command('write-tree',index=index).decode()
        commit = command('commit-tree',tree,'-p',head,data=b'private candidate\n').decode()
        def branch(name,body):
            private = staging.create_index(self.root,name,head,git)
            path = 'other.rs' if name=='right' else 'source.rs'
            staged = self.root/'.agent_tx/staging'/name/path; staged.parent.mkdir(parents=True); staged.write_bytes(body)
            r = {'path':path,'exit_code':0,'semantic_or_typecheck':False,'candidate_path':staged.relative_to(self.root).as_posix(),'candidate_sha256':staging.sha(body)}
            staged_oid = staging.index_staged(self.root,private,r,git)
            t = command('write-tree',index=private).decode()
            return command('commit-tree',t,'-p',head,data=b'private divergent candidate\n').decode(),staged_oid
        right,other_oid = branch('right',b'fn right() {}\n')
        index_before_preview = index.read_bytes()
        preview = staging.merge_preview(self.root,index,commit,right,git)
        self.assertEqual(preview['exit_code'],0)
        self.assertIn(oid.encode(),command('ls-tree',preview['tree'],'source.rs'))
        self.assertIn(other_oid.encode(),command('ls-tree',preview['tree'],'other.rs'))
        conflict,_ = branch('conflict',b'fn incompatible() {}\n')
        conflict_preview = staging.merge_preview(self.root,index,commit,conflict,git)
        self.assertEqual(conflict_preview['exit_code'],1); self.assertTrue(conflict_preview['conflicted'])
        self.assertEqual(index.read_bytes(),index_before_preview)
        self.assertEqual((self.root/'.git/index').read_bytes(),original_index)
        self.assertEqual(command('rev-parse','HEAD').decode(),head)
        self.assertEqual((self.root/'source.rs').read_bytes(),b'fn main() {}\n')

    def test_external_filter_and_merge_driver_sentinels_never_execute(self):
        git = Path(shutil.which('git')).resolve()
        def command(*args,data=None,index=None):
            env = {k:v for k,v in os.environ.items() if not k.startswith('GIT_')}
            if index:env.update(staging.git_environment(self.root,index))
            r = subprocess.run([str(git),'--no-optional-locks','-C',str(self.root),'-c','user.name=Temp','-c','user.email=temp@example.invalid',*args],env=env,input=data,capture_output=True)
            self.assertEqual(r.returncode,0,r.stderr); return r.stdout.strip().decode()
        command('init','-q'); command('add','source.rs'); command('commit','-qm','base')
        head = command('rev-parse','HEAD'); index = staging.create_index(self.root,'agent',head,git)
        driver = self.root/'driver.py'; driver.write_text("from pathlib import Path;Path('DRIVER_RAN').write_text('bad')\n")
        external = '"'+sys.executable+'" -I -B "'+str(driver)+'"'
        command('config','filter.spy.clean',external); (self.root/'.gitattributes').write_text('source.rs filter=spy\n')
        staged = self.root/'.agent_tx/staging/agent/source.rs'; staged.parent.mkdir(parents=True); staged.write_bytes(b'fn staged() {}\n')
        record = {'path':'source.rs','exit_code':0,'semantic_or_typecheck':False,'candidate_path':staged.relative_to(self.root).as_posix(),'candidate_sha256':staging.sha(staged.read_bytes())}
        with self.assertRaisesRegex(ValueError,'filter attribute'):staging.index_staged(self.root,index,record,git)
        self.assertFalse((self.root/'DRIVER_RAN').exists())
        command('config','--unset','filter.spy.clean'); (self.root/'.gitattributes').write_text('')
        oid = staging.index_staged(self.root,index,record,git); tree = command('write-tree',index=index)
        candidate = command('commit-tree',tree,'-p',head,data=b'candidate\n')
        command('config','merge.default','spy'); command('config','merge.spy.driver',external)
        with self.assertRaisesRegex(ValueError,'merge driver'):staging.merge_preview(self.root,index,head,candidate,git)
        self.assertFalse((self.root/'DRIVER_RAN').exists())


if __name__ == '__main__':
    try:
        unittest.main(verbosity=2)
    finally:
        supplied = os.environ.get('TG_BUS_TEST_OUTPUT')
        output = Path(supplied).resolve() if supplied else Path(tempfile.mkdtemp(prefix='tg-bus-test-report-')).resolve()
        if not output.is_relative_to(Path(tempfile.gettempdir()).resolve()):
            raise ValueError('Test inventory output must remain in explicit owned Temp')
        output.mkdir(parents=True,exist_ok=True)
        (output/'TEMP_INVENTORY.json').write_text(json.dumps(TEMPS,indent=2)+'\n',encoding='utf-8')
