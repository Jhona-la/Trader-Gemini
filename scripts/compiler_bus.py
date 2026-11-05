"""Private compiler feedback bus. No launch on import and no resource admission."""
import ctypes
import argparse
from ctypes import wintypes as w
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import uuid
import types


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def relative(name):
    require(isinstance(name, str) and name and '\\' not in name and ':' not in name
            and all(ord(c) >= 32 and ord(c) != 127 for c in name)
            and all(p not in ('', '.', '..') for p in name.split('/')), 'Unsafe relative path')
    return name


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('xb') as stream:
        stream.write(canonical(value) + b'\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def enqueue(repo, agent, phase, files, *, kind='diagnostic', prior_generation=None):
    """Persist and return; never capture sources, wait for Cargo, or start a worker."""
    require(re.fullmatch('[A-Za-z0-9_-]+', agent or '') and re.fullmatch('[a-z0-9-]+', phase or ''), 'Invalid request identity')
    require(kind in ('diagnostic', 'publication-phase'), 'Invalid request kind')
    inputs_changed(repo)  # A new explicit request also wakes deferred revisions.
    ident = uuid.uuid4().hex
    request = {'schema': 'trader-gemini-compiler-request-v1', 'id': ident, 'agent': agent,
               'kind': kind, 'phase': phase, 'files': sorted({relative(n) for n in files}),
               'prior_generation': prior_generation, 'created_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
    path = Path(repo) / '.agent_tx/compiler-bus/pending' / (ident + '.json')
    require(not path.exists(), 'Request collision')
    atomic_json(path, request)
    return request


def inputs_changed(repo):
    """Publisher notification; deferred failures do not wake themselves in a loop."""
    directory = Path(repo)/'.agent_tx/compiler-bus'
    pending = directory/'pending'; pending.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in sorted((directory/'deferred').glob('*.json')):
        destination = pending/path.name
        require(not destination.exists(), 'Deferred request collision preserved')
        try:
            path.rename(destination); count += 1
        except FileNotFoundError:
            continue
    return count


class DirectoryWake:
    """Windows directory event, not a compiler lease or agent admission semaphore."""
    def __init__(self, directory):
        require(os.name == 'nt', 'Windows event service required; drain_once is portable')
        self.api = ctypes.WinDLL('kernel32', use_last_error=True)
        for name, args, result in (
            ('FindFirstChangeNotificationW', [w.LPCWSTR, w.BOOL, w.DWORD], w.HANDLE),
            ('FindNextChangeNotification', [w.HANDLE], w.BOOL),
            ('FindCloseChangeNotification', [w.HANDLE], w.BOOL),
            ('CreateEventW', [ctypes.c_void_p, w.BOOL, w.BOOL, w.LPCWSTR], w.HANDLE),
            ('SetEvent', [w.HANDLE], w.BOOL), ('CloseHandle', [w.HANDLE], w.BOOL),
            ('WaitForMultipleObjects', [w.DWORD, ctypes.POINTER(w.HANDLE), w.BOOL, w.DWORD], w.DWORD)):
            fn = getattr(self.api, name); fn.argtypes = args; fn.restype = result
        self.change = self.api.FindFirstChangeNotificationW(str(Path(directory).resolve()), False, 1)
        if self.change == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        self.stop_handle = self.api.CreateEventW(None, True, False, None)
        if not self.stop_handle:
            self.api.FindCloseChangeNotification(self.change)
            raise ctypes.WinError(ctypes.get_last_error())

    def wait(self, milliseconds=0xffffffff):
        handles = (w.HANDLE * 2)(self.change, self.stop_handle)
        code = self.api.WaitForMultipleObjects(2, handles, False, milliseconds)
        if code == 0:
            if not self.api.FindNextChangeNotification(self.change):
                raise ctypes.WinError(ctypes.get_last_error())
            return True
        if code == 1:
            return False
        require(code == 0x102, 'Directory notification observation failed')
        return None  # A timeout never rearms the notification.

    def stop(self):
        require(self.api.SetEvent(self.stop_handle), 'Own stop event failed')

    def close(self):
        first = self.api.FindCloseChangeNotification(self.change)
        second = self.api.CloseHandle(self.stop_handle)
        require(first and second, 'Own event handle close failed')


class CompilerBus:
    """One ROOT-owned worker; clients remain independent while a pass runs.

    Gate is the sole reviewed capture/profile/COMMANDS provider. This private
    class deliberately has no worker CLI or substitute production capturer.
    """
    def __init__(self, repo, private_index, producer_reference, tools, gate):
        self.repo = Path(repo).resolve(); self.index = Path(private_index)
        self.producer, self.tools, self.gate = producer_reference, tools, gate
        self.directory = self.repo / '.agent_tx/compiler-bus'
        self.pending = self.directory / 'pending'
        self.pending.mkdir(parents=True, exist_ok=True)
        self.watcher = None

    def capture(self, generation):
        value = self.gate.capture_subject(self.repo, self.index, self.producer,
                                          tools=self.tools, generation=generation)
        require(isinstance(value, dict) and re.fullmatch('[0-9a-f]{64}', value.get('inputs_digest', '')), 'Invalid shared capture')
        return value

    def claim(self):
        groups, rejected = {}, []
        for path in sorted(self.pending.glob('*.json')):
            claimed = False; destination = path; identity = None
            try:
                info = path.lstat()
                identity = {'device':info.st_dev,'file_id':info.st_ino,'size':info.st_size,
                            'mtime_ns':info.st_mtime_ns,'attributes':getattr(info,'st_file_attributes',0)}
                require(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode)
                        and not identity['attributes'] & 0x400, 'Nonregular request preserved in place')
                destination = self.directory / 'processing' / path.name
                destination.parent.mkdir(exist_ok=True)
                require(not os.path.lexists(destination), 'Foreign claim collision preserved in place')
                path.rename(destination)
                claimed = True
                request = json.loads(destination.read_bytes())
                require(isinstance(request,dict) and set(request) == {'schema','id','agent','kind','phase','files','prior_generation','created_utc'}
                        and request['schema'] == 'trader-gemini-compiler-request-v1'
                        and request['id'] + '.json' == destination.name
                        and request['kind'] in ('diagnostic','publication-phase')
                        and isinstance(request['phase'],str) and isinstance(request['files'],list)
                        and all(relative(n) for n in request['files']), 'Malformed claimed request preserved')
                groups.setdefault((request['kind'], request['phase']), []).append((request, destination))
            except Exception as exception:
                if isinstance(exception,FileNotFoundError) and not claimed:
                    continue  # Another explicit worker may have won the rename.
                rejected.append((destination if claimed else path,claimed,identity,
                                 type(exception).__name__+': '+str(exception)))
        return groups, rejected

    def drain_once(self):
        receipts = []
        groups, rejected = self.claim()
        for path,claimed,identity,error in rejected:
            run = self.directory/'runs'/uuid.uuid4().hex; run.mkdir(parents=True)
            original = None
            if claimed:
                destination = run/'request.invalid.json'; path.rename(destination)
                original = self.reference(destination)
            rejection = {'schema':'trader-gemini-compiler-bus-rejection-v1','state':'FAILED',
                         'feedback_only':True,'request_id':path.stem,'error':error,
                         'original_request':original,'request_path':path.relative_to(self.repo).as_posix(),
                         'owned_claim':claimed,'entry_identity':identity,
                         'preserved_in_place':not claimed,'compiler_started':False}
            atomic_json(run/'rejection.json',rejection)
            receipts.append(rejection)
        for (kind, phase), requests in groups.items():
            generation = uuid.uuid4().hex; run = self.directory / 'runs' / generation
            run.mkdir(parents=True); log = run / 'cargo.log'
            before = after = None; before_ref = after_ref = None
            exit_code = None; error = None; messages = {}; runner_refs = []; runner_uncertain = False
            test_counts = {}
            known_phase = True; profile = None
            cpus = os.process_cpu_count() or os.cpu_count()
            argv = []; cwd = self.repo; candidate = None
            candidate_released = kind == 'diagnostic'; candidate_cleanup_error = None
            capture_attempted = False
            def capture_after():
                nonlocal after, after_ref, error, capture_attempted
                capture_attempted = True
                try:
                    after = self.capture(generation)
                    atomic_json(run/'after.json', after); after_ref = self.reference(run/'after.json')
                except Exception as exception:
                    error = (error + '; ' if error else '') + 'after: ' + type(exception).__name__ + ': ' + str(exception)
            try:
                known_phase = phase in self.gate.COMMANDS
                require(known_phase, 'Unknown reviewed phase')
                profile = self.gate.execution_env(self.repo, self.tools)
                require(cpus and profile['CARGO_BUILD_JOBS'] == str(cpus)
                        and Path(profile['CARGO_TARGET_DIR']).resolve() == (self.repo/'target').resolve(), 'Profile must use measured CPUs and the one target')
                env = dict(os.environ)
                for key, value in profile.items():
                    require(isinstance(key, str) and (value is None or isinstance(value, str)), 'Invalid shared environment profile')
                    if value is None:
                        env.pop(key, None)
                    else:
                        env[key] = value
                before = self.capture(generation)
                atomic_json(run/'before.json', before)
                before_ref = self.reference(run/'before.json')
                manager = lease = None
                if kind == 'publication-phase':
                    manager = self.gate.prepare_candidate(self.repo,self.index,self.producer,
                        tools=self.tools,generation=generation,subject=before)
                    lease = manager.__enter__()
                try:
                    if lease is not None:
                        cwd = Path(lease.cwd)
                    if lease is not None:
                        argv = lease.argv(phase)
                    else:
                        require(phase != 'fmt', 'Formatting requires a protected publication candidate')
                        argv = [self.tools['cargo']['path'], *self.gate.COMMANDS[phase]]
                    try:
                        try:
                            prior_runners = self.gate.runner_receipts(self.repo, self.producer)
                        except Exception:
                            runner_uncertain = True
                            raise
                        with log.open('xb') as output:
                            child = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                            try:
                                for line in child.stdout:
                                    output.write(line)
                                    summary = re.match(rb'^test result: (?:ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored; (\d+) measured; (\d+) filtered out;',line)
                                    if summary:
                                        test_counts.setdefault('suites',0); test_counts['suites'] += 1
                                        for key,value in zip(('passed','failed','ignored','measured','filtered_out'),summary.groups()):
                                            test_counts[key] = test_counts.get(key,0)+int(value)
                                    try:
                                        item = json.loads(line)
                                    except (ValueError, UnicodeError):
                                        continue
                                    if isinstance(item, dict) and item.get('reason') == 'compiler-message':
                                        message = item.get('message'); names = set()
                                        if not isinstance(message, dict):
                                            continue
                                        for span in message.get('spans', []):
                                            try:
                                                name = span['file_name'].replace('\\', '/')
                                                if Path(name).is_absolute():
                                                    source = Path(name)
                                                    slot = self.repo / '.agent_tx/staging/compiler-candidate'
                                                    if lease is not None and source.is_relative_to(slot):
                                                        name = source.relative_to(slot).as_posix()
                                                    else:
                                                        name = source.relative_to(self.repo).as_posix()
                                                names.add(relative(name))
                                            except (KeyError, ValueError, TypeError, AttributeError):
                                                names.add('_unmapped')
                                        for name in names or {'_unmapped'}:
                                            messages.setdefault(name, []).append(message)
                            finally:
                                # Even a log/parser error must drain and collect OUR child.
                                for _ in child.stdout:
                                    pass
                                exit_code = child.wait(); child.stdout.close()
                            output.flush(); os.fsync(output.fileno())
                        try:
                            current_runners = self.gate.runner_receipts(self.repo, self.producer)
                        except Exception:
                            runner_uncertain = True
                            raise
                        runner_refs = [r for r in current_runners if r not in prior_runners]
                    finally:
                        try:
                            if lease is not None:
                                candidate = lease.proof()
                        finally:
                            capture_after()  # Publication copies remain pinned through this observation.
                finally:
                    if lease is not None:
                        try:
                            manager.__exit__(*sys.exc_info())
                            candidate_released = True
                        except Exception as exception:
                            candidate_cleanup_error = type(exception).__name__ + ': ' + str(exception)
                            raise
            except Exception as exception:
                error = (error + '; ' if error else '') + type(exception).__name__ + ': ' + str(exception)
            if not capture_attempted:
                capture_after()  # Pre-launch failure; never a claim of a pinned candidate.
            stale = known_phase and (runner_uncertain or before is None or after is None
                                     or before['inputs_digest'] != after['inputs_digest'])
            state = 'STALE' if stale else ('PASSED' if exit_code == 0 and error is None and candidate_released else 'FAILED')
            if not log.exists():
                log.write_bytes((error or 'No compiler launched').encode())
            files = sorted({n for request, _ in requests for n in request['files']} | set(messages))
            receipt = {'schema':'trader-gemini-compiler-bus-receipt-v1', 'generation':generation,
                       'request_ids':[r['id'] for r, _ in requests], 'kind':kind, 'phase':phase,
                       'feedback_only':kind == 'diagnostic', 'state':state, 'argv':argv,
                       'cwd':str(cwd), 'env_profile':profile, 'available_cpus':cpus,
                       'before':before_ref, 'after':after_ref,
                       'before_inputs_digest':before['inputs_digest'] if before else None,
                       'after_inputs_digest':after['inputs_digest'] if after else None,
                       'log_ref':self.reference(log), 'exit_code':exit_code, 'error':error,
                       'diagnostics_by_file':files, 'runner_receipts':runner_refs,
                       'test_counts':test_counts,
                       'observed_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
                       'producer_reference':self.producer, 'writer':self.writer_reference(),
                       'candidate':candidate, 'candidate_released':candidate_released,
                       'candidate_cleanup_error':candidate_cleanup_error}
            atomic_json(run/'receipt.json', receipt)
            for name in files:
                atomic_json(self.repo/'.agent_tx/diagnostics'/(name+'.json'),
                            {'generation':generation,'state':state,'receipt':self.reference(run/'receipt.json'),
                             'messages':messages.get(name, [])})
            if stale:
                retry = {**requests[0][0], 'id':uuid.uuid4().hex,'agent':'bus-requeue',
                         'files':files,'prior_generation':generation,
                         'created_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
                atomic_json(self.directory/'deferred'/(retry['id']+'.json'),retry)
            for _, path in requests:
                path.rename(run/path.name)
            receipts.append(receipt)
        return receipts

    def reference(self, path):
        return {'path':path.relative_to(self.repo).as_posix(), 'sha256':sha(path.read_bytes())}

    def writer_reference(self):
        path = Path(__file__).resolve()
        name = path.relative_to(self.repo).as_posix() if path.is_relative_to(self.repo) else str(path)
        return {'path':name,'sha256':sha(path.read_bytes())}

    def serve(self):
        """Explicit service only. No polling, leases, resource checks, or Job limits."""
        require(self.watcher is None, 'This worker is already serving')
        self.watcher = DirectoryWake(self.pending)
        try:
            self.drain_once()
            while self.watcher.wait():
                self.drain_once()
        finally:
            self.watcher.close(); self.watcher = None

    def stop(self):
        require(self.watcher is not None, 'No own service to stop')
        self.watcher.stop()  # An already running Cargo pass is allowed to finish.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='operation',required=True)
    for name in ('enqueue','inputs-changed','once','serve'):
        command = commands.add_parser(name); command.add_argument('--repo',type=Path,required=True)
        if name=='enqueue':
            command.add_argument('--agent',required=True); command.add_argument('--phase',required=True)
            command.add_argument('--kind',choices=('diagnostic','publication-phase'),default='diagnostic')
            command.add_argument('--file',action='append',default=[])
        if name in ('once','serve'):
            command.add_argument('--private-index',type=Path,required=True)
            command.add_argument('--producer-path',required=True); command.add_argument('--producer-sha256',required=True)
            command.add_argument('--tools-json',type=Path,required=True)
            command.add_argument('--gate',type=Path,required=True); command.add_argument('--gate-sha256',required=True)
    args = parser.parse_args(); root = args.repo.resolve()
    require(root.is_dir(),'Repository directory required')
    if args.operation=='enqueue':
        print(json.dumps(enqueue(root,args.agent,args.phase,args.file,kind=args.kind))); return
    if args.operation=='inputs-changed':
        print(json.dumps({'resumed':inputs_changed(root)})); return
    raw = args.gate.read_bytes(); require(sha(raw)==args.gate_sha256,'Explicit reviewed Gate SHA changed')
    gate = types.ModuleType('explicit_compiler_bus_gate'); gate.__file__=str(args.gate.resolve())
    exec(compile(raw,str(args.gate),'exec'),gate.__dict__)
    producer = {'path':args.producer_path,'sha256':args.producer_sha256}
    worker = CompilerBus(root,args.private_index,producer,json.loads(args.tools_json.read_bytes()),gate)
    if args.operation=='once':
        print(json.dumps(worker.drain_once()))
    else:
        worker.serve()


if __name__=='__main__':
    try:
        main()
    except (ValueError,OSError,KeyError,TypeError) as exception:
        print(json.dumps({'state':'REJECTED','error':type(exception).__name__+': '+str(exception)}),file=sys.stderr)
        sys.exit(75)
