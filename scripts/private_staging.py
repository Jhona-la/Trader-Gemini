"""Private Rust syntax buffer and Git index helpers; never publish a live source."""
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import uuid


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def safe(root, path):
    root, path = Path(root).resolve(), Path(path).absolute()
    require(path.is_relative_to(root) and path.resolve() == path, 'Noncanonical private path')
    for part in (path, *path.parents):
        if part.exists() or part.is_symlink():
            info = part.lstat()
            require(not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 0x400, 'Reparse private path')
        if part == root:
            break
    return path


def identity(agent):
    require(re.fullmatch('[A-Za-z0-9_-]+', agent or ''), 'Invalid agent identity')
    return agent


def relative(name):
    require(isinstance(name, str) and name and '\\' not in name and ':' not in name
            and all(ord(c) >= 32 and ord(c) != 127 for c in name)
            and all(p not in ('', '.', '..', '.git') for p in name.split('/')), 'Invalid staged path')
    return name


def store(path, data):
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('xb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def stage_rust(root, agent, name, data, formatter, *, edition='2021'):
    root = Path(root).resolve(); relative(name); identity(agent)
    require(name.endswith('.rs') and isinstance(data, bytes) and edition in ('2015','2018','2021','2024'), 'Rust staging arguments differ')
    binary = Path(formatter['path']); require(binary.is_absolute() and sha(binary.read_bytes()) == formatter['sha256'], 'Reviewed parser changed')
    directory = safe(root, root/'.agent_tx/staging'/agent)
    directory.mkdir(parents=True, exist_ok=True)
    target = safe(root, directory/name); target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_name(target.name+'.'+uuid.uuid4().hex+'.pending')
    store(pending, data)
    config = directory/('rustfmt-'+uuid.uuid4().hex+'.toml')
    store(config, b'# Explicit empty syntax-check configuration.\n')
    argv = [str(binary),'--edition',edition,'--config-path',str(config),'--config','skip_children=true','--emit','stdout']
    result = subprocess.run(argv, input=data, capture_output=True, cwd=directory)
    record = {'schema':'trader-gemini-private-syntax-buffer-v1','path':name,'candidate_sha256':sha(data),
              'parser':formatter,'argv':argv,'exit_code':result.returncode,
              'stderr':result.stderr.decode('utf-8','replace'),'semantic_or_typecheck':False,
              'candidate_path':target.relative_to(root).as_posix() if result.returncode == 0 else None,
              'pending_path':pending.relative_to(root).as_posix(), 'live_source_written':False}
    if result.returncode == 0:
        safe(root, target); os.replace(pending, target)
    store(pending.with_suffix('.json'), (json.dumps(record, indent=2)+'\n').encode())
    return record


def git_environment(root, index):
    root, index = Path(root).resolve(), Path(index).absolute()
    require(index.parent == root/'.git' and index.name.startswith('index_'), 'Explicit private .git/index_agent required')
    safe(root, index)
    env = {k:v for k,v in os.environ.items() if not k.startswith('GIT_')}
    env.update(GIT_INDEX_FILE=str(index), GIT_OPTIONAL_LOCKS='0', GIT_NO_LAZY_FETCH='1')
    return env


def git(root, executable, index, *args, data=None):
    if args[0] == 'hash-object':
        names = [a.split('=',1)[1] for a in args if a.startswith('--path=')]
        require(len(names)==1, 'Explicit canonical filter path required')
        driver_policy(root,executable,index,names)
    result = subprocess.run([str(executable),'--no-optional-locks','-C',str(root),'-c','core.fsmonitor=false',*args],
                            env=git_environment(root,index),input=data,capture_output=True)
    require(result.returncode == 0, 'Private Git failed: '+result.stderr.decode('utf-8','replace'))
    return result.stdout


def driver_policy(root, executable, index, names, *, revision=None, merging=False):
    """Admit built-ins for the paths actually consumed, never external drivers."""
    prefix = [str(executable),'--no-optional-locks','-C',str(root),'-c','core.fsmonitor=false']
    env = git_environment(root,index)
    args = ['check-attr', *(['--source='+revision] if revision else []), '-z','--stdin','filter','merge']
    result = subprocess.run(prefix+args,env=env,input=b'\0'.join(n.encode() for n in names)+b'\0',capture_output=True)
    require(result.returncode==0, 'Cannot determine effective Git attributes')
    values = result.stdout.split(b'\0'); require(values[-1]==b'' and (len(values)-1)%3==0, 'Unknown Git attributes result')
    def config(key):
        value = subprocess.run(prefix+['config','--get',key],env=env,capture_output=True)
        require(value.returncode in (0,1), 'Unknown Git driver configuration')
        return value.stdout.decode().strip() if value.returncode==0 else None
    for offset in range(0,len(values)-1,3):
        _,attribute,value = values[offset:offset+3]
        if attribute==b'filter':
            require(value in (b'unspecified',b'unset'), 'External Git filter attribute is not admitted')
        if merging and attribute==b'merge':
            driver = value.decode()
            if driver=='unspecified':driver=config('merge.default') or 'text'
            if driver=='set':driver='text'
            if driver=='unset':driver='binary'
            require(driver in ('text','binary','union') and config('merge.'+driver+'.driver') is None,
                    'External Git merge driver is not admitted')


def create_index(root, agent, head, executable):
    root = Path(root).resolve(); identity(agent)
    require(re.fullmatch('[0-9a-f]{40}', head or ''), 'Exact commit required')
    index = safe(root, root/'.git'/('index_'+agent+'_'+uuid.uuid4().hex))
    require(not index.exists(), 'Private index already exists')
    git(root, executable, index, 'read-tree', head)
    return index


def index_staged(root, index, record, executable, *, mode='100644'):
    require(record['exit_code'] == 0 and record['semantic_or_typecheck'] is False and mode in ('100644','100755'), 'Successful syntax record required')
    name = relative(record['path']); candidate = safe(root,Path(root)/record['candidate_path'])
    require(candidate.is_relative_to(Path(root).resolve()/'.agent_tx/staging'), 'Candidate outside private staging')
    data = candidate.read_bytes(); require(sha(data) == record['candidate_sha256'], 'Staged bytes changed')
    oid = git(root,executable,index,'hash-object','-w','--path='+name,'--stdin',data=data).decode().strip()
    require(re.fullmatch('[0-9a-f]{40}', oid), 'Invalid Git blob')
    blob = git(root,executable,index,'cat-file','blob',oid)
    require(hashlib.sha1(b'blob '+str(len(blob)).encode()+b'\0'+blob).hexdigest()==oid, 'Stored Git blob integrity failed')
    git(root,executable,index,'update-index','--add','--cacheinfo',mode+','+oid+','+name)
    return oid


def merge_preview(root, index, left, right, executable):
    require(all(re.fullmatch('[0-9a-f]{40}', x or '') for x in (left,right)), 'Exact merge commits required')
    changed = git(root,executable,index,'diff','--no-ext-diff','--no-textconv','--name-only','-z',left,right)
    names = [n.decode() for n in changed.split(b'\0') if n]
    for revision in (left,right):
        if names:driver_policy(root,executable,index,names,revision=revision,merging=True)
    result = subprocess.run([str(executable),'--no-optional-locks','-C',str(root),'-c','core.fsmonitor=false','merge-tree','--write-tree',left,right],
                            env=git_environment(root,index),capture_output=True)
    require(result.returncode in (0,1), 'Merge-tree observation failed')
    first = result.stdout.splitlines()[0].decode('ascii')
    require(re.fullmatch('[0-9a-f]{40}', first), 'Missing merge-tree object')
    return {'tree':first,'exit_code':result.returncode,'conflicted':result.returncode == 1,
            'stdout':result.stdout.decode('utf-8','replace'),'stderr':result.stderr.decode('utf-8','replace'),
            'scope':'TREE_OBJECT_PREVIEW_ONLY_NO_REF_CAS', 'external_drivers_admitted':False}
