"""Reproduce a historical Depth parser defect using read-only cached Rust rlibs.

No Cargo, pipeline edits, network or trading. Requires matching compiled
dependencies; their exact hashes are recorded. Output must be new/empty.
"""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import time
import uuid

REF = 'e9c6a435eb1780da7f2c6e29103f3bd7d227d94c'
PARSER = 'crates/data-pipeline/src/parser.rs'
BLOB = '6c47744a94f5bea150e8a65c750b2a2f58c73cc4'
TOOLCHAIN = 'nightly-2026-06-30'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--repo', default='.')
    cli.add_argument('--memchr', required=True)
    cli.add_argument('--fast-float', required=True)
    cli.add_argument('--serde-json', required=True)
    cli.add_argument('--deps-dir', required=True)
    cli.add_argument('--output-dir')
    args = cli.parse_args()
    repo = Path(args.repo).resolve()
    output = Path(args.output_dir).resolve() if args.output_dir else repo / 'target/r4-depth-witnesses' / str(uuid.uuid4())
    if output.exists() and any(output.iterdir()):
        raise ValueError('Refusing to overwrite a nonempty output directory')
    output.mkdir(parents=True, exist_ok=True)
    receipt = {'source_commit': REF, 'source_path': PARSER,
               'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
               'runner_sha256': sha(Path(__file__).read_bytes()), 'status': 'IN_PROGRESS',
               'scope': 'Exact-source isolated parser characterization with synthetic JSON; not a repair or economic test',
               'commands': [], 'dependencies': {}}
    destination = output / 'receipt.json'

    def save():
        destination.write_bytes((json.dumps(receipt, indent=2, ensure_ascii=True) + '\n').encode('utf-8'))

    def run(label, argv):
        start = time.monotonic()
        proc = subprocess.run(argv, cwd=output, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
        raw = proc.stdout
        (output / f'{label}.log').write_bytes(raw)
        receipt['commands'].append({'label': label, 'argv': argv, 'exit_code': proc.returncode,
                                    'elapsed_seconds': round(time.monotonic() - start, 6), 'log_sha256': sha(raw)})
        save()
        if proc.returncode:
            raise RuntimeError(f'{label} exit {proc.returncode}')

    try:
        blob = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', f'{REF}:{PARSER}']).decode().strip()
        assert blob == BLOB
        raw = subprocess.check_output(['git', '-C', str(repo), 'show', f'{REF}:{PARSER}'])
        receipt.update(source_blob=blob, source_sha256=sha(raw), source_lines=len(raw.splitlines()))
        (output / 'parser.rs').write_bytes(raw)
        wrapper = Path(__file__).resolve().parent / 'fixtures/r4_depth_witness.rs'
        raw = wrapper.read_bytes()
        receipt['wrapper_sha256'] = sha(raw)
        (output / wrapper.name).write_bytes(raw)
        externs = []
        for name, argument in [('memchr', args.memchr), ('fast_float', args.fast_float), ('serde_json', args.serde_json)]:
            path = Path(argument).resolve()
            raw = path.read_bytes()
            receipt['dependencies'][name] = {'path': str(path), 'sha256': sha(raw), 'size_bytes': len(raw)}
            externs += ['--extern', f'{name}={path}']
        compiler = ['rustup', 'run', TOOLCHAIN, 'rustc']
        run('00-toolchain', compiler + ['--version', '--verbose'])
        executable = output / ('depth_witness.exe' if os.name == 'nt' else 'depth_witness')
        run('01-compile', compiler + ['--edition=2021', '-C', 'panic=abort', wrapper.name, *externs,
                                     '-L', f'dependency={Path(args.deps_dir).resolve()}', '-o', str(executable)])
        run('02-run', [str(executable)])
        receipt['status'] = 'EXISTING_DEPTH_DEFECT_REPRODUCED_NOT_REPAIRED'
        save()
        print(json.dumps({'receipt': str(destination), 'status': receipt['status']}))
        return 0
    except Exception as error:
        receipt.update(status='INCOMPLETE', error=str(error))
        save()
        raise


if __name__ == '__main__':
    raise SystemExit(main())
