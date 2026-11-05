"""Bounded Windows argv, cargo-fmt --all target closure, protected inputs only.

Metadata and rustfmt are real child processes. This helper never compiles and
never grants publication authority; zero_wait_gate independently checks its log.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

SCHEMA = 'trader-gemini-workspace-format-v1'
WINDOWS_COMMAND_LIMIT = 32767  # CreateProcessW, including terminating NUL.


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def regular(path):
    require(path.is_absolute() and path.resolve() == path, 'Noncanonical file')
    for item in (path, *path.parents):
        info = item.lstat()
        require(not item.is_symlink() and not getattr(info, 'st_file_attributes', 0) & 0x400, 'Reparse path')
    require(stat.S_ISREG(path.stat().st_mode), 'Nonregular file')
    return path.read_bytes()


def inside(slot, path):
    path = Path(path).resolve()
    require(path.is_relative_to(slot), 'Input escapes protected slot')
    regular(path)
    return path


def raw_map(slot):
    result = {}
    for directory, directories, files in os.walk(slot, followlinks=False):
        for name in directories:
            path = Path(directory) / name
            require(path.resolve() == path and not path.is_symlink() and
                    not getattr(path.lstat(), 'st_file_attributes', 0) & 0x400, 'Reparse directory')
        for name in files:
            path = Path(directory) / name
            result[path.relative_to(slot).as_posix()] = sha(regular(path))
    return dict(sorted(result.items()))


def units(argv):
    return len(subprocess.list2cmdline(argv).encode('utf-16-le')) // 2 + 1


def metadata_argv(cargo, manifest, configs):
    result = [cargo, 'metadata', '--locked', '--format-version', '1', '--no-deps', '--manifest-path', manifest]
    for config in configs:
        result += ['--config', config]
    return result


def planned_batches(rustfmt, config, inventory):
    result = []
    for edition in sorted({row['edition'] for row in inventory}):
        prefix = [rustfmt, '--edition', edition, '--config-path', config, '--check']
        batch = []
        for path in sorted(row['path'] for row in inventory if row['edition'] == edition):
            require(units([*prefix, path]) <= WINDOWS_COMMAND_LIMIT, 'Single target exceeds Windows argv limit')
            if batch and units([*prefix, *batch, path]) > WINDOWS_COMMAND_LIMIT:
                result.append([*prefix, *batch]); batch = []
            batch.append(path)
        if batch:
            result.append([*prefix, *batch])
    require(result, 'No Rust targets in local closure')
    return result


def child(argv, cwd):
    require(units(argv) <= WINDOWS_COMMAND_LIMIT, 'Child command exceeds Windows argv limit')
    result = {'argv': argv, 'cwd': str(cwd), 'exit_code': None, 'error': None,
              'stdout_b64': '', 'stderr_b64': '', 'stdout_sha256': sha(b''), 'stderr_sha256': sha(b'')}
    try:
        process = subprocess.run(argv, cwd=cwd, capture_output=True)
        result.update(exit_code=process.returncode,
                      stdout_b64=base64.b64encode(process.stdout).decode('ascii'),
                      stderr_b64=base64.b64encode(process.stderr).decode('ascii'),
                      stdout_sha256=sha(process.stdout), stderr_sha256=sha(process.stderr))
    except OSError as error:
        result['error'] = type(error).__name__ + ': ' + str(error)
    return result


def collect(slot, cargo, configs, cwd, records):
    pending = {str(inside(slot, slot / 'Cargo.toml'))}
    manifests = set(); targets = set()
    while pending:
        manifest = min(pending); pending.remove(manifest)
        if manifest in manifests:
            continue
        record = child(metadata_argv(cargo, manifest, configs), cwd); records.append(record)
        require(record['exit_code'] == 0 and record['error'] is None, 'Cargo metadata child failed')
        metadata = json.loads(base64.b64decode(record['stdout_b64']))
        require(metadata.get('version') == 1 and isinstance(metadata.get('packages'), list), 'Unknown Cargo metadata')
        packages = metadata['packages']
        require(packages, 'Empty Cargo metadata package set')
        returned = set()
        for package in packages:
            require(package['source'] is None, '--no-deps returned nonlocal package')
            package_manifest = str(inside(slot, package['manifest_path'])); returned.add(package_manifest)
            for target in package['targets']:
                path = str(inside(slot, target['src_path'])); edition = target['edition']
                require(edition in ('2015', '2018', '2021', '2024'), 'Unknown target edition')
                targets.add((path, edition))
            for dependency in package['dependencies']:
                if dependency.get('path') is not None:
                    pending.add(str(inside(slot, Path(dependency['path']) / 'Cargo.toml')))
        workspace_manifest = str(inside(slot, Path(metadata['workspace_root']) / 'Cargo.toml'))
        require(manifest in returned or manifest == workspace_manifest, 'Requested manifest missing from metadata')
        manifests.update(returned | {manifest}); pending.difference_update(manifests)
    return [{'path': path, 'edition': edition} for path, edition in sorted(targets)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--slot', required=True)
    parser.add_argument('--cargo', required=True); parser.add_argument('--cargo-sha256', required=True)
    parser.add_argument('--rustfmt', required=True); parser.add_argument('--rustfmt-sha256', required=True)
    parser.add_argument('--config-path', required=True)
    parser.add_argument('--cargo-config', action='append', default=[])
    args = parser.parse_args(argv)
    slot = Path(args.slot).resolve(); cwd = Path.cwd().resolve()
    report = {'schema': SCHEMA, 'slot': str(slot), 'cwd': str(cwd), 'config': args.config_path,
              'tools': {'cargo': {'path': args.cargo, 'sha256': args.cargo_sha256},
                        'rustfmt': {'path': args.rustfmt, 'sha256': args.rustfmt_sha256}},
              'cargo_configs': args.cargo_config, 'metadata': [], 'inventory': [], 'batches': [],
              'raw_before': {}, 'raw_after': {}, 'status': 'FAILED', 'exit_code': 1, 'error': None}
    try:
        require(str(slot) == args.slot and not cwd.is_relative_to(slot), 'Canonical slot/external cwd required')
        for item in report['tools'].values():
            require(sha(regular(Path(item['path']))) == item['sha256'], 'Native tool hash changed')
        config = inside(slot, args.config_path)
        require(str(config) == args.config_path and regular(config) == b'', 'Explicit empty formatter config required')
        for config_path in args.cargo_config:
            require(str(inside(slot, config_path)) == config_path, 'Noncanonical Cargo config')
        report['raw_before'] = raw_map(slot)
        report['inventory'] = collect(slot, args.cargo, args.cargo_config, cwd, report['metadata'])
        for command in planned_batches(args.rustfmt, args.config_path, report['inventory']):
            report['batches'].append(child(command, cwd))
        report['raw_after'] = raw_map(slot)
        require(report['raw_after'] == report['raw_before'], 'Protected formatter input changed')
        require(all(row['exit_code'] == 0 and row['error'] is None for row in report['batches']), 'Rustfmt child failed')
        report.update(status='PASSED', exit_code=0)
    except Exception as error:
        report['error'] = type(error).__name__ + ': ' + str(error)
    print(json.dumps(report, sort_keys=True, ensure_ascii=True, separators=(',', ':')), flush=True)
    return report['exit_code']


if __name__ == '__main__':
    sys.exit(main())
