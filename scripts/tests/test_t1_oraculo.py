"""Exercise real T-1 Bash in disposable repos with semaphore/Cargo stubs.

F7_TEST_SOURCE_ROOT selects only the T-1 source copied into each fixture, so
the identical contracts can compare an immutable baseline with the candidate.
No production semaphore or Cargo process is executed by this harness.
"""

import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest


SOURCE_ROOT = Path(os.environ.get("F7_TEST_SOURCE_ROOT",
                                 Path(__file__).resolve().parents[2]))
SCRIPT = SOURCE_ROOT / "scripts" / "t1_oraculo.sh"
BASH = (Path("C:/Program Files/Git/usr/bin/bash.exe") if sys.platform == "win32"
        else shutil.which("bash"))
RESOURCE_ENV = ("CARGO_TARGET_DIR", "CARGO_BUILD_JOBS", "T1_THREADS", "T1_LOG",
                "T1_SOLO_SELLO", "CARGO_INCREMENTAL", "CARGO_PROFILE_RELEASE_DEBUG",
                "T1_CARGO", "RUSTFLAGS", "T1_TEST_SEMAPHORE")


def canonical_path(value):
    """Compare Git Bash /c/... and native Windows paths without changing argv."""
    if sys.platform == "win32":
        value = re.sub(r"^/([A-Za-z])(?=/)", r"\1:", value)
    return os.path.normcase(os.path.abspath(value))


@unittest.skipUnless(BASH and Path(BASH).is_file(), "Bash is required")
class T1OraculoShellTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="t1 oraculo fixture ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.stub_bin = self.root / "stub bin"
        self.stub_bin.mkdir()
        recorder = self.root / "record_cargo.py"
        recorder.write_text(
            "import json, os, pathlib, sys\n"
            "keys = ('CARGO_TARGET_DIR', 'CARGO_BUILD_JOBS', 'T1_THREADS', "
            "'CARGO_INCREMENTAL', 'CARGO_PROFILE_RELEASE_DEBUG', 'SEMAFORO_RAIZ', "
            "'T1_TEST_SEMAPHORE', 'RUSTFLAGS')\n"
            "payload = {'argv': sys.argv[1:], 'cwd': os.getcwd(), "
            "'env': {k: os.environ.get(k) for k in keys}}\n"
            "record = pathlib.Path(os.environ['T1_STUB_RECORD'])\n"
            "previous = json.loads(record.read_text(encoding='utf-8')) if record.exists() else {}\n"
            "payload['call_count'] = previous.get('call_count', 0) + 1\n"
            "record.write_text(json.dumps(payload), encoding='utf-8')\n"
            "if os.environ['T1_STUB_RAN'] == '1':\n"
            "    print('Running tests/t1_cobertura_genetica (Cargo stub; no Rust)')\n"
            "print('test result: fixture cargo stub')\n"
            "sys.exit(int(os.environ['T1_STUB_RC']))\n",
            encoding="utf-8", newline="\n")
        stub = self.stub_bin / "cargo"
        stub.write_text(
            "#!/usr/bin/env bash\nexec " + shlex.quote(Path(sys.executable).as_posix())
            + " " + shlex.quote(recorder.as_posix()) + ' "$@"\n',
            encoding="utf-8", newline="\n")
        stub.chmod(0o755)
        self.fixture_count = 0
        self.prepare_repo()

    def prepare_repo(self):
        self.fixture_count += 1
        self.repo = self.root / f"repo {self.fixture_count}"
        self.primary_repo = self.repo
        (self.repo / "scripts").mkdir(parents=True)
        self.script = self.repo / "scripts" / "t1_oraculo.sh"
        self.script.write_text(SCRIPT.read_text(encoding="utf-8"),
                               encoding="utf-8", newline="\n")
        # This fixture proves routing only; it does not acquire a real build lease.
        (self.repo / "scripts" / "cargo_semaforo.sh").write_text(
            '#!/usr/bin/env bash\n'
            'sema_root=${SEMAFORO_RAIZ:-}\n'
            'if [ -z "$sema_root" ]; then\n'
            '  common=$(git rev-parse --path-format=absolute --git-common-dir) || exit 2\n'
            '  sema_root=$(dirname -- "$common")\n'
            'fi\n'
            'export SEMAFORO_RAIZ="$sema_root"\n'
            'export CARGO_TARGET_DIR="$sema_root/target"\n'
            'export T1_TEST_SEMAPHORE=$(( ${T1_TEST_SEMAPHORE:-0} + 1 ))\n'
            'exec cargo "$@"\n',
            encoding="utf-8", newline="\n")
        (self.repo / ".gitignore").write_text("target*/\nlogs/\n", encoding="utf-8")
        (self.repo / "fixture.rs").write_text("// fixture, never compiled\n", encoding="utf-8")
        self.record = self.root / f"cargo record {self.fixture_count}.json"
        self.git("init", "-q")
        self.git("config", "core.autocrlf", "false")
        self.git("config", "core.hooksPath", str(self.root / "no hooks"))
        self.git("add", "--", ".gitignore", "fixture.rs", "scripts/t1_oraculo.sh",
                 "scripts/cargo_semaforo.sh")
        # Commit only the disposable fixture: the real checkout is never staged.
        self.git("-c", "user.name=T1 Fixture", "-c", "user.email=t1@example.invalid",
                 "-c", "commit.gpgsign=false", "commit", "-qm", "fixture")

    def git(self, *args):
        result = subprocess.run(["git", "-C", str(self.repo), *args],
                                capture_output=True, text=True, check=True)
        return result.stdout

    def run_script(self, *args, overrides=None, rc=0, ran=True):
        env = os.environ.copy()
        for name in RESOURCE_ENV:
            env.pop(name, None)
        # A worktree's inherited Git index/directory must not escape the fixture.
        for name in tuple(env):
            if (name.startswith(("GIT_", "SEMAFORO_", "BASH_FUNC_"))
                    or name in ("BASH_ENV", "ENV")):
                env.pop(name)
        path_entries = [str(self.stub_bin)]
        if sys.platform == "win32":
            # Git/usr/bin/bash does not prepend its own coreutils directory.
            path_entries.append(str(Path(BASH).parent))
        path_entries.append(env.get("PATH", ""))
        env.update({"PATH": os.pathsep.join(path_entries),
                    "T1_STUB_RECORD": str(self.record), "T1_STUB_RC": str(rc),
                    "T1_STUB_RAN": "1" if ran else "0"})
        env.update(overrides or {})
        return subprocess.run([str(BASH), str(self.script), *args], cwd=self.repo,
                              env=env, capture_output=True, text=True, timeout=30)

    def assert_cargo_invocation(self, result):
        self.assertTrue(self.record.is_file(), result.stdout + result.stderr)
        payload = json.loads(self.record.read_text(encoding="utf-8"))
        self.assertEqual(1, payload["call_count"], "Cargo must be invoked once")
        extra = (["--target", "x86_64-pc-windows-gnu"]
                 if sys.platform.startswith("linux") else [])
        self.assertEqual(
            ["test", *extra, "--release", "-p", "backtest-engine", "--locked",
             "--test", "t1_cobertura_genetica", "--", "--exact",
             "t1_cobertura_genetica_del_oraculo_de_aptitud", "--nocapture",
             "--test-threads=1"], payload["argv"])
        self.assertEqual("0", payload["env"]["CARGO_INCREMENTAL"])
        self.assertEqual("0", payload["env"]["CARGO_PROFILE_RELEASE_DEBUG"])
        self.assertEqual("1", payload["env"]["T1_TEST_SEMAPHORE"],
                         "Cargo bypassed the fixture semaphore")
        self.assertEqual(canonical_path(str(self.repo)), canonical_path(payload["cwd"]))
        self.assertIsInstance(payload["env"]["SEMAFORO_RAIZ"], str,
                              "semaphore must publish its primary root")
        self.assertEqual(canonical_path(str(self.primary_repo)),
                         canonical_path(payload["env"]["SEMAFORO_RAIZ"]))
        self.assertIsNone(payload["env"]["RUSTFLAGS"])
        return payload["env"]

    def assert_no_outputs(self, result):
        self.assertNotEqual(0, result.returncode, result.stdout + result.stderr)
        self.assertFalse(self.record.exists(), "invalid resources reached Cargo")
        self.assertFalse((self.repo / "target").exists(), "invalid resources created target")
        self.assertFalse((self.repo / "target-portable").exists(),
                         "invalid resources created an alternate target")
        self.assertFalse((self.repo / "logs").exists(), "invalid resources created logs")
        self.assertEqual([], list(self.repo.rglob(".t1-sello")))

    def test_defaults_use_standard_target_and_serial_resources(self):
        result = self.run_script()
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        env = self.assert_cargo_invocation(result)
        self.assertEqual(canonical_path(str(self.repo / "target")),
                         canonical_path(env["CARGO_TARGET_DIR"]))
        self.assertEqual("1", env["CARGO_BUILD_JOBS"])
        self.assertEqual("1", env["T1_THREADS"])
        self.assertFalse((self.repo / "target-portable").exists())

    def test_thread_argument_overrides_env_without_increasing_build_jobs(self):
        result = self.run_script("3", overrides={"T1_THREADS": "2"})
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        env = self.assert_cargo_invocation(result)
        self.assertEqual("3", env["T1_THREADS"])
        self.assertEqual("1", env["CARGO_BUILD_JOBS"])

    def test_inherited_threads_and_build_jobs_are_preserved(self):
        result = self.run_script(overrides={"T1_THREADS": "2", "CARGO_BUILD_JOBS": "2"})
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        env = self.assert_cargo_invocation(result)
        self.assertEqual("2", env["T1_THREADS"])
        self.assertEqual("2", env["CARGO_BUILD_JOBS"])

    def test_cargo_build_jobs_uses_cargo_override_syntax(self):
        # Cargo accepts CPU offsets, "default", and signed integer strings.
        for value in ("-2", "default", "+02"):
            with self.subTest(value=value):
                self.prepare_repo()
                result = self.run_script("1", overrides={"CARGO_BUILD_JOBS": value})
                self.assertEqual(0, result.returncode, result.stdout + result.stderr)
                env = self.assert_cargo_invocation(result)
                self.assertEqual(value, env["CARGO_BUILD_JOBS"])
                self.assertEqual("1", env["T1_THREADS"])

    def test_explicit_canonical_target_is_preserved_and_log_stays_under_logs(self):
        target = self.primary_repo / "target"
        result = self.run_script(overrides={"CARGO_TARGET_DIR": str(target)})
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        env = self.assert_cargo_invocation(result)
        self.assertEqual(canonical_path(str(target)), canonical_path(env["CARGO_TARGET_DIR"]))
        self.assertTrue((target / ".t1-sello").is_file())
        self.assertTrue((self.repo / "logs" / "t1_oraculo.log").is_file())

    def test_worktree_uses_primary_target_without_creating_local_target(self):
        linked = self.root / "linked worktree with spaces"
        self.git("worktree", "add", "--detach", str(linked), "HEAD")
        self.repo = linked
        self.script = linked / "scripts" / "t1_oraculo.sh"
        result = self.run_script()
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        env = self.assert_cargo_invocation(result)
        self.assertEqual(canonical_path(str(self.primary_repo / "target")),
                         canonical_path(env["CARGO_TARGET_DIR"]))
        self.assertTrue((self.primary_repo / "target" / ".t1-sello").is_file())
        self.assertFalse((linked / "target").exists())
        self.assertFalse((linked / "target-portable").exists())
        self.assertTrue((linked / "logs" / "t1_oraculo.log").is_file())

    def test_alternate_target_is_rejected_before_side_effects(self):
        target = self.root / "alternate target"
        self.assert_no_outputs(self.run_script(overrides={"CARGO_TARGET_DIR": str(target)}))
        self.assertFalse(target.exists())

    def test_inherited_semaphore_root_cannot_redirect_shared_target(self):
        elsewhere = self.root / "alternate semaphore root"
        result = self.run_script(overrides={"SEMAFORO_RAIZ": str(elsewhere)})
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        env = self.assert_cargo_invocation(result)
        self.assertEqual(canonical_path(str(self.primary_repo / "target")),
                         canonical_path(env["CARGO_TARGET_DIR"]))
        self.assertFalse(elsewhere.exists())

    def test_fable_semaphore_hook_is_normalized_and_compatible(self):
        result = self.run_script(overrides={"T1_CARGO": "bash scripts/cargo_semaforo.sh"})
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        self.assert_cargo_invocation(result)

    def test_direct_or_ambiguous_cargo_hooks_are_rejected_before_side_effects(self):
        for value in ("cargo", "bash ./scripts/cargo_semaforo.sh",
                      "bash scripts/cargo_semaforo.sh; cargo"):
            with self.subTest(value=value):
                self.prepare_repo()
                self.assert_no_outputs(self.run_script(overrides={"T1_CARGO": value}))

    def test_rustflags_even_empty_are_rejected_before_side_effects(self):
        for value in ("", "-C debuginfo=2"):
            with self.subTest(value=value):
                self.prepare_repo()
                self.assert_no_outputs(self.run_script(overrides={"RUSTFLAGS": value}))

    def test_explicit_log_is_preserved(self):
        log = self.root / "custom oracle.log"
        result = self.run_script(overrides={"T1_LOG": str(log)})
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        self.assert_cargo_invocation(result)
        self.assertIn("Running tests/t1_cobertura_genetica", log.read_text(encoding="utf-8"))

    def test_relative_log_starting_with_dash_is_preserved(self):
        result = self.run_script(overrides={"T1_LOG": "-oracle.log"})
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        self.assert_cargo_invocation(result)
        log = self.repo / "-oracle.log"
        self.assertIn("Running tests/t1_cobertura_genetica", log.read_text(encoding="utf-8"))
        self.assertTrue((self.repo / "target" / ".t1-sello").is_file())
        self.assertEqual("", result.stderr)
        self.assertIn("test result: fixture cargo stub", result.stdout)

    def test_invalid_thread_arguments_are_rejected_before_side_effects(self):
        for value in ("0", "-2", "1.5", "not-an-integer"):
            with self.subTest(value=value):
                self.prepare_repo()
                self.assert_no_outputs(self.run_script(value))

    def test_invalid_thread_environment_is_rejected_before_side_effects(self):
        for value in ("0", "-2", "1.5", "not-an-integer"):
            with self.subTest(value=value):
                self.prepare_repo()
                self.assert_no_outputs(self.run_script(overrides={"T1_THREADS": value}))

    def test_invalid_build_jobs_are_rejected_before_side_effects(self):
        for value in ("0", "--2", "1.5", "not-an-integer"):
            with self.subTest(value=value):
                self.prepare_repo()
                self.assert_no_outputs(self.run_script("1", overrides={"CARGO_BUILD_JOBS": value}))

    def test_exit_code_is_preserved_and_seal_requires_execution_marker(self):
        for rc, ran in ((0, True), (23, True), (0, False), (23, False)):
            with self.subTest(rc=rc, ran=ran):
                self.prepare_repo()
                result = self.run_script(rc=rc, ran=ran)
                self.assertEqual(rc, result.returncode, result.stdout + result.stderr)
                self.assert_cargo_invocation(result)
                log = self.repo / "logs" / "t1_oraculo.log"
                self.assertTrue(log.is_file(), result.stdout + result.stderr)
                self.assertEqual(ran, "Running tests/t1_cobertura_genetica" in
                                 log.read_text(encoding="utf-8"))
                seal = self.repo / "target" / ".t1-sello"
                self.assertEqual(ran, seal.exists())
                if ran:
                    self.assertRegex(seal.read_text(encoding="utf-8").strip(),
                                     r"^v2 [0-9a-f]{40,64}$")


if __name__ == "__main__":
    unittest.main()
