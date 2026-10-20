"""Windows wrappers -> real MSYS/semaphore -> isolated Cargo recorder, never Rust.

SEMAFORO_RAIZ selects a temporary shared primary outside the temporary worktree.
Process/RAM seams avoid probing the host; barrido is disabled. F7_TEST_SOURCE_ROOT
optionally selects only the two PowerShell source files for baseline RED. The
semaphore always comes from this candidate. Native cargo.cmd also intercepts
direct Cargo calls made by the old wrappers; neither entry can compile Rust.
"""

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[2]
WRAPPER_SOURCE = Path(os.environ.get("F7_TEST_SOURCE_ROOT", str(ROOT))).resolve()
PROGRAM_FILES = Path(os.environ.get("ProgramFiles", "C:/Program Files"))
PWSH = PROGRAM_FILES / "PowerShell/7/pwsh.exe"
MSYS_BASH = PROGRAM_FILES / "Git/usr/bin/bash.exe"
WRAPPERS = ("build_loop.ps1", "run_sim_5.ps1")
EXPECTED_ARGV = ["run", "--bin", "multi_coin_simulator", "-j", "1"]


@unittest.skipUnless(sys.platform == "win32" and PWSH.is_file() and MSYS_BASH.is_file(),
                     "These contracts require Windows PowerShell 7 and real Git MSYS Bash")
class SimLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="sim launcher fixture ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.stub_bin = self.root / "cargo stub bin"
        self.stub_bin.mkdir()
        self.empty_bin = self.root / "empty bin"
        self.empty_bin.mkdir()
        # The wrapper prepends this exact MSYS directory. Fail before launching
        # anything if it unexpectedly contains a real Cargo installation.
        for name in ("cargo", "cargo.exe", "cargo.cmd", "cargo.bat"):
            self.assertFalse((MSYS_BASH.parent / name).exists(), name)
        recorder = self.root / "record cargo.py"
        recorder.write_text(
            "import json, os, pathlib, sys\n"
            "payload = {'argv': sys.argv[1:], 'cwd': os.getcwd(), 'env': {\n"
            "    key: os.environ.get(key) for key in (\n"
            "        'CARGO_TARGET_DIR', 'CARGO_INCREMENTAL', 'CARGO_BUILD_JOBS', 'RUSTFLAGS')},\n"
            "    'lock_held': (pathlib.Path(os.environ['SEMAFORO_RAIZ'])\n"
            "                  / 'target' / '.semaforo.lock').is_dir()}\n"
            "with pathlib.Path(os.environ['SIM_CARGO_CALLS']).open('a', encoding='utf-8') as stream:\n"
            "    stream.write(json.dumps(payload) + '\\n')\n"
            "print('sim launcher Cargo recorder; no Rust')\n"
            "sys.exit(int(os.environ['SIM_CARGO_RC']))\n",
            encoding="utf-8", newline="\n")
        # POSIX entry for Bash, native entry for the original PowerShell wrappers.
        cargo = self.stub_bin / "cargo"
        cargo.write_text(
            "#!/usr/bin/env bash\nexec " + shlex.quote(Path(sys.executable).as_posix())
            + " " + shlex.quote(recorder.as_posix()) + ' "$@"\n',
            encoding="utf-8", newline="\n")
        cargo.chmod(0o755)
        (self.stub_bin / "cargo.cmd").write_text(
            '@echo off\n"' + str(Path(sys.executable)) + '" "' + str(recorder)
            + '" %*\nexit /b %ERRORLEVEL%\n', encoding="utf-8", newline="\r\n")
        # A bare bash lookup must not select this WSL-like PATH decoy.
        (self.stub_bin / "bash.cmd").write_text(
            '@echo off\necho wrong-bash>"%SIM_BAD_BASH_MARKER%"\nexit /b 91\n',
            encoding="utf-8", newline="\r\n")
        self.case_count = 0

    def invoke(self, wrapper, *, rc=0, target=None, cargo_present=True,
               bash_present=True, semaphore_present=True, use_default_bash=False,
               ram="100", rustflags=None):
        self.case_count += 1
        case = self.root / f"case {self.case_count}"
        wrapper_root = case / "separate worktree with spaces"
        wrapper_root.mkdir(parents=True)
        script = wrapper_root / wrapper
        script.write_text((WRAPPER_SOURCE / wrapper).read_text(encoding="utf-8"),
                          encoding="utf-8", newline="\n")
        shared_primary = case / "shared primary with spaces"
        shared_primary.mkdir()
        if semaphore_present:
            semaphore = wrapper_root / "scripts/cargo_semaforo.sh"
            semaphore.parent.mkdir()
            semaphore.write_bytes((ROOT / "scripts/cargo_semaforo.sh").read_bytes())
        caller = case / "different caller with spaces"
        caller.mkdir()
        calls_path = case / "cargo calls.jsonl"
        state_path = case / "parent state.json"
        bad_bash_marker = case / "wrong bash selected.txt"
        parent = case / "invoke parent.ps1"
        parent.write_text(
            # PowerShell prepends its install directory on startup. Establish the
            # isolated caller PATH before the wrapper so restoration is exact.
            "$env:PATH = $env:SIM_ORIGINAL_PATH\n"
            "$ErrorActionPreference = 'Continue'\n"
            "$PSNativeCommandUseErrorActionPreference = $true\n"
            "function global:Start-Sleep { param([int]$Seconds) }\n"
            "$global:LASTEXITCODE = 0\n"
            "$failure = $null\n"
            "$callExit = $null\n"
            "try {\n"
            "    if ($env:SIM_USE_DEFAULT_BASH -eq '1') {\n"
            "        & $env:SIM_WRAPPER_PATH\n"
            "    } else {\n"
            "        & $env:SIM_WRAPPER_PATH -BashPath $env:SIM_BASH_PATH\n"
            "    }\n"
            "    $callExit = $LASTEXITCODE\n"
            "} catch {\n"
            "    $failure = $_.Exception.Message\n"
            "    $callExit = if ($LASTEXITCODE -ne 0) { $LASTEXITCODE } else { 1 }\n"
            "} finally {\n"
            "    $state = [ordered]@{\n"
            "        returned = $true\n"
            "        exit_code = $callExit\n"
            "        caller_location = (Get-Location).Path\n"
            "        path = $env:PATH\n"
            "        cargo_target_exists = Test-Path Env:CARGO_TARGET_DIR\n"
            "        cargo_target = $env:CARGO_TARGET_DIR\n"
            "        build_jobs = $env:CARGO_BUILD_JOBS\n"
            "        incremental = $env:CARGO_INCREMENTAL\n"
            "        rustflags_exists = Test-Path Env:RUSTFLAGS\n"
            "        rustflags = $env:RUSTFLAGS\n"
            "        error_action_preference = [string]$ErrorActionPreference\n"
            "        native_error_preference = [bool]$PSNativeCommandUseErrorActionPreference\n"
            "        failure = $failure\n"
            "    }\n"
            "    $state | ConvertTo-Json -Compress | Set-Content -LiteralPath $env:SIM_PARENT_STATE -Encoding utf8\n"
            "}\n"
            "exit ([int]$callExit)\n",
            encoding="utf-8", newline="\n")
        env = {key: value for key, value in os.environ.items()
               if not key.upper().startswith(("SEMAFORO_", "BASH_FUNC_"))
               and key.upper() not in {"BASH_ENV", "ENV", "CARGO_TARGET_DIR", "RUSTFLAGS", "PATH"}}
        original_path = (str(self.stub_bin if cargo_present else self.empty_bin)
                         + os.pathsep + str(Path(os.environ["SystemRoot"]) / "System32"))
        env.update({
            "PATH": original_path, "SIM_ORIGINAL_PATH": original_path,
            "CARGO_BUILD_JOBS": "7", "CARGO_INCREMENTAL": "fixture-incremental",
            "SEMAFORO_RAIZ": shared_primary.as_posix(),
            "SEMAFORO_AGENTE": "fixture-f7", "SEMAFORO_PROCESOS": "0",
            "SEMAFORO_RAM_GB": ram, "SEMAFORO_BARRIDO": "0",
            "SEMAFORO_PAUSA_S": "0", "SEMAFORO_ESPERA_MAX_MIN": "0",
            "SIM_WRAPPER_PATH": str(script), "SIM_CARGO_CALLS": str(calls_path),
            "SIM_PARENT_STATE": str(state_path), "SIM_CARGO_RC": str(rc),
            "SIM_BAD_BASH_MARKER": str(bad_bash_marker),
            "SIM_USE_DEFAULT_BASH": "1" if use_default_bash else "0",
            "SIM_BASH_PATH": str(MSYS_BASH if bash_present else case / "missing bash.exe"),
        })
        if target is not None:
            env["CARGO_TARGET_DIR"] = target
        if rustflags is not None:
            env["RUSTFLAGS"] = rustflags
        process = subprocess.run(
            [str(PWSH), "-NoLogo", "-NoProfile", "-NonInteractive", "-File", str(parent)],
            cwd=caller, env=env, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=90)
        calls = ([json.loads(line) for line in calls_path.read_text(encoding="utf-8").splitlines()]
                 if calls_path.exists() else [])
        state = (json.loads(state_path.read_text(encoding="utf-8-sig"))
                 if state_path.exists() else None)
        semaphore_log = shared_primary / "logs/semaforo.log"
        return SimpleNamespace(
            process=process, calls=calls, state=state, wrapper_root=wrapper_root,
            shared_primary=shared_primary, caller=caller, target=target, rustflags=rustflags,
            original_path=original_path, bad_bash_marker=bad_bash_marker,
            semaphore_log=semaphore_log.read_text(encoding="utf-8")
            if semaphore_log.exists() else "")

    def assert_restored(self, result):
        diagnostics = result.process.stdout + result.process.stderr
        self.assertIsNotNone(result.state, diagnostics)
        self.assertTrue(result.state["returned"], diagnostics)
        self.assertEqual(os.path.normcase(str(result.caller)),
                         os.path.normcase(result.state["caller_location"]), diagnostics)
        self.assertEqual(result.original_path, result.state["path"], diagnostics)
        self.assertEqual(result.target is not None, result.state["cargo_target_exists"], diagnostics)
        self.assertEqual(result.target, result.state["cargo_target"], diagnostics)
        self.assertEqual("7", result.state["build_jobs"], diagnostics)
        self.assertEqual("fixture-incremental", result.state["incremental"], diagnostics)
        self.assertEqual(result.rustflags is not None, result.state["rustflags_exists"], diagnostics)
        self.assertEqual(result.rustflags, result.state["rustflags"], diagnostics)
        self.assertEqual("Continue", result.state["error_action_preference"], diagnostics)
        self.assertTrue(result.state["native_error_preference"], diagnostics)
        self.assertFalse(result.bad_bash_marker.exists(), diagnostics)
        for folder in (result.wrapper_root, result.caller):
            self.assertFalse((folder / "target").exists(), diagnostics)
            self.assertEqual([], list(folder.glob("target-sim*")), diagnostics)
        self.assertFalse((result.shared_primary / "target/.semaforo.lock").exists(), diagnostics)
        if result.target is not None:
            self.assertFalse(Path(result.target).exists(), diagnostics)

    def assert_one_cargo(self, result):
        diagnostics = result.process.stdout + result.process.stderr
        self.assertEqual(1, len(result.calls), diagnostics)
        call = result.calls[0]
        self.assertEqual(EXPECTED_ARGV, call["argv"], diagnostics)
        self.assertEqual(os.path.normcase(str(result.wrapper_root)),
                         os.path.normcase(call["cwd"]), diagnostics)
        self.assertEqual(os.path.normcase(str(result.shared_primary / "target")),
                         os.path.normcase(os.path.normpath(call["env"]["CARGO_TARGET_DIR"])),
                         diagnostics)
        self.assertEqual("1", call["env"]["CARGO_BUILD_JOBS"], diagnostics)
        self.assertIsNone(call["env"]["RUSTFLAGS"], diagnostics)
        self.assertTrue(call["lock_held"], diagnostics)
        self.assertEqual(1, result.semaphore_log.count("ADQUIERE"), diagnostics)
        self.assertEqual(1, result.semaphore_log.count("LIBERA"), diagnostics)
        return call

    def test_success_uses_default_msys_and_semaphore_once(self):
        for wrapper in WRAPPERS:
            with self.subTest(wrapper=wrapper):
                result = self.invoke(wrapper, use_default_bash=True)
                self.assertEqual(0, result.process.returncode,
                                 result.process.stdout + result.process.stderr)
                self.assert_one_cargo(result)
                self.assert_restored(result)

    def test_native_failure_seven_is_propagated_without_retry(self):
        for wrapper in WRAPPERS:
            with self.subTest(wrapper=wrapper):
                result = self.invoke(wrapper, rc=7)
                self.assertEqual(7, result.process.returncode,
                                 result.process.stdout + result.process.stderr)
                self.assert_one_cargo(result)
                self.assert_restored(result)

    def test_caller_target_is_preserved_but_semaphore_uses_shared_target(self):
        for wrapper in WRAPPERS:
            with self.subTest(wrapper=wrapper):
                target = str(self.root / "caller private target with spaces")
                result = self.invoke(wrapper, target=target)
                self.assertEqual(0, result.process.returncode,
                                 result.process.stdout + result.process.stderr)
                self.assert_one_cargo(result)
                self.assert_restored(result)

    def test_caller_environment_is_restored_after_semaphore_failure(self):
        for wrapper in WRAPPERS:
            with self.subTest(wrapper=wrapper):
                target = str(self.root / "failure target with spaces")
                result = self.invoke(wrapper, rc=7, target=target)
                self.assertEqual(7, result.process.returncode,
                                 result.process.stdout + result.process.stderr)
                self.assert_one_cargo(result)
                self.assert_restored(result)

    def test_incremental_policy_is_specific_to_each_wrapper(self):
        for wrapper in WRAPPERS:
            with self.subTest(wrapper=wrapper):
                result = self.invoke(wrapper)
                self.assertEqual(0, result.process.returncode,
                                 result.process.stdout + result.process.stderr)
                call = self.assert_one_cargo(result)
                self.assert_restored(result)
                expected = "0" if wrapper == "run_sim_5.ps1" else "fixture-incremental"
                self.assertEqual(expected, call["env"]["CARGO_INCREMENTAL"])

    def test_missing_cargo_returns_127_and_restores_caller(self):
        for wrapper in WRAPPERS:
            with self.subTest(wrapper=wrapper):
                result = self.invoke(wrapper, cargo_present=False)
                self.assertEqual(127, result.process.returncode,
                                 result.process.stdout + result.process.stderr)
                self.assertEqual([], result.calls)
                self.assertEqual(1, result.semaphore_log.count("ADQUIERE"))
                self.assertEqual(1, result.semaphore_log.count("LIBERA"))
                self.assert_restored(result)

    def test_missing_bash_fails_before_semaphore_or_cargo(self):
        for wrapper in WRAPPERS:
            with self.subTest(wrapper=wrapper):
                result = self.invoke(wrapper, bash_present=False)
                self.assertEqual(1, result.process.returncode,
                                 result.process.stdout + result.process.stderr)
                self.assertEqual([], result.calls)
                self.assertEqual("", result.semaphore_log)
                self.assertFalse((result.shared_primary / "target").exists())
                self.assert_restored(result)

    def test_missing_semaphore_fails_before_cargo(self):
        for wrapper in WRAPPERS:
            with self.subTest(wrapper=wrapper):
                result = self.invoke(wrapper, semaphore_present=False)
                self.assertEqual(1, result.process.returncode,
                                 result.process.stdout + result.process.stderr)
                self.assertEqual([], result.calls)
                self.assertEqual("", result.semaphore_log)
                self.assertFalse((result.shared_primary / "target").exists())
                self.assert_restored(result)

    def test_denied_memory_returns_75_without_cargo(self):
        for wrapper in WRAPPERS:
            with self.subTest(wrapper=wrapper):
                result = self.invoke(wrapper, ram="0")
                self.assertEqual(75, result.process.returncode,
                                 result.process.stdout + result.process.stderr)
                self.assertEqual([], result.calls)
                self.assertIn("SIN TURNO", result.semaphore_log)
                self.assertNotIn("ADQUIERE", result.semaphore_log)
                self.assert_restored(result)

    def test_inherited_rustflags_are_rejected_even_when_empty(self):
        for wrapper in WRAPPERS:
            for rustflags in ("", "-C opt-level=2"):
                with self.subTest(wrapper=wrapper, rustflags=rustflags):
                    result = self.invoke(wrapper, rustflags=rustflags)
                    self.assertEqual(1, result.process.returncode,
                                     result.process.stdout + result.process.stderr)
                    self.assertEqual([], result.calls)
                    self.assertEqual("", result.semaphore_log)
                    self.assertFalse((result.shared_primary / "target").exists())
                    self.assert_restored(result)


if __name__ == "__main__":
    unittest.main()
