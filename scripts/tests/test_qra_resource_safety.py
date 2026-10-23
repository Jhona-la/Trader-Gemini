"""QRA-02: scripts reales sobre repos temporales; jamás usan target/ compartido.

Los wrappers BASH_ENV inyectan errores en comandos reales después del preflight,
sin costuras nuevas en producción. Funciona con GNU bash en Linux y Git Bash.
"""

from pathlib import Path
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import unittest


SCRIPTS = Path(__file__).resolve().parents[1]


def bash_disponible():
    bash = shutil.which("bash")
    if bash and "windowsapps" not in bash.lower():
        return bash
    for path in (r"C:\Program Files\Git\usr\bin\bash.exe",
                 r"C:\Program Files\Git\bin\bash.exe"):
        if Path(path).is_file():
            return path
    return None


BASH = bash_disponible()
POWERSHELLS = [path for path in (shutil.which("pwsh"), shutil.which("powershell")) if path]


def native_path(value):
    if sys.platform == "win32":
        value = re.sub(r"^/([a-zA-Z])/", r"\1:/", str(value))
    return os.path.normcase(os.path.realpath(value))


@unittest.skipIf(BASH is None or shutil.which("git") is None, "requiere bash GNU y git")
class ResourceSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="qra-resource-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        subprocess.run(["git", "init", "-q", str(self.root)], check=True,
                       capture_output=True)
        self.deps = self.root / "target" / "debug" / "deps"
        self.old = self.archivo("target/debug/deps/libfoo-0123456789abcdef.rlib", 2048, 72)
        self.new = self.archivo("target/debug/deps/libfoo-fedcba9876543210.rlib", 2048, 1)
        self.inc = self.archivo("target/debug/incremental/crate-1111/s-1/dep.bin", 4096, 72)
        self.sentinel = self.archivo("data/intocable.bin", 32, 100)

    def archivo(self, relative, size=64, age_h=0):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * size)
        timestamp = time.time() - age_h * 3600
        os.utime(path, (timestamp, timestamp))
        return path

    def correr(self, script, wrappers="", **changes):
        env = {key: value for key, value in os.environ.items()
               if not key.upper().startswith(("BARRIDO_", "SEMAFORO_", "APLICAR",
                                              "CLEAN_INCREMENTAL", "BASH_ENV"))}
        env.update(PATH=str(Path(BASH).parent) + os.pathsep + env.get("PATH", ""),
                   BARRIDO_RAIZ=self.root.as_posix(), BARRIDO_PROCESOS="0",
                   BARRIDO_INCREMENTAL_MAX_MB="100", BARRIDO_DEPS_MAX_MB="0.001",
                   APLICAR="1")
        for key, value in changes.items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = str(value)
        if wrappers:
            source = self.root / "wrappers.bash"
            source.write_text(wrappers, encoding="utf-8")
            env["BASH_ENV"] = source.as_posix()
        return subprocess.run([BASH, str(SCRIPTS / script)], cwd=self.root, env=env,
                              capture_output=True, text=True, timeout=60)

    def intactos(self):
        for path in (self.old, self.new, self.inc, self.sentinel):
            self.assertTrue(path.exists(), f"se borró el testigo {path}")

    def test_barrido_enumerador_deps_parcial_falla_cerrado(self):
        result = self.correr("barrido_target.sh", wrappers="""
find() {
  command find "$@" || return
  case "${1-}" in */deps) return 23 ;; esac
}
""")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.intactos()
        self.assertIn("falló la enumeración", result.stderr)
        self.assertNotIn("BARRIDO APLICADO", (self.root / "logs/higiene.log").read_text())

    def test_barrido_enumerador_incremental_parcial_falla_cerrado(self):
        result = self.correr("barrido_target.sh", BARRIDO_DEPS_MAX_MB="100",
                             BARRIDO_INCREMENTAL_MAX_MB="0.001", wrappers="""
find() {
  command find "$@" || return
  case "${1-}" in */incremental) return 23 ;; esac
}
""")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.intactos()

    def test_barrido_rm_falla_no_contabiliza_ni_declara_aplicado(self):
        result = self.correr("barrido_target.sh", wrappers="""
rm() {
  local path
  for path in "$@"; do
    case "$path" in *libfoo-*) echo 'permiso denegado de prueba' >&2; return 23 ;; esac
  done
  command rm "$@"
}
""")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.intactos()
        self.assertIn("DETENIDO: liberado≈0.0 MB", result.stdout)
        self.assertIn("variantes 0 archivos 0.0 MB", result.stdout)
        self.assertFalse((self.root / "target/.semaforo.lock").exists())

    def test_barrido_enumerador_procesos_falla_conserva_target(self):
        result = self.correr("barrido_target.sh", BARRIDO_PROCESOS=None, wrappers="""
powershell() { return 23; }
powershell.exe() { return 23; }
pwsh() { return 23; }
pwsh.exe() { return 23; }
tasklist() { return 23; }
pgrep() { return 2; }
""")
        self.assertEqual(result.returncode, 5, result.stdout + result.stderr)
        self.intactos()
        self.assertIn("cuenta desconocida", result.stderr)

    def test_barrido_sort_falla_conserva_target(self):
        result = self.correr("barrido_target.sh", wrappers="""
sort() {
  if [ "${1-}" = --version ]; then command sort "$@"; return; fi
  return 23
}
""")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.intactos()
        self.assertIn("falló ordenar deps", result.stderr)

    def test_higiene_no_borra_exes_por_edad_con_cerrojo_ajeno(self):
        exe = self.archivo("target/debug/deps/prueba-0123456789abcdef.exe", age_h=200)
        pdb = self.archivo("target/debug/deps/prueba-0123456789abcdef.pdb", age_h=200)
        lock = self.root / "target/.semaforo.lock"
        lock.mkdir()
        (lock / "dueno").write_text("agente ajeno\n")
        result = self.correr("higiene_recursos.sh")
        self.assertEqual(result.returncode, 5, result.stdout + result.stderr)
        self.intactos()
        self.assertTrue(exe.exists() and pdb.exists())
        self.assertIn("borrado por edad retirado", result.stdout)
        self.assertEqual((lock / "dueno").read_text(), "agente ajeno\n")

    def test_higiene_propaga_error_barrido(self):
        result = self.correr("higiene_recursos.sh", BARRIDO_INCREMENTAL_MAX_MB="invalido")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.intactos()

    def test_higiene_enum_procesos_falla_no_habilita_borrado_legacy(self):
        exe = self.archivo("target/debug/deps/prueba-0123456789abcdef.exe", age_h=200)
        result = self.correr("higiene_recursos.sh", BARRIDO_PROCESOS=None, wrappers="""
powershell() { return 23; }
powershell.exe() { return 23; }
pwsh() { return 23; }
pwsh.exe() { return 23; }
tasklist() { return 23; }
pgrep() { return 2; }
""")
        self.assertEqual(result.returncode, 5, result.stdout + result.stderr)
        self.assertTrue(exe.exists())
        self.intactos()

    def test_higiene_git_falla_no_borra_graphify_versionado(self):
        snapshots = [self.archivo(f"graphify-out/2026-10-{day:02}/testigo.txt")
                     for day in range(1, 6)]
        subprocess.run(["git", "add", "--", "graphify-out"], cwd=self.root,
                       capture_output=True, check=True)
        result = self.correr("higiene_recursos.sh", wrappers="""
git() {
  if [ "${1-}" = ls-files ]; then return 128; fi
  command git "$@"
}
""")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        for path in snapshots:
            self.assertTrue(path.exists(), f"git falló y se borró {path}")
        self.intactos()

    def test_higiene_find_logs_falla_no_borra(self):
        log = self.archivo("logs/viejo.gz", 2048, 1000)
        result = self.correr("higiene_recursos.sh", wrappers="""
find() {
  command find "$@" || return
  if [ "${1-}" = logs ]; then return 23; fi
}
""")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertTrue(log.exists())
        self.intactos()

    def test_higiene_rm_falla_no_contabiliza_y_sale_con_error(self):
        log = self.archivo("logs/viejo.gz", 2048, 1000)
        result = self.correr("higiene_recursos.sh", BARRIDO_DEPS_MAX_MB="100", wrappers="""
rm() {
  local path
  for path in "$@"; do
    case "$path" in *viejo.gz) return 23 ;; esac
  done
  command rm "$@"
}
""")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertTrue(log.exists())
        self.assertIn("liberaría/liberó ≈ 0.00 MiB", result.stdout)
        self.intactos()

    def test_higiene_suma_bytes_sin_redondear_cada_archivo(self):
        for index in range(2):
            self.archivo(f"logs/viejo{index}.gz", 512 * 1024, 1000)
        result = self.correr("higiene_recursos.sh", APLICAR="0")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("ficheros .gz a borrar: 2 (1.00 MB)", result.stdout)
        self.assertIn("liberaría/liberó ≈ 1.00 MiB", result.stdout)
        self.intactos()

    def test_higiene_conserva_logs_versionados_y_borra_solo_elegible(self):
        tracked = [self.archivo("logs/versionado.gz", 2048, 1000),
                   self.archivo("logs/versionado.log", 2048, 1000),
                   self.archivo("logs_motor_versionado.txt", 2048, 1000)]
        eligible = self.archivo("logs/elegible.gz", 2048, 1000)
        subprocess.run(["git", "add", "--", *(str(path.relative_to(self.root)) for path in tracked)],
                       cwd=self.root, capture_output=True, check=True)
        result = self.correr("higiene_recursos.sh", BARRIDO_DEPS_MAX_MB="100")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for path in tracked:
            self.assertTrue(path.exists(), f"se tocó el log versionado {path}")
            self.assertEqual(path.read_bytes(), b"x" * 2048)
        self.assertFalse(eligible.exists())
        self.intactos()

    def test_disk_du_falla_no_inventa_limpieza(self):
        result = self.correr("disk_audit.sh", wrappers="""
du() {
  if [ "${1-}" = --version ]; then command du "$@"; return; fi
  return 23
}
""")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("desconocido", result.stderr)
        self.assertNotIn("0 (limpio)", result.stdout)
        self.intactos()

    def test_disk_clean_incremental_rechazado_sin_tocar_target(self):
        result = self.correr("disk_audit.sh", CLEAN_INCREMENTAL="1")
        self.assertEqual(result.returncode, 5, result.stdout + result.stderr)
        self.assertIn("barrido_target.sh", result.stderr)
        self.intactos()

    def test_disk_sort_falla_no_declara_exito(self):
        result = self.correr("disk_audit.sh", wrappers="""
sort() {
  if [ "${1-}" = --version ]; then command sort "$@"; return; fi
  return 23
}
""")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("falló ordenar tamaños", result.stderr)
        self.intactos()


@unittest.skipIf(BASH is None or shutil.which("git") is None, "requiere bash GNU y git")
class LauncherAdmissionTests(unittest.TestCase):
    """Launchers reales -> semáforo real -> recorder, nunca el Cargo instalado."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="qra launch fixture ")
        self.addCleanup(self.temp.cleanup)
        self.fixture = Path(self.temp.name)
        self.root = self.fixture / "primary repo"
        self.root.mkdir()
        self.caller = self.fixture / "different caller"
        self.caller.mkdir()
        self.bin = self.fixture / "stub bin"
        self.bin.mkdir()
        self.trace = self.fixture / "cargo.jsonl"
        self.state = self.fixture / "parent.json"
        self.scripts = self.root / "scripts"
        self.scripts.mkdir()
        for name in ("quality_gate.ps1", "medir_camino_caliente.sh", "oracle_witness.sh",
                     "cargo_semaforo.sh"):
            shutil.copyfile(SCRIPTS / name, self.scripts / name)
        for relative in ("crates/god-engine-core/tests/medidor_camino_caliente.rs",
                         "crates/backtest-engine/src/lib.rs",
                         "crates/backtest-engine/tests/t1_cobertura_genetica.rs"):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("// fixture; no Rust compiler may run\n", encoding="utf-8")
        self.git("init", "-q")
        self.git("add", "--", "scripts", "crates")
        self.git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                 "commit", "-qm", "fixture")
        self.base = self.git("rev-parse", "HEAD").strip()
        (self.root / "fixture.txt").write_text("second synthetic revision\n")
        self.git("add", "--", "fixture.txt")
        self.git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                 "commit", "-qm", "fixture two")
        self.branch = self.git("rev-parse", "HEAD").strip()
        self.recorder = self.fixture / "cargo recorder.py"
        self.recorder.write_text('''import json, os, pathlib, re, sys
def native(value):
    if sys.platform == "win32":
        value = re.sub(r"^/([a-zA-Z])/", r"\\1:/", value)
    return pathlib.Path(value)
args = sys.argv[1:]
shared = native(os.environ["SEMAFORO_RAIZ"])
lock = shared / "target" / ".semaforo.lock"
target = native(os.environ.get("CARGO_TARGET_DIR", "UNSET"))
trace = pathlib.Path(os.environ["F0_TRACE"])
with trace.open("a", encoding="utf-8") as stream:
    stream.write(json.dumps({"argv": args, "cwd": os.getcwd(),
        "target": str(target), "jobs": os.environ.get("CARGO_BUILD_JOBS"),
        "lock_held": lock.is_dir(),
        "owner": (lock / "dueno").read_text() if lock.is_dir() else ""}) + "\\n")
if args[0] == os.environ.get("F0_FAIL_VERB"):
    print("diagnostic without the display filter prefix", file=sys.stderr)
    sys.exit(int(os.environ.get("F0_FAIL_CODE", "101")))
if args[0] == "test" and os.environ.get("F0_VARY_RAM") == "1":
    (shared / "deny-next").write_text("fixture only")
if "--no-run" in args:
    name = args[args.index("--test") + 1]
    artifact = target / "release" / "deps" / (name + "-fixture.exe")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"fixture only; never executed")
    print(json.dumps({"reason": "compiler-artifact", "executable": artifact.as_posix(),
                      "target": {"name": name}}))
elif args[0] == "fmt":
    print("warning: synthetic fmt notice with successful exit", file=sys.stderr)
elif args[0] == "clippy":
    for i in range(100): print("warning: synthetic generated 1 warning (%s)" % i)
''', encoding="utf-8", newline="\n")
        invocation = (shlex.quote(Path(sys.executable).as_posix()) + " "
                      + shlex.quote(self.recorder.as_posix()) + ' "$@"')
        cargo = self.bin / "cargo"
        cargo.write_text("#!/usr/bin/env bash\nexec " + invocation + "\n",
                         encoding="utf-8", newline="\n")
        cargo.chmod(0o755)
        python3 = self.bin / "python3"
        python3.write_text("#!/usr/bin/env bash\nexec "
                           + shlex.quote(Path(sys.executable).as_posix()) + ' "$@"\n',
                           encoding="utf-8", newline="\n")
        python3.chmod(0o755)
        (self.bin / "cargo.cmd").write_text(
            '@echo off\n"' + sys.executable + '" "' + str(self.recorder)
            + '" %*\nexit /b %ERRORLEVEL%\n', encoding="utf-8", newline="\r\n")
        # Protect Bash even if its own utility directory contains a real Cargo.
        # This is a fixture-only BASH_ENV, not a production sensor/launcher hook.
        self.bash_env = self.fixture / "sentinel.bash"
        self.bash_env.write_text("cargo() { " + invocation + "; }\nexport -f cargo\n"
                                 + '''if [ "${F0_VARY_RAM:-0}" = 1 ]; then
  unset SEMAFORO_RAM_GB
  powershell() {
    if [ -f "$SEMAFORO_RAIZ/deny-next" ]; then printf '1\\n'; else printf '10\\n'; fi
  }
  export -f powershell
fi
''',
                                 encoding="utf-8", newline="\n")

    def git(self, *args):
        result = subprocess.run(["git", *args], cwd=self.root, capture_output=True,
                                text=True, timeout=15, check=True)
        return result.stdout

    def environment(self, **overrides):
        env = {key: value for key, value in os.environ.items()
               if not key.upper().startswith(("SEMAFORO_", "BASH_FUNC_", "F0_"))
               and key.upper() not in {"BASH_ENV", "ENV", "RUSTFLAGS", "CARGO_TARGET_DIR",
                                       "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"}}
        env.update(PATH=str(self.bin) + os.pathsep + str(Path(BASH).parent)
                   + os.pathsep + env.get("PATH", ""),
                   BASH_ENV=self.bash_env.as_posix(), F0_TRACE=str(self.trace),
                   SEMAFORO_RAIZ=self.root.as_posix(), SEMAFORO_AGENTE="fixture-f0",
                   SEMAFORO_PROCESOS="0", SEMAFORO_RAM_GB="10", SEMAFORO_BARRIDO="0",
                   SEMAFORO_ESPERA_MAX_MIN="0", SEMAFORO_PAUSA_S="0",
                   CARGO_BUILD_JOBS="7", MEDIDOR_SOLO_COMPILAR="1",
                   MEDIDOR_TRABAJO=(self.fixture / "measurer worktrees").as_posix())
        env.update({key: str(value) for key, value in overrides.items()})
        return env

    def calls(self):
        return ([json.loads(line) for line in self.trace.read_text().splitlines()]
                if self.trace.exists() else [])

    def admitted(self, count):
        calls = self.calls()
        self.assertEqual(len(calls), count, calls)
        for call in calls:
            self.assertTrue(call["lock_held"], call)
            self.assertIn("fixture-f0", call["owner"])
            self.assertEqual(native_path(call["target"]), native_path(self.root / "target"))
            self.assertEqual(call["jobs"], "2")
        self.assertFalse((self.root / "target/.semaforo.lock").exists())
        return calls

    def bash(self, script, *args, **overrides):
        return subprocess.run([BASH, str(self.scripts / script), *args], cwd=self.root,
                              env=self.environment(**overrides), capture_output=True,
                              text=True, timeout=45)

    def quality(self, engine, deny=False, **overrides):
        parent = self.fixture / "parent.ps1"
        parent.write_text('''$env:PATH = $env:F0_PATH
$global:LASTEXITCODE = 0
if ($env:F0_DENY -eq '1') {
    & $env:F0_SCRIPT -BashPath $env:F0_BASH -ClippyDeny
} else {
    & $env:F0_SCRIPT -BashPath $env:F0_BASH
}
$code = $LASTEXITCODE
@{ path=$env:PATH; jobs=$env:CARGO_BUILD_JOBS; cwd=(Get-Location).Path } |
    ConvertTo-Json | Set-Content -LiteralPath $env:F0_STATE -Encoding utf8
exit $code
''', encoding="utf-8", newline="\n")
        env = self.environment(**overrides)
        env.update(F0_PATH=env["PATH"], F0_DENY="1" if deny else "0",
                   F0_SCRIPT=str(self.scripts / "quality_gate.ps1"),
                   F0_BASH=BASH, F0_STATE=str(self.state))
        result = subprocess.run([engine, "-NoProfile", "-NonInteractive", "-File", str(parent)],
                                cwd=self.caller, env=env, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=45)
        self.assertTrue(self.state.exists(), result.stdout + result.stderr)
        state = json.loads(self.state.read_text(encoding="utf-8-sig"))
        self.assertEqual(state["path"], env["PATH"])
        self.assertEqual(state["jobs"], "7")
        self.assertEqual(native_path(state["cwd"]), native_path(self.caller))
        return result

    def reset_calls(self):
        if self.trace.exists():
            self.trace.unlink()

    @unittest.skipUnless(POWERSHELLS, "requiere PowerShell; no certifica compatibilidad si omitido")
    def test_quality_todo_cargo_con_turno_fmt_warning_y_clippy_una_vez(self):
        for engine in POWERSHELLS:
            with self.subTest(engine=engine):
                self.reset_calls()
                result = self.quality(engine)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                calls = self.admitted(4)
                self.assertEqual([c["argv"][0] for c in calls], ["fmt", "check", "test", "clippy"])
                self.assertIn("Crates con warnings: 100", result.stdout)
                self.assertEqual(result.stdout.count("synthetic generated"), 30)
                self.assertIn("QUALITY GATE: PASS", result.stdout)

    @unittest.skipUnless(POWERSHELLS, "requiere PowerShell")
    def test_quality_clippy_deny_con_paquetes_separados(self):
        for engine in POWERSHELLS:
            with self.subTest(engine=engine):
                self.reset_calls()
                result = self.quality(engine, deny=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                calls = self.admitted(5)
                self.assertEqual(calls[-1]["argv"], ["clippy", "-p", "dark-alpha-engine",
                    "-p", "quantum-arena", "--all-targets", "--", "-D", "warnings"])

    @unittest.skipUnless(POWERSHELLS, "requiere PowerShell")
    def test_quality_error_no_desaparece_por_filtro_incluso_clippy(self):
        for engine in POWERSHELLS:
            for verb, count in (("fmt", 1), ("check", 2), ("test", 3), ("clippy", 4)):
                with self.subTest(engine=engine, verb=verb):
                    self.reset_calls()
                    result = self.quality(engine, F0_FAIL_VERB=verb)
                    self.assertEqual(result.returncode, 101, result.stdout + result.stderr)
                    self.admitted(count)
                    self.assertNotIn("QUALITY GATE: PASS", result.stdout)

    @unittest.skipUnless(POWERSHELLS, "requiere PowerShell")
    def test_quality_sin_ram_sale_75_sin_cargo_y_restaura_entorno(self):
        for engine in POWERSHELLS:
            with self.subTest(engine=engine):
                result = self.quality(engine, SEMAFORO_RAM_GB="1")
                self.assertEqual(result.returncode, 75, result.stdout + result.stderr)
                self.assertEqual(self.calls(), [])
                self.assertNotIn("QUALITY GATE: PASS", result.stdout)

    @unittest.skipUnless(POWERSHELLS, "requiere PowerShell")
    def test_quality_clippy_denegado_por_semaforo_real_sale_75(self):
        for engine in POWERSHELLS:
            with self.subTest(engine=engine):
                self.reset_calls()
                marker = self.root / "deny-next"
                if marker.exists():
                    marker.unlink()
                result = self.quality(engine, F0_VARY_RAM="1")
                self.assertEqual(result.returncode, 75, result.stdout + result.stderr)
                self.admitted(3)
                self.assertNotIn("QUALITY GATE: PASS", result.stdout)

    def test_medidor_dos_worktrees_comparten_turno_target_y_jobs(self):
        result = self.bash("medir_camino_caliente.sh", self.base, self.branch)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.admitted(2)
        self.assertNotEqual(native_path(calls[0]["cwd"]), native_path(calls[1]["cwd"]))
        for call in calls:
            self.assertFalse((Path(call["cwd"]) / "target").exists())

    def test_medidor_sin_ram_75_retira_solo_worktrees_creados(self):
        preexisting = self.fixture / "measurer worktrees" / self.base
        preexisting.parent.mkdir()
        self.git("worktree", "add", "--detach", str(preexisting), self.base)
        result = self.bash("medir_camino_caliente.sh", self.base, self.branch, SEMAFORO_RAM_GB="1")
        self.assertEqual(result.returncode, 75, result.stdout + result.stderr)
        self.assertEqual(self.calls(), [])
        self.assertTrue(preexisting.is_dir())
        self.assertFalse((preexisting / "target").exists())
        self.assertFalse((preexisting.parent / self.branch).exists())
        self.assertEqual(self.git("worktree", "list", "--porcelain").count("worktree "), 2)

    def test_medidor_error_cargo_propaga_y_no_deja_altas(self):
        result = self.bash("medir_camino_caliente.sh", self.base, self.branch, F0_FAIL_VERB="test")
        self.assertEqual(result.returncode, 101, result.stdout + result.stderr)
        self.admitted(1)
        self.assertEqual(self.git("worktree", "list", "--porcelain").count("worktree "), 1)

    def test_oracle_dos_cargos_con_turno_y_limite_de_identidad_explicito(self):
        result = self.bash("oracle_witness.sh", "--", "--nocapture")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.admitted(2)
        self.assertIn("--no-run", calls[0]["argv"])
        self.assertEqual(calls[1]["argv"][-2:], ["--", "--nocapture"])
        log = next((self.root / "outputs").glob("oracle_witness_*.log")).read_text(encoding="utf-8")
        self.assertIn("no certifican el ejecutable", log)

    def test_oracle_sin_ram_75_no_invoca_cargo(self):
        result = self.bash("oracle_witness.sh", SEMAFORO_RAM_GB="1")
        self.assertEqual(result.returncode, 75, result.stdout + result.stderr)
        self.assertEqual(self.calls(), [])

    def test_oracle_error_compilacion_propaga_sin_segunda_invocacion(self):
        result = self.bash("oracle_witness.sh", F0_FAIL_VERB="test")
        self.assertEqual(result.returncode, 101, result.stdout + result.stderr)
        self.admitted(1)

    def test_bash_launchers_no_admiten_target_o_config_alternativo(self):
        for argument in ("--target-dir=private", "--config=build.target-dir='private'", "-j16", "--jobs=16"):
            for script, args, overrides in (
                    ("medir_camino_caliente.sh", (self.base, self.branch), {"MEDIDOR_CARGO_ARGS": argument}),
                    ("oracle_witness.sh", (argument,), {})):
                with self.subTest(script=script, argument=argument):
                    result = self.bash(script, *args, **overrides)
                    self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                    self.assertEqual(self.calls(), [])


if __name__ == "__main__":
    unittest.main()
