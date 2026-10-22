"""QRA-02: scripts reales sobre repos temporales; jamás usan target/ compartido.

Los wrappers BASH_ENV inyectan errores en comandos reales después del preflight,
sin costuras nuevas en producción. Funciona con GNU bash en Linux y Git Bash.
"""

from pathlib import Path
import os
import shutil
import subprocess
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


if __name__ == "__main__":
    unittest.main()
