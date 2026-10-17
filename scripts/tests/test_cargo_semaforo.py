"""Contratos del semáforo de compilación (LG-25): turno, espera, cerrojo
huérfano y entorno forzado. Corre el script real con bash en una raíz
temporal; las costuras SEMAFORO_PROCESOS / SEMAFORO_RAM_GB fijan lo que en
producción mide el sistema operativo."""

from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import time
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "cargo_semaforo.sh"
BASH = shutil.which("bash")


@unittest.skipIf(BASH is None, "bash no disponible")
class CargoSemaforoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="semaforo-test-")
        self.addCleanup(self.temp.cleanup)
        self.raiz = Path(self.temp.name)
        self.cerrojo = self.raiz / "target" / ".semaforo.lock"
        self.bitacora = self.raiz / "logs" / "semaforo.log"

    def correr(self, *args, **env):
        entorno = dict(os.environ)
        entorno.pop("CARGO_BUILD_JOBS", None)
        entorno.update(
            SEMAFORO_RAIZ=str(self.raiz),
            SEMAFORO_AGENTE="prueba",
            SEMAFORO_PAUSA_S="0",
            SEMAFORO_PROCESOS="0",
            SEMAFORO_RAM_GB="10",
        )
        entorno.update({k: str(v) for k, v in env.items()})
        return subprocess.run(
            [BASH, str(SCRIPT), *args],
            env=entorno,
            capture_output=True,
            text=True,
            timeout=120,
        )

    def test_turno_libre_fuerza_target_compartido_y_dos_hilos(self):
        r = self.correr("check", SEMAFORO_SOLO_DECIDIR=1)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn(f"CARGO_TARGET_DIR={self.raiz.as_posix()}/target", r.stdout.replace("\\", "/"))
        self.assertIn("CARGO_BUILD_JOBS=2", r.stdout)
        self.assertFalse(self.cerrojo.exists(), "el cerrojo se libera al salir")
        log = self.bitacora.read_text(encoding="utf-8")
        self.assertIn("ADQUIERE", log)
        self.assertIn("LIBERA", log)

    def test_con_otro_build_vivo_no_compila_y_sale_75(self):
        r = self.correr("check", SEMAFORO_PROCESOS=2, SEMAFORO_ESPERA_MAX_MIN=0)
        self.assertEqual(r.returncode, 75, r.stderr)
        self.assertIn("procesos cargo/rustc ajenos", r.stderr)
        self.assertIn("SIN TURNO", self.bitacora.read_text(encoding="utf-8"))

    def test_con_poca_ram_no_compila(self):
        r = self.correr("check", SEMAFORO_RAM_GB="1.5", SEMAFORO_ESPERA_MAX_MIN=0)
        self.assertEqual(r.returncode, 75, r.stderr)
        self.assertIn("RAM libre 1.5 GB < 4 GB", r.stderr)

    def test_cerrojo_de_un_agente_sin_cuota_se_libera(self):
        # Cerrojo de hace 10 min dejado por otro agente; ningún cargo/rustc vivo.
        self.cerrojo.mkdir(parents=True)
        (self.cerrojo / "creado").write_text(str(int(time.time()) - 600))
        (self.cerrojo / "dueno").write_text("agente-dormido pid=1 cargo test" + chr(10))
        r = self.correr("check", SEMAFORO_SOLO_DECIDIR=1, SEMAFORO_ESPERA_MAX_MIN=0)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("cerrojo huérfano liberado", self.bitacora.read_text(encoding="utf-8"))

    def test_cerrojo_recien_tomado_no_se_roba_aunque_aun_no_haya_rustc(self):
        # Carrera: otro agente acaba de tomar el turno y su cargo aún no
        # arrancó (0 procesos). Dentro de la gracia el cerrojo es suyo.
        self.cerrojo.mkdir(parents=True)
        (self.cerrojo / "creado").write_text(str(int(time.time())))
        (self.cerrojo / "dueno").write_text("agente-rapido pid=9 cargo check" + chr(10))
        r = self.correr("check", SEMAFORO_SOLO_DECIDIR=1, SEMAFORO_ESPERA_MAX_MIN=0)
        self.assertEqual(r.returncode, 75, r.stderr)
        self.assertIn("cerrojo de agente-rapido", r.stderr)
        self.assertTrue(self.cerrojo.exists(), "el cerrojo ajeno sigue en pie")

    def test_cerrojo_vencido_por_ttl_se_libera(self):
        self.cerrojo.mkdir(parents=True)
        (self.cerrojo / "creado").write_text(str(int(time.time()) - 10 * 3600))
        (self.cerrojo / "dueno").write_text("agente-viejo pid=7 cargo build" + chr(10))
        r = self.correr("check", SEMAFORO_SOLO_DECIDIR=1, SEMAFORO_TTL_MIN=60, SEMAFORO_ESPERA_MAX_MIN=0)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("agente-viejo", self.bitacora.read_text(encoding="utf-8"))

    def test_respeta_jobs_explicitos(self):
        r = self.correr("check", SEMAFORO_SOLO_DECIDIR=1, CARGO_BUILD_JOBS=1)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("CARGO_BUILD_JOBS=1", r.stdout)


if __name__ == "__main__":
    unittest.main()
