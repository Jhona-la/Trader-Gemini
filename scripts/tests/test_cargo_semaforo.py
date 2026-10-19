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
        # RB-H2: con el target compartido, frescura por contenido aunque la
        # rama aún no traiga la clave en .cargo/config.toml.
        self.assertIn("CARGO_UNSTABLE_CHECKSUM_FRESHNESS=true", r.stdout)
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

    def test_tras_la_paciencia_los_builds_ajenos_no_bloquean_si_sobra_ram(self):
        # LG-25b: con builds ajenos (fuera del semáforo) que no toman el
        # cerrojo, pasada la paciencia y con RAM ≥ mínimo + margen, hay turno.
        r = self.correr(
            "check", SEMAFORO_SOLO_DECIDIR=1, SEMAFORO_PROCESOS=2,
            SEMAFORO_PACIENCIA_MIN=0, SEMAFORO_ESPERA_MAX_MIN=0,
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        log = self.bitacora.read_text(encoding="utf-8")
        self.assertIn("turno por paciencia: 2 procesos", log)
        self.assertIn("ADQUIERE", log)

    def test_la_paciencia_no_salta_la_ram(self):
        # Con builds ajenos y sólo 5 GB libres (< 4 + 2), la paciencia no basta.
        r = self.correr(
            "check", SEMAFORO_SOLO_DECIDIR=1, SEMAFORO_PROCESOS=2, SEMAFORO_RAM_GB="5",
            SEMAFORO_PACIENCIA_MIN=0, SEMAFORO_ESPERA_MAX_MIN=0,
        )
        self.assertEqual(r.returncode, 75, r.stderr)
        self.assertIn("procesos cargo/rustc ajenos", r.stderr)

    def test_la_paciencia_no_salta_el_cerrojo_de_otro_usuario(self):
        # Otro usuario del semáforo tiene el turno (cerrojo fresco) y su
        # cargo corre: la paciencia sólo ignora a los que van por fuera.
        self.cerrojo.mkdir(parents=True)
        (self.cerrojo / "creado").write_text(str(int(time.time())))
        (self.cerrojo / "dueno").write_text("agente-en-turno pid=5 cargo test" + chr(10))
        r = self.correr(
            "check", SEMAFORO_SOLO_DECIDIR=1, SEMAFORO_PROCESOS=1,
            SEMAFORO_PACIENCIA_MIN=0, SEMAFORO_ESPERA_MAX_MIN=0,
        )
        self.assertEqual(r.returncode, 75, r.stderr)
        self.assertIn("cerrojo de agente-en-turno", r.stderr)

    def test_sin_poder_contar_procesos_no_se_roba_un_cerrojo(self):
        # LG-25c: si no se sabe cuántos cargo/rustc hay («?»), un cerrojo
        # pasado de la gracia NO es huérfano (antes contaba como 0 procesos).
        self.cerrojo.mkdir(parents=True)
        (self.cerrojo / "creado").write_text(str(int(time.time()) - 600))
        (self.cerrojo / "dueno").write_text("agente-compilando pid=3 cargo check" + chr(10))
        r = self.correr("check", SEMAFORO_SOLO_DECIDIR=1, SEMAFORO_PROCESOS="?", SEMAFORO_ESPERA_MAX_MIN=0)
        self.assertEqual(r.returncode, 75, r.stderr)
        self.assertTrue(self.cerrojo.exists(), "el cerrojo ajeno sigue en pie")
        self.assertNotIn("huérfano", self.bitacora.read_text(encoding="utf-8"))

    def test_sin_powershell_en_el_path_no_se_roba_un_cerrojo(self):
        # LG-25c, el caso real: un entorno sin powershell ni tasklist en el
        # PATH (sólo las utilidades de bash). Antes `pgrep || echo 0` decía
        # «0 procesos» y el cerrojo de un build en marcha se robaba como
        # huérfano pasada la gracia.
        self.cerrojo.mkdir(parents=True)
        (self.cerrojo / "creado").write_text(str(int(time.time()) - 600))
        (self.cerrojo / "dueno").write_text("agente-compilando pid=3 cargo check" + chr(10))
        r = self.correr(
            "check", SEMAFORO_SOLO_DECIDIR=1, SEMAFORO_PROCESOS="", SEMAFORO_ESPERA_MAX_MIN=0,
            PATH=os.path.dirname(BASH),
        )
        self.assertEqual(r.returncode, 75, r.stdout + r.stderr)
        self.assertTrue(self.cerrojo.exists(), "el cerrojo ajeno sigue en pie")
        self.assertNotIn("huérfano", self.bitacora.read_text(encoding="utf-8"))

    def test_al_salir_no_se_borra_un_cerrojo_que_ya_es_de_otro(self):
        # LG-25c: otro agente declaró huérfano nuestro cerrojo y lo tomó
        # mientras compilábamos; al salir no se le suelta el turno.
        r = self.correr("check", SEMAFORO_SOLO_DECIDIR=1, SEMAFORO_PRUEBA_ROBO="otro-agente pid=8 cargo test")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(self.cerrojo.exists(), "el cerrojo del otro agente sigue en pie")
        self.assertIn("otro-agente", (self.cerrojo / "dueno").read_text(encoding="utf-8"))

    def test_respeta_jobs_explicitos(self):
        r = self.correr("check", SEMAFORO_SOLO_DECIDIR=1, CARGO_BUILD_JOBS=1)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("CARGO_BUILD_JOBS=1", r.stdout)


if __name__ == "__main__":
    unittest.main()
