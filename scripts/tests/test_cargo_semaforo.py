"""Contratos del semáforo de compilación (LG-25): turno, cerrojo huérfano y
entorno forzado. RÉGIMEN 2026-11-06 (mandato del operador, cero
restricciones de recursos): la RAM y los builds ajenos ya NO bloquean — el
tope SEMAFORO_RAM_MIN_GB es OPT-IN y los hilos van al 100 % de los núcleos.
Corre el script real con bash en una raíz temporal; las costuras
SEMAFORO_PROCESOS / SEMAFORO_RAM_GB fijan lo que en producción mide el
sistema operativo."""

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
            # FA-048: disco forzado holgado para que ninguna prueba dependa
            # del SSD real (la guarda DISCO-BAJO es OPT-OUT, default 40 GB).
            SEMAFORO_DISCO_GB="500",
        )
        # `None` BORRA la variable en vez de escribirla: las pruebas que
        # ejercitan la medición real no pueden heredar la costura que la salta.
        for clave, valor in env.items():
            if valor is None:
                entorno.pop(clave, None)
            else:
                entorno[clave] = str(valor)
        return subprocess.run(
            [BASH, str(SCRIPT), *args],
            env=entorno,
            capture_output=True,
            text=True,
            timeout=120,
        )

    def medidor_falso(self, respuesta):
        """Antepone al PATH un `powershell` de mentira que responde `respuesta`
        a la consulta de memoria y anota los argumentos que recibió."""
        bin_dir = self.raiz / "bin-medidor"
        bin_dir.mkdir(exist_ok=True)
        registro = self.raiz / "invocaciones.txt"
        shim = bin_dir / "powershell"
        shim.write_text(
            "#!/bin/bash\n"
            f'printf "%s\\n" "$*" >> "{registro.as_posix()}"\n'
            'case "$*" in\n'
            '  *Get-Process*) printf "0\\n" ;;\n'
            f"  *) printf '%s\\n' \"{respuesta}\" ;;\n"
            "esac\n",
            encoding="utf-8",
        )
        shim.chmod(0o700)
        return bin_dir, registro

    def test_turno_libre_fuerza_target_compartido_sin_tope_de_hilos(self):
        r = self.correr("check", SEMAFORO_SOLO_DECIDIR=1)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn(f"CARGO_TARGET_DIR={self.raiz.as_posix()}/target", r.stdout.replace("\\", "/"))
        # RÉGIMEN 2026-11-06: sin tope de hilos por defecto — el 100 % de los
        # núcleos salvo CARGO_BUILD_JOBS explícito (test_respeta_jobs_explicitos).
        self.assertIn("CARGO_BUILD_JOBS=todos", r.stdout)
        # RB-H2: con el target compartido, frescura por contenido aunque la
        # rama aún no traiga la clave en .cargo/config.toml.
        self.assertIn("CARGO_UNSTABLE_CHECKSUM_FRESHNESS=true", r.stdout)
        self.assertFalse(self.cerrojo.exists(), "el cerrojo se libera al salir")
        log = self.bitacora.read_text(encoding="utf-8")
        self.assertIn("ADQUIERE", log)
        self.assertIn("LIBERA", log)

    def test_builds_ajenos_no_bloquean(self):
        # RÉGIMEN 2026-11-06: los cargo/rustc ajenos (de quien compila por
        # fuera del semáforo) ya NO bloquean el turno — el file-lock interno
        # de cargo serializa el acceso real al target compartido.
        r = self.correr(
            "check", SEMAFORO_SOLO_DECIDIR=1, SEMAFORO_PROCESOS=2,
            SEMAFORO_ESPERA_MAX_MIN=0,
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("ADQUIERE", self.bitacora.read_text(encoding="utf-8"))

    def test_poca_ram_no_bloquea_por_defecto(self):
        # RÉGIMEN 2026-11-06: el tope de RAM es OPT-IN; con el default (0) la
        # RAM medida es TELEMETRÍA (se anota en la bitácora) y nunca bloquea.
        r = self.correr(
            "check", SEMAFORO_SOLO_DECIDIR=1, SEMAFORO_RAM_GB="1.5",
            SEMAFORO_ESPERA_MAX_MIN=0,
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("ram_libre=1.5GB", self.bitacora.read_text(encoding="utf-8"))

    def test_tope_ram_opt_in_bloquea(self):
        # El tope sigue existiendo para el agente que lo declare explícito.
        r = self.correr(
            "check", SEMAFORO_RAM_MIN_GB=4, SEMAFORO_RAM_GB="1.5",
            SEMAFORO_ESPERA_MAX_MIN=0,
        )
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

    def test_el_cerrojo_fresco_sigue_mandando_con_builds_vivos(self):
        # Otro usuario del semáforo tiene el turno (cerrojo fresco) y su cargo
        # corre: el cerrojo bloquea aunque los builds ajenos ya no bloqueen.
        self.cerrojo.mkdir(parents=True)
        (self.cerrojo / "creado").write_text(str(int(time.time())))
        (self.cerrojo / "dueno").write_text("agente-en-turno pid=5 cargo test" + chr(10))
        r = self.correr(
            "check", SEMAFORO_SOLO_DECIDIR=1, SEMAFORO_PROCESOS=1,
            SEMAFORO_ESPERA_MAX_MIN=0,
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
        # PATH, tampoco pgrep/pwsh aunque estén instalados en el host.
        # Antes `pgrep || echo 0` decía
        # «0 procesos» y el cerrojo de un build en marcha se robaba como
        # huérfano pasada la gracia.
        self.cerrojo.mkdir(parents=True)
        (self.cerrojo / "creado").write_text(str(int(time.time()) - 600))
        (self.cerrojo / "dueno").write_text("agente-compilando pid=3 cargo check" + chr(10))
        bin_prueba = self.raiz / "bin"
        bin_prueba.mkdir()
        for nombre in ("mkdir", "cat", "awk", "grep", "tr", "rm", "sleep", "date"):
            shim = bin_prueba / nombre
            shim.write_text(f'#!/bin/bash\nexec "/usr/bin/{nombre}" "$@"\n', encoding="utf-8")
            shim.chmod(0o700)
        entorno = {k: v for k, v in os.environ.items()
                   if not k.upper().startswith(("SEMAFORO_", "BASH_FUNC_"))
                   and k not in ("BASH_ENV", "ENV", "CARGO_BUILD_JOBS")}
        entorno.update(PATH=str(bin_prueba), SEMAFORO_RAIZ=str(self.raiz),
                       SEMAFORO_AGENTE="prueba", SEMAFORO_SOLO_DECIDIR="1",
                       SEMAFORO_PROCESOS="", SEMAFORO_RAM_GB="10",
                       SEMAFORO_ESPERA_MAX_MIN="0", SEMAFORO_PAUSA_S="0")
        probe = subprocess.run(
            [BASH, "--noprofile", "--norc", "-c",
             'for c in powershell powershell.exe pwsh pwsh.exe tasklist pgrep; do '
             'if command -v "$c" >/dev/null 2>&1; then exit 91; fi; done'],
            cwd=self.raiz, env=entorno, capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(probe.returncode, 0, probe.stdout + probe.stderr)
        r = subprocess.run(
            [BASH, "--noprofile", "--norc", str(SCRIPT), "check"],
            cwd=self.raiz, env=entorno, capture_output=True, text=True, timeout=20,
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

    def test_el_medidor_pide_memoria_disponible_y_no_solo_libre(self):
        # F1-097: la puerta que gobierna TODOS los builds medía en Windows
        # `FreePhysicalMemory` (sólo páginas libres) mientras en Linux medía
        # `MemAvailable` (libre + reclamable). No es la misma magnitud: con un
        # target/ de decenas de GB, el standby es la mayor parte de lo
        # reclamable y el semáforo anunciaba «2 GB» donde el sistema podía
        # entregar 9. Aquí se fija la magnitud, no el número.
        bin_dir, registro = self.medidor_falso("9.5")
        r = self.correr(
            "check", SEMAFORO_SOLO_DECIDIR=1,
            SEMAFORO_RAM_GB=None, SEMAFORO_PROCESOS=None,
            PATH=str(bin_dir) + os.pathsep + os.environ["PATH"],
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        invocaciones = registro.read_text(encoding="utf-8")
        self.assertIn("AvailableMBytes", invocaciones)
        self.assertNotIn("FreePhysicalMemory", invocaciones)
        # Y el valor medido es el que se anota como telemetría (RÉGIMEN
        # 2026-11-06: la RAM no gobierna el turno, se registra).
        self.assertIn("ram_libre=9.5GB", self.bitacora.read_text(encoding="utf-8"))

    def test_medicion_ilegible_es_telemetria_y_no_puerta(self):
        # F1-097 + RÉGIMEN 2026-11-06: una medición ilegible sigue siendo
        # DESCONOCIDA (nunca cero) en la bitácora, pero ya NO bloquea — no hay
        # puerta de RAM que gobernar por defecto.
        bin_dir, _ = self.medidor_falso("")
        r = self.correr(
            "check", SEMAFORO_SOLO_DECIDIR=1, SEMAFORO_ESPERA_MAX_MIN=0,
            SEMAFORO_RAM_GB=None, SEMAFORO_PROCESOS=None,
            PATH=str(bin_dir) + os.pathsep + os.environ["PATH"],
        )
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertNotIn("RAM libre  GB", r.stderr)
        self.assertIn("ram_libre=desconocida", self.bitacora.read_text(encoding="utf-8"))

    # ---- FA-048: guarda DISCO-BAJO (GB decimales, 1 GB = 1e9 B) ----------

    AVISO_DISCO = "[semaforo] disco libre desconocido; guarda DISCO-BAJO no aplicada"

    def bin_aislado(self, df=None, cargo=None):
        """PATH mínimo FUERA de la raíz: sólo las órdenes que usa el script,
        más `df` / `cargo` de mentira si se piden (sin df = df ausente)."""
        fuera = tempfile.TemporaryDirectory(prefix="semaforo-bin-")
        self.addCleanup(fuera.cleanup)
        bin_dir = Path(fuera.name)
        for nombre in ("mkdir", "cat", "awk", "grep", "tr", "rm", "sleep", "date"):
            shim = bin_dir / nombre
            shim.write_text(f'#!/bin/bash\nexec "/usr/bin/{nombre}" "$@"\n', encoding="utf-8")
            shim.chmod(0o700)
        for nombre, cuerpo in (("df", df), ("cargo", cargo)):
            if cuerpo is not None:
                shim = bin_dir / nombre
                shim.write_text("#!/bin/bash\n" + cuerpo, encoding="utf-8")
                shim.chmod(0o700)
        return bin_dir

    def correr_aislado(self, bin_dir, *args, **env):
        entorno = {k: v for k, v in os.environ.items()
                   if not k.upper().startswith(("SEMAFORO_", "BASH_FUNC_"))
                   and k not in ("BASH_ENV", "ENV", "CARGO_BUILD_JOBS")}
        entorno.update(PATH=str(bin_dir), SEMAFORO_RAIZ=str(self.raiz),
                       SEMAFORO_AGENTE="prueba", SEMAFORO_PROCESOS="0",
                       SEMAFORO_RAM_GB="10", SEMAFORO_ESPERA_MAX_MIN="0",
                       SEMAFORO_PAUSA_S="0")
        entorno.update({k: str(v) for k, v in env.items()})
        return subprocess.run(
            [BASH, "--noprofile", "--norc", str(SCRIPT), *args],
            cwd=self.raiz, env=entorno, capture_output=True, text=True, timeout=30,
        )

    @staticmethod
    def df_falso(kib):
        return ("printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\\n"
                f"C: 974851068 1 {kib} 93%% /c\\n'\n")

    def escrito_en_raiz(self):
        return sorted(p.relative_to(self.raiz).as_posix() for p in self.raiz.rglob("*"))

    def test_disco_bajo_bloquea_antes_de_compilar(self):
        traza = Path(tempfile.mkdtemp(prefix="semaforo-traza-")) / "cargo.trace"
        self.addCleanup(shutil.rmtree, traza.parent, True)
        bin_dir = self.bin_aislado(
            cargo=f'printf "CARGO %s\\n" "$*" >> "{traza.as_posix()}"; exit 0\n')
        r = self.correr_aislado(bin_dir, "check", "-p", "quantum-arena", SEMAFORO_DISCO_GB="10")
        salida = r.stdout + r.stderr
        self.assertEqual(r.returncode, 78, salida)
        self.assertIn("DISCO-BAJO", salida)
        self.assertIn("10 GB", salida)
        self.assertIn("40 GB", salida)
        self.assertFalse(traza.exists(), "con DISCO-BAJO no se invoca cargo")
        self.assertFalse((self.raiz / "target").exists(), "no se crea target/")
        self.assertFalse(self.cerrojo.exists(), "no se toma el cerrojo")
        # La línea DISCO-BAJO en logs/ es la ÚNICA escritura permitida.
        self.assertEqual(self.escrito_en_raiz(), ["logs", "logs/semaforo.log"])
        lineas = self.bitacora.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lineas), 1, lineas)
        self.assertIn("DISCO-BAJO", lineas[0])
        self.assertIn("10 GB < umbral 40 GB", lineas[0])

    def test_disco_suficiente_deja_pasarlo(self):
        # CONTROL (verde en base y en final): el flujo de decisión no se rompe.
        r = self.correr("check", SEMAFORO_SOLO_DECIDIR=1, SEMAFORO_DISCO_GB="500")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("turno concedido", r.stdout)

    def test_solo_decidir_informa_libre_y_umbral(self):
        r = self.correr("check", SEMAFORO_SOLO_DECIDIR=1, SEMAFORO_DISCO_GB="500")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("DISCO_LIBRE_GB=500", r.stdout)
        self.assertRegex(r.stdout, r"DISCO_MIN_GB=\d")
        self.assertIn("GB decimal, 1 GB = 1e9 B", r.stdout)
        # GD-04: la liberación anota también el libre de disco.
        self.assertIn("LIBERA (solo decidir) disco_libre=500GB",
                      self.bitacora.read_text(encoding="utf-8"))

    def test_disco_ilegible_no_bloquea_y_avisa(self):
        casos = {
            "costura ilegible": (self.bin_aislado(), {"SEMAFORO_DISCO_GB": "mucho"}),
            "df ausente": (self.bin_aislado(), {}),
            "df con exit != 0": (self.bin_aislado(df="echo roto >&2; exit 1\n"), {}),
            "df con salida ilegible": (self.bin_aislado(df="printf 'basura\\n'\n"), {}),
        }
        for nombre, (bin_dir, env) in casos.items():
            with self.subTest(nombre):
                if self.bitacora.exists():
                    self.bitacora.unlink()
                if nombre == "df ausente":
                    probe = subprocess.run(
                        [BASH, "--noprofile", "--norc", "-c", "command -v df && exit 91; exit 0"],
                        cwd=self.raiz, env={"PATH": str(bin_dir)},
                        capture_output=True, text=True, timeout=10)
                    self.assertEqual(probe.returncode, 0, probe.stdout)
                r = self.correr_aislado(bin_dir, "check", SEMAFORO_SOLO_DECIDIR=1, **env)
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                self.assertIn(self.AVISO_DISCO, r.stderr)
                log = self.bitacora.read_text(encoding="utf-8")
                self.assertIn("disco libre desconocido", log)
                self.assertIn("disco_libre=desconocida", log)
                self.assertIn("DISCO_LIBRE_GB=desconocida", r.stdout)

    def test_umbral_por_defecto_es_40(self):
        r = self.correr("check", SEMAFORO_SOLO_DECIDIR=1, SEMAFORO_DISCO_GB="500",
                        SEMAFORO_DISCO_MIN_GB=None)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertRegex(r.stdout, r"DISCO_MIN_GB=40(\s|$)")

    # Pruebas adicionales del integrador (fuera de la matriz, declaradas).

    def test_df_en_kib_se_convierte_a_gb_decimales(self):
        # GD-01/GD-02: libre = KiB x 1024 / 1e9. 39.062.500 KiB = 40,00 GB
        # exactos (37,25 GiB): pasa. Con lectura GiB del umbral bloquearía.
        # 39.000.000 KiB = 39,94 GB: bloquea.
        r = self.correr_aislado(self.bin_aislado(df=self.df_falso(39062500)),
                                "check", SEMAFORO_SOLO_DECIDIR=1)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("DISCO_LIBRE_GB=40.00", r.stdout)
        r = self.correr_aislado(self.bin_aislado(df=self.df_falso(39000000)),
                                "check", SEMAFORO_SOLO_DECIDIR=1)
        self.assertEqual(r.returncode, 78, r.stdout + r.stderr)
        self.assertIn("39.94 GB < umbral 40 GB", r.stderr)

    def test_borde_se_compara_sin_redondear(self):
        # NO APTO de COPILOT sobre b82d283: se comparaba el libre YA redondeado
        # a 2 decimales. 39.058.593 KiB = 39.995.999.232 B = 39,996 GB, que se
        # escribe «40.00» pero está bajo 40 GB: debe bloquear (78, DISCO-BAJO).
        r = self.correr_aislado(self.bin_aislado(df=self.df_falso(39058593)),
                                "check", SEMAFORO_SOLO_DECIDIR=1)
        self.assertEqual(r.returncode, 78, r.stdout + r.stderr)
        self.assertIn("DISCO-BAJO", r.stderr)
        self.assertIn("40.00 GB < umbral 40 GB", r.stderr)
        self.assertIn("DISCO-BAJO", self.bitacora.read_text(encoding="utf-8"))
        # Gemela justo por encima: 39.062.501 KiB = 40.000.001.024 B: pasa.
        self.bitacora.unlink()
        r = self.correr_aislado(self.bin_aislado(df=self.df_falso(39062501)),
                                "check", SEMAFORO_SOLO_DECIDIR=1)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("DISCO_LIBRE_GB=40.00", r.stdout)
        self.assertNotIn("DISCO-BAJO", r.stdout + r.stderr)
        # La costura con decimales se compara tal cual.
        r = self.correr_aislado(self.bin_aislado(), "check", SEMAFORO_SOLO_DECIDIR=1,
                                SEMAFORO_DISCO_GB="39.996")
        self.assertEqual(r.returncode, 78, r.stdout + r.stderr)
        self.assertIn("39.996 GB < umbral 40 GB", r.stderr)

    def test_liberacion_anota_disco_y_avisa_si_quedo_bajo(self):
        # GD-04: el build ya compilado no se bloquea; la línea LIBERA dice el
        # libre al liberar y que quedó bajo el umbral.
        contador = Path(tempfile.mkdtemp(prefix="semaforo-df-")) / "n"
        self.addCleanup(shutil.rmtree, contador.parent, True)
        df = (f'n=0; [ -f "{contador.as_posix()}" ] && n=$(cat "{contador.as_posix()}")\n'
              f'echo $((n+1)) > "{contador.as_posix()}"\n'
              'if [ "$n" = 0 ]; then kib=488281250; else kib=9765625; fi\n'
              "printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\\n"
              "C: 974851068 1 %s 93%% /c\\n' \"$kib\"\n")
        bin_dir = self.bin_aislado(df=df, cargo="exit 0\n")
        r = self.correr_aislado(bin_dir, "check")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        log = self.bitacora.read_text(encoding="utf-8")
        self.assertIn("disco_libre=500.00GB", log)
        self.assertIn("LIBERA codigo=0", log)
        self.assertIn("disco_libre=10.00GB DISCO-BAJO al liberar (< 40 GB)", log)


if __name__ == "__main__":
    unittest.main()
