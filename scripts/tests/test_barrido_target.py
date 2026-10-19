"""Contratos del barrido del target/ compartido (RB-H1, Codex RB 2026-10-19).

Presupuesto LRU de la caché incremental, variantes superadas de deps/,
fail-closed ante builds vivos o cuenta desconocida, cerrojo propio, ajeno o
robado, y la integración con el semáforo: el barrido corre ANTES de cargo,
como mucho cada SEMAFORO_BARRIDO_H horas, nunca con SEMAFORO_SOLO_DECIDIR ni
con turno por paciencia. Corre los scripts reales con bash sobre raíces
temporales; jamás toca el target/ del repositorio."""

from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import time
import unittest

SCRIPTS = Path(__file__).resolve().parents[1]
BARRIDO = SCRIPTS / "barrido_target.sh"
SEMAFORO = SCRIPTS / "cargo_semaforo.sh"


def _bash():
    """bash de GNU. En Windows `bash` puede ser el lanzador de WSL
    (WindowsApps), que sin WSL instalado no arranca: usar Git Bash."""
    b = shutil.which("bash")
    if b and "windowsapps" in b.lower():
        git_bash = r"C:\Program Files\Git\bin\bash.exe"
        return git_bash if os.path.exists(git_bash) else None
    return b


BASH = _bash()
H1, H2, H3, H4 = "0123456789abcdef", "fedcba9876543210", "00112233445566ff", "aaaaaaaaaaaaaaa1"
KB = 1024


def archivo(path: Path, size: int, edad_h: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    t = time.time() - edad_h * 3600
    os.utime(path, (t, t))
    return path


@unittest.skipIf(BASH is None, "bash de GNU no disponible")
class BarridoTargetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="barrido-test-")
        self.addCleanup(self.temp.cleanup)
        self.raiz = Path(self.temp.name)
        self.target = self.raiz / "target"
        self.inc = self.target / "debug" / "incremental"
        self.deps = self.target / "debug" / "deps"
        self.cerrojo = self.target / ".semaforo.lock"
        self.higiene = self.raiz / "logs" / "higiene.log"

    # --- árbol sintético ---------------------------------------------------
    def incremental(self, nombre, edad_h, size=40 * KB):
        return archivo(self.inc / nombre / "s-1" / "dep-graph.bin", size, edad_h)

    def arbol_estandar(self):
        """3 carpetas incrementales de 40 KB (50 h, 10 h, 1 h) y tres
        variantes de libfoo (48 h, 30 h, 1 h)."""
        self.inc_a = self.incremental("crate_a-1111", 50)
        self.inc_b = self.incremental("crate_b-2222", 10)
        self.inc_c = self.incremental("crate_c-3333", 1)
        self.foo1 = archivo(self.deps / f"libfoo-{H1}.rlib", 10 * KB, 48)
        self.foo2 = archivo(self.deps / f"libfoo-{H2}.rlib", 10 * KB, 30)
        self.foo3 = archivo(self.deps / f"libfoo-{H3}.rlib", 10 * KB, 1)

    def correr(self, **env):
        entorno = {k: v for k, v in os.environ.items()
                   if not k.upper().startswith(("BARRIDO_", "SEMAFORO_", "APLICAR"))}
        entorno.update(BARRIDO_RAIZ=str(self.raiz), BARRIDO_AGENTE="prueba",
                       BARRIDO_INCREMENTAL_MAX_MB="0.1", BARRIDO_PROCESOS="0")
        entorno.update({k: str(v) for k, v in env.items()})
        return subprocess.run([BASH, str(BARRIDO)], env=entorno,
                              capture_output=True, text=True, timeout=120)

    # --- simulación ----------------------------------------------------------
    def test_simulacion_mide_y_no_borra_nada(self):
        self.arbol_estandar()
        r = self.correr()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        for p in (self.inc_a, self.inc_b, self.inc_c, self.foo1, self.foo2, self.foo3):
            self.assertTrue(p.exists(), f"la simulación borró {p}")
        self.assertIn("borraría: 2 carpetas", r.stdout)
        self.assertIn("borraría 2 variantes", r.stdout)
        self.assertFalse(self.cerrojo.exists(), "la simulación no toma el cerrojo")

    # --- 1. incremental --------------------------------------------------------
    def test_incremental_lru_borra_lo_viejo_hasta_la_histeresis_y_conserva_lo_reciente(self):
        # 120 KB > 0,1 MB; objetivo 0,75 × 0,1 MB ≈ 76,8 KB: se van A (50 h)
        # y B (10 h), en ese orden; C (1 h), el crate que se está iterando, queda.
        self.arbol_estandar()
        r = self.correr(APLICAR=1)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertFalse(self.inc_a.exists())
        self.assertFalse(self.inc_b.exists())
        self.assertTrue(self.inc_c.exists(), "la caché más reciente sobrevive")
        self.assertIn("borradas: 2 carpetas", r.stdout)

    def test_incremental_dentro_del_presupuesto_no_se_toca(self):
        self.arbol_estandar()
        r = self.correr(APLICAR=1, BARRIDO_INCREMENTAL_MAX_MB=1)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        for p in (self.inc_a, self.inc_b, self.inc_c):
            self.assertTrue(p.exists())
        self.assertIn("dentro del presupuesto", r.stdout)

    def test_incremental_de_release_tambien_cuenta(self):
        viejo = archivo(self.target / "release" / "incremental" / "rel-9999" / "s-1" / "q.bin", 80 * KB, 72)
        nuevo = self.incremental("crate_c-3333", 1, size=60 * KB)
        r = self.correr(APLICAR=1)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertFalse(viejo.exists(), "la variante vieja de release se barre primero")
        self.assertTrue(nuevo.exists())

    # --- 2. variantes de deps ------------------------------------------------
    def test_variantes_conserva_la_mas_reciente_las_jovenes_y_las_unicas(self):
        self.arbol_estandar()
        unica_vieja = archivo(self.deps / f"libbar-{H4}.rlib", 10 * KB, 100)
        baz_joven1 = archivo(self.deps / f"libbaz-{H1}.rlib", 10 * KB, 10)
        baz_joven2 = archivo(self.deps / f"libbaz-{H2}.rlib", 10 * KB, 2)
        dll_vieja = archivo(self.deps / f"windows_implement-{H1}.dll", 10 * KB, 50)
        lib_vieja = archivo(self.deps / f"windows_implement-{H1}.dll.lib", 1 * KB, 50)
        dll_nueva = archivo(self.deps / f"windows_implement-{H2}.dll", 10 * KB, 1)
        lib_nueva = archivo(self.deps / f"windows_implement-{H2}.dll.lib", 1 * KB, 1)
        sin_hash = archivo(self.deps / "README", 1 * KB, 500)
        rel_vieja = archivo(self.target / "release" / "deps" / f"libqux-{H1}.rlib", 10 * KB, 40)
        rel_nueva = archivo(self.target / "release" / "deps" / f"libqux-{H2}.rlib", 10 * KB, 1)
        r = self.correr(APLICAR=1, BARRIDO_INCREMENTAL_MAX_MB=100)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        for borrada in (self.foo1, self.foo2, dll_vieja, lib_vieja, rel_vieja):
            self.assertFalse(borrada.exists(), f"variante superada no barrida: {borrada.name}")
        for viva in (self.foo3, unica_vieja, baz_joven1, baz_joven2, dll_nueva, lib_nueva, sin_hash, rel_nueva):
            self.assertTrue(viva.exists(), f"se borró algo que debía quedar: {viva.name}")

    def test_empate_de_mtime_no_borra_ninguna(self):
        a = archivo(self.deps / f"libtie-{H1}.rlib", 10 * KB, 48)
        b = archivo(self.deps / f"libtie-{H2}.rlib", 10 * KB, 48)
        os.utime(b, (a.stat().st_atime, a.stat().st_mtime))
        r = self.correr(APLICAR=1, BARRIDO_INCREMENTAL_MAX_MB=100)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertTrue(a.exists() and b.exists(), "sin una variante estrictamente más nueva no hay superada")

    # --- 3. fail-closed y cerrojo -----------------------------------------------
    def assert_intacto(self):
        for p in (self.inc_a, self.inc_b, self.inc_c, self.foo1, self.foo2, self.foo3):
            self.assertTrue(p.exists(), f"se borró {p} sin permiso")

    def test_con_un_build_vivo_no_borra_nada_y_sale_5(self):
        self.arbol_estandar()
        r = self.correr(APLICAR=1, BARRIDO_PROCESOS=1)
        self.assertEqual(r.returncode, 5, r.stdout + r.stderr)
        self.assert_intacto()
        self.assertIn("BARRIDO OMITIDO procesos=1", self.higiene.read_text(encoding="utf-8"))

    def test_con_la_cuenta_desconocida_no_borra_nada(self):
        self.arbol_estandar()
        r = self.correr(APLICAR=1, BARRIDO_PROCESOS="?")
        self.assertEqual(r.returncode, 5, r.stdout + r.stderr)
        self.assert_intacto()

    def test_un_cerrojo_ajeno_no_se_toca_ni_se_barre(self):
        self.arbol_estandar()
        self.cerrojo.mkdir(parents=True)
        (self.cerrojo / "creado").write_text(str(int(time.time())))
        (self.cerrojo / "dueno").write_text("agente-compilando pid=3 cargo test\n")
        r = self.correr(APLICAR=1)
        self.assertEqual(r.returncode, 5, r.stdout + r.stderr)
        self.assert_intacto()
        self.assertIn("agente-compilando", (self.cerrojo / "dueno").read_text())

    def test_toma_su_cerrojo_lo_libera_y_anota_en_la_bitacora(self):
        self.arbol_estandar()
        r = self.correr(APLICAR=1)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertFalse(self.cerrojo.exists(), "el cerrojo propio se libera al salir")
        self.assertIn("BARRIDO APLICADO", self.higiene.read_text(encoding="utf-8"))
        self.assertIn("liberado", r.stdout.strip().splitlines()[-1], "la última línea es el resumen")

    def test_si_le_roban_el_cerrojo_para_antes_de_borrar(self):
        self.arbol_estandar()
        r = self.correr(APLICAR=1, BARRIDO_PRUEBA_ROBO="otro-agente pid=8 cargo test")
        self.assertEqual(r.returncode, 5, r.stdout + r.stderr)
        self.assert_intacto()
        self.assertIn("otro-agente", (self.cerrojo / "dueno").read_text(), "no suelta el cerrojo ajeno")
        self.assertIn("DETENIDO", self.higiene.read_text(encoding="utf-8"))

    def test_con_cerrojo_propio_exige_la_ficha_del_turno(self):
        self.arbol_estandar()
        self.cerrojo.mkdir(parents=True)
        (self.cerrojo / "dueno").write_text("turno-real pid=1 cargo check\n")
        r = self.correr(APLICAR=1, BARRIDO_CERROJO_PROPIO=1, BARRIDO_FICHA="impostor")
        self.assertEqual(r.returncode, 5, r.stdout + r.stderr)
        self.assert_intacto()
        r = self.correr(APLICAR=1, BARRIDO_CERROJO_PROPIO=1, BARRIDO_FICHA="turno-real pid=1 cargo check")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertFalse(self.foo1.exists())
        self.assertTrue(self.cerrojo.exists(), "el cerrojo del turno es del semáforo: no lo suelta")

    def test_parametros_invalidos_no_borran_nada(self):
        self.arbol_estandar()
        for malo in ({"BARRIDO_INCREMENTAL_MAX_MB": "abc"}, {"BARRIDO_HISTERESIS": "1.5"},
                     {"BARRIDO_LOTE": "0"}, {"BARRIDO_VARIANTES_H": "-3"}):
            r = self.correr(APLICAR=1, **malo)
            self.assertEqual(r.returncode, 2, f"{malo}: {r.stdout}{r.stderr}")
            self.assert_intacto()

    def test_nunca_toca_nada_fuera_de_target(self):
        self.arbol_estandar()
        fuera_rlib = archivo(self.raiz / "data" / f"libfoo-{H1}.rlib", 10 * KB, 900)
        fuera_rlib2 = archivo(self.raiz / "data" / f"libfoo-{H2}.rlib", 10 * KB, 1)
        fuera_inc = archivo(self.raiz / "models" / "incremental" / "x-1" / "s" / "f", 900 * KB, 900)
        r = self.correr(APLICAR=1)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        for p in (fuera_rlib, fuera_rlib2, fuera_inc):
            self.assertTrue(p.exists(), f"tocó algo fuera de target/: {p}")

    def test_lotes_pequenos_borran_lo_mismo(self):
        self.arbol_estandar()
        for i in range(7):
            archivo(self.deps / f"libmany-{i:016x}.rlib", 1 * KB, 100 - i)
        r = self.correr(APLICAR=1, BARRIDO_LOTE=2, BARRIDO_CHEQUEO_S=0)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        quedan = sorted(p.name for p in self.deps.glob("libmany-*"))
        self.assertEqual(quedan, [f"libmany-{6:016x}.rlib"], "sólo la variante más nueva de libmany")


@unittest.skipIf(BASH is None, "bash de GNU no disponible")
class SemaforoBarridoTests(unittest.TestCase):
    """Integración: el semáforo barre con el turno tomado y antes de cargo."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="semaforo-barrido-")
        self.addCleanup(self.temp.cleanup)
        self.raiz = Path(self.temp.name)
        self.deps = self.raiz / "target" / "debug" / "deps"
        self.vieja = archivo(self.deps / f"libfoo-{H1}.rlib", 10 * KB, 48)
        self.nueva = archivo(self.deps / f"libfoo-{H2}.rlib", 10 * KB, 1)
        self.sello = self.raiz / "target" / ".barrido_ultimo"
        self.bin = self.raiz / "bin"
        self.bin.mkdir()
        self.traza = self.raiz / "cargo.traza"
        cargo = self.bin / "cargo"
        cargo.write_text(
            "#!/bin/bash\n"
            'if [ -e "$PRUEBA_TESTIGO" ]; then estado=presente; else estado=ausente; fi\n'
            "printf 'cargo %s testigo=%s\\n' \"$*\" \"$estado\" >> \"$PRUEBA_TRAZA\"\n"
            "exit 0\n",
            encoding="utf-8",
        )
        cargo.chmod(0o700)

    def correr(self, **env):
        entorno = {k: v for k, v in os.environ.items()
                   if not k.upper().startswith(("BARRIDO_", "SEMAFORO_", "APLICAR", "CARGO_BUILD_JOBS"))}
        entorno.update(
            PATH=str(self.bin) + os.pathsep + os.environ.get("PATH", ""),
            SEMAFORO_RAIZ=str(self.raiz), SEMAFORO_AGENTE="prueba", SEMAFORO_PAUSA_S="0",
            SEMAFORO_PROCESOS="0", SEMAFORO_RAM_GB="10", SEMAFORO_ESPERA_MAX_MIN="0",
            PRUEBA_TESTIGO=self.vieja.as_posix(), PRUEBA_TRAZA=self.traza.as_posix(),
        )
        entorno.update({k: str(v) for k, v in env.items()})
        return subprocess.run([BASH, str(SEMAFORO), "check"], env=entorno,
                              capture_output=True, text=True, timeout=120)

    def bitacora(self):
        return (self.raiz / "logs" / "semaforo.log").read_text(encoding="utf-8")

    def test_barre_con_el_turno_tomado_y_antes_de_compilar(self):
        r = self.correr()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("cargo check testigo=ausente", self.traza.read_text(), "el barrido corre ANTES de cargo")
        self.assertFalse(self.vieja.exists())
        self.assertTrue(self.nueva.exists())
        self.assertTrue(self.sello.exists(), "el sello marca el último barrido")
        self.assertIn("barrido automático codigo=0", self.bitacora())
        self.assertFalse((self.raiz / "target" / ".semaforo.lock").exists(), "el turno se libera")

    def test_con_sello_reciente_no_barre_pero_compila(self):
        self.sello.write_text(str(int(time.time())))
        r = self.correr()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertTrue(self.vieja.exists())
        self.assertIn("cargo check testigo=presente", self.traza.read_text())

    def test_desactivable(self):
        r = self.correr(SEMAFORO_BARRIDO=0)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertTrue(self.vieja.exists())
        self.assertNotIn("barrido", self.bitacora())

    def test_solo_decidir_no_barre_ni_compila(self):
        r = self.correr(SEMAFORO_SOLO_DECIDIR=1)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertTrue(self.vieja.exists())
        self.assertFalse(self.traza.exists())

    def test_turno_por_paciencia_con_builds_ajenos_no_barre(self):
        r = self.correr(SEMAFORO_PROCESOS=2, SEMAFORO_PACIENCIA_MIN=0)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("turno por paciencia", self.bitacora())
        self.assertTrue(self.vieja.exists(), "con builds ajenos vivos no se borra nada")
        self.assertIn("cargo check testigo=presente", self.traza.read_text())

    def test_un_fallo_del_barrido_no_bloquea_el_build(self):
        r = self.correr(BARRIDO_INCREMENTAL_MAX_MB="abc")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("barrido automático codigo=2", self.bitacora())
        self.assertFalse(self.sello.exists(), "un barrido fallido no sella: se reintenta en el próximo turno")
        self.assertIn("cargo check", self.traza.read_text())


if __name__ == "__main__":
    unittest.main()
