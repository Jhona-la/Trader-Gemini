"""Contratos del comprobador de citas del plan de Línea C (repositorio temporal)."""

import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest


TOOL = Path(__file__).resolve().parents[1] / "plan_linea_c.py"
SPEC = importlib.util.spec_from_file_location("plan_linea_c", TOOL)
plan = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plan)


class PlanLineaCCitasTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="plan-linea-c-test-")
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        self.git("init", "-q")
        self.git("config", "user.email", "plan-test@example.invalid")
        self.git("config", "user.name", "Plan Fixture")
        self.git("config", "commit.gpgsign", "false")
        archivos = {
            "crates/a/src/lib.rs": 10,
            "crates/b/src/lib.rs": 5,
            "src/bin/host.rs": 20,
            ".agents/MEMORIA.md": 3,
        }
        for ruta, n in archivos.items():
            destino = self.repo / ruta
            destino.parent.mkdir(parents=True, exist_ok=True)
            destino.write_text("".join("l%d\n" % i for i in range(n)), encoding="utf-8")
        self.git("add", ".")
        self.git("commit", "-q", "-m", "fixture")

    def git(self, *args):
        subprocess.run(["git", "-C", str(self.repo), *args], check=True,
                       capture_output=True, text=True)

    def comprobar(self, texto):
        return plan.comprobar(str(self.repo), "HEAD", texto)

    def test_cita_exacta_y_por_sufijo_unico(self):
        resumen, problemas = self.comprobar("ver crates/a/src/lib.rs:10 y host.rs:1-20")
        self.assertEqual(resumen["ok"], 2)
        self.assertEqual(problemas, [])

    def test_linea_fuera_del_archivo_falla(self):
        resumen, problemas = self.comprobar("crates/b/src/lib.rs:6")
        self.assertEqual(resumen["fuera_de_rango"], 1)
        self.assertEqual(problemas[0][1], "fuera_de_rango")

    def test_continuacion_usa_el_archivo_anterior_de_la_linea(self):
        resumen, _ = self.comprobar("host.rs:3, :19 y :21")
        self.assertEqual(resumen["ok"], 2)
        self.assertEqual(resumen["fuera_de_rango"], 1)

    def test_nombre_ambiguo_se_informa_sin_fallar(self):
        resumen, problemas = self.comprobar("lib.rs:4")
        self.assertEqual(resumen["ambigua"], 1)
        self.assertEqual(plan.main(["citas", self.escribir("lib.rs:4"), "--repo", str(self.repo)]), 0)

    def test_archivo_inexistente_falla(self):
        resumen, _ = self.comprobar("src/bin/otro.rs:1")
        self.assertEqual(resumen["inexistente"], 1)
        self.assertEqual(plan.main(["citas", self.escribir("src/bin/otro.rs:1"), "--repo", str(self.repo)]), 1)

    def test_ruta_que_empieza_por_punto(self):
        resumen, _ = self.comprobar("estado en .agents/MEMORIA.md:3 y ./src/bin/host.rs:2")
        self.assertEqual(resumen["ok"], 2)

    def test_horas_y_urls_no_son_citas(self):
        resumen, _ = self.comprobar("a las 22:55 UTC, ver https://example.invalid:443/x")
        self.assertEqual(sum(resumen.values()), 0)

    def test_puerto_entre_comillas_no_es_continuacion(self):
        resumen, _ = self.comprobar("host.rs:2 rasca un `:9091` inexistente")
        self.assertEqual(resumen["ok"], 1)
        self.assertEqual(resumen["fuera_de_rango"], 0)

    def test_alias_declarado_resuelve_el_nombre_corto(self):
        resumen, _ = plan.comprobar(str(self.repo), "HEAD", "lib.rs:10",
                                    {"lib.rs": "crates/a/src/lib.rs"})
        self.assertEqual(resumen["ok"], 1)
        resumen, _ = plan.comprobar(str(self.repo), "HEAD", "lib.rs:10",
                                    {"lib.rs": "crates/b/src/lib.rs"})
        self.assertEqual(resumen["fuera_de_rango"], 1)

    def escribir(self, texto):
        destino = Path(self.temp.name) / "doc.md"
        destino.write_text(texto, encoding="utf-8")
        return str(destino)


class PlanLineaCAsignacionTests(unittest.TestCase):
    def setUp(self):
        self.reglas, self.directorio = plan.REGLAS, plan.POR_DIRECTORIO
        self.addCleanup(self.restaurar)
        plan.REGLAS = {
            "C-F1": ["crates/*/Cargo.toml", "src/{config.rs, lib.rs}"],
            "C-F3": ["crates/core/src/lib.rs", "crates/core/"],
            "C-F9": ["crates/graph/", "crates/graph/Cargo.toml"],
            "C-F10": ["crates/*/tests/*"],
        }
        plan.POR_DIRECTORIO = {"src/": "C-F3"}

    def restaurar(self):
        plan.REGLAS, plan.POR_DIRECTORIO = self.reglas, self.directorio

    def fases(self, rutas):
        filas, sin_fase, inexistentes, empates = plan.asignar(rutas)
        return {r: f for r, f, _, _ in filas}, sin_fase, inexistentes, empates

    def test_gana_la_regla_mas_especifica(self):
        fases, _, _, _ = self.fases([
            "crates/core/src/lib.rs", "crates/core/src/otro.rs", "crates/core/Cargo.toml",
            "crates/core/tests/a.rs", "crates/graph/Cargo.toml", "crates/graph/src/lib.rs",
            "src/config.rs", "src/lib.rs"])
        self.assertEqual(fases["crates/core/src/lib.rs"], "C-F3")
        self.assertEqual(fases["crates/core/src/otro.rs"], "C-F3")
        self.assertEqual(fases["crates/core/Cargo.toml"], "C-F1")
        self.assertEqual(fases["crates/core/tests/a.rs"], "C-F10")
        self.assertEqual(fases["crates/graph/Cargo.toml"], "C-F9")
        self.assertEqual(fases["crates/graph/src/lib.rs"], "C-F9")

    def test_lo_no_nombrado_cae_en_su_directorio_y_se_anota(self):
        filas, sin_fase, _, _ = plan.asignar(["src/nuevo.rs", "otro/x.md",
                                              "src/config.rs", "src/lib.rs",
                                              "crates/core/src/lib.rs", "crates/graph/Cargo.toml"])
        tipo = {r: t for r, _, t, _ in filas}
        self.assertEqual(tipo["src/nuevo.rs"], "directorio")
        self.assertEqual(tipo["src/config.rs"], "explícita")
        self.assertEqual(sin_fase, ["otro/x.md"])

    def test_ruta_exacta_del_plan_que_no_existe_se_informa(self):
        _, _, inexistentes, _ = self.fases(["src/config.rs", "crates/core/src/lib.rs"])
        self.assertIn(("C-F1", "src/lib.rs"), inexistentes)
        self.assertIn(("C-F9", "crates/graph/Cargo.toml"), inexistentes)

    def test_dos_fases_con_la_misma_prioridad_es_un_empate(self):
        plan.REGLAS["C-F4"] = ["crates/core/src/lib.rs"]
        _, _, _, empates = self.fases(["crates/core/src/lib.rs"])
        self.assertEqual(empates, [("crates/core/src/lib.rs", ["C-F3", "C-F4"])])

    def test_llaves_se_expanden(self):
        self.assertEqual(plan.expandir("a/{x.rs, y.rs}"), ["a/x.rs", "a/y.rs"])
        self.assertEqual(plan.expandir("a/b.rs"), ["a/b.rs"])


if __name__ == "__main__":
    unittest.main()
