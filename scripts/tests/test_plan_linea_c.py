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


class PlanLineaCInventarioCiTests(unittest.TestCase):
    """`ci`: qué objetivos de test ejecuta alguna CI (C-F10, LC-F10-02)."""

    ARCHIVOS = {
        "Cargo.toml": (
            '[workspace]\nmembers = ["crates/a", "crates/b"]\n\n'
            '[package]\nname = "raiz"\nautobins = false\n\n'
            '[lib]\nname = "raiz_lib"\npath = "src/lib.rs"\n\n'
            '[[bin]]\nname = "host"\npath = "src/bin/host.rs"\n'
        ),
        "src/lib.rs": "#[test]\nfn r() {}\n",
        "src/bin/host.rs": "fn main() {}\n#[test]\nfn h() {}\n",
        "src/bin/suelto.rs": "fn main() {}\n#[test]\nfn s() {}\n",
        "crates/a/Cargo.toml": '[package]\nname = "a-crate"\n',
        "crates/a/src/lib.rs": "mod m;\n#[cfg(test)]\nmod t { #[test] fn x() {} }\n",
        "crates/a/src/m.rs": "#[test]\nfn y() {}\n",
        "crates/a/src/bin/tool.rs": "fn main() {}\n#[test]\nfn z() {}\n",
        "crates/a/tests/t1.rs": "#[test]\nfn a() {}\n#[tokio::test]\nasync fn b() {}\n",
        "crates/a/tests/t2.rs": "#[test]\n#[ignore = \"manual\"]\nfn c() {}\n",
        "crates/b/Cargo.toml": '[package]\nname = "b"\n',
        "crates/b/tests/solo.rs": "#[test]\nfn d() {}\n",
        "crates/b/tests/otro.rs": "#[test]\nfn e() {}\n",
        "crates/b/tests/nunca.rs": "#[test]\nfn f() {}\n",
        ".github/workflows/ci.yml": (
            "jobs:\n  x:\n    strategy:\n      matrix:\n        include:\n"
            "          - grupo: uno\n            paquetes: -p a-crate --lib --tests\n"
            "          - grupo: dos\n            paquetes: -p raiz --lib --bin host\n"
            "    steps:\n"
            "      - run: cargo +nightly-2026-06-30 test ${{ matrix.paquetes }} --locked -- --test-threads=1\n"
            "      - run: cargo +nightly-2026-06-30 test -p b --locked --test solo -- filtro::x --test-threads=1\n"
            "      # cargo test -p b --test nunca\n"
            "      - run: cargo +nightly-2026-06-30 check --workspace --all-targets --locked\n"
            "      - run: cargo test -p b --test nunca --no-run\n"
            "      - run: |\n"
            "          rustc +nightly-2026-06-30 --edition 2024 --test crates/b/tests/otro.rs -o x.exe\n"
        ),
    }

    def inventario(self):
        versionados = sorted(self.ARCHIVOS)
        return {(f[0], f[1], f[2]): f for f in plan.inventario_ci(versionados, self.ARCHIVOS.__getitem__)}

    def test_matriz_y_tests_seleccionan_lib_bins_y_tests_de_integracion(self):
        inv = self.inventario()
        for clave in [("a-crate", "a-crate", "lib"), ("a-crate", "tool", "bin"),
                      ("a-crate", "t1", "test"), ("a-crate", "t2", "test")]:
            self.assertEqual(inv[clave][6], "ci.yml:11", clave)
        self.assertEqual(inv[("raiz", "raiz", "lib")][6], "ci.yml:11")
        self.assertEqual(inv[("raiz", "host", "bin")][6], "ci.yml:11")

    def test_cuenta_tests_de_todo_el_arbol_de_la_lib_e_ignorados(self):
        inv = self.inventario()
        self.assertEqual(inv[("a-crate", "a-crate", "lib")][4], 2)
        self.assertEqual(inv[("a-crate", "t1", "test")][4], 2)
        self.assertEqual(inv[("a-crate", "t2", "test")][4:6], (1, 1))

    def test_autobins_false_deja_fuera_los_bins_no_declarados(self):
        inv = self.inventario()
        self.assertNotIn(("raiz", "suelto", "bin"), inv)

    def test_filtro_rustc_comentario_y_no_run(self):
        inv = self.inventario()
        self.assertEqual(inv[("b", "solo", "test")][6], "ci.yml:12 (filtro filtro::x)")
        self.assertEqual(inv[("b", "otro", "test")][6], "ci.yml:17")
        self.assertEqual(inv[("b", "nunca", "test")][6], "ninguna")

    def test_sin_seleccion_corre_lib_bins_y_tests(self):
        paquetes = plan.paquetes_del_workspace(sorted(self.ARCHIVOS), self.ARCHIVOS.__getitem__)
        objetivos, filtro, solo = plan.seleccion_de_orden("cargo test -p a-crate", paquetes)
        self.assertEqual(objetivos, {("a-crate", "lib", None), ("a-crate", "bin", "tool"),
                                     ("a-crate", "test", "t1"), ("a-crate", "test", "t2")})
        self.assertEqual((filtro, solo), (None, False))
        objetivos, filtro, _ = plan.seleccion_de_orden("cargo test --workspace -- --ignored", paquetes)
        self.assertIn(("b", "test", "nunca"), objetivos)
        self.assertEqual(filtro, "--ignored")


class PlanLineaCClavesTests(unittest.TestCase):
    """`claves` (C-F1): claves del registro que se leen y nadie escribe."""

    ARCHIVOS = {
        "crates/r/src/lib.rs": (
            "fn f(reg: &R, json: &J) {\n"
            '    let a = reg.get_for_coin_or(0, "huerfana", 1.0);\n'
            "    let b = reg.get_scoped_parameter(\n"
            "        sym, cid,\n"
            '        "escrita", "Motor");\n'
            '    let c = reg.get_value_or("en_tupla", 0.5);\n'
            '    let j = json.get("data");\n'
            '    let k = reg.get("leida_con_consumidor", "Motor");\n'
            '    let m = reg.get_value_or("otra_huerfana", f(1, 2));\n'
            '    let n = reg.get_for_coin_or(0, "via_cierre", {\n        x\n    });\n'
            '    let p = reg.get(nombre, "ConsumidorConVariable");\n'
            '    let q = reg.get_value_fast("por_parametro");\n'
            "}\n"
            "#[cfg(test)]\n"
            'mod t { fn g(reg: &R) { reg.set("huerfana", 1.0); reg.get_value_or("solo_en_test", 0.0); } }\n'
        ),
        "crates/w/src/lib.rs": (
            "fn w(reg: &R, x: f64) {\n"
            '    reg.set_for_coin(0, "escrita", x);\n'
            '    for (n, v) in [("en_tupla", x)] { reg.set(n, v); }\n'
            "    let set_reg = |k: &str, v: f64| reg.set(k, v);\n"
            '    set_reg("via_cierre", x);\n'
            '    reg.register(Parameter::new("por_parametro", Kind::Gene, x, "w"));\n'
            "}\n"
        ),
        "crates/w/tests/t.rs": 'fn t(reg: &R) { reg.set("leida_con_consumidor", 1.0); }\n',
        "crates/w/README.md": 'reg.set("otra_huerfana", 1.0)\n',
    }

    def claves(self):
        return plan.claves_sin_escritor(sorted(self.ARCHIVOS), self.ARCHIVOS.__getitem__)

    def test_lista_lecturas_sin_escritor_fuera_de_tests(self):
        claves = self.claves()
        self.assertEqual(sorted(claves), ["huerfana", "leida_con_consumidor", "otra_huerfana"])
        self.assertEqual(claves["huerfana"]["sitios"], ["crates/r/src/lib.rs:2"])
        self.assertEqual(claves["leida_con_consumidor"]["sitios"], ["crates/r/src/lib.rs:8"])

    def test_valor_por_defecto_de_las_lecturas_con_respaldo(self):
        claves = self.claves()
        self.assertEqual(claves["huerfana"]["por_defecto"], ["1.0"])
        self.assertEqual(claves["otra_huerfana"]["por_defecto"], ["f(1, 2)"])
        self.assertEqual(claves["leida_con_consumidor"]["por_defecto"], [])

    def test_escritor_en_varias_lineas_o_en_tupla_cuenta(self):
        claves = self.claves()
        self.assertNotIn("escrita", claves)
        self.assertNotIn("en_tupla", claves)
        self.assertNotIn("data", claves)
        self.assertNotIn("solo_en_test", claves)
        self.assertNotIn("via_cierre", claves)
        self.assertNotIn("por_parametro", claves)
        self.assertNotIn("ConsumidorConVariable", claves)


if __name__ == "__main__":
    unittest.main()
