#!/usr/bin/env python3
"""Contratos de CL-214: la puerta de fmt y clippy sobre las líneas cambiadas.

Cubren el parser del diff, la normalización de rutas de rustc/rustfmt (Linux y
Windows), la lectura de `rustfmt --emit json` y de `cargo clippy
--message-format=json`, la aplicación parcial de tramos de rustfmt y las
salidas 0/1/2 de `main` contra un repo git real en una carpeta temporal, con
un `cargo` sustituto que sólo vuelca salidas canónicas (sin red ni compilar).
"""

from pathlib import Path
import importlib.util
import json
import os
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest

SCRIPTS = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("puerta_lineas_cambiadas", SCRIPTS / "puerta_lineas_cambiadas.py")
P = importlib.util.module_from_spec(_spec)
sys.modules["puerta_lineas_cambiadas"] = P
_spec.loader.exec_module(P)


class Diff(unittest.TestCase):
    def test_lineas_nuevas_por_archivo_y_borrado_puro_no_cuenta(self):
        texto = textwrap.dedent(
            """\
            diff --git a/a.rs b/a.rs
            --- a/a.rs
            +++ b/a.rs
            @@ -3 +3 @@ fn x() {
            -    let a=1;
            +    let a = 1;
            @@ -10,2 +9,0 @@
            -borrada
            -borrada
            @@ -20,0 +19,3 @@
            +uno
            +dos
            +tres
            diff --git a/b.rs b/b.rs
            --- a/b.rs
            +++ /dev/null
            @@ -1 +0,0 @@
            -todo
            """
        )
        self.assertEqual(P.parsear_diff(texto), {"a.rs": {3, 19, 20, 21}})

    def test_linea_anadida_que_empieza_por_dos_mas_no_es_cabecera(self):
        texto = textwrap.dedent(
            """\
            --- a/c.rs
            +++ b/c.rs
            @@ -1,0 +1,2 @@
            +++ falso/otro.rs
            +normal
            @@ -9 +10 @@
            -x
            +y
            """
        )
        self.assertEqual(P.parsear_diff(texto), {"c.rs": {1, 2, 10}})

    def test_ruta_entre_comillas_de_git(self):
        texto = '--- a/x\n+++ "b/dir con \\"comillas\\"/ñ.rs"\n@@ -0,0 +1 @@\n+fn f() {}\n'
        self.assertEqual(P.parsear_diff(texto), {'dir con "comillas"/ñ.rs': {1}})


class Rutas(unittest.TestCase):
    def test_linux(self):
        raiz = Path("/repo/tg")
        self.assertEqual(P.normalizar("crates/a/src/lib.rs", raiz), "crates/a/src/lib.rs")
        self.assertEqual(P.normalizar("./src/main.rs", raiz), "src/main.rs")
        self.assertEqual(P.normalizar("/repo/tg/src/x.rs", raiz), "src/x.rs")
        self.assertIsNone(P.normalizar("/repo/tgx/src/x.rs", raiz))
        self.assertIsNone(P.normalizar("/home/u/.cargo/registry/src/x.rs", raiz))
        self.assertIsNone(P.normalizar("../fuera.rs", raiz))
        self.assertIsNone(P.normalizar("", raiz))

    def test_windows_relativa_absoluta_y_verbatim(self):
        raiz = Path("C:/Users/jhona/Documents/Proyectos/Trader Gemini")
        self.assertEqual(P.normalizar("crates\\a\\src\\lib.rs", raiz), "crates/a/src/lib.rs")
        self.assertEqual(
            P.normalizar("c:\\users\\jhona\\Documents\\Proyectos\\Trader Gemini\\src\\x.rs", raiz), "src/x.rs"
        )
        self.assertEqual(
            P.normalizar("\\\\?\\C:\\Users\\jhona\\Documents\\Proyectos\\Trader Gemini\\a.rs", raiz), "a.rs"
        )
        self.assertIsNone(P.normalizar("C:\\Users\\jhona\\.cargo\\registry\\x.rs", raiz))


class Fmt(unittest.TestCase):
    RAIZ = Path("/r")

    def _json(self, *archivos):
        return json.dumps([{"name": n, "mismatches": m} for n, m in archivos])

    def test_duplicados_de_un_mod_compartido_se_cuentan_una_vez(self):
        tramo = {"original_begin_line": 2, "original_end_line": 2, "expected_begin_line": 2,
                 "expected_end_line": 3, "original": "a\n", "expected": "b\nc\n"}
        salida = self._json(("/r/tests/comun/mod.rs", [tramo])) + "\n" + self._json(("/r/tests/comun/mod.rs", [tramo]))
        tramos, errores = P.parsear_fmt(salida, "", self.RAIZ)
        self.assertEqual(len(tramos), 1)
        self.assertEqual((tramos[0].ruta, tramos[0].inicio, tramos[0].fin), ("tests/comun/mod.rs", 2, 2))
        self.assertEqual(errores, [])

    def test_error_interno_con_ubicacion_y_sin_ubicacion(self):
        err = "\x1b[1merror[internal]\x1b[0m: left behind trailing whitespace\n --> /r/src/bin/g.rs:2483:2483:1\n"
        _, errores = P.parsear_fmt("", err, self.RAIZ)
        self.assertEqual([(e.ruta, e.inicio) for e in errores], [("src/bin/g.rs", 2483)])
        with self.assertRaises(P.PuertaRota):
            P.parsear_fmt("", "error: no se pudo leer la configuración\n", self.RAIZ)
        with self.assertRaises(P.PuertaRota):
            P.parsear_fmt("[{no es json", "", self.RAIZ)

    def test_aplicar_reemplazo_insercion_y_borrado(self):
        texto = "l1\nl2\nl3\nl4\n"
        tramos = [
            P.TramoFmt("f.rs", 1, 1, "l1\n", "L1\n"),
            P.TramoFmt("f.rs", 3, 3, "", "nueva\n"),  # inserción delante de l3
            P.TramoFmt("f.rs", 4, 4, "l4\n", ""),  # borrado
        ]
        self.assertEqual(P.aplicar_tramos(texto, tramos), "L1\nl2\nnueva\nl3\n")

    def test_aplicar_conserva_crlf_y_falla_cerrado_si_el_archivo_cambio(self):
        self.assertEqual(P.aplicar_tramos("a\r\nb\r\n", [P.TramoFmt("f.rs", 2, 2, "b\n", "B\n")]), "a\r\nB\r\n")
        with self.assertRaises(P.PuertaRota):
            P.aplicar_tramos("a\nb\n", [P.TramoFmt("f.rs", 2, 2, "otra\n", "B\n")])

    def test_insercion_pura_toca_sus_dos_vecinas(self):
        t = P.TramoFmt("f.rs", 10, 10, "", "x\n")
        self.assertTrue(P.tramo_toca(t, {"f.rs": {9}}))
        self.assertTrue(P.tramo_toca(t, {"f.rs": {10}}))
        self.assertFalse(P.tramo_toca(t, {"f.rs": {11}}))
        self.assertFalse(P.tramo_toca(P.TramoFmt("f.rs", 3, 5, "a\nb\nc\n", "abc\n"), {"f.rs": {2, 6}, "g.rs": {4}}))


def _mensaje(nivel, texto, spans, codigo=None):
    return json.dumps({
        "reason": "compiler-message",
        "message": {"level": nivel, "message": texto, "code": {"code": codigo} if codigo else None, "spans": spans},
    })


def _span(archivo, ini, fin=None, primario=True, expansion=None):
    s = {"file_name": archivo, "line_start": ini, "line_end": fin or ini, "is_primary": primario}
    if expansion:
        s["expansion"] = {"span": expansion}
    return s


class Clippy(unittest.TestCase):
    RAIZ = Path("/r")

    def test_avisos_errores_resumenes_y_macros(self):
        lineas = [
            json.dumps({"reason": "compiler-artifact"}),
            _mensaje("warning", "useless vec", [_span("src/a.rs", 7)], "clippy::useless_vec"),
            _mensaje("warning", "useless vec", [_span("src/a.rs", 7)], "clippy::useless_vec"),  # lib + test
            _mensaje("warning", "en macro", [_span("/home/u/.cargo/registry/m.rs", 1, expansion=_span("src/b.rs", 40))]),
            _mensaje("warning", "de una dependencia", [_span("/rustc/abc/library/core/src/x.rs", 3)]),
            _mensaje("warning", "3 warnings emitted", []),
            _mensaje("error", "aborting due to 1 previous error", []),
            _mensaje("note", "nota", [_span("src/a.rs", 1)]),
            _mensaje("error", "mismatched types", [_span("/rustc/abc/library/core/src/x.rs", 3)], "E0308"),
            _mensaje("error", "cannot find value", [_span("src/c.rs", 5, 6)], "E0425"),
        ]
        diags, errores = P.parsear_clippy("\n".join(lineas), self.RAIZ)
        self.assertEqual([(d.ruta, d.inicio) for d in diags], [("src/a.rs", 7), ("src/b.rs", 40)])
        self.assertEqual(diags[0].texto, "warning[clippy::useless_vec]: useless vec")
        self.assertEqual([(e.ruta, e.inicio, e.fin) for e in errores], [("", 0, 0), ("src/c.rs", 5, 6)])
        with self.assertRaises(P.PuertaRota):
            P.parsear_clippy("{roto", self.RAIZ)


CARGO_FALSO = '''#!{python}
import json, os, sys
modo = sys.argv[1]
if modo == "fmt":
    sys.stdout.write(open(os.environ["PUERTA_FMT"], encoding="utf-8").read())
    sys.exit(int(os.environ.get("PUERTA_FMT_RC", "0")))
sys.stdout.write(open(os.environ["PUERTA_CLIPPY"], encoding="utf-8").read())
sys.exit(int(os.environ.get("PUERTA_CLIPPY_RC", "0")))
'''


class Main(unittest.TestCase):
    """`main` de punta a punta contra un repo git real."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.raiz = Path(self.tmp.name).resolve()
        self.repo = self.raiz / "repo"
        self.repo.mkdir()
        git = lambda *a: subprocess.run(["git", *a], cwd=self.repo, check=True, capture_output=True)
        self.git = git
        git("init", "-q")
        git("config", "user.email", "puerta@test")
        git("config", "user.name", "puerta")
        (self.repo / "src").mkdir()
        (self.repo / "src" / "a.rs").write_text("fn a() {}\nfn b() {}\nfn c() {}\n", encoding="utf-8")
        (self.repo / "notas.md").write_text("x\n", encoding="utf-8")
        git("add", ".")
        git("commit", "-qm", "base")
        self.cargo = self.raiz / "cargo_falso"
        self.cargo.write_text(CARGO_FALSO.replace("{python}", sys.executable), encoding="utf-8")
        self.cargo.chmod(self.cargo.stat().st_mode | stat.S_IXUSR)
        self.fmt = self.raiz / "fmt.json"
        self.clippy = self.raiz / "clippy.json"
        self.fmt.write_text("", encoding="utf-8")
        self.clippy.write_text("", encoding="utf-8")
        self.env = {**os.environ, "PUERTA_FMT": str(self.fmt), "PUERTA_CLIPPY": str(self.clippy)}

    def tearDown(self):
        self.tmp.cleanup()

    def correr(self, *extra, env=None):
        r = subprocess.run(
            [sys.executable, str(SCRIPTS / "puerta_lineas_cambiadas.py"), "--base", "HEAD", "--cargo", str(self.cargo), *extra],
            cwd=self.repo, capture_output=True, text=True, env=env or self.env,
        )
        return r.returncode, r.stdout + r.stderr

    def test_sin_cambios_rs_sale_limpio_sin_llamar_a_cargo(self):
        (self.repo / "notas.md").write_text("y\n", encoding="utf-8")
        rc, salida = self.correr(env={**os.environ, "PUERTA_FMT": "/no/existe", "PUERTA_CLIPPY": "/no/existe"})
        self.assertEqual(rc, 0, salida)
        self.assertIn("nada que exigir", salida)

    def test_deuda_vieja_no_falla_y_la_linea_tocada_si(self):
        (self.repo / "src" / "a.rs").write_text("fn a() {}\nfn b() { let x=1; }\nfn c() {}\n", encoding="utf-8")
        ruta = str(self.repo / "src" / "a.rs")
        vieja = {"original_begin_line": 3, "original_end_line": 3, "expected_begin_line": 3,
                 "expected_end_line": 3, "original": "fn c() {}\n", "expected": "fn c() {}\n\n"}
        self.fmt.write_text(json.dumps([{"name": ruta, "mismatches": [vieja]}]) + "\n", encoding="utf-8")
        self.clippy.write_text(_mensaje("warning", "vieja", [_span("src/a.rs", 1)], "clippy::x") + "\n", encoding="utf-8")
        rc, salida = self.correr()
        self.assertEqual(rc, 0, salida)
        self.assertIn("en líneas cambiadas 0", salida)
        self.clippy.write_text(_mensaje("warning", "nueva", [_span("src/a.rs", 2)], "clippy::y") + "\n", encoding="utf-8")
        rc, salida = self.correr()
        self.assertEqual(rc, 1, salida)
        self.assertIn("::error file=src/a.rs,line=2,endLine=2,title=clippy::warning[clippy::y]: nueva", salida)

    def test_build_roto_falla_aunque_no_toque_lineas_cambiadas(self):
        (self.repo / "src" / "a.rs").write_text("fn a() {}\nfn b() {}\nfn c() { }\n", encoding="utf-8")
        self.clippy.write_text(_mensaje("error", "roto", [_span("src/otro.rs", 9)], "E0425") + "\n", encoding="utf-8")
        rc, salida = self.correr("--solo", "clippy", env={**self.env, "PUERTA_CLIPPY_RC": "101"})
        self.assertEqual(rc, 1, salida)
        self.assertIn("line=9", salida)
        self.clippy.write_text("", encoding="utf-8")
        rc, salida = self.correr("--solo", "clippy", env={**self.env, "PUERTA_CLIPPY_RC": "101"})
        self.assertEqual(rc, 1, salida)
        self.assertIn("cargo clippy salió con 101", salida)

    def test_salida_ilegible_o_base_inexistente_rompen_la_puerta(self):
        (self.repo / "src" / "a.rs").write_text("fn a() {}\nfn b() {}\nfn c() { }\n", encoding="utf-8")
        self.fmt.write_text("[{no json\n", encoding="utf-8")
        rc, salida = self.correr("--solo", "fmt")
        self.assertEqual(rc, 2, salida)
        r = subprocess.run(
            [sys.executable, str(SCRIPTS / "puerta_lineas_cambiadas.py"), "--base", "no-existe", "--cargo", str(self.cargo)],
            cwd=self.repo, capture_output=True, text=True, env=self.env,
        )
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)

    def test_archivo_nuevo_entra_entero_con_git_add(self):
        (self.repo / "src" / "nuevo.rs").write_text("fn n() {}\nfn m() {}\n", encoding="utf-8")
        self.git("add", "src/nuevo.rs")
        self.assertEqual(P.lineas_cambiadas(self.repo, "HEAD"), {"src/nuevo.rs": {1, 2}})


if __name__ == "__main__":
    unittest.main()
