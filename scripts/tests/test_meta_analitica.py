"""Contratos de la meta-analítica del impacto por cambio (RB-A1a, Codex RB 2026-10-19).

Construye repositorios git REALES en carpetas temporales (sin mocks): commits
con Cargo.lock, tests, docs, una rama fusionada con --no-ff, el TSV del
trinquete de pánicos y un archivo de datos por encima del umbral de texto;
corre scripts/meta_analitica.py y verifica el libro por commit, el resumen y
la lectura del log del semáforo. Usa las familias de átomo REALES del repo."""

from pathlib import Path
import importlib.util
import os
import subprocess
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1]
REPO_REAL = SCRIPTS.parent
FAMILIAS_REALES = REPO_REAL / "docs" / "audit" / "meta_analitica" / "familias_atomo.tsv"

_spec = importlib.util.spec_from_file_location("meta_analitica", SCRIPTS / "meta_analitica.py")
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

LOCK_1 = """version = 4

[[package]]
name = "a"
version = "0.1.0"

[[package]]
name = "god-engine-core"
version = "0.1.0"
dependencies = [
 "a",
]

[[package]]
name = "trader-gemini-v5"
version = "0.1.0"
dependencies = ["god-engine-core"]
"""

LOCK_2 = """version = 4

[[package]]
name = "a"
version = "0.1.0"

[[package]]
name = "god-engine-core"
version = "0.1.0"
dependencies = [
 "a",
 "polars 0.43.1 (registry+https://github.com/rust-lang/crates.io-index)",
]

[[package]]
name = "polars"
version = "0.43.1"
dependencies = [
 "polars-core",
]

[[package]]
name = "polars-core"
version = "0.43.1"

[[package]]
name = "trader-gemini-v5"
version = "0.1.0"
dependencies = ["god-engine-core"]
"""

TESTS_2 = "#[test]\nfn a() {}\n#[tokio::test]\nasync fn b() {}\n#[cfg(test)]\nmod m {}\n"
TESTS_3 = "#[tokio::test]\nasync fn b() {}\n#[cfg(test)]\nmod m {}\n"
TRINQUETE = ("# comentario\npaquete\tarchivo\tunwrap\texpect\tpanico\taserto\n"
             "p\ta.rs\t1\t2\t0\t0\nq\tb.rs\t0\t0\t3\t1\n")
SEMAFORO = (
    "2026-10-19T10:00:00Z\tAGY\tADQUIERE ram_libre=5.00GB jobs=2: cargo test -p x\n"
    "2026-10-19T10:10:00Z\tAGY\tLIBERA codigo=0 duracion=900s: cargo test -p x\n"
    "2026-10-19T11:00:00Z\tClaude-LineaG-lg27\tADQUIERE ram_libre=5.00GB jobs=2: cargo check\n"
    "2026-10-19T11:30:00Z\tClaude-LineaG-lg27\tLIBERA codigo=101 duracion=1800s: cargo check\n"
    "2026-10-19T12:00:00Z\tSol\tADQUIERE ram_libre=5.00GB jobs=2: cargo build\n"
    "2026-10-19T12:05:00Z\tCodex RB\tbarrido automático codigo=0: [barrido] SIMULADO\n"
)


class RepoTemporal:
    def __init__(self, raiz: Path):
        self.raiz = raiz
        self.git("init", "-q", "-b", "main")
        for clave, valor in (("user.name", "Ana"), ("user.email", "ana@x"),
                             ("core.autocrlf", "false"), ("commit.gpgsign", "false")):
            self.git("config", clave, valor)

    def git(self, *args, fecha=None, autor=None):
        entorno = dict(os.environ)
        if fecha:
            entorno["GIT_AUTHOR_DATE"] = entorno["GIT_COMMITTER_DATE"] = fecha
        if autor:
            entorno["GIT_AUTHOR_NAME"], entorno["GIT_AUTHOR_EMAIL"] = autor, "bot@x"
        r = subprocess.run(["git", *args], cwd=self.raiz, env=entorno, capture_output=True, text=True)
        if r.returncode != 0:
            raise AssertionError(f"git {args}: {r.stderr}")
        return r.stdout.strip()

    def escribir(self, archivos: dict):
        for ruta, contenido in archivos.items():
            p = self.raiz / ruta
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(contenido.encode("utf-8"))

    def commit(self, archivos: dict, mensaje: str, fecha: str, autor=None):
        self.escribir(archivos)
        self.git("add", "-A")
        self.git("commit", "-q", "-m", mensaje, fecha=fecha, autor=autor)
        return self.git("rev-parse", "HEAD")


class LibroDeImpacto(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="rb_a1_")
        raiz = Path(cls.tmp.name) / "repo"
        raiz.mkdir()
        r = RepoTemporal(raiz)
        cls.c1 = r.commit({"Cargo.lock": LOCK_1, "crates/x/src/lib.rs": "pub fn f() {}\n",
                           "docs/n.md": "hola\n"},
                          "RB-X1: base del libro", "2026-10-01T10:00:00+00:00")
        cls.c2 = r.commit({"Cargo.lock": LOCK_2, "crates/x/tests/t.rs": TESTS_2},
                          "Ola Ω200: dependencia nueva\n\nOráculo T-1: PASA 18/144.",
                          "2026-10-02T10:00:00+00:00")
        r.git("checkout", "-q", "-b", "rama")
        r.commit({"crates/x/tests/t.rs": TESTS_3,
                  "crates/x/src/lib.rs": "pub fn f() {}\npub fn g() {}\npub fn h() {}\n"},
                 "CL-99: quita una prueba", "2026-10-03T10:00:00+00:00")
        r.git("checkout", "-q", "main")
        r.git("merge", "-q", "--no-ff", "rama", "-m", "Merge PR #5: C-F9 limpieza",
              fecha="2026-10-04T10:00:00+00:00")
        cls.merge = r.git("rev-parse", "HEAD")
        grande = "x\n" * (1_300_000)  # 2,6 MB: por encima del umbral de texto (2 MB)
        cls.c5 = r.commit({"docs/audit/trinquete_panicos_produccion.tsv": TRINQUETE,
                           "data/grande.txt": grande},
                          "sin familia conocida", "2026-10-05T10:00:00+00:00", autor="Bot")
        cls.raiz = raiz
        cls.salida = Path(cls.tmp.name) / "salida"
        cls.semaforo = Path(cls.tmp.name) / "semaforo.log"
        cls.semaforo.write_bytes(SEMAFORO.encode("utf-8"))
        codigo = m.main(["--repo", str(raiz), "--salida", str(cls.salida), "--semaforo",
                         str(cls.semaforo), "--familias", str(FAMILIAS_REALES)])
        assert codigo == 0, codigo
        lineas = (cls.salida / "libro_impacto.tsv").read_text(encoding="utf-8").splitlines()
        cls.cabecera = lineas[0].split("\t")
        cls.filas = [dict(zip(cls.cabecera, l.split("\t"))) for l in lineas[1:]]
        cls.resumen = (cls.salida / "resumen.md").read_text(encoding="utf-8")
        cls.areas_lineas = (cls.salida / "area_impacto.tsv").read_text(encoding="utf-8").splitlines()
        cls.areas_cabecera = cls.areas_lineas[0].split("\t")
        cls.areas = [dict(zip(cls.areas_cabecera, l.split("\t"))) for l in cls.areas_lineas[1:]]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def fila(self, sha):
        return next(f for f in self.filas if sha.startswith(f["sha"]))

    def test_una_fila_por_commit_de_primer_padre_en_orden(self):
        self.assertEqual(self.cabecera, m.COLUMNAS)
        self.assertEqual([f["sha"] for f in self.filas],
                         [s[:12] for s in (self.c1, self.c2, self.merge, self.c5)],
                         "el commit de la rama fusionada no es de primer padre")

    def test_atomos_y_agente_desde_las_familias_reales(self):
        self.assertEqual((self.fila(self.c1)["agente"], self.fila(self.c1)["atomos"]), ("Codex RB", "RB-X1"))
        self.assertEqual((self.fila(self.c2)["agente"], self.fila(self.c2)["atomos"]), ("AGY", "Ω200"))
        self.assertEqual((self.fila(self.merge)["agente"], self.fila(self.merge)["atomos"]),
                         ("Claude Línea C", "C-F9,PR #5"))
        self.assertEqual((self.fila(self.c5)["agente"], self.fila(self.c5)["atomos"]), ("Bot", ""),
                         "sin familia, el agente es el autor")

    def test_lineas_por_clase_y_merge_contra_el_primer_padre(self):
        c1 = self.fila(self.c1)
        self.assertEqual((c1["prod_mas"], c1["docs_mas"], c1["build_mas"]),
                         ("1", "1", str(LOCK_1.count("\n"))))
        mg = self.fila(self.merge)
        self.assertEqual((mg["prod_mas"], mg["prod_menos"], mg["tests_mas"], mg["tests_menos"]),
                         ("2", "0", "0", "2"), "el merge cuenta su diff contra el primer padre")

    def test_delta_de_funciones_de_test(self):
        self.assertEqual([self.fila(s)["delta_tests"] for s in (self.c1, self.c2, self.merge, self.c5)],
                         ["0", "2", "-1", "0"], "#[cfg(test)] no es una función de test")

    def test_clausura_del_nucleo_y_del_host_por_version_del_lock(self):
        valores = [(self.fila(s)["clausura_nucleo"], self.fila(s)["clausura_host"], self.fila(s)["paquetes_lock"])
                   for s in (self.c1, self.c2, self.merge, self.c5)]
        self.assertEqual(valores, [("2", "3", "3"), ("4", "5", "5"), ("4", "5", "5"), ("4", "5", "5")])

    def test_trinquete_t1_y_umbral_de_texto(self):
        self.assertEqual([self.fila(s)["panicos_prod"] for s in (self.c1, self.c5)], ["", "7"])
        self.assertEqual([self.fila(s)["t1_k"] for s in (self.c1, self.c2)], ["", "18"])
        c5 = self.fila(self.c5)
        self.assertEqual(c5["otros_mas"], "0", "un archivo de datos sobre el umbral no suma líneas")
        self.assertEqual(c5["archivos"], "2")

    def test_resumen_con_las_cuatro_dimensiones(self):
        r = self.resumen
        self.assertIn("Clausura de dependencias de `god-engine-core` en `Cargo.lock`: 2 → 4 paquetes", r)
        self.assertIn(f"| +2 | `{self.c2[:12]}` | AGY |", r)
        self.assertIn("Δ neto de funciones de test en el rango: +1.", r)
        self.assertIn("Sitios de pánico de producción (trinquete RB-P1): 7 → 7", r)
        self.assertIn(f"| `{self.c2[:12]}` | 2026-10-02 | AGY | 18 |", r)
        self.assertIn("| AGY | 1 | 0.17 | 0.08 | 0 |", r)
        self.assertIn("| Claude-LineaG | 1 | 0.50 | 0.00 | 1 |", r)
        self.assertIn("2 cerrados, 1 sin liberar", r)

    def test_regenerar_es_determinista(self):
        otra = Path(self.tmp.name) / "salida2"
        self.assertEqual(m.main(["--repo", str(self.raiz), "--salida", str(otra), "--semaforo",
                                 str(self.semaforo), "--familias", str(FAMILIAS_REALES)]), 0)
        for nombre in ("libro_impacto.tsv", "area_impacto.tsv", "resumen.md"):
            self.assertEqual((otra / nombre).read_bytes(), (self.salida / nombre).read_bytes(), nombre)

    def test_desglose_area_conserva_los_totales_del_libro(self):
        self.assertEqual(self.areas_cabecera, list(m.COLUMNAS_AREA))
        suma = lambda col: sum(int(f[col]) for f in self.filas)
        for clave in ("prod_mas", "prod_menos", "tests_mas", "tests_menos"):
            self.assertEqual(sum(int(a[clave]) for a in self.areas), suma(clave), clave)
        # Los archivos del desglose son un subconjunto: docs, build, ci y otros
        # quedan fuera (transversales por clase, ya en el libro).
        self.assertLessEqual(sum(int(a["archivos"]) for a in self.areas),
                             sum(int(f["archivos"]) for f in self.filas))

    def test_area_x_con_sus_valores_y_la_seccion_del_resumen(self):
        (x,) = [a for a in self.areas if a["area"] == "x"]
        self.assertEqual((x["commits"], x["archivos"], x["prod_mas"], x["prod_menos"],
                          x["tests_mas"], x["tests_menos"]), ("3", "4", "3", "0", "6", "2"))
        self.assertEqual((x["ratio_tests"], x["share_prod_mas"]), ("2.0", "1.0"))
        self.assertEqual((x["primera_fecha"], x["ultima_fecha"]), ("2026-10-01", "2026-10-04"))
        r = self.resumen
        self.assertIn("## 6. Atribución por área (componentes)", r)
        self.assertIn("| x | 3 | 3 | 0 | 6 | 2 | 2.00 | 100.0% | 2026-10-01 | 2026-10-04 |", r)
        self.assertIn("Concentración: el top 3 de áreas concentra el 100.0%", r)
        self.assertIn("Ningún área con producción añadida queda por debajo del umbral", r)


class Piezas(unittest.TestCase):
    def test_clase_de_cada_ruta(self):
        casos = {"crates/a/src/lib.rs": "prod", "src/bin/x.rs": "prod", "crates/a/tests/t.rs": "tests",
                 "crates/a/benches/b.rs": "tests", "Cargo.lock": "build", "crates/a/Cargo.toml": "build",
                 ".cargo/config.toml": "build", "docs/x.tsv": "docs", "AGENT_SYNC.md": "docs",
                 ".agents/MEMORIA.md": "docs", "scripts/x.py": "scripts", "scripts/tests/t.py": "scripts",
                 ".github/workflows/w.yml": "ci", "models/m.json": "otros"}
        for ruta, esperado in casos.items():
            self.assertEqual(m.clase(ruta), esperado, ruta)

    def test_area_de_cada_ruta(self):
        casos = {"crates/a/src/lib.rs": "a", "crates/b/tests/t.rs": "b", "crates/a/benches/b.rs": "a",
                 "crates/": "crates", "crates": "otros", "src/main.rs": "host", "src/bin/x.rs": "host",
                 "x.rs": "host",
                 "Cargo.lock": "workspace", "Cargo.toml": "workspace", ".cargo/config.toml": "workspace",
                 "data/f.db": "datos", "docs/x.tsv": "docs", "AGENT_SYNC.md": "docs",
                 "scripts/x.py": "scripts", ".github/workflows/w.yml": "ci", "models/m.json": "otros"}
        for ruta, esperado in casos.items():
            self.assertEqual(m.area(ruta), esperado, ruta)

    def test_area_nunca_vacia_y_derivada_del_arbol(self):
        """Toda ruta cae en exactamente un área no vacía; las áreas de prod/tests
        bajo crates/ o src/ son componentes concretos del árbol (crate o host)."""
        rutas = ("crates/a/src/lib.rs", "crates/b/tests/t.rs", "crates/benches/b.rs",
                 "src/main.rs", "src/bin/x.rs", "Cargo.lock", "Cargo.toml",
                 ".cargo/config.toml", "data/x.bin", "docs/x.md", "AGENT_SYNC.md",
                 "scripts/x.py", ".github/workflows/w.yml", "models/m.json", "x.rs", "a/b/c.txt")
        for ruta in rutas:
            k, a = m.clase(ruta), m.area(ruta)
            self.assertTrue(a, ruta)
            if k in ("prod", "tests") and ruta.startswith("crates/"):
                self.assertEqual(a, ruta.split("/")[1], ruta)
            elif k in ("prod", "tests") and ruta.startswith("src/"):
                self.assertEqual(a, "host", ruta)
            elif k in ("prod", "tests"):
                self.assertIn(a, ("otros", "host", "datos", "workspace"), ruta)

    def test_t1_declarado_en_sus_variantes(self):
        for texto, k in (("T-1 18/144", 18), ("veredicto del oraculo T-1 de re-certificacion (PASA 16/144)", 16),
                         ("LG-docs: T-1 re-certificado (18/144)", 18), ("T-10 3/144", None),
                         ("sin oráculo", None)):
            self.assertEqual(m.t1_declarado({"asunto": texto, "cuerpo": ""}), k, texto)

    def test_agente_del_semaforo_sin_sufijos_de_tarea(self):
        for etiqueta, esperado in (("Claude-LineaG-lg27", "Claude-LineaG"), ("Claude-LineaG-merge150", "Claude-LineaG"),
                                   ("Codex RB", "Codex RB"), ("AGY", "AGY"), ("claude-lineaH", "claude-lineaH")):
            self.assertEqual(m.agente_semaforo(etiqueta), esperado, etiqueta)


class AreasDeuda(unittest.TestCase):
    """Repositorio propio y mínimo: un componente (host) con producción y sin
    tests, y otro (crates/y) con ambos. Verifica la alerta de deuda, la guarda
    de división (ratio 0.0, jamás NaN) y los shares."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="rb_area_")
        raiz = Path(cls.tmp.name) / "repo"
        raiz.mkdir()
        r = RepoTemporal(raiz)
        r.commit({"crates/y/src/lib.rs": "pub fn f() {}\n" * 3,
                  "src/bin/app.rs": "fn main() {}\n"},
                 "RB-Y1: dos componentes", "2026-10-03T10:00:00+00:00")
        r.commit({"crates/y/tests/t.rs": "#[test]\nfn a() {}\n"},
                 "RB-Y2: pruebas de y", "2026-10-04T10:00:00+00:00")
        cls.raiz = raiz
        cls.salida = Path(cls.tmp.name) / "salida"
        cls.semaforo = Path(cls.tmp.name) / "inexistente.log"
        codigo = m.main(["--repo", str(raiz), "--salida", str(cls.salida), "--semaforo",
                         str(cls.semaforo), "--familias", str(FAMILIAS_REALES)])
        assert codigo == 0, codigo
        lineas = (cls.salida / "area_impacto.tsv").read_text(encoding="utf-8").splitlines()
        filas = [dict(zip(lineas[0].split("\t"), l.split("\t"))) for l in lineas[1:]]
        cls.areas = {f["area"]: f for f in filas}
        cls.resumen = (cls.salida / "resumen.md").read_text(encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_orden_por_produccion_y_shares(self):
        self.assertEqual(list(self.areas), ["y", "host"], "orden por producción añadida")
        self.assertAlmostEqual(float(self.areas["y"]["share_prod_mas"]), 0.75)
        self.assertAlmostEqual(float(self.areas["host"]["share_prod_mas"]), 0.25)

    def test_guarda_de_division_y_alerta_de_deuda(self):
        self.assertEqual(float(self.areas["host"]["ratio_tests"]), 0.0, "0/1, no NaN")
        self.assertAlmostEqual(float(self.areas["y"]["ratio_tests"]), 2 / 3)
        self.assertIn("Áreas con producción añadida y ratio de tests < 0.05 "
                      "(candidatas a deuda de pruebas): host.", self.resumen)

    def test_commits_y_archivos_por_area(self):
        self.assertEqual((self.areas["y"]["commits"], self.areas["y"]["archivos"]), ("2", "2"))
        self.assertEqual((self.areas["host"]["commits"], self.areas["host"]["archivos"]), ("1", "1"))
        self.assertIn("Concentración: el top 3 de áreas concentra el 100.0%", self.resumen)


if __name__ == "__main__":
    unittest.main()
