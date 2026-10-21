"""Sonda de frescura en un target compartido entre dos árboles (RB-H2, Codex RB 2026-10-19).

Todos los agentes compilan en el único target/ de scripts/cargo_semaforo.sh
desde worktrees distintos. Cargo nombra los artefactos de los paquetes del
workspace con un hash de su ruta RELATIVA al workspace, así que dos árboles del
mismo paquete escriben los mismos archivos; por defecto decide la frescura por
mtime. La sonda arma dos árboles del mismo paquete con un target común: el
árbol 2 tiene fuentes más viejas que el build del árbol 1.

- Con la `.cargo/config.toml` del repositorio (checksum-freshness), cada árbol
  ejecuta su propio código.
- Control negativo, sin esa configuración: el árbol 2 ejecuta el binario del
  árbol 1. Es el fallo que corrige RB-H2; si un día cargo lo arreglara de
  serie, este control lo avisa y la clave podría revisarse.

Compila dos crates diminutos con el toolchain fijado en un directorio temporal
propio: jamás toca el target/ del repositorio."""

from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import time
import unittest

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / ".cargo" / "config.toml"
MANIFIESTO = b'[package]\nname = "sonda"\nversion = "0.1.0"\nedition = "2021"\n\n[workspace]\n'


def _canal():
    for linea in (REPO / "rust-toolchain.toml").read_text(encoding="utf-8").splitlines():
        clave, _, valor = linea.partition("=")
        if clave.strip() == "channel":
            return valor.strip().strip('"')
    return None


CARGO = shutil.which("cargo")
CANAL = _canal()


@unittest.skipUnless(CARGO and CANAL, "requiere cargo (rustup) y rust-toolchain.toml")
class SondaFrescuraTargetCompartido(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="rb_h2_")
        self.base = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def arbol(self, nombre, mensaje, horas_atras):
        d = self.base / nombre
        (d / "src").mkdir(parents=True)
        (d / "Cargo.toml").write_bytes(MANIFIESTO)
        fuente = d / "src" / "main.rs"
        fuente.write_bytes(('fn main() { println!("%s"); }\n' % mensaje).encode("utf-8"))
        t = time.time() - horas_atras * 3600
        os.utime(fuente, (t, t))
        return d

    def correr(self, arbol):
        entorno = {k: v for k, v in os.environ.items()
                   if not k.upper().startswith(("CARGO_UNSTABLE_", "CARGO_TARGET_DIR", "CARGO_BUILD_"))}
        entorno["CARGO_TARGET_DIR"] = str(self.base / "target")
        r = subprocess.run([CARGO, "+" + CANAL, "run", "-q"], cwd=arbol, env=entorno,
                           capture_output=True, text=True, timeout=600)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        return r.stdout.strip().splitlines()[-1]

    def ronda(self, config):
        if config is not None:
            (self.base / ".cargo").mkdir()
            (self.base / ".cargo" / "config.toml").write_bytes(config)
        # El árbol 2 se escribe con fuentes UNA HORA más viejas que el build
        # del árbol 1: el caso de un worktree sacado antes que el build ajeno.
        arbol2 = self.arbol("arbol2", "ARBOL-2", 1)
        arbol1 = self.arbol("arbol1", "ARBOL-1", 0)
        return self.correr(arbol1), self.correr(arbol2)

    def test_con_la_config_del_repo_cada_arbol_ejecuta_su_propio_codigo(self):
        self.assertEqual(self.ronda(CONFIG.read_bytes()), ("ARBOL-1", "ARBOL-2"))

    def test_control_negativo_por_mtime_el_arbol_viejo_ejecuta_el_codigo_ajeno(self):
        self.assertEqual(self.ronda(None), ("ARBOL-1", "ARBOL-1"))


if __name__ == "__main__":
    unittest.main()
