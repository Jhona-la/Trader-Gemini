#!/usr/bin/env python3
"""CL-214 — Puerta de `cargo fmt` y `cargo clippy` sobre las líneas cambiadas.

Medido sobre main a012146e (2026-10-31): `cargo fmt --check` falla en 577 de
860 archivos .rs y clippy da 797 avisos únicos en 253 archivos (4 de nivel
deny). Reformatear el repo de golpe choca con todas las sesiones que lo editan
en paralelo (AGENTS.md, `sesiones_concurrentes`). Esta puerta exige formato y
cero diagnósticos SÓLO en las líneas que un PR o un push añade o modifica
respecto de su base: la deuda nueva queda prohibida y la vieja baja cada vez
que alguien toca una de sus líneas, sin reformatear código ajeno.

Uso:
  python3 scripts/puerta_lineas_cambiadas.py --base origin/main
  python3 scripts/puerta_lineas_cambiadas.py --base origin/main --solo fmt
  python3 scripts/puerta_lineas_cambiadas.py --base origin/main --solo fmt --arreglar
  python3 scripts/puerta_lineas_cambiadas.py --base origin/main --solo clippy \
      --cargo-extra=--target=x86_64-pc-windows-gnu

El diff es de `--base` contra el árbol de trabajo: en local cuenta lo aún no
commiteado; un archivo nuevo necesita `git add` (o `git add -N`) para entrar.

`--arreglar` aplica, en el árbol de trabajo, sólo los cambios de rustfmt que
tocan líneas cambiadas; el resto del archivo queda como estaba. Compila
después: rustfmt puede mover un `use` entre dos tramos y aplicar uno solo de
ellos deja el import duplicado o ausente.

En clippy los lints de nivel deny se bajan a warn (`-W clippy::all`) para que
un deny en código viejo no aborte el crate y oculte los diagnósticos del resto
del grafo; ninguno se silencia: cualquier diagnóstico en una línea cambiada
falla, y un error de compilación falla esté donde esté.

Salidas: 0 limpio, 1 hallazgos en líneas cambiadas o build roto, 2 la propia
puerta no pudo medir (git, cargo o una salida ilegible): falla cerrada.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

CABECERA_HUNK = re.compile(r"^@@ -\d+(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
UBICACION = re.compile(r"-->\s+(.+?\.rs):(\d+)(?::\d+)*\s*$")


class PuertaRota(Exception):
    """La puerta no pudo medir: se sale con 2 y nunca se da por limpio."""


@dataclass(frozen=True)
class Hallazgo:
    ruta: str
    inicio: int
    fin: int
    herramienta: str
    texto: str


@dataclass(frozen=True)
class TramoFmt:
    ruta: str
    inicio: int
    fin: int
    original: str
    esperado: str


ESCAPES_GIT = {"\\": "\\", '"': '"', "t": "\t", "n": "\n", "r": "\r"}


def sin_comillas(texto: str) -> str:
    """Deshace las comillas C de git (con core.quotePath=false sólo escapa
    comillas, barras y controles; los bytes no ASCII salen tal cual)."""
    salida, i = [], 0
    while i < len(texto):
        c = texto[i]
        if c == "\\" and i + 1 < len(texto) and texto[i + 1] in ESCAPES_GIT:
            salida.append(ESCAPES_GIT[texto[i + 1]])
            i += 2
            continue
        salida.append(c)
        i += 1
    return "".join(salida)


def parsear_diff(texto: str) -> dict[str, set[int]]:
    """Líneas añadidas o modificadas por archivo, en la numeración del nuevo.

    Espera la salida de `git diff -U0`. Cuenta las líneas de cada tramo según
    su cabecera, así que una línea añadida que empiece por `++ ` no se toma por
    la cabecera `+++` de otro archivo. Un tramo `+c,0` es un borrado puro y no
    aporta líneas nuevas; un archivo borrado (`+++ /dev/null`) tampoco.
    """
    cambios: dict[str, set[int]] = {}
    actual: str | None = None
    pendientes = 0
    for linea in texto.splitlines():
        if pendientes > 0:
            pendientes -= 1
            continue
        if linea.startswith("+++ "):
            destino = linea[4:].rstrip("\t")
            if destino == "/dev/null":
                actual = None
                continue
            if destino.startswith('"') and destino.endswith('"'):
                destino = sin_comillas(destino[1:-1])
            actual = destino[2:] if destino.startswith("b/") else destino
            continue
        m = CABECERA_HUNK.match(linea)
        if m:
            viejas = 1 if m.group(1) is None else int(m.group(1))
            inicio = int(m.group(2))
            nuevas = 1 if m.group(3) is None else int(m.group(3))
            pendientes = viejas + nuevas
            if actual is not None and nuevas > 0:
                cambios.setdefault(actual, set()).update(range(inicio, inicio + nuevas))
    return {r: l for r, l in cambios.items() if l}


def lineas_cambiadas(raiz: Path, base: str) -> dict[str, set[int]]:
    """Diff de `base` contra el árbol de trabajo (en CI coincide con HEAD).

    Así cuenta también lo aún no commiteado al usarla en local; un archivo
    nuevo sin `git add` (o `git add -N`) no entra en el diff.
    """
    orden = [
        "git", "-c", "core.quotePath=false", "diff", "-U0", "--no-color",
        "--no-ext-diff", "--diff-filter=AMR", "-M", base, "--", "*.rs",
    ]
    r = subprocess.run(orden, cwd=raiz, capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise PuertaRota(f"git diff {base} falló ({r.returncode}): {r.stderr.strip()}")
    return parsear_diff(r.stdout)


def normalizar(ruta: str, raiz: Path) -> str | None:
    """Ruta relativa a la raíz con `/`, o None si queda fuera del repo.

    rustc y rustfmt dan rutas relativas (con `\\` en Windows) o absolutas
    (`C:\\…`, `\\\\?\\C:\\…`, `/…`). En una ruta con unidad la
    comparación con la raíz ignora mayúsculas, como el propio Windows.
    """
    if not ruta:
        return None
    r = ruta.replace("\\", "/")
    if r.startswith("//?/"):
        r = r[4:]
    base = raiz.as_posix().rstrip("/")
    con_unidad = re.match(r"^[A-Za-z]:/", r) is not None
    if con_unidad or r.startswith("/"):
        prefijo = base + "/"
        coincide = r.lower().startswith(prefijo.lower()) if con_unidad else r.startswith(prefijo)
        if not coincide:
            return None
        r = r[len(prefijo):]
    while r.startswith("./"):
        r = r[2:]
    if not r or r == ".." or r.startswith("../"):
        return None
    return r


def toca(lineas: set[int], inicio: int, fin: int) -> bool:
    if fin < inicio:
        fin = inicio
    return any(n in lineas for n in range(inicio, fin + 1))


def parsear_fmt(stdout: str, stderr: str, raiz: Path) -> tuple[list[TramoFmt], list[Hallazgo]]:
    """Tramos de `rustfmt --emit json` (un array por crate y línea) y errores.

    Un error de rustfmt sin ubicación (p. ej. un archivo que no parsea sin
    `-->`) rompe la puerta: no se puede saber si toca una línea cambiada.
    """
    tramos: list[TramoFmt] = []
    vistos: set[TramoFmt] = set()
    for linea in stdout.splitlines():
        linea = linea.strip()
        if not linea:
            continue
        try:
            archivos = json.loads(linea)
        except json.JSONDecodeError as e:
            raise PuertaRota(f"salida de rustfmt ilegible: {e}: {linea[:120]}") from e
        for archivo in archivos:
            ruta = normalizar(archivo["name"], raiz)
            if ruta is None:
                continue
            for m in archivo["mismatches"]:
                t = TramoFmt(
                    ruta, int(m["original_begin_line"]), int(m["original_end_line"]),
                    m["original"], m["expected"],
                )
                # Un `mod comun;` de tests/ entra en varios binarios y rustfmt
                # lo reporta una vez por cada uno.
                if t not in vistos:
                    vistos.add(t)
                    tramos.append(t)
    errores: list[Hallazgo] = []
    limpio = ANSI.sub("", stderr).splitlines()
    for i, linea in enumerate(limpio):
        if not linea.startswith("error"):
            continue
        ubicacion = None
        for siguiente in limpio[i + 1 : i + 4]:
            m = UBICACION.search(siguiente)
            if m:
                ubicacion = m
                break
        if ubicacion is None:
            raise PuertaRota(f"rustfmt falló sin ubicación: {linea.strip()}")
        ruta = normalizar(ubicacion.group(1), raiz)
        if ruta is None:
            continue
        n = int(ubicacion.group(2))
        errores.append(Hallazgo(ruta, n, n, "rustfmt", linea.strip()))
    return tramos, errores


def medir_fmt(raiz: Path, cargo: str) -> tuple[list[TramoFmt], list[Hallazgo]]:
    # `--emit json` no escribe archivos: sólo vuelca los tramos por stdout.
    r = subprocess.run(
        [cargo, "fmt", "--all", "--", "--emit", "json"],
        cwd=raiz, capture_output=True, text=True, encoding="utf-8",
    )
    if r.returncode != 0 and not r.stdout.strip():
        raise PuertaRota(f"cargo fmt falló ({r.returncode}) sin salida: {r.stderr.strip()[:400]}")
    return parsear_fmt(r.stdout, r.stderr, raiz)


def ubicar(span: dict, raiz: Path) -> tuple[str, int, int] | None:
    """Primer tramo dentro del repo siguiendo la cadena de expansión de macros."""
    s: dict | None = span
    while s:
        ruta = normalizar(s.get("file_name", ""), raiz)
        if ruta is not None:
            return ruta, int(s["line_start"]), int(s["line_end"])
        s = (s.get("expansion") or {}).get("span")
    return None


def parsear_clippy(stdout: str, raiz: Path) -> tuple[list[Hallazgo], list[Hallazgo]]:
    """(diagnósticos con ubicación en el repo, errores de compilación)."""
    vistos: set[tuple] = set()
    diags: list[Hallazgo] = []
    errores: list[Hallazgo] = []
    for linea in stdout.splitlines():
        linea = linea.strip()
        if not linea.startswith("{"):
            continue
        try:
            msg = json.loads(linea)
        except json.JSONDecodeError as e:
            raise PuertaRota(f"salida de cargo ilegible: {e}: {linea[:120]}") from e
        if msg.get("reason") != "compiler-message":
            continue
        m = msg["message"]
        nivel = m.get("level", "")
        if nivel not in ("warning", "error"):
            continue
        codigo = (m.get("code") or {}).get("code") or ""
        primarios = [s for s in m.get("spans", []) if s.get("is_primary")]
        texto = f"{nivel}[{codigo}]: {m.get('message', '')}" if codigo else f"{nivel}: {m.get('message', '')}"
        if not primarios:
            # Resúmenes («N warnings emitted», «aborting due to…») no tienen
            # ubicación; un error sin ubicación sí rompe el build.
            if nivel == "error" and not m.get("message", "").startswith("aborting due to"):
                errores.append(Hallazgo("", 0, 0, "rustc", texto))
            continue
        for s in primarios:
            ubicado = ubicar(s, raiz)
            if ubicado is None:
                # Fuera del repo (macro de una dependencia, std): un aviso no
                # es nuestro; un error rompe el build igual.
                if nivel == "error":
                    errores.append(Hallazgo("", 0, 0, "rustc", texto))
                continue
            ruta, ini, fin = ubicado
            clave = (ruta, ini, fin, texto)
            if clave in vistos:
                continue
            vistos.add(clave)
            h = Hallazgo(ruta, ini, fin, "clippy", texto)
            (errores if nivel == "error" else diags).append(h)
    return diags, errores


def medir_clippy(raiz: Path, cargo: str, extra: list[str]) -> tuple[list[Hallazgo], list[Hallazgo]]:
    orden = [
        cargo, "clippy", "--workspace", "--all-targets", "--locked",
        "--message-format=json", *extra, "--", "-W", "clippy::all",
    ]
    r = subprocess.run(orden, cwd=raiz, capture_output=True, text=True, encoding="utf-8")
    diags, errores = parsear_clippy(r.stdout, raiz)
    if r.returncode != 0 and not errores:
        cola = "\n".join(r.stderr.strip().splitlines()[-15:])
        errores.append(Hallazgo("", 0, 0, "cargo", f"cargo clippy salió con {r.returncode}:\n{cola}"))
    return diags, errores


def aplicar_tramos(texto: str, tramos: list[TramoFmt]) -> str:
    """Aplica tramos de rustfmt (no solapados) de abajo arriba sobre `texto`."""
    nl = "\r\n" if "\r\n" in texto else "\n"
    lineas = texto.replace("\r\n", "\n").split("\n")
    final_vacio = lineas and lineas[-1] == ""
    if final_vacio:
        lineas.pop()
    for t in sorted(tramos, key=lambda t: t.inicio, reverse=True):
        nuevas = t.esperado.replace("\r\n", "\n")
        nuevas_l = nuevas[:-1].split("\n") if nuevas.endswith("\n") else (nuevas.split("\n") if nuevas else [])
        if t.original == "":
            # Inserción pura: va delante de la línea `inicio` del original.
            lineas[t.inicio - 1 : t.inicio - 1] = nuevas_l
            continue
        actual = "".join(l + "\n" for l in lineas[t.inicio - 1 : t.fin])
        if actual != t.original.replace("\r\n", "\n"):
            raise PuertaRota(f"{t.ruta}:{t.inicio}: el tramo de rustfmt ya no coincide con el archivo")
        lineas[t.inicio - 1 : t.fin] = nuevas_l
    salida = "\n".join(lineas) + ("\n" if final_vacio else "")
    return salida.replace("\n", nl) if nl == "\r\n" else salida


def tramo_toca(t: TramoFmt, cambios: dict[str, set[int]]) -> bool:
    lineas = cambios.get(t.ruta)
    if not lineas:
        return False
    if t.original == "":
        # Una inserción pura toca la línea anterior y la siguiente.
        return toca(lineas, max(1, t.inicio - 1), t.inicio)
    return toca(lineas, t.inicio, t.fin)


def anotar(h: Hallazgo) -> str:
    texto = h.texto.replace("%", "%25").replace("\r", "").replace("\n", "%0A")
    if h.ruta:
        return f"::error file={h.ruta},line={h.inicio},endLine={max(h.inicio, h.fin)},title={h.herramienta}::{texto}"
    return f"::error title={h.herramienta}::{texto}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--base", required=True, help="commit o ref de la base del diff")
    p.add_argument("--solo", choices=("fmt", "clippy"))
    p.add_argument("--arreglar", action="store_true", help="aplica los tramos de rustfmt que tocan líneas cambiadas")
    p.add_argument("--cargo", default=os.environ.get("CARGO", "cargo"))
    p.add_argument("--cargo-extra", action="append", default=[], help="argumento extra para cargo clippy")
    a = p.parse_args(argv)
    top = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, encoding="utf-8")
    if top.returncode != 0 or not top.stdout.strip():
        print(f"::error title=puerta rota::no estoy dentro de un repo git: {top.stderr.strip()}")
        return 2
    raiz = Path(top.stdout.strip()).resolve()
    if a.arreglar and a.solo != "fmt":
        print("--arreglar sólo aplica con --solo fmt", file=sys.stderr)
        return 2
    try:
        cambios = lineas_cambiadas(raiz, a.base)
        total = sum(len(v) for v in cambios.values())
        print(f"Líneas .rs añadidas o modificadas frente a {a.base}: {total} en {len(cambios)} archivos")
        if not cambios:
            print("Sin líneas .rs cambiadas: nada que exigir.")
            return 0
        hallazgos: list[Hallazgo] = []
        if a.solo in (None, "fmt"):
            tramos, errores_fmt = medir_fmt(raiz, a.cargo)
            mios = [t for t in tramos if tramo_toca(t, cambios)]
            print(f"fmt: deuda total {len(tramos)} tramos en {len({t.ruta for t in tramos})} archivos; en líneas cambiadas {len(mios)}")
            if a.arreglar and mios:
                por_archivo: dict[str, list[TramoFmt]] = {}
                for t in mios:
                    por_archivo.setdefault(t.ruta, []).append(t)
                for ruta, ts in sorted(por_archivo.items()):
                    destino = raiz / ruta
                    with open(destino, encoding="utf-8", newline="") as f:
                        original = f.read()
                    with open(destino, "w", encoding="utf-8", newline="") as f:
                        f.write(aplicar_tramos(original, ts))
                    print(f"  arreglado {ruta}: {len(ts)} tramos")
                # Las líneas del archivo se movieron: se recalcula el diff.
                cambios = lineas_cambiadas(raiz, a.base)
                tramos, errores_fmt = medir_fmt(raiz, a.cargo)
                mios = [t for t in tramos if tramo_toca(t, cambios)]
            for t in mios:
                hallazgos.append(Hallazgo(t.ruta, t.inicio, max(t.inicio, t.fin), "rustfmt", "rustfmt reformatearía este tramo; corre la puerta con --solo fmt --arreglar"))
            hallazgos += [e for e in errores_fmt if toca(cambios.get(e.ruta, set()), e.inicio, e.fin)]
        if a.solo in (None, "clippy"):
            diags, errores = medir_clippy(raiz, a.cargo, a.cargo_extra)
            mios_c = [d for d in diags if toca(cambios.get(d.ruta, set()), d.inicio, d.fin)]
            print(f"clippy: deuda total {len(diags)} diagnósticos en {len({d.ruta for d in diags})} archivos; en líneas cambiadas {len(mios_c)}; errores de build {len(errores)}")
            hallazgos += errores + mios_c
    except PuertaRota as e:
        print(f"::error title=puerta rota::{e}")
        return 2
    for h in hallazgos:
        print(anotar(h))
    if hallazgos:
        print(f"{len(hallazgos)} hallazgos en líneas cambiadas o build roto.")
        return 1
    hechas = {None: "fmt y clippy", "fmt": "fmt", "clippy": "clippy"}[a.solo]
    print(f"Limpio: las líneas cambiadas pasan {hechas}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
