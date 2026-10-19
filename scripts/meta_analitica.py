#!/usr/bin/env python3
"""RB-A1a (Codex RB, 2026-10-19) — META-ANALÍTICA DEL IMPACTO HISTÓRICO DE CADA CAMBIO.

Mandato del operador: medir el impacto histórico de cada cambio en lo
matemático, el cómputo y lo financiero. Convierte la historia de primer padre
de una rama (main) en un libro de impacto por commit y un resumen por agente:

- cambio: átomos citados (familias en docs/audit/meta_analitica/
  familias_atomo.tsv), agente inferido, líneas ± por clase (producción, tests,
  build, docs, scripts, CI, otros) y Δ de funciones de test (`#[test]`,
  `#[tokio::test]`, …) en el diff contra el primer padre;
- cómputo: clausura de dependencias de god-engine-core (núcleo) y del host
  (trader-gemini-v5) y paquetes del Cargo.lock de cada commit; coste de build
  por agente (espera y compilación) desde logs/semaforo.log;
- salud del código: sitios de pánico de producción del trinquete RB-P1
  (docs/audit/trinquete_panicos_produccion.tsv) desde que existe;
- financiero: cobertura del oráculo T-1 (k/144) declarada en el commit.

Límites conocidos (deliberados): las pruebas unitarias dentro de src/ cuentan
como producción en las líneas (no así en el Δ de tests, que mira los
atributos); el T-1 sólo aparece si el commit lo declara; el coste de build es
el del log local del semáforo (por agente y día: el log aún no registra el
commit compilado; lo hará RB-A1b).

Uso:
  python scripts/meta_analitica.py [--rama origin/main] [--desde AAAA-MM-DD]
      [--salida docs/audit/meta_analitica] [--semaforo <raíz>/logs/semaforo.log]
      [--familias docs/audit/meta_analitica/familias_atomo.tsv] [--repo .]
Salida: libro_impacto.tsv y resumen.md, deterministas para una misma historia
y un mismo log. Código de salida: 0 hecho; 2 repositorio o rama inválidos.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import re
import subprocess
import sys
from pathlib import Path

SEP_REG = "\x1e"
SEP_CAMPO = "\x1f"
LOCK = "Cargo.lock"
TRINQUETE = "docs/audit/trinquete_panicos_produccion.tsv"
FAMILIAS = "docs/audit/meta_analitica/familias_atomo.tsv"
RAIZ_NUCLEO = "god-engine-core"
RAIZ_HOST = "trader-gemini-v5"
CLASES = ("prod", "tests", "build", "docs", "scripts", "ci", "otros")
# `T-1 18/144`, `oráculo T-1: PASA 16/144`, `T-1 re-certificado (18/144)`.
RE_T1 = re.compile(r"T-1\b[^\n]{0,80}?\b(\d{1,3})\s*/\s*144\b")
# Atributo de función de test: #[test], #[tokio::test], #[tokio::test(...)].
RE_TEST = re.compile(r"^#\[(?:[A-Za-z_]\w*::)?test(?:\]|\()")
RE_SEMAFORO = re.compile(r"^(\S+)\t([^\t]+)\t(ADQUIERE|LIBERA)\b(.*)$")

COLUMNAS = (
    ["sha", "fecha", "agente", "atomos", "archivos"]
    + [f"{c}_{s}" for c in CLASES for s in ("mas", "menos")]
    + ["delta_tests", "clausura_nucleo", "clausura_host", "paquetes_lock",
       "panicos_prod", "t1_k", "asunto"]
)


class ErrorGit(RuntimeError):
    pass


# Umbral de «texto» para los diffs (core.bigFileThreshold): por encima, git
# trata el archivo como binario y el libro cuenta 0 líneas. Medido el 19-10:
# el texto versionado más grande es el buzón (1,1 MB; MEMORIA 0,9 MB, núcleo
# 0,6 MB); los blobs mayores son datos (modelos, artefactos). Con 2 MB el libro
# de 1.075 commits tarda ~60 s en lugar de 270 s, sin perder ningún texto.
UMBRAL_TEXTO = "2m"


def git_bytes(repo: Path, *args: str, entrada: bytes | None = None) -> bytes:
    r = subprocess.run(["git", "-C", str(repo), "-c", "core.quotepath=off",
                        "-c", "core.bigFileThreshold=" + UMBRAL_TEXTO, *args],
                       input=entrada, capture_output=True)
    if r.returncode != 0:
        raise ErrorGit(f"git {' '.join(args[:3])}…: {r.stderr.decode('utf-8', 'replace').strip()}")
    return r.stdout


def git(repo: Path, *args: str) -> str:
    return git_bytes(repo, *args).decode("utf-8", "replace")


# ---------------------------------------------------------------- historia

def leer_commits(repo: Path, rama: str, desde: str | None) -> list[dict]:
    # %x1e / %x1f: los separadores salen de git, no viajan crudos en argv.
    formato = "%x1e" + "%x1f".join(["%H", "%aI", "%an", "%s", "%b"])
    args = ["log", "--first-parent", "--reverse", "--format=" + formato]
    if desde:
        args.append("--since=" + desde)
    salida = git(repo, *args, rama)
    commits = []
    for registro in salida.split(SEP_REG)[1:]:
        campos = registro.split(SEP_CAMPO)
        if len(campos) < 5:
            continue
        commits.append({
            "sha": campos[0].strip(),
            "fecha": campos[1].strip(),
            "autor": campos[2].strip(),
            "asunto": campos[3].strip(),
            "cuerpo": SEP_CAMPO.join(campos[4:]).strip(),
        })
    return commits


def _diff_args(desde: str | None) -> list[str]:
    args = ["log", "--first-parent", "--diff-merges=first-parent", "--no-renames"]
    if desde:
        args.append("--since=" + desde)
    return args


def leer_numstat(repo: Path, rama: str, desde: str | None) -> dict[str, list[tuple[int, int, str]]]:
    salida = git(repo, *_diff_args(desde), "--numstat", "--format=%x1e%H", rama)
    por_sha: dict[str, list[tuple[int, int, str]]] = {}
    for registro in salida.split(SEP_REG)[1:]:
        lineas = registro.splitlines()
        if not lineas:
            continue
        filas = []
        for linea in lineas[1:]:
            partes = linea.split("\t")
            if len(partes) != 3:
                continue
            mas, menos, ruta = partes
            # Binarios: «-\t-\truta».
            filas.append((int(mas) if mas.isdigit() else 0, int(menos) if menos.isdigit() else 0, ruta))
        por_sha[lineas[0].strip()] = filas
    return por_sha


def leer_delta_tests(repo: Path, rama: str, desde: str | None) -> dict[str, int]:
    """Funciones de test añadidas menos quitadas por commit (diff contra el
    primer padre, sólo .rs). `-G` limita la salida a los archivos tocados por
    un atributo de test; el conteo exacto lo hace RE_TEST."""
    salida = git(repo, *_diff_args(desde), "-U0", "-p", r"-G#\[.*test",
                 "--format=%x1e%H", rama, "--", "*.rs")
    delta: dict[str, int] = {}
    for registro in salida.split(SEP_REG)[1:]:
        lineas = registro.split("\n")
        d = 0
        for linea in lineas[1:]:
            if linea.startswith(("+++", "---")):
                continue
            if linea[:1] in "+-" and RE_TEST.match(linea[1:].strip()):
                d += 1 if linea[0] == "+" else -1
        delta[lineas[0].strip()] = d
    return delta


def clase(ruta: str) -> str:
    r = ruta.replace("\\", "/")
    nombre = r.rsplit("/", 1)[-1]
    if r.startswith(".github/"):
        return "ci"
    if r.startswith("scripts/"):
        return "scripts"
    if r.endswith(".md") or r.startswith(("docs/", ".agents/")):
        return "docs"
    if nombre in ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml") or r.startswith(".cargo/"):
        return "build"
    if r.endswith(".rs"):
        directorios = r.split("/")[:-1]
        return "tests" if ("tests" in directorios or "benches" in directorios) else "prod"
    return "otros"


# ---------------------------------------------------------------- blobs por commit

def blob_por_commit(repo: Path, shas: list[str], ruta: str) -> dict[str, str | None]:
    if not shas:
        return {}
    entrada = "".join(f"{s}:{ruta}\n" for s in shas).encode("utf-8")
    salida = git_bytes(repo, "cat-file", "--batch-check", entrada=entrada).decode("utf-8", "replace")
    res: dict[str, str | None] = {}
    for sha, linea in zip(shas, salida.splitlines()):
        partes = linea.split()
        res[sha] = partes[0] if len(partes) == 3 and partes[1] == "blob" else None
    return res


def leer_blobs(repo: Path, blobs) -> dict[str, str]:
    unicos = sorted({b for b in blobs if b})
    if not unicos:
        return {}
    datos = git_bytes(repo, "cat-file", "--batch", entrada="".join(b + "\n" for b in unicos).encode())
    res: dict[str, str] = {}
    i = 0
    for b in unicos:
        fin = datos.index(b"\n", i)
        cabecera = datos[i:fin].decode("ascii", "replace").split()
        tam = int(cabecera[2])
        res[b] = datos[fin + 1:fin + 1 + tam].decode("utf-8", "replace")
        i = fin + 1 + tam + 1
    return res


# ---------------------------------------------------------------- métricas

def grafo_lock(texto: str) -> dict[str, set[str]]:
    """nombre de paquete → nombres de sus dependencias (versiones fusionadas)."""
    grafo: dict[str, set[str]] = collections.defaultdict(set)
    for bloque in texto.split("[[package]]")[1:]:
        nombre = None
        deps: set[str] = set()
        dentro = False
        for linea in bloque.splitlines():
            l = linea.strip()
            if dentro:
                if l.startswith("]"):
                    dentro = False
                    continue
                entrada = l.rstrip(",").strip('"')
                if entrada:
                    deps.add(entrada.split()[0])
                continue
            if l.startswith("name = ") and nombre is None:
                nombre = l[len("name = "):].strip('"')
            elif l.startswith("dependencies = ["):
                if l.endswith("]"):
                    for parte in l[l.index("[") + 1:-1].split(","):
                        entrada = parte.strip().strip('"')
                        if entrada:
                            deps.add(entrada.split()[0])
                else:
                    dentro = True
        if nombre:
            grafo[nombre] |= deps
    return grafo


def clausura(grafo: dict[str, set[str]], raiz: str) -> int | None:
    """Paquetes alcanzables desde `raiz`, ella incluida; None si no está."""
    if raiz not in grafo:
        return None
    vistos = {raiz}
    pila = [raiz]
    while pila:
        for d in grafo.get(pila.pop(), ()):
            if d not in vistos:
                vistos.add(d)
                pila.append(d)
    return len(vistos)


def total_trinquete(texto: str) -> int:
    total = 0
    for linea in texto.splitlines():
        if not linea.strip() or linea.startswith(("#", "paquete\t")):
            continue
        campos = linea.split("\t")
        if len(campos) >= 6:
            total += sum(int(x) for x in campos[2:6] if x.strip().isdigit())
    return total


def leer_familias(ruta: Path) -> list[tuple[str, re.Pattern, str]]:
    familias = []
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        if not linea.strip() or linea.startswith("#"):
            continue
        campos = linea.split("\t")
        if len(campos) < 2 or campos[0] == "familia":
            continue
        familias.append((campos[0], re.compile(campos[1]), campos[2].strip() if len(campos) > 2 else ""))
    return familias


def atomos_y_agente(commit: dict, familias) -> tuple[list[str], str]:
    def buscar(texto: str) -> tuple[list[str], str]:
        vistos: list[str] = []
        agente = ""
        for _, patron, ag in familias:
            for m in patron.finditer(texto):
                if m.group(0) not in vistos:
                    vistos.append(m.group(0))
                if ag and not agente:
                    agente = ag
        return vistos, agente

    atomos, agente = buscar(commit["asunto"])
    primer_parrafo = commit["cuerpo"].split("\n\n", 1)[0]
    if not atomos:
        atomos, _ = buscar(primer_parrafo)
    if not agente:
        _, agente = buscar(primer_parrafo)
    return atomos, agente or commit["autor"]


def t1_declarado(commit: dict) -> int | None:
    m = RE_T1.search(commit["asunto"] + "\n" + commit["cuerpo"])
    return int(m.group(1)) if m and 0 <= int(m.group(1)) <= 144 else None


# ---------------------------------------------------------------- semáforo

def agente_semaforo(etiqueta: str) -> str:
    """«Claude-LineaG-lg27» → «Claude-LineaG»: quita los sufijos de tarea en
    minúsculas que algunos agentes añaden a SEMAFORO_AGENTE."""
    return re.sub(r"(?:-[a-z0-9]+)+$", "", etiqueta.strip()) or etiqueta.strip()


def leer_semaforo(ruta: Path | None) -> tuple[list[dict], int]:
    """Builds cerrados (espera y compilación en s) y adquisiciones sin
    liberar (builds matados). `duracion` del semáforo cuenta desde que el
    agente pidió turno: build = LIBERA − ADQUIERE; espera = duracion − build."""
    if ruta is None or not ruta.is_file():
        return [], 0
    abiertos: dict[tuple[str, str], list[dt.datetime]] = collections.defaultdict(list)
    builds: list[dict] = []
    for linea in ruta.read_text(encoding="utf-8", errors="replace").splitlines():
        m = RE_SEMAFORO.match(linea)
        if not m:
            continue
        try:
            ts = dt.datetime.fromisoformat(m.group(1).replace("Z", "+00:00"))
        except ValueError:
            continue
        agente, tipo, resto = agente_semaforo(m.group(2)), m.group(3), m.group(4)
        comando = resto.split(": ", 1)[1].strip() if ": " in resto else ""
        if tipo == "ADQUIERE":
            abiertos[(agente, comando)].append(ts)
            continue
        md = re.search(r"duracion=(\d+)s", resto)
        mc = re.search(r"codigo=(-?\d+)", resto)
        pila = abiertos.get((agente, comando))
        t0 = pila.pop() if pila else None
        build = (ts - t0).total_seconds() if t0 else None
        duracion = int(md.group(1)) if md else None
        espera = max(0.0, duracion - build) if (duracion is not None and build is not None) else None
        builds.append({"fin": ts, "agente": agente, "comando": comando,
                       "codigo": int(mc.group(1)) if mc else None,
                       "build_s": build, "espera_s": espera})
    sin_liberar = sum(len(v) for v in abiertos.values())
    return builds, sin_liberar


# ---------------------------------------------------------------- libro

def construir_libro(repo: Path, rama: str, desde: str | None, familias) -> list[dict]:
    commits = leer_commits(repo, rama, desde)
    shas = [c["sha"] for c in commits]
    numstat = leer_numstat(repo, rama, desde)
    delta_tests = leer_delta_tests(repo, rama, desde)
    blobs_lock = blob_por_commit(repo, shas, LOCK)
    blobs_trinquete = blob_por_commit(repo, shas, TRINQUETE)
    textos = leer_blobs(repo, list(blobs_lock.values()) + list(blobs_trinquete.values()))
    metricas_lock: dict[str, tuple] = {}
    for b in {b for b in blobs_lock.values() if b}:
        g = grafo_lock(textos[b])
        metricas_lock[b] = (clausura(g, RAIZ_NUCLEO), clausura(g, RAIZ_HOST), len(g))
    filas = []
    for c in commits:
        atomos, agente = atomos_y_agente(c, familias)
        fila: dict = {"sha": c["sha"][:12], "fecha": c["fecha"], "agente": agente,
                      "atomos": ",".join(atomos)}
        archivos = numstat.get(c["sha"], [])
        fila["archivos"] = len(archivos)
        for k in CLASES:
            fila[f"{k}_mas"] = 0
            fila[f"{k}_menos"] = 0
        for mas, menos, ruta in archivos:
            k = clase(ruta)
            fila[f"{k}_mas"] += mas
            fila[f"{k}_menos"] += menos
        fila["delta_tests"] = delta_tests.get(c["sha"], 0)
        nucleo, host, paquetes = metricas_lock.get(blobs_lock.get(c["sha"]) or "", (None, None, None))
        fila["clausura_nucleo"], fila["clausura_host"], fila["paquetes_lock"] = nucleo, host, paquetes
        bt = blobs_trinquete.get(c["sha"])
        fila["panicos_prod"] = total_trinquete(textos[bt]) if bt else None
        fila["t1_k"] = t1_declarado(c)
        fila["asunto"] = c["asunto"].replace("\t", " ")[:160]
        filas.append(fila)
    return filas


def _celda(v) -> str:
    return "" if v is None else str(v)


def escribir_tsv(filas: list[dict], ruta: Path) -> None:
    lineas = ["\t".join(COLUMNAS)]
    for f in filas:
        lineas.append("\t".join(_celda(f[c]) for c in COLUMNAS))
    ruta.write_text("\n".join(lineas) + "\n", encoding="utf-8", newline="\n")


# ---------------------------------------------------------------- resumen

def _deltas(filas: list[dict], clave: str) -> list[tuple[int, dict]]:
    """(Δ respecto del commit anterior con valor, fila) de una métrica de nivel."""
    out = []
    previo = None
    for f in filas:
        v = f[clave]
        if v is None:
            continue
        if previo is not None and v != previo:
            out.append((v - previo, f))
        previo = v
    return out


def resumen(filas: list[dict], builds: list[dict], sin_liberar: int, rama: str, punta: str) -> str:
    md: list[str] = []
    md.append("# Meta-analítica del impacto por cambio (RB-A1a)")
    md.append("")
    if not filas:
        md.append("Sin commits en el rango pedido.")
        return "\n".join(md) + "\n"
    md.append(f"Rama `{rama}` (primer padre) en `{punta}`: {len(filas)} commits, del "
              f"{filas[0]['fecha'][:10]} al {filas[-1]['fecha'][:10]}. Libro completo en "
              "`libro_impacto.tsv`; regenerar con `python scripts/meta_analitica.py`.")
    md.append("")

    md.append("## 1. Cambio por agente")
    md.append("")
    md.append("| Agente | Commits | Prod + | Prod − | Tests + | Tests − | Δ funciones de test | Docs + |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    por_agente: dict[str, list[dict]] = collections.defaultdict(list)
    for f in filas:
        por_agente[f["agente"]].append(f)
    for agente, fs in sorted(por_agente.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        s = lambda k: sum(f[k] for f in fs)
        md.append(f"| {agente} | {len(fs)} | {s('prod_mas')} | {s('prod_menos')} | {s('tests_mas')} | "
                  f"{s('tests_menos')} | {s('delta_tests'):+d} | {s('docs_mas')} |")
    md.append("")

    md.append("## 2. Cómputo")
    md.append("")
    ultimo = next((f for f in reversed(filas) if f["clausura_nucleo"] is not None), None)
    primero = next((f for f in filas if f["clausura_nucleo"] is not None), None)
    if ultimo and primero:
        md.append(f"Clausura de dependencias de `{RAIZ_NUCLEO}` en `Cargo.lock`: "
                  f"{primero['clausura_nucleo']} → {ultimo['clausura_nucleo']} paquetes "
                  f"(host `{RAIZ_HOST}`: {_celda(primero['clausura_host'])} → {_celda(ultimo['clausura_host'])}; "
                  f"lock: {_celda(primero['paquetes_lock'])} → {_celda(ultimo['paquetes_lock'])}).")
        md.append("")
        cambios = sorted(_deltas(filas, "clausura_nucleo"), key=lambda t: (-abs(t[0]), t[1]["sha"]))[:10]
        if cambios:
            md.append("| Δ clausura núcleo | Commit | Agente | Asunto |")
            md.append("|---:|---|---|---|")
            for d, f in cambios:
                md.append(f"| {d:+d} | `{f['sha']}` | {f['agente']} | {f['asunto'][:90]} |")
            md.append("")
    if builds:
        md.append(f"Builds del semáforo (log local): {len(builds)} cerrados, {sin_liberar} sin liberar "
                  "(matados o en curso).")
        md.append("")
        md.append("| Agente | Builds | Compilación (h) | Espera (h) | Fallidos |")
        md.append("|---|---:|---:|---:|---:|")
        b_ag: dict[str, list[dict]] = collections.defaultdict(list)
        for b in builds:
            b_ag[b["agente"]].append(b)
        for agente, bs in sorted(b_ag.items(), key=lambda kv: kv[0]):
            comp = sum(b["build_s"] or 0 for b in bs) / 3600
            esp = sum(b["espera_s"] or 0 for b in bs) / 3600
            fallos = sum(1 for b in bs if b["codigo"] not in (0, None))
            md.append(f"| {agente} | {len(bs)} | {comp:.2f} | {esp:.2f} | {fallos} |")
        md.append("")

    md.append("## 3. Salud del código")
    md.append("")
    total_tests = sum(f["delta_tests"] for f in filas)
    md.append(f"Δ neto de funciones de test en el rango: {total_tests:+d}.")
    con_panicos = [f for f in filas if f["panicos_prod"] is not None]
    if con_panicos:
        md.append(f"Sitios de pánico de producción (trinquete RB-P1): {con_panicos[0]['panicos_prod']} "
                  f"→ {con_panicos[-1]['panicos_prod']} desde `{con_panicos[0]['sha']}`.")
    else:
        md.append("Trinquete de pánicos RB-P1: aún sin línea base en este rango.")
    md.append("")

    md.append("## 4. Financiero")
    md.append("")
    t1 = [f for f in filas if f["t1_k"] is not None]
    if t1:
        md.append("| Commit | Fecha | Agente | T-1 (genes sensibles / 144) |")
        md.append("|---|---|---|---:|")
        for f in t1[-15:]:
            md.append(f"| `{f['sha']}` | {f['fecha'][:10]} | {f['agente']} | {f['t1_k']} |")
    else:
        md.append("Ningún commit del rango declara una medición T-1.")
    md.append("")

    md.append("## 5. Mayores cambios de producción")
    md.append("")
    md.append("| Commit | Agente | Prod + | Prod − | Átomos | Asunto |")
    md.append("|---|---|---:|---:|---|---|")
    for f in sorted(filas, key=lambda f: (-(f["prod_mas"] + f["prod_menos"]), f["sha"]))[:10]:
        md.append(f"| `{f['sha']}` | {f['agente']} | {f['prod_mas']} | {f['prod_menos']} | "
                  f"{f['atomos'][:40]} | {f['asunto'][:70]} |")
    md.append("")
    return "\n".join(md) + "\n"


# ---------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--repo", default=".")
    p.add_argument("--rama")
    p.add_argument("--desde")
    p.add_argument("--salida")
    p.add_argument("--semaforo")
    p.add_argument("--familias")
    a = p.parse_args(argv)
    try:
        repo = Path(git(Path(a.repo), "rev-parse", "--show-toplevel").strip())
        comun = Path(git(repo, "rev-parse", "--path-format=absolute", "--git-common-dir").strip())
        rama = a.rama
        if not rama:
            try:
                git(repo, "rev-parse", "--verify", "--quiet", "origin/main")
                rama = "origin/main"
            except ErrorGit:
                rama = "HEAD"
        punta = git(repo, "rev-parse", "--short=12", rama).strip()
    except ErrorGit as e:
        print(f"[meta_analitica] {e}", file=sys.stderr)
        return 2
    familias = leer_familias(Path(a.familias) if a.familias else repo / FAMILIAS)
    salida = Path(a.salida) if a.salida else repo / "docs" / "audit" / "meta_analitica"
    semaforo = Path(a.semaforo) if a.semaforo else comun.parent / "logs" / "semaforo.log"
    filas = construir_libro(repo, rama, a.desde, familias)
    builds, sin_liberar = leer_semaforo(semaforo)
    salida.mkdir(parents=True, exist_ok=True)
    escribir_tsv(filas, salida / "libro_impacto.tsv")
    (salida / "resumen.md").write_text(resumen(filas, builds, sin_liberar, rama, punta),
                                        encoding="utf-8", newline="\n")
    print(f"[meta_analitica] {len(filas)} commits de {rama} ({punta}); {len(builds)} builds del semáforo "
          f"→ {salida / 'libro_impacto.tsv'} y resumen.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
