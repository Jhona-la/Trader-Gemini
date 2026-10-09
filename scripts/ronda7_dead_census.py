#!/usr/bin/env python3
"""Censo R7-R0: codigo muerto y duplicado del arbol versionado.

Metodo determinista y barato (sin compilar). Cada fila lleva su evidencia
medida (conteo de apariciones) para que un humano u otra sesion pueda
re-verificarla con un grep antes de actuar.

Scaners:
  1. anotaciones #[allow(dead_code)] con el item que marcan
  2. modulos .rs huerfanos (sin declaracion `mod <nombre>;` alcanzable),
     tratando `X/mod.rs` como el modulo `X` declarado por el padre de `X`
  3. funciones pub cuya unica aparicion en todo el arbol versionado es su
     propia definicion
  4. dependencias de Cargo.toml cuyo identificador no aparece en NINGUN
     .rs del crate (src/, tests/, benches/, examples/, build.rs)
  5. pares de modulos homonimos en crates distintos (sospecha de fisica
     duplicada sin reconciliar)
  6. archivos .rs byte-identicos entre si (duplicacion literal)
  7. ficheros en src/bin/ que ningun [[bin]] del manifest declara por `name`
     ni por `path` (con autobins=false nunca se compilan), y [[bin]] cuyo
     `path` apunta a un archivo inexistente (manifest muerto)

Escribe TSV (una fila por hallazgo) e imprime conteos. No modifica codigo.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tomllib
from collections import Counter, defaultdict

ALLOW_RE = re.compile(r"#!?\[[^\]]*allow\(([^)]*dead_code[^)]*)\)\]")
MOD_DECL_RE = re.compile(r"^\s*(?:pub\s+)?mod\s+([a-zA-Z0-9_]+)\s*;", re.M)
FN_DEF_RE = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:const\s+|async\s+)?fn\s+([a-zA-Z0-9_]+)", re.M)
IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
SOURCE_SUBDIRS = ("src", "tests", "benches", "examples")


def tracked_files(repo, suffix):
    out = subprocess.check_output(
        ["git", "ls-files", "--", f"*{suffix}"], cwd=repo, text=True, encoding="utf-8"
    )
    return [p.replace("\\", "/") for p in out.splitlines() if p]


def read(repo, path):
    try:
        with open(os.path.join(repo, path), encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


def read_bytes(repo, path):
    try:
        with open(os.path.join(repo, path), "rb") as fh:
            return fh.read()
    except OSError:
        return b""


def allow_dead_code(blobs, rows):
    for path, text in sorted(blobs.items()):
        if "dead_code" not in text:
            continue
        lines = text.splitlines()
        for idx, line in enumerate(lines):
            if not ALLOW_RE.search(line):
                continue
            item = ""
            for look in lines[idx + 1 : idx + 6]:
                stripped = look.strip()
                if not stripped or stripped.startswith("//") or stripped.startswith("#"):
                    continue
                match = re.match(r"(?:pub\s+)?(?:const\s+|static\s+|async\s+|unsafe\s+)*"
                                 r"(?:extern\s+)?(?:fn|struct|enum|trait|impl|mod|type|use)\s+([A-Za-z0-9_]+)",
                                 stripped)
                item = match.group(1) if match else stripped[:60]
                break
            rows.append(("allow_dead_code", path, idx + 1, item, "-", "anotacion explicita del autor"))


def module_position(path):
    """(dirs_rel_dentro_de_src, stem, es_mod_rs) del archivo; None si no esta en src/."""
    parts = path.split("/")
    if "src" not in parts:
        return None
    src_i = len(parts) - 1 - parts[::-1].index("src")
    rel = parts[src_i + 1 :]
    if not rel or rel[0] == "bin":
        return None
    name = rel[-1]
    if name == "mod.rs":
        if len(rel) < 2:
            return None
        return tuple(rel[:-2]), rel[-2], True
    return tuple(rel[:-1]), name[:-3], False


PATH_ATTR_RE = re.compile(r'#\[\s*path\s*=\s*"([^"]+)"\s*\]')
MOD_NAME_RE = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+([a-zA-Z0-9_]+)\s*;", re.M)


def path_includes(blobs):
    """resuelta_ruta -> [(archivo_que_include, nombre_del_mod, va_por_cfg_test)].

    `#[path = "x.rs"] mod y;` compila `x.rs` bajo el nombre `y`, que NO
    coincide con el stem del archivo: sin esto el censo declara huerfano un
    modulo que si se compila (falso positivo medido en booktick_replay.rs).
    """
    out = defaultdict(list)
    for path, text in blobs.items():
        for match in PATH_ATTR_RE.finditer(text):
            tail = text[match.end() : match.end() + 400]
            name_match = MOD_NAME_RE.search(tail)
            if not name_match:
                continue
            head = text[max(0, match.start() - 200) : match.start()]
            cfg_test = head.rstrip().endswith("#[cfg(test)]")
            resolved = os.path.normpath(
                os.path.join(os.path.dirname(path), match.group(1))
            ).replace("\\", "/")
            out[resolved].append((path, name_match.group(1), cfg_test))
    return out


def orphan_modules(blobs, rows):
    decls = {p: set(MOD_DECL_RE.findall(t)) for p, t in blobs.items()}
    incluidos = path_includes(blobs)
    positions = {}
    for path in blobs:
        pos = module_position(path)
        if pos:
            positions[path] = pos
    for path, (dirs, stem, is_mod) in sorted(positions.items()):
        if stem in ("lib", "main"):
            continue
        parent_dir = "/".join(path.split("/")[:-1])
        if is_mod:
            parent_dir = "/".join(parent_dir.split("/")[:-1])
        candidates = []
        if dirs:
            candidates.append(f"{parent_dir}/{'/'.join(dirs)}.rs")
            candidates.append(f"{parent_dir}/{'/'.join(dirs)}/mod.rs")
        else:
            candidates.append(f"{parent_dir}/lib.rs")
            candidates.append(f"{parent_dir}/main.rs")
            candidates.append(f"{parent_dir}/mod.rs")
        for cand in candidates:
            if stem in decls.get(cand, ()):
                break
        else:
            if any(stem in names for names in decls.values()):
                continue  # declarado desde otro padre: ambiguo, no huerfano
            if path in incluidos:
                for sitio, nombre, cfg_test in sorted(set(incluidos[path])):
                    gate = "solo bajo #[cfg(test)]" if cfg_test else "tambien en produccion"
                    rows.append(("modulo_por_path", path, 0, nombre, 0,
                                 "%s.rs se compila como `mod %s;` desde %s (%s)"
                                 % (stem, nombre, sitio, gate)))
                continue
            rows.append(("modulo_huerfano", path, 0, stem, 0,
                         "ningun `mod %s;` declarable lo alcanza" % stem))


def unused_public_fns(blobs, token_counts, rows):
    for path, text in sorted(blobs.items()):
        for match in FN_DEF_RE.finditer(text):
            name = match.group(1)
            if len(name) < 4 or name in ("main", "new", "default", "clone", "drop"):
                continue
            contexto = text[max(0, match.start() - 120) : match.start()]
            if "#[test]" in contexto or "#[tokio::test]" in contexto or "#[ignore]" in contexto:
                continue
            if name.startswith("test_") or name.endswith("_test"):
                continue
            veces = token_counts[name]
            if veces <= 1:
                line = text[: match.start()].count("\n") + 1
                rows.append(("fn_sin_uso", path, line, name, veces,
                             "unica aparicion en el arbol = su definicion"))


def crate_sources(repo, crate_dir):
    paths = []
    for sub in SOURCE_SUBDIRS:
        d = os.path.join(repo, crate_dir, sub) if crate_dir else os.path.join(repo, sub)
        if os.path.isdir(d):
            for root, _dirs, files in os.walk(d):
                for fname in files:
                    if fname.endswith(".rs"):
                        rel = os.path.relpath(os.path.join(root, fname), repo).replace("\\", "/")
                        paths.append(rel)
    build_rs = os.path.join(repo, crate_dir, "build.rs") if crate_dir else os.path.join(repo, "build.rs")
    if os.path.exists(build_rs):
        rel = os.path.relpath(build_rs, repo).replace("\\", "/")
        paths.append(rel)
    return paths


def dep_tables(data):
    out = []
    for key in ("dependencies", "dev-dependencies", "build-dependencies"):
        for name in sorted(data.get(key, {}) or {}):
            out.append((name, key))
    for _target, tables in sorted((data.get("target") or {}).items()):
        for key in ("dependencies", "dev-dependencies", "build-dependencies"):
            for name in sorted(tables.get(key, {}) or {}):
                out.append((name, f"target.{key}"))
    return out


def unused_dependencies(repo, rows):
    for manifest in tracked_files(repo, "Cargo.toml"):
        crate_dir = os.path.dirname(manifest)
        try:
            with open(os.path.join(repo, manifest), "rb") as fh:
                data = tomllib.load(fh)
        except (OSError, ValueError):
            continue
        sources = [p for p in crate_sources(repo, crate_dir) if p in git_tracked]
        if not sources:
            continue
        tokens = Counter()
        for p in sources:
            tokens.update(IDENT_RE.findall(read(repo, p)))
        for dep, tabla in dep_tables(data):
            if tabla.startswith("dev-") or tabla.endswith("build-dependencies"):
                continue  # no afectan al artefacto compilado del crate
            alias = dep.replace("-", "_")
            veces = tokens[alias]
            if veces == 0:
                rows.append(("dep_sin_uso", manifest, 0, dep, veces,
                             "0 apariciones del identificador en los .rs del crate (verificar derive/macro)"))


def homonym_modules(blobs, rows):
    """Modulos .rs con el mismo nombre de archivo en crates distintos.

    No prueba duplicacion: homonimos como `orchestrator.rs` pueden ser
    dominios legitimos distintos. Es la lista de candidatos a revisar por
    fisica duplicada sin reconciliar, el patron que R6-A11 encontro en el
    doble Hodge (risk-engine/hodge.rs vs feature-engine/hodge_flow.rs).
    """
    por_stem = defaultdict(set)
    for path in blobs:
        crate = path.split("/")[1] if path.startswith("crates/") else "(raiz)"
        stem = os.path.basename(path)[:-3]
        if stem in ("lib", "main", "mod"):
            continue
        por_stem[stem].add((crate, path))
    for stem, where in sorted(por_stem.items()):
        crates = {c for c, _ in where}
        if len(crates) > 1:
            rutas = ",".join(sorted(p for _, p in where))
            for _crate, path in sorted(where):
                rows.append(("modulo_homonomo", path, 0, stem, len(crates),
                             "otro crate tiene %s.rs con el mismo nombre: %s" % (stem, rutas)))


def exact_duplicates(blobs, rows):
    by_hash = defaultdict(list)
    for path, text in blobs.items():
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        by_hash[digest].append(path)
    for digest, paths in sorted(by_hash.items()):
        if len(paths) > 1:
            for p in paths:
                rows.append(("archivo_igual", p, 0, digest, len(paths),
                             "contenido identico a: " + ",".join(x for x in paths if x != p)))


def undeclared_bins(repo, rows):
    for manifest in tracked_files(repo, "Cargo.toml"):
        crate_dir = os.path.dirname(manifest)
        try:
            with open(os.path.join(repo, manifest), "rb") as fh:
                data = tomllib.load(fh)
        except (OSError, ValueError):
            continue
        bins = data.get("bin") or []
        # Un [[bin]] declara el archivo por `path`, no por `name`: `name =
        # "quantum_benchmark"` con `path = "src/bin/benchmark.rs"` compila y
        # comparar solo contra el stem producia falsos positivos. En Windows
        # normpath invierte las barras: se normaliza SIEMPRE a "/" antes de
        # comparar contra `git ls-files`.
        declared = {b.get("name") for b in bins}

        def declared_path(raw):
            joined = os.path.join(crate_dir, raw.replace("\\", "/"))
            return os.path.normpath(joined).replace("\\", "/")

        declared_paths = {declared_path(b["path"]) for b in bins if b.get("path")}
        for b in bins:
            raw = b.get("path")
            if not raw:
                continue
            norm = declared_path(raw)
            if norm not in git_tracked:
                rows.append(("bin_ruta_inexistente", manifest, 0, b.get("name") or "", 0,
                             "[[bin]] apunta a %s que no esta versionado" % norm))
        bin_dir = os.path.join(repo, crate_dir, "src", "bin")
        if not os.path.isdir(bin_dir):
            continue
        autobins = (data.get("package") or {}).get("autobins", True)
        for fname in sorted(os.listdir(bin_dir)):
            if not fname.endswith(".rs"):
                continue
            stem = fname[:-3]
            rel = os.path.relpath(os.path.join(bin_dir, fname), repo).replace("\\", "/")
            if rel not in git_tracked:
                continue
            if stem in declared or rel in declared_paths:
                continue
            if not autobins:
                rows.append(("bin_no_declarado", rel, 0, stem, 0,
                             "autobins=false y ningun [[bin]] (ni por name ni por path) lo declara: nunca se compila"))
            elif bins:
                rows.append(("bin_no_listado", rel, 0, stem, 0,
                             "autobins=true con [[bin]] explicitos: confirmar que cargo lo toma"))


git_tracked = set()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    global git_tracked
    repo = os.path.abspath(args.repo)
    git_tracked = set(tracked_files(repo, ".rs"))
    blobs = {p: read(repo, p) for p in git_tracked}
    token_counts = Counter()
    for text in blobs.values():
        token_counts.update(IDENT_RE.findall(text))
    rows = []
    allow_dead_code(blobs, rows)
    orphan_modules(blobs, rows)
    unused_public_fns(blobs, token_counts, rows)
    unused_dependencies(repo, rows)
    homonym_modules(blobs, rows)
    exact_duplicates(blobs, rows)
    undeclared_bins(repo, rows)
    header = "tipo\truta\tlinea\tsimbolo\tveces\tnota\n"
    with open(args.output, "w", encoding="utf-8", newline="") as fh:
        fh.write(header)
        for row in sorted(rows):
            fh.write("\t".join(str(v) for v in row) + "\n")
    summary = Counter(r[0] for r in rows)
    print(json.dumps({"output": args.output, "total": len(rows),
                      "por_tipo": dict(sorted(summary.items()))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
