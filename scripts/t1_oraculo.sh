#!/usr/bin/env bash
# QS-P2 — Oráculo T-1 reproducible y en paralelo.
#
# Uso, desde la raíz de cualquier worktree:
#   scripts/t1_oraculo.sh [hilos]
#
# Variables opcionales: CARGO_TARGET_DIR (por defecto <worktree>/target-portable),
# T1_LOG (por defecto <target>/t1_oraculo.log).
#
# 1) Sello del árbol. Cargo no distingue dos worktrees del mismo paquete: con
#    un CARGO_TARGET_DIR compartido reutiliza el binario compilado desde OTRO
#    árbol si los .rs de éste son más viejos. El 2026-10-10 el T-1 «base»
#    corrió el binario de la rama C-22 sin un solo «Compiling». El script
#    guarda en el target un sello del árbol (tree de HEAD, diff sin commitear,
#    ficheros sin seguimiento y ruta). Si cambió desde el último build, toca
#    los .rs del worktree para que cargo recompile los crates del workspace.
#    Las dependencias externas no se recompilan.
# 2) Paralelismo. T1_THREADS = hilos (por defecto todos los núcleos): el test
#    evalúa los 144 genes en paralelo (QS-P1). Cada corrida imprime una huella
#    por gen ([T1-STATS]); dos corridas se comparan con
#      diff <(grep -o 'gene=[^ ]* huella=[0-9a-f]*' A | sort) \
#           <(grep -o 'gene=[^ ]* huella=[0-9a-f]*' B | sort)
# 3) En Linux compila para x86_64-pc-windows-gnu y ejecuta bajo wine
#    (os-guardian es sólo-Windows). En Windows, sin target extra.
set -euo pipefail

raiz=$(git rev-parse --show-toplevel)
cd "$raiz"

hilos=${1:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$raiz/target-portable}
export CARGO_INCREMENTAL=0 CARGO_PROFILE_RELEASE_DEBUG=0
export CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-$hilos}
export T1_THREADS=$hilos

extra=()
if [ "$(uname -s)" = "Linux" ]; then
  extra=(--target x86_64-pc-windows-gnu)
  # target-cpu=native emitió AVX512-FP16 en una VM sin esa extensión.
  export RUSTFLAGS=${RUSTFLAGS:-"-C target-cpu=x86-64-v2 -C opt-level=3"}
  export CARGO_TARGET_X86_64_PC_WINDOWS_GNU_RUNNER=${CARGO_TARGET_X86_64_PC_WINDOWS_GNU_RUNNER:-/usr/lib/wine/wine64}
  export WINEDEBUG=${WINEDEBUG:--all}
fi

mkdir -p "$CARGO_TARGET_DIR"
sello_archivo="$CARGO_TARGET_DIR/.t1-sello"
huella_local=$(
  {
    git diff HEAD
    git ls-files --others --exclude-standard -z | xargs -0 -r sha256sum
  } | sha256sum | cut -c1-16
)
sello="$(git rev-parse 'HEAD^{tree}') $huella_local $raiz"
if [ ! -f "$sello_archivo" ] || [ "$(cat "$sello_archivo")" != "$sello" ]; then
  echo "[t1_oraculo] el último build de $CARGO_TARGET_DIR es de otro árbol: se recompila el workspace"
  git ls-files --cached --others --exclude-standard -z -- '*.rs' | xargs -0 -r touch
fi

log=${T1_LOG:-$CARGO_TARGET_DIR/t1_oraculo.log}
inicio=$(date +%s)
set +e
cargo test "${extra[@]}" --release -p backtest-engine --locked \
  --test t1_cobertura_genetica -- --exact t1_cobertura_genetica_del_oraculo_de_aptitud \
  --nocapture --test-threads=1 >"$log" 2>&1
rc=$?
set -e

# El sello sólo se escribe si el binario llegó a ejecutarse: entonces es de
# este árbol.
if grep -q "Running tests/t1_cobertura_genetica" "$log"; then
  echo "$sello" >"$sello_archivo"
fi

echo "== T-1 $(git rev-parse --short HEAD) con $hilos hilos: rc=$rc en $(($(date +%s) - inicio)) s (log: $log)"
grep -E "Compiling (risk-engine|metacortex-engine|god-engine-core|backtest-engine)|evaluaciones en|COBERTURA|Sin cambio|test result|panicked" "$log" | tail -12 || true
exit "$rc"
