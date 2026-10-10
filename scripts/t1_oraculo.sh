#!/usr/bin/env bash
# QS-P2 — Oráculo T-1 reproducible y en paralelo.
#
# Uso, desde la raíz de cualquier worktree:
#   scripts/t1_oraculo.sh [hilos]
#
# Variables opcionales: CARGO_TARGET_DIR (por defecto <worktree>/target-portable),
# T1_LOG (por defecto <target>/t1_oraculo.log), T1_SOLO_SELLO=1 (sólo
# calcula el sello y lista lo que tocaría, sin compilar).
#
# 1) Sello del árbol. Cargo no distingue dos worktrees del mismo paquete: con
#    un CARGO_TARGET_DIR compartido reutiliza el binario compilado desde OTRO
#    árbol si los ficheros de éste son más viejos. El 2026-10-10 el T-1 «base»
#    corrió el binario de la rama C-22 sin un solo «Compiling».
#    El script guarda en el target el árbol git del ESTADO compilado
#    (ficheros seguidos con sus cambios sin commitear y ficheros nuevos no
#    ignorados). En la corrida siguiente toca SÓLO los ficheros que difieren
#    entre ese árbol y el estado actual (QS-P2b): cargo recompila los crates
#    afectados y sus dependientes, nada más. Sin sello válido, o si el
#    objeto del árbol anterior no existe, toca todos los .rs del worktree.
#    Las dependencias externas no se recompilan.
# 2) Paralelismo. T1_THREADS = hilos (por defecto todos los núcleos): el test
#    evalúa los 144 genes en paralelo (QS-P1). Cada corrida imprime una huella
#    por gen ([T1-STATS]); dos corridas se comparan con
#      diff <(grep -o 'gene=[^ ]* huella=[0-9a-f]*' A | sort) \
#           <(grep -o 'gene=[^ ]* huella=[0-9a-f]*' B | sort)
# 3) En Linux compila para x86_64-pc-windows-gnu y ejecuta bajo wine
#    (os-guardian es sólo-Windows). En Windows (Git Bash), sin target extra.
set -euo pipefail

raiz=$(git rev-parse --show-toplevel)
cd "$raiz"

hilos=${1:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo "${NUMBER_OF_PROCESSORS:-4}")}
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

# Árbol git del estado actual del worktree, sin tocar el índice real: copia
# del índice (conserva los datos de stat, así sólo se re-hashean los
# ficheros cambiados) + `git add -A`. Los target* y data/ están ignorados.
arbol_del_estado() {
  local tmp
  tmp=$(mktemp)
  cp "$(git rev-parse --git-path index)" "$tmp"
  GIT_INDEX_FILE=$tmp git add -A -- . >/dev/null 2>&1
  GIT_INDEX_FILE=$tmp git write-tree
  rm -f "$tmp"
}

mkdir -p "$CARGO_TARGET_DIR"
sello_archivo="$CARGO_TARGET_DIR/.t1-sello"
estado=$(arbol_del_estado)
sello="v2 $estado"
anterior=""
[ -f "$sello_archivo" ] && anterior=$(cat "$sello_archivo")

if [ "$anterior" = "$sello" ]; then
  echo "[t1_oraculo] el target ya está compilado desde este mismo estado"
  tocar=()
elif [[ "$anterior" == v2\ * ]] && git cat-file -e "${anterior#v2 }^{tree}" 2>/dev/null; then
  mapfile -t tocar < <(git diff --name-only "${anterior#v2 }" "$estado")
  echo "[t1_oraculo] cambiaron ${#tocar[@]} ficheros desde el último build de este target: se tocan sólo esos"
else
  echo "[t1_oraculo] sin sello válido en $CARGO_TARGET_DIR: se recompila el workspace"
  mapfile -t -d '' tocar < <(git ls-files --cached --others --exclude-standard -z -- '*.rs')
fi

if [ "${T1_SOLO_SELLO:-0}" = "1" ]; then
  echo "sello actual: $sello"
  echo "sello anterior: ${anterior:-<ninguno>}"
  printf '  %s\n' "${tocar[@]:0:20}"
  exit 0
fi

for f in "${tocar[@]}"; do
  [ -e "$f" ] && touch -- "$f"
done

log=${T1_LOG:-$CARGO_TARGET_DIR/t1_oraculo.log}
inicio=$(date +%s)
set +e
cargo test "${extra[@]}" --release -p backtest-engine --locked \
  --test t1_cobertura_genetica -- --exact t1_cobertura_genetica_del_oraculo_de_aptitud \
  --nocapture --test-threads=1 >"$log" 2>&1
rc=$?
set -e

# El sello sólo se escribe si el binario llegó a ejecutarse: entonces es de
# este estado.
if grep -q "Running tests/t1_cobertura_genetica" "$log"; then
  echo "$sello" >"$sello_archivo"
fi

echo "== T-1 $(git rev-parse --short HEAD) con $hilos hilos: rc=$rc en $(($(date +%s) - inicio)) s (log: $log)"
grep -E "Compiling (risk-engine|metacortex-engine|god-engine-core|backtest-engine)|evaluaciones en|COBERTURA|Sin cambio|test result|panicked" "$log" | tail -12 || true
exit "$rc"
