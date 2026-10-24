#!/usr/bin/env bash
# QS-P2 — Oráculo T-1 reproducible y en paralelo.
#
# Uso, desde la raíz de cualquier worktree:
#   scripts/t1_oraculo.sh [hilos]
#
# CARGO_TARGET_DIR sólo admite <checkout principal>/target (también en worktrees).
# T1_LOG (por defecto <worktree>/logs/t1_oraculo.log), T1_SOLO_SELLO=1 (sólo
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
# 2) Paralelismo. [hilos] prevalece sobre T1_THREADS (por defecto 1).
#    CARGO_BUILD_JOBS es independiente (por defecto 1); sus overrides conservan
#    la sintaxis de Cargo. El test evalúa los 144 genes (QS-P1) e imprime huella
#    por gen ([T1-STATS]); dos corridas se comparan con
#      diff <(grep -o 'gene=[^ ]* huella=[0-9a-f]*' A | sort) \
#           <(grep -o 'gene=[^ ]* huella=[0-9a-f]*' B | sort)
# 3) En Linux compila para x86_64-pc-windows-gnu y ejecuta bajo wine
#    (os-guardian es sólo-Windows). En Windows (Git Bash), sin target extra.
set -euo pipefail

raiz=$(git rev-parse --show-toplevel)
cd "$raiz"

hilos=${1:-${T1_THREADS:-1}}
jobs=${CARGO_BUILD_JOBS:-1}
if [[ ! "$hilos" =~ ^[+]?[0-9]+$ || "$hilos" =~ ^[+]?0+$ ]]; then
  echo "[t1_oraculo] hilos debe ser un entero positivo" >&2
  exit 2
fi
# Cargo admite offsets de CPU negativos, enteros con signo y default.
if [[ "$jobs" != default ]]; then
  if [[ ! "$jobs" =~ ^[+-]?[0-9]+$ || "$jobs" =~ ^[+-]?0+$ ]]; then
    echo "[t1_oraculo] CARGO_BUILD_JOBS debe ser un entero no nulo o default" >&2
    exit 2
  fi
fi
if [[ ${RUSTFLAGS+x} ]]; then
  echo "[t1_oraculo] RUSTFLAGS de entorno no está permitido; usa la configuración canónica .cargo/config.toml" >&2
  exit 2
fi
# El hook publicado por Fable se conserva como alias del mismo semáforo.
# No se interpreta texto de shell ni se permite omitir el semáforo obligatorio.
if [[ -n "${T1_CARGO:-}" && "$T1_CARGO" != 'bash scripts/cargo_semaforo.sh' ]]; then
  echo "[t1_oraculo] T1_CARGO sólo admite bash scripts/cargo_semaforo.sh" >&2
  exit 2
fi
comun=$(git rev-parse --path-format=absolute --git-common-dir)
raiz_principal=$(cd -- "$(dirname -- "$comun")" && pwd -P)
target_compartido="$raiz_principal/target"
normalizar_ruta() {
  case "$OSTYPE" in
    msys* | cygwin*)
      if ! command -v cygpath >/dev/null 2>&1; then
        echo "[t1_oraculo] necesito cygpath para comparar el target compartido" >&2
        return 2
      fi
      # -l sólo expande nombres 8.3 si existe la ruta completa. El target
      # puede no existir: expandir el ancestro existente y reponer el sufijo.
      local ancestro cola="" nombre expandida
      ancestro=$(cygpath -au -- "$1") || return 2
      while [[ "$ancestro" != / && "$ancestro" == */ ]]; do
        ancestro=${ancestro%/}
      done
      while [[ ! -e "$ancestro" ]]; do
        if [[ "$ancestro" == / ]]; then
          echo "[t1_oraculo] no encuentro un ancestro existente de: $1" >&2
          return 2
        fi
        nombre=${ancestro##*/}
        cola="/$nombre$cola"
        ancestro=${ancestro%/*}
        ancestro=${ancestro:-/}
      done
      expandida=$(cygpath -aml -- "$ancestro") || return 2
      if [[ -n "$cola" ]]; then
        printf '%s%s\n' "${expandida%/}" "$cola"
      else
        printf '%s\n' "$expandida"
      fi
      ;;
    *)
      if ! command -v realpath >/dev/null 2>&1; then
        echo "[t1_oraculo] necesito realpath para comparar el target compartido" >&2
        return 2
      fi
      realpath -m -- "$1"
      ;;
  esac
}
# MSYS puede exponer la misma carpeta como /tmp/... y C:/...; se comparan
# ambas rutas en el mismo formato, sin crear directorios para resolverlas.
target_normalizado=$(normalizar_ruta "$target_compartido")
if [[ -n "${CARGO_TARGET_DIR:-}" ]]; then
  target_explicito=$(normalizar_ruta "$CARGO_TARGET_DIR")
  if [[ "$target_explicito" != "$target_normalizado" ]]; then
    echo "[t1_oraculo] CARGO_TARGET_DIR debe ser el target compartido: $target_compartido" >&2
    exit 2
  fi
fi
semaforo="$raiz/scripts/cargo_semaforo.sh"
if [[ ! -f "$semaforo" ]]; then
  echo "[t1_oraculo] no encuentro el semáforo obligatorio: $semaforo" >&2
  exit 2
fi
export CARGO_TARGET_DIR=$target_compartido SEMAFORO_RAIZ=$raiz_principal
export CARGO_INCREMENTAL=0 CARGO_PROFILE_RELEASE_DEBUG=0
export CARGO_BUILD_JOBS=$jobs
export T1_THREADS=$hilos

extra=()
if [ "$(uname -s)" = "Linux" ]; then
  extra=(--target x86_64-pc-windows-gnu)
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

# Limitación heredada: este sello y touch ocurren antes del turno del semáforo;
# su escritura ocurre después. No acreditan frescura por contenido ni exclusión
# frente a otro worktree. RB-H2 es un cambio separado; aquí no se añaden locks.
mkdir -p -- "$CARGO_TARGET_DIR"
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

log=${T1_LOG:-$raiz/logs/t1_oraculo.log}
mkdir -p -- "$(dirname -- "$log")"
inicio=$(date +%s)
set +e
# F-A5 (Fable): el hook histórico sigue siendo válido, normalizado al Bash
# actual y al semáforo de este worktree; se adquiere un solo turno:
#   T1_CARGO="bash scripts/cargo_semaforo.sh" CARGO_BUILD_JOBS=2 scripts/t1_oraculo.sh 4
# Sin la variable se usa el mismo semáforo. Las rutas con espacios son literales.
"$BASH" "$semaforo" test "${extra[@]}" --release -p backtest-engine --locked \
  --test t1_cobertura_genetica -- --exact t1_cobertura_genetica_del_oraculo_de_aptitud \
  --nocapture --test-threads=1 >"$log" 2>&1
rc=$?
set -e

# Marcador histórico de ejecución; por sí solo no prueba procedencia del binario.
if grep -q "Running tests/t1_cobertura_genetica" -- "$log"; then
  echo "$sello" >"$sello_archivo"
fi

echo "== T-1 $(git rev-parse --short HEAD) con $hilos hilos: rc=$rc en $(($(date +%s) - inicio)) s (log: $log)"
grep -E "Compiling (risk-engine|metacortex-engine|god-engine-core|backtest-engine)|evaluaciones en|COBERTURA|Sin cambio|test result|panicked" -- "$log" | tail -12 || true
exit "$rc"
