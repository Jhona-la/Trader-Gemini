#!/usr/bin/env bash
# Testigo del oráculo T-1 (GLM 124, SOL-R4-01) — recibo de QUÉ fuentes corrió.
#
# Motivación (dos episodios reales):
# - Ciclo 123: otro agente integró mi rama a main y pusheó mientras mi
#   oráculo corría en background; la validez se probó a posteriori con el
#   mtime ad-hoc del binario.
# - CL-60b: con CARGO_TARGET_DIR compartido, cargo no distingue dos
#   worktrees del mismo paquete y puede correr el binario del OTRO árbol
#   (el PR #37 midió la base creyendo medir la rama).
#
# Este script hace el procedimiento completo con recibo:
#   1. TOCA el paquete para solicitar recompilación.
#   2. Compila SIN correr (--no-run).
#   3. Testigo PRE: HEAD, sha256 y mtime del binario, args, timestamp.
#   4. Invoca el oráculo por Cargo con otro turno del semáforo.
#   5. Testigo POST: sha256 de nuevo, exit code, duración.
# Todo queda en outputs/oracle_witness_<ts>.log.
#
# Uso: scripts/oracle_witness.sh [args-extra-de-cargo-test]
# Ejemplo por defecto: oráculo T-1 del plan (t1_cobertura_genetica, release).
# Límite: el glob y sus hashes observan un archivo del target compartido.
# La segunda invocación puede recompilar/elegir otro ejecutable: este recibo
# NO certifica la identidad del binario ejecutado ni aislamiento entre turnos.

set -uo pipefail

REPO=$(git rev-parse --show-toplevel) || exit 2
COMUN=$(git rev-parse --path-format=absolute --git-common-dir) || exit 2
RAIZ_PRINCIPAL=$(cd -- "$(dirname -- "$COMUN")" && pwd -P) || exit 2
SEMAFORO="$REPO/scripts/cargo_semaforo.sh"
if [[ ! -f "$SEMAFORO" || ${RUSTFLAGS+x} ]]; then
  echo 'necesito el semáforo del checkout y RUSTFLAGS de la configuración compartida' >&2
  exit 2
fi
for ARGUMENTO in "$@"; do
  [[ "$ARGUMENTO" = -- ]] && break
  case "$ARGUMENTO" in
    --target-dir | --target-dir=* | --config | --config=* | -j* | --jobs | --jobs=*)
      echo 'el oráculo no puede cambiar el target/config ni los jobs compartidos' >&2
      exit 2 ;;
  esac
done
cd -- "$REPO" || exit 2
export SEMAFORO_RAIZ="$RAIZ_PRINCIPAL" CARGO_TARGET_DIR="$RAIZ_PRINCIPAL/target"
export CARGO_BUILD_JOBS=2
PKG="backtest-engine"
PKG_DIR="crates/backtest-engine"
TEST_NAME="t1_cobertura_genetica"
TARGET_DIR="$CARGO_TARGET_DIR"
# Presupuesto de RAM compartido (convención del repo — Sol usa -j2 en
# worktrees): la compilación full-graph en release sin límite muere por
# OOM con otras sesiones vivas.
OUT_DIR="outputs"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$OUT_DIR/oracle_witness_${TS}.log"
BIN_GLOB="${TARGET_DIR}/release/deps/${TEST_NAME}-*.exe"
SCRIPT_START_EPOCH="$(date +%s)"

mkdir -p "$OUT_DIR"

{
  echo "# Testigo del oráculo — $(date -Is)"
  echo "# repo: $(pwd)"
  echo "# HEAD: $(git rev-parse HEAD)"
  echo "# rama: $(git branch --show-current)  (status: $(git status --porcelain | wc -l) archivos sucios)"
  echo "# jobs: ${CARGO_BUILD_JOBS} (RAM compartida)"
  echo '# LIMITE: hashes del archivo observado; no certifican el ejecutable de la segunda invocación Cargo'
} >> "$LOG"

# 1. TOCAR el paquete Y el test file del oráculo: con target compartido,
#    esto obliga a que el binario se reconstruya desde ESTE árbol
#    (mitigación CL-60b). El touch del TEST FILE fuerza el relink del
#    test exe aunque el delta del ciclo sea bin-only (los bins no son
#    dependencia del test target — incidente GLM 125: guard disparaba en
#    falso positivo porque el test exe no relinkeaba).
find "$PKG_DIR" -name '*.rs' -newer "$(git rev-parse --git-dir)/HEAD" -print -quit 2>/dev/null | grep -q . || \
  touch "$PKG_DIR/src/lib.rs"
TEST_SRC="$PKG_DIR/tests/${TEST_NAME}.rs"
[[ -f "$TEST_SRC" ]] && touch "$TEST_SRC"

# 2. Compilar SIN correr.
echo "[witness] compilando (--no-run)…" >&2
"$BASH" "$SEMAFORO" test -p "$PKG" --release --test "$TEST_NAME" --no-run >> "$LOG" 2>&1
CODIGO_COMPILACION=$?
if [[ "$CODIGO_COMPILACION" != 0 ]]; then
  echo "COMPILACION FALLIDA — ver $LOG" >&2
  exit "$CODIGO_COMPILACION"
fi

# 3. Testigo PRE + GUARD DE FRESCURA (incidente GLM 124, 2026-10-15 15:10:
#    con target compartido cargo recompiló el lib del worktree pero reutilizó
#    el test exe VIEJO del otro árbol — mtime 12:16 del ciclo anterior — y el
#    oráculo habría corrido fuentes ajenas. Este guard lo aborta).
BIN="$(ls -t "$TARGET_DIR"/release/deps/"$TEST_NAME"-*.exe 2>/dev/null | head -1)"
if [[ -z "$BIN" ]]; then
  echo "BINARIO NO ENCONTRADO ($BIN_GLOB)" >> "$LOG"
  exit 3
fi
BIN_EPOCH="$(date -r "$BIN" +%s)"
if [[ "$BIN_EPOCH" -lt "$SCRIPT_START_EPOCH" ]]; then
  {
    echo "GUARD CL-60b DISPARADO: el binario ($BIN, mtime $(date -Is -r "$BIN"))"
    echo "es ANTERIOR al inicio del testigo — cargo reutilizó un test exe de"
    echo "OTRO árbol. NO se corre el oráculo sobre fuentes no verificadas."
    echo "Solución: reintentar con turno del semáforo y fuentes verificadas;"
    echo "no crear targets alternativos."
  } >> "$LOG"
  echo "[witness] GUARD: binario ajeno/viejo — abortado (ver $LOG)" >&2
  exit 4
fi
{
  echo "PRE binario: $BIN"
  echo "PRE sha256:  $(sha256sum "$BIN" | cut -d' ' -f1)"
  echo "PRE mtime:   $(date -Is -r "$BIN")"
} >> "$LOG"
echo "[witness] binario: $BIN" >&2

# 4. Otra invocación con turno: puede recompilar; ver límite del recibo.
echo "[witness] corriendo oráculo…" >&2
START=$(date +%s)
"$BASH" "$SEMAFORO" test -p "$PKG" --release --test "$TEST_NAME" "$@" >> "$LOG" 2>&1
EXIT_CODE=$?
END=$(date +%s)

# 5. Testigo POST.
{
  echo "POST sha256: $(sha256sum "$BIN" | cut -d' ' -f1)"
  echo "POST mtime:  $(date -Is -r "$BIN")"
  echo "exit_code:   $EXIT_CODE"
  echo "duracion_s:  $((END - START))"
  echo "# fin — $(date -Is)"
} >> "$LOG"

echo "[witness] recibo completo: $LOG (exit $EXIT_CODE)" >&2
exit $EXIT_CODE
