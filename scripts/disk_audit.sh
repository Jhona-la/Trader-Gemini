#!/usr/bin/env bash
# Auditoría de disco del proyecto (GLM 125, petición del operador
# 2026-10-15: "el proyecto está ocupando mucha memoria, la va a
# desbordar") — mapa de consumidores por dueño + reglas de contención.
#
# Uso: bash scripts/disk_audit.sh          (solo medir, no toca nada)
#      CLEAN_INCREMENTAL=1 bash scripts/disk_audit.sh   (además borra
#      target/debug/incremental — SOLO si no hay cargo/rustc activos;
#      es caché regenerable de rustc, 25GB medidos la primera vez).
#
# Nada más se borra automáticamente: worktrees y targets ajenos son de
# sus dueños (regla del buzón MEMORIA-2026-10-15).

set -uo pipefail

echo "=== VOLUMEN ==="
df -h . | tail -1

echo ""
echo "=== PROYECTO (primer nivel, top 15) ==="
du -sh .[!.]* * 2>/dev/null | sort -rh | head -15

echo ""
echo "=== BUILD ACTIVO? (cargo/rustc vivos) ==="
BUILDS=$(powershell -NoProfile -Command "Get-Process cargo,rustc -ErrorAction SilentlyContinue | Measure-Object | Select-Object -ExpandProperty Count" 2>/dev/null | tr -d '\r')
echo "procesos cargo/rustc: ${BUILDS:-?}"

echo ""
echo "=== TARGET COMPARTIDO ==="
du -sh target/debug target/release 2>/dev/null
echo "incremental: $(du -sh target/debug/incremental 2>/dev/null | cut -f1 || echo '0 (limpio)')"
echo "exes en debug/deps: $(ls target/debug/deps/*.exe 2>/dev/null | wc -l)"

echo ""
echo "=== WORKTREES REGISTRADOS (du de cada uno) ==="
git worktree list --porcelain 2>/dev/null | grep '^worktree ' | cut -d' ' -f2 | while read -r wt; do
  # tamaño solo si está dentro del proyecto (los externos los mide su dueño)
  wt_u="$(cygpath -u "$wt" 2>/dev/null || echo "$wt")"
  case "$wt_u" in
    "$PWD"|"$PWD"/*) du -sh "$wt_u" 2>/dev/null ;;
  esac
done

if [[ "${CLEAN_INCREMENTAL:-0}" == "1" ]]; then
  echo ""
  if [[ "${BUILDS:-1}" == "0" ]]; then
    echo ">>> Borrando target/debug/incremental (caché regenerable)…"
    rm -rf target/debug/incremental
    echo ">>> Hecho. Libre ahora:"
    df -h . | tail -1
  else
    echo ">>> ABORTADO: hay ${BUILDS} procesos cargo/rustc activos — no se borra nada."
    exit 5
  fi
fi
