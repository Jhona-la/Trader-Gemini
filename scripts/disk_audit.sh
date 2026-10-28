#!/usr/bin/env bash
# Auditoría de disco del proyecto (GLM 125, petición del operador
# 2026-10-15: "el proyecto está ocupando mucha memoria, la va a
# desbordar") — mapa de consumidores por dueño + reglas de contención.
#
# Uso: bash scripts/disk_audit.sh          (solo medir, no toca nada)
#      CLEAN_INCREMENTAL=1 se rechaza: usar el barrido con cerrojo compartido.
#
# Nada más se borra automáticamente: worktrees y targets ajenos son de
# sus dueños (regla del buzón MEMORIA-2026-10-15).

set -uo pipefail

# Git usr/bin/bash puede heredar sólo el PATH de Windows. Conserva los shims
# válidos del llamador y recurre a GNU únicamente si falta una herramienta.
for herramienta in df du sort find; do
  version=$("$herramienta" --version 2>/dev/null)
  if [[ "$version" != *GNU* ]] && [[ "$version" != *uutils* ]] && [ -d /usr/bin ]; then
    PATH="/usr/bin:/bin:$PATH"
    break
  fi
done
for herramienta in df du sort find; do
  version=$("$herramienta" --version 2>/dev/null)
  if [[ "$version" != *GNU* ]] && [[ "$version" != *uutils* ]]; then
    echo "[disk_audit] falta $herramienta de GNU: medición desconocida" >&2
    exit 2
  fi
done
for herramienta in git tail head cut tr awk; do
  command -v "$herramienta" >/dev/null 2>&1 || {
    echo "[disk_audit] falta $herramienta: medición desconocida" >&2
    exit 2
  }
done
codigo=0
medir() {
  local salida
  if salida=$(du -sh -- "$@" 2>/dev/null); then
    printf '%s\n' "$salida"
  else
    echo "[disk_audit] tamaño desconocido: $*" >&2
    codigo=2
    return 1
  fi
}

echo "=== VOLUMEN ==="
df -h . | tail -1 || codigo=2

echo ""
echo "=== PROYECTO (primer nivel, top 15) ==="
shopt -s nullglob dotglob
entradas=(*)
if [ "${#entradas[@]}" -gt 0 ]; then
  if medidas=$(du -sh -- "${entradas[@]}" 2>/dev/null); then
    if ! ordenadas=$(printf '%s\n' "$medidas" | sort -rh); then
      echo "[disk_audit] falló ordenar tamaños: inventario desconocido" >&2
      codigo=2
    else
      # sort ya terminó: head no puede provocarle SIGPIPE al productor.
      head -15 <<< "$ordenadas" || codigo=2
    fi
  else
    echo "[disk_audit] inventario incompleto: tamaño del proyecto desconocido" >&2
    codigo=2
  fi
fi

echo ""
echo "=== BUILD ACTIVO? (cargo/rustc vivos) ==="
if BUILDS=$(powershell -NoProfile -Command "Get-Process cargo,rustc -ErrorAction SilentlyContinue | Measure-Object | Select-Object -ExpandProperty Count" 2>/dev/null | tr -d '\r '); then
  [[ "$BUILDS" =~ ^[0-9]+$ ]] || BUILDS="?"
else
  BUILDS="?"
fi
echo "procesos cargo/rustc: $BUILDS"

echo ""
echo "=== TARGET COMPARTIDO ==="
for ruta in target/debug target/release target/debug/incremental; do
  if [ -e "$ruta" ]; then
    medir "$ruta" || codigo=2
  else
    echo "$ruta: ausente"
  fi
done
if [ -d target/debug/deps ]; then
  if exes=$(find target/debug/deps -maxdepth 1 -type f -name '*.exe' -printf 'x\n' 2>/dev/null); then
    echo "exes en debug/deps: $(printf '%s' "$exes" | awk 'END { print NR }')"
  else
    echo "exes en debug/deps: desconocido (falló la enumeración)" >&2
    codigo=2
  fi
else
  echo "exes en debug/deps: 0 (directorio ausente)"
fi

echo ""
echo "=== WORKTREES REGISTRADOS (du de cada uno) ==="
if worktrees=$(git worktree list --porcelain 2>/dev/null); then
  while IFS= read -r linea; do
    case "$linea" in worktree\ *) wt=${linea#worktree } ;; *) continue ;; esac
    wt_u="$(cygpath -u "$wt" 2>/dev/null || echo "$wt")"
    case "$wt_u" in
      "$PWD"|"$PWD"/*) medir "$wt_u" || codigo=2 ;;
    esac
  done <<< "$worktrees"
else
  echo "[disk_audit] worktrees desconocidos: falló git" >&2
  codigo=2
fi

if [[ "${CLEAN_INCREMENTAL:-0}" == "1" ]]; then
  echo "[disk_audit] CLEAN_INCREMENTAL omitido: esta auditoría no toma el cerrojo compartido." >&2
  echo "Use APLICAR=1 bash scripts/barrido_target.sh (retención por presupuesto y cerrojo)." >&2
  exit 5
fi
exit "$codigo"
