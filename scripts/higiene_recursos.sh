#!/usr/bin/env bash
# F-H1 (Fable, 2026-10-17) — higiene automática de recursos del proyecto.
#
# Complementa a scripts/disk_audit.sh (que sólo mide; el borrado usa el
# cerrojo de barrido_target.sh). Este script aplica la POLÍTICA de retención que
# AGENT_SYNC §4 enuncia pero nadie ejecutaba, siempre con guardas:
#
#   1. logs/*.log y logs/*.txt con más de LOGS_COMPRIMIR_DIAS (7) días → gzip.
#      logs/*.gz con más de LOGS_BORRAR_DIAS (30) días → se borran.
#      Los `logs_motor_*.txt` sueltos en la raíz se MUEVEN a logs/ primero.
#   2. graphify-out/20*/ NO versionadas: se conservan las GRAPHIFY_CONSERVAR
#      (3) más recientes; el resto se borra (son instantáneas regenerables;
#      las versionadas en git no se tocan nunca).
#   3. target/debug/deps/*.exe viejos: sólo se listan. El borrado legacy
#      sin cerrojo se retiró; target/ se limpia exclusivamente en el paso 3b.
#   3b. RB-H1 (Codex RB, 2026-10-19): target/ por PRESUPUESTO vía
#      scripts/barrido_target.sh — LRU de target/*/incremental por encima de
#      BARRIDO_INCREMENTAL_MAX_MB y variantes superadas de deps/ con más de
#      BARRIDO_VARIANTES_H horas (el semáforo lo aplica solo cada 6 h).
#   4. `git worktree prune` (entradas de worktrees ya borrados del disco).
#
# NUNCA toca: data/, models/, .git/, worktrees con cambios, ficheros
# versionados, ni nada fuera de la raíz del repositorio.
#
# Uso:  bash scripts/higiene_recursos.sh            (simulación: lista y mide)
#       APLICAR=1 bash scripts/higiene_recursos.sh  (ejecuta)
# Entorno: LOGS_COMPRIMIR_DIAS, LOGS_BORRAR_DIAS, GRAPHIFY_CONSERVAR, EXES_DIAS,
#          HIGIENE_AGENTE (quién eres; va a logs/higiene.log).
# Salida: 0 ok; 2 error de medición/acción; 5 target omitido por guardas.
set -uo pipefail

for herramienta in find du sort df; do
  version=$("$herramienta" --version 2>/dev/null)
  if [[ "$version" != *GNU* ]] && [[ "$version" != *uutils* ]] && [ -d /usr/bin ]; then
    PATH="/usr/bin:/bin:$PATH"
    break
  fi
done
for herramienta in find du sort df; do
  version=$("$herramienta" --version 2>/dev/null)
  if [[ "$version" != *GNU* ]] && [[ "$version" != *uutils* ]]; then
    echo "[higiene] falta $herramienta de GNU: no limpio" >&2
    exit 2
  fi
done
for herramienta in git dirname mkdir date tail gzip mv rm sed awk; do
  command -v "$herramienta" >/dev/null 2>&1 || {
    echo "[higiene] falta $herramienta: no limpio" >&2
    exit 2
  }
done

raiz=$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null) || {
  echo "[higiene] no estoy dentro del repositorio" >&2
  exit 2
}
raiz=$(dirname "$raiz")
cd "$raiz" || exit 2

aplicar=${APLICAR:-0}
logs_comprimir=${LOGS_COMPRIMIR_DIAS:-7}
logs_borrar=${LOGS_BORRAR_DIAS:-30}
graphify_conservar=${GRAPHIFY_CONSERVAR:-3}
exes_dias=${EXES_DIAS:-3}
agente=${HIGIENE_AGENTE:-${SEMAFORO_AGENTE:-${USER:-${USERNAME:-agente}}}}
bitacora="$raiz/logs/higiene.log"
mkdir -p "$raiz/logs"
fallos=0

registrar() { printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$agente" "$*" >> "$bitacora"; }
bytes_de() { du -sb -- "$@" 2>/dev/null | awk '{s+=$1} END {printf "%.0f", s+0}'; }
mb_de() { bytes_de "$@" | awk '{printf "%.2f", $1/1048576}'; }
no_versionado() {
  local rastreados
  if ! rastreados=$(git ls-files -- "$1"); then
    echo "[higiene] git no pudo verificar $1: lo conservo" >&2
    fallos=1
    return 1
  fi
  if [ -n "$rastreados" ]; then
    echo "  [conservar] versionado: $1"
    return 1
  fi
  return 0
}
accion() {
  # accion <descripción> <comando...>: en simulación sólo imprime.
  local desc=$1; shift
  if [ "$aplicar" = "1" ]; then
    if "$@"; then
      echo "  [hecho] $desc"
    else
      echo "  [FALLÓ] $desc" >&2
      fallos=1
      return 1
    fi
  else
    echo "  [simularía] $desc"
  fi
}

echo "=== HIGIENE DE RECURSOS ($([ "$aplicar" = 1 ] && echo APLICANDO || echo SIMULACIÓN)) ==="
echo "raíz: $raiz"
df -h . | tail -1 || fallos=1
liberado_bytes=0

# ---------- 1. logs ----------
echo ""
echo "--- 1. logs/ (comprimir > ${logs_comprimir} d, borrar .gz > ${logs_borrar} d) ---"
for f in logs_motor_*.txt; do
  [ -f "$f" ] || continue
  no_versionado "$f" || continue
  accion "mover $f → logs/" mv -n "$f" "logs/$f"
done
a_comprimir=()
a_borrar=()
if ! listado_comprimir=$(find logs -maxdepth 1 -type f \( -name '*.log' -o -name '*.txt' \) -mtime +"$logs_comprimir" -size +1k 2>/dev/null) \
  || ! listado_borrar=$(find logs -maxdepth 1 -type f -name '*.gz' -mtime +"$logs_borrar" 2>/dev/null); then
  echo "[higiene] falló la enumeración de logs: no limpio" >&2
  exit 2
fi
[ -z "$listado_comprimir" ] || mapfile -t a_comprimir <<< "$listado_comprimir"
[ -z "$listado_borrar" ] || mapfile -t a_borrar <<< "$listado_borrar"
echo "ficheros a comprimir: ${#a_comprimir[@]} ($(mb_de "${a_comprimir[@]:-/dev/null}") MB)"
echo "ficheros .gz a borrar: ${#a_borrar[@]} ($(mb_de "${a_borrar[@]:-/dev/null}") MB)"
for f in "${a_comprimir[@]}"; do
  no_versionado "$f" || continue
  if bytes=$(bytes_de "$f"); then
    if accion "gzip $f" gzip -9 "$f"; then
      liberado_bytes=$((liberado_bytes + bytes * 8 / 10))
    fi
  else
    echo "[higiene] tamaño desconocido, omito $f" >&2
    fallos=1
  fi
done
for f in "${a_borrar[@]}"; do
  no_versionado "$f" || continue
  if bytes=$(bytes_de "$f"); then
    if accion "borrar $f" rm -f "$f"; then
      liberado_bytes=$((liberado_bytes + bytes))
    fi
  else
    echo "[higiene] tamaño desconocido, omito $f" >&2
    fallos=1
  fi
done

# ---------- 2. graphify-out dated ----------
echo ""
echo "--- 2. graphify-out/20*/ no versionadas (conservar ${graphify_conservar}) ---"
if [ -d graphify-out ]; then
  fechadas=()
  if ! listado_fechadas=$(find graphify-out -mindepth 1 -maxdepth 1 -type d -name '20*' 2>/dev/null | sort); then
    echo "[higiene] falló la enumeración de graphify-out: no limpio" >&2
    exit 2
  fi
  [ -z "$listado_fechadas" ] || mapfile -t fechadas <<< "$listado_fechadas"
  candidatas=()
  for d in "${fechadas[@]}"; do
    d=${d%/}
    # Sólo las que git NO versiona (ni un solo fichero rastreado dentro).
    if ! rastreados=$(git ls-files -- "$d"); then
      echo "[higiene] git no pudo verificar $d: no limpio graphify-out" >&2
      exit 2
    fi
    if [ -z "$rastreados" ]; then
      candidatas+=("$d")
    fi
  done
  n=${#candidatas[@]}
  echo "instantáneas no versionadas: $n"
  if [ "$n" -gt "$graphify_conservar" ]; then
    for d in "${candidatas[@]:0:$((n - graphify_conservar))}"; do
      if bytes=$(bytes_de "$d"); then
        if accion "borrar $d ($(mb_de "$d") MB)" rm -rf -- "$d"; then
          liberado_bytes=$((liberado_bytes + bytes))
        fi
      else
        echo "[higiene] tamaño desconocido, omito $d" >&2
        fallos=1
      fi
    done
  fi
fi

# ---------- 3. binarios de prueba viejos ----------
echo ""
echo "--- 3. target/debug/deps/*.exe > ${exes_dias} d (sólo inventario) ---"
codigo=0
exes=()
if [ -d target/debug/deps ]; then
  if ! listado_exes=$(find target/debug/deps -maxdepth 1 -type f -name '*.exe' -mtime +"$exes_dias" 2>/dev/null); then
    echo "[higiene] falló la enumeración de exes: no limpio target/" >&2
    exit 2
  fi
  [ -z "$listado_exes" ] || mapfile -t exes <<< "$listado_exes"
fi
echo "binarios candidatos: ${#exes[@]} ($(mb_de "${exes[@]:-/dev/null}") MB)"
if [ "${#exes[@]}" -gt 0 ]; then
  echo "  [omitido] borrado por edad retirado: target/ se limpia con cerrojo en el paso 3b."
fi

# ---------- 3b. target/ por presupuesto (RB-H1, Codex RB 2026-10-19) ----------
# La misma política que el semáforo aplica sola cada 6 h: LRU de la caché
# incremental por presupuesto y variantes superadas de deps (ver la cabecera de
# barrido_target.sh). Toma el cerrojo del semáforo; con builds vivos no borra.
echo ""
echo "--- 3b. target/: incremental por presupuesto + variantes superadas (barrido_target.sh) ---"
fuente=${BASH_SOURCE[0]//\\//}
dir_scripts=.
case "$fuente" in */*) dir_scripts=${fuente%/*} ;; esac
if [ -f "$dir_scripts/barrido_target.sh" ]; then
  APLICAR="$aplicar" BARRIDO_AGENTE="$agente" "$BASH" "$dir_scripts/barrido_target.sh" 2>&1 | sed 's/^/  /'
  codigo_barrido=${PIPESTATUS[0]}
  [ "$codigo_barrido" = "0" ] || codigo=$codigo_barrido
else
  echo "  [omitido] no encuentro $dir_scripts/barrido_target.sh"
  codigo=2
fi

# ---------- 4. worktrees ----------
echo ""
echo "--- 4. worktrees ---"
accion "git worktree prune" git worktree prune
git worktree list | sed 's/^/  /' || fallos=1

echo ""
liberado_mb=$(awk -v b="$liberado_bytes" 'BEGIN {printf "%.2f", b/1048576}')
echo "=== estimación lógica liberaría/liberó ≈ ${liberado_mb} MiB (gzip estimado al 80%) ==="
[ "$fallos" = "0" ] || codigo=2
modo=$([ "$aplicar" = 1 ] && echo APLICADO || echo SIMULADO)
[ "$codigo" = "0" ] || modo=INCOMPLETO
registrar "$modo liberado≈${liberado_mb}MiB comprimir=${#a_comprimir[@]} gz_borrar=${#a_borrar[@]} exes=${#exes[@]} codigo=$codigo"
exit "$codigo"
