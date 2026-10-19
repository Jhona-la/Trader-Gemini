#!/usr/bin/env bash
# F-H1 (Fable, 2026-10-17) — higiene automática de recursos del proyecto.
#
# Complementa a scripts/disk_audit.sh (que sólo mide y, a petición, borra
# target/debug/incremental). Este script aplica la POLÍTICA de retención que
# AGENT_SYNC §4 enuncia pero nadie ejecutaba, siempre con guardas:
#
#   1. logs/*.log y logs/*.txt con más de LOGS_COMPRIMIR_DIAS (7) días → gzip.
#      logs/*.gz con más de LOGS_BORRAR_DIAS (30) días → se borran.
#      Los `logs_motor_*.txt` sueltos en la raíz se MUEVEN a logs/ primero.
#   2. graphify-out/20*/ NO versionadas: se conservan las GRAPHIFY_CONSERVAR
#      (3) más recientes; el resto se borra (son instantáneas regenerables;
#      las versionadas en git no se tocan nunca).
#   3. target/debug/deps/*.exe (binarios de prueba, ≈1 GB cada uno con
#      debug=1) con más de EXES_DIAS (3) días → se borran, SÓLO si no hay
#      ningún cargo/rustc vivo (mismo guard que disk_audit.sh).
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
# Código de salida: 0 ok; 5 abortado por builds activos (paso 3 omitido).
set -uo pipefail

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

registrar() { printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$agente" "$*" >> "$bitacora"; }
mb_de() { du -sm "$@" 2>/dev/null | awk '{s+=$1} END {print s+0}'; }
accion() {
  # accion <descripción> <comando...>: en simulación sólo imprime.
  local desc=$1; shift
  if [ "$aplicar" = "1" ]; then
    "$@" && echo "  [hecho] $desc" || echo "  [FALLÓ] $desc" >&2
  else
    echo "  [simularía] $desc"
  fi
}

builds_activos() {
  if command -v powershell > /dev/null 2>&1; then
    powershell -NoProfile -Command \
      "(Get-Process cargo,rustc -ErrorAction SilentlyContinue | Measure-Object).Count" \
      2> /dev/null | tr -d '\r '
  else
    pgrep -c -x 'cargo|rustc' 2> /dev/null || echo 0
  fi
}

echo "=== HIGIENE DE RECURSOS ($([ "$aplicar" = 1 ] && echo APLICANDO || echo SIMULACIÓN)) ==="
echo "raíz: $raiz"
df -h . | tail -1
liberado_mb=0

# ---------- 1. logs ----------
echo ""
echo "--- 1. logs/ (comprimir > ${logs_comprimir} d, borrar .gz > ${logs_borrar} d) ---"
for f in logs_motor_*.txt; do
  [ -f "$f" ] || continue
  accion "mover $f → logs/" mv -n "$f" "logs/$f"
done
mapfile -t a_comprimir < <(find logs -maxdepth 1 -type f \( -name '*.log' -o -name '*.txt' \) -mtime +"$logs_comprimir" -size +1k 2>/dev/null)
mapfile -t a_borrar < <(find logs -maxdepth 1 -type f -name '*.gz' -mtime +"$logs_borrar" 2>/dev/null)
echo "ficheros a comprimir: ${#a_comprimir[@]} ($(mb_de "${a_comprimir[@]:-/dev/null}") MB)"
echo "ficheros .gz a borrar: ${#a_borrar[@]} ($(mb_de "${a_borrar[@]:-/dev/null}") MB)"
for f in "${a_comprimir[@]}"; do
  liberado_mb=$((liberado_mb + $(mb_de "$f") * 8 / 10))
  accion "gzip $f" gzip -9 "$f"
done
for f in "${a_borrar[@]}"; do
  liberado_mb=$((liberado_mb + $(mb_de "$f")))
  accion "borrar $f" rm -f "$f"
done

# ---------- 2. graphify-out dated ----------
echo ""
echo "--- 2. graphify-out/20*/ no versionadas (conservar ${graphify_conservar}) ---"
if [ -d graphify-out ]; then
  mapfile -t fechadas < <(ls -d graphify-out/20*/ 2>/dev/null | sort)
  candidatas=()
  for d in "${fechadas[@]}"; do
    d=${d%/}
    # Sólo las que git NO versiona (ni un solo fichero rastreado dentro).
    if [ -z "$(git ls-files "$d" | head -1)" ]; then
      candidatas+=("$d")
    fi
  done
  n=${#candidatas[@]}
  echo "instantáneas no versionadas: $n"
  if [ "$n" -gt "$graphify_conservar" ]; then
    for d in "${candidatas[@]:0:$((n - graphify_conservar))}"; do
      liberado_mb=$((liberado_mb + $(mb_de "$d")))
      accion "borrar $d ($(mb_de "$d") MB)" rm -rf "$d"
    done
  fi
fi

# ---------- 3. binarios de prueba viejos ----------
echo ""
echo "--- 3. target/debug/deps/*.exe > ${exes_dias} d (sólo sin builds activos) ---"
codigo=0
mapfile -t exes < <(find target/debug/deps -maxdepth 1 -type f -name '*.exe' -mtime +"$exes_dias" 2>/dev/null)
echo "binarios candidatos: ${#exes[@]} ($(mb_de "${exes[@]:-/dev/null}") MB)"
if [ "${#exes[@]}" -gt 0 ]; then
  builds=$(builds_activos)
  if [ "${builds:-1}" = "0" ]; then
    for f in "${exes[@]}"; do
      liberado_mb=$((liberado_mb + $(mb_de "$f")))
      accion "borrar $f" rm -f "$f" "${f%.exe}.pdb"
    done
  else
    echo "  [omitido] hay ${builds} procesos cargo/rustc activos — no se toca target/"
    codigo=5
  fi
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
  [ "$codigo_barrido" = "5" ] && codigo=5
else
  echo "  [omitido] no encuentro $dir_scripts/barrido_target.sh"
fi

# ---------- 4. worktrees ----------
echo ""
echo "--- 4. worktrees ---"
accion "git worktree prune" git worktree prune
git worktree list | sed 's/^/  /'

echo ""
echo "=== liberaría/liberó ≈ ${liberado_mb} MB ==="
registrar "$([ "$aplicar" = 1 ] && echo APLICADO || echo SIMULADO) liberado≈${liberado_mb}MB comprimir=${#a_comprimir[@]} gz_borrar=${#a_borrar[@]} exes=${#exes[@]} codigo=$codigo"
exit "$codigo"
