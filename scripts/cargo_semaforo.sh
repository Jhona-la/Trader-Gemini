#!/usr/bin/env bash
# LG-25 (Línea G) — semáforo de compilación compartido por todos los agentes.
#
# Por qué: la máquina tiene 16 GB útiles para builds y varios agentes compilan
# en paralelo (cada rustc ocupa 0,4–0,9 GB; AGY midió 6 rustc = 4,3 GB). La
# regla de AGENT_SYNC §4 («-j2, un build a la vez, target compartido») no la
# hacía cumplir nada. Este envoltorio la hace cumplir:
#   1. target ÚNICO: CARGO_TARGET_DIR = <raíz del repo>/target, también desde
#      cualquier worktree (lo deduce de `git rev-parse --git-common-dir`).
#      RB-H2: es seguro entre worktrees SÓLO con la frescura por contenido de
#      `.cargo/config.toml` ([unstable] checksum-freshness = true): dos
#      worktrees escriben los mismos artefactos y, por mtime, uno puede
#      ejecutar el código del otro (scripts/tests/test_frescura_target.py).
#   2. CARGO_BUILD_JOBS=2 salvo que se pida otra cosa explícitamente.
#   3. Espera mientras haya OTRO cargo/rustc vivo o la RAM libre esté por
#      debajo de SEMAFORO_RAM_MIN_GB (4 por defecto).
#   4. Cerrojo atómico (mkdir) en <raíz>/target/.semaforo.lock con su dueño.
#      Un cerrojo es huérfano si supera SEMAFORO_TTL_MIN (240), o si pasada
#      la gracia SEMAFORO_GRACIA_S (120 s) no hay ningún cargo/rustc vivo: un
#      agente sin cuota no bloquea a nadie y dos agentes no se roban el turno.
#   5. Una línea por adquisición y liberación en <raíz>/logs/semaforo.log
#      (agente, comando, RAM libre, duración).
#   6. LG-25b — paciencia: pasados SEMAFORO_PACIENCIA_MIN (30) minutos de
#      espera, los cargo/rustc AJENOS (de quien compila por fuera del
#      semáforo) dejan de bloquear si la RAM libre llega a
#      SEMAFORO_RAM_MIN_GB + SEMAFORO_RAM_EXTRA_GB (4 + 2). El cerrojo entre
#      usuarios del semáforo no se salta nunca.
#   7. RB-H1 — barrido automático del target/ compartido por PRESUPUESTO
#      (scripts/barrido_target.sh: LRU de incremental + variantes superadas
#      de deps) como mucho cada SEMAFORO_BARRIDO_H (6) horas, con el turno ya
#      tomado y 0 cargo/rustc vivos. Nunca bloquea el build; sello en
#      <raíz>/target/.barrido_ultimo; SEMAFORO_BARRIDO=0 lo apaga.
#
# Uso:   bash scripts/cargo_semaforo.sh <argumentos de cargo>
#        p. ej. bash scripts/cargo_semaforo.sh check --workspace --all-targets
# Entorno: SEMAFORO_AGENTE (quién eres), SEMAFORO_RAM_MIN_GB, SEMAFORO_TTL_MIN,
#          SEMAFORO_ESPERA_MAX_MIN (60; al agotarla sale con 75 sin compilar),
#          SEMAFORO_PAUSA_S (15), SEMAFORO_SOLO_DECIDIR=1 (imprime la decisión
#          y el entorno y sale sin compilar).
# Costuras de prueba (scripts/tests/test_cargo_semaforo.py): SEMAFORO_RAIZ,
#          SEMAFORO_PROCESOS (fuerza el nº de cargo/rustc ajenos),
#          SEMAFORO_RAM_GB (fuerza la RAM libre).
set -uo pipefail

raiz=${SEMAFORO_RAIZ:-}
if [ -z "$raiz" ]; then
  comun=$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null) || {
    echo "[semaforo] no estoy dentro del repositorio" >&2
    exit 2
  }
  raiz=$(dirname "$comun")
fi
agente=${SEMAFORO_AGENTE:-${USER:-${USERNAME:-agente}}}
ram_min=${SEMAFORO_RAM_MIN_GB:-4}
ttl_min=${SEMAFORO_TTL_MIN:-240}
gracia_s=${SEMAFORO_GRACIA_S:-120}
espera_max_min=${SEMAFORO_ESPERA_MAX_MIN:-60}
pausa_s=${SEMAFORO_PAUSA_S:-15}
# LG-25b: quien compila POR FUERA del semáforo no toma el cerrojo, así que
# esperar a «cero procesos ajenos» deja sin turno a los que sí lo usan (el
# 17-oct dos builds ajenos de dos horas dejaron sin turno a AGY, Fable y la
# Línea G). Pasada la paciencia, los ajenos dejan de bloquear si la RAM libre
# cubre el mínimo más un margen; el cerrojo entre usuarios sigue mandando.
paciencia_min=${SEMAFORO_PACIENCIA_MIN:-30}
ram_extra=${SEMAFORO_RAM_EXTRA_GB:-2}

export CARGO_TARGET_DIR="$raiz/target"
export CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-2}
# RB-H2: frescura por contenido también para las ramas cuya .cargo/config.toml
# aún no trae la clave (el target es compartido entre worktrees; ver punto 1).
export CARGO_UNSTABLE_CHECKSUM_FRESHNESS=true
mkdir -p "$raiz/target" "$raiz/logs"
cerrojo="$raiz/target/.semaforo.lock"
bitacora="$raiz/logs/semaforo.log"

# Cuántos cargo/rustc hay vivos. «?» si no se puede saber, y «?» cuenta como
# ocupado. LG-25c: antes, sin `powershell` en el PATH se caía a `pgrep`, que
# no existe en Git Bash, y `|| echo 0` decía «0 procesos» siempre: una
# instancia así robó como huérfano el cerrojo de un `cargo check` en marcha.
procesos_ajenos() {
  if [ -n "${SEMAFORO_PROCESOS:-}" ]; then
    echo "$SEMAFORO_PROCESOS"
    return
  fi
  local n="" ps listado codigo
  for ps in powershell powershell.exe pwsh pwsh.exe; do
    if command -v "$ps" > /dev/null 2>&1; then
      if listado=$("$ps" -NoProfile -Command \
        "(Get-Process cargo,rustc -ErrorAction SilentlyContinue | Measure-Object).Count" \
        2> /dev/null); then
        n=$(printf '%s' "$listado" | tr -d '\r ')
      else
        n=""
      fi
      break
    fi
  done
  if [ -z "$n" ] && command -v tasklist > /dev/null 2>&1; then
    # Un fallo de tasklist no es una observación de cero procesos.
    if listado=$(tasklist 2> /dev/null); then
      n=$(printf '%s\n' "$listado" | grep -ciE '^(cargo|rustc)\.exe')
    fi
  fi
  if [ -z "$n" ] && command -v pgrep > /dev/null 2>&1; then
    n=$(pgrep -c -x 'cargo|rustc' 2> /dev/null)
    codigo=$?
    # pgrep: 1 con cuenta 0 es «sin coincidencias»; 2/3 son errores.
    case "$codigo:$n" in
      0:* | 1:0) ;;
      *) n="" ;;
    esac
  fi
  case "$n" in
    '' | *[!0-9]*) echo "?" ;;
    *) echo "$n" ;;
  esac
}

ram_libre_gb() {
  if [ -n "${SEMAFORO_RAM_GB:-}" ]; then
    echo "$SEMAFORO_RAM_GB"
    return
  fi
  if command -v powershell > /dev/null 2>&1; then
    powershell -NoProfile -Command \
      "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB, 2)" \
      2> /dev/null | tr -d '\r '
  else
    awk '/MemAvailable/ {printf "%.2f", $2/1048576}' /proc/meminfo
  fi
}

menor_que() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a < b) }'; }
ahora() { date +%s; }
registrar() { printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$agente" "$*" >> "$bitacora"; }

cerrojo_huerfano() {
  [ -d "$cerrojo" ] || return 1
  local creado edad
  creado=$(cat "$cerrojo/creado" 2> /dev/null || echo 0)
  edad=$(( $(ahora) - creado ))
  if [ "$edad" -ge $(( ttl_min * 60 )) ]; then
    return 0
  fi
  # Gracia: quien acaba de tomar el turno aún no arrancó cargo (0 procesos);
  # sin ella dos agentes que ven 0 a la vez se roban el cerrojo.
  if [ "$edad" -lt "$gracia_s" ]; then
    return 1
  fi
  # Pasada la gracia, sin ningún cargo/rustc vivo nadie compila con él.
  [ "$(procesos_ajenos)" = "0" ]
}

inicio=$(ahora)
limite=$(( inicio + espera_max_min * 60 ))
while :; do
  ajenos=$(procesos_ajenos)
  ram=$(ram_libre_gb)
  motivo=""
  paciencia_agotada=0
  # LG-25b sólo puede ignorar una cuenta numérica conocida, nunca «?».
  if [[ "$ajenos" =~ ^[0-9]+$ ]] \
    && [ $(( $(ahora) - inicio )) -ge $(( paciencia_min * 60 )) ] \
    && ! menor_que "${ram:-0}" "$(awk -v a="$ram_min" -v b="$ram_extra" 'BEGIN { print a + b }')"; then
    paciencia_agotada=1
  fi
  if [ "${ajenos:-0}" != "0" ] && [ "$paciencia_agotada" = "0" ]; then
    motivo="$ajenos procesos cargo/rustc ajenos"
  elif menor_que "${ram:-0}" "$ram_min"; then
    motivo="RAM libre ${ram} GB < ${ram_min} GB"
  else
    if cerrojo_huerfano; then
      registrar "cerrojo huérfano liberado (dueño: $(cat "$cerrojo/dueno" 2> /dev/null))"
      rm -rf "$cerrojo"
    fi
    if mkdir "$cerrojo" 2> /dev/null; then
      ahora > "$cerrojo/creado"
      ficha="$agente pid=$$ t=$(ahora) cargo $*"
      printf '%s\n' "$ficha" > "$cerrojo/dueno"
      break
    fi
    motivo="cerrojo de $(cat "$cerrojo/dueno" 2> /dev/null || echo '?')"
  fi
  if [ "$(ahora)" -ge "$limite" ]; then
    echo "[semaforo] sin turno tras ${espera_max_min} min: $motivo; no compilo" >&2
    registrar "SIN TURNO ($motivo): cargo $*"
    exit 75
  fi
  echo "[semaforo] esperando: $motivo" >&2
  sleep "$pausa_s"
done

# LG-25c: sólo se libera el cerrojo PROPIO. Si otro lo declaró huérfano y lo
# tomó mientras este cargo corría, borrarlo al salir soltaría el turno ajeno.
liberar() {
  if [ "$(cat "$cerrojo/dueno" 2> /dev/null)" = "$ficha" ]; then
    rm -rf "$cerrojo"
  fi
}
trap liberar EXIT INT TERM
if [ -n "${SEMAFORO_PRUEBA_ROBO:-}" ]; then
  # Costura de prueba: otro agente toma el cerrojo mientras «compilamos».
  printf '%s\n' "$SEMAFORO_PRUEBA_ROBO" > "$cerrojo/dueno"
fi
if [ "${ajenos:-0}" != "0" ]; then
  registrar "turno por paciencia: ${ajenos} procesos cargo/rustc ajenos (fuera del semáforo), ram_libre=${ram}GB"
fi
registrar "ADQUIERE ram_libre=${ram}GB jobs=$CARGO_BUILD_JOBS: cargo $*"

if [ "${SEMAFORO_SOLO_DECIDIR:-0}" = "1" ]; then
  echo "[semaforo] turno concedido; CARGO_TARGET_DIR=$CARGO_TARGET_DIR CARGO_BUILD_JOBS=$CARGO_BUILD_JOBS CARGO_UNSTABLE_CHECKSUM_FRESHNESS=$CARGO_UNSTABLE_CHECKSUM_FRESHNESS"
  registrar "LIBERA (solo decidir)"
  exit 0
fi

# RB-H1 (Codex RB, 2026-10-19) — barrido automático del target/ compartido.
# target/ llegó a 77 GB (27,7 GB de incremental por churn de variantes y
# 30,9 GB de variantes superadas en deps; ver scripts/barrido_target.sh) porque
# la higiene dependía de que alguien la ejecutara a mano. Aquí corre sola: como
# mucho cada SEMAFORO_BARRIDO_H horas (6), con el turno YA tomado (nadie más
# compila por el semáforo) y sólo si la última cuenta de cargo/rustc fue 0 (con
# turno por paciencia hay builds ajenos vivos: no se barre). Nunca bloquea ni
# altera el build: cualquier fallo sólo se anota. Desactivable con
# SEMAFORO_BARRIDO=0. No corre con SEMAFORO_SOLO_DECIDIR (decide, no actúa).
barrido_si_toca() {
  [ "${SEMAFORO_BARRIDO:-1}" = "1" ] || return 0
  [ "${ajenos:-?}" = "0" ] || return 0
  local sello="$raiz/target/.barrido_ultimo" ultimo=0 cada_h=${SEMAFORO_BARRIDO_H:-6}
  local dir_scripts=. salida codigo_b fuente=${BASH_SOURCE[0]//\\//}
  if [ -f "$sello" ]; then ultimo=$(cat "$sello" 2> /dev/null || echo 0); fi
  case "$ultimo" in '' | *[!0-9]*) ultimo=0 ;; esac
  case "$cada_h" in '' | *[!0-9]*) cada_h=6 ;; esac
  [ $(( $(ahora) - ultimo )) -ge $(( cada_h * 3600 )) ] || return 0
  # Lanzado desde PowerShell con ruta Windows (C:\…\scripts\cargo_semaforo.sh)
  # BASH_SOURCE no trae «/»: se normalizan las barras antes de cortar.
  case "$fuente" in */*) dir_scripts=${fuente%/*} ;; esac
  if [ ! -f "$dir_scripts/barrido_target.sh" ]; then
    registrar "barrido omitido: no encuentro $dir_scripts/barrido_target.sh"
    return 0
  fi
  echo "[semaforo] barrido automático de target/ (RB-H1, cada ${cada_h} h)..." >&2
  salida=$(APLICAR=1 BARRIDO_RAIZ="$raiz" BARRIDO_CERROJO_PROPIO=1 BARRIDO_FICHA="$ficha" \
    BARRIDO_AGENTE="$agente" "$BASH" "$dir_scripts/barrido_target.sh" 2>&1)
  codigo_b=$?
  if [ "$codigo_b" = "0" ]; then ahora > "$sello"; fi
  echo "[semaforo] ${salida##*$'\n'}" >&2
  registrar "barrido automático codigo=$codigo_b: ${salida##*$'\n'}"
}
barrido_si_toca

cargo "$@"
codigo=$?
registrar "LIBERA codigo=$codigo duracion=$(( $(ahora) - inicio ))s: cargo $*"
exit "$codigo"
