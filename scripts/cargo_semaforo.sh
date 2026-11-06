#!/usr/bin/env bash
# LG-25 (Línea G) — semáforo de compilación compartido por todos los agentes.
#
# RÉGIMEN 2026-11-06 (mandato del operador: «cero restricciones de recursos»
# — acceso libre, total y sin restricciones a toda la RAM, todos los núcleos
# de CPU y todo el SSD; quedan eliminados los topes artificiales de RAM y de
# hilos de compilación, y las esperas por builds ajenos). Este envoltorio ya
# NO restringe recursos. Conserva sólo lo que protege la INTEGRIDAD del
# target/ compartido y la observabilidad:
#   1. target ÚNICO: CARGO_TARGET_DIR = <raíz del repo>/target, también desde
#      cualquier worktree (lo deduce de `git rev-parse --git-common-dir`).
#      RB-H2: es seguro entre worktrees SÓLO con la frescura por contenido de
#      `.cargo/config.toml` ([unstable] checksum-freshness = true): dos
#      worktrees escriben los mismos artefactos y, por mtime, uno puede
#      ejecutar el código del otro (scripts/tests/test_frescura_target.py).
#   2. Sin tope de hilos: CARGO_BUILD_JOBS sólo se exporta si el agente lo
#      pide; por defecto cargo usa el 100 % de los núcleos de la CPU.
#   3. RAM SIN puerta: la RAM disponible se MIDE y se anota como telemetría
#      (F1-097: AvailableMBytes en Windows / MemAvailable en Linux — la
#      memoria que el sistema puede entregar sin paginar), pero nunca bloquea
#      por defecto. SEMAFORO_RAM_MIN_GB queda como tope OPT-IN (default 0)
#      para quien lo declare explícitamente; una medición ilegible se anota
#      como `desconocida` y NO bloquea (ya no hay puerta que gobernar).
#   4. Cerrojo atómico (mkdir) en <raíz>/target/.semaforo.lock con su dueño:
#      un solo escritor del target a la vez (además del file-lock interno de
#      cargo). Un cerrojo es huérfano si supera SEMAFORO_TTL_MIN (240), o si
#      pasada la gracia SEMAFORO_GRACIA_S (120 s) no hay ningún cargo/rustc
#      vivo: un agente sin cuota no bloquea a nadie y dos agentes no se roban
#      el turno (LG-25c). Los cargo/rustc ajenos (de quien compila por fuera
#      del semáforo) ya NO bloquean: el file-lock de cargo serializa el
#      acceso real al target.
#   5. Una línea por adquisición y liberación en <raíz>/logs/semaforo.log
#      (agente, comando, RAM libre, duración).
#   6. RB-H1 — barrido automático del target/ compartido por PRESUPUESTO
#      (scripts/barrido_target.sh): ahora APAGADO por defecto (el mandato
#      prohíbe detenerse en auditorías de tamaño de carpeta);
#      SEMAFORO_BARRIDO=1 lo reactiva, cada SEMAFORO_BARRIDO_H (6) horas,
#      con el turno ya tomado y 0 cargo/rustc vivos. Nunca bloquea el build.
#
# Uso:   bash scripts/cargo_semaforo.sh <argumentos de cargo>
#        p. ej. bash scripts/cargo_semaforo.sh check --workspace --all-targets
# Entorno: SEMAFORO_AGENTE (quién eres), SEMAFORO_RAM_MIN_GB (0 = sin tope,
#          OPT-IN), SEMAFORO_TTL_MIN, SEMAFORO_ESPERA_MAX_MIN (60 de espera
#          del cerrojo; al agotarla sale con 75 sin compilar),
#          SEMAFORO_PAUSA_S (15), SEMAFORO_SOLO_DECIDIR=1 (imprime la decisión
#          y el entorno y sale sin compilar), SEMAFORO_BARRIDO (0 por defecto).
# Costuras de prueba (scripts/tests/test_cargo_semaforo.py): SEMAFORO_RAIZ,
#          SEMAFORO_PROCESOS (fuerza el nº de cargo/rustc ajenos),
#          SEMAFORO_RAM_GB (fuerza la RAM libre medida).
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
# RÉGIMEN 2026-11-06: tope de RAM OPT-IN (default 0 = sin puerta). La RAM se
# mide y se anota, pero ya no gobierna el turno salvo petición explícita.
ram_min=${SEMAFORO_RAM_MIN_GB:-0}
ttl_min=${SEMAFORO_TTL_MIN:-240}
gracia_s=${SEMAFORO_GRACIA_S:-120}
espera_max_min=${SEMAFORO_ESPERA_MAX_MIN:-60}
pausa_s=${SEMAFORO_PAUSA_S:-15}

export CARGO_TARGET_DIR="$raiz/target"
# Sin tope de hilos por defecto (mandato 2026-11-06): cargo usa el 100 % de
# los núcleos salvo que el agente pida CARGO_BUILD_JOBS explícitamente.
[ -n "${CARGO_BUILD_JOBS:-}" ] && export CARGO_BUILD_JOBS
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
  local salida
  if command -v powershell > /dev/null 2>&1; then
    # F1-097 — la MISMA magnitud que MemAvailable de Linux: memoria que el
    # sistema puede entregar sin paginar (libre + standby reclamable).
    # `Win32_PerfRawData_PerfOS_Memory` expone propiedades CIM en inglés, así
    # que no depende del idioma de Windows (el contador `\Memory\...` de
    # Get-Counter sí está localizado: en un Windows en español se llama
    # `\Memoria\MB disponibles`).
    salida=$(powershell -NoProfile -Command \
      "[math]::Round((Get-CimInstance Win32_PerfRawData_PerfOS_Memory).AvailableMBytes/1024, 2)" \
      2> /dev/null | tr -d '\r ')
  else
    salida=$(awk '/MemAvailable/ {printf "%.2f", $2/1048576}' /proc/meminfo)
  fi
  # Un valor ilegible NO es 0: se declara desconocido y el bucle decide con
  # esa información (fail-closed, pero con el motivo a la vista).
  if ram_conocida "$salida"; then
    echo "$salida"
  else
    echo "desconocida"
  fi
}

ram_conocida() {
  case "$1" in
    '' | *[!0-9.]* | *.*.*) return 1 ;;
    *) return 0 ;;
  esac
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
  ram=$(ram_libre_gb)
  # Etiqueta de bitácora: una medición ilegible se registra como
  # `ram_libre=desconocida`, nunca como `ram_libre=0GB`.
  if ram_conocida "$ram"; then ram_log="${ram}GB"; else ram_log="desconocida"; fi
  motivo=""
  # RÉGIMEN 2026-11-06: ni la RAM ni los cargo/rustc ajenos bloquean. El
  # tope SEMAFORO_RAM_MIN_GB es OPT-IN (default 0) y sólo decide con una
  # medición CONOCIDA; una lectura ilegible es telemetría (`desconocida`),
  # no puerta. Del cerrojo se encarga la rama else.
  if ram_conocida "$ram" && menor_que "$ram" "$ram_min"; then
    motivo="RAM libre ${ram} GB < ${ram_min} GB (tope OPT-IN de este agente)"
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
    registrar "SIN TURNO (ram_libre=${ram_log} | $motivo): cargo $*"
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
registrar "ADQUIERE ram_libre=${ram_log} jobs=${CARGO_BUILD_JOBS:-todos}: cargo $*"

if [ "${SEMAFORO_SOLO_DECIDIR:-0}" = "1" ]; then
  echo "[semaforo] turno concedido; CARGO_TARGET_DIR=$CARGO_TARGET_DIR CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-todos} CARGO_UNSTABLE_CHECKSUM_FRESHNESS=$CARGO_UNSTABLE_CHECKSUM_FRESHNESS"
  registrar "LIBERA (solo decidir)"
  exit 0
fi

# RB-H1 (Codex RB, 2026-10-19) — barrido automático del target/ compartido.
# target/ llegó a 77 GB (27,7 GB de incremental por churn de variantes y
# 30,9 GB de variantes superadas en deps; ver scripts/barrido_target.sh).
# RÉGIMEN 2026-11-06: APAGADO por defecto — el mandato del operador prohíbe
# detenerse en auditorías de tamaño de carpeta. SEMAFORO_BARRIDO=1 lo
# reactiva: como mucho cada SEMAFORO_BARRIDO_H horas (6), con el turno YA
# tomado y sólo si la cuenta de cargo/rustc es 0 (con builds ajenos vivos no
# se barre). Nunca bloquea ni altera el build: cualquier fallo sólo se anota.
# No corre con SEMAFORO_SOLO_DECIDIR (decide, no actúa).
barrido_si_toca() {
  [ "${SEMAFORO_BARRIDO:-0}" = "1" ] || return 0
  local ajenos
  ajenos=$(procesos_ajenos)
  [ "$ajenos" = "0" ] || return 0
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
  # LG-36: latido. El barrido no tiene cargo/rustc vivo y sólo renueva `creado`
  # mientras borra; en sus enumeraciones largas otro agente veía el cerrojo con
  # más de la gracia y 0 procesos, lo declaraba huérfano y compilaba a la vez.
  # Mientras el turno siga siendo propio, `creado` se renueva cada cuarto de gracia.
  local latido=$(( gracia_s / 4 ))
  [ "$latido" -ge 1 ] || latido=1
  (
    while sleep "$latido"; do
      [ "$(cat "$cerrojo/dueno" 2> /dev/null)" = "$ficha" ] || exit 0
      ahora > "$cerrojo/creado"
    done
  ) &
  local pid_latido=$!
  salida=$(APLICAR=1 BARRIDO_RAIZ="$raiz" BARRIDO_CERROJO_PROPIO=1 BARRIDO_FICHA="$ficha" \
    BARRIDO_AGENTE="$agente" "$BASH" "$dir_scripts/barrido_target.sh" 2>&1)
  codigo_b=$?
  kill "$pid_latido" 2> /dev/null
  wait "$pid_latido" 2> /dev/null
  if [ "$codigo_b" = "0" ]; then ahora > "$sello"; fi
  echo "[semaforo] ${salida##*$'\n'}" >&2
  registrar "barrido automático codigo=$codigo_b: ${salida##*$'\n'}"
}
barrido_si_toca
if [ -n "${SEMAFORO_PRUEBA_ROBO_BARRIDO:-}" ]; then
  # Costura de prueba: otro agente toma el cerrojo durante el barrido.
  printf '%s\n' "$SEMAFORO_PRUEBA_ROBO_BARRIDO" > "$cerrojo/dueno"
fi
# LG-36: si el turno dejó de ser propio (otro lo declaró huérfano durante el
# barrido), no se compila: dos builds a la vez es lo que el semáforo impide.
if [ "$(cat "$cerrojo/dueno" 2> /dev/null)" != "$ficha" ]; then
  echo "[semaforo] el turno se perdió durante el barrido; no compilo" >&2
  registrar "TURNO PERDIDO durante el barrido (dueño: $(cat "$cerrojo/dueno" 2> /dev/null || echo '?')): cargo $*"
  exit 75
fi
ahora > "$cerrojo/creado"

cargo "$@"
codigo=$?
registrar "LIBERA codigo=$codigo duracion=$(( $(ahora) - inicio ))s: cargo $*"
exit "$codigo"
