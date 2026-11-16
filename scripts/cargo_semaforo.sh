#!/usr/bin/env bash
# LG-25 (Línea G) — semáforo de compilación compartido por todos los agentes.
#
# RÉGIMEN 2026-11-06 (mandato del operador: «cero restricciones de recursos»
# — acceso libre, total y sin restricciones a toda la RAM, todos los núcleos
# de CPU y todo el SSD; quedan eliminados los topes artificiales de RAM y de
# hilos de compilación, y las esperas por builds ajenos). Este envoltorio ya
# NO ESTRANGULA recursos de forma artificial: ni topes de RAM o de hilos ni
# esperas por builds ajenos (eso sigue ELIMINADO por aquel mandato).
# FA-048 (2026-11-14) AÑADE un PISO DE SEGURIDAD distinto de aquellos topes:
# no llenar el SSD. Es regla del operador (REGLAS_DE_TRABAJO.md:16, «40 GB»
# libres; reafirmada el 14 nov: «que mi portátil no se llene en la ssd»,
# CONTRATO:9). Con el disco libre bajo el umbral no se compila (punto 7).
# Nota de gobernanza: .agents/AGENTS.md (higiene_disco_y_memoria) dice «no
# medir espacio de disco ni usar este script»; REGLAS:15-16 lo manda. Ese
# conflicto lo resuelve el operador o el coordinador, no este archivo.
# Conserva lo que protege la INTEGRIDAD del target/ compartido, el piso de
# disco y la observabilidad:
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
#   7. FA-048 — guarda DISCO-BAJO (OPT-OUT, a diferencia de la RAM, que es
#      OPT-IN): mide el libre del sistema de archivos que contiene la raíz
#      (df -Pk, KiB) en GB DECIMALES (1 GB = 1e9 B; libre = KiB x 1024 / 1e9)
#      y lo compara con SEMAFORO_DISCO_MIN_GB (default 40, el «40 GB» de
#      REGLAS:16; REGISTRO_DE_FALLAS.md:55 dice «40 GiB» = 42,95 GB, aquí se
#      fija GB = 1e9 B). Va ANTES de la primera escritura (crear target/): si
#      libre < umbral sale con 78 sin target/, sin cerrojo y sin cargo; su
#      única escritura es la línea DISCO-BAJO en logs/semaforo.log. NO borra
#      nada. Medición fallida (df ausente, código != 0, salida ilegible) =
#      `desconocida`: NO bloquea, pero avisa por stderr y en la bitácora
#      (la guarda queda ciega). Al liberar el turno se anota el libre de
#      nuevo y, si quedó bajo el umbral, la línea lo dice (no bloquea).
#
# Uso:   bash scripts/cargo_semaforo.sh <argumentos de cargo>
#        p. ej. bash scripts/cargo_semaforo.sh check --workspace --all-targets
# Entorno: SEMAFORO_AGENTE (quién eres), SEMAFORO_RAM_MIN_GB (0 = sin tope,
#          OPT-IN), SEMAFORO_TTL_MIN, SEMAFORO_ESPERA_MAX_MIN (60 de espera
#          del cerrojo; al agotarla sale con 75 sin compilar),
#          SEMAFORO_PAUSA_S (15), SEMAFORO_SOLO_DECIDIR=1 (imprime la decisión
#          y el entorno y sale sin compilar), SEMAFORO_BARRIDO (0 por defecto),
#          SEMAFORO_DISCO_MIN_GB (40 GB decimales por defecto; límite del
#          operador, no se deriva), SEMAFORO_PDB (1 = conservar los .pdb en
#          dev/test; ver COMPV2-P3).
# Códigos de salida: 0/código de cargo, 2 (fuera del repo), 75 (sin turno),
#          78 (DISCO-BAJO, no compila).
# Costuras de prueba (scripts/tests/test_cargo_semaforo.py): SEMAFORO_RAIZ,
#          SEMAFORO_PROCESOS (fuerza el nº de cargo/rustc ajenos),
#          SEMAFORO_RAM_GB (fuerza la RAM libre medida),
#          SEMAFORO_DISCO_GB (fuerza el disco libre medido, en GB).
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
# COMPV2-P3 (integrador, 2026-11-16) — sin PDB en dev/test (Windows MSVC).
# P0 (`cargo -v` y `--print link-args`, nightly-2026-06-30): con debug = 0 y
# strip = "symbols" rustc igual pasa `/DEBUG /PDBALTPATH:%_PDB%` a rust-lld y
# cada exe de prueba y de build script deja su .pdb (12,6 GB medidos el 15
# nov). Ni -C strip=debuginfo|none ni -C debuginfo=0 lo cambian; sólo
# -C link-arg=/DEBUG:NONE (la última /DEBUG manda). Va aquí, no en
# .cargo/config.toml, porque allí también cambiaría release (que hoy sí deja
# PDB y no se toca), y las rustflags dentro de [profile] son inestables.
# La variable del triple se SUMA a las rustflags de la config (no las
# sustituye). No aplica con --release, -r, --profile distinto de dev/test ni
# a `cargo bench`. SEMAFORO_PDB=1 la desactiva (quien necesite símbolos).
sin_pdb=no
if [ "${SEMAFORO_PDB:-0}" != "1" ]; then
  sin_pdb=si
  previo="" subcomando=""
  for arg in "$@"; do
    [ "$arg" = "--" ] && break
    if [ -z "$subcomando" ]; then
      case "$arg" in +* | -*) ;; *) subcomando=$arg ;; esac
    fi
    if [ "$previo" = "--profile" ]; then
      case "$arg" in dev | test) ;; *) sin_pdb=no ;; esac
    fi
    case "$arg" in
      --release | -r) sin_pdb=no ;;
      --profile=*) case "${arg#--profile=}" in dev | test) ;; *) sin_pdb=no ;; esac ;;
    esac
    previo=$arg
  done
  [ "$subcomando" = "bench" ] && sin_pdb=no
fi
if [ "$sin_pdb" = "si" ]; then
  CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUSTFLAGS="${CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUSTFLAGS:+$CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUSTFLAGS }-C link-arg=/DEBUG:NONE"
  export CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUSTFLAGS
fi
bitacora="$raiz/logs/semaforo.log"
registrar() { printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$agente" "$*" >> "$bitacora"; }

# FA-048 — guarda DISCO-BAJO (punto 7 de la cabecera). Unidad: GB decimales
# (1 GB = 1e9 B). El umbral es el límite de riesgo del operador (REGLAS:16):
# no se deriva ni se cambia aquí.
disco_min_defecto=40
disco_min=${SEMAFORO_DISCO_MIN_GB:-$disco_min_defecto}

# FA-048 v3 — comparación EXACTA, entero contra entero, sin coma flotante.
# Un valor en GB decimales («40», «39.996», «.5») se pasa a nanogigabytes
# (1 nGB = 1 B): parte entera * 10^9 + decimales rellenados a 9 cifras, en
# base 10 forzada (10#) para que un «08» no se lea en octal. Se aceptan como
# mucho 9 cifras enteras (sin desbordar 64 bits) y 9 decimales (la precisión
# del exacto de df); cualquier otra forma no es un valor: no imprime nada y
# falla. Antes awk comparaba en coma flotante y «39.999999999999999» se leía
# igual que 40.
a_nano_gb() {
  local v=$1 ent frac=
  case "$v" in
    *.*) ent=${v%%.*}; frac=${v#*.} ;;
    *) ent=$v ;;
  esac
  case "$ent$frac" in
    '' | *[!0-9]*) return 1 ;;
  esac
  [ "${#ent}" -le 9 ] && [ "${#frac}" -le 9 ] || return 1
  frac="${frac}000000000"
  echo $(( 10#${ent:-0} * 1000000000 + 10#${frac:0:9} ))
}

# ¿Lectura conocida? (primer campo de disco_libre_gb: un entero de nGB)
disco_conocido() {
  case "$1" in
    '' | *[!0-9]*) return 1 ;;
    *) return 0 ;;
  esac
}

if ! disco_min_nano=$(a_nano_gb "$disco_min"); then
  echo "[semaforo] SEMAFORO_DISCO_MIN_GB ilegible ('$disco_min'); uso el umbral por defecto ${disco_min_defecto} GB" >&2
  disco_min=$disco_min_defecto
  disco_min_nano=$(a_nano_gb "$disco_min")
fi

# Libre del sistema de archivos que contiene la raíz, en GB decimales, o
# «desconocida» ante CUALQUIER fallo de la medición (df ausente, código
# distinto de 0, salida ilegible). Sólo lee: no escribe nada.
# Imprime DOS campos: «<exacto> <para mostrar>». El exacto es el que se
# compara con el umbral, en nanogigabytes enteros (= bytes): con df son los
# bytes exactos (KiB * 1024, aritmética entera); con la costura
# SEMAFORO_DISCO_GB, su valor pasado por a_nano_gb (más de 9 decimales o mal
# formada = «desconocida»). El de mostrar es el exacto redondeado a 2
# decimales (df) o la costura tal cual: sólo para mensajes y bitácora. Antes
# se comparaba el redondeado y 39,996 GB («40.00») pasaba.
disco_libre_gb() {
  local salida kib bytes exacto nano
  if [ -n "${SEMAFORO_DISCO_GB+x}" ]; then
    if nano=$(a_nano_gb "$SEMAFORO_DISCO_GB"); then
      echo "$nano $SEMAFORO_DISCO_GB"
    else
      echo "desconocida"
    fi
    return
  fi
  if ! command -v df > /dev/null 2>&1; then
    echo "desconocida"
    return
  fi
  if ! salida=$(df -Pk "$raiz" 2> /dev/null); then
    echo "desconocida"
    return
  fi
  # df -P: una línea por sistema de archivos; la 4.ª columna es «Available»
  # en bloques de 1024 B (-k).
  kib=$(printf '%s\n' "$salida" | awk 'NR == 2 { print $4 }')
  # Más de 15 cifras de KiB desbordaría la aritmética entera de 64 bits:
  # se trata como salida ilegible (fail-open visible, igual que la basura).
  case "$kib" in
    '' | *[!0-9]* | ????????????????*) echo "desconocida" ;;
    *)
      bytes=$(( 10#$kib * 1024 ))
      exacto=$(printf '%d.%09d' $(( bytes / 1000000000 )) $(( bytes % 1000000000 )))
      echo "$bytes $(awk -v g="$exacto" 'BEGIN { printf "%.2f", g }')"
      ;;
  esac
}

disco_bajo() { (( 10#$1 < 10#$2 )); }

read -r disco disco_ver <<< "$(disco_libre_gb)"
if disco_conocido "$disco"; then
  disco_log="${disco_ver}GB"
  if disco_bajo "$disco" "$disco_min_nano"; then
    linea="DISCO-BAJO: disco libre ${disco_ver} GB < umbral ${disco_min} GB (GB decimal, 1 GB = 1e9 B); no compilo: sin target/, sin cerrojo, sin cargo, nada borrado: cargo $*"
    echo "[semaforo] $linea" >&2
    # Única escritura permitida antes de salir: la línea en logs/ (no target/).
    mkdir -p "$raiz/logs"
    registrar "$linea"
    exit 78
  fi
else
  disco_log="desconocida"
  disco_ver="desconocida"
  echo "[semaforo] disco libre desconocido; guarda DISCO-BAJO no aplicada" >&2
fi

mkdir -p "$raiz/target" "$raiz/logs"
cerrojo="$raiz/target/.semaforo.lock"
if [ "$disco_log" = "desconocida" ]; then
  registrar "disco libre desconocido; guarda DISCO-BAJO no aplicada (umbral ${disco_min} GB): cargo $*"
fi

# FA-047 — observabilidad al liberar: el libre de disco en ese momento y, si
# quedó bajo el umbral, dicho en la línea. No bloquea lo ya compilado.
disco_al_liberar() {
  local d d_ver
  read -r d d_ver <<< "$(disco_libre_gb)"
  if disco_conocido "$d"; then
    if disco_bajo "$d" "$disco_min_nano"; then
      echo "disco_libre=${d_ver}GB DISCO-BAJO al liberar (< ${disco_min} GB)"
    else
      echo "disco_libre=${d_ver}GB"
    fi
  else
    echo "disco_libre=desconocida"
  fi
}

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
registrar "ADQUIERE ram_libre=${ram_log} disco_libre=${disco_log} jobs=${CARGO_BUILD_JOBS:-todos}: cargo $*"

if [ "${SEMAFORO_SOLO_DECIDIR:-0}" = "1" ]; then
  echo "[semaforo] turno concedido; CARGO_TARGET_DIR=$CARGO_TARGET_DIR CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-todos} CARGO_UNSTABLE_CHECKSUM_FRESHNESS=$CARGO_UNSTABLE_CHECKSUM_FRESHNESS DISCO_LIBRE_GB=${disco_ver} DISCO_MIN_GB=${disco_min} (GB decimal, 1 GB = 1e9 B) SIN_PDB=${sin_pdb} CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUSTFLAGS=${CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUSTFLAGS:-}"
  registrar "LIBERA (solo decidir) $(disco_al_liberar)"
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
registrar "LIBERA codigo=$codigo duracion=$(( $(ahora) - inicio ))s $(disco_al_liberar): cargo $*"
exit "$codigo"
