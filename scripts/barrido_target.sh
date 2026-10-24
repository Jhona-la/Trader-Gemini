#!/usr/bin/env bash
# RB-H1 (Codex RB, 2026-10-19) — barrido del target/ compartido por PRESUPUESTO.
#
# Por qué (medido el 2026-10-19 sobre el checkout raíz): target/ pesaba 77 GB
# (37 GB el 17-10 según la auditoría de Línea G). higiene_recursos.sh sólo
# borra target/debug/deps/*.exe con más de 3 días y no veía las dos causas:
#   - target/debug/incremental: 27,7 GB en 2.710 carpetas crate-hash, TODAS
#     con menos de 2 días. Cada test de integración es un crate con su caché
#     incremental y cada combinación de features/flags de cada agente es otro
#     hash: el disco se llena por CHURN, sin envejecer, así que ningún umbral
#     de edad lo detecta. Hace falta un presupuesto.
#   - variantes superadas en deps/: 30,9 GB de los 37,6 de debug/deps (82 %)
#     eran versiones NO más recientes de un mismo artefacto (syn: 57
#     variantes; windows_implement: 67). El mtime de deps dice cuándo se
#     compiló, no cuándo se usó: barrer por edad borraría rlib vigentes y
#     forzaría a recompilar todo. Se conserva la variante más reciente.
#
# Política (sólo dentro de <raíz>/target; nunca data/, models/, .git ni
# worktrees):
#   1. incremental (debug y release): si su tamaño supera
#      BARRIDO_INCREMENTAL_MAX_MB (6144), se borran las carpetas crate-hash
#      MENOS recientemente modificadas hasta bajar a BARRIDO_HISTERESIS (0,75)
#      del presupuesto (LRU): sobrevive la caché del crate que alguien está
#      iterando y se van las variantes abandonadas. Caché regenerable.
#   2. deps (debug y release), RB-H1d: sólo si su tamaño total supera
#      BARRIDO_DEPS_MAX_MB (40960; medido el 19-10: deps entero pesaba 13,9 GB
#      tras el primer barrido, y 37,6 GB sin barrer). Entonces, de cada
#      artefacto <nombre>-<hash16><ext> con varias variantes se conserva la MÁS
#      RECIENTE
#      y, entre las demás que sean estrictamente más viejas y tengan más de
#      BARRIDO_VARIANTES_H (24) horas, se borran de la más vieja a la más nueva
#      hasta bajar a BARRIDO_HISTERESIS del presupuesto. Sin presión de disco no
#      se toca nada: `check` y `test` compilan variantes distintas del mismo
#      crate (otro perfil u otras features) y el mtime es de compilación, no de
#      uso, así que borrarlas sin necesidad forzaría recompilaciones (medido el
#      19-10: un `check` recompiló `syn` tras el barrido automático). Si un
#      build vuelve a pedir una variante borrada, cargo la recompila.
#   Orden de ejecución: primero deps (listado de segundos) y después la caché
#   incremental (su listado recorre cada archivo: minutos con ~3.000
#   carpetas), para aprovechar cada ventana sin builds ajenos.
#   3. Exclusión con los builds: con algún cargo/rustc vivo, o con la cuenta
#      desconocida («?», el criterio fail-closed de LG-25c), NO se borra nada.
#      Toma el cerrojo del semáforo (<raíz>/target/.semaforo.lock, protocolo
#      mkdir de LG-25) y renueva su `creado` mientras borra, para que ningún
#      semáforo lo declare huérfano a mitad del borrado; antes de cada lote
#      comprueba que el cerrojo sigue siendo suyo y que no apareció ningún
#      build: si no, para en seco. Llamado por el propio semáforo con el turno
#      ya tomado: BARRIDO_CERROJO_PROPIO=1 y BARRIDO_FICHA=<ficha del turno>.
#
# Uso:   bash scripts/barrido_target.sh            (simulación: mide y lista)
#        APLICAR=1 bash scripts/barrido_target.sh  (ejecuta)
# Entorno: BARRIDO_INCREMENTAL_MAX_MB, BARRIDO_DEPS_MAX_MB, BARRIDO_HISTERESIS,
#          BARRIDO_VARIANTES_H,
#          BARRIDO_LOTE (50 rutas por llamada a rm), BARRIDO_CHEQUEO_S (15),
#          BARRIDO_AGENTE (quién eres; va a logs/higiene.log).
# Costuras de prueba (scripts/tests/test_barrido_target.py): BARRIDO_RAIZ,
#          BARRIDO_PROCESOS (si no, SEMAFORO_PROCESOS), BARRIDO_PRUEBA_ROBO.
# Salida: 0 hecho o simulado; 5 omitido o detenido (builds vivos, cuenta
#         desconocida, cerrojo ajeno o robado); 2 fuera del repositorio,
#         parámetros inválidos, fallo de enumeración/borrado o GNU ausente. La última línea
#         impresa es el resumen (el semáforo la anota en su bitácora).
set -uo pipefail

raiz=${BARRIDO_RAIZ:-}
if [ -z "$raiz" ]; then
  comun=$(git rev-parse --path-format=absolute --git-common-dir 2> /dev/null) || {
    echo "[barrido] no estoy dentro del repositorio" >&2
    exit 2
  }
  raiz=${comun%/*}
fi
# Barras normales: las herramientas MSYS aceptan C:/… y así los prefijos de
# ruta comparan igual venga la raíz de git o de Windows.
raiz=${raiz//\\//}
raiz=${raiz%/}

aplicar=${APLICAR:-0}
max_mb=${BARRIDO_INCREMENTAL_MAX_MB:-6144}
deps_max_mb=${BARRIDO_DEPS_MAX_MB:-40960}
histeresis=${BARRIDO_HISTERESIS:-0.75}
variantes_h=${BARRIDO_VARIANTES_H:-24}
lote=${BARRIDO_LOTE:-50}
chequeo_s=${BARRIDO_CHEQUEO_S:-15}
agente=${BARRIDO_AGENTE:-${SEMAFORO_AGENTE:-${USER:-${USERNAME:-agente}}}}
target="$raiz/target"
cerrojo="$target/.semaforo.lock"
bitacora="$raiz/logs/higiene.log"

# ---------- validación: ante la duda, no se borra nada ----------
es_decimal() { [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]]; }
es_entero_pos() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }
if ! es_decimal "$max_mb" || ! es_decimal "$deps_max_mb" || ! es_decimal "$variantes_h" \
  || ! es_decimal "$histeresis" || ! es_entero_pos "$lote" || ! [[ "$chequeo_s" =~ ^[0-9]+$ ]]; then
  echo "[barrido] parámetros inválidos (max_mb=$max_mb deps_max_mb=$deps_max_mb histeresis=$histeresis" \
    "variantes_h=$variantes_h lote=$lote chequeo_s=$chequeo_s): no barro" >&2
  exit 2
fi
# Windows trae FIND.EXE y SORT.EXE homónimos en System32: en MSYS, las de GNU
# viven en /usr/bin y deben ganar.
for herramienta in find awk sort; do
  version=$("$herramienta" --version 2> /dev/null)
  if [[ "$version" != *GNU* ]] && [ -d /usr/bin ]; then
    PATH="/usr/bin:/bin:$PATH"
    break
  fi
done
for herramienta in find awk sort; do
  # Capturar antes de comparar: con pipefail, `| grep -q` puede dar un falso
  # negativo por SIGPIPE cuando grep sale antes que el productor.
  version=$("$herramienta" --version 2> /dev/null)
  if [[ "$version" != *GNU* ]]; then
    echo "[barrido] falta $herramienta de GNU en el PATH: no barro" >&2
    exit 2
  fi
done
if ! awk -v h="$histeresis" 'BEGIN { exit !(h > 0 && h <= 1) }'; then
  echo "[barrido] BARRIDO_HISTERESIS debe estar en (0, 1]: no barro" >&2
  exit 2
fi

ahora() { date +%s; }
registrar() {
  mkdir -p "$raiz/logs" 2> /dev/null
  printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$agente" "$*" >> "$bitacora"
}
mb() { awk -v b="$1" 'BEGIN { printf "%.1f", b / 1048576 }'; }

# Cuántos cargo/rustc hay vivos; «?» si no se puede saber (cuenta como
# ocupado). Mismo criterio que procesos_ajenos() de cargo_semaforo.sh.
procesos_build() {
  local costura=${BARRIDO_PROCESOS:-${SEMAFORO_PROCESOS:-}}
  if [ -n "$costura" ]; then
    echo "$costura"
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
    if listado=$(tasklist 2> /dev/null); then
      n=$(printf '%s\n' "$listado" | grep -ciE '^(cargo|rustc)\.exe')
    fi
  fi
  if [ -z "$n" ] && command -v pgrep > /dev/null 2>&1; then
    n=$(pgrep -c -x 'cargo|rustc' 2> /dev/null)
    codigo=$?
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

# ---------- cerrojo del semáforo ----------
ficha_propia=""
cerrojo_tomado=0
sigue_siendo_mio() { [ -n "$ficha_propia" ] && [ "$(cat "$cerrojo/dueno" 2> /dev/null)" = "$ficha_propia" ]; }
tomar_cerrojo() {
  if [ "${BARRIDO_CERROJO_PROPIO:-0}" = "1" ]; then
    ficha_propia=${BARRIDO_FICHA:-}
    if ! sigue_siendo_mio; then
      echo "[barrido] el cerrojo no es del turno que me llama: no barro" >&2
      return 1
    fi
    return 0
  fi
  mkdir -p "$target" 2> /dev/null
  if mkdir "$cerrojo" 2> /dev/null; then
    ficha_propia="barrido $agente pid=$$ t=$(ahora)"
    ahora > "$cerrojo/creado"
    printf '%s\n' "$ficha_propia" > "$cerrojo/dueno"
    cerrojo_tomado=1
    return 0
  fi
  echo "[barrido] cerrojo de $(cat "$cerrojo/dueno" 2> /dev/null || echo '?'): no barro" >&2
  return 1
}
liberar() {
  # Sólo el cerrojo PROPIO (LG-25c): si otro lo tomó, no se le suelta.
  if [ "$cerrojo_tomado" = "1" ] && sigue_siendo_mio; then
    rm -rf -- "$cerrojo"
  fi
}
trap liberar EXIT INT TERM

ultimo_chequeo=0
chequeo_seguro() {
  if ! sigue_siendo_mio; then
    echo "[barrido] el cerrojo dejó de ser mío: paro" >&2
    return 1
  fi
  local t n
  t=$(ahora)
  if [ $((t - ultimo_chequeo)) -ge "$chequeo_s" ]; then
    n=$(procesos_build)
    if [ "$n" != "0" ]; then
      echo "[barrido] apareció un build ($n procesos cargo/rustc): paro" >&2
      return 1
    fi
    printf '%s\n' "$t" > "$cerrojo/creado" # renovación: nadie lo ve huérfano
    ultimo_chequeo=$t
  fi
  return 0
}

# ---------- cola de borrado por lotes ----------
abortado=0
error_operacion=0
pendientes=()
pend_bytes=0
liberado=0
borrados=0
vaciar() {
  [ "${#pendientes[@]}" -gt 0 ] || return 0
  if [ "$abortado" != "0" ]; then
    pendientes=()
    pend_bytes=0
    return 1
  fi
  if [ "$aplicar" = "1" ]; then
    if ! chequeo_seguro; then
      abortado=1
      pendientes=()
      pend_bytes=0
      return 1
    fi
    if ! rm -rf -- "${pendientes[@]}"; then
      echo "[barrido] falló el borrado: paro; el lote fallido no se contabiliza" >&2
      abortado=1
      error_operacion=1
      pendientes=()
      pend_bytes=0
      return 1
    fi
  fi
  liberado=$((liberado + pend_bytes))
  borrados=$((borrados + ${#pendientes[@]}))
  pendientes=()
  pend_bytes=0
}
encolar() { # encolar <ruta> <bytes>: sólo dentro de target/, nunca el cerrojo
  case "$1" in
    "$cerrojo" | "$cerrojo"/*) return 0 ;;
    "$target"/?*) ;;
    *)
      echo "[barrido] ruta fuera de target, ignorada: $1" >&2
      return 0
      ;;
  esac
  pendientes+=("$1")
  pend_bytes=$((pend_bytes + $2))
  if [ "${#pendientes[@]}" -ge "$lote" ]; then vaciar; fi
}

if [ ! -d "$target" ]; then
  echo "[barrido] sin $target: nada que barrer"
  exit 0
fi
echo "=== BARRIDO DE target/ ($([ "$aplicar" = 1 ] && echo APLICANDO || echo SIMULACIÓN)) ==="
echo "raíz: $raiz"

if [ "$aplicar" = "1" ]; then
  n=$(procesos_build)
  if [ "$n" != "0" ]; then
    echo "[barrido] $n procesos cargo/rustc vivos o cuenta desconocida: no barro" >&2
    echo "[barrido] omitido: builds vivos ($n)"
    registrar "BARRIDO OMITIDO procesos=$n"
    exit 5
  fi
  if ! tomar_cerrojo; then
    echo "[barrido] omitido: cerrojo ajeno"
    registrar "BARRIDO OMITIDO cerrojo ajeno"
    exit 5
  fi
  if [ -n "${BARRIDO_PRUEBA_ROBO:-}" ]; then
    # Costura de prueba: otro agente se queda con el cerrojo antes del borrado.
    printf '%s\n' "$BARRIDO_PRUEBA_ROBO" > "$cerrojo/dueno"
  fi
fi

# ---------- 1. deps por presupuesto (listado de segundos: va primero) ----------
# RB-H1d: sólo se barre por encima del presupuesto. Variantes «no más
# recientes» de un mismo nombre pueden seguir en uso (otro perfil u otras
# features: `cargo check` y `cargo test` compilan variantes distintas del mismo
# crate) y su mtime es de compilación, no de uso: borrarlas sin presión de
# disco obligaría a recompilarlas. Con presión, se van primero las MÁS VIEJAS
# (lo más probable: restos de worktrees retirados y fuentes antiguas).
echo ""
echo "--- 1. deps: presupuesto ${deps_max_mb} MB; por encima, variantes no más recientes con más de ${variantes_h} h (de la más vieja a la más nueva) hasta ×${histeresis} ---"
ahora_s=$(ahora)
total_deps=0
candidatas=()
for deps in "$target/debug/deps" "$target/release/deps"; do
  [ -d "$deps" ] || continue
  if ! inventario=$(
    find "$deps" -maxdepth 1 -type f -printf '%T@\t%s\t%f\n' 2> /dev/null \
      | awk -F'\t' -v ahora="$ahora_s" -v horas="$variantes_h" '
          BEGIN { limite = ahora - horas * 3600 }
          {
            total += $2
            if (!match($3, /^(.*)-([0-9a-f]{16})(\..*)?$/, a)) next
            c = a[1] SUBSEP a[3]
            k = ++n[c]; mt[c, k] = $1 + 0; sz[c, k] = $2; nm[c, k] = $3
          }
          END {
            printf "T\t%.0f\t-\t-\n", total
            for (c in n) {
              if (n[c] < 2) continue
              im = 1
              for (k = 2; k <= n[c]; k++) if (mt[c, k] > mt[c, im]) im = k
              for (k = 1; k <= n[c]; k++)
                if (k != im && mt[c, k] < mt[c, im] && mt[c, k] < limite)
                  printf "C\t%.0f\t%.0f\t%s\n", mt[c, k], sz[c, k], nm[c, k]
            }
          }'
  ); then
    echo "[barrido] falló la enumeración de $deps: no barro" >&2
    registrar "BARRIDO ERROR enumeracion=$deps codigo=2"
    exit 2
  fi
  while IFS=$'\t' read -r tipo c2 c3 c4; do
    case "$tipo" in
      T) total_deps=$((total_deps + c2)) ;;
      C) candidatas+=("$c2"$'\t'"$c3"$'\t'"$deps/$c4") ;;
    esac
  done <<< "$inventario"
done
max_deps_b=$(awk -v mb="$deps_max_mb" 'BEGIN { printf "%.0f", mb * 1048576 }')
obj_deps_b=$(awk -v mb="$deps_max_mb" -v h="$histeresis" 'BEGIN { printf "%.0f", mb * 1048576 * h }')
echo "tamaño: $(mb "$total_deps") MB  variantes superadas candidatas: ${#candidatas[@]}"
lib0=$liberado
bor0=$borrados
if [ "$total_deps" -gt "$max_deps_b" ]; then
  restante=$total_deps
  if ! ordenadas=$(printf '%s\n' "${candidatas[@]}" | sort -t $'\t' -k1,1n); then
    echo "[barrido] falló ordenar deps: no barro" >&2
    registrar "BARRIDO ERROR ordenar=deps codigo=2"
    exit 2
  fi
  while IFS=$'\t' read -r _ b ruta; do
    [ -n "$ruta" ] || continue
    [ "$restante" -gt "$obj_deps_b" ] || break
    encolar "$ruta" "$b"
    restante=$((restante - b))
    [ "$abortado" = "0" ] || break
  done <<< "$ordenadas"
  vaciar
fi
var_borradas=$((borrados - bor0))
var_bytes=$((liberado - lib0))
if [ "$total_deps" -gt "$max_deps_b" ]; then
  echo "$([ "$aplicar" = 1 ] && echo borradas || echo borraría) $var_borradas variantes ($(mb "$var_bytes") MB) → quedan $(mb $((total_deps - var_bytes))) MB"
else
  echo "dentro del presupuesto: nada que barrer"
fi

# ---------- 2. incremental por presupuesto (LRU) ----------
# Su listado recorre todos los archivos de la caché (minutos con ~3.000
# carpetas): va después de deps para aprovechar mejor cada ventana sin builds.
echo ""
echo "--- 2. incremental: presupuesto ${max_mb} MB, bajar a ×${histeresis} (LRU por carpeta crate-hash) ---"
inc_borradas=0
inc_bytes=0
if [ "$abortado" != "0" ]; then
  echo "omitido: el barrido ya se detuvo"
else
  filas=()
  for base in "$target/debug/incremental" "$target/release/incremental"; do
    [ -d "$base" ] || continue
    if ! inventario=$(
      find "$base" -mindepth 2 -type f -printf '%T@\t%s\t%P\n' 2> /dev/null \
        | awk -F'\t' '{ split($3, c, "/"); d = c[1]; b[d] += $2; if ($1 + 0 > m[d]) m[d] = $1 + 0 }
            END { for (d in b) printf "%.0f\t%.0f\t%s\n", m[d], b[d], d }'
    ); then
      echo "[barrido] falló la enumeración de $base: no barro incremental" >&2
      registrar "BARRIDO ERROR enumeracion=$base codigo=2"
      exit 2
    fi
    while IFS=$'\t' read -r m b nombre; do
      [ -n "$nombre" ] || continue
      filas+=("$m"$'\t'"$b"$'\t'"$base/$nombre")
    done <<< "$inventario"
  done
  total_inc=0
  for fila in "${filas[@]}"; do
    b=${fila#*$'\t'}
    b=${b%%$'\t'*}
    total_inc=$((total_inc + b))
  done
  max_b=$(awk -v mb="$max_mb" 'BEGIN { printf "%.0f", mb * 1048576 }')
  obj_b=$(awk -v mb="$max_mb" -v h="$histeresis" 'BEGIN { printf "%.0f", mb * 1048576 * h }')
  echo "carpetas: ${#filas[@]}  tamaño: $(mb "$total_inc") MB"
  lib0=$liberado
  bor0=$borrados
  if [ "$total_inc" -gt "$max_b" ]; then
    restante=$total_inc
    if ! ordenadas=$(printf '%s\n' "${filas[@]}" | sort -t $'\t' -k1,1n); then
      echo "[barrido] falló ordenar incremental: no barro incremental" >&2
      registrar "BARRIDO ERROR ordenar=incremental codigo=2"
      exit 2
    fi
    while IFS=$'\t' read -r _ b ruta; do
      [ "$restante" -gt "$obj_b" ] || break
      encolar "$ruta" "$b"
      restante=$((restante - b))
      [ "$abortado" = "0" ] || break
    done <<< "$ordenadas"
    vaciar
  fi
  # Cifras de lo efectivamente procesado (un barrido detenido no cuenta de más).
  inc_borradas=$((borrados - bor0))
  inc_bytes=$((liberado - lib0))
  if [ "$total_inc" -gt "$max_b" ]; then
    echo "$([ "$aplicar" = 1 ] && echo borradas || echo borraría): $inc_borradas carpetas ($(mb "$inc_bytes") MB) → quedan $(mb $((total_inc - inc_bytes))) MB"
  else
    echo "dentro del presupuesto: nada que barrer"
  fi
fi

codigo=0
[ "$abortado" = "0" ] || codigo=5
[ "$error_operacion" = "0" ] || codigo=2
modo=$([ "$aplicar" = 1 ] && echo APLICADO || echo SIMULADO)
[ "$abortado" = "0" ] || modo="DETENIDO"
registrar "BARRIDO $modo inc_carpetas=$inc_borradas inc_mb=$(mb "$inc_bytes") variantes=$var_borradas var_mb=$(mb "$var_bytes") liberado_mb=$(mb "$liberado") codigo=$codigo"
echo ""
echo "[barrido] $modo: liberado≈$(mb "$liberado") MB (incremental $inc_borradas carpetas $(mb "$inc_bytes") MB; variantes $var_borradas archivos $(mb "$var_bytes") MB)"
exit "$codigo"
