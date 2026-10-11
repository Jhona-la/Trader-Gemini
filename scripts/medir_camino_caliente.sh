#!/usr/bin/env bash
# CL-60 — Puerta de rendimiento del camino caliente (plan de Línea C, §7 paso 5).
#
# Mide la base y la rama con el MISMO medidor y la MISMA cinta: copia
# crates/god-engine-core/tests/medidor_camino_caliente.rs de este checkout a
# un worktree de cada revisión, compila en release con el mismo compilador y
# aplica la regla del plan con `plan_linea_c.py rendimiento`.
#
# Corre base, rama y base otra vez: el ruido de la base sale de dos procesos
# distintos, y una deriva lenta de la máquina (temperatura, frecuencia) cae a
# los dos lados de la rama. Con BASE = RAMA da la comparación base contra base.
#
# CL-60b: con un CARGO_TARGET_DIR compartido, cargo no distingue dos worktrees
# del mismo paquete y reutiliza el binario del otro árbol si los fuentes de
# éste son más viejos: la «rama» del PR #37 corrió el binario de la base
# (0,57 s, sin un «Compiling»). Por eso, con un target compartido, se tocan
# los .rs del worktree antes de compilar; cada ejecutable se copia fuera del
# target nada más compilarlo; y la puerta compara la huella del binario que
# imprime el medidor.
#
# Uso:     scripts/medir_camino_caliente.sh BASE RAMA
# Entorno: MEDIDOR_EVENTOS, MEDIDOR_REPETICIONES… (ver el medidor);
#          MEDIDOR_CARGO_ARGS, p. ej. "--target x86_64-pc-windows-gnu" bajo wine
#          (el ejecutable copiado corre con CARGO_TARGET_<TRIPLE>_RUNNER);
#          MEDIDOR_TRABAJO, carpeta de los worktrees, los ejecutables y los
#          logs. Los worktrees se conservan; se quitan con `git worktree
#          remove`. Cada ejecutable se guarda con la revisión y una huella
#          del medidor, el compilador, RUSTFLAGS y MEDIDOR_CARGO_ARGS, y se
#          reutiliza mientras esa huella no cambie;
#          MEDIDOR_SOLO_COMPILAR=1 compila (o reutiliza) y sale sin medir,
#          para medir después con la máquina quieta.
# Código:  el de `plan_linea_c.py rendimiento` (0 pasa, 1 no pasa, 2 no
#          comparable).
set -euo pipefail

if [ $# -ne 2 ]; then
  echo "uso: $0 BASE RAMA" >&2
  exit 2
fi
repo=$(git rev-parse --show-toplevel)
medidor=crates/god-engine-core/tests/medidor_camino_caliente.rs
trabajo=${MEDIDOR_TRABAJO:-${TMPDIR:-/tmp}/medidor_camino_caliente}
read -r -a cargo_args <<< "${MEDIDOR_CARGO_ARGS:-}"
mkdir -p "$trabajo"

# Mismo compilador a los dos lados: el que fija este checkout, aunque la base
# sea anterior a rust-toolchain.toml.
if [ -z "${RUSTUP_TOOLCHAIN:-}" ] && [ -f "$repo/rust-toolchain.toml" ]; then
  RUSTUP_TOOLCHAIN=$(sed -n 's/^channel *= *"\(.*\)"/\1/p' "$repo/rust-toolchain.toml")
  export RUSTUP_TOOLCHAIN
fi

arbol() {
  local sha dir
  sha=$(git -C "$repo" rev-parse --verify "$1^{commit}")
  dir="$trabajo/$sha"
  if [ ! -d "$dir" ]; then
    git -C "$repo" worktree add --detach "$dir" "$sha" > /dev/null
  fi
  # Sólo si difiere: copiar renueva la fecha y obliga a reenlazar.
  cmp -s "$repo/$medidor" "$dir/$medidor" || cp "$repo/$medidor" "$dir/$medidor"
  echo "$dir"
}

cargo_medidor() {
  local dir=$1
  shift
  (cd "$dir" && cargo test --release --locked "${cargo_args[@]}" \
    -p god-engine-core --test medidor_camino_caliente "$@")
}

# El ejecutable corre con el runner que cargo usaría para el target pedido
# (wine para x86_64-pc-windows-gnu en Linux).
runner=()
for ((i = 0; i < ${#cargo_args[@]}; i++)); do
  if [ "${cargo_args[$i]}" = "--target" ] && [ $((i + 1)) -lt ${#cargo_args[@]} ]; then
    triple=${cargo_args[$((i + 1))]^^}
    var="CARGO_TARGET_${triple//-/_}_RUNNER"
    read -r -a runner <<< "${!var:-}"
  fi
done

huella=$( { cat "$repo/$medidor"; echo "${RUSTUP_TOOLCHAIN:-} ${RUSTFLAGS:-} ${cargo_args[*]}"; } \
  | sha256sum | cut -c1-12)

compilar() {
  local dir=$1 sha=$2 exe destino
  for destino in "$trabajo/medidor-$sha-$huella" "$trabajo/medidor-$sha-$huella.exe"; do
    if [ -f "$destino" ]; then
      echo "reutilizo $destino" >&2
      echo "$destino"
      return
    fi
  done
  if [ -n "${CARGO_TARGET_DIR:-}" ]; then
    git -C "$dir" ls-files -z -- '*.rs' | (cd "$dir" && xargs -0 touch)
    touch "$dir/$medidor"
  fi
  exe=$(cargo_medidor "$dir" --no-run --message-format=json-render-diagnostics | python3 -c '
import json, sys
for linea in sys.stdin:
    try:
        m = json.loads(linea)
    except ValueError:
        continue
    if m.get("reason") == "compiler-artifact" and m.get("executable") \
            and m["target"]["name"] == "medidor_camino_caliente":
        print(m["executable"])')
  [ -n "$exe" ] || { echo "no encontré el ejecutable del medidor en $dir" >&2; exit 2; }
  destino="$trabajo/medidor-$sha-$huella"
  case "$exe" in *.exe) destino="$destino.exe" ;; esac
  cp "$exe" "$destino"
  echo "$destino"
}

base_sha=$(git -C "$repo" rev-parse --verify "$1^{commit}")
rama_sha=$(git -C "$repo" rev-parse --verify "$2^{commit}")
base=$(arbol "$1")
rama=$(arbol "$2")
echo "base $1 en $base; rama $2 en $rama; compilador ${RUSTUP_TOOLCHAIN:-por defecto}" >&2
exe_base=$(compilar "$base" "$base_sha")
exe_rama=$exe_base
if [ "$rama_sha" != "$base_sha" ]; then
  exe_rama=$(compilar "$rama" "$rama_sha")
fi
if [ "${MEDIDOR_SOLO_COMPILAR:-0}" = "1" ]; then
  printf 'base %s\nrama %s\n' "$exe_base" "$exe_rama"
  exit 0
fi

medir() {
  (cd "$1/crates/god-engine-core" && "${runner[@]}" "$2" \
    --ignored --nocapture --test-threads=1 medir_camino_caliente) > "$3"
  grep '^MEDIDOR	' "$3" >&2 || true
}
medir "$base" "$exe_base" "$trabajo/base_1.log"
medir "$rama" "$exe_rama" "$trabajo/rama.log"
medir "$base" "$exe_base" "$trabajo/base_2.log"

fuentes=()
if ! git -C "$repo" diff --quiet "$base_sha" "$rama_sha" -- '*.rs' '*Cargo.toml' Cargo.lock; then
  fuentes=(--fuentes-distintas)
fi
python3 "$repo/scripts/plan_linea_c.py" rendimiento "${fuentes[@]}" \
  --base "$trabajo/base_1.log" "$trabajo/base_2.log" --rama "$trabajo/rama.log"
