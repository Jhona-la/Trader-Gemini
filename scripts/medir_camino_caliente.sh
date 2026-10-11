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
# Uso:     scripts/medir_camino_caliente.sh BASE RAMA
# Entorno: MEDIDOR_EVENTOS, MEDIDOR_REPETICIONES… (ver el medidor);
#          MEDIDOR_CARGO_ARGS, p. ej. "--target x86_64-pc-windows-gnu" bajo wine;
#          MEDIDOR_TRABAJO, carpeta de los worktrees y los logs. Los worktrees
#          se conservan para reusar la compilación; se quitan con
#          `git worktree remove`.
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

base=$(arbol "$1")
rama=$(arbol "$2")
echo "base $1 en $base; rama $2 en $rama; compilador ${RUSTUP_TOOLCHAIN:-por defecto}" >&2
cargo_medidor "$base" --no-run
cargo_medidor "$rama" --no-run

medir() {
  cargo_medidor "$1" -- --ignored --nocapture --test-threads=1 medir_camino_caliente > "$2"
  grep '^MEDIDOR	' "$2" >&2 || true
}
medir "$base" "$trabajo/base_1.log"
medir "$rama" "$trabajo/rama.log"
medir "$base" "$trabajo/base_2.log"

python3 "$repo/scripts/plan_linea_c.py" rendimiento \
  --base "$trabajo/base_1.log" "$trabajo/base_2.log" --rama "$trabajo/rama.log"
