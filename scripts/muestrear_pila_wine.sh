#!/usr/bin/env bash
# CL-63 — Muestras de pila de un hilo de un proceso de wine, para
# `scripts/perfil_pila_pe.py`.
#
# Cada muestra para el proceso con gdb (ptrace, unos 0,7 s), guarda los
# registros del hilo y 64 KiB de su pila desde RSP, y lo suelta. Entre
# muestras espera un tiempo al azar para no sincronizarse con el trabajo.
#
# Uso: scripts/muestrear_pila_wine.sh PID TID N CARPETA
#   PID      proceso de Linux que corre el .exe bajo wine
#   TID      hilo que trabaja (`ps -L -o tid,pcpu,comm -p PID`)
#   N        número de muestras
# Después:
#   cp /proc/PID/maps CARPETA/maps.txt     (mientras el proceso vive)
#   scripts/perfil_pila_pe.py CARPETA/maps.txt CARPETA medidor.exe
set -u

if [ $# -ne 4 ]; then
  echo "uso: $0 PID TID N CARPETA" >&2
  exit 2
fi
pid=$1 tid=$2 n=$3 carpeta=$4
mkdir -p "$carpeta"
cp "/proc/$pid/maps" "$carpeta/maps.txt"
for ((i = 0; i < n; i++)); do
  kill -0 "$pid" 2>/dev/null || break
  gdb -q -batch -p "$pid" \
    -ex "python [t for t in gdb.selected_inferior().threads() if t.ptid[1] == $tid][0].switch()" \
    -ex "info registers rip rsp rbp rbx rsi rdi r12 r13 r14 r15" \
    -ex "dump binary memory $carpeta/pila_$i.bin \$rsp \$rsp+65536" \
    > "$carpeta/regs_$i.txt" 2>&1
  sleep "0.$((RANDOM % 9 + 1))"
done
