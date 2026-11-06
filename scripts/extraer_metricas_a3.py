#!/usr/bin/env python3
"""Extracción de las métricas §28 del runbook A3 desde el log de una corrida.

Uso: python scripts/extraer_metricas_a3.py logs/corrida_a3b_20261106_1315.log

Métricas (runbook docs/RUNBOOK_CORRIDA_DEMO_A3_2026-11-06.md §28):
  1. Nº de trades (ciclos FILL de entrada)
  2. PnL bruto / neto / fees acumulados reportados por el motor
  3. WR reportado
  4. EV/trade = PnL neto / nº trades (cuando haya)
  5. Avisos de latencia elevada
  6. Salud: proceso vivo (lo verifica quien corre el script), snapshots
     Ready vs Blocked por causa, fallos TLS, purgas incompletas.
Es de SOLO LECTURA: no toca la corrida ni el .env.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    log = Path(sys.argv[1])
    texto = log.read_text(encoding="utf-8", errors="replace")
    lineas = texto.splitlines()

    fills = [l for l in lineas if "[FILL]" in l]
    flatten = [l for l in lineas if "[FLATTEN]" in l and "Purga" not in l]
    purgas = [l for l in lineas if "Purga incompleta" in l]

    # Última línea contable del motor (PnL/fees/WR que ÉL reporta).
    pnl = next((l for l in reversed(lineas) if "Gross PnL:" in l), "—")
    roi = next((l for l in reversed(lineas) if "ROI Net" in l), "—")
    fees_tel = next(
        (l for l in reversed(lineas) if "Cumulative Fees" in l), "—"
    )

    # PnL/fees de los propios fills (verdad del exchange, no del contador):
    # no todo fill trae PnL en el log; los que traen "pnl=" se suman.
    pnl_fills = 0.0
    comisiones = 0.0
    con_pnl = 0
    for l in fills:
        m = re.search(r"pnl=(-?\d+(?:\.\d+)?)", l)
        if m:
            pnl_fills += float(m.group(1))
            con_pnl += 1
        m = re.search(r"commission=(-?\d+(?:\.\d+)?)", l)
        if m:
            comisiones += float(m.group(1))

    latencia = sum(
        1 for l in lineas
        if re.search(r"latencia.*elevada|LATENCIA DE RED ELEVADA", l, re.I)
    )
    tls = sum(1 for l in lineas if "handshake TLS falló" in l)
    ready = sum(1 for l in lineas if "status: Ready" in l)
    blocked = {}
    for l in lineas:
        if "status: Blocked" in l:
            m = re.search(r'error: Some\("([^"]+)"\)', l)
            clave = m.group(1) if m else "sin-error"
            if "unresolved_orders: 0" not in l:
                clave = "ORDENES_SIN_RESOLVER"
            blocked[clave] = blocked.get(clave, 0) + 1
    ignicion = [l for l in lineas if "[IGNICIÓN]" in l]

    prints = [
        f"log: {log}  ({len(lineas)} líneas)",
        f"trades (fills de entrada+salida): {len(fills)}  |  cierres FLATTEN: {len(flatten)}",
        f"  fills con pnl en el log: {con_pnl}  pnl_fills={pnl_fills:+.6f}  commission_fills={comisiones:+.6f}",
        f"contador del motor: {pnl.strip()}",
        f"                   {roi.strip()}",
        f"telemetría de fees: {fees_tel.strip()}",
        f"avisos de latencia elevada: {latencia}",
        f"fallos de handshake TLS: {tls}",
        f"snapshots: Ready={ready}  Blocked={sum(blocked.values())} {blocked}",
        f"purgas incompletas (protection_health dirty): {len(purgas)}",
        f"ignición: {ignicion[0].strip() if ignicion else '—'}",
    ]
    if len(fills) >= 2:
        # EV crudo por ciclo completo (entrada+salida) con la verdad del fill.
        ciclos = len(fills) / 2
        prints.append(
            f"EV/trade (fills, bruto): {pnl_fills / ciclos:+.6f}  "
            f"| neto estimado: {(pnl_fills + comisiones) / ciclos:+.6f}"
        )
    print("\n".join(prints))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
