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

import json
import re
import sys
from pathlib import Path


def income_real(desde_ms: int | None, env_path: Path | None = None) -> None:
    """VERDAD DEL EXCHANGE: /fapi/v1/income desde el arranque de la corrida.

    El contador interno del motor sólo suma receipts de cierres internos
    (hallazgo H2, GLM-209): puede quedar en 0 aunque el exchange cobre y
    liquide de verdad. Esta sección lee el income del Testnet con las claves
    del .env y reporta el bruto/fees/neto REALES — la fuente que exige el
    criterio global (EV neto post-fees).
    """
    import hashlib
    import hmac
    import re as _re
    import time as _time
    import urllib.parse
    import urllib.request

    env: dict[str, str] = {}
    candidatos = [env_path, Path(".env")] if env_path else [Path(".env")]
    texto_env = None
    for c in candidatos:
        if c and c.is_file():
            texto_env = c.read_text(encoding="utf-8", errors="replace")
            break
    if texto_env is None:
        print("income real: no encuentro .env (ni junto al log ni en cwd)")
        return
    for ln in texto_env.splitlines():
        m = _re.match(
            r"^(BINANCE_TESTNET_API_KEY|BINANCE_TESTNET_SECRET_KEY)=(.*)$",
            ln.strip(),
        )
        if m:
            env[m.group(1)] = m.group(2).strip()
    if len(env) != 2:
        print("income real: faltan claves Testnet en .env")
        return
    base = "https://testnet.binancefuture.com"
    params: dict[str, object] = {"limit": 1000}
    if desde_ms:
        params["startTime"] = desde_ms
    params["timestamp"] = int(_time.time() * 1000)
    qs = urllib.parse.urlencode(params)
    sig = hmac.new(
        env["BINANCE_TESTNET_SECRET_KEY"].encode(), qs.encode(), hashlib.sha256
    ).hexdigest()
    req = urllib.request.Request(
        f"{base}/fapi/v1/income?{qs}&signature={sig}",
        headers={"X-MBX-APIKEY": env["BINANCE_TESTNET_API_KEY"]},
    )
    try:
        rows = json.load(urllib.request.urlopen(req, timeout=20))
    except Exception as e:  # noqa: BLE001 - reportar y seguir con el resto
        print(f"income real: consulta fallida: {e}")
        return
    tot: dict[str, float] = {}
    for r in rows:
        tipo = r.get("incomeType", "?")
        try:
            tot[tipo] = tot.get(tipo, 0.0) + float(r.get("income", "0") or 0)
        except (TypeError, ValueError):
            continue
    bruto = tot.get("REALIZED_PNL", 0.0)
    fees = tot.get("COMMISSION", 0.0) + tot.get("FUNDING_FEE", 0.0)
    trades = sum(1 for r in rows if r.get("incomeType") == "REALIZED_PNL")
    print("— VERDAD DEL EXCHANGE (/fapi/v1/income) —")
    print(f"  records: {len(rows)}  cierres REALIZED_PNL: {trades}")
    for k in sorted(tot):
        print(f"  {k:14s} {tot[k]:+.6f} USDT")
    print(f"  bruto={bruto:+.6f}  fees+funding={fees:+.6f}  NETO={bruto + fees:+.6f} USDT")
    if trades:
        print(f"  EV/trade NETO REAL: {(bruto + fees) / trades:+.6f} USDT")


def main() -> int:
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        return 2
    usar_income = len(sys.argv) == 3 and sys.argv[2] == "--income"
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
    if usar_income:
        # Ventana del income: el nombre del log lleva el arranque
        # (corrida_a3b_YYYYMMDD_HHMM.log). El mtime NO sirve: en un log vivo
        # es "ahora". Si el nombre no parsea, se pide todo (la ventana la
        # acota quien lee el informe).
        desde_ms = None
        m = re.search(r"_(\d{8})_(\d{4})$", log.stem)
        if m:
            import datetime as _dt

            try:
                arranque = _dt.datetime.strptime(
                    m.group(1) + m.group(2), "%Y%m%d%H%M"
                ).astimezone()
                desde_ms = int(arranque.timestamp() * 1000) - 60_000
            except ValueError:
                desde_ms = None
        # El .env vive en el raíz del repo (padre de logs/), no junto al script.
        env_path = log.resolve().parent.parent / ".env"
        income_real(desde_ms, env_path if env_path.is_file() else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
