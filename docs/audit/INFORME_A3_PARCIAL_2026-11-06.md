# INFORME A3 — PARCIAL (~1.3 h de la corrida A3-bis) — GLM, 2026-11-06

> Primer informe de medición con trades REALES del sistema. Corrida:
> `logs/corrida_a3b_20261106_1315.log`, binario 13:15 (fix GLM-208), PID 42748,
> encendida con `TG_IGNICION_SIN_DARWIN=1` (rechazo de Darwin logueado antes de
> pisarse). Extractor reproducible: `python scripts/extraer_metricas_a3.py <log>`.

## Métricas §28 al corte (~13:15→14:30 local)

| Métrica | Valor | Nota |
|---|---|---|
| Trades (ciclos completos) | **1** (2 fills + FLATTEN) | BCHUSDT BUY 266.39 → SELL 266.36 |
| PnL bruto | ≈ −0.0006 USD | −0.03 × 0.02 (redondea a 0.0000) |
| Fees | **0.0000 reportados** | ver hallazgo H2 |
| WR | 0/1 | |
| EV/trade | ≈ −0.0006 bruto | indistinguible de 0 con 1 trade |
| Avisos latencia elevada | **0** | antes del régimen nuevo: 200+ por sesión |
| Fallos TLS (CN/root) | 25 | vía Fast-Lane por IP; conecta por alternativa |
| Snapshots OMS | Ready 4 · Blocked 186 | ver hallazgo H3 |
| Ticks | 5000/klines procesados, 0.18–1.0 ms/tick | |

## Lo que FUNCIONA (verificado en vivo)
- Ignición con puerta auditable: Darwin RECHAZADO (metrica_entrenamiento_invalida)
  logueado ANTES de habilitar por env — el contrato del ciclo 204 vive.
- OMS autoriza (`Ready, unresolved: 0`) — el fix GLM-208 desbloqueó la operativa.
- **Primer ciclo de trade real de la historia del sistema** (entrada MARKET,
  salida LIMIT, FLATTEN limpio).
- Balance real Testnet $7,321.12 por API; fees VIP reales extraídos por símbolo.
- 0 avisos de latencia elevada en ~1.3 h.

## Hallazgos (por severidad)
- **H2 — contabilidad de fees/PnL no registra el ciclo**: el motor reporta
  `Fees: 0.0000` y `PnL: 0.0000` tras un ciclo que debía comisionar ≈0.0032
  (maker 0.02% + taker 0.04% sobre ~$5.3). O el contador no ingiere los fills
  del user-stream, o el Testnet no cobra fees (income API lo dirá: B3.6/FEE-06).
  SIN resolver esto, el informe de 24 h sale todo en cero y el criterio global
  (EV neto post-fees) NO es medible. Dueño natural: fees walk-forward V39 (QSR).
- **H3 — OMS_UNCONFIRMED_LOCAL_RESERVATION bloquea el ~97% de los snapshots**
  (186 de 190) pese a estar TODO cerrado tras el FLATTEN: estado pegajoso
  (slot `is_open()` tras cierre, o `exchange_confirmed` nunca levado). No impidió
  el primer trade pero reduce la capacidad de entrada. Dueño natural: OMS
  (host_oms gate, líneas 600-608) — mismo archivo que GLM-208.
- **H4 — purga incompleta en 144 símbolos del Testnet** al flatten (basura del
  Testnet: opciones sintéticas, delivery futures, nombres de prueba chinos):
  `protection_health dirty`. No bloquea; ruido que conviene acotar al universo
  activo en la purga.
- H5 — TLS CN/root mismatch en la vía Fast-Lane por IP (25 fallos, reintenta y
  conecta por vía alternativa): latencia, no corte.

## Próximo corte
Extracción §28 completa a las 24 h (2026-11-07 ~13:15 local) con el mismo
script; veredicto del criterio global requiere: H2 resuelto (medir fees de
verdad) y suficientes trades. Los BLOQUEANTES #2/#3 del arranque (A1) se
calibran con ese informe.
