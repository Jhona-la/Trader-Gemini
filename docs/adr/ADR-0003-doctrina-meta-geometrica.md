# ADR-0003 — Doctrina de la meta: crecimiento geométrico medido, no número fijo

- **Estado**: Aceptado (directiva del operador)
- **Fecha**: 2026-09-29
- **Responsables**: Operador (re-encuadre); GLM (instrumentación XLV·L,
  XLVII·B, XLVIII·A); ratificado en buzón

## Contexto

La meta original (+100% cada 3 días ≈ 26% diario compuesto) es
matemáticamente posible en backtest e insostenible en mercado real
(costos, slippage, capacidad, ruina). Dos mediciones del sistema lo
cuantificaron: la aritmética exige ~10 trades/día con edge sostenido
(XLV·L) y el campeón midió 0.02 trades/día sobre tapes reales — brecha
620× (XLVII·B), con el cuello de botella en evidencia ML, no en sizing.

## Decisión

1. La meta operativa es **maximizar el crecimiento geométrico sujeto a
   restricciones de riesgo, DD, costos y capacidad** — medida con el
   panel ex-post (CAGR, Sharpe/Sortino anualizados, Calmar, MaxDD,
   CVaR95, turnover, WR, PF; `backtest_engine::metrics`).
2. La ruta hacia el volumen pasa por **cobertura de modelos del roster
   + calidad con lift real sobre tapes** (secuencia: trainer FMT →
   promover → watcher hot-reload 10 s desbloquea B3.25 en caliente) —
   NO por relajar vetos de riesgo.
3. Toda métrica degenerada se reporta explícita (Sortino infinito sin
   pérdidas; CVaR NaN con <20 trades: la cola no es estimable).

## Consecuencias

- El panel está cableado al replay (`ReplayStats.metrics`) — cada
  corrida reporta el panel completo; la brecha contra cualquier objetivo
  es medible, no opinable.
- Los cambios de física exigen re-baseline del oráculo T-1 (ver
  ADR-0001).
- "Promesas de crecimiento exponencial" quedan fuera de vocabulario
  operativo: sólo números del panel.
