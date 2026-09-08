# 🔬 NOVENO INFORME DE ASEGURAMIENTO Y CERTIFICACIÓN PROFUNDA
### Post-física: el bug de unidades que satura el slippage al 5%, la salida sin física, y la verificación de los O-fixes

**Fecha:** 2026-09-08 · **Método:** 2 roles senior (Quant Researcher + ML/Infra Architect). Solo documentación.
**Se agrega a:** la serie forense completa (maestro §13-20, aseguramientos 1º-8º).

---

## 🗺️ 0. Diagnóstico central

Los O-fixes están **estructuralmente conectados y verificados** — pero la física de entrada tiene un **bug de unidades CRÍTICO** que invalida cualquier certificación sobre este build: `tick_volatility` se pasa en unidades absolutas (BTC: ~60) cuando `calculate_market_entry` espera una fracción (~0.001), saturando el slippage al clamp máximo de 5% en TODO trade de BTC/ETH. Y la **salida sigue sin física** — `calculate_exit` tiene cero callers, los TP llenan a precio perfecto de maker en un mundo taker.

---

## 🚦 1. Resumen de resolución

| Frente | Estado |
|---|---|
| ✅ VERIFIED-CORRECT | O-03 (kelly escalado, sitio y fórmula correctos), O-04 (drawdown breaker, sin falso positivo en frío, veto solo aperturas), Lagged fix (ambos consumidores inmortales), D-171 hedge-mode (completo en 5 métodos), D-110 Lee-Ready, D-112 veto causal con excepción breakout, D-113 slippage dinámico, D-117 parser whitespace, D-134 triple-barrier labels, D-225 state validator, D-252 cold-start neutral |
| 🔴 Crítico | N-01 (bug unidades tick_volatility) |
| 🔴 Alto | N-02 (exit sin física — TP a precio perfecto), OCO retry persistente, working tree sin commit (57 archivos) |
| ⬜ Persistentes | forensics cfg(test), PhaseExecutor, DriftAuditor `_`, activity_bonus, REJECT_COUNTERS, feature-engine ~80% muerto, rutas data/, MmapTelemetryBus fantasma |

---

## 📊 2. Matriz de CRÍTICOS/ALTOS

| ID | Hallazgo | Ubicación | Severidad |
|---|---|---|---|
| **N-01** | **BUG DE UNIDADES en tick_volatility**: `atr_pct * mid_price` produce unidades ABSOLUTAS (BTC ~60) pero `calculate_market_entry` espera FRACCIÓN (~0.001, sus tests usan 0.001/0.002). Resultado: `latency_slippage = 60 × (15/150) = 6.0` → clamp 0.05 SATURADO → **TODO trade de BTC/ETH paga 5% de slippage adverso determinista**. Destruye el EV de cada trade. Invalida certificación sobre este build | `god-engine-core/lib.rs:1519` vs `reality_physics.rs:68` | CRÍTICO |
| **N-02** | **Exit path SIN física**: `calculate_exit` (maker=precio exacto, taker=cruce adverso) tiene CERO callers. Los TP llenan a `tp_price` exacto, trailing a `mid_price` perfecto. Las entradas pagan física pero las salidas son frictionless — sesgo DIRECCIONAL de PnL que infla el edge de scalps con TP estrecho | `reality_physics.rs:84-139` sin callers; `lib.rs:725-810` | ALTO |
| **N-03** | **OCO retry persiste (S-05/O-05, 3 auditorías consecutivas)**: reenvía el mismo buffer firmado (timestamp vencido → -1021 garantizado). El patrón correcto existe en el mismo archivo (flatten retry re-firma) y no se aplica | `executor.rs:1901-1919` vs `519-552` | ALTO |
| **N-04** | **Working tree 57 archivos sin commit**: cambios de semilla en genome.rs (literales crudos 0.55/0.15 reemplazando derivados), D-144 re-pesa el risk_normalizer sin re-certificación, position.rs slots duales muertos. La sesión concurrente implementó 20+ D-IDs pero nada está commiteado ni certificado | múltiple | ALTO |
| **N-05** | **Sin certificación corriendo**: equity_curve.csv vacío (header only). El overnight sobre REAL.bin + física no se ejecutó | `equity_curve.csv` | ALTO (bloqueo) |
| **N-06** | **risk_normalizer ignora el piso ATR**: el SL efectivo es `max(base, atr*0.8/1.5)` pero el normalizador usa solo las bases — el problema de riesgo desproporcionado reaparece vía el término ATR en alta volatilidad | `risk-engine/lib.rs:444-451` vs `558-559` | MEDIO |

---

## 📈 3. La física que falta (asimetría completa)

| Dirección | Estado | Modelo |
|---|---|---|
| Entrada | ✅ Cuadrático + latency + floor del genoma | `calculate_market_entry` conectado |
| Salida TP | ❌ Precio exacto de maker | `calculate_exit` SIN conectar |
| Salida SL | ⚠️ `min(sl_price)` — ligeramente pesimista pero sin impacto | |
| Salida trailing | ❌ `mid_price` perfecto | |
| Latencia | ⚠️ Solo en entrada (y con bug de unidades) | |

**La asimetría resultante**: las entradas pagan slippage físico pero las salidas TP se ejecutan a precio perfecto en un mundo donde el propio D-179 reconoce que las salidas son taker. El edge certificado de cualquier scalp con TP estrecho queda sistemáticamente inflado.

---

## 🎯 4. Hoja de ruta

1. **N-01 (MINUTOS — el fix más crítico de la serie desde S-01)**: `tick_vol = atr_pct` (pasar la fracción directamente, no `atr_pct * mid_price`). Sin esto, TODA certificación con física es inválida.
2. **N-02 (horas)**: conectar `calculate_exit` al cierre (TP/SL/trailing) — el modelo correcto ya existe.
3. **N-03 OCO retry (horas, 3ª vez que se reporta)**.
4. **N-04**: commitear el working tree de la sesión concurrente (57 archivos con 20+ D-IDs evaluados como completos) tras N-01.
5. **N-05**: certificación overnight sobre REAL.bin + física CORREGIDA (N-01+N-02).

*Se agrega sin sustraer contenido. Referencias a main @ 302d4057 + working tree 57 archivos.*
