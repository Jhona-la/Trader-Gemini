# PLAN MAESTRO DE EVIDENCIA Y FASES — Trader Gemini

> Directriz del operador (2026-11-05): «Lo quiero todo: primero ver que sí
> funciona, luego evidencia de cómo funciona, llenar los vacíos
> matemáticos. Haz análisis de qué falta, audita, y haz un plan por fases
> — así sean cientos — y ejecútalo.»
>
> Autor: GLM (por mandato directo del operador). Para TODO el ecosistema.
> Estado: VIGENTE desde su publicación.

## 0. Diagnóstico (la una línea)

~40 teorías certificadas COMO CÓDIGO (CI verde en main), **cero evidencia
de rentabilidad**. El eslabón roto no es la matemática: es el bucle
evidencial — **ignición → Demo → medición → decisión**. El CI verde
certifica que cada pieza hace lo que su contrato dice; no certifica que
el sistema gane dinero. Mientras no exista una corrida Demo medida, todo
juicio sobre rentabilidad es fe, no ciencia.

Evidencia que sostiene el diagnóstico:
- QRA-BOOT-01 (buzón, 2026-10-22): la ignición NO está certificada —
  `Err` de fetch_open_positions interpretado como cero posiciones, bypass
  de Testnet vía `is_demo_mode`, readiness sin evidencia de conciliación.
- El operador reporta PÉRDIDAS y 200+ avisos de WARNING: LATENCIA DE RED
  ELEVADA en producción.
- Los observables de régimen (σz, λ_KS, β_RG, D^α Caputo, LPPLS,
  deformación FR — GLM; sombras de Sol; Kähler/firmas de AGY) viven en
  el motor SIN consumidor decisional: instrumentos de un avión sin piloto.

## 1. Criterio global PRE-DEFINIDO (publicado ANTES de cualquier corrida)

**«El sistema funciona» ≡ las tres condiciones, medidas en Demo Binance
(Testnet) out-of-sample:**

1. **EV neto post-fees > 0** sostenido durante 7 días corridos.
2. **Drawdown acotado**: DD máx < banda pre-definida antes de encender
   (sugerencia inicial: 15% del capital de Demo — ajustable por el
   consejo, pero FIJADA antes de la corrida).
3. **Paridad Demo/backtest**: la desviación de PnL entre la corrida Demo
   y el backtest de los mismos períodos queda dentro de una banda
   estadística pre-definida (si Demo se desvía, el modelo no representa
   la realidad y hay que arreglar el modelo, no el informe).

**Un EV neto negativo MEDIDO es un hallazgo, no un fracaso**: dice
exactamente dónde está el agujero (fees, latencia o señal). Cuarenta
teorías sin medir valen menos que un número medido.

La meta de 100%/3 días queda como HORIZONTE, no como criterio: ×1.26
diario compuesto no tiene soporte histórico verificable en ningún
sistema. El criterio medible de esta fase es EV neto post-fees > 0; el
escalado viene después (Fase D).

## 2. FASE A — EVIDENCIA (prioridad #1 del operador)

| #  | Tarea | Dueño sugerido | Criterio de salida (medible) |
|----|-------|----------------|------------------------------|
| A1 | Auditoría de ignición: mapa del arranque real de `src/bin/god_engine.rs` vs. la secuencia del mandato §5; base: QRA-BOOT-01 | Quien tome el hilo (QRA mapeó los hallazgos) | Lista de huecos con `archivo:línea` publicada en docs/audit/ |
| A2 | Cierre de huecos de ignición: OMS reconciliado al arranque, discriminador Testnet/paper real (no `is_demo_mode` agrupado), adopción atómica de posiciones, readiness con evidencia | Dueño del host (AGY) + QRA como revisor | Contrato de ignición en verde en CI |
| A3 | **Corrida Demo 24 h**: encender el bin en Testnet con secuencia mínima (balance real de la wallet Demo → OMS → warmup → operar), loggeando PnL bruto/neto post-fees, WR, EV/trade, nº de trades | El agente que ejecute A2 + telemetría de Sol | Informe con números reales: el PRIMER dato del sistema completo (verde o rojo) |
| A4 | Pruebas de caos: microcortes de red, latencia inyectada, reconexión; circuit breaker que regrese a modo seguro | Línea I (ya trabaja reconexiones WS: I-11..I-13) | El freno funciona, medido en el log |
| A5 | Extensión 72 h → 7 d + banda estadística de desviación Demo/backtest | El dueño de A3 | **Veredicto del criterio global** publicado |

## 3. FASE B — CONEXIÓN TEORÍA→DECISIÓN

| #  | Tarea | Dueño sugerido | Criterio de salida |
|----|-------|----------------|--------------------|
| B1 | Experimento A/B EN DEMO: puerta base vs. base+observable (LPPLS estable + κ Hawkes; deformación FR con pico de época). Misma señal, mismo período, única diferencia = el observable | Dueño de la puerta + GLM (observables) | **ΔEV neto medido** con N suficiente; publicado verde o rojo |
| B2 | Promoción de ganadores / retiro de perdedores, cada uno con contrato de consumo (calibración de nulo incluida) | Consejo | Ningún observable vivo sin experimento de valor asociado |

## 4. FASE C — VACÍOS MATEMÁTICOS (paralela a A/B; núcleos independientes)

| #  | Tarea | Dueño | Estado |
|----|-------|-------|--------|
| C1 | Núcleo β_q: métrica Fisher empírica q-gaussiana (β_q = 1/(5−3q)) | GLM | Diseño CERRADO (memoria GLM-195 + buzón 195); átomo mecánico |
| C2 | Ricci-q como observable %64 + wiring | GLM | Tras C1 |
| C3 | Acoples decisionales: κ×LPPLS, umbral FR, gen α Caputo | Consejo | Esperando decisión (la calibración honesta de cada núcleo ya está publicada) |
| C4 | Friston/inferencia activa (F-026), LTC/Mamba, SBE binario, Ed25519, Stefan | Dueños: learning/silicio (AGY, Codex) | Pendientes |
| C5 | **Auditoría de hardcodes/sesgos en config/ y lib/** (pedido explícito del operador) | Libre (auditar y reportar; cada fix un átomo) | Pendiente |

## 5. FASE D — ESCALADO (BLOQUEADA hasta A5 verde)

| #  | Tarea | Criterio |
|----|-------|----------|
| D1 | Top 10 dinámico con rotación medida | Rotación decide por estadística medida, no por hardcode |
| D2 | Kelly fraccional multivariante sobre colas medidas (no supuestas) | Dimensionamiento calibrado con las colas REALES de la corrida A5 |
| D3 | Conmutación Demo→Producción con freno de emergencia | Solo tras criterio global sostenido |

## 6. Reglas de orden

1. **A antes que B** (sin corrida no hay A/B que medir).
2. **C en paralelo** (los núcleos son independientes del bucle evidencial).
3. **D bloqueada** hasta que A5 dé el veredicto.
4. Todo informe de corrida DEBE incluir: ROI/PnL/WR **antes y después
   de fees**, período exacto, dataset, y comparación contra backtest del
   mismo período (mandato §28 — obligatorio desde la primera corrida).
5. Ningún agente declara «el sistema funciona» fuera del criterio del §1.

— GLM, por directriz del operador, 2026-11-05.
