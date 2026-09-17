# 🔬 SEGUNDO INFORME DE ASEGURAMIENTO Y CERTIFICACIÓN — POST R1/R2/R3
### Auditoría de raíz a cima tras los commits r1 (`588bf9b5`), r2 (`55e21637`), r3-parcial (`aa24f171`) + trabajo concurrente sin commit

**Fecha:** 2026-09-07 · **Método:** 5 auditores paralelos (regresión de commits, genoma bt/prod, módulos 1+6, módulos 2+7, módulos 3+4+5+8). Solo documentación.
**Se agrega a:** INFORME_FORENSE_MAESTRO.md §13, INFORME_ASEGURAMIENTO_F0F3.md, PLAN_REPARACION_EVOLUCION_SENIOR.md.

---

## 🗺️ 0. Topología del grafo tras las reparaciones

El grafo mejoró en sus Nodos Terminales (ejecución firma correctamente, mode-aware, con cancelación de pierna hermana y failover) y en el Nodo Raíz (procedencia de datos versionada, lookahead de generadores eliminado, M2 correcto). **El diagnóstico central persiste y se agrava en el Nodo de Decisión y los Nodos de Memoria**: la ruta de decisión viva (cuántica) ignora el motor de riesgo reparado, y la memoria evolutiva adquirió una **regresión de lógica nueva** que congela la evolución exactamente cuando hay edge. Hay además **dos bypass del gate de genoma** que vacían de autoridad al embudo versión que f3 instaló.

---

## 🚦 1. Resumen de estado de resolución

| Área | Estado |
|---|---|
| ✅ Certificado correcto | Kelly R1.4 (fórmula, rampa, guard, gen revivido); firma OCO mode-aware completa (payload_start correcto, política única ambas piernas); failover -4061 del flatten; motor sibling-cancel con firma e idempotencia; reduce_only one-way; insolvencia antes de aperturas; interlock de latencia genómico; ws_client (keepalive, timeout dinámico); flight-recorder con seqlock real; parser f64 y tolerante a reorden; procedencia BinTick (3 magics); M2 World Bank correcto y escalado; macro sintético marcado; lookahead de generadores eliminado; geometría DarkAlpha mitigada por despacho in_dim; swing-NN features reales; Consejo INVOCADO (parcial); darwin gate→apply |
| ⚠️ Reparado con defecto residual | R1.1 (bounds unificados PERO constructor `new_baseline` fuera de bounds en 7 genes — el gap inverso); R1.3 (split bayesiano PERO muerto en producción: la ruta cuántica lo bypassa); R1.6 (expected_volatility=ATR% PERO su consumidor crítico — el router — es código huérfano); R3.1 (OCO mode-aware PERO `is_hedge_mode` nunca puede ser false → rama one-way inalcanzable) |
| 🔴 Crítico nuevo | Inversión de lógica del daemon (CHAMPION PROTECTED + `if true`); bypass del gate por evolution-engine; piso Kelly 5% en ruta cuántica; cierre cruzado de espejos |
| 🔴 Pendiente crítico histórico | K-17 colisión Continuous↔Swing; conformal constante; colinealidad OBI; DarkAlpha ReLU/tanh + normalizadores mutantes; champion_path; refresh_models clobber; sin separación de entornos; R3.4; R3.7; forensic Lagged; etiquetas triple-barrier; aggTrades OBI ±1 |

**Conteo de esta ronda:** ~55 hallazgos (6 críticos, ~18 altos, ~20 medios, ~11 bajos) + 30+ certificaciones de corrección.

---

## 📊 2. Matriz de hallazgos CRÍTICOS/ALTOS de esta ronda

| ID | Severidad | Hallazgo | Ubicación |
|---|---|---|---|
| **N-01** | CRÍTICO | **Inversión de lógica del gate evolutivo (regresión nueva)**: `if current_shadow_sharpe >= 2.0 { …CHAMPION PROTECTED…; return; }` — cuando el edge ESTÁ validado no se muta nada; la evolución solo corre sobre ruido. Y el `if true {` posterior vuelve CÓDIGO MUERTO el kill-switch EWMA/drift (líneas ~480-498 inalcanzables). La producción no puede ni mejorar con edge ni frenarse por drift | `evolution-engine/src/online_daemon.rs:270-287, 480-498` |
| **N-02** | CRÍTICO | **Bypass del gate de genoma por dos vías**: (a) `evolution-engine/lib.rs:~411` hace `current_alpha.apply_to_arena(&arena)` + `save()` SIN pasar por `promote()` — el embudo único solo lo respeta online_daemon; (b) `from_vector` no aplica la reparación RR (solo existe en `mutate`), y un sample CMA crudo con tp=lo/sl=hi viola RR → o el gate lo rechaza (bloqueo) o la vía (a) lo elude (arena con genoma inválido) | `evolution-engine/src/lib.rs:165,411`; `genome.rs:1646+` |
| **N-03** | CRÍTICO | **Piso Kelly 5% forzado en la ÚNICA ruta de producción**: `evaluate_quantum_order_by_horizon` clampa `kelly_fraction` a mínimo 0.05 — una moneda en régimen perdedor (Kelly legítimo 0) apuesta 5% igual. Pisa todo el trabajo R1.4 del kelly.rs en la puerta de entrada | `risk-engine/src/lib.rs:350-354` |
| **N-04** | CRÍTICO | **Cierre cruzado de espejos de posición**: el cierre scalp llama `close_with_fee()` incondicional sobre `positions.position` — si ese slot contiene el espejo del SWING, resta el margen del swing vivo de `used_margin` mientras el swing sigue abierto (doble sustracción + espejo huérfano); simétrico en cierre swing | `god-engine-core/src/lib.rs:768-771, 999-1002, 1438-1452, 1523-1537` |
| **N-05** | ALTO | **`is_hedge_mode` nunca se pone en false** (arranca true, solo `store(true)` en ensure_hedge_mode): si `ensure_hedge_mode` falla en una cuenta one-way, el bot sigue firmando TODO con positionSide → -4061 en cada pierna OCO. El failover del flatten corrige local pero NO escribe el flag: el resto del motor queda en modo equivocado. La rama one-way de R3.1 es inalcanzable en producción real | `executor.rs:256, 294, 355, 374, 508-544` |
| **N-06** | ALTO | **`new_baseline` fuera de bounds en 7 genes** (constructor vs arrays divergen — el gap INVERSO de R1.1): idx 12 scalp_obi (0.025 < lo 0.05), idx 27 dynamic_atr_min (1e-6 vs [1e-4, 0.01] → ×100 al clamp), idx 29 ema_trend (×10), idx 38 explosive_conf (0.9995 > hi 0.99), idx 81 turbo_z (2.5 > 1.5), idx 111 hawkes_vol (×2), idx 135 iceberg (×20). El baseline NO es estable bajo round-trip y el gate RECHAZARÍA el baseline puro | `genome.rs` new_baseline vs bounds |
| **N-07** | ALTO | **Riesgo de estado global**: un solo `RiskEngine` para 30 monedas — picos de drawdown contaminados cross-coin; `smoothed_split` (R1.3) es estado compartido跨-monedas Y además está muerto en producción (solo `benchmark.rs` llama `evaluate_order`); la ruta cuántica usa capital 100% para ambos horizontes | `god-engine-core/src/lib.rs:116`; `risk-engine/src/lib.rs:334-364` |
| **N-08** | ALTO | **Etiquetas de entrenamiento sesgadas (triple-barrier sin SL efectivo)**: la condición de SL es matemáticamente imposible (`fut_mid <= long_sl && fut_mid >= short_sl` con long_sl < short_sl) — código muerto; una secuencia que rompe SL y luego rebota al TP se etiqueta GANADORA. El ML aprende P(win) inflada | `src/bin/feature_exporter.rs:76-100` |
| **N-09** | ALTO | **aggTrades "CERTIFICADO" con microestructura unilateral**: la nueva vía vision_sync escribe `TGMTICK1` (real) con spread constante modelado y `bid_qty`/`ask_qty` = 0 en un lado → OBI = ±1.0 en cada tick, exactamente el defecto que `historical.rs` documenta evitar. El magic "real" otorga falsa garantía de pureza L2 a features degeneradas | `src/bin/binance_vision_sync.rs` (--aggtrades) |
| **N-10** | ALTO | **El consumidor del fix R1.6 es código huérfano**: `route_order`/`QuantumOrderRouter` no tiene callers de producción (solo tests) — la defensa anti-slippage por volatilidad NO existe en producción; el fix de `expected_volatility` es correcto pero cosmético | `executor`/`router.rs:19,92` |
| **N-11** | ALTO | **DarkAlpha train/inference roto (persiste)**: `fit` entrena con ReLU, `predict` infiere con tanh; y los normalizadores Welford mutan DURANTE predict (media/varianza se desplazan tick a tick) — backtest ≠ live garantizado, no-determinismo por orden de eventos | `dark-alpha-engine/src/lib.rs:560-561 vs 653-654; 420, 539` |
| **N-12** | ALTO | **Consejo sin feedback (K-07 quedó a medias)**: `record_outcome` jamás se llama — los seniors deliberan con pesos estáticos para siempre; y el snapshot llega sembrado de constantes (`graph_correlation: 0.0, do_calculus_risk: 0.0, slippage 1.5/3.0`) — los seniors Causal/Grafos evalúan fantasmas | `consejo_seniors.rs:547`; `god-engine-core/src/lib.rs:1379-1388, 1464-1473` |
| **N-13** | ALTO | **Genoma bt/prod — persiste el núcleo**: champion_path nunca escrito (muta `default()`); `refresh_models` pisa cada 1000 ticks todo hot-swap no promovido (sin cache de generación; la cosecha del forest aplica sin promote); sin separación de entornos (backtest promueve al store de producción); ghost task 6h `cargo run --bin evolution` quemando CPU y escribiendo config que nada recarga; forest con 5/6 features constantes (`shadow_evaluate_with_features` sin callers); daemon muta 10 de 139 genes | Ver detalle en §4 |

---

## 🔬 3. Módulo por módulo — estado residual

### Módulo 1 y 6 (Ingesta/Estado/Telemetría)
**Certificado:** M2 correcto, macro sintético marcado y sin consumidores ciegos, BinTick versionado (3 magics con `kind`), flight-recorder con seqlock real, parser f64/reorden-tolerante, ws_client endurecido.
**Pendiente:** forensic_auditor muere por `Lagged` (H3, sin cambios); cola SegQueue ilimitada con "FIX #1447" que es comentario, no fix; lakehouse offset-corrupto al llenarse sin rotación ni persistencia; mmap_bus con tearing residual en vuelta de anillo (falta seqlock completo de lectura); `SQLITE_OPEN_NO_MUTEX` latente; historical.rs L2 sintético sin marca de procedencia; clamp [0,100] que borra tasas negativas (inconsistente con su propio test); features 40-53 del feature_exporter congeladas.

### Módulo 2 y 7 (IA/Señales/Cuántica)
**Certificado:** expected_volatility semántica+timing correctos; consejo invocado; swing-NN features reales; geometría por in_dim; multifractal parcialmente consumido; registry lock-free; sin O(n²) en signal path.
**Pendiente:** N-10 (router huérfano), N-11 (ReLU/tanh + normalizadores), N-12 (consejo sin aprender), conformal_p_value constante 0.95 con gen `conformal_alpha` anulado, colinealidad OBI ~8 módulos (hawkes_intensity = 1+|obi|·2 → el gate `hawkes>=1.2` equivale a `|obi|>=0.1`), cutoff_floor aún anulable por min_confidence_btc alto, tramos no genómicos (0.65/0.60 fallback, 0.70+x·0.30), Shockwave con bug dimensional (velocidad absoluta / ATR fraccional), ppo_engine/lead_lag/spectral computados y descartados, API scoped sin callers, test que reescribe el modelo de producción, clonación de Arc por lectura en estrategias (existe `get_value_fast` sin usar).

### Módulo 3, 4, 5, 8 (Estrategia/Ejecución/Riesgo/Backtest)
**Certificado:** firmas OCO correctas; failover; sibling-cancel; insolvencia; kelly.rs; sin pipelines híbridos legacy (el core bajó de ~3187 a 1628 líneas, `process_event` delega limpio).
**Pendiente:** N-03 (piso 5%), N-04 (espejos), N-05 (is_hedge_mode), C-2 K-17 colisión Continuous (sin avance), R3.4 no aplicado (confirmado: sin lifecycle_rank ni dedup de comisión por fill duplicado), R3.7 header-only, TP/SL restauración ±2.5/±1.5 y literales 2.0/2.5 del core que pisan `tp_rr_ratio_btc`, simulator qty sin leverage, divergencia de fees 4 motores, CEB AUTOEVOLUCIÓN FORZADA + activity bonus + doble increment_tick, auditores decorativos (`let _drift_auditor`), retry OCO reutilizando firma/timestamp/coid, `dual` toggling que persiste entre símbolos en el flatten, gen 108 con meseta semántica <1000.

---

## 🧬 4. Genoma backtest↔producción — ranking residual de bloqueos

1. **N-01** (nuevo): gate invertido + kill-switch drift muerto.
2. **refresh_models** pisa hot-swaps no promovidos cada 1000 ticks; cosecha del forest sin promote.
3. **Semillas default**: champion_path nunca escrito; `start_evolution_loop` arranca `Genotype::default()` — el trabajo evolutivo acumulado jamás es punto de partida.
4. **N-02**: bypass del embudo por evolution-engine + from_vector sin reparación RR.
5. **Sin separación de entornos** (rutas fijas del store; el backtest final promueve al genoma de producción).
6. **N-06**: baseline vs bounds divergen en 7 genes — round-trips con clamping silencioso ×2–×100.
7. Ghost 6h + forest 5/6 features constantes + subespacio de 10 genes.
8. Watchdog de rollback sin des-autocorrelación ni lag de maduración (20 trades clustered disparan reversión).

**Progreso real desde la primera auditoría:** darwin gate→apply ✔; daemon walk-forward OOS con fees reales y momentum en sigmas ✔; forest con retornos reales (no vacío) ✔; clamping consistente con bounds ✔; rollback watchdog implementado ✔. El núcleo del problema (herencia, autoridad del embudo, separación de entornos, refresh_models) sigue intacto.

---

## 🎯 5. Acciones priorizadas (actualización de la hoja de ruta)

1. **N-01 (URGENTE, minuto 1)**: restaurar la semántica del daemon — mutar cuando hay edge validado (o mantener champion y explorar nichos alternativos), eliminar el `if true`, revivir el kill-switch drift.
2. **N-03**: eliminar el `clamp(0.05, 1.0)` de la ruta cuántica — usar los clamps genómicos.
3. **N-05**: `is_hedge_mode.store(false)` en el failover -4061 y en pre-flight fallido; abortar arranque sin modo garantizado.
4. **N-02**: reparación RR dentro de `from_vector`; enrutar TODO hot-swab por `promote()`.
5. **N-06**: alinear `new_baseline` con los bounds (o los bounds con el baseline recalibrado a tick) — test de round-trip del baseline puro.
6. **N-04/N-07**: horizonte como discriminador del espejo antes de cerrar; `RiskEngine` por moneda o picos por moneda.
7. **N-08/N-09**: SL como barrera real del loop de etiquetado; OBI agrupado (no ±1) en vision_sync.
8. **E-fase** (persistente): herencia desde envelope activo, cache de generación en refresh_models, separación de entornos del store, cablear `shadow_evaluate_with_features`.

---

*Fin del segundo informe de aseguramiento. Se agrega sin sustraer contenido a la serie forense. Referencias al árbol en `aa24f171` + working tree concurrente (2026-09-07 tarde).*
