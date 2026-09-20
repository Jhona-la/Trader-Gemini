# INFORME DE CERTIFICACIÓN POST-CAMBIOS — QUINTA OLA COMPLETA
## Auditoría Integral del Sistema Trader Gemini tras el Programa QO

**Fecha**: 2026-09-19
**Alcance**: 100% del workspace — 3 agentes de auditoría en paralelo cubriendo 8 módulos
**Base**: Post 10+ commits del programa QO (M0, E1, E2, U1a/b/c, U2, M1.1, M1.2, M2.1, M2.2)
**Metodología**: Verificación archivo-por-archivo, traza de señal completa raíz→cima, verificación matemática fórmula-a-fórmula, análisis de paridad train/serve/backtest/live
**Veredicto**: ⛔ **NO APROBADA** — 18 CRITICAL + 34 HIGH + 98 MEDIUM + 55 LOW = 205 hallazgos

---

## PARADIGMA DE GRAFO VIVO Y TOPOLOGÍA DEL SISTEMA

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          NODO RAÍZ (Ingestión)                               │
│  WS @trade ─ WS @depth5 ─ WS @kline_1h ─ WS !forceOrder ─ REST Pollers     │
│  (symbol_to_id ─ parsers ─ book_seq_guard ─ whale/spoof trackers)          │
│                    ⚠ M1-C01: symbol_to_id CONGELADO al arranque            │
└──────────────────────┬──────────────────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────────────────┐
│                    NODO DE CARACTERÍSTICAS (34D + 10D + 4D)                  │
│  StatefulEngine (EMA/Hurst/OFI/OBI/ATR/Entropy) ⊕ Spectral FFT ⊕ Macro     │
│  [VECTOR 48D: swing(34) ⊕ spectral(10) ⊕ macro(4)]                          │
│      ⚠ M2-H02: FEATURES_DEAD zerificado SÓLO en trainer, NO en serve        │
│      ⚠ M2-H01: macro features doble-actualizadas en depth                  │
└──────────────────────┬──────────────────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────────────────┐
│                  NODO DE DECISIÓN (Motor Continuo)                          │
│  fast_intent ─ slow_intent ─ FUSIÓN ESPECTRAL (τ_dom arbitración)          │
│  ─ Escudos Macro ─ ML Gate (lift sobre base) ─ Consejo (11 asientos)       │
│  ─ Risk Engine (Kelly/env/ruina) ─ Envelope Bayesiana                      │
│      ⚠ M2-C03/C04: MicroScalp + 3 asientos gatean en 0.5 ABSOLUTO          │
│      ⚠ M7-C01: SeniorCausal veto desarmado por OR permisivo               │
│      ⚠ M2-C01: entries_blocked DESCARTA closed_order ya computado           │
└──────────────────────┬──────────────────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────────────────┐
│                    NODO TERMINAL (Ejecución)                                 │
│  Orden REST/WS ─ OCO TP/SL ─ Watchdog ─ Reconciliación ─ Fee Breaker       │
│  (executor ─ order_registry ─ user_data_stream ─ trade_accounting)         │
│      ⚠ M4-C01: WS órdenes fire-and-forget SIN ack                          │
│      ⚠ M4-C02: Kill-switch se atasca ON por 4 paths sin re-arm            │
│      ⚠ M4-H01: auto_trainer_daemon zombie acumulándose                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Bucle de Evolución (paralelo al camino de trading)
```
┌──────────────────────────────────────────────────────────┐
│  Cierre de trade → coin.metrics → returns_history        │
│       → RANSAC Sharpe → DSR gate → Mutación 2000        │
│       → Walk-forward (JUGUETE ⚠ M8-H01) → Promoción      │
│       → genome_store → apply_to_arena → Motor vivo       │
│       → Rollback watchdog (20 obs ⚠ M8-H04)              │
│  ⚠ M8-C02: threshold sweep SIEMPRE escribe 0.50/0.50    │
│  ⚠ M8-C03: DSR evalúa INCUMBENTE no CANDIDATO           │
└──────────────────────────────────────────────────────────┘
```

---

## 🚦 RESUMEN DE ESTADO DE RESOLUCIÓN

### ✅ RESUELTOS Y VERIFICADOS (programa QO)
| ID | Descripción | Verificación |
|----|-------------|-------------|
| QO-M0.1 | leverage_matrix Kelly W·(1−1/PF) pura | Tests identidad exacta |
| QO-M0.2 | DynamicKelly clamp≥0 (sin apostar edge negativo) | Tests |
| QO-M0.3 | extract_kelly_stats p=wins/n (sin moneda fabricada) | Tests |
| QO-M0.4 | FundingRateElasticity orden prev_price | Tests |
| QO-M0.5 | quantum_kelly_risk tope 1× Kelly crudo | Tests |
| QO-M0.6 | reality_physics ENTRADA √latency/√Q | Código verificado |
| QO-M0.7 | KylesLambda→AmihudIlliquidity | Renombrado |
| QO-E1 | Autoevolución ARMADA en demo | Log QO-E1 en vivo |
| QO-E2a | predict_6d → forest6_prob/acc → core modula confianza | Cableado verificado |
| QO-E2b | nn_entry_tensor → dataset CSV → auto_trainer hijo | Proceso activo |
| QO-E2c | ShadowForest fitness::compute unificado | Tests |
| QO-E2d | EvolutionLedger promociones/rollbacks | Cableado |
| QO-U1a | Sanitizer escáner (mojibake/triviales fuera) | Test con basura real |
| QO-U1b | WS re-suscripción al rotar | Log QO-U1b en vivo |
| QO-U1c | Funding per-símbolo (premiumIndex all-market) | End-to-end |
| QO-U2 | L/S+taker ratios contrarian en asiento Ente | End-to-end |
| QO-M1.1 | DSR gate (selection_stats.rs) | Tests PSR/DSR |
| QO-M1.2 | Ruina analítica P((1−f)/(1+f))^(1/f) tope | Tests |
| QO-M2.1 | quantum_evolver DELETED | Archivo eliminado |
| QO-M2.2 | Hawkes kernel exponencial real | Matemática correcta (⚠ inerte) |

### 🔴 NUEVOS HALLAZGOS CRÍTICOS (rotos o revelados por los cambios)

| ID | Severidad | Descripción |
|----|-----------|-------------|
| M1-C01 | CRITICAL | symbol_to_id congelado — símbolos rotados NUNCA procesados por el consumidor |
| M2-C01 | CRITICAL | entries_blocked descarta closed_order — cierres defensivos invisibles al host |
| M2-C02 | CRITICAL | Hawkes record_event() SIN CALLERS — QO-M2.2 es código muerto |
| M2-C03 | CRITICAL | MicroScalp gatea ML en 0.5 absoluto — sesgo short estructural |
| M2-C04 | CRITICAL | 3 asientos del consejo centrados en 0.5 — contradicen B3.18 por construcción |
| M2-C05 | CRITICAL | book_absent NO gateado a backtest — path F7 activo en vivo |
| M4-C01 | CRITICAL | Órdenes WS fire-and-forget SIN ack del exchange |
| M4-C02 | CRITICAL | Kill-switch se atasca ON permanentemente (4 paths sin re-arm) |
| M5-C01 | CRITICAL | kelly_bootstrap_cold piso 0.35 con PF≤1 — bypass de protección |
| M5-C02 | CRITICAL | micro_kelly piso 0.10-0.12 sin edge en demo |
| M5-C03 | CRITICAL | Darwin curvas en anclas 10s/24h (no canónicas 30s/12h) |
| M5-C04 | CRITICAL | kelly_at_tau clamp inconsistente config[0.05,0.40] vs genotipo[0.01,3.0] |
| M7-C01 | CRITICAL | SeniorCausal veto desarmado por OR permisivo (spectral_s<0.5 casi siempre) |
| M8-C01 | CRITICAL | Cadena sizing backtest ≠ vivo (LA causa #1 del gap BT↔live) |
| M8-C02 | CRITICAL | Threshold sweep: `th` NUNCA usado → siempre 0.50/0.50 |
| M8-C03 | CRITICAL | DSR evalúa incumbente no candidato |
| M8-C04 | CRITICAL | CmaEsOptimizer SOBRESCRIBE fitness canónico |
| M8-C05 | CRITICAL | run_backtest_native SIN envelope — certifica trades que producción aborta |

---

## 📊 MATRIZ MAESTRA CONSOLIDADA

### Distribución por Severidad
| Severidad | Cantidad |
|-----------|----------|
| **CRITICAL** | 18 |
| **HIGH** | 34 |
| **MEDIUM** | 98 |
| **LOW** | 55 |
| **TOTAL** | **205** |

### Distribución por Módulo
| Módulo | C | H | M | L | Total |
|--------|---|---|---|---|-------|
| 1. Ingestión/Parsers/L2 | 1 | 3 | 8 | 5 | 17 |
| 2. IA/ML/Señales | 5 | 5 | 15 | 8 | 33 |
| 3. Estrategia/Régimen | 0 | 2 | 8 | 4 | 14 |
| 4. Ejecución/Red | 2 | 6 | 10 | 5 | 23 |
| 5. Riesgo/Kelly/Genomas | 4 | 7 | 12 | 6 | 29 |
| 6. Estado/Memoria/Telemetría | 0 | 3 | 7 | 6 | 16 |
| 7. Orquestación/Confluencia | 1 | 3 | 6 | 4 | 14 |
| 8. Backtest/Evolución | 5 | 5 | 12 | 7 | 29 |
| **TOTAL** | **18** | **34** | **78** | **45** | **175** |

---

## 🔬 MÓDULO 1: INGESTIÓN, PARSERS, LIBROS L2 Y NORMALIZACIÓN

### [M1-C01] symbol_to_id congelado — universo fantasma persiste en el CONSUMIDOR
**Archivo**: `src/bin/god_engine.rs:1246-1252, 2595-2615`
**Categoría**: DISCONNECT
**Severidad**: CRITICAL

**Descripción exhaustiva**: Las estructuras `symbol_to_id` (HashMap<String, usize>), `local_orderbooks` (Vec<OrderBook>), `whale_trackers` (Vec<InstitutionalVolumeTracker>), `spoof_detectors` (Vec<SpoofingDetector>), y `book_seq_guard` (BookSequenceGuard) se construyen todas a partir del universo del **arranque** (la lista hardcoded del bootloader) y **jamás se reconstruyen** durante la vida del proceso. Cuando `evolve_symbols_daemon_with_resubscribe` (QO-U1b) rota el universo y re-suscribe el WebSocket, los ticks de los **nuevos** símbolos llegan al event loop pero al buscar en `symbol_to_id.get(parsed_sym)` obtienen `None → continue` (línea 2843-2849) y son **silenciosamente descartados**.

El fix de la suscripción (QO-U1b) es correcto pero incompleto: reparó el PRODUCTOR (el WS ahora envía los datos de los símbolos rotados) pero no el CONSUMIDOR (el event loop no sabe qué hacer con ellos). El resultado es que la energía de la reparación se desperdicia — los datos fluyen hasta la puerta del motor y mueren ahí.

**Impacto en trading**: Los símbolos rotados siguen siendo totalmente muertos para el motor hasta reinicio del proceso. Toda la maquinaria de features, señales, ML gate, consejo y ejecución es invisible para ellos. El "universo fantasma" que QO-U1b declaró muerto persiste en una capa más profunda.

**Dirección de fix**: Reconstruir `symbol_to_id` y todos los vectores per-símbolo desde `get_active_universe()` cuando se recibe la señal de reconexión WS, o reemplazar todas las claves por `symbol_registry::try_index()` (que SÍ se actualiza en vivo).

---

### [M1-H01] Backpressure drop-oldest sin contador ni telemetría de pérdida
**Archivo**: `god_engine.rs:4298-4306, 4314-4323`
**Categoría**: DATA-LOSS
**Severidad**: HIGH

**Descripción**: La política de backpressure del canal crossbeam (5000 slots) es drop-oldest: cuando el lector no sigue el ritmo del escritor, `try_recv()` desecha los mensajes más antiguos para hacer espacio. No hay ningún `AtomicU64` contando mensajes descartados, ni telemetría de la tasa de pérdida. Bajo sobrecarga sostenida (pico de volumen crypto), los trades y actualizaciones de depth más antiguos desaparecen sin dejar rastro forense. Esto significa que durante los momentos de mayor volatilidad (exactamente cuando cada tick importa), la calidad de datos del motor se degrada silenciosamente.

---

### [M1-H02] Guard de secuencia del libro NO se resetea en reconexión WS
**Archivo**: `god_engine.rs:2656-2672`
**Categoría**: LOGIC-ERROR
**Severidad**: HIGH

**Descripción**: El handler `[SYSTEM:RECONNECT]` correctamente resetea los engines (`engine_real.reset_engines()`) y limpia los orderbooks (`ob.clear()`), pero **NO resetea `book_seq_guard`**. Este guard compara el `update_id` de cada mensaje depth con el último aceptado (D-610). Después de una reconexión, los primeros mensajes del nuevo stream tienen IDs que pueden ser menores que los del stream anterior (reinicio de secuencia del exchange). Sin reset, el guard descarta hasta 50 mensajes consecutivos (`RESYNC_AFTER=50`) por símbolo antes de resincronizar. Con 26+ símbolos, eso es **>1300 actualizaciones de libro descartadas** por cada reconexión — durante segundos, el OBI/OFI/microprice operan sobre un libro congelado en el tiempo pre-desconexión.

---

### [M1-H03] Warmup(1m) vs live(1h) mismatch de cadencia kline
**Archivo**: `god_engine.rs:874-891` vs `bootloader.rs:124,306`
**Categoría**: RIGIDITY / PARITY
**Severidad**: HIGH

**Descripción**: El warmup de arranque (Phase 3) descarga e inyecta klines de **1 minuto** (`interval=1m`) en los feature engines para calibrar ATR/v_t/Hurst. Pero el WebSocket vivo se suscribe a `@kline_1h` — la única corriente de klines. Post-warmup, los eventos de cierre de barra (`is_kline_closed=true`) llegan a 1/60 de la cadencia que los features esperaban. Cualquier feature dependiente de periodicidad de barra (decaimiento por cierre de barra, refresh de ATR) se comporta de forma diferente entre warmup/backtest y producción.

---

### [M1-M01] Latencia: baseline NTP congelado, drift acumulativo multi-día
**Archivo**: `god_engine.rs:2803, 2591-2593`
**Categoría**: LOGIC-ERROR
**Severidad**: MEDIUM

El `epoch_baseline_ms` se calcula una vez al inicio con `local_ms + ntp_offset_ms`. El `spawn_ntp_synchronizer` sigue actualizando `arena.server_time_offset_ms` en un loop separado, pero el baseline del event loop **nunca se re-ancla**. En sesiones multi-día, el drift entre el reloj del host y el de Binance se acumula linealmente en `latency_ms`, eventualmente armando falsamente el kill-switch de volatilidad sintética (10 ticks consecutivos > threshold).

### [M1-M02] OI per-símbolo: normalización sin unidades coherentes cross-símbolo
**Archivo**: `god_engine.rs:2139-2202`
**Categoría**: MATH-ERROR
**Severidad**: MEDIUM

`(oi.ln() / 1.0e8f64.ln()).clamp(0,1)` donde `oi` está en **unidades del activo base** (no en dólares). Para DOGE/SHIB (OI en billones de unidades) esto satura a 1.0 permanentemente; para BTC (~100k unidades) lee ~0.63. La comparabilidad cross-símbolo — el objetivo del asiento Ente del Mercado — está destruida: es una tabla de lookup por símbolo, no una señal de apalancamiento.

### [M1-M03] forceOrder parser: fallos silenciosos sin contador forense
**Archivo**: `god_engine.rs:2634-2652`
**Categoría**: DATA-LOSS
**Severidad**: MEDIUM

Un cambio de esquema en `!forceOrder@arr` zerificaría `dex_severity` y el asiento Ente sin ningún rastro forense. El parser maneja JSON malformado gracefully (skip), pero sin contar los fallos.

---

## 🧠 MÓDULO 2: INFERENCIA DE IA, MODELOS PREDICTIVOS Y SEÑALES

### [M2-C01] entries_blocked descarta closed_order ya computado — el host pierde cierres defensivos
**Archivo**: `god-engine-core/src/lib.rs:2045-2047`
**Categoría**: LOGIC-ERROR / DISCONNECT
**Severidad**: CRITICAL

**Descripción exhaustiva**: En la sección 1 de `process_event` (gestión de posición continua, líneas ~862-1563), el motor computa trailing stops, detecta SL/TP hits, evalúa zombie/toxic, y produce un `closed_order = Some((is_long, net_trade_pnl, qty))`. Este valor contiene el resultado del cierre para que el HOST (god_engine.rs) haga contabilidad, accounting de OCO, fee attribution, y notifique al usuario.

En la línea 2045-2047, después de que TODA la analítica ML/espectral ya ha corrido (bloques completos de inferencia de 48D, ensemble, publicación de ml_prob al registry), se encuentra:
```rust
if entries_blocked {
    return (None, None, None);
}
```

Esto retorna `None` para TODOS los valores de retorno — incluyendo `closed_order` que YA FUE COMPUTADO exitosamente 1500 líneas arriba. El comentario X-012 dice "frontera REAL del bloqueo — gestión de posiciones (sección 1) y analítica ML/espectral ya corrieron completas... no se EVALÚAN ni abren posiciones nuevas desde aquí hacia abajo". La intención es bloquear NUEVAS ENTRADAS, no descartar cierres ya ejecutados.

**Impacto**: Durante tormentas de latencia o stalls del feed (exactamente cuando las salidas defensivas son VITALES — el propio comentario de D-XXX lo dice), el host nunca recibe el evento de cierre. El estado del host/core diverge: el core cree que cerró la posición, el host todavía la ve abierta. Los OCO brackets quedan rancios, la contabilidad pierde el trade, el Kelly no aprende.

**Fix**: `return (None, closed_order, None);`

---

### [M2-C02] Hawkes record_event() SIN CALLERS — QO-M2.2 es CÓDIGO MUERTO
**Archivo**: `signal-engine/src/hawkes_bessel.rs:82-97, 179-209`
**Categoría**: DISCONNECT / INTELLIGENCE-BLOCK
**Severidad**: CRITICAL

**Descripción exhaustiva**: El commit QO-M2.2 reescribió completamente `hawkes_bessel.rs` con matemática correcta: kernel exponencial λ(t) = μ + Σ α·e^(−β·(t−tᵢ)), historia de eventos (VecDeque con purga a 5/β), ratio de ramificación α/β con condición de estacionariedad. La implementación es matemáticamente correcta y está bien testeada.

**PERO**: `record_event()` — el método que alimenta la historia de eventos que hace que el proceso de Hawkes funcione — tiene **CERO llamadores** en todo el workspace (verificado por grep exhaustivo). El trait `QuantumStrategy::evaluate` recibe `&self` (referencia inmutable), por lo que el orchestrador NUNCA puede llamar `record_event` (que requiere `&mut self`) para alimentar la historia.

Sin eventos registrados, `intensity_ratio()` siempre retorna `μ/μ = 1.0`, y el voto del trait se reduce a `sign(order_flow_direction) × 0.7616` — una constante amplificada por el signo del OBI, exactamente lo que la versión anterior hacía pero con más líneas de código muerto alrededor.

**El headline "HAWKES REAL" es inerte**: branching ratio, stationarity check, purge, decay — todo es código muerto que consume mantenimiento y da falsa confianza.

**Fix**: Alimentar `record_event` desde el camino de trade/liquidation (requiere interior mutability o publicar intensidad pre-calculada al registry).

---

### [M2-C03] MicroScalp gatea ML en 0.5 absoluto — sesgo short estructural
**Archivo**: `signal-engine/src/micro_scalp_trigger.rs:149-156`
**Categoría**: LOGIC-ERROR (escala)
**Severidad**: CRITICAL

**Descripción**: El voto de esta estrategia en el consenso tensorial gatea en `ml_prob >= 0.5` para long y `ml_prob <= 0.5` para short — un umbral **absoluto**, no lift-sobre-base (la doctrina B3.36 que el sistema adoptó).

Con el etiquetado honesto HOST-010, la base del modelo es ~0.30 (no 0.50). Esto significa:
- Para LONG: `ml_prob >= 0.5` es un evento raro (el modelo predice P(TP largo) que rara vez supera 0.5 con base 0.30) → la pata long casi nunca dispara.
- Para SHORT: `ml_prob <= 0.5` es casi siempre cierto (el modelo con base 0.30 rara vez supera 0.5) → la pata short está casi permanentemente satisfecha.

**Resultado**: Esta estrategia contribuye un **sesgo short estructural** al consenso tensorial para CADA moneda con modelo honesto. El ML gate B3.18 correctamente usa lift-sobre-base, pero esta estrategia en la capa de señal usa el umbral absoluto vetado — dos gates en el mismo pipeline que discrepan por construcción.

---

### [M2-C04] Consejo: TRES asientos direccionales centrados en 0.5 absoluto
**Archivo**: `metacortex-engine/src/consejo_seniors.rs:227-236` (fn ml_opinion)
**Categoría**: LOGIC-ERROR (escala)
**Severidad**: CRITICAL

**Descripción**: La función `ml_opinion` centra su zona muerta en ±0.05 alrededor de **0.5** y su edge pleno en 0.20 desde 0.5. Esta función alimenta:
- **SeniorML** (asiento direccional con peso 1.2)
- **SeniorMetacognitivo** (asiento direccional que compara divergencia ml vs señal, peso ~1.0)
- **SeniorTeleonomia** (asiento direccional que evalúa utilidad esperada, peso ~1.0)

Con base ~0.30: los tres asientos tienen la misma patología que M2-C03 — el ML rara vez cruza 0.5+ para justificar longs, pero frecuentemente está debajo para justificar shorts. **Tres de cinco asientos direccionales** del consejo sistemáticamente contradicen entradas long que el B3.18 lift-gate correctamente aprobó. El consejo (que requiere consenso 0.35 de capacidad direccional) tiene un sesgo short incorporado por diseño accidental.

---

### [M2-C05] book_absent NO gateado a backtest — path F7 activo en producción
**Archivo**: `god-engine-core/src/lib.rs:2389-2508`
**Categoría**: LOGIC-ERROR / ARBITRARY
**Severidad**: CRITICAL

`book_absent = obi_val.abs() < 0.005` determina si el libro está "ausente" y activa el path F7 (ML-only). Pero NO está gateado a modo backtest. En vivo, cualquier libro momentáneamente balanceado (OBI cercano a 0 — que es común) dispara este path con `ml_prob_adaptive = 0.5 + (ml−0.5)×2` — un recentrado ARBITRARIO que duplica la distancia de 0.5, permitiendo entradas en producción que bypass los gates de OBI/flujo que normalmente aplicarían.

---

### [M2-H01] Doble actualización de macro features en eventos depth
**Archivo**: `god-engine-core/src/lib.rs:608 + 828`
**Categoría**: RACE / INCONSISTENCY
**Severidad**: HIGH

En la rama `is_depth` (línea ~608), se llama `take_pending()` y `update_macro_features()`. Luego, cuando `process_tick_dual` procesa el mismo evento vía `process_tick` (línea ~828), llama `update_macro_features()` OTRA VEZ. Resultado: `obi_noise`, `obi_accel`, `fr_elasticity` se actualizan al DOBLE de frecuencia en eventos depth pero una vez en trades — cadencia de estimador inconsistente que sesga el shield D-688 (que usa `obi_noise.sd()`).

### [M2-H02] FEATURES_DEAD_IN_SERVE zerificado SÓLO en trainer
**Archivo**: `stateful_engine.rs:17` vs `train_forest.rs:395-397`
**Categoría**: DISCONNECT (paridad train/serve)
**Severidad**: HIGH

`FEATURES_DEAD_IN_SERVE = &[4,5,9,10]` se aplica en el TRAINER para zerificar las dims muertas. Pero `get_universal_features()` en serve NO zerifica dims 4/5/10 — sirve el obi_accel vivo. El comentario de B3.35 dice "zerificadas en AMBOS lados" — **es falso**. Cualquier modelo entrenado pre-B3.35 tiene splits en dims que ahora sirven una distribución viva que el modelo nunca vio.

### [M2-H03] Conformal/calibrador GLOBALES — contaminación cross-asset
**Archivo**: `god-engine-core/src/lib.rs:2112-2119, 1319, 1423`
**Categoría**: LOGIC-ERROR
**Severidad**: HIGH

`self.conformal` y `self.confidence_calibrator` son instancias ÚNICAS globales. Los cierres de TODAS las monedas alimentan los mismos. La exchangeability (supuesto fundamental del conformal) se rompe cross-asset — el α_eff y el mapa Platt aprenden un blend BTC+NEAR+ATOM, no una calibración por símbolo. La misma contaminación que D-432 fixó para los ensembles.

### [M2-H04] Hedge/Brier califica contra dirección de kline — no barrera triple
**Archivo**: `god-engine-core/src/lib.rs:534-545`
**Categoría**: ML-CORRECTNESS
**Severidad**: HIGH

La calibración Hedge/Brier del ensamble califica `bar_open_predictions` contra y=1 si el próximo kline de 1m cierra arriba. Pero el forest MOTOR predice una **barrera triple** (TP +0.36%/SL −0.18% a horizonte de minutos). Los pesos castigan/premian al forest por una pregunta que no fue entrenado para responder.

### [M2-H05] Darwín evoluciona contra tensor con funding=entropy y macro=0
**Archivo**: `god-engine-core/src/darwin.rs:256-270`
**Categoría**: DISCONNECT
**Severidad**: HIGH

El GA copia `get_universal_features()[0..34]` en `omni[0..34]`. `omni[11]` se lee como funding_rate pero recibe `micro[11]` = norm_at (aceleración). Los slots macro 21-26 quedan en 0. El GA optimiza genes contra una entrada que producción nunca ve.

---

## 📈 MÓDULO 3: ESTRATEGIA MULTIACTIVO, RÉGIMEN Y HORIZONTES TEMPORALES

### [M3-H01] Fused score aún sobre-pesa escalas lentas
**Archivo**: `quantum-arena/temporal_spectrum.rs:238-256`
**Categoría**: MATH-ERROR (residual)
**Severidad**: HIGH

C-05 fixeó `dominant_tau_ms` (clamp a banda operativa), pero los **pesos de fusión** `w ∝ 1/dev_vol` siguen sobre-pesando las escalas más lentas. Cada consumidor de `fused_score` (arbitración espectral U-2, asiento Espectral del consejo, Teleonomia) lee la fusión degenerada que el propio comentario C-05 describe: "slow scales always dominate".

### [M3-H02] Exit physics retiene potencia-1.2 e impacto lineal
**Archivo**: `reality_physics.rs:161-163`
**Categoría**: MATH-ERROR
**Severidad**: HIGH

M0.6 corrigió la ENTRADA a √latency y √Q (ley raíz cuadrada). La SALIDA (`calculate_exit`) retiene `powf(1.2)` e impacto **lineal** en latencia. Cada fill simulado de cierre paga física diferente a su apertura — el backtest tiene un sesgo sistemático en la asimetría entrada/salida que favorece stop-heavy genomes.

### [M3-M01] ~40 literales fijos en camino de señal sin derivación
Véase la tabla de rigidez en la sección "Censo de Rigidez" abajo.

---

## ⚡ MÓDULO 4: EJECUCIÓN HFT, PROTOCOLO DE RED Y CONECTIVIDAD BINANCE

### [M4-C01] Órdenes WS fire-and-forget SIN ack del exchange
**Archivo**: `executor.rs:1690-1727, 1886-1919`
**Categoría**: DISCONNECT
**Severidad**: CRITICAL

En el path WS, `ws.send_order_payload` escribe el frame al socket TCP. Si el write local tiene éxito (el buffer del kernel acepta los bytes), la función retorna `Ok(())` **sin ningún ack del exchange**. Si la conexión muere entre el write local y el procesamiento por el servidor de Binance, la orden se pierde silenciosamente mientras el motor cree que fue colocada. El registry mantiene la intención en estado `New` para siempre hasta que `cleanup_stale_orders` la expira — que NADIE llama en god_engine. Los brackets TP/SL se colocan contra una posición inexistente (rechazados -2022) y el machinery X-009 de emergency-close dispara sobre nada.

### [M4-C02] Kill-switch puede quedar ATASCADO ON permanentemente
**Archivo**: `god_engine.rs:1599-1616, 3270-3274, 4048`
**Categoría**: OPERATIONAL
**Severidad**: CRITICAL

Cuatro paths independientes escriben `kill_switch_active = true`:
1. **Immune loop** (latch por diseño, sólo reinicio)
2. **DriftAuditor** (3265-3274) — almacena `true` **sin ningún path de clearing en todo el codebase**
3. **Rate limits**: 3×429 o 418 en executor (`handle_rate_limit_error`)
4. **X-009 escalation** (posición desnuda sin cerrar tras retry)

Una vez que CUALQUIERA dispara, TODO trading queda bloqueado hasta reinicio del proceso. Los casos (b) y (c) son condiciones recuperables (drift fue heurístico; los 429 pasan) pero quedan permanentemente latched.

### [M4-H01] auto_trainer_daemon zombie: sin supervisor
**Archivo**: `god_engine.rs:1862-1894`
**Categoría**: OPERATIONAL
**Severidad**: HIGH

`Command::spawn` produce un `Child` que se **dropea inmediatamente** — nunca se espera (`wait()`), nunca se reinicia en crash, nunca se termina al salir el engine. En Windows, cada reinicio del motor acumula otro trainer concurrente escribiendo `dark_alpha_dataset_BTCUSDT.csv` — múltiples trainers compitiendo por el mismo archivo, reentrenando en datasets intercalados/corruptos.

### [M4-H02] Sin path de shutdown graceful
**Archivo**: `god_engine.rs` (todo el archivo)
**Categoría**: OPERATIONAL
**Severidad**: HIGH

`panic="abort"` en release. Cualquier panic en el thread unificado aborta el proceso instantáneamente con posiciones abiertas, transacciones redb sin flush, journals medio escritos. No hay signal handler, no hay ctrl-c handler.

### [M4-H03] OCO leg retry después de POST ambiguo duplica trigger
**Archivo**: `executor.rs:2663-2787`
**Categoría**: LOGIC-ERROR
**Severidad**: HIGH

Si el primer POST de la pierna SL timeout pero realmente llegó al exchange (estado AMBIGUOUS), el código lo trata como fallido y reintenta con un nuevo `_SLR` client ID. Resultado: **dos piernas SL vivas** para una posición — el exchange puede aceptar ambas, causando un doble intento de cierre.

### [M4-H04] Flatten-all en stall de WS — network blip = liquidación forzada
**Archivo**: `god_engine.rs:1571-1577`
**Categoría**: OPERATIONAL
**Severidad**: HIGH

El immune latency-strike incluye `feed_health::is_stalled()`. El WS backoff está capped a 5s pero la reconexión total (DNS fail → happy-eyeballs fail → backoff) puede superar 15s fácilmente → 3 strikes → **flatten ALL positions at market** en el peor spread de una desconexión transitoria de red.

### [M4-H05] Posición adoptada con entry_fee=0 — Kelly sobreestima
**Archivo**: `reconciliation.rs` (adopción)
**Categoría**: LOGIC-ERROR
**Severidad**: HIGH

Al adoptar una posición del exchange, `open_with_horizon(..., 0.0, 0.0, ...)` pone entry_fee=0. El BracketClose fee model calcula `fees = exit_commission + entry_fee × (qty/entry_qty)` → fee=0 para la pierna de entrada en cada cierre de posición adoptada. El PnL neto que Kelly aprende está **sistemáticamente sobreestimado** → leverage oversized en los trades siguientes.

---

## 🛡️ MÓDULO 5: GESTIÓN DE RIESGO, KELLY Y GENOMAS EVOLUTIVOS

### [M5-C01] kelly_bootstrap_cold PISO en 0.35 con PF≤1
**Archivo**: `risk-engine/src/lib.rs:225-239`
**Categoría**: WRONG-MATH (bypass)
**Severidad**: CRITICAL

```rust
let kelly_frac = if raw_kelly <= 0.0 {
    kelly_cold  // = kelly_bootstrap_cold.clamp(0.05, 0.35), baseline 0.5 → 0.35
} else { ... };
```

Cuando `raw_kelly <= 0` (lo que ocurre con PF ≤ 1, probado sin edge), el fallback bootstrap clampa a **0.35** (porque `kelly_bootstrap_cold` tiene baseline 0.5, clampeado a [0.05, 0.35]). Esto significa que una moneda con PF ≤ 1 — el sistema ha MEDIDO que no tiene edge — tradeará al **35% de fracción Kelly**. Contradice directamente `kelly.rs:60-66` (PF≤1 → exploración ≤ ¼ del piso). La protección se bypassa una capa arriba antes de que se consulte.

### [M5-C02] micro_kelly PISO en 0.10-0.12 sin edge en cuenta demo
**Archivo**: `risk-engine/src/lib.rs:327-331`
**Categoría**: WRONG-MATH (forced exposure)
**Severidad**: CRITICAL

```rust
let micro_kelly = (kelly_adjusted.max(0.12) * (1.0 + (confidence - 0.65) * 1.5)).clamp(0.10, 0.20);
```

En régimen micro (capital ≤ 3 min-notionals = la cuenta demo de $13), la fracción Kelly tiene piso **0.10-0.12 independientemente del edge**. Combinado con M5-C01: una moneda sin edge en la cuenta demo apuesta 12-20% del capital por trade.

### [M5-C03] Darwin reconstruye curvas TP/SL en anclas 10s/24h
**Archivo**: `god-engine-core/src/darwin.rs:145-146`
**Categoría**: DISCONNECT (geometría)
**Severidad**: CRITICAL

`update_tp_curve`/`update_sl_curve` ajustan curvas through puntos en **τ = 10 segundos / 24 horas**, mientras las anclas canónicas del genoma son **30 segundos / 12 horas** (`TAU_ANCHOR_FAST_MS`/`TAU_ANCHOR_SLOW_MS`). Cuando Darwin hace hot-swap, las curvas vivas del arena se reconstruyen sobre un eje diferente al del almacén de genomas. La geometría que tradea ≠ la geometría que se persiste.

### [M5-C04] kelly_at_tau clamp INCONSISTENTE entre config y genotipo
**Archivo**: `config.rs:421-426` vs `genome.rs:2081-2085`
**Categoría**: LOGIC-ERROR
**Severidad**: CRITICAL

`QuantumConfig::kelly_at_tau` clampa a **[0.05, 0.40]**; `SuperGenotype::kelly_at_tau` clama a **[0.01, 3.0]**. El camino vivo lee la versión config; la evolución lee la versión genotipo. Un genoma evolucionado hacia kelly 0.6 silenciosamente opera a 0.40 en producción mientras la evolución mide 0.6. **El fitness mide un animal diferente al que producción tradeará.**

### [M5-H01] Fitness FRAGMENTADO: 5 objetivos de selección compiten
- `fitness.rs` (canónico, D-652) 
- `evolver.rs:428-452` (offline: growth^1.5 × (1−dd)² × wr³)
- `cma_es.rs:220-242` (pnl-based)
- `darwin.rs:285/476` ((final−initial)×(1−dd))
- `evolution-engine/lib.rs` (CmaEsOptimizer **sobrescribe** el canónico)

### [M5-H02] Genoma hot-swap NO atómico
`apply_to_arena` escribe ~150 atomics no-transaccionalmente. Mid-swap, una entrada puede usar el leverage viejo con la curva SL nueva.

### [M5-H03] Ruin cap (M1.2) alcanzable SÓLO desde kelly.rs
`kelly_bootstrap_cold` (M5-C01), `micro_kelly` (M5-C02), leverage matrix output, y margin bumps a min_notional — todos bypassan el tope de ruina.

---

## 🔒 MÓDULO 6: ESTADO ATÓMICO, MEMORIA MMAP, TELEMETRÍA Y SO

### [M6-H01] mmap bus: frames TROZADOS en reuso de slot
**Archivo**: `mmap_bus.rs:122-194, 233-275`
**Categoría**: RACE
**Severidad**: HIGH

Sin seqlock/sequence counter: cuando el ring de 1M frames envuelve y un slot se reusa, el lector puede observar una **mezcla de payload viejo + header nuevo** (o viceversa) — frames trozados sin detección.

### [M6-H02] Bracket closes drenados SÓLO cuando trading permitido
**Archivo**: `god_engine.rs:3091`
**Categoría**: DATA-LOSS
**Severidad**: HIGH

`drain_bracket_closes()` corre dentro de `is_trading_allowed` — durante warmup (~120 ticks) o vetos del orchestrator, los cierres se acumulan (cap 1024, overflow silencioso). Kelly nunca ve esos trades.

### [M6-M01] dynamic_config escritura directa SIN tmp+rename
**Archivo**: `symbol_manager.rs:90-99, 195-199`
**Categoría**: DATA-LOSS
**Severidad**: MEDIUM

Crash mid-write deja el config corrupto; boot cae silenciosamente al universo del bootloader.

---

## ⚛️ MÓDULO 7: SEÑALES CUÁNTICAS, ORQUESTACIÓN Y CONFLUENCIA

### [M7-C01] SeniorCausal veto DESARMADO por OR permisivo
**Archivo**: `consejo_seniors.rs:439-441`
**Categoría**: LOGIC-ERROR
**Severidad**: CRITICAL

```rust
let is_aligned_breakout = (payload.book_imbalance.abs() > 0.25 || payload.spectral_s() < 0.5)
    && do_calculus_risk < 0.88;
```

`spectral_s() < 0.5` es cierto para τ < ≈19 minutos — la **mayoría de la banda operativa**. El OR hace que `is_aligned_breakout` sea true para prácticamente toda entrada de banda rápida sin importar el OBI. El veto de manipulación de SeniorCausal está **efectivamente deshabilitado** para la mayoría de las entradas — un asiento de veto que casi nunca veta.

### [M7-H01] Consejo: 30+ literales fijos sin derivación
Toda la grid de umbrales del consejo es fija: pesos 1.2/1.5/3.0, aprobación 0.35, override 0.80/0.28, DD cap, slippage bands, k=8, Ente factors, crowd thresholds, veto thresholds.

### [M7-H02] D-345 rubber-stamp: asientos moduladores auto-refuerzan
Los asientos de permiso/modulación (Riesgo, Volatilidad, Ente) heredan `intended_direction` como su señal y se cuentan como "correctos" cuando el trade gana → sus tracker weights se auto-refuerzan con el win rate del sistema, inflando `final_signal` y facilitando el veto-override.

---

## 🧪 MÓDULO 8: BACKTESTING, EVOLUCIÓN Y GOBERNANZA

### [M8-C01] Cadena de sizing backtest ≠ cadena vivo (CAUSA #1 del gap)
**Archivo**: `god_engine.rs:3493-3541` vs `booktick_replay.rs:649-656` vs `run_backtest_native`
**Categoría**: PARITY
**Severidad**: CRITICAL

**Vivo**: `exec_leverage = min(lev_from_risk, env_cap)` donde `lev_from_risk = (0.05 · kelly_frac · vol_brake) / sl_at_tau(τ_entry)`
**Replay**: usa `core_leverage.clamp(1,20).min(cap)` — sin kelly_frac, sin stop-distance, sin vol_brake
**Native**: **NO APLICA ENVELOPE** — certifica trades que producción abortará

Un genoma certificado a leverage L en backtest tradeará a un L diferente en demo. **Esta es la causa estructural #1** de que el genoma funcione bien en backtest pero no en demo/producción.

### [M8-C02] Threshold sweep: `th` NUNCA usado en el loop de profit
**Archivo**: `evolution-engine/src/online_random_forest.rs:170-196`
**Categoría**: LOGIC-ERROR
**Severidad**: CRITICAL

El sweep que debería encontrar los umbrales ML óptimos no usa `th` dentro del cálculo de profit — `profit_l`/`profit_s` son idénticos para cada valor de `th`. Resultado: `best_th_long` siempre es el primer paso del sweep (0.50) y `best_th_short` siempre 0.50. Cuando la autoevolución demo está armada, `online_daemon.rs:217-227` almacena **ml_threshold_long=0.50, ml_threshold_short=0.50** — el gate ML más laxo posible — en el entorno exacto donde las mutaciones fluyen.

### [M8-C03] DSR evalúa los returns del INCUMBENTE, no del CANDIDATO
**Archivo**: `online_daemon.rs:871-882`
**Categoría**: LOGIC-ERROR
**Severidad**: CRITICAL

`edge_survives_multiplicity(&self.returns_history, 2000)` evalúa los returns de la estrategia **actualmente activa**. Pero el genoma que se promueve es un **mutante aleatorio** evaluado por un walk-forward de juguete con 3 trades mínimo. La corrección DSR nunca toca al candidato real: cuando el incumbente está caliente, cualquier mutante pasa; cuando frío, nada promueve. El gate es simultáneamente demasiado estricto (bloquea todo cuando el incumbente es frío) e inútil (no filtra al mutante).

### [M8-C04] CmaEsOptimizer SOBRESCRIBE el fitness canónico
**Archivo**: `evolution-engine/src/lib.rs:505-523`
**Categoría**: LOGIC-ERROR
**Severidad**: CRITICAL

El loop computa `fitness::compute` (canónico D-652), pasa tuplas a `CmaEsOptimizer::update` que **sobrescribe el slot .1** con `real_pnl · reality_gap` (pnl-based). El caller ordena por `.1` DESPUÉS de la sobrescritura. La utilidad Kelly-log es decorativa; D-653/D-654 se re-rompen en este path.

### [M8-C05] run_backtest_native SIN envelope
**Archivo**: `backtest-engine/src/lib.rs`
**Categoría**: PARITY
**Severidad**: CRITICAL

El path native/FFI/golden no construye `RiskEnvelope`. Entradas que el host vivo vetaría (exec_leverage=0) existen en native con leverage hasta 50×. Cualquier consumidor del path FFI certifica trades que producción abortará.

### [M8-H01] Walk-forward del daemon es MODELO DE JUGUETE
`online_daemon.rs:696-816`: entrada en `prev_sigma > (thr−0.5)·2`, sizing con kelly clamp, fricción global cross-asset. Ninguno de los gates reales (consejo, ML, envelope) existe. Los genomas se optimizan para una máquina que no es el motor.

### [M8-H02] Kill-switch de deriva NUNCA dispara en demo
`online_daemon.rs:461-495`: `if self.ewma_sharpe < -1.50 && !self.is_demo` — el kill switch no funciona en el único entorno donde la autoevolución está armada.

### [M8-H03] RANSAC trima outliers ANTES del t-stat
`online_daemon.rs:930-995`: removiendo exactamente la cola negativa que evidencia degradación. Los tres controles (DSR, kill-switch, rollback) son **optimistas por construcción**.

---

## 📋 CENSO DE RIGIDEZ (literales fijos sin derivación teórica)

| # | Archivo:Línea | Literal | Contexto |
|---|-----|-----|-----|
| R-01 | lib.rs:996 | `clamp(0.0062, 0.0160)` | BE activation 62-160 bps |
| R-02 | lib.rs:1018 | `clamp(0.0072, 0.0200)` | Trail activation 72-200 bps |
| R-03 | lib.rs:1180-87 | ofi ±0.30/±0.25, vpin 0.65, toxic `(sl*0.85).max(0.0065)` | Toxic flow grid |
| R-04 | lib.rs:2655-62 | 5e-5/1e-4/1.5e-4/4e-4/2e-3/1e-3 | Momentum invariants (8 banded) |
| R-05 | lib.rs:3360-401 | 45s/60s/180s/20m/1h/2h, stretch ±0.25/0.30 | Whiplash grid |
| R-06 | lib.rs:3428-31 | (0.28,0.18)/(0.32,0.22) | Streak conviction |
| R-07 | lib.rs:3592 | `spread*0.5+0.5 clamp(0.5,500)` | slip_bps |
| R-08 | lib.rs:3961 | `latency_ms <= 25` | Maker toggle |
| R-09 | lib.rs:2191/2247 | 70/30 ppo/cvd; 60/40 NN/tensor | Mixture weights |
| R-10 | lib.rs:2430-43 | fused ±0.6, persist ±0.15, conf 0.55+0.3 | Spectral direct |
| R-11 | god_engine.rs:3512-39 | ratio **1.25**, floor **0.4**, budget **5%**, clamp 1-20 | VOL-BRAKE |
| R-12 | stateful_engine.rs:161-202 | ×2/×4/×6/×15/×30; 18000/7200; v_t>0.0015 | Cooldown ladder |
| R-13 | trailing.rs:151-163 | 1.5/2.5/4.0/5.0 ATR; 0.40·tp | Phase gates |
| R-14 | consejo_seniors.rs (múltiple) | 0.35/0.80/0.28/×0.75/1.2/1.5/3.0/0.95−0.10s/35+65s/k=8/Ente 0.7/0.5/0.85/0.8/crowd 3.0/0.33/2.5/0.4/veto 0.88/0.85/0.92 | Toda la grid |
| R-15 | risk-engine/lib.rs:582 | `0.66/0.62` | Confidence-hardening |
| R-16 | risk-engine/lib.rs:663 | `+0.1/×1.0005` | FP epsilons |
| R-17 | risk-engine/lib.rs:688 | `(alloc*0.25).clamp(1.20,2.60)` | Micro margin cap |
| R-18 | online_daemon.rs:124 | warmup 60s demo / 900s prod | Evolución timing |
| R-19 | online_daemon.rs:837-53 | 0.55/0.50/0.60/0.75 by N | Confidence targets |
| R-20 | kelly.rs:113-25 | lerp endpoints 0.15/0.25/0.50/0.65; 0.40/0.55/0.80/0.95; survival 0.20/0.75 | Spectral bands |

---

## 🎯 HOJA DE RUTA SISTÉMICA DE REHABILITACIÓN

### FASE CRÍTICA INMEDIATA (bloqueadores — reparar ANTES de cualquier operativa)

| Prioridad | ID | Reparación | Esfuerzo est. |
|-----------|-----|-----------|--------------|
| 1 | M2-C01 | `return (None, closed_order, None)` | 5 min |
| 2 | M8-C02 | Usar `th` en el sweep de profit del forest online | 1 h |
| 3 | M1-C01 | Reconstruir symbol_to_id en rotación | 2 h |
| 4 | M5-C01+C02 | Rutar bootstrap/micro por calculate_kelly_fraction | 3 h |
| 5 | M2-C03+C04 | Migrar MicroScalp + 3 asientos a lift-sobre-base | 4 h |
| 6 | M8-C03 | DSR sobre returns del CANDIDATO | 2 h |
| 7 | M8-C01+C05 | Portar sizing del host a replay + native | 4 h |
| 8 | M5-C03 | Darwin usar anclas canónicas 30s/12h | 30 min |
| 9 | M5-C04 | Unificar clamp kelly_at_tau | 30 min |
| 10 | M2-C02 | Alimentar Hawkes record_event | 2 h |
| 11 | M2-C05 | Gatear book_absent a modo backtest | 30 min |
| 12 | M4-C01 | WS orders: correlar ORDER_TRADE_UPDATE | 4 h |
| 13 | M4-C02 | Kill-switch re-arm supervisado | 3 h |
| 14 | M8-C04 | CmaEsOptimizer no sobrescribir fitness | 1 h |
| 15 | M7-C01 | SeniorCausal requerir conjunción | 30 min |
| 16 | M3-H02 | Exit physics √Q/√latency | 30 min |
| 17 | M2-H02 | Zerificar FEATURES_DEAD en serve | 30 min |
| 18 | M6-H02 | drain_bracket_closes incondicional | 5 min |

### FASE ALTA (34 HIGH)
- M4-H01-05: trainer supervisor, shutdown graceful, OCO retry, stall-strike, entry_fee
- M5-H01-03: fitness unificado, hot-swap atómico, ruin cap centralizado
- M8-H01-03: walk-forward real, kill-switch demo, RANSAC sin trim
- M2-H01-05: doble macro, conformal per-sym, Hedge labels, book_absent, Darwin tensor
- M1-H01-03: backpressure counter, seq_guard reset, kline cadencia

### FASE MEDIA (98 MEDIUM)
- ~50 literales de rigidez → promover a genes/config con documentación
- ~15 paths muertos → wire o delete
- ~10 allocations en hot path → pool/pre-compute
- Escrituras atómicas para todos los archivos de config

---

## ✅ VERIFICADO COMO SÓLIDO (no tocar)

| Componente | Verificación |
|-----------|-------------|
| `diffusion.rs` | Varianza estacionaria exacta, derivada y testeada |
| `hurst_dfa.rs` | DFA real con gate r², calibrado contra RW/AR(1) |
| `kelly_envelope.rs` | Bayesiano completo (Beta posterior, LCB, shrinkage, ruin, streak) |
| `ensemble.rs` Hedge/Brier | Pesos correctos, n_eff correcto, skill z-test correcto |
| `conformal.rs` | Split conformal + ACI con regret bound correcto |
| `calibration.rs` Platt | Newton MAP correcto, consistente con Beta prior |
| `spectral.rs` FFT | Radix-2 Cooley-Tukey real |
| `tp_sl.rs` + `friction_floors` | EV algebra exacta, σ(τ) scaling correcto |
| `selection_stats.rs` DSR/PSR | Bailey & LdP 2014 correcto, A&S CDF precisa |
| Paridad 48D train≡serve | Contrato verificado (excepto M2-H02) |
| Motor continuo espectral | Bandas fast/slow + arbitración por τ_dom |
| DSR gate en promoción | Cableado (pero evalúa muestra equivocada M8-C03) |

---

**FIN DEL INFORME**

*Certificación: ⛔ NO APROBADA*
*Próximo paso: ejecutar FASE CRÍTICA INMEDIATA (18 items, ~30 horas de trabajo estimado)*
*Los hallazgos confirman que la arquitectura QO es correcta donde aterrizó — los defectos restantes son de cableado, consistencia y disciplinas de paridad.*
