# 🔬 INFORME DE ASEGURAMIENTO Y CERTIFICACIÓN PROFUNDA — POST FASES 0–3

**Fecha:** 2026-09-07 · **Alcance:** 23 crates + `src/`, 286 archivos Rust, auditoría multi-agente de raíz a cima.
**Naturaleza:** AUDITORÍA DOCUMENTAL EXCLUSIVA (sin cambios de código). Este informe **se agrega** a la serie forense existente (`INFORME_FORENSE_MAESTRO.md`, `FORENSIC_INTELLIGENCE_AUDIT.md`, `WALKTHROUGH_FORENSE_SISTEMICO.md`).
**Commits certificados:** `ddae98f4` (f0-f2: desbloqueo sistémico) y `286a6a7a` (f3: des-hardcodeo evolutivo).

---

## 🗺️ 0. Paradigma de Grafo Vivo y Topología del Sistema

El sistema se modela como grafo dirigido de datos→decisión→ejecución. Los nodos se clasifican:

- **Nodo Raíz** (fuente de verdad del mercado): `data-pipeline` (WS parsers, klines, macro), `data-ingest`, bins históricos.
- **Nodos de Transformación**: `feature-engine`, `dark-alpha-engine` (tensor 54D), `signal-engine` (orquestador de votos), `metacortex-engine`.
- **Nodo de Decisión**: `god-engine-core::process_tick_dual` / `process_event`, `risk-engine`, genoma activo.
- **Nodos Terminales** (que tocan dinero real): `execution-engine` (REST/WS), `god_engine.rs` (lazo de producción).
- **Nodos de Memoria/Evolución**: `quantum-arena` (genoma/genome_store), `evolution-engine`, `storage-engine`.

**Diagnóstico topológico central de esta auditoría:** el grafo está PARTIDO. Existen múltiples inteligencias calculadas en Nodos de Transformación que nunca alcanzan el Nodo de Decisión (bloqueos de inteligencia), el Nodo de Decisión tiene DOS sistemas de riesgo de los cuales solo uno está cableado (y el cableado descarta su output), y los Nodos de Memoria tienen tres promotores de genoma con semánticas contradictorias y sin exclusión mutua. La divergencia backtest/producción reportada por el operador es una consecuencia topológica, no un accidente.

---

## 🚦 1. Resumen de Estado de Resolución

| Estado | Puntos | Nota |
|---|---|---|
| ✅ Resueltos y verificados | Kelly congelado (PF≤1→0.0 eterno), techo micro `×2.30`, kill-switch desarmado con cap≤0, truncado silencioso a 3 días, Sharpe ×724, roundtrip_fee 0.0004, is_win bruto/neto, genoma 139D roto, módulos perdidos ntp/ws_executor (compilan) | Confirmado por suite en verde |
| ⚠️ Resueltos con defectos introducidos o incompletos | Gate de `promote()` (incompatible con `from_vector` → evolución CMA-ES bloqueada), split bayesiano (satura; inestable por tick), cutoff_floor (anula el gen `ml_threshold_*`), `expected_volatility` (semántica rota tras la recalibración), router genómico (fallback re-introduce literales vía hot-swap) | Detalle en §3–§5 |
| 🔴 Pendientes críticos estructurales | Lookahead en TODOS los generadores de ticks, ejecución sin `reduceOnly`, OCO sin motor de cancelación, riesgo de producción que descarta el `ValidatedOrder`, consejo de seniors jamás invocado, pipeline de telemetría mmap fantasma, `champion_path` nunca escrito, contaminación cruzada backtest↔producción del almacén de genomas | Detalle en §2 |

**Conteo total de esta ronda:** ~180 puntos de fallo documentados (17 CRÍTICOS, ~45 ALTOS, ~60 MEDIOS, ~58 BAJOS), que se agregan a los 305+ históricos del informe maestro.

---

## 📊 2. Matriz Maestra Consolidada — Hallazgos CRÍTICOS de esta ronda

| ID | Módulo | Hallazgo | Archivo:línea | Categoría |
|---|---|---|---|---|
| **K-01** | 5/Evolución | **El gate de `promote()` rechaza a TODA la evolución CMA-ES**: los clamps de `from_vector` son disjuntos de los bounds de validación (`scalp_sl_base` clamp-min 0.0025 > upper bound 0.002; `scalp_tp_base` 0.006 > 0.005; `swing_tp_base` 0.15 > 0.10). Ningún genoma reconstruido por vector puede ser promovido — el pipeline versionado quedó funcionalmente muerto tras f3 | `genome.rs:1645-1648` vs `genome.rs:1776-1790` vs `genome_store.rs:95-107` | Incoherencia de invariantes |
| **K-02** | 3/8 | **Lookahead estructural en los 4 generadores de ticks**: la ordenación intra-vela y el OBI se construyen con `is_bullish` (close futuro). El backtest "funciona" porque el desbalance de volumen codifica la respuesta | `backtest-engine/src/lib.rs:212-231`; `parquet_to_bin.rs:71-135`; `multi_coin_simulator.rs:61-66`; features omni en `backtest-engine/src/lib.rs:116-183` | Lookahead bias |
| **K-03** | 4 | **`execute_reduce_only_market` NO envía `reduceOnly=true`**: su contrato documentado es "garanteed to only reduce" pero el payload nunca lo incluye; divergencia local/exchange → -2022 → posición desnuda sin retry | `executor.rs:1500-1560` | reduceOnly / fallo silencioso |
| **K-04** | 4 | **OCO sin motor de cancelación y sin reduceOnly**: el doc promete "el motor interno cancela la otra pierna"; no existe tal motor ni suscripción ORDER_TRADE_UPDATE que cancele. Tras TP, el STOP queda vivo | `executor.rs:1672-1730` | OCO / reconciliación |
| **K-05** | 5 | **El camino de riesgo de producción descarta el `ValidatedOrder` completo**: solo usa `order.signal`; volumen, leverage, TP/SL, maker_only y todo el trabajo de `evaluate_single_intent` se tiran; el sizing se recalcula con literales (`0.70`, `clamp(0.25,0.70)`, `5.05/lev`). El `evaluate_order` bayesiano no tiene ningún caller en producción | `god-engine-core/src/lib.rs:1327-1394` | Coherencia de riesgo |
| **K-06** | 4 | **Hot-swap del executor en la transición mainnet pierde arena, registro y NTP**: nuevo executor con `arena: empty()` y registry fresco; el UserDataStreamer mantiene el registry VIEJO → acks REST y fills WS divergen; el router decae al fallback hardcodeado | `god_engine.rs:1262-1281` | Ciclo de vida / WS-vs-REST |
| **K-07** | 2/7 | **`ConsejoDeliberacion` (10 Seniors, 4 vetos) jamás invocado**: construido en el engine, `grep deliberar` = 0 callers fuera de tests. La gobernanza adversarial completa está muerta | `god-engine-core/src/lib.rs:44,131` | Bloqueo de inteligencia |
| **K-08** | 2/7 | **Filtro conformal degenerado**: `conformal_p_value` es la constante 0.95 escrita cada tick; no existe set de calibración conformal en el repo; umbral 1−α=0.90 < 0.95 siempre → tautología, 100% de señales pasan | `god-engine-core/src/lib.rs:933-934` | Filtro decorativo |
| **K-09** | 5/Evolución | **Divergencia raíz backtest↔producción del genoma** (8 sub-causas, §6): telemetría mmap jamás escrita, gate t-stat que requiere cientos de trades, `champion_path` nunca escrito (muta `default()`), `refresh_models` que pisa hot-swaps cada 1000 ticks, almacén compartido sin separación de entornos | `online_daemon.rs` múltiples | Ver §6 |
| **K-10** | 5 | **`darwin.rs` aplica el genoma ANTES del gate** (el bug exacto corregido en online_daemon f3 persiste en el otro promotor): promote rechazado → arena de producción operando con genoma inválido | `darwin.rs:415-443` | Ordenamiento atómico |
| **K-11** | 6 | **Torn reads en el bus mmap de telemetría**: el head se incrementa antes de escribir el payload; sin seqlock, frames entrelazados de dos eventos | `storage-engine/src/mmap_bus.rs:91-139` | Race |
| **K-12** | 6 | **Cola de telemetría global ilimitada**: `init_*` calcula capacidad y jamás la usa (`SegQueue` sin cota); ante spam de errores → OOM | `telemetry-engine/src/atomic_telemetry.rs:35-39,57-61` | Unbounded queue |
| **K-13** | 6 | **El auditor forense muere permanentemente ante un burst** (`while let Ok` trata `Lagged` como terminal) justo cuando más eventos hay; mismo patrón en dashboard WS | `telemetry-server/src/forensic_auditor.rs:82` | Silent failure |
| **K-14** | 6 | **Datos macro SINTÉTICOS silenciosos**: Yahoo v7 muerto → random-walk de VIX/SP500/DXY escrito como historia real sin marcador; y el indicador del World Bank para `wb_us_m2_supply` es "Broad money % of GDP" (≈90) en vez de M2 $B (≈21000) — salto de 2 órdenes de magnitud | `macro_history_sync.rs:56-78`; `omni_multiplexer.rs:549` | Garbage-in |
| **K-15** | 5 | **Fallback de leverage que viola el axioma del envelope Kelly**: con evidencia insuficiente, micro-cuentas fuerzan `5.05/cap` (hasta 5x) para alcanzar minNotional — exactamente el "camino suicida" que el propio docstring de F5.1 prohíbe | `god_engine.rs:1457-1463`; `kelly_envelope.rs:200-206` | Riesgo de ruina |
| **K-16** | 4 | **`flatten_all_positions` asume hedge-mode si la lectura del modo falla**: en el primitivo de seguridad del kill-switch, un fallo de parse envía positionSide a una cuenta one-way → todas las órdenes de cierre fallan (-4061) en el escenario catastrófico | `executor.rs:368-370` | Fail-unsafe |
| **K-17** | 3 | **Migración "Continuous" ~20% completa**: codificación de horizonte autocontradictoria (Continuous colisiona con Swing; `horizon()` jamás lo devuelve), posición unificada abierta con TP=SL=0.0 espejada en `scalp_position` (doble contabilidad), pipeline híbrido continuo+dual conviviendo en el mismo tick con el cierre continuo reportado como "scalp" | `position.rs:15,130-135,154-159`; `god-engine-core/src/lib.rs:1181-1207` | Migración incompleta |

---

## 🔬 3. Módulo 1 y 6 — Ingestión, Parsers, L2, Normalización, Estado Atómico, Mmap, Telemetría, OS

### CRÍTICOS (además de K-11..K-14)
- **[C] Lookahead en `parquet_to_bin`** (`src/bin/parquet_to_bin.rs:47-121`): el OBI de los sub-ticks t+15/35/55s se deriva de `is_bullish = close >= open` — el cierre de la vela que aún no ocurrió. Todo modelo entrenado sobre `{symbol}_ticks.bin` ve el futuro; en producción el OBI real no tiene esa correlación → degradación sistemática del edge. La trayectoria open→high→low→close es además un path fijo supuesto. **Este hallazgo contamina la totalidad del pipeline evolutivo que consume `_ticks.bin`.**
- **[C] Macro sintético silencioso** (`macro_history_sync.rs:56-78`): 1000 filas de random-walk como "historia VIX/SP500/DXY", fechas falsas cíclicas 2022-01, sin marcador de sinteticidad; `period2` congelado en jul-2025.

### ALTOS
- **[A] `LakehouseMmap::append_tensor` corrompe el offset** (`data-pipeline/src/lakehouse_mmap.rs:72-75`): `fetch_add` antes del chequeo de capacidad → cada intento fallido avanza el offset; buffer inutilizable permanentemente; sin commit marker (lectores ven frames a medio escribir).
- **[A] Precisión f32 para precios persistidos** (`data-pipeline/src/storage.rs:9-19`): ~7 dígitos — a BTC≈100k el tick 0.1 no es representable; exóticos pierden dígitos. Forense/backtest cuantizados no uniformemente.
- **[A] `LockFreeBus::try_pop`/`push` sin seqlock** (`telemetry-server/src/lockfree_bus.rs:32-73`): tail avanza antes de leer el slot; productor puede sobrescribir mientras el consumidor copia. MPMC declarada sin la serialización que la hace válida.
- **[A] Persistencia fire-and-forget con ventana remove→rename** (`omniscient-registry/src/lib.rs:199-222`; idéntico patrón en `os-guardian/src/crash_dump.rs:60-63` — irónico en el dump de emergencia — y `data-pipeline/src/market_context.rs:99-102`): crash entre remove y rename = sistema SIN snapshot; hilo huérfano pierde la escritura silenciosamente tras un `Ok(())` ya devuelto.
- **[A] Fees con fallback silencioso** (`data-pipeline/src/api_client.rs:164-175`): fallo de parse → asume 0.02%/0.04% sin distinguir "dato ausente" de "dato real"; VIP/BNB-distinto → PnL live diverge del backtest sin alarma.
- **[A] `wb_us_m2_supply` con el indicador equivocado** (`omni_multiplexer.rs:549`): `FM.LBL.BMNY.GD.ZS` es "Broad money % of GDP", no M2 — ver K-14.
- **[A] Flight recorder sin msync y head volátil** (`telemetry-server/src/flight_recorder.rs:59-81`): tras reinicio, `head=0` sobrescribe los primeros N eventos del registro previo (justo los del crash); sin msync, el post-mortem depende del writeback de Windows.
- **[A] Parser L1 por "parsea o no" sin discriminar stream/evento** (`data-pipeline/src/parser.rs:18-21`): `BookTickerEvent::parse_from_json` acepta cualquier payload que contenga `"b":`... — contaminación del libro ante cualquier cambio de schema de Binance; `is_buyer_maker` interpreta no-booleanos como false.
- **[A] SQLite síncrono dentro de tasks async y `NO_MUTEX`** (`forensic_auditor.rs:103-152`; `state_db.rs:18-23`, `persistence.rs:10-15`): bloquea workers tokio (jitter al runtime completo); `Connection` compartida vía `&self` con NO_MUTEX es UB de SQLite.
- **[A] Libro L2 sintético en `fetch_agg_trades`** (`data-pipeline/src/historical.rs:253-269`): spread 1.5 bps inventado + depth base — el OBI del backtest es sistemáticamente más "limpio" que el live; divergencia no medida.

### MEDIOS (resumen detallado)
- **[M] Stasis bayesiana del WS descarta 2 ticks legítimos tras saltos reales** (`ws_client.rs:298-310`) — el libro queda congelado en el precio viejo mientras el mercado se movió.
- **[M] Latencia condicional a offset de reloj** (`ws_client.rs:327-348`): con drift, `last_ws_latency_ms` queda saturada al timeout → el interlock de pánico dispara por reloj, no por red.
- **[M] Rutas relativas `data/...` en todo el módulo** (9 sitios): lanzamiento desde otro CWD crea una jerarquía vacía silenciosa.
- **[M] Formatos binarios sin header/magic/versión** (mmap_bus, telemetry_mmap, flight record, BinTick) + `transmute` de `TelemetrySnapshot`: cualquier cambio de layout lee basura sin detectar.
- **[M] "Ghost Flusher" que no persiste nada** (`zero_copy_bus.rs:141-169`): el comentario lo admite; `is_active` nunca cae.
- **[M] `append_tick` sin persistir write_cursor; checksum XOR decorativo** (`storage.rs`).
- **[M] Swallows sistemáticos en persistence**: `insert_tick` convierte NaN→0.0 y lo INSERTA (contamina `tick_data`); `try_upsert_kline` devuelve Ok ante vela inválida; `PRAGMA page_size` post-WAL = no-op.
- **[M] `TokenBucket` con contrato de unidad ambiguo (tokens/ms vs /s = 1000x)** y pérdida de crédito fraccional (`data-ingest/src/lib.rs:21-48,68-73`).
- **[M] `latency_ns = cycles/3` asume 3GHz exactos** (`profiler.rs:104-108`) + `_rdtsc` sin lfence.
- **[M] `onchain_feed` acopla SHADOW_MODE→testnet y asume coin 0 = BTC** (`onchain_feed.rs:97-109,149-161`) con universo dinámico.
- **[M] `download_history` con header parseado como dato (funciona por accidente) y filas corruptas saltadas sin contador** (`download_history.rs:120-155`).
- **[M] Fallback del seqlock del flight-recorder sin flag de corrupción** (`flight-recorder/src/lib.rs:117-138`).
- **[M] Detector de gaps con semántica de secuencia estricta + jitter determinista `seed%1000`** (`resilient_stream.rs:29-47,78`) — el anti-thundering-herd no mitiga nada.

### BAJOS (resumen)
Telemetría `tensor_drift: 0.0` muerta y `truncate(true)` destructivo (`telemetry_mmap.rs:142`); win_rate por defecto 0.55 fabricado para el dashboard y `effective_max_leverage` esperado por el JS que el server no expone; `min_working_set` con división entera y `INITIALIZED` prematuro en os-guardian; `.unwrap()` sobre RwLock/Mutex envenenables (dns_optimizer, os-guardian telemetry); `fast_parse_f64` sin notación exponencial (`1e-8`→1.0); clamp `[0,100]` que borra tasas negativas reales (el propio test documenta −0.50 como válido — contradicción interna); `OnlineNormalizer::default` con `f64::MIN`; canal bounded con `try_send` y contador de dropeo que nadie lee; `Parameter.timestamp` jamás actualizado; `force_push` del logger que descarta los mensajes MÁS VIEJOS durante un incidente; `with_extension("tmp")` que colisiona con puntos en el nombre; `macro_data.rs` abortando las 3 series por un solo error `?`.

**Patrón transversal del módulo:** (1) `let _ =`/`.ok()` como política de errores en TODA la persistencia; (2) "estado publicado antes que el dato" — la misma clase de bug de protocolo aparece en mmap_bus (K-11), lakehouse_mmap y lockfree_bus; (3) remove+rename como sustituto de rename atómico.

---

## 🧠 4. Módulo 2 y 7 — Inferencia IA, Señales, Cuántica, Orquestación, Confluencia

### CRÍTICOS (además de K-07, K-08)
- **[C] Multicolinealidad estructural del "ensamble cuántico"**: `quantum_oscillator`, `stochastic_resonance`, `soliton_wave`, `coaxial_breakout`, `hawkes_bessel`, `vecm_zscore`, `cointegration_zscore` son TODOS re-derivaciones del mismo `obi_val` (`god-engine-core/src/lib.rs:883-899` escritura; fallbacks en cada módulo de signal-engine). El `TensorVoteOrchestrator` pondera 14 "estrategias" que son una sola variable contada ~7 veces; `ensemble_boost = 1+(n−1)·0.1` multiplica confianza por pseudo-diversidad. Correlación de fallos en cascada; confianza inflada donde OBI no predice. **No fue corregido por la recalibración FASE 2.**
- **[C] Incongruencia dimensional 54D vs 34D + 8 features constantes + normalizadores mágicos**: coexisten tres geometrías (`default_model()=34D`, fallback 54D, ruta 12D); dentro del tensor 54D, índices 40/46/48/49/50/51-53 son constantes (15% del input es ruido estructural); constantes fallback 1.04/1.02/1.00/0.75/1.05 y divisores /5000,/18000,/20,/4,/80,/1000 sin derivación (`dark-alpha-engine/src/lib.rs:475-478`; `god-engine-core/src/lib.rs:99-104,1085-1100,265-312`).
- **[C] Stale-features en la NN Swing**: solo se refresca cuando `coin.ml_prob` (prob del NanoForest de SCALP) está en extremos o ==0.5; el resto del tiempo la señal swing hereda la probabilidad de otro modelo de otro dominio (`god-engine-core/src/lib.rs:1081-1104`).
- **[C] RegRESIÓN de FASE 2 — `expected_volatility` ya no es volatilidad**: tras quitar `max_volatility` de la convicción, el campo quedó como máximo |peso| de señal (∈[0,1] adimensional), pero el router lo compara contra `scalp_sl_base*0.5` (~0.05% de precio) → el gate de volatilidad es SIEMPRE verdadero; la defensa anti-slippage por volatilidad no existe y el IOC dispara con cualquier convicción > min_confidence_btc en mainnet (`orchestrator.rs:103` vs `router.rs:84-92`).
- **[C] RegRESIÓN de FASE 2 — el gen `ml_threshold_long/short` queda anulado por `cutoff_floor`**: con baselines del genoma, `cutoff_floor ≈ 0.12` y `long_dist*2 ≈ 0.14` → el cutoff efectivo es siempre el floor para CUALQUIER `ml_threshold` en [0.44, 0.975]; la evolución optimiza un gen que el orquestador neutraliza; y `min_confidence_btc > 0.95` saturaría el floor a 0.90 bloqueando casi toda señal (`orchestrator.rs:156-171`).

### ALTOS
- **[A] `evaluate_consensus` con peso swing `1.20` hardcoded y convicciones no normalizadas por horizonte** (`orchestrator.rs:220-231`).
- **[A] Umbral tensor-swing 0.60 fijo en el consumidor final** que duplica y contradice los cutoffs del orquestador (`god-engine-core/src/lib.rs:1287-1290`); confianza base `0.70+|score|*0.30` fabricada (`lib.rs:1001,1008`).
- **[A] 20+ umbrales internos hardcoded en `evaluate()` de las estrategias** (turbo_scalper `hawkes>=1.2 && |obi|>=0.2` pesos 0.6/0.4; micro_scalp_trigger; swing_conformal `|z|>=1.5`; trend_runner `hurst<=0.52`): gatean el voto real y ningún gen puede moverlos.
- **[A] Las APIs parametrizadas/genómicas de los filtros NO tienen ningún caller de producción** — solo tests. La versión calibrada está muerta; vive la hardcoded.
- **[A] Pseudo-ciencia sin derivación**: el "pozo anarmónico" es espejo lineal del OBI (λ irrelevante); la "resonancia estocástica" es ganancia ~(0.5,1) siempre; el "solitón" se reduce a `sign(vel)*|amp|` (x=t=0 fijos); el "Rankine-Hugoniot" es un tanh sin γ; el "Hamiltoniano cuántico" es random search cuya "energía" no depende del error salvo por término constante — la selección no puede aprender (`quantum_oscillator.rs`, `stochastic_resonance.rs:33-38`, `soliton_wave.rs:88-93`, `supersonic_shockwave.rs:47-53`, `quantum_evolver.rs:41-76`).
- **[A] Normalización Welford mutada durante `predict()`** (`dark-alpha-engine/src/lib.rs:536-540,416-427`): cada inferencia desplaza media/varianza → dos predicciones del mismo vector difieren; drift no estacionario del input; backtest no determinista por orden de eventos. Además el decay post-2000 mezcla unidades m2/varianza.
- **[A] `fit()` entrena con ReLU pero `predict()` infiere con tanh** (`lib.rs:653-655` vs `:559-562`): el modelo entrenado no es el modelo inferido.

### MEDIOS
- **[M] Dead code masivo en feature-engine**: `quantum_tensor_store`, `tensor_ring`, `simd_neural_network`, Hawkes real, Kalman — cero consumidores. La "tensor store" del nombre no alimenta nada.
- **[M] Features calculadas y descartadas en el hot-path**: espectro multifractal completo tirado; `SpectralCycleEngine` acumulando sin lectores; `lead_lag_engine` actualizado sin consumir su alpha; `ppo_engine` y `online_learner` construidos y jamás usados.
- **[M] `route_order(&TensorDecision,...)` huérfano**: el "Ruteo Cuántico" no recibe TensorDecision real de producción (solo tests sintéticos).
- **[M] Metacortex: 9 de 11 módulos sin consumidores** (immune_system, hot_swap_controller, epigenoma_store, cazador_constantes, reminiscence_and_adn, shadow_graph_auditor, evolutionary_templates, fases_autonomous, compiler_sandbox).
- **[M] Defectos de lógica del consejo aunque se cableara**: aprobación fail-open con capacidad direccional 0; `dissenting_log` distorsionado cuando final_signal==0; inversión de señal por WR<0.5 con el mayor peso tras pocas muestras (indefendible estadísticamente); `effective_wr = 1−wr` trata 20% WR como 80% de confianza opuesta.
- **[M] Scaler sin fallback correcto** (std≤1e-8 deja features crudos → saturación de tanh) y layer-norm que destruye la normalización per-canal.
- **[M] ATRs multi-escala sintéticos** (`atr_5s = atr*2.236`, `atr_1m = atr*7.746`): el CoaxialBreakout que los consume produce `squeeze ≈ 0.995` constante — degenerado siempre-pasa. `hawkes_intensity = 1+|obi|*2` — "Hawkes" sin proceso de Hawkes.
- **[M] Simetría long/short**: `Short` reporta `abs()` del net_confidence — semántica inconsistente para consumidores por signo.

### BAJOS
División redundante por `len().max(1)`; test con `hawkes_ratio=-3.0` contradiciendo el dominio no-negativo declarado; `vecm_beta_hedge` reutilizado como multiplicador de Z-threshold sin documentar; test que ESCRIBE en `models/DarkAlpha_BTCUSDT.json` (muta artefactos de producción desde la suite); ruta relativa del servidor 4D; offset Nash `spread*0.25` sin derivación; `RenyiTsallisEntropyEngine` con defaults fijos (la versión parametrizada no se usa); perceptron con peso Hebbiano jamás actualizado (vive en 1.0 con clamp al leer).

---

## 📈 5. Módulo 3 y 8 — Estrategia, Régimen, Horizontes, Backtesting, Auditoría Interna, Gobernanza

### CRÍTICOS (además de K-02, K-17)
- **[C] Selección evolutiva del backtest continuo in-sample con bonus de actividad distorsionante** (`continuous_evolution_backtest.rs:445-509`): `activity_bonus` hasta 0.06 USD = 0.46% del capital de $13 — derrota mutantes planos; umbral mágico +0.30; rama "AUTOEVOLUCIÓN FORZADA" adopta cualquier mutante PnL>0 con 1 día de muestra; el genema elegido con los datos del día D se promueve a producción (líneas 526-533).
- **[C] Ver detalle en K-02**: los cuatro generadores (run_backtest_native, parquet_to_bin, multi_coin_simulator, omni features) fabrican la microestructura con el close futuro.

### ALTOS
- **[A] Divergencia de modelos de fee/min-notional/leverage entre los 4 backtests y live**: `avg_fee_est 0.0006` vs genoma `max_fee_pct` vs specs 0.0002/0.0005 vs maker-en-TP live; min_notional $1 (MCS) vs $5 (CEB/live); el TP live asume fill maker que ningún backtest replica.
- **[A] `run_vectorized_hybrid` con fills exactos sin slippage de stop, wins contados bruto en TP y neto en reversión, margen sobre-extendido, y `capital.max(0.0)` que absorbe quiebras** (`vectorized.rs:148-216,236-246`).
- **[A] `NetworkJitterSimulator` es código muerto**: la latencia no se simula en ningún backtest; en live gatea market-making y dispara el interlock de pánico — divergencia estructural backtest/live.
- **[A] Tensor 54D con semántica DISTINTA por binario**: BE pone precios cross-exchange en 0-3; CEB pone mid/volumen/spread; MCS pone bid/ask — un modelo entrenado contra una disposición se evalúa contra otra.
- **[A] Enumeración de la migración Continuous no propagada** (9 grupos): `horizon()` nunca devuelve Continuous; el match Continuous del orquestador es rama muerta (ningún caller lo pasa); `QuantumStrategy::horizon()` solo Scalp/Swing; state_db `HorizonIntent` binario (string desconocido→Swing); consejo_seniors sin opinión Continuous; online_learning entrena el modelo equivocado (los trades continuos se reportan como scalp); trajectory_auditor binario; TODAS las estrategias declaran Scalp o Swing; `process_event` devuelve el 4-tuple dual.
- **[A] `forensics.rs` completo es `#[cfg(test)]`**: las "auditorías forenses" anti-lookahead solo existen bajo `cargo test`; además su detector correlaciona una media móvil incremental contra 5 precios hardcodeados y escribe `.forensic_violation` desde un test.
- **[A] `PhaseExecutor::run` decorativo**: las fases Gamma/Delta leen RAM/CPU y devuelven strings; invocado y descartado en el hot-loop del simulador.
- **[A] Auditores reales nunca cableados**: `AuditorInterno` (Tribunal Diario), `TrajectoryAuditor`, `BehavioralAuditorEngine`, `CyberneticResilienceShield` — 0 referencias; `DriftAuditor` instanciado como `let _drift_auditor` y jamás usado, con implementación duplicada divergente en `src/simulation/`.
- **[A] Doble `increment_tick` en CEB**: el champion recibe tick manual + interno (2x) vs shadows (1x) → refresco de modelos y gates por tick sesgados contra el baseline que se compara en la élite.

### MEDIOS
- **[M] Números mágicos por estrategia** (censo): scalp `15_000`ms mínimo y VPIN 0.85; swing `z clamp(1.2,3.0)`, `macd<=0.0005`, `conf=macd*hurst*50`; turbo z-proxy tratado como z-score; trend_runner `0.02/0.08/(tp*10).tanh()`; coaxial `sqrt(5)/sqrt(60)/*4.0`; los "10 nichos ecológicos" de CEB con rangos congelados a mano.
- **[M] Bug de lógica en `SwingEngine::evaluate_trend`**: Short exige `price > fast_val` pero Long exige `price >= slow_val` (asimetría sin justificación); Bollinger con varianza de la media equivocada; boost de confianza con el threshold del lado opuesto como pivot.
- **[M] Voto por peso no por convicción** en el orquestador: una estrategia que escupe ±1.0 constante domina el "consenso bayesiano".
- **[M] Registro de features sintéticas inyectadas como reales** (hawkes=OBI reempaquetado, vecm==coint==OBI duplicado en dos claves) — diversidad ilusoria.
- **[M] Métricas de reporte**: sharpe por trade sin anualizar; `out_stats[6] = final_cap − 13.0` con capital fijo hardcodeado en FFI.
- **[M] CEB: condición de quiebra intra-día con capital del día anterior; bases de fitness y capital operativo distintas (con/sin flotante)**.
- **[M] CEB: nichos no re-seedan; campeón puede provenir de promociones previas contaminadas**.
- **[M] `StateValidator::validate_parity` tolera y no actúa** (kill-switch comentado); contabilidades que cuadran por azar.
- **[M] `VerificadorResultados` con `latency_us:150` hardcodeado, anualización `sqrt(min(500))`, `alpha − fee_drag*0.5` sin justificación**.
- **[M] `audit-engine/main.rs` parsea solo lib.rs (nada de estrategias); canal de 100 que dropea resultados**.

### BAJOS
tick_replayer sin validar alineación y silenciando archivos faltantes; RNG `%1000` con sesgo leve; último micro-tick cayendo 2s antes del cierre real de la vela; funding cada 480 velas no alineado a 00/08/16 UTC; `maint_margin_rate` único para todos los tiers; step_size "ultra fine" que elimina el redondeo de lotes real (fills imposibles pasan); spread fijo independiente de volatilidad; estado Welford contaminado según orden de vetos; drift acumulado sin acción; fee de cierre flotante maker-only optimista; PnL flotante final con `closes.last()` para ambos horizontes y contando la vía continua solo por el espejo; `ParityAlert` dropeable; umbral RAM 13.6GB hardcodeado; `open()` legacy defaultea Scalping.

---

## ⚡ 6-bis. Módulo 4 y 5 — Ejecución HFT, Red, Riesgo, Kelly, Genomas

### CRÍTICOS (además de K-03, K-04, K-06, K-10, K-15, K-16)
- **[C] Ver K-05**: el sistema de riesgo completo (`evaluate_order` bayesiano con split, gates EV, TP/SL del genoma) no tiene callers; producción usa `evaluate_quantum_order_by_horizon`, descarta su output, y además esa función hace `kelly_fraction.clamp(0.05, 1.0)` — **obliga mínimo 5% de fracción Kelly aunque el motor diga 0**.

### ALTOS
- **[A] `ws_executor.rs` fue re-reemplazado por la sesión concurrente y ahora es una fachada muerta**: `sender` siempre `None`, `set_connected` sin callers, `WsOrderMessage { api_secret: String }` transporta el secreto en claro por un canal ilimitado, y NO existe construcción de firma HMAC — si se conectara, todas las órdenes fallarían con `-1022 Signature`. (La reconstrucción con HMAC de f0-f2 fue sobrescrita.) Hoy benigno (siempre cae a REST) pero es una trampa de activación.
- **[A] Cliente no-tipado trata 5xx como definitivo → retry del OCO puede duplicar piernas** (si Binance sí registró la orden, el retry crea doble SL/TP); la resolución ambigua solo está implementada en `execute_order`, no en las piernas.
- **[A] `apply_ack`/`apply_trade_update` permiten regresión FILLED→NEW/Unknown** (status sin guard monótono) → órdenes "reviven", falsos suspicious, cancels espurios.
- **[A] Doble conteo de comisiones REST(max)+WS(+=) para la misma orden.**
- **[A] `reconcile()` no compara cantidades** (fill parcial perdido pasa como sano) y `apply_to_registry` inserta registros sintéticos `adopted_{symbol}_{update_time}` no deduplicados — crecimiento no acotado; no existe reconciliación periódica (solo arranque) ni cancelación de expiradas en el exchange.
- **[A] Cierre local inmediato + close REST fire-and-forget**: fallo del close (rate limit, -2022, cooldown) deja el arena creyendo flat con posición real abierta y el OCO remoto vivo; PnL registrado de un cierre que no ocurrió.
- **[A] Restauración de posiciones con TP/SL hardcodeados ±2.5%/±1.5%** ignorando el genoma y el ATR; margen inventado `(qty*entry)/10` (asume 10x).
- **[A] `evaluate_single_intent` usa bases de SCALP para órdenes SWING** (stops ~10x más estrechos de lo debido, rompiendo la invariante RR que el gate valida) — amortizado hoy por K-05, estallará al reconectar.
- **[A] Rate-limit accounting 100% por headers de respuesta** (N requests en vuelo cruzan el límite antes del primer header); reset read-then-store no atómico; el `QuantumMultiplexer` anti-ban no tiene callers.
- **[A] `promote()` sin lock inter-proceso**: `remove+rename` con ventana sin `active.json` (un lector del hot-path hace bootstrap desde el espejo legacy → linaje corrupto); tres promotores pueden colisionar en generación — el histórico "inmutable" puede sobrescribirse y el rollback del watchdog revertir a la generación equivocada.
- **[A] `kill_switch_active` del drift-detector del daemon nunca se rearma** ("hasta reentrenar" mintiendo: no hay código que lo desarme) → freeze permanente con solo un println.

### MEDIOS (selección; el detalle completo queda en el artefacto)
`kelly.rs:59` clamp puede PANICAR con min>max vía genoma legacy sin gate; `strategy_base_fraction` es un gen muerto (jamás leído tras el check de finitud); rampa de exploración con `||` que mezcla criterios PF y WR (WR<0.05 con PF>1 pisa el Kelly legítimo); `avg_price` del WS usa el precio del ÚLTIMO fill en vez de `ap`; `on_positions` recibe solo el delta (trampa de contrato); maker-chase con fallback a market de qty completa ante error ambiguo; IOC redondeado al lado NO agresivo (anti-propósito en breakout); router con fallback que re-introduce los literales erradicados; drawdown hard-stop "neutered" (`scalp_valid=true` siempre + pico re-inflado ×1.05 por brecha — puede sangrar 20-30% en escalones); literales $13-15 sembrados en tres puntos del sizing; leverage_matrix con `kelly.max(0.01)` forzando fracción positiva con edge negativo y `veto_threshold_btc` con semántica prestada; doble conteo de margen por la posición espejo; daemon mutando desde disco desincronizado con retornos agregados globales (t-stat de una estrategia inexistente); `record_trade` con doble EMA y payoff sesgado; capital bifurcado arena/AtomicU64 sin puente de vuelta; `spec()` con panic! en la ruta de restauración; doble mapeo de coin_id potencialmente divergente; NTP sin filtro de outliers (offset contaminado ±1s por RTT de 2s); `hot_swap.execute_swap()` no-op que imprime "MAINNET"; rama WS del flatten contando cerradas sin registro.

### BAJOS
27 hallazgos tabulados en el artefacto: bloques `if true {}` muertos; `hot_swap_credentials` sin actualizar el WsExecutor embebido; races menores de reset; SymbolFilter::default ficticio; redondeos inconsistentes entre endpoints; `positionAmt` solo-string; `Retry-After` sin log; keepalive eterno; cancel sin clasificar 418; USE_TESTNET divergente; heap de símbolos con top_10 corto sin aviso; simulador con qty sin leverage (subestima exposición ~Nx — divergencia sim/real); posiciones adoptadas infiriendo side del id; `load_active()` con efectos de escritura en una "carga"; `kelly_clamp_min` potencialmente negativo desde baseline; `unwrap()` ante rollback de reloj; warmup con high/low sintéticos; spawn de `cargo run` bloqueando la task async; bootloader exigiendo exactamente 30 slots; ping del WS dependiente de la versión; `pending_qty` firmado sin positionSide; evidencia del split con escala n²; tolerancia 0.75 DD para ≤$50.

---

## 🧬 6. AUDITORÍA ESPECIAL: por qué el genoma impacta en backtest y no en producción/demo

Cadena causal completa (ordenada por palanca):

1. **El pipeline de telemetría→forest es un fantasma.** `online_daemon.rs:92-115` lee `data/telemetry.mmap` buscando frames (subsystem 12, frame 30) que NADIE escribe: `MmapTelemetryBus` no tiene callers fuera de tests, las constantes ni siquiera existen en el módulo, el archivo no existe en disco. El Shadow Forest solo recibe el fallback con features constantes degradadas (`obi=±0.5, spread=1.0, atr=0.002, hurst=0.50` — `online_random_forest.rs:46-58`): sus "umbrales óptimos" son ruido ajustado a etiquetas win/loss. **El umbral adaptativo de producción entrena sobre constantes.**
2. **El gate aritmético hace que la promoción casi nunca dispare.** t-stat ≥ 2.0 con edge típico de scalping (+0.05% media, 0.5% std) exige ~400 trades cerrados; con ventana de 1000 retornos, warmup 900s y eval cada 180s, la primera promoción puede tardar días-semanas — y si nunca dispara, el daemon solo puede PARAR trading (drift/kill), nunca mejorarlo. En backtest la promoción es diaria con regla permisiva.
3. **Cuando dispara, muta el genoma equivocado.** `champion_path` (`config_dir/genotypes/online_champion.json`) no lo escribe nadie y no existe: cada ciclo clona `SuperGenotype::default()` y promueve una perturbación aleatoria del default, **destruyendo el linaje del campeón de backtest** en `active.json`.
4. **`refresh_models` pisa los hot-swaps cada 1000 ticks.** El engine relee `active.json` y re-aplica; el harvest del shadow forest se aplica SIN promote → revertido en ≤1000 ticks; los umbrales que el daemon escribe cada 500ms son sobreescritos por los valores del genoma en disco.
5. **Contaminación cruzada sin separación de entornos.** El backtest promueve al MISMO `active.json` del que bootea producción (rutas relativas, misma máquina): un overfit histórico se convierte en genoma live en segundos (vía refresh_models); las promociones débiles de producción degradan al campeón entre corridas. No existe sufijo demo/ vs prod/.
6. **Espacio de mutación disparejo.** El backtest muta 139 genes vía 10 nichos; el daemon muta ~10 genes con ruido uniforme — ninguno de los genes con los que el backtest gana (`tech_threshold`, `weight_obi`, `maker_*`, `explosive_*`, trailing) se muta en producción. El fitness del daemon es un juguete (clamps ±sl/tp, fee plano, la mayoría de candidatos saltan el filtro) — optimiza un modelo de la estrategia, no la estrategia. Darwin muta 12 campos sobre una ventana de minutos (overfit trivial) y reconstruye el genoma descartando el drift de los otros ~127 genes.
7. **Física dispareja.** Producción aplica el envelope Kelly F5.1 (evidencia insuficiente → f=0 → entradas bloqueadas), interlock de latencia, filtro de flash-crash, kill-switch inmune — nada de eso existe en backtest. Un genoma que "funciona" en backtest puede simplemente no operar en producción por leverage 0.
8. **Tarea fantasma de 6h**: `god_engine.rs:447-469` spawnea `cargo run --bin evolution` que escribe `dynamic_config.json`... y el watcher dice "requires restart" y no hace nada. CPU pura, impacto cero.
9. **CMA-ES híbrido degradado**: mezcla CMA+PSO rompiendo los supuestos de actualización de covarianza; Cholesky fallback que regulariza silenciosamente; semilla `Genotype::default()` nunca cargada del almacén.
10. **Nota de verificación**: un agente reportó que el binario de backtest llamaría a un `process_tick` inexistente; se verificó compilación del workspace en verde — ese punto específico se descarta, pero la divergencia de rutas (`process_event` en producción vs `process_tick` 5-tupla en backtest con doble increment_tick) es real y está documentada arriba.
11. **Complemento**: la transición mainnet tiene `is_demo_mode` y `USE_TESTNET` independientes sin cross-check (se puede operar mode=prod contra endpoints testnet sin advertencia); el boot auto-promociona desde el espejo legacy como efecto secundario de LEER (bump de generación espurio); existe una cuarta ruta de persistencia no sancionada (`genesis_genome.json`).

**Conclusión:** el genoma de producción está evolutivamente paralizado (1-2), des-heredado (3), revertido cíclicamente (4) y contaminado en ambas direcciones (5). El impacto asimétrico que observa el operador es la conjunción de los cinco.

---

## 🧾 7. Censo de arbitrariedades persistente (post-f3)

El commit f3 erradicó router 0.85/0.015, RR 2.0 y cutoff 0.08, pero persisten 24 arbitrariedades catalogadas con anclaje propuesto (tabla completa en auditoría de origen; extracto de las de mayor impacto):

| # | Ubicación | Valor | Anclaje teórico/genómico propuesto |
|---|---|---|---|
| B-01 | `consejo_seniors.rs:132-133` | Hurst 0.55/0.42/0.65/0.35 | Percentil empírico de la distribución de Hurst o genes |
| B-02 | `consejo_seniors.rs:238-239` | 25/75 bps slippage máx | k·ATR_t + spread L2 observado |
| B-06/07 | `online_daemon.rs:60,155` | warmup 60/900s, eval 60/180s | n mínimo para varianza acotada del posterior Beta; semi-período de `regime_duration_ms` |
| B-10 | `orchestrator.rs:89` | `ensemble_boost = 1+(n−1)·0.1` | `1/√(1+ρ(n−1))` con ρ medido — corrige además K-multicolinealidad |
| B-12 | `god-engine-core/lib.rs:982-1006` | `0.70 + score·0.30` | Calibración ROC score→probabilidad |
| B-13/14 | `god-engine-core/lib.rs:581,617` | `tp.max(sl·2.5).max(0.0080)`, `trail=tp·0.60` | `tp_rr_ratio_btc` (ya genómico en f3 pero NO aquí); genes `scalp_trail_*` existentes |
| B-15 | `god-engine-core/lib.rs:1317` | max_position_size 50000.0 | leverage_cap × capital × concentración máxima |
| B-16 | `correlation_guard.rs:26,43,60` | `<30.0`, fallback capital 13.0 | `min_notional/(f_ruina·leverage)` |
| B-19 | `kelly.rs:33` | `·0.25` exploración (nuevo de f0-f2) | shrinkage n/(n+k) del EdgePosterior ya existente |

---

## 🎯 8. Hoja de Ruta de Rehabilitación (priorizada por palanca)

**Nivel 0 — Desbloqueo evolutivo (sin esto, nada de lo demás importa):**
1. Unificar clamps de `from_vector` con `get_lower/upper_bounds` (fuente única de verdad por macro) — desbloquea el gate K-01.
2. Separar almacenes de genoma por entorno (`genomes/demo/`, `genomes/prod/`) y añadir lock de archivo en `promote()`.
3. Escribir `online_champion.json` (o leer del envelope activo) y reordenar darwin a gate→apply.

**Nivel 1 — Verdad de datos:**
4. Eliminar el lookahead de los 4 generadores de ticks (OBI neutral o derivado de la vela PREVIA; trayectoria intra-bar sin orden por dirección).
5. Escribir el bus mmap de telemetría desde el engine (frames 12/30 reales) o eliminar el pipeline fantasma y alimentar el forest con features vivas.
6. Corregir el indicador M2 del World Bank y marcar/eliminar el macro sintético.

**Nivel 2 — Integridad de ejecución:**
7. `reduceOnly=true` en cierre y OCO + motor de cancelación de pierna hermana (ORDER_TRADE_UPDATE).
8. Fix del hot-swap mainnet (re-inyectar arena/registry/NTP al nuevo executor).
9. Reconciliación periódica programada + resolución ambigua en piernas OCO + guard monótono de status.

**Nivel 3 — Conexión de inteligencias:**
10. Cablear `ConsejoDeliberacion` en `process_tick_dual` (es el veto de gobernanza diseñado) y las APIs parametrizadas de filtros en lugar de los `evaluate()` hardcoded.
11. Reconectar el `ValidatedOrder` completo en producción (K-05) y unificar TP/SL por horizonte.
12. Conformal real con set de calibración, o eliminación honesta del filtro.

**Nivel 4 — Sistema continuo universal:**
13. Completar la migración Continuous (codificación 3-bucket sin colisión, sin espejo de posición, reporting por horizonte real).
14. Des-colinear el ensamble (diversificar fuentes: CVD real, trades, funding) y re-derivar `expected_volatility` como volatilidad real (EWMA de retornos).

---

*Fin del informe de aseguramiento. Se agrega sin sustraer contenido a los informes forenses previos. Todas las referencias archivo:línea corresponden al árbol en el commit `286a6a7a`.*
