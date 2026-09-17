# 🌊 DUODÉCIMA OLA FORENSE — AUDITORÍA DE ASEGURAMIENTO INTEGRAL DEL ESPECTRO CONTINUO (X-001 a X-045)

> **Artefacto:** INFORME_ASEGURAMIENTO_DUODECIMA_OLA_X.md — anexo a INFORME_FORENSE_MAESTRO.md (ola D-1..D-450 previa).
> **Alcance:** raíz a cima, posterior a las olas D y a las intervenciones F0-F8 (dos sesiones de ingeniería concurrentes). Auditoría de SOLO LECTURA: sin escritura de código. Prefijo **X-** (sin colisión con series D/E/G/H/N/S/U).
> **Motivo del operador:** miles de cambios ⇒ auditar desajustes, comportamientos no esperados, desconexiones; **pregunta raíz: por qué el genoma impacta en backtest pero no en producción/demo**; erradicar el binario scalp/swing; censo de rigidez de filtros; bloqueos de inteligencia; arbitrariedades sin teoría; lógica, latencia, cuellos de botella.
> **Método:** 5 auditorías paralelas de solo lectura (genoma-lifecycle, rigidez/filtros, core-lógica, ejecución/riesgo, orquestador/datos) + síntesis causal. Toda afirmación con evidencia archivo:línea verificada.

---

## 🧬 1. RESPUESTA CAUSAL A LA PREGUNTA RAÍZ (genoma: backtest sí, producción no)

Cadena completa de 6 eslabones, cada uno verificado:

1. **El GA evoluciona contra un simulacro (X-006).** El backtest del motor de evolución (`backtest-engine/src/lib.rs`) re-sintetiza 30 micro-ticks por vela con puente browniano hacia el cierre conocido; fabrica OFI (`bid_ratio` 0.8/0.2 según el delta del bar previo, `lib.rs:105-113`); alimenta omni-54 con macro CONGELADO hardcodeado (DXY=104.2, SP500=5120, `lib.rs:135-203`); inventa timestamps uniformes de 60 s (`lib.rs:316`) sin relación con el inter-arribo real; y fabrica `is_buyer_maker` como `tick_ret<0` (`lib.rs:302`). El "alpha" que el GA encuentra es ajuste a artefactos de la síntesis, no del mercado. `evolver.rs:307-340` (ticks reales + `process_tick`) aún diverge: su omni usa PRECIOS donde producción alimenta PORCENTAJES (`omni[0]=bid_price`), y su slippage es penalización fija por nocional (`:365`).
2. **El campeón ni siquiera llega (X-001, X-002).** Los evolvers promueven a la raíz COMPARTIDA del almacén (`TG_GENOME_ENV` no la fijan); producción corre como `prod|demo` y `load_active` hereda la línea compartida solo si `genomes/{env}/active.json` no existe — ocurrió una vez (primer arranque; disco verificado: raíz compartida en generación 1). Además no existe watcher de genomas: la carga es solo al arranque y el "HOT-RELOAD Watcher" solo mira `dynamic_config.json` (y su reacción es loguear "requires restart") y `models/*.json`.
3. **Y si llegara, aplicaría el binario (X-003).** `apply_to_arena` escribe TP/SL derivados de las curvas y a continuación PISA los mismos átomicos con las anclas — última escritura gana. El continuo F8 es decorativo en el camino vivo.
4. **La evolución de curvas es borrada sistemáticamente (X-004/X-005).** El vector genético tiene 140 slots — todos anclas; (a,b) sin slots ni bounds. Todo `to_vector→from_vector` (sanitización del GenomeStore, CMA-ES, candidatos de online_daemon) reconstruye curvas DESDE anclas. La mutación doble independiente anclas/curvas crea dos verdades divergentes; el reparo RR no cubre curvas (TP(τ)<SL(τ) posible sin detección).
5. **La única evolución en runtime es local y divergente.** online_daemon muta desde el genoma corriente (walk-forward sobre <100 retornos); shadow-forest cosecha umbrales ML; ninguno carga al campeón promovido. El arena deriva lejos del certificado.
6. **Darwin inerte por defecto** (dos flags de entorno) y su salida pasa por otro reconstructor que vuelve a borrar curvas.

**Veredicto:** la no-transferencia del edge no es mala suerte: es la arquitectura. El GA optimiza features que producción jamás ve; su resultado no llega al motor; y el mecanismo de llegada aplicaría los valores que el GA ni siquiera creyó evolucionar.

---

## 📊 2. MATRIZ CONSOLIDADA (X-001 a X-045)

| ID | Subsistema | Ubicación | Tipo de fallo | Severidad |
|---|---|---|---|---|
| X-001 | genome_store | `genome_store.rs:34-42,151-172`; `god_engine.rs:95` | Trapdoor ambiental unidireccional: promociones a raíz compartida invisibles para prod/demo para siempre | 🔴 CRÍTICO |
| X-002 | god_engine | `god_engine.rs:351-352, 881-882, 407-455` | Genoma sin watcher de runtime: evolución = cero efecto hasta reinicio manual | 🔴 CRÍTICO |
| X-003 | genome | `genome.rs:879-896 vs 944-956` | apply_to_arena pisa curvas con anclas: el continuo F8 es decorativo en producción | 🔴 CRÍTICO |
| X-004 | genome | `genome.rs:1684-1830, 2051-2086` | Coeficientes (a,b) ausentes del vector genético (dim=140) y de los bounds: evolución de curvas muerta en toda vía vectorial | 🔴 CRÍTICO |
| X-005 | genome | `genome.rs:1465-1496, 1657-1662` | Mutación doble divergente anclas/curvas; reparo RR no cubre curvas | 🔴 CRÍTICO |
| X-006 | backtest-engine | `lib.rs:105-113,135-203,221-316`; `evolution.rs:108-153`; `evolver.rs:307-365` | Evolución sobre microestructura sintética (OFI/omni/timestamps/maker fabricados) | 🔴 CRÍTICO |
| X-007 | ejecución | `executor.rs:1270-1335`; `god_engine.rs:1758-1764` | Ghost position: orden aceptada tras read-timeout ⇒ rollback local de posición viva en exchange, sin reconciliación automática | 🔴 CRÍTICO |
| X-008 | ejecución | `god_engine.rs:1561-1572` | Cierre: OCO cancelado primero; close fallido ⇒ solo log; core ya cerró local ⇒ posición naked sin TP/SL | 🔴 CRÍTICO |
| X-009 | ejecución | `god_engine.rs:1791-1792` | Emergency-close del OCO con errores tragados (`let _`) sin escalada a kill-switch | 🔴 CRÍTICO |
| X-010 | inmune | `god_engine.rs:806`; `state.rs:379`; grep writers | Rama de latencia MUERTA: `last_ws_latency_ms` sin writers en el feed real (siempre 0) ⇒ breach jamás dispara ni ante feed muerto | 🔴 CRÍTICO |
| X-011 | user-data | `god_engine.rs:624-634, 1414-1424` | Streamer zombie post-transición: 2 streams vivos; ACCOUNT_UPDATE de testnet pisa capital mainnet | 🔴 CRÍTICO |
| X-012 | core | `lib.rs:644-666 vs 586-591` | Interlock de latencia retorna ANTES de gestión de posiciones: en desconexiones no se evalúa SL/trailing/zombie/toxic | 🔴 CRÍTICO |
| X-013 | inmune/capital | `god_engine.rs:788, 600-610` | Doble plano de capital: inmune mide dd en arena simulado; la verdad del exchange escribe otro AtomicU64 | 🟠 HIGH |
| X-014 | evolución | `evolution.rs:542-551` | Campeón del SA exportado a `dynamic_config.json` que nadie lee en runtime | 🟠 HIGH |
| X-015 | evolución | `polars_evolver.rs:69,133-154` | Promueve genomas ALEATORIOS (`Genotype::new_random`) por el embudo | 🟠 HIGH |
| X-016 | espectro | `temporal_spectrum.rs` (grep de callers) | Espectro sin consumidores: `fused_score`/`signal_at`/`dominant_tau_ms` cero llamadas; O(19)/evento de overhead puro; solo `process_event` lo actualiza (dual/tick divergen) | 🟠 HIGH |
| X-017 | warmup | `bootloader.rs:287-330`; `god_engine.rs:1102-1109`; `stateful_engine.rs:305-358` | Klines 1m logueados como "(1h)"; OHLC inyectado como banda constante ±0.05% con volumen fijo 10 ⇒ `v_t`/ATR mal calibrado por horas (alimenta stops vivos) | 🟠 HIGH |
| X-018 | shadow-forest | `god_engine.rs:1316-1323`; `random_forest.rs:87-116` | 11× `process_event` por tick (10 motores sombra) en el hilo TIME_CRITICAL | 🟠 HIGH |
| X-019 | core | `lib.rs:490-497, 392-404` | `refresh_models()`: lectura de disco + parse JSON del sobre de genoma cada 1000 ticks DENTRO del hot loop | 🟠 HIGH |
| X-020 | arena | `god_engine.rs:719-760, 941-961` | `arena_shadow`: 40+MB VirtualLock muertos (nunca procesa un tick) + leak del canal `tx_shadow` (modelos NN encolados eternamente) | 🟠 HIGH |
| X-021 | símbolos | `symbol_manager.rs:75-87`; `god_engine.rs:192-206,520` | "Universo vivo" parcial: specs/registry sí; pero WS-subscriptions, `symbol_to_id` y slots de engine congelados en la lista hardcodeada — mensaje de éxito engañoso | 🟠 HIGH |
| X-022 | riesgo | `god_engine.rs:1624-1627,1693`; `kelly_envelope.rs:195-205` | Envolvente Kelly advisory-only: `.max(core_leverage)` pisa el cap; `operable=false` aún ejecuta (bloqueo `exec_leverage==0` inalcanzable); constante 5.05 duplicada entre capas | 🟠 HIGH |
| X-023 | ensamble | `lib.rs:473-484, 1520-1546` | Calificación con lead-time cero: se califica la predicción del tick inmediatamente previo al cierre del kline, no la que impulsó la entrada | 🟠 HIGH |
| X-024 | ensamble | `lib.rs:1929-1953` | El gate swing consume predicción NN cruda — BYPASS del ensamble (los pesos aprendidos solo afectan el scalp) | 🟠 HIGH |
| X-025 | simulador | `multi_coin_simulator.rs:433-470` | 2× `process_event` por tick (depth+trade) ⇒ volumen/OFI duplicados vs producción; `predict_for_coin` actualiza estadísticas Welford 2× | 🟠 HIGH |
| X-026 | flatten | `executor.rs:445-459, 492-597` | -2022 (TP llenó entre snapshot y close) sin manejo: println + Ok ⇒ inmune reporta "Aplanado" con posición abierta; WS fast-path cuenta cierres fire-and-forget | 🟠 HIGH |
| X-027 | rollback | `god_engine.rs:1676-1690` | Rollback binario con `fetch_add(-margin)` sin clamp (puede ir <0 ⇒ margen libre inflado); el core usa clamp | 🟠 HIGH |
| X-028 | telemetría | `god_engine.rs:1629,1651` | `record_entry` de trayectoria y `tx_log_worker` emitidos ANTES del spawn de orden; no revertidos en rollback ⇒ scores/logs contaminados | 🟡 MED |
| X-029 | régimen | `lib.rs:1693-1694, 2062-2095` | Hurst 0.45/0.50 hardcodeado en 6 sitios de decisión mientras los genes `hurst_*` del genoma están huérfanos | 🟠 HIGH |
| X-030 | OCO | `god_engine.rs:1637-1638` | Fallback TP/SL ±40/±30 bps fijos: ignora curvas, τ y ATR — última fuente binaria de TP/SL en el camino vivo | 🟠 HIGH |
| X-031 | risk-engine | `lib.rs:262-265` | Split contable falso: `scalp_wr` y `swing_wr` leen el MISMO `coin.metrics.win_rate` | 🟠 HIGH |
| X-032 | genoma | `config.rs:20-160` (censo) | ~20 anclas per-pole SIN curva (trailing×10, kelly×2, hurst×2, hawkes, obi×2, accel×2, capital_split): evolucionan como pares binarios, ciegos al espectro | 🟠 HIGH |
| X-033 | arquitectura | censo §4 | Doble pipeline de evaluación completo (gates, modelos, intents) + MoE dual + trayectoria dual-track: la decisión sigue siendo binaria | 🟠 HIGH |
| X-034 | tick-ring | `state.rs:157-176` | Torn reads: len y head cargados por separado (Acquire) ⇒ `prev_price` desgarrado en el filtro flash-crash durante wrap del anillo | 🟡 MED |
| X-035 | espectro | `temporal_spectrum.rs:118-129` | Gap blindness: dt no acotado; tras reconexión α→1 y TODAS las EWMAs del espectro hacen reset silencioso | 🟡 MED |
| X-036 | core | `lib.rs:520-526, 700` | `update_macro_features` ejecutado 2× por evento depth | 🟡 MED |
| X-037 | ensamble | grep global | Guard de degeneración (prob≥0.9999⇒0.5) ELIMINADO: un modelo saturado contribuye probabilidades extremas ~50+ barras antes de decaer | 🟡 MED |
| X-038 | red | `.env:62`; `god_engine.rs:1957-1965` | IPs AWS-Tokyo stale (2023) compiten vía select_ok contra GeoDNS fresco; `BinanceStreamer` (ws_client.rs con el fix F2.2) es código muerto — la conexión real usa otro camino | 🟡 MED |
| X-039 | hot-loop | `god_engine.rs:1292, 2023-2025, 1356` | Allocs/tick: `snapshot_recent(1)` devuelve Vec; `msg.into_data()+clone()`; `orchestrator.write()` exclusivo por tick; SipHash por lookup de símbolo | 🟡 MED |
| X-040 | inmune | `god_engine.rs:785` | `STOP_TRADING.LOCK` relativo al CWD: lanzado desde otro directorio jamás se encuentra | 🟡 MED |
| X-041 | OS | `os-guardian/lib.rs:163-177`; `god_engine.rs:714,1930` | `SetThreadIdealProcessor(1)` en AMBOS hilos críticos colisionando con el pinning explícito core_ids[1]/[2]; 2× TIME_CRITICAL sin análisis de starvation | 🟡 MED |
| X-042 | config | `config.rs:216`; risk-engine `lib.rs:9` | `regime_duration_ms` sin consumidor vivo (su módulo `macro_regime_swing_optimizer` jamás se invoca) | 🟢 LOW |
| X-043 | contabilidad | `god_engine.rs:1528-1529` | `gross = pnl + fee` reconstruido con fees actuales al precio actual — ignora entry fee y física de fees ⇒ `total_fees` inconsistente con lo deducido real | 🟢 LOW |
| X-044 | telemetría | `ensemble.rs weights()/log_briers()` | Peso/calibración del ensamble SIN visibilidad: nunca leídos | 🟢 LOW |
| X-045 | memoria | `lib.rs:486` vs `god_engine.rs:1508` | Kill-switch: load Relaxed vs store SeqCst — formalmente no sincronizado (benigno en x86) | 🟢 LOW |

---

## 🔬 3. DETALLE EXPANDIDO DE LOS CRÍTICOS

**X-001 — Trapdoor ambiental.** `GenomeEnvelope` enraíza por `TG_GENOME_ENV` (prod|demo|backtest|compartido). god_engine fija la variable (`:95`); los evolvers no. Promoción→raíz compartida. `load_active` solo cae a la compartida si falta la del entorno, y al hacerlo la **lava** vía `from_vector` (matando curvas). Una vez que existe `genomes/prod/active.json` (primer arranque), la herencia se cierra PARA SIEMPRE: cada promoción del evolver es invisible a producción incluso tras reinicios. Esclusa de un sentido.

**X-002 — Sin watcher.** Carga única en arranque (`:351→:881`). Los `apply_to_arena` runtime (online_daemon `:715,305`; cosecha shadow `god_engine.rs:1896`) mutan el genoma corriente; ninguno carga el campeón del almacén. La única vía de refresco es `refresh_models()` cada 1000 ticks (X-019) que sí lee `GenomeEnvelope::load_active()` — PERO en el entorno actual (X-001) ese archivo es la línea local muerta, no el campeón compartido. El refresco existe y refresca nada.

**X-003 — Orden de stores.** `genome.rs:879-896` almacena `tp_horizon_curve.eval(...)` en `scalp/swing_tp_base`; `:944-956` vuelve a almacenar `self.scalp_tp_base` (ancla) en el mismo atómico. La segunda escritura anonima la primera. Grep: `tp_horizon_curve` no tiene más consumidores. Conclusión dura: **la migración F8 no cambió un solo valor efectivo en producción.**

**X-004 — Dimensión 140.** `to_vector`/`from_vector`/bounds: 140 anclas, cero coeficientes de curva. GenomeStore::sanitize, CMA-ES y online_daemon hacen roundtrip vectorial ⇒ curvas reconstruidas desde anclas. La "pendiente evolucionable" declarada en F8 no es evolucionable en ninguna vía real.

**X-005 — Dos verdades.** `mutate_with_rng` muta anclas (líneas 1489-1496) y curvas (1465-1468) independientemente: tras una mutación la curva ya no pasa por las anclas. El invariante RR (1657-1662) repara solo anclas: las curvas divergentes pueden violar TP(τ)>SL(τ) sin gate.

**X-006 — El simulacro.** Además de lo ya descrito: el GA optimiza `ev_fee_multiplier=1.0` (correcto desde F3.6) pero contra un feed donde el `is_buyer_maker`决定 OFI… y el live drop de glitches/flash-crash (`god_engine.rs:1291-1303`) NO existe en backtest: el GA nunca penaliza secuencias que producción descarta, ni ve las que producción sí opera (omni real desde F4.1).

**X-007 — Ghost por transporte.** `execute_raw_qty_with_client_id` devuelve Err en timeout de lectura aunque Binance haya ACEPTADO la orden; el binario hace rollback local (`.1758-1764`) de una posición viva en el exchange. La reconciliación de arranque la detectaría — en el PRÓXIMO reinicio. Entre tanto: posición sin gestión local.

**X-008 — Orden letal de cierre.** Cancelar OCO antes de cerrar es protocolo invertido: si el close falla, la protección ya no existe. Con `let _` en el error (`:1568`) y el core ya cerrado localmente: naked real + limpio local.

**X-009 — Emergency tragado.** El path de emergencia del OCO externo (3 retries → cancel-all + market close + rollback) ignora los errores de cancel-all y close (`:1791-1792`): el peor momento para silenciar es exactamente ese.

**X-010 — Latencia inmune muerta (verificado por grep).** Writers de `last_ws_latency_ms`: `ws_client.rs:254,405` (muerto — X-038) y `src/data/ws_client.rs:70,77` (no usado). El feed real (`god_engine.rs:2018-2053`) tiene su propio watchdog pero NO escribe la métrica. Valor queda 0 ⇒ strikes siempre en 0 ⇒ breach imposible. El inmune es tuerto a la latencia.

**X-011 — Zombie streamer.** Sin handle abortable; transición spawnea el segundo; ambos sinks → mismo AtomicU64 de capital local. ACCOUNT_UPDATE de TESTNET sobrescribe el balance MAINNET leído para sizing.

**X-012 — Interlock mata salidas.** `:644-666` retorna antes de `manage_positions`: durante el pico de latencia (desconexión parcial, exactamente el momento del cisne) no se evalúa NINGÚN exit. El comentario promete cierres defensivos; el código los prohíbe.

---

## 🚫 4. CENSO DE RIGIDEZ SCALP/SWING (qué es VISTA, qué es RÍGIDO)

| Zona | Evidencia | Clasificación |
|---|---|---|
| config.rs (~24 campos) + genome (~24 anclas) | tp/sl teóricamente VISTA (curvas… salvo X-003); ~20 restantes (trail×10, kelly×2, hurst×2, hawkes, obi×2, accel×2, capital_split) RÍGIDAS per-pole | RÍGIDO (X-032) |
| state.rs: `ScalpState`(12)+`SwingState`(11)+márgenes duales | Rama binaria `is_pos_swing` en core `:1175-1330` | RÍGIDO |
| position.rs: `DualPosition{scalp,swing}` + enum 3 slots | `:272-298` | RÍGIDO |
| core: doble pipeline eval completo | scalp `:1680-1927` vs swing `:1957-2118`; modelos duales; intents duales; fusión con inercia ×1.50 | RÍGIDO (X-033); TP/SL/trailing/tensor = VISTA (lerps) |
| stateful_engine: features duales [12] vs [34] | `:389,429`; `can_open_scalp(250)`; `last_scalp_was_loss` | RÍGIDO |
| signal-engine: motores + consenso dual + swing ×1.20 | `orchestrator.rs:230-248,425` | RÍGIDO |
| risk-engine: CapitalSplitter dual + Kelly dual + wr falso | `:105-357, 262-265` | RÍGIDO (X-031) |
| moe_neat_arena: `scalp_moe`/`swing_moe` duales | `:19-51` | RÍGIDO |
| reconciliation: adoptadas→Swing; alimenta métricas duales | `:206-362` | RÍGIDO |
| multifractal: viabilidades con cortes H duros | `:186-187` | RÍGIDO |
| backtest continuous_evolution: rescates/clamps de anclas | `:273-430` | RÍGIDO |
| god_engine binario | labels = cosmético; trayectoria dual-track = rígido | MIXTO |

**Lectura del consejo:** la migración F8 construyó el continuo de DATOS (espectro, curvas) pero la DECISIÓN sigue binaria de punta a punta. El binario no fue erradicado: fue envuelto.

---

## 🧮 5. CENSO DE FILTROS: FÍSICOS vs ARBITRARIOS

**Físicos/datos (se mantienen):** spread-book 50% (F2.1); stasis 6σ (piso 0.8% documentado); gate de spread escalado por ATR; break-even fee-based (el ×3 es arbitrario-menor); flash-crash genómico.

**Arbitrarios sin teoría (hallazgo colectivo — cada uno es deuda de justificación o eliminación):**

| # | Filtro | Valor | Ubicación |
|---|---|---|---|
| 1 | Fallback OCO TP/SL | ±40/±30 bps fijos | `god_engine.rs:1637-1638` (X-030) |
| 2 | Piso stop_pct | 0.0015 | `god_engine.rs:1599,1607` |
| 3 | Frontera micro-cuenta envolvente | $50; z=0.85/k=10 vs 1.64/k=50 | `god_engine.rs:1609-1627` |
| 4 | Alta confianza | 0.80/0.20 | `god_engine.rs:1632` |
| 5 | Hurst régimen | 0.45/0.50 ×6 (genes huérfanos) | `lib.rs:1693-2095` (X-029) |
| 6 | Activación trailing | tp×0.60 | `lib.rs:839` |
| 7 | Piso tendencia | 0.52 | `lib.rs:1963` |
| 8 | Mapa confianza | (0.50+0.40\|s\|)∈[0.51,0.90] | `lib.rs:1712` |
| 9 | Gates tensor | 0.70/0.60 | `lib.rs:1813,2065-2080` |
| 10 | Cooldown scalp | 250 ticks | `lib.rs:1691` |
| 11 | Gate MACD swing | tp×0.003/hurst | `lib.rs:2003-2005` |
| 12 | Pesos fusión | ×1.50/×1.10/×1.20 | `lib.rs:2144,2133`; `orchestrator.rs:425` |
| 13 | Split Bayes residual | 60/40 | `lib.rs:1675-1676` |
| 14 | Viabilidad multifractal | H>0.60/<0.40 | `multifractal.rs:186-187` |
| 15 | Glitch/confirmación WS | 20%/3 ticks | `ws_client.rs:335-342` |
| 16 | Clamps spread-entry | 0.0006/0.0025 (y el 0.25×) | `lib.rs:1686` |
| 17 | Constante 5.05 | duplicada entre capas | `god_engine.rs:1622`; `lib.rs:2268,2277` |
| 18 | Clamps core SL/TP | 1.5×ATR y [0.0010,0.0300]/[0.0020,0.0800] | `lib.rs:814-821` |

---

## 🧠 6. BLOQUEOS DE INTELIGENCIA (capacidad construida que el sistema ignora)

1. Espectro temporal completo SIN consumidores (X-016) — se paga O(19)/evento por nada.
2. Curvas de horizonte: pisadas (X-003), no evolucionables (X-004), divergentes (X-005).
3. Ensamble: calificado con lead cero (X-023), bypaseado por swing (X-024), sin guard de degeneración (X-037), pesos invisibles (X-044).
4. Genes `hurst_*` configurados y nunca leídos (X-029); `regime_duration_ms` sin consumidor (X-042).
5. Campeón del SA exportado a un archivo sin lectores (X-014); Darwin inerte tras dos flags.
6. `Lakehouse::record_tensor` sin callers — el almacén tensorial vive apagado.
7. Estructura: watcher de genomas inexistente + trapdoor ambiental + polars random — la inteligencia evolucionada está DESCONECTADA de la ejecución por diseño accidental.
8. Warmup que degrada (X-017): el swing-state arranca horas con ATR sintético plano, des-calibrando exactamente al modelo que más lo necesita.

## ⚡ 7. CUELLOS DE BOTELLA Y LATENCIA (orden de impacto)

1. ShadowForest 11× por evento en hilo crítico (X-018). 2. refresh_models disco+JSON en hot loop (X-019). 3. arena_shadow 40+MB VirtualLock muerto + leak de canal (X-020). 4. Allocs por tick: Vec snapshot, msg clone, write-lock, SipHash (X-039). 5. CPU: ideal-processor(1) duplicado + 2× TIME_CRITICAL sin estudio de starvation (X-041).

## 🎯 8. DICTAMEN DEL CONSEJO DE ROLES SENIOR (ola X)

- **Arquitecto de Sistemas:** tres verdades de genoma (anclas/curvas/vector) de las que solo una opera; una esclusa ambiental desconecta evolución de producción; F8 está construido y desconectado en tres cuartos. La factura de dos sesiones concurrentes sin integración formal.
- **Quant Sénior:** el edge del backtest es ajuste a artefactos sintéticos y estructuralmente intraducible (X-001/002/003/006). Responder "por qué no impacta" exigió 6 eslabones — todos rotos.
- **Risk Officer:** envolvente consultiva (X-022), inmune tuerto a latencia (X-010), doble plano de capital (X-013), salidas bloqueadas en crisis (X-012): las capas de defensa tienen modos silenciosos de no existir.
- **SRE HFT:** los CRÍTICOS de ejecución comparten patrón "estado local optimista + error de exchange tragado" (X-007/008/009/026): el sistema reportará salud mientras hay posiciones naked.
- **Investigador de IA:** el ensamble aprende la cantidad equivocada y el swing no pasa por él; la calibración viva sigue siendo promesa. El espectro — la pieza más cara conceptualmente — no decide nada todavía.

## 🗺️ 9. HOJA DE RUTA DE REHABILITACIÓN 1-A-1 (por dependencia causal)

1. **Continuo real:** X-003 (curvas mandan en apply_to_arena, stores únicos) → X-004/X-005 (curvas en el vector + mutación única + RR sobre curvas) → X-016 (consumidores: señales leen fused_score/signal_at; TP/SL por τ dominante; horizon continuo con τ viva) → X-032/X-033 (familias restantes a curvas; colapsar el dual-pipeline).
2. **Circuito evolutivo cerrado:** X-001/X-002 (una raíz; watcher runtime con apply atómico) → X-014/X-015 (todos los evolvers por el embudo, con candidatos reales) → X-006 (GA evoluciona contra el forense: aggTrades + omni histórico REAL).
3. **Protocolo de ejecución a prueba de fantasmas:** X-007 (reconcile-then-rollback), X-008 (close-antes-de-cancelar), X-009 (errores de emergencia escalan al inmune), X-026 (-2022 con re-snapshot y retry).
4. **Defensa que existe de verdad:** X-010 (writer de latencia en el feed real), X-011 (abort del streamer viejo), X-012 (interlock bloquea entradas, nunca salidas), X-013 (un plano de capital: el del exchange), X-040 (lock por ruta absoluta), X-022 (envolvente autoritativa).
5. **Aprendizaje honesto:** X-023/X-024/X-037/X-044 (pairing entrada↔outcome con lead real; un solo camino por el ensamble; restaurar guard; exponer pesos).
6. **Rendimiento:** X-017 (warmup OHLCV real), X-018 (shadow a hilo aparte o muestreado), X-019 (refresh fuera del hot loop), X-020 (liberar arena_shadow + canal), X-039 (cero allocs/tick).
7. **Des-arbitrarización:** X-029/X-030 + censo §5 — cada filtro queda como gen-curva, constante física documentada, o eliminado.

---
*Fin de la Duodécima Ola (X-001 a X-045). Serie X reservada a esta ola y sus re-verificaciones. Anexo referenciado desde INFORME_FORENSE_MAESTRO.md.*
