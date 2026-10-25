# Censo de motores muertos o desconectados (C-F7b), 2026-10-18

Línea C, sólo lectura. Tabla completa, con una fila por símbolo: `censo_motores_2026-10-18.tsv`, en esta misma carpeta.
Las líneas citadas se verificaron sobre `wt-ci` en 5d98d5d5 (árbol idéntico a main 710039b9). Main está hoy en c826b1f7, con Ω155 de AGY ya fusionado: cambiaron `god-engine-core/src/lib.rs`, `risk-engine/src/{lib,tp_sl}.rs`, `online_daemon.rs`, `perceptron_gate.rs` y `temporal_spectrum.rs` (+310/−72). Antes de cada átomo hay que volver a buscar cada ancla.

## 1. Qué es y cómo se hizo

El censo cubre 576 símbolos de los 23 crates: funciones, tipos, módulos, dependencias, claves del registro y genes, sin lector vivo o con un lector dudoso. Se repartieron en 4 lotes. En cada grupo, un verificador y dos escépticos confirmaron con `rg` el estado de hoy, la naturaleza del símbolo y la propuesta, y el censo cerró un `final` por fila. Después, seis analistas sintetizaron por tema (`reconectar`, `motores_retirar`, `revisar`, `dueno`, `retiro_plan`, `islas`) y un crítico de completitud revisó la cobertura, las contradicciones, las rupturas de compilación y los genes. **Donde el crítico contradice a la síntesis, manda el crítico**, y aquí se marca «(crítico)». Esta redacción añade tres correcciones que el crítico no vio (dos rupturas de compilación, en los lotes 42 y 44, y una aserción que F263 necesita, en el lote 30), verificadas con `rg` sobre c826b1f7; se marcan «(redacción)».

El TSV conserva las 18 columnas del censo y añade seis:

| Columna | Contenido |
|---|---|
| `veredicto_sintesis` | Veredicto final de la síntesis. Si ningún ítem cubre la fila, se copia el `final` del censo |
| `prioridad_sintesis` | Prioridad del ítem dentro de su lista. En `retiro_plan` coincide con el número de lote |
| `coste` | S, M o L, según la síntesis |
| `dueno_zona` | Dueño o zona que hay que coordinar |
| `plan_resumido` | Resumen del plan, de 300 caracteres como máximo |
| `fuente` | Clave de la síntesis; lleva `+critica` si el crítico cambió el ítem |

Regla de asignación: si un id aparece en varios ítems, gana el del crítico. Si no hay ítem del crítico, gana el ítem con menos ids. Hay 49 filas sin ítem (39 CONSERVAR y 10 YA_RESUELTO), que conservan el veredicto del censo, y el crítico decidió otras 49.

## 2. Recuento por veredicto final

| Veredicto | Filas (síntesis) | Filas (censo) | Qué significa |
|---|---:|---:|---|
| RETIRAR | 363 | 367 | Se borra en un lote atómico. |
| ZONA_AJENA | 53 | 27 | Se entrega a su dueño (§6). |
| UNIFICAR | 51 | 44 | Hay dos versiones: queda una, tras compararlas (regla 2). |
| CONSERVAR | 43 | 45 | Vivo o necesario; no se toca. |
| RECONECTAR_SOMBRA | 28 | 21 | Se conecta sin cambiar ninguna decisión y se mide (regla 4). |
| ENSAMBLE_SOMBRA | 20 | 0 | Candidato frente a un rival vivo, puntuado prequencialmente. |
| YA_RESUELTO | 11 | 11 | Ya está en main. |
| DECISION_DUENO | 7 | 38 | Sin recomendación cerrada (F268; ws_client/parser, lote 54). |
| RECONECTAR / REVISAR_ALCANCE | 0 | 11 / 12 | Ya reasignados por la síntesis. |
| **Total** | **576** | **576** | |

Las 38 filas que el censo dejó al dueño ya tienen recomendación de la síntesis: 22 RETIRAR, 12 RECONECTAR_SOMBRA, 3 UNIFICAR y 1 ENSAMBLE_SOMBRA. **Todas siguen pendientes del sí del dueño** (§5). Las 7 DECISION_DUENO que quedan vienen de lotes de retiro (F268 y el lote 54).
Contexto: 23 crates y 133.758 líneas de `src`. Unas 21.500 líneas (≈ 14 %) están en módulos muertos enteros. `polars` está declarado en 6 manifiestos y no se usa en producción.

## 3. Reconexiones en sombra, por prioridad

Dos arneses compartidos, que se construyen una sola vez:

- **LibroSombra**: resuelve contrafactualmente el primer toque de TP o SL para vetos y TP alternativos.
- **SkillSombra**: la maquinaria de `SkillMotores` (AcumIc, e-proceso, olvido 1/64, madurez 30) con una familia propia, sin tocar el Bonferroni de 480.

Con unos 41 cierres cada 50 h, las sombras por intención, vela o bloque dan evidencia en horas; las sombras por cierre tardan semanas. Ninguna cambia una decisión hasta pasar su contrato, y todas en las dos mitades. Las claves `sombra_*` nuevas no chocan con ninguna viva (crítico). La prioridad es la de la síntesis; los empates entre listas se ordenan por valor.

1. **F412, F413, F271: discrepancia del Auditor Interno** (crítico; RECONECTAR_SOMBRA; M; urgente).
   - Por qué: hoy veta con datos tautológicos (disc 0,231 > 0,20), y el veto no se recupera solo.
   - Plan: las tres patas devuelven `None` mientras no haya una fuente independiente; se quita el `registry.set` global (core lib.rs:5088); test RED→GREEN 13 → 16,9 → 13.
   - Sombra: fills del host (ORDER_TRADE_UPDATE) en `sombra_wr_disc` y `sombra_friccion_real_menos_modelada_bp`. Es el mismo átomo que el paso 1 de F407. T-1 y golden.
   - Archivos: auditor_interno.rs, core lib.rs, consejo_seniors.rs, god_engine.rs.
2. **F250, F272, F400: LibroSombra con `ultimo_rechazo`** (RECONECTAR_SOMBRA; M).
   - Por qué: dice qué vetos destruyen valor.
   - Plan: un thread-local `ULTIMA_GEOMETRIA` tras `tpsl_gate` (risk lib.rs:1015) e `IntencionVetada` en el núcleo. Sólo vetos posteriores a la geometría (suelo_tp_sl, confianza, EV, viabilidad, min_notional, fee, orquestador; consejo y ML como fuentes 19 y 20).
   - Contrato: H0 «el veto acierta» (E[R neto] ≤ 0), error estándar por bloques de τ, Bonferroni sobre unas 10 fuentes, n_min = (z·ŝ/δ)² con δ = fricción/SL (≈ 60 por fuente, inferido).
   - Crítico: indexar las fuentes por `veto_registry`/`REJECT_NAMES` (F267), no con una tabla nueva, y no desplazar los índices 7, 9 y 14 (F251).
   - Archivos: risk lib.rs, shadow_ledger.rs, core lib.rs; test nuevo `libro_sombra_e2e_contract.rs` (pureza bit a bit).
3. **F43: compuerta conformal de HiddenLiquidity** (RECONECTAR_SOMBRA; M).
   - Por qué: defecto vivo. risk lib.rs:674-687 aplica Kelly ×1,25 ante cualquier absorción, sin mirar la dirección; el −33 % nunca se alcanza.
   - Átomo: `conformal_kelly_mult = 1,0`, reescribir `conformal_kelly_microstructure_contract.rs` y T-1. Se retira `evaluate_conformal_gate`.
   - Sombra: iceberg por `is_buyer_maker` y reposición; `would_veto` frente a un veto al azar y a muro_en_contra; N ≥ 100 por moneda y bootstrap en dos mitades.
   - Aviso a GLM: `hidden_liquidity_pressure` es una razón de muros.
4. **F189: σ de Garman-Klass con OHLC del exchange** (ENSAMBLE; S).
   - Plan: `observar_vela_exchange` sin tocar `process_event`; el host (god_engine.rs:3512) aporta O, H y L.
   - Contrato: QLIKE/MSE frente a SigmaMinuto y `sigma_forecast_at(60 s)`; Diebold-Mariano HAC por moneda en dos mitades.
   - Archivos: normalizer.rs, core lib.rs, god_engine.rs; test `gk_sigma_sombra_contract.rs`. StatisticalNormalizer se retira en otro commit.
5. **F323, R152: `flujo_at` multiescala y el arnés SkillSombra** (ENSAMBLE; M).
   - Contrato: IC(flujo) > IC(cvd) a τ_op; H0 IC = 0 con e-proceso de familia. Se mide primero offline, en `spectral_bench` sobre aggTrades; la intensidad queda para después.
   - Requiere hacer pública `AcumIc` (skill_motores.rs, Línea G y Qoder).
6. **F418, F419, F135: DriftAuditor con sombra = 0,95·real, un mock** (RECONECTAR_SOMBRA; S).
   - P1: `centinela_contable`, sin cambio de conducta.
   - P2: Δdiv = Δcontrol − Δreal por cierre, con cortes de segmento en cada generación del control; e-proceso/CUSUM con H0 E[Δdiv] = 0, sin veto.
   - Que el control vea todos los mensajes cuesta CPU: es una decisión del dueño (§5, tema 3). Se secuencia con F273.
7. **F263: fricción por latencia con Hurst** (ENSAMBLE; M).
   - Offline: E|Δmid(L)|/mid para L entre 20 y 500 ms, frente a la ley viva atr·√(L/60 s) y la `_hurst`, por error log-ratio en dos mitades. En vivo: telemetría frente a `slippage_bps`.
   - Si gana: un átomo en los 9 sitios con T-1. Si no: se retira `_hurst`. Los métodos `TpSl::probabilidad_*` se retiran igual.
8. **F95: ratios LS y taker** (RECONECTAR_SOMBRA; S).
   - Plan: el poller recibe el roster en vez de `take(30)`; claves `sombra_ls_*` con edad.
   - Contrato: columna −ln LS; H0 IC = 0; un dato más viejo que 2 barras no opina.
9. **F222: `LiquidationObservation.is_buy` con signo** (ENSAMBLE; S).
   - Plan: dos envolventes (compra y venta) y `sombra_liq_neta`. N = número de cascadas distintas; rivales sgn(cvd_z) y una moneda al aire.
10. **F283, F437: residuo de `OnlineLearningModule`** (ENSAMBLE; M).
    - Hoy entrena con los rasgos del cierre (core :5506). Plan: snapshot de rasgos en la entrada por ranura y un aprendiz por moneda.
    - Contrato: log-loss frente a la base Platt; e-proceso en dos mitades.
11. **F193: área de Lévy** (ENSAMBLE; S). Columna en SkillSombra; rivales −r_k y la persistencia CL-30. Se mide offline primero.
12. **F204, F462: extensión de TP de VolatileMomentumBooster** (crítico; ENSAMBLE; M).
    - Importar `hawkes_bessel` en strategy-core crearía un ciclo; por eso la excitación normalizada la pasa quien llama.
    - Contrato: ΔR pareado al tocar el TP (core lib.rs:4599); rivales m = 1 y S-2; N ≥ 100 toques.
    - Gen 34 es un polizón que muta online_daemon.rs:1793: se congela.
13. **F195: OBI L2 de 5 niveles** (ENSAMBLE; S). Criterio IC(L2) > IC(L1) en las dos mitades; sólo en vivo. ART-2 y `obi_acceleration` se retiran aparte.
14. **F175: `calculate_expanded_tp`** (crítico; ENSAMBLE; S).
    - Recibe la dispersión al horizonte τ como argumento; una llamada signal → risk sería un ciclo. NaN ⇒ None.
    - Contrato: ΔR pareado frente al TP vivo en el LibroSombra. F1-118 ya está en main.
15. **F322, R150: pronóstico de volumen y nocional** (RECONECTAR_SOMBRA; S).
    - Plan: `habilidad_volumen`, espejo de la de volatilidad; se publica sólo con habilidad > 0 (R² OOS ya medido: +0,25…+0,50).
    - Su valor crece con el capital; con 13 USD el impacto es despreciable.
16. **F178: Kalman → «precio eficiente»** (crítico; ENSAMBLE; M).
    - Candidatos: mid, microprecio con β = 1 y β = ½ (F168), β̂ de Stoikov por RLS, Roll-Kalman.
    - Contrato: habilidad = 1 − MSE/MSE_persistencia, N ≥ 10⁴, dos mitades.
    - Consumidor: referencia de salida y trailing, y el IOC del host (god_engine.rs:4875), no router.rs:123, que está muerto. Usar `try_new`.
    - Después se retiran price_ring, jerk_t y `TensorRing::jerk`.
17. **F214, F463: MakerEngine** (crítico; RECONECTAR_SOMBRA; M).
    - Plan: cotizar sólo cuando hay intención de entrada. Cambia `process_tick_dual` ⇒ T-1.
    - Contrato: llenado en τ y markout a 1, 10 y 60 s; N ≥ 300.
    - La comparación va en el host, porque `passive_join_price` es pub(crate) y F233 lo dejaría huérfano. Gen 108 se congela. Tema 2.
18. **F184: potenciales de Hodge φ_i** (ENSAMBLE; S). Rivales ema_ofi y lead_lag; la divergencia se lee de la variable local (core lib.rs:4055), no de la clave que retira F187 (crítico).
19. **F242: contadores de trade_accounting** (RECONECTAR_SOMBRA; S). Se publican; si hubo censura, las estadísticas de la moneda pasan a «sin evidencia».
20. **F416, F275: coherencia del TrajectoryAuditor** (ENSAMBLE; M). τ de la orden; Brier a t/τ = ½ frente a mag_score. Valor bajo.
21. **F182: rasgo [6] sin olvido** (crítico; ENSAMBLE; M).
    - Plan: EWMA paralela en tiempo físico (`update_elapsed`); IC/log-loss fuera de muestra.
    - Si gana, Línea E reentrena con paridad D-753. Tema 3.
22. **F473: OBI de Hyperliquid para BTC** (ENSAMBLE; M). Normalizado por la tasa de mensajes; BTC como líder de las altcoins. La red está bloqueada en la nube.
23. **R151: concentración de entes** (ENSAMBLE; S). Sólo en vivo; rival `whale_burst_z`. Valor bajo: el espectro ya la midió como no predecible.
24. **F192: transfer entropy frente a hawkes_cross** (crítico; ENSAMBLE; M).
    - Offline sobre llegadas de trades; nulo por desplazamiento circular de bloques con BH; mejora de log-loss.
    - Usar `transfer_entropy_observada` (`te_binaria_observada` no existe).

Infraestructura en sombra que depende del sí del dueño (§5):

| Ids | Qué | Contrato o criterio | Coste | Tema |
|---|---|---|---|---|
| F253 | p_hit_sl_first, r15_vetaria, n_eff_grupo… → ValidatedOrder → EntryLearningEvidence → `decision_snapshots` | Brier frente a RR/(1+RR), N ≥ 200 por mitad | M | 1 |
| F252 | lh04_* (7 claves) por la misma vía | Riesgo al stop realizado frente al objetivo LH-04 | S | 1 |
| F276, F401 | StateValidator como invariante del replay (`run_booktick_replay_with_observer`) | Tolerancia ulp·Σ\|flujos\|·n_ops; falta la dev-dependency (crítico) | S | 3 |
| F235 | Conciliación con `GET /fapi/v1/openOrders` | Primero registra huérfanos; después cancela los propios | S | 2 |
| F392, F308 (+R25, R26, crítico) | Zona raw v2: frames WS crudos, FNV, rotación atómica; MAGIC/VERSION entran en v2 | Reproduce con el mismo parser vivo | M | 2 |
| F93 | Bybit/OKX por venue, con la hora del evento (World Bank se retira) | R² OOS, N ≥ 2000; slot inválido si cae el WS | M | 2 |
| F72, R173 | Telegram `send_alert` para un conjunto cerrado de eventos urgentes | Estrangulado como el log | S | 1 |

## 4. Plan de retiro por crate

Protocolo de cada lote:

- Un commit atómico `RET-n`. `git status` y `git add` sólo de los archivos propios.
- Volver a buscar cada ancla antes de editar.
- Regenerar y commitear `Cargo.lock`, porque la CI usa `--locked`.
- `cargo check --target x86_64-pc-windows-gnu --workspace --all-targets`.

Riesgos de todo el plan:

- `include_str!`: corpus de veto_registry.rs:549-565, lectores de executor.rs, cf4_rasgo_ml_congelado.rs → stateful_engine.rs y symbol_manager.rs:347.
- Tests con `#[path]`, bins autodescubiertos y `replay-contracts.yml:89/:121`.
- R62 crea la primera sección `[dev-dependencies]`.

| Lote | Crate | Ids | Qué se borra (y qué queda) | Compilación y orden |
|---|---|---|---|---|
| 1 | flight-recorder | F2–F6 | Crate entero (members:15) | Nadie depende de él |
| 2 | strategy-core | F80, F470, R78–R82 | polars, smartcore, data-pipeline, telemetry-server y rand | `-p strategy-core --all-targets`; evolution Cargo:20 es de AGY |
| 3 | quantum-arena | F314–F320, F328, R148, R155, R160, R202 | 7 módulos isla, TickSource, reqwest y tokio | **(crítico)** congelar el gen 37 tras F-A2; borrar universe_selection_contract.rs |
| 4 | god-engine-core | F152, F383, R62–R64 | telemetry-engine y winapi; data-pipeline pasa a dev | Primera `[dev-dependencies]` |
| 5 | signal-engine | F63, F153 | dark-alpha-engine y telemetry-engine | Sólo el manifiesto |
| 6 | telemetry-engine | F149–F151, F379–F382, R83, R171 | TradeStats, TensorRecord y DnsOptimizer | **(crítico)** editar también atomic_telemetry.rs (serde); quitar crossbeam, conservar crossbeam-queue |
| 7 | audit-engine (bin) | F281, F406, F420, R47, R48 | main.rs y sus dependencias | Antes de los lotes 8 y 9 |
| 8 | graph-4d | F139–F148, F343–F351, R65 | Crate entero y `pub use ::graph_4d` | Después del lote 7 |
| 9 | phase-runner | F81–F89, F353–F359, F361, R188, R201 | Crate entero; uso descartado en multi_coin_simulator:505 | Después del lote 7; `[[bin]]` con `--all-targets` |
| 10 | backtest-engine | F41, F248, R49–R51 (+F42, R2, R3, K24) | csv, chrono, lazy_static y execution-engine | **(crítico)** F42, R2, R3 y K24 esperan a F9-4 de Sol; las dependencias pueden ir antes |
| 11 | data-ingest | F215–F221, F370–F376, R91, R92 | TokenBucket, ZeroCopyReader, Polars, TensorParser, selector, WorldBank (quedan cola_ws y liquidation) | Antes del lote 12; con F377 (mismo test) |
| 12 | data-pipeline | F94, F96–F106, F108–F110, R5, R6, R53, R54, R93–R101 | 13 módulos muertos, HistoricalLoader y campos de OmniState | **(crítico)** sin spot_feed.rs:129-137 no compila; decidir spot_futures_arb_spread |
| 13 | raíz src/features | F13, F197, F363, F479, R17–R20, R179–R183, R196, R197 | `src/features/` y `pub mod features` | **(crítico)** antes, portar FIX #1454 con test RED→GREEN (R194, R195, R198) |
| 14 | raíz | F79, F474 | dashboard.rs, su bin y crossterm | Junto a R29 |
| 15 | raíz | F471, F476, F478, F484 | src/risk, src/quantum_arena.rs, NanoForest y 5 métodos de EnvManager | 0 importadores |
| 16 | raíz | F477, R185, R199 | src/trailing.rs | Regla 2: comparar antes con trailing.rs del núcleo |
| 17 | raíz y signal | F177, F196, F475 | multi_asset_orchestrator.rs y re-export de StatArbEngine | Test `#[path]`; aviso a Línea G |
| 18 | graph-architecture | F10, F14, F15, F17, F366–F368, R67 | Bin, index.html, quote, tokio y warp | Depende de la respuesta a F12 (tema 4) |
| 19 | storage-engine | F302–F304, F306, F307, F313, F387–F391, F399, R76, R77, R115, R166 | lakehouse, history_store, ledger, temporal_store y `set_lakehouse` | 2 commits (núcleo y storage); con F384 |
| 20 | metacortex | F284, F312, F398, F438, R142 | 3 funciones de online_learning | Quitar storage-engine del manifiesto |
| 21 | telemetry-server | F64–F71, R172 | 8 buses y loggers muertos, más el campo flight_recorder del núcleo | Antes del lote 22; con F7 y el lote 1 |
| 22 | os-guardian | F19–F23, R73, R145 | crash_dump, zero_latency_telemetry, pin/lock e Uuid de init_guardian | Sólo Windows |
| 23 | telemetry-server | F73, F74, F76–F78 | 3 eventos, 2 rutas y update_financials | Cambia el esquema forense: avisar |
| 24 | god-engine-core | R117–R125 | 9 funciones de math_kernels | Riesgo nulo |
| 25 | god-engine-core | F181, R127, R128 | 2 métodos y el segundo lead_lag | No tocar la cadena que exige cf4 :136 |
| 26 | god-engine-core | R116 | revert_ghost_position (×2) | Volver a buscar en los `include_str!` |
| 27 | núcleo, arena y feature | F62, F187, F338, K12 | Claves de registro sin lector | **K12 cambia conducta**: arranque frío 0,50 → 0,1; test |
| 28 | feature-engine | F190, F198, R110–R113 | QuantumTensorStore, is_resonant, load_from_disk, d0_f_alpha y Navier | Migrar 2 contratos Navier |
| 29 | strategy-core | F200, F201, F203, F453–F455, F468, R168, R169 | StrategyOrchestrator, evaluate_scoped y 2 campos de SignalIntent | Volver a buscar `drift:` en todo el workspace |
| 30 | execution-engine | F225, F226, F228–F230, F247, R15, R107 | router, shadow, hot_swap, multiplexer y socket | **(redacción)** antes, migrar la aserción :45-46 de router_latency_diffusion_contract.rs, que usa F263 |
| 31 | execution-engine | F233, F236–F239, F243–F246, R103, R104, R106 | execute_order, build_payload y otra API muerta | Reescribir ioc_fill_contract.rs:164-185; genes 108, 133 y 134 |
| 32 | quantum-arena | F321, F331, F340, R146, R153, R154 | Feedback epigenético y tensor_arena | Cambia el layout de GlobalArena |
| 33 | quantum-arena | F334, F336, F337, R147, R157–R159 | 6 métodos de TemporalSpectrum y API de EProceso | Línea CL edita temporal_spectrum |
| 34 | quantum-arena | F325, F326, F329, F339, R156 | position, protection_health y spec() | **F325 cambia conducta** (minNotional tras redondeo) |
| 35 | dark-alpha-engine | F44–F46, F51, F52, F55, F56, F59, F61, R89 | Métodos y capas sin llamador | Recortar 3 contratos |
| 36 | metacortex | F287, F441 | epigenoma_store.rs | El código que emite CazadorConstantes apuntaría a una función inexistente |
| 37 | metacortex | F293, F446 | reminiscence_and_adn.rs | Corregir el log de god_engine.rs:2391 |
| 38 | metacortex y núcleo | F282, F436 | ShadowGraphAuditor | 2 commits |
| 39 | audit-engine | F273, F274, F277, F278, F280, F402, F403, F405, F411, F414, F415, F417, R84, R86–R88 | Behavioral, Cybernetic, StateValidator y send_* | **(crítico)** F270 y F410 pasan al bloque F407; F273 se secuencia con F418 |
| 40 | audit, núcleo y config | F409 (F271 sale) | TribunalThresholds y su sincronización | **(crítico)** va en el bloque F407 + F408 + F409 + F410 + F270, tras el ítem 1 de §3 |
| 41 | risk-engine | F154, F249, F251, F269, R23, R24, R164 | Alias por horizonte y REJ_* sin uso | Marcar Retirado conservando los índices 7, 9 y 14; congelar btc/eth_volatility_multiplier |
| 42 | risk-engine | F254–F256, F259, F260, R165 | calculate_dynamic_allocation, RegimeDetector, 2 funciones de guard y cargar_pico | **(redacción)** `guard::enforce_minimum_notional` está vivo (lib.rs:1674): no borrar guard.rs entero; congelar los genes 115-121 |
| 43 | storage-engine | F309–F311, F394–F397, R167 | Constantes, wrapper, derive y cargas de EvolutionLedger | online_daemon.rs:1057 es de AGY; con F305 |
| 44 | núcleo y arena | F327, R114 | SystemBootloader y state_continuity | **(redacción)** ROSTER y SystemDiagnostics viven en bootloader.rs (god_engine, copulas_manifest y cf5_roster): borrar sólo el struct; **R114 cambia conducta** en testnet |
| 45 | raíz host | F472, F480, F481, R30, R176 | parse_binance_depth, evolve_symbols_daemon, 4 métodos de OrderBook y fred_daily | Test `include_str!` de symbol_manager |
| 46 | feature-engine | F191 | ShannonEntropyEngine | Regla 2: comparar con math_kernels; portar el test |
| 47 | dark-alpha-engine | F47, F50, F58 | transform, fit y AtomicF64 | Regla 2 contra el entrenador; toca el estado PPO |
| 48 | risk-engine | F261 | ExposureBudget | Comparar con el nocional LH-04 |
| 49 | omniscient-registry | F431 | get_scoped_strict(_or) | Conservar la sonda NaN/None |
| 50 | quantum-arena | F324 | margin_multiplier legacy | **Cambia conducta** (min_margin_floor); golden y T-1 |
| 51 | telemetry-engine | F155, F386 | BINARY_QUEUE | Macros `$crate`: compilar el workspace |
| 52 | execution-engine | F232 | WsExecutor (stub) | Después del lote 31 |
| 53 | risk-engine | F268 | 22 claves sin escritor | DECISION_DUENO, tema 1 |
| 54 | data-pipeline | F90–F92, F111, R12, R13 (R25, R26 → F392) | ws_client, parser y validation | DECISION_DUENO, tema 2; portar antes la aduana |
| 55 | quantum-arena | F332, F333 | 3 métodos de SuperGenotype | Esperar a F-A2 de Fable (genome.rs) |

Orden obligatorio:

- 7 antes de 8 y 9; 11 antes de 12; 21 antes de 22; 30 antes de 31, y 31 antes de 52; 39 antes de 40.
- F205/F456 antes de F206/F457/F458.
- F7 con el lote 1 y F71; F360 y F352 con los lotes 7 y 8; R29 con el lote 14.
- F305 con F394; F384 con F304/F388.

**Retiros fuera de los lotes** (motores, revisar y dueño; una línea por crate):

- **audit-engine**: F279/F404 (resilience.rs) y F408 (audit_biases). F407 va en el bloque del lote 40 (crítico).
- **dark-alpha**: F48/R90: Oja y predict_with_plasticity; F57: clip PPO, porque el gen 100 es inerte; F53: UNIFICAR la carga de arranque (tema 3).
- **data-ingest y data-pipeline**: F377: 7 campos de liquidación (tema 1); F107: feature_vm en un commit aparte del lote 12 (crítico).
- **evolution**: F125, NSGA (crítico).
- **execution**: F227 simulator.rs y F234 iceberg (tema 2); F241, el trío FX; R105, F231 y R108/R109, en el mismo commit (crítico).
- **feature-engine**: F180/R129/R130: UNIFICAR, MLP frente a bosque en Línea E (crítico); F185 (espectro f-alpha) y F188 (correlation.rs); F199: builders; update_elapsed se queda.
- **núcleo**: R126 fused_compute_step.
- **graph-4d y architecture**: F352, F360, F11, F362, F12 y F369 (tema 4).
- **metacortex**: F286/F439 Prospect (crítico; tema 4) y F295 (UNIFICAR las cotas del consejo).
- **quantum-arena**: F335 (crítico: editar también espectral_multiactivo_regime_adaptation_contract.rs).
- **risk-engine**: F264 (update_continuous_sde; freno_estrategia.rs:62 tiene `expect`) y F265 (ruin_probability).
- **raíz**: F482 (UNIFICAR, refresco horario de SymbolSpec), F483, R29 y R194/R195/R198 (vía lote 13).
- **signal-engine**: F168 (β = ½ entra en F178) y F170 (envolventes de Navier).
- **storage**: F305 EvolutionLedger y F393 (productor raw a god_engine).
- **strategy-core**: F205/F456 y F206/F457/F458: UNIFICAR; Línea G decide la absorción OU (crítico); F212: sólo el impl (crítico); F465: parámetros de Kähler.
- **telemetry**: F384 PositionLedger y F385, que congela el gen 37 (crítico).

Genes polizones que hay que congelar sin cambiar DIMENSION 144 (precedente LG-18/F1-026), tras F-A2: 37, 34, 108, 115–121, 133/134, 135/136 y btc/eth_volatility_multiplier. El gen 100 es inerte por cota. Retirarlos del vector lo decide el consejo.
Cambios de conducta que exigen test y golden/T-1: K12, F324, R114, F220, F325, el paso 1 de F43, el ítem 1 de §3 (con F407) y la firma de F463.

## 5. Decisiones del dueño

Cada tema se presenta como una sola tarjeta de decisión. En todos los temas, la alternativa a una opción es pedir una excepción concreta: cada ítem lleva en el TSV su pregunta «si discrepas».

**Tema 1. Telemetría y registro sin lector ni escritor.**
Ítems: F252, F253, F268 (22 claves), F7, F305, F384, F241, F377, F72 y R173.

- Pregunta: ¿unificamos la telemetría de decisión en un único vehículo con lector, y retiramos los almacenes que nadie lee?
- Opciones:
  - (a) Vehículo único `ValidatedOrder → EntryLearningEvidence → decision_snapshots` en la BD forense, puntuado contra `etiqueta_barrera`. Telegram `send_alert` sólo para eventos urgentes. Se retiran los dos flight recorder, EvolutionLedger, PositionLedger, el trío FX y los 7 campos de liquidación. F268 se resuelve con una tabla por clave: constante, gen o medida.
  - (b) Retirar también F252/F253 (los tests leerían la función) y prescindir de Telegram.
  - (c) Conservar almacenes paralelos (caja negra F7, SQL de promociones F305, SQLite de posiciones F384, comisiones en BNB F241).
- **Recomendada: (a).** Es una sola fuente con lector, cumple la regla 1 y da justo la evidencia que falta para calibrar p_hit_sl_first y R-15. Coste S–M.

**Tema 2. Ejecución y microestructura sin camino operativo hoy.**
Ítems: F214/F463, F234, F235, F392/F308/F393/R25/R26, el lote 54 (F90–F92, F111, R12, R13) y F93.

- Pregunta: ¿IOC como única ruta viva, con las demás medidas en sombra, o cerramos del todo lo que no es IOC?
- Opciones:
  - (a) IOC (B3.29), y además: maker como contrafactual (llenado y markout); iceberg nativo retirado, con los genes 135/136 congelados; conciliación de openOrders que primero registra y después cancela los huérfanos propios; captura raw v2 opt-in (≈ 1–5 GB/día, inferido), con el productor en god_engine y BinanceStreamer retirado; Bybit/OKX en sombra, y World Bank retirado.
  - (b) Retirar maker, zona raw, OmniDataHub y conciliación.
- **Recomendada: (a).** La sombra no cambia ninguna orden y mantiene abierta la ruta pasiva, la única que podría bajar la comisión taker, que es la fricción dominante con 13 USD. Las órdenes en reposo no se pueden descartar (manuales o una segunda instancia).

**Tema 3. Integridad del modelo, del veto y de la contabilidad.**
Ítems: F407 con el ítem 1 de §3, F53, F182, F199, F276/F401 y el coste de CPU de F418.

- Pregunta: ¿el Auditor Interno veta sólo por discrepancia entre la contabilidad interna y los fills del exchange, dejando el drawdown a SeniorRiesgo?
- Opciones:
  - (a) Sí, y además: el tribunal se retira; una sola ruta de carga de DarkAlpha, congelada; el rasgo [6] nuevo frente al viejo, en sombra; se retiran los builders que sólo usan los tests, salvo `update_elapsed`; StateValidator pasa a ser un invariante del replay.
  - (b) El drawdown también en el auditor, con la misma definición; normalización online a propósito; slot 6 congelado a 0; StateValidator retirado.
- **Recomendada: (a).** Elimina el veto tautológico y la doble definición de drawdown, y sólo reentrena si la sombra gana. Arreglar el veto absorbente es urgente con cualquier opción.

**Tema 4. Teorías y herramientas sin datos, sin aptitud o duplicadas.**
Ítems: F286/F439, F107, F125, F465, F482/F483/F385, R29, F352/F360, F11/F362/F12/F369 y los genes polizones.

- Pregunta: ¿retiramos las teorías y herramientas sin dato ni lector, y sólo se reconstruye, como trabajo nuevo con contrato, lo que el dueño pida?
- Opciones:
  - (a) Retirar. El universo es el ROSTER fijo, con refresco horario de SymbolSpec, y los polizones se congelan sin cambiar DIMENSION.
  - (b) Reconstruir alguna: Prospect con funding o top traders, en sombra; búsqueda de rasgos offline (Línea E, L); NSGA-II sin WR (AGY); +4 genes Kähler (cambia DIMENSION); rotación de universo al arrancar; dashboard o visor reparados, `/api/profiler`.
- **Recomendada: (a).** Ninguna tiene un dato independiente que la alimente ni un contrato que la puntúe; rehacerla desde cero cuesta lo mismo que repararla.

## 6. Zonas ajenas: qué se entrega y a quién

53 filas en 17 ítems. Primero, los ítems ZONA_AJENA; después, los avisos ligados a ítems propios.

**AGY (8 ítems, 20 filas)**

- F29, F34, F37: FFI `ffi_run_*` y NetworkJitterSimulator. Si se retiran, conservar los archivos del corpus de veto_registry y actualizar la doc de tp_sl.rs:217-247.
- F39 (con Línea E): BinTick tiene 11 copias repr(C); unificarlo en una crate hoja.
- K10, F156, F158, F161, F163, F171 (con Línea G): alias de consenso y campos de TensorDecision sin lector tras el lote 30; `min_confidence` no tiene escritor.
- F54, F121, F123, F127, F132, R102: AntiBiasGovernor (comparar deflated_sharpe y validate_oos con el walk-forward vivo), EntropyFitness, polars_evolver. Phenotype va al consejo (#605).
- F120 CmaEsOptimizer: probarlo como ES isótropo con CSA en el arnés de F122.
- F122 blx_alpha y cauchy_mutate: brazo en sombra en [0,1] por banda, sin sumar al DSR vivo.
- F488: evolution.rs perturba las vistas sin `rebuild`; perturbar las anclas o retirar.
- F157 hedge_eta (con Línea G): sombra frente a SkillMotores con pérdida por moneda, o retirarlo.
- Avisos: F80: telemetry-server en evolution/Cargo.toml sin uso; F57: ppo_clip_eps es inerte; gen 34 congelado; F392, F276 y el lote 43 (online_daemon.rs:1057); F305, con Sol.

**Línea G (1 ítem, 10 filas)**

- F202, F208, F209, F211, F459–F461, F466, F469, R170 (votantes): retirar TradeHorizon y `horizon()` en un commit coordinado sobre 34 archivos. Además, statarb_beta y la API de Johansen, StatArb y Yang-Mills, comparando antes de borrar.
- Avisos: la absorción OU de F206/F457 la decide Línea G: LibroSombra o retirar; F168, F170, F157; la visibilidad de `AcumIc`; el consumidor de F93; lote 17 (F475, F177, F196).

**GLM/Qoder (2 ítems, 7 filas)**

- F24–F27, R21, R22: sensores de os-guardian (PMU, eBPF, anomalías) sin consumidor. Reconexión certificada o retirada, con su Cargo.toml.
- F179 `last_hawkes_ratio`: columna en SkillSombra frente a ema_ofi, o retirar.
- Avisos: F43: `hidden_liquidity_pressure` es una razón de muros; F212/F464: J_i ≈ −tanh(0,5·tanh OFI) y la fuga del registro global; F192: defectos de hawkes_cross; `contagion_publisher.rs:38` tiene `unwrap`.

**Línea E (2 ítems, 5 filas)**

- F485, F486, R31, R32: reconectar `verify_model_lineage` en la carga de modelos (FMT-037/190), no retirarla.
- F186 `analyze_event_spectrum_v2`: sombra frente al legacy; si gana, versionar el tensor y reentrenar con paridad D-753.
- Avisos: F180, MLP frente a bosque; reentreno de F182; F47/F50/F58 frente al entrenador; F53: auto_trainer debe guardar el scaler; F107, si se pide; el rasgo de F178, si gana.

**Codex (2 ítems, 4 filas)**

- F257, F258, F266, R161: random_matrix, vetos legados de CorrelationGuard y copulas_store. Retirar o poner en sombra; si se retiran vetos, marcarlos Retirado sin renumerar.
- Avisos: F188: ρ̄ queda sin consumidor; F253: `lxxii_lambda_grupo`, en correlation_guard.rs:719; F192: `net_role`.

**Sol (1 ítem, 4 filas)**

- F42, R2, R3, K24: esperar a F9-4 y después aplicar el lote 10 (UNIFICAR K24 con `arena.config.tech_threshold`).
- Avisos: F305: online_daemon.rs es su siguiente átomo; F222: la fila de lib.rs:1980 está obsoleta; mirar cerca de :1920.

**Fable y Línea H (1 ítem, 3 filas)**

- F31–F33, vectorized (modelo L2 de deslizamiento): compararlo con la fricción viva antes de borrar; el re-export de lib.rs:12 es de AGY.
- Avisos: genome.rs está en edición (F-A2). Esperan F234, F332/F333 y los genes 37, 108, 34 y 115–121; F252 (LH-04).

## 7. Islas: veredicto por crate

| Crate | Veredicto | Coste | Vivo | Nota |
|---|---|---|---|---|
| flight-recorder | RETIRAR | S | 0 % (258 líneas) | Nadie lo enlaza; duplica el de telemetry-server |
| phase-runner | RETIRAR | S | muerto (214) | Sólo bin muerto y simulador que descarta el resultado |
| graph-4d | RETIRAR | S | muerto (174) | Duplica graph-architecture |
| audit-engine | UNIFICAR | M | 65 % (1718/2654) | Viven DriftAuditor, TrajectoryAuditor y auditor_interno |
| data-ingest | UNIFICAR | S | ≈ 27 % (≈ 320/1183) | Sólo cola_ws y liquidation |
| storage-engine | UNIFICAR | M | 31 % (979/3142) | mmap_bus y evolution_ledger |
| data-pipeline | UNIFICAR | L | 1909 vivas + 672 herramienta / 6610 | 17 módulos sin consumidor |
| telemetry-server | UNIFICAR | M | ≈ 70 % (3436/4932) | 8 módulos sin consumidor |
| metacortex-engine | UNIFICAR | M | ≈ 73 % (4697/6396) | |
| evolution-engine | UNIFICAR | M | 76 % (5502/7214) | |
| telemetry-engine | UNIFICAR | S | ≈ 50 % (≈ 190 muertas de 375) | |
| raíz (trader-gemini-v5) | UNIFICAR | M | lib 1433/3613 | 11 dependencias sin uso |
| strategy-core | CONSERVAR | S | 82 % | |
| quantum-arena | CONSERVAR | S | 95 % | |
| execution-engine | CONSERVAR | M | 85 % (1866 muertas/12393) | |
| feature-engine | CONSERVAR | M | 85 % | |
| god-engine-core | CONSERVAR | S | 97,8 % | |
| backtest-engine | CONSERVAR | M | 771 muertas/6352 | Herramienta de certificación |
| os-guardian | CONSERVAR | S | 82 % | |
| graph-architecture | CONSERVAR | S | herramienta | Sólo si el dueño rechaza F12 (tema 4); si no, RETIRAR |
| omniscient-registry | CONSERVAR | S | vivo | |
| dark-alpha-engine | CONSERVAR | S | sin módulos muertos | |
| risk-engine | CONSERVAR | S | vivo | guard sigue vivo (confirma la corrección del lote 42) |
| signal-engine | CONSERVAR | S | vivo | |

## 8. Lo que queda abierto o sin verificar

- El crítico recibió `retiro_plan` truncado en el lote 17 y no recibió `islas`. El `sintesis.json` actual trae los 67 ítems y los 24, así que **el crítico no revisó los lotes 18–55 ni las islas**. Esta redacción encontró en ellos dos rupturas de compilación (lotes 42 y 44) y una aserción que F263 necesita (lote 30); nadie más las ha revisado.
- Las anclas de línea se verificaron en 5d98d5d5. Ω155 (c826b1f7) cambió seis archivos fuente que citan muchos ítems.
- Inferido, sin medir: que T-1 no cambia con los retiros (último conocido: 18/144); la paridad tras portar FIX #1454; si el fixture de T-1 dispara el veto del auditor; el serde de TribunalThresholds; el γ₁ < 0 del mid; el volumen de la captura raw; la región de Bybit/OKX; si los modelos JSON del operador llevan scaler.
- Modelos: hay `_MOTOR` de ATOM, BNB y NEAR, pero no hay `BTCUSDT_MOTOR` (lo verificó el crítico en wt-ci).
- Símbolos fuera del censo, con un peso de unas 1.900 líneas: en el núcleo: `slippage_predictor`, `order_flow_aggregator`, `latency_accelerator` y los bins `test_ml`/`nano_benchmark` (564 líneas); en metacortex: `compiler_sandbox`, `evolutionary_templates`, `fases_autonomous`, `hot_swap_controller` y `trigger_wavelet_mutation` (≈ 1232); en evolution: `meta.rs` (98).
- Lote 36: el flujo de tokens de CazadorConstantes apuntaría a una función inexistente. Los tests que sólo comparan texto siguen verdes.
- Violaciones de la regla 6 vistas de paso, no corregidas: `freno_estrategia.rs:62`, `KalmanFilter1D::new`, `ewma.rs:19/:40` y `math_kernels.rs:1030`, con `expect`; `contagion_publisher.rs:38` y `raw_zone::from_bytes`, con `unwrap`.
- Genes sin ningún lector ya hoy, fuera del alcance del censo: 36, 42, 43, 49, 50, 64, 65, 70–74, 77–79, 84, 85, 90–94, 96, 97, 102–105, 111 y 122–128. Cuentan como no sensibles en T-1.
- Nada de este censo autoriza a operar ni valida la meta de crecimiento. Es un inventario con un plan, sin ejecutar.

## 9. Ejecución (se actualiza con cada tanda)

El censo es una foto del 2026-10-18. Esta sección registra lo ejecutado y
lo que cambió desde entonces; las filas del TSV no se reescriben.

**Fusionado** (un commit por lote, sin T-1 salvo CL-159, la tanda 7 y la 11; sin
cambio de conducta salvo RET-10b, RET-41, RET-51a, CL-154, RET-13a (CL-157,
inerte con entradas validadas) y RET-27b (CL-158, inalcanzable), que cambian
diagnóstico o conducta inalcanzable; RET-44a (CL-156), que cambia la
conducta viva con USE_TESTNET=true, la demo; RET-51b (CL-160), que cambia
el orden y el vaciado del log del host; CL-159, que cambia el tamaño de
las órdenes de LH-04 (T-1 14/144, la cuenta de main, fila N13), y la
tanda 7 (fila N13): CL-161 y CL-163 cambian el tamaño sólo con entradas
corruptas, con `lh04_muestras_min` por encima de n o con un gen 33 por
debajo de f, CL-162 cambia la mutación y CL-164 es telemetría, y la
tanda 8: CL-165 cambia la conducta sólo con entradas corruptas y CL-166 es
telemetría, y la tanda 9: CL-167 cambia la conducta sólo con un bias
activo corrupto y f = 0, CL-168 es el formato del informe y CL-169 la
lectura del libro de la sombra y de su informe (telemetría; sólo cambia
con un acumulado no finito), y la tanda 10: CL-170 y CL-171 completan esa
lectura (telemetría, sólo con un contador o una marca de ruina corruptos), y
la tanda 11 (fila N12): CL-172 y CL-173 cambian la conducta sólo junto a la
frontera del tope de riesgo por evento (T-1 14/144, la misma lista que
main), y la tanda 12 (filas N2, N4, N9, N10 y N15): RET-N2 y RET-N4 retiran
código sin llamadores, CL-174 y CL-175 dan el mismo voto bit a bit en todo
camino alcanzable y CL-176 es el log del host; como
dice su lote):

| PR | Lotes |
|---|---|
| #80 | 24 |
| #81 | 2, 5, 15, 28a |
| #120 | 4, 25, 26, 20 |
| #122 | 3a (F317–F320, F328, R160), 37, 49, 17 (F177, F475), 29, 43a (F309 envoltorio, F395, R167, F310/F396: seis constantes que sólo usaban los tests, que pasan a locales del módulo de tests, y SUBSYSTEM_OS_MEM, sin ningún lector; F311, F397), 46, 16 |
| #123 | 6, 7 (resto del manifiesto), 8 (sin F352), 10 (más K24 como RET-10b, sólo diagnóstico), 11, 12a |
| #124 | 21a (F64–F70, R172), 22a (F19–F23, R145; sin R73), 23a (F76, F78), 30 (F226, R15, F228–F230, R107, parte de F247), 32, 33 (sin capital, umbral_familia ni n), 34 (F325 parcial: sale roundtrip_fee_cost, queda validate_order; sin open_positions_count), 35 (sin F44; más RET-35c, tres avisos de clippy en pruebas) |
| #126 | 47 (parcial: 47a F47 y 47b F58, con CL-154), 38 (commit 2: F282, F436), 39 (39a: F277, F403, F278, F402, F280, F405, R86, R88, F415 y parte de F274/F414; 39b: F411), 41 (F251 sin margen_insuf), 42a (F254/R165, F260), 51a (F386 y la parte neutra de F155), 19 (19a: R115; 19b: F302, F303, F306, F307, F313, F387, F389–F391, F399, R166, R76, R77), 27a (F62, F187 y cuatro claves de F338), 31 (sin F244, F233, get_status ni cleanup_stale_orders), 18a (la dependencia quote de graph-architecture: F15, R67 y su parte de F368) |
| #130 | 50 (residuo: 50a legacy_view, LegacyRegimeView y su re-export; 50b, el T12 de cf3_fuente_contract como guarda de F1-054), 44 (44a CL-156: G021; 44b: R114, F327), 13 (13a CL-157: FIX #1454 portado al OmniStrategyEngine vivo; 13b: src/features y los dos #[path] de FMT-140; sin F315, F385, R105 ni F184); más RV-12a, RV-plan y RV-docs (revisión de las tandas 2 a 4) |
| #138 | 27b (K12: RET-27b, CL-158), 48 (CL-159: LH-04 con la fricción al stop; RET-48: ExposureBudget, exposure_budget(_spectral) y project_quantity), 34 (resto de F325: RET-34b, validate_order; RET-34c, la OPEN de la fila N8), 51 (51b CL-160: telemetry! escribe en LOG_QUEUE; 51c: BINARY_QUEUE, StackBuffer, init/pop_telemetry y crossbeam-queue de telemetry-engine); más RV5-T12 y RV5-docs (revisión de la tanda 5) |
| #139 | N13, el plan del panel para los genes de Kelly (CL-161: clases de tamaño explícitas; CL-163: el gen 33 como techo κ de la f de LH-04; CL-162: genes 10, 11, 20 y 32 congelados en la mutación; CL-164: sombra del Kelly legado frente a LH-04); más RV6-T y RV6-docs (revisión de la tanda 6) |
| #140 | N13, correcciones de la revisión de la tanda 7 (CL-165: LH-04 valida el bias crudo y las entradas antes de toda clase; CL-166: la sombra Kelly congela la regla arruinada, suma R y R² y el forense la imprime); más RV7-T y RV7-docs; abre N14 (banda del gen 33) |
| #142 | N13, correcciones de la revisión de la tanda 8 (CL-167: el bias activo se valida también con f = 0; CL-168: el informe publica null como log-riqueza de una regla arruinada); más RV8-T y RV8-docs, y las de su propia revisión antes de fusionar (CL-169: el libro de la sombra lee sin respaldo y un acumulado no finito sale null; RV9-T, RV9-docs) |
| #143 | N13, precisiones de la revisión de las correcciones del #142 (CL-170: el contador y las marcas de ruina de la sombra también sin respaldo; RV10-T, RV10-docs), y la de su propia revisión antes de fusionar (CL-171: el libro no repara un contador corrupto; RV11-docs) |
| #144 | N12, la fricción en la pérdida al stop (CL-172: la viabilidad de la orden mínima; CL-173: el tope de ruina F1-092; RV12-T, RV12-docs); f_min_ejecutable de la envolvente queda para el dueño |
| #145 | N4 (RET-N4: Position::entry_tau_ms(), sin llamadores), N2 (RET-N2: ewma_price_at, scale_gain_at y volatility_at, tras la regla 2), N9 (CL-174: sin VPIN medido la sombra TrendRunner se abstiene), N10 (CL-175: la presión adversarial de Nash es el cvpin medido), N15 (CL-176: la escalada desnuda reporta el fallo de la purga de brackets); más RV13-docs; abre N10b, N15, N16 y N17 |
| #146 | N5, la cuota del exchange para las salidas (CL-177: uso por contador con su sello y sin bajar por una respuesta tardía; CL-178: las salidas sin consulta respetan el freno 429/418 y el barrido no purga dos veces; CL-179: las críticas pasan hasta el límite del exchange con las críticas en vuelo contadas); más N5-docs; abre N18 a N22; y sus dos revisiones antes de fusionar (CL-180: el máximo exacto de las cabeceras vigentes, con el sello futuro acotado por la siguiente cabecera; CL-181: un contador sin coste no frena por lo que está en vuelo; CL-182: la cabecera se registra con el cupo vivo; CL-183: en X-026 un re-snapshot sin respuesta no da la posición por plana; RV14-c, RV14-T, RV14-docs, RV15-c, RV15-docs) |
| #147 | N20, peticiones fuera de la cuota (CL-184: exchangeInfo dentro del freno; CL-185: un exchangeInfo sin filtros no borra la caché; CL-186: un símbolo ausente de un exchangeInfo completo no se vuelve a descargar; CL-187: modo hedge y hora dentro del freno, con la cabecera del modo registrada; CL-188: un freno por ejecutor compartido por los clones de su cliente; CL-189: la hora del NTP y el listenKey miran y arman el freno; CL-190: la cancelación de la pierna hermana mira el freno; CL-191: sin el último expect de producción del ejecutor); más N20-docs; abre N23 a N39; y sus dos revisiones antes de fusionar (CL-192: el freno no deja un símbolo en la caché negativa; CL-193: el 2xx del NTP y del listenKey reinicia la racha de 429; CL-194: su 429 ya registrado no cuenta dos veces; CL-195: la caché negativa decide por la descarga de esa llamada; RV16-T, RV17-T, RV16-docs) |

**Hecho por otros agentes**: lote 9 entero (Sol, d18b3735), lote 45 entero
(Codex RB, 906a9c6a), parte del 7 (Sol retiró el bin main.rs de
audit-engine en d18b3735), mitad del 38 (AGY, 7bdd3bb3) y la parte central
del 50 (AGY, 508f1597). AGY re-exportó net_multiplexer y le puso un test
de contrato (C1 de f4_001, PR #114); sigue sin lector de producción
(SOLO_TESTS).

**Reverificación del 2026-10-24 sobre main 6c485c05** (triaje por lote y
un escéptico por lote listo):

- Listos (con los subconjuntos que dice cada plan): 6, 7, 8, 10, 11, 12,
  18 (sólo la dependencia quote), 19, 21, 22, 23 (F76/F78), 27 (sin K12),
  30, 32, 33, 34, 35 (sin F44), 38 (commit 2), 39, 41, 42, 47 (parcial),
  51. Los lotes 1 y 8 obligan a editar el paso F1-095 de
  replay-contracts.yml (prueba flight-recorder y graph-4d).
- Lote 31: el escéptico lo refuta tal como estaba. F244
  (UserDataStreamer::dispatch) es la entrada pública por bytes del diseño
  F2 de Codex QRA (QRA_F2_CAPITAL_DISENO_2026-10-23.md:103) y AGY le añadió
  llamadas de test el 2026-10-23. Los otros 11 ítems pueden ir sin F244.
- Bloqueados por tema del dueño: 1 (tema 1), 14 (tema 4), 40 (tema 3),
  53 (F268, tema 1), 54 (ws_client, parser y validation, tema 2).
  Por orden: 36 (CazadorConstantes). Por zona: 55. Cambian conducta: 48
  (13 y 44 hechos en la tanda 5). Revisar: 52 (exige 30 y 31 antes).

**Correcciones al censo** que salieron al ejecutar:

- F431 (lote 49): la columna 23 dice «BORRA: las 2 funciones», pero
  get_scoped_strict se conserva: siete sitios de test (lf1, cf3) necesitan
  la lectura cruda con NaN. Sólo se retiró get_scoped_strict_or.
- Lote 16 (F477): el rótulo estaba invertido. El legado era el único que
  leía trail_f1/f2/runner como distancias por fase; el núcleo los descarta
  y usa literales 1,20/0,90/0,65/0,45. En vivo, trail_step y la ranura
  runner no tienen fenotipo (trail_act sí, por trail_max). Va a C-F5 DER /
  R8-A (G017) o a los genes polizones del tema 4.
- F208/F459 (StatArbEngine::update, zona de Línea G): tras RET-17 pasa de
  SIN_LLAMADOR a SOLO_TESTS.
- Lote 46 (F191): el gemelo no es «el vivo antes de D-713» (aquél usaba
  `5 + 100·r`); mide la concentración en nivel de un flujo acotado. Las
  cifras de la regla 2 están en el commit RET-46. Generadores de las filas
  4 y 5, que el commit no dice: fila 4 (OBI 0,8 ± 0,05) = Lcg(0x0B1),
  0,8 + 0,05·N(0,1) recortado a [−1, 1] (sin el recorte el vivo da
  0,9696082620, no 0,9696102199); fila 5 = Lcg(0x5A1), +1 si u < 0,5 y −1
  si no.

- Lote 8 (F352): el retiro del indexador de AST no depende del tema 4.
  Las dos opciones que se plantean al dueño (construir la topología 4D de
  liquidez como motor nuevo o corregir el documento) no usan este código.
  Siguen desfasados, hasta que responda, docs/INVENTORY.md:32,
  ARCHITECTURE.md:153 y los planes maestros del 2026-10-13 (:519, :572) y
  del 2026-10-14 (:203). La tarjeta del tema 4 debe decir que el crate ya
  no existe.
- Lote 10 (K24): la lectura del umbral técnico en SIGNAL-PATH era de una
  clave de registro que nadie escribe (imprimía siempre 0). Va como
  RET-10b, sólo diagnóstico, con test.
- Lote 11: se separa de F377 (tema 1). Compartían archivo de test, pero
  sus líneas no se solapan.
- Lote 11 (F220/F375): no se portó a dynamic_ranker la deduplicación
  (ConflictingDuplicate) ni el desempate por símbolo de DynamicSelector.
  El ranker vivo no deduplica (un símbolo repetido en el feed ocupa dos
  asientos) y desempata con partial_cmp, así que manda el orden del feed.
  Queda como átomo abierto con test, en dynamic_ranker.rs, coordinado con
  AGY (Ω215) y Sol (F1-091).
- Lote 12: se parte en 12a (hecho) y 12b. El 12b lleva F94 (spot_feed y
  los mocks de AGY y Sol), F105/R97 (HistoricalLoader, que ejercita el
  contrato cero_dato de Sol en CI) y F108/R94/R95 (api_client, único
  lector de producción de cero_dato::monto_de_json); F214/F463 siguen en
  el tema 2 y F107 en el tema 4.

- Lote 21: el tema 1 del triaje no aplica a sus ítems; sólo F71
  (flight_recorder y su campo del núcleo) queda retenido, con F7 y el
  lote 1. R172 duplica F68. F67 (MmapTelemetry) se retira sin portar su
  WR ponderado, que ya vive en audit-engine telemetry.rs:162-171; queda
  abierto que /api/state promedia el WR por moneda (cambio de conducta,
  con su test). Las pruebas de lib de telemetry-server bajan de 73 a 57.
- Lote 22: R73 (winapi) no entra; va a GLM con sus sensores (F24–F27,
  R21, R22). uuid: con os-guardian fuera, fast-rng pasa a execution-engine
  y omniscient-registry, los dos que llaman a Uuid::now_v7, para que el
  generador de clientOrderId no cambie. Sin verificar: que
  SetProcessAffinityMask de init_guardian no deshaga en Windows el pin
  del núcleo unificado del host (F21). os-guardian pasa de 15 a 11 tests.
- Lote 30: F247 queda partido. Sale sólo el default de
  cancel_all_symbol_orders; los de cancel_position_oco_orders y
  query_order los hereda SimulatedExecutor y van con F227 (tema 2). F225
  (router) se queda con AGY: el host no llama a route_order ni a
  route_trailing_stop, y con el router se quedan los lectores de
  TensorDecision (K10, F156…).
- Lote 32 (R154): el centinela event_time_ms == 0 de
  apply_spectral_epigenetic_feedback_with_time NO cae, contra lo que dice
  la fila: el núcleo aún puede pasar 0. Lo fija el test
  ret32_event_time_cero_usa_el_reloj_de_pared.
- Lote 33 (F336): capital() y umbral_familia() se quedan, pero no por un
  lector vivo: sólo los leen las API F1-078 de AGY, sin llamador de
  producción (fila N3). Las cinco pruebas de spectral_gradient_at se
  migraron a signal_at, que está vivo y no tenía certificada su forma por
  tramos.
- Lote 34: F325 queda a medias. Sale roundtrip_fee_cost; la unificación de
  validate_order cambia conducta (minNotional tras redondeo) y va en un
  lote propio con golden y T-1 (execution-engine executor.rs:1433/1512 y
  risk-engine capital_regime.rs:407). Anclas de R156 al día: los
  duplicados vivos de maker+taker son core lib.rs:7359,
  booktick_replay.rs:401 (AGY), active_universe.rs:138/:160 y
  symbol_ranker_engine.rs:119; ninguno se tocó.
  Resto de F325 (tanda 6): RET-34b retira validate_order y lleva sus 12
  pruebas a execution-engine/tests/order_quantity_contract.rs (13, contra
  BinanceFixedFilter::validate_and_quantize, fixed_filter.rs:340). La
  premisa del censo, «el ejecutor recalcula qty = volume_usd·L/precio y
  no revisa min_notional», describe a build_payload (executor.rs:1466),
  sin llamador de producción (F233): las rutas vivas reciben la cantidad
  del núcleo y sólo la bajan al paso, y el tick al alza ya lo cerró
  CL-98b (capital_regime.rs:468). Eran cuatro cuantizadores, no tres:
  faltaba el canónico; project_quantity salió con el lote 48. Queda un
  defecto más estrecho, la fila N8, con su OPEN (RET-34c). Lo que el
  «viejo frente a nuevo» de RET-34b no decía (revisión de la tanda 6, ya en
  la cabecera de order_quantity_contract.rs): el canónico redondea el
  precio al tick antes de juzgar minNotional (validate_order juzgaba al
  precio crudo), from_symbol_spec fija min_price = tick y techos de 1e7 a
  precio y cantidad, e is_valid_config rechaza pasos o mínimos sin decimal
  exacto de ≤ 10 cifras. Hoy no cambia conducta: from_symbol_spec no tiene
  llamador de producción.
- Lote 35: el único recorte de contrato es F44
  (hidden_liquidity_conformal_contract.rs:47), que pasa a RET-35b cuando
  Codex ROOT libere Hidden04. neural_evidence_contract.rs:144-157 sigue
  probando una ida y vuelta bincode que ya no usa ninguna ruta de
  producción (la producción guarda JSON).

- Lote 47: F47 queda cerrado en parte. Las dos ramas vivas del z de
  inferencia comparten z_acotado, pero `normalize` (el z de `fit`) sigue
  aparte hasta F50. F50 es regla 2 para Línea E: `fit` es SGD/BCE por
  muestra con Welford mutante (lib.rs:701-832) y los entrenadores usan
  Adam + Scaler (train_dark_alpha.rs, auto_trainer_daemon.rs); no son
  equivalentes por construcción.
- CL-154 (hallazgo de Línea G, linea_g/hallazgos.tsv:10): AtomicF64::new
  canoniza −0,0 y NaN como store y los CAS. Cierra el caso en todos los
  átomos de GlobalArena, no sólo en los pesos PPO de RET-47b.
- Lote 38: cerrado. El commit 1 (núcleo) lo hizo AGY en 7bdd3bb3
  (F1-109) y el 2 (MetacortexEngine) es RET-38.
- Lote 39: sigue abierto. R84 lo retiró GLM (9fb74315). Quedan R87 y
  TradeExecution (recibo §52 de GLM, sin emisor de cierre), ParityAlert y
  send_parity_alert (state_validator, tema 3) y F273/F417; «39 antes de
  40» todavía no se cumple.
- Lote 41: V-TECH-003 «otros» pasa a Retirado (índice 9 conservado). En
  V-LOGIC-009 la puerta EV, fuera de la sonda, rechaza sin p_ganar en vez
  de saltarse; hoy es inalcanzable. La prueba
  ret41_las_compuertas_retiradas_no_tienen_emisor impide que una entrada
  Retirada recupere un emisor sin cambiar el registro (choque posible con
  LG5 de Codex ROOT). F269 va al lote 41b.
- Lote 42: sólo 42a. F255 (fallback_discrete_veto), F256 (RegimeDetector)
  y F259 (check_drawdown_limit) siguen muertos, pero esperan la
  conformidad de AGY, que los declaró conservados en F1-119 y F1-085. La
  teoría de la caída flotante (G055) no se descarta: su vehículo es la
  equidad caja + Σ pnl_unrealized en evaluate_quantum_order.
- Lote 51: cerrado. 51a fue preparatorio y neutro. 51b (CL-160) no baja
  LOG_QUEUE a telemetry-engine, como pedía el TSV para evitar el ciclo:
  telemetry-engine guarda sólo un destino (un `fn(fmt::Arguments)`) y el
  host conecta `telemetry_server::macros::linea_de_telemetria`, que
  escribe en LOG_QUEUE como `telemetry_log!`. El código de LG-11, LG-19 y
  CL-144 no se mueve ni cambia. En el host, las líneas de telemetry!
  salen en el orden del hilo que las emite, el apagado las vacía y no se
  reservan los 33 MiB de BINARY_QUEUE, que un hilo sondeaba cada ≈1 ms
  cuando estaba vacía. Sin destino (herramientas, backtests, evolver) se
  descartan como antes, pero sin la cola de 8,25 MiB que nadie leía. 51c
  retira la cola, StackBuffer, init/pop_telemetry y crossbeam-queue.
  Cierra también G7-src06 de Línea G y f1_aristas_muertas.tsv:155 de
  Fable. Quedan para el dueño o Línea G si las herramientas deben imprimir
  esas líneas (cambia su salida) y un contador de descartes del log.
- Lote 19: partido. ledger.rs (F304/F388), su re-export y
  ownership_read_contract esperan a F384 (tema 1). Sale bytemuck, que el
  censo no nombraba (sólo lo usaba lakehouse.rs). Para la tarjeta del
  tema 1: el lakehouse SQLite (depth, telemetría y tensores) era un
  almacén que nadie leía; si el dueño conserva los almacenes, se rehace
  desde git.
- Lote 27: sólo 27a. K12 (nash_equilibrium_drift) cambia el voto Nash en
  arranque frío y va aparte con test y golden; la variante «literal 0,50
  sin clave» es equivalente bit a bit. Los re-exports de Wasserstein se
  quedan: la evidencia del censo era falsa (el núcleo usa
  WassersteinKantorovichEngine por esa ruta).
- K12 (lote 27b, RET-27b/CL-158): no cambia la conducta en ningún camino
  alcanzable, contra lo que decían la fila y el triaje del 2026-10-24. El
  0,50 sólo se leía en el primer evento de un núcleo nuevo: process_event
  llama siempre al dual al final (lib.rs:4160) y el dual escribe cvpin por
  moneda y global sin condición (:6183). En ese evento el espectro acaba
  de sembrarse (temporal_spectrum.rs:512-533 vuelve sin calcular
  momentum_z): x ≡ 0 y el voto Nash es nulo con cualquier presión. La
  caída al cvpin global de otra moneda (LC-F1-06) sólo actúa en el primer
  evento de cada moneda, que tampoco tiene desplazamientos. 0,1, 0,50 y la
  abstención dan el mismo voto bit a bit en vivo, en el replay y en el T-1.
  Se elige la abstención (espectro nulo): ninguna p ∈ [0; 0,9] expresa
  «adversario desconocido» (la escala 0 vota tanh(x₀) con cualquier p) y
  no queda constante. Dar peso 0 al motor que se abstiene es F3.1-01
  (Qoder: las abstenciones cuentan en Σw) y no entra.
  game_theory_adversarial_pressure sigue sin escritor (fila N10). Su 0,1
  del motor escalar (game_theoretic_nash.rs:203) es inalcanzable por lo
  mismo: el orquestador evalúa en lib.rs:7255, después de :6183; queda con
  Línea G (vetos.tsv:1236). Lo que sí actúa en el arranque es el VPIN sin
  volumen = 0 (math_kernels.rs:236-243; linea_g/lazos.tsv:55). Cierra
  R7-R3-E-2 y la parte de Nash de F3-A4 (BARRIDO_EXHAUSTIVO_FASES.md:3060,
  :396). La sombra TrendRunner tiene la misma clase de defecto (fila N9).
- Lote 31: F244 (dispatch) se queda como costura pública del contrato
  E2E del diseño F2 de Codex QRA; la fila pasa a CONSERVAR salvo que QRA y
  AGY acuerden otra entrada. F233 sigue con AGY (Ω203, Ω220) y el tema 2.
  get_status (lo lee el contrato del Átomo 2.6 de AGY) y
  cleanup_stale_orders (lo lee registry_open_diagnostics.rs:15, el
  diagnóstico 107 de GLM; el commit f736bf15 se lo atribuye a AGY)
  esperan a F235 (tema 2). Los genes 133
  y 134 (umbrales de cuota) siguen sin lector vivo.

- Revisión adversarial de las tandas 2 a 4 (un verificador por tanda;
  lo que se corrige va en la tanda 5):
  - RET-8: syn conserva extra-traits. La pide también
    crates/graph-architecture/Cargo.toml:7, así que retirar graph-4d no la
    desactiva; el mensaje de b1fefe8e lo dice al revés.
  - RET-12a: el comentario de macro_ml_features seguía diciendo que el
    poller vivo publica niveles FRED (DTWEXBGS, NASDAQCOM, VIXCLS); publica
    cierres de Yahoo (^VIX, ^GSPC, ^IXIC, DX-Y.NYB). Corregido en RV-12a.
  - RET-42a: el título dice «sin lector … ni en tests», que sólo vale para
    calculate_dynamic_allocation (R165). cargar_pico (F260) lo leían
    cf4_pico_reinicio y cf5c_pico_plano_y_venue, que pasan a leer_pico. El
    cuerpo llama polizón sin lector a «CoinMetrics.profit_factor»: es
    ScalpState.profit_factor (arena.coins[i].metrics) y lo lee el Kelly
    LH-04 en risk-engine lib.rs:1508-1519. No es polizón; los únicos
    polizones del lote son los campos portfolio_* de config.rs.
  - Las REGLAS de scripts/plan_linea_c.py nombraban 41 rutas inexistentes
    tras la tanda 4: 8 de antes de la tanda 1 (5 de lotes del censo: 15 y
    28a del #81, y el 9 y parte del 7 de Sol, d18b3735; 3 de otros agentes: macro_data.rs
    y market_context.rs por RB-R1, 566a9c8c, y vectorized.rs por Ω160,
    fcab8e1c), 9 de la tanda 1, 13 de los
    lotes 8 y 12a, 4 del 30 y 7 de los lotes 19, 38 y 39 (con
    history_store.rs). Salen todas en RV-plan y `asignacion` vuelve a
    código 0. Desde ahora, el lote que retira un archivo lo quita de
    REGLAS.
  - Refutados: que la deuda de V-TECH-003 prometa más de lo que prueba
    ret41 (la forma numérica `rej(9)` la veta
    test_zero_raw_magic_number_rejections_in_risk_engine_source y `rej` sólo
    se llama desde lib.rs), y que el recuento de os-guardian del paso F1-095
    se olvidara (ya se avisó a Sol en la tanda 3).

- Lote 50: la fila 199 del plan («cambia conducta (min_margin_floor)»)
  ya no aplica: el recorte en línea de free_cap lo quitó AGY en F1-054
  (508f1597) y este lote no lo valida. F3.3-10 queda cerrado: el T12 pasa
  a guarda de regresión. Si alguien conecta
  effective_quantum_margin_multiplier sobre free_cap en el núcleo, T12
  falla a propósito y se sustituye por una prueba de propiedad sobre la
  función; si la conecta en risk-engine (available_margin_headroom), T12
  no se entera. La revisión de la tanda 5 vio que el T12 de RET-50b sólo
  cazaba las dos grafías literales (`free_cap *=` y `0.95 *
  directional_flux`): un sombreado `let free_cap = free_cap * …`, un
  recorte en la puerta o la misma copia con otra variable pasaban. RV5-T12
  lo endurece: entre el cálculo de free_cap y su puerta, free_cap se asigna
  una sola vez y no aparece margin_multiplier. La revisión de la tanda 6
  vio que un sombreado con tipo (`let free_cap: f64 = free_cap * …`), uno
  por tupla o una copia en otra variable seguían pasando; RV6-T cuenta
  free_cap entero, que sólo puede aparecer en su cálculo y en la puerta. F2.3-05 sigue abierto: el
  núcleo publica crash_flux ≈ 0,41 en ruido y risk-engine resta
  0,25·max(flux, presión sistémica) del colchón en
  continuous_exposure_limit (Hermite entre 0,80 y 0,90 de presión y 0
  desde 0,90). El flux cuenta cuando supera la presión sistémica (hasta
  0,80 de presión) o 0,80 (en la banda Hermite, aunque quede por debajo
  de la presión); desde 0,90 el límite es 0 y no cuenta.
  risk-engine ya lo leía antes de F1-054, que sólo quitó el recorte
  duplicado del núcleo.
- Lote 44: como ya corregía la redacción (§4, fila 44), bootloader.rs se
  queda (ROSTER y SystemDiagnostics son el arranque vivo) y sólo sale
  SystemBootloader. El paso 1 es CL-156: el host pasa a
  execute_phase_3_warmup el venue del ejecutor (is_testnet), no
  is_demo_mode, y la URL sale de warmup_klines_url; ninguna lectura nueva
  de USE_TESTNET. LC-F1-40 sigue abierto: la sonda de OI elige venue con
  is_demo_env (god_engine.rs:2939), igual que symbol_manager.rs:36/:49.
  Golden y T-1 no recorren el calentamiento del host. Con USE_TESTNET=true
  y un BEST_WS_ENDPOINT de mainnet, el WS va a mainnet y el calentamiento a
  testnet (antes, los dos a mainnet). Ese caso es parte de LC-F1-40 (un
  venue único). FMT-150 (Codex, auditoría XIII) queda cerrado por retiro
  en lo que auditaba: sale el cargador de posiciones de state_continuity
  (load_checkpoint y su checksum XOR). Queda
  EspectralMultiactivo::cargar_checkpoint (F1-078, fila N3), sin llamador
  de producción y sin versión ni comprobación de integridad: quien lo
  cablee hereda el contrato de formato que pedía FMT-150.
- Lote 13: R182 y R197 decían que portar FIX #1454 «sobra»; se portó
  porque lo pedía el crítico (§4, tabla del lote 13), con dos pruebas
  RED→GREEN, y es inerte en vivo: los dos llamadores validan antes
  (stateful_engine.rs:718-736 y :1061-1077) y la sanitización sólo actúa
  si un cociente desborda f32. El riesgo del plan «graph-4d se queda sin
  usuario» ya no aplica (RET-8); graph-architecture se queda por
  src/bin/graph_server.rs:1 (lote 18). Se cierra FMT-140. feature-engine
  pierde 10 pruebas en los dos binarios #[path] (3 propias y 7 de los
  módulos incluidos) y gana 2; la lib de la raíz pierde 9. La copia de
  correlation duplicaba feature_engine::MarketCorrelationHeatmap, que
  tampoco tiene consumidor de producción (F188, R179: SOLO_TESTS,
  RETIRAR); el mensaje de RET-13b la cuenta por error entre las vivas.
- Lote 48 (F261): regla 2 antes de retirar. ExposureBudget ya dividía el
  riesgo por SL + fricción y LH-04 no: N = f·C/SL convertía la fracción
  de Kelly en una pérdida al stop de f·(SL + c)/SL, un 29 % más con SL
  0,5 % y 14,5 pb. CL-159 porta el término (`nocional_por_riesgo`,
  kelly_envelope.rs:379, N = f·C/(SL + c)) a LH-04 (risk-engine lib.rs:1608)
  y a sus dos medidas del riesgo realizado; ADR-0016 lleva el corrigendum
  de D1 (la derivación es la de linea_g/ola6_derivacion_friccion.md:61).
  En el fixture de LH-04, medido bajo wine, el stop real vale ≈ 5,9·ATR%
  y el tamaño baja un 1,4 % (ATR 1 %) o un 2,7 % (ATR 0,5 %). T-1 14/144, la de main (N13). De
  las 6 pruebas nuevas de CL-159 sólo las 3 de contrato
  (cl159_lh04_friccion_al_stop_contract) fallan contra el lib.rs de main;
  las 3 unitarias prueban la función nueva, sin estado previo (el cuerpo
  del PR #138 las daba a las 6 por RED→GREEN). RET-48 retira
  la API (ExposureBudget, exposure_budget(_spectral) y project_quantity):
  6 de sus 9 contratos y el c4 de AGY (Ω215) migran a la función nueva;
  las 3 de lotes no se migran, las cubren round_to_step_size
  (executor.rs, snap_floor) y D-750/ADR-0016 D4. Siguen sin la fricción
  el tope de ruina F1-092 (lib.rs:1820-1837, con expected_loss),
  `capital_regime::orden_viable` (:387, ya en linea_g/vetos.tsv:1129) y
  f_min_ejecutable de max_leverage_spectral (kelly_envelope.rs:322): fila
  N12 (los dos primeros los cierran CL-172 y CL-173 en la tanda 11; el
  tercero espera al dueño). `lh04_f_realizado` cambia de semántica (ahora con la fricción): es
  F252, decisión del dueño, sin lectores fuera de pruebas. El fix sketch
  de G045 (Línea G) nombraba la API retirada.

**Filas nuevas** (salieron al ejecutar; no están en el TSV):

- N1: spot_bid_qty y spot_ask_qty (quantum-arena state.rs:260-261) sin
  lector de producción desde RET-23a (/api/tensor era el último).
  RECONECTAR como OBI del spot (candidata al «OBI L2» de FASE 3,
  docs/PLAN_MAESTRO_SOL_2026-10-18.md:132) o RETIRAR con su escritor y los
  tests de spot_feed.
  Sin ejecutar en la tanda 12: es de Sol (spot_feed.rs, F1-091; lote 12b,
  MEMORIA). Corrección del puntero: el «OBI L2» de FASE 3 es F195, el OBI
  de la profundidad L2 de futuros, que el bookTicker del spot (nivel 1) no
  da. Retirar toca también statarb_live_physics_contract.rs:121
  (apply_spot_tick con 8 argumentos), spot_feed_contract.rs:80-89 y :133,
  y 8 llamadas a update_spot_data en 6 archivos de pruebas del núcleo
  (cf3_reversion_base, comun_base_spot, cf3_spot_congelado_ttl,
  cf3_espejo_conformal_contract, cf3_vecm_base_constante,
  statarb_live_physics_contract). Los precios del spot siguen vivos (core
  lib.rs, StatArb; online_daemon.rs, fila N16).
- N2: ewma_price_at, volatility_at y scale_gain_at (temporal_spectrum.rs)
  sólo con lectores de test desde RET-33. volatility_at es un estimador
  de volatilidad: antes de retirarlo, regla 2 contra el σ vivo
  (sigma_forecast_at, dev_moment_by).
  Ejecutada en la tanda 12 (RET-N2), con la comparación de la regla 2 en
  el commit: volatility_at mide la media de |p − EWMA_τ|/EWMA_τ, no σ(τ)
  del retorno. Sobre caminatas gaussianas, √π·volatility_at/(σ√τ) da en
  los nodos 0,99–1,05 de media (1,49 con τ ≈ dt), con un CV de 0,37–0,53
  por consulta, y la interpolación lineal sesga +5 a +9 % entre nodos. El
  σ(τ) vivo es sigma_forecast_at (geométrica, con habilidad medida) o el
  respaldo ATR; el estimador sigue por nodo como ewma_dev_vol.
  El plan de Línea G para revivir Coaxial (PLAN_LINEA_G_CLAUDE_2026-10-12.md
  §17.6, :1197-1199, y el paso 2 de §17.10, :1287; buzón :14527-14528) nombraba
  volatility_at como su σ(τ) real a 1 s, 5 s y 60 s (revisión de la tanda
  12). La entrada por nodo sigue en `scales[k].ewma_dev_vol`
  (E|dev| = σ√(τ/π) con dt ≪ τ: ×√π para σ√τ), y leer entre nodos pide la
  interpolación geométrica y el factor de resolución (con τ ≈ dt el nodo
  sesga +49 %); para el σ(τ) del retorno, sigma_forecast_at. Avisado en el
  buzón.
- N3: API F1-078 de persistencia y diagnóstico sin consumidor de
  producción: SnapshotParEscala, guardar/cargar_checkpoint,
  exportar/importar_snapshots, estado_gate_ville y el campo
  state_continuity.espectral_snapshots (AGY y Línea C; toca el lote 44,
  R114). RET-44b retiró ArenaCheckpoint con su campo espectral_snapshots;
  queda la API de espectral_multiactivo.rs (guardar/cargar_checkpoint y
  exportar/importar_snapshots), sin llamador de producción.
  Sin ejecutar en la tanda 12: espera a AGY (autor de F1-078) y, si se
  quiere persistencia, al dueño (¿cubre «Guardar todo», CL-108, la
  evidencia Ville multiactivo?). Por defecto, RETIRAR como RET-44b:
  warm_start_par fabrica aciertos = 3n/4 (espectral_multiactivo.rs:693-694),
  y con GRAPA encendido eso daría λ = 0,35 con un acierto del 75 %
  inventado; el doc del módulo (:76) dice «tras reinicio es 0 h», falso en
  producción (el núcleo arranca en frío). Si se quiere persistir, por
  EstadoAprendido, con clave por símbolo (el universo cambia de orden
  entre arranques), no por un segundo archivo.
- N4: Position::entry_tau_ms() (position.rs:683) sin llamadores.
  Ejecutada en la tanda 12 (RET-N4); el campo sigue vivo.
- N5 (F229, cambio de conducta): la cuota de peticiones por criticidad
  (salidas a 98 %, entradas a 90 %) que el censo pedía portar del
  QuantumMultiplexer (quantum_multiplexer.rs:93-97 en 18584be2, retirado
  en RET-30); hoy salidas y entradas comparten HEADROOM 0,8
  (executor.rs:1391-1412). Átomo propio con test.
  Sin ejecutar en la tanda 12 (su revisión de diseño lo paró): el 0,98 de
  salidas es un literal sin derivar, y en la franja 80–98 % varias
  peticiones de salida en vuelo pueden recibir varios 429 a la vez; al
  tercero RATE_BRAKE (executor.rs:1262-1268) bloquea también las salidas
  hasta el enfriamiento + 60 s, y repetidos arriesgan un 418. Hace falta
  una reserva de salidas derivada de una cota de peticiones en vuelo.
  Además flatten_all_positions envía sus cierres sin pasar por la cuota, y
  las lecturas (income, órdenes abiertas, saldo) comparten el tramo de
  salidas. Faltan en la lista de documentos BINANCE_API_MAP.md:75 y
  linea_g/vetos.tsv:690.
  Ejecutada en la tanda 13, en tres átomos, con una reserva sin literal
  nuevo: la regla del exchange más un libro de las críticas en vuelo.
  - CL-177: cada contador (peso 1 min, órdenes 10 s y 1 min) lleva su
    sello, y una respuesta tardía no rebaja el uso dentro de su ventana
    (se guarda la mayor vigente y la última). Antes un sello compartido
    renovaba la edad de los tres y la última en llegar pisaba el valor.
  - CL-178: los envíos que no miraban el freno 429/418 lo miran
    (`check_backoff`): los dos cierres y el GET de modo del aplanado, el
    DELETE de la purga por símbolo, los reintentos D-02 de la OCO y el
    reintento silencioso de `get_payload_account`. Los 429 de los cierres
    del aplanado arman el cooldown (antes sólo se imprimían), y el barrido
    de huérfanas ya no purga dos veces cada símbolo plano en hedge.
  - CL-179: dos tramos. Entradas y lecturas siguen frenando al 80 %; las
    críticas (positionRisk, cancelaciones, piernas, cierres reduce-only,
    OCO, consulta de una orden, listado algo de un símbolo, cierres del
    aplanado) pasan mientras uso + en vuelo + coste ≤ L, con el coste
    reservado bajo el candado y soltado tras la respuesta. Así N críticas
    concurrentes no caben en el hueco de una y no reciben varios 429 a la
    vez, la objeción que paró la tanda 12. Los costes salen del mapa de la
    API; lo no documentado se marca SUPUESTO y toma el valor del análogo
    más caro (binance_api.rs). El barrido de huérfanas va por el tramo de
    lectura: es limpieza.
  - Decisión por defecto, sin tarjeta: durante un cooldown o un freno no
    sale nada de lo que pasa por `check_backoff`, tampoco los cierres. Es
    la regla del exchange (seguir enviando tras un 429 acaba en 418) y lo
    que ya hacían las salidas consultadas. Las excepciones vivas están en
    N20.
  - Revisión antes de fusionar (revisor por pieza y escéptico por
    hallazgo, 16 confirmados y 3 refutados): CL-180, el uso es el máximo
    exacto de las cabeceras vigentes (con sólo la mayor y la última se
    perdía una intermedia aún vigente) y un sello futuro, por un reloj que
    retrocede, se acota a ahora con la siguiente cabecera de ese contador
    (sin ella sigue vigente, lo conservador); CL-181, en el tramo crítico un
    contador en el que la petición no gasta sólo la rechaza con el uso ya
    por encima del límite (un cierre de peso 0 se frenaba por el peso de un
    positionRisk en vuelo); CL-182, la cabecera de una crítica se registra
    antes de soltar su cupo; RV14-c, el aplanado devuelve el freno al
    empezar y comentarios corregidos (el 418 y el tercer 429 arman
    RATE_BRAKE, no el kill-switch); RV14-T, guardas de fuente más
    estrictas. Segunda revisión, sobre esos arreglos (8 confirmados, 1
    refutado): CL-183, en X-026 un positionRisk fresco que no respondió
    (freno, cuota o red) se leía como «ya plana» y se purgaban los brackets
    de una posición quizá viva; ahora queda como no cerrada (defecto previo
    a la tanda, que CL-179 hace más probable: positionRisk cuesta 5 de peso
    y la purga 1); RV15-c y RV15-docs, comentarios, la guarda del listado
    algo y los documentos.
  - Queda sin derivar el 0,8 de entradas y lecturas: su franja (480 de
    peso/min, 60 órdenes/10 s, 240/min) es ahora la reserva crítica. El
    panel de diseño estimó, sin medir, que cubre el tráfico crítico del
    motor hasta unas 13 posiciones en peso y 5 en órdenes de 10 s con dos
    entradas por ventana; derivarla pide publicar el número de posiciones y
    la tasa de entradas (fila N22).
- N6: darwin.rs:151-154 (Sol, F9-4) escribe tp/sl_curve_a/b sin
  begin_write/end_write; tras RET-32 el único escritor de curvas bajo
  seqlock es genome.rs (F1-073 de AGY). update_*_curve no lo arreglaría:
  recibía anclas, no a/b.
- N7: los defaults de cancel_position_oco_orders y query_order del trait
  ExecutionProvider (resto de F247), con F227 en el tema 2.
  Sin ejecutar en la tanda 12. El default de cancel_position_oco_orders
  (Ok(0), «purga hecha») es fail-open latente: sólo lo hereda
  SimulatedExecutor, sin consumidor de producción; el de query_order falla
  cerrado. Hacerlos obligatorios toca simulator.rs (AGY, rescate Ω214 del
  B15-01 de Codex): opción 1, con el visto bueno de AGY, métodos explícitos
  con la conducta de hoy; opción 2, si el dueño retira F227 en el tema 2,
  borrar el simulador. Línea C no elige.
- N8 (F325, cambio de conducta; titular desde 30fc7ee8: AGY implementa,
  Kimi V33 revisa RED/GREEN y Línea C consulta, V33_REVISIONES_2026-10-24.tsv
  fila 12): un
  largo IOC en la orden mínima del lote sale bajo minNotional. El riesgo
  dimensiona esa orden al precio medio con +0,10 USD (risk-engine
  lib.rs:1421-1427), el núcleo garantiza el mínimo al precio de entrada
  modelado ask·(1+s) (core lib.rs:10121-10132), el host pone el límite IOC
  de compra en ask·(1+dyn_slip) redondeado hacia abajo al tick
  (god_engine.rs:5111-5128), por debajo de esa entrada, y Binance juzga el
  nocional al precio de la orden. Con paso 0,1, tick 0,001 y ask 1,219
  salen 41 lotes a 1,219 = 4,9979 USD (-4164). Es raro: ≈ 0,02 % en los
  lotes gruesos de la sonda. OPEN:
  execution-engine/tests/f325_min_notional_ioc_open_diagnostics.rs, con
  guardas de texto sobre esa cadena. El arreglo no es cuantizar con
  validate_and_quantize tal cual: deja la orden mínima un lote por debajo
  en ≈ 11 % de los puntos del retículo (4 342 de 39 992: los pasos 0,001,
  0,01, 0,1, 1, 0,025, 0,05, 0,5 y 0,2 por n = 1…4 999 lotes, con
  q = n·paso en f64, y el piso binario de validate_and_quantize da n − 1);
  hay que contar
  en unidades enteras del paso al precio límite, antes de la reserva. Las
  specs de arranque con minNotional 5 para todo símbolo (F482, tema 4) son
  otra fuente del mismo rechazo. Golden y T-1 no lo ven (el backtest no
  cuantiza al paso).
- N9: la sombra TrendRunner lee `cvpin` con 0,50 por defecto
  (core lib.rs:3713), la misma clase que K12: sólo actúa en el primer
  evento de un núcleo nuevo. Átomo propio con test.
  Ejecutada en la tanda 12 (CL-174): sin cvpin finito la sombra se abstiene
  (VotoEspectral::default()); el mismo voto bit a bit en todo camino
  alcanzable, porque el primer evento tiene H = 0,5 y x ≡ 0.
- N10: game_theory_adversarial_pressure no tiene escritor; la leen la
  sombra de Nash del núcleo (lib.rs:3497) y GameTheoreticNashEngine
  (game_theoretic_nash.rs:196), que caen a cvpin.
  Ejecutada en la tanda 12 (CL-175): las dos lecturas piden cvpin
  directamente, con la misma cadena de respaldo. El 0,1 del motor escalar y
  su recorte siguen (Línea G, vetos.tsv:1236).
- N10b (nueva, de la tanda 12): game_theory_long_payoff y
  game_theory_short_payoff (game_theoretic_nash.rs, evaluate_for_coin)
  tampoco tienen escritor y caen a max(±OFI, 0). Son el gancho de la
  teoría «modelar un juego con pagos medidos» (linea_g/teoria.tsv:266): a
  Línea G, reconstruir Nash con pagos medidos o retirar las claves.
- N11: StanfordDiscoveryEngine (evolution-engine stanford_discovery.rs,
  Fase 6 de AGY, PR #127) sólo tiene lectores de test; nada de producción
  llama a run_discovery_cycle. A AGY: conectar y certificar o retirar.
- N12: la pérdida al stop sin la fricción en el tope de ruina F1-092,
  `orden_viable` y f_min_ejecutable (lote 48). Con CL-159, LH-04 dimensiona
  con SL + c y esos tres siguen con SL: admiten más pérdida real de la que
  declaran (un 29 % más con SL 0,5 % y 14,5 pb). Átomo propio con test (AGY es el último
  autor de F1-092). Ejecutada en la tanda 11 (PR #144): CL-172 y CL-173
  cierran la viabilidad y F1-092; f_min_ejecutable queda para el dueño
  (ADR-0013, Decisión 5).
- N13 (no es de esta tanda): main mide en el T-1 14/144 (9,7 %), bajo el
  trinquete de 11,0 %. Bisección por commit con el oráculo de los genes
  10, 33 y 141 (release bajo wine): 24dfc19c8 sensible, primer commit malo
  01ce0a57 (F1-116, Ω157: lh04_muestras_min de 10 a 1). Desde ahí LH-04
  dimensiona toda orden con n ≥ 1, y el margen Kelly legado (raw_exposure,
  risk-engine lib.rs, `let raw_exposure`), el único camino por el que los genes 10, 11 y
  33 (kelly.rs) y 143 (risk_normalizer, `let risk_normalizer`) movían el TAMAÑO en el
  fixture del T-1, no fija ningún tamaño; F1-087 (ab2992a6) cierra además
  el caso f = 0. El 141 (pendiente de la curva TP) no pasa por
  risk_normalizer: llega por tp_at_tau. Fuera del fixture estos genes
  tienen más lectores (corrección de RV6): 10 y 11 en el pre-examen del
  daemon vivo (mutación directa y kelly_at_tau, online_daemon.rs), y 141 y
  143 por tp_at_tau/sl_at_tau en los brackets del host, el fallback del
  núcleo, la viabilidad del spread y leverage_matrix.rs. El gen 20 perdió
  su única lectura en F1-092 (dd783b74, leverage_matrix.rs:135-142). Se
  ganan 29, 62 y 107: 17 − 6 + 3 = 14 (el 142 depende de la plataforma).
  El T-1 completo de main 58f90ced y el de esta tanda dan la misma lista:
  [1, 17, 18, 24, 27, 29, 39, 62, 68, 69, 107, 129, 130, 131]. Panel de diseño (tres diseñadores y tres jueces): congelar
  10, 11, 20 y 32 en la mutación; reconectar 33 como techo κ de f en LH-04
  (sólo aprieta, inerte con 0,55); cerrar con clases explícitas el resto de
  fail-open del margen legado, y comparar en sombra el Kelly legado con
  LH-04 antes de retirar la cadena. Ninguna de esas piezas devuelve
  cobertura en el fixture, donde el riesgo realizado lo acota el margen
  (≈ 2-4 % frente a f ≤ 25 %). Volver a certificar el trinquete en 14/144
  es decisión del dueño, igual que derivar el apalancamiento del riesgo
  de LH-04.
  Ejecutado en la tanda 7 (PR #139):
  - CL-161: una orden la dimensiona LH-04 o va a la orden mínima (sonda,
    sin edge o n < lh04_muestras_min); lo demás se rechaza con
    REJ_INVALID_INPUT (risk-engine lib.rs, `lh04_entradas_validas`). Cierra
    dos fail-open del margen legado: PF no finito y lh04_muestras_min del
    registro por encima de n. Corrección de la tanda 8: la comprobación
    miraba el bias ya recortado a [0, 1] y sólo veía el NaN (un bias +∞
    dimensionaba con la f entera), un PF negativo y la sonda no se validaban, y
    la clase «f en cero por la racha» no se alcanza (el bias vivo está en
    [0,5; 2]). Lo cierra CL-165.
  - CL-163: f = min(½·Kelly-LCB, tope de ruina, κ) con κ = kelly_clamp_max
    (`tope_gen`). Sólo aprieta. Corrección de la tanda 8: en el motor
    legado el gen acotaba una fracción de MARGEN (raw_exposure), no la
    pérdida al stop. Como f ≤ 0,25, κ sólo muerde en [0,1; 0,25): la línea
    base (0,55) y los genomas guardados (0,8245) no cambian nada. Y la
    orden sólo cambia si κ baja también del riesgo que los techos de margen
    ya dejan (≈ 1,9 % del capital en el fixture de cl163, con L = 1; ver
    N14). κ = 0 no sale de un genoma (lo[33] = 0,1). Fijado a mano por
    debajo de kelly_clamp_min (gen 32, ≥ 0,001), lo rechaza antes FMT-212
    con REJ_INVALID_INPUT, no la cadena legada con REJ_EXPOSURE_ZERO. Al
    retirar esa cadena seguirá así mientras FMT-212 exija clamp_min ≤ κ.
  - CL-162: 10, 11, 20 y 32 se copian sin mutar en mutate_with_rng
    (genome.rs:1883-1921); el daemon ya no muta 10/11 y
    restaurar_genes_no_observables_en_examen los devuelve al incumbente
    (online_daemon.rs:585), así que el pre-examen ya no ordena por una
    Kelly que en el examen no fija ningún tamaño ni pérdida al stop
    (corrección de la tanda 8: a través del margen legado, 10, 11 y 32 sí
    eligen el apalancamiento de la orden mínima en D-130 y, con él, la
    puerta REJ_FEE_IMPACT). Fuera los recortes de 10/11 con literales
    de continuous_evolution_backtest.rs (nichos 2 y 9 y blindaje).
  - CL-164: MedidaOrden lleva f_lh04, f_legado y SL + c, y cada cierre
    suma ln(1 + f·R) por regla (LH-04, legado y el mínimo de las dos) con
    R = retorno neto/(SL + c) en sombra_kelly_logw_{lh04,legado,min};
    las ruinas (1 + f·R ≤ 0) van a sombra_kelly_ruina_* y los cierres a
    sombra_kelly_n (risk-engine lib.rs, `registrar_sombra_kelly`; núcleo
    lib.rs, al cerrar). Tanda 8 (CL-166): una regla arruinada tiene
    log-riqueza −∞; el libro la marca (ruina = 1) y deja de sumarle
    cierres, suma R y R² y el forense lo imprime (SOMBRA_KELLY).
  - T-1 14/144 (9,7 %), la misma lista que main: [1, 17, 18, 24, 27, 29, 39, 62, 68, 69, 107, 129, 130, 131].
  Queda para el dueño: re-certificar el trinquete; la ventana y la regla
  de decisión de la sombra (propuesta: retirar kelly.rs si ni el legado ni
  el mínimo ganan a LH-04 fuera de muestra en las dos mitades del
  forense, cada una una corrida con su línea SOMBRA_KELLY; una regla con
  ruina pierde), y si se adopta el mínimo como regla de tamaño. La
  columna «legado» es el número de kelly.rs leído como f al stop. Los
  moduladores de raw_exposure (gen 143, convicción D-750, interpolación
  micro D-641, Ω134) no entran en la sombra ni se comparan. Retirar
  raw_exposure exige además otra fuente para el apalancamiento de la orden
  mínima, que hoy D-130 elige con el margen legado (corrección de la
  tanda 8).
  Ejecutado en la tanda 8 (PR #140), por la revisión adversarial del
  #139 (17 hallazgos confirmados, 3 refutados):
  - CL-165: la validez va antes de toda clase y sobre el bias crudo:
    win rate no finito o fuera de [0, 1], PF no finito o negativo y bias
    no finito o ≤ 0 se rechazan con REJ_INVALID_INPUT, también en la
    sonda. Un bias finito y positivo fuera de [0,5; 2] sólo encoge f o la
    deja igual. La orden mínima es la sonda, sin edge o n < mínimo. PF = 0
    («sólo pérdidas», dominio [0, ∞) de `estado_aprendido::validar`) no es
    corrupto y sigue en la orden mínima (CL-165b).
  - CL-166: la sombra congela la regla arruinada, suma R y R²
    (`sombra_kelly_suma_r`, `_suma_r2`) y `informe_sombra_kelly` da una
    línea por moneda; el forense la imprime al final (SOMBRA_KELLY).
  - RV7-T: la prueba de κ mira la orden. En su fixture el techo de margen
    (L = 1) acota el riesgo emitido en ≈ 1,9 % del capital, así que
    κ = 0,10 sólo cambia la f objetivo; con κ = ½ de ese riesgo el
    nocional baja a la mitad. cf5d fija R ≈ −1 al cerrar en el stop por
    el camino del núcleo (−1,29 observado).
  - RV7-docs: premisas de CL-163 (margen, banda) y de los genes 10/11/32
    (D-130) en el código, las pruebas, aquí y en el buzón.
  Ejecutado en la tanda 9 (PR #142), por la revisión adversarial del #140
  (11 hallazgos confirmados; 4 de ellos, el PF = 0, ya los cerró CL-165b
  dentro del #140):
  - CL-167: CL-165 sólo leía el bias si LH-04 podía dimensionar (f > 0);
    con f = 0 (sonda, sin edge, n < mínimo) un bias ±∞ o ≤ 0 salía a la
    orden mínima. Ahora se lee y se valida en toda clase. Corrección de
    la tanda 9 a la línea de CL-165 de arriba: «también en la sonda» es
    cierto desde CL-167. Lo que se valida es el bias ACTIVO: con un cierre
    registrado y el reloj ya pasado, `epigenetic_bias_at` lleva un
    almacenado ±∞ o NaN a 1,0 y uno s ≤ 0 a max(0,5; 1 − (1 − s)·
    e^{−Δ/30 min}), 0,5 sólo durante 30·ln(2·(1 − s)) min (≈ 21 con s = 0,
    ≈ 42 con s = −1), y no se rechaza (un almacenado corrupto sólo se ve
    sin cierre registrado o en el mismo ms del cierre; precisión de la
    tanda 10).
  - CL-168: `informe_sombra_kelly` publica `"logw_<regla>":null` para una
    regla arruinada (antes, la suma congelada hasta la ruina, 0 si se
    arruinaba en el primer cierre). Un acumulado no finito salía como
    0.000000 hasta CL-169.
  - RV8-T: la R del libro se fija exacta, −SL/(SL + c) con la fricción del
    fixture (el rango anterior admitía −1, un divisor sin c). La orden de
    cf5d sale con L = 1, así que ese cierre no distingue un R sobre el
    margen de uno sobre el nocional. Corrige el mensaje de 9f9fc77e y el
    comentario de cf5d (RV7-T), que decían que un R sobre el margen (×L)
    ponía roja la prueba; la línea de RV7-T de arriba sólo dice R ≈ −1, y
    eso sigue siendo cierto.
  Antes de fusionar el #142, su revisión adversarial confirmó 7 hallazgos
  (1 medio, 6 leves; 1 refutado):
  - CL-169: el libro de la sombra y su informe leen el registro de la
    moneda sin respaldo (`get_for_coin_strict`). Con `get_for_coin_or` un
    acumulado no finito llegaba como 0, el informe publicaba un 0.000000
    creíble y el cierre siguiente reiniciaba la suma. Ahora sale como null
    y se queda.
  - RV9-T: en el fixture de cl164 un hueco que arruina a LH-04 arruina a
    las tres reglas (f_lh04 ≤ f_legado), así que la comprobación de las
    «vivas» de CL-168 no comprobaba nada. La prueba exige null en las
    tres y otra fija una regla viva junto a una arruinada (4,5 stops
    arruinan al legado). cl161 fija la cobertura real de CL-167 tras un
    cierre. cf5d cierra una tercera posición con L = 10 y exige que la
    sombra reciba el retorno sobre el nocional (antes ninguna prueba con
    L ≥ 2 veía ese argumento; `realized_ret` ya lo fijaba
    close_outcome_contract a través de la recompensa del PPO).
  - RV9-docs: las precisiones de arriba, en el código, MEMORIA, el buzón y
    los cuerpos de los PR #140 y #142.
  Ejecutado en la tanda 10 (PR #143), por la revisión de esas correcciones
  (6 hallazgos confirmados, todos leves y en 4 temas; 1 refutado):
  - CL-170: CL-169 dejó con respaldo, en `registrar_sombra_kelly`, el
    contador n y las marcas de ruina. Un n no finito volvía a 1 en el
    cierre siguiente sobre las sumas de todos los cierres (media falsa y
    creíble) y el informe borraba la línea; una marca +∞ era «viva» para
    el libro y «arruinada» para el informe. Ahora las dos lecturas son
    estrictas, con el mismo predicado que el informe (ausente o NaN, viva;
    > 0, ruina), y un n presente pero inválido sale como null en la línea.
    Con esto es cierto lo que decía CL-169: el libro y su informe leen
    todas sus claves sin respaldo.
  - RV10-T: la prueba de la cobertura real de CL-167 añade el régimen
    relajado (s = 0 una hora después del cierre llega como 1 − e^{−2}).
  - RV10-docs: «uno ≤ 0 a 0,5» sólo vale durante 30·ln(2·(1 − s)) min tras
    el cierre; corregido aquí, en el código, la cabecera de cl161, MEMORIA
    y los cuerpos de los PR. CL-169 cambia también el escritor del libro,
    no sólo el formato del informe (cabecera de este bloque, fila de
    AGENT_SYNC).
  La revisión de la tanda 10 antes de fusionarla confirmó 1 hallazgo más,
  leve (0 refutados), corregido en el mismo PR:
  - CL-171: CL-170 sólo dejaba quieto un n no finito. El libro guardaba
    n + 1 para cualquier n finito, y el informe llamaba corrupto a uno
    menor que 1: un n de 0, 0,5 o −2 salía null una vez y volvía a ser
    válido uno o pocos cierres después, sobre las sumas de todos los
    cierres (con 5 cierres y n = 0, una r_media seis veces la verdadera).
    Un n fraccionario ≥ 1 pasaba por válido. Ahora el libro y el informe
    usan `contador_sombra_valido` (entero finito ≥ 1): sin n el libro
    escribe 1, con uno válido n + 1 y con otro no escribe nada.
  - RV11-docs: estas precisiones aquí, en MEMORIA, AGENT_SYNC, el buzón y
    el cuerpo del PR #143.
  Ejecutado en la tanda 11 (PR #144), fila N12 (diseño con un panel de tres
  diseñadores y un juez; revisión adversarial antes de fusionar: 2 hallazgos
  leves, 2 refutados):
  - CL-172: la viabilidad D-750 (`capital_regime::orden_viable`, un solo
    llamador de producción) recibe `expected_loss + roundtrip_fee`, la
    pérdida al stop por unidad de nocional con la misma c de LH-04 (CL-159)
    y de D-744. Ni carry ni MMR. La orden mínima con
    N_min·SL ≤ tope·C < N_min·(SL + c) pasa de salir a REJ_VIABILIDAD
    (roja sobre main con SL 63,68 % a 13 USD: 3,2521 USD al stop con un
    tope de 3,25).
  - CL-173: el tope de ruina F1-092 es
    `nocional_por_riesgo(tope, C, stop_real, roundtrip_fee) / L`, la función
    de LH-04. Una orden de LH-04 sin rescate no se recorta ni en un ulp (su
    f nunca pasa del tope); muerde sólo tras los rescates (piso ×1,0005,
    D-130, Sol-A2). Rojo sin el arreglo: con un lote grueso (orden mínima de
    50 USD a 140 USD) la orden rescatada y la sonda salían con 35,0087 USD
    al stop y tope 35.
  - RV12-T: la guarda bit a bit de CL-173 corre con L = 3 (300 USD, SL a
    69,4), donde la forma ingenua del cociente sí recortaría un ulp (con
    L = 2 el cociente es exacto y no distinguía nada). Comprobado por
    mutación: con F1-092 en la forma ingenua la guarda falla.
  - RV12-docs: ADR-0016 (corrigendum de CL-159 y viñeta D-750), las dos
    filas de linea_g/vetos.tsv (REJ_VIABILIDAD), esta sección, MEMORIA,
    AGENT_SYNC y el buzón.
  - Sitio 3, f_min_ejecutable de `max_leverage_spectral` (host, replay y
    juez del daemon): NO se toca. Es política de riesgo del dueño
    (ADR-0013, Decisión 5) y la decisión está en el hilo. Por la réplica
    del diseño, contar c ahí sólo endurece, pero ensancha ×(1 + c/SL) la
    ventana no monótona 0 < f < f_min (sin decidir) y puede mover el T-1.
    Diseño preparado, a la espera de esa decisión: un helper único de c
    para el gate y el host, el parámetro del stop pasa a ser la pérdida al
    stop SL + c con el cuerpo intacto, y un solo commit para host, replay y
    daemon (paridad B3.19).
- N14 (de la revisión de la tanda 7): la banda del gen 33, [0,1; 1,0], es
  inerte por encima del axioma de ruina (f ≤ 0,25): el 83 % de la banda,
  la línea base (0,55) y los tres genomas guardados (0,8245). Por lectura
  del código, estrecharla a [0,1; 0,25] no cambia ninguna orden: FMT-212,
  kelly.rs, la cadena legada y LH-04 aplican el axioma después de κ. Pero
  hoy rompería el rollback y la promoción entre entornos de un genoma con
  κ > 0,25. `load_generation` (rama JSON) y `promote_across_env` promueven
  el genoma crudo, sin la normalización de `desde_json` (la rama binaria
  de `load_generation` sí normaliza), y el gate `validate` rechaza un gen
  fuera de banda. Además, con la banda estrecha κ muerde sólo si los
  techos de margen dejan un riesgo mayor que κ (≥ 10 % del capital, con
  apalancamiento alto). A Fable/Línea H (genome.rs, genome_store.rs):
  normalizar esas dos lecturas como `load_active` y luego estrechar la
  banda, o congelar el gen 33 como el 10, 11, 20 y 32.
- N15 (de la tanda 12): la escalada desnuda del host (god_engine.rs, cierre
  de emergencia de NAKED-ESCALATION) descartaba con `let _ =` el resultado
  de cancel_position_oco_orders y logueaba «brackets purgados» siempre;
  D-698 hizo que el ejecutor devolviera Err al fallar una pierna, y los
  otros dos cierres del host (PURGE OCO y EMERGENCY) ya lo reportaban.
  Ejecutada en la tanda 12 (CL-176): el fallo va a telemetry_err! y el
  éxito se loguea sólo con Ok. Sin cambio de órdenes. Guarda estática
  execution-engine/tests/n15_purga_tras_escalada_contract.rs. Aviso a
  Línea I (EDITANDO god_engine.rs en I-6, otro bloque).
- N15b (de la revisión de la tanda 12, a Línea I): la pierna que sobrevive
  a esa purga puede quedar armada sobre la posición siguiente.
  `purge_orphan_legs` corre al principio de la auditoría con la foto de
  posiciones donde la escalada aún está abierta, así que esa auditoría no
  la purga; en la siguiente (hasta 60 s: el camino Err no llama a
  `protection_health::mark_dirty`) sólo la purga si no hay posición en el
  símbolo (pierna BOTH o sin lado) o en ese lado (hedge), y nada bloquea
  una reentrada mientras tanto. `ensure_position_protected` cuenta además
  esa pierna BOTH o sin lado como protección de la posición nueva. En
  hedge, una pierna sin positionSide hace fallar la purga sin cancelarse
  (executor.rs:3835-3840) y el watchdog no la toca mientras viva el lado
  contrario. Lo mismo vale para el cierre con la posición ya inexistente
  del mismo bloque (su comentario decía «en esta misma auditoría»; CL-176
  lo corrige). Propuesta: en el Err, `mark_dirty` y reintento de la purga
  para (símbolo, lado), con la reentrada en ese símbolo y lado bloqueada
  hasta un Ok. Cambia conducta: átomo propio con test.
- N16 (de la tanda 12, sin verificar como defecto): online_daemon.rs:306-325
  (observed_market_env) mide el medio spread del examen walk-forward con
  el libro del SPOT (spot_bid/spot_ask), no con el de futuros, y las
  características del bosque en :1304-1310 y :1450-1456 usan el spread_bps
  del spot con 1,0 por defecto. Posible cruce de mercado. A Sol.
- N17 (de la tanda 12): los genes executor_max_orders_10s y
  executor_max_weight_1m (genome.rs:305-306, 280 y 2200 por defecto) no
  tienen lector desde LG-12, ni sus copias en config.rs:178-179 (los
  límites vivos son 2400/300/1200 de
  binance_api.rs:28-30); de ahí salían los comentarios «0,8·2200» y
  «0,8·280» de tests_lg3 (executor.rs; CL-177 los pasó a 2400/300). Ya están congelados en la mutación
  desde LG-18 (genome.rs:2037-2040, prueba lg18_los_genes_sin_lector_no_mutan;
  linea_g/genes_censo.tsv:135-136), pero new_random los sigue sorteando
  (genome.rs:1098-1099, [100; 500) y [1000; 5000)). A Fable: retirarlos
  del vector (DIMENSION y genomas guardados) o fijarlos también en
  new_random.
- N18 (de la tanda 13, a Línea I, cambio de conducta): con un 429 activo
  (cooldown de 60 s por defecto) la colocación de brackets tras una
  entrada llenada falla en sus tres intentos (god_engine.rs:5424), el
  cierre de emergencia y su reintento a 150 ms también (:5536-5547), y la
  escalada X-009 arma el kill-switch permanente (:5582) por un freno que
  el exchange levanta solo. CL-179 saca ese camino de la franja del 80 %,
  no del cooldown. Candidato: distinguir el rechazo local por freno
  temporal (RATE_LIMIT_COOLDOWN, RATE_BRAKE) y reintentar al acabar antes
  de escalar; el vigilante (CL-44) ya audita la posición mientras tanto.
- N19 (de la tanda 13, a Línea I, sin verificar como defecto): el aplanado
  inmune se marca hecho (latched, god_engine.rs:2420) aunque devuelva
  FLATTEN_INCOMPLETE, y desde CL-178 un 429 a mitad del aplanado deja los
  cierres restantes sin enviar. Lo cubre el vigilante (CL-44: re-bracket
  o cierre de escalada); falta decidir si se reintenta el aplanado al
  acabar el freno.
- N20 (de la tanda 13): peticiones fuera de la cuota del ejecutor, sin
  freno 429/418 ni libro en vuelo: ensure_hedge_mode (GET y POST de
  positionSide/dual), que OmsSession::refresh llama cada 60 s
  (host_oms.rs:484, god_engine.rs:2856-2861); fetch_hedge_mode;
  fetch_all_symbol_filters, con su reintento; fetch_server_time; el bucle
  NTP (ntp.rs:106), que usa el cliente directamente; y el user-data
  stream, que también lo usa: create_listen_key (POST listenKey,
  client.rs:192) en el bucle de reconexión con reintento de 500 ms a 30 s
  (user_data_stream.rs:133-170) y keep_alive_listen_key (PUT,
  client.rs:226) cada 25 min (user_data_stream.rs:213), sin leer las
  cabeceras de uso; y la cancelación de la pierna hermana tras un fill
  (DELETE /fapi/v1/algoOrder por clientAlgoId, user_data_stream.rs:667-715,
  un tokio::spawn por fill de _TP/_TPR/_SL/_SLR/wdTP_/wdSL_), que no mira el
  freno, no pide cupo, tira la cabecera del Ok y cuyo 429/418 no llega a
  handle_rate_limit_error (el streamer no tiene el OrderExecutor). Al
  meterla en la cuota hay que decidir qué hace una crítica rechazada por el
  freno; hoy su Err ya marca la protección sucia y la purga el vigilante. get_payload_account sólo mira el freno antes de su
  reintento: el primer GET de fetch_hedge_mode y ensure_hedge_mode sale
  siempre. Átomo propio (Línea C).
  Ejecutada en la tanda 14 (CL-184 a CL-191), sólo en execution-engine
  (executor.rs y client.rs), sin tocar ntp.rs, user_data_stream.rs,
  host_oms.rs ni el host. Lo que queda va a las filas N26 a N33.
  - CL-184: la descarga de exchangeInfo y su reintento miran el freno
    (sólo el freno, sin la franja del 80 %: los cierres necesitan los
    filtros).
  - CL-185: un exchangeInfo sin filtros utilizables (cuerpo ilegible,
    «TEXT_ERR…», lista vacía) es un error y no borra la caché buena
    (`filtros_de_exchange_info`); antes era un mapa vacío que
    get_symbol_filter guardaba encima.
  - CL-186: un símbolo ausente de un exchangeInfo completo va a la caché
    negativa (F1-105), así que cuesta como mucho una descarga por símbolo
    cada 60 s, no una por llamada.
  - CL-187: el GET de modo de fetch_hedge_mode y ensure_hedge_mode va por
    el tramo de lectura y registra su cabecera; el POST de activación
    mira el freno; la hora de `ExecutionProvider::fetch_server_time` mira
    el freno. Todo llamador de `get_payload_account` frena antes de
    llamarla (guarda).
  - CL-188 (refactor sin cambio de conducta): el freno 429/418 vive en
    `client::FrenoExchange`, uno por ejecutor y compartido por los clones
    de su cliente (el NTP y el user-data stream los toman con
    `client().clone()`), con el mismo reloj que `OrderExecutor::arena`
    (ahora `Arc<ArcSwapOption<GlobalArena>>`).
  - CL-189: la hora del NTP y el alta y la renovación del listenKey, que
    salen por un clon sin ejecutor, miran el freno y lo arman con su
    propio 429/418 (una sola vez: los transportes del ejecutor no
    registran y el ejecutor no llama a estas rutas). Antes su 429 se leía
    como «REJECTED http=429»; ahora vuelve como «FRENO_ARMADO:
    HTTP_429_RATE_LIMITED retry_after=N» (CL-194). Sólo lo leen los logs.
  - CL-190: la cancelación de la pierna hermana (user-data stream) mira el
    freno en `cancel_order_payload`; no registra (su 429 lo imprime el
    stream, fila N27).
  - CL-191 (fila N35): sale el último `.expect` de producción de
    executor.rs y client.rs (`record_query_resolution`, inalcanzable), con
    una guarda que cuenta unwrap, expect y panic en los dos archivos.
  - Decisión por defecto, sin tarjeta: durante un cooldown o un freno no
    salen tampoco el NTP, el listenKey ni la cancelación hermana. El NTP se
    salta sus ticks (corrige al acabar), un keepalive rechazado lo cubre la
    validez de 60 min del listenKey salvo un freno de más de unos 35 min, y
    la cancelación rechazada marca la protección sucia, como un rechazo del
    exchange, y la purga el vigilante al acabar el freno.
  - Revisión antes de fusionar (revisor por grupo de átomos y dos
    escépticos, 10 confirmados y 3 refutados): CL-192, durante el freno un
    símbolo sin spec en el registro entraba en la caché negativa por el
    rechazo local de la descarga y la entrada (60 s) sobrevivía al freno;
    CL-193, el 2xx del NTP y del listenKey reinicia la racha de 429 (desde
    CL-189 sus 429 cuentan, y en papel, con el ejecutor casi sin tráfico,
    tres 429 aislados en un día armaban RATE_BRAKE); CL-194, el 429/418 que
    una ruta sin ejecutor ya registró vuelve marcado y no cuenta dos veces
    aunque el stream o el NTP lo registren (la guarda no leía a esos
    llamadores); RV16-T, las pruebas de CL-186, CL-187 y CL-188 prueban lo
    que dicen (mutaciones comprobadas). El aviso falso del arranque con el
    modo sin consultar va a la fila N39; el freno a cero tras el cambio de
    ejecutor (refutado como defecto de la tanda: ya pasaba) es N26.
    Segunda revisión, sobre esos arreglos (1 confirmado, baja, y 1
    dividido): CL-195, CL-192 decidía con el freno armado al llegar al paso
    5, no con lo que pasó en la descarga; ahora no guarda sólo si la
    descarga de esa llamada la paró el freno o la cuota (su rechazo local o
    su 429/418) y no hay un exchangeInfo guardado, que sí prueba la
    ausencia; RV17-T, la guarda de CL-193 exige el reinicio fuera de la rama
    de error.
- N21 (de la tanda 13): las respuestas de error tiran sus cabeceras de uso,
  así que tras un 4xx/5xx el uso no se actualiza. En client.rs:296, :420 y
  :508 extract_limits sólo corre en la rama 2xx; en :355
  (execute_order_payload_body) corre con cualquier estado, 429/418
  incluidos, y en :471 (get_payload_with_timeout) con cualquiera salvo
  429/418, que vuelven antes; en esas dos el Err sólo lleva un String y los
  límites leídos se pierden. Un timeout AMBIGUOUS pudo contar en el
  exchange sin traer respuesta. En los dos casos, al soltar el cupo de una
  crítica su coste sale del libro en vuelo sin entrar en el uso (lo vio la
  revisión en los reintentos D-02 de la OCO). Candidato: devolver los
  límites también en el Err y, con un Err ambiguo, sumar el coste a un uso
  local sellado que caduque con su ventana. Átomo propio, con la fila N20.
  En la tanda 14, CL-189 sólo arregla las tres rutas sin ejecutor (hora del
  NTP y listenKey): su 429/418 ya no es «REJECTED http=429», arma el freno
  con su Retry-After. El resto pasa a N23 (cabeceras de los siete
  transportes), N24 (coste sin respuesta) y N25 (marca AMBIGUOUS).
- N22 (de la tanda 13): literales y pesos sin derivar en la cuota. El 0,8
  de entradas y lecturas (executor.rs, `HEADROOM`) y los +60 s del freno
  escalado de M4-C02 (`until.saturating_add(60_000)`). Y tres pesos del
  mapa que el panel de diseño recuerda distintos en la documentación de
  Binance, sin poder comprobarlo desde la nube (sus hosts están
  bloqueados): commissionRate 1 frente a 20, positionSide/dual 1 frente a
  30, income 20 frente a 30. OmsSession::refresh lee commissionRate de cada
  símbolo del universo cada 60 s (host_oms.rs:498): con peso 20 y 20
  símbolos serían 400 de peso por minuto, el 17 % del límite. Comprobar
  desde el PC contra la documentación y derivar el 0,8 con el número de
  posiciones y la tasa de entradas publicados. Falta también comprobar si
  las ventanas del exchange están alineadas al minuto o son móviles. Si
  están alineadas, una cabecera menor que llega más tarde que la vida
  máxima de una petición (timeouts de client.rs: 1,2 s en órdenes, 5 s en
  cuenta, 10 s en exchangeInfo) tras la mayor prueba un reinicio, y el
  contador podría soltar la mayor antes de su caducidad (hoy la conserva,
  lo conservador; CL-180).
  Desde la tanda 14 los 429 del NTP y del listenKey cuentan también para
  el freno escalado (CL-189), así que sus +60 s se arman más a menudo; con
  N18 (X-009 arma el kill-switch permanente si un rechazo local bloquea
  los brackets) eso sube la prioridad de N18. El riesgo añadido es pequeño:
  un 429 del NTP o del listenKey significa que la IP ya pasó el límite, y la
  siguiente petición del ejecutor recibiría el suyo.
- N23 (de la tanda 14, resto de N21, Línea C): las respuestas de error de
  los siete transportes de las peticiones firmadas (client.rs
  `execute_order_payload`, `_body`, `cancel_order_payload`,
  `get_payload_with_timeout`, `post_payload`) siguen tirando su cabecera de
  uso. Diseño: un `ErrorRest { texto, limits }` con `Deref<Target = str>` y
  `Display`, sin `From<ErrorRest> for String`; registrar por
  `observar_limites` (nunca `update_limits`, que reinicia la racha) y nunca
  la cabecera de un 429/418. Al hacerlo, juntar los cinco
  `let _ = self.handle_rate_limit_error(` del ejecutor y cuidar que ningún
  envío registre dos veces su 429 (en `execute_order` el 429 ya vuelve por
  `handle_rate_limit_error`). Hace falta un banco de pruebas en loopback para
  las rutas `full_url`.
- N24 (de la tanda 14, Línea C, con N22): el coste de una petición sin
  respuesta (timeout AMBIGUOUS o error de transporte) sale del libro en
  vuelo sin entrar en el uso, y el reintento de `get_payload_account` envía
  dos veces con una sola reserva. Candidato: `Cupo::sin_respuesta(ahora)`,
  una muestra local sellada que caduca con su ventana.
- N25 (de la tanda 14, Línea C, cambio de conducta): la marca AMBIGUOUS es
  desigual. En `execute_order_payload` un 5xx, un 408 o -1006/-1007 salen
  como «Binance API Error»; en `cancel_order_payload`,
  `get_payload_with_timeout` y `post_payload` un error de transporte no es
  AMBIGUOUS; `get_payload_with_timeout` devuelve Ok con «TEXT_ERR…» si no
  pudo leer el cuerpo, y `_typed` tira los límites de un 2xx cuyo ACK no
  se puede leer. Cambia cómo se resuelven las órdenes: átomo propio.
- N26 (de la tanda 14; método de Línea C, línea del host de Línea I o AGY):
  el cambio de ejecutor en caliente (god_engine.rs:4410-4431, mismo
  USE_TESTNET y la misma arena que el anterior; `new_with_shared`) estrena
  un freno a cero aunque el exchange siga frenando la IP. Parche:
  `OrderExecutor::heredar_cuota(&self, previo) -> bool` sólo con la misma
  URL base y la misma clave (plazos y `consecutive_429` con `fetch_max`,
  muestras de uso copiadas sin anidar candados; nunca `en_vuelo` ni el
  kill-switch), llamado por el host antes de publicar el nuevo
  (god_engine.rs:4431) o en `OmsSession::replace_executor`. El método entra
  con su llamada.
- N27 (de la tanda 14; Codex QRA, Línea I desde el 2026-10-26, con Línea C):
  lo que queda de la cancelación de la pierna hermana
  (user_data_stream.rs:631-731). (a) Su 429/418 sólo se imprime: registrarlo
  con `client.freno().registrar(&e)` y salir del bucle de candidatos con un
  HTTP_4xx. (b) Los candidatos `_SLR`/`_TPR` están muertos desde LG-6
  (:650-657). (c) No pide cupo crítico y tira la cabecera: con un
  `Arc<OrderExecutor>` cancelaría por `cancel_algo_order` (necesita el
  cableado del host). (d) La era demo firma con la hora local sin corregir
  (:675-677, arena None) aunque con USE_TESTNET=true opera la testnet real:
  firmar con `client.freno().ahora_ms()`. (e) URLs literales (:688-694).
- N28 (de la tanda 14; dueño de `start()`, Codex QRA o Línea I): el ritmo del
  bucle del user-data stream. El backoff se reinicia antes de conectar el
  WS (:135, un bucle de ~0,5 s); tras un cierre del servidor espera 1 s
  fijo (:260-261); durante un freno debería dormir hasta su fin con jitter
  (pide `FrenoExchange::activo_hasta_ms()`) y reintentar el keepalive al
  acabar; el bucle del listenKey corre también en papel (C8).
- N29 (de la tanda 14, Línea I): el sondeo de OI del host
  (god_engine.rs:3066-3144) tiene su propio cliente reqwest de 8 s, envía N
  peticiones seguidas cada 120 s, se traga el 429/418 con `if let Ok` y elige
  el entorno con `is_demo_env`. Si el entorno es el del ejecutor: saltar el
  ciclo con el freno armado (`exec.load().client().freno()`, hace falta un
  acceso público), clasificar y registrar el 429/418 y cortar al primero.
- N30 (de la tanda 14, Sol): sondeos de data-pipeline en la misma IP.
  omni_multiplexer pide PAXG cada 60 s y premiumIndex, OI y hasta 60
  `/futures/data` cada 120 s (:578, :673-773); dynamic_ranker,
  ticker/24hr (peso 40) y exchangeInfo cada hora (:81-140); spot_feed abre
  N peticiones ante cualquier no-2xx (:284-290). Todos se tragan el
  429/418. data-pipeline no puede nombrar `FrenoExchange`: el host
  inyectaría un asa del freno (un trait o un cierre en quantum-arena). Parar
  el abanico del spot con un 429/418.
- N31 (de la tanda 14; god-engine-core, coordina Línea I): el calentamiento
  de velas del bootloader (bootloader.rs:245-284, llamado en
  god_engine.rs:1541) no mira el estado HTTP: un 429/418 da una serie vacía
  sin aviso (unos 150 de peso con N = 30).
- N32 (de la tanda 14, Línea C, documentación): los binarios de operador
  (system_health, income_report, oco_probe, algo_fetch_probe,
  audit_forensic_backtest) usan la misma IP y cuenta con su propia cuota.
  Regla operativa: no correrlos durante un freno. Falta documentarla.
- N33 (de la tanda 14, Línea C): sin un exchangeInfo guardado y con la
  descarga fallando fuera de un freno, get_symbol_filter vuelve a descargar
  en cada llamada (hasta unos 20 s cada una con una red lenta). Candidato:
  un plazo global de reintento tras una descarga fallida y la precarga al
  arrancar.
- N34 (de la tanda 14, Línea I): entre 0,8·L y L, `purge_orphan_legs` se
  traga el rechazo del tramo de lectura de `fetch_all_open_algo_orders`
  (`if let Ok`, god_engine.rs:286) y la auditoría limpia la marca de sucio
  (:3495). Durante un 429/418 la auditoría para antes (:3251), así que
  CL-190 es seguro. Pregunta abierta de N5: ¿openAlgoOrders sin símbolo es
  lectura aunque la purga sea protectora?
- N35 (de la tanda 14): el `.expect` de `record_query_resolution`. Hecha en
  CL-191.
- N36 (de la tanda 14, Línea I): el comentario D-702 (god_engine.rs:1916-1921)
  dice que la era demo es papel; con USE_TESTNET=true es falso
  (`set_paper_trading(is_demo_mode && !is_env_testnet)`, :1433): esa era
  opera la testnet real con un streamer sin arena.
- N37 (de la tanda 14; Línea I, I-5, o AGY, Ω209; inferido, sin medir): el
  filtro de Kalman del NTP es lento con el RTT de producción (K ≈ 0,006,
  constante de tiempo de unos 41 min). Un salto de +1,5 s del reloj del
  equipo dejaría toda petición firmada, salidas incluidas, en -1021 unos
  17 min.
- N38 (de la tanda 14, Línea C, con N22): las respuestas 2xx de las rutas sin
  ejecutor (NTP, listenKey) no observan el uso: su cabecera no entra en los
  contadores. Desde CL-193 sí reinician la racha de 429. Decidir con N22
  (alineación de las ventanas y los +60 s).
- N39 (de la revisión de la tanda 14, Línea I): el preflight de arranque
  (god_engine.rs:1747-1756) imprime «No se pudo garantizar modo hedge … TODA
  orden fallará (-4061)» ante cualquier error de ensure_hedge_mode. Desde
  CL-187 puede ser un rechazo local (uso por encima del 80 %, cooldown o
  RATE_BRAKE) sin haber consultado el modo, con `is_hedge_mode` aún en true
  y las órdenes funcionando. Candidato: con un error que empieza por RATE_,
  registrar «modo no consultado por cuota; lo confirma el refresco OMS».

**Restos de lotes ejecutados**: del 3, net_multiplexer (a AGY, para
unificar con data_pipeline::http_cliente) y active_universe/
symbol_ranker_engine (tema 4 y gen 37); del 17, F196 (tema 2, por
F214/F463); del 43, F394 (tema 1, con F305) y los literales 12/30 de
online_daemon.rs (a Sol); del 28, el 28b (Navier, espera al dueño de
Ω121); del 7, detect_collisions de omniscient-registry (devuelve
siempre un vector vacío y sólo lo prueban tests; átomo propio); del 11,
bump/take_pending/peek_pending de god-engine-core liquidation_feed.rs
(sólo tests); del 12a, comentarios que siguen nombrando módulos
retirados: data-pipeline cero_dato.rs:88-89 y
tests/cero_dato_contract.rs:24, :33 (de Sol), dynamic_ranker.rs:101 y
parser.rs:142-143 (cita tensor_parser); del 50, la familia
margin_multiplier*/with_quantum_phase/effective_quantum_margin_multiplier y
la cadena quantum_phase (sin consumidor de producción; a AGY, que la
amplió en el Átomo 3.3), los campos margin_contraction_scale,
min_margin_floor y w_quantum_coupling de SpectralRegimeParams, que sólo
ella lee, los dos from_registry_or_default (SpectralRegimeParams y
EspectralMultiactivoParams, sin ningún llamador desde e468491b) y
cushion_symmetry_contract.rs:9-99 (AGY, Ω59: prueba una copia local de la
fórmula que F1-054 quitó del núcleo); del 44, los stubs
execute_phase_4_training (devuelve Some(()) y el host lo guarda en
_swing_nn, god_engine.rs:1529) y execute_phase_6_hft de SystemDiagnostics,
el comentario de TensorConfig «We always use mainnet data»
(god_engine.rs:1146) y lo que F327 quería llevar al diario de posiciones
(los extremos del trailing; el diario vivo, position_journal.jsonl, ya
lleva ts y age_hours, y PositionSnapshot nunca tuvo productor, así que no
hay datos que portar: es el abierto de MEMORIA «adopción con
entry_time_ms = now») y el reintento de la fase 3 retirada (hasta 3
intentos con 500 ms tras un error de transporte, y el aviso «Falló la
carga de klines»), que no se portó: el calentamiento vivo
(bootloader.rs:261) hace un solo intento y el fallo sólo se ve como
«Inyectadas 0 K-lines» en el host (átomo propio, con test); del 13, el guard de finitud del resultado de la
copia ewma (R180), que el Ewma vivo no tiene y sólo actúa cerca de
1e308; del 21, el comentario de god_engine.rs:5458-5460 (zona AGY), que
cita read_recent_events de un módulo que ya no existe; del 42,
genome.rs:2013-2014 (Fable) y f1_084_genome_portfolio_perf_freeze_contract.rs:7
(AGY), que nombran calculate_dynamic_allocation; del 39,
docs/DISENO_META_ANALITICA_INTEGRAL.md:19 (GLM), que da behavioral_auditor.rs
como fuente de las sombras (es auditor_interno.rs), y
docs/TRIAJE_ROJOS_PERPETUOS.md:12, que cita auditor_open_diagnostics.rs:40
(ahora :39); del 19, el recuento de replay-contracts.yml
(storage-engine pasa de 39 a 17 tests de lib; el de telemetry-engine lo
puso al día RET-51c); del 27, los bloques de MEMORIA de las olas Ω137, Ω142 y Ω143 y de
la liquidez oculta documentan claves que ya no se publican.
