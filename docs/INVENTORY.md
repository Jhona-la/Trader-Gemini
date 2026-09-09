# INVENTARIO MAESTRO DEL SISTEMA (F0.8 — Censo archivo-por-archivo)

> Generado en la Fase 0 del Plan Maestro (2026-08-16). Este documento es la
> fuente de verdad de QUÉ existe, QUÉ está vivo, QUÉ está archivado y QUÉ
> valor hay que rescatar. Se actualiza al cierre de cada fase.

## 1. Mapa del workspace (23 crates vivos + paquete raíz)

| Crate | LOC | Archivos | Rol | Estado |
|---|---|---|---|---|
| god-engine-core | 3.658 | 15 | Motor central scalp+swing, orchestrator, darwin, ml_inference | VIVO (hot-path) |
| quantum-arena | 3.477 | 18 | Arena lock-free, genome, símbolos, estado global | VIVO (hot-path) |
| data-pipeline | 2.636 | 23 | WS, multiplexers, parser, macro/onchain feeds, api_client | VIVO |
| execution-engine | 2.529 | 12 | Executor, cliente REST, router, shadow, simulator | VIVO (crítico F1) |
| metacortex-engine | 2.100 | 16 | Consejo de seniors, sistema inmune, templates evolutivas | VIVO — ConsejoDeliberacion en el core (F4.8 verificado) |
| telemetry-server | 1.984 | 13 | Servidor warp, Telegram bot, flight recorder | VIVO |
| evolution-engine | 1.877 | 12 | Evolución, polars_evolver, online_daemon | VIVO (rediseño F4.5) |
| audit-engine | 1.100 | 11 | Drift/trajectory auditors | VIVO — DriftAuditor arma kill-switch (E-03); TrajectoryAuditor track por posición (F4.8) |
| feature-engine | 1.075 | 15 | Features técnicas, correlación | VIVO |
| risk-engine | 875 | 11 | Kelly dual, DD dinámico, EV gate | VIVO (recalibrar F5.1) |
| storage-engine | 809 | 7 | Lakehouse (record_tensor), WAL | VIVO (expandir F2.5) |
| strategy-core | 712 | 10 | Intenciones de señal | VIVO |
| os-guardian | 686 | 9 | Prioridad CPU, afinidad, auditoría memoria | VIVO (adaptar F5.3) |
| dark-alpha-engine | 571 | 3 | MLP swing 54→64→32→1 + Adam | VIVO (paridad F4.1) |
| backtest-engine | 499 | 2 | Backtest sintético + vectorizado | A REEMPLAZAR (F3) |
| signal-engine | 447 | 14 | TurboScalp, TensorVote (huérfanos) | VIVO — 14 módulos de física, TODOS con consumidores (F4.8) |
| data-ingest | 345 | 4 | Ingesta histórica | VIVO |
| graph-architecture | 267 | 2 | Análisis sintáctico del grafo de llamadas | HERRAMIENTA F4.8 |
| telemetry-engine | 232 | 3 | Macros de telemetría | VIVO |
| omniscient-registry | 205 | 2 | Registro de capacidades | VIVO — arena.registry lee/escribe pesos hebbianos por símbolo |
| phase-runner | 150 | 2 | Fases (stub sleep 10ms) | VIVO — audit-engine main + multi_coin_simulator (auditoría estática) |
| graph-4d | 76 | 1 | Grafo 4D ligero | VIVO — audit-engine main (grafo del workspace) |
| flight-recorder | 14 | 1 | Shim | VIVO |
| **src/ (raíz)** | **7.812** | **63** | 20 binarios + lib auxiliares | MIXTO (censo §3) |

## 2. Código muerto archivado (../TraderGemini_archivo/) — valor a rescatar

| Archivado | Motivo | Valor rescatable (migrar en F4) |
|---|---|---|
| crates/state-engine | Fork divergente de god-engine-core, solo referenciado por shims no compilados | `calculate_entropy`, `calculate_l2_entropy` (→ Chrono-Stability F4.10); `fast_tanh_simd`, `process_4x_simd`, `update_weighted` (SIMD); `update_online`, `get_adaptive_leverage`, `get_dynamic_tp_sl` (revisar vs implementación viva) |
| src/core/ (20 archivos) | Capa de shims 1-línea no declarada en lib.rs | Ninguno (eran re-exports) |
| src/telemetry/ (6 archivos) | Dup de crates/telemetry-server (el Telegram vivo está en telemetry-server) | diff rápido antes de descartar |
| crates/quantum-engine | Directorio vacío (0 .rs) | Ninguno |
| aits_research/ | Python/PyTorch (directriz: cero Python) | Referencia histórica |
| premium-dashboard/ | React/TS no conectado | Referencia visual |
| monitoring_tools/ | Grafana+Prometheus vendorizados (98 MB) | Re-descargar si se monta stack F6 |
| scratch/, sistema_inmune/, memoria/, wandb/, cerebro/, research/, optimization/, hardware/, f32/ | Experimentos huérfanos / basura de raíz | memoria/ tiene 12 JSONs de estado runtime — revisar en F2.7 |

## 3. Binarios de src/bin/ (20)

| Bin | Propósito | Notas |
|---|---|---|
| god_engine | Motor principal live | 1.203 líneas; punto de entrada F1/F7 |
| evolution / evolver | GA + simulated annealing | Rediseño F4.5 (contaminación OOS) |
| train_dark_alpha / auto_trainer_daemon | Entrenamiento MLP swing | Vigilancia F4.2 |
| audit_forensic_backtest | Backtest honesto (fees reales) | BASE del motor canónico F3 |
| multi_coin_simulator | Sim multi-moneda | Usa modelos reales |
| download_history / binance_vision_sync / macro_history_sync | Ingesta de datos | F2 |
| feature_exporter / feature_validator / parquet_to_bin | Pipeline de features | F2 |
| graph_server | Grafo 3D | Revivir en F6.4 |
| dashboard / test_forest / test_forest2 / latency_flow_audit / quantum_benchmark / config_compiler | Utilidades | Revisar en F4.8 |

## 4. Mapa de calor de hardcode (densidad de números mágicos)

Top archivos (conteo heurístico de literales numéricos fuera de comentarios):
1. **quantum-arena/src/genome.rs: 805** — esperado (bounds de genes), PERO incluye fees/funding evolucionables = BUG F4.5 (el GA puede evolucionar los costos hacia 0)
2. god-engine-core/src/lib.rs: 173
3. telemetry-server/src/lib.rs: 128
4. god-engine-core/src/math_kernels.rs: 111
5. src/bin/evolution.rs: 100
6. src/bin/god_engine.rs: 79

**Valores hardcodeados críticos conocidos (de la auditoría):**
- `god_engine.rs:279` — `darwin_approved = true` (gate de producción anulado)
- `god-engine-core/lib.rs` — interlock de latencia obsoleta con umbral 3.000.000 ms (≈50 min)
- `god_engine.rs:472` — máscara de afinidad 0xFFFF (16 cores fijos)
- `god_engine.rs:738` — `&[0.0; 54]` como features omni de la NN swing
- `darwin.rs:204` — leverage forzado 25-35x
- `guard.rs:19` — tolerancia DD 99% para capital pequeño
- `vectorized.rs:70,84` — fees/funding derivables del genoma
- `executor.rs:284` — precio `{:.4}` ignorando tickSize real

## 5. Linaje de archivos runtime (F2.7 confirmará generadores)

| Archivo/Dir | Generador | Estado |
|---|---|---|
| state.json/state2/state3.json | ¿? (sospecha: bins legacy) | SIN CONFIRMAR — no trackear hasta linaje |
| exchange_info.json | runtime (cache de /fapi/v1/exchangeInfo) | Usado por dynamic_ranker, simulator, executor |
| config_dir/genotypes/*.json | evolver.rs / evolution-engine | Hot-cargados por god_engine (<60s) |
| models/*.json | train_dark_alpha + forests | Hot-reload por mtime |
| data/ (1.9 GB) | download_history, binance_vision_sync | Parquet + *_ticks.bin |
| STOP_TRADING.LOCK | LAUNCHER.bat / EMERGENCY_SHUTDOWN.bat | NO leído por ningún .rs — decorativo hasta F5.2 |

## 6. Deuda conocida heredada (de auditoría, pendiente por fase)

- Double-fill en execute_maker_chase (executor.rs:573-587) → **F1.3**
- Respuestas de orden sin parsear (orderId/fills/rejects) → **F1.1**
- Sin newClientOrderId en market orders → **F1.2**
- Camino maker con firma/payload inconsistentes → **F1.4**
- Look-ahead en backtest sintético → **F3.2**
- FFI out_stats overflow → **F3.3**
- OOS contaminado en evolution.rs → **F3.6/F4.5**
- DNS pineado en ws_client → **F2.2**
- Macro feed Yahoo muerto (features congeladas) → **F2.4**
- mapeo lows=bid_qty/volumes=ask_qty en evolution.rs:109-115 → **F2.3**
- Relaxed load+store en win_rate/profit_factor → **F4.9**
- os-guardian TIME_CRITICAL + 0xFFFF → **F5.3**
