---
trigger: always_on
---

├── utils/                         # 🔧 UTILIDADES - ESTABILIDAD SISTEMA (67 archivos)
│   ├── __init__.py                # 📦 Exports utilidades
│   ├── # --- LOGGING & OBSERVABILIDAD ---
│   ├── logger.py                  # 📝 Logger principal (11KB) - Structured logging
│   ├── transparent_logger.py      # 📝 Logger transparente proxy
│   ├── log_analyzer.py            # 🔍 Analizador logs (13KB) - Pattern matching anomalías
│   ├── telemetry.py               # 📡 Telemetría (8KB) - Métricas Prometheus/Grafana
│   ├── metrics.py                 # 📊 Métricas trading (4KB) - Sharpe, Sortino, etc.
│   ├── metrics_exporter.py        # 📤 Exportador métricas (17KB) - Push a backends
│   ├── analytics.py               # 📊 Analytics avanzado (12KB) - Reporting detallado
│   ├── reporter.py                # 📋 Generador reportes (4KB) - Output formateado
│   │
│   ├── # --- ERROR HANDLING & RESILIENCIA ---
│   ├── error_handler.py           # 🔧 Handler errores (7KB) - Retry, fallback, recovery
│   ├── circuit_breaker.py         # ⚡ Circuit breaker (3KB) - Protección cascading failures
│   ├── health_check.py            # 🏥 Health check (2KB) - Endpoint salud
│   ├── health_supervisor.py       # 🏥 Supervisor salud (6KB) - Monitoreo continuo
│   ├── heartbeat.py               # 💓 Heartbeat (2KB) - Señal de vida periódica
│   ├── sentinel.py                # 🛡️ Centinela (37KB) - Vigilante integral del sistema
│   ├── watchdog (→keep_alive.py)  # 🐕 Keep alive (2KB) - Anti-timeout
│   │
│   ├── # --- PERFORMANCE & OPTIMIZACIÓN ---
│   ├── math_kernel.py             # ⚡ Kernel matemático (21KB) - NumPy/Numba optimizado
│   ├── math_helpers.py            # 🔢 Helpers matemáticos (1KB) - Funciones auxiliares
│   ├── axioma_math.py             # 📐 Matemáticas axiomáticas (4KB) - Precisión financiera
│   ├── statistics_pro.py          # 📊 Estadísticas avanzadas (19KB) - Tests, distribuciones
│   ├── hft_buffer.py              # ⚡ Buffer HFT (7KB) - Ring buffer zero-copy
│   ├── hft_experience_buffer.py   # ⚡ Buffer experiencia HFT (3KB) - Replay rápido
│   ├── memory_pool.py             # 🧠 Pool memoria (10KB) - Pre-allocación objetos
│   ├── memory_alignment.py        # ⚡ Alineamiento memoria (1KB) - Cache-line optimal
│   ├── shared_memory.py           # 🔗 Memoria compartida (3KB) - Inter-process comm
│   ├── shm_utils.py               # 🔗 Utils shared memory (2KB) - Helpers SHM
│   ├── binary_packer.py           # 📦 Empaquetador binario (2KB) - Serialización rápida
│   ├── fast_json.py               # ⚡ JSON rápido (6KB) - orjson/ujson wrapper
│   ├── fast_packer.py             # ⚡ Packer rápido (1KB) - MessagePack
│   ├── fast_strings.py            # ⚡ Strings rápidas (1KB) - Operaciones optimizadas
│   ├── bloom_filter.py            # 🌸 Bloom filter (3KB) - Deduplicación O(1)
│   ├── token_bucket.py            # 🪣 Token bucket (4KB) - Rate limiting token-based
│   ├── cpu_affinity.py            # 💻 Afinidad CPU (4KB) - Thread pinning
│   ├── timer_resolution.py        # ⏱️ Resolución timer (1KB) - High-precision timing
│   ├── os_tuner.py                # ⚙️ Tuner OS (4KB) - TCP/Socket optimization
│   ├── network_optimizer.py       # 🌐 Optimizador red (1KB) - DNS/Latency tuning
│   ├── dns_cache.py               # 🌐 Cache DNS (2KB) - Resolución rápida
│   │
│   ├── # --- GESTIÓN DE DATOS ---
│   ├── data_manager.py            # 📊 Gestor datos (11KB) - CRUD operaciones datos
│   ├── data_sync.py               # 🔄 Sincronización datos (2KB) - Multi-source sync
│   ├── clean_data.py              # 🧹 Limpieza datos (5KB) - Outlier removal, gaps
│   ├── common.py                  # 📋 Funciones comunes (4KB) - Helpers compartidos
│   ├── time_helpers.py            # ⏰ Helpers temporales (1KB) - Conversión timestamps
│   ├── time_sync.py               # ⏰ Sincronización tiempo (2KB) - NTP check
│   ├── ntp_monitor.py             # ⏰ Monitor NTP (3KB) - Drift detection
│   │
│   ├── # --- TRADING ESPECÍFICO ---
│   ├── cooldown_manager.py        # ❄️ Gestor cooldowns (12KB) - Anti-overtrading
│   ├── position_cleaner.py        # 🧹 Limpiador posiciones (8KB) - Orphan cleanup
│   ├── safe_leverage.py           # 🔒 Apalancamiento seguro (14KB) - Validación leverage
│   ├── mae_tracker.py             # 📉 Tracker MAE/MFE (3KB) - Maximum adverse excursion
│   ├── efficacy_tracker.py        # 📊 Tracker eficacia (7KB) - Win rate por estrategia
│   ├── wallet_heartbeat.py        # 💰 Heartbeat wallet (5KB) - Balance monitoring
│   │
│   ├── # --- EVOLUCIÓN & RL ---
│   ├── evolution_kernels.py       # 🧬 Kernels evolución (4KB) - Operadores genéticos rápidos
│   ├── rl_buffer.py               # 🤖 Buffer RL (4KB) - Experience replay
│   │
│   ├── # --- SEGURIDAD & DEPLOYMENT ---
│   ├── security.py                # 🔐 Seguridad (2KB) - Encriptación, hashing
│   ├── security_scanner.py        # 🔍 Scanner seguridad (4KB) - Vulnerability check
│   ├── session_manager.py         # 🔑 Gestor sesiones (11KB) - API session lifecycle
│   ├── prod_handshake.py          # 🤝 Handshake producción (14KB) - Validación pre-deploy
│   ├── reloader.py                # 🔄 Hot reloader (18KB) - Live code reload
│   ├── atomic_guard.py            # ⚛️ Guard atómico (2KB) - Operaciones atómicas
│   ├── dep_graph.py               # 📊 Grafo dependencias (12KB) - Dependency analysis
│   │
│   ├── # --- MONITORING AVANZADO ---
│   ├── system_monitor.py          # 💻 Monitor sistema (4KB) - Resources tracking
│   ├── latency_monitor.py         # ⏱️ Monitor latencia (2KB) - P50/P95/P99
│   ├── thread_monitor.py          # 🧵 Monitor threads (1KB) - Deadlock detection
│   ├── debug_tracer.py            # 🔍 Tracer debug (2KB) - Execution tracing
│   ├── chaos_engine.py            # 🌪️ Motor caos (2KB) - Chaos engineering
│   ├── notifier.py                # 📢 Notificador (4KB) - Telegram/Discord alerts
│   ├── wandb_tracker.py           # 📊 Tracker W&B (11KB) - Experiment tracking
│   └── audit_futures.py           # 🔍 Auditor futuros (3KB) - Validación contratos
│
├── models/                        # 🤖 MODELOS ML ENTRENADOS (25 activos)
│   ├── BTCUSDT_xgb.json           # 📦 Modelo XGBoost BTC
│   ├── ETHUSDT_xgb.json           # 📦 Modelo XGBoost ETH
│   ├── SOLUSDT_xgb.json           # 📦 Modelo XGBoost SOL
│   ├── ... (25 modelos por activo) # 📦 Un modelo por cada par trading
│   └── WIFUSDT_xgb.json           # 📦 Modelo XGBoost WIF
│
├── sophia/                        # 🧠 (Descrito arriba - Capa IA)
├── ml/                            # 🤖 (Descrito arriba - Infraestructura ML)
│
├── scripts/                       # 🔧 SCRIPTS OPERACIONALES (40 archivos)
│   ├── run_multi_horizon_backtest.py   # 📊 Backtest multi-horizonte (147KB) - Motor principal BT
│   ├── supervisor_24h.py               # 🕐 Supervisor 24h (23KB) - Daemon monitoreo
│   ├── profitability_diagnosis.py      # 🔍 Diagnóstico rentabilidad (14KB)
│   ├── preflight_check.py             # ✈️ Preflight check (9KB) - Validación pre-vuelo
│   ├── optuna_oracle_tuner.py         # 🎯 Tuner Optuna (10KB) - Optimización Bayesiana
│   ├── run_optuna_oracle.py           # 🏃 Runner Optuna (7KB) - Script ejecución
│   ├── validate_genotype_evolution.py  # 🧬 Validador evolución (6KB)
│   ├── validate_hft_stack.py          # ⚡ Validador stack HFT (5KB)
│   ├── validate_capital.py            # 💰 Validador capital (3KB)
│   ├── report_sim_adaptive.py         # 📊 Reporte simulación adaptativa (5KB)
│   ├── production_reset.py            # 🔄 Reset producción (4KB)
│   ├── train.py                       # 🤖 Entrenamiento ML (4KB)
│   ├── umbral_sincro.py               # ⏰ Umbral sincronización (4KB)
│   ├── health_check.py                # 🏥 Health check (3KB)
│   ├── ... (26 scripts adicionales)   # 🔧 Diagnóstico, limpieza, inspección
│   └── dashboard/                     # 📈 Dashboard scripts
│
├── tests/                         # 🧪 TESTING COMPLETO (147+ archivos)
│   ├── # --- TESTS UNITARIOS ---
│   ├── unit/                      # 🧪 Tests unitarios aislados
│   ├── test_strategies.py         # 🧠 Tests estrategias (11KB)
│   ├── test_risk_manager.py       # 🔒 Tests riesgo (11KB)
│   ├── test_risk_engine.py        # 🔒 Tests motor riesgo (16KB)
│   ├── test_flow.py               # 🔄 Tests flujo completo (14KB)
│   ├── test_api.py                # 🔌 Tests API (13KB)
│   │
│   ├── # --- TESTS INTEGRIDAD ---
│   ├── test_sync_integrity.py     # 🔗 Tests integridad sync (22KB)
│   ├── test_schema_resilience.py  # 📋 Tests resiliencia schema (19KB)
│   ├── test_fuerza_delta.py       # ⚡ Tests fuerza delta (20KB)
│   ├── test_production_integrity.py  # 🏭 Tests integridad producción (6KB)
│   │
│   ├── # --- TESTS CHAOS & STRESS ---
│   ├── test_chaos_engineering.py  # 🌪️ Tests chaos engineering (30KB)
│   ├── chaos_test.py              # 🌪️ Tests caos extendido (23KB)
│   ├── stress_test.py             # 💪 Tests estrés (20KB)
│   ├── byzantine_test.py          # ⚔️ Tests fallos bizantinos (17KB)
│   ├── black_swan_backtest.py     # 🦢 Tests cisne negro (19KB)
│   │
│   ├── # --- TESTS ML & SOPHIA ---
│   ├── test_sophia_intelligence.py  # 🧠 Tests Sophia (7KB)
│   ├── test_nemesis.py              # ⚔️ Tests Nemesis (12KB)
│   ├── test_neural_bridge.py        # 🧠 Tests puente neural (3KB)
│   ├── test_online_learning.py      # 🤖 Tests aprendizaje online (2KB)
│   ├── test_ml_leakage.py           # 🔍 Tests anti-leakage (3KB)
│   │
│   ├── # --- BENCHMARKS ---
│   ├── benchmark_total_latency.py   # ⏱️ Benchmark latencia total (4KB)
│   ├── benchmark_risk_latency.py    # ⏱️ Benchmark riesgo (2KB)
│   ├── benchmark_fused_kernel.py    # ⚡ Benchmark kernel fusionado (1KB)
│   ├── benchmark_simd_math.py       # ⚡ Benchmark SIMD (1KB)
│   │
│   ├── # --- BACKTESTING ---
│   ├── run_backtest.py              # 📊 Backtest completo (59KB)
│   ├── walk_forward.py              # 📊 Walk-forward test (15KB)
│   ├── run_massive_horizons.py      # 📊 Backtesting masivo (4KB)
│   │
│   ├── # --- VALIDACIÓN & AUDITORÍA ---
│   ├── float_precision_audit.py     # 🔢 Auditoría precisión float (14KB)
│   ├── mutation_tester.py           # 🧬 Testing mutaciones (23KB)
│   ├── omni_certification.py        # ✅ Certificación omni (15KB)
│   ├── expectancy_analysis.py       # 📊 Análisis expectancy (16KB)
│   │
│   ├── # --- HERRAMIENTAS TEST ---
│   ├── mocks/                       # 🎭 Mocks para testing
│   ├── integration/                 # 🔗 Tests integración
│   ├── concurrency/                 # 🔄 Tests concurrencia
│   ├── security/                    # 🔐 Tests seguridad
│   └── audits/                      # 🔍 Auditorías automatizadas
│
├── tools/                         # 🛠️ HERRAMIENTAS ANÁLISIS
│   ├── walk_forward_tester.py     # 📊 Tester walk-forward (10KB)
│   ├── monte_carlo_sim.py         # 🎲 Simulación Monte Carlo (5KB)
│   ├── massive_report_generator.py  # 📋 Generador reportes masivos (4KB)
│   ├── oracle_remediation.py      # 🔧 Remediación oráculo (6KB)
│   └── convergence_audit.py       # 🔍 Auditoría convergencia (2KB)
│
├── analysis/                      # 📊 ANÁLISIS & ENTRENAMIENTO
│   ├── train_supreme.py           # 🤖 Entrenamiento supremo (4KB)
│   ├── train_genesis.py           # 🤖 Entrenamiento génesis (4KB)
│   ├── correlation_audit.py       # 📈 Auditoría correlación (4KB)
│   ├── audit_performance.py       # 📊 Auditoría performance (3KB)
│   ├── validate_matrix.py         # ✅ Validación matriz (4KB)
│   ├── log_analyzer.py            # 🔍 Analizador logs (5KB)
│   └── smoke_test_supreme.py      # 🧪 Smoke test supremo (2KB)
│
├── docs/                          # 📚 DOCUMENTACIÓN COMPLETA
│   ├── ARCHITECTURE.md            # 🏗️ Arquitectura del sistema
│   ├── ARCHITECTURE_METAL_CORE.md # ⚡ Arquitectura metal/core
│   ├── PROJECT_BIBLE.md           # 📖 Biblia del proyecto (30KB)
│   ├── STRATEGIES.md              # 🧠 Documentación estrategias
│   ├── RISK.md                    # 🔒 Documentació
