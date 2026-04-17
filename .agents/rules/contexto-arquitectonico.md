---
trigger: always_on
---

🎯 CONTEXTO CRÍTICO - PROYECTO TRADER GEMINI (SCALPING + SWING BINANCE)
PROYECTO: Trader Gemini - Sistema HFT Multi-Horizonte en Binance OBJETIVO: Operaciones Scalping (1s-5min) + Swing (1h-4h), alta frecuencia, máxima estabilidad CAPITAL: $13 USD → Duplicar cada 15 días ÚLTIMA ACTUALIZACIÓN: 2026-04-01

📁 ARQUITECTURA TRADER GEMINI - MAPA COMPLETO ACTUALIZADO
Trader Gemini/
├── config.py                      # ⚙️ CONFIGURACIÓN CENTRAL - Todas las constantes y parámetros
├── conftest.py                    # 🧪 Fixtures globales pytest - Setup compartido de tests
├── main.py                        # 🚀 PUNTO DE ENTRADA - Orquestador principal del bot (42KB)
├── setup.py                       # 📦 Empaquetado Python - Distribución del proyecto
│
├── core/                          # 🚨 NÚCLEO CRÍTICO - NO MODIFICAR SIN CONFIRMACIÓN
│   ├── engine.py                  # 🚨 MOTOR PRINCIPAL (29KB) - Event loop, coordinación multi-horizonte
│   ├── events.py                  # 🚨 Sistema mensajería (Signal/Order/Fill) - Bus de eventos tipado
│   ├── portfolio.py               # 🚨 Gestión estados/balances/PnL (67KB) - Ledgers virtuales Scalping/Swing
│   ├── market_regime.py           # 🚨 Clasificador HMM tendencia/volatilidad (20KB) - Filtro estratégico
│   ├── market_regime_hmm.py       # 🚨 Modelo Hidden Markov avanzado (5KB) - Detección regímenes
│   ├── enums.py                   # 📋 Enumeraciones globales (Horizon, Side, OrderType)
│   ├── resolution_state.py        # 📋 Estado de resolución temporal
│   │
│   ├── # --- MOTOR EVOLUTIVO & IA ADAPTATIVA ---
│   ├── evolution.py               # 🧬 Motor evolutivo Darwiniano (15KB) - Mutación/selección estrategias
│   ├── genotype.py                # 🧬 Representación genética de parámetros (3KB) - DNA de estrategia
│   ├── gene_bank.py               # 🧬 Banco de genes persistente (3KB) - Almacén genotipos exitosos
│   ├── shadow_darwin.py           # 🧬 Evolución shadow paralela (15KB) - Simulación A/B evolutiva
│   ├── shadow_optimizer.py        # 🧬 Optimizador shadow complementario
│   ├── reward_system.py           # 🧬 Sistema recompensas RL (4KB) - Feedback para evolución
│   ├── meta_optimizer.py          # 🧬 Meta-optimización Optuna (10KB) - Hyperparameter tuning
│   ├── self_tuner.py              # 🧬 Auto-ajuste adaptativo (12KB) - Calibración continua
│   │
│   ├── # --- INTELIGENCIA DE MERCADO ---
│   ├── market_scanner.py          # 🔭 Scanner multi-activo (5KB) - Detección oportunidades
│   ├── liquidity_guardian.py      # 🛡️ Guardián de liquidez (4KB) - Validación profundidad orderbook
│   ├── orderbook.py               # 📊 Procesador orderbook Python (2KB) - Análisis bid/ask
│   ├── c_orderbook.pyx            # ⚡ Orderbook Cython nativo (2KB) - Procesamiento ultra-rápido
│   ├── c_orderbook.cp314-win_amd64.pyd  # ⚡ Binario compilado Cython - Latencia nano
│   ├── sentiment_processor.py     # 🌐 Procesador sentimiento (5KB) - Análisis Fear&Greed
│   ├── world_awareness.py         # 🌍 Conciencia macro-económica (3KB) - Eventos globales
│   ├── correlation_manager.py     # 📈 Gestor correlaciones (4KB) - Matriz correlación activos
│   ├── swarm_correlator.py        # 🐝 Correlador enjambre (3KB) - Detección patrones multi-activo
│   ├── sovereign_oracle.py        # 🔮 Oráculo soberano (5KB) - Predicción agregada final
│   ├── multiverse_simulator.py    # 🌌 Simulador multiverso (4KB) - Monte Carlo escenarios
│   │
│   ├── # --- MACHINE LEARNING & APRENDIZAJE ---
│   ├── online_learning.py         # 🤖 Aprendizaje online (19KB) - Adaptación en tiempo real
│   ├── online_learning_kernels.py # ⚡ Kernels optimizados aprendizaje (7KB) - Cómputo vectorizado
│   ├── neural_bridge.py           # 🧠 Puente neural Sophia↔Core (8KB) - Interfaz IA
│   ├── ml_governance.py           # 🛡️ Gobernanza ML (5KB) - Anti-leakage, validación modelos
│   ├── fused_strategy_kernel.py   # ⚡ Kernel fusionado estrategias (3KB) - Single-pass cómputo
│   │
│   ├── # --- GESTIÓN DE FLUJO ---
│   ├── strategy_selector.py       # 🎯 Selector estrategias por horizonte (10KB) - Router Scalping/Swing
│   ├── adaptive_balancer.py       # ⚖️ Balanceador adaptativo (7KB) - Distribución capital dinámico
│   ├── order_manager.py           # 📋 Gestor órdenes (6KB) - Cola y lifecycle de órdenes
│   ├── data_handler.py            # 📊 Handler de datos interno (10KB) - Transformación OHLCV
│   ├── state_manager.py           # 💾 Gestor estado persistente (9KB) - Crash recovery
│   ├── simulation.py              # 🎮 Motor simulación (7KB) - Paper trading y sandbox
│   │
│   ├── # --- INFRAESTRUCTURA & PERFORMANCE ---
│   ├── api_manager.py             # 🔌 Gestor API centralizado (16KB) - Rate limiting, retry
│   ├── rate_limiter.py            # 🚦 Rate limiter Binance (2KB) - Protección throttling
│   ├── pre_flight.py              # ✈️ Verificación pre-vuelo (7KB) - Checks arranque sistema
│   ├── gc_tuner.py                # ⚙️ Tuner Garbage Collector (3KB) - Optimización GC Python
│   ├── jit_warmup.py              # 🔥 Warmup JIT/Numba (1KB) - Pre-compilación rutas calientes
│   ├── memory.py                  # 🧠 Gestor memoria cognitiva (6KB) - Cache inteligente
│   ├── watchdog.py                # 🐕 Watchdog sistema (3KB) - Heartbeat y auto-recovery
│   ├── system_monitor.py          # 📊 Monitor sistema (2KB) - CPU/RAM/Latencia
│   ├── secure_store.py            # 🔐 Almacén seguro credenciales (2KB) - Encriptación keys
│   ├── transparent_logger.py      # 📝 Logger transparente (8KB) - Auditoría decisiones
│   │
│   ├── # --- EXPLAINABILITY ---
│   ├── xai_engine.py              # 🔍 Motor explicabilidad (5KB) - SHAP/LIME por trade
│   ├── forensics.py               # 🔬 Motor forense (3KB) - Análisis post-trade
│   ├── audit_phase2.py            # 📋 Auditoría fase 2 (8KB) - Validación profunda
│   │
│   └── interfaces/                # 🔌 Interfaces abstractas
│       └── exchange.py            # 🔌 Interfaz Exchange (1KB) - Contrato para ejecutores
│
├── risk/                          # 🔒 SEGURIDAD MÁXIMA - MÁXIMA CAUTELA
│   ├── risk_manager.py            # 🔒 Gestión riesgo COMPLETA (81KB) - Size/SL/TP por horizonte
│   ├── kill_switch.py             # 🔒 Parada emergencia (8KB) - Circuit breaker drawdown
│   └── risk_manager.BAK           # 💾 Backup risk manager anterior
│
├── execution/                     # ⚡ EJECUCIÓN DELICADA - PRECISIÓN ABSOLUTA
│   ├── binance_executor.py        # ⚡ Ejecutor órdenes Binance (73KB) - Market/Limit/SL/TP
│   ├── liquidity_guardian.py      # 🛡️ Guardián liquidez ejecución (10KB) - Slippage protection
│   ├── cost_guard.py              # 💰 Guardián costos (2KB) - Fee tracking y optimización
│   ├── user_data_stream.py        # 📡 Stream datos usuario (11KB) - Websocket account updates
│   └── live_smoke_test.py         # 🧪 Smoke test producción (3KB) - Validación live
│
├── strategies/                    # 🧠 LÓGICA TRADING - VALIDAR ESTADÍSTICAMENTE
│   ├── __init__.py                # 📦 Registro estrategias
│   ├── strategy.py                # 📋 Clase base abstracta - Interfaz para todas las estrategias
│   ├── technical.py               # 🧠 Estrategia Técnica Híbrida (89KB) - Scalping/Trend principal
│   ├── ml_strategy.py             # 🤖 Estrategia ML XGBoost (183KB) - Predicción por horizonte
│   ├── ml_worker.py               # ⚙️ Worker ML asíncrono (7KB) - Entrenamiento background
│   ├── statistical.py             # 📊 Estrategia estadística (26KB) - Mean reversion/Z-score
│   ├── sniper_strategy.py         # 🎯 Estrategia Sniper (18KB) - Ultra-selectiva scalping
│   ├── arbitrage.py               # 💱 Arbitraje cross-pair (5KB) - Detección spreads
│   ├── stat_arb.py                # 📐 Arbitraje estadístico (6KB) - Cointegración pairs
│   ├── phalanx.py                 # 🛡️ Estrategia Phalanx (7KB) - Formación defensiva multi-signal
│   ├── quant_math.py              # 🔢 Matemáticas cuantitativas (4KB) - Funciones helper trading
│   │
│   └── components/                # 🔧 Componentes modulares de estrategias
│       ├── adaptive_engine.py     # 🧬 Motor adaptativo (5KB) - Auto-calibración parámetros
│       ├── feature_engineering.py # 📊 Ingeniería features (20KB) - 100+ features técnicos
│       ├── microstructure.py      # 🔬 Microestructura mercado (6KB) - Análisis tick-level
│       ├── signal_generator.py    # 📡 Generador señales (5KB) - Agregación multi-fuente
│       └── models/                # 🤖 Modelos ML
│           └── factory.py         # 🏭 Factory de modelos (2KB) - Creación XGBoost/LightGBM
│
├── sophia/                        # 🧠 CAPA INTELIGENCIA ARTIFICIAL - CEREBRO SUPERIOR
│   ├── __init__.py                # 📦 Registro módulo Sophia
│   ├── intelligence.py            # 🧠 Inteligencia central (93KB) - Orquestador IA decisiones
│   ├── nemesis.py                 # ⚔️ Motor adversarial (47KB) - Stress testing estrategias
│   ├── narrative.py               # 📖 Generador narrativas (8KB) - Explicación humana decisiones
│   ├── axioma.py                  # 📐 Motor axiomático (5KB) - Reglas lógicas inmutables
│   ├── post_mortem.py             # 🔬 Análisis post-mortem (11KB) - Autopsia trades perdedores
│   └── rewards.py                 # 🏆 Sistema recompensas IA (3KB) - Feedback loop Sophia
│
├── ml/                            # 🤖 MACHINE LEARNING - INFRAESTRUCTURA
│   └── replay_buffer.py          # 🔄 Buffer experiencia RL (5KB) - Almacén transiciones
│
├── data/                          # 📊 FLUJO DATOS - INTEGRIDAD CRÍTICA
│   ├── data_provider.py           # 📊 Fuente única verdad OHLCV (1KB) - Router de datos
│   ├── binance_loader.py          # 📊 Conector Binance real-time (84KB) - Websockets + REST
│   ├── database.py                # 💾 Persistencia SQLite WAL (14KB) - Trades, estados, métricas
│   ├── feature_store.py           # 📊 Feature store (5KB) - Cache features calculados
│   ├── user_stream.py             # 📡 Stream usuario Binance (15KB) - Balance/Position updates
│   ├── sentiment_loader.py        # 🌐 Loader sentimiento (7KB) - Fear&Greed, social metrics
│   ├── historical_loader.py       # 📚 Loader histórico (3KB) - Descarga masiva velas
│   ├── historic_loader.py         # 📚 Loader histórico alt (3KB) - Formato alternativo
│   ├── download_history.py        # ⬇️ Descargador historial (2KB) - Script bulk download
│   ├── ibkr_loader.py             # 🏦 Loader Interactive Brokers (4KB) - Multi-broker soporte
│   ├── audit_genesis.py           # 🔍 Auditor datos génesis (2KB) - Validación integridad
│   ├── engineer_genesis.py        # ⚙️ Ingeniero features (2KB) - Pipeline transformación
│   ├── ingest_genesis.py          # 📥 Ingestión datos (3KB) - ETL pipeline
│   ├── ingest_supreme.py          # 📥 Ingestión suprema (2KB) - Pipeline optimizado
│   ├── cache_parquet/             # 💽 Cache Parquet - Datos OHLCV comprimidos
│   ├── gene_bank/                 # 🧬 Banco genes persistido
│   ├── genotypes/                 # 🧬 Genotipos almacenados
│   └── historical/               # 📚 Datos históricos descargados
│
├── dashboard/                     # 📈 MONITOREO - MANTENER FUNCIONAL
│   ├── app.py                     # 📈 Dashboard Streamlit (61KB) - Interfaz monitoreo completa
│   └── data/                      # 📊 Datos dashboard
│
├── utils/                         # 🔧 UTILIDADES - ESTABILIDAD SISTEMA (67 archivos)
│   ├── __init__.py                # 📦 Exports utilidades
│   ├── # --- LOGGING & OBSERVABILIDAD ---
│   ├── logger.py                  # 📝 Logger principal (11KB) - Structured logging
│   ├── transparent_logger.py      # 📝 Logger transparente proxy
│   ├── log_analyzer.py            # 🔍 Analizador logs (13KB) - Pattern matching anomalías
│   ├── telemetry.py               # 📡 Telemetría (8KB) - Métricas Prometheus/Grafana
│   ├── metrics.py                 # 📊 Métricas trading (4KB) - Sharpe, Sortino, etc.
│   ├── metrics_exporter.py        # 📤 Exportador métricas (17KB) - Push a backends
