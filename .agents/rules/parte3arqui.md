---
trigger: always_on
---

├── docs/                          # 📚 DOCUMENTACIÓN COMPLETA
│   ├── ARCHITECTURE.md            # 🏗️ Arquitectura del sistema
│   ├── ARCHITECTURE_METAL_CORE.md # ⚡ Arquitectura metal/core
│   ├── PROJECT_BIBLE.md           # 📖 Biblia del proyecto (30KB)
│   ├── STRATEGIES.md              # 🧠 Documentación estrategias
│   ├── RISK.md                    # 🔒 Documentación riesgo
│   ├── DEPLOYMENT.md              # 🚀 Guía despliegue
│   ├── COMMANDS.md                # 💻 Comandos disponibles
│   ├── COMMANDS_V3.md             # 💻 Comandos V3
│   ├── RECOVERY_PROTOCOL.md       # 🔄 Protocolo recuperación
│   ├── GOD_MODE_PROTOCOL.md       # 🚀 Protocolo God Mode
│   ├── GOD_MODE_LAUNCH_REPORT.md  # 📋 Reporte lanzamiento
│   ├── OMEGA_PROTOCOL.md          # 🌌 Protocolo Omega
│   ├── SYMBOLS.md                 # 💱 Símbolos trading
│   ├── TRADING_KNOWLEDGE.md       # 📊 Conocimiento trading
│   ├── ANALYSIS_LOSSES.md         # 📉 Análisis pérdidas
│   ├── VALIDATION_REPORT.md       # ✅ Reporte validación
│   └── PULSO-NOUVEAUCRAFT.md      # 🎨 Pulso Nouveaucraft
│
├── launchers/                     # 🚀 SCRIPTS LANZAMIENTO (.bat)
│   ├── START_FUTURES.bat          # ▶️ Iniciar modo futuros
│   ├── START_SPOT.bat             # ▶️ Iniciar modo spot
│   ├── START_GROWTH.bat           # ▶️ Iniciar modo crecimiento
│   ├── PREFLIGHT_CHECK.bat        # ✈️ Verificación pre-vuelo
│   ├── MASSIVE_BACKTESTER.bat     # 📊 Backtester masivo
│   ├── MOCK_MULTIVERSE.bat        # 🌌 Mock multiverso
│   ├── BOTTLENECK_HUNTER.bat      # 🔍 Cazador cuellos botella
│   ├── EMERGENCY_SHUTDOWN.bat     # 🚨 Apagado emergencia
│   ├── DASHBOARD_FUTURES.bat      # 📈 Dashboard futuros
│   ├── DASHBOARD_SPOT.bat         # 📈 Dashboard spot
│   ├── INFRA_MANAGER.bat          # ⚙️ Gestor infraestructura
│   ├── REPORT_DISCREPANCY.bat     # 📋 Reporte discrepancias
│   ├── UMBRAL_SINCRO.bat          # ⏰ Sincronización umbral
│   ├── WANDB_SYNC_ALL.bat         # 📊 Sync W&B
│   └── HELP.bat                   # ❓ Ayuda comandos
│
├── deployment/                    # 🚀 INFRAESTRUCTURA DEPLOYMENT
│   ├── build_instructions.md      # 📋 Instrucciones build
│   ├── entrypoint.sh              # 🐳 Entrypoint Docker
│   ├── prometheus.yml             # 📊 Config Prometheus
│   ├── promtail.yml               # 📊 Config Promtail
│   ├── loki-config.yaml           # 📊 Config Loki
│   ├── elk/                       # 📊 Stack ELK (Elasticsearch/Logstash/Kibana)
│   └── grafana/                   # 📈 Dashboards Grafana
│
├── dev_ops/                       # ⚙️ DEVOPS
│   └── purge_protocol.py         # 🧹 Protocolo purga (1KB) - Limpieza sistema
│
├── hardware/                      # 💻 HARDWARE SPECS
│   └── fpga_spec.md              # 🔧 Especificación FPGA - Futuro HW aceleración
│
├── grafana/                       # 📈 GRAFANA
│   └── sophia_view_dashboard.json # 📊 Dashboard Sophia Grafana
│
├── Dockerfile                     # 🐳 Docker config
├── docker-compose.yml             # 🐳 Docker Compose (multi-service)
├── LAUNCH_GOD_MODE.bat            # 🚀 Lanzador God Mode
├── MASTER_LAUNCHER.bat            # 🚀 Lanzador maestro
├── requirements.txt               # 📦 Dependencias Python
├── pytest.ini                     # 🧪 Config pytest
├── mypy.ini                       # 🔍 Config mypy
└── README.md                      # 📖 README principal
📊 ESTADÍSTICAS DEL PROYECTO
Capa Archivos Tamaño Total Aprox. Criticidad
core/ 52 ~350KB 🚨 MÁXIMA
risk/ 2 (+1 BAK) ~89KB 🔒 MÁXIMA
execution/ 5 ~96KB ⚡ ALTA
strategies/ 11 + 5 components ~400KB 🧠 ALTA
sophia/ 7 ~165KB 🧠 ALTA
data/ 14 ~130KB 📊 ALTA
utils/ 67 ~450KB 🔧 MEDIA
scripts/ 40 ~300KB 🔧 MEDIA
tests/ 147+ ~900KB 🧪 MEDIA
tools/ 5 ~27KB 🛠️ BAJA
TOTAL ~350+ archivos ~3MB+ código 
🔄 FLUJO DE DATOS PRINCIPAL
WebSocket/REST
Scalping 1s-5m
Scalping Sniper
ML Predictivo
Estadístico
Arbitraje
Defensivo
Decisión IA
Aprobado
Fill
PnL Update
Régimen
Emergency Stop
Genotype Update
binance_loader.py
data_provider.py
feature_engineering.py
strategy_selector.py
technical.py
sniper_strategy.py
ml_strategy.py
statistical.py
arbitrage.py / stat_arb.py
phalanx.py
sophia/intelligence.py
events.py Signal
risk_manager.py
engine.py
binance_executor.py
portfolio.py
dashboard/app.py
market_regime.py
kill_switch.py
evolution.py
💡 REGLAS NEGOCIO TRADER GEMINI
Órdenes LIMIT exclusivamente (minimizar slippage)
Cierre automático por kill_switch.py en drawdown >2%
Backtesting tick-by-tick con datos Binance reales
Validación walk-forward para evitar overfitting
Monitoreo tiempo real via dashboard/app.py
Gestión dual Scalping/Swing con ledgers virtuales independientes
Evolución Darwiniana de parámetros via genotype.py
Sophia IA como cerebro decisor final
25 modelos XGBoost entrenados por activo
Capital micro $13 USD - Sizing ultra-conservador
🔄 WORKFLOWS ESPECÍFICOS GEMINI (EJECUTAR AUTOMÁTICAMENTE)
📈 WORKFLOW: CAMBIO_ESTRATEGIA_TECHNICAL
Para modificar technical.py, ml_strategy.py, statistical.py, sniper_strategy.py, phalanx.py, arbitrage.py, stat_arb.py:

🔍 BUSCAR código similar en strategies/ y strategies/components/
📊 ANALIZAR impacto en portfolio.py, market_regime.py y sophia/intelligence.py
⚠️ VALIDAR con risk_manager.py y kill_switch.py
🧬 VERIFICAR compatibilidad con evolution.py y genotype.py
🧪 BACKTEST 1 semana con run_multi_horizon_backtest.py datos reales
📈 ACTUALIZAR STRATEGIES.md y dashboard/app.py
👨‍🏫 EXPLICAR cambios usando método profesor completo
⚡ WORKFLOW: MODIFICACION_CORE_CRITICO
Para cambios en core/, risk/, execution/:

🚨 EVALUAR criticidad: engine.py > risk_manager.py > binance_executor.py > portfolio.py
🛡️ SANDBOX testing obligatorio (simulation.py)
👥 REQUERIR confirmación explícita para módulos críticos
🔄 PLAN reversión detallado paso a paso
📖 ACTUALIZAR ARCHITECTURE.md y RISK.md
🧪 PRUEBA resiliencia con chaos_test.py y stress_test.py
📚 WORKFLOW: DOCUMENTACION_SISTEMA
Para documentar o explicar cualquier parte:

📋 REVISAR documentación existente en docs/
👨‍🏫 EXPLICAR usando QUÉ-POR QUÉ-PARA QUÉ-CÓMO-CUÁNDO-DÓNDE-QUIÉN
✍️ ACTUALIZAR/CREAR ARCHITECTURE.md, STRATEGIES.md, etc.
✅ VERIFICAR que documentación es clara y completa
🔄 VINCULAR documentación con código específico
🧪 WORKFLOW: EXAMEN_PRE_PRODUCCION
Antes de ejecutar el bot en producción:

🔍 ANÁLISIS ESTÁTICO: Revisar código completo Trader Gemini/
🧪 PRUEBAS UNITARIAS: pytest tests/ completo
📊 BACKTEST COMPLETO: run_multi_horizon_backtest.py 1 mes datos Binance reales
⚡ PRUEBA LATENCIA: benchmark_total_latency.py < 50ms
🌪️ PRUEBA RESILIENCIA: chaos_test.py + byzantine_test.py + black_swan_backtest.py
📈 VALIDACIÓN MÉTRICAS: Sharpe > 2.0, Drawdown < 1.5%, Win Rate > 55%
🧠 VALIDACIÓN SOPHIA: test_sophia_intelligence.py + test_nemesis.py
🧬 VALIDACIÓN EVOLUCIÓN: validate_genotype_evolution.py
📋 GENERAR informe de salud del sistema completo
❓ CHECKLIST PRE-IMPLEMENTACIÓN GEMINI (OBLIGATORIO)
Antes de cualquier cambio, verificar:

✅ ¿Afecta latency de engine.py? [SÍ/NO]
✅ ¿Preserva kill_switch.py funcional? [SÍ/NO]
✅ ¿Mantiene data_provider.py/binance_loader.py integridad? [SÍ/NO]
✅ ¿Actualiza dashboard/app.py correctamente? [SÍ/NO]
✅ ¿Logging en logger.py incluido? [SÍ/NO]
✅ ¿Error handling en error_handler.py? [SÍ/NO]
✅ ¿Documentación actualizada? [SÍ/NO]
✅ ¿Explicación completa (modo profesor)? [SÍ/NO]
✅ ¿Sophia AI compatibility? [SÍ/NO]
✅ ¿Evolution/Genotype compatibility? [SÍ/NO]
✅ ¿Scalping AND Swing horizons covered? [SÍ/NO]
📊 MÉTRICAS GEMINI OBLIGATORIAS
Latencia total: < 50ms (engine + execution)
Sharpe ratio: > 2.0 en backtest 3 meses
Max drawdown: < 1.5% por sesión
Uptime: > 99.5% (error_handler.py + sentinel.py + watchdog.py)
Recuperación fallos: < 2 segundos (state_manager.py + crash recovery)
Precisión señales: > 60% en walk-forward testing
Coverage tests: > 80% módulos críticos
🔒 DEPENDENCIAS CRÍTICAS CONOCIDAS
Módulo Consumidor Dependencia Config Impacto
BinanceLoader Config.Strategies.ML_LOOKBACK_BARS Historial necesario
RiskManager Config.Risk.MAX_DRAWDOWN Límites pérdida
TechnicalStrategy Config.Strategies.TECHNICAL_PARAMS Señales trading
Portfolio Config.Data.RESOLUTION Timeframes OHLCV
SophiaIntelligence Config.Strategies.* Decisiones IA
Evolution Config.Strategies.* Rangos mutación
StrategySelector Config.Strategies.STRATEGY_SPECIALIZATION_MAP Router horizonte
KillSwitch Config.Risk.MAX_DRAWDOWN Parada emergencia
BinanceExecutor Config.Execution.* Parámetros órdenes
🧠 CAPAS ARQUITECTÓNICAS (Bottom → Top)
┌─────────────────────────────────────────────────────────┐
│ 7. MONITOREO: dashboard/app.py + grafana/               │
├─────────────────────────────────────────────────────────┤
│ 6. SOPHIA IA: intelligence.py → nemesis.py → axioma.py  │
├─────────────────────────────────────────────────────────┤
│ 5. EVOLUCIÓN: evolution.py → genotype.py → gene_bank.py │
├─────────────────────────────────────────────────────────┤
│ 4. ESTRATEGIAS: technical → ml → statistical → sniper   │
├─────────────────────────────────────────────────────────┤
│ 3. RIESGO: risk_manager.py → kill_switch.py             │
├─────────────────────────────────────────────────────────┤
│ 2. CORE: engine.py → events.py → portfolio.py           │
├─────────────────────────────────────────────────────────┤
│ 1. DATOS: binance_loader.py → data_provider.py → DB     │
├─────────────────────────────────────────────────────────┤
│ 0. INFRA: utils/ (67 módulos) + deployment/ + hw/       │
└─────────────────────────────────────────────────────────┘
