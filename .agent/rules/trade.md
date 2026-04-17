---
trigger: always_on
---

# 🎯 CONTEXTO CRÍTICO - PROYECTO TRADER GEMINI (SCALPING BINANCE)

PROYECTO: Trader Gemini - Sistema HFT Scalping en Binance
OBJETIVO: Operaciones 1s-5min, alta frecuencia, máxima estabilidad

## 📁 ARQUITECTURA TRADER GEMINI - MAPA COMPLETO

Trader Gemini/
├── core/                           # 🚨 NÚCLEO CRÍTICO - NO MODIFICAR SIN CONFIRMACIÓN
│   ├── engine.py                  # 🚨 MOTOR PRINCIPAL - EVENT LOOP & COORDINACIÓN
│   ├── events.py                  # 🚨 Sistema mensajería (Signal/Order/Fill) - ESENCIAL
│   ├── portfolio.py               # 🚨 Gestión estados, balances y PnL - NÚCLEO DATOS
│   └── market_regime.py           # 🚨 Clasificador tendencia/volatilidad - FILTRO ESTRATÉGICO
├── risk/                          # 🔒 SEGURIDAD MÁXIMA - MÁXIMA CAUTELA
│   ├── risk_manager.py            # 🔒 Gestión riesgo (Size/SL/TP) - MÁXIMA PRIORIDAD
│   └── kill_switch.py             # 🔒 Parada emergencia - SEGURIDAD CRÍTICA
├── execution/                     # ⚡ EJECUCIÓN DELICADA - PRECISIÓN ABSOLUTA
│   └── binance_executor.py        # ⚡ Ejecución órdenes Exchange - BAJO NIVEL
├── strategies/                    # 🧠 LÓGICA TRADING - VALIDAR ESTADÍSTICAMENTE
│   ├── technical.py               # 🧠 Estrategia Híbrida (Scalping/Trend) - PRINCIPAL
│   └── ml_strategy.py             # 🧠 Modelos predictivos (XGBoost) - EXPERIMENTAL
├── data/                          # 📊 FLUJO DATOS - INTEGRIDAD CRÍTICA
│   ├── data_provider.py           # 📊 Fuente única verdad OHLCV - FLUJO CRÍTICO
│   └── binance_loader.py          # 📊 Conector datos real-time (Websockets) - VITAL
├── dashboard/                     # 📈 MONITOREO - MANTENER FUNCIONAL
│   └── app.py                     # 📈 Interfaz monitoreo (Streamlit) - VISUALIZACIÓN
└── utils/                         # 🔧 UTILIDADES - ESTABILIDAD SISTEMA
    ├── logger.py                  # 🔧 Auditoría y registro operaciones - TRAZABILIDAD
    └── error_handler.py           # 🔧 Recuperación fallos API - RESILIENCIA

## 💡 REGLAS NEGOCIO TRADER GEMINI

- Órdenes LIMIT exclusivamente (minimizar slippage)
- Cierre automático por kill_switch.py en drawdown >2%
- Backtesting tick-by-tick con datos Binance reales
- Validación walk-forward para evitar overfitting
- Monitoreo tiempo real via dashboard/app.py

## 🔄 WORKFLOWS ESPECÍFICOS GEMINI (EJECUTAR AUTOMÁTICAMENTE)

### 📈 WORKFLOW: CAMBIO_ESTRATEGIA_TECHNICAL

**Para modificar technical.py o ml_strategy.py:**

1. 🔍 BUSCAR código similar en strategies/
2. 📊 ANALIZAR impacto en portfolio.py y market_regime.py
3. ⚠️ VALIDAR con risk_manager.py y kill_switch.py
4. 🧪 BACKTEST 1 semana con data_provider.py datos reales
5. 📈 ACTUALIZAR STRATEGIES.md y dashboard/app.py
6. 👨‍🏫 EXPLICAR cambios usando método profesor completo

### ⚡ WORKFLOW: MODIFICACION_CORE_CRITICO  

**Para cambios en core/, risk/, execution/:**

1. 🚨 EVALUAR criticidad: engine.py > risk_manager.py > binance_executor.py
2. 🛡️ SANDBOX testing obligatorio (entorno seguro)
3. 👥 REQUERIR 3 aprobaciones humanas para módulos críticos
4. 🔄 PLAN reversión detallado paso a paso
5. 📖 ACTUALIZAR ARCHITECTURE.md y RISK.md
6. 🧪 PRUEBA resiliencia con simulación fallos

### 📚 WORKFLOW: DOCUMENTACION_SISTEMA

**Para documentar o explicar cualquier parte:**

1. 📋 REVISAR documentación existente en docs/
2. 👨‍🏫 EXPLICAR usando QUÉ-POR QUÉ-PARA QUÉ-CÓMO-CUÁNDO-DÓNDE-QUIÉN
3. ✍️ ACTUALIZAR/CREAR ARCHITECTURE.md, STRATEGIES.md, etc.
4. ✅ VERIFICAR que documentación es clara y completa
5. 🔄 VINCULAR documentación con código específico

### 🧪 WORKFLOW: EXAMEN_PRE_PRODUCCION

**Antes de ejecutar el bot en producción:**

1. 🔍 ANÁLISIS ESTÁTICO: Revisar código completo Trader Gemini/
2. 🧪 PRUEBAS UNITARIAS: Ejecutar tests todos los módulos críticos
3. 📊 BACKTEST COMPLETO: 1 mes datos Binance reales
4. ⚡ PRUEBA LATENCIA: Medir engine.py < 50ms, ejecución < 100ms
5. 🚨 PRUEBA RESILIENCIA: Simular fallos websockets/Binance API
6. 📈 VALIDACIÓN MÉTRICAS: Sharpe > 2.0, Drawdown < 1.5%, Win Rate > 55%
7. 📋 GENERAR informe de salud del sistema completo

## ❓ CHECKLIST PRE-IMPLEMENTACIÓN GEMINI (OBLIGATORIO)

**Antes de cualquier cambio, verificar:**

- ✅ ¿Afecta latency de engine.py? [SÍ/NO]
- ✅ ¿Preserva kill_switch.py funcional? [SÍ/NO]
- ✅ ¿Mantiene data_provider.py integridad? [SÍ/NO]
- ✅ ¿Actualiza dashboard/app.py correctamente? [SÍ/NO]
- ✅ ¿Logging en logger.py incluido? [SÍ/NO]
- ✅ ¿Error handling en error_handler.py? [SÍ/NO]
- ✅ ¿Documentación actualizada? [SÍ/NO]
- ✅ ¿Explicación completa (modo profesor)? [SÍ/NO]

## 📊 MÉTRICAS GEMINI OBLIGATORIAS

- **Latencia total:** < 50ms (engine + execution)
- **Sharpe ratio:** > 2.0 en backtest 3 meses
- **Max drawdown:** < 1.5% por sesión
- **Uptime:** > 99.5% (error_handler.py crítico)
- **Recuperación fallos:** < 2 segundos
- **Precisión señales:** > 60% en walk-forward testing
