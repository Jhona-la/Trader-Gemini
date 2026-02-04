---
description: 
---

NOMBRE: gemini_detailed_workflow  
DESCRIPCIÓN: Workflow detallado específico para Trader Gemini

PASOS ESPECÍFICOS:

1. 🎯 RECONOCIMIENTO ARQUITECTURA GEMINI
   - Identificar módulos: core/, risk/, strategies/, execution/, data/
   - Módulos críticos: engine.py, risk_manager.py, binance_executor.py
   - Dependencias entre módulos y flujos de datos

2. ⚠️ EVALUACIÓN RIESGOS ESPECÍFICOS TRADING
   - Impacto PnL: potenciales pérdidas financieras
   - Latencia: mantener <50ms total sistema
   - Kill-switch: verificar funcionalidad emergencia
   - Métricas: Sharpe >2.0, Drawdown <1.5%, Win Rate >55%

3. 📊 VALIDACIONES AUTOMÁTICAS GEMINI
   - Estrategias: BACKTEST 1 semana datos Binance reales
   - Core crítico: PRUEBAS SANDBOX obligatorias
   - Producción: EXAMEN COMPLETO 1 mes + métricas
   - Latencia: medición engine.py + execution end-to-end

4. 🔗 INTEGRACIÓN MÓDULOS ESPECÍFICA
   - technical.py → market_regime.py → portfolio.py
   - data_provider.py → strategies/ → risk_manager.py
   - engine.py → todos los módulos (coordinación)

5. 📚 DOCUMENTACIÓN GEMINI COMPLETA
   - ARCHITECTURE.md: arquitectura específica Trader Gemini
   - STRATEGIES.md: explicación technical.py y ml_strategy.py
   - RISK.md: gestión riesgos y procedimientos emergencia
   - DEPLOYMENT.md: despliegue y configuración Binance

6. 👥 APROBACIÓN POR CRITICIDAD GEMINI
   - Módulos críticos (engine.py, risk_manager.py): 3 aprobaciones
   - Estrategias (technical.py): 1 aprobación + backtest exitoso
   - Datos y utils: 1 aprobación automática con pruebas

7. 🔄 IMPLEMENTACIÓN SEGURA GEMINI
   - Logging obligatorio en logger.py
   - Error handling en error_handler.py
   - Rollback automático para cambios críticos
   - Monitoreo dashboard/app.py en tiempo real
