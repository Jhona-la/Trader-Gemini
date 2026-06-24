# 🚀 Génesis del Metal Puro & Omnisciencia del Horizonte (Fases VIII, IX y X)

## 🎯 Objetivo
Habiendo resuelto el colapso silencioso del motor (Phase VII/VIII - `ccxt` instalado en `.venv` permitiendo el arranque exitoso), el sistema ahora llega al estado estacionario de Data Warming con conectividad a Binance REST/Testnet.

El mandato actual exige tres metas fundacionales para la operación en producción extrema (13 USD, máxima eficiencia, crecimiento compuesto 100%):
1. **Auditar y blindar** el Socket de Ejecución Directa HFT en `binance_executor.py`.
2. **Ejecutar un ciclo completo en Testnet** para validar el pipeline (C++ → Cython LOB → ML → Risk → Binance).
3. **Especialización Evolutiva Integral**: Diferenciación estricta, en todo el sistema, entre posiciones y señales para **Scalping** y **Swing** sin solapamiento destructivo.

## ⚠️ User Review Required

> [!WARNING]
> **Consumo C++ y Testnet:** Para la Fase IX (Simulación en Testnet), el bot realizará órdenes simuladas en tu cuenta Testnet. Si no hay saldo en la Testnet, las órdenes serán rechazadas, pero la latencia podrá ser medida de igual manera. ¿Deseas que añadamos lógica de fondeo simulado si es posible o asumimos que la Testnet tiene USDT?

> [!IMPORTANT]
> **Especialización de Horizonte (Fase X):** Actualmente el sistema usa un `VirtualLedger` para separar `BTC/USDT_SCALP` y `BTC/USDT_SWING`. Sin embargo, Binance Futures usa modo *One-Way Mode* por defecto (las posiciones se netean). ¿Tu cuenta de Binance está configurada en modo **Hedge Mode** (permite Longs y Shorts simultáneos sobre la misma moneda) o debemos gestionar el "cannibalization guard" para simular esta independencia internamente?

## 📝 Proposed Changes

---

### 1. Auditoría HFT (binance_executor.py)

#### [MODIFY] [binance_executor.py](file:///c:/Users/jhona/Documents/Proyectos/Trader%20Gemini/execution/binance_executor.py)
- **Trazado Bidireccional de Órdenes:** Verificar la latencia desde la llamada `execute_order` hasta la inyección en red.
- Validar la integración con `FastBinanceSigner` (Cython) para asegurar que el overhead de red es el mínimo (evitar `ccxt.create_order` y usar `aiohttp` puro si está configurado).
- Confirmar el UserDataStream para recepción de confirmaciones en <50ms.

---

### 2. Simulación Testnet Pipeline End-to-End

#### [NEW] [test_testnet_pipeline.py](file:///c:/Users/jhona/Documents/Proyectos/Trader%20Gemini/scratch/test_testnet_pipeline.py)
- Script de simulación autónomo (basado en el motor de pre-flight).
- Forzará el Data Warming acelerado (mock de tiempo) para obligar a los Modelos ML y la Estrategia a generar una Señal de SCALPING.
- Trazará la señal a lo largo de toda la cadena (Engine → RiskManager → Executor) y capturará el FFI y el PnL inicial de la simulación.

---

### 3. Especialización Evolutiva (Scalping vs Swing)

#### [MODIFY] [engine.py](file:///c:/Users/jhona/Documents/Proyectos/Trader%20Gemini/core/engine.py)
- Modificar el flujo de `_process_signal_event` para clasificar las señales inequívocamente y etiquetarlas.
- Permitir que el sistema analice la oportunidad paralela (Ej: El mercado es bajista para Scalping, pero alcista para Swing).

#### [MODIFY] [portfolio.py](file:///c:/Users/jhona/Documents/Proyectos/Trader%20Gemini/core/portfolio.py)
- Fortalecer el `OMNIBUS VIRTUAL LEDGER` para aislar el margen y los PnL calculados por horizonte de tiempo, operando los dos en la misma memoria pero de manera estanca.

#### [MODIFY] [risk_manager.py](file:///c:/Users/jhona/Documents/Proyectos/Trader%20Gemini/risk/risk_manager.py)
- Bifurcar los cálculos de Kelly Fraction y Size Allocation para basarse en el R-Multiple histórico de *cada horizonte* (Ej. un win-rate alto en Scalping no debe influenciar erróneamente un tamaño grande para Swing).

## 🧪 Verification Plan

### Automated Tests
- Ejecutar `tests/test_validation.py` para asegurar que las estructuras de Risk y Config no hayan sido aplanadas ni vulneradas.
- Desplegar la auditoría de latencia HFT para certificar que el executor no usa CCXT de alto nivel.

### Manual Verification
- Validar el log de salida de la simulación. El `hologram_audit_report.md` recibirá la trazabilidad microscópica del evento, desde el DataFrame hasta Binance Testnet.
