# 🏛️ TRADER GEMINI: ENCICLOPEDIA OMNIBÚS DE ESTRATEGIAS (V7.2 - SUPREMO)

Este es el **Manuscrito Técnico Definitivo**. Ningún parámetro de `config.py` o lógica de `technical.py` queda fuera. Este documento fusiona la visión macro-orgánica con los detalles microscópicos necesarios para duplicar $13 USD.

---

## 🏛️ I. ARQUITECTURA DEL ORGANISMO INTEGRAL

El sistema opera mediante una jerarquía de 7 capas que procesan datos desde el hardware hasta la explicación humana:

1.  **Infraestructura (`Aegis`)**: Optimización de hardware, afinidad de CPU y aceleración AVX2.
2.  **Sensores (`DataProvider`)**: Ingesta de datos crudos (OHLCV) y normalización.
3.  **Sinapsis (`NeuralBridge`)**: Construcción del Tensor de Estado Unificado.
4.  **Instintos (`Execution Strategies`)**: Lógica competitiva de Scalping, Swing y Sniper.
5.  **Conciencia (`Meta-Brain`)**: Selección de estrategia y alocación `Anti-Whipsaw`.
6.  **Oráculo (`Reasoning Oracle`)**: Atribución de Skill vs Luck y razonamiento causal.
7.  **Némesis (`Adversarial Feedback`)**: El fiscal que audita y deconstruye cada fallo.

---

## ⚙️ II. CONFIGURACIÓN MAESTRA DE INFRAESTRUCTURA

### ⚡ Protocolo Aegis-Ultra (Hardware)
Configuraciones en `config.py` para latencias nano-segundo:
- `Aegis.CORE_PINNING = True`: Fija los hilos del bot a núcleos físicos específicos para evitar context switching.
- `Aegis.USE_AVX2 = True`: Habilita vectorización matemática pesada en los kernels JIT.
- `Aegis.ZERO_COPY_DATA = True`: Acceso directo al RingBuffer de datos sin copias en memoria.

### 📡 Observabilidad y Alertas
- **Telegram/Email**: Notificaciones en tiempo real para cambios de PnL y errores críticos.
- `ALERT_MAX_DRAWDOWN = 0.05`: Alerta visual si el drawdown total toca el 5%.
- `ALERT_MIN_SHARPE = 1.2`: Notificación si la eficiencia de la sesión cae de niveles institucionales.

---

## ⚙️ III. CONFIGURACIÓN GLOBAL DE TRADING ($13 USD)

| Parámetro | Valor | Justificación Técnica |
| :--- | :--- | :--- |
| `INITIAL_CAPITAL` | $13.0 | Capital de inicio para micro-scalping. |
| `BINANCE_LEVERAGE` | 10x | Garantiza que $3.90 de margen alcancen el mínimo de $5 de Binance Futures. |
| `POS_SIZE_MICRO` | 30% | Aloca ~$3.90 por trade, permitiendo 2 operaciones concurrentes. |
| `MAX_RISK_TRADE` | 5.0% | Tolerancia de pérdida máxima por señal sobre el capital total. |
| `MAX_SLIPPAGE` | 0.1% | Límite para órdenes LIMIT Post-Only para evitar pérdida por spread. |

---

## 🧠 IV. ESTRATEGIA TÉCNICA HÍBRIDA (`technical.py`) - MICRO-LÓGICA

Esta estrategia es el motor principal. Aquí se detalla la lógica de puntuación y adaptabilidad.

### 1. Sistema de Puntuación de Confluencia
Para que se dispare una señal, el sistema suma puntos según el estado de múltiples timeframes:
- **Base Score**: `MeanReversion (+0.6)` o `Momentum (+0.5)`.
- **Tendencia Confirmada**: `+0.3` si EMA Fast > Slow y Precio > EMA Trend.
- **RSI de Memoria**: `+0.2` si RSI está en zona de equilibrio (40-60).
- **RSI Extremo**: `+0.4` si RSI > 70 o < 30 (Dispara Mean Reversion).
- **Volume Ratio**: `+0.2` si el volumen actual es > 1.2x la media móvil.
- **Umbral de Calidad**: La señal solo se emite si el Score Total > **0.40 - 0.55** (adaptativo).

### 2. DPE: Dynamic Parametric Evolution (DPE)
El bot no usa RSI 30/70 fijos, sino que los recalcula dinámicamente:
- **RSI Dinámico**: Usa los **percentiles 15 y 85** de las últimas 200 velas. El mercado define qué es "sobrevendido".
- **ADX Dinámico**: Usa el `Mean(ADX) + 0.5 * StdDev(ADX)`. El mercado define qué es "tendencia".

### 3. Escalado de Riesgo Asimétrico (ATR-Scaling)
El Stop Loss (SL) y Take Profit (TP) no son fijos, dependen de la volatilidad:
- **Base Multiplier**: `ATR * 1.5 (SL)` / `ATR * 3.0 (TP)`. (Variables según Scalping/Swing).
- **VolRatio Multiplier**: Si la volatilidad actual es 1.2x la media, el SL se amplía un **20%** para evitar mechas falsas.
- **Regime Multiplier**: En mercados `CHOPPY`, el SL se cierra un **25%** para proteger el capital.

### 4. Definición Quirúrgica de Setups
- **Mean Reversion**: Mecha en Banda Bollinger + RSI Extremo + Volumen Alto.
- **Proximity Scalping**: Especial para $13. BB position < 25% + RSI "leaning" (<45) + Volumen moderado. Es más permisivo.
- **Momentum (VCP)**: `Volatility Contraction Pattern`. Expansión de bandas + Aceleración MACD + ADX > 20.

---

## 🤖 V. ESTRATEGIA ML XGBOOST (`ml_strategy.py`)

- **Conjunto de 25 Modelos**: Un modelo entrenado específicamente para cada activo del basket.
- **Engine V5.10**: Optimiza 100+ features divididos en Momentum, Volatilidad y Flujo de Volumen.
- **Retraining cada 240 velas**: Los modelos "aprenden" mientras operan para no quedar obsoletos.
- **Confidence Oracle**: Requiere un margen de beneficio proyectado del **1.5%** por el modelo de IA antes de autorizar el trade.

---

## 🎯 VI. ESTRATEGIA SNIPER HFT (`sniper.py`)

- **Mapa de Agresión por Régimen**:
    - `TRENDING_BULL`: 8x Leverage | Aggressive Threshold (-0.05).
    - `CHOPPY`: 1x Leverage | Caution Threshold (+0.05).
    - `ZOMBIE`: 1x Leverage | No Trade Threshold (+1.0).
- **Order Flow Depth**: Analiza los primeros 20 niveles del libro de órdenes buscando desequilibrios del 30% (`Imbalance`).

---

## 💾 VII. ESTADO COGNITIVO (COGNITIVE MEMORY)

El bot recuerda su desempeño reciente por activo y setup:
- **ALPHA STATE**: Si el activo tiene una racha ganadora, el bot desbloquea **ALL_SETUPS** y aumenta la agresividad.
- **INJURED STATE**: Si el activo ha perdido recientemente, el bot bloquea setups débiles y solo permite entradas con **"IA Brutal"** (Score > 0.75).
- **NORMAL STATE**: Comportamiento estándar según perfiles.

---

## 🔢 VIII. CIMIENTOS Y ESTADÍSTICA (MATH PROOFS)

- **Hurst Exponent**: Cálculo de persistencia en 20 velas. Gemini filtra señales donde H está entre 0.45 y 0.55 (ruido blanco).
- **RANSAC Robustness**: Los cálculos de tendencia y bandas se realizan descartando el 25% de los datos que son ruido o manipulación de mercado.
- **Brier Audit**: Puntuación de calibración de confianza. Si el bot gana pero predijo con baja confianza, Némesis lo penaliza como "Luck".

---

## 💰 IX. MICRO-ECONOMÍA DE SUPERVIVENCIA ($13 USD)

Para duplicar capital cada 15 días:
- **Kelly Adaptativo (0.3)**: Usamos el 30% de la fracción de Kelly óptima para evitar la ruina estadística.
- **Fee Optimization**: El bot prefiere órdenes `LIMIT` (Post-Only). Sabe que el 0.05% de comisión adicional de órdenes `MARKET` destruiría la cuenta rápidamente.
- **Slippage Forensic**: Cada trade es auditado. Si el spread real supera el 0.05%, el bot anula la operación antes de entrar.

---
**Tratado Omnibús de Estrategias y Configuraciones Trader Gemini V7.2 - EL MANUAL SUPREMO**
**"Integración total, de lo nano a lo macro."**
