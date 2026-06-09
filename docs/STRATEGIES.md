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
---
**Tratado Omnibús de Estrategias y Configuraciones Trader Gemini V7.2 - EL MANUAL SUPREMO**
**"Integración total, de lo nano a lo macro."**

---

## 📈 X. EVOLUCIÓN Y REACTIVACIÓN DEL DOBLE HORIZONTE (FASE 3 OPTIMIZACIÓN)

El bot implementa un motor de **Doble Horizonte de Operación** que ejecuta estrategias de **Scalping** y **Swing** de forma adaptativa, evolutiva e integral sin interrumpirse ni anularse mutuamente:

### 1. Dinámica Intradía de Scalping (HFT)
* **Objetivo:** Capturar micro-movimientos rápidos y capitalizar la cuenta con interés compuesto exponencial en días.
* **Calibración:** TP de 0.45% y SL de 0.30% (relación Payoff 1.5x) con un máximo de 45 barras de retención.
* **Consistencia:** Protegido contra el Fee Drag mediante órdenes LIMIT Post-Only y un stop zombie ultrarrápido de 60 minutos.

### 2. Dinámica Estructural de Swing (Tendencial)
* **Objetivo:** Capturar tendencias macro en marcos de tiempo de 1h y 4h.
* **Calibración:** TP de 4.5% y SL de 2.5% con una duración máxima de 96 barras.
* **Reactivación de Señal:** Se redujo el `strength_threshold` (umbral de confluencia) de 0.55 a **0.45** en `Horizons.Swing`. Esto permite que el sistema emita señales Swing válidas en momentos óptimos del ciclo de mercado sin ser bloqueadas por el sobre-filtrado de consenso.
* **Aislamiento Total:** El silo del 70% del margen para Scalping y 30% para Swing garantiza que el bot pueda operar ambos motores en paralelo y en ambas direcciones (LONG/SHORT) simultáneamente por símbolo, sin confusión ni pisadas de pies en el Ledger Virtual.

---

## 🔬 XI. COMPLIANCE NEXUS: FLUJO DE SEÑALES INMUTABLES Y MÁQUINA DE ESTADOS (MODO PROFESOR)

Para cumplir con la especificación de integridad de la arquitectura **NEXUS**, el ciclo de vida de los eventos de señal ha sido rediseñado para garantizar inmutabilidad absoluta y trazabilidad total:

### 1. El Ciclo de Vida del Evento de Señal
- **QUÉ**: Es una máquina de estados determinista aplicada a cada evento de señal (`SignalEvent`). Los estados válidos son: `GENERATED` → `EVALUATING` → `APPROVED` / `REJECTED` / `BLOCKED_COLLISION` / `EXPIRED` → `EXECUTED`.
- **POR QUÉ**: Previamente, las señales se enviaban al bus de eventos y eran modificadas sobre la marcha (en caliente) por el motor o árbitro. Esto violaba la inmutabilidad de los dataclasses congelados (`frozen=True`) produciendo caídas de ejecución críticas (`FrozenInstanceError` o `AttributeError`).
- **PARA QUÉ**: Garantizar la reproducibilidad matemática de las decisiones de trading, depurar fallos en auditorías forenses y mantener el orden del flujo del sistema al congelar el estado físico de los datos.
- **CÓMO**: `SignalEvent` se define como un dataclass congelado en `core/events.py`. Cuando un componente necesita actualizar propiedades (ej. `state`, `tension`, `strength`), debe invocar `dataclasses.replace(event, **updates)` para generar y retornar una nueva copia inmutable mutada, en lugar de modificar la existente.
- **CUÁNDO**: Se activa en el bucle principal de procesamiento del Engine durante cada etapa del pipeline de arbitraje y ejecución.
- **DÓNDE**:
  - Definición de estados: `core/events.py` (clase `SignalState`).
  - Origen del evento: `strategies/technical.py` o `strategies/ml_strategy.py` (`GENERATED`).
  - Evaluación y filtrado: `core/engine.py` y `core/meta_arbitrator.py` (`EVALUATING` → `APPROVED` / `REJECTED`).
  - Ejecución de órdenes: `risk/risk_manager.py` (`APPROVED` → `EXECUTED`).
- **QUIÉN**: El componente `core/engine.py` (Engine Loop) coordina el flujo, apoyado por `MetaArbitrator` para el veto y `RiskManager` para la conversión a orden firme.

### 2. Trazabilidad y Snapshots del Estado Neuronal
- **QUÉ**: Es la inyección obligatoria de metadatos de auditoría en cada señal, incluyendo un `prediction_id` único, un `features_snapshot` (los valores de los indicadores al momento de disparar) y un `namespace` formateado estrictamente como `SIGNAL::{STRATEGY}::{ASSET}::{DIRECTION}::{TIMESTAMP}::{HASH}`.
- **POR QUÉ**: Para evitar sesgos de anticipación (Look-Ahead Bias) y simular con precisión de nanosegundo qué información exacta tenía el bot cuando decidió entrar al mercado.
- **PARA QUÉ**: Validar el poder predictivo del sistema de 90+ indicadores, verificar que las señales se alinean con la realidad histórica del backtest y prevenir colisiones operativas.
- **CÓMO**: Durante la fase `__post_init__` del `SignalEvent`, se genera un identificador único SHA-256 a partir del timestamp y la estrategia, y se consolida el namespace inmutable de forma automática.
- **CUÁNDO**: En el instante preciso de la creación de la señal en la capa de la estrategia.
- **DÓNDE**: En el constructor `SignalEvent` (`core/events.py`).
- **QUIÉN**: La clase `SignalEvent` autogestiona su inicialización de metadatos de integridad.

### 3. Aceleración y Optimización de Latencia para Fused Path (Fase 60)
- **QUÉ**: Bypass y simplificación selectiva del procesamiento de ticks en `HybridScalpingStrategy` cuando se opera bajo el modo de Red Neuronal Directa (`use_fused_path`).
- **POR QUÉ**: El cálculo de múltiples marcos de tiempo (MTF) e indicadores técnicos redundantes, junto con la inferencia de Sophia, generaban un cuello de botella de `5.2 ms` de latencia por símbolo, impidiendo la ejecución HFT sub-1ms requerida para scalping institucional.
- **PARA QUÉ**: Alcanzar latencias inferiores a `800 μs` por símbolo y jitter promedio de `100 μs` para asegurar una respuesta libre de deslizamiento (slippage) y 100% determinista.
- **CÓMO**: 
  - Almacenar los perfiles de parámetros en un caché por símbolo (`self._symbol_params_cache`) copiado de forma rápida para eliminar la consolidación redundante en cada tick.
  - Verificar `use_fused_path` en las primeras líneas de `generate_signals`, obligando a `get_multi_timeframe_data` a computar únicamente el marco temporal primario (`primary_only=True`).
  - Omitir de forma total la ejecución pesada de `sophia.analyze` para señales de tipo `FUSED_PATH`, protegiendo el resto de flujos internos con comprobaciones condicionales robustas contra valores nulos en el reporte.
- **CUÁNDO**: Se evalúa y activa en cada tick de datos durante la ingesta del `MarketEvent` en la ruta caliente del event loop.
- **DÓNDE**: En la clase `HybridScalpingStrategy` en [strategies/technical.py](file:///c:/Users/jhona/Documents/Proyectos/Trader%20Gemini/strategies/technical.py).
- **QUIÉN**: Coordinado de forma asíncrona por el event loop e implementado en la lógica de confluencia y motor evaluador de señales de la estrategia técnica.

---

## 📊 XII. MOTOR DE CALIBRACIÓN MASIVA PARALELA Y PERFILES DINÁMICOS (MODO PROFESOR)

Para maximizar el crecimiento compuesto sobre el capital de $13 USD y evitar stops estáticos ineficientes, el sistema integra calibraciones óptimas por horizonte temporal consumidas dinámicamente desde `optimal_profiles.json`:

### 1. Ingesta de Perfiles de Calibración
- **QUÉ**: Es la carga dinámica de parámetros calibrados (ATR multipliers de entrada, R:R ratios, veto thresholds y strength thresholds) guardados en `optimal_profiles.json`.
- **POR QUÉ**: Previamente, los parámetros de SL y TP eran estáticos en `config.py` y compartidos por igual para todas las monedas. Esto no permitía adaptarse a las características y correlación de cada activo en particular (ej. SOL es más volátil que BTC).
- **PARA QUÉ**: Duplicar el capital de $13 USD de manera exponencial en el menor tiempo posible adaptando el stop-loss y el take-profit al rango de oscilación dinámico particular de cada activo.
- **CÓMO**:
  1. Durante la instanciación de la estrategia (`HybridScalpingStrategy`), se lee `optimal_profiles.json` y se parsea en un diccionario de perfiles.
  2. En `get_symbol_params()`, si el símbolo y el horizonte (Scalping/Swing) existen en el JSON, se sobrescriben dinámicamente los parámetros de base (`strength_threshold`).
  3. Al calcular los stops dinámicos en `_calculate_dynamic_risk_params()`, se inyectan los multiplicadores de stop-loss (`sl_atr`) y take-profit (`tp_atr`) específicos calibrados.
- **CUÁNDO**: Ocurre en tiempo real durante la evaluación de confluencia y generación de señales en cada tick, y al momento de actualizar los stops de las posiciones.
- **DÓNDE**: Carga del JSON y overrides en [technical.py](file:///c:/Users/jhona/Documents/Proyectos/Trader%20Gemini/strategies/technical.py) y [asset_parameter_engine.py](file:///c:/Users/jhona/Documents/Proyectos/Trader%20Gemini/core/asset_parameter_engine.py).
- **QUIÉN**: La clase `HybridScalpingStrategy` e `AssetParameterEngine`.

### 2. Sincronización del Exit Oracle (Sophia Veto Thresholds)
- **QUÉ**: Ajuste del umbral de veto dinámico (`veto_threshold`) en `SophiaExitOracle` utilizando los resultados de la simulación óptima por símbolo y horizonte.
- **POR QUÉ**: El exit oracle veta las salidas o entradas si la predicción de red neuronal o la volatilidad indica un régimen adverso. El umbral estático de veto a veces bloqueaba buenas salidas técnicas.
- **PARA QUÉ**: Permitir salidas tempranas adaptadas al nivel de ruido y dirección de cada moneda en su respectivo horizonte temporal.
- **CÓMO**: El oráculo de salida `evaluate_exit()` lee el archivo de perfiles y, si existe un `veto_threshold` calibrado para ese símbolo/horizonte, lo adopta como umbral de decisión, reemplazando el valor por defecto.
- **CUÁNDO**: En cada llamada de evaluación de salida técnica para las posiciones abiertas.
- **DÓNDE**: En la clase `SophiaExitOracle` dentro de [exit_oracle.py](file:///c:/Users/jhona/Documents/Proyectos/Trader%20Gemini/sophia/exit_oracle.py).
- **QUIÉN**: `SophiaExitOracle`.

