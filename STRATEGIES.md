> ⚠️ **NOTA F0 (2026-08-16):** Las referencias a `technical.py` / `ml_strategy.py` corresponden a la era Python (archivada). Las estrategias vivas están en Rust: `crates/strategy-core`, `crates/god-engine-core` (scalp+swing), `crates/signal-engine`. Documentación viva: `docs/INVENTORY.md`.

# 🧠 TRADER GEMINI: DOCUMENTACIÓN DE ESTRATEGIAS (STRATEGIES.md)
**Documento Maestro de Orquestación Algorítmica y Señales**

## 1. PROPÓSITO
Este documento detalla la estructura, flujo y métricas de las dos estrategias fundamentales del sistema: `TechnicalStrategy` (`technical.py`) y `MLStrategy` (`ml_strategy.py`). Ambas operan bajo los paradigmas de **Performance Cuántica** y **Consciencia de Horizonte (Scalping vs Swing)**.

---

## 2. TECHNICAL STRATEGY (`technical.py`)

### QUÉ ES
Un motor de análisis técnico de alta velocidad que emite señales basadas en confluencias clásicas de indicadores matemáticos.
### POR QUÉ SE USA
Para capturar expansiones de volatilidad y micro-reversiones en tiempos donde la red neuronal (ML) aún está convergiendo o en regímenes donde la acción del precio es estrictamente algorítmica.
### PARA QUÉ SIRVE
Actúa como un cazador de anomalías estadísticas (Bandas de Bollinger, RSI Estocástico, MACD). 
### CÓMO FUNCIONA
1. **Extracción de Features:** Recibe el último vector de precios desde `feature_engine`.
2. **Evaluación Multicapa:**
   - *Filtro de Tendencia:* Evalúa cruces de EMAs (20 vs 50 vs 200).
   - *Filtro de Momentum:* Revisa la compresión del RSI.
   - *Filtro de Volatilidad:* Mide el ancho de Bollinger y el ATR.
3. **Validación de Horizonte:** Ajusta los umbrales dependiendo si el horizonte activo es SCALPING o SWING.
### CUÁNDO SE ACTIVA
En cada tick (epoch) del motor si no está bloqueada por el `RiskManager` (ej. en modo "Alta Correlación").
### QUIÉN LO MANEJA
El `OmniscientEngine` y el `MetaCoordinator`, quienes combinan su puntaje de confianza con el modelo ML a través del `ConsensusFilter`.

---

## 3. ML STRATEGY (`ml_strategy.py`)

### QUÉ ES
El cerebro predictivo central de Trader Gemini. Un ensamble de clasificadores y regresores Gradient Boosting (XGBoost/LightGBM) que predice la Magnitud Máxima Favorable (MFE).
### POR QUÉ SE USA
Para superar el *lag* (retraso) inherente a los indicadores técnicos. Los modelos predictivos pueden inferir una ruptura direccional basándose en la topología de la micro-estructura del mercado antes de que las EMAs se crucen.
### PARA QUÉ SIRVE
Para proporcionar un "Puntaje de Convicción" (Confidence) y predecir el Take Profit dinámico exacto de una operación (Magnitud).
### CÓMO FUNCIONA (PERFORMANCE CUÁNTICA)
1. **Extracción (GIL-Free):** Usa `df.to_numpy()[-1].reshape(1, -1)` en lugar de métodos Pandas (eliminado en Phase 23-32) para evitar el *overhead* de validación de objetos en la ruta crítica.
2. **Predicción Dual:**
   - `xgb_regressor_long`: Infiere la excursión alcista esperada.
   - `xgb_regressor_short`: Infiere la excursión bajista esperada.
3. **Alineación de Horizonte:** Emite una señal con un `tp_target` y `sl_target` calibrado dinámicamente según la certeza del modelo.
4. **Optimización de Memoria Asíncrona:** A diferencia de arquitecturas tradicionales que usan `gc.collect()` sincrónico, Gemini confía en la recolección Gen0 de Python para no bloquear el GIL durante los milisegundos críticos.
### CUÁNDO SE ACTIVA
Se alimenta con matrices pre-procesadas por el `FeatureEngineer`. Sólo emite señales cuando la confianza cruza los umbrales dictaminados por el `AssetParameterEngine` (típicamente > 65% para Scalping).
### QUIÉN LO MANEJA
Invocado asincrónicamente por el motor de inferencia principal y auditado por el `PredictionTracker` para evitar la degradación del modelo a lo largo del tiempo.

---

## 4. OMNISCORE (Consensus Filter)
Ambas estrategias alimentan al `ConsensusFilter`. 
Si ML grita "SHORT" con 80% de confianza, pero Technical grita "LONG" con 90% (Divergencia Severa), el sistema penaliza la señal y exige protección adicional (Stop Loss más corto o cancelación total de la orden). El objetivo no es tener la razón siempre, es **garantizar la supervivencia matemática (Crecimiento Compuesto del 100% cada 3 días)** limitando las entradas a escenarios de altísima confluencia.

---

## 5. ESTRATEGIAS CUÁNTICAS NATIVAS EN RUST (`signal-engine` & `strategy-core`)

En la arquitectura moderna en Rust, las estrategias se organizan en dos motores de alta fidelidad sin retrasos de intérprete:

### 5.1 Ensamble Cuántico Tensorial (`crates/signal-engine`)
1. **`CoaxialBreakout`:** Detección de rupturas de volatilidad multicanal con bandas elásticas en micro-segundos.
2. **`QuantumOscillator`:** Superposición de probabilidades y fuerza restaurativa armónica para reversiones medias.
3. **`SolitonWave`:** Modelado no lineal de paquetes de energía en el libro de órdenes (ondas solitónicas).
4. **`SupersonicShockwave`:** Detección de discontinuidades de Mach en la propagación de órdenes en el libro L2.
5. **`StochasticResonance`:** Amplificación de señales débiles sumergidas en el ruido blanco del mercado.
6. **`HawkesBessel`:** Procesos puntuales auto-excitados para cuantificar la intensidad de clusters de trades.
7. **`RenyiTsallisEntropy`:** Medición de entropía no extensiva para detectar transiciones de fase y regímenes caóticos.
8. **`GameTheoreticNash`:** Equilibrio Minimax contra algoritmos predadores institucionales y bots de arbitraje.
9. **`PerceptronGate`:** Puerta perceptrónica hebbiana con actualización adaptativa de pesos por trade.
10. **`TurboScalper`:** Señalador HFT basado en desbalance de volumen delta e intensidad direccional de ticks.
11. **`MicroScalpTrigger`:** Inferencia conforme para entradas de alta probabilidad en ventanas de segundos.
12. **`SwingConformalFilter`:** Filtrado de ruido intradiario para posiciones multihour/multiday en tendencias macro.
13. **`TrendRunner`:** Expansión dinámica del Take Profit en regímenes de persistencia de Hurst ($H > 0.65$).

### 5.2 Arbitraje Estadístico y Modelos Macro (`crates/strategy-core`)
1. **`MultivariateCointegration`:** Detección de cointegración multiactivo con ajuste de reversión Ornstein-Uhlenbeck.
2. **`JohansenVecm`:** Modelo de Corrección de Error Vectorial con beta dinámico adaptativo para pares líderes/rezagados.
3. **`StatArbEngine`:** Arbitraje de divergencias con reversión rápida a la media en ventanas O(1).
4. **`ScalpEngine`:** Microestructura pura con filtro de toxicidad VPIN y aceleración OBI.
5. **`SwingEngine`:** Cruces EWMA y bandas de desviación estándar con filtros bayesianos de confluencia.

