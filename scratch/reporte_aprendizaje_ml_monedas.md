# Reporte Forense de Adaptabilidad, Aislamiento y Evolutividad de Modelos de ML - Trader Gemini

## 📋 Introducción y Contexto Estratégico
Este reporte detalla los hallazgos de la auditoría forense y de diseño realizada sobre las estrategias de Machine Learning (`strategies/ml_strategy.py`), el rastreador de predicciones (`core/prediction_tracker.py`), el motor de aprendizaje en línea (`core/online_learning.py`) y la gobernanza de modelos (`core/ml_governance.py`) en la arquitectura Trader Gemini.

El objetivo principal de esta auditoría es garantizar que los modelos de Machine Learning y los mecanismos de feedback evolutivo respeten las idiosincrasias y regímenes particulares de cada criptoactivo (BTC, ETH, SOL, XRP, etc.) de forma aislada, evitando la **dilución del poder predictivo** causada por la mezcla global de datos. Esto es crítico para lograr nuestro objetivo: **operar con un capital de $13 USD, maximizar el Win Rate en Scalping/Swing y duplicar el capital exponencialmente cada 15 días mediante crecimiento compuesto.**

---

## 🔍 Hallazgo 1: Dilución Global de Métricas y Veto Colateral en PredictionTracker
### 👨‍🏫 Método Profesor
*   **QUÉ:** La clase `PredictionTracker` calcula la precisión direccional (`direction_accuracy`) y los parámetros de ejecución (`get_execution_params`) de manera **global por horizonte**, mezclando las señales de todas las monedas bajo una misma estrategia.
*   **POR QUÉ:** Las instancias de `MLStrategy` para todas las monedas comparten el mismo identificador de estrategia (`strategy_id`). El identificador se construye en `strategies/ml_strategy.py` como `self.strategy_id = f"{lbl}_{base_label}_{self.horizon_str}"`, lo cual genera cadenas fijas como `"[SCL]_ML_HYBRID_ULTIMATE_V2_SCALPING"` para scalping y `"[SWG]_ML_HYBRID_ULTIMATE_V2_SWING"`. Al llamar a `PredictionTracker.record_signal(strategy_id=self.strategy_id, symbol=self.symbol, ...)` y calcular las métricas agregadas en `_refresh_metrics`, el sistema agrupa los resultados utilizando únicamente la clave `strategy_id`.
*   **PARA QUÉ:** Evitar que monedas altamente volátiles o ruidosas (por ejemplo, SOL o XRP en momentos de shock) diluyan las métricas de activos más predecibles como BTC, evitando dos fallos críticos:
    1.  **Fuga de Reclamos (Falsos Negativos):** Si una moneda experimental tiene mal desempeño y baja la precisión global de scalping por debajo del 55%, el gate `should_reject_signal` vetará automáticamente las señales de **todas las monedas** (incluido BTC), provocando un deadlock operativo.
    2.  **Falsos Positivos:** Si BTC tiene un rendimiento excelente y mantiene la métrica global por encima del 55%, las señales deficientes de una moneda inestable no serán vetadas por el gate de la estrategia, lo que resultará en pérdidas financieras para cuentas de capital reducido.
*   **CÓMO:** En `core/prediction_tracker.py`, el método `_refresh_metrics` recorre `self._signals.items()`. Dado que `self._signals` es un diccionario indexado por `strategy_id` (que no incluye el símbolo), el rastreador acumula señales de múltiples activos en el mismo `ring`. Las métricas consolidadas (accuracy, MFE, MAE) se promedian colectivamente. El método `should_reject_signal(strategy_id, horizon)` no recibe el parámetro `symbol`, lo que imposibilita la evaluación por activo.
*   **CUÁNDO:** Se ejecuta dinámicamente en el arranque del sistema (`_refresh_metrics` inicial) y cada vez que el RiskManager genera una orden o evalúa si debe vetar una señal de entrada.
*   **DÓNDE:** En `core/prediction_tracker.py` (Línea 408 y Línea 553) y en la inicialización de `MLStrategy` en `strategies/ml_strategy.py` (Línea 347).
*   **QUIÉN:** El `PredictionTracker` en coordinación con `MLStrategy` y `RiskManager`.

> [!WARNING]
> **Impacto cuantitativo directo:** Mezclar la volatilidad de XRP/USDT con la de BTC/USDT en el cálculo del `limit_offset_pct` genera órdenes límite pasivas ineficientes para BTC (perdiendo fills clave) y órdenes demasiado agresivas para XRP (sufriendo un slippage excesivo).

---

## 🔍 Hallazgo 2: Mezcla de Transiciones Globales en el Singleton PPOAgent
### 👨‍🏫 Método Profesor
*   **QUÉ:** El motor de aprendizaje por refuerzo `ppo_agent` es un **singleton global único** que procesa experiencias de manera indiferenciada entre monedas y horizontes, impidiendo que la IA aprenda políticas de tamaño de posición adaptadas a cada activo.
*   **POR QUÉ:** En `ml/ppo_agent.py`, el agente se instancia como un singleton global: `ppo_agent = PPOAgent()`. Además, el vector de estado de 15 dimensiones utilizado en la inferencia en `ml_strategy.py` (líneas 2134-2143) solo incluye métricas técnicas generales y de régimen de mercado, **omitiendo por completo cualquier etiqueta de identificación de la moneda o del horizonte (scalping/swing).**
*   **PARA QUÉ:** Evitar que las transiciones de recompensa de un activo muy dinámico y con bajas comisiones alteren la política de agresividad de activos lentos y caros, optimizando el sizing y la consistencia matemática del Sharpe Ratio por activo de manera independiente.
*   **CÓMO:** Durante la resolución de cada trade en `MLStrategy.update_recursive_weights()`, se registra la experiencia en `self.ppo_memory`. Posteriormente, se inicia un hilo secundario que ejecuta `_learn_ppo_batch()`, llamando a `ppo_agent.update()` con estados, acciones y recompensas recopilados localmente. Al entrenar todos los hilos sobre la misma red global, los gradientes de múltiples monedas colisionan y sobrescriben continuamente el comportamiento del modelo, creando amnesia catastrófica inter-activos.
*   **CUÁNDO:** Durante el procesamiento del evento `FILL` en el motor de ejecución, específicamente en la fase de aprendizaje en batch del PPO.
*   **DÓNDE:** Ubicado en `ml/ppo_agent.py` y en `strategies/ml_strategy.py` (Líneas 1713-1724 y 2127-2146).
*   **QUIÉN:** El submódulo PPO del Quant Developer y el Arquitecto Senior.

---

## 🔍 Hallazgo 3: Aislamiento Correcto en Modelos Base Supervisados y Aprendizaje Online Recursivo
### 👨‍🏫 Método Profesor
*   **QUÉ:** A diferencia del PPO y el PredictionTracker, los modelos supervisados de base (RandomForest, XGBoost, GradientBoosting) y el ajuste recursivo de sus pesos del ensamble funcionan en **aislamiento estricto por moneda**.
*   **POR QUÉ:** La arquitectura de Trader Gemini instancia una clase `MLStrategy` por cada símbolo y horizonte. Cada instancia carga sus propios archivos serializados desde `.models/{symbol}_{horizon}/` y mantiene un objeto `OnlineLearner` local.
*   **PARA QUÉ:** Garantizar que los patrones de precios y los pesos asignados a cada modelo de la trinidad (RF, XGB, GB) respondan puramente al comportamiento histórico de cada activo, protegiendo las decisiones de entrada y salida contra correlaciones espurias del mercado global.
*   **CÓMO:**
    1.  `MLGovernance` actúa como gatekeeper utilizando SQLite para almacenar registros vinculados a la columna `symbol`. Las rutas de producción son específicas de cada moneda (ej: `.models/BTC_USDT_v1_20260607`).
    2.  `OnlineLearner` actualiza los atributos individuales `self.base_rf_weight`, `self.base_xgb_weight` y `self.base_gb_weight` basándose en el historial de PnL local del portfolio para ese símbolo específico.
*   **CUÁNDO:** Se activa de forma continua en cada iteración de predicción (inferencia aislada) y al cierre de cada operación en mercados activos (actualización de pesos online).
*   **DÓNDE:** Gestionado en `core/ml_governance.py` y ejecutado en `strategies/ml_strategy.py` (Líneas 1557-1710).
*   **QUIÉN:** El Arquitecto Senior y la clase `MLGovernance`.

---

## 🔍 Hallazgo 4: Ausencia de Identificación de Horizonte Temporal en la Política del PPO
### 👨‍🏫 Método Profesor
*   **QUÉ:** El vector de características de entrada del `ppo_agent` no incluye información sobre el horizonte temporal (Scalping vs. Swing), forzando al modelo a aprender una única política de dimensionamiento de posición para objetivos de rentabilidad diametralmente opuestos.
*   **POR QUÉ:** El array de características `ppo_state` (líneas 2134-2143) incluye probabilidades del ensamble, RSI, ADX, Hurst y el régimen de mercado (Trending, Volatile, Ranging), pero **no contiene ninguna bandera o codificación para `horizon`**.
*   **PARA QUÉ:** Asegurar que el PPO entienda que una operación de scalping busca capturar micro-movimientos rápidos con apalancamiento alto (requiriendo un sizing estricto y un veto ágil), mientras que una operación de swing tolera retrocesos mayores para capturar tendencias macro.
*   **CÓMO:** El modelo PPO predice una acción continua $a \in [-1, 1]$ que modula la agresividad del sizing. Al no saber si está operando en un marco temporal de 1 minuto (scalping) o de 4 horas (swing), el PPO tiende a promediar sus predicciones, generando sizing subóptimo en ambos horizontes.
*   **CUÁNDO:** Ocurre durante la fase de toma de decisiones antes de propagar la señal al RiskManager.
*   **DÓNDE:** En `strategies/ml_strategy.py` (Línea 2134).
*   **QUIÉN:** El Quant Developer y el Risk Manager.

---

## 🔍 Hallazgo 5: Incompatibilidad de Firmas y Excepciones Silenciosas en el Algoritmo de Aprendizaje Online (update_ppo_batch)
### 👨‍🏫 Método Profesor
*   **QUÉ:** Hay una incompatibilidad en la firma del método `update_ppo_batch` entre su definición en `core/online_learning.py` y sus invocaciones en `strategies/ml_strategy.py`, lo que causa que cualquier intento de entrenamiento por refuerzo online de los pesos del ensamble (Omega Mind) falle con un `TypeError` silencioso.
*   **POR QUÉ:** 
    1. En `strategies/ml_strategy.py` (Línea 1700), se llama a `self.online_learner.update_ppo_batch` pasando el argumento `returns=rewards`, pero la firma del método en `core/online_learning.py` (Línea 320) define el parámetro como `rewards`, no `returns`.
    2. En `strategies/ml_strategy.py` (Línea 5474), se llama al mismo método pasando `next_states=next_states` y `dones=dones`, parámetros que no existen en la definición del método en `core/online_learning.py`.
    Dado que ambas invocaciones están envueltas en bloques `try-except` genéricos que solo registran el error en los logs, el fallo ocurre de forma silenciosa sin detener el bot, pero anulando por completo la retroalimentación de aprendizaje online para los pesos del ensamble.
*   **PARA QUÉ:** Asegurar que el ajuste adaptativo de los pesos de la trinidad de modelos (RF, XGB, GB) funcione de forma continua en producción, permitiendo al sistema re-priorizar dinámicamente los pesos basándose en la rentabilidad de las últimas operaciones sin errores de ejecución.
*   **CÓMO:** Al cerrarse las operaciones y acumular un lote de 32 registros, Python eleva un `TypeError`. Este error es capturado por el bloque `except Exception as e` en `_learn_ppo_batch()` o en la función de actualización de pesos de la estrategia, imprimiendo `"PPO Update Failed: ..."` o `"PPO Batch Learn Error: ..."` en los logs y saltándose la actualización. Como consecuencia, los pesos del ensamble permanecen estáticos.
*   **CUÁNDO:** Ocurre cada vez que la estrategia intenta actualizar sus pesos de ensemble de manera recursiva tras acumular suficientes transacciones (lote de 32 trades).
*   **DÓNDE:** Definido en `core/online_learning.py` (Línea 320) e invocado en `strategies/ml_strategy.py` (Línea 1700 y Línea 5474).
*   **QUIÊN:** El componente `OnlineLearner` en interacción con `MLStrategyHybridUltimate`.

---

## 🛠️ Propuesta de Remediación Técnica (Código y Arquitectura)

Para resolver las deficiencias detectadas sin romper la estructura de configuración ni alterar la lógica crítica del event loop de `core/engine.py`, se proponen los siguientes cambios estructurados:

### 1. Parche para PredictionTracker (Aislamiento por Símbolo)
Debemos modificar la inicialización de `self.strategy_id` en `MLStrategy` para que incluya de forma explícita el símbolo. Esto segregará las señales en el `PredictionTracker` de forma automática.

En `strategies/ml_strategy.py` (Línea 347):

```diff
-        self.strategy_id = f"{lbl}_{base_label}_{self.horizon_str}"
+        # Incluir el símbolo para forzar el aislamiento en el PredictionTracker
+        self.strategy_id = f"{lbl}_{base_label}_{self.horizon_str}_{self.symbol.replace('/', '_')}"
```

Asimismo, debemos corregir el método `get_strategy_metrics` en `core/prediction_tracker.py` para asegurar que el RiskManager pueda interrogarlo de forma precisa utilizando el ID enriquecido:

```diff
     def get_strategy_metrics(self, strategy_id: str,
-                             horizon: str = None) -> Optional[Dict]:
+                             horizon: str = None,
+                             symbol: str = None) -> Optional[Dict]:
         """
         📊 Returns aggregated prediction metrics for a strategy.
         """
         self._refresh_metrics()
 
+        # Si se proporciona el símbolo y el ID de la estrategia no lo contiene,
+        # intentamos buscar la clave segregada por símbolo.
+        if symbol:
+            symbol_safe = symbol.replace('/', '_')
+            if symbol_safe not in strategy_id:
+                strategy_id = f"{strategy_id}_{symbol_safe}"
+
         metrics = self._metrics_cache.get(strategy_id)
```

Y en `risk/risk_manager.py`, modificar las llamadas para pasar el parámetro `symbol`:

```diff
         if self.prediction_tracker:
             should_reject, reject_reason = self.prediction_tracker.should_reject_signal(
-                strategy_id, horizon
+                strategy_id, horizon, symbol=event.symbol
             )
```

---

### 2. Parche para el Agente PPO (Inyección de Contexto y Coexistencia)
Para evitar la dilución del modelo PPO sin aumentar la sobrecarga de memoria que supondría tener 42 instancias de redes neuronales de PyTorch en CPU, sugerimos inyectar el **contexto del activo y el horizonte** en el vector de estado del PPO, incrementando sus dimensiones de 15 a 18.

En `ml/ppo_agent.py` (Modificar dimensión del estado a 18):
```diff
 class PPOAgent:
     def __init__(self, state_dim: int = 18, lr: float = 3e-4, gamma: float = 0.99, clip_eps: float = 0.2):
-        self.state_dim = state_dim
+        # Expandido a 18 dimensiones para inyectar contexto de moneda y horizonte
+        self.state_dim = 18
```

En `strategies/ml_strategy.py` (Línea 2134):
```diff
                 ppo_state = np.array([
                     ensemble_proba[0], ensemble_proba[1], # 2
                     deep_probs.get("SHORT", 0), deep_probs.get("LONG", 0), # 2
                     confluence, atr_pct, vol_ratio, rsi, # 4
                     current_row.get("trend_power", 0), # 1
                     current_row.get("adx", 0), # 1
                     current_row.get("volume_zscore", 0), # 1
                     math_hurst, # 1
                     reg_trend, reg_vol, reg_range, # 3
+                    # Inyección de Contexto Coexistente (3 dimensiones adicionales)
+                    1.0 if self.horizon_str == "SCALPING" else 0.0, # Scalping Flag
+                    1.0 if "BTC" in self.symbol else (0.5 if "ETH" in self.symbol else 0.0), # Peso de Liquidez
+                    current_row.get("normalized_spread", 0.0) # Spread característico del activo
                 ], dtype=np.float32)
```

---

### 3. Parche para update_ppo_batch (Alineación de Firmas)
Para corregir la firma del método y evitar que ocurran TypeErrors durante el entrenamiento online, se propone corregir los argumentos en las llamadas dentro de `strategies/ml_strategy.py` de la siguiente manera:

En `strategies/ml_strategy.py` (Línea 1700):
```diff
                 new_weights, abs_advantages = self.online_learner.update_ppo_batch(
                     weights=current_weights,
                     states=states,
                     actions=actions,
                     old_log_probs=old_log_probs,
-                    returns=rewards,
+                    rewards=rewards,
                     advantages=advantages,
                 )
```

En `strategies/ml_strategy.py` (Línea 5474):
```diff
                 new_weights, advantages = self.online_learner.update_ppo_batch(
                     weights=current_w,
                     states=states,
                     actions=actions,
                     rewards=rewards,
-                    next_states=next_states,
                     old_log_probs=log_probs,
-                    dones=dones,
                 )
```

---

## 📈 Conclusiones Estratégicas para la Gestión de $13 USD

Para duplicar un capital de $13 USD en 15 días mediante interés compuesto y scalping/swing coexistente, **no hay margen de error**. Un Win Rate de 100% (o cercano al límite matemático) requiere que el bot bloquee instantáneamente cualquier señal dudosa.

1.  **El peligro del promedio:** Mantener las métricas del `PredictionTracker` consolidadas globalmente enmascara fallos graves en monedas secundarias, exponiendo nuestro pequeño capital a drawdowns destructivos. La segregación por símbolo en el Tracker debe ser prioritaria.
2.  **Eficiencia en comisiones:** La inyección del spread del activo en el estado del PPO permitirá al modelo aprender a reducir drásticamente el tamaño de posición en monedas con alto spread y baja liquidez, protegiendo los $13 USD de comisiones innecesarias.
3.  **Coexistencia sin interferencias:** Al independizar las métricas y alimentar al PPO con el horizonte temporal exacto, el motor de Scalping (alta frecuencia) y el motor de Swing (largo plazo) pueden ejecutarse en paralelo sobre la misma cuenta sin pisarse ni anularse mutuamente.
