# 🔬 Auditoría Forense: Precisión Predictiva y Optimización de Ejecución
**Fecha:** 2026-04-23
**Objetivo:** Análisis data-driven de la exactitud direccional del motor Trader Gemini y calibración de la ejecución mediante órdenes LIMIT.

> [!WARNING]
> **Hallazgo Crítico:** La auditoría revela que la exactitud direccional general de las estrategias actuales se encuentra por debajo del umbral del 60%. Esto justifica matemáticamente por qué el capital se erosionaba en scalping y valida la necesidad imperativa de nuestro nuevo sistema de rechazo estricto en el `RiskManager`.

---

## 📊 Matriz de Precisión por Estrategia (Baseline Data)

### 1. `[SCL] ML_HYBRID_ULTIMATE_ENSEMBLE_V3` (Scalping)
* **Precisión Direccional Base:** `45.36%` ❌ (Pobre)
* **Exactitud por Ventana Temporal (Decaimiento):**
  * Barras 1-5: ~47%
  * Barras 30-60: cae estrepitosamente a **38.7% - 39.7%**.
  * Barras 120: se recupera a 48.9%.
* **Excursión (MFE vs MAE):** MFE promedio de +0.33%, pero MAE promedio de -0.47%. El riesgo/beneficio inherente está invertido.
* **Duración Óptima del Edge (TTL):** 52 barras.
* **Decisión de Ejecución LIMIT:** **RECHAZO ACTIVO**. Su precisión es inferior al 60%. El `PredictionTracker` emitirá veto.

### 2. `[SWG] ML_HYBRID_ULTIMATE_ENSEMBLE_V3` (Swing)
* **Precisión Direccional Base:** `54.64%` ⚠️ (Aceptable pero bajo)
* **Exactitud por Ventana Temporal:**
  * Barras 1-15: ~52% - 54%
  * Barras 30-60: Alcanza su pico predictivo con **61.28%** a las 30 barras.
* **Excursión (MFE vs MAE):** MFE de +0.47% y MAE de -0.33%. Perfil asimétrico positivo (ganancias mayores a pérdidas temporales).
* **Duración Óptima del Edge (TTL):** 105 barras.
* **Decisión de Ejecución LIMIT:** El promedio de los primeros 15 bars es bajo (54.6%), pero tiene *picos > 60%*. Bajo la nueva regla (Límite 60%), actualmente será vetado hasta que mejore el factor de confianza inicial, o requiere reentrenamiento.

### 3. `[SCL] Technical Momentum_SCALPING (MEAN_REV)`
* **Precisión Direccional Base:** `56.45%` ⚠️ (Aceptable)
* **Exactitud por Ventana Temporal:**
  * Se mantiene bastante estable. Pico predictivo a las 30 barras (**60.00%**).
* **Excursión (MFE vs MAE):** MFE +0.48% vs MAE -0.56%.
* **Duración Óptima del Edge (TTL):** 94 barras.
* **Decisión de Ejecución LIMIT:** Aunque es la estrategia de mejor rendimiento bruto, se queda por debajo del umbral de rechazo del 60% global.

---

## ⚙️ Optimización de la Ejecución LIMIT (Implementada)

Hemos integrado los hallazgos anteriores directamente en el enrutador de ejecución del `RiskManager`.

### 1. Sistema de Veto Predictivo
Se modificó `prediction_tracker.py` y `risk_manager.py` para establecer una compuerta estricta.
- **Acción:** Cualquier señal de una estrategia con una precisión predictiva (`direction_accuracy`) menor al 60% es **rechazada sistemáticamente** antes de reservar margen. 
- **Impacto:** Con los datos actuales, el sistema entrará en una "pausa preventiva", protegiendo el capital de $13 USD de falsos positivos y ruido algorítmico, y obligando a los workers en background a evolucionar modelos hasta romper la barrera del 60%.

### 2. Pricing Dinámico de Órdenes LIMIT BBO
El `limit_offset_pct` se inyecta en la metadata del `OrderEvent`, de modo que el motor asimila la precisión para construir el precio:
- **Precisión > 75% (Agresiva):** Inyección de un offset de `-0.0002` (adentrarse en el spread) para forzar un llenado prioritario dado que el edge predictivo es altísimo.
- **Precisión 60-75% (Conservadora):** Offset pasivo de `+0.0001` (descansar sobre BBO).
- **Precisión < 60%:** No se genera orden (rechazo).

### 3. Ajuste Dinámico del Take Profit (Zombie Feature Remediada)
El `Take Profit` inicial ya no es una constante ciega. El `RiskManager` ahora recorta dinámicamente el `TP` para no superar el 90% del `MFE` histórico de la estrategia. 
- **Ejemplo:** Si el modelo Scalping genera un setup al 0.5%, pero sabemos que su MFE se asfixia en +0.33%, el `RiskManager` recalibra automáticamente el límite TP para evitar tocar techos inalcanzables que causan *decay* de retorno.

## 🚀 Próximos Pasos Recomendados
1. **Auditoría de Retraining:** Todos los modelos XGBoost están produciendo un *win-rate direccional* inferior al 55% en sus primeros 15 barras. Esto sugiere **Model Decay**. La prioridad número 1 es invocar a Optuna para hyper-parametrizar y volver a entrenar.
2. **Dashboard Visual:** Revisar la métricas a través de la interfaz de Streamlit (`dashboard/app.py`), donde esta matriz dictaminará la salud general de las predicciones en tiempo real.
