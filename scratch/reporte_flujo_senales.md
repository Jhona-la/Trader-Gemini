# 🕵️ REPORTE FORENSE: AUDITORÍA DE FLUJO DE SEÑALES (TRADER GEMINI)

## 1. RESUMEN EJECUTIVO

Este análisis forense traza la señal de trading desde su concepción algorítmica en las estrategias hasta su despacho final al exchange. La auditoría revela que el sistema cuenta con un ecosistema hiper-protegido de **más de 35 compuertas de filtrado (gates)**. 

Si bien este diseño protege el capital de pérdidas severas, introduce un cuello de botella crítico para cuentas de tamaño atómico ($13 USD). La interacción de reglas de sizing institucionales, límites sectoriales basados en notional bruto y la falta de propagación estructurada de errores de riesgo genera rechazos silenciosos (ej. `RISK_GATE_UNKNOWN` y `PHASE_1_POSITION_CAP`), reduciendo la probabilidad de supervivencia de las señales a menos del 10% y asfixiando el poder de interés compuesto de la estrategia de scalping.

---

## 2. TRAZABILIDAD DEL CICLO DE VIDA DE LA SEÑAL

### ETAPA 1: Generación de Señales (`strategies/technical.py` y `strategies/ml_strategy.py`)
*   **INPUT**:
    *   `MarketEvent` emitido por el data handler (ticks de precio, libro de órdenes, volumen).
    *   Parámetros técnicos estáticos de `Config.Strategies`.
*   **TRANSFORM**:
    *   Se calculan indicadores matemáticos (EMA, ATR, RSI, MACD, etc.) y se computan predicciones probabilísticas en la capa de Machine Learning.
    *   Se aplican filtros tempranos en la estrategia: **Oracle/Regime Filter**, **Volatility Threshold (ATR)**, **Confluence Filter** y consulta a la IA local (**Sophia**).
*   **OUTPUT**:
    *   Objeto `SignalEvent` (campos: `symbol`, `signal_type` [LONG/SHORT/EXIT], `strength`, `horizon` [SCALPING/SWING], `confidence`, `metadata`, etc.).
*   **LOSS POINTS (Pérdidas Silenciosas)**:
    *   *Filtro ATR Temprano:* Si la volatilidad del par es menor que el umbral estático de la estrategia, se ejecuta un `return None` sin registrar registro de descarte en telemetría.
    *   *Excepción en ML Strategy:* Si ocurre un error al procesar variables temporales (como el atributo huérfano `horizon_type` en `ml_strategy.py`), la ejecución de la estrategia falla y `engine.py` captura el error con un `except Exception: pass` silencioso, perdiendo la señal.
*   **DUPLICATION POINTS (Duplicaciones)**:
    *   Si una estrategia se registra por símbolo individual (`BTCUSDT`) y simultáneamente en la lista general (`ALL`), el motor ejecutará la llamada a `calculate_signals` dos veces para el mismo tick, generando señales gemelas.

### ETAPA 2: Event Loop y Enrutamiento del Motor (`core/engine.py`)
*   **INPUT**:
    *   `SignalEvent` colocado en el buffer `PriorityBoundedQueue`.
*   **TRANSFORM**:
    *   Se valida el Time-To-Live (TTL) de la señal contra la hora del reloj maestro.
    *   Se valida que el precio de mercado esté disponible y no sea anómalo.
    *   Se aplican las restricciones del `TemporalSupervisor` (bloqueo durante fases de startup).
    *   Se optimizan asíncronamente los TP/SL mediante simulaciones de trayectoria en `multiverse_simulator`.
    *   Se ejecuta el filtro de evasión de Funding Fees (bloqueo en los minutos XX:45 a XX:59 en horas de snapshot).
    *   Se verifica si hay shock de volatilidad en el par (`market_regime.is_volatility_shock`), activando un congelamiento de 5 minutos si es positivo.
    *   Se verifica que no existan conflictos direccionales inmediatos entre horizontes de Scalping y Swing (`competing_strategies`).
    *   Se aplica el veto macro de `MultiHorizonOracle` (evaluando el clash vector en velas de 1d y 1w).
    *   Se aplica el veto global de `Sophia-Global` según el umbral `SOPHIA_MIN_CONFIDENCE`.
    *   Se evalúa la salud del ecosistema con `OmegaProtocol`.
    *   Se aplican filtros de segmentación institucional (`SegmentPolicyEngine`).
    *   Se calcula el score final de la señal con `SignalScorer` y se valida la viabilidad de Breakeven.
*   **OUTPUT**:
    *   Llamada asíncrona a `meta_arbitrator.submit_intent(event)`.
*   **LOSS POINTS (Pérdidas Silenciosas)**:
    *   *Discrepancia en Sophia:* `SOPHIA_MIN_CONFIDENCE` estaba hardcodeado en `engine.py` en 0.70, mientras que las estrategias usaban 0.60 de Config. Esto mataba en silencio señales con confianza de entre 60% y 69.9% en el motor principal, a pesar de haber sido aprobadas por la estrategia.
    *   *Filtro de Breakeven:* Si el precio actual está demasiado cerca del target y el R:R efectivo cae por debajo de los mínimos, la señal se descarta silenciosamente.
*   **DUPLICATION POINTS**:
    *   Falta de deduplicación de IDs de señales entrantes en la cola rápida de eventos del motor.

### ETAPA 3: Consenso y Arbitraje de Intenciones (`core/meta_coordinator.py`)
*   **INPUT**:
    *   `TradeIntent` derivado de la señal aprobada por el motor.
*   **TRANSFORM**:
    *   El `ConsensusFilter.check_signal()` evalúa la señal a través de **10 gates numéricos** (toxicidad, listas negras, desbalances de orderflow, etc.).
    *   Se aplican las invariantes duras del sistema en `invariants.py` (Self-Hedging, liquidez crítica, cap de exposición).
    *   Se evalúan vetos de grafo en `GraphIntelligenceLayer` (contagio de riesgo sistémico).
    *   Se valida la viabilidad de apertura en `get_asset_intelligence().verify_opening`.
    *   Se resuelven conflictos direccionales dentro del mismo horizonte temporal.
*   **OUTPUT**:
    *   El `TradeIntent` aprobado es encolado en `meta_arbitrator.approved_queue`.
*   **LOSS POINTS (Pérdidas Silenciosas)**:
    *   *Consensus Veto:* Si el filtro de consenso o el grafo de contagio vetan el trade, la intención es descartada. Aunque se incrementa la telemetría métrica, no se emite una alerta estructurada de vuelta al flujo principal de ejecución de la orden.
*   **DUPLICATION POINTS**:
    *   Falta de validación de unicidad de `thought_id` en la cola de intenciones del árbitro.

### ETAPA 4: Pre-Flight del Motor y RiskManager (`core/engine.py` -> `risk/risk_manager.py`)
*   **INPUT**:
    *   `TradeIntent` aprobado extraído por `_drain_meta_arbitrator()`.
*   **TRANSFORM**:
    *   Se valida la intención pre-flight contra la Capa 7 del `OmniscientRegistry` (calor de portafolio y límites absolutos).
    *   Se delega al `RiskManager.generate_order()`.
    *   Se valida el Kill Switch global.
    *   Se validan límites de frecuencia diarios.
    *   Se aplican los vetos de correlación sistémica (`CorrelationManager`) y sentimiento del mercado.
    *   Se evalúa la precisión histórica de la estrategia en `PredictionTracker` (veto si accuracy < 55%).
    *   Se realiza el chequeo de dirección segura (`_validate_directional_safety`): si hay una posición contraria activa que está perdiendo y ha madurado, genera una orden de `FLIP_EXIT` primero.
    *   Se calcula la exposición sectorial permitida (`_get_sector_exposure`).
    *   Se calcula el dimensionamiento de posición en `size_position` mediante Criterio de Kelly (o apalancamiento agresivo del 95% para cuentas de $13).
    *   Se intenta reservar el margen en `portfolio.reserve_cash()`. Si no es suficiente, se ejecuta el protocolo de Margin Fitting ajustando el tamaño al notional mínimo de Binance ($5.5 - $6.0).
    *   Se evalúa si la orden entra como LIMIT (Maker) o MARKET (Taker) según momentum y volatilidad.
*   **OUTPUT**:
    *   Objeto `OrderEvent` listo para el ejecutor de órdenes.
*   **LOSS POINTS (Pérdidas Silenciosas)**:
    *   *RISK_GATE_UNKNOWN:* Si el sizing falla por falta de margen libre o el check de VaR del portafolio veta la entrada, `generate_order` imprime en consola con `print()` y retorna `None`. El motor captura este `None` y loguea un warning genérico sin especificar cuál de los 10+ sub-filtros internos del `RiskManager` gatilló el descarte.
    *   *PHASE_1_POSITION_CAP:* Bloqueo estricto para cuentas de menos de $50 USD. Evita abrir una posición si ya hay una activa en el mismo horizonte. Impide operar scalping si hay un trade abierto en ese horizonte temporal.
*   **DUPLICATION POINTS**:
    *   Si se genera la orden pero la reconexión de red del WebSocket del ejecutor reenvía el payload sin un id único deduplicable de Binance.

---

## 3. RESPUESTAS A LAS PREGUNTAS CRÍTICAS DE LA AUDITORÍA

### A. ¿Cuántos filtros/gates existen entre la generación de la señal y la ejecución?
Existen exactamente **más de 35 gates y sub-gates** distribuidos a lo largo del pipeline de Trader Gemini:
1.  **4 gates en la Estrategia:** Oracle, Volatility (ATR), Confluence, IA Sophia Local.
2.  **11 gates en el Event Loop del Motor:** TTL, Price, Temporal Constraints, Funding Evasion, Shock Regime Freeze, Competing Strategies, Multi-Horizon Oracle Veto, Sophia-Global Veto, Omega Protocol, Segment Policy, Signal Scorer / Breakeven Viability.
3.  **13 gates en el MetaCoordinator/Consensus:** ConsensusFilter (10 sub-gates numéricos), Invariantes (Self-Hedging, Liquidez, Max Exposure), Graph Contagion Veto, Asset Intel Verification, Conflict Resolution.
4.  **1 gate en Pre-Flight:** Omniscient Registry (Capa 7).
5.  **11 gates en el RiskManager:** Kill Switch, Frequency Limits, Regime Veto, Tension Veto, Correlation, Sentiment, Liquidity Guardian, Prediction Confidence Gate, Directional/Flip Safety, Sector Exposure, Sizing / Margin Reservation / Portfolio VaR.

### B. ¿Cuál es la probabilidad de que una señal sobreviva a todos los gates?
Matemáticamente, si asumimos una tasa de aprobación alta del **95%** en cada una de las 35 compuertas (lo cual es muy optimista para filtros predictivos y de volatilidad), la probabilidad acumulada de supervivencia es:
$$P(\text{survival}) = 0.95^{35} \approx 16.6\%$$
En condiciones reales de alta volatilidad o bajo balance ($13 USD), compuertas como `PHASE_1_POSITION_CAP`, `SOPHIA_MIN_CONFIDENCE` (cuando estaba descalibrada), `FEE_DRAG_ATR` y la reserva de margen de sector colapsan la tasa de supervivencia a **menos del 5-10%**. Esto anula la frecuencia operativa requerida para el Scalping de alta probabilidad.

### C. ¿Are there any gates that are mathematically impossible to pass?
En el contexto de una cuenta de **$13 USD**, el gate de **Límite de Exposición de Sector (`max_sector_exposure`)** era matemáticamente impracticable en su diseño original:
*   El límite institucional estaba fijado en **35% de la equity** ($4.55 USD de exposición).
*   Binance Futures impone un **notional mínimo por orden de $5.00 USD**.
*   Aún con apalancamiento de 10x (margen requerido = $0.50 USD), la exposición bruta noional calculada por el sistema para una sola orden era de $5.00 USD, lo cual representaba un **38.4% de la cuenta total**.
*   Por lo tanto, la primera orden de cualquier sector consumía el 38.4%, excediendo instantáneamente el límite del 35% y haciendo **matemáticamente imposible abrir una segunda posición en el mismo sector**, o bloqueando la primera si se calculaba de manera restrictiva.
*   *Estado:* Mitigado recientemente mediante el ajuste de cap adaptativo por tamaño de cuenta (microcuentas < $100 USD escalan al **95% de exposición** para permitir operar).

### D. ¿Existe confusión de horizontes (SCALPING vs SWING) en algún gate?
Sí, se han identificado dos puntos críticos de colisión de horizontes:
1.  **En `ml_strategy.py`:** Uso inconsistente de variables de horizonte. La estrategia utiliza referencias a `horizon_type` (que no existe como atributo de instancia inicializado en la clase base) en lugar de utilizar el parámetro unificado `self.horizon` (que se mapea como `'SCALPING'` o `'SWING'`). Esto causa fallos silenciosos al recuperar parámetros de configuración o infiere horizontes equivocados.
2.  **En `risk_manager.py` (Límites de Posición):**
    El gate `PHASE_1_POSITION_CAP` originalmente contaba todas las posiciones activas de forma global. Esto causaba que una posición de Swing a largo plazo bloqueara por completo el flujo de Scalping de alta frecuencia en el mismo horizonte o viceversa, impidiendo la coexistence pacífica e integral de ambos motores. Aunque se mitigó aislando por horizonte (`open_positions_for_horizon`), el sistema de dimensionamiento aún sufre si los límites agregados de margen no distinguen de manera inteligente el capital asignado dinámicamente a cada uno.

---

## 4. PROPUESTA DE MEJORA PARA EL LOGGING (FORENSIC OBSERVABILITY)

Para resolver la opacidad detrás de `RISK_GATE_UNKNOWN` y ver con precisión qué compuerta rechazó cada orden, proponemos la implementación de un **Protocolo de Rechazo Estructurado**.

### QUÉ
Reemplazar el retorno simple de `None` en `RiskManager.generate_order()` por el enriquecimiento del objeto `TradeIntent` / `SignalEvent` con metadatos específicos del veto, o bien lanzar una excepción estructurada (`RiskVetoException`) que propague la causa exacta de la falla.

### POR QUÉ
Actualmente, el `Engine` y los logs del backtest solo registran que el `RiskManager` rechazó el trade, pero la lógica exacta de la muerte del trade se pierde en impresiones de consola (`print`) no estructuradas que no se integran al `logger.py` ni a la base de datos de telemetría forense.

### PARA QUÉ
Permitir que el dashboard, el motor de backtesting y los operadores identifiquen instantáneamente si el trade murió por:
1.  `PHASE_1_POSITION_CAP` (Límite de posición por microcuenta).
2.  `MARGIN_INSUFFICIENT` (Margen libre insuficiente).
3.  `FEE_DRAG` (ATR insuficiente para fees).
4.  `PREDICTION_GATE` (Precisión histórica de la estrategia < 55%).
5.  `PORTFOLIO_VAR` (VaR excedido).
6.  `HIGH_CORRELATION` (Riesgo de correlación sistémica).

### CÓMO (Código Propuesto)

1.  **Definir una enumeración de razones de rechazo o usar strings estructurados en `core/enums.py` o dentro de `RiskManager`:**
    ```python
    # En risk/risk_manager.py
    class RejectionReason:
        KILL_SWITCH = "KILL_SWITCH_ACTIVE"
        FEE_DRAG = "FEE_DRAG_ATR_INSUBSTANTIAL"
        FREQUENCY_LIMIT = "DAILY_TRADE_LIMIT_EXCEEDED"
        REGIME_VETO = "REGIME_ALIGNMENT_VETO"
        REGIME_TENSION = "REGIME_TENSION_EXCESSIVE"
        HIGH_CORRELATION = "SYSTEMIC_CORRELATION_VETO"
        SENTIMENT_DIVERGENCE = "SENTIMENT_DIVERGENCE_VETO"
        LIQUIDITY_VACUUM = "LIQUIDITY_VACUUM_VETO"
        PREDICTION_GATE = "STRATEGY_ACCURACY_BELOW_THRESHOLD"
        DIRECTIONAL_SAFETY = "DIRECTIONAL_DUPLICATION_BLOCKED"
        MARGIN_INSUFFICIENT = "MARGIN_INSUFFICIENT_FOR_ENTRY"
        SECTOR_EXPOSURE = "SECTOR_EXPOSURE_LIMIT_EXCEEDED"
        PORTFOLIO_VAR = "PORTFOLIO_VAR_BUDGET_EXCEEDED"
        ORPHAN_GUARD = "ORPHAN_STRATEGY_BLOCKED"
        TEMPORAL_STARTUP_BLOCK = "TEMPORAL_STARTUP_OBSERVATION_ACTIVE"
    ```

2.  **Modificar `generate_order` para inyectar la causa en los metadatos de la señal antes de retornar `None`:**
    ```python
    # En risk/risk_manager.py
    def _reject_trade(self, signal_event, reason: str):
        if not hasattr(signal_event, 'metadata') or signal_event.metadata is None:
            object.__setattr__(signal_event, 'metadata', {})
        signal_event.metadata['rejection_reason'] = reason
        signal_event.metadata['rejected_at'] = time.time()
        
        # Log estructurado con logger en lugar de print
        logger.warning(f"🛑 [RISK VETO] Signal {signal_event.symbol} rejected. Reason: {reason}")
        
        # Opcional: Registrar en telemetría forense si está disponible
        if self.portfolio and hasattr(self.portfolio, 'db') and self.portfolio.db:
            try:
                self.portfolio.db.log_thought(
                    thought_id=getattr(signal_event, 'thought_id', 'N/A'),
                    symbol=signal_event.symbol,
                    message=f"Risk Veto: {reason}",
                    regime=self.current_regime
                )
            except Exception:
                pass
        return None
    ```

3.  **Actualizar los puntos de descarte en `generate_order` para usar `_reject_trade`:**
    *   *Ejemplo:*
        ```python
        if not self._validate_kill_switch():
            return self._reject_trade(signal_event, RejectionReason.KILL_SWITCH)
        ```
    *   *Para `size_position` (en la llamada interna):*
        ```python
        # Capturar la razón específica de size_position
        params = self.size_position(...)
        if not params:
            # size_position debería poder reportar si falló por TEMPORAL_STARTUP, PHASE_1_POSITION_CAP, etc.
            # Propagamos una razón por defecto si no se inyectó ya en el signal_event
            reason = signal_event.metadata.get('rejection_reason', RejectionReason.MARGIN_INSUFFICIENT)
            return self._reject_trade(signal_event, reason)
        ```

4.  **Actualizar `core/engine.py` en `_execute_approved_intent` para recuperar y loguear el error estructurado:**
    ```python
    # En core/engine.py -> _execute_approved_intent
    order_event = self.risk_manager.generate_order(event, current_price)
    if order_event:
        # ... flujo normal de encolado de orden ...
    else:
        # Recuperar la razón estructurada de los metadatos de la señal
        reason = event.metadata.get('rejection_reason', 'RISK_GATE_UNKNOWN')
        logger.warning(f"🛑 [ENGINE] RiskManager REJECTED signal for {event.symbol}. Reason: {reason}")
        self.metrics['discarded_events'] += 1
        
        # Registrar en Interaction Monitor con el detalle del error estructurado
        try:
            if _INTERACTION_MONITOR_AVAILABLE and get_interaction_monitor:
                strat_id = getattr(event, 'strategy_id', 'Strategy')
                get_interaction_monitor().log_interaction(
                    source=strat_id,
                    action="Signal Rejected",
                    details=f"RiskManager rejected signal for {event.symbol}. Reason: {reason}"
                )
        except Exception as e:
            logger.debug(f"Interaction monitor logging failed: {e}")
    ```

---

## 5. CONCLUSIÓN (MÉTODO PROFESOR)

### ¿QUÉ?
La **Auditoría Forense del Flujo de Señales** es un rastreo sistemático del ciclo de vida de los eventos de trading (Signal -> Intent -> Order -> Fill) a través de las compuertas lógicas de Trader Gemini para identificar cuellos de botella y descartes silenciosos de órdenes.

### ¿POR QUÉ?
Se realiza porque en el entorno de producción y backtesting se observaba una baja tasa de ejecución de órdenes y un logueo opaco catalogado bajo `RISK_GATE_UNKNOWN`, lo cual impedía diagnosticar por qué estrategias con alto poder predictivo no lograban colocar operaciones en la cuenta de $13 USD.

### ¿PARA QUÉ?
Para reconfigurar el balance óptimo entre la seguridad de la cuenta y la frecuencia de trading (WR del 100% y crecimiento compuesto exponencial), removiendo restricciones sobredimensionadas para micro-cuentas y habilitando logs precisos de nanosegundos que eviten la ceguera operativa.

### ¿CÓMO?
Analizando el código estático de las estrategias (`technical.py`, `ml_strategy.py`), el motor de eventos (`engine.py`), el árbitro central (`meta_coordinator.py`, `consensus_filter.py`, `invariants.py`) y el validador de riesgo (`risk_manager.py`), correlacionando sus variables de control y detectando fallos lógicos en cascada.

### ¿CUÁNDO?
Debe ejecutarse antes de iniciar campañas operativas reales de scalping/swing con dinero real, y de forma continua en cada ciclo de integración y desarrollo del bot de trading.

### ¿DÓNDE?
A lo largo de toda la arquitectura core y de riesgo del proyecto en `C:\Users\jhona\Documents\Proyectos\Trader Gemini`.

### ¿QUIÉN?
Manejado de forma integrada por el **Quant Developer** (lógica matemática), el **Risk Manager** (seguridad de stops y caps), el **SRE/DevOps** (logs estructurados y latencia) y coordinado por el **Arquitecto Senior**.
