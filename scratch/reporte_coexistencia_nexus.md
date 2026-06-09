# Reporte de Coexistencia Multihormonal y Prevención de Colisiones (Nexus Protocol)

Este reporte detalla los hallazgos de la auditoría arquitectónica realizada sobre los componentes principales del sistema (`core/portfolio.py`, `risk/risk_manager.py`, `core/engine.py` y `strategies/`) para validar el soporte simultáneo de estrategias de Scalping y Swing sin colisiones, solapamiento de órdenes o interferencia en la gestión de riesgo y límites.

---

## Hallazgo 1: Soporte de Posiciones Simultáneas y Aislamiento por Horizontes (El Libro Mayor Virtual)

### QUÉ
Aislamiento lógico y contable de las posiciones y precios promedio mediante un libro mayor virtual (`virtual_ledger`) en lugar de usar un estado unificado de posición agregada.

### POR QUÉ
Binance Futures, operando incluso en Modo Cobertura (Hedge Mode / Dual-Side Position), solo permite tener una posición física `LONG` y otra `SHORT` por símbolo. Si una estrategia de Scalping abre un `LONG` y otra de Swing abre un `LONG` en el mismo símbolo, Binance las consolida físicamente en un solo trade con un tamaño combinado y un único precio promedio de entrada. Si el sistema no realizara un seguimiento lógico separado, la salida de Scalping (que opera con targets de ~0.20%) cerraría total o parcialmente la posición de Swing (que opera con targets de ~4.5%), arruinando su rentabilidad y lógica interna.

### PARA QUÉ
Evitar la interferencia mutua entre estrategias con horizontes temporales distintos (ej. que cierres rápidos de Scalping anulen operaciones de Swing), garantizar el cálculo exacto del PnL de cada trade según su horizonte, y permitir que ambas lógicas operen en paralelo de manera integral en cuentas con capital micro ($13 USD).

### CÓMO
El componente `Portfolio` genera claves únicas en su diccionario interno `virtual_ledger` siguiendo el formato `{symbol}_{horizon}_{pos_side}` (ej. `BTC/USDT_SCALPING_LONG` y `BTC/USDT_SWING_LONG`).
1. Al recibir un `FillEvent`, se invoca `_update_virtual_ledger()`.
2. Se extrae la dirección y el horizonte del fill (proporcionados en los metadatos o por el origen de la señal).
3. Se calcula el precio promedio (`avg_price`), la cantidad (`quantity`) y el PnL aislado para ese horizonte y lado específicos de forma totalmente segregada, ignorando el tamaño consolidado en el exchange.

### CUÁNDO
Se ejecuta de manera reactiva inmediata cada vez que se procesa un `FillEvent` que proviene del gestor de ejecuciones (`BinanceExecutor` / `UserDataStream`).

### DÓNDE
Ubicado en `core/portfolio.py` dentro de la función interna `_update_virtual_ledger()` (Líneas 1015-1200).

### QUIÊN
Manejado por la clase `Portfolio` (responsable del mantenimiento del estado contable y del libro mayor local).

---

## Hallazgo 2: Gestión de Stops Virtuales en el Risk Manager (Check Stops)

### QUÉ
Monitoreo de niveles de Stop-Loss (SL), Take-Profit (TP) y Trailing Stops a nivel de software (virtuales), y desactivación de órdenes de stop físicas en el Exchange.

### POR QUÉ
Si se enviaran órdenes físicas de SL/TP a Binance para múltiples horizontes en la misma dirección (ej. un stop de Scalping al -0.35% y un stop de Swing al -1.5%), las órdenes físicas se cruzarían. El primer stop en activarse (el de Scalping al -0.35%) se ejecutaría contra la posición consolidada total de Binance. Esto resultaría en el cierre parcial o total de la posición de Swing antes de tiempo en el exchange, dejando la sub-posición de Swing en el `virtual_ledger` como una "posición fantasma" sin respaldo real.

### PARA QUÉ
Asegurar que cada horizonte evalúe y ejecute sus stops de forma independiente, de modo que el cruce de un stop de Scalping solo genere una orden de cierre por el tamaño exacto de la sub-posición de Scalping, dejando la posición física de Swing intacta en el Exchange.

### CÓMO
1. El ejecutor físico (`binance_executor.py`) tiene un bypass explícito en `_place_protective_orders()` que desactiva el envío de SL/TP reales a la API de Binance.
2. El motor principal (`engine.py`) invoca en cada tick de mercado a `RiskManager.check_stops()`.
3. `check_stops` itera sobre el `virtual_ledger` evaluando cada clave activa `{symbol}_{horizon}_{pos_side}`.
4. Calcula el PnL no realizado basado en el `avg_price` de esa clave virtual y el precio actual del `MarketEvent`.
5. Si el precio cruza el TP o SL virtual de dicho horizonte, genera un `SignalEvent` con `signal_type=SignalType.EXIT` y el `horizon` correspondiente.
6. El motor procesa este evento y el executor envía una orden normal de reducción (`reduceOnly=True` y `positionSide=LONG/SHORT`) a Binance por la cantidad exacta de esa clave virtual.

### CUÁNDO
Se activa en cada ciclo del motor principal cuando llega un `MarketEvent` (tick de precio) para un activo en el cual el portafolio tiene posiciones abiertas en el `virtual_ledger`.

### DÓNDE
Ubicado en `risk/risk_manager.py` en la función `check_stops()` (Líneas 2485-3260).

### QUIÊN
Manejado por el `RiskManager` en confluencia con el `BinanceExecutor`.

---

## Hallazgo 3: Prevención de Solapamiento, Colisiones y Flips Direccionales (Locks y FLIP-EXIT)

### QUÉ
Control de exclusión mutua mediante candados temporales (`exit_pending_time`) y flips direccionales atómicos de alta confianza para prevenir bloqueos de capital (margin locks).

### POR QUÉ
En una cuenta micro de $13 USD, el margen disponible es extremadamente limitado. Si una estrategia de Scalping decide hacer un "Flip" (ej. está en LONG y recibe una señal fuerte de SHORT), intentar abrir la posición SHORT antes de que Binance confirme el cierre total de la LONG fallará por falta de margen (doble reserva de margen). Asimismo, retrasos de red podrían causar que se envíen múltiples órdenes de cierre para la misma posición si no se bloquea la evaluación.

### PARA QUÉ
Garantizar transiciones atómicas de posición en el exchange sin solapamiento de órdenes y proteger la liquidez de la cuenta contra "margin leaks" o rechazos de la API de Binance.

### CÓMO
1. **Exit Lock**: Al enviar una orden de salida, `Portfolio` registra un timestamp en `exit_pending_time` para esa sub-posición. Mientras este candado esté activo y no haya expirado (timeout), `RiskManager` y `Engine` bloquean cualquier nueva orden de apertura para esa clave. Al procesar el fill de cierre, el lock se limpia.
2. **FLIP-EXIT atómico**: Si se recibe una señal de sentido contrario en el mismo horizonte (ej. señal de SHORT estando en LONG), `RiskManager.generate_order()` intercepta el flujo. Si la señal tiene alta confianza (>0.80) y el trade anterior ha madurado, genera primero un `SignalEvent` especial de `FLIP_EXIT`. Este cierra la posición activa. Una vez que se libera el margen en el exchange, el motor procesa la nueva entrada en la dirección opuesta en el siguiente tick.

### CUÁNDO
Se activa durante la validación de señales en `generate_order()` y al procesar salidas en el flujo de órdenes.

### DÓNDE
Ubicado en `risk/risk_manager.py` -> `generate_order()` (Líneas 1960-2067) y en `core/portfolio.py` (Líneas 1073, 1146, 1192).

### QUIÊN
Manejado coordinadamente por `Portfolio` y `RiskManager`.

---

## Hallazgo 4: Sizing Kelly, Drawdowns y Límites Segregados por Horizonte

### QUÉ
Parametrización dinámica y límites de drawdown, apalancamiento y tamaño de posición diferenciados por horizonte temporal.

### POR QUÉ
Las estrategias de Scalping requieren stops extremadamente ajustados (0.40% SL y 0.60% TP base) y un mayor apalancamiento para capitalizar micro-tendencias. Swing requiere stops amplios (1.5% SL, 4.5% TP base) para tolerar el ruido del mercado y un apalancamiento menor para evitar liquidaciones. Si el gestor de riesgo evaluara los límites de forma agregada, el drawdown de un trade de Swing podría activar el switch defensivo de Scalping, reduciendo incorrectamente su tamaño o congelando su operativa.

### PARA QUÉ
Mantener un Win Rate esperado del 100% en Scalping mediante una altísima selectividad de trades, y permitir que Swing capture tendencias de largo plazo con reglas de maduración adecuadas.

### CÓMO
1. **Diferenciación de Configuración**: `check_stops` y `generate_order` cargan parámetros desde `Config.Horizons.Scalping` o `Config.Horizons.Swing` de forma aislada.
2. **Sizing Kelly Modificado**: `size_position()` calcula el tamaño de posición aplicando un multiplicador de mérito que toma en cuenta la precisión reciente del horizonte calculada por el `PredictionTracker`.
3. **Cap de Posiciones por Balance**: Para balances bajos (<$50 USD), se limita estrictamente a un máximo de 1 posición abierta por horizonte, protegiendo el margen remanente.
4. **Leverage Dinámico**: Se invoca `_ensure_leverage` en el executor para ajustar el apalancamiento del exchange antes de enviar la orden de cada horizonte (ej. 20x para Scalping, 5x para Swing).

### CUÁNDO
Durante la inicialización de cada trade en `generate_order()`, el ajuste de leverage en `binance_executor.py` y la evaluación de salidas de mercado.

### DÓNDE
Definido en `config.py`, evaluado en `risk/risk_manager.py` -> `size_position()` y ejecutado en `execution/binance_executor.py`.

### QUIÊN
Manejado por el `RiskManager`.

---

## Hallazgo 5: Resolución de Conflictos Temporales y Veto Macro (Clash Vector & Meta-Arbitrator)

### QUÉ
Arbitraje y resolución de señales contradictorias entre múltiples horizontes (Scalping vs Swing) en el mismo símbolo mediante un vector de choque direccional.

### POR QUÉ
Si Scalping genera una señal `LONG` y Swing genera una señal `SHORT` simultáneamente para el mismo par (ej. BTC/USDT), el bot entraría en ambas posiciones. Físicamente en Binance tendríamos un LONG y un SHORT abiertos. Aunque se mantengan contablemente separadas en el `virtual_ledger`, esta operativa cruzada anula el beneficio neto y duplica el pago de comisiones (comisión por apertura/cierre en ambas), lo cual es inaceptable para cuentas de $13 USD.

### PARA QUÉ
Prevenir que las estrategias "compitan" entre sí en el mismo activo, reduciendo los costos de transacción y priorizando la dirección con mayor probabilidad macro.

### CÓMO
1. Cuando `Engine` procesa una señal de entrada en `_process_signal_event()`, comprueba si hay una posición activa en el horizonte opuesto mediante `get_horizon_position(event.symbol, opposing_horizon)`.
2. Si las direcciones chocan (ej. señal LONG de Scalping vs posición SHORT de Swing), se emite una advertencia de integridad sistémica.
3. El `Engine` somete la señal al `MetaArbitrator` llamando a `meta_arbitrator.submit_intent(event)`.
4. El árbitro utiliza `MultiHorizonOracle.evaluate_clash_vector()` para contrastar la dirección de la señal local contra el contexto macro de timeframes largos (1d, 1w) y el sentimiento de flujo de órdenes.
5. Si el Clash Vector es alto (>0.85), se aplica un **Veto Absoluto** y la señal se descarta. Si es menor, se aplica un **Veto Suave (Soft Veto)** penalizando la confianza de la señal proporcionalmente.
6. Solo las señales aprobadas son liberadas del canal del `MetaArbitrator` para que el `RiskManager` genere la orden física.

### CUÁNDO
En cada generación de señal de entrada de compra (`LONG`) o venta (`SHORT`).

### DÓNDE
Ubicado en `core/engine.py` -> `_process_signal_event()` (Líneas 970-1040) y `sophia/intelligence.py` -> `MultiHorizonOracle.evaluate_clash_vector()`.

### QUIÊN
Manejado coordinadamente por `Engine`, `MetaArbitrator` y `MultiHorizonOracle`.

---

## Conclusión de la Auditoría de Coexistencia
La arquitectura de Trader Gemini posee un diseño de aislamiento de horizontes sumamente robusto. La combinación del **Virtual Ledger** para la contabilidad, **Stops Virtuales** para la gestión de riesgo por software, e **Integridad Sistémica vía Meta-Arbitrador** garantiza que el sistema puede operar Scalping y Swing de forma concurrente sobre los mismos activos sin generar colisiones ni anular posiciones. 

Este blindaje es la piedra angular técnica que permite proteger la cuenta micro de $13 USD y maximizar el crecimiento de capital exponencial.
