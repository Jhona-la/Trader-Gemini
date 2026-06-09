# Reporte de Auditoría de Portfolio Multiactivo y Margen bajo Restricción de Capital ($13 USD)

Este reporte documenta los hallazgos detallados de la auditoría forense y de diseño realizada sobre los módulos centrales de administración de portafolio y asignación de capital de **Trader Gemini** (`core/portfolio.py`, `core/adaptive_balancer.py`, `core/correlation_manager.py`, y `core/compounding_engine.py`). El objetivo es comprender cómo se comporta el sistema al operar múltiples activos de forma paralela en una cuenta micro de $13 USD, identificando bloqueos, colisiones y oportunidades de optimización.

---

## Hallazgo 1: Desconexión del Motor de Interés Compuesto Dinámico (Silos Rígidos)

### QUÉ
Bypass completo y desactivación de la lógica de asignación dinámica del motor de interés compuesto (`core/compounding_engine.py`), reemplazándola por silos de asignación de margen estáticos en el portafolio (`MICROSCALPING_MARGIN_CAP = 0.40`, `SCALPING_MARGIN_CAP = 0.40`, `SWING_MARGIN_CAP = 0.20`).

### POR QUÉ
El desarrollador anterior hardcodeó la distribución estática de margen (40% Microscalping, 40% Scalping, 20% Swing) en `portfolio.py` para prevenir que un horizonte temporal con alto volumen (como Scalping) consumiera todo el capital libre de la cuenta, dejando sin fondos a otros horizontes (como Swing). Sin embargo, bajo un capital micro de $13 USD, esta rigidez imposibilita el crecimiento adaptativo. Un silo estático de 20% para Swing equivale a un margen disponible de $2.60 USD, lo cual hace imposible abrir una posición estándar de Swing sin recurrir constantemente a parches de sobredimensión de margen o al padding artificial de notional.

### PARA QUÉ
El motor de interés compuesto (`CompoundingEngine`) fue diseñado para ser **evolutivo y adaptativo**. Su propósito original es asignar porcentajes dinámicos basados en la curva de equidad: con $13 USD, debería asignar ~85-90% del capital a Scalping/Microscalping (donde el alto apalancamiento y el volumen rápido multiplican el capital velozmente) y retrasar la asignación a Swing hasta que la cuenta supere los $50 USD. La desconexión del motor de interés compuesto provoca que el portafolio opere de manera estática y subóptima para cuentas micro.

### CÓMO
En `core/portfolio.py`, el método `_get_available_cash_internal()` calcula el capital asignado utilizando las variables fijas cargadas desde la configuración. La función original del motor `compounding_engine.get_horizon_allocation(equity)` no es invocada en ninguna parte del código de producción, convirtiéndose en código muerto. Para evitar que el silo estático del 20% de Swing bloquee las entradas, el sistema depende del "Soft Cap" que expande el capital disponible al 90% del total de la cuenta si el balance es menor a $50 USD.

### CUÁNDO
Ocurre en cada tick del ciclo de vida del bot cuando se genera una señal de entrada y se calcula la disponibilidad de efectivo para dimensionar el margen necesario (`reserve_cash` / `get_available_cash`).

### DÓNDE
Ubicado en `core/portfolio.py` (Líneas 500-600) y en `core/compounding_engine.py` (módulo desconectado).

### QUIÉN
La clase `Portfolio` actúa como bypass contable, aislando e ignorando los cálculos de la clase `CompoundingEngine`.

---

## Hallazgo 2: Veto de Correlación Sectorial entre Horizontes (Bloqueo No Adaptativo)

### QUÉ
Bloqueo no adaptativo y prematuro de señales debido a la aplicación de un veto de correlación sectorial agregado y global que no discrimina entre los diferentes horizontes de inversión (Scalping vs Swing).

### POR QUÉ
El `CorrelationManager` bloquea una nueva señal si el activo tiene una correlación histórica mayor al 85% con cualquier posición abierta en la cuenta. Bajo un capital de $13 USD, el filtro de seguridad restringe las monedas operables únicamente a **BTC, ETH y SOL**. Debido a la naturaleza del mercado de criptomonedas, la correlación de corto plazo entre estos tres activos suele ser superior a 0.85. Si el bot tiene un trade abierto en BTC en el horizonte de Scalping, cualquier señal subsiguiente en ETH o SOL (incluso en el horizonte de Swing) será bloqueada automáticamente por el veto de alta correlación.

### PARA QUÉ
La restricción de correlación sectorial busca mitigar el riesgo sistémico de acumular riesgo en la misma dirección de mercado. Sin embargo, no tiene sentido técnico que un Scalping LONG rápido (holding time estimado de 5-15 minutos) bloquee una señal de Swing SHORT o LONG de mediano plazo (holding time de 2-3 días) en un activo altamente correlacionado, ya que operan en diferentes dimensiones temporales, con diferentes estructuras de stops y dinámicas de precios. El bloqueo agregado actual reduce la frecuencia de operaciones drásticamente de manera no adaptativa.

### CÓMO
El método `CorrelationManager.check_correlation_risk()` toma una nueva señal, extrae las posiciones activas globales desde `portfolio.positions.keys()` y calcula la correlación cruzada mediante la matriz de correlación histórica. Si detecta un par que exceda el límite global de `MAX_CORRELATION = 0.85`, retorna un veto direccional, descartando la señal antes de su análisis de sizing.

### CUÁNDO
Se ejecuta de manera síncrona en el flujo del `Engine` al recibir un `SignalEvent` de entrada, antes de enviarlo al `RiskManager`.

### DÓNDE
Ubicado en `core/correlation_manager.py` (Líneas 70-130) y consumido por el event loop del motor.

### QUIÊN
Manejado por el `CorrelationManager` en coordinación con el despachador de señales del `Engine`.

---

## Hallazgo 3: Protocolo de Ajuste de Margen para Cuenta Micro (El Protocolo de $13)

### QUÉ
Mecanismo de redimensionamiento de margen dinámico ("Margin Fitting") y uso de "Soft Cap" en cuentas micro (< $50 USD) para cumplir con el tamaño notional de orden mínimo requerido por Binance Futures ($5.00 USD).

### POR QUÉ
Las matemáticas de gestión de riesgo tradicionales ( Kelly o 2% de riesgo por trade) aplicadas a un capital total de $13 USD darían como resultado tamaños nominales inferiores a $1.00 USD. Por ejemplo, arriesgar el 2% de $13 USD equivale a una pérdida máxima permitida de $0.26 USD. A 10x de apalancamiento en Scalping, el margen requerido es insignificante y el tamaño de la orden (notional) estaría por debajo del límite mínimo físico de Binance ($5.00 USD), causando el rechazo instantáneo por parte del exchange y la inoperabilidad total del bot.

### PARA QUÉ
Asegurar que todas las señales validadas por el sistema puedan traducirse en órdenes físicamente aceptadas por la API de Binance, permitiendo la continuidad de la operativa y el crecimiento compuesto acelerado sin bloqueos técnicos por tamaño insuficiente de orden.

### CÓMO
El sistema implementa dos capas de adaptación para cuentas micro:
1. **Límite de Silos Relajado (Soft Cap)**: Si `equity < 50.0`, `Portfolio._get_available_cash_internal()` anula el límite de silo rígido del horizonte y permite tomar hasta el 90% del efectivo total libre de la cuenta.
2. **Uso Agresivo del Margen**: `RiskManager.size_position()` detecta el balance bajo y escala el margen directamente a valores extremos: 95% del efectivo disponible para Scalping/Microscalping (~$11.115 USD de margen) y 50% para Swing (~$5.85 USD de margen).
3. **Padding de Notional**: Si el notional calculado final es menor a $6.00 USD, el Risk Manager incrementa automáticamente el tamaño nominal a $6.00 USD (con margen de error sobre los $5.00 USD de Binance), siempre y cuando el margen de reserva no supere el headroom global del 95% de la cuenta.

### CUÁNDO
Se ejecuta obligatoriamente en cada cálculo de tamaño de orden de entrada antes del envío de la orden física.

### DÓNDE
Definido en `risk/risk_manager.py` en `size_position()` (Líneas 1416-1600) y consumido en `core/portfolio.py`.

### QUIÊN
Orquestado síncronamente por el `RiskManager` consumiendo el efectivo del `Portfolio`.

---

## Hallazgo 4: Balanceador Adaptativo de Carga (Priorización ATR del Event Loop)

### QUÉ
Orquestador dinámico de ordenamiento y priorización de la cola de ejecución del event loop, priorizando los símbolos con mayor volatilidad relativa (ATR) para optimizar la reserva de capital ultra limitado ($13 USD).

### POR QUÉ
Cuando se opera con un capital micro de $13 USD, la primera orden de Scalping enviada consume el 95% del margen libre de la cuenta. Esto significa que las señales que lleguen milisegundos después serán inevitablemente rechazadas por falta de capital. Si el event loop procesa las estrategias en orden alfabético o secuencial simple, una moneda lenta, pesada o en consolidación (baja volatilidad) podría secuestrar el único espacio de margen de la cuenta, bloqueando una señal en un activo de alta volatilidad con mayor probabilidad de movimiento rápido y rentabilidad.

### PARA QUÉ
Garantizar la máxima eficiencia en el uso de los $13 USD de capital, priorizando los activos que ofrecen una mayor relación de movimiento/tiempo (volatilidad ATR), asegurando que el escaso margen disponible se asigne al trade más prometedor del tick actual.

### CÓMO
1. Cada 30 segundos, el `AdaptiveBalancer` recibe los datos de ATR y precio actual para cada símbolo de la lista activa.
2. Calcula la volatilidad relativa y la suaviza utilizando una Media Móvil Exponencial (EMA con factor de suavizado `alpha = 0.3`) para evitar oscilaciones rápidas debido al ruido temporal.
3. Asigna pesos lineales de prioridad entre 0.1 (mínima prioridad) y 1.0 (máxima prioridad) según su ranking de volatilidad.
4. El método `get_processing_order()` devuelve la lista de símbolos ordenados de mayor a menor peso.
5. El `Engine` utiliza este orden para priorizar y evaluar las estrategias asociadas a cada símbolo de forma secuencial en su event loop.

### CUÁNDO
El cálculo de prioridades ocurre periódicamente cada 30 segundos. El orden resultante se utiliza en cada tick de mercado (`MarketEvent`) para priorizar la ejecución del hilo de estrategias.

### DÓNDE
Ubicado en `core/adaptive_balancer.py` y consumido directamente en el bucle principal de `core/engine.py`.

### QUIÊN
El `AdaptiveBalancer` es el encargado del cálculo, alimentado por el `DataHandler` e invocado en el ciclo principal por el `Engine`.

---

## Hallazgo 5: Cuello de Botella por Límite de Posición Única (Cap de Fase 1)

### QUÉ
Restricción estricta de capacidad que limita la cuenta a un máximo de **1 posición activa por horizonte temporal** (Scalping, Microscalping y Swing) cuando el balance total de la cuenta es menor a $50 USD.

### POR QUÉ
En una cuenta micro de $13 USD, la primera posición abierta en Scalping requiere un margen aproximado de $11.115 USD (95% del capital total). Por lo tanto, el balance libre restante es de apenas $1.885 USD. Si el sistema no limitara rígidamente la cantidad de posiciones concurrentes, un intento de abrir una segunda posición de Scalping (por ejemplo, en otra moneda) podría provocar una sobreexposición catastrófica o fallar por margen insuficiente. El cap de 1 posición actúa como una contención de riesgo físico extrema.

### PARA QUÉ
Prevenir llamadas de margen (Margin Calls) y la liquidación total de la cuenta micro, protegiendo el capital disponible y garantizando que el bot mantenga un headroom de seguridad del 5% libre para absorber tarifas de comisión (funding rates y trade fees) y fluctuaciones de precio adversas.

### CÓMO
En el método `size_position()`, el `RiskManager` itera sobre el `virtual_ledger` del `Portfolio` y suma las posiciones activas de cada horizonte. Si la equidad de la cuenta es `< 50.0` y el contador de posiciones del horizonte de la señal recibida es `>= 1`, la señal es bloqueada inmediatamente regresando `None`, incluso si hay margen remanente suficiente para una pequeña posición de notional mínimo.

### CUÁNDO
Ocurre en cada ciclo de dimensionamiento de señales de entrada de forma síncrona en el Risk Manager.

### DÓNDE
Ubicado en `risk/risk_manager.py` (Líneas 1423-1429).

### QUIÊN
Manejado por el `RiskManager`.

---

## Conclusiones sobre la Operatoria Multiactivo con $13 USD

La arquitectura actual presenta un diseño altamente defensivo que prioriza la supervivencia de la cuenta micro de $13 USD a expensas de la flexibilidad operativa. 

### Puntos Fuertes Identificados:
1. **Evita la sobreexposición y liquidación**: El cap de 1 posición y el margen agresivo concentran la fuerza de la cuenta micro en un solo trade de alta probabilidad, permitiendo cumplir con las políticas físicas de Binance sin dispersar el capital en múltiples órdenes inviables.
2. **Bucle Seguro**: El event loop procesa las señales en serie de manera secuencial, lo que elimina cualquier riesgo de carrera (Race Condition) en la reserva de margen contable entre múltiples subprocesos concurrentes.
3. **Priorización Eficiente**: El `AdaptiveBalancer` evita que el capital micro sea secuestrado por activos lentos al reorganizar constantemente el flujo del event loop según la volatilidad real.

### Puntos Críticos a Resolver (Divergencias e Ineficiencias):
1. **Bypass del CompoundingEngine**: El sistema es incapaz de evolucionar de manera dinámica hacia una distribución de capital estándar a medida que el balance sube de $13 USD, debido a que el portafolio utiliza silos rígidos hardcodeados.
2. **Veto de Correlación Agregado Injusto**: El bot no puede ejecutar estrategias de Swing y Scalping en diferentes monedas de forma paralela si el mercado está altamente correlacionado, ya que el veto de correlación no distingue horizontes.

---
**Auditoría realizada y documentada por el Equipo Integrado de Trader Gemini.**
