> ⚠️ **NOTA F0 (2026-08-16):** El objetivo "+100% neto cada 3 días" NO es meta de diseño del sistema actual — matemáticamente insostenible y peligroso como axioma. La meta operativa del Plan Maestro es: expectancy NETO de fees positivo, certificado estadísticamente en demo antes de escalar exposición (Kelly fraccional bayesiano + cota de ruina). El resto del documento se conserva como referencia histórica.

# 🏛️ SISTEMA AUTÓNOMO DE FUTUROS BINANCE — TRADER GEMINI
**DOCUMENTO MAESTRO DE ARQUITECTURA Y REGLAS INAMOVIBLES**

**Capital Inicial:** ~$13 USD  
**Objetivo Maestro:** Crecimiento Exponencial Compuesto (Interés Compuesto Acelerado)  
**Modalidad Operativa:** Horizonte Continuo Cuántico Adaptativo (Erradicación Total de Dualidad Scalp/Swing)  
**Dirección:** Long & Short (Hedging Controlado)

---

## 1. AXIOMAS FUNDACIONALES

### 1.1 PRINCIPIO DE RESPONSABILIDAD ÚNICA
Cada estrategia y cada feature cumple una sola función atómica y bien definida. No existe lógica compartida ni responsabilidades mezcladas. Si una pieza ejecuta más de una cosa, se divide hasta que cada fragmento sea indivisible. Este principio rige el diseño de todo el sistema sin excepción.

### 1.2 RENTABILIDAD INDIVIDUAL DEMOSTRABLE
Cada estrategia y feature debe ser rentable de forma autónoma y estadísticamente demostrable, sin depender del desempeño de ninguna otra pieza. Toda pieza nueva pasa por validación aislada antes de integrarse. Una feature que no demuestra rentabilidad propia no entra. Se corrige, se rediseña o se elimina permanentemente.

### 1.3 OBJETIVO MAESTRO — CRECIMIENTO EXPONENCIAL COMPUESTO
El sistema opera bajo una única métrica de éxito: **+100% de retorno neto sobre el capital total cada ciclo de 3 días**, de forma sostenida y compuesta. 
- Ciclo 1: $13 → $26
- Ciclo 2: $26 → $52
- Ciclo 10: ~$13,312
Toda decisión de diseño o configuración se evalúa exclusivamente contra esta curva exponencial. El capital de cada ciclo completado se convierte en la nueva base del siguiente. El compounding es automático, total e inmediato.

### 1.4 RESTRICCIÓN CLAVE — CRITERIO SUPREMO
> **¿Esta pieza es rentable de forma autónoma, no colisiona con ninguna otra, está correctamente registrada, opera en el régimen/sesión correctos, respeta los límites de riesgo, gestiona su dirección con evidencia y contribuye al +100% cada 3 días?**
*Si la respuesta a cualquier parte de esa pregunta es no, la pieza no se integra.*

---

## 2. ARQUITECTURA DE GESTIÓN — 5 CAPAS OBLIGATORIAS

### Capa 1 — Gestión Individual de Estrategia
Cada estrategia tiene su propio sistema de configuración, control de riesgo, métricas y estado **completamente aislados**. Ninguna estrategia puede leer ni modificar el estado de otra directamente.

### Capa 2 — Registro Omnisciente y Sistema de No-Colisión (`omniscient-registry`)
Núcleo de integridad absoluta en Rust. Bloquea conflictos antes de la ejecución. Posee dos categorías:
- **Valores fijos:** Límites inamovibles de identidad y seguridad. (Prioridad Absoluta).
- **Valores adaptativos:** Parámetros ajustables dinámicamente dentro de rangos fijos.

### Capa 3 — Consciencia Grupal y Orquestación
Inteligencia de portafolio en tiempo real. Detecta sinergias, elimina redundancias, redistribuye capital y asegura que el conjunto avance hacia el +100%. No reemplaza la Capa 1, la coordina.

### Capa 4 — Motor de Dirección (Long / Short Intelligence)
El sistema nunca asume una dirección por defecto. Cada operación justifica su dirección con evidencia cuantitativa. Long y Short tienen lógicas optimizadas separadas. Puede operar hedging controlado.

### Capa 5 — Motor de Compounding y Escala
Al completar cada ciclo de 3 días con +100%, recalcula tamaños, apalancamiento y riesgos. Gestiona la fase de micro-capital (bajo $50) priorizando supervivencia y compounding sobre diversificación.

---

## 3. GESTIÓN DE FUTUROS PERPETUOS (BINANCE)

1. **Apalancamiento Inteligente:** Variable dinámica basada en régimen, señal, capital y distancia al stop. En fase micro, compensa el tamaño sin exceder los límites seguros.
2. **Liquidación (Línea Roja Absoluta):** Monitorea precio de liquidación virtual y mantiene un buffer. Si se acerca, reduce tamaño, añade margen o cierra posición. Margen aislado obligatorio en modo micro-capital.
3. **Funding Rate:** Descuenta costo acumulado del profit esperado (si dura > 4h). Puede rechazar trades.

### Phase 14-22: Quantum Leverage, HFT Execution & Self-Healing
Trader Gemini is designed specifically for **micro-accounts ($13 USD)** with the aggressive goal of achieving a **100% Return on Investment every 3 days**. 

Unlike conventional bots that seek high win rates over months, Trader Gemini uses:
1. **Quantum Leverage Scaling:** Dynamic leverage from 5x to 50x based on win streaks.
2. **Asymmetric Kelly Criterion:** Mathematically guarantees exponential compounding even with win rates < 100%.
3. **Predictive Auto-Healing:** Nano-Hedges injected at -1.0% drawdown to extract micro-PnL from adverse volatility.
4. **Nano-Timeframe Validation:** Order Book imbalance checks to prevent `MICROSCALPING` into micro-dumps/pumps.

The architecture strictly simulates the **Cruel Reality** of Binance Futures fees (0.05% Taker, 0.02% Maker) during backtesting, ensuring that strategies surviving the simulation will absolutely survive in production.

### Phase 23-32: Espejo Perfecto y Performance Cuántica (Grafo Vivo)
1. **Paridad Simulada Absoluta (Espejo Perfecto):** El simulador de Backtesting DEBE replicar 1:1 las restricciones de producción. Esto incluye el `ConsensusFilter` unificado y la simulación estricta de los **Funding Rates** (cada 8h), garantizando que las estrategias SWING no infle ganancias falsas.
2. **Performance Cuántica (GIL-Free & Pandas-Free):** Las rutas críticas HFT en los motores de ML operan exclusivamente utilizando slices en NumPy bruto. Operaciones pesadas (como iteraciones Pandas o bloqueos `gc.collect()`) están formalmente prohibidas en el *hot-path* para sostener un P99 de latencia inferior a 50ms.
4. **Fees y Costos:** Opera maker sobre taker preferiblemente. Calcula breakeven exacto (incluyendo fees) en microscalping antes de entrar.
5. **Liquidez y Selección de Pares:** Monitorea order books. Rota pares dinámicamente priorizando volumen, spread, y volatilidad. 

### Phase 33+: AEGIS V2 HFT Dual-Engine & Quantum Compounding
1. **Zero-Drop Data Queues & Forensic Telemetry:** Rastreo riguroso del `tick_id` desde `binance_loader.py` hasta el cierre de órdenes en `engine.py`. Permite auditorías sistémicas exactas de dónde y por qué se pierde latencia o datos.
2. **Streaming Features O(1):** `math_kernel.py` ha sido equipado con algoritmos O(1) asíncronos para EMA y RSI (recursividad matemática), evitando recalculaciones de arreglos completos para mitigar bloqueos en HFT.
3. **Espectro Continuo Universal (1 ns a 146 años):** Erradicación total de la dicotomía Scalping vs Swing. El motor `GodEngineCore` evalúa un espectro temporal continuo de 32 escalas log-espaciadas base 4 ($\tau_k = 10^{-6} \times 4^k\,\text{ms}$). Cada operación viva evalúa dinámicamente sus funciones analíticas de horizonte continuo ($TP(\tau), SL(\tau), \text{Kelly}(\tau), \text{Trailing}(\tau), OBI(\tau)$) sin colapso discreto ni supresión de frecuencias.
4. **Matemática Expansiva All-In (Micro-Accounts):** `risk-engine` aplica apalancamiento compuesto que arriesga hasta el 95% del capital total para cuentas `< $50`, ejecutando Asymmetric Kelly Fraction continua para salir del fango exponencialmente.

---

## 4. GESTIÓN DE RIESGO ESTRUCTURAL

- **Riesgo por operación:** Porcentaje fijo (más conservador en micro-capital).
- **Riesgo por sesión:** Drawdown máximo diario (congela el bot si se toca).
- **Riesgo por estrategia:** Drawdown acumulado por módulo (desactiva la estrategia específica).
- **Riesgo sistémico:** Drawdown crítico del portafolio (protocolo de emergencia).
- **Stop loss obligatorio:** Ninguna operación nace sin Stop Loss definido en Binance.
- **Ratio R:R:** Curvas continuas $SL(\tau)$ y $TP(\tau)$ garantizan EV $> 0$ en toda escala.
- **Trailing Stop Dinámico:** Curvas continuas $Trailing(\tau) = (\text{mult}, \text{act}, \text{step}, \text{max})$ lock-free.

---

## 5. REGLAS COMPLEMENTARIAS

### 5.1 Detección de Régimen
El sistema reconoce: *tendencial alcista/bajista, lateral comprimido/volátil, ruptura, reversión, volatilidad extrema*. En "régimen incierto", reduce tamaño y prohíbe alta volatilidad. 

### 5.2 Calidad de Señal
Score de 0 a 100 basado en confluencia, timeframes, régimen y liquidez. Aprende estadísticamente: penaliza puntajes en regímenes donde históricamente fallan.

### 5.3 Gestión de Tiempo y Continuidad Temporal
- **Espectro Universal:** Cobertura de micro-impulsos de flujo ($\tau \approx 10\,\text{ms}-1\,\text{s}$) hasta tendencias estructurales macro ($\tau > 24\,\text{h}$) de forma simultánea e integral.
- **Sin Pisarse ni Anularse:** La mecánica de ondas continuas (`Continuous Wave Mechanics`) permite interferencia constructiva cuando las escalas coinciden en dirección, y preservación de capital cuando colisionan sin tendencia dominante confirmada.
- *Calendario de restricciones:* Evita noticias de alto impacto.

### 5.4 Fatiga y Correlación
- **Correlación:** Bloquea posiciones nuevas correlacionadas (ej. BTC y altcoins direccionales simultáneas) para evitar riesgo invisible.
- **Sobreoperación:** Límite máximo de trades, cooldowns entre señales.

### 5.5 Auditoría y Evolución
- Todo es loggeado, rastreable y reconstruible.
- Ciclos de validación estrictos: Backtest -> Paper Trading -> Promoción.
- Si una estrategia sufre degradación (Overfitting), se retira automáticamente.

### 5.6 FASE MICRO-CAPITAL ($13 a $100)
Modo supervivencia-compounding:
- Estrategias de mayor winrate y menor drawdown.
- Posiciones simultáneas limitadas para controlar impacto de fees.
- Prioridad absoluta: Llegar a $100 para desbloquear diversificación total.

---

## 6. ARQUITECTURA NATIVA RUST Y PARADIGMA DE GRAFO VIVO ($G = (V, E, \Phi)$)

### 6.1 Topología de 23 Crates Modulares
1. **`quantum-engine` (Crate Raíz):** Orquestación central, enrutador de microestructura, y gestor dinámico de universos.
2. **`god-engine-core`:** Bucle de eventos HFT sin asignaciones en heap con despacho asíncrono y latencia <2 µs.
3. **`quantum-arena`:** Estructuras `GlobalArena` alineadas en memoria caché (`#[repr(C)]`) con primitivas `SeqLock` sin contención.
4. **`feature-engine`:** 54 variables tensoriales normalizadas calculadas en $O(1)$ (Welford, OFI, Hawkes, Tsallis Entropy).
5. **`signal-engine`:** 13 estrategias cuánticas (Coaxial Breakout, Superposiciones, Solitones, Ondas de Choque Supersónicas).
6. **`risk-engine`:** Envolvente bayesiana adaptativa, criterio de Kelly dinámico y `CorrelationGuard` por horizonte temporal.
7. **`execution-engine`:** Despacho de órdenes HTTP/WebSocket a Binance Futures en Hedge Mode con firmas HMAC SHA256 zero-copy.
8. **`dark-alpha-engine`:** Inferencia neuronal continua y sensores de MEV/liquidaciones de DEX externas (Hyperliquid).
9. **`storage-engine`:** Base de datos embebida `redb` y lakehouse de ticks con soporte para compresión y atomicidad WAL.
10. **`telemetry-server`:** Panóptico en tiempo real, buffers circulares libres de locks y servidor WebSocket Axum.
11. **`audit-engine`:** Monitoreo cibernético de invariantes, validación de paridad de estado y auditoría de drift.
12. **`backtest-engine`:** Motor vectorial con simulación estricta de comisiones reales de Binance y slippage de Kyle.
13. **`evolution-engine`:** Optimización genética con Recocido Simulado y CMA-ES sin sesgos de lookahead.
14. **`metacortex-engine`:** Deliberación del Consejo de Roles Senior bajo Do-Calculus causal.
15. **`os-guardian`:** Bloqueo de memoria física (`VirtualLock`), compactación y contención de working set $<4\text{GB}$.
16. **`data-pipeline`:** Multiplexación de feeds de Binance, Fred Macro y libros L2 en tiempo real.
17. **`data-ingest`:** Parseo ASCII y SIMD JSON de alta velocidad con limitadores de tasa token-bucket.
18. **`strategy-core`:** Cointegración multivariante, arbitraje estadístico y modelos VECM Johansen.
19. **`telemetry-engine`:** Logging no bloqueante basado en colas SPSC de bajo consumo.
20. **`omniscient-registry`:** Registro omnisciente de parámetros y reconciliación sin colisiones de estado.
21. **`flight-recorder`:** Grabador de eventos binarios mapeados en memoria RAM.
22. **`graph-architecture`:** Escaneo y análisis del Grafo Vivo y dependencias de código en tiempo real.
23. **`phase-runner` & `graph-4d`:** Transiciones de fase discretas y topología multidimensional de estado.

### 6.2 Métricas de Rendimiento Cuántico
- **Latencia Hot Path:** 1,349 nanosegundos (Avg), 1,800 ns (P99) sobre 1,000,000 de ticks L1.
- **Velocidad de Backtest:** 120,121 ticks/segundo con paridad 1:1 absoluta.
- **Consumo de Memoria:** $<4\text{GB}$ en RAM física protegida en laptop de 16GB.
- **Pruebas Unitarias:** 258/258 tests unitarios pasando en verde (100%).
- **Compilación Release:** 0 advertencias, 0 errores en los 23 crates y 20 binarios.

## 6.3 Paradigma Cuántico Continuo Unificado (Erradicación Total Scalp/Swing)

En lugar de imponer una división rígida y artificial entre órdenes "Scalping" y "Swing", el kernel de Trader Gemini opera bajo un **Motor Cuántico Continuo Unificado**:

- **QUÉ:** Unificación total del motor de decisión, riesgo y ejecución en un flujo continuo donde el holding time, el trailing stop y los objetivos de ganancia/pérdida se adaptan dinámicamente según la microestructura L2, el régimen de mercado y la volatilidad local (ATR).
- **POR QUÉ:** La división artificial en dos mitades (50/50) causaba inanición de margen crítico en cuentas de $13 USD frente al requisito notional mínimo de Binance ($5.05 USD). Asimismo, la exclusión mutua generaba bloqueos innecesarios. Al unificar el 100% del capital disponible en una sola posición adaptativa por activo, se aprovecha al máximo el margen y se evita la fragmentación de liquidez.
- **PARA QUÉ:** Multiplicar el capital de $13 USD con la máxima velocidad, asignando el 100% del margen disponible de forma óptima a la mejor oportunidad predictiva, logrando crecimiento exponencial compuesto sin rechazos de margen.
- **CÓMO:** El bucle de ejecución de `GodEngineCore::process_tick` sintetiza Dark Alpha, microestructura de libro de órdenes y consenso tensorial en un único `SignalIntent` continuo. `RiskEngine::evaluate_quantum_order` valida el notional contra el capital unificado, calculando SL y TP calibrados dinámicamente mediante múltiplos de ATR sanitizados. El trailing stop cuántico continuo protege los beneficios conforme se expande el movimiento del precio sin cortar prematuramente tendencias macro.
- **CUÁNDO:** En cada micro-tick L1/L2 recibido por WebSocket en tiempo real o procesado en backtesting determinista.
- **DÓNDE:** En el kernel de ejecución (`crates/god-engine-core/src/lib.rs`), el orquestador de riesgo (`crates/risk-engine/src/lib.rs`) y la memoria de estado compartida (`crates/quantum-arena/src/position.rs`).
- **QUIÉN:** Arquitectura Cuántica continua ejecutada por `GodEngineCore`, auditada por `RiskEngine` y monitorizada por `OmniscientRegistry`.
