# 🧠 TRADER GEMINI: DOCUMENTACIÓN MAESTRA DE ESTRATEGIAS (STRATEGIES.md)
**Documento Maestro de Orquestación Algorítmica, Señales Cuánticas y Paradigma Continuo**

---

## 1. PARADIGMA CONTINUO CUÁNTICO UNIFICADO (ZERO DUALITY)

### QUÉ ES
Unificación total del motor de estrategias y gestión de posiciones en un **flujo cuántico continuo** (`TradeHorizon::Continuous`). Erradica la bifurcación artificial entre "Scalping" y "Swing", consolidando las decisiones en un único vector de convicción direccional adaptativo.

### POR QUÉ SE HACE
En cuentas de micro-capital ($13 USD), dividir el saldo en dos mitades (50% Scalp / 50% Swing) generaba fragmentación de margen. Dado que el requisito de notional mínimo en Binance Futures es de $5.05 USD, fragmentar el saldo provocaba que las órdenes fallaran por margen insuficiente (`Margin is insufficient`) o requirieran locks de exclusión mutua que bloqueaban operaciones válidas.

### PARA QUÉ SIRVE
Permite que el 100% del capital ($13 USD) se asigne de forma íntegra a la señal con mayor ventaja estadística (Edge). La posición evoluciona de forma fluida: si la microestructura detecta agotamiento en milisegundos, toma ganancias de inmediato; si el momentum se expande en una tendencia macro ($Hurst > 0.55$), la posición se mantiene activa y se protege mediante trailing stop continuo sin ser cerrada prematuramente por una etiqueta rígida.

### CÓMO FUNCIONA
1. **Extracción Tensorial:** En cada tick de mercado, `FeatureEngine` calcula 54 variables matemáticas en $O(1)$ (OFI, VPIN, Hurst, Entropía de Tsallis, Aceleración).
2. **Consenso de Ensamble:** Las 13 estrategias cuánticas de `signal-engine` y los modelos neuronales de `dark-alpha-engine` emiten votos ponderados por convicción bayesiana.
3. **Generación de Señal Continua:** Se emite un único `SignalIntent` continuo con su precio objetivo y distancia de stop derivados de la volatilidad instantánea (ATR sanitizado).
4. **Validación de Riesgo:** `RiskEngine::evaluate_quantum_order` evalúa la orden contra el 100% del capital unificado, dimensionando el apalancamiento mediante Kelly continuo y protegiendo el drawdown sin particiones artificiales.
5. **Gestión de Posición Unificada:** La orden se registra en `coin.positions.position`. Durante cada tick posterior, el trailing stop dinámico ajusta el nivel de salida en RAM en $< 100$ nanosegundos.

### CUÁNDO SE ACTIVA
En cada micro-tick L1/L2 recibido en tiempo real a través de WebSockets (`execution-engine`) o simulado en el motor de backtesting determinista (`backtest-engine`).

### DÓNDE RESIDE
- Kernel de Ejecución: `crates/god-engine-core/src/lib.rs` (`Quantum Unified Continuous Execution Engine`)
- Orquestador de Riesgo: `crates/risk-engine/src/lib.rs` (`evaluate_quantum_order`)
- Estado y Memoria: `crates/quantum-arena/src/position.rs` (`PositionManager::position`)
- Tipos de Horizonte: `crates/strategy-core/src/types.rs` (`TradeHorizon::Continuous`)

### QUIÉN LO EJECUTA
El bucle principal de `GodEngineCore`, auditado en tiempo real por `RiskEngine` y gobernado por el genoma evolutivo en `QuantumConfig`.

---

## 2. ENSAMBLE CUÁNTICO TENSORIAL (`crates/signal-engine`)

El ensamble de señales opera sin asignaciones dinámicas en memoria (Zero-Heap Allocation) y evalúa 13 modelos matemáticos en paralelo:

1. **`CoaxialBreakout`:** Detección de rupturas de volatilidad multicanal con bandas elásticas en micro-segundos.
2. **`QuantumOscillator`:** Superposición de probabilidades y fuerza restaurativa armónica para reversiones medias.
3. **`SolitonWave`:** Modelado no lineal de paquetes de energía en el libro de órdenes (ondas solitónicas).
4. **`SupersonicShockwave`:** Detección de discontinuidades de Mach en la propagación de órdenes en el libro L2.
5. **`StochasticResonance`:** Amplificación de señales débiles sumergidas en el ruido blanco del mercado.
6. **`HawkesBessel`:** Procesos puntuales auto-excitados para cuantificar la intensidad de clusters de trades institucionales.
7. **`RenyiTsallisEntropy`:** Medición de entropía no extensiva para detectar transiciones de fase y regímenes caóticos.
8. **`GameTheoreticNash`:** Equilibrio Minimax contra algoritmos predadores institucionales y bots de arbitraje.
9. **`PerceptronGate`:** Puerta perceptrónica hebbiana con actualización adaptativa de pesos por trade.
10. **`TurboScalper`:** Señalador HFT basado en desbalance de volumen delta e intensidad direccional de ticks.
11. **`MicroScalpTrigger`:** Inferencia conforme para entradas de alta probabilidad en micro-ventanas.
12. **`SwingConformalFilter`:** Filtrado de ruido intradiario para extensiones de tendencia macro.
13. **`TrendRunner`:** Expansión dinámica del Take Profit en regímenes de persistencia de Hurst ($H > 0.65$).

---

## 3. ARBITRAJE ESTADÍSTICO Y COINTEGRACIÓN (`crates/strategy-core`)

1. **`MultivariateCointegration`:** Detección de cointegración multiactivo con ajuste de reversión Ornstein-Uhlenbeck.
2. **`JohansenVecm`:** Modelo de Corrección de Error Vectorial con beta dinámico adaptativo para pares líderes/rezagados.
3. **`StatArbEngine`:** Arbitraje de divergencias con reversión rápida a la media en ventanas $O(1)$.
4. **`ConformalPrediction`:** Intervalos de confianza predictivos para TP y SL garantizados sin sesgos de sobreajuste.

---

## 4. MOTOR DE INFERENCIA NEURONAL CONTINUA (`crates/dark-alpha-engine`)

- **Inferencia en Tiempo Real:** Evaluación continua de redes neuronales optimizadas con SIMD (AVX2/AVX-512) para proyectar la probabilidad de éxito y la excursión direccional esperada.
- **Detección de Toxicidad y Flujo Institucional:** Cuantifica la toxicidad del libro de órdenes para evitar ejecuciones frente a participantes informados con asimetría de información.
- **Fusión Cuántica:** Las predicciones neuronales se integran directamente con los tensores de microestructura para alimentar el `SignalIntent` continuo en `GodEngineCore`.
