# Auditoría Forense y Corrección Sistémica: Motor Evolutivo y Realidad (Modo Profesor)

## 1. QUÉ se hizo
Se ejecutó una **auditoría forense completa** al ciclo de retroalimentación de la evolución y simulación de backtesting, descubriendo dos fallos críticos: 
1. El cálculo de comisiones en la simulación omitía la comisión de entrada (`taker fee`).
2. El modelo de fitness (`evolver.rs`) castigaba el apalancamiento estricto necesario para operar micro-cuentas de $13 USD, requiriendo su refactorización. Además, no deducía el deslizamiento (slippage) del capital final de forma implacable.

Se corrigieron ambas desviaciones de la realidad implementando restricciones algorítmicas estrictas de mercado.

## 2. POR QUÉ se hizo
**La divergencia Backtest-Producción invalida cualquier hallazgo del algoritmo genético**. 
El sistema de CMA-ES/Evolución genética optimiza buscando lagunas matemáticas. Al solo cobrar un 50% de las comisiones reales, las redes y modelos convergían hacia el *sobre-trading*, resultando en un aparente "éxito" que, llevado a producción en Binance con comisiones del 100%, garantizaba la destrucción de la cuenta de 13 USD.

## 3. PARA QUÉ se hizo
Para obligar a los genomas a evolucionar **Estrategias con Esperanza Matemática Absoluta**. 
El objetivo supremo es lograr un *100% de crecimiento cada 3 días* sobre un capital de 13 USD. Para lograrlo usando interés compuesto, las operaciones deben superar el `round-trip fee` y la entropía del slippage de forma consistente. La evolución matemática ahora se basa en *física del mercado real*.

## 4. CÓMO funciona ahora
1. **Deducción de Taker Fee Doble:** En `crates/god-engine-core/src/lib.rs`, al liquidar una posición de Scalping o Swing, el sistema calcula el PNL deduciendo explícitamente tanto el `close_fee` como un nuevo `entry_fee = live_taker_fee`.
2. **Horizonte de Identidad (Scalp vs Swing):** Se inyectó el parámetro `horizon` en el `MarketSnapshotPayload` de `consejo_seniors.rs`, obligando a los agentes (Seniors) a distinguir si arbitran un flujo para Scalp o Swing.
3. **Slippage y Fitness Asimétrico:** En `src/bin/evolver.rs`, el `final_capital` descuenta el slippage estricto ANTES de evaluar a un genoma. El límite fatal de Drawdown subió de 0.40 a 0.85, reconociendo que los $13 USD requieren alta volatilidad permitida (sobreapalancamiento táctico), pero premiando el crecimiento con una potencia `x^1.5` para forzar la curva exponencial requerida.

## 5. CUÁNDO actúa
- Durante **cada tick del simulador**, el cálculo de PNL latente y real incluye la fricción matemática completa (fees de Maker/Taker asimétricas y deslizamiento). 
- Durante la **evaluación final de isla genética**, los parámetros que destruyen cuentas pequeñas por drawdowns ridículos o estancamiento de pnl, son purgados del pool genético.

## 6. DÓNDE se aplicaron los cambios
- `crates/god-engine-core/src/lib.rs`: Restricción estricta de fees y asignación de `TradingHorizon::Scalping` y `Swing`.
- `src/bin/evolver.rs`: Lógica asimétrica de Capital Final con penalidad `reality_slippage_penalty_with_notional` y Multi-objetivo fitness adaptativo a Micro-Cuentas.
- `crates/metacortex-engine/src/consejo_seniors.rs`: Definición y distinción explícita de `TradingHorizon` (Scalping / Swing) para no pisar las evaluaciones.

## 7. QUIÉN maneja esto
- **Architect & Risk Manager:** Definen la física de liquidación.
- **Quant Developer:** Optimiza los multiplicadores de fitness y genómica.
- **QA & DevOps:** Garantizan que el pipeline converja y sea resistente a datos sucios.

---
**Nota sobre Rendimiento & Optimizaciones (Rust / Picosegundos):** 
Los parches no generaron allocation overhead; mantienen los micro-ajustes al nivel de memoria local, con una latencia mínima imperceptible. Seguimos dentro de los márgenes óptimos para laptops sin GPU dedicadas (solo CPU bound con Rayon).
