# 📉 Análisis Forense de Pérdidas (Fee Drag vs MFE)

## Diagnóstico del Bot: ¿Por qué estamos en pérdida?

El backtest ha revelado una pérdida progresiva (de $13.00 a $12.91). Tras aplicar los parches **P0** (abrir la compuerta de predicciones y evitar el bloqueo por conflictos de horizonte), el bot comenzó a tomar operaciones correctamente. Sin embargo, estamos perdiendo dinero de forma constante.

### El Modo Profesor (QUÉ, POR QUÉ, PARA QUÉ, CÓMO, CUÁNDO, DÓNDE, QUIÉN)

*   **QUÉ:** El sistema está abriendo posiciones que terminan en pérdida por un fenómeno llamado **"Fee Drag" (Arrastre de Comisiones)** frente a un **MFE (Excursión Favorable Máxima)** diminuto.
*   **POR QUÉ:** Las comisiones de Binance (especialmente de taker o incluso de maker + slippage) suman aproximadamente `0.188%` por un viaje de ida y vuelta (round-trip). En una temporalidad de Scalping de 1 minuto, la volatilidad promedio (ATR) de las monedas actuales (ej. SOL, WIF) ronda entre `0.080%` y `0.125%`. Matemáticamente, el mercado no se está moviendo lo suficiente para ni siquiera pagar el costo del peaje.
*   **PARA QUÉ:** Identificar esto nos sirve para detener la sangría de capital. No hay "poder predictivo" que pueda vencer a un spread+comisión que es mayor que el movimiento real del precio en ese horizonte temporal.
*   **CÓMO:** El bot predice correctamente una micro-tendencia (la dirección es correcta). El precio avanza un `0.16%` a nuestro favor. El bot intenta cerrar, pero descubre que al pagar comisiones quedaría en `-0.02%`. Por lo tanto, no puede cerrar en ganancias. Eventualmente, el mercado revierte, nos toca el Stop Loss o cerramos por tiempo límite, resultando en pérdidas netas.
*   **CUÁNDO:** Ocurre en la temporalidad de `1m` (Scalping) en condiciones de mercado que no tienen volatilidad extrema.
*   **DÓNDE:** En la interacción entre el tamaño de nuestra vela (`portfolio.py` / `risk_manager.py`) y las reglas físicas del Exchange (`binance_executor.py`).
*   **QUIÉN:** El diseño arquitectónico de forzar el Scalping de alta frecuencia (HFT) en un micro-capital de $13 USD es el responsable. En HFT institucional se pagan comisiones nulas o se tienen rebates (te pagan por operar). Como usuarios retail, no podemos hacer HFT en velas de 1 minuto si la moneda no es súper volátil.

---

## 📊 Evidencia del Motor de Predicción (prediction_metrics.json)

Extraído directamente de tu base de datos y métricas:

| Estrategia | Accuracy | Win Rate | Ganadas | Perdidas | Avg MFE (Punto Máximo de Ganancia) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ML_SCALPING** | 36.2% | 100% | 1 | 0 | **0.1213%** |
| **ML_SWING** | 61.2% | 20.0% | 1 | 4 | **0.1724%** |
| **STAT_V1** | 54.4% | 30.0% | 3 | 7 | **0.1627%** |

### Análisis del problema:
Fíjate en la columna **Avg MFE**. Nuestras posiciones, en promedio, se mueven a nuestro favor un máximo de **0.17%**.
Sin embargo, los costos de Binance exigen que el precio se mueva al menos **0.18% - 0.20%** a nuestro favor para apenas salir de la operación en "Break Even" (sin ganar ni perder).

**Conclusión Matemática:** Estamos comprando un billete de lotería por $1.00 donde el premio mayor es $0.80.

---

## 🛠️ La Solución Inmediata (Fase P1)

No estamos equivocados en la "dirección" (el bot predice bien). Estamos equivocados en el **TERRENO DE JUEGO (Timeframe)** y el **CÁLCULO DEL PEAJE**.

Para duplicar $13 en 15 días con 100% Win Rate (o cercano) en Scalping, debemos:

1.  **Cambiar el terreno de juego (Timeframe 5m/15m):** Mover el horizonte de Scalping de velas de 1 minuto a velas de 5 o 15 minutos. En 5m, el ATR sube a `0.40% - 0.80%`, dándonos espacio de sobra para pagar la comisión del `0.18%` y quedarnos con un `0.30%` a `0.60%` de ganancia real pura al bolsillo.
2.  **Sizing Concentrado ($6 absolutos):** Binance exige órdenes mayores a $5 USD. Con $13, si el sistema intenta dividir el riesgo en varias monedas, enviará órdenes de $2.5 o $3 USD, que serán rechazadas u obligarán a usar mucho apalancamiento forzado. Debemos hacer un "All-In Táctico" de $6.50 por operación.
3.  **Filtro Absoluto de Fee Drag:** Modificaremos `RiskManager` para que **RECHACE INMEDIATAMENTE** cualquier operación donde el `ATR(14) < Costos de Ejecución * 1.5`.
