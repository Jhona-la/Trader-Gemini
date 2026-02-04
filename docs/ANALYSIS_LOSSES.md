# 📉 ANÁLISIS DE PÉRDIDAS - TRADER GEMINI (Feb 2026)

> **Regla de Oro:** "El 100% de las pérdidas tocaron el Stop Loss completo. El sistema de defensa dinámico no está actuando antes del impacto."

---

## 1. 🔍 Radiografía de las 33 Operaciones Perdedoras

Este análisis disecciona las 33 operaciones fallidas del backtest de 30 días (88 trades totales).

### 1.1 Patrones Temporales (Hora UTC)

| Sesión | Horario | # Losses | % Total | Observación |
|---|---|---|---|---|
| **Asian** | 00:00 - 07:00 | 10 | 30% | Losses dispersas, bajo volumen. |
| **London** | 08:00 - 12:00 | 3 | 9% | ✅ **Zona más segura**. |
| **Overlap** | 13:00 - 14:00 | 4 | 12% | Inicio de volatilidad. |
| **NY (Open)** | 15:00 - 18:00 | **13** | **39%** | 🚨 **ZONA CRÍTICA DE PELIGRO**. |
| **NY (Close)** | 19:00 - 23:00 | 3 | 9% | Cierre de sesión tranquilo. |

> **Hallazgo #1:** El **39% de las pérdidas** se concentran en solo 4 horas de la sesión de Nueva York (15:00-18:00 UTC). La volatilidad direccional de NY rompe los rangos de scalping.

### 1.2 Volatilidad y Régiem

*   **ATR Promedio (Losses):** 0.1321%
*   **ATR Promedio (Winners):** 0.1369%
*   **Deltas:** La diferencia es despreciable. **No es la volatilidad per se** lo que mata el trade, sino la dirección repentina.
*   **Régimen:** Clasificadas 100% como "RANGING" (Mercado lateral).
    *   *Hipótesis:* La estrategia trata de operar reversiones a la media (Ranging) justo cuando el mercado rompe el rango (Breakout en NY), quedando atrapada.

### 1.3 Efectividad Stop-Loss

| Tipo de Salida | Cantidad | % | Conclusión |
|---|---|---|---|
| **Full Loss (-1.5%)** | 33 | 100% | ❌ El precio nunca dio respiro. |
| **Trailing Stop** | 0 | 0% | El trade fue en contra desde el inicio. |
| **Early Close** | 0 | 0% | No hubo señal de salida anticipada. |

> **Hallazgo #2:** Entradas "Cuchillo Cayendo". El precio cruza la entrada y va directo al SL sin rebote.

---

## 2. 🛡️ Recomendaciones para Risk Manager (`risk_manager.py`)

### A. Implementar "New York Filter"
**Problema:** La estrategia de scalping sufre en la apertura agresiva de NY.
**Acción:** Configurar un multiplicador de riesgo por horario.
```python
# Pseudo-código para risk_manager.py
hour = event.timestamp.hour
if 14 <= hour <= 17:
    risk_multiplier = 0.5  # Reducir tamaño a la mitad en NY Open
```

### B. Ajuste de Stop Loss (ATR-Based)
**Problema:** SL fijo de 1.5% es arbitrario. En baja volatilidad (0.13%), 1.5% es una eternidad (11x ATR). Si el precio mueve 11x ATR en contra, la tesis del trade murió hace mucho.
**Acción:** Cambiar SL fijo a SL dinámico más ajustado.
*   **Propuesta:** `SL = Entry ± (ATR * 3)`
*   *Impacto:* Si ATR=0.13%, SL = 0.39%. Reduciría la pérdida promedio por trade de -1.5% a -0.4%, mejorando drásticamente el Drawdown.

### C. Filtro de "Range Breakout"
**Problema:** Operar reversión (RSI Oversold) durante un breakout fuerte causa pérdidas inmediatas.
**Acción:** Validar ADX < 25 ESTRICTO antes de entrar en reversión. Si ADX sube, prohibir contra-tendencia.

---

## 3. 🎯 Próximos Pasos Sugeridos
1.  **Modificar `risk_manager.py`:** Implementar el ajuste de SL basado en ATR (3x-5x) en lugar de porcentaje fijo.
2.  **Backtest de Validación:** Correr el backtest nuevamente con SL = 3x ATR.
    *   *Predicción:* Win Rate podría bajar levemente (ruido), pero Profit Factor y Sharpe subirán al cortar pérdidas rápido.
