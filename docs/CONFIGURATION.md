# 🔧 CONFIGURATION GUIDE — TRADER GEMINI (PRO)

Este documento es la **Fuente Única de Verdad** para la configuración del motor Trader Gemini. Todos los parámetros aquí descritos residen físicamente en `config.py` y son inyectados en el `Metal-Core` durante el arranque.

---

## 🏷️ I. PROTOCOLO DE ETIQUETADO (STRATEGY_LABELS)

Para permitir la auditoría forense de micro-cuentas ($13 USD), cada estrategia debe estar explícitamente etiquetada. Esto permite al `Portfolio` rutear las ganancias y pérdidas a ledgers aislados.

```python
STRATEGY_LABELS = {
    "technical": "[SCL] Hybrid Engine",
    "ml_strategy": "[SWG] XGBoost Supreme",
    "statistical": "[SCL] Stat-Arb V1",
    "sniper": "[SCL] Sniper Ultra"
}
```

*   **[SCL]**: Scalping. Operaciones de alta frecuencia, targets cortos, protección agresiva.
*   **[SWG]**: Swing. Operaciones de baja frecuencia, targets largos, mayor tolerancia al ruido.

---

## 📈 II. ESPECIALIZACIÓN POR HORIZONTE

El sistema ya no usa parámetros globales para todas las temporalidades. Se aplican diccionarios especializados:

### 1. `SCALPING_PARAMS` (Frecuencia HFT)
Optimizado para capturar micro-movimientos en velas de 1m-5m.
- `tp_pct`: 0.0025 (0.25%) — Cubre el fee y genera ganancia rápida.
- `sl_pct`: 0.0050 (0.50%) — Corte quirúrgico de pérdidas.
- `rsi_buy`: 35 / `rsi_sell`: 65 — Umbrales relajados para mayor frecuencia.
- `bb_std`: 1.5 — Bandas más estrechas para detectar explosiones de volatilidad.

### 2. `SWING_PARAMS` (Estructural)
Optimizado para tendencias de 1h-4h.
- `tp_pct`: 0.015 (1.50%) — Captura de movimientos direccionales.
- `sl_pct`: 0.02 (2.00%) — Espacio para "respirar" ante el ruido del mercado.
- `ema_trend`: 200 — Filtro de tendencia institucional.

---

## 🔒 III. GESTIÓN DE RIESGO PARA MICRO-CUENTAS

### `MAX_DRAWDOWN` (Kill-Switch)
- **Valor Recomendado**: 0.02 (2.0%).
- **Lógica**: Si la cuenta de $13 cae por debajo de $12.74, el `KillSwitch` aniquila todas las posiciones. Esto protege el capital base para una re-entrada posterior.

### `POSITION_SIZE_MICRO_ACCOUNT`
- **Valor**: 0.30 (30% equity).
- **Apalancamiento**: 10x-20x.
- **Resultado**: Una posición nocional de ~$39-78 USD, lo cual supera el mínimo de Binance ($5 USD) permitiendo operar con capital pequeño.

---

## 🚦 IV. PARÁMETROS DE RED Y LATENCIA

- `REST_TIMEOUT`: 5s (Reducido de 20s para evitar bloqueos del Event Loop).
- `WS_RECONNECT_INTERVAL`: 10s.
- `HEARTBEAT_INTERVAL`: 30s.

---

> [!CAUTION]
> **ADVERTENCIA:** No modifique la estructura jerárquica de `Config.Strategies` o `Config.Risk`. El motor de inyección de `engine.py` depende de esta estructura para la validación de tipos en tiempo de ejecución.
