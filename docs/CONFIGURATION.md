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

## 📈 II. ESPECIALIZACIÓN POR HORIZONTE (SSOT UNIFICADO)

El sistema implementa una **Fuente Única de Verdad (Single Source of Truth - SSOT)** de manera dinámica a nivel de módulo. Las variables redundantes `Config.Strategies.SCALPING_PARAMS` y `Config.Strategies.SWING_PARAMS` han sido enlazadas directamente a `Config.Horizons.Scalping` y `Config.Horizons.Swing` al final de `config.py`. Cualquier cambio se propaga instantáneamente a ambas interfaces sin discrepancias.

### 1. `SCALPING_PARAMS` / `Horizons.Scalping` (Frecuencia HFT)
Optimizado para capturar micro-movimientos en velas de 1m-5m mitigando el Fee Drag con un Payoff altamente positivo:
- `tp_pct`: 0.0045 (0.45%) — [ROUND3-OPT] Incrementado desde 0.30% para mitigar el Fee Drag de Binance y asegurar PnL neto positivo.
- `sl_pct`: 0.0030 (0.30%) — [ROUND3-OPT] Ajustado para un Payoff ratio saludable (1.5:1 TP/SL) adaptado al ruido de 1m.
- `rsi_buy`: 35 / `rsi_sell`: 65 — Umbrales relajados para mayor frecuencia.
- `bb_std`: 1.5 — Bandas más estrechas para detectar explosiones de volatilidad.
- `max_hold_bars`: 45 — [ROUND3-OPT] Reducido drásticamente de 120 a 45 barras (45m) para cortar trades zombis estancados en pérdidas.

### 2. `SWING_PARAMS` / `Horizons.Swing` (Estructural)
Optimizado para tendencias en velas de 1h-4h:
- `tp_pct`: 0.045 (4.50%) — Captura de movimientos estructurales.
- `sl_pct`: 0.025 (2.50%) — Espacio para "respirar" ante fluctuaciones en marcos temporales altos.
- `strength_threshold`: 0.45 — [ROUND3-OPT] Reducido de 0.55 a 0.45 para reactivar Swing y evitar el sobre-filtrado de confluencia que impedía la generación de señales.
- `ema_trend`: 200 — Filtro de tendencia institucional (Golden/Death Cross).

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

## 🔒 V. INTEGRACIÓN NEXUS Y AJUSTE DE CACHÉ (MODO PROFESOR)

Para garantizar la estabilidad en la ejecución y prevenir el envejecimiento de las señales bajo condiciones de alta volatilidad, la especificación **NEXUS** introduce nuevos parámetros y políticas de caché:

### 1. TTL Absoluto de Señal y Validación de Expiración
- **QUÉ**: Es la configuración y validación de la validez temporal de una señal utilizando un timestamp absoluto de expiración (`expiration_timestamp`), calculado al momento de generarse como `timestamp + ttl_seconds`.
- **POR QUÉ**: Antes, la expiración de la señal se calculaba dinámicamente en el Engine comparando la edad relativa con un umbral global (`MAX_SIGNAL_AGE`). Si el reloj del sistema experimentaba deriva (drift) o había demoras en la cola de procesamiento, las señales se validaban incorrectamente, pudiendo ejecutar órdenes basadas en señales viejas (anomalías HFT).
- **PARA QUÉ**: Evitar el deslizamiento temporal de las ejecuciones, garantizando que ninguna señal de Scalping que haya pasado más de N segundos en el bus de eventos sea procesada por el módulo de ejecución real.
- **CÓMO**: Se define `ttl` por defecto en cada señal (ej. 30 segundos). El Engine ejecuta `_validate_signal_ttl(event)` comparando `time.time() > event.expiration_timestamp`. Si se supera, el estado pasa a `SignalState.EXPIRED` y se veta la ejecución.
- **CUÁNDO**: Ocurre en el instante inmediato anterior a que el Engine intente derivar la señal al Risk Manager.
- **DÓNDE**: Configurado en la estructura de estrategia/horizonte (`config.py`) y evaluado en el bucle del Engine (`core/engine.py`).
- **QUIÉN**: El Engine es el encargado absoluto de realizar este control.

### 2. Caché del Puente Neuronal (`NeuralBridge`)
- **QUÉ**: Es el almacenamiento temporal y seguro (Thread-Safe) de las inferencias estadísticas e ideas neuronales generadas por los modelos ML para cada activo.
- **POR QUÉ**: El cálculo e inferencia ML consume ciclos de CPU significativos. Multiplicar esto por 42 símbolos saturaba los recursos en la micro-cuenta de $13 USD. El NeuralBridge actuaba como un stub sin memoria (devolvía diccionarios vacíos), haciendo inútiles los intentos de consenso de las estrategias.
- **PARA QUÉ**: Optimizar el uso de CPU, amortiguar la latencia de consulta y permitir que estrategias como `SniperStrategy` y el consenso accedan a las expectativas direccionales del modelo de forma instantánea (< 1ms).
- **CÓMO**: Se implementa un diccionario en memoria protegido por un lock de exclusión mutua (`threading.Lock`). Cada insight guardado tiene un tiempo de vida (TTL) de **300 segundos**. Si una estrategia consulta un insight que superó este TTL, el NeuralBridge lo descarta y retorna un contenedor vacío.
- **CUÁNDO**: Al publicar un insight desde el sub-proceso de inferencia (`publish_insight`) y al consultarlo desde cualquier estrategia de confluencia (`query_insight`).
- **DÓNDE**: Implementado en el módulo central de sinapsis (`core/neural_bridge.py`).
- **QUIÉN**: La clase singleton `NeuralBridge` (`neural_bridge`) es la propietaria del almacenamiento y verificación del ciclo del caché.

---

## 🚦 VI. FASES DE CAPITAL Y LÍMITES CONCURRENTES (MODO PROFESOR)

Para evitar la sobreexposición y garantizar un crecimiento compuesto defensivo adaptado al volumen real de la cuenta, el gestor de riesgos implementa un control matricial de Tiers de capital.

### 1. Definición y Matriz de Tiers de Capital
- **QUÉ**: Reglas de restricción operativa que limitan la cantidad de posiciones simultáneas y los activos operables basándose en la equidad (`equity`) neta de la cuenta.
- **POR QUÉ**: Cuentas extremadamente pequeñas (como la cuenta inicial de $13 USD) tienen nulo margen de error frente a drawdowns de correlación cruzada. Si el bot abre 5 posiciones simultáneas de $4 de margen, la cuenta se liquida en segundos ante un mechazo generalizado. A medida que el capital crece, la diversificación se expande de forma controlada.
- **PARA QUÉ**: Prevenir margin calls catastróficas en fases tempranas y permitir escalabilidad institucional en fases avanzadas.
- **CÓMO**: El gestor de riesgos evalúa la equidad actual en cada cálculo de `size_position()` y clasifica la cuenta en uno de los 6 Tiers:
  - **Tier 1 (Micro-Scalper)**: Equity < $50.00. Máximo **1 posición abierta**. Símbolos permitidos limitados a los 4 más líquidos: `BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `BNBUSDT`. Las comisiones de transacción (fees) estimadas para la entrada y salida deben representar menos del 1.0% de la equidad total de la cuenta.
  - **Tier 2 (Growth Retail)**: $50.00 <= Equity < $200.00. Máximo **2 posiciones abiertas**. Símbolos permitidos: Tier 1 + `ADAUSDT` y `XRPUSDT`.
  - **Tier 3 (Standard Retail)**: $200.00 <= Equity < $1,000.00. Máximo **4 posiciones abiertas**. Símbolos permitidos: Tier 2 + `LTCUSDT`, `LINKUSDT`, `DOTUSDT` y `AVAXUSDT`.
  - **Tier 4 (Advanced Trader)**: $1,000.00 <= Equity < $10,000.00. Máximo **8 posiciones abiertas**. Todos los símbolos de la cesta operativa están permitidos.
  - **Tier 5 (Institutional Core)**: $10,000.00 <= Equity < $100,000.00. Máximo **12 posiciones abiertas**. Todos los símbolos permitidos.
  - **Tier 6 (Sovereign Engine)**: Equity >= $100,000.00. Máximo **12 posiciones abiertas**. Todos los símbolos de la cesta permitidos con apalancamientos optimizados.
- **CUÁNDO**: Validado dinámicamente antes de autorizar cualquier tamaño de trade en `size_position`.
- **DÓNDE**: `risk/risk_manager.py` (método `size_position()`).
- **QUIÉN**: El `RiskManager`.

---

## 🧬 VII. CONFIGURACIÓN DEL REGISTRO DE ADN DE ESTRATEGIAS (STRATEGY_DNA) (MODO PROFESOR)

El sistema enforza un mapa genético estricto para cada estrategia operada en Trader Gemini para evitar discrepancias lógicas entre la apertura y el cierre de las posiciones.

### 1. Definición y Mapeo Genético de Estrategias
- **QUÉ**: La estructura `STRATEGY_DNA` en `core/senior_auditor.py` que almacena los parámetros e indicadores necesarios para auditar la salud de cada trade.
- **POR QUÉ**: Diferentes estrategias requieren diferentes lógicas de invalidación y salidas técnicas (ej. TFTF requiere fuerza de tendencia ADX, mientras que Mean Reversion Bollinger requiere descompresión lateral). Centralizar el ADN evita que un motor genérico cierre un trade por la razón incorrecta.
- **PARA QUÉ**: Asegurar que las posiciones abiertas por el bot conservan su tesis operativa original a lo largo de todo su ciclo de vida.
- **CÓMO**: Se configuran las propiedades fundamentales en el diccionario `STRATEGY_DNA`:
  - `regime_requerido`: Tipo de régimen de mercado para la admisión (`TENDENCIAL`, `LATERAL`, `POST-RANGO`, `CUALQUIERA`, etc.).
  - `indicadores_de_invalidacion`: Lista de métricas técnicas que invalidan inmediatamente la tesis (ej. `ADX < 20` para TFTF, `ob_extremum_violated` para OB_RETEST).
  - `tiempo_maximo_de_validez`: Ventana temporal estricta de vida del trade (ej. 90 segundos para LCA, 48 horas para OCS).
- **CUÁNDO**: Configurado estáticamente y leído por la clase `SeniorAuditor` durante las fases de apertura, seguimiento y cierre.
- **DÓNDE**: Definido en `core/senior_auditor.py`.
- **QUIÉN**: El `SeniorAuditor`.

---

> [!CAUTION]
> **ADVERTENCIA:** La unificación SSOT dinámica garantiza que cualquier cambio en `Config.Horizons` se aplique a `Config.Strategies`. No altere el código de mapeo dynamic al final de `config.py` para evitar rotura de referencias en el motor de inyección de `engine.py`.

---

## 🧪 VIII. CONFIGURACIÓN DE ENTORNOS Y OVERRIDES DE PRUEBAS (MODO PROFESOR)

Para garantizar la estabilidad en la ejecución de las pruebas unitarias y de integración, así como el comportamiento determinista, el sistema anula ciertos componentes de red en el entorno de pruebas (`TEST`).

### 1. Desactivación de Notificaciones de Red y Logs de Terceros
- **QUÉ**: Forzado de las variables de entorno `TELEGRAM_ENABLED=False`, `EMAIL_ENABLED=False`, y `WANDB_MODE=disabled` al iniciar la suite de pruebas en `conftest.py`.
- **POR QUÉ**: El archivo de variables locales `.env` del usuario puede tener habilitadas las integraciones de producción (mensajería y alertas reales). Si pytest las ejecuta, causará tráfico HTTP/SMTP real a la API de Telegram y Gmail. Esto genera cuellos de botella por timeouts, bloquea los hilos no-demonio de `Notifier` y cuelga el proceso de pytest al finalizar.
- **PARA QUÉ**: Evitar llamadas de red reales a APIs externas en entornos de pruebas aislados, previniendo latencias de red y fugas de hilos que bloqueen el desmontaje de la suite.
- **CÓMO**: Se configuran directamente en `os.environ` antes de que pytest importe los componentes de trading.
- **CUÁNDO**: Ocurre automáticamente en la inicialización global de pytest (`conftest.py`).
- **DÓNDE**: Definido al inicio de [conftest.py](file:///c:/Users/jhona/Documents/Proyectos/Trader%20Gemini/conftest.py).
- **QUIÉN**: El cargador de entorno de pytest.



