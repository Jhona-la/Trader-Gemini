# 🧠 REPORTE DE AUDITORÍA FORENSE: CALIBRACIÓN Y PERSONALIDAD DE ACTIVOS
**Trader Gemini — Módulo Maestro de Inteligencia y Gestión de Riesgo**

Este reporte detalla los hallazgos de la auditoría exhaustiva realizada sobre los módulos centrales encargados de la comprensión, clasificación y calibración de las particularidades de cada activo operado por el sistema:
*   [core/asset_classifier.py](file:///c:/Users/jhona/Documents/Proyectos/Trader%20Gemini/core/asset_classifier.py)
*   [core/asset_intelligence.py](file:///c:/Users/jhona/Documents/Proyectos/Trader%20Gemini/core/asset_intelligence.py)
*   [core/asset_parameter_engine.py](file:///c:/Users/jhona/Documents/Proyectos/Trader%20Gemini/core/asset_parameter_engine.py)
*   [risk/risk_manager.py](file:///c:/Users/jhona/Documents/Proyectos/Trader%20Gemini/risk/risk_manager.py)

---

## 🏛️ ESTRUCTURA Y COMPONENTES AUDITADOS

### 1. Clasificador de Activos (`AssetClassifier`)

*   **QUÉ:** Es un clasificador taxonómico estático que categoriza cada par de trading en una de tres clases institucionales: `AssetClass.MAJOR` (Monedas Principales), `AssetClass.ALT` (Altcoins Estándar) o `AssetClass.MEME` (Monedas de Alta Especulación/Memes).
*   **POR QUÉ:** Las diferentes monedas tienen dinámicas microestructurales fundamentalmente distintas. Las Majors (BTC, ETH) presentan baja volatilidad relativa y liquidez masiva; los Memes (DOGE, SHIB) experimentan explosiones de volatilidad impulsadas por el sentimiento social y no respetan patrones técnicos tradicionales como la reversión a la media.
*   **PARA QUÉ:** Sirve para aplicar filtros preliminares de riesgo en la apertura de posiciones, determinar qué estrategias son compatibles con cada activo y modular la agresividad del stop loss y la confianza requerida para disparar un trade.
*   **CÓMO:**
    1.  Extrae la base de la moneda (ej. de `BTC/USDT` o `BTCUSDT` obtiene `BTC`).
    2.  Verifica si la base pertenece al conjunto `KNOWN_MAJORS` (`{"BTC", "ETH"}`) o al conjunto `KNOWN_MEMES` (`{"DOGE", "SHIB", "PEPE", "FLOKI", "BONK", "WIF"}`).
    3.  Si pertenece a Majors se clasifica como `AssetClass.MAJOR`. Si pertenece a Memes se clasifica como `AssetClass.MEME`. De lo contrario, se clasifica por defecto como `AssetClass.ALT`.
    4.  Utiliza una caché interna (`self._cache`) en memoria para indexar el resultado del análisis del string del símbolo y evitar ejecuciones de split repetitivas en cada tick del motor.
*   **CUÁNDO:** Se activa cada vez que el motor de parámetros o el gestor de riesgos requiere clasificar un nuevo activo ingresado al sistema o inicializar su perfil.
*   **DÓNDE:** Clase `AssetClassifier` en [core/asset_classifier.py](file:///c:/Users/jhona/Documents/Proyectos/Trader%20Gemini/core/asset_classifier.py).
*   **QUIÉN:** Módulo de soporte taxonómico, consumido por `AssetIntelligence` y el `AssetParameterEngine`.

---

### 2. Módulo Maestro de Inteligencia (`AssetIntelligence`)

*   **QUÉ:** Es el componente central de control de políticas que gobierna los perfiles detallados de los activos (tiers, niveles de liquidez, perfiles de volatilidad) y enforza los pipelines estrictos de apertura de 7 pasos (A1-A7) y cierre de 7 pasos (C1-C7).
*   **POR QUÉ:** Permite evitar la simplificación del trading a través de parámetros uniformes que expondrían la cuenta de $13 USD a ruina estadística rápida. Es el "guardián institucional" de la cuenta.
*   **PARA QUÉ:**
    *   Filtrar señales no compatibles con el comportamiento del activo.
    *   Exigir un umbral de confianza proporcional a la volatilidad del activo (ej. BTC exige 0.58 de confianza, mientras que DOGE exige 0.60 y los genéricos 0.68).
    *   Prevenir congestiones de red o riesgos regulatorios (vetos en tiempo real para SOL y XRP).
    *   Enforzar buffers de liquidación virtual y salidas rápidas por tiempo para proteger el capital.
*   **CÓMO:**
    *   **Perfiles Estáticos Preestablecidos:** Inicializa perfiles específicos (`AssetProfile`) para `BTC/USDT`, `ETH/USDT`, `BNB/USDT`, `SOL/USDT`, `XRP/USDT` y `DOGE/USDT` con métricas asignadas de forma rígida en código (ej. Beta base, stop ATR multiplier base, fracción de Kelly, estrategias permitidas y restricciones). Si una moneda no está predefinida, utiliza una plantilla genérica rígida.
    *   **Pipeline de Apertura (A1-A7):**
        *   **A1 (Régimen):** Filtra estrategias basándose en la tendencia de mercado (ej. la estrategia de seguimiento de tendencia `TFTF` se bloquea en regímenes `CHOPPY` o `RANGING`).
        *   **A2 (Session/Timing):** Aplica cooldowns macro y ventanas óptimas.
        *   **A3 (Señal Primaria):** Compara la confianza de la señal (calculada usando el máximo entre `strength`, `ml_confidence` y `meta_confidence`) contra el umbral mínimo del activo (`profile.min_signal_threshold`).
        *   **A4 (Confirmación Multicapa):** Evalúa métricas avanzadas (ej. pullback volume ratio para TFTF, fuerza del Order Block > 1.5x ATR para OB_RETEST, clusters de liquidez para CASCADE).
        *   **A5 (Riesgo y Sizing):** Limita la cartera a un máximo de 3 posiciones simultáneas y valida que el tamaño nominal estimado sea ≥ $5.00 USD (mínimo de Binance).
        *   **A6 (No Colisión):** Aplica vetos rápidos basados en el estado de red global (ej. detiene operaciones en SOL si se detecta caída de red de Solana, o XRP si el caso de la SEC activa alertas).
        *   **A7 (Ejecución):** Ejecuta la auditoría del `SeniorAuditor`.
    *   **Pipeline de Cierre (C1-C7):**
        *   **C7 (Emergencia):** Cierra posiciones instantáneamente si se activa el Kill Switch global o vetos regulatorios de red.
        *   **C1 (Stops/TP):** Chequea stops. Incorpora un "Virtual Liquidation Buffer" (salida anticipada si la pérdida vulnerable se acerca al precio de liquidación del apalancamiento) y "Short Squeeze Emergency" (salida de cortos ante subidas rápidas).
        *   **C2 (Invalidación de Contexto):** Cierra la posición si la estructura que originó el trade muere (ej. en TFTF si el ADX cae por debajo de 20 o si hay divergencias de CVD de 3 velas en contra).
        *   **C5 (Tiempo Límite):** Límite máximo de retención para evitar capital atrapado (1 hora para Scalping, 48 horas para Swing).
        *   **C6 (Reversión/Auditoría):** Salidas por señales contrarias y validación del `SeniorAuditor`.
*   **CUÁNDO:** Se ejecuta de manera continua en cada tick del motor para verificar el estado de las posiciones abiertas (`verify_closing`) y en cada recepción de señal candidata (`verify_opening`).
*   **DÓNDE:** Clase `AssetIntelligence` en [core/asset_intelligence.py](file:///c:/Users/jhona/Documents/Proyectos/Trader%20Gemini/core/asset_intelligence.py).
*   **QUIÉN:** Módulo de Inteligencia de Activos del núcleo, consumido por `MetaCoordinator` y `RiskManager`.

---

### 3. Motor de Parámetros Dinámicos (`AssetParameterEngine`)

*   **QUÉ:** Es el motor matemático responsable de calcular dinámicamente el Take Profit (TP) y Stop Loss (SL) para cada activo basándose en su volatilidad histórica real y su microestructura del libro de órdenes actual.
*   **POR QUÉ:** La volatilidad no es un valor estático. Un Stop Loss fijo de 2.0% es demasiado amplio para BTC (que tiene un ATR típico de 0.5% en temporalidades menores), permitiendo pérdidas excesivas, pero demasiado ajustado para una altcoin de alta volatilidad (ej. WIF con ATR de 5.0%), provocando salidas prematuras por ruido del mercado.
*   **PARA QUÉ:** Optimizar la colocación de stops para que estén lo suficientemente alejados para tolerar el ruido del mercado y situar objetivos de TP matemáticamente factibles, garantizando siempre que la relación Riesgo/Beneficio (R:R) sea estadísticamente ganadora (mínimo de 1.5:1, target de 2.0:1).
*   **CÓMO:**
    1.  **Cálculo de Volatilidad OHLCV:** Extrae datos históricos y computa:
        *   **ATR-14 (Average True Range)** normalizado como porcentaje del precio actual.
        *   **Daily Range %** promedio del activo.
        *   **Volatilidad %** calculada como la desviación estándar de los retornos logarítmicos del precio de cierre.
    2.  **Aislamiento Temporal de ATR (Forensic-V130):** Calcula y guarda el ATR en tres temporalidades diferentes para evitar desfases de horizontes de trading:
        *   `atr_1m_pct` para el horizonte `MICROSCALPING`.
        *   `atr_5m_pct` para el horizonte `SCALPING` (representa ~70 minutos de contexto).
        *   `atr_1h_pct` para el horizonte `SWING` (representa ~14 horas de contexto).
    3.  **Fórmulas Adaptativas de Stop Loss y Take Profit:**
        *   **Scalping:**
            *   $SL = \text{Clamp}(ATR_{14} \times \text{dynamic\_scalp\_mult}, 0.15\%, 1.50\%)$
                *   *dynamic\_scalp\_mult* varía por activo: BTC = 0.40; ETH/BNB = 0.60; SOL/XRP/Memes = 0.85; otros = 0.70.
            *   $TP = \text{Clamp}(SL \times 2.0, 0.30\%, 1.50\%)$
        *   **Swing:**
            *   $SL = \text{Clamp}(ATR_{14} \times 1.50, 0.80\%, 5.00\%)$
            *   $TP = \text{Clamp}(SL \times 2.0, 1.50\%, 10.0\%)$
        *   **Filtro R:R Mínimo:** Si al aplicar los clamps el ratio $TP / SL < 1.5$, el motor ensancha el TP dinámicamente hasta alcanzar un ratio exacto de 1.5:1, en lugar de ajustar el SL, protegiendo al trade de salidas prematuras por ruido.
    4.  **Short Intelligence (Asimetría en Cortos):**
        *   Dado que las caídas y subidas de mercado presentan asimetrías de momentum (las subidas en short squeezes suelen ser extremadamente rápidas y violentas), el motor expande el stop loss en posiciones cortas (`SHORT`) multiplicando el multiplicador ATR por valores de 1.8x a 3.0x según la moneda (BTC=1.8, ETH=2.0, BNB/XRP=2.3, SOL=2.5, Memes=3.0). El TP se calcula consecuentemente como $SL \times 2.0$ y el clamp superior del TP se extiende hasta 1.5 veces el máximo habitual.
    5.  **Calibración Evolutiva:**
        *   En producción, el motor almacena perfiles en caché y verifica periódicamente su vigencia (`RECALIBRATE_INTERVAL_S = 3600` segundos = 1 hora).
        *   Si los datos están obsoletos o vacíos, consulta al `data_provider` o `data_handler` para traer las últimas 200 barras en el timeframe operativo (`1m`, `5m` o `1h`) y recalcular las métricas.
        *   Permite la inyección y lectura de calibraciones optimizadas offline a través del archivo `optimal_profiles.json`. Si un activo y horizonte se encuentran en dicho archivo, el motor utiliza los multiplicadores optimizados `sl_atr_mult` y `tp_rr_ratio` guardados allí.
*   **CUÁNDO:** Se consulta dinámicamente cada vez que se requiere calcular el tamaño de una posición (`size_position`) o generar una orden (`generate_order`) en el `RiskManager`, y recalcula la volatilidad del activo cada 1 hora.
*   **DÓNDE:** Clase `AssetParameterEngine` en [core/asset_parameter_engine.py](file:///c:/Users/jhona/Documents/Proyectos/Trader%20Gemini/core/asset_parameter_engine.py).
*   **QUIÊN:** Módulo de calibración dinámica de stop y profit, invocado por `RiskManager`.

---

## 🔍 INVESTIGACIÓN: ¿ADAPTATIVO O RÍGIDO GENERALIZADO?

A continuación se resume si los parámetros clave del sistema se calculan de manera adaptativa (basados en datos de mercado reales de cada activo) o si presentan valores estáticos o rígidos:

| Parámetro | ¿Es Adaptativo por Activo? | Mecanismo y Origen de Datos | ¿Es Evolutivo con el Tiempo? | Grado de Rigidez / Observaciones |
| :--- | :---: | :--- | :---: | :--- |
| **Take Profit (TP)** | **SÍ** | Multiplicador proporcional al Stop Loss dinámico ($SL \times 2.0$), el cual se basa en el ATR temporal del activo. | **SÍ** | Calibrado cada 1 hora según volatilidad histórica reciente. Posee límites rígidos de seguridad (*clamps* máximos/mínimos) para evitar targets inalcanzables en scalping. |
| **Stop Loss (SL)** | **SÍ** | Calculado directamente multiplicando el ATR-14 (de 1m, 5m o 1h según horizonte) por un factor de ruido de la moneda. | **SÍ** | Calibrado cada 1 hora. Los multiplicadores base son estáticos en código pero asimétricos para posiciones en Short. Posee clamps de seguridad estrictos. |
| **Apalancamiento** | **NO** | Parámetro estático definido por horizonte en `RiskManager.horizon_params` (`MICROSCALPING` = 20x, `SCALPING` = 10x, `SWING` = 5x). | **NO** | No cambia automáticamente con el ATR o la liquidez de la moneda. Sin embargo, puede ser sobreescrito si la señal enviada por la estrategia incluye un valor de leverage recomendado en su metadata. |
| **Tamaño de Posición (Sizing)** | **SÍ** (Cuentas > $50) / **RÍGIDO** (Cuentas < $50) | Para cuentas grandes: Criterio de Kelly dinámico usando tasa de acierto y ratio win/loss del activo. Para cuentas de $13: Margen fijo (95% para scalping, 50% para swing). | **SÍ** (Cuentas > $50) / **NO** (Cuentas < $50) | Las cuentas micro (< $50) operan bajo un protocolo rígido de **compounding agresivo** debido a limitaciones de Binance Futures, requiriendo usar el 95% o 50% del balance como margen para superar el nocional mínimo de $5.00. |

---

## 🔬 ANÁLISIS EN EL CONTEXTO DE LA CUENTA DE $13 USD

Para una cuenta de capital ultra-pequeño ($13 USD) con el objetivo de duplicación exponencial en 15 días con 100% de win rate en scalping, los hallazgos de esta auditoría forense revelan las siguientes ventajas y áreas críticas de atención:

### Ventajas de la Arquitectura Actual para $13 USD
1.  **Filtro de Símbolos en Fase 1 (Líneas 1417-1422 de `risk_manager.py`):**
    El sistema restringe las operaciones estrictamente a **BTC, ETH y SOL** cuando el capital es inferior a $50 USD. Esto es altamente beneficioso para proteger el capital contra spreads de orden elevados y deslizamientos (*slippage*) desastrosos que ocurren comúnmente en altcoins ilíquidas (Tier 4 y especulativos).
2.  **Sizing de Compounding Agresivo (95% de margen):**
    En lugar de arriesgar una fracción Kelly pequeña (que para $13 resultaría en una orden de $0.50 de margen que Binance rechaza por tamaño mínimo de orden), el sistema calcula el sizing escalando al 95% del capital como margen ($12.35). Con 10x de apalancamiento, esto genera una orden nominal de ~$123.50, cumpliendo cómodamente el mínimo de Binance ($5.00 nocional) y maximizando el poder del interés compuesto diario.
3.  **Cap de Posiciones Abiertas (Línea 1424):**
    Restringe a máximo 1 posición activa por horizonte temporal. Esto garantiza que la posición abierta de Scalping use todo el capital disponible sin competir por el margen con otros trades paralelos de la misma categoría, evitando liquidaciones colaterales por llamadas de margen.
4.  **Aislamiento de Horizontes (Scalping y Swing a la vez):**
    La arquitectura de `RiskManager` y `AssetIntelligence` está diseñada para soportar operaciones en ambos horizontes simultáneamente sin pisarse. El ledger de capital está segmentado y el límite de posiciones activas se aplica "por horizonte", permitiendo tener abierta una posición de Scalping (a 10x, arriesgando 95% del capital asignado) y una de Swing (a 5x, arriesgando 50% de su capital correspondiente) sin interferencia mutua.

### Riesgos y Puntos de Falla Críticos
1.  **Riesgo de Ruina por Apalancamiento Agresivo (95%):**
    Arriesgar el 95% del balance en un solo trade de scalping significa que un movimiento en contra que supere el Stop Loss provocará la pérdida de casi toda la cuenta. Para que el capital se duplique exponencialmente, la tasa de acierto (Win Rate) debe ser de hecho cercana al 100%, o el Stop Loss debe ser implacablemente preciso.
2.  **Régimen de Mercado Desconocido (`UNKNOWN`):**
    Se detectó un parche en `core/asset_intelligence.py` (líneas 256-263) que desactiva el veto por régimen "UNKNOWN". Esto evita bloqueos de backtest pero en producción real podría permitir que el sistema tome trades de seguimiento de tendencia (`TFTF`) durante mercados de lateralización extrema (*choppy*) si el clasificador HMM o de régimen no ha sincronizado correctamente.
3.  **Falta de Apalancamiento Adaptativo por Volatilidad:**
    Aunque el Stop Loss en porcentaje se adapta dinámicamente según el ATR, el apalancamiento es estático (10x para Scalping). Esto significa que en momentos de expansión de volatilidad (donde el ATR sube y por ende el Stop Loss se ensancha a 1.2% en lugar del 0.25% promedio), el tamaño nominal en Binance se mantiene idéntico. Un Stop Loss de 1.2% a 10x apalancamiento representa una pérdida del 12% del capital de margen de un solo golpe. El apalancamiento debería reducirse inversamente a la expansión del ATR para mantener el riesgo monetario constante.

---

## 📈 RECOMENDACIONES DE MEJORA Y OPTIMIZACIÓN

1.  **Apalancamiento Inversamente Proporcional a la Volatilidad (ATR):**
    Modificar la asignación de apalancamiento en `size_position` para que no sea fija (10x o 20x), sino que se calcule dinámicamente:
    $$\text{Leverage} = \frac{\text{Riesgo Máximo de Margen}}{\text{Stop Loss \% Calculado}}$$
    Esto asegura que si el ATR aumenta y el stop loss se ensancha, el apalancamiento se reduzca automáticamente para mantener el riesgo controlado en $13 USD.
2.  **Optimización y Calibración Periódica de `optimal_profiles.json`:**
    Garantizar que el optimizador evolutivo actualice el archivo `optimal_profiles.json` de manera constante basándose en el historial reciente de trades. Esto permitirá al sistema "aprender" si el multiplicador de stop actual (ej. 0.40x ATR) está siendo tocado por ruido normal con demasiada frecuencia, o si los objetivos de profit en short squeezes deben refinarse.
3.  **Centralización de Datos Históricos en Backtest y Producción:**
    Asegurar que los datos de entrada del `data_handler` en vivo posean la misma resolución y cobertura que los datos históricos parquet para evitar divergencias críticas en la calibración del ATR entre backtesting y producción real.
