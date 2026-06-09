# 🛡️ TRADER GEMINI: COMPLIANCE & RISK MANAGEMENT

Este documento certifica el Nivel The Seguridad Militar de la aplicación y la Matriz The Mitigación de Pérdidas de **Trader Gemini**, un requerimiento vital para escalamiento algorítmico asincrónico real en operaciones Scalper.

---

## 🔐 I. DIAGRAMA DE BÓVEDA (VAULTING)

La memoria the la Máquina Virtual de Python es vulnerable al *Heap-Scraping*. Implementamos cifrado de grado militar sobre las credenciales utilizando AES-GCM-256 (Ephemeral In-Memory Vault).

```mermaid
graph LR
    subgraph Storage & Environment
        Env[(.env / OS)]
    end

    subgraph Security Vault (config.py)
        EP[Key Derivation<br/>Ephemeral Fernet Key]
        SEC[SecureString Object<br/>Encrypted Bytes]
    end

    subgraph Execution Area (Restringido)
        CCXT[Binance ccxt Client]
    end

    Env -->|Raw Secret/API| EP
    activate EP
    EP -->|Encrypt AES-256| SEC
    deactivate EP
    
    SEC -.->|Garbage Collection Seguro| SEC 

    CCXT -->> SEC: Solicitar Decrypt Temporal .get_unmasked()
    activate SEC
    SEC -->|Raw Secret (Vida: 1ms)| CCXT
    deactivate SEC
    
    classDef secure fill:#4a148c,stroke:#ab47bc,stroke-width:2px,color:#fff;
    classDef api fill:#e65100,stroke:#fb8c00,stroke-width:2px,color:#fff;
    
    class EP,SEC secure;
    class CCXT api;
```

---

## 📐 II. MAPEO DE LA CAPA 'CRITERIO-AXIOMA' (Mathematical Integrity)

Cada orden y cálculo de PnL atraviesa una auditoría financiera de doble-chequeo basada en **Numpy Float64** (64-bit IEEE 754) procesado mediante kernels **Numba JIT** (`utils/math_kernel.py`). Esto elimina el overhead de `decimal.Decimal` (28 dígitos) que, aunque preciso, era 1,000x veces más lento e incompatible con la latencia nano requerida. Mantener una precisión de ~15-17 decimales reales es suficiente para los límites del Exchange de Binance.

```mermaid
graph TD
    subgraph Fill Event Data
        Fill[Asset PnL C++]
        Fee[Commission Dynamic Rate]
    end

    subgraph Portfolio Matrix (Axioma Audit)
        DF[Float64 JIT Check]
        Acc[Accounting Eq:<br/>Final == Init + PnL - Fees]
        Tr[Nano-Precision Tracker]
    end

    subgraph Risk Manager Sync
        Veto[Drift Veto Threshold:<br/>Tolerancia 0.001%]
    end

    Fill & Fee --> DF
    DF --> Acc
    Acc --> Tr
    Tr --> Veto
    
    classDef fill fill:#004d40,color:#fff;
    classDef check fill:#b71c1c,color:#fff;
    
    class Acc,Tr fill;
    class Veto check;
```

---

## 🚷 III. MANUAL DE 'SOVEREIGN-DEPLOY' (KILL-SWITCHES)

Los mecanismos de protección detienen el motor pasiva o agresivamente dependiendo de la clase del evento adverso, asfixiando automáticamente el capital expuesto y deteniendo las hemorragias.

### Nivel 1: "El Purgatorio" (Soft-Switch / Local Cooldown)

* **Activador:** 3 Asesinatos. (Tres Trades perdidos de forma consecutiva por el mismo Símbolo o Estrategia PPO).
* **Acción:** Suspensión the generación the señales del símbolo por tiempo definido (eg. 1h Cooldown).
* **Objetivo:** Apaciguar estrategias reaccionarias atrapadas en latigazos M1 de baja liquidez.

### Nivel 2: "Equity Critical" (Hard-Switch / Engine Panic)

* **Activador:** El saldo local ($13.00 USDT inicial base) retrocede del Drawdown Máximo Permanente estipulado (ejemplo: balance detectado < `$12.50 USDT`). O si Time-Drift contra Binance es severo (> 100ms sostenido).
* **Acción:**
    1. Envía señal The liquidación a valor de Mercado Vía REST inmediata.
    2. Modifica localmente un "Lock-File" de bloqueo total (`STOP_TRADING.LOCK`).
    3. Impone `sys.exit()` o rompe Event-Loop asincróno.
* **Objetivo:** Detener un desplome general the la cuenta o la estrategia por falta the consenso en la meta-data. Cero operaciones sin supervisión.

### Nivel 3: "Nuclear Override" (Manual Human Panic)

* **Activador:** El Big Red Button del `Dashboard` local Streamlit por intervención del Autor (Jhona).
* **Acción:** Invoca el `KillSwitch` explícitamente y aniquila todo hilo activo cancelando órdenes the LOB in-flight sin esperar "Ack" de CCXT.

---

## 🛑 IV. MATRIZ DE RIESGOS (BLACK SWANS)

Escenarios the Catástrofe (Cisnes Negros) históricamente presentes en Crypto y sus defensas ya implementadas en el **Metal-Core**:

| Evento Riesgoso | Impacto Potencial | Mitigación Implementada | Score Residual |
| :--- | :--- | :--- | :--- |
| **Velas 'Flash-Crash' / Anomalías de Aguja** | Desvío del SL, Ruina Inmediata | GARCH estipula TP/SLs conservadores lejanos al Spread. | **BAJO** |
| **Pérdida De Conexión WS Binance** | Órdenes Fantasma, Datos Estancados | Uvloop Keep-Alive. Re-conexion Automática y purga `buffer_reset`. | **MEDIO** |
| **Slippage Extremo In-Flight** | Llenado a peor precio que el LOB bid/ask | "Liquidity/Slippage Awareness Check" previo a emitir cualquier Orden Límite. | **BAJO** |
| **Underflow Matemático (Softwares ML)** | Modelos votando 100% Largo vs 100% Corto por NaN | "Tensor Axioma"; Clipping de la matriz L-Prob `[-100, 80]` pre-Softmax y validación JIT de números reales. | **NULO** |
| **Eventos de Fallo the Máquina / OS** | Corrupción del Historial, Des-sincronía PnL | Base The datos SQLite forzada a WAL mode + Local Parquet Data Caching persistente. | **MUY BAJO** |

### 🟢 VI. AUDITORÍA DE FRICCIÓN (FEE DRAG) — MICRO-CUENTA $13

Para cuentas pequeñas, la comisión de Binance es el principal depredador. El sistema implementa:

1.  **Isolated Fee Tracker**: El `portfolio.py` calcula y resta la comisión estimada (0.05% Taker) antes de reportar el `net_pnl`.
2.  **Piso de Rentabilidad (Minimum Alpha)**: Se exige un movimiento mínimo del **0.15%** solo para cubrir los costos de ida y vuelta (*round-trip fees*). Si la estrategia no provee este Alpha neto, las señales son bloqueadas por el `RiskManager`.
3.  **Filtrado de Micro-Slippage**: El `FillEvent` captura la diferencia entre el precio solicitado y el precio de llenado. Si el slippage acumulado es > 0.05%, se activa un `Cooldown` operativo.

---

> Todo Cisne Negro conocido tiene una barrera técnica the protección en este sistema. El sistema opera the manera Agresiva bajo supervisión defensiva Extrema.

---

## 🛡️ V. GESTIÓN DE RIESGO HORIZON-AWARE (SCALABILITY)

Para soportar operaciones desde 1D hasta 30D, el sistema implementa una escala de riesgo dinámica basada en la volatilidad temporal.

### 1. Escalamiento de Stop Loss (ATR-Based)
En lugar de porcentajes fijos, el SL/TP se calcula como un múltiplo del ATR (Average True Range). Este múltiplo se escala por la raíz cuadrada del tiempo ($\sqrt{H}$) para compensar el aumento natural del ruido en horizontes largos.

*   **Formula**: $SL = Price - (ATR \times Multiplier \times \sqrt{Horizon\_Days})$

### 2. Drawdown Adaptativo en KillSwitch
El límite de pérdida máxima permitida (`MAX_DRAWDOWN`) ya no es estático del 1.5%. Para horizontes de Swing/Position (15D-30D), el KillSwitch permite fluctuaciones más amplias (hasta 4-5%) para evitar liquidaciones prematuras de posiciones con tesis estructurales de largo plazo, siempre manteniendo la integridad del capital base.

### 3. Pisos de Capital Dinámicos
El `CRITICAL_CAPITAL_FLOOR` se ajusta automáticamente según el valor del Tick y el Horizonte, garantizando que el bot siempre tenga margen suficiente para ejecutar órdenes LIMIT sin entrar en colisión con los mínimos del exchange de Binance bajo alta volatilidad.

### 4. Aislamiento y Control de Microscalping (Ultra HFT)
Para operaciones en el horizonte de `MICROSCALPING` (TP=0.25% y SL=0.20%), el sistema utiliza un control estricto de fricción. Las llaves del libro mayor virtual se aíslan bajo el formato `{symbol}_MICROSCALPING_{side}` para evitar colisiones de órdenes, y el Risk Manager las procesa de manera nativa mapeando las salidas y actualizando los precios utilizando el flujo real de datos, protegiendo la cuenta de $13 USD contra el Fee Drag mediante la preferencia de comisiones Maker.

---

## 🔒 VI. CONTROL DE PROPIEDAD EXCLUSIVA Y PREFIJOS DE SEGURIDAD (MODO PROFESOR)

Para evitar interferencias destructivas en el Ledger Virtual (por colisiones de órdenes entre los motores de Scalping y Swing), la arquitectura implementa una política estricta de propiedad de posición:

### 1. Propiedad Exclusiva de Posiciones (Strict Ownership Verification)
- **QUÉ**: Es una validación lógica que impide que una estrategia modifique, actualice los stops o cierre una posición que fue abierta por otra estrategia diferente.
- **POR QUÉ**: Previamente, cualquier orden de salida o señal `EXIT` emitida por un módulo (ej. `TechnicalStrategy` de Scalping) podía cerrar indiscriminadamente posiciones de otra estrategia (ej. `MLStrategy` de Swing) si compartían el mismo símbolo. Esto arruinaba la gestión de posición y provocaba pérdidas por cierre prematuro de posiciones ganadoras a largo plazo.
- **PARA QUÉ**: Aislar las tesis operativas por horizonte temporal. Un trade Swing debe ser gestionado exclusivamente por sus parámetros Swing, y un trade Scalping por sus propios stops rápidos.
- **CÓMO**: En `core/portfolio.py` (método `_update_virtual_ledger()`) y en `risk/risk_manager.py` (métodos `check_stops()` y `_generate_exit_order()`), cada orden de salida se somete a verificación. Si el `strategy_id` del emisor no coincide exactamente con el `opener_strategy_id` registrado en la posición, la orden de salida es vetada de inmediato, a menos que cumpla con las excepciones de seguridad del sistema.
- **CUÁNDO**: Se evalúa en cada iteración del Risk Manager al chequear stops, y en el Portfolio al procesar eventos de ejecución de llenado (Fills).
- **DÓNDE**: Implementado en el Ledger Virtual (`core/portfolio.py`) y en la capa de control de riesgo (`risk/risk_manager.py`).
- **QUIÉN**: El `Portfolio` (como custodio del Ledger) y el `RiskManager` (como árbitro de órdenes) co-ejecutan la validación.

### 2. Prefijos de Seguridad y Salidas del Sistema (Whitelisted Safety Exits)
- **QUÉ**: Son excepciones a la regla de propiedad exclusiva que permiten que ciertos sistemas de seguridad globales o módulos transversales del bot ejecuten cierres forzados sobre cualquier posición.
- **POR QUÉ**: Si la regla de propiedad fuera 100% rígida, un sistema de emergencia general (como el Kill-Switch, el trailing stop adaptativo de la cuenta o el cierre por tiempo extremo) no podría cerrar una posición abierta por otra estrategia en caso de pánico, resultando en pérdidas catastróficas.
- **PARA QUÉ**: Permitir la coexistencia de la propiedad exclusiva con salvaguardas sistémicas globales frente a cisnes negros o fallos de conexión.
- **CÓMO**: Se define una lista blanca de prefijos válidos de salida defensiva: `HARD_` (Stops duros / KillSwitch), `SPAP_` (Sophia Panic Exits), `TRAIL_` (Sistemas Trailing transversales), y `WEAK_` (Filtros de debilidad estructural). Si la orden o señal de salida empieza con uno de estos prefijos (ej. `"HARD_SL"` o `"SPAP_L_CAPTURE"`), la verificación de propiedad exclusiva se omite, permitiendo el cierre seguro de la posición.
- **CUÁNDO**: Al procesar y validar el emisor de la orden de salida en la capa de riesgo.
- **DÓNDE**: En `risk/risk_manager.py` (clase `RiskManager`) y `core/portfolio.py`.
- **QUIÉN**: El `RiskManager` valida el prefijo en la señal de salida y autoriza la emisión de la orden a Binance.


## ⚖️ VII. MÓDULO MAESTRO — INTELIGENCIA DIFERENCIAL DE ACTIVOS Y SISTEMAS DE APERTURA/CIERRE (MODO PROFESOR)

### 1. Inteligencia Diferencial de Activos (Asset-Specific Intelligence)
- **QUÉ**: Una capa de inteligencia transversal que clasifica y parametriza los activos del basket operativo según cuatro dimensiones fundamentales: Jerarquía de Liderazgo (Tiers 0-4), Perfil de Liquidez (Niveles 1-5), Perfil de Volatilidad (Perfiles A-D) y Mapa de Catalizadores (Tipos 1-6).
- **POR QUÉ**: Los mercados de criptomonedas no son homogéneos. BTC (Tier 0) se comporta de forma institucional y eficiente, mientras que DOGE (Tier 4) responde puramente a dinámicas de sentimiento y manipulación retail. Tratar a todos los activos con los mismos parámetros de stop loss, sizing y compatibilidad de estrategia destruye cuentas con capitales ajustados ($13 USD).
- **PARA QUÉ**: Asegurar que cada señal se evalúa bajo las reglas del activo correspondiente, reduciendo significativamente el drawdown, optimizando los stops dinámicos (multiplicadores ATR) y adaptando la fracción de Kelly (desde 1/2 en BTC hasta 1/4 en DOGE).
- **CÓMO**: Implementado mediante la clase `AssetIntelligence` en `core/asset_intelligence.py` que almacena los perfiles específicos de BTC, ETH, BNB, SOL, XRP y DOGE.
- **CUÁNDO**: Se activa en el flujo de entrada de señales (`verify_opening`) y en la monitorización de salidas (`verify_closing`).
- **DÓNDE**: `core/asset_intelligence.py`.
- **QUIÉN**: Diseñado por el Arquitecto Senior y el Quant Developer del equipo Trader Gemini.

### 2. Pipeline de Apertura Secuencial (A1-A7)
- **QUÉ**: Un sistema de 7 filtros ineludibles que valida y calibra la apertura de nuevas posiciones.
- **POR QUÉ**: Previene la entrada de operaciones en regímenes erróneos, timing desfavorable, con señales débiles, o bajo riesgo regulatorio y cortes de red.
- **PARA QUÉ**: Maximizar la probabilidad de acierto (Win Rate) a mercado real y proteger el capital inicial.
- **CÓMO**:
  1. **A1 (Régimen)**: Valida con el detector de régimen. TFTF requiere tendencia; Mean Reversion está prohibido en tendencias.
  2. **A2 (Timing)**: Bloquea aperturas durante cooldowns macro o períodos de baja liquidez.
  3. **A3 (Puntuación Primaria)**: Exige un umbral de confianza específico para cada activo (BTC >= 0.72, DOGE >= 0.78).
  4. **A4 (Confluencia y Compatibilidad)**: Filtra por la matriz de estrategias permitidas (ej. Mean Reversion prohibida en DOGE).
  5. **A5 (Riesgo y Sizing)**: Enforza un límite de 3 posiciones abiertas simultáneas y valida el tamaño mínimo de Binance ($5 USD).
  6. **A6 (No-Colisión y Catalizadores)**: Bloquea operaciones bajo cortes de red (outages en SOL) o riesgos regulatorios (XRP).
  7. **A7 (Calibración)**: Calcula los niveles de stop y liquidación virtual.
- **CUÁNDO**: En cada evaluación de señal en `MetaCoordinator` antes del motor de ejecución.
- **DÓNDE**: En `core/asset_intelligence.py` (`verify_opening`) and `core/meta_coordinator.py`.
- **QUIÉN**: Ejecutado por el `MetaCoordinator`.

### 3. Pipeline de Cierre Dinámico (C1-C7)
- **QUÉ**: Un sistema de 7 prioridades que monitoriza las salidas de las posiciones activas en tiempo real.
- **POR QUÉ**: El edge de una estrategia reside principalmente en su salida. Salir tarde destruye las micro-cuentas por mechas en contra.
- **PARA QUÉ**: Asegurar salidas limpias ante stops duros, invalidación de tesis técnica, trailing adaptativo, límites de tiempo o emergencias.
- **CÓMO**:
  1. **C1 (Stop Loss Inicial)**: Cierre inamovible basado en estructura.
  - **C2 (Invalidación de Contexto)**: Cierra proactivamente ante cambios adversos (ej. ADX de TFTF cae por debajo de 20).
  - **C3 (Cierre Parcial Progresivo)**: Materializa ganancias en objetivos de beneficio (R1, R2).
  - **C4 (Trailing Stop Adaptativo)**: Sigue la tendencia tras asegurar beneficios.
  - **C5 (Tiempo Límite)**: Expiración incondicional (1 hora para Scalping, 48 horas para Swing).
  - **C6 (Reversión de Señal)**: Cierra al invertirse la dirección del timeframe.
  - **C7 (Emergencia)**: Liquidación a mercado inmediata ante cisnes negros o outages detectados.
- **CUÁNDO**: Evaluado en cada tick del ledger virtual en `check_stops()`.
- **DÓNDE**: `core/asset_intelligence.py` (`verify_closing`) y `risk/risk_manager.py`.
- **QUIÉN**: Orquestado por el `RiskManager`.


## ⏳ VIII. AUDITORÍA TEMPORAL Y COMPORTAMIENTO EVOLUTIVO DEL SISTEMA (MODO PROFESOR)

Para blindar el capital de $13 USD y garantizar el crecimiento compuesto exponencial bajo el mandato del **PROMPT SUPREMO**, la arquitectura incorpora un motor supervisor que controla el comportamiento temporal, el envejecimiento de capital depositado y los niveles de degradación de la rentabilidad del bot.

### 1. Checklist de Inicialización de Fase 0 (System Init Validation)
- **QUÉ**: Protocolo de verificación estricto de 9 pasos previos a la operación que se ejecuta en los primeros 5 minutos del bot.
- **POR QUÉ**: Previene la activación de órdenes reales en situaciones de inestabilidad de red, falta de liquidez en buffers, desincronización de base de datos o fallos del event loop.
- **PARA QUÉ**: Garantizar que el sistema inicia en un estado 100% óptimo y seguro.
- **CÓMO**: El supervisor evalúa: latencia de WebSocket (<200ms), recepción activa de ticks para todos los pares configurados en `TRADING_PAIRS`, presencia de datos históricos en el Feature Store, carga exitosa de modelos ML, accesibilidad a bases de datos, consistencia del `OmniscientRegistry`, Portfolio Heat = 0, Kill Switches cargados (no activos) y disponibilidad de fondos reales de Binance.
- **CUÁNDO**: En el arranque inmediato del bot.
- **DÓNDE**: `core/temporal_supervisor.py` (`verify_initialization_checklist()`).
- **QUIÉN**: El `TemporalSupervisor`.

### 2. Control de Sub-fases del Primer Ciclo
- **QUÉ**: Restricciones de tamaño de posición (size) y puntuación de señal (score) basadas en el tiempo transcurrido desde el arranque del ciclo (72 horas).
- **POR QUÉ**: Evita la sobreexposición en los primeros momentos del arranque, permitiendo al bot "calentar" sus buffers y confirmar si la estrategia se encuentra en fase alineada ("On-Track").
- **PARA QUÉ**: Evitar pérdidas masivas inmediatas por falsos arranques de mercado.
- **CÓMO**:
  1. `OBSERVACION` (Minuto 0-30): Veto total (`allowed = False`) de toda señal.
  2. `HORA_1` (Minuto 30-60): Máximo 25% del sizing permitido, penalización de +15 puntos en la exigencia de score.
  3. `HORA_2_4` (Horas 1-4): Máximo 50% de sizing, +10 en score. A la hora 2 se evalúa el rendimiento acumulado: si el P&L es inferior a -1.0%, se fuerza el `conservative_mode` reduciendo a la mitad el tamaño del trade por el resto de la sesión.
  4. `HORA_4_8` (Horas 4-8): Máximo 70% de sizing.
  5. `OPERACION_NORMAL` (Horas 8+): 100% del sizing permitido.
- **CUÁNDO**: Interceptado dinámicamente antes de enviar señales al Risk Manager.
- **DÓNDE**: `core/temporal_supervisor.py` (`apply_temporal_constraints()`).
- **QUIÉN**: Co-ejecutado por el `TemporalSupervisor` y validado por el `RiskManager`.

### 3. Protocolo de Inyección de Capital (Gradual Capital Deployment)
- **QUÉ**: Detección automatizada de depósitos externos y su incorporación progresiva en el capital disponible para el trading durante un periodo de 4 semanas.
- **POR QUÉ**: Un incremento brusco del balance debido a un depósito manual altera drásticamente los cálculos de tamaño de posición por Kelly, pudiendo causar un sobredimensionamiento catastrófico antes de validar que los modelos se comportan correctamente bajo el nuevo volumen.
- **PARA QUÉ**: Atenuar el impacto de nuevas inyecciones de capital y evitar el aumento descontrolado del riesgo de ruina.
- **CÓMO**: El supervisor calcula la diferencia de efectivo contra el P&L de trading (`delta_cash - delta_pnl`). Si es mayor a $1.00 USD, detecta una inyección y registra el evento en el `OmniscientRegistry`. El capital inyectado se somete a un filtro de no-despliegue gradual:
  - Semana 1 (Días 0-7): 75% del depósito no es deployable (25% disponible).
  - Semana 2 (Días 8-14): 50% del depósito no es deployable (50% disponible).
  - Semana 3 (Días 15-21): 25% del depósito no es deployable (75% disponible).
  - Semana 4 (Días 22-28+): 0% no deployable (100% disponible).
- **CUÁNDO**: Evaluado en el loop de fondo cada minuto.
- **DÓNDE**: `core/temporal_supervisor.py` (`get_deployable_capital_reduction()`).
- **QUIÉN**: El `TemporalSupervisor` calcula la reducción y el `RiskManager` la descuenta en `size_position()`.

### 4. Niveles de Degradación Sistémica (Systemic Degradation Alerts)
- **QUÉ**: Evaluación estadística automatizada de la performance del sistema al finalizar cada ciclo de 72 horas para declarar niveles de degradación operativa.
- **POR QUÉ**: Identifica el deterioro de la ventaja estadística (edge) debido a cambios drásticos en el régimen del mercado, reduciendo el riesgo de forma proactiva antes de que la cuenta sufra un drawdown severo.
- **PARA QUÉ**: Garantizar la antifragilidad adaptativa y aplicar salvaguardas sin intervención humana.
- **CÓMO**: El supervisor evalúa Profit Factor, Win Rate, SHS y Drawdown de los ciclos completados:
  - **Nivel 1 (Alerta Amarilla)**: PF entre 1.2 y 1.5, o caída de Win Rate > 10% consecutiva, o SHS < 70. Acción: reduce el sizing de entrada al 70% y aumenta los filtros de score en +15 puntos.
  - **Nivel 2 (Alerta Naranja)**: PF < 1.2 consecutiva, Drawdown > 30%, o SHS < 60. Acción: reduce el sizing al 50% y aumenta la exigencia de score en +30 puntos.
  - **Nivel 3 (Alerta Roja)**: Pérdida neta en el ciclo, Drawdown > 50%, o SHS < 40. Acción: dispara de inmediato el Kill Switch general deteniendo la operación del bot.
- **CUÁNDO**: Al final de cada ciclo de 72 horas.
- **DÓNDE**: `core/temporal_supervisor.py` (`_execute_cycle_transition()`).
- **QUIÉN**: El `TemporalSupervisor` y el `RiskManager`.

---

## 🕵️ IX. MÓDULO AUDIT SENIOR (ACS, ACI, AEA, ACR, ATA)

El **MÓDULO AUDIT SENIOR** introduce un gobierno omnisciente del ciclo de vida de cada trade, garantizando que el motor de ejecución razone íntimamente sobre el contexto de la tesis que abrió la posición.

### 🧬 1. Estrategia ADN Registry
El sistema formaliza el mapa genético (`STRATEGY_DNA`) de 11 estrategias operativas (TFTF, OB_RETEST, LCA, MRBB, WYCKOFF, VBA, MBV, FRA, SC, STATARB, OCS). Cada una registra:
- Tesis y condiciones necesarias/suficientes.
- Indicadores críticos de confirmación e invalidación.
- Ventana temporal de validez técnica.
- Asimetrías de apalancamiento y mitigación por activo.

### 🚦 2. Roles de Auditoría Senior
1. **ACS (Auditor de Coherencia Estratégica)**: Valida la coherencia de régimen de mercado al momento de apertura (ej. TFTF exige tendencia; MRBB exige lateralización) e intercepta cierres por invalidación de indicadores clave (ej. caída de ADX < 20).
2. **ACI (Auditor de Continuidad de Inteligencia)**: Vigila constantemente que el bot no opere a "ciegas" debido a cortes de datos o desactualización de variables.
3. **AEA (Auditor de Especificidad Activo-Estrategia)**: Calibra el sizing y valida el alineamiento direccional con el líder del mercado (BTC) para los Tiers inferiores.
4. **ACR (Auditor de Captura de Rentabilidad)**: Optimiza los ratios de Take Profit y persigue el precio mediante stops de arrastre basados en el perfil del activo.
5. **ATA (Auditor de Trazabilidad y Aprendizaje)**: Escribe una bitácora detallada y persistente en `logs/audit_chronicle.json` registrando cada `ENTRY`, `HEARTBEAT` y `EXIT` para cerrar el bucle de aprendizaje automático (learning loops).

### 🚨 3. Protocolo de Degradación por Ceguera
El auditor de continuidad (ACI) mide el retraso (*lag*) del feed de datos en tiempo real. Si detecta anomalías de red o desconexiones, activa progresivamente:

| Nivel de Degradación | Condición de Activación | Acción y Penalización Aplicada |
| :--- | :--- | :--- |
| **Nivel 1: Alerta Parcial** | *Lag* > Límite del Horizonte (ej. > 45s en Scalping) | Generación de advertencia y registro de Warning. Se activa el modo defensivo. |
| **Nivel 2: Reducción Crítica** | *Lag* > 3x Límite o Predicción Expirada | Reducción inmediata a la mitad (Halving) del tamaño de la posición o congelamiento de márgenes. |
| **Nivel 3: Cierre de Emergencia** | *Lag* > 10x Límite | Salida inmediata a mercado (Emergency Exit) para salvar la cuenta de $13 USD de cisnes negros. |
