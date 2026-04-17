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

