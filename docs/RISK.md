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

Cada orden y cálculo the PnL atraviesa una auditoría financiera de doble-chequeo basada en la clase `decimal.Decimal` (28 dígitos) que imposibilita de facto el *Floating-Point Drift* ("Ghost Money" / Dinero Fantasma de los Errores de Redondeo en CPU).

```mermaid
graph TD
    subgraph Fill Event Data
        Fill[Asset PnL C++]
        Fee[Commission Dynamic Rate]
    end

    subgraph Portfolio Matrix (Axioma Audit)
        DF[Decimal PnL Check]
        Acc[Accounting Eq:<br/>Final == Init + PnL - Fees]
        Tr[Precision Drift Tracker]
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
| **Underflow Matemático (Softwares ML)** | Modelos votando 100% Largo vs 100% Corto por NaN | "Tensor Axioma"; Clipping the la matriz L-Prob `[-100, 80]` pre-Softmax. | **NULO** |
| **Eventos de Fallo the Máquina / OS** | Corrupción del Historial, Des-sincronía PnL | Base The datos SQLite forzada a WAL mode + Local Parquet Data Caching persistente. | **MUY BAJO** |

> Todo Cisne Negro conocido tiene una barrera técnica the protección en este sistema. El sistema opera the manera Agresiva bajo supervisión defensiva Extrema.
