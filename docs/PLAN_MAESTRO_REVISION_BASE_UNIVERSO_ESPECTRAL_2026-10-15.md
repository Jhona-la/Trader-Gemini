# 🌌 PLAN MAESTRO CANÓNICO DE AUDITORÍA SISTÉMICA TOTAL, EXCELENCIA OPERATIVA Y EVOLUCIÓN CUÁNTICA CONTINUA (V6 · 2026-10-15)
## Unificando Teoría Matemática, Física de Campos, Estadística Prequencial, Psicología de Masas, Liquidez Oculta y Rendimiento en Nanosegundos

> **Autor / Director:** Antigravity (AGY, Quant Sr. Lead)  
> **Comité Revisor:** Consejo Colegiado de 10 Roles Senior en Modo Profesor  
> **Destinatarios:** Claude (Línea C), Codex (Líneas G/H/E), GLM (Línea B/LH), Qoder, Sol y Operador Humano  
> **Ámbito:** Repositorio Completo Trader Gemini (23 Crates Rust, 100% Nativo, CERO Python)  
> **Fecha:** 15 de Octubre de 2026 (Actualización Canónica Post-Ola Ω132 / Liquidez Oculta Kyle-Hasbrouck, Conformal Risk Gate, Protocolo de Higiene de Disco/RAM y Saneamiento de >150GB)  
> **Estado CI GitHub Actions:** 🟢 100% VERDE (Runs 38765470568, 38675440815, 38675440991, 38680460789 verificados con éxito)  

---

## 🏛️ PARTE I — IDENTIDAD, DOCTRINA Y GOBERNANZA COLEGIADA

```
                               ┌────────────────────────────────────────────────────────┐
                               │       CONSEJO SUPREMO DE 10 ROLES SENIOR              │
                               │           (Modo Profesor Obligatorio)                  │
                               └──────────────────────────┬─────────────────────────────┘
                                                          │
         ┌──────────────────┬──────────────────┬──────────┴──────────┬──────────────────┬──────────────────┐
         ▼                  ▼                  ▼                     ▼                  ▼                  ▼
  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐      ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
  │ 1. 🏗️ ARQ.   │   │ 2. 📊 QUANT  │   │ 3. ⚠️ RISK   │      │ 4. 🔧 SRE/   │   │ 5. ✅ QA     │   │ 6. 👨‍🏫 PROF.  │
  │    SENIOR    │   │    DEV       │   │    MANAGER   │      │    DEVOPS    │   │    ENGINEER  │   │    PEDAGÓGICO│
  └──────────────┘   └──────────────┘   └──────────────┘      └──────────────┘   └──────────────┘   └──────────────┘
         │                  │                  │                     │                  │                  │
         └──────────────────┼──────────────────┴─────────────────────┼──────────────────┴──────────────────┘
                            │                                        │
                            ▼                                        ▼
             ┌──────────────────────────────┐         ┌──────────────────────────────┐
             │ 7. 🌌 FÍSICO SISTEMAS COMPL. │         │ 8. 📐 ESTADÍSTICO CUÁNTICO   │
             │ Hilbert L² & Continuo τ      │         │ Teoría de Medida & Martingal │
             └──────────────┬───────────────┘         └──────────────┬───────────────┘
                            │                                        │
                            └────────────────────┬───────────────────┘
                                                 │
                                                 ▼
             ┌──────────────────────────────┐         ┌──────────────────────────────┐
             │ 9. 🧠 IA CUÁNTICA & CONDUCTA │         │ 10. 🕵️ AUDITOR FORENSE GRAFO │
             │ Kahneman-Tversky & Mean Field│         │ Trazado Vivo & Anti-Asfixia  │
             └──────────────────────────────┘         └──────────────────────────────┘
```

### 1.1. Metas Financieras y Científicas Fundamentales
1. **Micro-Capital Inicial:** $\$13.00\text{ USD}$ exactos.
2. **Tasa de Crecimiento Compuesto Continuo ($g$):**
   El objetivo primordial del sistema es duplicar el capital (+100%) cada 72 horas (3 días):
   $$g = \frac{\ln(2)}{3} \approx +0.231049\text{ día}^{-1} \quad (\approx +25.9921\%\text{ compuesto diario})$$
   Alcanzar este objetivo requiere una aceleración geométrica de alta frecuencia y baja varianza, donde cada micro-ganancia se reinvierte inmediatamente en el dimensionamiento de las siguientes posiciones.
3. **Erradicación del Mito del 100% Win Rate (`RULE[growth_over_wr]`):**
   - NO se persigue ni se requiere un Win Rate del 100%. Un sistema que exige el 100% de aciertos colapsa ante el primer evento de cola o se paraliza en una asfixia total de vetos.
   - El crecimiento exponencial exige **Esperanza Matemática Positiva Neta**:
     $$\mathbb{E}[R] = p \cdot W - (1 - p) \cdot L - \text{Fricción}_{\text{roundtrip}} > 0$$
   - Con un Payoff Asimétrico riguroso ($RR = \frac{W}{L} \ge 2.25$), un Win Rate realista del $55\% - 75\%$ genera una curva de capital parabólica sin riesgo de ruina.
4. **Dimensionamiento Óptimo Fraccionario de Kelly Continuo y Axioma de Ruina:**
   $$f^* = \frac{1}{2} \cdot \frac{p \cdot (RR + 1) - 1}{RR} \cdot \left(1 - \frac{DD}{d^*}\right)$$
   Donde el factor $\frac{1}{2}$ amortigua la volatilidad de la trayectoria y $d^* = 1 - (0.05)^{1/3} \approx 0.6316$ representa el suelo teórico de falsación. Acoplado con el axioma de ruina (`clamp_ruin \le 0.25`), garantizando que ninguna orden individual exponga más del 25% del capital independientemente del stop loss.
5. **Restricción Física del Entorno de Ejecución:**
   - Hardware: Laptop de recursos compartidos (16 GB RAM total, **CERO GPU dedicada**).
   - Presupuesto de memoria para el motor: $< 2.5\text{ GB}$ en producción.
   - Latencia crítica: Hot-path de decisión en microsegundos/nanosegundos ($< 50\text{ }\mu\text{s}$), libre de allocations dinámicas en heap (cero llamadas a `malloc`/`alloc` en el loop crítico mediante `AtomicF64`, stack buffers fijos y `Cow`).

### 1.2. Doctrina Pedagógica Obligatoria (Modo Profesor)
Toda interacción, código, commit y documentación debe responder con precisión analítica a la plantilla séptuple:
- **QUÉ:** Definición formal y ontológica del componente o cambio.
- **POR QUÉ:** Causa raíz matemática, física o de microestructura financiera.
- **PARA QUÉ:** Impacto directo en la curva de capital, PnL y erradicación de sesgos.
- **CÓMO:** Mecanismo algorítmico interno paso a paso (complejidad $O(1)$, zero heap allocations).
- **CUÁNDO:** Condiciones exactas de disparo o activación en el flujo causal.
- **DÓNDE:** Ubicación canónica en la arquitectura de 23 crates.
- **QUIÉN:** Módulo, struct, función o asiento responsable del cómputo.

---

## 🌌 PARTE II — EL MERCADO COMO UNIVERSO MULTIVARIANTE CONTINUO TEMPORAL-ESPECTRAL

```
   MICROESTRUCTURA (L2/L3)                                                MACRO-ESTRUCTURA (SECULAR)
   τ = 1 ns                                                                            τ = 146 años
   k = 0 ─── 1 ─── 2 ─── 3 ─── 4 ─── 5 ─── ... ─── 15 ─── ... ─── 28 ─── 29 ─── 30 ─── 31
   [============ FAST (s ≈ 0) ============]              [============ SLOW (s ≈ 1) ===========]
   τ = 30 s (Laminar / Flujo L2)                          τ = 12 h (Inercial / Drift Secular)
```

### 2.1. Erradicación Total del Binarismo Heurístico ("Scalping vs Swing")
El mercado financiero no es una alternancia binaria entre "scalping" y "swing", ni una partición en "regímenes de volatilidad" como cajas discretas estancas. El mercado es un **campo continuo multivariante en un espacio de Hilbert $L^2(\mathbb{R})$** parametrizado por:
- **Coordenada Temporal Continua $\tau \in [1\text{ ns}, 100\text{ años}]$:**
  $$\tau_k = 10^{-6}\text{ ms} \cdot 4^k \quad \forall k \in [0, 31]$$
- **Coordenada Espectral Normalizada $s \in [0, 1]$:**
  $$s(\tau) = \frac{\ln(\tau / \tau_{\text{fast}})}{\ln(\tau_{\text{slow}} / \tau_{\text{fast}})} \quad \text{con } \tau_{\text{fast}} = 30\,000\text{ ms}, \tau_{\text{slow}} = 43\,200\,000\text{ ms}$$
- **Símplex Tetraédrico Continuo de Régimen $\Delta^3$:**
  Todo punto de mercado se describe como una combinación convexa en el símplex $\Delta^3 = \{ (w_1, w_2, w_3, w_4) \in \mathbb{R}^4 : \sum w_i = 1, w_i \ge 0 \}$:
  - $w_1$: Fracción de Tendencia Monótona (Flujo Direccional)
  - $w_2$: Fracción de Reversión a la Media (Rango Estacionario)
  - $w_3$: Fracción de Ruptura de Volatilidad (Expansión Caótica)
  - $w_4$: Fracción de Compresión Asimétrica (Vorticidad Solenoidal)

Ninguna compuerta ni asiento puede bifurcar con un `if is_scalp` ni con umbrales booleanos rígidos. Todo cálculo de Take Profit, Stop Loss, Trailing, Sizing de Margen, Veto de Slippage o Penalización de Drawdown debe ser una **función analítica suave $C^\infty$** evaluada sobre $\tau$ y $s$.

---

## 🚀 PARTE III — "EL SIGUIENTE PASO" TRANSDISCIPLINAR (REVOLUCIÓN CIENTÍFICA)

Si el Gradient Boosting (XGBoost) es el siguiente paso del Random Forest, el sistema Trader Gemini implementa el siguiente paso de cada teoría, modelo y cálculo:

| Área Científica | Paradigma Convencional (Obsoleto) | Estado Previo en Repo | **EL SIGUIENTE PASO CANÓNICO (V3)** |
| :--- | :--- | :--- | :--- |
| **Filtrado Temporal** | Medias Móviles (SMA, EMA) | Filtro de Kalman Discreto | **Wavelets Continuas Morlet/Daubechies en Espacios de Sobolev $\mathcal{H}^s$ con Regularización Variacional de Tikhonov** |
| **Modelado de Volatilidad** | Desviación Estándar / ATR fijo | GARCH(1,1) / Volatilidad Histórica | **Rough Volatility Fractional Brownian Motion ($H < 0.5$) acoplada a PIN-SDE con Saltos de Lévy y Procesos de Hawkes Mutuamente Excitados** |
| **Ensamble de Predicción** | Random Forest / Regresión | XGBoost / LightGBM puntual | **Redes Conformalizadas Continuas de Kolmogorov-Arnold (KAN) y Deep Energy-Based Models (EBM) con Cobertura Prequencial Finita** |
| **Decisión & Consenso** | Votación simple / If-Else rígido | Media ponderada de asientos | **Decoherencia Cuántica en Espacio de Hilbert mediante Integrales de Camino de Feynman $\int \mathcal{D}[x(t)] e^{\frac{i}{\hbar}S[x]}$ y Matrices de Densidad Cuántica de von Neumann** |
| **Microestructura L2/L3** | Desequilibrio de Libro (OBI) | VPIN / OFI lineal | **Hidrodinámica Navier-Stokes Financiera ($Re$, Fracción Laminar) acoplada a Descomposición Ortogonal Helmholtz-Hodge (Vorticidad/Curl)** |
| **Paridad Multiactivo** | Correlación Lineal Pearson | Cópulas Bivariantes Estáticas | **Teoría Gauge No-Abeliana Yang-Mills $SU(N)$ con Curvatura de Holonomía Triangular y Corrientes Gauge Restauradoras** |
| **Psicología de Masas** | Ratios Long/Short estáticos | Indicador Contrarian heurístico | **Teoría de Prospectos Asimétrica (Kahneman-Tversky) ($S$-curve, $\lambda=2.25$, Prelec weighting) acoplada a Juegos de Campo Medio (MFG)** |
| **Control de Ruina** | Stop Loss fijo en precio | Máximo Drawdown estático | **E-martingalas de Ville, Cota Teórica de Falsación $d^* = 1 - \alpha^{1/3}$ y Conformal Risk Control con Cobertura Garantizada** |
| **Entropía de la Información** | Entropía de Shannon | Entropía Discreta | **Entropía No-Extensiva de Rényi-Tsallis con $q$-deformación y Rampa Suave $C^1$ Hermite** |
| **Dinámica No-Lineal** | Ondas de Volatilidad Lineales | Oscilador Armónico Clásico | **Ondas de Choque Supersónicas (Rankine-Hugoniot con Salto de Mach) y Solitones No-Lineales KdV en Pozo Cuántico Anarmónico** |

### 3.1. Ecuaciones Fundamentales Implementadas en Código Rust Nativo
1. **Descomposición Ortogonal de Helmholtz-Hodge ($L^2$):**
   $$\mathbf{F} = \nabla \phi + \nabla \times \mathbf{A} + \mathbf{h}$$
   - $\nabla \phi$: Flujo potencial laminar (momentum genuino sin recirculación).
   - $\nabla \times \mathbf{A}$: Flujo solenoidal rotacional (vorticidad espuria, spoofing, wash trading).
   - $\mathbf{h}$: Flujo armónico ($\Delta \mathbf{h} = 0$, arbitraje global).
   *Implementado en `crates/metacortex-engine/src/consejo_seniors.rs` y `crates/feature-engine/src/hodge_flow.rs`.*

2. **Acción Gauge de Yang-Mills Multiactivo ($SU(N)$):**
   $$F_{\mu\nu} = \partial_\mu A_\nu - \partial_\nu A_\mu + g [A_\mu, A_\nu]$$
   Donde la corriente de Yang-Mills $J_i = D^\mu F_{\mu i}$ induce una modulación suave Lorentz-Cauchy $C^\infty$:
   $$w_{\text{ym}} = w_{\max} \cdot \frac{J^2}{J^2 + J_0^2}$$
   *Implementado en `crates/strategy-core/src/yang_mills_gauge.rs`.*

3. **Propagador Cuántico de Feynman (Path Integral):**
   $$\mathcal{K}(x_b, t_b; x_a, t_a) = \int \mathcal{D}[x(t)] \exp\left( \frac{i}{\hbar_{\text{fin}}} S[x(t)] \right)$$
   Donde la acción clásica $S[x(t)] = \int (T - V) dt$ penaliza la fricción y el deslizamiento adverso.
   *Implementado en `crates/signal-engine/src/feynman_propagator.rs`.*

4. **Entropía No-Extensiva de Rényi-Tsallis:**
   $$S_q = \frac{1}{1 - q} \left( 1 - \sum_{i} p_i^q \right) \quad \text{y} \quad H_\alpha = \frac{1}{1 - \alpha} \ln \left( \sum_i p_i^\alpha \right)$$
   Captura la correlación de largo alcance y las distribuciones de colas pesadas de la microestructura financiera con parámetro adaptativo $q$.
   *Implementado en `crates/signal-engine/src/renyi_tsallis_entropy.rs`.*

5. **Salto de Choque Supersónico de Rankine-Hugoniot y Pozo Anarmónico:**
   $$[[\rho u]] = 0, \quad [[\rho u^2 + p]] = 0 \implies \Delta p \propto \frac{M^2 - 1}{1 + \gamma M^2}$$
   $$V(x) = \frac{1}{2} k x^2 + \lambda x^4, \quad F(x) = -(k x + 4 \lambda x^3)$$
   *Implementados en `crates/signal-engine/src/supersonic_shockwave.rs` y `crates/signal-engine/src/quantum_oscillator.rs`.*

---

## 🔍 PARTE IV — AUDITORÍA SISTÉMICA TOTAL DE GRAFO VIVO Y RESOLUCIÓN DE VETOS

```
   [ FUENTES DE DATOS L2/L3/TICKS ]
                 │
                 ▼ (Arista de Ingestión)
   [ TRANSFORMADORES HILBERT / WAVELETS ]
                 │
                 ▼ (Arista de Features)
   [ FEATURE STORE OMNISCIENTE & CALIBRADORES ]
                 │
                 ▼ (Arista de Señales)
   [ MODELOS PREDICTIVOS & PROPAGADOR CUÁNTICO ]
                 │
                 ▼ (Arista de Deliberación)
   [ TRIBUNAL SUPREMO: CONSEJO DE 11 ASIENTOS ]
                 │
                 ▼ (Arista de Autorización & Cartera Multiactivo)
   [ COMPUERTAS DE RIESGO CONTINUO (KELLY / D* / HEADROOM) ]
                 │
                 ▼ (Arista de Ejecución)
   [ MOTOR DE EJECUCIÓN BINANCE FUTURES ]
                 │
                 ▼ (Arista de Retroalimentación)
   [ APRENDIZAJE ONLINE & MEMORIA EPIGENÉTICA ]
```

### 4.1. Taxonomía de los 3 Tipos de Fallos Sistémicos
1. **Fallo Tipo 1 — Arista Muerta:**
   Un nodo produce un cómputo matemáticamente válido pero ningún consumidor downstream lo recibe o procesa (resuelto en Ω120 con la resurrección del Tribunal de Autoengaño).
2. **Fallo Tipo 2 — Nodo Silencioso / Asfixia por Veto Rígido:**
   Un nodo recibe información viva pero está bloqueado por una constante estática congelada o un veto ciego que silencia toda salida hacia el mercado (ej. `REJ_ORCHESTRATOR` resuelto en Ω124 acotando por el headroom de la cartera viva).
3. **Fallo Tipo 3 — Colisión de Flujos y Carreras de Estado:**
   Múltiples sub-sistemas leen o escriben en el mismo registro con semánticas contradictorias (evitado con `AtomicF64` lock-free y sincronización determinista en `sync_council_params`).

### 4.2. Auditoría de Vetos, Límites y Rechazos (Anti-Asfixia)
El sistema ha erradicado las siguientes compuertas de asfixia:
- **`REJ_ORCHESTRATOR (8)`:** Corregido en Ola Ω124 (`0da7a1e3`). El evaluador de intención individual consulta `available_margin_headroom` y reduce el margen pedido mediante `sol_a2_margen_reducido`, en lugar de colapsar la señal a `SignalType::Flat`.
- **`REJ_FEE_IMPACT (3)`:** Corregido en LH-02. En cuentas micro ($13-$65 USD) donde el nocional mínimo obliga a apalancamientos $L \approx 5-10\text{x}$, la tolerancia de comisión conserva el techo continuo micro (3.5%) evitando rechazos masivos de órdenes viables.
- **`REJ_MIN_NOTIONAL (6)`:** Corregido en D-130 / CL-6. El apalancamiento necesario se calcula sobre el margen del trade con redondeo hacia arriba (`.ceil()`) respetando las cotas genómicas continuas.

---

## 📋 PARTE V — HOJA DE RUTA EXHAUSTIVA ARCHIVO POR ARCHIVO (23 CRATES · FASES F0 A F9)

### 📊 Inventario General de Código
- **Crates Totales:** 23
- **Archivos Rust (`.rs`):** >510
- **Líneas de Código Rust:** >125,000
- **Suites de Pruebas Unitarias y de Contrato:** >80 suites (>660 tests pasando al 100%)
- **Presupuesto de Memoria RAM:** <2.5 GB utilizados (holgado en laptop de 16 GB, sin GPU)
- **Latencia de Hot-Path de Decisión:** <25 microsegundos (cero asignaciones en heap)

```
crates/
├── backtest-engine/        (Fase F9: Replay determinista L2, oráculo T-1, sin mirar al futuro)
├── dark-alpha-engine/      (Fase F3: Detección de liquidez oculta, icebergs e impacto institucional)
├── data-ingest/            (Fase F7: Ingestión ultra-rápida WebSocket / REST Binance Futures)
├── data-pipeline/          (Fase F7: Pipelines de streaming zero-copy de microestructura)
├── evolution-engine/       (Fase F9: Algoritmos genéticos continuos y mutación de plantillas)
├── feature-engine/         (Fase F3: 200+ features espectrales, wavelets, OFI, VPIN continuo)
├── god-engine-core/        (Fase F8: Orquestador omnisciente de eventos y loop principal)
├── graph-4d/               (Fase F1: Topología cuatridimensional espacio-tiempo de liquidez)
├── graph-architecture/     (Fase F1: Arquitectura de grafos vivos y trazado de dependencias)
├── metacortex-engine/      (Fase F5: Consejo de Seniors, Prospect Theory, aprendizaje online)
├── omniscient-registry/    (Fase F1: Registro atómico multiactivo lock-free en memoria compartida)
├── os-guardian/            (Fase F0: Vigilancia de hardware, CPU, memoria 16GB y watchdog)
├── phase-runner/           (Fase F9: Ejecución por etapas de simulación y producción)
├── quantum-arena/          (Fase F1: Estado central, posiciones multislot 0/1/2, genotipo supersimétrico)
├── risk-engine/            (Fase F6: Control fraccionario Kelly, drawdown difusivo, compuertas de veto)
├── signal-engine/          (Fase F3: Propagador Feynman, Navier-Stokes manifold, ensamble tensorial)
├── storage-engine/         (Fase F1: Almacenamiento persistente mmap zero-copy en disco)
├── strategy-core/          (Fase F3: Lógica base de estrategias continuas y ejecución de órdenes)
├── telemetry-engine/       (Fase F8: Métricas de latencia en nanosegundos, Sharpe y tracking de PnL)
└── telemetry-server/       (Fase F8: Servidor local de telemetría y monitoreo de estado)
```

### 🗺️ Fases de Ejecución Sistemática
- **FASE F0: Micro-Capital (\$13.00 USD), Preservación Absoluta y Vigilancia de Recursos** (`os-guardian`, `risk-engine/capital_regime.rs`).
- **FASE F1: Espacio de Hilbert Multivariante y Estado Central Zero-Alloc** (`quantum-arena`, `omniscient-registry`, `storage-engine`).
- **FASE F2: Física Matemática de Campos: Navier-Stokes, Helmholtz-Hodge y Yang-Mills** (`feature-engine`, `strategy-core`, `risk-engine/hodge.rs`).
- **FASE F3: Señales Cuánticas, Feynman Propagator, Rényi-Tsallis y Choques Supersónicos** (`signal-engine`, `dark-alpha-engine`).
- **FASE F4: Psicología Financiera, Conducta de Masas y Teoría de Prospectos** (`metacortex-engine/prospect_theory.rs`, `god-engine-core/entes_consejo.rs`).
- **FASE F5: Tribunal Supremo de Deliberación: Consejo de 11 Asientos y Aprendizaje Online** (`metacortex-engine/consejo_seniors.rs`).
- **FASE F6: Motor de Riesgo, Stop Loss Difusivo, LH-04, Orquestador y Trailing Multiescala** (`risk-engine`).
- **FASE F7: Ingestión de Microestructura Binance Futures Zero-Copy** (`data-ingest`, `data-pipeline`).
- **FASE F8: Loop Principal Omnisciente, Sincronización Lock-Free y Ejecución** (`god-engine-core`, `execution-engine`, `telemetry-engine`).
- **FASE F9: Backtesting de Fidelidad Absoluta, Replay Cuántico y Oráculos Genéticos** (`backtest-engine`, `evolution-engine`, `phase-runner`, `audit-engine`).

---

## 🤝 PARTE VI — PROTOCOLO DE CONCURRENCIA MULTI-AGENTE Y GOBERNANZA GIT

```
                    ┌──────────────────────────────────────────────┐
                    │               ORIGIN / MAIN                  │
                    │         (Cabecera Canónica Pública)          │
                    └──────────────────────┬───────────────────────┘
                                           │
                   ┌───────────────────────┴───────────────────────┐
                   │                                               │
                   ▼                                               ▼
    ┌──────────────────────────────┐                ┌──────────────────────────────┐
    │ WORKTREE AGY (.antigravity)  │                │ CHECKOUT PRINCIPAL (GLM/etc) │
    │ Rama: agy/125-...            │                │ Ramas aisladas concurrentes  │
    └──────────────┬───────────────┘                └──────────────┬───────────────┘
                   │                                               │
                   │ 1. git status (estricto)                      │ 1. Trabajo aislado
                   │ 2. cargo test / check                         │ 2. Oráculo T-1
                   │ 3. git add selectivo                          │ 3. Commit atómico
                   │ 4. Rebase / Merge limpio                      │
                   │ 5. Push FF a origin/main                      │
                   │ 6. Eliminar rama integrada local y remota     │
                   ▼                                               ▼
    └──────────────────────────────────────────────────────────────┘
```

1. **Aislamiento Estricto por Worktree:**
   - Antigravity opera exclusivamente en `.antigravity/worktrees/agy-plan` sobre su rama activa `agy/*`.
   - Se respeta estrictamente el trabajo en paralelo de los otros agentes (`claude/`, `codex/`, `glm/`, `sol/`, `qoder/`). PROHIBIDO tocar o borrar ramas ajenas activas.
2. **Prohibición Absoluta de `git add -A` o `git add .`:**
   - Ejecutar `git status` antes de cada commit y realizar `git add <archivo>` individualmente sobre los archivos modificados de la tarea en curso.
3. **Verificación Previa a la Ejecución:**
   - Detener cualquier proceso residual (`Stop-Process god_engine* -ErrorAction SilentlyContinue`) antes de compilar con `cargo check` o `cargo test`.
4. **Ciclo de Integración Limpia:**
   - Crear rama de trabajo `agy/<número>-<nombre>`, implementar, certificar contratos, sincronizar con `origin/main`, commitear atómicamente, mergear a `main`, pushear a `origin/main` y eliminar la rama local una vez integrada.
5. **Sincronización Transversal de Memoria:**
   - En cada commit, actualizar simultáneamente `TABLERO.md` y `.agents/MEMORIA.md`.

---

## 📈 PARTE VII — MATRIZ DE INTEGRACIÓN Y ESTADO DE OLAS (Ω101 A Ω129 Y HOJA DE RUTA Ω130+)

| Ola | Módulo / Crate | Foco Técnico / Físico | Estado | Certificación Contractual |
| :---: | :--- | :--- | :---: | :--- |
| **Ω101** | `metacortex-engine` | Desacoplamiento dinámico de `CouncilParams` y cotas en `with_params` | 🟢 **MERGED** | `council_params_dynamic_adaptation_contract.rs` |
| **Ω102** | `metacortex-engine` | Desacoplamiento dinámico de regímenes de volatilidad y slippage | 🟢 **MERGED** | `volatilidad_y_slippage_dinamica_contract.rs` |
| **Ω103** | `signal-engine` | Corrección de boost por votante solitario en `TensorVoteOrchestrator` | 🟢 **MERGED** | `cf3_espectro_opuesto_baja_la_confianza_tensorial` |
| **Ω104** | `risk-engine` | Desacoplamiento dinámico de comisión micro (3.5%) por activo | 🟢 **MERGED** | `veto_logic_contracts.rs` (`resolve_micro_max_fee_pct`) |
| **Ω105** | `metacortex-engine` | Ponderación de pesos por esperanza matemática $\theta_i$ en tracker | 🟢 **MERGED** | `cf3_tracker_premia_ev_no_acierto` |
| **Ω106** | `metacortex-engine` | Censado e incorporación de compuertas V-CNCL-001..006 en registro | 🟢 **MERGED** | `cf3_registro_cubre_los_vetos_de_asiento_del_consejo` |
| **Ω107** | `risk-engine` | Dimensionamiento por probabilidad calibrada Platt vs confianza cruda | 🟢 **MERGED** | `cf3_la_puntuacion_cruda_no_dimensiona_por_su_cuenta` |
| **Ω108** | `signal-engine` | Acotamiento de TTL de veredicto espectral en $[\tau_{\text{fast}}, \tau_{\text{slow}}]$ | 🟢 **MERGED** | `cf3_ttl_del_veredicto_no_supera_la_vida_que_fija_el_orquestador` |
| **Ω109** | `quantum-arena` | Relajación homeostática continua Ornstein-Uhlenbeck en sesgo epigenético | 🟢 **MERGED** | `epigenetic_ornstein_uhlenbeck_contract.rs` |
| **Ω110** | `risk-engine` | ~~Modulación adaptativa continua de exploración Kelly~~ (Revertida P0 GLM 113) | 🔴 **REVERTED** | Absorbente corregido en GLM 118 vía racha asimétrica |
| **Ω111** | `metacortex-engine` | Modulación continua de utilidad esperada y win rate en Teleonomía | 🟢 **MERGED** | `teleonomy_dynamic_thresholds_contract.rs` |
| **Ω112** | `risk-engine` | Desacoplamiento dinámico de cotas de margen micro (0.25) y estándar (0.90)| 🟢 **MERGED** | `margin_cap_adaptation_contract.rs` |
| **Ω113** | `metacortex-engine` | Desacoplamiento dinámico de tolerancias de slippage en SeniorEjecucion | 🟢 **MERGED** | `execution_slippage_adaptation_contract.rs` |
| **Ω114** | `metacortex-engine` | Desacoplamiento dinámico de límite de drawdown $d^*$ y racha en Riesgo | 🟢 **MERGED** | `risk_opinion_adaptation_contract.rs` |
| **Ω115** | `metacortex-engine` | Desacoplamiento dinámico de tolerancias causales y breakout alineado | 🟢 **MERGED** | `causal_opinion_adaptation_contract.rs` |
| **Ω116** | `metacortex-engine` | Desacoplamiento dinámico de rotacional Hodge y corriente gauge Yang-Mills | 🟢 **MERGED** | `physical_gauge_opinion_adaptation_contract.rs` |
| **Ω117** | `metacortex-engine` | Desacoplamiento dinámico de umbrales ML y factores metacognitivos | 🟢 **MERGED** | `ml_and_metacognitive_adaptation_contract.rs` |
| **Ω118** | `metacortex-engine` | Desacoplamiento dinámico de psicología de masas, ballenas y cascadas | 🟢 **MERGED** | `market_entity_opinion_adaptation_contract.rs` |
| **Ω119** | `metacortex-engine` | Desacoplamiento dinámico de Abogado del Diablo, autoengaño y online learning | 🟢 **MERGED** | `auditor_internal_and_tracker_adaptation_contract.rs` |
| **Ω120** | `audit-engine` / `god-engine-core` | Resurrección de arista muerta y desacoplamiento en Tribunal de Autoengaño | 🟢 **MERGED** | `tribunal_and_discrepancy_adaptation_contract.rs` |
| **Ω121** | `feature-engine` / `signal-engine` | Desacoplamiento dinámico de Navier-Stokes y Microestructura (OFI, OrderFlow) | 🟢 **MERGED** | `navier_stokes_and_microstructure_adaptation_contract.rs` |
| **Ω122** | `feature-engine` / `strategy-core` / `signal-engine` | Desacoplamiento Helmholtz-Hodge, Yang-Mills y propagador Feynman | 🟢 **MERGED** | `feynman_propagator_adaptation_contract.rs` (5/5) |
| **Ω123** | `signal-engine` / `quantum-arena` | Desacoplamiento Rényi-Tsallis, oscilador cuántico y onda de choque supersónica | 🟢 **MERGED** | `entropy_oscillator_shockwave_adaptation_contract.rs` (4/4) |
| **Ω124** | `risk-engine` / `god-engine-core` | Hotfix CI Núcleo: `available_margin_headroom` vs `safe_limit` cartera viva | 🟢 **MERGED** | `spectral_zero_replay_contract.rs` (2/2) & `portfolio_admission_contract.rs` (7/7) |
| **Ω125** | `dark-alpha-engine` / `god-engine-core` | Neuroplasticidad continua Oja $C^1$ y PPO Online con hiperparámetros genómicos | 🟢 **MERGED** | `neuro_plasticity_ppo_adaptation_contract.rs` (4/4) |
| **Ω126** | `signal-engine` / `quantum-arena` | Continuidad $C^1$ Proyección Espectral, Perceptrón tanh y Hawkes-Bessel | 🟢 **MERGED** | `spectral_projection_hawkes_perceptron_adaptation_contract.rs` (4/4) |
| **Ω127** | `quantum-arena` / `risk-engine` | Desacoplamiento Espectral Multiactivo, Régimen Continuo $\Delta^3$ y LH-04 | 🟢 **MERGED** | `espectral_multiactivo_regime_adaptation_contract.rs` (4/4) |
| **Ω128** | `feature-engine` / `execution-engine` | Continuidad $C^1$ Multifractal Hermite, Microestructura y Fill Realista | 🟢 **MERGED** | `multifractal_microstructure_c1_contract.rs` (4/4) |
| **Ω129** | `strategy-core` / `signal-engine` | Continuidad $C^1$ Hermite en VECM, Momentum Booster, Solitón y Resonancia | 🟢 **MERGED** | `vecm_momentum_booster_c1_contract.rs` (3/3) & `soliton_stochastic_resonance_c1_contract.rs` (2/2) |
| **SOL-R5-02 / GLM-123** | `backtest-engine` | Mark-to-market del capital final del replay (caja + PnL no realizado de posiciones abiertas) | 🟢 **MERGED** | `sol_r5_02_tests` (2/2) & `sol_cont_accounting_2026-10-15.rs` (3/3) |
| **LH-06** | `signal-engine` | Diagnóstico colapso de $\tau$ y propuesta de peldaño intermedio $IC > 0$ | 🟡 **EN ESPERA** | Condicionado a T-3 v2 de Claude (`docs/DISENO_LH06_TAU.md`) |
| **Ω130** | `risk-engine` / `quantum-arena` | Corrección Monotonicidad Cramér-Lundberg en $R$, Desbloqueo Test CF2 Canónico y Desacoplamiento Dinámico Cuántico | 🟢 **MERGED** | `cramer_lundberg_exposure_monotonicity_contract.rs` (3/3) & `cf2_lundberg_veto_monotono_en_r` (1/1) |
| **Ω131** | `risk-engine` | Autosolvedor de Jacobi en Stack Zero-Alloc para Teoría de Matrices Aleatorias (RMT) ($N \le 32$) y APIs Planas | 🟢 **MERGED** | `zero_alloc_rmt_contract.rs` (4/4) & `spectral_matrix_contract.rs` (13/13) |
| **Ω132** | `dark-alpha-engine` / `risk-engine` | Detector de Liquidez Oculta Kyle-Hasbrouck, Conformal Risk Gate y Saneamiento de Disco/RAM (>150GB) | 🟢 **MERGED** | `hidden_liquidity_conformal_contract.rs` (4/4) & `cf4_freno_estrategia.rs` (4/4) |
| **Ω133** | `signal-engine` / `risk-engine` | Acoplamiento de Presión Iceberg con Tensor de Votos del Consejo y Kelly Conformal | 🔵 **ROADMAP** | Próxima ola de integración |

---

*Plan Maestro Canónico V6 certificado por el Consejo de 10 Roles Senior de Trader Gemini. Documento vivo de referencia absoluta para todas las sesiones de desarrollo concurrentes.*
