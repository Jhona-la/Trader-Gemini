# 🌌 PLAN MAESTRO INTEGRAL DE REVISIÓN DESDE LA BASE · UNIVERSO MULTIVARIANTE CONTINUO TEMPORAL ESPECTRAL
## Gobernanza Científica, Escalera Evolutiva de Teorías, Auditoría de Vetos y Sincronización Sistémica Multi-Agente
### Antigravity (Quant Sr. Lead & Consejo Colegiado de 10 Roles Senior) · Ola Ω207 · 22 de Octubre de 2026
**Corte Operativo:** `main` consolidado (`bec441023` + Ola Ω207) | **Workspace:** 23 crates nativos en Rust, 644 archivos `.rs` compilables, 292 suites contractuales | **Hardware Base:** Laptop 16 GB RAM sin GPU dedicada (`-j 1` mandatorio).

---

## 🏛️ PARTE I — IDENTIDAD, CONSEJO COLEGIADO Y MODO PROFESOR SUPREMO

Este documento establece la refundación metodológica y operativa del sistema autónomo **Trader Gemini** tras la culminación total de la Fase 2 (125/125 hallazgos cerrados en `f1_hallazgos.tsv`). Actuando bajo la dirección del **Quant Sr. Lead**, el sistema es gobernado por un **Consejo Colegiado de 10 Roles Senior**, donde cada concepto, algoritmo, veto de riesgo y decisión arquitectónica debe ser justificado mediante el método pedagógico completo:

$$\text{MÉTODO PROFESOR} \equiv \{\text{QUÉ}, \text{POR QUÉ}, \text{PARA QUÉ}, \text{CÓMO}, \text{CUÁNDO}, \text{DÓNDE}, \text{QUIÉN}\}$$

```
                                  ┌─────────────────────────────────────────────────────────────┐
                                  │             QUANT SR. LEAD & ARQUITECTO SUPREMO             │
                                  └──────────────────────────────┬──────────────────────────────┘
                                                                 │
                                ┌───────────────────┬────────────┴───────────┬───────────────────┐
                                ▼                   ▼                        ▼                   ▼
                           ┌─────────────┐   ┌─────────────┐          ┌─────────────┐     ┌─────────────┐
                           │ 1. 🏗️ ARCH   │   │ 2. 📊 QUANT │          │ 3. ⚠️ RISK  │     │ 4. 🔧 SRE   │
                           │ Arquitecto  │   │ Developer   │          │ Manager     │     │ DevOps      │
                           │ Nanosegundo │   │ Microestruc.│          │ Actuaria D3 │     │ Lock-Free   │
                           └──────┬──────┘   └──────┬──────┘          └──────┬──────┘     └──────┬──────┘
                                  │                 │                        │                   │
                                  ├─────────────────┴────────────┬───────────┴───────────────────┤
                                  ▼                              ▼                               ▼
                           ┌─────────────┐                ┌─────────────┐                 ┌─────────────┐
                           │ 5. ✅ QA    │                │ 6. 👨‍🏫 PROF  │                 │ 7. 🌌 FÍSICO│
                           │ Contratos   │                │ Pedagógico  │                 │ Variedad L² │
                           │ Falsación   │                │ Explicación │                 │ Navier-Stok.│
                           └──────┬──────┘                └──────┬──────┘                 └──────┬──────┘
                                  │                              │                               │
                                  ├──────────────────────────────┴───────────────────────────────┤
                                  ▼                                                              ▼
                     ┌──────────────────────────────┐              ┌──────────────────────────────┐
                     │ 8. 📐 ESTADÍSTICO MATEMÁTICO │              │ 9. 🧠 IA CUÁNTICA & PSICO    │
                     │ Malliavin & Martingalas Ville│              │ Kahneman-Tversky & Herding   │
                     └──────────────┬───────────────┘              └──────────────┬───────────────┘
                                    │                                             │
                                    └──────────────────────┬──────────────────────┘
                                                           ▼
                                           ┌──────────────────────────────┐
                                           │ 10. 🕵️ AUDITOR FORENSE GRAFO  │
                                           │ Trazado de Flujos & Antídoto │
                                           └──────────────────────────────┘
```

### Definición de los 10 Roles Senior Integrados

1. **🏗️ Arquitecto Senior (Sistemas de Ultra-Baja Latencia & Modularidad Zero-Cost)**:
   - *Misión*: Garantizar que el hot-path opere en escala de nanosegundos (< 250 ns por evaluación de orden).
   - *Reglas*: Cero asignaciones dinámicas en el bucle crítico (`zero-alloc`), alineación estricta a líneas de caché L1/L2 (64 bytes), estructuras `repr(C, align(64))`, y sustitución de mutex bloqueantes por atómicos lock-free o seqlocks.
2. **📊 Quant Developer Senior (Microestructura de Mercado & Ecuaciones Estocásticas)**:
   - *Misión*: Modelar la formación de precios no como series temporales discretas aisladas, sino como procesos estocásticos de salto-difusión con microestructura real (Lee-Ready, Stoikov, Hawkes multivariante).
   - *Reglas*: Prohibición absoluta de síntesis artificiales de datos (p. ej. VIX derivado del retorno local); estricta causalidad temporal $t \le T_{\text{actual}}$.
3. **⚠️ Risk Manager Senior (Actuaria de Ruina, Kelly Fraccionario & Circuit Breakers D3)**:
   - *Misión*: Proteger incondicionalmente el capital semilla de $13 USD evitando la ruina absorbente ($W_t \le 0$) mientras se maximiza la tasa de crecimiento compuesto logarítmico.
   - *Reglas*: Dimensionamiento vía LCB (Lower Confidence Bound) jerárquico bayesiano; aplicación del freno duro actuarial de Cramér-Lundberg D3 ($d^* \approx 0.6316$); no absorción de los mecanismos de rearme.
4. **🔧 SRE / DevOps Senior (Estabilidad Operativa, Conexiones Binance & Latencia Lock-Free)**:
   - *Misión*: Blindar la resiliencia de la infraestructura de trading contra desconexiones, paquetes perdidos, saturación de sockets y fallos del sistema operativo.
   - *Reglas*: Detección de latencia degradada mediante ventana de 3 strikes; desacoplamiento total de telemetría vía archivos mapeados en memoria (`mmap_bus`) con instrucciones SIMD non-temporal stores (`_mm_stream_si128`).
5. **✅ QA Engineer Senior (Certificación Contractual & Falsación de Hipótesis)**:
   - *Misión*: Validar que ningún cambio de código entre a producción sin una suite de pruebas contractuales formales en Rust que falsen rigurosamente el comportamiento esperado.
   - *Reglas*: Prohibición de tests tautológicos; validación de controles negativos y verificación de invariantes numéricos finitos.
6. **👨‍🏫 Profesor Pedagógico (Estructura de Explicación Integral)**:
   - *Misión*: Desglosar la totalidad de los componentes del sistema para que cada agente y operador humano comprenda el fundamento profundo de cada línea de código.
   - *Reglas*: Todo informe y análisis debe desglosarse en QUÉ, POR QUÉ, PARA QUÉ, CÓMO, CUÁNDO, DÓNDE y QUIÉN.
7. **🌌 Físico Teórico de Sistemas Complejos (Variedad de Hilbert, Navier-Stokes & Hamiltonianos Cuánticos)**:
   - *Misión*: Modelar el mercado como un sistema dinámico continuo disipativo en un espacio de Hilbert multivariante.
   - *Reglas*: Formulación del flujo de órdenes como dinámica de fluidos viscosos compresibles (Navier-Stokes); descripción de la micro-turbulencia y conservación de momento en cascadas de liquidación.
8. **📐 Estadístico Matemático Avanzado (Teoría Espectral, Cálculo de Malliavin & Martingalas)**:
   - *Misión*: Proporcionar el instrumental analítico riguroso para la toma de decisiones bajo incertidumbre radical y colas pesadas de Pareto-Lévy.
   - *Reglas*: Uso de martingalas no negativas de Ville para pruebas secuenciales libres de hipótesis distribucionales; cálculo de Malliavin para sensibilidad estocástica instantánea.
9. **🧠 Psicólogo Cuantitativo & IA Cuántica (Dinámica de Masas, Teoría de Prospectos & Redes Cuánticas)**:
   - *Misión*: Incorporar la asimetría conductual de los participantes humanos y algoritmos institucionales en las hipersuperficies de decisión del oráculo.
   - *Reglas*: Función de valor de Kahneman-Tversky con aversión a la pérdida ($\lambda \approx 2.25$); modelado de pánico colectivo y auto-excitación de ventas forzadas.
10. **🕵️ Auditor Forense de Grafos (Trazado de Flujos, Detección de Aristas Muertas & Colisiones)**:
    - *Misión*: Tratar al sistema completo como un grafo vivo y rastrear bidireccionalmente cada señal desde el websocket de Binance hasta la orden enviada y el estado interno.
    - *Reglas*: Clasificación estricta de fallos en Tipo 1 (arista muerta), Tipo 2 (nodo silencioso) y Tipo 3 (colisión de flujos).

---

## 🌌 PARTE II — EL PARADIGMA SUPREMO: UNIVERSO MULTIVARIANTE CONTINUO TEMPORAL ESPECTRAL

### 1. El Espacio Funcional de Hilbert $\mathcal{H} = L^2(\mathbb{R}^{d \times \tau}, \mu)$

El mercado de activos criptográficos no es un conjunto de barras discretas de 1 minuto, 5 minutos o 1 día. Pensar en términos de dicotomías simplistas como **"Scalping vs Swing"** o **"Régimen de Rango vs Régimen de Tendencia"** es un error conceptual que fragmenta la realidad del mercado.

El mercado es una **variedad continua multivariante temporal espectral**:
$$\mathcal{M} = \left\{ X(t) \in \mathbb{R}^d \mid t \in [0, \infty) \right\}$$
donde $X(t)$ representa el tensor de estados continuos (precios mid, profundidades L2/L3, micro-flujo de agresores, tasas de financiación, métricas macroeconómicas y volatilidades implícitas).

El espacio de observación admisible es el espacio de Hilbert de funciones cuadrado-integrables:
$$\mathcal{H} = L^2\left(\mathbb{R}^{d \times \tau}, \mu\right) = \left\{ f: \mathbb{R}^{d \times \tau} \to \mathbb{C} \ \middle|\ \int |f(x, \tau)|^2 d\mu(x, \tau) < \infty \right\}$$
dotado del producto interno hermítico:
$$\langle f, g \rangle_{\mathcal{H}} = \int_{\mathbb{R}^{d \times \tau}} f^*(x, \tau) g(x, \tau) \, d\mu(x, \tau)$$

### 2. Unificación Espectral Continua: Operadores de Proyección $\mathcal{P}_\omega$

Lo que comúnmente se denomina "Scalping" o "Swing" no son estrategias distintas, sino **proyecciones ortogonales sobre subespacios espectrales de diferentes longitudes de onda o frecuencias $\omega$**:
$$\mathcal{P}_{\Delta \omega} f = \int_{\omega_{\min}}^{\omega_{\max}} \left[ \mathcal{W}_\psi f \right](\omega, t) \, \psi_{\omega, t} \, d\omega$$
donde $\mathcal{W}_\psi f$ es la **Transformada Wavelet Continua Analítica (CWT)** con base de Cauchy-Morlet:
$$\left[ \mathcal{W}_\psi f \right](\omega, t) = \frac{1}{\sqrt{|\omega|}} \int_{-\infty}^\infty f(s) \, \psi^*\left(\frac{s - t}{\omega}\right) ds$$

- **Altas Frecuencias ($\omega \in [10^2 \text{ Hz}, 1 \text{ Hz}]$)**: Microestructura pura, desbalance de libro de órdenes (OBI), dinámicas de colas de agresores Lee-Ready, propagación de ondas de presión acústica en el spread.
- **Frecuencias Medias ($\omega \in [10^{-2} \text{ Hz}, 10^{-4} \text{ Hz}]$)**: Flujos intradiarios, correlaciones cruzadas multiactivo, absorción de liquidez institucional, equilibrio de carry/funding.
- **Bajas Frecuencias ($\omega \in [10^{-5} \text{ Hz}, 10^{-7} \text{ Hz}]$)**: Derivas macroeconómicas globales, ciclos de liquidez de bancos centrales, regímenes de volatilidad estocástica de largo plazo.

El sistema debe operar de forma **integral y simultánea** sobre todo el espectro, sin anularse ni pisarse, proyectando las decisiones de asignación de capital a lo largo de este continuo.

---

## ⚡ PARTE III — LA ESCALERA EVOLUTIVA: "¿CUÁL ES EL SIGUIENTE PASO?"

Para construir un sistema verdaderamente autoadaptativo y autoevolutivo, aplicamos el principio de la frontera científica: **Si XGBoost fue el siguiente paso natural tras el Random Forest, debemos identificar e integrar el siguiente paso cuántico y riguroso de cada teoría y cálculo del sistema**:

| Componente | Teoría Básica | Siguiente Paso Convencional | **El Siguiente Paso Cuántico / Espectral Superior** |
| :--- | :--- | :--- | :--- |
| **Horizontes Temporales** | Scalping vs Swing discreto | Agregación multi-timeframe (1m, 15m, 1h) | **Espacio de Hilbert Espectral Continuo $\mathcal{H}$** con proyecciones ortogonales $\mathcal{P}_\omega$ mediante Wavelets analíticas continuas. |
| **Modelos Predictivos** | Random Forest / Árboles CART | XGBoost / LightGBM | **Operadores de Koopman $\mathcal{K}^t$ & Ecuaciones Diferenciales Neuronales Informadas por la Física (PINDE)**: Transformación no lineal al espacio infinito de Hilbert donde la dinámica caótica se vuelve unitaria lineal. |
| **Dinámica del Libro (L2)** | Nivel 1/2 OBI simple | Micro-price de Stoikov / Hawkes bivariante | **Ecuaciones de Navier-Stokes Compresibles Estocásticas para la Densidad del Flujo de Órdenes**: Modelado de la liquidez como un fluido compresible con viscosidad $\nu$ (fricción de mercado) y ondas de choque acústicas en cascadas de liquidación. |
| **Correlación Multiactivo** | Pearson $\Sigma$ lineal | Matriz de Covarianza Reducida RMT (Marchenko-Pastur) | **Teoría Gauge no Abeliana de Yang-Mills $SU(N)$ & Formas Armónicas de Hodge-de Rham**: Curvatura del campo gauge $F_{\mu\nu}$ que calcula holonomías de arbitraje en bucles cerrados en $O(N)$ nanosegundos sin inversión matricial $O(N^3)$. |
| **Regímenes de Volatilidad** | Clasificación discreta (Bull/Bear) | Cadenas Ocultas de Markov (HMM) | **Ecuación Diferencial Parcial de Fokker-Planck-Kolmogorov**: Evolución continua en colector Riemanniano de la función de densidad de probabilidad $\frac{\partial \rho}{\partial t} = -\nabla \cdot (\mu \rho) + \frac{1}{2} \nabla^2 (D \rho)$. |
| **Medida de Información** | Entropía de Shannon / Log-Loss | Divergencia de Kullback-Leibler | **Entropías Generalizadas No Extensivas de Tsallis ($S_q$) y Rényi ($R_\alpha$) con Cálculo de Malliavin**: Sensibilidad estocástica en presencia de distribuciones de colas pesadas de Pareto-Lévy ($\alpha < 2$). |
| **Gestión de Ruina & Sizing** | Fracción fija de capital | Criterio de Kelly estático / Sortino | **Teoría Actuarial de Ruina de Cramér-Lundberg D3 con Control Estocástico de Bellman**: Sizing adaptativo $\kappa_t^* = \frac{\mu_t - r}{\sigma_t^2} \cdot \left(1 - \left(\frac{W_t}{W_{crit}}\right)^{-\gamma}\right)$ que garantiza cota de ruina $< 10^{-6}$ para $13 USD. |
| **Plasticidad del Sistema** | Constantes mágicas hardcodeadas | Optimización periódica batch (Optuna) | **Plasticidad Epigenética Continua Lock-Free**: Ingestión continua en nanosegundos desde el `OmniscientRegistry` a registros atómicos alineados a caché, donde cada hiperparámetro es un atractor dinámico auto-calibrado. |
| **Psicología Conductual** | Indicador Fear & Greed externo | Sentiment analysis por NLP diferido | **Micro-Pánico de Masas de Kahneman-Tversky con Ecuaciones de Campo Medio (MFG)**: Detección instantánea de cascadas de aversión a la pérdida ($\lambda = 2.25$) en los micro-ticks de agresores pasivos vs activos. |
| **Infraestructura de Datos** | SQLite / Logs en disco síncronos | Redis en memoria / Canales Tokio mpsc | **Bus de Telemetría Mmap Lock-Free con Instrucciones SIMD Non-Temporal Stores (`_mm_stream_si128`)**: Escritura directa en memoria física en $< 150 \text{ ns}$ sin contaminar la caché L1/L2 del procesador. |

---

## 🛡️ PARTE IV — AUDITORÍA FORENSE DE VETOS, BLOQUEOS Y CIRCUIT BREAKERS

Un sistema de trading cuantitativo de alto apalancamiento que inicia con un micro-capital de **$13 USD** se enfrenta a una fragilidad extrema: una pérdida acumulada de apenas unos pocos dólares destruye el poder de compra mínimo (Binance exige $5 USD de nocional por orden). Por tanto, los vetos y circuit breakers son esenciales, pero deben ser matemáticamente óptimos y **no absorbentes**.

### 1. Censo Riguroso de Códigos de Rechazo en `crates/risk-engine`

| Código | Identificador | Causa Matemática / Operativa | ¿Veto Justificado o Parizante? |
| :--- | :--- | :--- | :--- |
| `0` | `REJ_FLAT_COIN` | La señal orquestada tiene dirección neutra (`Flat`). | **Justificado**: No genera costo de transacción innecesario. |
| `1` | `REJ_EXPOSURE_ZERO` | El tamaño de posición calculado colapsa a 0. | **Justificado**: Filtra micro-ruido sin valor esperado positivo. |
| `2` | `REJ_CORRELATION` | Límite de exposición térmica o correlación multiactivo excedido. | **Justificado**: Evita concentración excesiva de riesgo en activos cointegrados. |
| `3` | `REJ_SPEC` | Violación de especificación de símbolo Binance (tick size, step size). | **Justificado**: Previene rechazo inmediato en el motor de Binance. |
| `4` | `REJ_EV` | Esperanza matemática de la operación $\mathbb{E}[\Delta W] \le 0$. | **Justificado**: Principio supremo; jamás operar con ventaja negativa. |
| `5` | `REJ_FEE_IMPACT` | La fricción de comisiones devora la ganancia esperada ($2 \times \text{fee} \ge \text{target}$). | **Justificado**: Protege el capital de morir por fricción microscópica. |
| `6` | `REJ_MIN_NOTIONAL` | Valor nocional de la orden $< \$5.00 \text{ USDT}$. | **Justificado**: Restricción dura de la API de Binance. |
| `7` | `REJ_MARGIN_INSUFFICIENT`| Saldo libre en la cuenta insuficiente para el margen de la posición. | **Justificado**: Previene rechazo de margen en el exchange. |
| `8` | `REJ_ORCHESTRATOR` | El orquestador de señales veta la entrada por consenso discordante. | **Auditado**: Debe requerir consenso espectral flexible, no bloqueo ciego. |
| `9` | `REJ_OTROS` | Condiciones de contorno no contempladas en categorías específicas. | **Deuda**: Debe descomponerse en causas unívocas. |
| `10` | `REJ_DRAWDOWN` | **FRENO DURO DE LA CUENTA**: Caída $\ge d^* \approx 0.6316$ o acumulación de caídas en ventana 200 stops $\ge K(w,l)$ o `kill_switch_active`. | **CRÍTICO / JUSTIFICADO**: Blindaje supremo contra ruina; ahora probado como no absorbente vía `reset_ventana_stops()` y rearme limpio. |
| `11` | `REJ_TP_SL_FLOOR` | Distancia de brackets más corta que el suelo de fricción roundtrip. | **Justificado**: Evita que una fluctuación aleatoria ejecute un TP en pérdida neta. |
| `12` | `REJ_CONFIDENCE` | Confianza bayesiana del oráculo inferior al umbral adaptativo. | **Auditado**: Modulado suavemente por temperatura en frío. |
| `13` | `REJ_VIABILIDAD` | Verificación de viabilidad geométrica entre TP, SL y ATR. | **Justificado**: Garantiza asimetría positiva en la relación ganancia/pérdida. |
| `14` | `REJ_SIN_EVIDENCIA` | Muestra de trades $N < N_{\min}$ sin prior bayesiano válido. | **Auditado**: No debe congelar el arranque en frío (resuelto vía sonda exploratoria D-750). |
| `15` | `REJ_INVALID_INPUT` | Datos de entrada corruptos, infinitos o NaN. | **Justificado**: Protección numérica de precisión finita IEEE-754. |
| `16` | `REJ_TARGET_GEOMETRY` | Precios objetivo TP/SL negativos o no finitos. | **Justificado**: Robustez matemática. |
| `17` | `REJ_SONDA_OCUPADA` | Ranura de sonda exploratoria en uso actualmente. | **Justificado**: Evita saturación de órdenes de prueba. |
| `18` | `REJ_ESTRATEGIA_FALSADA` | Estrategia declarada falsada por prueba secuencial SPRT de Wald. | **Justificado**: Apaga modelos que han perdido su ventaja empírica. |

### 2. Principio de No Absorción y Rearme Limpio

Un veto es patológico si se convierte en un **estado absorbente**: una vez activado, el sistema nunca puede salir de él de manera autónoma, congelando indefinidamente la capacidad de generar interés compuesto. 

En la Ola Ω207 se ha certificado formalmente que:
1. **Rearme de Ventana de Stops (F1-011)**: El operador o el protocolo de recuperación pueden invocar `arena.reset_ventana_stops()` consumiendo `rearme_frenos.txt` o ante reinicio con nuevo pico, restaurando los contadores acumulados a 0.0 sin destruir la memoria aprendida del oráculo.
2. **Defensa en Profundidad del Kill-Switch**: Cuando el sistema inmune activa `arena.kill_switch_active` (por latencia sostenida > umbral en 3 strikes, insolvencia, o cerrojo `STOP_TRADING.LOCK`), `RiskEngine::validate_order` veta inmediatamente toda nueva entrada con `REJ_DRAWDOWN`. Una vez desactivada la condición anómala, el latch se libera de forma lock-free y el flujo de órdenes se reanuda limpiamente.

---

## 🗺️ PARTE V — PLAN OPERATIVO DE RECORRIDO SISTÉMICO DE LOS 23 CRATES (FASE POR FASE)

Para erradicar la totalidad de los fallos de lógica, constantes arbitrarias y sesgos ocultos, se establece el siguiente recorrido sistemático a través de los 644 archivos `.rs` del workspace:

```
FASE 1: FUNDAMENTOS MATEMÁTICOS & ESTRUCTURAS DE DATOS (Crítica Zero-Alloc)
  ├── crates/math-engine/           (Transformadas espectrales, wavelets, operadores Koopman)
  ├── crates/quantum-arena/         (Estado global, configuración epigenética, posiciones, monedas)
  └── crates/omniscient-registry/   (Canal lock-free de propagación epigenética)

FASE 2: INGESTA CONTINUA & PIPELINES DE TIEMPO REAL (Ultra-Baja Latencia)
  ├── crates/data-ingest/           (Zero-copy binary reader, conectores de baja latencia)
  ├── crates/data-pipeline/         (Multiplexor omni 54D, lakehouse mmap, persistencia causal)
  └── crates/storage-engine/        (MmapTelemetryBus, seqlock de 64 bytes, AVX non-temporal stores)

FASE 3: GENERACIÓN DE FEATURES & TENSORES ESPECTRALES (Invariantes Causales)
  ├── crates/feature-engine/        (Microestructura, multifractal, entropía Tsallis/Rényi, Navier-Stokes)
  └── crates/quantum-tensor/        (Álgebra tensorial de Hilbert, proyecciones espectrales)

FASE 4: INTELIGENCIA ARTIFICIAL CUÁNTICA & APRENDIZAJE CONTINUO (Sin Fuga Temporal)
  ├── crates/evolution-engine/      (Online daemon, Shadow Forest causal, mutación y selección)
  └── crates/model-engine/          (Inferencia en nanosegundos, perceptrones de microestructura)

FASE 5: SEÑALES, CONSENSO DEL CONSEJO & TRIBUNAL (Arbitraje Espectral)
  ├── crates/signal-engine/         (Orquestador de voces, descomposición espectral de señales)
  └── crates/audit-engine/          (Auditor interno, validación de invariantes de estado)

FASE 6: MOTOR DE RIESGO & DIMENSIONAMIENTO ACTUARIAL (Supervivencia & Interés Compuesto)
  ├── crates/risk-engine/           (Freno duro D3, Kelly jerárquico bayesiano, 19 gates de veto)
  └── crates/portfolio-engine/      (Asignación de capital, balance multiactivo, curvas de crecimiento)

FASE 7: EJECUCIÓN, RECONCILIACIÓN & VIGILANTE INMUNE (Efectores de Mercado)
  ├── crates/execution-engine/      (Enrutamiento de órdenes Binance, cancelación atómica, brackets)
  ├── crates/god-engine-core/       (Event loop principal, unificación tick/dual, interlocks de latencia)
  └── src/bin/god_engine.rs         (Host de producción, vigilante inmune asíncrono, aplanado total)
```

---

## 🤝 PARTE VI — PROTOCOLO DE CONCURRENCIA Y COORDINACIÓN MULTI-AGENTE

Para garantizar armonía absoluta entre las sesiones de agentes que editan el repositorio en paralelo:

1. **Ramas Efímeras por Agente**: Toda sesión crea una rama aislada con su nombre (`antigravity/ola-omegaXXX`). Tras compilar, verificar contratos y hacer merge limpio a `main`, empuja a `origin/main` y elimina la rama efímera.
2. **Staging Quirúrgico Exclusivo**: `git add -A` está terminantemente prohibido. Se añaden únicamente los archivos explícitamente modificados por la tarea (`git add crates/... docs/...`).
3. **Respeto a los Procesos y Worktrees Vivos**: Antes de ejecutar compilaciones pesadas o alterar archivos, verificar en `AGENT_SYNC.md` y `Get-Process` que no se interfiera con pruebas de largo recorrido de otros agentes (como la prueba de ortogonalidad espectral de GLM 146).
4. **Higiene de Recursos en Portátil de 16 GB RAM**: Ejecutar invariablemente `cargo test` y `cargo check` con el flag `-j 1` para prevenir colapso de memoria swap o saturación del enlazador MSVC.
