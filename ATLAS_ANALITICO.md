# ATLAS ANALÍTICO DEL MOTOR — Qué calcula, para qué existe, cómo se interpreta

> **Actualización 2026-09-24:** las secciones históricas se conservan. La
> [adenda de continuidad y causalidad](docs/AUDITORIA_CONTINUIDAD_ESPECTRAL_2026-09-24.md)
> corrige la interpretación de persistencia, pesos, entropía, Hurst y resolución
> temporal; también actualiza el estado de ramas. Las afirmaciones anteriores
> de certificación no sustituyen ese corte verificable.

> Compendio central de TODA la matemática del sistema. Cada componente se
> documenta con: **(F)** la fórmula que computa, **(P)** el propósito — qué
> decisión alimenta —, **(I)** cómo interpretar sus valores, y **(∩)** dónde
> vive. Este documento no sustituye los doc-comments de cada módulo: los
> indexa y les añade la capa de interpretación operativa.
>
> Convención de lectura: σ = volatilidad por unidad de tiempo; τ = horizonte
> temporal en milisegundos; f = fricción (fees+slippage) por pierna; bps =
> puntos básicos (1 bp = 0.01%).

---

## I. MICROESTRUCTURA Y FLUJO — «quién está empujando ahora»

### OBI — Order Book Imbalance
- **(F)** `OBI = (Qbid − Qask) / (Qbid + Qask)` sobre los niveles vivos del libro L2.
- **(P)** La lectura instantánea de presión: alimenta las ramas de señal de
  microestructura (`range_obi`, `effective_obi_*`) y al asiento Flujo del consejo.
- **(I)** ∈ [−1, 1]. |OBI| > 0.25 = desequilibrio real (umbral del asiento
  Causal en conjunción); OBI ≈ 0 = libro en equilibrio → las ramas que lo
  exigen quedan inactivas. **Sin libro** (stream de depth caído) vale 0 — por
  eso D-736 reemplazó las lecturas de `current_obi` por `effective_obi_*` en
  las ramas de reversión: la rama long leía el valor muerto y jamás disparaba.
- **(∩)** `parsers.rs` → `god-engine-core/src/lib.rs` (bloque analytics).

### OFI — Order Flow Imbalance
- **(F)** Normaliza el delta de cantidades del best bid/ask entre eventos:
  flujo agresor neto aproximado sin necesitar el tape completo.
- **(P)** Entrada al tensor 54D (slot 39) y a los triggers de explosividad.
- **(I)** ∈ [−1, 1]; > 0 compra agresora dominante. Un NaN histórico dejaba el
  OFI «clavado para siempre» (D-709) — hoy se sanitiza a 0.
- **(∩)** tensor omni 54D, `OmniSynth` (darwin), `run_backtest_native::omni_sim`.

### CVD — Cumulative Volume Delta
- **(F)** Suma acumulada de (volumen comprador − vendedor).
- **(P)** Confirmación direccional de ramas de señal; se actualiza DENTRO de
  `process_event` (D-708): una sola fuente para vivo, forense y replay —
  llamarla también desde el host lo contaba dos veces.
- **(I)** Su PENDIENTE importa más que su nivel: CVD subiendo con precio
  plano = acumulación silenciosa.
- **(∩)** `god-engine-core/src/lib.rs` (núcleo), slots 30/31 del tensor.

### Proceso de Hawkes — auto-excitación de trades
- **(F)** `λ(t) = μ + Σᵢ α·e^(−β(t−tᵢ))`; ratio de ramificación `η = α/β < 1`
  (estacionariedad). `intensity_ratio = λ/μ` mide cuántas veces más activo
  que la base está el flujo.
- **(P)** Detectar clusterización endógena (cascadas de actividad que se
  alimentan a sí mismas) — el régimen donde el momentum se auto-refuerza y
  los stops en cascada se contagian.
- **(I)** `intensity_ratio ≈ 1` = flujo Poisson (sin memoria). > 2 =
  auto-excitación fuerte; cerca de η → 1 el proceso es crítico (cascadas).
  **ESTADO ACTUAL (M2-C02 ABIERTO)**: la matemática existe en
  `hawkes_bessel.rs` pero `record_event` no tiene callers en producción — lo
  publicado al registry como `hawkes_intensity` es un proxy de
  aceleración/ATR. No interpretar ese campo como λ Hawkes hasta cerrar el
  hallazgo.
- **(∩)** `signal-engine/src/hawkes_bessel.rs` (matemática), wiring pendiente.

### Whale bursts — z-score de ráfaga de volumen
- **(F)** Z-score del notional del trade contra la historia del símbolo
  (tracker por coin); `is_burst` cuando z supera el umbral del tracker.
- **(P)** Asiento Ente del Mercado del consejo (P-5): quién opera AHORA.
- **(I)** z > 3 con dirección = ballena visible; el registro publica
  `whale_burst_z` ∈ [0, 10] al registry por símbolo.
- **(∩)** host `god_engine.rs` (whale_trackers) → registry scoped.

### Spoofing — evaporación de muros L2
- **(F)** Decay del tamaño de los muros de los 5 niveles del depth stream:
  un muro que aparece grande y se retira sin ejecución eleva el score.
- **(P)** Detectar liquidez falsa antes de apoyar una entrada en ella.
- **(I)** Score alto = el libro miente; el asiento Ente lo penaliza.
- **(∩)** `feature-engine` SpoofingDetector (P-6, niveles del parser depth5).

### Open Interest / Long-Short / Taker ratio
- **(F)** Ratios por símbolo de los endpoints públicos de Binance
  (`/futures/data/*`), publicados al registry.
- **(P)** Apalancamiento de la manada (OI), posicionamiento (L/S), agresión
  (taker). El asiento Ente y el contrarian de manada (CROWD_LS_FEAR = 3.0)
  los consumen.
- **(I)** L/S > 3.0 = unanimidad long apalancada (percentil alto retail) →
  el consejo aplica contrarian eufórico. **Limitación documentada (M1-M02)**:
  la normalización de OI por unidades del activo destruye la comparabilidad
  cross-símbolo (DOGE satura a 1.0) — leer por símbolo, no entre símbolos.
- **(∩)** `data-pipeline/src/omni_multiplexer.rs` (QO-U1c/U2).

---

## II. ESPECTRO TEMPORAL CONTINUO — «el mercado como banda, no como reloj»

### Las 32 escalas
- **(F)** τ geométrica en base 4 desde 1 µs hasta ~2.18 años
  (`τᵢ = 4ⁱ µs`, paso ln 4). Cada escala mantiene su propia EMA de precio,
  desviación `dev`, vol de sorpresa `ewma_dev_vol` y z-score `momentum_z`;
  `signal = tanh(z)` ∈ [−1, 1].
- **(P)** Reemplazo del binario scalp/swing: TODAS las escalas se actualizan
  en cada evento (observación continua, O(19)); el motor interpola en esta
  banda para cualquier horizonte.
- **(I)** `signal_at(τ)` = lectura puntual; señal > 0 = la escala ve impulso
  alcista. Un precio con señales alineadas en muchas escalas = tendencia de
  banda ancha; señales dispares = transición de régimen.
- **(∩)** `quantum-arena/src/temporal_spectrum.rs` (F8).

### Persistencia — el análogo discreto de |Hurst − 0.5|
- **(F)** EMA de acuerdo de signo entre desviaciones consecutivas de la
  escala: `persistence += α·(agree − persistence)` con agree ∈ {0,1}.
- **(P)** Medida de CONTENIDO INFORMATIVO por escala: 0.5 = puro ruido
  (cambio de signo aleatorio, H = 0.5), 1.0 = tendencia pura (H → 1),
  0.0 = anti-persistencia perfecta (H → 0, mean-reversion).
- **(I)** Es LA señal de régimen: persistencia alta en escalas lentas =
  mercado tendencial que merece dejar correr; baja = mercado que revierte y
  hay que asegurar rápido. La interpolación espectral S-2/S-3 la usa como
  parámetro t ∈ [0,1] en todos los lerp del motor.
- **(∩)** `temporal_spectrum.rs` (por escala).

### Fusión espectral — por qué pesa cada escala
- **(F)** `wᵢ = max(0.05, (persistenceᵢ − 0.5)·2)`;
  `fused_score = Σwᵢ·signalᵢ / Σwᵢ`, clamp [−1,1].
- **(P)** La lectura única del espectro que consumen la arbitración espectral
  (U-2), el asiento Espectral del consejo y Teleonomía.
- **(I)** **(M3-H01, corregido)** La fusión anterior ponderaba por
  1/dev_vol — como la vol de sorpresa de las escalas lentas es
  sistemáticamente menor, SIEMPRE dominaban. Ahora el peso es por
  información: una escala impredecible aporta peso suelo (5%, conserva la
  diversificación del promedio de ensamble), una escala persistente manda.
  `fused_score` alto = alineación direccional en las escalas que SABEN.
- **(∩)** `temporal_spectrum.rs::update` (M3-H01).

### τ dominante — la escala que manda HOY
- **(F)** La τ de la escala con mayor |contribución| (w·signal), acotada a la
  banda operativa [30 s, 12 h] (`TAU_ANCHOR_FAST/SLOW_MS`).
- **(P)** Decide la GEOMETRÍA del trade: las curvas TP/SL del genoma se
  evalúan en esta τ (`tp_at_tau`/`sl_at_tau`); el horizonte de la posición y
  el trailing respiran con ella.
- **(I)** τ corta dominante = régimen de micro-impulso (trades rápidos,
  stops cercanos); τ larga = régimen tendencial. **(C-05)** Sin la acotación,
  la τ cruda quedaba pegada al extremo lento (31 ≈ 146 años — medido en vivo)
  y las curvas extrapolaban brackets absurdos (+65%/−32%).
- **(∩)** `temporal_spectrum.rs` (dominant_tau_ms).

### Hurst por DFA — estimador multiescala real
- **(F)** Detrended Fluctuation Analysis: `F(τ) ∝ τ^H` con regresión
  log-log sobre MÚLTIPLES τ; gate de calidad r²; calibrado contra RW y AR(1).
- **(P)** El Hurst GENÓMICO (confianza espectral, remapeo de horizonte, ley
  fractal de TP/SL) — el espectro de 32 escalas es su observación viva.
- **(I)** H > 0.5 persistente (tendencia), H < 0.5 anti-persistente
  (reversión). **(D-615)** La versión anterior medía curtosis (E|r|/√E[r²]·√(2/π)
  ≡ 1 para gaussiana) — era ciega exactamente a lo que Hurst mide.
- **(∩)** `feature-engine/src/hurst_dfa.rs`.

### Estadística de difusión — z comparables entre horizontes
- **(F)** Varianza estacionaria EXACTA de (precio − EMA) para paseo aleatorio:
  `Var(e) = (1−α)²σ²/(α(2−α))`, covarianzas entre EMAs de distinto α, σ por
  ATR con factor de Parkinson `√(8/π)`. Cada tendencia → z.
- **(P)** Un solo umbral (z bilateral 95%) sustituye todos los literales en
  bps que comparaban unidades incompatibles (D-621).
- **(I)** |z| < 1.96 = movimiento compatible con ruido; z > 3 en varias
  EMAs = desplazamiento real.
- **(∩)** `god-engine-core/src/diffusion.rs` (D-621/D-624/D-685).

---

## III. PROBABILIDAD, MODELOS Y CALIBRACIÓN — «cuánto creerle al modelo»

### NanoForest (GBDT) — el modelo por símbolo
- **(F)** Gradient-boosted decision trees Rust nativo (`train_forest`),
  vector 48D = swing(34) ⊕ espectral(10) ⊕ macro(4), validación cross-month,
  init_score = log-odds de la base del símbolo.
- **(P)** `ml_prob = sigmoid(forest(x) + init_score)` es LA opinión ML que
  gatea toda entrada (B3.18).
- **(I)** **LA BASE ES DEL SÍMBOLO** (18-23% con etiquetado honesto): prob
  0.30 NO significa «casi seguro que baja» — significa «7 puntos sobre SU
  base». Por eso el gate es por LIFT (§ Gate ML abajo). Modelos {SYM}_MOTOR =
  dirección; _VOL = σ futura (R² 0.106 en NEAR); _VOLU = profundidad.
- **(∩)** `src/bin/train_forest.rs`, `models/*.json`, hot-reload en host.

### Etiquetado honesto — triple barrera
- **(F)** Cada ejemplo se etiqueta con la primera barrera tocada:
  SL −0.18% / TP +0.36% dentro del horizonte (HOST-010).
- **(P)** El etiquetado 50/50 de dirección producía bases irreales; el triple
  barrera produce la base 18-23% contra la que todo lift se mide.
- **(I)** Base ≈ 0.2 = el modelo «en reposo»; un edge real se ve como lift
  sostenido sobre esa base, no como prob > 0.5.
- **(∩)** trainer + `HOST-010`.

### Gate ML por LIFT (B3.18 + B3.36 + D-715)
- **(F)** Genes reparados por `ml_gate_thresholds` (canónico: largo ≥ ½ ≥
  corto, no-finitos neutralizados) → lift ∈ [0.02, 0.25] por lado →
  `entrada ⇔ ml_prob ≥ base_modelo + lift_eff` con
  `lift_eff = lift·(1 − 0.3·agree)` donde agree = acuerdo espectral (S-6).
- **(P)** TODA entrada del motor pasa esta puerta (más el consejo y la
  envolvente).
- **(I)** El lift son «los puntos que el modelo debe superar a SU propia
  base». Acuerdo espectral baja la exigencia ×0.7 (dos fuentes independientes
  alineadas); divergencia la sube ×1.3. Sin modelo del roster: NO se opera.
- **(∩)** `god-engine-core/src/lib.rs` (gate B3.18), `calibration.rs`.

### Conformal + ACI — intervalos con garantía distribucional
- **(F)** Split conformal: cuantiles de scores de calibración → banda de
  predicción con cobertura 1−α garantizada; ACI (Adaptive Conformal
  Inference) ajusta α online con regret bound si la cobertura se desvía.
- **(P)** `conformal_by_coin`: convierte la prob del modelo en banda con
  garantía — cuándo la predicción ES señal y cuándo ruido.
- **(I)** Si el realized cae fuera de la banda más de lo que α promete, el
  ACI ensancha automáticamente: leer el ancho de banda como incertidumbre
  VIVA del símbolo. Por-moneda (M2-H03) para no mezclar distribuciones.
- **(∩)** `god-engine-core/src/conformal.rs`.

### Platt por símbolo — calibración de temperatura
- **(F)** Regresión logística MAP (Newton) sobre (score, outcome) con prior
  Beta — aplana/enciende los scores del modelo a probabilidades reales.
- **(P)** `calibrator_by_coin`: corrige sobre/under-confidence ANTES de que
  el lift gate lea la prob.
- **(I)** Un calibrador con pendiente < 1 = el modelo exagera; > 1 =
  conservador. Se espera deriva lenta — por eso es por-moneda y online.
- **(∩)** `god-engine-core/src/calibration.rs`.

### Ensamble Hedge/Brier — pesos por conocimiento real
- **(F)** Pesos de expertos por pérdida Brier (exponencial, Hedge), n_eff por
  divergencia, z-test de skill; actualiza por kline cerrado con labels de
  BARRERA (D-738: el PnL se cuenta UNA vez).
- **(P)** Combina las perspectivas de modelo en `ml_prob_pure` — la que el
  gate lee SIN sesgo de spot (MOD2/7-029).
- **(I)** Peso alto = ese experto predice barreras mejor AQUÍ; n_eff bajo =
  los expertos dicen lo mismo (falsa diversidad).
- **(∩)** `god-engine-core/src/ensemble.rs`.

---

## IV. RIESGO Y SIZING — «cuánto arriesgar por lo que sabemos»

### Kelly exacta desde Profit Factor
- **(F)** `f* = W·(1 − 1/PF)` (FIX #374) — ¡PF, no odds! Con PF ≤ 1:
  exploración ≤ ¼ del piso (sin edge probado no hay apuesta).
- **(P)** Fracción base de todo el sizing.
- **(I)** f* > 0 ⇔ edge medido; su magnitud escala con W y PF conjuntos.
  Clamp del genoma + ruin cap (abajo) la acotan SIEMPRE.
- **(∩)** `risk-engine/src/kelly.rs` (QO-M0.1 pure fn en leverage_matrix).

### Envolvente Kelly Bayesiana — el juicio de evidencia
- **(F)** Posterior Beta(wins, losses) del edge → LCB(z por régimen de
  capital); shrinkage por evidencia `n/(n+k)`; streak-bound de ruina; tope
  25% (axioma). `risk_fraction` = fracción final; `max_leverage` = techo
  de apalancamiento operable.
- **(P)** LA fracción que el host usa: `lev_from_risk = 0.05·f/sl(τ_entry)`
  (CERT-M8-C01: MISMA fórmula en backtest nativo, replay y vivo).
- **(I)** n < 30 trades ⇒ fracción de bootstrap (leverage 1) — no es
  timidez, es que NO HAY evidencia. Streak negativa reciente ⇒ f_ruina
  aprieta. El z baja (1.64→0.85) en cuentas micro: menos capital = exigir
  MENOS certeza estadística para operar una unidad (capital_regime).
- **(∩)** `risk-engine/src/kelly_envelope.rs`, host sizing, `live_envelope_gate`.

### Ruina centralizada — streak-bound (M5-H03)
- **(F)** `f_cap = 1 − SURVIVAL_FLOOR^(1/streak(q))` con
  `streak = ln(200)/ln(q)` y SURVIVAL_FLOOR = 5%; después axioma 25%.
- **(P)** El techo transversal: bootstrap, micro-kelly, leverage matrix y
  DynamicKelly pasan TODOS por aquí.
- **(I)** q = prob. de pérdida (LCB si hay datos; 0.60 conservador si no):
  q=0.45 ⇒ cap ≈ 0.35; q=0.75 ⇒ cap ≈ 0.10 — peor calidad, menos fracción
  por trade. La exponencial QO-M1.2 clásica queda como diagnóstico
  (monótona decreciente con piso e⁻²: jamás fue un tope válido — ver
  `ruin.rs` para la demostración).
- **(∩)** `risk-engine/src/ruin.rs`.

### VOL-BRAKE — el freno de volatilidad prevista
- **(F)** El predictor {SYM}_VOL estima σ futura; si
  `pronóstico/base > 1.25`, la fracción de riesgo se encoge proporcionalmente.
- **(P)** El sizing reacciona a la volatilidad QUE VIENE, no sólo a la pasada.
- **(I)** R² 0.106 (NEAR): 10.6% de la varianza futura de σ explicada OOS —
  modesto pero real; el freno sólo actúa en pronósticos extremos.
- **(∩)** host sizing + modelos _VOL.

### Régimen de capital — z/k continuos
- **(F)** `micro_weight(capital, min_notional)` interpola z (1.64→0.85) y
  k (50→10) según cercanía al mínimo operable del exchange.
- **(P)** Mismo rigor estadístico a toda escala de cuenta (D-641: sin saltos
  en $20).
- **(I)** Cuenta pequeña ⇒ LCB menos estricto y shrinkage más corto — el
  sistema NO se congela por ser micro, pero el axioma 25% sigue intacto.
- **(∩)** `risk-engine/src/capital_regime.rs`.

---

## V. GEOMETRÍA DEL TRADE — «dónde entrar, dónde salir, cuánto respirar»

### Curvas TP/SL por horizonte (el genoma geométrico)
- **(F)** `param(τ) = exp(a + b·ln τ)` — el genoma evoluciona (a, b) por
  curva; las anclas 30 s/12 h se RE-DERIVAN de los coeficientes
  (`derive_anchors_from_curves`), con `enforce_curve_rr` manteniendo RR ≥
  mínimo en todo τ.
- **(P)** Un stop que escala con el horizonte REAL de la posición: la misma
  geometría para el trade de 40 s y el de 8 h, sin escalones.
- **(I)** b > 0 = los brackets se ensanchan con τ (física correcta: σ ∝ √τ).
  El GA muta (a, b), JAMÁS las anclas directamente (C-10: mutar anclas se
  autodestruía en el roundtrip).
- **(∩)** `quantum-arena/src/config.rs` (curvas), genome to/from_vector.

### Pisos de fricción — nada opera debajo del costo
- **(F)** `friction_floor = fees + slippage_modelo + latencia_precio` por
  pierna; todo EV de entrada lo atraviesa.
- **(P)** El enemigo histórico del sistema fueron los fees (−108% de
  arrastre): ningún trade se certifica si su esperanza no supera la fricción.
- **(I)** Es el SUELO de todas las comparaciones EV; subir cuando el
  maker-spread o la latencia empeoran.
- **(∩)** `tp_sl.rs` + friction floors (EV algebra exacta).

### Escudo cuántico — escalera de protección espectral
- **(F)** UNA escalera relativa al TP: breakeven a fracción espectral
  lerp(0.30, 0.50)·TP del recorrido, half/profit/runner en fracciones
  crecientes; interpolada por persistencia (S-2). Piso físico
  max(fricción ida-vuelta, 2×ATR), TECHO fracción del TP (D-727): NADA se
  arma por encima del objetivo. D-711: una SOLA definición de be_trigger —
  la duplicidad de cotas dejaba la protección en limbo.
- **(P)** La gestión de la posición abierta: asegurar progresivamente lo
  ganado sin decapitar los winners (la escalera ×fee antigua disparaba toda
  dentro del rango del TP: WR 73% con RRR 0.30).
- **(I)** Persistencia alta = escalera tardía (deja correr); baja = temprana
  (asegura el retroceso). BE activo = el trade ya no puede perder (neto de
  fricción).
- **(∩)** `god-engine-core/src/trailing.rs` (B3.27 + S-2 + D-711 + D-727).

### Escalera del núcleo (BE/trailing en process_event)
- **(F)** BE activa cuando el recorrido cubre fricción y fracción espectral
  del TP; trailing a lerp(0.60, 0.80)·TP, siempre < TP×0.95 (D-727).
- **(P)** La protección del camino caliente del core (la de trailing.rs es
  la del escudo por defecto; ambas comparten el invariante).
- **(I)** Ver teoría en § Escudo cuántico — misma matemática, dos capas.
- **(∩)** `god-engine-core/src/lib.rs` (hunks BE/trail).

---

## VI. EVOLUCIÓN Y SELECCIÓN — «mejorarse a sí mismo sin mentirse»

### Fitness único — utilidad logarítmica penalizada por ruina
- **(F)** `fitness = ln(final/initial) − λ·dd²`; sin trades mínimos ⇒
  INVIABLE (−∞). Único objetivo en TODOS los promotores (D-652..655).
- **(P)** Selección de genomas: crecimiento compuesto con aversión a la
  ruina — crecer poco sin drawdown > crecer mucho con él.
- **(I)** Negativo = destruyó capital o no operó lo mínimo; la penalización
  cuadrática en dd hace que un 50% de drawdown sea casi irrecuperable para
  el fitness (por diseño: es lo que tarda 100% en recuperar).
- **(∩)** `evolution-engine/src/fitness.rs` (+ `fitness_compute` en core).

### DSR — Deflated Sharpe Ratio (Bailey & López de Prado 2014)
- **(F)** `DSR = Φ(((SR − SR₀)·√(T−1))/√(1 − γ₃SR + ((γ₄−1)/4)SR²))` con
  SR₀ el máximo esperado por pura suerte entre N pruebas (multiplicidad),
  γ₃/γ₄ skew/kurtosis de los retornos; CDF por A&S 7.1.26.
- **(P)** Con 2 000 candidatos por ronda, el mejor por suerte supera
  CUALQUIER umbral fijo — el DSR es la puerta de promoción (QO-M1.1):
  sólo un edge que sobrevive la corrección por multiplicidad y colas pasa.
- **(I)** DSR < umbral = «el edge no es estadísticamente distinguible de
  probar 2 000 monedas al aire». CERT-M8-C03: evalúa los retornos del
  CANDIDATO, jamás los del incumbente.
- **(∩)** `evolution-engine/src/selection_stats.rs`.

### Walk-forward del daemon — el motor real de juez (M8-H01)
- **(F)** Pre-screen de momentum (ventana corta) rankea 2 000 mutantes →
  top-24 + incumbente se evalúan con `wf_evaluate_real`: GodEngineCore
  completo (consejo, ML gate, física de fees) + envolvente del host sobre
  micro-ticks Brownian-bridge de las series por-moneda (8 ticks/retorno,
  máx 8 monedas, ventana 400). Capital semilla = capital VIVO.
- **(P)** Los genomas se optimizan para LA MÁQUINA QUE OPERA — antes el
  juez era un simulador de momentum con física propia.
- **(I)** El incumbente SIEMPRE compite (anti-regresión, espíritu D-740):
  si ningún mutante lo supera en el motor real, no hay hot-swap.
- **(∩)** `evolution-engine/src/online_daemon.rs::wf_evaluate_real`.

### Shadow Forest — el mundo de control
- **(F)** 10 universos clonados con el mismo motor, muestreo 1-in-10, mismo
  latency_panic (M8-H04), fitness `fitness::compute` con peak_capital
  reseteado en replant (F-56).
- **(P)** Contraparte de realidad: cosecha el mejor genoma SOLO si supera al
  control por fitness unificado.
- **(I)** Leaderboard por universo = diversidad de estrategias vivas; el
  spread control-vs-mejor es la señal de que la evolución aporta.
- **(∩)** `evolution-engine/src/random_forest.rs`.

### CMA-ES + recocido del daemon
- **(F)** Optimización por matriz de covarianza (D-140: covarianza viva
  preservada); reality_gap es factor multiplicativo ≤ 1, JAMÁS sobrescribe
  el fitness (M8-C04).
- **(P)** Exploración continua del espacio de 139+ genes entre rondas.
- **(∩)** `evolution-engine/src/cma_es.rs`.

### RANSAC de Sharpe — control de degradación
- **(F)** Sharpe robusto con t-stat sobre la muestra COMPLETA (M8-H03: sin
  trim de outliers — recortar la cola negativa es recortar la evidencia de
  degradación).
- **(P)** Insumo del kill-switch de deriva del daemon (activo también en
  demo, M8-H02).
- **(∩)** `online_daemon.rs`.

---

## VII. DEFENSA, EJECUCIÓN Y ESTADO — «sobrevivir a lo inesperado»

### Sistema inmune (kill-switch por capas)
- **(F)** Vigilante 5 s: (1) STOP_TRADING.LOCK del operador (la puerta
  humana gobierna, D-697), (2) drawdown vs límite del genoma sobre el pico
  del PLANO MÁS CONSERVADOR (arena vs exchange), (3) latencia obsoleta
  SOSTENIDA con medición (M4-H04: stall de WS solo no cuenta sin evidencia
  de latencia). Al disparar: kill-switch + flatten-all. Drift-kill con
  AUTO-REARME tras 10 cierres limpios (M4-C02); freno de rate-limit
  429/418 AUTO-EXPIRABLE (no latch eterno).
- **(I)** Cada capa ataca un modo de fallo DISTINTO: humano, solvencia,
  transporte, contaminación de datos, abuso del exchange.
- **(∩)** host immune + `drift_kill_*` + `rate_brake_until_ms`.

### Reconciliación — la verdad del exchange
- **(F)** Cada 60 s: positionRisk ↔ estado local. Adopciones con margen del
  leverage REAL (S-06/D-726) + entry_fee imputado (M4-H05 vía open_with_fee,
  R7-0) + validación precio/cantidad (D-729: entrada sin precio no es
  entrada). Órdenes por algoId (D-702); piernas huérfanas purgadas (B2.6).
- **(I)** «Adopted Reg: N» = posiciones del exchange absorbidas; el estado
  local NUNCA es la verdad final — el exchange sí.
- **(∩)** `execution-engine/src/reconciliation.rs`.

### Lazo de confirmación de órdenes (M4-C01)
- **(F)** `OrderResolution {Accepted, Rejected, Timeout}`: el OK del kernel
  al enviar NO es ack del exchange; la confirmación llega por
  ORDER_TRADE_UPDATE o poll REST acotado — NUNCA re-envío (duplicación).
- **(∩)** `execution-engine/src/executor.rs`.

### Bus mmap con seqlock (M6-H01)
- **(F)** Frames de 64 B; seq u32 en bytes 12-15: escritor marca impar
  (invalidate) → sfence → payload NT → commit par → sfence; lector acepta
  sólo seq par y estable.
- **(P)** Telemetría predicción-vs-realidad sin locks ni lecturas rasgadas
  — el non-temporal store NO está ordenado por sí solo (el «commit al final»
  sin fence intermedio era la puerta del tearing).
- **(I)** Frame descartado = se escribió durante tu lectura: la pérdida de
  UN frame de telemetría es gratis; la lectura rasgada, no.
- **(∩)** `storage-engine/src/mmap_bus.rs`.

### Shutdown graceful (M4-H02)
- **(F)** ^C #1 → flag + centinela → drenaje (flatten-all + persistencia de
  la envolvente + exit 0); ^C #2 → salida forzada.
- **(I)** Ya no existen posiciones huérfanas por cierre del proceso.
- **(∩)** host `god_engine.rs` (loop principal).

---

## VIII. CÓMO LEER LA TELEMETRÍA — guía de interpretación rápida

| Señal en log/telemetría | Significado operativo |
|---|---|
| `ml_prob` ≈ base del símbolo (0.18-0.23) | El modelo no ve edge — normal |
| `ml_prob` ≥ base + lift | Candidata a entrada (falta consejo + envelope) |
| `envelope_vetoes` alto | Capital/evidencia no sostienen el sizing pedido |
| `fused_score` ±alto | Alineación espectral en escalas con persistencia |
| `τ dominante` baja → alta | Régimen micro-impulso → tendencia de banda |
| `Adopted Reg: N` | N posiciones absorbidas del exchange (no eran nuestras en libro) |
| `strike NO aplicado (sólo transport)` | Stall de WS sin latencia medida: no cuenta |
| `WF-REAL ... mejor fitness` | El juez-motor evaluó al top-K; incumbent compite |
| `kill-switch ... auto-rearme en 10` | Drift de contabilidad: corta pero NO para siempre |
| `shadow forest reentrenado N obs` | Contraparte de control viva con N ejemplos |

---

## INVENTARIO DE RAMAS (estado de la unificación, 2026-09-19)

| Rama | Estado |
|---|---|
| `main` | **= origin/main.** Contiene TODO el trabajo vivo |
| `claude/decima-ola-auditoria-2` | FUNDIDA (d9f54ce4) — integración dual de la sesión paralela |
| `claude/decima-ola-auditoria-forense` | Ya contenida en main (no-op) |
| `claude/kind-jackson-455bb4` | Ya contenida en main (no-op) |
| `develop` | Ya contenida en main (no-op) |
| `subagent-*` (×3) | **ARCHIVO**: limpieza de artefactos que main ya no rastrea (base jun-29; fusionar solo regresaría estados viejos de Cargo.toml) |
| `backup-before-cleanup` | **ARCHIVO**: sus commits AÑADEN artefactos de build al repo — fusionarla dañaría main permanentemente |
| `v7-unificacion-wip` | **ARCHIVO**: WIP superseded — sus definiciones fueron portadas selectivamente en 2026-09-08 (ver memoria del episodio V7) |

*Generado en R8 (2026-09-19). Mantener junto a los doc-comments de cada
módulo: este atlas es el índice interpretativo; el módulo es la fuente.*

---

## IX. ADENDA 2026-09-24 — Qué mide realmente el estado espectral

Esta sección añade precisión al historial sin eliminarlo. Su fuente es el
código inspeccionado sobre `edb7194e` y la corrección local del contrato
temporal. El detalle, las reproducciones y pendientes están en la
[auditoría CES-001 a CES-017](docs/AUDITORIA_CONTINUIDAD_ESPECTRAL_2026-09-24.md).

### Tiempo, resolución y evidencia

- **Fórmula:** `τ_i = 10^-6 ms · 4^i`, i=0…31. La malla representa 1 ns–146
  años; la actualización recibe timestamps enteros en ms. Hay aproximadamente
  1.66 intervalos por década. El coste del banco es O(32).
- **Propósito:** mantener un conjunto de filtros de respuesta temporal
  distinta e interpolar consultas en log(τ).
- **Interpretación:** representar 1 ns no permite distinguir eventos de 1 ns;
  representar 100 años no aporta historia secular. Hacen falta resolución,
  cobertura e incertidumbre por escala. Los nodos no son muestras independientes.

### Normalización de sorpresa y persistencia de signo

- **Fórmula:** `α=-expm1(-Δt/τ)`; `dev=(precio-EWMA_previa)/EWMA_previa`;
  `v←v+α(|dev|-v)`; `s=tanh(dev/v)` con las protecciones del módulo.
- **Propósito:** comparar amplitudes relativas de desviación sin dejar que
  el precio nominal determine la escala de la señal.
- **Interpretación:** v es una media suavizada de desviación absoluta. El
  campo llamado `momentum_z` no usa una desviación típica y no implica un
  p-valor gaussiano. La señal tampoco es una probabilidad.
- **Persistencia:** `p←p+α(agree-p)`, con `agree` en {-1,0,1}. El dominio de p
  es [-1,1], no [0,1]. `hurst_at` devuelve `(p+1)/2` por compatibilidad:
  debe leerse como índice reescalado, no como estimación de Hurst.

### Una medida común para el aprendizaje y sus consumidores

- **Fórmula vigente:** `w=clamp(2·|p|·g,0.02,3)`; `F=Σw·s/Σw`;
  `e=w·|s|`; `τ*=exp(Σe·lnτ/Σe)` cuando existe masa.
- **Propósito:** fusión, coherencia, densidad, proyecciones y centroides deben
  describir el mismo estado aprendido. El cierre de un trade refresca F sin
  esperar al siguiente evento.
- **Interpretación:** g es una ganancia adaptativa, no evidencia de habilidad
  por sí misma. Las constantes conservadas son política heredada pendiente
  de calibración. El centroide no es necesariamente un máximo ni un horizonte
  ejecutable. Las salidas operativas y las curvas aún conservan recortes
  históricos; esta adenda no los declara eliminados.

### Entropía de masa y dirección son análisis diferentes

- **Fórmula:** `q_i=e_i/Σe`; `H=-Σq_i ln(q_i)/ln(32)`.
- **Propósito:** describir cuánto se reparte la masa entre nodos.
- **Interpretación:** H=1 es masa uniforme, H=0 concentración en un nodo.
  Treinta y dos señales +1 con iguales pesos tienen H=1 y consenso alcista
  perfecto. Por ello «H alta = caos térmico» no es una inferencia válida.
  El veto que usa esa interpretación sigue abierto como CES-007.

### Aprendizaje y trazabilidad

La calificación predictiva debe utilizar la predicción de entrada y su
objetivo exacto. Pérdida neta por costes no implica movimiento adverso del
precio. El evaluador de un candidato debe mantener fija la evidencia de
mercado; que use el motor real no vuelve real un tape sintético. Estos
contratos siguen abiertos y están descritos en CES-010 a CES-012.

### Estado de integración en este corte

Tras fetch: main 43 commits por delante de origin/main. La rama activa de
auditoría aporta 13 commits no contenidos; main aporta 46 no contenidos en
esa rama. Una simulación de merge encontró conflictos en 11 archivos.
El inventario del 19 de septiembre es histórico: no acredita unificación
del estado actual. No se publicaron estos cambios durante esta revisión.
