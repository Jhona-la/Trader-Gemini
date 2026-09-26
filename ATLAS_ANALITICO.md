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

---

## Ampliación científica FMT — 24 de septiembre de 2026

Este anexo se añade sin sustituir las explicaciones anteriores. Las
interpretaciones históricas incompatibles con la evidencia siguiente no
deben usarse como garantía vigente. Referencia detallada:
[Auditoría de fundamentos científicos y diseño espectral](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_2026-09-24.md>).

### Qué calculan realmente las cantidades y para qué pueden servir

| Cálculo actual | Significado identificable | Uso potencial y límite de interpretación |
|---|---|---|
| EMA de `abs(Δprecio)` publicada como velocidad | Actividad monetaria por evento, sin signo | Medir intensidad de movimiento; no orientar Long/Short como velocidad direccional |
| Ganancia Kalman `P/(P+R)` | Peso de innovación bajo un modelo de covarianzas | Estimar estado latente; exige P/Q/R con unidades compatibles y reloj explícito |
| Cociente `E|r|/sqrt(E r²)` | Descriptor de forma de distribución | Puede discriminar distribuciones; no estima memoria temporal Hurst por sí solo |
| Bin FFT de 64 retornos | Oscilación por índice de evento | Describir regularidad local; no es frecuencia en Hz sin relación con timestamps |
| Densidad `sqrt(a/pi) exp(-a x²)` | Densidad gaussiana antes de recortes | Probabilidades requieren integración; un valor puntual puede superar 1 |
| Fuerza `-kx-4λx³` | Gradiente negativo de potencial clásico | Feature no lineal de reversión; no acredita dinámica cuántica |
| `A sech(A(x-vt))` | Envolvente de perfil | Feature localizada; no acredita solución dinámica NLS del libro |
| Tsallis bid/ask con q=1,5 | Concentración de una distribución binaria | Su máximo es 0,5857864; umbral 0,60 no filtra libros válidos por entropía |
| Diferencia normalizada de votos | Acuerdo relativo entre scores | No equivale a probabilidad calibrada ni evidencia independiente |
| Multiplicación de pesos por gradiente recortado | Regla heurística de adaptación | Puede compararse contra expertos online; no es PPO sin razón de políticas |
| P-valor conformal | Rango de no-conformidad frente a calibración | No es probabilidad de ganar; distinguir cobertura marginal y selectiva |
| `p(1-1/PF)` | Kelly bajo parametrización binaria | No determina el óptimo de retornos continuos multiactivo con colas y costes |
| Límite derivado de una racha | Supervivencia frente a una secuencia elegida | No certifica probabilidad de ruina a horizonte universal |
| RMS entre dos vectores de features | Distancia dependiente de normalización | No es exponente de Lyapunov ni covariance |
| Entropía de Long/Short/Flat | Diversidad marginal de acciones | Aleatoriedad puede maximizarla; no prueba robustez o adaptación |

### Modelo unificado propuesto

El objeto de análisis es un campo `Z(t, log(τ/τ₀), activo, canal)` con
incertidumbre, soporte observacional y procedencia. Representar τ desde
1 ns hasta 100 años no implica disponer de observaciones a esa resolución
ni de evidencia secular. La evolución entre eventos y la asimilación de
nuevas observaciones son operaciones distintas.

El genoma debe parametrizar funciones temporales y contextuales con
dominios adecuados; las antiguas anclas de scalping/swing pueden ser
vistas transitorias de compatibilidad, no la definición del universo.
La epigenética debe poseer estado, evidencia, actualización, persistencia
y consumidores verificables. Un grafo de nombres conectados no basta.

Cada fórmula nueva necesita un pasaporte: objetivo, variables y unidades,
supuestos, fuente, estimación, incertidumbre, reloj, consumidor, coste,
prueba de falsación y resultado fuera de selección. El informe desarrolla
13 líneas de investigación: modelos de estado, scattering, Hawkes marcado,
firmas de caminos, grafos/Hodge discreto, aprendizaje online, changepoints,
conformal, inferencia secuencial, Kelly robusto, control de inventario,
evolución restringida y Koopman/redes tensoriales.

### Estado verificable de esta ampliación

Se documentan FMT-001 a FMT-026 con causas, límites y criterios de cierre;
no se implementaron sus reparaciones. Pasaron 141 tests existentes de
feature-engine, signal-engine y dark-alpha-engine. Los contraejemplos
algebraicos identifican propiedades que esa batería no cubre.

La investigación bibliográfica utilizó fuentes primarias y distinguió
transferencias propuestas de resultados demostrados. La discusión de los
Problemas del Milenio incluye la actualización oficial de Clay de
septiembre de 2026; no equipara anuncios, pruebas verificadas y utilidad
financiera. La sofisticación nominal no es un criterio de admisión.

## Ampliación científica II — contratos, identificabilidad y evolución temporal (24-09-2026)

Se añade, sin sustituir los análisis anteriores, la
[auditoría científica II](</C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_II_2026-09-24.md>).
Contiene FMT-027 a FMT-046, con evidencias, contraejemplos, alcance
operativo y criterios de cierre. Son 20 entradas documentales adicionales,
no 20 incidentes de producción acreditados ni un recuento deduplicado
contra todos los informes históricos.

### Qué calculan las cantidades y dónde cambia su significado

| Cantidad | Interpretación válida | Riesgo identificado |
|---|---|---|
| Probabilidad del bosque entrenado con barreras | Probabilidad de primer TP condicionada a resolución antes del límite | No equivale a dirección de vela, éxito de corto ni beneficio neto; FMT-028 |
| Salida calibrada | Estimación de probabilidad para un evento especificado, si el ajuste es válido | Newton puede saturarse y el consumidor confunde cero con ausencia; FMT-030 |
| Habilidad del ensamble | Diagnóstico del agregado bajo una población y un régimen de inferencia | Se atribuye el error solo a una red; dependencia temporal invalida garantías nominales no justificadas; FMT-031/032 |
| Semivida OU | Tiempo físico de reducción esperada de una desviación bajo κ positivo estimado | La API fuerza reversión e ignora el timestamp; FMT-033/034 |
| Z de basis | Distancia normalizada de una relación de precios | No demuestra rango de cointegración ni estimación Johansen; FMT-035 |
| Kelly mediante PF | Identidad de p y pagos de una misma población | Mezclar p de señal con PF histórico puede cambiar el signo de la asignación; FMT-044 |
| Correlación con cesta propia | Dependencia de un activo con un benchmark que lo incluye | No es correlación por pares ni beta; FMT-046 |

### Continuidad temporal y justificaciones que sí se conservan

El dominio temporal debe modelarse conjuntamente, con incertidumbre y
soporte observacional explícitos. Los estados exponenciales pueden
aproximar un kernel continuo sin crear estilos operativos separados.
Una malla numérica es admisible si su error se controla; declarar una
escala de 1 ns no acredita observación ni capacidad de decisión a 1 ns.

Se documenta una demostración positiva: para curvas TP/SL de potencia,
probabilidad fija y coste fijo no negativo, comprobar EV no negativa en
ambos extremos certifica todo el intervalo por concavidad de una
transformación logarítmica. Hay que corregir primero los límites para
exponentes negativos —FMT-040—. La demostración no se extiende sin más
a probabilidades/costes dependientes de escala o a splines arbitrarios.

### Investigación propuesta y conexión con el genoma

T14–T20 añaden memoria de Volterra, OU/VECM identificado, covariación y
lead–lag asíncronos, primer paso con censura, geometría de sensibilidad,
métodos cuántico-inspirados con costes completos y aprendizaje
numéricamente verificable. Se detallan unidades, hipótesis, consumidores,
complejidad, baselines y pruebas de rechazo. Son propuestas, no mejoras
de rendimiento demostradas ni algoritmos ya implementados.

El análisis de sensibilidad distingue gen mutable de gen identificable
y de gen efectivamente consumido. El grafo diagnóstico propuesto
conecta raíces observadas, estado, predicción tipada, decisión, fills,
resultado maduro y publicación versionada. No se certifica esa
conectividad a partir de los grafos de sintaxis actuales —FMT-043—.

### Verificación y estado

107 tests existentes de strategy-core y god-engine-core aprobados.
23 archivos Rust adicionales leídos completos, con manifiesto y hashes;
otras conexiones se revisaron por tramos. No se afirma revisión íntegra
de los 1.119 archivos versionados. Los 20 hallazgos siguen abiertos en
esta adenda: no se modificó código, no se promovieron genomas ni se
hicieron operaciones, despliegues, commit, push o merge.

## Ampliación científica III — autoevolución y validez de la evidencia (24-09-2026)

Se añade la [auditoría científica III](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_III_2026-09-24.md>),
con FMT-047–071: 25 entradas documentales, abiertas, con alcance operativo
o auxiliar identificado. No se suman como incidentes ni como fallos
deduplicados frente a todo el histórico.

### Contratos que explican la diferencia entre evolución aparente y efectiva

| Cálculo o mecanismo | Para qué debería servir | Diferencia encontrada |
|---|---|---|
| Fitness logarítmica | Comparar crecimiento con preferencias de riesgo explícitas | La función completa no es aditiva ni invariante al apalancamiento; FMT-047 |
| Suficiencia de operaciones | Saber si hay evidencia para certificar un candidato | La abstención se convierte en peor resultado que una pérdida finita; FMT-048 |
| Replay de candidatos | Comparar políticas ante el mismo mercado | Usa PnL del incumbente como precios y spread dependiente del candidato; FMT-049/050 |
| OOS/DSR | Medir soporte estadístico bajo el protocolo correspondiente | Reutilización adaptativa y parámetros de ventana completa; FMT-051 |
| Bosque online | Aprender un evento definido a partir de features causales | Dataset heterogéneo, umbrales sobre clases y beneficio tratado como dirección; FMT-052–054 |
| Watchdog evolutivo | Detectar deterioro atribuible y reaccionar sin borrar otras protecciones | Relectura de evidencia, propietario y generación no verificados; FMT-055/056/071 |
| Mutación/compilación inmune | Probar que el candidato corrige una conducta | Tests de literales, módulo no integrado y artefacto no vinculado; FMT-060–064 |

### Nueva transferencia científica, con supuestos y rechazo

T21 incorpora evaluación off-policy/doblemente robusta: exige acción,
contexto, recompensa y soporte; no permite convertir beneficios en
cotizaciones. T22 propone presupuestos y grafos de hipótesis para la
reutilización adaptativa de datos; repartir alpha no repara tests inválidos.
T23 estudia crecimiento con restricciones de drawdown y estado de
trayectoria, incluidos procesos de Azéma–Yor, sin importar garantías de
mercados ideales a fills discretos. T24 propone pruebas metamórficas,
derivadas del contrato y su tolerancia, que sí ejecuten el comportamiento.

El anexo incluye un diagrama de conexiones realmente inspeccionadas,
contraejemplos y una hoja de ruta. No basta que dos nodos intercambien
f64: la arista debe preservar evento, unidades, timestamp, objetivo y
generación. Las plantillas auxiliares aún pueden generar arquitectura
dual, aunque no se localizó su consumidor vivo; se distingue este
productor de los simples nombres heredados.

Verificación de la ronda: 17 tests existentes aprobados, 20 archivos
Rust adicionales completos con hashes y conexiones grandes leídas por
tramos. Los manifiestos de las tres rondas reúnen 67 Rust distintos;
no equivalen a la totalidad del proyecto. No se modificó código,
armado, configuración operativa, genomas o ramas. Las propuestas siguen
pendientes de implementación y validación.


---

## Ampliación científica IV — aprendizaje servido y conexión del genoma (2026-09-24)

Informe completo y visualización diagnóstica:
[Auditoría de fundamentos científicos IV](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_IV_2026-09-24.md>).

Se agregan FMT-072–094, 23 hallazgos abiertos, sin sustituir los anexos
anteriores. Destacan una segunda lectura Hebbiana que aún usa una clave
incorrecta y cae al global; el entrenador de índices 54D que admite
modelos de otra dimensión; entrenamiento sin Welford frente a validación
con Welford; pérdida de validación que no penaliza predicciones ausentes;
rollback parcial; features Omni dependientes de denominación; y OFI
con profundidad inicial no guardada.

La derivada del colchón respecto del gen vale 1−w y se anula cuando
la escasez llega a w=1. Una interpolación suave no garantiza que el
genoma conserve efecto. El informe distingue ese mecanismo del veto
global de régimen, las restricciones legítimas y las APIs auxiliares
que todavía no tienen consumidor vivo localizado.

T25 explica el transporte afín de coordenadas y sus límites frente a
clipping y estado Adam. T26 propone información dirigida en tiempo
continuo para medir conexiones predictivas, no para declarar causalidad
económica u omnisciencia. Se concreta también T07 mediante posterior
de cambio. Se consultaron fuentes primarias y el catálogo oficial de
Clay; no se presentan teorías prestigiosas como garantías de alpha.

Cobertura: 15 Rust completos nuevos, 4.781 líneas, hashes y conexiones
dirigidas. Total de las cuatro rondas: 82 Rust distintos, no todo el
proyecto. Pasaron 95 tests existentes seleccionados. Se documenta por
qué finitud, clipping o una batería verde no certifican el contrato.
No hubo reparación operativa, cambios de genomas, trading ni Git.

## Ampliación científica V — riesgo realizable y memoria temporal (2026-09-24)

Documento detallado: [Auditoría de fundamentos científicos V](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_V_2026-09-24.md>).

Se añaden **FMT-095–113: 19 fichas abiertas**, sin reemplazar el historial.
El hallazgo conectado prioritario es FMT-113: Kelly y vol_brake modifican
un leverage, pero el host conserva la cantidad candidata; el adaptador
de margen puede subir ese leverage. Notional, riesgo al stop y margen
no son intercambiables. El replay repite la lógica: igualdad entre
entornos no demuestra corrección del contrato.

La revisión detalla también: cambio implícito de probabilidad al
aplicar el RR genómico; contradicción entre piso difusivo y cap de SL;
escalón micro de 55 bps; bootstrap que ignora evidencia adversa madura;
excepción al riesgo mínimo; doble EMA del payoff y regularización
monetaria; contexto de cierre sin identidad/as-of; evidencia encolada
sin confirmación durable y muestra censurada por congestión.
La interpolación espectral conserva continuidad pero no todas las
identidades declaradas entre sus campos.

Las capacidades auxiliares se etiquetan como tales: router, simulador,
compounder y almacén temporal no tienen consumidor vivo localizado
en la búsqueda realizada. Sus defectos no se presentan como causa
demostrada del backtest principal. Se preserva el reconocimiento de
CES-018 conectado y de las correcciones espectrales en nodos.

T27 propone factibilidad y filtros de seguridad sobre la acción final;
T28 propone reducción de orden orientada a observabilidad. Ambas tienen
condiciones, coste y criterios de refutación; la verificación del texto
completo de T28 queda pendiente. El catálogo T01–T26 se conserva.
No se atribuye ventaja cuántica ni resolución de problemas del milenio.

Cobertura adicional: 8 Rust completos, 3.615 líneas; acumulado FMT:
90 Rust distintos de 289, con lecturas dirigidas separadas.
Pasaron 34 tests existentes. No es auditoría integral ni certificación
productiva. Sólo documentación: sin código operativo, genomas o Git.

## Ampliación científica VI — reparación local y riesgo monetario verificable (2026-09-24)

Documento detallado: [Auditoría científica VI y rehabilitación verificable del riesgo](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_VI_2026-09-24.md>).

Esta ronda sí modifica código, bajo la autorización de mejora posterior:
tres archivos de risk-engine y dos suites nuevas de integración. Conserva
los informes previos como evidencia histórica; no convierte sus estados
antiguos en afirmaciones sobre el código actual.

FMT-096 queda corregido localmente: el target RR no reduce el TP base ni
cambia la hipótesis probabilística. FMT-098/099 se corrigen en la envolvente:
sin edge no se financia exposición micro implícita y el mínimo del exchange
no permite exceder el presupuesto. El consumidor maduro de replay tiene
regresiones de veto, rollback y aceptación con evidencia positiva.

Se agregan **FMT-114–116**, con reparación local y contraejemplos reproducidos:
datos no finitos y varianza beta desbordada; filtro de notional inválido;
pisos de probabilidad/payoff que inventaban edge. Dos casos anteriores
permitían f=25 % y f≈0,504748 % pese a no tener ventaja bajo sus inputs.

La nueva API ExposureBudget hace explícitos B=Cf, N=|Q|P y pérdida nominal
N(d+c). Proyecta cantidad sobre lotes y mínimos sin aumentar la propuesta
ni relajar el presupuesto. Los costes son una entrada explícita. No
equivale a una garantía frente a gaps, liquidación o riesgo de cartera.

**FMT-113 permanece abierto:** host/replay aún no aplican esa proyección
a la cantidad final. Sus cambios concurrentes fueron preservados. Sigue
existiendo el bootstrap externo n<30; tampoco se repararon en esta ronda
el piso/cap de barreras, doble EMA, identidad de cierres o durabilidad.

La investigación primaria distingue restricción probabilística de Kelly,
heurística de racha y presupuesto nominal. El informe explica variables,
unidades, supuestos y límites; no atribuye ventaja cuántica ni convierte
el dominio representable 1 ns–100 años en resolución observacional.

Verificación: **76 tests seleccionados aprobados; 24 nuevos, de los cuales
12 fueron observados fallando antes de la reparación**. Compila el host
con cargo check; no se ejecutó el motor. Cobertura acumulada de lectura
completa: 91 Rust preexistentes distintos de 289, más los dos tests nuevos
contados aparte. No es auditoría integral ni certificación productiva.
Sin despliegue, trading, cambios de genomas, reinicios, commit, push,
merge o fetch.

## Ampliación científica VII — soporte espectral y selección evolutiva (2026-09-24)

Informe completo: [Auditoría científica VII](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_VII_2026-09-24.md>).

Se añaden **FMT-117–123**, siete fichas nuevas. FMT-117 sigue abierto en
el flujo temporal: una estimación sobre muestras no acredita cobertura
universal ni regularidad del reloj. Los otros seis tienen reparación
local; también se repara el orden de fitness negativo de FMT-011.

El DFA publica pendiente cruda y soporte efectivo, conserva homogeneidad
frente a amplitud, no pierde retornos finitos por overflow del cociente y
no convierte una pendiente fuera del modelo en Hurst válido mediante
clipping. Se corrigió además el generador de tests que producía sólo
innovaciones negativas. R² pasa a describirse como ajuste, no certeza.

En CMA, una penalización ya no mejora fitness negativo ni un NaN gana
por convertirse en −1e9. El caller de evolución usa una ruta explícita
de backtest: su crecimiento de capital no se interpreta como Sharpe live.
No disponer de referencia live sigue siendo ausencia de evidencia, no
validación del candidato. Los gates de promoción no se ejecutaron.

Se añaden dieciséis tests; seis regresiones heredadas fallaron antes de
repararlas. Otra detectó y corrigió un problema de precisión introducido
en la primera versión de esta ronda, que se documenta sin atribuirlo al
código anterior. El informe registra resultados, comandos y hashes.

T29 desarrolla diagnóstico de escalamiento con soporte y falsación de
modelos; T30, contratos de evidencia y utilidad. Las fuentes de DFA y CMA
acotan las garantías, no certifican alpha ni ventaja cuántica.

Cobertura acumulada: 92 Rust preexistentes distintos de 289; no auditoría
integral. FMT-003/012/013/113 y otros permanecen abiertos. Se conservaron
los cambios ajenos de host/replay/risk-lib. Sin despliegue, operación,
genomas, reinicios, commit, push, merge o fetch.

## 2026-09-24 — Ronda científica VIII: continuidad y evidencia evolutiva

Se añade [el anexo VIII](docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_VIII_2026-09-24.md),
sin sustituir las siete rondas anteriores. Documenta FMT-124–129 con
causa, contraejemplo, impacto, reparación o criterio de cierre. Tres
nuevos hallazgos se reparan localmente: contrato estructural de generación,
atracción PSO sin observaciones y amortiguación CSA mal transcrita.

FMT-013 avanza: el supervisor compone cambios relativos de exploración
sin sobrescribir sigma aprendido. Dos generaciones deterministas conservan
0,1837367 y 0,1495964 para su siguiente muestreo. Esto no certifica la
política heurística del supervisor ni la geometría completa del híbrido.

Se añaden 16 tests, cinco con demostración rojo→verde; 25 tests distintos
aprobados en total y cargo check del host satisfactorio. T31 propone
invariancias de geometría de información como criterios de contraste;
T30 se amplía con identidad de objetivo, propuesta y evidencia.

Importante: el host llama a OnlineEvolutionDaemon, no se localizó un
llamador operativo de start_evolution_loop/CMA. No se afirma impacto ya
conectado a demo o producción. Persisten comparabilidad de memorias,
población sólo aparentemente adaptativa y escalas de muestreo/update
incompatibles, además de FMT-012/048/058/059/113/117 y otros previos.

Cobertura completa acumulada sin aumento artificial: 92 Rust preexistentes
de 289. Esta ronda profundiza archivos ya cubiertos. El grafo del anexo
distingue rutas y autoridades; graphify-out ajeno permanece intacto.
Sin operación, despliegue, genomas, reinicios, commit/push/merge/fetch.

## 2026-09-24 — Ronda científica IX: evidencia operativa y autoridad de parada

Se añade [el anexo IX](docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_IX_2026-09-24.md),
centrado en LiveEvolutionDaemon, cuya llamada desde el host sí está
localizada. FMT-055 avanza parcialmente: media studentizada corregida,
normalización sin piso absoluto, degeneración/error tipados y EWMA sólo
con revisiones nuevas. Se corrigen mensajes de confianza bayesiana
inexistente; la fórmula del gate sigue siendo heurística y pendiente.

FMT-056 avanza en autoridad: el daemon puede activar la parada compartida,
pero ya no desactivarla por recuperación del score. Se explicita el coste
operativo: el rearme requiere una autoridad que conozca todas las causas.
Rollback condicional de linaje y conservación ante fallo siguen abiertos.

Nuevos FMT-130–132: entrega no confirmada al ledger, upsert de pesos usado
como historial y deltas de PnL/capital posterior presentados como retornos.
Incluyen contraejemplos, impacto y criterios de cierre; no se declaran
reparados. T32 aborda evaluación off-policy con soporte y causalidad;
T22 se amplía con inferencia secuencial y límites de consultas repetidas.

Verificación: 47 tests distintos aprobados, 22 nuevos y cuatro rojo→verde;
cargo check del host satisfactorio. No equivale a ensayo económico ni
prueba del proceso desplegado. Cobertura completa: 93 Rust preexistentes
de 289; se añade la lectura íntegra de evolution_ledger.rs, sin modificarlo.
Se conservan informes y cambios ajenos. Sin operación, despliegue,
genomas, reinicios, commit/push/merge/fetch.

## Continuación X — contrato temporal del genoma y campo espectral

[Auditoría científica X](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_X_2026-09-24.md>) y [artefacto estructurado X](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_X_2026-09-24.json>).

Se repara FMT-040: la política de fricción admite un mínimo para pendientes
positivas, un máximo para negativas y una banda vacía real cuando corresponde.
FMT-041 queda parcial: se corrige su interpretación sin retirar el guard.
FMT-112 explicita interpolación independiente de observables y masa nodal,
conservando los valores existentes. FMT-133 corrige la derivada local por
tramos. FMT-134 queda abierto: lectores del mismo genoma discrepan en dominio,
OBI y trailing; tres pruebas diagnósticas reproducen esa diferencia.

44 tests seleccionados aprobados, 21 nuevos, 12 regresiones rojo→verde;
cargo check del host pasó sin ejecutar el motor. T33 propone un experimento
de espacio de escalas causal y selección normalizada, no una integración
activada ni una ventaja cuántica. Cobertura acumulada: 94/289 Rust preexistentes
leídos completos; se añade config.rs. No se acredita auditoría íntegra.
Sin genomas activos, trading, despliegue, reinicios ni publicación Git.

## Adenda científica XI — contrato de lectura y adaptación estadística (2026-09-24)

Detalle y fórmulas en [auditoría XI](</C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XI_2026-09-24.md>); manifiesto verificable en [artefacto XI](</C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XI_2026-09-24.json>).
Se conserva íntegramente el diagnóstico anterior; esta adenda actualiza
estados, no borra evidencia histórica ni renumera la matriz maestra.

FMT-134 tiene reparación local: genotipo y QuantumConfig comparten política
runtime-v1, y los getters del genoma reconstruyen curvas derivadas desde
genes autoritativos. Se mantienen las fórmulas/límites de la configuración
operativa; cambian las consultas directas del genoma que antes discrepaban.
runtime-v1 no es una generación ni convierte cargas individuales en snapshot.
FMT-070/042, FMT-113 y el soporte temporal de CES-008/009 siguen pendientes.

Tres hallazgos nuevos: FMT-135 corrige probabilidad NaN al construir P²;
FMT-136 corrige overflow intermedio que rompía equivalencia de unidades;
FMT-137 retira la afirmación de adaptación temporal de un cuantil acumulado.
Su mecanismo de olvido sigue sin implementarse. No se encontró consumidor
operativo del estimador: no se atribuyen estos fallos auxiliares al PnL vivo.
T34 define una propuesta de cuantiles con memoria física y evaluación causal,
contrastada con fuentes primarias; no está conectada a ejecución.

63 tests distintos aprobados, 15 nuevos y siete rojo→verde. Los tres testigos
de FMT-134 de X ahora exigen igualdad. cargo check del host pasó sin ejecutarlo.
Cobertura conservadora: 95/289 Rust preexistentes leídos completos; 194
pendientes. Sin certificación integral, rentabilidad, ventaja cuántica,
trading, genomas activos, despliegue, reinicios ni publicación Git.

## Adenda científica XII — memoria física y diagnóstico con procedencia (2026-09-24)

La [auditoría XII](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XII_2026-09-24.md>) y el [artefacto XII](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XII_2026-09-24.json>) agregan FMT-138–144, actualizan
FMT-092 y desarrollan T35. Se conserva el material anterior.

Welford ya rechaza alpha>1 y no vuelve a tratar una varianza exponencial
como suma de cuadrados; su primera observación no inventa un prior cero.
EWMA valida configuración y añade una API opt-in de memoria física con
alpha=-expm1(-dt/tau), derivada de una ODE con entrada retenida constante.
Se comprueba partición de intervalos y dt=1 ns con tau=100 años, sin
prometer resolución de mercado ni cómputo operativo nanosegundo a nanosegundo.
La nueva API no está conectada al motor ni aprende tau automáticamente.

system_health deja de afirmar salud del motor desde un átomo local,
distingue su registro de órdenes del registro operativo, consulta el modo
de cuenta mediante GET sin activarlo y etiqueta income como filas de una
respuesta, no trades ni WR. Siguen pendientes IPC, completitud contable,
salud causal por sesión/fuente y unificación de copias estadísticas legacy.

85 tests distintos aprobados: 24 nuevos, siete regresiones numéricas y tres
controles de fuente de defectos previos rojo→verde. Otro control exige el
getter nuevo; no es una reproducción adicional de bug numérico. Dos tests
nuevos caracterizan defectos legacy abiertos. cargo check del motor y del
diagnóstico pasó sin ejecutarlos. Cobertura acreditada: 101/289 Rust
preexistentes, 188 pendientes. Sin trading, exchange autenticado, despliegue,
genomas activos, reinicios ni operaciones de publicación Git.


## Adenda XIII — evidencia persistida y continuidad causal (2026-09-24)

Se conserva íntegro el atlas anterior. La [auditoría XIII](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XIII_2026-09-24.md>) y su [artefacto](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XIII_2026-09-24.json>) añaden FMT-145–154 y T36 sin renumerar la matriz histórica.

La topología raíz→estado→decisión→efecto terminal exige conservar identidad de eventos, soporte temporal y estado suficiente de cada nodo. No basta almacenar posiciones ni renombrar etiquetas. El parser vivo del warmup admitía infinitos y geometrías OHLCV inválidas; StateDb reinterpretaba cualquier etiqueta desconocida como swing; HistoryStore podía reconstruir datos, omitir corrupción y cambiar de destino ante un fallo de apertura. Esos contratos locales se corrigieron con pruebas.

Siguen abiertos el esquema multifuente/multiescala, la recuperación del estado aprendido, la integridad completa de checkpoints, la fase de entrenamiento vacía y los datos/timestamps de WalStorage. Se preservan los lectores de etiquetas legacy para no borrar historia; no representan aprobación de motores separados.

T36 propone comparar trazas con y sin fallo bajo idénticos eventos/políticas, incluyendo deduplicación de efectos terminales. El contraste de fuentes primarias mediante Firecrawl orientó ese protocolo, no añadió una infraestructura nueva ni demostró beneficio económico.

Verificación: 28 tests distintos pasan, 22 nuevos; doce reproducciones rojo→verde. Tres tests son caracterizaciones que confirman defectos aún abiertos. Compilación del binario sin ejecución y controles de formato pasan. Cobertura: 107/289 Rust preexistentes leídos completos, 182 pendientes. No certificación integral, migración activa, trading, promoción de genomas ni publicación Git.

## Adenda XIV — consenso único, contratos de evidencia y adaptación causal (2026-09-24)

Ampliación, no sustitución, del atlas ni de los 305 puntos históricos. Referencia: [informe científico XIV](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XIV_2026-09-24.md>) y [artefacto verificable XIV](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XIV_2026-09-24.json>).

La unidad conceptual sigue siendo una decisión sobre evidencia multivariante y soporte temporal declarado. Las APIs legacy de consenso ahora evalúan una sola vez el ensamble continuo y no aplican una preferencia 1,20 a la etiqueta swing. La firma dual devuelve copias de esa decisión por compatibilidad; no acredita dos expertos independientes. El núcleo observado ya llamaba directamente al consenso continuo: no se atribuye a producción una duplicación exclusiva de los adaptadores.

| Nexo del grafo vivo | Qué se corrigió o verificó | Frontera abierta |
| --- | --- | --- |
| Aprobación → fase → admisión | FMT-155: sin autorización automática por esperar ticks; revocación efectiva en la consulta | Certificado versionado, DemoVerify real y chequeo final de despacho |
| Capital/slots → guard de riesgo | FMT-093: rechazo de capital/configuración/márgenes no válidos | Snapshot, reservas atómicas, nocional/delta y fórmula económica del límite |
| Init → registro → voto | FMT-157: init fallido no inserta estrategia | Transacción del registro y conjunto de componentes obligatorios |
| Adaptadores → decisión | FMT-156: evaluación única y sin sesgo nominal swing | Deduplicación global por evento y snapshot causal |
| ATR → compresión | FMT-160: cero/cero y negativos ya no producen squeeze casi máximo | Productor de escalas sintéticas y fallbacks del registro |
| Mach → heurística acotada | FMT-161: identidad con cuadrado recíproco evita inf/inf para entradas finitas | Unidades de la entrada y justificación física |
| Genoma/modelo → voto | FMT-158/159: identificadas variable Hawkes equivocada, base ausente y genes de helpers no usados en voto | Integración coordinada con el núcleo protegido |
| Targets → telemetría → fitness | FMT-162–164: caracterizaciones y contrato documentados | TP que contrae base, extensión con pérdidas y defaults sin calidad |

T37 propone expertos con abstención/disponibilidad, masa conservada sobre log(tau) y etiquetas causales maduradas. Firecrawl Research Index ayudó a contrastar hipótesis de AdaNormalHedge y aprendizaje con feedback retardado. La integral descrita en XIV sólo agrega predicciones del mismo target; no convierte probabilidades de eventos distintos en una probabilidad conjunta. Una implementación finita necesita soporte, cuadratura, error y política de abstención. T37 no fue desplegada ni se atribuyó ventaja económica/cuántica.

Validación XIV: 84 tests distintos pasan; 21 nuevos; once reproducciones rojo→verde. Cinco tests diagnósticos confirman defectos que continúan abiertos, incluido FMT-023. Compilación sin ejecución y formato pasan. Cobertura acreditada: 114/289 Rust preexistentes completos, 175 pendientes, siete lecturas nuevas. No auditoría integral, promoción de genomas, trading ni publicación Git.

## Adenda XV — volatilidad como estado espectral, no etiqueta universal (2026-09-24)

Continuación aditiva: [informe XV](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XV_2026-09-24.md>) y [artefacto verificable XV](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XV_2026-09-24.json>). La matriz histórica y todos los análisis previos permanecen.

El grafo real mantiene dos contratos distintos: un régimen local llamado Continuous y un código global BTC que puede vetar cualquier entrada long. FMT-091 no se cierra con renombrar una variante. En paralelo, FMT-166 identifica regime_duration_ms y regime_atr_multiplier en configuración, mutación y persistencia, sin lectura decisoria localizada. No se borran genes ni se inventa otra fórmula para conectarlos.

FMT-165 corrige cinco mecanismos del Kalman: denominador e innovación desbordados, piso absoluto de covarianza, ruido negativo y mutación de R tras observación inválida. Las expresiones de ganancia y varianza posterior se evalúan de forma estable, con APIs fallibles y commit local de estado. FMT-002 sigue abierto en el caller: Q por evento y R proporcional a precio no se vuelven físicamente correctos con ese arreglo.

FMT-046 incorpora validación conjunta y staging reutilizable de correlación, diferenciando error/ausencia/resultado. Se declara el estimando real: media de correlaciones con una cesta que incluye al propio activo, no beta ni matriz multiactivo. FMT-140 se amplía con la copia root: sigue imputando retornos en cortes inválidos y un test diagnóstico demuestra divergencia futura.

FMT-004 recibe una API FFT V2 opt-in, con máscara de calidad, centrado previo, potencia unilateral normalizada y pruebas de Parseval/invariancias. La ruta ML continúa usando la función legacy y conserva sus defectos hasta migrar esquema y evaluar/reentrenar el modelo. V2 no representa hertz, todas las escalas ni una implementación de T38.

T38 propone estado espectral multivariante evolutivo con soporte e incertidumbre, sustentado por búsqueda de fuentes primarias mediante Firecrawl. Se distingue escala observacional de horizonte de decisión; se exige positividad matricial, medida de integración explícita y causalidad. El suavizado simétrico offline no se puede importar como feature disponible en el instante de trading.

Verificación XV: 93 tests distintos pasan, 25 nuevos; siete rojo→verde y cinco diagnósticos de fallos abiertos. Compilación sin ejecución, formato e integridad de fuentes intervenidas comprobados. Cobertura conservadora: 115/289 Rust completos, 174 pendientes; sólo src/features/correlation.rs se agrega como nueva lectura acreditada. Sin genomas activos, cuentas, despliegue ni publicación Git.

## Adenda XVI — campo multiactivo y contratos de evidencia (2026-09-24)

Se conserva el atlas anterior. Desarrollo, contraejemplos y criterios de cierre en el [informe XVI](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XVI_2026-09-24.md>) y el [artefacto XVI](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XVI_2026-09-24.json>).

T39 propone evaluar covariación asíncrona con ruido, procedencia y soporte; no se implementa una matriz de riesgo ni se declara ventaja cuántica. La investigación primaria distingue covariación realizada, predicción y cointegración. Una copia de caché no agrega evidencia; un continuo temporal no se obtiene mezclando instrumentos ni asignando valor cero a datos ausentes.

| Arista auditada | Evidencia XVI | Límite |
| --- | --- | --- |
| Scores/metilación/capital → fracciones | FMT-088: abstención y API de error; escala numérica estable | K sigue rígido; no factibilidad por venue ni cartera óptima |
| Precios/pesos → cesta → proxy OU | FMT-167/034: commit lógico y L1 estable | Tiempo físico ignorado; salto persistente bloquea |
| Tick → caché de par → StatArb | FMT-036: identidad exacta y sin replay por terceros | Solo adaptador BTCUSDT/ETHUSDT; sin edad por pata |
| Libro → maker → quote | FMT-168: sin suelo absoluto en libro válido | Fallback inválido aún cruzable y sin estado de calidad |
| Ventana → z → intención | FMT-169: abs(z)≤sqrt(n−1), umbral imposible para n=2 y 1,5 | Estrategia abierta; no hedge multípata codificado |
| Quote → process_event → host | FMT-170: _maker descartado | No equivalencia entre cómputo de quote y ejecución |
| Gen maker → libro sintético → fitness | FMT-050 reconfirmado | El candidato modifica el entorno de evaluación |

Verificación: 48 tests distintos, 22 nuevos, doce rojo→verde y seis diagnósticos de deuda abierta. Cuatro fuentes y cinco archivos de tests propios. Check del binario pasa sin ejecutarlo, con tres advertencias previas de evolution-engine. Cobertura 116/289 Rust completos, 173 pendientes; únicamente maker.rs incrementa la cobertura. No cuentas, despliegue, genomas activos ni publicación Git. Main/59a76de4 es el corte local, no una verificación remota.

## Adenda XVII — universo elegible, identidad y coste de adaptación (2026-09-24)

Se preserva el atlas previo. Desarrollo en el [informe XVII](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XVII_2026-09-24.md>) y el [artefacto XVII](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XVII_2026-09-24.json>).

La topología debe separar oportunidades, identidad estable, vigilancia de posiciones y admisión de exposición. Una representación continua por activo/tiempo/escala requiere esa raíz coherente.

| Arista | Diagnóstico XVII | Estado |
| --- | --- | --- |
| Capital/specs → admisión | FMT-171: capital, dimensiones, forced IDs y mínimos | Contratos reparados; cupo y scores legacy pendientes |
| Ticker diario → score | FMT-172: retorno no identifica variación; banda fija al 15% | Abierto; sin integración espectral |
| Feed inválido → universo | FMT-173: fallback fabricaba diez símbolos | Parser checked, sin fabricación, duplicados tratados |
| Ranking mutable → estado por ID | FMT-174: caché del host y lookup dinámico divergen | Abierto, diagnóstico aislado |
| Cantidad → lote validado | FMT-175: NaN y redondeo que aumenta cantidad | Abierto, helper sin caller operativo localizado |
| Política → elegibilidad | FMT-176: blacklist incondicional, defaults y tests duplicados | Abierto en ruta del manager |
| Publicación → specs/WS/archivos | FMT-177: sin epoch común y sin refresco con lista estable | Abierto, trazado estático |

T40 propone aprendizaje online con disponibilidad, feedback y coste de transición, contrastado con fuentes primarias mediante Firecrawl. Regret frente a un benchmark no garantiza rentabilidad. No se implementa ni se atribuye ventaja cuántica.

35 tests distintos pasan, 23 nuevos: doce rojo→verde, seis refuerzos y cinco diagnósticos abiertos. Dos fuentes y tres archivos de pruebas. Check sin ejecutar motor, formato y diff pasan. Ocho nuevas lecturas completas elevan cobertura a 124/289 Rust; 165 pendientes. Sin genomas activos, trading, despliegue ni publicación Git.

## Adenda XVIII — evidencia, unidades y posiciones por pierna (2026-09-24)

Se conserva el contenido anterior. Continúa en el [informe XVIII](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XVIII_2026-09-24.md>) y su [artefacto estructurado](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XVIII_2026-09-24.json>).

T41 distingue continuidad de la representación analítica y restricciones discretas del terminal. La cadena instrumento/epoch → observación con unidades → estimación → intención → ejecución → posición/ledger → evolución debe conservar identidad e incertidumbre. Un timestamp en nanosegundos no crea observaciones; una suma neta no identifica las piernas; una suma de comisiones sin moneda no identifica un coste.

| Arista | Hallazgo | Estado XVIII |
| --- | --- | --- |
| Cantidad → malla de lotes | FMT-175: proyección q_min+n·paso, dominio finito y límite de precisión | Reparación parcial; decimal/serialización integral pendiente |
| JSON → números contables | FMT-178: NaN/inf aceptados como strings | Frontera finita reparada, signos legítimos preservados |
| JSON → identidad/estado | FMT-179: defaults aceptan ACK vacío | Abierto; esquemas por endpoint pendientes |
| Respuesta POST → resolución | FMT-180: parse fallido y timeout tratados como no ambiguos | Clasificación reparada en cliente tipado; ciclo integral pendiente |
| Fills → coste | FMT-181: mezcla de monedas y alteración de rebates | Abierto; ledger firmado por moneda pendiente |
| Ticker → selección | FMT-182: segundo selector fabrica BTC y permite duplicados | Abierto, sin caller operativo localizado |
| Código API → categoría | FMT-183: -1105/-4164 mal nombrados | Nombres canónicos y aliases deprecados corregidos |
| Snapshot remoto → posición | FMT-184: hedge neteado; reversión con dirección vieja | P1 abierto, reproducido sin exchange |

50 pruebas distintas pasan, 23 nuevas: doce rojo→verde, cinco refuerzos y seis diagnósticos abiertos. Cuatro lecturas completas nuevas elevan el acumulado a 128/289 Rust; 161 pendientes. Check pasa con tres warnings preexistentes. Tres fuentes modificadas, sin ejecución del motor ni genomas operativos, cuentas, deploy, commit o push. No se certifica rentabilidad ni todo el proyecto.

## Adenda XIX — evidencia causal y significado de la separación espectral (2026-09-24)

Continuación en el [informe XIX](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XIX_2026-09-24.md>) y el [artefacto XIX](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XIX_2026-09-24.json>). Se preserva el atlas anterior.

FMT-184 queda parcialmente contenido: conciliación con resultado tipado rechaza ajustes no representables, sin netear un hedge a plano ni asignar el total remoto a uno de varios slots. Ausencia no es cero y cantidades pequeñas no se borran por epsilon absoluto. El wrapper registra motivos, pero el host aún no los consume como gate global de admisión. Conservar estado no equivale a representar la cuenta completa.

| Arista | Diagnóstico XIX | Estado |
| --- | --- | --- |
| REST/WS → snapshot acumulado | FMT-185: acumulado nuevo con precio/nocional viejo y terminales intercambiables | Reparación parcial; igual cobertura y tiempo causal pendientes |
| Escala → admisión | FMT-186: corte fijo0,80 interpretado como ortogonalidad; fallback30s | Abierto, con contraejemplo matemático |
| Entrada/cierre → dataset | FMT-187: tensor54D sin propiedad generacional | Abierto, reproducción secuencial y trazado de productores |
| Campos atómicos → snapshot | FMT-188: escritores públicos no versionados | Abierto; estrés open/close no cubre todos los writers |
| Demora → expiración | FMT-189: limpieza local fabrica EXPIRED | Abierto auxiliar, sin caller operativo localizado |

T42 separa unión de fills, selección de snapshots y contratos generacionales. Para kernels exponenciales normalizados, el producto interno es sech(Δlogτ/2); con separación0,80 supera0,92. No es ortogonalidad, ni prueba de correlación empírica de las señales del sistema. La diversidad de escalas requiere estimandos y validación, no una nueva etiqueta física.

59 pruebas distintas pasan: 26 nuevas —14 rojo→verde,5 refuerzos,6 diagnósticos abiertos,1 contraejemplo— y33 anteriores. Check/formato/diff pasan. Dos lecturas nuevas completas: position y order_registry. Cobertura130/289 Rust;159 pendientes. Dos fuentes modificadas, sin operar cuentas, promover genomas, desplegar ni publicar Git.


## Adenda XX — paridad del predictor y validación temporal (2026-09-24)

Informe ampliado: [Fundamentos XX](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XX_2026-09-24.md>). [Artefacto XX](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XX_2026-09-24.json>). Se conservan todas las teorías y matrices anteriores.

El grafo auditado es dato/label → fit → representación serializada → predictor servido → decisión → resultado atribuible. Se verificó que esa cadena cambiaba de función: árboles concatenados conservaban hijos locales, y la exportación no aplicaba shrinkage. FMT-190 corrige índices globales, hojas ponderadas y gate/diagnósticos del artefacto retenido. FMT-191 corrige el residual de regresión y−f: con suma positiva y hoja G/(H+λ), f−y aumentaba el error.

FMT-037 se actualiza, sin renumerarlo: validación de arrays, offsets, hijos, ciclos y finitud; recorrido acotado y predict_raw_checked. Se mantiene el bias-only explícito de los oráculos. Falta contrato semántico de esquema/target/tiempo y presupuesto de carga.

| Nuevo ID | Estado | Significado |
| --- | --- | --- |
| FMT-190 | Reparación local, migración pendiente | Entrenamiento y serving deben representar la misma función |
| FMT-191 | Signo local corregido | El residual debe reducir la pérdida bajo el convenio aditivo |
| FMT-192 | Abierto | Padre completo e hijo submuestreado no forman una partición coherente |
| FMT-193 | Abierto | El corte temporal requiere intervalos de labels y separación de selección/evaluación |
| FMT-194 | Abierto | max-samples no limita la cantidad ni garantiza el stride solicitado |

Revalidación de FMT-052/053/054: el dataset online mezcla semánticas; umbrales sobre clases binarias no discriminan; beneficio no equivale a dirección y accuracy de entrenamiento no acredita consumo vivo. No se cuenta esa persistencia como nuevos hallazgos.

T43 exige invariancia numérica del artefacto y una pérdida explícita antes de añadir complejidad teórica. El continuo multivariado requiere representación y error medible; no se ha implementado aquí un motor temporal universal.

Inventario aislado: 20 bosques JSON, 8 rechazados por aristas entre árboles y 12 aceptados solo estructuralmente; un JSON neuronal se clasifica aparte. Los modelos no se repararon, promovieron ni cargaron globalmente. Futuro despliegue del validador requiere plan de migración y de ausencia de modelo.

31 pruebas funcionales distintas pasan: 19 nuevas (12 rojo→verde y 7 refuerzos) y 12 anteriores. Un diagnóstico adicional inventaría modelos. Check de ambos binarios pasa; sin ejecutar motor/entrenamiento real. Lectura nueva completa del trainer: cobertura 131/289 Rust, 158 pendientes. Dos fuentes de producción intervenidas, sin publicación Git ni actividad de cuenta.


## Adenda XXI — medida estadística, ventanas de información y aprendizaje auxiliar (2026-09-24)

Continuación del paradigma de grafo vivo; no sustituye la matriz histórica ni las rondas anteriores. [Informe científico XXI](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XXI_2026-09-24.md>) · [Artefacto XXI](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XXI_2026-09-24.json>).

El nodo de aprendizaje debe conservar la identidad de su medida, objetivo y evidencia: población del ajuste → features/label/intervalo → partición causal → modelo congelado → evaluación posterior → decisión de publicación. Se corrigen FMT-192 y FMT-194 localmente: el probe solo propone umbrales y la ganancia usa todos los elementos del multiconjunto bootstrap; el stride nunca densifica la malla solicitada y un contador impone el máximo de intentos por archivo.

FMT-193 pasa a reparación parcial: ventanas cerradas de etiqueta, purga por fin de información y test posterior obligatorio para --promote. Sin test solo se produce candidato de investigación. El gate no es una prueba estadística de alpha ni controla reutilización del holdout entre experimentos. La media baseline procede de train; el resultado de regresión se denomina skill relativo, no R² convencional del test.

| Hallazgo nuevo | Arista auditada | Estado |
|---|---|---|
| FMT-195 | ticks → feature_exporter → CSV | Abierto: 500 ticks rotulados 5m, TP inalcanzables y canales macro constantes |
| FMT-196 | OHLCV → feature_validator → veredicto | Abierto: proxy de cierre llamado OBI, ambigüedad TP/SL optimista y ausencia de inferencia estadística |
| FMT-197 | CSV → parser neuronal → normalización | Abierto: dimensiones variables, targets no finitos y ceros de imputación no declarada |
| FMT-198 | ajuste neuronal → destino de modelo | Abierto: sin separación temporal/evaluación independiente ni candidato diferenciado |
| FMT-199 | eventos → targets vol/volu | Abierto: RMS y profundidad por evento no equivalen a magnitudes por reloj |

FMT-028 se reconfirma: el complemento de éxito condicionado del largo no certifica éxito del corto; los timeouts descartados cambian la pregunta probabilística. No se ha demostrado activación productiva de los binarios auxiliares ni que el CSV alimente train_forest.

T44 añade contratos de medida/intervalos/evidencia al diseño continuo multivariante. Un horizonte parametrizable no equivale a cobertura científica entre 1 ns y 100 años. La aproximación necesita soporte observado, incertidumbre, error numérico y presupuesto. No se integraron ecuaciones físicas/cuánticas sin variables identificadas y evaluación falsable.

Verificación: 37 pruebas funcionales distintas pasan, 18 nuevas —4 rojo→verde— y 19 anteriores. Check de god_engine/train_forest pasa; no se ejecutó motor ni entrenamiento real. Tres lecturas nuevas completas elevan cobertura a 134/289 Rust preexistentes; 155 siguen pendientes. Solo train_forest cambia como fuente en XXI; se preservan siete fuentes de referencia y 41 modelos por hash. Sin commit/push/merge/fetch, cambio de cuentas, genomas o promoción.

## Adenda XXII — identidad neuronal, coordenadas y verificación diagnóstica (2026-09-24)

Continuación del grafo vivo sin sustituir el historial ni la matriz de 305 puntos. [Informe XXII](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XXII_2026-09-24.md>) · [Artefacto XXII](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XXII_2026-09-24.json>).

La identidad de un nodo predictivo incluye pesos, transformación, estado, esquema, unidad, soporte y autorización. Preparar buffers no debe podar coeficientes; congelar no debe aprender con la evaluación; una ausencia no es una probabilidad neutral. El flujo neuronal auditado es CSV legacy → parser estricto → TRAIN/Scaler/Adam → candidato de investigación → evaluación y autorización pendientes → host → decisión → terminal. No se implementa aquí la totalidad de ese ciclo.

| Hallazgo | Estado actualizado |
|---|---|
| FMT-073 | Reparación local: freeze no calienta ni modifica estadísticas; ambas APIs comparten selección de coordenadas; estado frío sin Scaler se abstiene |
| FMT-074 | Parcial: finitud/formas/salida única, error de Scaler, propagación de no finitos y rollback de fit; faltan manifiesto, soporte y política hasta el ensemble |
| FMT-076 | Local: buffers no podan; is_subnormal reemplaza el corte arbitrario 1e−7; no se recuperan modelos ya podados |
| FMT-197 | Local: CSV exacto 54D con target binario finito, sin imputación ni omisiones silenciosas; target_5m no se valida semánticamente |
| FMT-198 | Contención parcial: salida solo en artifacts/training, exclusiva y diferenciada; sin holdout causal ni autorización de promoción |
| FMT-200, nuevo | Abierto: sonda supuestamente exacta construye 44D frente a 48D del bosque; sus causas de None son incompletas y la carga puede escribir caché |
| FMT-201, nuevo | Abierto: diagnóstico consume cabecera no verificada, castea bytes sin contrato portable de alineación y compara transformaciones distintas |

La doble normalización Scaler+Welford en la inferencia ordinaria no se confirmó: son ramas excluyentes. Se amplía T25/T44, sin crear otra teoría duplicada. El transporte afín de coordenadas tiene límites por clipping, escalas cero y estado del optimizador; no se aplicó silenciosamente. El count mínimo 2 establece varianza algebraicamente definida, no confianza estadística.

En el host se confirma la lectura estática de models/DarkAlpha_BTCUSDT.json. Fuera de BTC no se ejecuta la red, pero se remite Some(0.5) al ensemble: no es ausencia de evidencia a nivel de interfaz. No se probó activación de un modelo particular en un proceso actual ni se cuantificó el efecto del voto neutral.

Verificación: 94 tests distintos pasan, 26 nuevos; 93 funcionales y un control de tiempo medio, sin p99 productivo. Doce reprodujeron fallos antes del parche. Check de god_engine/train_dark_alpha/train_forest/auto_trainer_daemon pasa; no se ejecutaron motores, entrenamientos reales ni los diagnósticos que escriben caché. Se mantienen los tres warnings previos de evolution-engine.

Cobertura acumulada: 136/289 Rust preexistentes completos, 153 pendientes. Nuevos: ml_path_probe.rs (74 líneas) y test_ml.rs (247). Se releen núcleo DarkAlpha y trainer sin duplicar cobertura. Dos fuentes modificadas, un test nuevo; ocho fuentes protegidas y 41 modelos conservan hashes. Sin commit/push/merge/fetch ni promoción, cuentas o genomas modificados. La auditoría global y la paridad backtest/demo/live permanecen abiertas.

## Adenda XXIII — procedencia, primeros pasos y superficie temporal de etiquetas (2026-09-24)

[Informe XXIII](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XXIII_2026-09-24.md>) · [Artefacto XXIII](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XXIII_2026-09-24.json>). Se preservan historial y matriz de 305 puntos.

FMT-195 pasa a reparación parcial mediante JSONL v2 de investigación: horizontes de reloj explícitos, barreras por lado independientes, no-hit y censura sin descarte, intervalos informacionales, hash de input, procedencia y ausencia de features. La salida no reinterpreta ni sobrescribe el CSV anterior y no se incorpora silenciosamente al trainer legacy. No hay nuevo modelo entrenado ni promoción.

| ID | Arista | Estado |
|---|---|---|
| FMT-069 | bytes→replay | Nueva ruta LE validada sin casts; los lectores anteriores siguen pendientes |
| FMT-202 | aggTrades→pseudo-libro→REAL | Contenido derivado no acredita L2; v2 lo declara; productores históricos abiertos, relacionado con D-233/D-721 |
| FMT-203 | vela final→subticks anteriores | Abierto: volumen final, spread/fallback dependiente de futuro y ubicación intrabar no identificada; residual de D-691/D-722 |
| FMT-204 | cantidades→maker→flujo | El productor y bq>aq invierten el indicador original; v2 omite canal 6, otras rutas pendientes |
| FMT-205 | adquisición→tape | Abierto: éxito sin registros, omisiones, cobertura no registrada y orden mensual no estable |

La comparación opuesta del diagnóstico FMT-201 se reinterpreta con la nueva traza: para este generador concreto aq>bq coincide con buyer-maker, bq>aq lo invierte. No se convierte por ello al diagnóstico en correcto ni a cualquier profundidad en trade observado. FMT-200/201 no se modifican ni ejecutan.

T17/T44 se concretan como contrato de evidencia; T24 aporta pruebas metamórficas. Las probabilidades de primer paso por horizonte y la dependencia multiactivo aún deben estimarse/calibrarse. El horizonte cubierto solo indica que el tape alcanza ese tiempo; no prueba ausencia de huecos o barreras intermedias. Una confirmación posterior amplía information_end y debe considerarse en la purga.

Validación: 70 tests distintos pasan, 34 nuevos. Check de cuatro binarios y ayuda del exportador pasan; sin corpus real, descargas o entrenamiento. Dos lecturas nuevas —binance_vision_sync, 483 líneas; parquet_to_bin, 199— elevan cobertura a 138/289 Rust; 151 pendientes. Dos fuentes Rust modificadas, dos añadidas y regla de caché pública en .gitignore. Doce fuentes protegidas y 41 modelos mantienen hashes. Sin operaciones Git remotas, genomas, órdenes, reinicios ni promoción.

## Adenda XXIV — estado por activo, transiciones y filtros (2026-09-25)

Informe: [Auditoría XXIV](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XXIV_2026-09-25.md>). Artefacto: [JSON XXIV](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XXIV_2026-09-25.json>). Se conserva todo el contenido previo.

Nueva lectura completa: StatefulEngine, 1.390 líneas iniciales. HawkesProcessEngine se releyó, pero ya estaba contado. Cobertura acumulada 139/289 fuentes Rust preexistentes; 150 pendientes. Los 1.119 archivos base y 24 manifiestos no se declaran auditados íntegramente.

| Hallazgo | Evidencia causal | Resultado |
|---|---|---|
| FMT-206 | Warmup inicializa last_price, pero EMA/Kalman de ticks seguían sin siembra | Corregida transición por primer tick aceptado |
| FMT-207 | Reset dejaba last_inst_v, dir_velocity y last_trade_is_sell antiguos | Campos limpiados; separar riesgo/modelo/observación sigue pendiente |
| FMT-208 | Ticks atrasados o volumen/OHLCV inválidos alteraban prefijo | APIs fallibles y exportador v2 propagan rechazo; host legacy no |
| FMT-209 | clamp con min>max, wrap de reloj y tau NaN admitido | Guardas numéricas reparadas, sin reducir mínimo del caller |
| FMT-210 | Hawkes admite impulso tardío, confunde t=0 y devuelve ratio neutro en lote vacío | Abierto, reproducido por tres diagnósticos |

El nombre Continuous no elimina las bandas a 60.000/1.800.000 ms, EMA por eventos ni FFT de muestras heterogéneas. Un diagnóstico confirma distinta admisión entre tau=59.999 y 60.000 ms con el mismo historial. Otro confirma que la vela se ancla al evento de cruce, no a una malla regular. FMT-004 se amplía; FMT-001/002 permanecen abiertos en signo/unidades y Kalman/Δt.

T01/T24 se concretan como contratos de transición y pruebas metamórficas. T03 exige eventos identificados, kernels/marcas y compensadores; las fuentes primarias consultadas mediante Firecrawl sustentan esos requisitos, no rentabilidad del módulo. No se añade una ecuación cuántica ni un kernel aprendido sin calibración.

Resultado: 103 tests funcionales y 5 diagnósticos abiertos pasan; 23 funcionales y 5 diagnósticos son nuevos. Quince regresiones distintas se observaron fallar antes de corregirse. Un diagnóstico verde no es un fallo cerrado. Check de cuatro binarios pasa; sin suite completa, PnL ni latencia productiva. Dos fuentes modificadas, dos suites nuevas, cinco referencias y 41 modelos preservados. Sin entrenamiento/promoción, genomas, operaciones Git remotas, órdenes ni reinicios.

## Adenda XXV — 2026-09-25: qué justifica cada veto

[Informe científico XXV](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XXV_2026-09-25.md>) y [artefacto JSON](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XXV_2026-09-25.json>). Extensión, no sustitución de la matriz histórica.

Se reparó el contrato de identidad TP/SL entre filtro EV y orden devuelta; se validan dominios numéricos antes de clamp/comparaciones; incertidumbre de ejecución ya no se cuenta como rechazo confirmado de brackets. Se mantienen políticas de drawdown, margen, fricción y promoción. Nuevos contadores: entrada_invalida y geometria_invalida, sin desplazar los trece índices anteriores.

| ID | Resultado verificable | Estado restante |
|---|---|---|
| FMT-211 | TP/SL explícitos se resuelven antes de EV y se reutilizan | Calibración de p y pérdida sobre cantidad final abiertas |
| FMT-212 | Inf de confianza, pico NaN e intervalo Kelly inválido se rechazan | Falta snapshot/versionado de todo el estado |
| FMT-213 | AMBIGUOUS, -1006/-1007 y números incidentales no acreditan rechazo | Falta outcome tipado y evidencia por pierna/solicitud |
| FMT-214 | Diagnóstico devuelve leverage 4,121212121212122 tras floor inicial | Proyección conjunta de leverage/lotes/margen pendiente |
| FMT-215 | Helpers admiten desconocidos o amplían máximos uno a dos | Contratos auxiliares y política por definir; no todos tienen caller vivo |
| FMT-216 | Semilla 199/tasa 0,5 deja banda admisible vacía | Veto justificado; reparar contrato del generador/test, no debilitar promoción |

El espectro sigue truncado en un lector a 1.000..43.200.000 ms; el tipo de intención no representa nanosegundos. Coherencia/entropía modulan un umbral heurístico, no prueban certidumbre física. La concentración por número de posiciones y Crash global no sustituyen riesgo conjunto multiactivo. Los comentarios se precisaron sin añadir una teoría ornamental.

Verificación XXV: 16 contratos nuevos, seis diagnósticos nuevos, nueve rojo→verde. En las ejecuciones ampliadas se observaron 164 contratos funcionales sin fallo, diez diagnósticos abiertos que pasan al reproducir deuda y un test preexistente de promoción con fallo y pase aleatorios: no se declara suite global verde. Check de cuatro binarios pasa. Cobertura 141/289 Rust; 148 pendientes. Los 41 modelos se preservan; sin trading, promoción, reinicios ni operaciones Git remotas.

## Adenda XXVI — acción ejecutable y auditoría de vetos (2026-09-25)

Se conserva íntegra la evidencia anterior. Véanse [informe XXVI](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XXVI_2026-09-25.md>) y [artefacto verificable](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XXVI_2026-09-25.json>).

FMT-214 queda parcialmente contenido: las reasignaciones intervenidas respetan leverage entero y techo genómico; el ejecutor deja de truncar. El rescate original devolvía 4,121212121212122 incluso con techo genómico 1. Continúan pendientes el presupuesto de pérdida final y la factibilidad conjunta de margen/lote/precio.

FMT-217 corrige el veto de coste dependiente de ramas: cL_final se comprueba sobre cada acción final de la ruta de riesgo. El presupuesto efectivo sigue interpolando max_fee_pct hacia 0,035 en el extremo micro; no se presenta ese literal como teoría calibrada ni como techo duro genómico universal.

FMT-218 refuerza el dominio numérico del payload y la admisión básica previa al éxito paper. Una compra maker con referencia menor que el primer tick ya no se desplaza por encima de ella en este constructor. La prueba inicial confundía ese caso con precio cero; se corrigió su explicación. MARKET no exige un tick de precio límite que no utiliza. Pasividad local completa y otras rutas no quedan certificadas.

FMT-175 se amplía con dos reproducciones en el ejecutor: N=5,10255 termina en 3 después del lote y q=0,3999999999 termina en 0,4 por tolerancia. FMT-219 reproduce una venta maker cuyo valor límite pasa de 40 a 44 sin nuevo presupuesto. Los tres tests pasan al reproducir deuda; no son reparaciones. Se requiere un plan de acción común previo a efectos de cuenta, con valoración, filtros, coste y presupuesto conservados.

El informe explica la diferencia entre espectro continuo y actuador discreto, incluye el grafo raíz→decisión→proyección→terminal→aprendizaje y separa vetos numéricos, económicos, de datos y del venue. Persisten el recorte temporal, las categorías globales de régimen y los problemas de dependencia/snapshot; no se promete omnisciencia ni ventaja cuántica.

XXVI añade 14 contratos funcionales y tres diagnósticos; convierte un diagnóstico anterior en regresión. Ocho reproducciones rojo→verde contabilizadas conservadoramente. Ejecuciones distintas: 116 funcionales y 11 diagnósticos abiertos, 127 pases. Check offline de cuatro binarios pasa con warnings previos. Cobertura 142/289 Rust, 147 pendientes; 41 modelos preservados. Sin órdenes, promoción, entrenamiento, reinicios ni commit/push/merge/fetch. La prueba de promoción inestable de XXV sigue abierta.

## Adenda XXVII — consumidor real, capacidades y confirmación (2026-09-25)

[Informe XXVII](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XXVII_2026-09-25.md>) y [artefacto XXVII](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XXVII_2026-09-25.json>). El histórico se conserva. La ruta de entrada del host invocaba raw/maker/iceberg, no execute_order/build_payload: no se extrapolan las reparaciones XXVI a consumidores que las omitían.

FMT-220 contiene la rama iceberg: campos nombrados evitan intercambiar precio y cantidad visible; se elimina el ID fijo del host. La API se conserva, pero el adaptador USD-M rechaza explícitamente iceberg nativo no respaldado por el contrato de /fapi/v1/order, también en paper, antes de efectos de configuración. No se sustituye por una orden visible o MARKET. Un algoritmo de órdenes hijas requiere implementación y validación aparte.

FMT-221 conecta un despachador común al host: configurar leverage, incluido 1×, confirmar símbolo/valor y sólo entonces enviar. Un fallo impide el envío; una respuesta HTTP exitosa por sí sola ya no confirma configuración. Se preserva AMBIGUOUS de envío. Falta coherencia concurrente por cuenta/símbolo, aplicación de tiers y tratamiento de cambios de configuración inciertos.

FMT-218 se amplía a raw_qty: una cantidad NaN producida por paso subnormal ya no pasa por la comparación contra cero ni se declara éxito paper. FMT-113 permanece: cambiar L con q fija puede modificar margen sin modificar pérdida nominal N·d. También permanecen los tres diagnósticos de FMT-175/219.

Nuevo FMT-222, P1 abierto: consulta de reconciliación fallida retorna adopted=true y después exchange_confirmed=true. Preservar un estado incierto no lo confirma. Identidad por símbolo y rollback de todos los slots requieren un contrato por intención/reserva; no se declara reparado con una prueba de helper.

Diez contratos nuevos, dos rojo→verde observados. Ejecutados 126 funcionales y 11 diagnósticos abiertos, 137 pases distintos; check de cuatro binarios pasa. Cobertura 143/289 Rust, 146 pendientes. Un módulo nuevo y una suite; cuatro fuentes existentes intervenidas. Sin trading, cuentas, promoción, entrenamiento, reinicios ni publicación Git. No se certifica paridad productiva ni espectro universal completado.

## Adenda XXVIII — evidencia, reconciliación y recompensa (2026-09-25)

[Informe XXVIII](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XXVIII_2026-09-25.md>) y [artefacto XXVIII](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XXVIII_2026-09-25.json>). Se conserva el histórico y su numeración. Nuevos FMT-223…226; FMT-222 recibe contención, no cierre global.

FMT-222: la rama ambigua ya no acredita una entrada por presencia agregada del símbolo, ni por consulta fallida. Tampoco hace rollback por no encontrar posición. Conserva estado, marca protección dirty y consulta la intención con el mismo ejecutor del despacho. No certifica fill ni resolución de hijos; faltan ledger, reservas por identidad y generación. La rama Ok y el rollback genérico siguen pendientes.

FMT-223: consulta fallida, incluidos -2013 y errores de identidad/dominio, permanece no concluyente. GET exige campos explícitos e identidad antes de escribir el registro; terminal cero no borra fills ya conocidos. Accepted aún incluye NEW y no significa ejecución. La validación WS, los motivos estructurados y la transacción de registro siguen abiertos.

FMT-224: el parser activo deja de convertir errores/faltantes en cartera vacía y elimina el corte arbitrario de exposición 1e-8. Valida filas y piernas completas; no colapsa hedge balanceado. El arranque todavía usa unwrap_or_default y la segunda ruta PositionRiskEntry conserva defaults: la protección del consumidor global no queda certificada.

FMT-225, P1 abierto: en el core, exchange_confirmed protege la métrica de PnL pero no todas las actualizaciones de capital, aprendizaje espectral, ensamble y mmap. Se documenta el flujo y su impacto causal; no se añade un gate global que confunda backtest con live. Requiere OutcomeEvidence por procedencia, intención, fills y generación.

FMT-226: maker-chase ya no envía MARKET completa ante cualquier error inicial, ni remanente cuando la consulta sigue activa. Se exige evidencia terminal. Pendientes: fills causales, agregado padre/hijos, fallback tipado y paridad del transporte; retener un intento incierto puede reducir disponibilidad y no debe etiquetarse como fracaso predictivo.

16 pruebas nuevas, cinco reproducciones rojo→verde; 66 funcionales y 11 diagnósticos abiertos, 77 pases distintos. Check offline de cuatro binarios pasa con warnings previos. Cobertura permanece 143/289 Rust, 146 pendientes; los nuevos archivos no inflan el inventario histórico. 41 modelos verificados sin cambios. Sin cuentas, órdenes, entrenamiento/promoción, reinicios ni commit/push/merge/fetch. La continuidad temporal y multiactivo no sustituye identidad de ejecución ni evidencia de recompensa.

## Adenda XXIX — procedencia, horizonte y pérdida de evidencia (2026-09-25)

[Informe XXIX](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XXIX_2026-09-25.md>) y [artefacto XXIX](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XXIX_2026-09-25.json>). Se añaden FMT-227…232 y se amplía FMT-225, sin sustituir la matriz histórica ni sus conclusiones.

FMT-225: una entrada no confirmada ya no modifica capital ni ingresa el cierre como evidencia de aprendizaje en el contexto del host. Se conserva la propuesta de salida. Es contención: el slot/margen todavía cambian antes del gate, y confirmar la entrada no acredita el fill de salida. Las estimaciones legacy de entradas confirmadas siguen sin ser ledger de liquidaciones.

FMT-227: el constructor común pasa a simulación aislada para las salidas de cierre. Conserva aprendizaje y capital de su instancia, pero no publica el resultado por CSV/mmap/trauma. El host selecciona procedencia explícita; el constructor inmune diferido no crea directorios. No se certifica aislamiento general, causalidad de todas las features ni separación de todas las cuentas/proveedores.

FMT-228: eliminada una de dos ingestiones del mismo cierre. Con α=0,05, el peso efectivo de la doble EMA era 0,0975; vuelve a 0,05. El contraste con la fuente primaria PPO distingue múltiples épocas sobre datos fijados de doble conteo de evidencia. FMT-006 permanece abierto: recortar pesos no implementa el objetivo PPO.

FMT-229: Kelly leía entry_tau_ms después de que close_with_fee lo pusiera a cero, recurriendo a 30s. Se reutiliza el horizonte continuo anterior al cierre. La regresión detectaba 0,056671945 frente a 0,086238142 del horizonte correcto; ahora coincide. No se certifica optimalidad de la fórmula de riesgo ni soporte end-to-end de nanosegundos.

FMT-230, abierto reproducido: el lector mmap avanza su cursor sobre un slot en progreso y no recupera su commit posterior. FMT-231, abierto estático: interpreta cabecera antes de validar tamaño/formato. FMT-232, abierto estático: kill-switch devuelve antes de gestionar posiciones; requiere política por causa y autoridad defensiva, no eliminar el veto sin sustitución.

Diez pruebas nuevas, cuatro rojo→verde. Resultado distinto: 82 funcionales + 6 diagnósticos abiertos = 88 pases, una ignorada. Check de cuatro binarios pasa. Nueva lectura integral de mmap_bus.rs: cobertura 144/289 Rust, 145 pendientes. Se verifican 41 modelos sin cambios. Sin cuentas, órdenes, promoción/entrenamiento operativos, reinicios ni publicación Git. La auditoría completa del proyecto sigue pendiente.

## Adenda XXX — admisión, posiciones y vetos con evidencia reproducible (2026-09-25)

[Informe XXX](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XXX_2026-09-25.md>) y [artefacto XXX](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XXX_2026-09-25.json>). Se agregan FMT-233…236 y se actualizan FMT-230/231/232; no se sustituye ni renumera la matriz histórica.

FMT-231: un test aislado sobre archivo vacío terminó en STATUS_ACCESS_VIOLATION; ahora se valida longitud antes de mapear y antes de interpretar la cabecera. Ausencia/error deja de ser un lote vacío válido. El daemon registra interrupción y recuperación sin certificar continuidad. Formato, truncamiento posterior, protocolo concurrente y pérdidas siguen abiertos.

FMT-233: el writer rechaza un archivo existente no vacío e incompleto y preserva sus bytes; ya no lo rellena de ceros como reparación implícita. Sólo un archivo vacío/nuevo se inicializa. No se acredita inicialización concurrente ni recuperación versionada.

FMT-234: lectura de ownership con Result, SQLite sólo lectura y todas las etiquetas de procedencia. No imputa errores como cero ni omite etiquetas desconocidas; el adaptador dual devuelve None si no puede representar todo. No se encontraron consumidores operativos de PositionLedger: corrección auxiliar, no reparación del ledger real.

FMT-235, abierto: push_event descarta snapshot absoluto cero y exposición 1e−13; el writer además aplica epsilon 1e−8. No confundir inventario real con tamaño mínimo de orden. Dos diagnósticos confirman el rechazo antes del writer.

FMT-236, abierto: cola llena descarta un evento válido sin ACK; errores SQL/commit no llegan al productor. Saturación reproducida en canal propio, modos de persistencia por inspección. Encolado, persistido y conciliado necesitan estados e identidad separados.

FMT-230 sigue perdiendo el frame reservado que se publica después. FMT-232 pasa de sospecha estática a reproducción sobre el core: el flag global también bloquea una propuesta local de cierre por stop. No se retiró el veto ni se afirmó ausencia de brackets externos.

35 pases distintos: 30 funcionales/compatibilidad y cinco diagnósticos abiertos; 22 tests nuevos. Cinco aserciones rojo→verde y un aborto de proceso de test contenido por la guarda. Check offline de cuatro binarios pasa. Cobertura 145/289 Rust preexistentes, 144 pendientes; nueva lectura ledger.rs (294 líneas en HEAD). 41 modelos sin cambios. Sin cuentas, órdenes, entrenamiento/promoción operativos, reinicios ni publicación Git. Continuidad temporal no equivale a evidencia infinita ni a procesamiento exhaustivo nanosegundo a nanosegundo.

## Adenda XXXI — consejo, vetos y atribución causal de aprendizaje (2026-09-25)

[Informe XXXI](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XXXI_2026-09-25.md>) y [artefacto XXXI](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XXXI_2026-09-25.json>). Se añaden FMT-237…245 conservando matriz e informes históricos. El consejo está conectado al core; las reparaciones no se presentan como arreglos de toda la ejecución real.

FMT-237: deliberación valida los nueve flotantes antes omitidos, dominios, parámetros, win-rate y multiplicadores externos. Error de integridad deja de convertirse en permiso por clamp/default. No acredita freshness ni elimina imputaciones del productor.

FMT-238: la excepción de breakout exige signo de OBI alineado con la entrada; magnitud grande contraria ya no exonera el veto. Permanecen pendientes calibración de umbrales y la interpretación causal del proxy VPIN.

FMT-239: el umbral configurable de cascada gobierna activación y protección contra override, no sólo una rama posterior a un veto hardcoded. Default conservado; configuración inválida rechazada.

FMT-240: Teleonomia compara ML contra la base del modelo. En el fixture p=base=0,3 y espectro cero, la falsa señal −0,154448 se vuelve cero. No se acredita utilidad económica por corregir el centro.

FMT-241: abstención y moduladores enmascarados no cuentan como ensayos/éxitos; la expulsión de la ventana usa la misma semántica. FMT-242: retorno neto de una operación se atribuye con el lado ejecutado; el core ya no premia la señal alcista por un corto ganador. No equivale a precisión contrafactual.

FMT-243, abierto: n agregado no corresponde necesariamente al wr del activo; extracción usa wr crudo y deliberación shrinkage. k=8 implica Beta(4,4) bajo ensayos compatibles, no Beta(1,1); se corrige la descripción sin afirmar posterior calibrado.

FMT-244, abierto: una sola raíz direccional OBI puede producir consenso 100% por el denominador variable y su voto derivado. FMT-245, abierto: la coordenada s colapsa 1ns/30s y 12h/cien años a extremos iguales; continuo acotado no es soporte universal.

23 tests nuevos, 13 contratos rojo→verde válidos. Resultado distinto 37 funcionales + 4 diagnósticos abiertos = 41 pases; check offline de cuatro binarios pasa. Nueva lectura integral consejo_seniors.rs (1.521 líneas base): cobertura 146/289 Rust, 143 pendientes. 41 modelos sin cambios. El error inicial de disco no fue un test ejecutado; reintento serial sin limpiar cachés. Sin cuentas, órdenes, promoción/entrenamiento operativos, reinicios ni publicación Git.

## Adenda XXXII — identidad de decisión, generación y entrega de evidencia al veto

[Informe detallado XXXII](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XXXII_2026-09-25.md>) · [Artefacto estructurado XXXII](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XXXII_2026-09-25.json>). Continuación aditiva de XXXI; no sustituye la matriz histórica de 305 puntos ni certifica todo el proyecto.

FMT-243, avance parcial: una sola deliberación devuelve decisión, opiniones, votos elegibles, wr efectivo y n observado. El core deja de extraer con wr crudo antes de volver a evaluar. El tracker aporta n/pesos bajo un mismo read-lock. La mezcla n global/wr por activo sigue ABIERTA y tiene diagnóstico reproducible.

FMT-246: roles únicos, pesos/señales por identidad y validación de opiniones/overflow. Reordenar agentes ya no reasigna pesos; duplicar rol no crea evidencia. Peso o confianza cero significan capacidad nula, no integridad inválida.

FMT-247: vínculo local símbolo/coin/slot/generación/lado creado sólo tras apertura válida y consumido una vez. El array legacy deja de entrenar. Sin vínculo coherente se omite crédito al consejo, no se veta por ello el cierre. La apertura respeta el bool de publicación; validación previa y compensación local conservan el orden reserva→publicación. No equivale a ledger de fills ni a concurrencia integral certificada.

FMT-248: ratio de flujo escalado por máximo; elimina overflow de B+S y umbral dependiente de unidades. Helper sin caller operativo localizado. FMT-249 ABIERTO: macro consume la severidad y el consejo puede encontrar cero; canal global sin activo/tiempo. Documentación corregida: 10k USD→2/3 y100k→5/6, no probabilidad/percentil.

FMT-250 ABIERTO: offset sin calidad/frescura y RTT con reloj de pared. FMT-251 ABIERTO latente: wr/EV no acreditan promoción y execute_swap sólo cambia un booleano. FMT-252: descripción honesta del helper de latencia; wrap extremo sigue ABIERTO, sin caller operativo localizado.

23 tests nuevos; ocho aserciones rojo→verde. Selección:60 funcionales/compatibilidad +6 diagnósticos abiertos =66 pases únicos. Check offline de cuatro binarios pasa, tres warnings previos. Cinco lecturas completas nuevas: ntp, hot_swap de execution, order_flow_aggregator, latency_accelerator y liquidation_feed; cobertura151/289 Rust,138 pendientes. 41 modelos preservados. Sin cuentas/órdenes, promoción operativa, reinicios o publicación Git.

## Adenda XXXIII — liquidaciones por símbolo, tiempo causal y sentido del veto

[Informe detallado XXXIII](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XXXIII_2026-09-25.md>) · [Artefacto estructurado XXXIII](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XXXIII_2026-09-25.json>). Continuación aditiva; no sustituye la matriz histórica ni certifica todos los archivos.

FMT-249: reparación local de la entrega. El buffer global sin símbolo/tiempo podía contaminar el siguiente activo y vaciar la evidencia antes del consejo. Se reproduce 0,95 atribuido indebidamente. El core ahora posee estados por símbolo e instancia; features/consejo usan una vista as-of no destructiva. Repetir ticks no consume ni suma el nivel. Igual E con contenido distinto conserva máximo y registra ambigüedad, no presume identidad de orden. Evidencia futura bloquea entradas, pero el test confirma continuidad del cierre defensivo. Historia causal, calidad del feed y universo transaccional permanecen abiertos.

FMT-253: el parser operativo conserva símbolo/E/T/lado y valida esquema, valores y unidades. Usa ap*z como ejecución acumulada reportada del snapshot UM, no p*q solicitado ni volumen incremental. CM explícito no se reinterpreta como UM; un registro inválido no borra los válidos del array. Las API antiguas permanecen compatibles y diagnosticadas: 1e3→1 y q*p incorrecto siguen OPEN fuera de la nueva ruta. El decoder nuevo no certifica claves JSON duplicadas, transiciones X ni rendimiento bajo ráfagas.

FMT-254 ABIERTO: un feed de snapshots muestreados no identifica tape completo, intensidad o probabilidad de cascada. La severidad logarítmica fija implica, con umbral0,85, una frontera≈125.892,54 unidades de cotización sin ajuste por liquidez. Semivida10s no demuestra adaptación genómica; desde score1 el cruce de ese umbral ocurre≈2.344,65ms después. La envolvente máxima decaída evita sumar acumulados sin identidad, pero no conserva volumen. Sigue pendiente historia equivalente para backtest/demo/prod y normalización multi-activo validada.

FMT-255 ABIERTO: apply_event del kernel genérico puede retroceder last_timestamp_ms y envejecer dos veces al volver al mismo instante; también carece de dominio numérico completo. Nuevo estado de liquidación evita ese contrato, pero el helper no se da por reparado. FMT-256: se corrige overflow del punto medio del filtro de spread mediante r=bid/ask y 2(1-r)/(1+r). Se conserva el límite0,50 y se retira la afirmación de que es una ley de mercado.

26 tests nuevos; tres aserciones válidas rojo→verde. Selección final100pases=91funcionales/compatibilidad+9OPEN. Check offline de cuatro binarios pasa, tres warnings previos. Lecturas completas nuevas tensor_parser.rs y validation.rs:153/289Rust,136pendientes. 41modelos sin cambio; prefijos documentales conservados. Sin cuentas/órdenes, promoción operativa, reinicios ni publicación Git. El informe incluye topología raíz→estado→decisión→terminal, semántica de cálculos, matriz de vetos y agenda de investigación con requisitos de evidencia.

## Adenda XXXIV — memoria causal, costes y evolución verificable — 2026-09-25

[Informe detallado XXXIV](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XXXIV_2026-09-25.md>) · [Artefacto estructurado XXXIV](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XXXIV_2026-09-25.json>). Adición al historial; la matriz anterior y sus observaciones de aquel snapshot permanecen intactas.

FMT-255, continuación: reparado el contrato local del acumulador exponencial. Las APIs try_* distinguen configuración/tasa/estado/impulso inválidos, reloj anterior y overflow; validan toda la transición antes de publicar nivel y reloj. StatefulEngine valida también funding/OBI, conserva el evento anterior ante rechazo y contabiliza rechazos. El test OPEN anterior del retroceso se convierte en regresión reparada. No se declara resuelta la trazabilidad de frescura hasta la decisión ni adaptación genómica de la semivida10s; las firmas legacy mantienen limitaciones de observabilidad del error. new(h) conserva firma, no comportamiento ante configuración inválida: ahora panic explícito, con try_new para entrada externa.

FMT-257: continuidad del slippage auxiliar reparada. exp(1+2|d|) daba límite e cuando d→0−, frente a valor1 en cero; se corrige a exp(2|d|). No se cambia gamma ni límites económicos. Siguen OPEN profundidad ausente→1,5bps, piso monetario que rompe cambio de unidades y urgencia rígida. No se localizó caller operativo del helper. FMT-258 ABIERTO: RealityPhysics sí participa en costes del core; maker presupone fill a precio base, sin cola ni probabilidad. Latencia negativa puede esconder NaN detrás de un piso finito; precio extremo finito puede producir infinito. La frontera tau60s elige maker/taker; es una fuente potencial de sesgo de selección, no prueba cuantificada de pérdida.

FMT-259 ABIERTO: Darwin legacy evalúa candidatos con drawdown0,95 forzado, baseline sin ese override; la diferencia depende del valor de arranque. Usa primeros30slots, contexto sintético y la misma ventana para selección/comparación; +5%fitness no es significación estadística. FMT-260 ABIERTO: aplica al estado vivo antes de confirmar persistencia; error posterior no revierte RAM. La escritura de átomicos individuales tampoco publica una generación coherente. La ruta está condicionada por flags desactivados por defecto en código; no se activó ni se inspeccionó su estado de proceso.

FMT-261 ABIERTO: el cazador de constantes produce genes ordinales cuya identidad se desplaza al insertar literales, reutiliza namespace global, transforma const fn/inline const con llamadas de runtime y omite literales dentro de macros. Las pruebas inspeccionan AST serializado, no compilan el resultado. Hace falta catálogo estable con unidades/dominio/consumidores; convertir literales en genes no demuestra autoevolución. No se localizó uso operativo de este transformador.

Verificación final:105pases únicos=92funcionales/compatibilidad+13OPEN;25tests nuevos=13funcionales+12OPEN;7fallos válidos rojo→verde. Check offline de cuatro binarios pasa, warnings previos registrados. Lecturas completas nuevas:reality_physics, darwin, slippage_predictor y cazador_constantes;157/289Rust,132pendientes.41modelos sin cambio. Investigación primaria distingue impacto de metaorden de fill individual; no acredita coeficientes universales ni ventaja cuántica. Sin órdenes, cuentas, promociones/reinicios operativos o publicación Git.

## Adenda XXXV — comparación de genomas y auditoría de los vetos — 2026-09-25

[Informe XXXV](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XXXV_2026-09-25.md>) · [Artefacto XXXV](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XXXV_2026-09-25.json>). Se conserva el historial; las conclusiones siguientes corresponden al nuevo snapshot.

FMT-262: reparación local del dominio numérico del fitness. Un cociente de capitales positivos finitos podía desbordarse antes del log o colapsar a cero. La fórmula compartida usa diferencia de logs cuando el cociente no es representable. La API checked distingue capital/DD inválido y soporte insuficiente; los wrappers mantienen −infinito por compatibilidad. OOS inválido ya no se convierte en factor neutro y prior no finito no contamina la contracción de muestra corta. No se altera el mínimo heredado ni se certifica su suficiencia estadística.

FMT-259, continuación: candidato y baseline usan ahora un replay común y el mismo límite de drawdown capturado; desaparece el override0,95 exclusivo del candidato. Persisten defaults no congelados, contexto sintético, límite de población, evaluación en la misma cinta y valoración terminal incompleta. FMT-263: se elimina el baseline ficticio −999999; ambos scores deben ser finitos para la comparación. Superar el margen relativo no prueba utilidad positiva, optimalidad, cambio estructural o generalización. Darwin legacy no se activó ni promovió genomas.

FMT-264: alerta de agotamiento reparada; antes ratio≤3 nunca podía satisfacer ratio>4. Se mantiene el umbral, y el consumidor sólo emite telemetría. OPEN: datos ausentes/NaN equivalen a alineación perfecta, ticks antiguos rebobinan el track, hay dos depósitos scalp/swing y capacidad30 en el host; el error TCE se anuncia invertido como fidelidad. FMT-265 OPEN: drift NaN devuelve Ok, faltan identidad de pareja y validación de configuración; shadow=0,95·real no es predicción independiente.

FMT-266 OPEN: el rearme no reinicia crédito de cierres limpios ante nueva discrepancia, usa el valor previo de fetch_add y libera un booleano compartido sin propiedad de causa. FMT-267 OPEN: la recurrencia llamada SPRT omite varianza/término de centrado de la LLR normal; bounds nominales no dan garantías del1%. Sin caller operativo localizado del auditor comportamental. Documentación científica y afirmaciones de latencia precisadas; no se inventa una nueva calibración.

Cobertura acumulada: 160/289 Rust preexistentes, 129 pendientes; nuevas lecturas completas de trajectory_auditor, drift_auditor y behavioral_auditor. La verificación final y su separación entre contratos y diagnósticos OPEN quedan detalladas en el informe/JSON. 41 modelos preservados. Sin actividad de cuenta, órdenes, entrenamiento/promoción operativos, reinicio del motor ni publicación Git. La investigación distingue preferencia de drawdown, utilidad logarítmica y restricciones probabilísticas; no incorpora teorías por prestigio.


## Ronda XXXVI — causalidad del veto, recuperación propia y límites de la auditoría

[Informe detallado XXXVI](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XXXVI_2026-09-25.md>) · [Artefacto XXXVI](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XXXVI_2026-09-25.json>). Ampliación aditiva del historial, no certificación global.

FMT-265, continuación: API checked de drift con razones distintas para límite, pareja, PnL, diferencia y acumulador inválidos, overflow y exceso finito. El wrapper legacy ya no devuelve Ok(NaN); el host usa errores tipados. Se preserva0,05 por observación, no como prueba estadística. OPEN: shadow=0,95real, suma firmada que cancela, ausencia de outcome_id/generación/deduplicación y cierre de salida no confirmado.

FMT-266, continuación: DriftRecovery reinicia la racha ante todo error y libera exactamente en la décima observación válida. Sólo controla un veto privado de entradas del core; no limpia el latch global ni el executor. Permite propuestas locales de cierre defensivo. OPEN: evidencia de recuperación compartida entre activos, repeticiones aceptadas, falta de vía de recuperación sin cierres y veto aplicado desde la siguiente decisión. La entrada ya generada en la llamada actual requiere arbitraje transaccional.

FMT-268: StateValidator sustituye/omite evidencia inválida, lee sin snapshot contable, omite flujos externos como términos explícitos y retorna unit. FMT-269: las dos políticas de salud discrepan a1500ms y no llevan ventana/denominador. FMT-270: ChaosMonkey usa reloj mod100, no semilla reproducible; probabilidades inválidas simulan éxito/drop y latencias submilisegundo se omiten. Sin callers operativos externos localizados de estos helpers.

FMT-271: spawn y try_send de telemetría ignoran fallos; una alerta puede perderse sin contador. FMT-272: capital/(PnL medio×50) es escenario lineal, no pronóstico compuesto; se corrigen etiqueta y descripción preservando cálculo. FMT-273: rollback_position(coin_id) cierra todos los slots, incluido uno confirmado ajeno, reproducido de forma aislada. FMT-274: tests forenses sin aserción sobre datos/pipeline imprimen certificado; correlación de cuatro parejas sintéticas no prueba ausencia de leakage.

155 pruebas únicas:133 contratos/compatibilidad,5 smoke heredados y17 OPEN;21 nuevas (14 funcionales,7 OPEN),3 OPEN anteriores reclasificados tras reparación.4 RED iniciales→GREEN. Check god_engine/evolver/walkforward_evolver correcto. Cobertura165/289 Rust,124 pendientes;5 nuevas lecturas completas.41 modelos intactos. Host editado en fuente, no desplegado. Sin órdenes/cuentas, promoción, reinicios ni publicación Git. Investigación de inferencia secuencial utilizada para precisar supuestos, no para declarar garantías o implantar un contraste sin evidencia causal.

## Ronda XXXVII — propiedad de reservas, sizing y límites espectrales

[Informe detallado XXXVII](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XXXVII_2026-09-25.md>) · [Artefacto XXXVII](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XXXVII_2026-09-25.json>). Adenda sin borrar ni reescribir estados históricos; no certificación total.

FMT-273/FMT-266: el host captura una reserva por símbolo, coin, slot y generación. Rechazos ordinarios y veto tardío memoria/drift cancelan sólo ese ocupante no confirmado. Confirmación y cancelación cooperantes comparten lock; la compensación sólo la ejecuta el cancelador ganador. El adaptador legacy ya no barre slots ajenos. Persisten escritores externos, ausencia de ledger durable, generación prevista antes de open, mutación del registro y falta de claim-submit. Una emergencia confirmada no se trata como rechazo ni devuelve fee de entrada; conserva estado para reconciliación, con deduplicación todavía pendiente.

FMT-275: eliminado PF mínimo1,01 que fabricaba ventaja, veto WR40% y pisos que reinflaban riesgo. Dominios finitos y límites ordenados; tamaño bajo mínimo se abstiene. Los multiplicadores de confianza/Hurst/racha/DD siguen siendo heurísticos y el resultado no demuestra nocional Kelly óptimo. FMT-276: capital inválido no se convierte en13; capacidad cero, cociente monetario sin base.max(1). Capacidad por raíz, suelo de riqueza y split siguen sin estimación espectral. No se encontró caller operacional de estas dos APIs.

FMT-277: confirmación por reserva en lugar de slot fijo; el bucle deja de reservar una segunda propuesta que sobrescriba el único retorno. No se demuestra selección óptima, fill completo ni migración de los demás lectores fijos. FMT-278: PnL de emergencia tenía signo contrario para long y short; usa ahora gross_pnl común. Se verifica aritmética y cableado estático; siguen pendientes identidad, salida confirmada, fees completos y deduplicación.

FMT-186 continúa abierto: tres slots, fallback30s para tau inválido/pequeño y cortes0,80/1,50 no prueban ortogonalidad; comentarios precisados y cuatro diagnósticos OPEN. FMT-113 continúa abierto: margen propio ya reservado participa otra vez en free_margin, pudiendo provocar rechazo o aumento de leverage; cambiar leverage no reduce pérdida de nocional fijo. Requiere ledger/proyección conjunta, no otro umbral.

109 pruebas únicas:103 funcionales/compatibilidad,1 estática y5 OPEN.24 nuevas=19funcionales+1estática+4OPEN; un OPEN previo reclasificado.6 RED numéricos y1 RED de cableado pasan tras cambios. Check offline de tres binarios; warnings heredados. Cobertura166/289 Rust,123 pendientes; nueva lectura completa capital_compounder.41 modelos conservados. Investigación Kelly/DRO usada para limitar garantías y describir riesgo/nocional; no optimizador nuevo ni ventaja cuántica. Sin órdenes/cuentas, despliegue, promoción, reinicio o publicación Git.

## Ronda XXXVIII — evidencia contable, aprendizaje y causalidad de los vetos

[Informe detallado XXXVIII](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XXXVIII_2026-09-25.md>) · [Artefacto XXXVIII](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XXXVIII_2026-09-25.json>). Adenda sin reemplazar el historial ni certificar la matriz completa.

FMT-279/FMT-278: checked_gross_pnl rechaza precios/cantidad inválidos, overflow y producto no nulo perdido hasta cero. El wrapper legacy sigue retornando0 ante error y no certifica un resultado plano. Host y emergencia consumen la API checked antes de aprender/encolar. OPEN: supplied PnL finito puede contradecir precios; falta outcome_id, moneda, generación, fill confirmado y deduplicación. El resultado plano actualiza media ganadora pero se registra como false en el indicador win; acumuladores globales aún pueden desbordar.

FMT-280: JSON estructurado conserva claves, escapa textos y no trunca a decimales fijos. Contador INVALID_RECORDS antes de I/O/cola. OPEN: canal I/O sin cota, errores de disco ignorados, fallback síncrono, cola de aprendizaje1024 con pérdidas, poisoning y recuperación sin as-of/generación. Serialización válida no equivale a durabilidad ni a muestra íntegra.

FMT-281: cinco OPEN prueban que ShadowExecutor acepta precios NaN/Inf, cantidad redondeada a0 y limit inválido; su kill sólo imprime y el ACK no produce fills/posiciones/capital. Sin constructor operacional localizado fuera de tests: no se imputa a este stub la discrepancia real de demo. Se precisa su documentación, no se lo convierte en simulador completo.

FMT-282: veto operacional por comisiones usa una ventana derivada de un gen temporal global, no del espectro propio de cada activo. Cuenta filas REALIZED_PNL no nulas como trades; agrega sin usar asset; pagina hasta4 sin estado de completitud; renueva hasta=ahora+horas con la misma evidencia. Umbral3, piso monetario1e-9, horas enteras y persistencia sin ACK no prueban viabilidad futura. Se documenta rehabilitación causal, sin eliminar la protección ni inventar calibración.

FMT-283: tipos de bracket ahora exactos; NOT_STOP_MARKET ya no pasa. Los nombres de clientOrderId siguen siendo convención, no firma ni prueba de propiedad. FMT-181 continúa: aceptar fees firmadas en serializer no corrige abs del productor ni falta de conversión monetaria.

78 pruebas únicas:69 funcionales/compatibilidad,2 estáticas y7 OPEN.22 nuevas=15 funcionales+1 estática+6 OPEN;5 RED de comportamiento y1 estático pasan tras reparación; ningún OPEN previo reclasificado. Check offline de tres binarios correcto con warnings heredados. Cobertura168/289 Rust,121 pendientes; dos lecturas completas nuevas (trade_accounting y shadow), tres relecturas sin incremento.41 modelos intactos. Investigación primaria de ingresos utilizada para distinguir filas, operaciones y numerario. Sin órdenes/cuentas, entrenamiento/promoción operativos, build/reinicio del motor o publicación Git.

## Ronda XXXIX — cobertura de income y significado de los informes

[Informe detallado XXXIX](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XXXIX_2026-09-25.md>) · [Artefacto XXXIX](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XXXIX_2026-09-25.json>). Continuación aditiva, no sustitución de la matriz histórica ni certificación global.

FMT-282, consulta: colector probado con páginas numeradas e intervalo inclusivo fijo. Se distinguen PageExhausted, PageBudgetExceeded, NoProgress y Simulated. El wrapper usado por el host entrega filas sólo ante agotamiento observado; el límite4/20 no certifica cobertura. Se valida filtro escalar, rango, tamaño, identidad mínima y finitud. Activo/trade_id forman parte de la tupla visible; cambio de importe bajo esa misma tupla es conflicto.1.001 registros del mismo milisegundo sobreviven en el test. OPEN: sin snapshot, retención certificada, identidad de cuenta ni reconciliación de revisiones que cambien la tupla.

FMT-282, alcance: el control de moneda se aplica por símbolo y sobre las tres clases de la política legacy. Un grupo mixto no suprime la evaluación de otro homogéneo; transferencias globales no contaminan esos grupos. No se convierte FX ni se libera explícitamente el veto existente; su expiración con datos insuficientes sigue sin política causal. Permanecen n>=3filas, abs de comisiones, escala global, sumas legacy, renovación con la misma evidencia y persistencia incompleta.

FMT-284: income_report deja de afirmar WR bruto=WR neto por trade. Conserva proporción de filas positivas como %FILAS+, sin inferir operaciones independientes. Agrega por moneda/símbolo y total por moneda; muestra R+C+F como subtotal seleccionado y OTROS separado, sin llamar beneficio a transferencias. Sumas checked; razón(C+F)/|R| sin epsilon monetario y N/D cuando indefinida/no representable. --days inválido, overflow y pre-epoch se rechazan; la ventana mostrada es la consultada. No ROI ni WR neto inventados.

66 pruebas únicas:58 funcionales/compatibilidad,5 estáticas y3 OPEN.28 nuevas=23funcionales+3estáticas+2OPEN;3RED iniciales estáticos→GREEN, no RED funcionales declarados. Check offline de god_engine/income_report/evolver/walkforward_evolver correcto, warnings heredados.41modelos intactos. Cobertura169/289Rust,120pendientes: nueva lectura completa income_report; relecturas y módulo nuevo no inflan el inventario. Investigación Firecrawl reutiliza fuente oficial del mismo día; no snapshot ni unicidad global inferidos. Sin cuentas/órdenes, entrenamiento/promoción operativos, build/reinicio o publicación Git.


---

## Anexo Ola XL (2026-09-25 tarde) — fusión PR #5 + FMT-285

Entre rondas: WIP XXX–XXXIX versionado (ac136633) y PR #5 fusionado semánticamente (merge 6fdccd64, ~90 hunks; decisiones en issuecomment-5841402608). Defectos del PR reparados al fusionar: NaN capital/stop/posterior admitía órdenes en max_leverage (fail-closed); deadlock D-750↔D-751 mataba el arranque frío (oráculo 0/144 medido) — cláusula D-751b: arranque frío total ⇒ sonda mínima D-750. Techo micro re-auditado 5×→4×.

FMT-285 (§13.2): partition_income separa aceptadas/cuarentena(InvalidRecord recuperable, ConflictingIdentity no recuperable)/duplicados exactos con inventario conservado; SymbolInterval por símbolo = intervalo observado del recorrido, no retención; symbols_with_quarantine debilita por símbolo, no global. Pendiente: cablear al recorrido con transporte (FMT-285b), identidad con payload decimal (§13.1), FX as-of (§13.3).

Re-baseline PENDIENTE: oráculo T-1 #[ignore] con diagnóstico — física de viabilidad D-751/D-756 inviable sobre fixture sintético (spread 4pb vs fricción 7pb a τ corto; diag: spread 0.0004 ≤ max 0.000002). Recalibrar sobre tape real; el trinquete sólo sube desde re-medición documentada. Exportador de paridad D-753 recuperable: git show 3ee49b05:src/bin/feature_exporter.rs. Workspace verde (T-1 ignorado); cobertura lecturas sin cambio 169/289.


---

## Anexo Ola XLI (2026-09-26) — cadena de vetos, campo de régimen y teoría nueva

**Censo**: ~70 puntos de veto en 6 etapas (tabla maestra en AUDITORIA_XLI §1/§6); 9 deadlocks/incoherencias reparados (D1 sonda sin roster, D2 proxy de riesgo, D3 X-016 frío, D4 una-puerta-ML, D5 Stouffer unificado, D6 orchestrator, D7 quórum dd, D9 banda resonante) + 2 bugs reales del merge (re-sembra ts=0; lector spectral_coherence con peso legacy ≠ decisión D-742 — doctrina de DOS MASAS declarada: aprendizaje=persistencia×ganancia, decisión=observable). **Oráculo T-1: 0→19/144 (13,2% > trinquete 11,5%)**.

**B1 Campo de régimen** (spectral_regime.rs): coordenadas continuas (H(τ)×3, τ*, dlnτ*/dt, entropía, marea) + crash-ness [0,1]; margen de largos ×(1−0.95·crash_flux) con marea adversa; enum = vista legacy. **C1 Marchenko-Pastur** (random_matrix.rs): borde (1+√(N/T))²; grupo AllNoise ⇒ el ruido no veta (D-748 con denoising RMT). **C2 Kolmogorov**: S_p(τ) del núcleo espectral (3er momento EWMA), ζ(p) log-log, χ=(1−ζ3)⁺; pisos P80/P85 ×(1+0.5χ). **C3 Fisher de escala**: Σ(Δq/Δlnτ)²/q de la masa — identificabilidad del régimen (walk-forward: cablear). Tabla milenio (C4): NS→multiescala (integrado), Y-M→familia RMT como teorema (integrado), Hodge→grafo de señales (candidato), RH/P≠NP/BSD→sin variable identificada (rechazado por protocolo). Deuda A3 con diseño: CVD z-tipificado, X-016 percentilado, pisos fríos.


---

## Anexo Ola XLII (2026-09-26) — z-tipificación de flujo/campo + Wasserstein espectral

**A3 completo**: CVD z-tipificado (veto F8-P10 por Z95 contra su propia distribución σ_EWMA; confluencias ±0.05→z>0, 0.15→0.5σ; fallback literal en calentamiento); X-016 z-tipificado (entropía >μ+2σ, marea adversa >2σ, EWMAs por símbolo). Pisos fríos 0.40/0.15 RE-CLASIFICADOS: fallan seguro (conservadores), se quedan como contrato de arranque. **Consumidores**: Fisher→walk-forward (gate de identificabilidad: campo difuso = ronda aplazada, evolucionar sobre ruido memoriza ruido); crash_pressure=0.25·crash_flux contrae colchón de largos en orchestrator. **Teoría D — W₁ de Wasserstein espectral**: transporte óptimo 1-D exacto (Σ|ΔCDF|·Δlnτ) entre masas espectrales a lag N (anillo 256); unidades = ejes de escala movidos; complementa Fisher (concentración estática vs MOVIMIENTO). Publicado al registry. Trinqueta del oráculo: 13.2%→**13.9% (20/144)**.


---

## Anexo Ola XLIII (2026-09-26 tarde) — Hayashi-Yoshida asíncrona + BOCPD sobre W₁

**HY**: el veto D-748 correlacionaba monedas muestreando ticks en rejilla común — Epps effect: la correlación decae con la desincronía, el par "independiente" era a veces el más desincronizado. R_HY(A,B)=Σ_{overlap>0} rᵢrⱼ normalizada (Hayashi-Yoshida 2005), O(n+m), cableada ANTES del Pearson (fallback); la matriz del grupo MP hereda la covarianza corregida. 3 tests: sincronas→HY==1; señal común con relojes desplazados→mantiene ~0.98; sin solape→None. **BOCPD sobre W₁** (Adams-MacKay, hazard 1/100, R=128): p_transition = masa posterior de segmento nuevo sobre el transporte espectral. La falsación delató la emisiva fija (ciega al salto) → estadística por segmento (predictiva N(media_r,0.35) vs prior ancho N(0,2.0)): reposo→~hazard, salto 2 ejes→p>0.5. **Medición honesta (3 corridas oráculo)**: el descuento de convicción ×0.70 midió 20→15 genes en DOS calibraciones — el fixture sintético tiene transporte fluctuante y gravaba la sesión entera; ACTUADOR a telemetría hasta calibrar sobre tape real (doctrina D-751), DETECTOR operativo. Trinqueta: 20/144 restaurado y PASA.
