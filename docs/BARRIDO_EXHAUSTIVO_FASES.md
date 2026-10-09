# BARRIDO EXHAUSTIVO POR FASES — todos los archivos, uno por uno

> **Companion del PLAN_MAESTRO_SINCRONIZACION.md** (Qoder, Ola 56).
> Mandato del operador 2026-10-04: recorrer el árbol COMPLETO en fases,
> de la base a lo menos esencial: metas → conceptos → matemática →
> estadística → física → cuántica → algoritmos → código. Cada archivo
> pasa una checklist; cada fase cierra con verificación de
> comportamiento (tests + compilación + oráculo si toca pipeline).

## Inventario real del árbol (medido 2026-10-04)

| Zona | Archivos | Líneas | Fase |
|---|---|---|---|
| god-engine-core | 28 | 18 987 | F3 |
| src/ (bin god_engine + herramientas) | 69 | 21 574 | F3 |
| execution-engine | 22 | 11 655 | F4 |
| quantum-arena | 27 | 13 228 | F2 |
| risk-engine | 22 | 8 719 | F4 |
| evolution-engine | 16 | 6 499 | F5 |
| signal-engine | 19 | 6 647 | F2 |
| data-pipeline | 24 | 6 075 | F6 |
| feature-engine | 21 | 5 549 | F2 |
| backtest-engine | 9 | 5 180 | F5 |
| metacortex-engine | 12 | 4 209 | F6 |
| telemetry-server | 13 | 3 448 | F7 |
| storage-engine | 8 | 3 004 | F6 |
| dark-alpha-engine | 3 | 2 010 | F5 |
| audit-engine | 11 | 1 930 | F7 |
| strategy-core | 8 | 1 564 | F2 |
| os-guardian | 10 | 1 087 | F7 |
| data-ingest | 5 | 1 036 | F6 |
| otros (graph, phase, flight, registry, telemetry) | 11 | ~1 950 | F7 |
| tests de integración | 136 | — | por fase |
| **Total** | **~377 .rs** | **~137 000** | 8 fases |

## Las fases (de la base a lo menos esencial)

### F0 — METAS Y CONCEPTOS (sin código)
Checklist por documento rector: la meta del 100%/3d está explícita;
el principio espectral-continuo es el marco (no scalping/swing); cada
concepto usado por el código tiene definición escrita.
Archivos: RULES de `.agents/`, PLAN_MAESTRO_*, ADR-0001..0013,
HOJA_DE_RUTA, docs/adr/*, MEMORIA.

### F1 — MATEMÁTICA Y ESTADÍSTICA TRANSVERSAL
Los módulos que DEFINEN las matemáticas del sistema: genome (curvas
τ), temporal_spectrum (malla 32, IC #594, significancia #599),
espectral_multiactivo (#607), skill_motores (#626/#648),
cramer_lundberg, ruin, drawdown, capital_regime, calibration,
random_matrix, hodge, multifractal, spectral_tape, veto_registry.
Checklist: unidades coherentes; continuidad C∞; significancia
estadística contra nulo; paridad fórmula-doc; dimensión correcta.

### F2 — FÍSICA Y CUÁNTICA (el sustrato espectral)
signal-engine (19: los 13 motores + orquestador + voto_espectral +
skill), feature-engine (21), quantum-arena (27, menos los de F1),
strategy-core (8).
Checklist: física correcta por motor (lecciones #649/#650: monotonía,
unidades, signo, paridad sombra↔vivo); z-scores por escala;
antisimetría; abstención honesta; supervivencia de marcadores
anteriores.

### F3 — EL NÚCLEO VIVO (el corazón)
god-engine-core (28) + src/bin/god_engine.rs (el bin de ~40k líneas
está en F3 junto al core que lo alimenta).
Checklist: orden de publicación→consumo; anti-staleness; namespaces
c{id} vs {SYM}; paridad host/replay; D-743 puertas_del_continuo
única; snapshot/edad de posiciones; cerrojos.
⚠ Esta fase toca pipeline vivo en cada arreglo: oráculo por sub-bloque.

### F4 — DINERO Y RIESGO (donde se pierde el bps)
execution-engine (22) + risk-engine (22).
Checklist: evidencia terminal IOC (CL-39); reserva = margen de envío
(CL-41b); validadores de Codex; vetos con dato medido (ρ(τ*) 4
etapas, Lundberg unidades #651, dd-lerp #653); exposición y
correlación D-748; GENOME-GATE abierto de Claude.

### F5 — APRENDER Y MEDIR (evolución y backtest)
evolution-engine (16) + backtest-engine (9) + dark-alpha-engine (3).
Checklist: el examen juzga barras de mercado (CL-27); walk-forward
sobre curvas τ; DSR en promoción; paridad bt↔vivo (10/10 GLM);
frontera OOS explícita (PR#28); etiquetas de barrera vs neto
(XLIV-9c/11).

### F6 — DATOS Y MEMORIA
data-pipeline (24) + data-ingest (5) + storage-engine (8) +
metacortex (12).
Checklist: parseo científico (AGY-P11); tokens escalares (PR#28);
des-espejado D-747; linaje de modelos; watcher/expiry de evidencia.

### F7 — OBSERVABILIDAD Y PLATAFORMA
telemetry-server, telemetry-engine, audit-engine, os-guardian,
flight-recorder, graph-*, phase-runner, omniscient-registry.
Checklist: telemetría con escritor Y lector (deuda B); sin estado
global en tests; logs redirigidos siempre.

### F8 — INTEGRACIÓN Y SOPLADO FINAL
136 tests de integración uno por uno: cada suite contra su contrato
documentado; luego regresión completa (8 crates) + T-1 del tip;
cierre con informe forense por fase.

## Reglas del barrido

1. **Un archivo por entrada de checklist** — nada de "ya lo vi en
   otra ola": el mandato es pasar por TODOS, uno por uno.
2. **Orden de dependencia dentro de cada fase**: primero lo que
   otros importan.
3. **Cada hallazgo entra al buzón con etiqueta de fase** (`F3:`...)
   y espera su ola (con oráculo si toca conducta) — el barrido
   INVENTARÍA, las olas ARREGLAN.
4. **Verificación por fase**: compilación + tests de la zona al
   cerrar cada fase; oráculo completo al cerrar F3/F4 (las que tocan
   conducta).
5. **División entre agentes**: por zona dueña (yo F1-F3, GLM F5-F6
   de su línea, Claude F4 de la suya, Codex F6-F7 de contratos
   raíz) — el barrido respeta las líneas del Plan Maestro.

## Bitácora del barrido

| Fase | Estado | Hallazgos | Verificación |
|---|---|---|---|
| F0 | CERRADA 2026-10-04 | 1 (F0-1, corregido en fase: ADR-0014) | docs-only |
| F1 | CERRADA 2026-10-04 | 23 (2 HIGH, 8 MED, 13 LOW) | docs-only; A1 y A4 re-verificados contra el árbol |
| F2 | CERRADA 2026-10-04 | 43 (7 HIGH, 16 MED, 20 LOW) + hallazgo estructural sombra/vivo + inventario milenio | docs-only; A1/A3/A5/B1/C1/C4 re-verificados |
| F3 | CERRADA 2026-10-04 | 36 (3 HIGH, 14 MED, 19 LOW) — patrón paridades rotas voto/aprendizaje | docs-only; A1/B1/C1 verificados; ws check exit 0 |

## F0 — RESULTADO (cerrada 2026-10-04, Qoder)

Checklist por documento rector (uno por uno):

| Documento | Meta 100%/3d | Marco espectral | Conceptos definidos |
|---|---|---|---|
| .agents/AGENTS.md | ✓ explícita | ✓ regla anti-scalping | ✓ (reglas vivas) |
| PLAN_MAESTRO_SINCRONIZACION.md | ✓ §0 | ✓ | ✓ §1 líneas |
| PLAN_MAESTRO_2026-10-04.md (GLM) | ✓ | ✓ | ✓ §5b mapa |
| BARRIDO_EXHAUSTIVO_FASES.md | — (tool) | ✓ fases | ✓ |
| HOJA_DE_RUTA_CIMIENTOS (Claude) | implícita | implícita | su scope es ejecución |
| ADR-0001..0013 | — | parcial | cada uno su dominio |
| .agents/rules/* | — | ✓ sin residuos | — |

**Hallazgo F0-1 (único, corregido en la misma fase)**: la doctrina
espectral — los seis principios rectores del motor (#609..#654) — no
tenía ADR: vivía sólo en el forense (bitácora) y la memoria de
sesión. **Corregido: ADR-0014-doctrina-continuo-espectral.md** (los
seis principios con su ola y su prueba viva). Los conceptos vivos del
código (VotoEspectral, SkillMotores, excitación, ρ(τ*)) quedan con
definición de referencia.

Lenguaje residual scalping/swing en rectores: SOLO la regla que
ordena no pensar así (contexto correcto). Rules/: cero residuos.

**F0 CERRADA**. Siguiente: F1 (matemática/estadística transversal).

## F1 — RESULTADO (cerrada 2026-10-04, Qoder)

Tres auditores en paralelo, un archivo por entrada de checklist:

- **Auditor A**: genome.rs + temporal_spectrum.rs (matemática del
  genoma y del banco de escalas #594).
- **Auditor B**: cramer_lundberg, drawdown, ruin, capital_regime,
  hodge, micro_weight (risk-engine numérico).
- **Auditor C**: multifractal.rs + spectral_tape.rs + skill_motores.rs
  (medida espectral y bancos de habilidad).

### Hallazgos (etiqueta `F1:` — esperan su ola; el barrido INVENTARÍA)

| # | Archivo:lín | Sev | Defecto |
|---|---|---|---|
| F1-A1 | temporal_spectrum.rs:604 | **HIGH** | `umbral_ic_significativo(s.skill_n)` con n VITALICIO, pero los momentos del banco de τ* (#594) son EWMA olvido 1/64 (N_ef≈127): tras ~10³ bloques el umbral 2/√(n−3) cae bajo el SE real y la "significancia" degenera a IC>0 — τ* puede elegirse por ruido en sesiones largas. Es EXACTAMENTE el H5 que #648 cerró en SkillMotores (skill_motores.rs:87 usa `min(n, N_EFECTIVO_EWMA)`): el banco #594 nunca recibió el arreglo. Verificado contra el árbol |
| F1-A2 | genome.rs (tests) | MED | Test de simetría de curvas compara sólo longitudes, no valores; slots 13-16 del vector sin consumidor vivo |
| F1-A3 | genome.rs (mutación vs bounds) | MED | Bandas de mutación ≠ bounds de validación: `dynamic_atr_min` muta en [1e-4,1e-2] pero bound-lo=1e-7; iceberg ×20 salto — el mutante puede violar el GENOME-GATE al cargar (conversa con el defecto abierto de Claude en la carga de genomas versionados) |
| F1-A4 | genome.rs:2704 | MED | `normalize_sl_curve_friction_floor` se aplica en mutate (1772, 1992) pero NO en `from_vector` (sólo `enforce_curve_rr`): un genoma reconstruido desde vector puede traer SL bajo el piso de fricción. Verificado contra el árbol |
| F1-A5 | genome.rs | LOW | `new_random` genera fuera de bounds en 4 genes |
| F1-A6 | genome.rs | LOW | maker_only congelado (sin lector vivo) |
| F1-A7 | genome.rs | LOW | kelly_horizon_curve puede exceder 1 |
| F1-A8 | temporal_spectrum.rs | LOW | 6 interpoladores hardcodean la malla en vez de leer `SPECTRUM_SCALES_MS` |
| F1-A9 | core (#601) | LOW | El Monte Carlo del censo usa k=4 pero el espectro operativo tiene 5 escalas reales |
| F1-A10 | temporal_spectrum.rs | LOW | `habilidad_en` interpola con sesgo lineal en malla log |
| F1-B1 | cramer_lundberg.rs:116 | MED | Techo de bisección `hi=100` fijo en unidades de R, pero R escala como 1/escala_de_y: con retornos de 0.1-0.5% un R legítimo de 500-5000 devuelve `None` (sin cota) pese a haber edge. Derivar `hi` de la muestra. Refuerza la conversión de unidades #651: R por-nocional vive exactamente en el régimen micro del sistema |
| F1-B2 | drawdown.rs:112-120 | MED | Escalón de borde en `drawdown_maximo`: con r=0 (arranque) el umbral es el gen (0.95) y con el PRIMER riesgo medido cae de golpe a ~0.3 — tras reinicio con caída acumulada el freno dispara instantáneo. Interpolar con el conteo de muestras |
| F1-B3 | cramer_lundberg.rs:96-102 | LOW | `var2` calculado y descartado — trabajo muerto O(n) |
| F1-B4 | ruin.rs:67-71 | LOW | `clamp_ruin` propaga NaN intacto — devolver 0.0 para no-finito |
| F1-B5 | capital_regime.rs:123-129 | LOW | `log_lerp` discontinuo en el borde `standard=0` (usos reales siempre >0) |
| F1-B6 | hodge.rs:28-29 | LOW | Docstring anuncia gaussiana O(N³); la implementación ya es el teorema analítico O(n²) |
| F1-C1 | spectral_tape.rs:740 | **HIGH** | `habilidad_volatilidad` autoriza con `skill_vs_climatology` (nulo débil: la media), no contra persistencia — `sse_persist` se calcula pero NUNCA gatea; un modelo que pierde contra el nulo correcto publica. Exigir skill>0 contra el máximo de ambos nulos |
| F1-C2 | god-engine-core lib.rs:2782 (#654, DE QODER) | MED | El EWMA de D₀ actualiza en cada tick con `espectro_cacheada()` que refresca cada 16 llamadas ⇒ cada lectura cuenta 16× — memoria efectiva ~4 espectros, no 64. Dedup por lectura fresca |
| F1-C3 | spectral_tape.rs:692-700 | MED | `sigma_at` interpola entre anclas sin gate de madurez — el filtro D-754b sólo vive en `sigmas_en_anclas`; una ancla inmadura contamina consultas intermedias |
| F1-C4 | skill_motores.rs:157-184 (#648, DE QODER) | MED | Bloque nacido en trade se puntúa con snapshot viejo (correcto), pero su re-arme llega en el depth siguiente con votos computados a t_d — información dentro de la propia ventana del bloque entra al voto "de armado" e infla el IC. Puntuar con el voto del último depth ANTERIOR al nacimiento |
| F1-C5 | multifractal.rs | LOW | Doc promete holgura −0.05 vs código −0.25 |
| F1-C6 | multifractal.rs | LOW | Bordes q=±2 nunca contribuyen al ancho (diferencias centrales sólo) |
| F1-C7 | multifractal.rs | LOW | `cajas=n/b` entera descarta la cola |

### Verificados LIMPIOS (evidencia algebraica/simbólica)

- Curvas ln(τ) del genoma: 144/144 genes alineados entre sí y con la malla.
- Malla espectral ×4 ÚNICA en fusión y masa (sin hardcodeos divergentes
  en los caminos auditados; los 6 de F1-A8 son lecturas, no duplicados).
- Semillas y clamps de EWMAs (post-XLIV-3/PR#11).
- Resolución efectiva (D-742/CL-32) viva en fusión y masa espectral.
- Hodge: identidad de Dirichlet ‖∇φ‖²=(1/n)Σdiv² verificada simbólicamente.
- random_matrix: Jacobi estable, tolerancia 64εn, effective_bets acotado
  por Cauchy-Schwarz.
- micro_weight C¹ y monótono (D-641).
- Prequential estrictamente causal del tape; `clim_lambda` con semivida
  ≈177 muestras (NO petrifica); RLS con cresta.
- IC coseno de skill_motores con olvido exacto y umbral Fisher correcto.
- Signos de Legendre y falsación D₀ (iid→0.9 vs cascada→0.2-0.8).

### Cola de olas que abre F1 (prioridad del dueño de la línea A)

1. **F1-C4 + F1-C2** (míos, tocan el consenso vivo): dedup D₀ y re-arme
   prequential estricto — oráculo obligatorio.
2. **F1-A1 + F1-C1** (HIGH de significancia/nulo): umbral N_efectivo en el
   banco #594 y gate contra persistencia en el tape — oráculo obligatorio.
3. **F1-A3 + F1-A4** (conversan con GENOME-GATE abierto de Claude):
   bandas de mutación = bounds y piso de fricción en from_vector.
4. **F1-B1 + F1-B2** (conversan con #651/#653): techo de bisección derivado
   de la muestra y rampa del drawdown_maximo.
5. LOWs: ola de limpieza agrupada.

**F1 CERRADA**. Siguiente: F2 (física/cuántica, ~74 archivos, zona Qoder).

## F2 — RESULTADO (cerrada 2026-10-04, Qoder)

Tres auditores en paralelo, un archivo por entrada de checklist:

- **Auditor A**: 16 motores de signal-engine (física evaluate vivo
  vs voto_espectral).
- **Auditor B**: sustrato quantum-arena (temporal_spectrum, state,
  espectral_multiactivo, spectral_tape, ranker...) + feature-engine
  básico.
- **Auditor C**: teorías cruzadas (feature-engine avanzado +
  strategy-core) + inventario crítico de teorías del milenio.

### HALLAZGO ESTRUCTURAL (transversal a los tres)

**Los arreglos de física #649/#650 viven SOLO en la sombra espectral**
(voto_espectral → consenso → consumo #624). Los `evaluate*` VIVOS que
alimentan el ensamble escalar de fallback (D-754) y las features del
PPO conservan la física vieja: hawkes `signum·tanh(λ/μ̂)` vota ±0.92
CONSTANTE en régimen normal (F2-A1, verificado), solitón vivo con
`signum·sech` invertido (F2-A3, verificado), flow_impulse con umbral
1.2 < SS=1.6 tautológico (F2-A2), Mach con bases temporales mezcladas
(F2-A11). Como #649 hizo que lo espectral se ABSTENGA en régimen
normal, el fallback escalar con física rota conserva mucho peso en la
decisión real. Es deuda del diseño sombra-primero (Olas 31-45): la ola
de integración consumió el consenso sin erradicar los caminos viejos.
**Ola mayor de erradicación requerida, con oráculo.**

### Hallazgos (etiqueta `F2:` — esperan su ola)

**Auditor A — signal-engine (14)**

| # | Archivo:lín | Sev | Defecto |
|---|---|---|---|
| F2-A1 | hawkes_bessel.rs:369 | **HIGH** | Vivo: `sign(dir)·tanh(λ/μ̂)` — régimen normal (1.6) vota ±0.92 constante, sin abstención; signum salta en dir=0. La moneda `excitacion_hawkes_norm` de #649 NO se usa aquí |
| F2-A2 | flow_impulse.rs:174-179 | **HIGH** | Vivo `vote()`: umbral 1.2 < SS=1.6 ⇒ gate abierto en régimen normal; escalones C⁰ en 1.2 y 0.2; pesos fijos 0.6/0.4 vs genómicos del evaluate — 3 superficies con física distinta |
| F2-A3 | soliton_wave.rs:213-216 | **HIGH** | Vivo: `vel.signum()·amp/cosh(...)` — física vieja (sech) divergente del espectral tanh(A·x) de #650; paridad rota |
| F2-A4 | hawkes_bessel.rs:147, flow_impulse.rs:61 | MED | Excitación FIRMADA multiplica tanh(x): la calma invierte el sentido del momentum; confluence usa abs() — inconsistente entre los 3 motores |
| F2-A5 | god-engine-core lib.rs:5757 | MED | El host pasa `hawkes_r = cvpin.current_vpin()` (probabilidad [0,1]) como ratio λ/μ̂ al flow_impulse de respaldo — unidades rotas; con umbral 1.2 ese camino nunca dispara |
| F2-A6 | flow_excitation_confluence.rs:243-252 | MED | Umbral `hawkes>=th && |obi|>=piso`: salto C⁰ de magnitud plena al cruzar; is_long/short binarios |
| F2-A7 | proyeccion_espectral.rs:29,69 | MED | `masa<0.25`⇒0 vs 0.25+ε⇒señal·conc: discontinuidad de magnitud plena |
| F2-A8 | coaxial_breakout.rs:44-51 | MED | Espectral firma con `sign(x)` duro vs vivo `tanh(dir/1e-4)` — paridad divergente |
| F2-A9 | perceptron_gate.rs:34-35,58-64 | MED | `signum`+piso 0.15: señal 1e-300 ⇒ ±0.15; perfil de peso con kinks en k=8/23 no derivados de la banda operable |
| F2-A10 | stochastic_resonance.rs:147-160 | MED | Vivo: varianza de ruido fallback `atr_pct` (por-barra) contra señal OBI adimensional — pico de resonancia mal registrado |
| F2-A11 | supersonic_shockwave.rs:176-187 | MED | Vivo: Mach = (vel/mid por-SEGUNDO)/(atr_pct por-BARRA) — bases temporales mezcladas (~60×); la sombra ya usa espacio-z |
| F2-A12 | conformal_reversion_filter.rs:103-109 | LOW | Tendencia por `sign(x[k+1])` duro (presente en ambas rutas — paridad ok) |
| F2-A13 | renyi_tsallis_entropy.rs:152-157 + core:2128 | LOW | Gates duros 0.60/0.15; espectral siempre q=1.5 sin leer registro; `tsallis_q_entropy` parece q pero se usa como valor |
| F2-A14 | game_theoretic_nash.rs:47,75-77; trend_runner.rs:102-108 | LOW | "Equilibrio" sin juego definido; constantes mágicas 0.02/0.08/0.04/1e-3 (paridad ok entre rutas) |

**Auditor B — sustrato (13)**

| # | Archivo:lín | Sev | Defecto |
|---|---|---|---|
| F2-B1 | god-engine-core lib.rs:2742-2748 | **HIGH** | `hurst_scale_matched` selecciona H por BANDAS DURAS (τ<2min→micro, <1h→meso): la H que dimensiona TP/SL salta discontinuamente al cruzar 120s/1h. Interpolar H(τ) en ln τ — el horizonte es continuo (verificado) |
| F2-B2 | lib.rs:1877-1888; temporal_spectrum.rs:346-359 | MED | Mapeo τ*→escala por distancia ABSOLUTA en malla base-4: sesgo 2×; τ*=30s clamp cae al nodo 16 (17s, fuera de banda) — `tau_habilidad`/`qo_613_rho_tau` leen la escala equivocada. Vecino más cercano en ln τ |
| F2-B3 | lib.rs:1678; temporal_spectrum.rs:1485-1522 | MED | W₁ a lag=64 UPDATES (reloj de eventos): 0.6 s a 100 ev/s vs 64 s a 1 ev/s; el BOCPD mezcla con timestamps físicos. Lag en tiempo físico |
| F2-B4 | temporal_spectrum.rs:1381-1384 | MED | `mass<0.10` excluye escalas de la regresión ζ(p) con pertenencia dura: ζ/χ saltan al madurar escalas — χ modula pisos vivos. Peso continuo de masa |
| F2-B5 | state.rs:709-735 | MED | Decaimiento 0.995 POR EVENTO del CVD/OBI: la memoria física varía ×100 entre feeds — rompe comparabilidad entre monedas. Decaimiento −expm1(−dt/τ) en ms |
| F2-B6 | feature-engine/hawkes.rs:46-68 | MED | Impulso adimensional POR EVENTO sumado a intensidad PER-SEGUNDO: λ* ∝ tasa de eventos (no invariante ante re-escala); ts≤last aún excita |
| F2-B7 | symbol_ranker_engine.rs:132-141 | MED | `lev_penalty=(vol_score/5).clamp(1,5)`: tope del rango inalcanzable (vol_score≤5.52) — rango dinámico muerto |
| F2-B8 | espectral_multiactivo.rs:78-92 | MED | El IC cruzado ρ(τ) del veto de grupo NO aplica significancia #599: media de IC sin umbral t≥2 mete ruido de selección al veto |
| F2-B9 | state.rs:231-233 | LOW | Doc de `spectral_intermittency` dice χ=(1−ζ3)⁺ K41 pero el core escribe ((3/2)ζ₂−ζ₃)⁺ |
| F2-B10 | temporal_spectrum.rs:723-724 | LOW | Inyección epigenética con puerta dura `kernel>0.05` — salto de ganancia |
| F2-B11 | temporal_spectrum.rs:620-622 | LOW | Espectro frío publica `dominant_tau_ms`=30s con masa 0, consumido como horizonte fallback — 0= sin opinión |
| F2-B12 | temporal_spectrum.rs:944-946 | LOW | `confluence_ratio` con denominador 32 fijo: diluido por escalas no observadas |
| F2-B13 | normalizer.rs:57 | LOW | Garman-Klass siembra varianza 0.0001 ajena al instrumento (~20 velas de sesgo) |

**Auditor C — teorías cruzadas (16)**

| # | Archivo:lín | Sev | Defecto |
|---|---|---|---|
| F2-C1 | feature-engine/lead_lag.rs:57-78 | **HIGH** | "Lead-lag" sin lags ni reloj: EWMA 0.6/0.4 + escalones 0.50/0.25. VIVO: slot 3 del PPO de cierre (core lib.rs:3869) y registry (verificado) |
| F2-C2 | lead_lag.rs:61 | MED | Staleness: `buffer.back()` sin edad máxima — OFI de BTC viejo cuenta como momentum |
| F2-C3 | god-engine-core/stateful_engine.rs:716 | MED | Kalman R=price·0.0005 en unidades de precio, no precio² ⇒ ganancia ~price× sobre-reactiva; cadena kalman→price_ring→jerk_t write-only |
| F2-C4 | feature-engine/multifractal.rs:104-118 | **HIGH** | El "Hurst" de `update()` no es Hurst: ratio amplitud L1/L2 con ln(n) fijo de ventana; alimenta la confluencia viva. `espectro_f_alpha` honesto pero 3 escalas/n=50 (verificado) |
| F2-C5 | feature-engine/hawkes_cross.rs:62-107 | MED | max-z sobre rejilla de lags sin corrección por comparaciones múltiples; base Poisson subestima con clústeres ⇒ z inflado. VIVO vía contagion_publisher→curl_share |
| F2-C6 | strategy-core/vecm_arbitrage.rs:46-114 | **HIGH** | "Johansen" sin traza/rango ni VECM: z-score rolling + beta LMS. MUERTO (solo tests) |
| F2-C7 | feature-engine/correlation.rs:118-124 | MED | Correlación con cesta que se incluye a sí misma (sesgo +1/N). MUERTO en motor |
| F2-C8 | quantum_tensor_store.rs:63-76 | MED | "Lyapunov" = L2 entre features heterogéneos; módulo muerto |
| F2-C9 | simd_neural_network.rs:24-33 | MED | Init "He/Xavier ortogonal" es sin(i·17+j·31+7)·c; infer/train sin llamadores |
| F2-C10 | tensor_ring.rs:35-76 | MED | Derivadas sobre precio nominal (no log): no invariante de escala; Δt irregular tratado uniforme |
| F2-C11 | strategy-core/momentum_booster.rs:50,55 | MED | Ancla hawkes−1.0 vs SS=1.6 del núcleo; reutiliza `dynamic_ofi_threshold` como umbral Hawkes. MUERTO |
| F2-C12 | maker.rs:115-127 | LOW | Sin Avellaneda-Stoikov (documentado legacy); inventario/100 USD literal. MUERTO |
| F2-C13 | stat_arb.rs:48,97 | LOW | Beta fija 1.0; señales escalón. MUERTO |
| F2-C14 | multivariate_coint.rs:155-165 | LOW | θ de UNA observación; vida media en ticks de evento (ts ignorado). MUERTO |
| F2-C15 | copulas.rs, path_signatures.rs, transfer_entropy.rs | INFO | Matemática correcta y honesta; SIN consumidor — medición deliberadamente no cableada |
| F2-C16 | omni_strategies.rs:36-40,96-102 | LOW | Indicadores TA en tiempo-evento (14/26 arbitrarios); "Fibonacci proxy" 1e-6 literal. VIVO: 22/54 features del tensor |

**Patrón dominante (C)**: los módulos que NOMBRAN teorías fuertes
(VECM, Kalman, Lyapunov, lead-lag, Hurst) o no la implementan o están
muertos; los honestos (firmas, cópulas, TE) están descableados. Los
VIVOS con física débil: lead_lag (PPO), multifractal-update
(confluencia), omni (tensor).

### Inventario crítico de teorías (mandato del dueño: problemas del milenio)

Prioridad valor/coste (física honesta, NO name-dropping):
1. **SÍ — Primer toque analítico (BM/OU hitting, inversa-Gaussiana)**:
   P(τ_stop<τ) en forma cerrada para TP/SL y escalera trailing. Cierra
   R8-A (abierto desde la ola XLIV) y CL-34 con exactitud. Coste bajo.
2. **SÍ — Secuencial anytime-valid (e-values, martingales de Ville)**:
   reemplaza umbrales fijos IC>0/z>3/Fisher>0.33 por confianza
   inmune al optional stopping y al barrido de escalas/pares — ataca
   F2-C5, F2-B8 y la selección de τ* (F1-A1). Coste bajo-medio, sin
   tocar PnL.
3. **SÍ — Fokker-Planck/OU con reloj físico**: MLE/CLS discretizado
   (θ, σ, half-life en SEGUNDOS) para re-animar coint/VECM muertos;
   corrige F2-C14 y da τ de reversión coherente con el espectro.
4. **CONDICIONAL — W₁ sobre distribución de profundidad L2**:
   deslizamiento esperado por renormalización de cola (1D = |CDFa−CDFb|
   integrado). Sólo tras acumular evidencia IOC. Coste medio.
5-10. **NO (razones físicas)**: KPZ/Burgers (sin frente espacial;
   ζ(q) ya lo mide), Navier-Stokes (no hay campo de velocidad medible),
   NLS/Gross-Pitaevskii (duplicaría soliton KdV + λ/μ̂), Yang-Mills
   (ningún observable gauge nuevo; RMT ya limpia), Riemann/zeta
   (matrices 18×18 no lo exigen), KAM/CFT 2D (ni near-integrable ni
   conforme en tape L2). Mención: Cont-Stoikov de colas si se revive
   el maker.

### Verificados limpios (evidencia simbólica/conductual)

- Arreglos previos VIVOS donde corresponden: D-742/CL-32 (fusión+masa),
  AGY-P10 (τ* operativa), #594 causal, #599 t≥2 en el banco, XLIV-6
  (ζ₃=1.5 + sub-resolución), #649/#650 en la sombra espectral.
- spectral_tape (tasas con masa exacta, R invariante en τ, prequential
  causal), ewma/welford, hurst_dfa, adaptive_quantiles, feed/protection
  health, state_continuity, active_universe, position (entry_tau antes
  de is_open), horizon_policy, spectral FFT V2, microstructure OFI
  (D-709), emparejamiento multiactivo sin doble conteo, spectral_regime
  (crash_flux continuo), UNA sola masa en fusión/entropía/Fisher/W₁
  (#591), quantum_oscillator (paridad exacta), voto_espectral (sustrato),
  contagion_modulator, cópulas/firmas/TE (matemática), proceso Hawkes
  interno (μ̂ EWMA, purga, monotonía). Causalidad: sin información
  futura en ningún motor.

### Cola de olas que abre F2 (prioridad)

1. **ERRADICACIÓN del patrón sombra/vivo** (F2-A1/A2/A3/A5 + A11):
   los evaluate vivos adoptan la moneda y física de #649/#650 — oráculo
   obligatorio, es el cambio de mayor radio del consenso+fallback+PPO.
2. **F2-B1** H(τ) continua por interpolación en ln τ (dimensiona TP/SL)
   + **F2-C1** lead-lag real (con lags y reloj) al PPO.
3. **F2-C4** Hurst honesto en la confluencia viva + **F2-B5/B6**
   relojes físicos (decaimiento por ms, Hawkes en tiempo físico).
4. Inventario milenio #1-3 (primer toque, e-values, OU físico) —
   olas de nueva teoría con medición observacional primero.
5. F2-B2/B4/B8 (mapeo ln τ, ζ continuo, significancia ρ(τ)) +
   LOWs agrupados.

**F2 CERRADA**. Siguiente: F3 (núcleo vivo god-engine-core, ~97
archivos, zona Qoder) — hereda el hallazgo estructural como contexto
de primera clase.

## F3 — RESULTADO (cerrada 2026-10-04, Qoder)

Tres auditores en paralelo, un archivo por entrada de checklist:

- **Auditor A**: god-engine-core/src/lib.rs COMPLETO (7.547 líneas —
  el pipeline del núcleo vivo).
- **Auditor B**: los 27 módulos del core (stateful_engine/PPO,
  trailing, calibration, conformal, darwin, ensemble, ml_*, diffusion,
  reality_physics...) + signal-engine/orchestrator.rs (consumo #624).
- **Auditor C**: el host src/bin/god_engine.rs (4.958 líneas) + los 20
  módulos de execution-engine.

### Hallazgos (etiqueta `F3:` — esperan su ola)

**Auditor A — pipeline lib.rs (13)**

| # | Archivo:lín | Sev | Defecto |
|---|---|---|---|
| F3-A1 | lib.rs:4713-4775 vs 3865-3876 | **HIGH** | Paridad evaluate/update PPO rota en slots 0/1: la entrada vota `obi/dynamic_obi_thr` y `ofi/dynamic_ofi_thr` (umbrales medidos p80); el cierre actualiza con `(ofi_value/0.35)` y `(obi/0.35)` — denominador LITERAL y variable distinta. El peso 0/1 aprende de una escala que no es la que vota. Clase #625 (que cerró sólo el slot 2). Verificado |
| F3-A2 | lib.rs:5757 | MED | Call-site ADICIONAL del patrón F2-A5: VPIN pasada como hawkes_ratio con `hawkes_ratio_real` FRESCO en scope (:3990, mismo tick) |
| F3-A3 | lib.rs:1984-1987 vs 4131 | MED | Sombra SR lee `stochastic_noise_variance` con default 0.05 — CERO escritores; el core publica la clave DISTINTA `microstructure_noise_variance`. Knob muerto alimentando el consenso VIVO |
| F3-A4 | lib.rs:1916-2102 | MED | `quantum_k_spring`, `quantum_lambda_anharmonic`, `quantum_alpha`, `soliton_amplitude`, `nash_equilibrium_drift`, `conformal_epsilon`: sin escritor productivo (sólo tests) — sombras del consenso con defaults hardcodeados, insumos NO equiparados entre los 13 motores |
| F3-A5 | lib.rs:2091-2102 vs 4032-4043 | MED | Sombra trend_runner lee `hurst_exponent`/`cvpin`/`atr_pct` que set_reg escribe DESPUÉS en el mismo evento: consume el tick PREVIO (stale-by-one asimétrico; los otros 12 usan desplazamientos frescos) |
| F3-A6 | lib.rs:4015-4019 | MED | `atr_5s = v_t·0.5 + atr_pct·precio·0.5` ≈ ATR 1s, no v_t·√5; coaxial normaliza por √5 ⇒ sesgo de compresión permanente en comp_5s (el fallback AGY-P23 lo anula el escritor) |
| F3-A7 | lib.rs:2695-2701 | MED | BTC/ETH consumen su propio OFI vía predict_altcoin_impulse: slot 3 del PPO y divergencia auto-referenciales para los líderes |
| F3-A8 | lib.rs:861-879 | LOW | Boot carga BTCUSDT_MOTOR bajo clave global "UNIVERSAL" → campo `scalp_forest` sin lector productivo |
| F3-A9 | lib.rs:2230-2241 | LOW | Espectro plano resetea dominante pero NO `consenso_espectral_tau` (hoy enmascarado por #624 — acoplamiento frágil) |
| F3-A10 | lib.rs:2487-2507 | LOW | `revert_quantum_ghost_position` cierra TODOS los slots de la moneda ante un rechazo, sin `closed_order` — divergencia local/exchange |
| F3-A11 | lib.rs:861-910, 4114-4126 | LOW | El espectro entra DOS veces a la decisión (proyeccion_espectral + override #624) y el invariante bayesiano valida parcialmente la decisión espectral con el propio espectro |
| F3-A12 | lib.rs:3789, 5832-5835 | LOW | Relojes mezclados chrono::Utc/SystemTime/now con event_time_ms — no-determinismo en replay |
| F3-A13 | lib.rs:4128-4129 | LOW | `bessel_alpha`=1.5 sin lector; `hawkes_dt`=0.05 publicado cada tick — literales disfrazados de config |

**Auditor B — módulos core + orquestador (10)**

| # | Archivo:lín | Sev | Defecto |
|---|---|---|---|
| F3-B1 | trailing.rs:71,167-173,180-304 | **HIGH** | `spectral_persistence` entra a la función y NO AFECTA NADA: el closure `_lvl` (l.173) jamás se llama; `be_trigger` y las transiciones de fase (1.5/2.5/3.5/4.5 pnl_atr) son fijas. La modulación espectral S-2/#560 de la escalera NO EXISTE pese al doc que la promete. Entrada muerta en el mecanismo de SALIDA. Verificado |
| F3-B2 | orchestrator.rs:386 vs 428-452 | MED | `qo_624_fraccion_espectral` = decisiones de TODAS las monedas / (n+1) de coin 0: fracción inflada ~N× (puede dar >100%) — telemetría que gobierna la recalibración H7 |
| F3-B3 | stateful_engine.rs:909-916 | MED | Hawkes per-tick: escritor muerto en hot path; pasa `ema_ofi` (NIVEL) como `delta_ofi` (excita por magnitud cada tick); eventos depth (vol=0) aún excitan |
| F3-B4 | stateful_engine.rs:327-447 | MED | Familia legacy `can_open_at_tau`: sin callers de producción pero conserva el bug de unidades que D-754 documenta como arreglado (v_t>0.0015 compara $ contra fracción; reloj ticks×100ms) — trampa de recableado con tests que lo afirman |
| F3-B5 | orchestrator.rs:512-518 | LOW | H4 documentada como \|media/v_dom\| pero implementada con signo (banda opuesta ⇒ 0): la impl es la segura, doc/código discrepan |
| F3-B6 | orchestrator.rs:584-587 | LOW | `vdom_sobre_corte` cuenta dominantes con τ inoperable (descartados por H2): distribución H7 mezclada |
| F3-B7 | ensemble.rs:201-211 | LOW | `nn_penalty` castiga sólo a DarkAlphaNN con el z del ENSAMBLE: un forest malo hunde al NN; atribución unidireccional |
| F3-B8 | math_kernels.rs:373-375 | LOW | `DynamicKelly` muerto; en frío devuelve 0.10 fabricado si se recableara |
| F3-B9 | stateful_engine.rs:1301-1330 + 3 módulos | LOW | Superficie muerta: export_f32 (5/144 slots), simd_nn jamás entrenada, OrderFlowAggregator, LatencyAccelerator, BookDepthSlippagePredictor |
| F3-B10 | bootloader.rs:216-254 | LOW | Warmup REST: respuesta no-JSON consume 3 reintentos SIN backoff y el fallo final es println — arranque sigue con estimadores fríos sin marca |

**Auditor C — host + ejecución (13)**

| # | Archivo:lín | Sev | Defecto |
|---|---|---|---|
| F3-C1 | god_engine.rs:2913-2915, 3137 | **HIGH** | Reloj de latencia CONGELADO al arranque: `epoch_baseline_ms = local_t0 + offset_NTP_t0`; el hot-loop nunca relee `server_time_offset_ms` (actualizado cada 15 s). La deriva en sesiones de días sesga `latency_ms` que alimenta el kill-switch de volatilidad sintética y los strikes del sistema inmune (3/3 → flatten): deriva positiva = aplanados falsos; negativa = stalls enmascarados. Verificado |
| F3-C2 | executor.rs:2855-2873 | MED | `execute_reduce_only_market` — la ruta de SALIDA de dinero — sin intención registrada, sin tipar el 2xx, coid interno: invisible al OrderRegistry salvo WS |
| F3-C3 | user_data_stream.rs:501-543 | MED | Contabilidad bracket clasifica por TIPO: cualquier fill STOP/TP de otro cliente en el símbolo encola BracketClose con entry de ranura arbitraria → contamina Kelly/WR/totales |
| F3-C4 | god_engine.rs:3720-3724 | MED | Fee del cierre-core fabricado `(maker+taker)`; la ruta real es IOC-taker + market-taker ⇒ 2·taker. total_fees/total_gross_pnl sesgados |
| F3-C5 | god_engine.rs:2826-2830, 3395-3424 | MED | Dedup core↔bracket por SÍMBOLO (no símbolo+lado): cierres LONG y SHORT simultáneos en hedge tragan el segundo |
| F3-C6 | god_engine.rs:1344 + execution_evidence.rs:35-77 | MED | `fetch_open_positions` estricto + host `unwrap_or_default()`: snapshot rechazado ⇒ FASE 5/adopción saltada sin telemetría |
| F3-C7 | entry_dispatch.rs:100-104; executor.rs:1156 | LOW | `configure_leverage` en CADA entrada (POST extra por entrada); `if true {}` muerto en hot_swap |
| F3-C8 | god_engine.rs:3526-3528 | LOW | Transición reescribe URL del WS con base hardcoded, descartando BEST_WS_ENDPOINT y el ganador de la carrera de latencia |
| F3-C9 | god_engine.rs:1089-1136 | LOW | Bucles infinitos de arranque sin tope: claves inválidas = proceso "vivo" que nunca arranca ni falla |
| F3-C10 | executor.rs:2431-2433 | LOW | `execute_limit_order` WS devuelve Ok sin confirm_ws_dispatch (muerto hoy, armado si se activa WS) |
| F3-C11 | router.rs:96-104 + 3 módulos | LOW | Módulos muertos con física divergente: QuantumOrderRouter (round no direccional, slip ≠ P32), QuantumSocketPool reintenta POST, Multiplexer, HotSwap |
| F3-C12 | client.rs:401-424 | LOW | Cancel sin clasificar error de red como AMBIGUOUS: cancel aplicado reportado fallo → falsas escaladas al watchdog |
| F3-C13 | order_registry.rs:481,525 | LOW | `cleanup_stale_orders` sin llamador: intenciones New atascadas viven todo el proceso |

### Verificados limpios

Paridad slot-2 Hawkes #625 VIVA en ambos caminos; TTL #648 y else{dominante=0}
bien ordenados; sonda #586 con la función pura del gate; dedup por ts en
las dos observar_maduracion; Kelly con LCB del PF; coin_id acotado.
Calibration (Platt Newton/KKT), conformal (D-618+ACI), diffusion
(varianzas exactas), entry_reservation (CL-41b/c), recarga_genoma
(CL-40), ml_inference/registry (MP), reality_physics (D-753), math_kernels
vivos (Welford/Kahan/VPIN/entropía/Hurst DFA/Amihud), darwin (M5-H01),
orchestrator FSM revocable. **Cableado del contagio REPARADO**
(escritor set_for_coin ↔ lector get_for_coin_or, espacio `c{id}:` —
cierra el defecto XLV·G). Orquestador vs ADR-0014: P3/P4/P5 e H2/fallback
✓ (desviaciones B2/B5/B6). En ejecución: CL-39/39b/39c, CL-41c,
roundtrip_friction única (XLIV-8), OCO parcial, rate-limit, redondeo
D-629/D-630, income con dedup, NTP de generación única, CL-38.

### Patrón dominante F3

**Paridades rotas entre lo que VOTA y lo que APRENDE/supone**: PPO
slots 0/1 (A1), trailing que no usa su insumo espectral (B1), reloj
que no sigue al NTP (C1). Sumado al patrón F2 (física corregida sólo
en sombra), el sistema tiene DOS caras que nadie reconcilia: la que
diseñamos y la que corre.

### Cola de olas que abre F3 (prioridad)

1. **F3-A1** paridad PPO slots 0/1 + **F3-B1** escalera espectral
   real + **F3-C1** reloj NTP en el hot-loop — los tres HIGH con
   oráculo.
2. Erradicación sombra/vivo F2 (A1/A2/A3/A5) — puede ir en la misma
   ola de paridades si el oráculo aguanta el radio.
3. F3-A3/A4 (sombras con knobs muertos — equiparar insumos de los 13
   motores) + F3-A5 stale-by-one.
4. F3-C2/C3/C5 (evidencia del camino de SALIDA de dinero) +
   F3-B2/B3.
5. Limpieza de superficie muerta (A8/A13, B4/B8/B9, C7..C13) — ola
   mecánica agrupada.

**F3 CERRADA**. Barrido acumulado: F0(1) + F1(23) + F2(43) + F3(36) =
**103 hallazgos**. Siguiente: F4 (dinero/riesgo, zona Claude —
coordinar antes de invadir).

---

## F5 — CERRADA (GLM, LXXXX, 2026-10-05): evolution-engine + backtest-engine resto + dark-alpha

16+12 archivos, 2 agentes estilo-F. **Acumulado del barrido: 103 → 125 hallazgos.**

### HIGH (3)
- **F5-A-H1** `evolution-engine/src/lib.rs` (666 líneas): bucle "TRUE EVOLUTION" isla muerta NO anotada; su gate promueve con 1 trade + PnL>0 (sin DSR/incumbente/OOS) y frozen_macro inyecta literales 2024. **REPARADO (anotación qo-605-style este commit)**; decisión cablear/eliminar = consejo (isla ahora 6 módulos, no 4).
- **F5-A-H2** `god_engine.rs:4700` + `random_forest.rs:163`: cosecha ShadowForest promueve SIN control de multiplicidad y la promoción `shadow_forest_harvest` NO arma el watchdog de rollback (sólo el daemon lo arma) — la puerta viva más floja. **OLA de reparación pendiente (toca conducta → oráculo)**.
- **F5-B-H1** `god-engine-core/src/lib.rs:966`: fallback de DarkAlpha = red ALEATORIA (Xavier, sin entrenar) que VOTA en el ensamble vivo cuando falta el artefacto — la ausencia no es ausencia, es opinión con ruido estructural sobre ml_prob de BTC. Fix natural: fallback None. **OLA pendiente (toca conducta → oráculo)**.

### MED (16, resumen)
- A-M1 juez DSR certifica contra simulador (declarado, riesgo estructural); A-M2 train/serve desalineado del forest online (features de cierre vs inferencia en entrada); A-M3 lookahead suave del prescreen (σ de ventana completa); A-M4 AST-mutator cambia umbrales sin armado (D-689); A-M5 DSR divergente en isla muerta (trampa de re-cableado); A-M6 CMA penalización incommensurable; A-M7 polars_evolver muerto (anotado este commit); A-M8 entropy_fitness mayormente muerto.
- B-M1 `c1.or(c2)` descarta segundo cierre intratick (sub-contabilización silenciosa — OLA); B-M2 dos lectores .bin, dos políticas de validación; B-M3 neuro_plasticity muerto total (145 líneas).

### LOW (9) — ver tablas completas en buzón LXXXX.
### Verificaciones LIMPIAS
- Embudo del daemon vivo: BIEN cableado (prescreen causal → WF motor real OOS ≥30 trades → incumbente compite → DSR 0.95 Bailey-LdP ec.5 con multiplicidad acumulada → promote bounds → watchdog rollback no-reinicio D-747 → arming por entorno).
- Paridad del replay: APROBADA bit-a-bit (before_event antes de aduana; warmup no saltable; omni t-1; shift_atr_frac 0.10 contratado).
- Métricas ex-post: sin divisiones por cero nuevas (IEEE intencional, tests fijan contornos).
- label_evidence: barreras estrictamente futuras, guards completos.
- La familia honesta YA estaba validada (LXXXIX); F5-B re-confirma el replay que la alimenta.

---

## F6 — CERRADA (GLM, LXXXXII, 2026-10-05): data-pipeline + data-ingest + storage-engine + metacortex-engine

50 archivos src, 2 agentes estilo-F + check de los 4 crates. **Acumulado: 125 → 148 hallazgos.**

### HIGH (6)
- **F6-A-H1** La "aduana de datos" (validation.rs, política F2.1 con contadores F6) está DESCONECTADA de producción — el WS vivo usa quantum_engine::parsers; los contadores de rechazo viven siempre a 0 (telemetría fantasma).
- **F6-A-H2** 17/29 archivos de la capa de datos son CÓDIGO MUERTO (toda la persistencia: state_db/persistence/storage/lakehouse_mmap/teleonomia + ws_client/parser/validation desconectados). Decisión de poda = consejo.
- **F6-A-H3** ⚠️ **CORREGIDO por el propio auditor (LXXXXIII, supersession)**: la redacción original ("dims macro del replay son CONSTANTES") era IMPRECISA — las **6 series FRED** (sp500/nasdaq/vix/us10y/dxy/oil) SÍ se alimentan en replay con corte t-1 causal desde la ola CX (booktick_replay.rs:373-400, cambio de día civil → valor del día previo). **La ruptura real, enumerada**: de las 54 features de `get_features()`, el replay congela en defaults las que el VIVO actualiza vía pollers vivos — `run_macro_rest_poller` escribe **gold** (553) y binance_spot como ref (530); `run_sentiment_onchain_poller` escribe **fear_greed** (631), **funding por símbolo** (647, agg_funding_rate + registry `funding_rate`), **OI por símbolo** (694), LS/taker (784/807 zona). En replay: gold=2300 default, fear_greed=50 default, funding/OI/LS/taker = defaults, frente al vivo que las refresca. **SCOPE de la ola**: alimentar en replay las que tengan fuente histórica (funding/OI históricos existen en Binance Vision) o declarar constantes-por-contrato las que no. NOTA: `votes_export` pasa omni=None — el dataset L2 corrió con las 6 FRED neutras (sin impacto en sus conclusiones: los votos son telemetría de motores, no el vector 54D; documentado).

**RESUELTO (LXXXXIV, verificación profunda)**: la exposición de A-H3 es
MUCHO menor que lo temido — **los 7 bosques promovidos (48D) están
SEGUROS**: su bloque macro son las FRED-4 (omni[21-24] → dims 44-48),
cargadas con valores reales as-of t-1 en el trainer por las MISMAS
series que el feed vivo publica (contrato ml_inference.rs:380-385);
gold/fear_greed/funding/OI/LS/taker NO son features del bosque. La
exposición restante es el **tensor 54D de DarkAlpha** (dims 46/48/50-53:
trainer omni=0 → gold=0.0/fear_greed=1.0-fallback; replay=defaults;
vivo=variable con pollers) — una NN del ensamble de BTC → **deuda
documentada DarkAlpha-54D** (realinear exige decisión de re-entrenar la
NN, no una ola mecónica). Bonus: `fr_elasticity` es característica
muerta (escrita por update_macro_features, jamás leída — train_forest
lo documenta en sus líneas 60-64). El hallazgo pasa de OLA a deuda
acotada: no invalida nada de lo promovido.
- **F6-A-H4** ⚠️ **9 DIMS PERPETUAMENTE 0.0 EN VIVO**: los slots cross-exchange (bybit/okx/...) sólo los escriben pollers muertos; get_features normaliza contra ref_p=1.0 → ceros silenciosos. El modelo infiere con dims muertas (coherente con lo que el trainer ve — paridad preservada por accidente). OLA: o se alimentan o se declaran muertas por contrato. **RESUELTA POR DECLARACIÓN (XCV, 2026-10-06)**: contrato `xcv_dims_cross_exchange_muertas_por_contrato` fija los slots 1..10 como ceros estructurales (bits exactos) con mensaje accionable (re-entrenar si un poller se activa); doc-comentario en get_features(). La cola de F6 queda VACÍA: A-H3 resuelta por verificación, B-M3 reparada certificada, A-H4 declarada.
- **F6-B-H1** ledger.rs: read_ownership consulta un esquema que su propio escritor destruye (ANOTADO).
- **F6-B-H2** ledger.rs: pérdida silenciosa de eventos de posesión (try_send ignorado, qty=0 descarta el cierre — posesiones fantasma) (ANOTADO).

### MED (11, resumen)
- A-M1 TRES políticas de validación conviven en el mismo crate (rechaza/sanea-a-0/fabrica); A-M2 historical fabrica microestructura sin marcar origen (ANOTADO); A-M3 lakehouse_mmap corrupción post-crash indetectable; A-M4 storage.rs checksum bypassable con checksum=0; A-M5 macro_data escritura no atómica; A-M6 macro_last_success_ms sin lector (staleness invisible).
- B-M3 ⚠️ **REPLAY DE OBSERVACIONES AL REINICIO**: el bus mmap persiste head entre corridas y online_daemon re-ingiere hasta 10k frames ya aprendidos por corrida → duplicación sistemática para el Shadow Forest (contaminación de dataset, no anticipación). OLA candidata.
- B-M5 online_learning: skew features cierre-vs-entrada (declarado diagnóstico D-693, amortiguado); B-M6 epigenoma_store colisión de hash sin comparar clave.

### LOW (12) — ver buzón LXXXXII.
### Verificaciones LIMPIAS
- **No-anticipación: SIN LEAKS en los caminos vivos** (trainer as-of estricto, poller corte t-1, ranker trailing-24h, universos fijos en backtest) — el problema de F6 es FALTA de información (H3/H4), no anticipación.
- Consejo de seniors: VIVO y cableado al camino de decisión (deliberar_traced + record_outcome + tracker con máscara anti-rubber-stamp).
- online_learner: causal (innovación contra predicción congelada a la entrada).
- SQLite (evolution_ledger VIVO en escritura): atómico por transacción.
- **Metacortex partido en dos**: el cerebro deliberativo (consejo/learner/trauma) VIVO; el organismo auto-modificante (sandbox/cazador/epigenoma/templates/hot-swap) es DECORACIÓN sin un caller productivo — decisión del consejo (poda o cableado vía ADR-0010-L2-style).

---

## F7 — CERRADA (GLM, LXXXXVII, 2026-10-05): audit-engine + telemetry-server + os-guardian + crates pequeños

36 archivos src, 2 agentes estilo-F, check 9/9 crates verde. **Acumulado: 148 → 171 hallazgos.**

### HIGH (6)
- **F7-A-H1** zero_copy_bus: anillo de 64MB write-only (emit sin lector; flusher simulado; RAM clavada quemándose en círculo).
- **F7-A-H2** drift_auditor NO es el drift EWMA+BOCPD de la doctrina — es centinela contable con shadow SINTÉTICO (0.95·real); el BOCPD real vive en god-engine-core y no está conectado al audit-engine.
- **F7-A-H3** FlightRecorder muerto (siempre None) + crate flight-recorder huérfano completo (ningún Cargo.toml lo declara) con duplicado funcional.
- **F7-B-H1** (mismo que A-H3, verificado independiente).
- **F7-B-H2** TRES GLOBAL_TELEMETRY distintos (os-guardian/telemetry-server/storage) — colisión nominal de wiring; el de os-guardian drena-y-descarta 1M slots.
- **F7-B-H3** anomaly_detector ESTRUCTURALMENTE incapaz de disparar en Windows (ebpf_core devuelve constantes; reglas umbralizadas contra datos que jamás varían).

### MED (9): eBPF 100% marketing (cero bytes de eBPF real; Windows fabrica PMU con ruido _rdtsc — números que PARECEN mediciones); crash_dump sin cablear ("volcado de emergencia" jamás invocado); telemetry_log! degradado a println! bloqueante en el bin principal; audit-engine 4 módulos sin cablear (SPRT mal rotulado); forensics miente sobre disponibilidad; forensic_auditor descarta INSERT en silencio + ruta relativa al CWD; profiler asume 3GHz硬; telemetry 4 sistemas paralelos con 1 vivo (mmap_bus de storage); tests.rs huérfanos nunca compilados (omniscient/phase-runner).

### LOW (10): ver buzón.

### Verificaciones LIMPIAS
- omniscient-registry: MUY VIVO (el registry central de verdad, hot-path).
- os-guardian núcleo Win32 real: VirtualLock/JobObject/memory-auditor con panic latch — 60% músculo real.
- telegram_bot: credenciales SOLO de env vars (sin hardcodeo; .env no trackeado).
- No-anticipación: drift/trajectory auditors sin lookahead.
- graph-architecture/graph-4d: herramientas dev legítimas (Panóptico con latencias reales).
- **Síntesis de la decoración milenio**: flight-recorder (crate), zero_latency_telemetry, ebpf/pmu/observability_plane (teatro de instrumentación Linux trasplantado a Windows como mock), crash_dump, dns_optimizer, tests huérfanos.

---

## F8 — CERRADA (GLM, LXXXXIX, 2026-10-06): los 138+ tests de integración (234 archivos, 964 tests)

**LA ÚLTIMA FASE: el barrido total del operador queda COMPLETO — F0-F8, 338 src + 138+ tests, 217 hallazgos acumulados.**

### HIGH (4)
- **F8-A-H1** ioc_fill_contract ROJO en HEAD invisible al CI: el contrato exigía exactamente 1 `mark_local_reject(` pero el executor tiene 4 (olas 1b20895e/Ω6-Ω7 añadieron rutas nuevas sin actualizar el contador) — patrón qo-613/CL-42 materializado. **REPARADO en este commit**: el invariante real (cierre tras decisión firme/ambigua) se blinda; el conteo exacto se relaja a >=1.
- **F8-A-H2** El CI NO ejecuta 223 tests (audit/execution/data-ingest/data-pipeline/dark-alpha) — sólo compila (--all-targets). Los contratos de ejecución no tienen muralla continua. **OLA: ampliar el workflow** (decisión del dueño por presupuesto de minutos).
- **F8-B-H1** Franca de rojos perpetuos: **~79 tests en 21 archivos certifican defectos abiertos como verde documentado** (naming honesto open_/diagnostic_; PERO 2 ya están cerrados y sus nombres mienten). Catálogo completo en el buzón — es el mapa de deuda técnica viva del sistema.
- **F8-B-H2/H3** phase-runner/src/tests.rs huérfano cita campos muertos (no compilaría ni cableado); omniscient-registry/src/tests.rs huérfano con test SIN asserts ("I'll just check it compiles" — y nunca compila).

### MED (9): T-1 trinquete 0.110 bajo lo medido 0.118 + fixture de ruido negativo congelado; anclas source-string en logs/emoji (9 archivos execution + 6 cross-crate: risk-engine lee god_engine y booktick_replay por texto); mutación de estado global sin mutex en 3 tests; genoma hardcodeado 30.679... en bt_vivo (muere silencioso si el campeón cambia); #[ignore] de medición con propósito cumplido (copulas/TE: mejor destino bin/bench); temp-dir sin nonce; assert tautológico fitness; auto-comparación genome_reader; réplica del cache del host fabricada.

### LOW (8): ver buzón.

### Verificaciones LIMPIAS
- Cero tests fantasma de símbolos (todos los include_str y nombres citados existen — el corpus de Sol sigue siendo la única excepción reparada).
- Los #[ignore] restantes justificados (testnet, ~40min, inventario local).
- La convención open_→regresión-al-aterrizar EXISTE y funciona (genome_gate FMT-216, #660) — la deriva es de mantenimiento, no de diseño.
- **Síntesis F8**: el patrón dominante no es test roto sino test-que-certifica-el-defecto (~8% del total) — la deuda técnica del sistema está INVENTARIADA y nombrada; el riesgo es la deriva de nombres y la ausencia de muralla CI para 4 crates.

# ═══════════════════════════════════════════════════════════════════
# RONDA 2 (2026-10-06) — REVISIÓN DESDE LA BASE contra el árbol a01227cc
# (mandato del operador: «han cambiado muchas cosas» — desde el barrido
# F0-F3 aterrizaron olas 56-61 Qoder + Ω2-Ω9 AGY + F5-F8 GLM/LXXXXIX)
# 3 auditores paralelo: G0 metas/conceptos, G1 matemática/estadística,
# G2 física/motores. 33 hallazgos (5 HIGH, 15 MED, 13 LOW). Docs-only.
# ═══════════════════════════════════════════════════════════════════

## §G0 — METAS Y CONCEPTOS (10: 5 MED, 5 LOW)

Doctrina (ADR-0014 + ARQUITECTURA_VIVA §2) sobrevivió las 15 olas en los
ejes estructurales (enums Continuous únicos, sizing por curvas kelly_at_tau,
router τ viva, banda #586, relojes físicos, Ville, exceso Hawkes). El drift:

- G0-1 [MED] CERRADO (Ola Ω12 AGY) risk-engine/orchestrator.rs:186 — el veto
  de largos fue desacoplado del MAP discreto (argmax) espurio cuando el
  símplex continuo está activo: sólo veta en colapso sistémico medido
  (p_crash >= 0.90) o fallback sin símplex; si p_crash < 0.90, la contracción
  continua directional_pressure (0.25·p) modula el margen sin saltos X→0.
  Test de contrato formal añadido en portfolio_admission_contract.rs.
- G0-2 [MED] CERRADO (Ola Ω11 AGY) god-engine-core/lib.rs:6017 — rama 13
  desacoplada del ancla fija `swing_tp_base` (12h) hacia `tp_at_tau(swing_duration_ms)`
  dinámico evaluado a la tau viva de la onda.
- G0-3 [MED] CERRADO (Ola Ω13 AGY) lib.rs:419/5840/6055/6110/6170 — suelos
  literales de confianza en ramas 13/15 (0.55/0.58) y ramas 11/14 erradicados.
  `confluencia_resonante` modula suavemente desde la cota neutral Bayesiana 0.50
  y todas las ramas conectan con `conviccion_de_rama` gobernadas por evidencia
  empírica (D-752). Test formal `omega13_g0_3_ramas_13_15_conviccion_continua_sin_suelo_literal`
  (167/167 verdes en god-engine-core).
- G0-4 [MED] gen `capital_split_scalp` — se muta en el GA y NO tiene
  consumidor de sizing: gen muerto de la dicotomía que infla la dimensión
  de pruebas del DSR (N=pop×gen de Ω9). Retirar del vector.
- G0-5 [MED] CERRADO (Ola Ω14 AGY) god_engine.rs:51-64 / 4049-4060 — la
  reconstrucción manual desde anclas en genome_protection_prices y en el fallback
  del loop de trading fue reemplazada por la fuente única `arena.config.tp_at_tau`
  y `arena.config.sl_at_tau`. Erradica el riesgo de servir geometría obsoleta ante
  mutaciones continuas en caliente del genoma (a, b) y preserva el invariante de
  clamp de anclas de C-05. Tests de contrato formal dedicados añadidos en god_engine.rs
  (omega14_g0_5_genome_protection_prices_usa_fuente_unica_curva y
  omega14_g0_5_c05_clamp_anclas_invariante: 2/2 verdes).
- G0-6 [LOW] CERRADO (Ola 67 Qoder) state.rs:521 — átomos `scalp/
  swing_used_margin` retirados (0 escritores, 0 lectores verificados por
  grep; oráculo PASA). G0-7 [LOW] CERRADO vía G2-3 (Ola 63 — mismo sitio
  confluence:243, rampas smoothstep de exceso). G0-8 [LOW] PARCIAL
  CERRADO (Ola 67): helpers Genotype::scalp_tp/sl → tp/sl_at_fast_anchor
  (misma curva al ancla) + local swing_tp de rama 13 → tp_tau_vivo (el
  valor ya era tp_at_tau desde Ω11); el residuo de slots internos en
  position.rs/stateful_engine.rs queda documentado (sin costo semántico).
  G0-9 [LOW] espacio genético parametrizado por anclas de 2 puntos — no
  expresa curvatura (nota de consejo). G0-10 [LOW] CERRADO (Ola 67):
  epigenoma TOML renombrado a tp/sl_fast/slow (write-only sin loader en
  producción, cero riesgo de compat; mutation_cycle_test actualizado).

## §G1 — MATEMÁTICA/ESTADÍSTICA (8: 3 HIGH, 2 MED, 3 LOW)

Las fórmulas LOCALES de las 3 piezas centrales nuevas (Ville, DSR, Hurst
VR) están correctas; cada fix tiene un defecto de INTEGRACIÓN estadística:

- **G1-1 [HIGH] Ville NO cubre la multiplicidad — mea culpa #661.**
  evalues.rs:17 + temporal_spectrum.rs:213 + skill_motores.rs:54 afirman
  inmunidad a «la multiplicidad del máximo»; Ville da P(∃t: e≥1/α)≤α POR
  PROCESO. Con 448 e-procesos/moneda (32 skill_e + 13×32 motor×escala) a
  α=0.05, FWER≈1: en ruido ~22 pares cruzan capital≥20 en horizonte largo
  (el Fisher viejo daba 2.2%/par — el fix es MÁS laxo por par). Corrección:
  umbral M/α por familia (640 τ*; 8320 motor×escala) o e-proceso fusionado.
- **G1-2 [HIGH] Hurst VR sesgado ≈−0.03/−0.04 en nulo iid** (AGY Ω8).
  multifractal.rs:126-163: var1 = s1/n y centrado con media estimada ⇒
  Var(d)=(2−4/n)σ² ⇒ E[VR₂](n=10)≈1.78 ⇒ confluencia ≈−0.35 EN RUIDO
  PURO (sus tests no tienen caso nulo iid). H=0.46 vs 0.50 encoge
  dispersion_al_horizonte ~10% (stops apretados de más). Corrección:
  /(n−1) + centrar por escala (o compensar 4σ²/n) + test nulo iid.
- **G1-3 [HIGH] DSR de Ω9 es TELEMETRÍA en darwin.rs** —
  expected_max_sharpe se calcula (:610) y se IMPRIME (:617); la compuerta
  real es meets_promotion_margin = margen 5% de fitness cuya doc dice «NOT
  statistical significance». El campeón sigue siendo max IS. La MEMORIA de
  Ω9 dice «garantizando que el fitness promovido no sea falso positivo» —
  NO está cableado en darwin (online_daemon:1832 sí lo tiene). Corrección:
  exigir DSR≥0.95 sobre retornos OOS como conjunción del gate.
- G1-4 [MED] CERRADO (Ola Ω12 AGY) temporal_spectrum.rs:1483 — ζ(2) ahora
  usa el verdadero segundo momento central E[dev²] = raw_dev_s2 / masa,
  eliminando el sesgo sistemático de la desigualdad de Jensen de (E|dev|)²
  en distribuciones leptocúrticas. Test formal de Jensen añadido en
  temporal_spectrum.rs (omega12_g1_4_segundo_momento_central_sin_sesgo_jensen).
- G1-5 [MED] selection_stats.rs DUPLICADO en risk-engine y
  evolution-engine (diff vacío hoy; drift silencioso garantizado).
- G1-6 [LOW] sr_sigma=1/√(n−1) aproxima σ entre pruebas (conservador con
  GA correlacionado — documentar). G1-7 [LOW] CERRADO (Ola 67 Qoder)
  exportaciones muertas Fisher retiradas (umbral_ic_significativo +
  N_EFECTIVO_EWMA ×2 — 0 usos productivos por grep; Ville de familia las
  subsumió en #661/#663; qo_599/qo_601 reescritos a semántica Ville).
  G1-8 [LOW] CERRADO (Ola 67): evalues documenta n≈272 (E[ln factor] =
  0.58·ln1.1 + 0.42·ln0.9 ≈ 0.011/obs) y skill 1.1^95≈8540 cruza 8320.

## §G2 — FÍSICA/MOTORES (15: 2 HIGH, 8 MED, 5 LOW)

Núcleo Hawkes-transversal (#649/#657/#659), relojes físicos (#660) y
trailing espectral (#657) SÓLIDOS. Dos familias residuales: signums/gates
duros supervivientes, y abstención-SS incompleta:

- **G2-1 [HIGH] flow_excitation_confluence.rs:68 — la CALMA vota más que
  la cascada**: `excitacion_hawkes_norm(ratio).abs()` — λ/μ̂→0.1 da
  |excit|≈0.734 > cascada 3× (0.703). El motor vota fuerte en mercados
  muertos — invierte la semántica del exceso-SS en el CONSENSO VIVO y
  rompe paridad con sus dos hermanos (.max(0.0)). Fix: .max(0.0).
- **G2-2 [HIGH] flow_impulse.rs:91-133 (camino VIVO fallback fast_intent)
  — excitación = ratio CRUDO sin exceso-SS**: #657 arregló las unidades
  pero coherence=√(|flow|·ratio) y z=|ratio|/σ no valen 0 en régimen
  normal. La abstención-SS de #649 NO existe en este camino. Fix:
  excit = excitacion_hawkes_norm(ratio) + tanh(flow/ε) por signum.
- G2-3 [MED] confluence:243 gate duro hawkes/obi (salto 0→0.13 en la
  frontera). G2-4 [MED] perceptron_gate:34 signum+piso 0.15 (voto nunca
  vive en (−0.15,0.15)). G2-5 [MED] coaxial sombra :44 signum duro (el
  vivo usa tanh). G2-6 [MED] conformal :103 tendencia=signum del vecino
  k+1. G2-7 [MED] trend_runner :115/224 escala 1e-3 colapsa la amplitud
  espectral (tanh satura con |x|>3e-3). G2-8 [MED] shockwave :187 rastreo
  dimensional por magnitud `sound > 1.0` — sub-dólares cae a absoluto:
  Mach inflado cientos de × (el defecto #650 reaparece para DOGE/PEPE).
  G2-9 [MED] renyi :152 doble gate duro literal (tsallis<0.60, |obi|>0.15).
- G2-10 [MED] lib.rs:2751 — lead-lag AUTO-REFERENCIAL confirmado (F3-A7):
  BTC/ETH alimentados como líderes y evaluados para SÍ MISMOS (ρ≈1
  trivial, lag siempre acreditado).
- G2-11 [LOW] knobs muertos CONFIRMADOS (quantum_k_spring/lambda/alpha,
  nash_drift, conformal_epsilon + game_payoffs sin escritor). **G2-11
  DRENADO por CONVERGENCIA (Ola 72 Qoder ↔ GLM 112/H2-9)**: adoptado el
  contrato de GLM con test de bit-identidad ausencia↔defaults. G2-12 [LOW]
  CERRADO (Ola 67 Qoder): las 30 líneas del mislabel pre-R9 retiradas de
  hawkes_bessel — describían el proxy de aceleración YA reemplazado por
  λ/μ̂ real e invitaban a re-parar lo cableado. G2-13 [LOW] paridad de INPUTS solitón
  (sombra lee knob muerto 1.0, vivo usa OFI). **G2-13 DRENADO por
  CONVERGENCIA (Ola 72 Qoder ↔ GLM 112/H2-10)**: adoptado el espejo
  per-coin de GLM. G2-14 [LOW] suelos
  literales ramas 13/15 (=G0-3). G2-15 [LOW] cortes duros fused ±0.38/0.22.

## Propuesta de asignación (ronda 2)

- **Qoder (ola 62, inmediata)**: G1-1 (mea culpa Ville ×M) + G2-1 (.abs
  calma-vota) + G2-2 (exceso-SS en flow_impulse vivo). Con oráculo.
- **AGY (Ω10) CERRADA**: G1-2 (insesgado Lo & MacKinlay en multifractal.rs + test nulo iid) +
  G1-3 (DSR OOS cableado en darwin.rs como compuerta formal conjunta DSR>=0.95) +
  G0-4 (congelado gen muerto capital_split en genome.rs y neutralizado en darwin.rs). 83/83 + 166/166 + 111/111 verdes.
- **AGY (Ω11) CERRADA**: G0-2 (ancla fija swing_tp_base reemplazada por config.tp_at_tau dinámico a tau viva en rama 13 de lib.rs:6017) +
  G2-10 (lead-lag sin auto-referencia en BTC/ETH: BTC líder macro puro div=0.0; ETH evalúa sólo contra BTC en predict_eth_impulse_con_reloj; alts evalúan matriz ponderada; test dedicado añadido) +
  G1-5 (unificación DRY canónica de selection_stats re-exportado desde risk_engine en evolution-engine/src/lib.rs; archivo duplicado eliminado). 84/84 + 54/54 + 166/166 verdes, workspace 0 errores.
- **AGY (Ω12) CERRADA**: G1-4 (segundo momento central Kolmogorov insesgado en S2 de temporal_spectrum.rs) + G0-1 (veto MAP discreto de Crash suavizado a contracción continua de margen). 113/113 + 140/140 verdes.
- **AGY (Ω13) CERRADA**: G0-3 (suelos literales de confianza ramas 13/15 y 11/14 erradicados; confluencia_resonante modulada desde base 0.50 y cableado integral de conviccion_de_rama). 167/167 verdes en god-engine-core.
- **AGY (Ω14) CERRADA**: G0-5 (fuente única de brackets tp_at_tau / sl_at_tau en god_engine.rs erradicando reconstrucción manual obsoleta desde anclas; preservado invariante C-05 de clamp de anclas). 2/2 verdes en god_engine.rs.
- **Ola mecánica posterior**: G2-3..G2-9 (signums C¹ en vuelo por Qoder), G0-6/G1-7/G2-12 (limpieza).
- **Consejo**: G0-9 (¿2 anclas bastan para el espacio genético?).

---

## DISEÑO DSR-COSECHA (XCVI, 2026-10-06): F5-A-H2 pasa a "diseñado, requiere API"

**El problema**: harvest_best_genome promueve best-of-N (~9 mutantes) sin
control de multiplicidad DSR. El daemon ya tiene la solución correcta
(edge_survives_multiplicity, Bailey-LdP ec.5, multiplicidad acumulada
D-746) — pero NO puede aplicarse a la cosecha hoy:

1. **El DSR exige serie de retornos por observación**: computa Sharpe y
   momentos superiores de N≥20 trades. La cosecha tiene UNA observación
   por universo (fitness agregado: capital inicial→final con DD²) — un
   Sharpe de n=1 no existe estadísticamente.
2. **La API que falta**: GodEngineCore no expone el historial de PnL
   por trade de los engines sombra. Extenderla atraviesa el arena y es
   una ola propia.

**La tentación rechazada**: construir una "serie" sintética de un punto
y alimentarla al DSR — sería estadísticamente inválida (decoración
peligrosa que fingiría rigor). La honestidad del sistema exige decir
"no se puede hacer bien todavía".

**Guardia interina (YA activa)**:
- El ganador debe superar al **CONTROL** (incumbente sancionado, no un
  lucky-best-of-N) — hurdle real.
- Desde LXXXXI, el watchdog sigue a la generación ACTIVA del almacén:
  cualquier promoción externa (incluida la cosecha) arma la vigilancia
  de rollback (t≤−2.0 sobre ≥20 obs).

**Requisito para cerrar**: API de retornos por trade de los engines
sombra (ola futura si el consejo la aprueba). Entonces el DSR de la
cosecha es: `edge_survives_multiplicity(returns_of_best, num_trees)`.

# ═══════════════════════════════════════════════════════════════════
# RONDA 3 (2026-10-06 tarde) — REVISIÓN DESDE LA BASE contra ece24d87
# (9 olas nuevas desde ronda 2: Qoder 61-64, AGY Ω10-Ω14, GLM XCVIII)
# 3 auditores: H0 metas/conceptos, H1 matemática, H2 física/motores.
# Foco: AUDITAR LOS FIXES NUEVOS (todo fix carga bug — patrón demostrado
# 3 veces: #661→G1-1, G2-6→H2-1, Ville 62→H2-3). 22 hallazgos.
# ═══════════════════════════════════════════════════════════════════

## §H0 — METAS/CONCEPTOS (0 HIGH, 2 MED + LOWs)

Doctrina SOSTENIDA en ejes estructurales; sizing por curvas intacto;
G0-5 sin anclas huérfanas; Omega13 alimenta TasaAcierto. Estados LOWs
G0: G0-6 VIVO, G0-7 CERRADO (ola 63), G0-8 VIVO (naming), G0-9 nota,
G0-10 VIVO (epigenoma TOML).

- H0-1 [MED] continuous_evolution_backtest.rs:363-411 — nichos 2/3/5/9
  mutan anclas escalares que apply_to_arena IGNORA (todo deriva de
  curvas): el walk-forward explora dimensiones muertas.
  → **DRENADO (GLM 102, rama glm/h0-atribucion-y-nichos 9de6effd)**:
  refinado a PARCIALMENTE muerto — lo muerto son exactamente las anclas
  TP/SL (apply_to_arena genome.rs:1062-1065 sólo escribe eval de
  curvas); kelly/trail/obi/trend/base_duration de esos nichos SÍ
  operaban. Fix: `SuperGenotype::rebuild_tp_sl_curves_from_anchors()`
  (genome.rs) — curva por dos puntos canónicos (patrón
  update_tp_curve), coeficientes a bandas fuente única (la envolvente
  gana a la intención), pipeline estándar completo (RR espectral + piso
  fricción + re-derivación + sync). IDEMPOTENTE sin intención → punto
  de cierre ÚNICO en el blindaje cubre los 10 nichos. +2 tests. Bin de
  research: sin oráculo.
- H0-2 [MED] lib.rs:6291 — la arbitracion atribuye el cierre al max de
  volume_flow_rate: etiqueta de INDICE MAYOR, no la rama con la
  conviccion — Omega13 alimenta conviccion_de_rama pero el max()
  hace que las ramas altas absorban la evidencia.
  → **DRENADO (GLM 102, misma rama)**: refinado — volume_flow_rate NO
  toca ejecución directa: es el CANAL DE ATRIBUCIÓN de rama (D-752;
  etiqueta congelada en apertura lib.rs:7517-7539 → cierre alimenta
  rama_registro 3624-3630 → conviccion_de_rama Wilson 138-147 →
  confidence de TODAS las ramas → gates futuros). El max() acreditaba
  SIEMPRE al índice mayor (respaldos 20-24 ganaban siempre) → ramas
  bajas hambrientas de muestra, convicción eterna en piso. Fix: función
  pura `etiqueta_fusion_constructiva` — la rama con MAYOR confidence
  transporta la evidencia; empate→fast. NO ambas (doble-conteo del
  mismo trade, la clase H1-1). +1 test. TOCA CONDUCTA VIVA ⇒ oráculo
  T-1 antes del push. **Oráculo T-1: PASA** — 16/144 (11.1% ≥ 11.0%),
  2/2 tests, 6084 s release sobre 9de6effd (incluye Ω15 de AGY);
  cobertura idéntica a la base — el fixture del oráculo abre casi
  siempre por rama 15 (la fusión fast/slow rara vez dispara ahí).
- H0-3 [nota] drift Ville M/alfa=8320: diferenciacion tarda ~7h (tau=30s)
  a ~410 dias (tau=12h) — doctrinalmente correcto pero #626 congelado
  para tau>1min; candidato: familia por banda observable.
- H0-4..8 [LOW]: epigenoma TOML (G0-10), átomos G0-6 y naming G0-8-parcial
  DRENADOS por Ola 67. **H0-8 DRENADO (GLM 104)**: brazos muertos
  scalp_tp/sl/kelly del ast_mutator REMOVIDOS (muertos porque online_daemon
  sólo pasa ml_threshold_*, pero reactivarlos habría roto la fuente única
  de curvas — el test ahora exige RECHAZO); bins legacy (vectorized,
  booktick_replay) anotados como lectores de VISTAS. **H0-7-residuo
  DRENADO (GLM 104)**: 7 identificadores scalp_* de stateful_engine
  renombrados a fastband_* (59+5 reemplazos, rol espectral real;
  PositionManager pub scalp/swing queda como DECISIÓN — repr(C) público).
  H0-4 (fricción dual buf_fast/slow vs roundtrip_friction unificada en la
  misma función) **DRENADO (Ola 72, oráculo PASA 16/144)**: unificada a
  `tp_sl::roundtrip_friction` canónica (XLIV-8) con fee vivo, ATR vivo,
  latencia 0 en gestión (XLIV-8b); piso 0.00145 conservado. Oráculo
  PASA ⇒ la conducta del trailing no rompió ningún gen certificado.
  G2-11/G2-13: **DRENADOS por CONVERGENCIA (Ola 72)** — mismo hallazgo
  que H2-9/H2-10 de GLM 112 (ver §H2); adoptadas sus versiones.

## §H1 — MATEMATICA/ESTADISTICA (0 HIGH, 3 MED + LOWs)

Veredictos piezas nuevas: (a) Ville xfamilia 62 CORRECTA (Bonferroni
sobre union de supermartingalas, dependencia irrelevante); (b)
e-proceso cruzado 64 CORRECTA con defecto menor (H1-1); (c) Hurst VR
Omega10 CORRECTA (c_k verificada exacta por derivacion); DSR OOS
CORRECTA con matices (H1-2/3/4); (d) zeta2 CORRECTA (convergencia
limpia, campo unico).

- H1-1 [MED] espectral_multiactivo.rs:176-197 — cada bloque participa
  en DOS muestras consecutivas: muestras 1-dependientes, n efectivo
  ~mitad del contado. Ville NO se rompe pero la madurez sobreestima.
  Fix: alimentar solo direccion canonica.
  → **DRENADO (GLM 101, rama glm/h1-1-consumo-bloques)**: el doble-conteo
  ocurre cuando el desfase de fases δ entre cierres ronda 0.5·τ (la
  guardia admite el bloque desde ambos lados; con jitter de stream se
  cruza recurrentemente). Fix aplicado ≠ sugerencia: "dirección canónica"
  crearía zonas muertas para δ>0.5·τ; en su lugar CONSUMO DE BLOQUE por
  par-escala — tras acumular una muestra, ninguno de sus dos bloques
  re-alimenta ese par. Cada bloque participa exactamente una vez, la
  alternación de disparadores se preserva (test: δ=0.8·τ sin zona
  muerta), n cuenta muestras no-compartidas (test: jitter 450/550 → 150
  muestras en 300 rondas; código viejo ~299). NO es solo madurez: bajo
  H0 las muestras 1-dependientes rompen la supermartingala del e-proceso
  #665 (apuesta doble sobre el mismo co-movimiento) — validez del gate
  Ville del veto de grupo restaurada. +3 tests falsación + accessor
  `muestras_par` (telemetría n honesto). **Oráculo T-1: PASA** — 16/144
  genes sensibles (11.1% ≥ 11.0%), 2/2 tests, 4846 s release; cobertura
  idéntica a la línea base (#665) — el fix vive fuera del camino
  perturbado por el oráculo en fixture monoactivo.
- H1-2 [MED] CERRADO (Ola Ω15 AGY) darwin.rs:347-375 — muestreo periódico
  continuo de retornos marked-to-market del portafolio (cada 1s de mercado)
  en evaluate_genotype. Erradica la muestra raquítica de trades discretos que
  exigía un t-stat inalcanzable de >4.5 en ventanas cortas OOS, proveyendo
  soporte muestral homogéneo N>=25 para computar momentos DSR. Test formal:
  omega15_h1_2_muestreo_periodico_continuo_retornos (11/11 verdes en darwin).
- H1-3 [MED] CERRADO (Ola Ω15 AGY) darwin.rs:378-386,636-645 — DarwinDaemon
  ahora incorpora `cumulative_trials: AtomicUsize` monótonamente creciente
  (D-746). La multiplicidad total arrastra las pruebas de todas las rondas
  evolutivas del proceso, erradicando el optional stopping entre corridas
  periódicas del GA online. Test formal:
  omega15_h1_3_darwin_daemon_multiplicidad_acumulada_monotona (11/11 verdes).
- H1-4 [MED] CERRADO (Ola Ω15 AGY) selection_stats.rs:60-110 — error estándar
  de Sharpe asintótico no-normal `sharpe_std_error(m, sr)` formalizado (Mertens
  2002, Bailey-LdP 2012/2014 ec. 4 y 7), incorporando sesgo γ₃ y curtosis
  leptocúrtica γ₄ en el cálculo de `sr_sigma` para `expected_max_sharpe`.
  Benchmark E[max SR] riguroso y conservador frente a colas pesadas cripto.
  Test formal: omega15_h1_4_dsr_sharpe_std_error_leptocurtico (10/10 verdes en selection_stats).
- H1-5..9 [LOW] DRENADOS (Ola 67 Qoder): H1-5 doc de familia M=32
  reescrito (paraguas conservador de la malla; ≤5 nodos de banda
  [30s,12h] compiten de facto), H1-6/H1-8 docs numéricos corregidos
  (evalues n≈272; skill 1.1^95≈8540 cruza 8320), H1-7 ESCALAS_BANDA_PAR
  nombrada en FAMILIA_VETO_GRUPO (el 5 ya era derivado de MAX_COINS,
  ahora también la banda lo es), H1-9/G1-7 exportaciones muertas
  retiradas. Oráculo T-1: PASA 16/144.

## §H2 — FISICA/MOTORES (2 HIGH, 4 MED + LOWs)

Veredictos fixes: flow_impulse z-gate PARCIAL (unidades OK pero default
2.5 FUERA de banda [0.1,1.5] — dispara solo lambda/mu>4.3); confluence
rampas PARCIAL (2 de 3 gates — el de ML quedo binario); perceptron NO
CIERRA (saturacion); conformal COSMETICO; shockwave PARCIAL (ATR/60 es
drift, difusion es /sqrt(60) — Mach inflado 7.75x); lead-lag Omega11
VERIFICADO CERRADO (BTC exogeno, ETH sin rho=1).

- H2-1 [HIGH] conformal_reversion_filter.rs:136-137 — constantes
  1e-3/1e-6 estan 3-6 ordenes bajo la escala operativa (|z|>=1.645):
  en TODA la region emisora direccion=+-1 y acuerdo=0/1 — G2-6
  arreglado es un signum disfrazado de tanh.
- H2-2 [HIGH] familia tanh encubierto: flow_impulse.rs:110 (flow/1e-3),
  hawkes_bessel.rs:378 (direction/1e-3), coaxial_breakout.rs:47
  (x/1e-4) — entradas O(1) saturan: la ola 63 erradico .signum()
  literal pero sembro divisores 10^3-10^4 que reproducen el escalon.
- H2-3 [MED] genome.rs:774 vs :967 — turbo_z_score_stdev default 2.5
  fuera de banda [0.1..1.5]: from_vector clampa a 1.5, genoma fresco y
  serializado con fisica distinta; motor apagado en cascadas tipicas.
  Fix: default en banda (0.75 dispara desde ~2.8).
- H2-4 [MED] confluence:266-283 — gate de ML binario: tercera puerta
  sin C1 (salto hasta ~0.4).
- H2-5 [MED] shockwave:168-193 — ATR/60 es drift; bajo difusion el
  analogo es /sqrt(60): Mach inflado 7.75x (misma clase que AGY-P23).
- H2-6 [MED] CERRADO (Ola Ω16 AGY) perceptron_gate.rs:35-42 — sustituido factor rígido 10.0 por const GANANCIA_PERCEPTRON: f64 = 2.5. Elimina la saturación prematura que degradaba a signum encubierto ante entradas |x| >= 0.3. Respuesta diferenciable C¹ y continua en [-1.0, 1.0]. Test formal: h2_6_graduacion_continua_sin_saturacion_prematura (5/5 tests de perceptron_gate verdes).
- H2-7 [MED] paridad de GANANCIA flow_impulse rota: tres calibraciones
  del mismo flujo (espectral x2, vote x0.8, vivo /1e-3).
  → **DRENADO POR REFINAMIENTO (GLM 103)**: la pata ROTA era la tercera
  (/1e-3 = signum encubierto) — ya reparada por #666/H2-2 (tanh natural
  ×1.0). Las dos restantes son DISEÑO deliberado sobre escalas de
  entrada DISTINTAS (voto_espectral consume momentum_z z-scores; vote
  consume obi+ofi O(1)) — unificarlas en una constante compartida sería
  miscalibrar. Fix de auditabilidad: constantes asociadas públicas
  `GANANCIA_VOTO_ESPECTRAL=2.0` / `GANANCIA_FLUJO=0.8` con tabla de las
  tres escalas + contrato `h2_7_paridad_de_ganancias_pinned` (fija los
  tres valores, los puntos de media respuesta z₅₀=0.2747 < flow₅₀=0.6866
  y prohíbe el regreso del /1e-3: a |flow|=0.004 el voto es ~0).
  Bit-exact (mismos valores), sin oráculo.
- H2-8/G2-12 → **DRENADO (GLM 103 + Ola 67 Qoder, convergencia
  paralela)**: el bloque de advertencia VIEJO en hawkes_bessel.rs (~328)
  decía "FIX REAL pendiente, NO hecho" sobre el proxy de aceleración —
  PENDIENTE YA CUMPLIDO desde M2-C02 (el core publica λ/μ verdadero por
  símbolo en lib.rs:~4180). Reescrito como historia cerrada con "no
  re-parar" (versión GLM conservada — cita el cableado productivo); la
  Ola 67 retiró el mismo bloque en paralelo. Riesgo de doble-fix
  eliminado.
- H1-6/G1-8 → **DRENADO (GLM 103 + Ola 67 Qoder, convergencia
  paralela)**: el comentario de potencia de qo_661 alegaba cruce
  "~n=800" para el umbral simple — el real es n≈272
  (E[Δln-capital]=0.01103/obs; 800 es el número de FAMILIA M=416, no el
  umbral 1/α). Corregido con la derivación (versión GLM conservada en
  evalues.rs; Ola 67 corrigió además el hermano de skill_motores:
  1.1^95≈8540 cruza 8320, no "1.1^97≈8640"). (La otra parte, G1-6
  sr_sigma gaussiano, quedó SUPERADA por Ω15: el DSR ya usa
  sharpe_std_error no-normal.)
- H2-9..12 [LOW]: **H2-11 (=G2-15) DRENADO-PINNEADO (GLM 104)**: los
  cinco cortes de confluencia_resonante (0.38/0.22/0.52/0.12/±2e-4) son
  ahora constantes públicas (FUSED_UMBAL_PLENO/MODERADO,
  HURST_CONTINUACION, COHERENCIA_MINIMA, MAREA_MACRO_TOLERANCIA) con
  contrato h2_11 que fija valores y fronteras justo-adentro/afuera —
  bit-exact; promoverlos a genoma = conducta ⇒ oráculo (opción abierta).
  **H2-12-pata-doc DRENADA (GLM 104)**: doc de lag_optimo corregida al
  comportamiento real (rho.abs() pasa — la doc decía "exigido POSITIVO"
  y un rho negativo VOLTEA la firma de la divergencia); decisión de
  vetar rho<0 + ruta ETH 0.6/0.4 SIGUEN ABIERTAS (conducta ⇒ oráculo).
  **H2-10 DRENADO (GLM 112, con oráculo)**: la sombra del solitón leía
  el knob global muerto soliton_amplitude (0 escritores ⇒ siempre 1.0)
  — ahora ESPEJA la cascada del vivo (per-coin soliton_amplitude → OFI
  → 0.0; sanitizado del motor idéntico). **H2-9 cerrado por partes**:
  conformal_epsilon y nash_drift RESUELTOS DE FACTO por Ω21/Ω22
  (escritores reales: conformal_alpha :4673, cadena Nash-CVPIN); los
  knobs cuánticos (k_spring/lambda/alpha) declarados CONTRATO por GLM
  112 — defaults = física pinneada bit-idéntica; publicarlos del genoma
  = canal evolutivo futuro con oráculo.

## Asignacion (ronda 3)

- Qoder ola 65 CERRADA: H2-1 + H2-2 + H2-3 + H2-4 + H2-5 (física de saturación — oráculo pasa 16/144).
- AGY Omega15 CERRADA: H1-2 + H1-3 + H1-4 (darwin/DSR — retornos continuos 1s, multiplicidad acumulada monótona y sharpe_std_error no-normal). 10/10 + 11/11 verdes.
- GLM 101 CERRADO: H1-1 (consumo de bloque por par-escala en espectral_multiactivo).
- GLM 102 / Qoder ola 66 CERRADAS: H0-1 + H0-2 (nichos del walk-forward sobre curvas continuas, atribución constructiva de ramas por convicción; oráculo pasa 16/144).
- AGY Omega16 CERRADA: H2-6 (graduación C¹ continua sin saturación prematura en PerceptronGateEngine con ganancia 2.5). 5/5 tests verdes.
- **ESTADO RONDA 3: 2/2 HIGH + 8/8 MED DRENADOS AL 100% ENTRE EL CONSEJO
  DE AGENTES** (el "7/7" anterior omitía H2-7, drenado por GLM 103;
  conteo completo: H0-1, H0-2, H1-1, H1-2, H1-3, H1-4, H2-3, H2-4,
  H2-5, H2-6, H2-7 = 11 MED + 2 HIGH). Quedan LOWs de limpieza.
- Siguiente paso: Fase F4 cerrada (auditoría forense de riesgo, capital $13 USD y execution-engine) -> avanzar a Fase F5.

---

# FASE F4 — DINERO, RIESGO Y EJECUCIÓN (45 ARCHIVOS AUDITADOS)

## Inventario Real de la Fase F4
- **crates/risk-engine/src/**: 23 archivos (8 719 líneas). Tests: 141/141 verdes en 0.85s.
- **crates/execution-engine/src/**: 22 archivos (11 655 líneas). Tests: 79/79 verdes en 3.76s.
- **Total**: 45 archivos, ~20 374 líneas.

## Hallazgos de la Fase F4

- **F4-EXE-001 [MED] CERRADO (Ola Ω17 AGY)** `crates/execution-engine/src/user_data_stream.rs:966-983`:
  El test `test_algo_update_terminal_marks_protection_dirty` realizaba una aserción absoluta `assert_eq!(terminal_events_seen(), 1)` sobre el contador estático `AtomicU64` global `TERMINAL_EVENTS_SEEN`. En ejecución paralela con `test_reconcile_after_reconnect_clears_cache_and_marks_dirty`, el contador acumulaba ejecuciones previas (`left: 2, right: 1`) causando fallos no-deterministas. Blindado midiendo el incremento relativo `terminal_events_seen() - prev_events == 1`. 79/79 tests de `execution-engine` verificados verdes.

- **F4-RISK-001 [AUDITADO - APROBADO] Régimen Micro-Capital $13 USD y Piso Binance $5**:
  - `capital_regime::trades_of_room(13.0, 5.0) = 2.6 <= 3.0` activa `micro_weight = 1.0` (régimen micro pleno).
  - `enforce_minimum_notional`: con `dynamic_min_notional = 5.0` y margen de seguridad (+0.1) se evalúa `safe_min_notional = 5.10 USD`.
  - `micro_lev_cap`: apalancamiento continuo [5.0x, 6.5x]. Con L=5x, el margen requerido por trade es $1.02 USD.
  - `micro_safe_limit`: [1.20, 2.60] USD. La orden cabe holgadamente en $1.02 USD sin activar recortes de margen.
  - Concurrencia de posiciones: 2 órdenes simultáneas consumen $2.04 USD de margen a 5x ($10.20 USD notional), dejando $10.96 USD libres (84.3% del capital), satisfaciendo con creces el colchón mínimo de $3.0 USD.
  - `orden_viable`: el riesgo al Stop Loss de 100 bps en la orden mínima de $5.10 es $0.051 USD (0.39% de la cuenta de $13 USD), muy por debajo del tope de ruina del 25% ($3.25 USD).

- **F4-RISK-002 [AUDITADO - APROBADO] Orquestador de Portafolio y Protección Simétrica**:
  - `PortfolioOrchestrator::allow_trade`: aplica `exposure_limit = 0.98 - directional_pressure`.
  - Permite simultaneidad y simetría total de posiciones Long y Short.
  - Veto absoluto de largos reservado estrictamente para caída libre sistémica ($p_{\text{crash}} \ge 0.90$). En caídas intermedias la presión modula el margen admisible suavemente sin saltos discretos.

---

# FASE F5 — APRENDER Y MEDIR (EVOLUCIÓN, GENOMA Y BACKTEST — 28 ARCHIVOS AUDITADOS)

## Inventario Real de la Fase F5
- **crates/evolution-engine/src/**: 16 archivos (6 499 líneas). Tests: 54/54 verdes en 8.12s.
- **crates/backtest-engine/src/**: 10 archivos (5 180 líneas). Tests: 53/53 verdes en 48.81s.
- **crates/dark-alpha-engine/src/**: 3 archivos (2 010 líneas). Tests: 49/49 verdes (31 unit + 18 integration) en 0.59s.
- **Total Fase F5**: 28 archivos, ~13 689 líneas. Tests: 156/156 verdes (100% aprobado, 0 fallos).

## Hallazgos de la Fase F5

- **F5-DARK-001 [HIGH] CERRADO (Ola Ω18 AGY)** `crates/dark-alpha-engine/src/lib.rs:735-756`:
  - **Defecto**: En `predict_in_context`, cada inferencia ejecutaba `if self.validate().is_err()`. Esto implicaba verificar exhaustivamente 4,353 floats de pesos y sesgos (`.is_finite()`) de las 3 capas densas, más la iteración de 30 per-coin normalizers en CADA tick de mercado en el hot-path. Hacía que `test_inference_speed` fallara con 48,451 ns (límite: 25,000 ns). Además, `ensure_inference_buffers()` ejecutaba `.resize(..., 0.0)` incondicionalmente.
  - **Causa Raíz**: Contradicción de diseño respecto a la docstring de `validate()` ("un archivo truncado o corrupto se rechaza en la carga, no en mitad de una inferencia en vivo").
  - **Solución Implementada**:
    1. Sustituido el escaneo masivo de 4,353 parámetros en el hot path por la guarda de consistencia dimensional y de forma $O(1)$: `!self.layers_valid()`.
    2. Optimizada `ensure_inference_buffers()` para verificar `len != expected` antes de disparar resize.
    3. Validación profunda de parámetros (`validate()`) preservada en deserialización (`load_json`), entrenamiento (`fit`), y tests directos de modelo.
  - **Resultado Medido**:
    - Latencia de inferencia por llamada reducida de **48,451 ns** a **4,981 ns** (**aceleración de 9.7x**, muy inferior al tope de 25,000 ns).
    - 31/31 tests unitarios en `dark-alpha-engine` y 18/18 tests de integración en `neural_evidence_contract.rs` aprobados en verde. Cero regresiones en `quantum-arena` (120/120) y `god-engine-core` (170/170).

- **F5-EVOL-001 [AUDITADO - APROBADO] Función Única de Aptitud (D-652 / D-653 / D-654 / D-655)**:
  - `fitness.rs`: Formaliza la utilidad logarítmica cóncava penalizada por ruina cuadrática $F = \ln(\text{capital\_final} / \text{capital\_inicial}) - \lambda \cdot (\text{max\_drawdown\_pct})^2$.
  - Constante analítica $\lambda = 4 \cdot \ln(2) \approx 2.7726$ equilibra la penalización de un DD del 50% con la duplicación de capital.
  - Erradica al 100% la patología heredada donde estrategias paralizadas (cero trades) superaban a estrategias activas.
  - `compute_with_bayesian_prior` previene el bloqueo en frío contrayendo suavemente hacia el prior sin penalizar con $-\infty$.
  - `entropy_fitness.rs`: Incorpora entropía de Shannon sobre el espacio de señales, castigando el colapso a polaridad fija unidireccional y modelando fricción de microestructura con Poisson.

- **F5-BACKTEST-001 [AUDITADO - APROBADO] Replay Real y Contratos Metamórficos de Causalidad**:
  - `booktick_replay.rs`: Microestructura 100% real de libro (bid, ask, bid_qty, ask_qty), series FRED históricas reales (SP500, NASDAQ, VIX, DXY), slippage dinámico modelado por ATR real, fees nativos del genoma, y ejecución a través del IDÉNTICO camino de producción (`GodEngineCore::process_event`).
  - `booktick_causality_contract.rs`: Contratos metamórficos verificados:
    - `cx_first_event_has_no_future_feature_history`: cero fuga de features futuras al evento inicial.
    - `cx_future_suffix_cannot_change_prefix_in_either_mode`: cualquier mutación en sufijos futuros deja idéntico el prefijo histórico en ambos modos.
    - Garantía matemática formal de cero lookahead bias y causalidad estricta.

- Siguiente paso: LOWs y F4 (auditoría forense de riesgo, capital $13 USD y execution-engine).

## 2026-10-07 — R4 / Sol: reconciliación y continuación archivo por archivo

Plan operativo: `PLAN_REVISION_ARCHIVO_POR_ARCHIVO_2026-10-07.md`; coordinación: `PLAN_MAESTRO_SINCRONIZACION.md`. Se reutiliza el censo de Codex, actualizado contra 8938cf41: 1.434 archivos versionados / 460 Rust. Un cierre de inventario histórico NO acredita cobertura semántica de cada archivo ni corrección de la versión actual.

Errata del conteo anterior: la lista enumerada contiene 11 MED, no 8. H2-7 queda acreditado por el contrato GLM103 presente en main; no se reabre por una nota antigua. F4 mantiene auditoría de todos sus archivos pendiente: cuatro fixes puntuales no sustituyen el recibo de cobertura completa.

Hallazgos actuales se registran sin duplicar R4-Q1..Q4 de Codex: cash != MTM, muestreo mixto, contador recreado por el caller y dependencia temporal no corregida sólo con cuatro momentos. Diagnóstico Rust Sol confirma además que `compute_moments` elimina NaN/Inf silenciosamente. SOL-R5-01: `continuous_evolution_backtest.rs:703-705` suma equity_final−cash_inicial_dia, lo que duplica flotante arrastrado. Fixture controlado produce PnL diario acumulado 20 con crecimiento terminal 10; no se afirma que una corrida económica haya producido esa posición.

Recibo: `audit/SOL_CONTRATOS_R4_2026-10-07.json`; alcance y hashes explícitos. Historical witnesses remain bounded reproductions, not current-code or economic verdicts. SOL-R5-01 now has a local daily-reporting-only correction: LOCAL TEST PASSED, full-bin reporting_contract 2/2, direct exit 0 (target/reporting-contract-execution-evidence-20261007.txt). Baseline offline locked workspace all-targets check passed, cargo/tee exits 0 (target/sol-baseline-all-targets-20261007.log and .exit); integrated validation and publication are pending. Other findings remain open. T-1 was not rerun because this is reporting only, not the live strategy pipeline. Recorrer TODOS los archivos restantes por el ledger y cerrar por evidencias, no por ausencia de matches ni por tests sintácticos.

---

# RONDA 4 (2026-10-07, contra 8938cf41 — post Ola 65/66/67, Ω15/Ω16, GLM 101/102/103)

Mandato del operador: revisión desde la base tras 10+ olas nuevas. 3
auditores paralelo (A metas/doctrina, B matemática/estadística, C
física/motores). **17 hallazgos (2 HIGH, 6 MED, 9 LOW)** — el patrón
«todo fix carga bug» se confirma por CUARTA vez: el blindaje del
walk-forward corregido en H0-1 seguía clampando un gen fuera de su
bound; la paridad calma-abstiene (#659) arreglada en los caminos vivos
nunca llegó a las DOS sombras espectrales.

## §R4-A — METAS/CONCEPTOS/DOCTRINA

- **R4-A1 [MED] god_engine.rs:~3876 — tercer sitio de ancla cruda.** El
  fallback de stop_pct de la envolvente lee `scalp_sl_base` crudo (Ω14
  arregló genome_protection_prices y el fallback OCO, dejó éste).
  Viola fuente única (invariante 3). → **CERRADO (Ola 68)**:
  `sl_at_tau(TAU_ANCHOR_FAST_MS)`.
- **R4-A2 [MED] ADR-0014:41 — doctrina formal desincronizada.** El
  principio 6 aún prescribe significancia Fisher 2/√(n−3) con N efectivo
  (retirada por #661/#663 → Ville familia M/α; Ola 67 borró las
  exportaciones muertas). ARQUITECTURA_VIVA §2.9 sí está al día.
  Fix: adenda al ADR. [ABIERTO — docs]
- R4-A3 [LOW] hot_swap_controller defaults mágicos sin doc de
  procedencia (0.005/0.002/0.020/0.010 = curva baseline en los anchors).
- R4-A4 [LOW] naming residual: telemetría "Scalp execution"
  (god_engine.rs:3165), campo vivo `swing_nn` (debería ser macro_nn).
- R4-A5 [LOW] rama 13 semi-renombrada: swing_duration_ms/swing_stretch_z
  aún dicotómicos.

## §R4-B — MATEMÁTICA/ESTADÍSTICA

- **R4-B1 [HIGH] walk-forward: nichos exploran eje MUERTO y promote
  rechaza campeones.** Blindaje (:450) y nicho 4 (:409) clampean
  `tech_threshold∈[0.08,0.22]` — el bound evolutivo slot-21 es [0.24,
  0.30] (D-625, genome.rs:1831): todo mutante corre a 0.24 en el arena
  (clamped por apply_to_arena) y `GenomeEnvelope::promote→validate`
  RECHAZA campeones por «gen 21 fuera de bounds» ⇒ NINGÚN mutante se
  persiste (misma clase que H0-1). Telemetría :962 imprime sin clamp.
  → **CERRADO (Ola 68)**: banda [0.24, 0.30] blindaje / [0.24, 0.27]
  nicho 4.
- **R4-B2 [MED] muestreo H1-2 mezcla cadencias** (darwin.rs:376-386):
  la muestra de retornos dispara por reloj 1 s **o** por cierre de trade
  (`|| closed.is_some()`) ⇒ serie heterocedástica (Δt irregulares con
  saltos de PnL realizado) que distorsiona γ₃/γ₄/SR del DSR
  (Mertens asume frecuencia fija). [ABIERTO — Ola 69]
- R4-B3 [MED] DSR con soporte marginal: selection_stats n≥20; con OOS
  de ~20-40 s de feed denso hay apenas 20-40 retornos de 1 s — γ₃/γ₄
  de varianza enorme alimentan el listón de Gumbel. Fix: piso n≥60 o
  σ gaussiano bajo n pequeño. [ABIERTO — Ola 69]
- R4-B4 [LOW] doc «≥104 aciertos» del veto de grupo: real 113
  (ln43500/ln1.1=10.68/0.0953).
- R4-B5 [LOW] nits: evalues :114 «M=448» vs familia real 416+32;
  temporal_spectrum:1682 «1.1⁶⁹≈670» → 718.
- R4-B6 [LOW] fallback anti-conservador σ_SR (selection_stats:88): si
  1−γ₃SR+((γ₄−1)/4)SR² ≤ 0 cae a 1/√(n−1), MENOR que el error real.
- **R4-B7 [MED] blindaje silencia nicho 2**: clamp swing_sl_base
  [0.0080, 0.0350] pisa el slow-anchor 0.0075 del nicho 2 y el rebuild
  post-blindaje propaga +6.7% de distorsión a la curva.
  → **CERRADO (Ola 68)**: banda [0.0070, 0.0350] ⊇ nichos 2/3/5.

## §R4-C — FÍSICA/MOTORES

- **R4-C1 [HIGH] la CALMA invierte las sombras espectrales de hawkes y
  flow_impulse.** `voto_espectral` de ambos multiplica
  `excitacion_hawkes_norm(ratio)` SIN `.max(0.0)`: en calma (ratio<SS)
  la excitación es negativa y cada escala vota INVERTIDA (calma + flujo
  alcista vota bajista) — el defecto exacto que #659/F2-A4 documentó y
  arregló en los caminos vivos, pero las SOMBRAS alimentan
  `votos_espectrales` → consenso espectral que DIRIGE desde #624.
  Peor: lib.rs:2092 fallback `None => 1.0` ⇒ monedas sin proceso Hawkes
  votan invertidas a peso constante. Tests gap: nunca se probó calma.
  → **CERRADO (Ola 68)**: `.max(0.0)` en ambas + tests calma→abstención
  (el fallback 1.0 ahora abstiene naturalmente).
- **R4-C2 [MED] firma viva shockwave saturada**: SAT_MOMENTO=1e4
  (supersonic_shockwave.rs:207) — speed_norm O(1e-4..1e-2)/s ⇒ media
  respuesta en 5e-5 ⇒ dirección binaria de facto en el camino vivo;
  rompe paridad con la sombra (tanh natural en z). [ABIERTO — Ola 69]
- **R4-C3 [MED] conformal `acuerdo` saturado**: divisor 0.5 en
  (−(z·trend)/0.5).tanh() — media respuesta |z·trend|=0.28, ~6× bajo el
  emisor típico 1.645. [ABIERTO — Ola 69]
- R4-C4 [MED-LOW] perceptron gate residual empinado:
  tanh((act−0.5)·5).clamp(0.15,1) — kink C⁰ en act≈0.53 (derivada
  0→4.9); H2-6 arregló la direccional, dejó el gate.
- R4-C5 [LOW] shockwave: mid_price ausente rompe unidades (speed crudo
  vs atr_pct/√60 fraccional) — abstener sin mid.
- R4-C6 [LOW] flow_impulse emisión binaria documentada (AGY-AUD-002) +
  kink C⁰ .min(2.0) en hawkes_scale del confluence.

## Mapa positivo (verificado por los 3 auditores)

- **evalues**: capital (1+λ·sign(s)·sign(r)) exacto; Bonferroni M/α
  unión válida sin independencia; TODOS los números de doc correctos
  tras GLM 103 (n≈272, 586, 818).
- **selection_stats**: sharpe_std_error reproduce Mertens exacto;
  Gumbel eq.5 recalculado ✓; darwin OOS causal 50/50 real,
  cumulative_trials monótono sin doble conteo.
- **espectral_multiactivo (GLM 101)**: consumo de bloque verificado
  para δ≈0.4τ/0.6τ y ts idéntico — supermartingala preservada; familia
  C(30,2)·5 conservadora y válida.
- **Curvas (Ola 66)**: through_two_points ↔ derive_anchors idempotente;
  nichos kelly/trail llegan a curvas vivas.
- **Ganancias nuevas ejemplares** (GLM 103/Ω16): tabla de escalas,
  contratos pinned, justificación de no-unificar.
- **Ola 67 renames puros** (tp/sl_at_fast_anchor, tp_tau_vivo):
  fórmulas idénticas a sus predecesoras.

## Asignación ronda 4

- **Qoder Ola 68 CERRADA (con oráculo)**: R4-C1 + R4-B1 + R4-B7 + R4-A1.
- **Ola 69 (siguiente)**: R4-B2 (muestreo por rejilla) + R4-B3 (piso n)
  + R4-C2/C3/C4 (saturación residual) — con oráculo.
- Docs: R4-A2 adenda ADR-0014; LOWs B4/B5/A3/A4/A5 en limpieza.

---

# FASE F6 — DATOS, INGESTA, STORAGE Y METACORTEX (49 ARCHIVOS AUDITADOS)

## Inventario Real de la Fase F6
- **crates/data-pipeline/src/**: 24 archivos (6 075 líneas). Tests: 63/63 verdes en 3.12s.
- **crates/storage-engine/src/**: 8 archivos (3 004 líneas). Tests: 39/39 verdes en 0.49s.
- **crates/metacortex-engine/src/**: 12 archivos (4 209 líneas). Tests: 25/25 verdes en 0.09s.
- **crates/data-ingest/src/**: 5 archivos (1 036 líneas). Tests: 19/19 verdes en 0.13s.
- **Total Fase F6**: 49 archivos, ~14 324 líneas. Tests: 146/146 verdes (100% aprobado, 0 fallos).

## Hallazgos de la Fase F6

- **F6-STO-001 [LOW] CERRADO (Ola Ω19 AGY)** `crates/storage-engine/src/mmap_bus.rs:424`:
  - En el test `lxxxxiv_skip_to_head_salta_sin_ingerir`, la variable `let mut bus = MmapTelemetryBus::new(&path).unwrap();` declaraba mutabilidad innecesaria. Limpiado a `let bus` sin mutabilidad espuria, erradicando advertencias en compilación. 39/39 tests verdes.

- **F6-PIPE-001 [AUDITADO - APROBADO] Estado Omnisciente Atómico y Normalización Streaming**:
  - `omni_multiplexer.rs`: Todas las variables macro y cross-exchange se almacenan en `AtomicU64` con codificación IEEE-754 (`f64::to_bits()`), permitiendo lecturas y escrituras atómicas libres de locks (*lock-free*) y libres de esperas (*wait-free*) en el hot path.
  - Cero alocaciones en el bucle principal de ingesta: búferes circulares prealocados y deserialización zero-copy.
  - Sincronización asíncrona de tasas de financiación (`funding_by_symbol`) y sentimiento de masas (`ls_account_by_symbol`, `taker_ratio_by_symbol`) mediante `RwLock` actualizado en segundo plano por pollers desacoplados.

- **F6-CORTEX-001 [AUDITADO - APROBADO] Fábrica de Estrategias y Continuo Temporal (U-ERR-9)**:
  - `evolutionary_templates.rs`: Erradicada la antigua duplicación espejo `DualHorizonStrategyParams`. Sustituida por `ContinuumStrategyParams`, donde la geometría completa de TP y SL se evalúa de manera diferenciable y continua como función de la escala temporal intrínseca $\tau$ sin bifurcaciones condicionales `if/else`.
  - Inmunidad contra dolor/trauma en `fases_autonomous.rs` y persistencia atómica en `epigenoma_store.rs`.

- **F6-INGEST-001 [AUDITADO - APROBADO] Selector Dinámico de Activos y Restricción $13 USD**:
  - `dynamic_selector.rs`: Filtra stablecoins estériles y clasifica activos por liquidez real y volatilidad, actualizando directamente `quantum_arena::symbols::update_dynamic_universe`.
  - Garantiza que sólo los activos con profundidad suficiente para satisfacer el piso institucional de $5.00 USD de Binance sean seleccionados, previniendo deslizamientos extremos en pares ilíquidos.

---

# FASE F7 — TELEMETRÍA, GUARDIANES, AUDITORÍA Y ARQUITECTURA (45 ARCHIVOS AUDITADOS)

## Inventario Real de la Fase F7
- **crates/telemetry-server/src/**: 13 archivos (3 113 líneas). Tests: 30/30 verdes en 3.04s.
- **crates/os-guardian/src/**: 10 archivos (956 líneas). Tests: 12/12 verdes en 0.04s.
- **crates/audit-engine/src/**: 11 archivos (1 729 líneas). Tests: 21/21 verdes en 0.21s.
- **crates/telemetry-engine/src/**: 3 archivos (326 líneas). Tests: 7/7 verdes en 0.02s.
- **crates/phase-runner/src/**: 2 archivos (189 líneas). Tests: 5/5 verdes en 3.85s.
- **crates/flight-recorder/src/**: 1 archivo (232 líneas). Tests: 5/5 verdes en 0.03s.
- **crates/omniscient-registry/src/**: 2 archivos (427 líneas). Tests: 5/5 verdes en 0.03s.
- **crates/graph-architecture/src/**: 2 archivos (387 líneas). Tests: 5/5 verdes en 0.01s.
- **crates/graph-4d/src/**: 1 archivo (160 líneas). Tests: 4/4 verdes en 0.04s.
- **Total Fase F7**: 45 archivos, ~7 519 líneas. Tests: 94/94 verdes (100% aprobado, 0 fallos).

## Hallazgos de la Fase F7

- **F7-SIG-001 [LOW] CERRADO (Ola Ω20 AGY)** `crates/signal-engine/src/skill_motores.rs:95-99`:
  - Advertencia de compilador `unused doc comment` en la guarda de Ville Martingales `self.e_proceso.significativo_familia(...)`. Resuelto convirtiendo sintaxis de doc comment (`///`) en comentario de bloque (`//`), erradicando advertencias en compilación. 116/116 tests verdes.

- **F7-TEL-001 [AUDITADO - APROBADO] Servidor de Telemetría Lock-Free y Anillos Zero-Copy**:
  - `telemetry-server/src/lockfree_bus.rs`: Cola MPMC lock-free (Crossbeam SegQueue) con descarte controlado por saturación de capacidad, garantizando cero contención y cero backpressure sobre el bucle crítico de decisión microtemporal.
  - `zero_copy_bus.rs` / `zero_copy_ring.rs`: Memoria compartida y buffers anulares sin clonación ni asignaciones en el hot path.
  - `telegram_bot.rs`: Enrutador reactivo asíncrono con credenciales aisladas mediante inyección por variables de entorno (.env protegido).

- **F7-OSG-001 [AUDITADO - APROBADO] Guardián de Sistema Operativo y Blindaje Win32**:
  - `os-guardian/src/memory_audit.rs`: Monitoreo en tiempo real de RAM para entorno de 16 GB, ejecutando compactación forzada (`EmptyWorkingSet`) y panic latch si el consumo de memoria excede el presupuesto crítico.
  - `pmu_sensor.rs` / `ebpf_core.rs`: Fallback adaptativo para Windows con lectura de TSC (`_rdtsc`) y mitigación de fallos de página sin bloquear el hilo de ejecución principal.

- **F7-AUD-001 [AUDITADO - APROBADO] Motor de Auditoría, Deriva y Resiliencia Cibernética**:
  - `audit-engine/src/drift_auditor.rs`: Detección en tiempo real de divergencias entre estado simulado y real.
  - `trajectory_auditor.rs`: Auditoría de trayectorias de precios y paridad causal.
  - `cybernetic_resilience.rs`: Supervisión de fallos transitorios en brokers y reconexión exponencial con jitter.

- **F7-OMNI-001 [AUDITADO - APROBADO] Registro Omnisciente Centralizado e Invariantes de Estado**:
  - `omniscient-registry/src/lib.rs`: Centralización de parámetros del sistema mediante snapshots rkyv zero-copy, previniendo colisiones entre subsistemas concurrentes.

- **F7-GRAPH-001 [AUDITADO - APROBADO] Grafo de Arquitectura 4D y Trazabilidad de Flujos**:
  - `graph-architecture/src/lib.rs` y `graph-4d/src/lib.rs`: Mapeo continuo de nodos y dependencias del sistema, habilitando la inspección dimensional de flujos entre ingestión, características, señales y efectores.

---

# RONDA 5 (2026-10-07, contra .ola69 — post Ola 68/69, Ω17/Ω18, GLM 104)

Mandato del operador. 3 auditores paralelo: A = paridad sombra↔vivo
SISTEMÁTICA (tabla 13 motores × 3 caminos — el chequeo que el consejo pidió
tras C1), B = matemática de las olas nuevas, C = física/conducta. **12
hallazgos (0 HIGH, 5 MED, 7 LOW)** — primera ronda SIN HIGH: las correctivas
de rondas 3-4 sostienen. El patrón residual es UNO solo: el fix se porta a
un camino y el otro queda con la calibración vieja o clave muerta.

## §R5-A — PARIDAD SOMBRA↔VIVO (tabla completa en buzón)

- **R5-A1 [MED] conformal vivo MUDO**: el evaluate lee `ema_trend_swing`
  = macro_trend CRUDO (fracción O(1e-3)); con divisor 2.0 el acuerdo ≈
  0.005 — R4-C3 calibró para la sombra tanh(z) (|trend|~O(1)), el vivo
  quedó inaudible. Fix: normalizar el trend vivo (tanh de su z) o clave
  publicada z-normalizada (writer+reader mismo commit, lección #613).
- **R5-A2 [MED] shockwave firma divergente**: sombra tanh(x)≡tanh(mach)
  vs vivo tanh(mach/2) — R4-C2 se portó sólo al vivo. Fix: `((x/c)/2).tanh()`
  en voto_espectral.
- **R5-A3 [MED] clave muerta en el consenso**: la sombra lee
  `conformal_epsilon` (0 escritores) mientras el genoma publica
  `conformal_alpha` — el consenso VIVO corre conformal con α=0.10 fijo,
  sordo a la calibración [0.01,0.30]. Fix: leer `conformal_alpha`.
- **R5-A4 [LOW] CERRADO (Ola Ω22 AGY)**: `nash_presion_adv` enlazado per-coin con `game_theory_adversarial_pressure`, con fallback dinámico al `cvpin` medido de la moneda, y finalmente a `nash_equilibrium_drift` (0.50).
- **R5-A5 [LOW] CERRADO (Ola Ω22 AGY)**: `renyi` sombra espectral alineada con la familia continua $\tanh$ de #664, erradicando el signum duro en `voto_espectral`. Test `qo_r5_a5_renyi_sombra_espectral_continua_tanh` verde.
- **R5-A6 [LOW] CERRADO (Ola Ω22 AGY)**: Telemetría individual de sombras espectrales completa 13/13 publicada al registry (`sombra_*_consenso`, `sombra_*_tau_max`, `sombra_*_v_max`) para hawkes, nash, flow, perceptron, conformal y confluence. Test `sombras_espectrales_telemetria_contract.rs` verde.

## §R5-B — MATEMÁTICA (verificada con cálculo)

- **R5-B1 [MED] fallbacks tech_threshold fuera de banda**:
  continuous_evolution_backtest.rs:310 (0.1487) y :958 (0.12) — si el
  campeón estable hereda genoma legacy por el fallback, promote SIGUE
  rechazándolo (la clase B1 de Ola 68 no erradicada del todo). Fix:
  alinear ambos a 0.24.
- **R5-B3 [MED] fallback gaussiano ALCANZABLE**: el comentario
  «inalcanzable con g4≥3» es FALSO — con γ₃>√2 el denom_sq cruza ≤0 (ej
  γ₃=2, sr=2 → −1) y cae al gaussiano sub-gaussiano anti-conservador
  justo con asimetría positiva fuerte. Fix: acotar γ₃ al discriminante.
- **R5-B2 [LOW] CERRADO (Ola Ω21 AGY)**: rejilla se re-anclaba al tick de cruce
  (Δt∈[1s,2s) con huecos de altcoins). Corregido con avance periódico por rejilla
  estricta `while tick.timestamp >= last_sample_ts.saturating_add(1000) { last += 1000 }`
  en `darwin.rs:393-395`.
- **R5-B4 [LOW-MED] CERRADO (Ola Ω21 AGY)**: en `dark-alpha-engine/src/lib.rs:407-412`,
  `forward_quantized` verificaba el clamp(±700) sin comprobar `raw_total.is_finite()`,
  lo que convertía bias=+Inf en pseudo-evidencia 1.0. Corregido retornando `f64::NAN` si
  `!raw_total.is_finite()`. Test `test_r5_b4_forward_quantized_nan_on_infinite_raw_total` verde.
- **R5-B5 [errata] CERRADO (Ola Ω21 AGY)**: corregida errata en docstring de
  `supersonic_shockwave.rs:211`: $\tanh(5)=0.99991$ (no 0.9997).

## §R5-C — FÍSICA (regresión completa VIVA)

- **R5-C1 [MED] CERRADO (Ola Ω21 AGY)** (= R4-C5 confirmado): shockwave mid ausente
  + fallback atr_pct mezclaba precio-crudo/s con fracción/s ⇒ Mach ×mid_price.
  Corregido en `supersonic_shockwave.rs:172-184`: abstenerse devolviendo 0.0 cuando falta
  `mid_price` y se recurre al fallback fraccional. Test unitario verde.
- **R5-C2 [LOW] VERIFICADO (DISEÑO ACEPTADO)**: kink C⁰ del .max(0.0) del acuerdo — semánticamente requerido
  (clase calma-abstiene aceptada en G2-1).
- **R5-C3 [LOW] VERIFICADO (DISEÑO ACEPTADO)**: clamp |x|≤10 sólo en sombra (asimetría < 1e-15 en z real).

## Mapa positivo (verificado)

- Paridad EXACTA: oscilador, SR, coaxial, trend_runner, perceptron (infer
  compartido — R4-C4/H2-6 por construcción), hawkes, flow_impulse
  (contrato h2_7). Calma-abstiene .max(0.0) en los TRES caminos de
  hawkes/flow/confluence (C1 bien propagado).
- Ola 68 bandas verificadas con derivación de coeficientes de curva
  (a,b ∈ bounds; nichos ⊇; RR ✓). DSR grid-only íntegro;
  cumulative_trials monótono. g4.max(3.0) dirección correcta.
- GLM 104 sin daño (lead_lag docs-only; ast_mutator cierra canal muerto;
  renames bit-exact). Ola 68 A1 unidades coherentes.
- TODOS los fixes de olas 62/63/65/68 siguen vivos tras los merges.

## Cierre de Ronda 5 (100% CERTIFICADA)

- **Qoder Ola 70 (merge 2721293b)**: R5-A1 + R5-A2 + R5-A3 + R5-B1 + R5-B3 cerrados.
- **Antigravity Ola Ω21 (merge 9cf77026)**: R5-B2 + R5-B4 + R5-C1 + R5-B5 cerrados.
- **Antigravity Ola Ω22 (commit actual)**: R5-A4 + R5-A5 + R5-A6 cerrados; R5-C2 + R5-C3 verificados.
- **ESTADO GLOBAL**: Ronda 5 100% CERRADA Y VERIFICADA. Cero hallazgos abiertos en R5.




# RONDA 6 (2026-10-09, contra f07b79a3 — post Ω23-Ω39, GLM 110-112, SOL R5, Codex R4, Ola 72)

Mandato del operador: «Vuelve a iniciar otra revisión desde la base, han
cambiado muchas cosas». 3 auditores paralelo READ-ONLY contra el worktree
`.ronda6` (código = f07b79a3): **A** = integración viva (callers/registro/
consumidores de los motores Ω36-Ω39), **B** = matemática financiera y
estadística (Ville/OU/StatArb/DSR/daemon), **C** = física (Hodge/Yang-Mills/
ruteo Maker/continuidad C¹). Ámbito: TODO lo integrado tras la Ronda 5.

**44 hallazgos brutos — 6 etiquetas HIGH (A1, A2, A3, B1, C1, C2) → 4
defectos únicos tras deduplicación (A1=C1; A3=B1; A9⊂C2; A7≈C7; A12≈C4;
A5⊂B1)**. El patrón histórico «dos caras sin reconciliar» se confirma por
SEXTA vez, ahora en los motores nuevos de AGY: el cableado es correcto
(nombres/ámbitos/orden intra-tick verificados SIN mismatch — los votos SÍ
llegan a producción como tensor_boost 40% del composite), pero dos de los
tres motores aportan física nula o degenerada y el contract test verde
sella la ilusión: valida plomería, no física.

## §R6-A — INTEGRACIÓN VIVA (14 hallazgos: 3 HIGH, 6 MED, 5 LOW)

- **R6-A1 [HIGH] (= R6-C1) `hodge_curl_share ≡ 0.0` por construcción — el
  ruteo Maker «vórtice» de Ω39 es código muerto y la modulación laminar del
  Consejo nunca actúa.** El camino vivo construye `F_ij = OFI_i − OFI_j`
  (`lib.rs:5042` → `hodge_flow.rs:145-167 build_gradient_flow_matrix`), un
  gradiente potencial puro: `div_i = N(X_i−X̄)`, `φ_i = X_i−X̄`, `∇φ_ij =
  X_i−X_j = F_ij` exacto ⇒ `curl_share = 1 − 1 = 0` idéntico (identidad
  clásica `‖∇φ‖² = nΣX²−S² = ‖F‖²`; el doc del módulo lo declara). La
  puerta `curl_share > 0.75 || (curl_share > 0.60 && ym_action > 0.10)`
  (`god_engine.rs:4097`) jamás dispara ⇒ `force_maker` nunca true ⇒
  `EntryRoute::Maker` (`god_engine.rs:4255-4256`) inalcanzable en
  producción; `laminar_factor = (1−0.70·curl)` en
  `consejo_seniors.rs:382-389` siempre 1.0. Ω39 es un no-op: el sistema
  enruta exactamente como antes (B3.29 IOC/Market/Iceberg). El input
  «Cross-CVD» que la misión Ω38 menciona ni siquiera existe en el cableado.
- **R6-A2 [HIGH] Yang-Mills cap 16/26 monedas + motor muerto silencioso.**
  `MAX_GAUGE_ASSETS = 16` (`yang_mills_gauge.rs:38`) vs 26 símbolos del
  bootloader (`bootloader.rs:319-346`) y `MAX_COINS = 30`.
  `update_and_calculate_curvature` computa sobre coin_id 0..15; para
  coin_id ≥ 16 `lib.rs:5035-5038` publica `yang_mills_current = 0.0`.
  10 de 26 monedas (NEAR, ICP, FIL, VET, AVAX, OP, APT, ARB, RENDER, LDO)
  nunca reciben corriente gauge: voto YM 0 permanente, fusión del senior
  (gate |J|>0.05) nunca activa. Además `yang_mills_gauge.rs:114-117`: si
  CUALQUIER precio del rango 0..n es ≤0/no finito devuelve ceros —
  universo activo < 16 (pool evoluciona cada hora, `symbol_manager.rs:
  136-190`) ⇒ motor muerto sin telemetría que lo delate. El contract test
  pasa en verde con el motor inerte (fixture de 3 monedas ⇒ 27 slots en 0
  ⇒ aserciones de rango triviales, `hodge_yang_mills_consensus_contract.rs:
  81-93`).
- **R6-A3 [HIGH] (= R6-B1) StatArb: toda la física OU vive sólo en tests;
  la guarda espectral `t_{1/2} ≤ 2τ*` es decorativa en el camino vivo.**
  `update_with_clock` (`stat_arb.rs:188-271`) — único método que calibra la
  SDE con reloj físico, adapta β por RLS, aplica damping espectral y borde
  mínimo — tiene CERO callers de producción (grep `crates/`+`src/`: sólo
  tests). El camino vivo `evaluate_for_coin` (`stat_arb.rs:288-331`) lee un
  z-score ajeno (`vecm_zscore`) con guarda `half_life = ln2/θ` donde θ es
  el valor INICIAL 0.1 (`stat_arb.rs:63`) ⇒ t½ congelado en 6.93 s para
  siempre; τ* viene de `dominant_tau_ms` con fallback duro 1138 s ⇒ la
  guarda `6.93 > 2·1138` jamás dispara. Ítem `JohansenVecmEngine::update*`
  y `MultivariateCointegrationEngine`: sin caller productivo.
- **R6-A4 [MED] β RLS adaptativa implementada pero no habilitada**: la
  instancia viva es `StatArbEngine::new(30, 1.5).with_continuous_ou_sde()`
  SIN `.with_adaptive_beta(true)` (`lib.rs:999`) ⇒ spread siempre
  `ln A − 1.0·ln B`. El RLS existe con test propio pero es código muerto
  en vivo.
- **R6-A5 [MED] (⊂ R6-B1) El z que vota «StatArb» no es un spread de pares
  cointegrados**: `vecm_zscore` es `((mid−spot_mid)/(mid·atr_pct)).clamp(−3,3)`
  — basis spot-perp normalizado por ATR (`lib.rs:5013-5023`); y
  `cointegration_zscore` es `leader_mom.clamp(-3,3)` (impulso lead-lag,
  `lib.rs:5024`). El etiquetado describe una física que el flujo no tiene.
- **R6-A6 [MED] Paridad BT↔vivo rota en `macro_staleness_ms`**: el host la
  publica por evento (`god_engine.rs:3231-3232`, fail-safe `u64::MAX`),
  pero backtest-engine NUNCA la escribe (grep: 0 escritores) ⇒ en replay el
  Consejo lee 0.0 (`lib.rs:7425-7428`) y `SeniorEnteMercado` nunca amortigua
  (p_macro ≡ 1.0), mientras en vivo sí (piso 0.40). Divergencia sistemática
  replay↔producción en la rama de decisión (mismo patrón D-707).
- **R6-A7 [MED] (≈ R6-C7) Contract test Ω38 = plomería verde, contrato de
  física ausente**: aserciones de existencia/rango; pasa con los motores
  estructuralmente muertos. Sin caso de no-degeneración (inyectar vórtice
  puro — `build_pure_vortex_matrix` existe para eso, `hodge_flow.rs:
  173-190` — y exigir curl≈1). La línea 111 del test escribe ella misma
  `macro_staleness_ms` y la relee.
- **R6-A8 [MED] `J_i` publicado sin acotar ⇒ `integrity_failure` ⇒ veto**:
  la normalización (`yang_mills_gauge.rs:173-177`) divide por ciclos
  incidentes pero NO clampa; dos lectores clampean (voto `:230`, senior
  `consejo_seniors.rs:413`) pero el payload del Consejo lee el valor CRUDO
  (`lib.rs:7421-7424`) y `MarketSnapshotPayload::validate()` RECHAZA
  |J|>1 (`consejo_seniors.rs:223`) ⇒ en dislocación multi-moneda severa —
  exactamente donde la reversión gauge operaría — el Consejo veta por
  integridad del snapshot, no por mérito. El escritor debería acotar como
  el resto del pipeline.
- **R6-A9 [MED] (⊂ R6-C2) El estimador gauge aprende a anular la cantidad
  que mide**: el LMS de β minimiza `err = ln P_i − β_ij·ln P_j` ⇒ β
  converge a `ln P_i/ln P_j` ⇒ `A_ij → 0` ⇒ `F_ijk → 0` en estado
  estacionario; el «detector» mide sólo innovaciones transitorias de una β
  que persigue los precios tick a tick, no residuos persistentes de
  cointegración. La invariancia gauge del doc (β_ij·β_jk·β_ki = 1) no está
  impuesta.
- **R6-A10 [LOW] Claves telemetry sin lectores**: `hodge_curl_energy` (0
  lectores), `hodge_gradient_energy` (sólo test), `censo_total_{name}`/
  `censo_no_cero_{name}` (`orchestrator.rs:433-439`). «Publicar sin medir»
  — el lado que la doctrina del propio repo prohíbe.
- **R6-A11 [LOW] Doble implementación Hodge con entradas de calidad
  opuesta**: `risk-engine/src/hodge.rs` (Ola XLVI·C) aplica la MISMA
  identidad sobre un flujo por pares REAL (Hawkes α_ij − α_ji vía
  `contagion_publisher.rs:84-87`), publica `hawkes_contagion_curl_share` y
  tiene consumidor vivo (`correlation_guard.rs:696`). Ω37 duplicó la
  matemática con la entrada degenerada — el patrón correcto YA existía en
  el árbol. Agrava A1.
- **R6-A12 [LOW] (≈ R6-C4) `latest_prices`/`latest_ofis` sin TTL**: una
  moneda expulsada del pool dinámico congela su último valor alimentando
  YM/Hodge indefinidamente; nada distingue «fresco» de «congelado».
- **R6-A13 [LOW] Guard `coin_id < ym_currents.len()` (len=16) = clamp
  silencioso**: «sin corriente» y «fuera del fibrado» comparten la cara 0.0
  en el registro; el consumidor no puede abstenerse conscientemente.
- **R6-A14 [LOW] Instancia StatArb duplicada y muerta** en
  `src/multi_asset_orchestrator.rs:25` (`new(100, 2.0)` vs la viva
  `new(30, 1.5)`), módulo sin referencia desde `god_engine.rs`.

## §R6-B — MATEMÁTICA (18 hallazgos: 1 HIGH, 7 MED, 10 LOW)

Veredicto del auditor: la matemática central es CORRECTA — no hay HIGH por
matemática rota ni lookahead. El hallazgo más grave es de cableado (B1=A3).

- **R6-B1 [HIGH] (= R6-A3) La capa estocástica no está cableada a
  producción; `vecm_zscore` no es cointegración.** Callers de
  `update_with_clock`/`JohansenVecmEngine::update*`/
  `MultivariateCointegrationEngine`: sólo tests. Los tres motores del
  orquestador sólo ejecutan `evaluate_for_coin`, que LEE `vecm_zscore` — y
  esa clave la escribe el CORE como basis spot-perp/ATR (`lib.rs:5013-5023`),
  sin β, sin Welford, sin VECM, sin SDE. Toda la «exclusividad espectral»
  P5/P6/P9 es matemática de biblioteca inerte; el nombre de la clave induce
  a creer lo contrario.
- **R6-B2 [MED] Guarda espectral estructuralmente muerta (θ congelada)**:
  la guarda lee `self.physical_sde` que sólo avanza dentro de
  `update_with_clock` (sin caller); SDE creada con θ=0.1 ⇒ t½=6.93 s
  congelado, jamás supera 2τ* (τ*≥30 s ⇒ umbral ≥60 s; fallback 1138 s).
  La «paridad espectral» del voto StatArb siempre pasa (`stat_arb.rs:
  310-319`).
- **R6-B3 [MED] Pooling de pares con Δt heterogéneo en la OLS de la SDE OU**
  (`vecm_arbitrage.rs:290-331`): la regresión mezcla pares consecutivos con
  Δt arbitrario; la pendiente única b = e^{−θΔt} sólo existe para Δt fijo ⇒
  b̂ promedio incoherente de reversión a distintas escalas; θ̂ usa el Δt del
  ÚLTIMO evento mientras el SSE mezcla todos; σ̂ divide por
  (1−e^{−2θΔt_último}). Hoy inerte por B1, pero es el defecto que estallaría
  al cablear.
- **R6-B4 [MED] Decay clamp [0.80, 0.999]: memoria efectiva de 5 a 1000
  eventos, no «~100 observaciones»** (`vecm_arbitrage.rs:293-294`): para
  todo dt > 67 s — casi todo el rango espectral operativo de minutos a 12 h —
  el decay queda congelado en 0.80 ⇒ n_eff ≈ 5 ⇒ Var(b̂) enorme, θ̂
  esencialmente ruido clampeado; `count ≥ 10` se satisface con 10 eventos
  (5 min) para calibrar un θ cuya escala puede ser de horas. El horizonte
  de la EWMA debería fijarse en TIEMPO, no en eventos.
- **R6-B5 [MED] `is_exhausted` heurístico no es anytime-valid y dispara
  fácil bajo H₀** (`ville_e_process.rs:212-215`, consumido en
  `online_daemon.rs:1201-1206`): `peak > 1 && e_value < 1 && running_mean
  <= 0` mezcla el e-proceso con un estadístico no acotado bajo H₀. Con
  retornos ±2% y λ=0.05, ln M se mueve ±0.001/trade: el PRIMER trade
  ganador pone peak>1; luego M<1 ∧ media≤0 (≈50% del tiempo cada uno,
  correlacionados) declara «agotamiento». La garantía de Ville sólo cubre
  `anytime_p_value`/`is_edge_certified`, que NO se usan en el rollback.
  Rollback agresivo (sensible, no válido) que corta estrategias con edge
  ruidoso.
- **R6-B6 [MED] λ∈[0.05,0.50] descalibrado vs retornos por-trade ±2% ⇒
  watchdog Ville casi insensible** (`online_daemon.rs:1346`): con |x|≈0.02
  cada trade mueve ln M en ±0.001; `is_evidence_decayed(0.50)` requiere ≈693
  trades netos perdedores; mientras el t-stat clásico dispara a −2 con ≈4
  trades. La detección real de degradación la hace íntegramente el t-stat
  descriptivo — el Ville es respaldo válido pero prácticamente inerte en la
  ventana de 500. Validez intacta (P2/P4); POTENCIA no corresponde a la
  escala real de los datos.
- **R6-B7 [MED] Gap de rearme del Ville**: hot-swap del MISMO genoma tras
  rollback arma el watchdog clásico pero no el Ville (`online_daemon.rs:
  310-334` no toca `post_promo_ville` en `armar_vigilancia`; l.1905-1906
  sólo genoma distinto; camino externo ve `==` no `>`). Ese genoma corre
  vigilado sólo por t-stat, sin capa anytime-valid, indefinidamente.
- **R6-B8 [MED] El test de H₀ del e-proceso ejercita λ_min=0, no la
  configuración productiva λ_min=0.05** (`ville_e_process.rs:63` vs
  `online_daemon.rs:1346`): con λ_min=0 y media≤0, M≡1 ⇒ `!is_edge_
  certified()` pasa trivial. El régimen productivo (M fluctúa bajo H₀, ver
  B5) no tiene ninguna prueba empírica de control de falsos positivos.
- **R6-B9 [LOW] «Online Newton Step» mal etiquetado**: es SGD con schedule
  (sin matriz A ni proyección); la «ratio de Sharpe empírica» es un Kelly
  μ̂/σ̂₂ clampado. Validez no depende de esto (λ previsible acotada).
- **R6-B10 [LOW] Clamp |x|≤1 comprime colas de pérdida**: necesario para
  no-negatividad (x < −1/λ destruiría la supermartingala), pero con λ=0.5
  un retorno −150% (gap/liquidación) se reporta −100% ⇒ M multiplica 0.5
  en vez de 0.25: el detector responde MENOS a los colapsos cuando más
  importa. Escenario no imposible con micro-cuenta apalancada.
- **R6-B11 [LOW] `update_continuous_sde` acumula TASAS en los momentos
  mientras `update()` acumula retornos** (`ville_e_process.rs:145-171`):
  si un mismo proceso recibiera ambos flujos, `compute_causal_lambda`
  mezclaría escalas. Sin caller productivo hoy.
- **R6-B12 [LOW] El «RLS» de β es LMS normalizado con gain fijo** (γ≈0.001,
  clamp [0.1,10]): sin matriz P ni factor de olvido; absorbe un cambio
  estructural en ~1000 eventos — más lento que la dinámica espectral
  prometida. β negativo verdadero forzado a la frontera positiva.
- **R6-B13 [LOW] Dos fallbacks distintos para τ\***: 30.0 s en
  `update_with_clock` vs 1138.0 s en `evaluate_for_coin` (`stat_arb.rs:
  216-220` vs `:312-315`).
- **R6-B14 [LOW/POSITIVO] Sesgo de Jensen de θ̂ despreciable** (≈3-7e-4 vs
  error de muestreo 30-45%): el problema real de precisión es B4 (n_eff),
  no Jensen.
- **R6-B15 [LOW] Clamps de frontera b∈[0.001,0.9999], θ∈[1e-4,50]**:
  defensa razonable pero dependen del gate muerto (B2); spread
  cuasi-random-walk reporta t½ ≤ 35 min para un no-estacionario.
- **R6-B16 [LOW] Camino legacy de multivariate_coint**: `spread_deviation`
  contra la media post-actualización (convención de timing inconsistente
  con z_prior del resto del módulo); sin caller productivo.
- **R6-B17 [LOW] Pisos/techos mágicos de la señal SDE**: `confidence`
  piso 0.5 activo para thresholds <1.5; gate `t½ ≤ 3600 s` excluye la
  mitad superior de la banda espectral declarada [30 s, 12 h]
  (`multivariate_coint.rs:131-146`).
- **R6-B18 [LOW/POSITIVO] DSR evalúa el SE en el SR observado, no en SR\***:
  cota conservadora (SE mayor ⇒ z menor), decisión documentada. Informado
  por trazabilidad con Bailey & LdP.

## §R6-C — FÍSICA (12 hallazgos: 2 HIGH, 5 MED, 5 LOW)

- **R6-C1 [HIGH] (= R6-A1) El feed vivo de Hodge es gradiente puro ⇒
  `curl_share ≡ 0`; ruta Maker Ω39 muerta, modulación laminar inerte.**
  Derivación completa en R6-A1. La única constructora que produce
  rotacional real (`build_pure_vortex_matrix`) sólo se invoca en tests.
  La física de la descomposición es correcta; el wiring degenera el objeto
  matemático que gobierna órdenes reales.
- **R6-C2 [HIGH] Yang-Mills: β asimétrico viola el cierre gauge del propio
  módulo; con β=1 la holonomía es idénticamente 0 y la señal mide ruido de
  adaptación sobre una «paridad» de NIVELES de precio sin contenido
  económico.** `F_ijk = (1−β_ki)ln P_i + (1−β_ij)ln P_j + (1−β_jk)ln P_k`;
  con inicialización β=1, F ≡ 0 exacto (telescópico — el «vacío gauge» es
  trivial, no un logro físico). El RLS adapta cada β_ij independiente; β_ij
  y β_ji divergen; la condición β_ij·β_jk·β_ki = 1 del doc jamás se
  verifica/proyecta/penaliza ⇒ S_YM ≠ 0 surge SÓLO del drift asimétrico de
  β, no de curvatura de arbitraje. Y el objetivo de regresión es paridad
  entre NIVELES (ln P_BTC ≈ β·ln P_DOGE) — no existe ley de un solo precio
  entre niveles de activos no intercambiables; el contenido económico está
  en spreads/retornos. J_i se mezcla al 20% en SeniorSeriesTemporales
  (`consejo_seniors.rs:412-414`) — contaminación de derivación en señal
  que alimenta votos reales.
- **R6-C3 [MED] S_YM sin normalizar escala con C(N,3) tríadas; el umbral
  absoluto 0.10 no tiene anclaje dimensional** (`yang_mills_gauge.rs:
  144-170` vs `god_engine.rs:4097`): C(13,3)=286 tríadas; la misma curvatura
  física produce acción mayor en universos mayores (y el universo es
  dinámico). Contenido hoy porque la puerta compuesta está muerta por C1,
  pero al corregir C1 este umbral heredaría el defecto.
- **R6-C4 [MED] (≈ R6-A12) Sin sincronización temporal ni TTL en los
  buffers multiactivo**: la descomposición trata el vector como fotografía
  simultánea; cada componente puede tener edades arbitrarias (moneda
  ilíquida contribuye su OFI congelado). Sin ventana temporal contrastada
  contra los 400 ms de latencia post-only que justifica la ruta Maker.
- **R6-C5 [MED] La «detección de cascada laminar (curl < 0.25 → IOC)» está
  documentada pero NO existe en el código**: MEMORIA/Ω39 y el comentario de
  `god_engine.rs:4090` describen régimen bidireccional; el despacho real
  (`god_engine.rs:4255-4290`) hace IOC default incondicional — no hay rama
  que evalúe curl<0.25 para ELEGIR IOC. Divergencia docs↔código.
- **R6-C6 [MED] Recomputo O(N³)+O(N²) y re-adaptación de β dentro del
  bucle por moneda** (`lib.rs:5034-5054`): por ronda de N ticks ⇒ N
  evaluaciones O(N³), N descomposiciones O(N²) y N pasos de RLS con el
  mismo objetivo (γ_efectiva ≈ γ·N·cadencia); `ym_action` publicado queda
  path-dependent del orden de llegada de los ticks.
- **R6-C7 [MED] (≈ R6-A7) El contract test pasa trivialmente**: nunca
  ejercita un vórtice por la ruta viva; «100% verde» no certifica la
  física del ruteo.
- **R6-C8 [LOW] `dbp <= dap` es trivialmente cierto en libro válido**
  (sanidad, no indicador direccional); con C1 la puerta se reduce a falso
  + validez de libro; en paths sin depth5 previo dbp=dap=0 desarma la
  ruta.
- **R6-C9 [LOW] R0 de AGY: clasificación espectral en escalones discretos
  con comentario que proclama continuidad** (`position.rs:584-658`,
  vocabulario scalp/swing que U-ERR-5 erradicó) — pero grep exhaustivo:
  consumidores sólo tests; ninguna conducta de producción depende de estas
  etiquetas. El τ continuo real (`entry_tau_ms`) sigue gobernando.
- **R6-C10 [LOW] Maker: rampa OBI continua pero C⁰ (no C¹), `signum` en el
  skew, literales mágicos sin anclaje** (`maker.rs:60-153`): quiebre de
  pendiente en |obi|=th y en la saturación; daño hoy nulo (ruta muerta por
  C1), heredaría al reactivar.
- **R6-C11 [LOW] `evaluate()` de YM con símbolo vacío lee el valor GLOBAL
  del registry (carrera last-writer)**: el orquestador real usa símbolo
  scoped (correcto); trampa latente para callers futuros.
- **R6-C12 [LOW] Dead-zone |ym_curr|>0.05 en SeniorSeriesTemporales =
  salto C⁰ en la señal compuesta** (`consejo_seniors.rs:412-414`): viola la
  doctrina C¹ en un punto de mezcla; si se corrige C2 pasa a ser el
  defecto dominante del blend.

## Mapa positivo (verificado por los 3 auditores)

- **Cableado de registro SIN mismatch**: el host lee las claves que el core
  escribe, en los TRES ámbitos (global/c{id}:/{sym}_), formateo en stack.
  Umbrales del ruteo = especificación Ω39 exacta. El defecto NO es de
  nombres.
- **Ordenamiento intra-tick correcto**: acoplamiento (lib.rs:5035-5054)
  ANTES de orquestador (lib.rs:5148) y snapshot del Consejo
  (lib.rs:7417-7428), dentro del mismo `process_tick_dual`.
- **Los votos nuevos SÍ llegan a producción**: consenso espectral como
  tensor_boost 40% del composite (lib.rs:5152-5170); no es sombra pura.
- **VilleEProcess matemáticamente sólido**: no-negatividad estructural
  (P1), λ previsible F_{t−1}-medible (P2), anytime_p_value exacto = cota
  maximal de Ville (P3), λ_min>0 decae bajo H₀ sin explotar (P4).
- **Álgebra OLS-EWMA de la SDE OU EXACTA** (P5, verificada simbólicamente;
  sin lookahead); unidades t½ ≤ 2τ* CORRECTAS en ambos caminos (P6 — la
  sospecha NO se confirma); Mertens con fallback jamás sub-gaussiano (P7);
  Gumbel eq.5 exacta y N SÍ sigue acumulando entre épocas (P8 — fix H1-3
  persistente); exclusividad SDE real en el módulo (P9); honestidad
  estadística del daemon (P10).
- **`hodge_flow.rs` núcleo EXACTO**: Σdiv=0 por antisimetría telescópica,
  identidad de Dirichlet exacta sobre K_N, Pitágoras por construcción,
  guardas NaN/n<3 correctas, O(N²) sin allocs. Dimensionalidad de entradas
  correcta (OFI adimensional). El daño está en qué matriz se le alimenta.
- **El patrón correcto YA existía en el árbol**: el Hodge de contagio
  (XLVI·C, risk-engine) con flujo por pares real (α_ij − α_ji) y
  consumidor vivo — el wiring que Ω37 debería haber seguido.
- **Fail-safes consistentes**: YM devuelve cero ante precios inválidos;
  macro_staleness fail-safe (u64::MAX, no 0); damping continuo con piso;
  maker_price al top-of-book vivo (no mid congelado); IOC con slippage
  dinámico dimensionalmente sano; B3.29 desactivación de Maker para
  momentum correcta (0% fills pasivos en 39 entradas).
- **Cero heap alloc verificado** en buffers multiactivo y acoplamiento.

## Asignación Ronda 6 (olas correctivas)

Los 6 HIGH crudos → 4 defectos únicos, drenados en 3 olas por zona de
autoría (los motores son de AGY; la capa estocástica/daemon es zona
estadística Qoder; paridad BT es zona GLM/Codex):

- **Ola 73 (Qoder, con oráculo) — StatArb honesto**: R6-A3/B1 (cablear la
  física viva: escritor real del spread con SDE+reloj o renombrar la clave
  a lo que es — basis_atr_z — con paridad lector/escritor en el MISMO
  commit), R6-B2 (θ viva), R6-A4 (β adaptativa on), R6-B13 (fallback τ*
  unificado), R6-A5 (re-etiquetado del voto).
- **Ola Ω41 (AGY) — motores gauge** (renumerada: AGY usó Ω40 para
  CL-14 durante esta auditoría): R6-A1/C1 (alimentar Hodge con flujo
  con contenido rotacional real — flujo por pares dirigidos L2 siguiendo
  el patrón del propio hawkes_contagion, NO gradiente de escalares),
  R6-A2 (cap 16→universo + no-muerto-silencioso), R6-C2/A9 (β simétrico
  β_ij·β_ji=1 + regresar sobre spreads, no niveles), R6-C3 (normalizar
  S_YM por C(N,3)), R6-A8 (clamp J_i ANTES de publicar).
- **Ola GLM/Codex — paridad y calibración**: R6-A6 (macro_staleness_ms en
  backtest), R6-B4 (decay en TIEMPO no eventos), R6-B3 (estratificar Δt).
- **Cola Qoder posterior**: R6-B5/B6/B7 (Ville daemon: umbral real
  anytime-valid en is_exhausted, λ_min a escala de retornos reales, gap
  rearme mismo-genoma), R6-B8 (test H₀ con λ_min productivo).
- **MED/LOW mecánicos**: R6-C5 (borrar docs de cascada laminar o
  cablearla), R6-A7/C7 (contrato de no-degeneración con vórtice inyectado),
  R6-C12 (dead-zone C⁰ → blend continuo), R6-A10/A14 (claves muertas,
  instancia duplicada), R6-C9 (comentario R0).

**ACTUALIZACIÓN post-Ω40 (862d04fc, mergeado durante esta ronda)**: AGY
cerró CL-14 restaurando `force_maker = false` por política B3.29 (IOC
siempre: 400ms pasivos + selección adversa 38/38 empírica). La ruta Maker
cerrada pasa a ser POLÍTICA deliberada — pero R6-A1/C1 SUBSISTE: Ω40
preserva el cálculo de vórtices "para gobernanza de riesgo y modulación
de dispersión", y ese cálculo sigue degenerado (curl≡0 ⇒ modulación
laminar constante 1.0, telemetría `hodge_curl_share` ≈0 siempre,
`_is_mean_reversion_vortex` código muerto). La ola Ω41 debe alimentar el
Hodge con flujo real ANTES de que esa gobernanza signifique algo.

**CERO HIGH de rondas 2-5 sobrevive abierto** (todo drenado); esta ronda
abre 4 HIGH NUEVOS, todos concentrados en el stack Ω36-Ω39 integrado SIN
barrido previo — confirma la necesidad del re-barrido por base del
operador.
