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
