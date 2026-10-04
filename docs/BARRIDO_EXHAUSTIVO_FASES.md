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
