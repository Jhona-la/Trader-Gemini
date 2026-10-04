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
| F0 | F1-F3 ya cubiertas por las auditorías sistemáticas de la sesión (3 auditores, ~25 hallazgos, 12 cerrados en olas #648-#653) — se documentan los restos | — | — |
